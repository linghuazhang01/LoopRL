"""EP-GRPO actor and worker for verl 0.6.1 (``rltt_baseline.extrap`` on the RLTT training path).

Requires verl 0.6.1 and ray; import only inside the training environment. The launcher installs
``ExtrapActorRolloutRefWorker`` in place of the vendor RLTT worker when ``extrap.enabled``, and
points the vLLM rollout at ``rltt_baseline.vllm_loop_cd`` with the same LoopCD settings
(``config.add_extrap_overrides``), so the rollouts are samples of the policy trained here.

``ExtrapActor`` reuses ``PLCActor``'s plumbing (padded per-loop forward, per-loop entropy metrics,
loop audit hook, loss aggregation, clip settings) with its own loss:

- ``compute_log_prob`` runs the per-loop forward with ``return_extrap_log_probs`` under
  ``no_grad`` and returns the extrapolated policy's log-probs and entropy as verl's
  ``old_log_probs`` / ``entropys`` (so verl's rollout-vs-training check compares the same
  distribution), and keeps the per-loop log-probs ``old_log_probs_loops`` ``[B, L, T]`` for the
  loop audit and metrics.
- ``update_policy`` recomputes the extrapolated log-probs with gradient and backpropagates
  ``extrap.extrap_policy_loss`` (on-policy GRPO on pi_w, per-token weights c_j from the same
  forward, stop-gradient), + the vendor entropy bonus and the final-policy KL (both 0 here).

The model directory must provide ``return_extrap_log_probs``: rebuild ``LOOPRL_MODEL_DIR`` with
``rltt_baseline.prepare_model``.
"""

import os
import sys
from typing import Dict, Optional

import ray
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.single_controller.base.decorator import (
    Dispatch,
    make_nd_compute_dataproto_dispatch_fn,
    register,
)
from verl.utils.device import get_device_id
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import (
    prepare_dynamic_batch,
    rearrange_micro_batches,
    restore_dynamic_batch,
)
from verl.workers.fsdp_workers import ActorRolloutRefWorker

from rltt_baseline import extrap, loss_agg, plc
from rltt_baseline.loop_audit import attach_loop_audit
from rltt_baseline.per_loop_forward import extrap_forward
from rltt_baseline.plc_workers import OLD_LOOPS_KEY, PLCActor
from rltt_baseline.workers import MetricsRLTTActor


class ExtrapActor(PLCActor):
    def __init__(self, config, actor_module, actor_optimizer=None, settings=None,
                 total_ut_steps: int = 4,
                 loss_agg_settings: Optional[loss_agg.LossAggSettings] = None):
        if settings is None:
            raise ValueError("ExtrapActor needs ExtrapSettings")
        # PLCActor.__init__ is PLC-specific; build the metrics actor directly
        MetricsRLTTActor.__init__(self, config, actor_module, actor_optimizer,
                                  loop_weighting="uniform", total_ut_steps=total_ut_steps)
        settings.validate(total_ut_steps)
        self.settings = settings  # not PLCSettings: the loop audit records no PLC weights
        self.loss_agg = loss_agg_settings or loss_agg.LossAggSettings()
        self.init_module = None
        if not self._model_forward_accepts("return_extrap_log_probs"):
            raise RuntimeError("extrap.enabled needs a model directory whose modeling_ouro.py has "
                               "return_extrap_log_probs: rebuild LOOPRL_MODEL_DIR with "
                               "rltt_baseline.prepare_model")
        self._old_extras: Optional[Dict[str, torch.Tensor]] = None

    def _forward_micro_batch_plc(self, micro_batch, temperature: float):
        """Last-loop entropy, per-loop log-probs ``[B, L, T]`` and the extrapolated policy's
        tensors ``[B, L]``. Keeps the name the loop audit hooks."""
        self._captured = None
        entropy, ell, ext = extrap_forward(self.actor_module, micro_batch, temperature,
                                           self.settings.model_kwargs())
        if torch.is_grad_enabled() and self._captured is not None:
            self._pending.append(self._loop_metrics(micro_batch))
        self._captured = None
        return entropy, ell, ext

    def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False):
        self.actor_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        data = data.select(batch_keys=["responses", "input_ids", "attention_mask", "position_ids"])
        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs, entropies, ells = [], [], []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                _, ell, ext = self._forward_micro_batch_plc(inputs, temperature)
            log_probs.append(ext["log_probs"])
            entropies.append(ext["entropy"])
            ells.append(ell)
        log_prob = torch.cat(log_probs, dim=0)
        entropy = torch.cat(entropies, dim=0)
        ell = torch.cat(ells, dim=0)
        if use_dynamic_bsz:
            log_prob = restore_dynamic_batch(log_prob, batch_idx_list)
            entropy = restore_dynamic_batch(entropy, batch_idx_list)
            ell = restore_dynamic_batch(ell, batch_idx_list)
        self._old_extras = {OLD_LOOPS_KEY: ell}
        return log_prob.contiguous(), (entropy if calculate_entropy else None)

    def update_policy(self, data: DataProto) -> Dict[str, object]:
        settings = self.settings
        self._pending = []
        self.actor_module.train()
        temperature = data.meta_info["temperature"]
        loss_agg_mode = self.config.loss_agg_mode
        clip = self._clip_settings()

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids",
                       "advantages", "old_log_probs"]
        missing = [k for k in select_keys if k not in data.batch.keys()]
        if missing:
            raise KeyError(f"EP-GRPO update_policy: batch lacks {missing}")
        use_ref_log_prob = "ref_log_prob" in data.batch.keys()
        if use_ref_log_prob:
            select_keys.append("ref_log_prob")
        row_normalized = self.loss_agg.row_normalized
        batch = data.select(batch_keys=select_keys).batch
        metrics: Dict[str, object] = {}
        tracker = loss_agg.LossTracker()
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        for _ in range(self.config.ppo_epochs):
            for mini_batch in dataloader:
                if self.config.use_dynamic_bsz:
                    max_token_len = (self.config.ppo_max_token_len_per_gpu
                                     * self.ulysses_sequence_parallel_size)
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch,
                                                               max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (self.config.ppo_mini_batch_size
                                                  // self.config.ppo_micro_batch_size_per_gpu)
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()
                for micro in micro_batches:
                    micro = micro.to(torch.cuda.current_device())
                    response_length = micro["responses"].size(1)
                    response_mask = micro["attention_mask"][:, -response_length:]
                    advantages = micro["advantages"]
                    row_div = (loss_agg.row_divisors(self.loss_agg, response_mask)
                               if row_normalized else None)

                    _, ell_new, ext = self._forward_micro_batch_plc(micro, temperature)
                    log_prob = ext["log_probs"]
                    weights = extrap.token_weights(ext["omega"], ext["margin"], settings)
                    pg_loss, step_metrics = extrap.extrap_policy_loss(
                        log_prob, micro["old_log_probs"].float(), advantages, response_mask,
                        weights, loss_agg_mode=loss_agg_mode, row_div=row_div, **clip)
                    policy_loss = pg_loss
                    terms = loss_agg.pop_loss_terms(step_metrics)

                    entropy = ext["entropy"]
                    entropy_loss = plc.agg_loss(entropy, response_mask, loss_agg_mode)
                    entropy_coeff = float(self.config.entropy_coeff)
                    if entropy_coeff != 0:
                        entropy_term = entropy_coeff * plc.agg_term(
                            entropy, response_mask, loss_agg_mode, row_div)
                        policy_loss = policy_loss - entropy_term
                        terms["entropy"] = -float(entropy_term.detach())

                    ref_key = "ref_log_prob" if use_ref_log_prob else "old_log_probs"
                    kld = plc.low_var_kl(log_prob, micro[ref_key].float())
                    kl_loss = plc.agg_loss(kld, response_mask, loss_agg_mode)
                    kl_coef = float(self.config.kl_loss_coef)
                    kl_term = plc.agg_term(kld, response_mask, loss_agg_mode, row_div) * kl_coef
                    policy_loss = policy_loss + kl_term
                    if kl_coef != 0:
                        terms["kl"] = float(kl_term.detach())

                    if self.config.use_dynamic_bsz:
                        scale = response_mask.shape[0] / self.config.ppo_mini_batch_size
                        loss = policy_loss * scale
                    else:
                        scale = 1.0 / self.gradient_accumulation
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()
                    terms["total"] = float(policy_loss.detach())
                    tracker.add(terms, scale)

                    step_metrics.update({
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/entropy_loss": entropy_loss.detach().item(),
                        "actor/kl_loss": kl_loss.detach().item(),
                        "actor/kl_coef": kl_coef,
                        "actor/total_loss": policy_loss.detach().item(),
                    })
                    if row_div is not None:
                        step_metrics["loss_agg/row_div_mean"] = row_div.mean().item()
                    step_metrics.update(extrap.extrap_stats(
                        log_prob.detach(), entropy, ext["omega"], ext["margin"], weights,
                        ell_new[..., -1].detach(), response_mask))
                    append_to_dict(metrics, step_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item(),
                                         **tracker.flush()})

        self.actor_optimizer.zero_grad()
        for record in self._pending:
            append_to_dict(metrics, record)
        self._pending = []
        return metrics


@ray.remote
class ExtrapActorRolloutRefWorker(ActorRolloutRefWorker):
    """verl worker that builds ``ExtrapActor`` and adds its per-loop old tensors to the batch."""

    def __init__(self, config, role: str, **kwargs):
        raw = config.actor.get("extrap", None)
        if raw is None:
            raise ValueError("actor_rollout_ref.actor.extrap is missing")
        settings = extrap.ExtrapSettings.from_mapping(OmegaConf.to_container(raw, resolve=True))
        # verl 0.6.1 materializes actor as FSDPActorConfig, which has no extrap field
        native_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        OmegaConf.set_struct(native_config.actor, False)
        native_config.actor.pop("extrap")
        super().__init__(native_config, role, **kwargs)
        self.settings = settings
        self.total_ut_steps = config.get("total_ut_steps", 4)
        raw_agg = config.get("loss_agg", None)
        self.loss_agg_settings = loss_agg.LossAggSettings.from_mapping(
            OmegaConf.to_container(raw_agg, resolve=True) if raw_agg is not None else None)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if self._is_actor:
            self.actor = ExtrapActor(
                config=self.actor.config,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
                settings=self.settings,
                total_ut_steps=self.total_ut_steps,
                loss_agg_settings=self.loss_agg_settings,
            )
            attach_loop_audit(self.actor, self.config.get("loop_audit"), self.rank)
            if self.rank == 0:
                print(f"[rltt_baseline] ExtrapActor {self.settings.describe()} "
                      f"{self.loss_agg_settings.describe()} pid={os.getpid()} "
                      f"python={sys.executable}", flush=True)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_log_prob(self, data: DataProto):
        output = super().compute_log_prob(data)
        extras = self.actor.pop_old_extras() if self._is_actor else None
        if extras:
            output = output.union(DataProto.from_dict(
                tensors={k: v.to("cpu") for k, v in extras.items()}))
        return output
