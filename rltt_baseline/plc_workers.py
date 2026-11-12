"""PLC actor and worker for verl (``rltt_baseline.plc`` wired into the RLTT training path).

Requires verl 0.6.1 and ray; import only inside the training environment. The launcher
installs ``PLCActorRolloutRefWorker`` in place of the vendor RLTT worker when ``plc.enabled``.

``PLCActor`` extends ``MetricsRLTTActor`` (per-loop entropy metrics, vendor forward unchanged)
and replaces the vendor's REINFORCE loss:

- ``compute_log_prob`` runs the per-loop forward under ``no_grad`` and returns the final-loop
  log-probs / entropies like verl; it also keeps ``old_log_probs_loops`` ``[B, L, T]``, which the
  worker adds to the batch. The final loop of ``old_log_probs_loops`` is the ``old_log_probs``
  tensor itself, so ``lam = 0`` is exactly verl's GRPO loss on this forward.
- ``update_policy`` recomputes the per-loop log-probs with gradient, forms the earlier-loop
  weights from the old tensors (fixed for the whole update) and backpropagates
  ``plc_policy_loss``: GRPO on the final loop plus the truncated-IS surrogate and forward-KL trust
  region of the earlier loops, + the vendor's entropy bonus and final-loop KL. It also logs the
  cross-loop ratio statistics (``plc.cross_loop_stats``).

Per-token weight statistics are logged under ``plc/``, one scalar key per loop
(``plc/<name>/loop_<t>``, ``plc.flatten_loop_metrics``): verl's ``reduce_metrics`` takes the mean
of each key's per-micro-batch list, which would also average a per-loop list over its loops.
"""

import os
import sys
from typing import Dict, Optional, Tuple

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

from rltt_baseline import plc
from rltt_baseline.loop_audit import attach_loop_audit
from rltt_baseline.workers import MetricsRLTTActor

OLD_LOOPS_KEY = "old_log_probs_loops"


class PLCActor(MetricsRLTTActor):
    def __init__(self, config, actor_module, actor_optimizer=None, settings=None,
                 total_ut_steps: int = 4):
        if settings is None:
            raise ValueError("PLCActor needs PLCSettings")
        super().__init__(
            config, actor_module, actor_optimizer,
            # the vendor flag only decides whether its forward asks for the exit PDF
            loop_weighting="uniform",
            total_ut_steps=total_ut_steps,
        )
        self.settings = settings
        if not self._model_supports_per_loop_log_probs():
            raise RuntimeError("PLC needs the per-loop model (rltt_baseline.prepare_model output "
                               "with return_per_loop_log_probs)")
        self._old_extras: Optional[Dict[str, torch.Tensor]] = None

    # ---------- forward ----------

    def _forward_micro_batch_plc(self, micro_batch, temperature: float
                                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Final-loop entropy ``[B, L]`` and per-loop log-probs ``[B, L, T]`` of the response
        tokens."""
        response_length = micro_batch["responses"].size(-1)
        self._captured = None
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)
            labels = torch.nn.functional.pad(input_ids[:, 1:], (0, 1), value=0)
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_per_loop_log_probs=True,
                return_exit_pdf=False,
                log_prob_labels=labels,
                log_prob_temperature=temperature,
                use_cache=False,
            )
            window = slice(-response_length - 1, -1)
            ell = torch.stack([lp[:, window] for lp in output.per_loop_log_probs], dim=-1).float()
            entropy = output.per_loop_entropies[-1][:, window]
            del output
        if torch.is_grad_enabled() and self._captured is not None:
            self._pending.append(self._loop_metrics(micro_batch))
        self._captured = None
        return entropy, ell

    # ---------- old policy ----------

    def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False):
        """verl's ``compute_log_prob`` contract (final loop), plus the per-loop log-probs."""
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

        entropies, ells = [], []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, ell = self._forward_micro_batch_plc(inputs, temperature)
            entropies.append(entropy)
            ells.append(ell)

        entropys = torch.cat(entropies, dim=0)
        ell = torch.cat(ells, dim=0)
        if use_dynamic_bsz:
            entropys = restore_dynamic_batch(entropys, batch_idx_list)
            ell = restore_dynamic_batch(ell, batch_idx_list)

        self._old_extras = {OLD_LOOPS_KEY: ell}
        return ell[..., -1].contiguous(), (entropys if calculate_entropy else None)

    def pop_old_extras(self) -> Optional[Dict[str, torch.Tensor]]:
        extras, self._old_extras = self._old_extras, None
        return extras

    # ---------- update ----------

    def _clip_settings(self) -> dict:
        cfg = self.config
        return {
            "clip_ratio": float(cfg.clip_ratio),
            "clip_ratio_low": cfg.get("clip_ratio_low", None),
            "clip_ratio_high": cfg.get("clip_ratio_high", None),
            "clip_ratio_c": float(cfg.get("clip_ratio_c", 3.0)),
        }

    def _loop_weights(self, ell_old: torch.Tensor
                      ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Earlier-loop weights ``[B, L, T]`` and the trust-region weights ``[B, L, T]`` (alpha on
        t < T; None when ``kl_coef = 0``), from the old tensors."""
        settings = self.settings
        weights = plc.compute_loop_weights(ell_old, settings.weight_mode, settings.trunc_cap)
        kl_weights = (plc.compute_loop_weights(ell_old, "early_prior")
                      if settings.kl_coef > 0 else None)
        return weights, kl_weights

    def update_policy(self, data: DataProto) -> Dict[str, object]:
        settings = self.settings
        self._pending = []
        self.actor_module.train()
        temperature = data.meta_info["temperature"]
        loss_agg_mode = self.config.loss_agg_mode
        clip = self._clip_settings()

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids",
                       "advantages", "old_log_probs", OLD_LOOPS_KEY]
        missing = [k for k in select_keys if k not in data.batch.keys()]
        if missing:
            raise KeyError(f"PLC update_policy: batch lacks {missing}; the PLC worker's "
                           "compute_log_prob must run before update_policy")
        use_ref_log_prob = "ref_log_prob" in data.batch.keys()
        if use_ref_log_prob:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        metrics: Dict[str, object] = {}
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
                    ell_old = micro[OLD_LOOPS_KEY].float()
                    weights, kl_weights = self._loop_weights(ell_old)

                    entropy, ell_new = self._forward_micro_batch_plc(micro, temperature)

                    pg_loss, step_metrics = plc.plc_policy_loss(
                        ell_new, ell_old, advantages, response_mask, weights, settings.lam,
                        loss_agg_mode=loss_agg_mode, kl_coef=settings.kl_coef,
                        kl_weights=kl_weights, **clip)
                    policy_loss = pg_loss

                    entropy_loss = plc.agg_loss(entropy, response_mask, loss_agg_mode)
                    entropy_coeff = float(self.config.entropy_coeff)
                    if entropy_coeff != 0:
                        policy_loss = policy_loss - entropy_loss * entropy_coeff

                    # KL to the frozen reference on the final loop only (RLTT Eq. 7)
                    ref_key = "ref_log_prob" if use_ref_log_prob else "old_log_probs"
                    ref_log_prob = micro[ref_key]
                    kld = plc.low_var_kl(ell_new[..., -1], ref_log_prob.float())
                    kl_loss = plc.agg_loss(kld, response_mask, loss_agg_mode)
                    kl_coef = float(self.config.kl_loss_coef)
                    policy_loss = policy_loss + kl_loss * kl_coef

                    if self.config.use_dynamic_bsz:
                        scale = response_mask.shape[0] / self.config.ppo_mini_batch_size
                        loss = policy_loss * scale
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    step_metrics.update({
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/entropy_loss": entropy_loss.detach().item(),
                        "actor/kl_loss": kl_loss.detach().item(),
                        "actor/kl_coef": kl_coef,
                        "actor/total_loss": policy_loss.detach().item(),
                    })
                    if settings.log_weight_stats:
                        step_metrics.update(plc.weight_stats(weights, ell_old, response_mask))
                    step_metrics.update(plc.cross_loop_stats(
                        ell_old, advantages, response_mask, settings.trunc_cap))
                    # one scalar key per loop: verl's reduce_metrics averages each key's list
                    append_to_dict(metrics, plc.flatten_loop_metrics(step_metrics))

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        for record in self._pending:
            append_to_dict(metrics, record)
        self._pending = []
        return metrics


@ray.remote
class PLCActorRolloutRefWorker(ActorRolloutRefWorker):
    """verl worker that builds ``PLCActor`` and adds its per-loop old tensors to the batch."""

    def __init__(self, config, role: str, **kwargs):
        raw = config.actor.get("plc", None)
        if raw is None:
            raise ValueError("actor_rollout_ref.actor.plc is missing")
        settings = plc.PLCSettings.from_mapping(OmegaConf.to_container(raw, resolve=True))
        # verl 0.6.1 materializes actor as FSDPActorConfig, which has no PLC field.
        # Keep PLC settings on this worker and pass an isolated native config onward.
        native_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        OmegaConf.set_struct(native_config.actor, False)
        native_config.actor.pop("plc")
        super().__init__(native_config, role, **kwargs)
        self.settings = settings
        self.total_ut_steps = config.get("total_ut_steps", 4)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if self._is_actor:
            self.actor = PLCActor(
                # Reuse verl's typed config, including its native actor defaults.
                config=self.actor.config,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
                settings=self.settings,
                total_ut_steps=self.total_ut_steps,
            )
            attach_loop_audit(self.actor, self.config.get("loop_audit"), self.rank)
            if self.rank == 0:
                print(f"[rltt_baseline] PLCActor {self.settings.describe()} pid={os.getpid()} "
                      f"python={sys.executable}", flush=True)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_log_prob(self, data: DataProto):
        output = super().compute_log_prob(data)
        extras = self.actor.pop_old_extras() if self._is_actor else None
        if extras:
            output = output.union(DataProto.from_dict(
                tensors={k: v.to("cpu") for k, v in extras.items()}))
        return output
