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

With ``plc.init_kl_coef > 0`` the worker also loads a frozen copy of the initial model (the same
``actor_rollout_ref.model.path`` the actor starts from, built by verl's own reference-model code:
fp32 master weights, the actor's FSDP mixed precision, parameters offloaded to CPU). Each
``compute_log_prob`` runs it on the same micro-batches, without gradient, after the actor's own
pass, and adds its per-loop log-probs ``init_log_probs_loops`` ``[B, L, T]`` to the batch;
``update_policy`` pulls the earlier loops back towards them (``plc.init_anchor_penalty``). The
frozen model is never loaded from a checkpoint, so a resumed run keeps the same anchor. Its forward
goes through ``_per_loop_forward`` directly, not through ``_forward_micro_batch_plc``, so neither
the loop audit nor the per-loop entropy metrics see it.

With ``plc.loop_loss: kd`` the update forward also asks the model for the earlier loops'
full-vocabulary distillation losses towards the stop-gradient final loop (``return_loop_kd``, on
the response positions only) and backpropagates ``plc.plc_kd_loss`` instead: GRPO on the final
loop plus ``lam * alpha_t`` times those losses; the KD statistics go to ``plc/kd_*/loop_t``. The
old-policy pass is unchanged.

With a row-normalized ``loss_agg.mode`` (``actor_rollout_ref.loss_agg``, from the config
section ``loss_agg``) every micro-batch computes per-response divisors ``d_i`` from its own
``response_mask`` (``rltt_baseline.loss_agg.row_divisors``: ``norm_len`` for drgrpo, the response
length for seq-mean-token-mean) and every loss term aggregates as ``(1 / B) sum_i S_i / d_i``
instead of the micro-batch token mean. With the default ``token-mean``
nothing changes. The logged ``actor/entropy_loss`` and ``actor/kl_loss`` stay token means.

Per-token weight statistics are logged under ``plc/``, one scalar key per loop
(``plc/<name>/loop_<t>``, ``plc.flatten_loop_metrics``): verl's ``reduce_metrics`` takes the mean
of each key's per-micro-batch list, which would also average a per-loop list over its loops.
"""

import inspect
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
from verl.utils.fsdp_utils import fsdp_version
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import (
    prepare_dynamic_batch,
    rearrange_micro_batches,
    restore_dynamic_batch,
)
from verl.workers.fsdp_workers import ActorRolloutRefWorker

from rltt_baseline import loss_agg, plc
from rltt_baseline.loop_audit import attach_loop_audit
from rltt_baseline.workers import MetricsRLTTActor

OLD_LOOPS_KEY = "old_log_probs_loops"
INIT_LOOPS_KEY = "init_log_probs_loops"


class PLCActor(MetricsRLTTActor):
    def __init__(self, config, actor_module, actor_optimizer=None, settings=None,
                 total_ut_steps: int = 4, init_module=None,
                 loss_agg_settings: Optional[loss_agg.LossAggSettings] = None):
        if settings is None:
            raise ValueError("PLCActor needs PLCSettings")
        super().__init__(
            config, actor_module, actor_optimizer,
            # the vendor flag only decides whether its forward asks for the exit PDF
            loop_weighting="uniform",
            total_ut_steps=total_ut_steps,
        )
        self.settings = settings
        self.loss_agg = loss_agg_settings or loss_agg.LossAggSettings()
        if settings.uses_init_model and init_module is None:
            raise ValueError("plc.init_kl_coef > 0 needs the frozen initial model (init_module)")
        self.init_module = init_module if settings.uses_init_model else None
        if not self._model_supports_per_loop_log_probs():
            raise RuntimeError("PLC needs the per-loop model (rltt_baseline.prepare_model output "
                               "with return_per_loop_log_probs)")
        if settings.loop_loss == "kd" and not self._model_forward_accepts("return_loop_kd"):
            raise RuntimeError("plc.loop_loss=kd needs a model directory whose modeling_ouro.py "
                               "has return_loop_kd: rebuild LOOPRL_MODEL_DIR with "
                               "rltt_baseline.prepare_model")
        self._old_extras: Optional[Dict[str, torch.Tensor]] = None

    # ---------- forward ----------

    def _model_forward_accepts(self, name: str) -> bool:
        """Whether the unwrapped HF model's ``forward`` takes the keyword ``name``."""
        model = self.actor_module
        for _ in range(10):
            inner = None
            for attr in ("_fsdp_wrapped_module", "_orig_mod"):
                if hasattr(model, attr):
                    inner = getattr(model, attr)
                    break
            if inner is None:
                break
            model = inner
        try:
            return name in inspect.signature(model.forward).parameters
        except (TypeError, ValueError):
            return False

    def _forward_micro_batch_plc(self, micro_batch, temperature: float, with_kd: bool = False):
        """Final-loop entropy ``[B, L]`` and per-loop log-probs ``[B, L, T]`` of the response
        tokens, from the actor. With ``with_kd`` (``loop_loss: kd`` updates) also the earlier
        loops' distillation losses ``[B, L, T - 1]`` and their ``KD_STATS``."""
        self._captured = None
        divergence = self.settings.kd_divergence if with_kd else None
        entropy, ell, kd, kd_stats = self._per_loop_forward(self.actor_module, micro_batch,
                                                            temperature, divergence)
        if torch.is_grad_enabled() and self._captured is not None:
            self._pending.append(self._loop_metrics(micro_batch))
        self._captured = None
        if with_kd:
            return entropy, ell, kd, kd_stats
        return entropy, ell

    @staticmethod
    def _per_loop_forward(module, micro_batch, temperature: float,
                          kd_divergence: Optional[str] = None):
        """The padded per-loop forward of ``module`` (the actor, or the frozen initial model):
        ``(entropy, ell, kd, kd_stats)``; ``kd`` / ``kd_stats`` are None unless
        ``kd_divergence`` is given."""
        response_length = micro_batch["responses"].size(-1)
        kd_kwargs = {}
        if kd_divergence is not None:
            kd_kwargs = {"return_loop_kd": True, "loop_kd_divergence": kd_divergence,
                         "loop_kd_window": response_length}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)
            labels = torch.nn.functional.pad(input_ids[:, 1:], (0, 1), value=0)
            output = module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_per_loop_log_probs=True,
                return_exit_pdf=False,
                log_prob_labels=labels,
                log_prob_temperature=temperature,
                use_cache=False,
                **kd_kwargs,
            )
            window = slice(-response_length - 1, -1)
            ell = torch.stack([lp[:, window] for lp in output.per_loop_log_probs], dim=-1).float()
            entropy = output.per_loop_entropies[-1][:, window]
            kd = kd_stats = None
            if kd_divergence is not None:
                kd, kd_stats = output.loop_kd, output.loop_kd_stats
            del output
        return entropy, ell, kd, kd_stats

    # ---------- old policy ----------

    def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False):
        """verl's ``compute_log_prob`` contract (final loop), plus the per-loop log-probs (and
        those of the frozen initial model when the anchor is on)."""
        self.actor_module.eval()
        if self.init_module is not None:
            self.init_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        data = data.select(batch_keys=["responses", "input_ids", "attention_mask", "position_ids"])
        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        entropies, ells, init_ells = [], [], []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, ell = self._forward_micro_batch_plc(inputs, temperature)
                if self.init_module is not None:
                    # same micro-batches on every rank, in the same order: FSDP collectives match
                    _, init_ell, _, _ = self._per_loop_forward(self.init_module, inputs,
                                                               temperature)
                    init_ells.append(init_ell)
            entropies.append(entropy)
            ells.append(ell)

        entropys = torch.cat(entropies, dim=0)
        ell = torch.cat(ells, dim=0)
        init_ell = torch.cat(init_ells, dim=0) if init_ells else None
        if use_dynamic_bsz:
            entropys = restore_dynamic_batch(entropys, batch_idx_list)
            ell = restore_dynamic_batch(ell, batch_idx_list)
            if init_ell is not None:
                init_ell = restore_dynamic_batch(init_ell, batch_idx_list)

        self._old_extras = {OLD_LOOPS_KEY: ell}
        if init_ell is not None:
            self._old_extras[INIT_LOOPS_KEY] = init_ell
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
                      ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Earlier-loop weights ``[B, L, T]``, and the trust-region and anchor weights
        ``[B, L, T]`` (alpha on t < T, 0 on T; None when ``kl_coef`` / ``init_kl_coef`` is 0),
        from the old tensors."""
        settings = self.settings
        weights = plc.compute_loop_weights(ell_old, settings.weight_mode, settings.trunc_cap)
        alpha = (plc.compute_loop_weights(ell_old, "early_prior")
                 if settings.kl_coef > 0 or settings.init_kl_coef > 0 else None)
        kl_weights = alpha if settings.kl_coef > 0 else None
        init_weights = alpha if settings.init_kl_coef > 0 else None
        return weights, kl_weights, init_weights

    def update_policy(self, data: DataProto) -> Dict[str, object]:
        settings = self.settings
        self._pending = []
        self.actor_module.train()
        temperature = data.meta_info["temperature"]
        loss_agg_mode = self.config.loss_agg_mode
        clip = self._clip_settings()

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids",
                       "advantages", "old_log_probs", OLD_LOOPS_KEY]
        if self.init_module is not None:
            select_keys.append(INIT_LOOPS_KEY)
        missing = [k for k in select_keys if k not in data.batch.keys()]
        if missing:
            raise KeyError(f"PLC update_policy: batch lacks {missing}; the PLC worker's "
                           "compute_log_prob must run before update_policy")
        use_ref_log_prob = "ref_log_prob" in data.batch.keys()
        if use_ref_log_prob:
            select_keys.append("ref_log_prob")
        row_normalized = self.loss_agg.row_normalized
        batch = data.select(batch_keys=select_keys).batch
        metrics: Dict[str, object] = {}
        tracker = loss_agg.LossTracker()  # loss/<term>: the exact objective of each update
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
                    weights, kl_weights, init_weights = self._loop_weights(ell_old)
                    ell_init = (micro[INIT_LOOPS_KEY].float()
                                if self.init_module is not None else None)
                    # per-response divisors d_i (a function of each row alone)
                    row_div = (loss_agg.row_divisors(self.loss_agg, response_mask)
                               if row_normalized else None)

                    kd_stats = None
                    if settings.loop_loss == "kd":
                        entropy, ell_new, kd, kd_stats = self._forward_micro_batch_plc(
                            micro, temperature, with_kd=True)
                        pg_loss, step_metrics = plc.plc_kd_loss(
                            ell_new, ell_old, advantages, response_mask, kd, settings.lam,
                            loss_agg_mode=loss_agg_mode, init_kl_coef=settings.init_kl_coef,
                            logp_init=ell_init, init_kl_form=settings.init_kl_form,
                            init_kl_log_ratio_cap=settings.init_kl_log_ratio_cap,
                            row_div=row_div, **clip)
                        del kd
                    else:
                        entropy, ell_new = self._forward_micro_batch_plc(micro, temperature)
                        pg_loss, step_metrics = plc.plc_policy_loss(
                            ell_new, ell_old, advantages, response_mask, weights, settings.lam,
                            loss_agg_mode=loss_agg_mode, kl_coef=settings.kl_coef,
                            kl_weights=kl_weights, init_kl_coef=settings.init_kl_coef,
                            logp_init=ell_init, init_weights=init_weights,
                            init_kl_form=settings.init_kl_form,
                            init_kl_log_ratio_cap=settings.init_kl_log_ratio_cap,
                            row_div=row_div, **clip)
                    policy_loss = pg_loss
                    terms = loss_agg.pop_loss_terms(step_metrics)

                    # entropy_loss / kl_loss are logged as token means; the loss terms use the
                    # configured aggregation (agg_term = agg_loss when row_div is None)
                    entropy_loss = plc.agg_loss(entropy, response_mask, loss_agg_mode)
                    entropy_coeff = float(self.config.entropy_coeff)
                    if entropy_coeff != 0:
                        entropy_term = entropy_coeff * plc.agg_term(
                            entropy, response_mask, loss_agg_mode, row_div)
                        policy_loss = policy_loss - entropy_term
                        terms["entropy"] = -float(entropy_term.detach())

                    # KL to the frozen reference on the final loop only (RLTT Eq. 7)
                    ref_key = "ref_log_prob" if use_ref_log_prob else "old_log_probs"
                    ref_log_prob = micro[ref_key]
                    kld = plc.low_var_kl(ell_new[..., -1], ref_log_prob.float())
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
                    if settings.log_weight_stats:
                        step_metrics.update(plc.weight_stats(weights, ell_old, response_mask))
                    step_metrics.update(plc.cross_loop_stats(
                        ell_old, advantages, response_mask, settings.trunc_cap))
                    if kd_stats is not None:
                        step_metrics.update(plc.kd_stats(kd_stats, response_mask))
                    # one scalar key per loop: verl's reduce_metrics averages each key's list
                    append_to_dict(metrics, plc.flatten_loop_metrics(step_metrics))

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item(),
                                         **tracker.flush()})

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
        # actor_rollout_ref.loss_agg (launcher, only when loss_agg.mode is row-normalized)
        raw_agg = config.get("loss_agg", None)
        self.loss_agg_settings = loss_agg.LossAggSettings.from_mapping(
            OmegaConf.to_container(raw_agg, resolve=True) if raw_agg is not None else None)

    def _build_init_module(self):
        """Frozen copy of the initial model, built like verl 0.6.1's reference model.

        ``_build_model_optimizer(role="ref")`` from the actor's own model path, override config
        (``early_exit_threshold: null`` included), padding / kernel switches and FSDP config: fp32
        master weights as the actor's (this worker is an actor), the same bf16 mixed precision,
        and CPU-offloaded parameters (forced for ``role="ref"``). No optimizer, never
        checkpointed or reloaded.
        """
        from verl.utils.config import omega_conf_to_dataclass
        from verl.utils.fs import copy_to_local

        model_cfg = self.config.model
        override_model_config = OmegaConf.to_container(
            OmegaConf.create(model_cfg.get("override_config", {})))
        local_path = copy_to_local(model_cfg.path, use_shm=model_cfg.get("use_shm", False))
        module = self._build_model_optimizer(
            model_path=local_path,
            fsdp_config=omega_conf_to_dataclass(self.config.actor.fsdp_config),
            optim_config=None,
            override_model_config=override_model_config,
            use_remove_padding=model_cfg.get("use_remove_padding", False),
            use_fused_kernels=model_cfg.get("use_fused_kernels", False),
            trust_remote_code=model_cfg.get("trust_remote_code", False),
            use_liger=model_cfg.get("use_liger", False),
            role="ref",
        )[0]
        module.eval()
        if self.rank == 0:
            print(f"[rltt_baseline] PLC anchor: frozen initial model from {model_cfg.path}",
                  flush=True)
        return module

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        self.init_module_fsdp = None
        if self._is_actor and self.settings.uses_init_model:
            self.init_module_fsdp = self._build_init_module()
        if self._is_actor:
            self.actor = PLCActor(
                # Reuse verl's typed config, including its native actor defaults.
                config=self.actor.config,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
                settings=self.settings,
                total_ut_steps=self.total_ut_steps,
                init_module=self.init_module_fsdp,
                loss_agg_settings=self.loss_agg_settings,
            )
            attach_loop_audit(self.actor, self.config.get("loop_audit"), self.rank)
            if self.rank == 0:
                print(f"[rltt_baseline] PLCActor {self.settings.describe()} "
                      f"{self.loss_agg_settings.describe()} pid={os.getpid()} "
                      f"python={sys.executable}", flush=True)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_log_prob(self, data: DataProto):
        output = super().compute_log_prob(data)
        init_module = getattr(self, "init_module_fsdp", None)
        if init_module is not None and self.world_size > 1:  # as verl does for its reference
            if fsdp_version(init_module) == 1:
                init_module._handle.reshard(True)
            elif fsdp_version(init_module) == 2:
                init_module.reshard()
        extras = self.actor.pop_old_extras() if self._is_actor else None
        if extras:
            output = output.union(DataProto.from_dict(
                tensors={k: v.to("cpu") for k, v in extras.items()}))
        return output
