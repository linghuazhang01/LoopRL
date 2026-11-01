"""RLTT actor and worker that additionally log per-loop actor metrics.

Added metrics (per micro-batch during ``update_policy``, averaged by verl):
- ``rltt/loop_<t>_entropy``: mean entropy of loop t's next-token distribution over response
  tokens (the vendor actor logs only the final loop's entropy, as ``actor/entropy``);
- ``rltt/expected_exit_step``: mean over response tokens of sum_t t * exit_pdf_t, when
  ``loop_weighting=exit_pdf``;
- ``actor/total_loss``: pg_loss + kl_coef * kl_loss - entropy_coeff * entropy_loss, i.e. the
  loss the vendor actor backpropagates (before gradient-accumulation scaling).

The vendor actor computes per-loop entropies but discards all but the last; they are read
here with a forward hook on the actor module, so the vendor loss code runs unchanged.
Requires verl 0.6.1 and ray; import only inside the training environment.
"""

import os
import sys

import ray
import torch
import verl_rltt.rltt_fsdp_workers  # noqa: F401  (applies the vendor target_modules patch)
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.py_functional import append_to_dict
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl_rltt.rltt_actor import RLTTDataParallelPPOActor

from rltt_baseline.loop_audit import attach_loop_audit


class MetricsRLTTActor(RLTTDataParallelPPOActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._captured = None
        self._pending = []
        self.actor_module.register_forward_hook(self._capture_output)

    def _capture_output(self, module, inputs, output):
        entropies = getattr(output, "per_loop_entropies", None)
        if entropies is None:
            return
        exit_pdf = getattr(output, "exit_pdf", None)
        self._captured = (
            [e.detach() for e in entropies],
            exit_pdf.detach() if exit_pdf is not None else None,
        )

    def _forward_micro_batch_rltt(self, micro_batch, temperature):
        self._captured = None
        result = super()._forward_micro_batch_rltt(micro_batch, temperature)
        if torch.is_grad_enabled() and self._captured is not None:
            self._pending.append(self._loop_metrics(micro_batch))
        self._captured = None
        return result

    def _loop_metrics(self, micro_batch):
        entropies, exit_pdf = self._captured
        response_length = micro_batch["responses"].size(-1)
        mask = micro_batch["attention_mask"][:, -response_length:].float()
        denom = mask.sum().clamp_min(1.0)
        record = {}
        for t, ent in enumerate(entropies):
            ent = ent[:, -response_length - 1:-1].float()
            record[f"rltt/loop_{t + 1}_entropy"] = ((ent * mask).sum() / denom).item()
        if exit_pdf is not None:
            pdf = exit_pdf[:, -response_length - 1:-1, :].float()
            steps = torch.arange(1, pdf.shape[-1] + 1, device=pdf.device, dtype=pdf.dtype)
            expected = (pdf * steps).sum(-1)
            record["rltt/expected_exit_step"] = ((expected * mask).sum() / denom).item()
        return record

    def update_policy(self, data):
        self._pending = []
        metrics = super().update_policy(data)
        for record in self._pending:
            append_to_dict(metrics, record)
        self._pending = []
        pg = metrics.get("actor/pg_loss", [])
        kl = metrics.get("actor/kl_loss", [])
        kl_coef = metrics.get("actor/kl_coef", [])
        ent = metrics.get("actor/entropy_loss", [])
        if pg and len(pg) == len(kl) == len(kl_coef) == len(ent):
            coeff = float(self.config.entropy_coeff)
            totals = [p + c * k - coeff * e for p, k, c, e in zip(pg, kl, kl_coef, ent)]
            append_to_dict(metrics, {"actor/total_loss": totals})
        return metrics


@ray.remote
class MetricsRLTTActorRolloutRefWorker(ActorRolloutRefWorker):
    """Same as the vendor ``RLTTActorRolloutRefWorker`` but builds ``MetricsRLTTActor``."""

    def __init__(self, config, role: str, **kwargs):
        super().__init__(config, role, **kwargs)
        self.loop_weighting = config.get("loop_weighting", "uniform")
        self.progressive_alpha = config.get("progressive_alpha", 1.0)
        self.total_ut_steps = config.get("total_ut_steps", 4)
        self.learned_weights_init = config.get("learned_weights_init", "uniform")
        self.learned_weights_temp = config.get("learned_weights_temp", 1.0)
        self.learned_weights_lr = config.get("learned_weights_lr", None)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if self._is_actor:
            self.actor = MetricsRLTTActor(
                config=self.config.actor,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
                loop_weighting=self.loop_weighting,
                progressive_alpha=self.progressive_alpha,
                total_ut_steps=self.total_ut_steps,
                learned_weights_init=self.learned_weights_init,
                learned_weights_temp=self.learned_weights_temp,
                learned_weights_lr=self.learned_weights_lr,
            )
            attach_loop_audit(self.actor, self.config.get("loop_audit"), self.rank)
            if self.rank == 0:
                print(f"[rltt_baseline] MetricsRLTTActor loop_weighting={self.loop_weighting} "
                      f"pid={os.getpid()} python={sys.executable}", flush=True)
