"""OPD: on-policy distillation of Ouro's last loop towards the frozen base model's LoopCD policy.

Pure torch, no verl dependency (``tests/test_opd.py``). ``rltt_baseline.verl09_engine`` computes
the teacher's log-probs with a frozen copy of the initial model, ``rltt_baseline.verl09_losses``
applies the loss. Only the verl 0.9.1 engine path implements it (the launcher refuses a verl 0.6.1
training run). Design: ``research/opd_loopcd_design_2026-11-22.md`` (not tracked).

Teacher. With l_t = log P^(t)_base of the frozen initial model (temperature 1) and the
extrapolation strength w (``omega``):

    log pi_T = log_softmax( ((1 + w) l_4 - w l_r) / temperature ),  r = ``ref_loop`` (1)

i.e. "distribution 4 + w (distribution 4 - distribution 1)" in log space, which is LoopCD-Logits
(Liu et al., arXiv 2610.02185, ``rltt_baseline.loop_cd``) of the base model. The teacher is fixed
for the whole run (it is never the current model), so distillation has a fixed point instead of
re-extrapolating the student every step.

Student and rollouts. The student is the trained model's last loop P^(4) (the loop evaluated with
w_eval = 0); vLLM samples the rollouts from it, as in GRPO / RLTT / PLC (on-policy).

Two estimators of the per-position divergence (``estimator``):

- ``sampled`` (reverse KL only): on every response token y_j,

      A_j = sg[ log pi_T(y_j) - log P^(4)_old(y_j) ]            (minus the k1 estimate of KL)
      loss = -agg_j[ PPO_j(A_j) ],  ratio = P^(4)(y_j) / P^(4)_old(y_j)   (1: one update per step)

  With y ~ P^(4) the expected gradient of the per-token term is the gradient of
  KL(P^(4)(.|s) || pi_T(.|s)) at every visited prefix s (E_{y ~ p}[grad log p(y)] = 0). The
  teacher's sampled-token log-probs come from a frozen engine in the ``compute_log_prob`` pass.

- ``full``: the exact divergence over the whole vocabulary at every visited prefix,

      reverse: KL(P^(4)(.|s) || pi_T(.|s)),   forward: KL(pi_T(.|s) || P^(4)(.|s)),
      loss = agg_j[ KL_j ]

  computed inside the student's training forward (``rltt_per_loop.opd_full_kd``) from the hidden
  states of a frozen teacher model that every actor rank holds (``verl09_workers``).

Both use the token's own term only (no reward-to-go over later tokens) on the student's own
trajectories. The verifiable reward is not part of the objective: the driver still scores the
rollouts (``reasoning/*`` metrics) and computes GRPO advantages, which this loss ignores.
"""

from dataclasses import asdict, dataclass, fields
from typing import Dict, Mapping, Optional, Tuple

import torch

from rltt_baseline import plc

ESTIMATORS = ("sampled", "full")
DIVERGENCES = ("reverse", "forward")


@dataclass(frozen=True)
class OPDSettings:
    """The ``opd`` config section; ``from_mapping`` validates a dict of user values."""

    enabled: bool = False
    omega: float = 0.5   # w: teacher log pi_T = (1 + w) l_4 - w l_r of the base; 0 = base P^(4)
    ref_loop: int = 1    # reference loop r of the teacher (1-based); the source is the last loop
    estimator: str = "sampled"   # sampled-token k1 (reverse only) | full-vocabulary KL
    divergence: str = "reverse"  # reverse KL(P4 || pi_T) | forward KL(pi_T || P4) (full only)

    @classmethod
    def from_mapping(cls, values: Optional[Mapping]) -> "OPDSettings":
        values = dict(values or {})
        names = {f.name for f in fields(cls)}
        unknown = set(values) - names
        if unknown:
            raise KeyError(f"unknown opd keys {sorted(unknown)}; allowed {sorted(names)}")
        merged = {**asdict(cls()), **values}
        if not isinstance(merged["enabled"], bool):
            raise TypeError(f"opd.enabled must be a bool, got {merged['enabled']!r}")
        for name in ("estimator", "divergence"):
            if not isinstance(merged[name], str):
                raise TypeError(f"opd.{name} must be a string, got {merged[name]!r}")
        omega = merged["omega"]
        if isinstance(omega, bool) or not isinstance(omega, (int, float)):
            raise TypeError(f"opd.omega must be a number, got {omega!r}")
        merged["omega"] = float(omega)
        ref = merged["ref_loop"]
        if isinstance(ref, bool) or not isinstance(ref, int):
            raise TypeError(f"opd.ref_loop must be an integer, got {ref!r}")
        settings = cls(**merged)
        settings.validate()
        return settings

    def validate(self, total_loops: Optional[int] = None) -> None:
        if self.omega < 0:
            raise ValueError("opd.omega must be >= 0 (negative values interpolate towards the "
                             "reference loop)")
        if self.ref_loop < 1:
            raise ValueError("opd.ref_loop must be >= 1")
        if total_loops is not None and self.ref_loop >= total_loops:
            raise ValueError(f"opd.ref_loop must be < the {total_loops} loops of the model")
        if self.estimator not in ESTIMATORS:
            raise ValueError(f"opd.estimator must be one of {ESTIMATORS}, got {self.estimator!r}")
        if self.divergence not in DIVERGENCES:
            raise ValueError(f"opd.divergence must be one of {DIVERGENCES}, got "
                             f"{self.divergence!r}")
        if self.estimator == "sampled" and self.divergence != "reverse":
            raise ValueError("opd.estimator=sampled estimates the reverse KL only (the student "
                             "samples the tokens); use opd.estimator=full for the forward KL")

    @property
    def full_vocab(self) -> bool:
        return self.estimator == "full"

    def student_model_kwargs(self) -> dict:
        """Keyword arguments of the student's per-loop forward for the full-vocabulary loss; the
        caller adds ``opd_teacher_hidden_states``, ``opd_teacher_lm_head`` and ``opd_window``."""
        return {"return_opd_kd": True, "opd_omega": self.omega, "opd_divergence": self.divergence}

    def teacher_model_kwargs(self) -> dict:
        """Keyword arguments of the per-loop forward (``modeling_ouro.py``) that give the
        teacher's log-probs on the sampled tokens: LoopCD-Logits with a fixed strength, no clip.
        The teacher runs without gradient, so ``extrap_ref_grad`` is irrelevant (kept False)."""
        return {"return_extrap_log_probs": True, "extrap_omega": self.omega,
                "extrap_ref_loop": self.ref_loop, "extrap_adaptive": False,
                "extrap_delta_cap": None, "extrap_ref_grad": False}

    def describe(self) -> str:
        return (f"omega={self.omega} ref_loop={self.ref_loop} estimator={self.estimator} "
                f"divergence={self.divergence} (teacher: frozen initial model)")


def opd_full_loss(kd: torch.Tensor, response_mask: torch.Tensor, loss_agg_mode: str = "token-mean",
                  row_div: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, object]]:
    """``agg[KL]`` of the per-position full-vocabulary divergence ``kd`` ``[B, L]`` (with grad)."""
    loss = plc.agg_term(kd, response_mask, loss_agg_mode, row_div)
    value = loss.detach().item()
    return loss, {"loss_term/kd": value, "loss_term/pg": 0.0}


@torch.no_grad()
def opd_full_stats(kd: torch.Tensor, kl_other: torch.Tensor, teacher_entropy: torch.Tensor,
                   top1_agree: torch.Tensor, settings: "OPDSettings",
                   response_mask: torch.Tensor) -> Dict[str, float]:
    """Token means over the response of both KL directions, the teacher's entropy and the
    student / teacher top-1 agreement (full-vocabulary estimator)."""
    mask = response_mask.float()
    denom = mask.sum().clamp_min(1.0)

    def mean(x):
        return float((x.float() * mask).sum() / denom)

    reverse, forward = ((kd, kl_other) if settings.divergence == "reverse" else (kl_other, kd))
    return {
        "opd/reverse_kl": mean(reverse),
        "opd/forward_kl": mean(forward),
        "opd/teacher_entropy": mean(teacher_entropy),
        "opd/top1_agree": mean(top1_agree),
    }


def opd_advantages(teacher_log_prob: torch.Tensor, old_log_prob: torch.Tensor) -> torch.Tensor:
    """Per-token advantage ``log pi_T(y) - log P^(4)_old(y)`` (no gradient)."""
    return (teacher_log_prob.detach().float() - old_log_prob.detach().float())


def opd_policy_loss(log_prob: torch.Tensor, old_log_prob: torch.Tensor,
                    teacher_log_prob: torch.Tensor, response_mask: torch.Tensor,
                    clip_ratio: float, loss_agg_mode: str = "token-mean",
                    clip_ratio_low: Optional[float] = None,
                    clip_ratio_high: Optional[float] = None, clip_ratio_c: float = 3.0,
                    row_div: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, Dict[str, object]]:
    """``-agg[PPO(A)]`` on the last loop's ``[B, L]`` log-probs with the token-level advantage
    ``A = log pi_T - log P^(4)_old`` (``plc.grpo_policy_loss`` with a per-token advantage)."""
    if teacher_log_prob.shape != log_prob.shape:
        raise ValueError(f"teacher log-probs {tuple(teacher_log_prob.shape)} do not match the "
                         f"student's {tuple(log_prob.shape)}")
    adv = opd_advantages(teacher_log_prob, old_log_prob).to(log_prob.dtype)
    return plc.grpo_policy_loss(log_prob, old_log_prob, adv, response_mask, clip_ratio,
                                loss_agg_mode=loss_agg_mode, clip_ratio_low=clip_ratio_low,
                                clip_ratio_high=clip_ratio_high, clip_ratio_c=clip_ratio_c,
                                row_div=row_div)


@torch.no_grad()
def opd_stats(teacher_log_prob: torch.Tensor, old_log_prob: torch.Tensor,
              response_mask: torch.Tensor) -> Dict[str, float]:
    """Token means over the response of the sampled tokens' teacher and student log-probs and two
    estimates of KL(P^(4) || pi_T): k1 = log p - log t (unbiased, can be negative) and
    k3 = t/p - 1 - log(t/p) (non-negative)."""
    mask = response_mask.float()
    denom = mask.sum().clamp_min(1.0)
    t, s = teacher_log_prob.float(), old_log_prob.float()
    log_ratio = t - s

    def mean(x):
        return float((x * mask).sum() / denom)

    return {
        "opd/teacher_log_prob": mean(t),
        "opd/student_log_prob": mean(s),
        "opd/reverse_kl_k1": mean(-log_ratio),
        "opd/reverse_kl_k3": mean(torch.exp(log_ratio.clamp(max=50.0)) - 1.0 - log_ratio),
        "opd/teacher_prefers_frac": mean((log_ratio > 0).float()),
    }
