"""OPD: on-policy distillation of Ouro's last loop towards the frozen base model's LoopCD policy.

Pure torch, no verl dependency (``tests/test_opd.py``). ``rltt_baseline.verl09_engine`` computes
the teacher's log-probs with a frozen copy of the initial model, ``rltt_baseline.verl09_losses``
applies the loss. Only the verl 0.9.1 engine path implements it (the launcher refuses a verl 0.6.1
training run). Design: ``research/opd_loopcd_design_2026-11-22.md`` (not tracked).

Teacher. With l_t = log P^(t)_base of the frozen initial model (temperature 1) and the
extrapolation strength w (``omega``):

    log pi_T = log_softmax( ((1 + w) l_4 - w l_r) / temperature ),  r = ``ref_loop`` (1)

i.e. "distribution 4 + w (distribution 4 - distribution 1)" in log space, which is LoopCD-Logits
(Liu et al., arXiv 2610.02185, ``rltt_baseline.loop_cd``) of the base model. With the default
``teacher: frozen`` the teacher is fixed for the whole run (it is never the current model), so
distillation has a fixed point instead of re-extrapolating the student every step.

``teacher: online`` (ELPO, user 2026-11-26) instead computes the same formula from the CURRENT
model, with a stop-gradient:

    log pi_T = sg[ log_softmax( ((1 + w) log P^(4)_theta - w log P^(r)_theta) / temperature ) ]

No teacher copy is loaded. ``full`` reads it from the loop-r and last-loop hidden states of the
student's own training forward; ``sampled`` / ``k2`` take its sampled-token log-probs from the
actor's own ``compute_log_prob`` pass (``teacher_log_probs`` at the rollout parameters, no extra
forward). With one optimiser update per step both are the rollout policy's LoopCD distribution,
refreshed every step. The fixed point of distillation then moves with the student (the idealised
argument of the ELPD draft predicts the contrast log P^(4) - log P^(r) to keep growing); this is
the "refreshed teacher" ablation of the frozen teacher.

Student and rollouts. The student is the trained model's last loop P^(4) (the loop evaluated with
w_eval = 0); vLLM samples the rollouts from it, as in GRPO / RLTT / PLC (on-policy).

Three estimators of the per-position divergence (``estimator``):

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

- ``k2`` (reverse KL only; KDRL, Xu et al., arXiv 2506.02208, Eq. 6): on every response token,

      R_j = log pi_T(y_j) - log P^(4)(y_j),   loss = agg_j[ R_j^2 / 2 ]

  The gradient flows through the student's log-prob only: grad = -R_j grad log P^(4)(y_j), the
  REINFORCE step with the per-token reward R_j, whose expectation over y ~ P^(4) is the gradient
  of KL(P^(4)(.|s) || pi_T(.|s)) (the value of R^2 / 2 is a biased KL estimate, its gradient is
  not). Same teacher engine as ``sampled``; no PPO ratio or clip (one update per step).

All three use the token's own term only (no reward-to-go over later tokens) on the student's own
trajectories.

Reward. ``rl_coef`` (default 0) adds GRPO on the batch's outcome advantages, the same element-wise
PPO term as ``trainer: grpo`` (``plc.grpo_policy_loss``), and ``kd_coef`` (beta, default 1) scales
the divergence:

    loss = rl_coef * GRPO(A_outcome) + kd_coef * D

With the defaults the verifiable reward is not part of the objective (pure OPD, the loss is
unchanged): the driver still scores the rollouts (``reasoning/*`` metrics) and computes GRPO
advantages, which the loss then ignores. ``estimator: k2, rl_coef: 1, kd_coef: beta`` is KDRL's
joint objective J_GRPO - beta * k2 with the base model's LoopCD policy as the teacher.
"""

from dataclasses import asdict, dataclass, fields
from typing import Dict, Mapping, Optional, Tuple

import torch

from rltt_baseline import plc

ESTIMATORS = ("sampled", "full", "k2")
TEACHERS = ("frozen", "online")  # frozen initial model | the current model (stop-gradient)
SAMPLED_TOKEN_ESTIMATORS = ("sampled", "k2")   # need the teacher's sampled-token log-probs
DIVERGENCES = ("reverse", "forward")


@dataclass(frozen=True)
class OPDSettings:
    """The ``opd`` config section; ``from_mapping`` validates a dict of user values."""

    enabled: bool = False
    omega: float = 0.5   # w: teacher log pi_T = (1 + w) l_4 - w l_r of the base; 0 = base P^(4)
    ref_loop: int = 1    # reference loop r of the teacher (1-based); the source is the last loop
    estimator: str = "sampled"   # sampled-token k1 PPO | full-vocabulary KL | sampled-token k2
    divergence: str = "reverse"  # reverse KL(P4 || pi_T) | forward KL(pi_T || P4) (full only)
    rl_coef: float = 0.0  # weight of GRPO on the outcome advantages (0: the reward is not used)
    kd_coef: float = 1.0  # beta: weight of the divergence term
    teacher: str = "frozen"  # frozen initial model | online: the current model (stop-gradient)

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
        for name in ("estimator", "divergence", "teacher"):
            if not isinstance(merged[name], str):
                raise TypeError(f"opd.{name} must be a string, got {merged[name]!r}")
        for name in ("omega", "rl_coef", "kd_coef"):
            value = merged[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"opd.{name} must be a number, got {value!r}")
            merged[name] = float(value)
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
        if self.estimator in SAMPLED_TOKEN_ESTIMATORS and self.divergence != "reverse":
            raise ValueError(f"opd.estimator={self.estimator} estimates the reverse KL only (the "
                             "student samples the tokens); use opd.estimator=full for the forward "
                             "KL")
        for name in ("rl_coef", "kd_coef"):
            value = getattr(self, name)
            if not value >= 0 or value == float("inf"):
                raise ValueError(f"opd.{name} must be a finite number >= 0, got {value!r}")
        if self.rl_coef == 0 and self.kd_coef == 0:
            raise ValueError("opd.rl_coef and opd.kd_coef are both 0: the loss would be empty")
        if self.teacher not in TEACHERS:
            raise ValueError(f"opd.teacher must be one of {TEACHERS}, got {self.teacher!r}")
        if self.online_teacher:
            if self.omega == 0:
                raise ValueError("opd.teacher=online with opd.omega=0 makes the teacher the "
                                 "student itself: the loss and its gradient are zero")

    @property
    def full_vocab(self) -> bool:
        return self.estimator == "full"

    @property
    def online_teacher(self) -> bool:
        """The teacher is the current model's LoopCD distribution (stop-gradient), computed from
        the student's own training forward; no frozen copy is loaded."""
        return self.teacher == "online"

    @property
    def actor_teacher_log_probs(self) -> bool:
        """The actor's own ``compute_log_prob`` pass gives ``teacher_log_probs`` (online teacher
        with a sampled-token estimator); no teacher engine is loaded."""
        return self.online_teacher and self.sampled_teacher

    @property
    def frozen_teacher_model(self) -> bool:
        """Every actor rank holds a frozen copy of the initial model (``full``, frozen teacher)."""
        return self.full_vocab and not self.online_teacher

    @property
    def sampled_teacher(self) -> bool:
        """The loss needs the teacher's sampled-token log-probs (``teacher_log_probs`` from the
        ``compute_log_prob`` pass: the frozen engine, or the actor itself with the online
        teacher): estimators ``sampled`` and ``k2``."""
        return self.estimator in SAMPLED_TOKEN_ESTIMATORS

    @property
    def pure_distillation(self) -> bool:
        """``rl_coef = 0, kd_coef = 1``: the original OPD loss, unchanged."""
        return self.rl_coef == 0 and self.kd_coef == 1

    def student_model_kwargs(self) -> dict:
        """Keyword arguments of the student's per-loop forward for the full-vocabulary loss; the
        caller adds ``opd_window`` and, for the frozen teacher, ``opd_teacher_hidden_states`` and
        ``opd_teacher_lm_head``. The online teacher reads the student's own loops instead."""
        kwargs = {"return_opd_kd": True, "opd_omega": self.omega,
                  "opd_divergence": self.divergence}
        if self.online_teacher:
            kwargs.update(opd_online_teacher=True, opd_ref_loop=self.ref_loop)
        return kwargs

    def teacher_model_kwargs(self) -> dict:
        """Keyword arguments of the per-loop forward (``modeling_ouro.py``) that give the
        teacher's log-probs on the sampled tokens: LoopCD-Logits with a fixed strength, no clip.
        The teacher runs without gradient, so ``extrap_ref_grad`` is irrelevant (kept False)."""
        return {"return_extrap_log_probs": True, "extrap_omega": self.omega,
                "extrap_ref_loop": self.ref_loop, "extrap_adaptive": False,
                "extrap_delta_cap": None, "extrap_ref_grad": False}

    def describe(self) -> str:
        return (f"omega={self.omega} ref_loop={self.ref_loop} estimator={self.estimator} "
                f"divergence={self.divergence} rl_coef={self.rl_coef} kd_coef={self.kd_coef} "
                f"teacher={self.teacher} ("
                + ("teacher: the current model, stop-gradient" if self.online_teacher
                   else "teacher: frozen initial model") + ")")


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


def opd_k2_loss(log_prob: torch.Tensor, teacher_log_prob: torch.Tensor,
                response_mask: torch.Tensor, loss_agg_mode: str = "token-mean",
                row_div: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Dict[str, object]]:
    """KDRL's k2 term ``agg[(log pi_T(y) - log P^(4)(y))^2 / 2]`` on the sampled tokens of the last
    loop's ``[B, L]`` log-probs (gradient through the student only: ``-R grad log P^(4)``)."""
    if teacher_log_prob.shape != log_prob.shape:
        raise ValueError(f"teacher log-probs {tuple(teacher_log_prob.shape)} do not match the "
                         f"student's {tuple(log_prob.shape)}")
    log_ratio = teacher_log_prob.detach().to(log_prob.dtype) - log_prob
    loss = plc.agg_term(0.5 * log_ratio.square(), response_mask, loss_agg_mode, row_div)
    return loss, {"loss_term/kd": loss.detach().item(), "loss_term/pg": 0.0}


def opd_loss(settings: "OPDSettings", log_prob: torch.Tensor, old_log_prob: torch.Tensor,
             response_mask: torch.Tensor, clip_ratio: float, *,
             advantages: Optional[torch.Tensor] = None,
             teacher_log_prob: Optional[torch.Tensor] = None, kd: Optional[torch.Tensor] = None,
             loss_agg_mode: str = "token-mean", clip_ratio_low: Optional[float] = None,
             clip_ratio_high: Optional[float] = None, clip_ratio_c: float = 3.0,
             row_div: Optional[torch.Tensor] = None
             ) -> Tuple[torch.Tensor, Dict[str, object]]:
    """``rl_coef * GRPO(advantages) + kd_coef * D`` on the last loop.

    ``D`` is the estimator's divergence term: ``opd_policy_loss`` (``sampled``, needs
    ``teacher_log_prob``), ``opd_full_loss`` (``full``, needs the student forward's ``kd``) or
    ``opd_k2_loss`` (``k2``, needs ``teacher_log_prob``). With ``rl_coef = 0, kd_coef = 1`` the
    loss and metrics are exactly those of the estimator alone (the original OPD). Otherwise the
    metrics report ``loss_term/pg`` (the weighted GRPO term) and ``loss_term/kd`` (the weighted
    divergence) and keep verl's clip statistics for the GRPO term only.
    """
    clip = dict(clip_ratio_low=clip_ratio_low, clip_ratio_high=clip_ratio_high,
                clip_ratio_c=clip_ratio_c)
    if settings.sampled_teacher and teacher_log_prob is None:
        raise ValueError(f"OPD estimator={settings.estimator} needs teacher_log_probs from the "
                         "frozen teacher engine")
    if settings.full_vocab:
        if kd is None:
            raise ValueError("OPD estimator=full needs opd_kd from the student's forward")
        kd_loss, kd_metrics = opd_full_loss(kd, response_mask, loss_agg_mode, row_div)
    elif settings.estimator == "k2":
        kd_loss, kd_metrics = opd_k2_loss(log_prob, teacher_log_prob, response_mask,
                                          loss_agg_mode, row_div)
    else:
        kd_loss, kd_metrics = opd_policy_loss(log_prob, old_log_prob, teacher_log_prob,
                                              response_mask, clip_ratio, loss_agg_mode,
                                              row_div=row_div, **clip)
    if settings.pure_distillation:
        return kd_loss, kd_metrics
    metrics: Dict[str, object] = {}
    loss = settings.kd_coef * kd_loss
    if settings.rl_coef > 0:
        if advantages is None:
            raise ValueError("opd.rl_coef > 0 needs the batch's outcome advantages")
        pg_loss, pg_metrics = plc.grpo_policy_loss(log_prob, old_log_prob, advantages,
                                                   response_mask, clip_ratio,
                                                   loss_agg_mode=loss_agg_mode, row_div=row_div,
                                                   **clip)
        metrics.update({k: v for k, v in pg_metrics.items() if not k.startswith("loss_term/")})
        loss = settings.rl_coef * pg_loss + loss
        metrics["loss_term/pg"] = settings.rl_coef * pg_loss.detach().item()
    else:
        metrics["loss_term/pg"] = 0.0
    metrics["loss_term/kd"] = settings.kd_coef * kd_loss.detach().item()
    return loss, metrics


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
