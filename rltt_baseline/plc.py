"""Posterior Loop Credit (PLC): token x loop weights for the per-loop RLTT objective.

Pure torch, no verl dependency, so everything here is unit-tested on CPU
(``code/tests/test_plc.py``). ``rltt_baseline.plc_workers`` wires it into the verl actor.

Notation, for a batch ``B`` of responses of length ``L`` from a ``T``-loop model:

    ell[b, j, t]  = log P^(t)(y_j | x, y_<j)      log-prob of the sampled token at loop t
    logq[b, j, t] = log q_t(s_j)                   log prior over the exit depth
    w[b, j, t]    = softmax_t(logq + gamma * ell_old)   loop weight, stop-gradient

``gamma = 1`` is the exact posterior ``q_t P^(t)(y_j) / sum_s q_s P^(s)(y_j)``, the policy-gradient
coefficient of the exit mixture ``pi_q = sum_t q_t P^(t)``; ``gamma = 0`` gives ``w = q``, i.e. RLTT
with that prior. Two prior-free modes name the uniform-prior special cases explicitly (they need
``prior = uniform`` and ignore ``gamma``):

    weight_mode = uniform        w[b, j, t] = 1 / T                      (RLTT-uniform)
    weight_mode = logp_softmax   w[b, j, t] = softmax_t(ell_old[b, j, t])
                                            = P^(t)(y_j) / sum_s P^(s)(y_j)

``logp_softmax`` is the policy-gradient coefficient of ``log (1/T) sum_t P^(t)(y_j)``.

The policy loss mixes the final-loop PPO term with the loop-weighted one:

    loss_pg = -agg_j[ (1 - lam) * PPO_j^(T) + lam * sum_t w[j, t] * PPO_j^(t) ]

where ``PPO^(t)`` is verl's clipped objective (``compute_policy_loss_vanilla`` in verl 0.6.1,
dual clip included) on loop t's ratio ``exp(ell_new - ell_old)``. ``lam = 0`` is GRPO on the
final loop. With ``prior = exit`` the gate can be trained with the gate part of the same
mixture-policy gradient plus Ouro's Stage I entropy regulariser:

    loss_gate = gate_coef * agg_j[ A_j * CE(sg[w_j], q_new_j) ]
                - gate_entropy_coef * agg_j[ H(q_new_j) ]

(``CE(w, q) = -sum_t w_t log q_t``; Ouro uses beta = 0.1 in pre-training and 0.05 in its last
stage, arXiv 2510.25741 Table 2 / Section 4.) The optional importance factor
``is_j = min(sum_t q_t exp(ell_old[j, t] - ell_old[j, T]), is_clip)`` on the loop term turns the
self-normalised weights into plain per-loop importance weights ``q_t P^(t) / P^(T)``.
"""

from dataclasses import asdict, dataclass, fields
from typing import Dict, Mapping, Optional, Tuple

import torch

WEIGHT_MODES = ("rltt", "posterior", "uniform", "logp_softmax")
PRIOR_FREE_WEIGHT_MODES = ("uniform", "logp_softmax")
PRIORS = ("uniform", "progressive", "exit")
LOSS_AGG_MODES = ("token-mean", "seq-mean-token-sum", "seq-mean-token-mean",
                  "seq-mean-token-sum-norm")
LOG_EPS = 1e-8


@dataclass(frozen=True)
class PLCSettings:
    """The ``plc`` config section; ``from_mapping`` validates a dict of user values."""

    enabled: bool = False
    weight_mode: str = "posterior"      # rltt | posterior | uniform | logp_softmax
    prior: str = "uniform"              # uniform | progressive | exit
    progressive_alpha: float = 1.0      # prior=progressive: q_t ∝ t^alpha
    gamma: float = 1.0                  # posterior temperature; 0 = RLTT
    lam: float = 1.0                    # loop-weighted share of the policy loss; 0 = GRPO
    eps_floor: float = 0.0              # w <- (1 - eps) w + eps / T
    use_gate_loss: bool = False         # needs prior=exit
    gate_coef: float = 1.0
    gate_entropy_coef: float = 0.05     # Ouro Stage I entropy regulariser (last-stage value)
    use_is_correction: bool = False
    is_clip: float = 5.0
    log_weight_stats: bool = True

    @classmethod
    def from_mapping(cls, values: Optional[Mapping]) -> "PLCSettings":
        values = dict(values or {})
        names = {f.name for f in fields(cls)}
        unknown = set(values) - names
        if unknown:
            raise KeyError(f"unknown plc keys {sorted(unknown)}; allowed {sorted(names)}")
        merged = {**asdict(cls()), **values}
        for f in fields(cls):
            value = merged[f.name]
            if f.type == "bool" or isinstance(getattr(cls(), f.name), bool):
                if not isinstance(value, bool):
                    raise TypeError(f"plc.{f.name} must be a bool, got {value!r}")
            elif isinstance(getattr(cls(), f.name), float):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError(f"plc.{f.name} must be a number, got {value!r}")
                merged[f.name] = float(value)
            elif not isinstance(value, str):
                raise TypeError(f"plc.{f.name} must be a string, got {value!r}")
        settings = cls(**merged)
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.weight_mode not in WEIGHT_MODES:
            raise ValueError(f"plc.weight_mode must be one of {WEIGHT_MODES}")
        if self.prior not in PRIORS:
            raise ValueError(f"plc.prior must be one of {PRIORS}")
        if self.gamma < 0:
            raise ValueError("plc.gamma must be >= 0")
        if not 0.0 <= self.lam <= 1.0:
            raise ValueError("plc.lam must be in [0, 1]")
        if not 0.0 <= self.eps_floor < 1.0:
            raise ValueError("plc.eps_floor must be in [0, 1)")
        if self.progressive_alpha < 0:
            raise ValueError("plc.progressive_alpha must be >= 0")
        if self.gate_coef < 0 or self.gate_entropy_coef < 0:
            raise ValueError("plc.gate_coef and plc.gate_entropy_coef must be >= 0")
        if self.is_clip <= 0:
            raise ValueError("plc.is_clip must be positive")
        if self.use_gate_loss and self.prior != "exit":
            raise ValueError("plc.use_gate_loss needs plc.prior=exit (the gate is the prior)")
        if self.weight_mode in PRIOR_FREE_WEIGHT_MODES and self.prior != "uniform":
            # the weights do not use the prior, but the IS correction does; keep both uniform
            raise ValueError(f"plc.weight_mode={self.weight_mode} needs plc.prior=uniform")

    @property
    def needs_exit_pdf(self) -> bool:
        return self.prior == "exit" or self.use_gate_loss

    def describe(self) -> str:
        return (f"weight_mode={self.weight_mode} prior={self.prior} gamma={self.gamma} "
                f"lam={self.lam} eps_floor={self.eps_floor} gate_loss={self.use_gate_loss} "
                f"gate_coef={self.gate_coef} gate_entropy_coef={self.gate_entropy_coef} "
                f"is_correction={self.use_is_correction}")


# ---------- priors and weights ----------

def loop_prior_logq(prior: str, num_loops: int, alpha: float = 1.0,
                    exit_pdf: Optional[torch.Tensor] = None,
                    like: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``log q_t``: shape ``[1, 1, T]`` for the static priors, ``[B, L, T]`` for ``exit``.

    ``exit_pdf`` is the model's exit distribution (``sum_t = 1``), used detached; ``like`` gives
    the device / dtype of the static priors.
    """
    if prior == "exit":
        if exit_pdf is None:
            raise ValueError("prior=exit needs the exit_pdf tensor")
        return torch.log(exit_pdf.detach().float().clamp_min(LOG_EPS))
    device = like.device if like is not None else None
    if prior == "uniform":
        q = torch.full((num_loops,), 1.0 / num_loops, device=device)
    elif prior == "progressive":
        t = torch.arange(1, num_loops + 1, device=device, dtype=torch.float32)
        q = t.pow(alpha) / t.pow(alpha).sum()
    else:
        raise ValueError(f"unknown prior {prior!r}")
    return q.log().view(1, 1, num_loops)


def compute_loop_weights(ell_old: torch.Tensor, logq: torch.Tensor, gamma: float,
                         weight_mode: str = "posterior", eps_floor: float = 0.0) -> torch.Tensor:
    """``w[b, j, t]``, detached, ``sum_t = 1``.

    ``ell_old``: ``[B, L, T]`` old-policy per-loop log-probs of the sampled token;
    ``logq``: broadcastable to it (``uniform`` and ``logp_softmax`` ignore it and ``gamma``).
    """
    ell_old = ell_old.detach().float()
    if weight_mode == "rltt":
        z = logq.expand_as(ell_old)
    elif weight_mode == "posterior":
        z = logq + gamma * ell_old
    elif weight_mode == "uniform":
        z = torch.zeros_like(ell_old)
    elif weight_mode == "logp_softmax":
        z = ell_old
    else:
        raise ValueError(f"unknown weight_mode {weight_mode!r}")
    w = torch.softmax(z, dim=-1)
    if eps_floor > 0:
        w = (1.0 - eps_floor) * w + eps_floor / w.shape[-1]
    return w.detach()


def importance_weights(ell_old: torch.Tensor, logq: torch.Tensor, is_clip: float) -> torch.Tensor:
    """``is_j = min(sum_t q_t exp(ell_old[j, t] - ell_old[j, T]), is_clip)``, shape ``[B, L]``."""
    ell_old = ell_old.detach().float()
    ratio = torch.exp(logq + ell_old - ell_old[..., -1:]).sum(-1)
    return ratio.clamp(max=is_clip).detach()


# ---------- losses ----------

def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str) -> torch.Tensor:
    """verl 0.6.1 ``core_algos.agg_loss`` for ``[B, L]`` matrices."""
    loss_mask = loss_mask.to(loss_mat.dtype)
    if loss_agg_mode == "token-mean":
        return (loss_mat * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)
    if loss_agg_mode == "seq-mean-token-sum":
        seq = (loss_mat * loss_mask).sum(-1)
        seq_mask = (loss_mask.sum(-1) > 0).to(loss_mat.dtype)
        return (seq * seq_mask).sum() / seq_mask.sum().clamp_min(1.0)
    if loss_agg_mode == "seq-mean-token-mean":
        counts = loss_mask.sum(-1)
        seq = (loss_mat * loss_mask).sum(-1) / (counts + 1e-8)
        seq_mask = (counts > 0).to(loss_mat.dtype)
        return (seq * seq_mask).sum() / seq_mask.sum().clamp_min(1.0)
    if loss_agg_mode == "seq-mean-token-sum-norm":
        return (loss_mat * loss_mask).sum() / loss_mask.shape[-1]
    raise ValueError(f"invalid loss_agg_mode {loss_agg_mode!r}; allowed {LOSS_AGG_MODES}")


def ppo_objective(logp_new: torch.Tensor, logp_old: torch.Tensor, advantages: torch.Tensor,
                  clip_ratio: float, clip_ratio_low: Optional[float] = None,
                  clip_ratio_high: Optional[float] = None, clip_ratio_c: float = 3.0
                  ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Per-element clipped PPO objective (to be maximised), as verl 0.6.1's vanilla loss.

    ``advantages`` must broadcast against ``logp_new``. Returns the objective and the
    per-element ``ratio``, ``clipped`` (upper / lower clip active) and ``clipped_lower``
    (dual-clip active) indicators for metrics.
    """
    low = clip_ratio if clip_ratio_low is None else clip_ratio_low
    high = clip_ratio if clip_ratio_high is None else clip_ratio_high
    if clip_ratio_c <= 1.0:
        raise ValueError("clip_ratio_c must be > 1")
    negative_approx_kl = (logp_new - logp_old).clamp(-20.0, 20.0)
    ratio = torch.exp(negative_approx_kl)
    losses1 = -advantages * ratio
    losses2 = -advantages * ratio.clamp(1.0 - low, 1.0 + high)
    clipped1 = torch.maximum(losses1, losses2)
    losses3 = -advantages * clip_ratio_c
    clipped2 = torch.minimum(losses3, clipped1)
    losses = torch.where(advantages < 0, clipped2, clipped1)
    info = {
        "ratio": ratio.detach(),
        "negative_approx_kl": negative_approx_kl.detach(),
        "clipped": (losses2 > losses1).to(ratio.dtype).detach(),
        "clipped_lower": ((clipped1 > losses3) & (advantages < 0)).to(ratio.dtype).detach(),
    }
    return -losses, info


def _masked_mean_per_loop(values: torch.Tensor, mask3: torch.Tensor) -> list:
    denom = mask3.sum(dim=(0, 1)).clamp_min(1.0)
    return ((values * mask3).sum(dim=(0, 1)) / denom).tolist()


def plc_policy_loss(logp_new: torch.Tensor, logp_old: torch.Tensor, advantages: torch.Tensor,
                    response_mask: torch.Tensor, weights: torch.Tensor, lam: float,
                    clip_ratio: float, loss_agg_mode: str = "token-mean",
                    clip_ratio_low: Optional[float] = None, clip_ratio_high: Optional[float] = None,
                    clip_ratio_c: float = 3.0, is_weights: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, Dict[str, object]]:
    """``-agg_j[(1 - lam) PPO^(T) + lam sum_t w PPO^(t)]`` and its metrics.

    ``logp_new`` / ``logp_old``: ``[B, L, T]``; ``advantages``: ``[B]`` or ``[B, L]``;
    ``response_mask``: ``[B, L]``; ``weights``: ``[B, L, T]`` (treated as constants);
    ``is_weights``: optional ``[B, L]`` factor on the loop term.
    """
    if logp_new.shape != logp_old.shape or logp_new.dim() != 3:
        raise ValueError("logp_new and logp_old must both have shape [B, L, T]")
    if advantages.dim() == 1:
        advantages = advantages[:, None]
    adv3 = advantages.to(logp_new.dtype)[..., None]
    weights = weights.detach().to(logp_new.dtype)
    objective, info = ppo_objective(logp_new, logp_old.detach(), adv3, clip_ratio,
                                    clip_ratio_low, clip_ratio_high, clip_ratio_c)
    loop_term = (weights * objective).sum(-1)
    if is_weights is not None:
        loop_term = loop_term * is_weights.detach().to(loop_term.dtype)
    final_term = objective[..., -1]
    per_token = (1.0 - lam) * final_term + lam * loop_term
    loss = -agg_loss(per_token, response_mask, loss_agg_mode)

    with torch.no_grad():
        mask = response_mask.to(objective.dtype)
        mask3 = mask[..., None].expand_as(objective)
        metrics = {
            "actor/pg_clipfrac": agg_loss(info["clipped"][..., -1], mask, "token-mean").item(),
            "actor/ppo_kl": agg_loss(-info["negative_approx_kl"][..., -1], mask,
                                     "token-mean").item(),
            "actor/pg_clipfrac_lower": agg_loss(info["clipped_lower"][..., -1], mask,
                                                "token-mean").item(),
            "plc/clipfrac_per_loop": _masked_mean_per_loop(info["clipped"], mask3),
            "plc/ratio_mean_per_loop": _masked_mean_per_loop(info["ratio"], mask3),
            "plc/loop_term_mean": agg_loss(loop_term, mask, "token-mean").item(),
            "plc/final_term_mean": agg_loss(final_term, mask, "token-mean").item(),
        }
        if is_weights is not None:
            metrics["plc/is_weight_mean"] = agg_loss(is_weights.float(), mask, "token-mean").item()
    return loss, metrics


def gate_loss(exit_pdf_new: torch.Tensor, weights: torch.Tensor, advantages: torch.Tensor,
              response_mask: torch.Tensor, gate_coef: float, gate_entropy_coef: float,
              loss_agg_mode: str = "token-mean") -> Tuple[torch.Tensor, Dict[str, object]]:
    """``gate_coef * agg[A * CE(sg[w], q)] - gate_entropy_coef * agg[H(q)]`` and metrics.

    ``exit_pdf_new``: ``[B, L, T]`` exit distribution of the current gate, with gradient.
    """
    if advantages.dim() == 1:
        advantages = advantages[:, None]
    q = exit_pdf_new.float()
    logq = torch.log(q.clamp_min(LOG_EPS))
    w = weights.detach().float()
    ce = -(w * logq).sum(-1)                                   # [B, L]
    entropy = -(q * logq).sum(-1)                               # [B, L]
    adv = advantages.to(ce.dtype)
    loss = gate_coef * agg_loss(adv * ce, response_mask, loss_agg_mode)
    loss = loss - gate_entropy_coef * agg_loss(entropy, response_mask, loss_agg_mode)

    with torch.no_grad():
        mask = response_mask.to(ce.dtype)
        mask3 = mask[..., None].expand_as(q)
        steps = torch.arange(1, q.shape[-1] + 1, device=q.device, dtype=q.dtype)
        kl_w_q = (w * (torch.log(w.clamp_min(LOG_EPS)) - logq)).sum(-1)
        metrics = {
            "gate/ce_mean": agg_loss(ce, mask, "token-mean").item(),
            "gate/entropy_mean": agg_loss(entropy, mask, "token-mean").item(),
            "gate/kl_w_q_mean": agg_loss(kl_w_q, mask, "token-mean").item(),
            "gate/expected_exit_step": agg_loss((q * steps).sum(-1), mask, "token-mean").item(),
            "gate/q_mean_per_loop": _masked_mean_per_loop(q, mask3),
            "gate/loss": loss.detach().item(),
        }
    return loss, metrics


# ---------- diagnostics ----------

@torch.no_grad()
def weight_stats(weights: torch.Tensor, ell_old: torch.Tensor,
                 response_mask: torch.Tensor) -> Dict[str, object]:
    """Per-batch summaries of the weights and of the loop-to-loop log-prob increments."""
    w = weights.float()
    ell = ell_old.float()
    mask = response_mask.to(w.dtype)
    mask3 = mask[..., None].expand_as(w)
    n = mask.sum().clamp_min(1.0)
    num_loops = w.shape[-1]
    entropy = -(w * torch.log(w.clamp_min(LOG_EPS))).sum(-1)
    delta = ell[..., 1:] - ell[..., :-1]                        # [B, L, T-1]
    mask_delta = mask[..., None].expand_as(delta)
    n_delta = mask_delta.sum(dim=(0, 1)).clamp_min(1.0)
    return {
        "plc/w_mean_per_loop": _masked_mean_per_loop(w, mask3),
        "plc/eff_loops_mean": ((entropy.exp() * mask).sum() / n).item(),
        "plc/argmax_is_final_frac": (((w.argmax(-1) == num_loops - 1).to(w.dtype) * mask).sum()
                                     / n).item(),
        "plc/delta_mean_per_transition": ((delta * mask_delta).sum(dim=(0, 1)) / n_delta).tolist(),
        "plc/delta_neg_frac_per_transition": (((delta < 0).to(w.dtype) * mask_delta).sum(dim=(0, 1))
                                              / n_delta).tolist(),
        "plc/delta_abs_lt_0.01_frac": (((delta.abs() < 0.01).to(w.dtype) * mask_delta).sum()
                                       / (n * max(num_loops - 1, 1))).item(),
        "plc/final_logp_mean": ((ell[..., -1] * mask).sum() / n).item(),
    }


def low_var_kl(logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    """verl ``low_var_kl``: ``exp(ref - logp) - (ref - logp) - 1``, clamped to [-10, 10]."""
    kl = (ref_logp - logp).clamp(-20.0, 20.0)
    return (torch.exp(kl) - kl - 1).clamp(-10.0, 10.0)


def entropy_from_log_probs(logq: torch.Tensor) -> torch.Tensor:
    return -(logq.exp() * logq).sum(-1)


__all__ = [
    "PLCSettings", "WEIGHT_MODES", "PRIOR_FREE_WEIGHT_MODES", "PRIORS", "LOSS_AGG_MODES",
    "loop_prior_logq", "compute_loop_weights", "importance_weights",
    "agg_loss", "ppo_objective", "plc_policy_loss", "gate_loss",
    "weight_stats", "low_var_kl", "entropy_from_log_probs",
]
