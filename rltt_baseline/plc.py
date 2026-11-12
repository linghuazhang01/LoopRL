"""PLC: GRPO on the final loop plus an off-policy auxiliary term on the earlier loops.

Pure torch, no verl dependency, so everything here is unit-tested on CPU
(``tests/test_plc.py``). ``rltt_baseline.plc_workers`` wires it into the verl actor.

Notation, for a batch ``B`` of responses of length ``L`` from a ``T``-loop model:

    ell[b, j, t] = log P^(t)(y_j | x, y_<j)      log-prob of the sampled token at loop t

vLLM samples from the final loop T, which is therefore the behaviour policy of the earlier
loops. The final loop is trained by GRPO unchanged; each earlier loop t < T by a
truncated-importance-weighted PPO surrogate plus a forward-KL trust region towards the final loop:

    loss = -agg_j[ PPO_j^(T)(A) ]                                          (GRPO, unchanged)
           + lam * sum_{t<T} alpha_t * ( -agg_j[ rho[j, t] PPO_j^(t)(A) ]     (surrogate)
                                         + kl_coef * agg_j[ k1[j, t] ] )    (trust region)
    rho[j, t] = min(trunc_cap, P_old^(t)(y_j) / P_old^(T)(y_j)),  stop-gradient
    k1[j, t]  = log P_old^(T)(y_j) - log P^(t)(y_j)

``PPO^(t)`` is verl's clipped objective (``compute_policy_loss_vanilla`` in verl 0.6.1, dual clip
included) on loop t's ratio ``exp(ell_new - ell_old)``; ``agg`` is the actor's
``loss_agg_mode``. ``alpha_t = 1 / (T - 1)`` (uniform over the earlier loops). ``k1`` is the
sampled estimate of ``KL(P^(T) || P^(t))`` on the final loop's own samples (unbiased, no ratio);
its gradient is the cross-entropy of loop t on the rollout tokens. ``trunc_cap`` bounds the ratio
from above only (variance); ``kl_coef`` sets how far loop t may move from the final loop. With one
update per step the per-token gradient on loop t is ``-(lam / N) alpha_t (rho A + kl_coef)
d log P^(t)``. ``lam = 0`` is GRPO on this code path; ``kl_coef = 0`` is the surrogate alone;
``weight_mode = early_prior`` drops the ratio (``rho = 1``). The gate gets no gradient.
"""

import math
from dataclasses import asdict, dataclass, fields
from typing import Dict, Mapping, Optional, Tuple

import torch

WEIGHT_MODES = ("is_trunc", "early_prior")
LOSS_AGG_MODES = ("token-mean", "seq-mean-token-sum", "seq-mean-token-mean",
                  "seq-mean-token-sum-norm")
LOG_EPS = 1e-8


@dataclass(frozen=True)
class PLCSettings:
    """The ``plc`` config section; ``from_mapping`` validates a dict of user values."""

    enabled: bool = False
    weight_mode: str = "is_trunc"       # is_trunc | early_prior (rho = 1)
    lam: float = 0.1                    # coefficient of the earlier-loop term; 0 = GRPO
    trunc_cap: float = 1.0              # is_trunc: upper cap c of the cross-loop ratio (>= 1)
    kl_coef: float = 0.2                # forward-KL trust region of the earlier loops
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
            default = getattr(cls(), f.name)
            if isinstance(default, bool):
                if not isinstance(value, bool):
                    raise TypeError(f"plc.{f.name} must be a bool, got {value!r}")
            elif isinstance(default, float):
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
        if self.lam < 0:
            raise ValueError("plc.lam must be >= 0")
        if self.trunc_cap < 1.0:
            raise ValueError("plc.trunc_cap must be >= 1 (it caps the ratio from above only)")
        if self.trunc_cap != 1.0 and self.weight_mode != "is_trunc":
            raise ValueError("plc.trunc_cap only applies to plc.weight_mode=is_trunc")
        if self.kl_coef < 0:
            raise ValueError("plc.kl_coef must be >= 0")

    def describe(self) -> str:
        return (f"weight_mode={self.weight_mode} lam={self.lam} trunc_cap={self.trunc_cap} "
                f"kl_coef={self.kl_coef}")


# ---------- weights ----------

def compute_loop_weights(ell_old: torch.Tensor, weight_mode: str = "is_trunc",
                         trunc_cap: float = 1.0) -> torch.Tensor:
    """``w[b, j, t]`` ``[B, L, T]``, detached; rows do not sum to 1.

    ``ell_old``: old-policy per-loop log-probs of the sampled token. For t < T,
    ``is_trunc``: ``w[.., t] = alpha_t * min(c, exp(ell_old[.., t] - ell_old[.., T]))`` and
    ``early_prior``: ``w[.., t] = alpha_t``, with ``alpha_t = 1 / (T - 1)`` and ``c = trunc_cap``;
    ``w[.., T] = 0`` (the final loop is trained by the GRPO term).
    """
    if weight_mode not in WEIGHT_MODES:
        raise ValueError(f"unknown weight_mode {weight_mode!r}; allowed {WEIGHT_MODES}")
    if trunc_cap < 1.0:
        raise ValueError("trunc_cap must be >= 1")
    ell_old = ell_old.detach().float()
    num_loops = ell_old.shape[-1]
    if num_loops < 2:
        raise ValueError("the earlier-loop term needs at least two loops")
    alpha = torch.full_like(ell_old[..., :-1], 1.0 / (num_loops - 1))
    if weight_mode == "is_trunc":
        log_ratio = (ell_old[..., :-1] - ell_old[..., -1:]).clamp(max=math.log(trunc_cap))
        alpha = alpha * torch.exp(log_ratio)
    return torch.cat([alpha, torch.zeros_like(ell_old[..., -1:])], dim=-1).detach()


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
                    clip_ratio_c: float = 3.0, kl_coef: float = 0.0,
                    kl_weights: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, Dict[str, object]]:
    """``-agg[PPO^(T)] - lam * agg[sum_t w PPO^(t)] + lam * kl_coef * agg[sum_t kl_w k1^(t)]``.

    ``logp_new`` / ``logp_old``: ``[B, L, T]``; ``advantages``: ``[B]`` or ``[B, L]``;
    ``response_mask``: ``[B, L]``; ``weights``: ``[B, L, T]`` from ``compute_loop_weights``
    (treated as constants, 0 on the final loop). The trust region (``kl_coef > 0``) uses
    ``k1^(t) = logp_old[.., T] - logp_new[.., t]``, the sampled ``KL(P_old^(T) || P^(t))`` on the
    final loop's samples, weighted by ``kl_weights`` ``[B, L, T]`` (``alpha`` on t < T, 0 on T).
    """
    if logp_new.shape != logp_old.shape or logp_new.dim() != 3:
        raise ValueError("logp_new and logp_old must both have shape [B, L, T]")
    if kl_coef < 0:
        raise ValueError("kl_coef must be >= 0")
    if kl_coef > 0 and (kl_weights is None or kl_weights.shape != logp_new.shape):
        raise ValueError("kl_coef > 0 needs kl_weights of shape [B, L, T]")
    if advantages.dim() == 1:
        advantages = advantages[:, None]
    adv3 = advantages.to(logp_new.dtype)[..., None]
    weights = weights.detach().to(logp_new.dtype)
    ppo, info = ppo_objective(logp_new, logp_old.detach(), adv3, clip_ratio, clip_ratio_low,
                              clip_ratio_high, clip_ratio_c)
    loop_term = (weights * ppo).sum(-1)
    final_term = ppo[..., -1]
    loss = -(agg_loss(final_term, response_mask, loss_agg_mode)
             + lam * agg_loss(loop_term, response_mask, loss_agg_mode))
    k1 = kl_term = None
    if kl_coef > 0:
        k1 = logp_old.detach()[..., -1:] - logp_new                       # [B, L, T]
        kl_term = (kl_weights.detach().to(k1.dtype) * k1).sum(-1)          # [B, L]
        loss = loss + lam * kl_coef * agg_loss(kl_term, response_mask, loss_agg_mode)

    with torch.no_grad():
        mask = response_mask.to(ppo.dtype)
        mask3 = mask[..., None].expand_as(ppo)
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
        if kl_term is not None:
            # k1 is the loss's estimate; k3 = r - 1 - log r (r = P^(t) / P_old^(T)) estimates the
            # same KL with lower variance and is >= 0 per token, for monitoring only
            log_r = (-k1).clamp(-20.0, 20.0)
            k3 = torch.exp(log_r) - 1.0 - log_r
            metrics["plc/kl_term_mean"] = agg_loss(kl_term, mask, "token-mean").item()
            metrics["plc/kl_k1_per_loop"] = _masked_mean_per_loop(k1.to(mask.dtype), mask3)
            metrics["plc/kl_k3_per_loop"] = _masked_mean_per_loop(k3.to(mask.dtype), mask3)
    return loss, metrics


# ---------- diagnostics ----------

@torch.no_grad()
def cross_loop_stats(ell_old: torch.Tensor, advantages: torch.Tensor,
                     response_mask: torch.Tensor, trunc_cap: float = 1.0) -> Dict[str, object]:
    """Per earlier loop t < T: the cross-loop ratio ``rho = P_old^(t)(y) / P_old^(T)(y)`` of the
    sampled tokens (mean of ``min(c, rho)``, share above the cap, share below 0.1) and the mean
    surrogate weight ``|min(c, rho) * A|``, to set against ``kl_coef``."""
    if advantages.dim() == 1:
        advantages = advantages[:, None]
    ell = ell_old.float()
    log_rho = ell[..., :-1] - ell[..., -1:]                                 # [B, L, T-1]
    rho_bar = torch.exp(log_rho.clamp(max=math.log(trunc_cap)))
    mask = response_mask.to(ell.dtype)
    mask3 = mask[..., None].expand_as(log_rho)
    adv = advantages.to(ell.dtype).expand_as(mask)[..., None]
    return {
        "plc/xloop_rho_bar_mean_per_loop": _masked_mean_per_loop(rho_bar, mask3),
        "plc/xloop_capped_frac_per_loop": _masked_mean_per_loop(
            (log_rho > math.log(trunc_cap)).to(ell.dtype), mask3),
        "plc/xloop_rho_lt_0.1_frac_per_loop": _masked_mean_per_loop(
            (log_rho < math.log(0.1)).to(ell.dtype), mask3),
        "plc/xloop_pg_weight_abs_mean_per_loop": _masked_mean_per_loop(
            (rho_bar * adv).abs(), mask3),
    }


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
    delta = ell[..., 1:] - ell[..., :-1]                        # [B, L, T-1]
    mask_delta = mask[..., None].expand_as(delta)
    n_delta = mask_delta.sum(dim=(0, 1)).clamp_min(1.0)
    return {
        "plc/w_mean_per_loop": _masked_mean_per_loop(w, mask3),
        "plc/delta_mean_per_transition": ((delta * mask_delta).sum(dim=(0, 1)) / n_delta).tolist(),
        "plc/delta_neg_frac_per_transition": (((delta < 0).to(w.dtype) * mask_delta).sum(dim=(0, 1))
                                              / n_delta).tolist(),
        "plc/delta_abs_lt_0.01_frac": (((delta.abs() < 0.01).to(w.dtype) * mask_delta).sum()
                                       / (n * max(num_loops - 1, 1))).item(),
        "plc/final_logp_mean": ((ell[..., -1] * mask).sum() / n).item(),
    }


def flatten_loop_metrics(metrics: Mapping[str, object]) -> Dict[str, object]:
    """One scalar key per loop for the list-valued metrics, for verl's metric reduction.

    verl appends each micro-batch's metrics to a list and ``reduce_metrics`` takes ``np.mean``
    of that list, which would also average a per-loop list over its loops. A list under
    ``<name>_per_loop`` becomes ``<name>/loop_<t>`` (entry i is loop i + 1). Most have T entries;
    the ``xloop_*`` statistics have T - 1 (the earlier loops 1..T-1 against the final loop). The
    ``loop_T`` entries of ``kl_k1`` / ``kl_k3`` / ``ratio_mean`` / ``clipfrac`` compare the final
    loop with its own old policy; ``kl_weights`` is 0 there, so they are not part of the trust
    region. A list under ``<name>_per_transition`` becomes ``<name>/loop_<t>_to_<t+1>``. Scalars
    pass through unchanged.
    """
    flat: Dict[str, object] = {}
    for key, value in metrics.items():
        if not isinstance(value, (list, tuple)):
            flat[key] = value
        elif key.endswith("_per_loop"):
            base = key[:-len("_per_loop")]
            for i, v in enumerate(value):
                flat[f"{base}/loop_{i + 1}"] = v
        elif key.endswith("_per_transition"):
            base = key[:-len("_per_transition")]
            for i, v in enumerate(value):
                flat[f"{base}/loop_{i + 1}_to_{i + 2}"] = v
        else:
            raise ValueError(f"list-valued metric {key!r} needs a _per_loop or _per_transition "
                             "suffix (verl would average it to one number)")
    return flat


def low_var_kl(logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    """verl ``low_var_kl``: ``exp(ref - logp) - (ref - logp) - 1``, clamped to [-10, 10]."""
    kl = (ref_logp - logp).clamp(-20.0, 20.0)
    return (torch.exp(kl) - kl - 1).clamp(-10.0, 10.0)


__all__ = [
    "PLCSettings", "WEIGHT_MODES", "LOSS_AGG_MODES",
    "compute_loop_weights", "agg_loss", "ppo_objective", "plc_policy_loss",
    "cross_loop_stats", "weight_stats", "flatten_loop_metrics", "low_var_kl",
]
