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

Optional anchor to the initial model (``init_kl_coef > 0``; off by default): each earlier loop is
also pulled back towards its own distribution in the frozen starting model, on the same sampled
tokens,

    + lam * init_kl_coef * agg_j[ sum_{t<T} alpha_t * pen(d[j, t]) ]
    d[j, t] = log P^(t)(y_j) - log P_init^(t)(y_j)

with ``pen`` chosen by ``init_kl_form`` (all three are 0 with zero gradient at d = 0):

    k3          r = P_init / P_new :  e^{-d} + d - 1   (verl low_var_kl for small |d|; verl
                clamps the value at 10, d < -2.48, with zero gradient beyond); per-token slope
                1 - e^{-d}: at most 1 on raised tokens, unbounded on lowered ones
    k3_reverse  r = P_new / P_init :  e^{d} - d - 1;   slope e^{d} - 1: unbounded on raised tokens
                (the direction the rollout tokens of the final loop push), at most 1 on lowered
    k2          d^2 / 2;                               slope d, symmetric in log space

Beyond ``|d| > init_kl_log_ratio_cap`` the penalty continues linearly with its slope at the cap,
so the value stays continuous and the per-token gradient is bounded. Unlike ``k1`` above, the
anchor is a fixed point of loop t itself: the sampled-token ``k1`` to the initial model would have
the same gradient as the ``k1`` to the final loop, because ``log P_init`` carries no gradient.

``loop_loss`` selects the earlier-loop term. ``surrogate`` (default) is everything above.
``kd`` replaces the surrogate and the ``k1`` trust region by full-vocabulary distillation towards
the stop-gradient final loop on the same rollouts (the final loop's own samples), with fixed
weights and no advantage:

    loss = -agg_j[ PPO_j^(T)(A) ]  +  lam * sum_{t<T} alpha_t * agg_j[ D_j^(t) ]
    D_j^(t) = KL( sg P^(T)(.|s_j) || P^(t)(.|s_j) )     kd_divergence = forward
            = KL( P^(t)(.|s_j) || sg P^(T)(.|s_j) )     kd_divergence = reverse

over the full vocabulary at every response position ``s_j = (x, y_<j)``, ``alpha_t = 1 / (T - 1)``.
The gradient on loop t's logits is ``(lam / N) alpha_t (P^(t) - P^(T))`` (forward) or
``(lam / N) alpha_t P^(t) (log P^(t) - log P^(T) - D)`` (reverse). The forward form is the exact
expectation, over the sampled token, of the ``k1`` trust region (its gradient ``P^(t) - e_y``
averages to ``P^(t) - P^(T)`` under ``y ~ P^(T)``). ``kd`` needs ``weight_mode = early_prior``
and ``kl_coef = 0``; the anchor (``init_kl_coef``) can be added on top. ``D`` comes from the
model forward (``return_loop_kd``, ``rltt_per_loop.per_loop_kd``), since it needs every loop's
logits.
"""

import math
from dataclasses import asdict, dataclass, fields
from typing import Dict, Mapping, Optional, Tuple

import torch

WEIGHT_MODES = ("is_trunc", "early_prior")
INIT_KL_FORMS = ("k3", "k3_reverse", "k2")
LOOP_LOSSES = ("surrogate", "kd")
KD_DIVERGENCES = ("forward", "reverse")
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
    init_kl_coef: float = 0.0           # anchor of the earlier loops to the initial model; 0 = off
    init_kl_form: str = "k3_reverse"    # k3 | k3_reverse | k2 (see the module docstring)
    init_kl_log_ratio_cap: float = 3.0  # |log P^(t) - log P_init^(t)| where pen turns linear
    loop_loss: str = "surrogate"        # surrogate (above) | kd (full-vocabulary distillation)
    kd_divergence: str = "forward"      # kd: forward KL(sg P^(T) || P^(t)) | reverse

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
        if self.init_kl_coef < 0:
            raise ValueError("plc.init_kl_coef must be >= 0")
        if self.init_kl_form not in INIT_KL_FORMS:
            raise ValueError(f"plc.init_kl_form must be one of {INIT_KL_FORMS}")
        if self.init_kl_log_ratio_cap <= 0:
            raise ValueError("plc.init_kl_log_ratio_cap must be > 0")
        if self.init_kl_coef > 0 and self.lam == 0:
            # the anchor is scaled by lam: it would load the frozen model for a zero gradient
            raise ValueError("plc.init_kl_coef > 0 needs plc.lam > 0 (the anchor is scaled by "
                             "lam; its strength is lam * init_kl_coef)")
        if self.loop_loss not in LOOP_LOSSES:
            raise ValueError(f"plc.loop_loss must be one of {LOOP_LOSSES}")
        if self.kd_divergence not in KD_DIVERGENCES:
            raise ValueError(f"plc.kd_divergence must be one of {KD_DIVERGENCES}")
        if self.loop_loss == "kd":
            if self.weight_mode != "early_prior":
                raise ValueError("plc.loop_loss=kd uses the fixed weights alpha_t: set "
                                 "plc.weight_mode=early_prior")
            if self.kl_coef != 0:
                raise ValueError("plc.loop_loss=kd replaces the k1 trust region: set "
                                 "plc.kl_coef=0")
        elif self.kd_divergence != "forward":
            raise ValueError("plc.kd_divergence only applies to plc.loop_loss=kd")

    @property
    def uses_init_model(self) -> bool:
        """True when the worker must load the frozen initial model (``init_kl_coef > 0``)."""
        return self.enabled and self.init_kl_coef > 0

    def describe(self) -> str:
        if self.loop_loss == "kd":
            text = (f"loop_loss=kd kd_divergence={self.kd_divergence} lam={self.lam} "
                    f"init_kl_coef={self.init_kl_coef}")
        else:
            text = (f"weight_mode={self.weight_mode} lam={self.lam} trunc_cap={self.trunc_cap} "
                    f"kl_coef={self.kl_coef} init_kl_coef={self.init_kl_coef}")
        if self.init_kl_coef > 0:
            text += (f" init_kl_form={self.init_kl_form} "
                     f"init_kl_log_ratio_cap={self.init_kl_log_ratio_cap}")
        return text


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


def init_anchor_penalty(logp_new: torch.Tensor, logp_init: torch.Tensor, form: str,
                        log_ratio_cap: float = 3.0) -> torch.Tensor:
    """Per-element anchor penalty from ``d = logp_new - logp_init`` (``logp_init`` detached).

    ``k3``: ``e^{-d} + d - 1`` (verl ``low_var_kl`` with ``r = P_init / P_new``); ``k3_reverse``:
    ``e^{d} - d - 1`` (``r = P_new / P_init``); ``k2``: ``d^2 / 2``. For ``|d| > log_ratio_cap``
    the penalty is extended linearly with its slope at the cap (continuous value, gradient
    ``slope(+-cap)``), so no token's gradient can blow up and none is dropped.
    """
    if form not in INIT_KL_FORMS:
        raise ValueError(f"unknown init_kl_form {form!r}; allowed {INIT_KL_FORMS}")
    if log_ratio_cap <= 0:
        raise ValueError("log_ratio_cap must be > 0")
    d = logp_new - logp_init.detach()
    dc = d.clamp(-log_ratio_cap, log_ratio_cap)
    if form == "k3":
        value, slope = torch.exp(-dc) + dc - 1.0, 1.0 - torch.exp(-dc)
    elif form == "k3_reverse":
        value, slope = torch.exp(dc) - dc - 1.0, torch.exp(dc) - 1.0
    else:
        value, slope = 0.5 * dc * dc, dc
    # inside the cap d - dc is 0 with zero gradient; outside, dc is constant and the slope term
    # carries the gradient
    return value + slope.detach() * (d - dc)


def _masked_mean_per_loop(values: torch.Tensor, mask3: torch.Tensor) -> list:
    denom = mask3.sum(dim=(0, 1)).clamp_min(1.0)
    return ((values * mask3).sum(dim=(0, 1)) / denom).tolist()


def plc_policy_loss(logp_new: torch.Tensor, logp_old: torch.Tensor, advantages: torch.Tensor,
                    response_mask: torch.Tensor, weights: torch.Tensor, lam: float,
                    clip_ratio: float, loss_agg_mode: str = "token-mean",
                    clip_ratio_low: Optional[float] = None, clip_ratio_high: Optional[float] = None,
                    clip_ratio_c: float = 3.0, kl_coef: float = 0.0,
                    kl_weights: Optional[torch.Tensor] = None, init_kl_coef: float = 0.0,
                    logp_init: Optional[torch.Tensor] = None,
                    init_weights: Optional[torch.Tensor] = None,
                    init_kl_form: str = "k3_reverse", init_kl_log_ratio_cap: float = 3.0
                    ) -> Tuple[torch.Tensor, Dict[str, object]]:
    """``-agg[PPO^(T)] - lam * agg[sum_t w PPO^(t)] + lam * kl_coef * agg[sum_t kl_w k1^(t)]``
    ``+ lam * init_kl_coef * agg[sum_t init_w pen^(t)]``.

    ``logp_new`` / ``logp_old``: ``[B, L, T]``; ``advantages``: ``[B]`` or ``[B, L]``;
    ``response_mask``: ``[B, L]``; ``weights``: ``[B, L, T]`` from ``compute_loop_weights``
    (treated as constants, 0 on the final loop). The trust region (``kl_coef > 0``) uses
    ``k1^(t) = logp_old[.., T] - logp_new[.., t]``, the sampled ``KL(P_old^(T) || P^(t))`` on the
    final loop's samples, weighted by ``kl_weights`` ``[B, L, T]`` (``alpha`` on t < T, 0 on T).
    The anchor (``init_kl_coef > 0``) uses ``pen^(t) = init_anchor_penalty(logp_new[.., t],
    logp_init[.., t])`` with ``logp_init`` ``[B, L, T]`` the frozen initial model's per-loop
    log-probs of the same tokens, weighted by ``init_weights`` ``[B, L, T]`` (``alpha`` on t < T,
    0 on T). ``init_kl_coef = 0`` leaves the loss bitwise unchanged.
    """
    if logp_new.shape != logp_old.shape or logp_new.dim() != 3:
        raise ValueError("logp_new and logp_old must both have shape [B, L, T]")
    if kl_coef < 0:
        raise ValueError("kl_coef must be >= 0")
    if kl_coef > 0 and (kl_weights is None or kl_weights.shape != logp_new.shape):
        raise ValueError("kl_coef > 0 needs kl_weights of shape [B, L, T]")
    if init_kl_coef < 0:
        raise ValueError("init_kl_coef must be >= 0")
    if init_kl_coef > 0 and (logp_init is None or logp_init.shape != logp_new.shape
                             or init_weights is None or init_weights.shape != logp_new.shape):
        raise ValueError("init_kl_coef > 0 needs logp_init and init_weights of shape [B, L, T]")
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
    init_pen = init_term = None
    if init_kl_coef > 0:
        init_pen = init_anchor_penalty(logp_new, logp_init.to(logp_new.dtype), init_kl_form,
                                       init_kl_log_ratio_cap)                 # [B, L, T]
        init_term = (init_weights.detach().to(init_pen.dtype) * init_pen).sum(-1)
        loss = loss + lam * init_kl_coef * agg_loss(init_term, response_mask, loss_agg_mode)

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
        if init_term is not None:
            # per loop, including the final one (weight 0 there: how far it moved, not trained)
            d = (logp_new - logp_init).detach().to(mask.dtype)
            metrics["plc/init_term_mean"] = agg_loss(init_term, mask, "token-mean").item()
            metrics["plc/init_pen_per_loop"] = _masked_mean_per_loop(
                init_pen.detach().to(mask.dtype), mask3)
            metrics["plc/init_log_ratio_per_loop"] = _masked_mean_per_loop(d, mask3)
            metrics["plc/init_beyond_cap_frac_per_loop"] = _masked_mean_per_loop(
                (d.abs() > init_kl_log_ratio_cap).to(mask.dtype), mask3)
    return loss, metrics


def plc_kd_loss(logp_new: torch.Tensor, logp_old: torch.Tensor, advantages: torch.Tensor,
                response_mask: torch.Tensor, kd: torch.Tensor, lam: float, clip_ratio: float,
                loss_agg_mode: str = "token-mean", clip_ratio_low: Optional[float] = None,
                clip_ratio_high: Optional[float] = None, clip_ratio_c: float = 3.0,
                init_kl_coef: float = 0.0, logp_init: Optional[torch.Tensor] = None,
                init_kl_form: str = "k3_reverse", init_kl_log_ratio_cap: float = 3.0
                ) -> Tuple[torch.Tensor, Dict[str, object]]:
    """``-agg[PPO^(T)] + lam * agg[sum_{t<T} alpha_t kd^(t)]`` (``loop_loss: kd``), plus
    ``lam * init_kl_coef * agg[sum_{t<T} alpha_t pen^(t)]`` when the anchor is on.

    ``logp_new`` / ``logp_old``: ``[B, L, T]`` (only the final loop's PPO enters the loss);
    ``kd``: ``[B, L, T - 1]``, the per-position distillation losses of the earlier loops
    (``rltt_per_loop.per_loop_kd``, with grad); ``alpha_t = 1 / (T - 1)``; ``logp_init``:
    ``[B, L, T]`` as in ``plc_policy_loss``. ``lam = 0`` is GRPO on this code path.
    """
    if logp_new.shape != logp_old.shape or logp_new.dim() != 3:
        raise ValueError("logp_new and logp_old must both have shape [B, L, T]")
    num_loops = logp_new.shape[-1]
    if tuple(kd.shape) != (*logp_new.shape[:2], num_loops - 1):
        raise ValueError(f"kd must have shape [B, L, T - 1], got {tuple(kd.shape)}")
    if init_kl_coef < 0:
        raise ValueError("init_kl_coef must be >= 0")
    if init_kl_coef > 0 and (logp_init is None or logp_init.shape != logp_new.shape):
        raise ValueError("init_kl_coef > 0 needs logp_init of shape [B, L, T]")
    if advantages.dim() == 1:
        advantages = advantages[:, None]
    adv3 = advantages.to(logp_new.dtype)[..., None]
    ppo, info = ppo_objective(logp_new, logp_old.detach(), adv3, clip_ratio, clip_ratio_low,
                              clip_ratio_high, clip_ratio_c)
    final_term = ppo[..., -1]
    alpha = 1.0 / (num_loops - 1)
    kd_term = kd.float().sum(-1) * alpha                                       # [B, L]
    loss = -agg_loss(final_term, response_mask, loss_agg_mode)
    if lam > 0:
        loss = loss + lam * agg_loss(kd_term, response_mask, loss_agg_mode)
    init_pen = init_term = None
    if init_kl_coef > 0:
        init_pen = init_anchor_penalty(logp_new, logp_init.to(logp_new.dtype), init_kl_form,
                                       init_kl_log_ratio_cap)                 # [B, L, T]
        init_term = init_pen[..., :-1].sum(-1) * alpha
        loss = loss + lam * init_kl_coef * agg_loss(init_term, response_mask, loss_agg_mode)

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
            "plc/final_term_mean": agg_loss(final_term, mask, "token-mean").item(),
            "plc/kd_term_mean": agg_loss(kd_term.detach(), mask, "token-mean").item(),
            # entry t (t < T): the earlier loop's own distillation loss D^(t)
            "plc/kd_loss_per_loop": _masked_mean_per_loop(
                kd.detach().to(mask.dtype), mask[..., None].expand_as(kd)),
        }
        if init_term is not None:
            d = (logp_new - logp_init).detach().to(mask.dtype)
            metrics["plc/init_term_mean"] = agg_loss(init_term, mask, "token-mean").item()
            metrics["plc/init_pen_per_loop"] = _masked_mean_per_loop(
                init_pen.detach().to(mask.dtype), mask3)
            metrics["plc/init_log_ratio_per_loop"] = _masked_mean_per_loop(d, mask3)
            metrics["plc/init_beyond_cap_frac_per_loop"] = _masked_mean_per_loop(
                (d.abs() > init_kl_log_ratio_cap).to(mask.dtype), mask3)
    return loss, metrics


@torch.no_grad()
def kd_stats(stats: Mapping[str, torch.Tensor], response_mask: torch.Tensor
             ) -> Dict[str, object]:
    """Per earlier loop t < T, masked means of the ``rltt_per_loop.KD_STATS`` tensors
    ``[B, L, T - 1]`` (``plc/kd_<name>/loop_t`` after ``flatten_loop_metrics``), plus the share
    of response tokens where loop t is sharper than the final loop (``entropy_gap < 0``)."""
    mask = response_mask.float()
    out: Dict[str, object] = {}
    for name, value in stats.items():
        mask3 = mask[..., None].expand_as(value)
        out[f"plc/kd_{name}_per_loop"] = _masked_mean_per_loop(value.float(), mask3)
        if name == "entropy_gap":
            out["plc/kd_below_teacher_frac_per_loop"] = _masked_mean_per_loop(
                (value < 0).float(), mask3)
    return out


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
    region. Likewise the ``loop_T`` entries of ``init_pen`` / ``init_log_ratio`` /
    ``init_beyond_cap_frac`` show how far the final loop moved from the initial model; the anchor
    weight is 0 there. A list under ``<name>_per_transition`` becomes
    ``<name>/loop_<t>_to_<t+1>``. Scalars
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
    "PLCSettings", "WEIGHT_MODES", "LOSS_AGG_MODES", "INIT_KL_FORMS", "LOOP_LOSSES",
    "KD_DIVERGENCES", "compute_loop_weights", "agg_loss", "ppo_objective", "init_anchor_penalty",
    "plc_policy_loss", "plc_kd_loss", "kd_stats",
    "cross_loop_stats", "weight_stats", "flatten_loop_metrics", "low_var_kl",
]
