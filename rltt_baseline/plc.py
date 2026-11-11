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

GRPO with an off-policy loop auxiliary term (``objective = additive``, ``weight_mode =
is_trunc``). The final loop T is the behaviour policy of the earlier loops, whose updates are a
truncated-importance-weighted PPO surrogate plus a forward-KL trust region towards the final loop:

    loss = -agg_j[ PPO_j^(T)(A) ]                                          (GRPO, unchanged)
           + lam * sum_{t<T} alpha_t * ( -agg'_j[ rho[j, t] PPO_j^(t)(A~) ]   (surrogate)
                                         + kl_coef * agg'_j[ k1[j, t] ] )   (trust region)
    rho[j, t] = min(trunc_cap, P_old^(t)(y_j) / P_old^(T)(y_j)),  stop-gradient
    k1[j, t]  = log P_old^(T)(y_j) - log P^(t)(y_j)

``alpha`` is the prior restricted to t < T and renormalised. ``k1`` is the sampled estimate of
``KL(P^(T) || P^(t))`` on the final loop's own samples (unbiased, no ratio); its gradient is the
cross-entropy of loop t on the rollout tokens. ``trunc_cap`` bounds the ratio from above only
(variance); ``kl_coef`` sets how far loop t may move from the final loop. With one update per
step the per-token gradient on loop t is ``-(lam / N) alpha_t (rho A + kl_coef) d log P^(t)``.
``kl_coef = 0`` is the surrogate alone.

``A~ = A`` unless ``loop_baseline = group``: ``A~[i, t] = A_i - b[g(i), t]`` with
``b[g, t] = sum_{i in g} A_i m_it / sum_{i in g} m_it``, ``m_it`` the weight mass of response i at
loop t under the loop aggregation (token mean of ``w`` for seq-mean-token-mean, token sum for
token-mean), which makes the weighted advantage mass of every group zero at every loop.
``weight_mode = early_prior`` drops the ratio (``rho = 1``). The final-loop term is GRPO
unchanged (``lam = 0`` is GRPO) and the gate gets no gradient.
"""

import math
from dataclasses import asdict, dataclass, fields
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch

WEIGHT_MODES = ("rltt", "posterior", "uniform", "logp_softmax", "is_trunc", "early_prior")
# weight modes of the additive loop term: earlier loops only, the final loop has weight 0
EARLY_LOOP_WEIGHT_MODES = ("is_trunc", "early_prior")
PRIOR_FREE_WEIGHT_MODES = ("uniform", "logp_softmax")
PRIORS = ("uniform", "progressive", "exit")
OBJECTIVES = ("mix", "additive")
LOOP_BASELINES = ("none", "group")
LOSS_AGG_MODES = ("token-mean", "seq-mean-token-sum", "seq-mean-token-mean",
                  "seq-mean-token-sum-norm")
# loop aggregations under which the group baseline zeroes the weighted advantage mass exactly
BASELINE_AGG_MODES = ("token-mean", "seq-mean-token-mean")
LOG_EPS = 1e-8


@dataclass(frozen=True)
class PLCSettings:
    """The ``plc`` config section; ``from_mapping`` validates a dict of user values."""

    enabled: bool = False
    weight_mode: str = "posterior"      # rltt | posterior | uniform | logp_softmax | is_trunc |
                                        # early_prior
    prior: str = "uniform"              # uniform | progressive | exit
    progressive_alpha: float = 1.0      # prior=progressive: q_t ∝ t^alpha
    gamma: float = 1.0                  # posterior temperature; 0 = RLTT
    lam: float = 1.0                    # loop term coefficient (mix: its share); 0 = GRPO
    objective: str = "mix"              # mix: (1-lam) final + lam loop | additive: final + lam loop
    loop_baseline: str = "none"         # none | group: per-group, per-loop advantage baseline
    loop_agg_mode: str = "same"         # aggregation of the loop term; same = actor loss_agg_mode
    trunc_cap: float = 1.0              # is_trunc: upper cap c of the cross-loop ratio (>= 1)
    kl_coef: float = 0.0                # early-loop modes: forward-KL trust region to final loop
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
        if self.objective not in OBJECTIVES:
            raise ValueError(f"plc.objective must be one of {OBJECTIVES}")
        if self.loop_baseline not in LOOP_BASELINES:
            raise ValueError(f"plc.loop_baseline must be one of {LOOP_BASELINES}")
        if self.loop_agg_mode != "same" and self.loop_agg_mode not in LOSS_AGG_MODES:
            raise ValueError(f"plc.loop_agg_mode must be 'same' or one of {LOSS_AGG_MODES}")
        baseline_modes = ("same",) + BASELINE_AGG_MODES
        if self.loop_baseline == "group" and self.loop_agg_mode not in baseline_modes:
            raise ValueError(f"plc.loop_baseline=group needs plc.loop_agg_mode in "
                             f"{BASELINE_AGG_MODES} (or same)")
        if self.lam < 0 or (self.objective == "mix" and self.lam > 1.0):
            raise ValueError("plc.lam must be in [0, 1] (objective=mix) or >= 0 (additive)")
        if self.weight_mode in EARLY_LOOP_WEIGHT_MODES:
            # alpha comes from a static prior; is_trunc already is an importance weight
            mode = f"plc.weight_mode={self.weight_mode}"
            if self.prior not in ("uniform", "progressive"):
                raise ValueError(f"{mode} needs plc.prior uniform or progressive")
            if self.objective != "additive":
                raise ValueError(f"{mode} needs plc.objective=additive "
                                 "(its loop term has no final-loop weight)")
            if self.eps_floor != 0.0 or self.use_is_correction:
                raise ValueError(f"{mode} needs eps_floor 0 and use_is_correction false")
        if self.trunc_cap < 1.0:
            raise ValueError("plc.trunc_cap must be >= 1 (it caps the ratio from above only)")
        if self.trunc_cap != 1.0 and self.weight_mode != "is_trunc":
            raise ValueError("plc.trunc_cap only applies to plc.weight_mode=is_trunc")
        if self.kl_coef < 0:
            raise ValueError("plc.kl_coef must be >= 0")
        if self.kl_coef > 0 and self.weight_mode not in EARLY_LOOP_WEIGHT_MODES:
            raise ValueError(f"plc.kl_coef > 0 needs plc.weight_mode in {EARLY_LOOP_WEIGHT_MODES} "
                             "(the trust region of the earlier loops towards the final loop)")
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
                f"lam={self.lam} objective={self.objective} loop_baseline={self.loop_baseline} "
                f"loop_agg_mode={self.loop_agg_mode} trunc_cap={self.trunc_cap} "
                f"kl_coef={self.kl_coef} eps_floor={self.eps_floor} "
                f"gate_loss={self.use_gate_loss} gate_coef={self.gate_coef} "
                f"gate_entropy_coef={self.gate_entropy_coef} "
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
                         weight_mode: str = "posterior", eps_floor: float = 0.0,
                         trunc_cap: float = 1.0) -> torch.Tensor:
    """``w[b, j, t]``, detached; ``sum_t = 1`` except for ``is_trunc`` and ``early_prior``.

    ``ell_old``: ``[B, L, T]`` old-policy per-loop log-probs of the sampled token;
    ``logq``: broadcastable to it (``uniform`` and ``logp_softmax`` ignore it and ``gamma``).
    ``is_trunc``: ``w[.., t] = alpha_t * min(c, exp(ell_old[.., t] - ell_old[.., T]))`` for
    t < T with ``alpha = softmax(logq[.., :T-1])`` (the prior restricted to t < T),
    ``c = trunc_cap`` and ``w[.., T] = 0``; ``gamma`` is ignored. Rows do not sum to 1.
    ``early_prior``: the same without the ratio, ``w[.., t] = alpha_t`` for t < T and 0 for the
    final loop.
    """
    if trunc_cap < 1.0:
        raise ValueError("trunc_cap must be >= 1")
    ell_old = ell_old.detach().float()
    if weight_mode in EARLY_LOOP_WEIGHT_MODES:
        alpha = torch.softmax(logq.expand_as(ell_old)[..., :-1].float(), dim=-1)
        if weight_mode == "is_trunc":
            log_ratio = (ell_old[..., :-1] - ell_old[..., -1:]).clamp(max=math.log(trunc_cap))
            alpha = alpha * torch.exp(log_ratio)
        w = torch.cat([alpha, torch.zeros_like(ell_old[..., -1:])], dim=-1)
        return w.detach()
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


def sequence_weight_mass(weights: torch.Tensor, response_mask: torch.Tensor,
                         loop_agg_mode: str) -> torch.Tensor:
    """``m[i, t]``: the share of response i's tokens in loop t's term, up to a factor that is
    the same for every response of the step. ``[B, L, T]`` weights -> ``[B, T]``.

    seq-mean-token-mean weighs response i by ``1 / L_i`` (token mean of ``w``); token-mean
    weighs every token alike (token sum of ``w``).
    """
    mask = response_mask.to(weights.dtype)[..., None]
    total = (weights * mask).sum(dim=1)
    if loop_agg_mode == "token-mean":
        return total
    if loop_agg_mode == "seq-mean-token-mean":
        return total / mask.sum(dim=1).clamp_min(1.0)
    raise ValueError(f"the group baseline needs loop_agg_mode in {BASELINE_AGG_MODES}, "
                     f"got {loop_agg_mode!r}")


def sequence_advantages(advantages: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """``A_i`` from verl's token-level outcome advantages ``[B, L]`` (constant on the response)."""
    mask = response_mask.to(advantages.dtype)
    return (advantages * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)


def group_loop_baseline(seq_adv: torch.Tensor, mass: torch.Tensor,
                        group_ids: Sequence) -> torch.Tensor:
    """``b[i, t] = sum_{k in g(i)} A_k m_kt / sum_{k in g(i)} m_kt``, shape ``[B, T]``.

    With ``A~[i, t] = A_i - b[i, t]`` the weighted advantage mass ``sum_{i in g} m_it A~[i, t]``
    of every group is zero at every loop. A group whose mass at a loop is zero gets ``b = 0``.
    """
    if len(group_ids) != seq_adv.shape[0] or mass.shape[0] != seq_adv.shape[0]:
        raise ValueError("seq_adv, mass and group_ids must cover the same responses")
    index: Dict[object, int] = {}
    groups = torch.tensor([index.setdefault(g, len(index)) for g in group_ids],
                          dtype=torch.long, device=seq_adv.device)
    mass = mass.double()
    adv = seq_adv.double()[:, None]
    num = torch.zeros(len(index), mass.shape[1], dtype=torch.float64, device=mass.device)
    den = torch.zeros_like(num)
    num.index_add_(0, groups, adv * mass)
    den.index_add_(0, groups, mass)
    b = torch.where(den > 0, num / den.clamp_min(1e-300), torch.zeros_like(num))
    return b[groups].to(seq_adv.dtype)


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
                    clip_ratio_c: float = 3.0, is_weights: Optional[torch.Tensor] = None,
                    objective: str = "mix", loop_advantages: Optional[torch.Tensor] = None,
                    loop_agg_mode: Optional[str] = None, kl_coef: float = 0.0,
                    kl_weights: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, Dict[str, object]]:
    """The PLC policy loss and its metrics.

    ``objective = mix``:      ``-agg_j[(1 - lam) PPO^(T) + lam sum_t w PPO^(t)]``
    ``objective = additive``: ``-agg_j[PPO^(T)] - lam * agg'_j[sum_t w PPO^(t)]``
                              ``+ lam * kl_coef * agg'_j[sum_t kl_weights * k1^(t)]``

    ``agg`` uses ``loss_agg_mode`` and ``agg'`` uses ``loop_agg_mode`` (default: the same).
    ``logp_new`` / ``logp_old``: ``[B, L, T]``; ``advantages``: ``[B]`` or ``[B, L]``, used by
    the final-loop term; ``loop_advantages``: optional ``[B, L, T]`` advantages of the loop term
    (default: ``advantages`` for every loop); ``response_mask``: ``[B, L]``; ``weights``:
    ``[B, L, T]`` (treated as constants); ``is_weights``: optional ``[B, L]`` factor on the loop
    term. The trust region (``kl_coef > 0``, additive only) uses
    ``k1^(t) = logp_old[.., T] - logp_new[.., t]``, the sampled ``KL(P_old^(T) || P^(t))`` on the
    final loop's samples, weighted by ``kl_weights`` ``[B, L, T]`` (the prior on t < T, 0 for the
    final loop).
    """
    if logp_new.shape != logp_old.shape or logp_new.dim() != 3:
        raise ValueError("logp_new and logp_old must both have shape [B, L, T]")
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    if kl_coef < 0:
        raise ValueError("kl_coef must be >= 0")
    if kl_coef > 0:
        if objective != "additive":
            raise ValueError("kl_coef > 0 needs objective=additive")
        if kl_weights is None or kl_weights.shape != logp_new.shape:
            raise ValueError("kl_coef > 0 needs kl_weights of shape [B, L, T]")
    loop_mode = loss_agg_mode if loop_agg_mode in (None, "same") else loop_agg_mode
    if advantages.dim() == 1:
        advantages = advantages[:, None]
    adv3 = advantages.to(logp_new.dtype)[..., None]
    weights = weights.detach().to(logp_new.dtype)
    clip = (clip_ratio, clip_ratio_low, clip_ratio_high, clip_ratio_c)
    ppo, info = ppo_objective(logp_new, logp_old.detach(), adv3, *clip)
    if loop_advantages is None:
        loop_ppo, loop_info = ppo, info
    else:
        if loop_advantages.shape != logp_new.shape:
            raise ValueError("loop_advantages must have shape [B, L, T]")
        loop_ppo, loop_info = ppo_objective(logp_new, logp_old.detach(),
                                            loop_advantages.detach().to(logp_new.dtype), *clip)
    loop_term = (weights * loop_ppo).sum(-1)
    if is_weights is not None:
        loop_term = loop_term * is_weights.detach().to(loop_term.dtype)
    final_term = ppo[..., -1]
    if objective == "mix" and loop_mode == loss_agg_mode:
        per_token = (1.0 - lam) * final_term + lam * loop_term
        loss = -agg_loss(per_token, response_mask, loss_agg_mode)
    else:
        final_coef = (1.0 - lam) if objective == "mix" else 1.0
        loss = -(final_coef * agg_loss(final_term, response_mask, loss_agg_mode)
                 + lam * agg_loss(loop_term, response_mask, loop_mode))
    k1 = kl_term = None
    if kl_coef > 0:
        k1 = logp_old.detach()[..., -1:] - logp_new                       # [B, L, T]
        kl_term = (kl_weights.detach().to(k1.dtype) * k1).sum(-1)          # [B, L]
        loss = loss + lam * kl_coef * agg_loss(kl_term, response_mask, loop_mode)

    with torch.no_grad():
        mask = response_mask.to(ppo.dtype)
        mask3 = mask[..., None].expand_as(ppo)
        metrics = {
            "actor/pg_clipfrac": agg_loss(info["clipped"][..., -1], mask, "token-mean").item(),
            "actor/ppo_kl": agg_loss(-info["negative_approx_kl"][..., -1], mask,
                                     "token-mean").item(),
            "actor/pg_clipfrac_lower": agg_loss(info["clipped_lower"][..., -1], mask,
                                                "token-mean").item(),
            "plc/clipfrac_per_loop": _masked_mean_per_loop(loop_info["clipped"], mask3),
            "plc/ratio_mean_per_loop": _masked_mean_per_loop(loop_info["ratio"], mask3),
            "plc/loop_term_mean": agg_loss(loop_term, mask, "token-mean").item(),
            "plc/final_term_mean": agg_loss(final_term, mask, "token-mean").item(),
        }
        if is_weights is not None:
            metrics["plc/is_weight_mean"] = agg_loss(is_weights.float(), mask, "token-mean").item()
        if loop_advantages is not None:
            metrics["plc/loop_adv_abs_mean_per_loop"] = _masked_mean_per_loop(
                loop_advantages.abs().to(mask.dtype), mask3)
        if kl_term is not None:
            # k1 is the loss's estimate; k3 = r - 1 - log r (r = P^(t) / P_old^(T)) estimates the
            # same KL with lower variance and is >= 0 per token, for monitoring only
            log_r = (-k1).clamp(-20.0, 20.0)
            k3 = torch.exp(log_r) - 1.0 - log_r
            metrics["plc/kl_term_mean"] = agg_loss(kl_term, mask, "token-mean").item()
            metrics["plc/kl_k1_per_loop"] = _masked_mean_per_loop(k1.to(mask.dtype), mask3)
            metrics["plc/kl_k3_per_loop"] = _masked_mean_per_loop(k3.to(mask.dtype), mask3)
    return loss, metrics


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
    "PLCSettings", "WEIGHT_MODES", "EARLY_LOOP_WEIGHT_MODES", "PRIOR_FREE_WEIGHT_MODES", "PRIORS",
    "LOSS_AGG_MODES",
    "OBJECTIVES", "LOOP_BASELINES", "BASELINE_AGG_MODES",
    "loop_prior_logq", "compute_loop_weights", "importance_weights",
    "sequence_weight_mass", "sequence_advantages", "group_loop_baseline",
    "agg_loss", "ppo_objective", "plc_policy_loss", "cross_loop_stats", "gate_loss",
    "weight_stats", "low_var_kl", "entropy_from_log_probs",
]
