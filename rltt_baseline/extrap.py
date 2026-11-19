"""EP-GRPO: on-policy GRPO on the extrapolated (LoopCD) policy of Ouro.

Pure torch, no verl dependency (``tests/test_extrap.py``). ``rltt_baseline.extrap_workers`` wires
it into the verl 0.6.1 actor, ``rltt_baseline.verl09_losses`` into the verl 0.9 engine. Design:
``research/extrapolated_rl_loss_design_2026-11-19.md`` (not tracked).

Policy. With l_t = log P^(t) (temperature 1), the last loop s = T and a reference loop r < T,

    pi_w = softmax( ((1 + w) l_s - w sg[l_r]) / temperature )

which is LoopCD-Logits (Liu et al., arXiv 2610.02185, ``rltt_baseline.loop_cd``) followed by the
sampler's temperature. vLLM samples the rollouts from pi_w (``rltt_baseline.vllm_loop_cd``), so the
objective is plain on-policy GRPO on pi_w:

    loss = -agg_j[ PPO_j(c_j A) ],   ratio = pi_w^new(y_j) / pi_w^old(y_j)

- The gradient on the last loop's logits is (1 + w) c A (e_y - pi_w) / temperature, centred on the
  sampling distribution (E_{y ~ pi_w}[e_y - pi_w] = 0): no prefix-dependent drift of the kind that
  off-policy cross-loop training has (PLC design note, section 2).
- ``w = 0`` is GRPO on the last loop, on this code path.
- The reference loop is stop-gradient by default (``ref_grad: false``): with a gradient, a positive
  advantage lowers the sampled token at loop r (gradient -w A (e_y - pi_w)), so pi_w improves by
  making the weak loop weaker ("contrast hacking"). ``ref_grad: true`` is the ablation.

Per-token weight c_j (stop-gradient, fixed within the update; 1 with the default margin_alpha 0):

    c_j = 1 + margin_alpha * g_j,   g_j = 1 - (p_(1) - p_(2))   (last loop top-2 margin, T 1)

The (1 + w) factor of the gradient is kept (user 2026-11-19: no 1 / (1 + w) normalization). With a
fixed w it scales the whole gradient by a constant, which AdamW (and the grad-norm clip) largely
cancel; with ``adaptive`` it also gives contested positions (large w_j) more weight.
``margin_alpha`` concentrates learning on contested positions; it
depends on the position only, so it adds no drift (unlike a token weight such as Qwen's
1 + alpha (1 - pi(y)), arXiv 2603.22117). ``adaptive`` uses w_j = w * g_j per position (LoopCD
Eq. 4) in both the rollout and the loss. ``delta_cap`` clips l_s - l_r (not in the paper).
"""

from dataclasses import asdict, dataclass, fields
from typing import Dict, Mapping, Optional, Tuple

import torch

from rltt_baseline import loop_cd, plc


@dataclass(frozen=True)
class ExtrapSettings:
    """The ``extrap`` config section; ``from_mapping`` validates a dict of user values."""

    enabled: bool = False
    omega: float = 0.5              # strength w (the cap w_max with ``adaptive``); 0 = GRPO
    adaptive: bool = False          # per-position w * (1 - top-2 margin), LoopCD Eq. 4
    ref_loop: int = 1               # reference loop r (1-based); the source is the last loop
    ref_grad: bool = False          # gradient through the reference loop (ablation)
    margin_alpha: float = 0.0       # position weight 1 + alpha * (1 - top-2 margin); 0 = off
    delta_cap: Optional[float] = None  # clip l_s - l_r to [-cap, cap]; None = paper's LoopCD

    @classmethod
    def from_mapping(cls, values: Optional[Mapping]) -> "ExtrapSettings":
        values = dict(values or {})
        names = {f.name for f in fields(cls)}
        unknown = set(values) - names
        if unknown:
            raise KeyError(f"unknown extrap keys {sorted(unknown)}; allowed {sorted(names)}")
        merged = {**asdict(cls()), **values}
        for name in ("enabled", "adaptive", "ref_grad"):
            if not isinstance(merged[name], bool):
                raise TypeError(f"extrap.{name} must be a bool, got {merged[name]!r}")
        for name in ("omega", "margin_alpha"):
            value = merged[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"extrap.{name} must be a number, got {value!r}")
            merged[name] = float(value)
        ref = merged["ref_loop"]
        if isinstance(ref, bool) or not isinstance(ref, int):
            raise TypeError(f"extrap.ref_loop must be an integer, got {ref!r}")
        cap = merged["delta_cap"]
        if cap is not None:
            if isinstance(cap, bool) or not isinstance(cap, (int, float)):
                raise TypeError(f"extrap.delta_cap must be a number or null, got {cap!r}")
            merged["delta_cap"] = float(cap)
        settings = cls(**merged)
        settings.validate()
        return settings

    def validate(self, total_loops: Optional[int] = None) -> None:
        if self.omega < 0:
            raise ValueError("extrap.omega must be >= 0 (negative values interpolate towards "
                             "the reference loop)")
        if self.margin_alpha < 0:
            raise ValueError("extrap.margin_alpha must be >= 0")
        if self.ref_loop < 1:
            raise ValueError("extrap.ref_loop must be >= 1")
        if total_loops is not None and self.ref_loop >= total_loops:
            raise ValueError(f"extrap.ref_loop must be < the {total_loops} loops of the model")
        if self.delta_cap is not None and not self.delta_cap > 0:
            raise ValueError("extrap.delta_cap must be > 0 or null")

    def loop_cd(self) -> loop_cd.LoopCDConfig:
        """The rollout's LoopCD settings (logit form; same policy as the loss)."""
        return loop_cd.LoopCDConfig(ref_loop=self.ref_loop, omega=self.omega,
                                    adaptive=self.adaptive, space="logits",
                                    delta_cap=self.delta_cap)

    def model_kwargs(self) -> dict:
        """Keyword arguments of the per-loop model's forward (``modeling_ouro.py``)."""
        return {"return_extrap_log_probs": True, "extrap_omega": self.omega,
                "extrap_ref_loop": self.ref_loop, "extrap_adaptive": self.adaptive,
                "extrap_delta_cap": self.delta_cap, "extrap_ref_grad": self.ref_grad}

    def describe(self) -> str:
        return (f"omega={self.omega} adaptive={self.adaptive} ref_loop={self.ref_loop} "
                f"ref_grad={self.ref_grad} margin_alpha={self.margin_alpha} "
                f"delta_cap={self.delta_cap}")


def token_weights(strength: torch.Tensor, margin: torch.Tensor,
                  settings: ExtrapSettings) -> torch.Tensor:
    """Per-token weight c_j = 1 + margin_alpha * margin (stop-gradient). ``strength`` only fixes
    the shape and device (the (1 + w) factor of the gradient is not normalized away)."""
    return torch.ones_like(strength, dtype=torch.float32) + settings.margin_alpha * \
        margin.detach().float()


def extrap_policy_loss(log_prob: torch.Tensor, old_log_prob: torch.Tensor,
                       advantages: torch.Tensor, response_mask: torch.Tensor,
                       weights: torch.Tensor, clip_ratio: float,
                       loss_agg_mode: str = "token-mean", clip_ratio_low: Optional[float] = None,
                       clip_ratio_high: Optional[float] = None, clip_ratio_c: float = 3.0,
                       row_div: Optional[torch.Tensor] = None
                       ) -> Tuple[torch.Tensor, Dict[str, object]]:
    """``-agg[PPO(c A)]`` on the extrapolated policy's ``[B, L]`` log-probs.

    The weight is folded into the advantage: c > 0, so the clipped PPO objective of c A is c times
    that of A (dual clip included). ``weights`` all 1 is ``plc.grpo_policy_loss``.
    """
    if advantages.dim() == 1:
        advantages = advantages[:, None]
    if (weights <= 0).any():
        raise ValueError("extrap token weights must be positive")
    weighted = advantages.to(log_prob.dtype) * weights.detach().to(log_prob.dtype)
    return plc.grpo_policy_loss(log_prob, old_log_prob, weighted, response_mask, clip_ratio,
                                loss_agg_mode=loss_agg_mode, clip_ratio_low=clip_ratio_low,
                                clip_ratio_high=clip_ratio_high, clip_ratio_c=clip_ratio_c,
                                row_div=row_div)


@torch.no_grad()
def extrap_stats(log_prob: torch.Tensor, entropy: torch.Tensor, strength: torch.Tensor,
                 margin: torch.Tensor, weights: torch.Tensor, last_loop_log_prob: torch.Tensor,
                 response_mask: torch.Tensor) -> Dict[str, float]:
    """Token means over the response: the policy's entropy, strength, margin, weight, and how far
    the sampled tokens' log-prob under the policy is from the last loop alone."""
    mask = response_mask.float()
    denom = mask.sum().clamp_min(1.0)

    def mean(x):
        return float((x.float() * mask).sum() / denom)

    return {
        "extrap/entropy": mean(entropy),
        "extrap/omega_mean": mean(strength),
        "extrap/margin_mean": mean(margin),
        "extrap/weight_mean": mean(weights),
        "extrap/log_prob_minus_last_loop": mean(log_prob - last_loop_log_prob),
    }
