"""LoopCD: contrastive decoding between an earlier loop and the last loop (no vLLM import).

Liu et al. (Apple), "Decoding Looped Transformers Better for (Almost) Free", arXiv 2610.02185.
The last loop s (= the model's ``total_ut_steps``) is guided away from an earlier reference loop r:

- ``logits`` (LoopCD-Logits, Eq. 3): z' = z_s + w (z_s - z_r). Up to a per-position constant this is
  (1 + w) log p_s - w log p_r, i.e. p' ∝ p_s (p_s / p_r)^w (paper App. A.3). The result is not
  normalized (sum_v p_s^{1+w} p_r^{-w} = exp(w D_{1+w}(p_s || p_r)) >= 1 for w > 0), so
  ``guided_logprobs`` returns log_softmax of it. Temperature and top-p are applied afterwards by the
  sampler, as in the paper ("guided logits before any temperature or truncation").
- ``hidden`` (LoopCD-Hidden, Eq. 2): h' = h_s + w (h_s - h_r) on the loop states before the final
  norm (Ouro has no coda: only the final RMSNorm separates the loop from the vocabulary
  projection), then norm and lm_head once. Fixed strength only, as in the paper.

Strength: fixed ``omega``; or ``adaptive`` (Eq. 4), where ``omega`` is the cap w_max and each
position uses w = w_max [1 - (p_(1) - p_(2))], with p_(1), p_(2) the two largest probabilities of
softmax(z_s) at temperature 1.

Paper settings for Ouro (Table A4, reference h_1 for every logit configuration): math reasoning on
Ouro-1.4B-Thinking fixed w 0.5 / adaptive cap 1.0, on Ouro-2.6B-Thinking 0.5 / 1.5; greedy answer
and code generation 0.3 / 0.5 (1.4B) and 0.3 / 1.0 (2.6B). Generation collapses beyond w ~ 0.6.

``delta_cap`` (this repository, not in the paper; default off) clips the per-token contrast
log p_s - log p_r to [-cap, cap] in the logit form, bounding how much a token that the reference
loop rules out can be boosted. With it the result is no longer the paper's LoopCD.

The vLLM model is ``rltt_baseline.vllm_loop_cd``; the settings travel in the HF config as the flat
``loop_cd_*`` keys below (via vLLM ``hf_overrides``).
"""

import math
from dataclasses import asdict, dataclass
from typing import Optional

import torch

ARCHITECTURE = "OuroLoopCDForCausalLM"
LIB = "rltt_baseline.vllm_loop_cd"
SPACES = ("logits", "hidden")
HF_KEYS = {"ref_loop": "loop_cd_ref_loop", "omega": "loop_cd_omega",
           "adaptive": "loop_cd_adaptive", "space": "loop_cd_space",
           "delta_cap": "loop_cd_delta_cap"}


@dataclass(frozen=True)
class LoopCDConfig:
    ref_loop: int = 1  # reference loop r, 1-based; the source loop is the model's last loop
    omega: float = 0.5  # fixed strength w, or the cap w_max when ``adaptive``
    adaptive: bool = False
    space: str = "logits"
    delta_cap: Optional[float] = None

    def validate(self, total_loops: int) -> "LoopCDConfig":
        if self.space not in SPACES:
            raise ValueError(f"loop_cd space must be one of {SPACES}, got {self.space!r}")
        if not 1 <= self.ref_loop < total_loops:
            raise ValueError(f"loop_cd ref_loop must be in [1, {total_loops - 1}] for a model with "
                             f"{total_loops} loops (the source is the last loop), got "
                             f"{self.ref_loop}")
        if not math.isfinite(self.omega):
            raise ValueError(f"loop_cd omega must be finite, got {self.omega}")
        if self.adaptive and self.omega < 0:
            raise ValueError("loop_cd adaptive needs omega (the cap) >= 0")
        if self.space == "hidden" and (self.adaptive or self.delta_cap is not None):
            raise ValueError("loop_cd space=hidden supports fixed strength only (no adaptive, "
                             "no delta_cap), as in the paper")
        if self.delta_cap is not None and not self.delta_cap > 0:
            raise ValueError(f"loop_cd delta_cap must be > 0 or None, got {self.delta_cap}")
        return self

    def hf_overrides(self) -> dict:
        return {HF_KEYS[k]: v for k, v in asdict(self).items()}


def from_hf_config(config) -> LoopCDConfig:
    """Read the ``loop_cd_*`` keys of an HF config and validate against its loop count."""
    values = {k: getattr(config, key) for k, key in HF_KEYS.items() if hasattr(config, key)}
    cfg = LoopCDConfig(**values)
    return cfg.validate(int(getattr(config, "total_ut_steps", 4)))


def adaptive_omega(z_s: torch.Tensor, omega_max: float) -> torch.Tensor:
    """Per-position strength w_max [1 - (p_(1) - p_(2))] from the source logits, shape (..., 1)."""
    top2 = torch.softmax(z_s.float(), dim=-1).topk(2, dim=-1).values
    return omega_max * (1.0 - (top2[..., :1] - top2[..., 1:2]))


def guided_logprobs(z_s: torch.Tensor, z_r: torch.Tensor, cfg: LoopCDConfig) -> torch.Tensor:
    """Normalized log-probabilities of LoopCD-Logits (fp32), from raw source/reference logits."""
    log_s = torch.log_softmax(z_s.float(), dim=-1)
    log_r = torch.log_softmax(z_r.float(), dim=-1)
    # the contrast uses normalized log-probs: raw z_s - z_r carries a per-position offset
    # log Z_s - log Z_r, which cancels in the softmax but would shift the delta_cap window
    delta = log_s - log_r
    if cfg.delta_cap is not None:
        delta = delta.clamp(-cfg.delta_cap, cfg.delta_cap)
    omega = adaptive_omega(z_s, cfg.omega) if cfg.adaptive else cfg.omega
    return torch.log_softmax(log_s + omega * delta, dim=-1)


def guided_hidden(h_s: torch.Tensor, h_r: torch.Tensor, omega: float) -> torch.Tensor:
    """LoopCD-Hidden state h_s + w (h_s - h_r) (pre-norm loop states)."""
    return h_s + omega * (h_s - h_r)


def run_loops(hidden: torch.Tensor, positions: torch.Tensor, layers, norm, total_loops: int,
              cfg: LoopCDConfig) -> torch.Tensor:
    """The loop of vLLM's ``OuroModel.forward`` with LoopCD's readout (used by the vLLM model).

    ``layers`` follow vLLM's ``layer(positions, hidden, ut, residual) -> (hidden, residual)`` and
    ``norm(x, residual)`` is vLLM's fused add + RMSNorm, which may work in place and returns
    ``(norm(x + residual), x + residual)``. Returns ``cat([h_s, h_r], -1)`` (post-norm states) for
    ``logits`` and ``norm(h_s + w (h_s - h_r))`` on the pre-norm states for ``hidden``.
    """
    ref = None
    for ut in range(total_loops):
        residual = None
        for layer in layers:
            hidden, residual = layer(positions, hidden, ut, residual)
        hidden, pre_norm = norm(hidden, residual)
        if ut == cfg.ref_loop - 1:
            # clone: the next loop's layers reuse these buffers in place
            ref = (pre_norm if cfg.space == "hidden" else hidden).clone()
    if cfg.space == "hidden":
        return norm(guided_hidden(pre_norm, ref, cfg.omega))
    return torch.cat([hidden, ref], dim=-1)


def register() -> None:
    """Register the vLLM architecture lazily (``module:class``), safe before CUDA init."""
    from vllm import ModelRegistry

    if ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(ARCHITECTURE, f"{LIB}:OuroLoopCDForCausalLM")


def engine_overrides(cfg: Optional[LoopCDConfig], total_loops: int) -> dict:
    """``hf_overrides`` for the vLLM engine; stock Ouro architecture when ``cfg`` is None."""
    overrides = {"total_ut_steps": int(total_loops)}
    if cfg is not None:
        cfg.validate(int(total_loops))
        overrides.update({"architectures": [ARCHITECTURE], **cfg.hf_overrides()})
    return overrides
