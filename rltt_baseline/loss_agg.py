"""Loss aggregation that does not depend on how verl cuts a mini-batch into micro-batches.

Pure torch, no verl dependency (unit-tested on CPU in ``tests/test_loss_agg.py``). The config
section ``loss_agg`` selects the mode; ``rltt_baseline.plc_workers`` (PLC) and
``rltt_baseline.grpo_workers`` (GRPO) apply it.

verl 0.6.1 aggregates each micro-batch on its own (``loss_agg_mode``; ``token-mean`` divides by
the micro-batch's own token count), scales it by ``B / N`` before ``backward`` (``B`` rows in the
micro-batch, ``N`` rows of the mini-batch on this rank; dynamic batch) and FSDP averages the
gradients of the ``R`` data-parallel ranks. With ``token-mean`` a token's weight therefore depends
on which rows share its micro-batch. The row-normalized modes divide each response ``i`` by a
divisor ``d_i`` that depends on that response alone and aggregate a micro-batch as

    agg = (1 / B) * sum_i S_i / d_i,        S_i = sum_j loss[i, j] * mask[i, j]

so after verl's ``B / N`` scaling and the FSDP mean the update minimises
``sum_i S_i / d_i / (N R)`` over the whole mini-batch, whatever the micro-batch split. Because
``d_i`` is a function of row ``i`` only, every micro-batch computes it from its own
``response_mask``; nothing has to be prepared before the split.

Modes (``N R`` responses per mini-batch, ``L_i`` response tokens of response ``i``):

``token-mean``
    verl's own per-micro-batch aggregation (default; the loss is unchanged bit for bit).
``drgrpo``
    ``d_i = norm_len`` (Dr. GRPO, arXiv 2503.20783, constant ``M``):
    ``loss = 1 / (N R M) * sum_i S_i``. Every token has the same weight, independent of the
    lengths of the responses.
``seq-mean-token-mean``
    ``d_i = L_i``: ``loss = 1 / (N R) * sum_i S_i / L_i``, each response averaged over its own
    tokens, then a mean over all responses. This is the original GRPO objective (DeepSeekMath,
    arXiv 2402.03300: ``1/G sum_i 1/|o_i| sum_t``) and verl's mode of the same name; Dr. GRPO
    points out its length bias (a long wrong response gets a smaller per-token penalty). A row
    without response tokens adds 0 and still counts in ``N R`` (verl would skip it; with EOS in
    the mask a response always has a token).

Logged loss (``loss/<term>``, ``loss/total``). verl's ``actor/pg_loss`` is not the optimized
loss: each micro-batch appends a value and verl takes the mean of the list over micro-batches and
ranks (verl's actor appends ``pg_loss * B / N``, i.e. the loss divided by the number of
micro-batches; PLC appends the unscaled micro-batch value). The actors therefore also sum, per
mini-batch, every loss term times the ``B / N`` scale it gets before ``backward``
(``LossTracker``): one value per rank, the exact objective of that rank's update; verl's mean over
the ranks is then the objective of the whole update (FSDP averages the ranks' gradients).
``loss/total`` is the loss that is backpropagated; the terms sum to it.
"""

from dataclasses import asdict, dataclass, fields
from typing import Dict, Mapping, MutableMapping, Optional

import torch

LOSS_AGG_MODES = ("token-mean", "drgrpo", "seq-mean-token-mean")
# per-micro-batch metric keys of the loss terms (signed, coefficients included), collected by
# LossTracker and removed from the per-micro-batch metrics
LOSS_TERM_PREFIX = "loss_term/"


@dataclass(frozen=True)
class LossAggSettings:
    """The ``loss_agg`` config section; ``from_mapping`` validates a dict of user values."""

    mode: str = "token-mean"           # one of LOSS_AGG_MODES
    norm_len: Optional[int] = None     # drgrpo: the constant M (tokens), e.g. 16384

    @classmethod
    def from_mapping(cls, values: Optional[Mapping]) -> "LossAggSettings":
        values = dict(values or {})
        names = {f.name for f in fields(cls)}
        unknown = set(values) - names
        if unknown:
            raise KeyError(f"unknown loss_agg keys {sorted(unknown)}; allowed {sorted(names)}")
        merged = {**asdict(cls()), **values}
        if not isinstance(merged["mode"], str):
            raise TypeError(f"loss_agg.mode must be a string, got {merged['mode']!r}")
        norm_len = merged["norm_len"]
        if norm_len is not None:
            if isinstance(norm_len, bool) or not isinstance(norm_len, (int, float)) \
                    or float(norm_len) != int(norm_len):
                raise TypeError(f"loss_agg.norm_len must be an integer, got {norm_len!r}")
            merged["norm_len"] = int(norm_len)
        settings = cls(**merged)
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.mode not in LOSS_AGG_MODES:
            raise ValueError(f"loss_agg.mode must be one of {LOSS_AGG_MODES}, got {self.mode!r}")
        if self.mode == "drgrpo":
            if self.norm_len is None or self.norm_len <= 0:
                raise ValueError("loss_agg.mode=drgrpo needs loss_agg.norm_len > 0 (the constant "
                                 "M of Dr. GRPO, e.g. the maximum response length 16384)")
        elif self.norm_len is not None:
            raise ValueError("loss_agg.norm_len only applies to loss_agg.mode=drgrpo")

    @property
    def row_normalized(self) -> bool:
        """True when the actors replace verl's per-micro-batch aggregation."""
        return self.mode != "token-mean"

    def describe(self) -> str:
        if self.mode == "drgrpo":
            return f"loss_agg=drgrpo norm_len={self.norm_len}"
        return f"loss_agg={self.mode}"


def row_divisors(settings: LossAggSettings, response_mask: torch.Tensor) -> torch.Tensor:
    """``d_i`` ``[B]`` (float32, on ``response_mask``'s device) for a ``[B, L]`` micro-batch."""
    if not settings.row_normalized:
        raise ValueError("loss_agg.mode=token-mean has no row divisors (verl aggregates itself)")
    if settings.mode == "drgrpo":
        return torch.full(response_mask.shape[:1], float(settings.norm_len), dtype=torch.float32,
                          device=response_mask.device)
    # seq-mean-token-mean: L_i; an empty row has S_i = 0 whatever its divisor
    return response_mask.detach().to(torch.float32).sum(-1).clamp_min(1.0)


def agg_rows(loss_mat: torch.Tensor, loss_mask: torch.Tensor,
             row_div: torch.Tensor) -> torch.Tensor:
    """``(1 / B) * sum_i (sum_j loss[i, j] mask[i, j]) / d_i`` for a ``[B, L]`` micro-batch."""
    if row_div.shape != loss_mat.shape[:1]:
        raise ValueError(f"row_div must have shape [{loss_mat.shape[0]}], got "
                         f"{tuple(row_div.shape)}")
    mask = loss_mask.to(loss_mat.dtype)
    # float32 at least: d_i is up to 16384 and the sums run over thousands of tokens
    dtype = torch.promote_types(loss_mat.dtype, torch.float32)
    per_row = (loss_mat * mask).sum(-1).to(dtype) / row_div.to(device=loss_mat.device,
                                                                dtype=dtype)
    return per_row.sum() / max(loss_mat.shape[0], 1)


def loss_term_metrics(terms: Mapping[str, torch.Tensor]) -> Dict[str, float]:
    """``loss_term/<name>`` floats of the loss terms of one micro-batch (before verl's scale)."""
    return {LOSS_TERM_PREFIX + name: float(value.detach()) for name, value in terms.items()}


def pop_loss_terms(metrics: MutableMapping[str, object]) -> Dict[str, float]:
    """Remove the ``loss_term/*`` entries from a micro-batch's metrics and return them by name."""
    return {key[len(LOSS_TERM_PREFIX):]: float(metrics.pop(key))
            for key in [k for k in metrics if k.startswith(LOSS_TERM_PREFIX)]}


class LossTracker:
    """Per mini-batch, ``sum_micro scale * term`` for every loss term (``scale`` = the factor the
    actor multiplies the micro-batch loss by before ``backward``, ``B / N``). ``flush`` returns
    ``loss/<term>`` for this rank's update and starts the next mini-batch."""

    def __init__(self):
        self._sums: Dict[str, float] = {}

    def add(self, terms: Mapping[str, float], scale: float) -> None:
        for name, value in terms.items():
            self._sums[name] = self._sums.get(name, 0.0) + float(value) * float(scale)

    def flush(self) -> Dict[str, float]:
        out = {f"loss/{name}": value for name, value in self._sums.items()}
        self._sums = {}
        return out


def rltt_loss_terms(pg, kl, kl_coef, ent, entropy_coeff, rows, mini_batch_rows):
    """``loss/<term>`` of one vendor RLTT update from its per-micro-batch metrics: the vendor
    scales micro-batch ``i`` by ``rows[i] / mini_batch_rows`` before ``backward``."""
    tracker = LossTracker()
    for p, k, c, e, b in zip(pg, kl, kl_coef, ent, rows):
        terms = {"pg": p, "total": p + c * k - entropy_coeff * e}
        if c != 0:
            terms["kl"] = c * k
        if entropy_coeff != 0:
            terms["entropy"] = -entropy_coeff * e
        tracker.add(terms, b / mini_batch_rows)
    return tracker.flush()


__all__ = ["LOSS_AGG_MODES", "LOSS_TERM_PREFIX", "LossAggSettings", "row_divisors", "agg_rows",
           "loss_term_metrics", "pop_loss_terms", "LossTracker", "rltt_loss_terms"]
