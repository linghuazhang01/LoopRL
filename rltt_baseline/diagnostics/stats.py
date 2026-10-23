"""Statistics of per-loop gradients from two half batches.

For each shared parameter tensor the probe stores the Gram matrix of the 2T vectors
[g_1^A .. g_T^A, g_1^B .. g_T^B] (A, B = two disjoint halves of the prompt groups).
Gram matrices add over disjoint parameter blocks, so any group (a layer, all attention
weights, the whole model) is summarised by summing its tensors' Gram matrices.

From a Gram matrix G (full-batch g_t = g_t^A + g_t^B):
- ``norm``: ||g_t|| per loop;
- ``cos``: cos(g_t, g_s) on the full batch;
- ``rho``: ||sum_t g_t|| / sum_t ||g_t|| (1 = aligned; 1/sqrt(T) for orthogonal equal norms;
  lower = cancellation);
- ``noise_same_loop``: cos(g_t^A, g_t^B), the split-half agreement of one loop (noise floor);
- ``cross_loop_split``: cos(g_t^A, g_s^B) for s != t, averaged over both orders. Loops conflict
  beyond noise when this is clearly below ``noise_same_loop``.
"""

import math
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import torch


def gram(vectors: Sequence[torch.Tensor]) -> torch.Tensor:
    flat = torch.stack([v.reshape(-1).double() for v in vectors])
    return (flat @ flat.T).cpu()


def _cos(g: torch.Tensor, i: int, j: int) -> float:
    denom = math.sqrt(max(g[i, i].item(), 0.0) * max(g[j, j].item(), 0.0))
    return g[i, j].item() / denom if denom > 0 else float("nan")


def summarize(g: torch.Tensor, loops: int) -> Dict[str, object]:
    """Metrics from a (2T x 2T) Gram matrix of [halves A loops..., halves B loops...]."""
    t = loops
    # full-batch vectors f_k = a_k + b_k
    pick = torch.zeros(t, 2 * t, dtype=g.dtype)
    for k in range(t):
        pick[k, k] = 1.0
        pick[k, t + k] = 1.0
    full = pick @ g @ pick.T
    norms = [math.sqrt(max(full[k, k].item(), 0.0)) for k in range(t)]
    total = math.sqrt(max(full.sum().item(), 0.0))
    cos = [[_cos(full, i, j) for j in range(t)] for i in range(t)]
    noise = [_cos(g, k, t + k) for k in range(t)]
    cross = []
    for i in range(t):
        for j in range(t):
            if i != j:
                cross.append(_cos(g, i, t + j))
    pairs = [cos[i][j] for i in range(t) for j in range(i + 1, t)]
    return {
        "norm": norms,
        "norm_total": total,
        "rho": total / sum(norms) if sum(norms) > 0 else float("nan"),
        "cos": cos,
        "cos_pair_mean": sum(pairs) / len(pairs),
        "noise_same_loop": noise,
        "noise_same_loop_mean": sum(noise) / t,
        "cross_loop_split_mean": sum(cross) / len(cross),
    }


_LAYER = re.compile(r"^layers\.(\d+)\.")


def module_type(name: str) -> str:
    if name.startswith("norm") or name.startswith("early_exit_gate"):
        return name.split(".")[0]
    if ".self_attn." in name:
        return "attention"
    if ".mlp." in name:
        return "mlp"
    return "layer_norm"


def aggregate(grams: Dict[str, torch.Tensor], loops: int) -> Dict[str, Dict[str, object]]:
    """Summaries for: all, each module type, each layer, each layer x module type."""
    buckets: Dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(2 * loops, 2 * loops,
                                                                      dtype=torch.float64))
    for name, g in grams.items():
        kind = module_type(name)
        keys = ["all", f"type:{kind}"]
        match = _LAYER.match(name)
        if match:
            keys += [f"layer:{int(match.group(1)):02d}",
                     f"layer:{int(match.group(1)):02d}:{kind}"]
        for key in keys:
            buckets[key] = buckets[key] + g
    return {key: summarize(g, loops) for key, g in sorted(buckets.items())}


def hidden_grad_norms(norms: Iterable[Sequence[float]]) -> Dict[str, List[float]]:
    """Mean over sequences of ||dL/dh_t|| and of its ratio to ||dL/dh_T||."""
    rows: List[Tuple[float, ...]] = [tuple(float(x) for x in seq) for seq in norms]
    if not rows:
        return {"mean": [], "ratio_to_last": []}
    t = len(rows[0])
    mean = [sum(r[k] for r in rows) / len(rows) for k in range(t)]
    valid = [r for r in rows if r[-1] > 0]
    ratio = [sum(r[k] / r[-1] for r in valid) / max(1, len(valid)) for k in range(t)]
    return {"mean": mean, "ratio_to_last": ratio}
