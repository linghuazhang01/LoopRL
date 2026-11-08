"""Which tokens push RLTT's exit gate and each loop's entropy, split by the sign of the advantage.

Reads the per-token loop audit (``<output_dir>/audit/step_*/tokens_rank*.npz`` and
``responses.jsonl``, written with ``save_tokens: true``), recomputes the GRPO advantage of every
response from its reward and group (verl: unbiased std, epsilon 1e-6), and evaluates the first
order effect of one step of the RLTT loss

    L = -(1/N) sum_tokens A * sum_t w_t * ell_t,    ell_t = log P^(t)(y),  w = exit PDF

on two things, each split into the part from A > 0 tokens and the part from A < 0 tokens:

1. Gate push. ``gate_push/loop_t``: the gradient of -L with respect to the exit weights,
   projected onto the simplex, ``sum_tokens c * (ell_t - mean_s ell_s)`` with ``c`` the token's
   coefficient ``A / N``. A positive value means the update moves exit mass onto loop t.
   ``gate_dw/loop_t`` is the same step taken on each token's own gate logits
   (``lambda_t = sigmoid(z_t)``, ``w_t = lambda_t prod_{s<t}(1 - lambda_s)``), mapped back to the
   exit weights: the change of ``w_t`` per unit step size. Only computed when the audit has
   ``exit_pdf``; for a GRPO run it is what the RLTT loss would do on the same rollouts.

2. Entropy push. ``entropy_push/loop_t``: the first-order change of loop t's entropy if each
   token's loop-t logits took one gradient step of -L, ``dz = c_t (e_y - p)`` with
   ``c_t = c * w_t``: ``dH = -c_t [p_y (log p_y + H) - sum_v p_v^2 (log p_v + H)]``. The sum over
   the vocabulary uses the audit's top-k probabilities (the rest of the vocabulary is left out;
   its mass is ``1 - topk_mass``). ``--weighting last`` puts all weight on the last loop, which is
   GRPO's gradient (ratio 1, one update per step).

Both are logit-space directions: they treat every token's logits as free parameters. The real
update goes through parameters shared by all loops, so they say which way each loss term pulls,
not how far the model moves. The summary checks the predicted sign of each step against the
observed change of the per-loop means at the next audited step.

Token weights: ``--norm token`` (default) weights every response token of the step by 1/N,
verl's token-mean over the whole step; ``--norm sequence`` weights each response's tokens by
1/(|y| * responses), the paper's (1/g) sum_i (1/|y_i|).

    python -m rltt_baseline.diagnostics.gate_push --audit-dir <attempt 1>/audit@1-28 \
        [--audit-dir <attempt 2>/audit@29-100 ...] --out gate_push.csv [--steps 20-100]
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from rltt_baseline.diagnostics.losses import grpo_advantages

PDF_EPS = 1e-6
NEEDED = ("sampled_log_prob", "entropy", "topk_probs", "offsets")
WEIGHTINGS = ("exit_pdf", "plc_weight", "last", "uniform")


def gate_logits_from_pdf(pdf: torch.Tensor) -> torch.Tensor:
    """Invert ``exit_pdf_from_gates``: (n, T) exit PDF -> (n, T - 1) gate logits."""
    pdf = pdf.double()
    remaining = torch.ones_like(pdf[:, 0])
    logits = []
    for t in range(pdf.shape[1] - 1):
        lam = (pdf[:, t] / remaining.clamp_min(PDF_EPS)).clamp(PDF_EPS, 1 - PDF_EPS)
        logits.append(torch.log(lam) - torch.log1p(-lam))
        remaining = remaining * (1 - lam)
    return torch.stack(logits, dim=-1)


def pdf_from_gate_logits(logits: torch.Tensor) -> torch.Tensor:
    """(n, T - 1) gate logits -> (n, T) exit PDF; the last loop's gate is never read.

    The recursion of ``exit_pdf_from_gates`` kept in the input dtype (that one computes in
    float32, too coarse for the Jacobian-vector product here).
    """
    remaining = torch.ones_like(logits[:, 0])
    pdf = []
    for t in range(logits.shape[1]):
        lam = torch.sigmoid(logits[:, t])
        pdf.append(lam * remaining)
        remaining = remaining * (1 - lam)
    pdf.append(remaining)
    return torch.stack(pdf, dim=-1)


def gate_push(ell: np.ndarray, coef: np.ndarray) -> np.ndarray:
    """Simplex-projected gradient of ``sum coef * w . ell`` w.r.t. w, summed over tokens: (T,)."""
    centered = ell - ell.mean(axis=1, keepdims=True)
    return (coef[:, None] * centered).sum(axis=0, dtype=np.float64)


def gate_dw(ell: np.ndarray, pdf: np.ndarray, coef: np.ndarray) -> np.ndarray:
    """Change of the exit weights after a unit gradient step on each token's gate logits: (T,)."""
    z = gate_logits_from_pdf(torch.from_numpy(pdf)).requires_grad_(True)
    ell_t = torch.from_numpy(ell).double()
    coef_t = torch.from_numpy(coef).double()
    objective = (coef_t[:, None] * pdf_from_gate_logits(z) * ell_t).sum()
    (grad,) = torch.autograd.grad(objective, z)
    _, dw = torch.func.jvp(pdf_from_gate_logits, (z.detach(),), (grad,))
    return dw.sum(0).numpy()


def entropy_push(ell: np.ndarray, entropy: np.ndarray, topk_probs: np.ndarray,
                 coef: np.ndarray) -> np.ndarray:
    """Per-token first-order entropy change of each loop under ``dz_t = coef_t (e_y - p_t)``.

    Args:
        ell, entropy: (n, T) log-prob of the sampled token and entropy of each loop.
        topk_probs: (n, T, k) the k largest probabilities of each loop.
        coef: (n, T) step coefficient of each token and loop.

    Returns:
        (n, T) dH; the vocabulary sum ``sum_v p_v^2 (log p_v + H)`` runs over the top k only.
    """
    p_y = np.exp(ell)
    top = topk_probs.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_top = np.where(top > 0, np.log(top), 0.0)
    second = (top ** 2 * (log_top + entropy[..., None])).sum(-1)
    return -coef * (p_y * (ell + entropy) - second)


def parse_steps(text: Optional[str]) -> Optional[Tuple[int, int]]:
    if not text:
        return None
    low, _, high = text.partition("-")
    return int(low), int(high or low)


def parse_audit_dir(text: str) -> Tuple[Path, Optional[Tuple[int, int]]]:
    """``DIR`` or ``DIR@LOW-HIGH``: the steps this directory may supply (inclusive)."""
    path, _, steps = str(text).rpartition("@") if "@" in str(text) else (str(text), "", "")
    return Path(path), parse_steps(steps)


def step_dirs(audit_dirs: Sequence, steps: Optional[Tuple[int, int]],
              stride: int) -> List[Tuple[int, Path]]:
    """Audited steps with token shards; a step found in several directories uses the last one.

    Each entry of ``audit_dirs`` is ``DIR`` or ``DIR@LOW-HIGH``; with a range the directory only
    supplies those steps (a resumed run's earlier attempt also holds steps that were replayed).
    """
    found: Dict[int, Path] = {}
    for entry in audit_dirs:
        directory, allowed = parse_audit_dir(entry)
        for path in sorted(directory.glob("step_*")):
            step = int(path.name.split("_")[-1])
            if allowed is not None and not allowed[0] <= step <= allowed[1]:
                continue
            if (path / "responses.jsonl").exists() and any(path.glob("tokens_rank*.npz")):
                found[step] = path
    chosen = sorted(s for s in found if steps is None or steps[0] <= s <= steps[1])
    if stride > 1 and chosen:
        chosen = [s for s in chosen if (s - chosen[0]) % stride == 0]
    return [(s, found[s]) for s in chosen]


def response_advantages(step_dir: Path) -> Dict[Tuple[int, int], float]:
    """GRPO advantage of every matched response, keyed by (rank, index) in the token shards."""
    lines = [json.loads(line) for line in open(step_dir / "responses.jsonl")]
    matched = [line for line in lines if line.get("uid") is not None and line["reward"] is not None]
    adv = grpo_advantages([line["reward"] for line in matched], [line["uid"] for line in matched])
    return {(line["rank"], line["index"]): a for line, a in zip(matched, adv)}


def _weights(arrays: Dict[str, np.ndarray], weighting: str, loops: int) -> np.ndarray:
    n = arrays["sampled_log_prob"].shape[0]
    if weighting == "last":
        w = np.zeros((n, loops))
        w[:, -1] = 1.0
        return w
    if weighting == "uniform":
        return np.full((n, loops), 1.0 / loops)
    return arrays[weighting].astype(np.float64)


def analyze_step(step_dir: Path, weighting: str = "exit_pdf",
                 norm: str = "token") -> Optional[dict]:
    """Metrics of one audited step (see the module docstring).

    Returns None when ``weighting`` is ``exit_pdf`` or ``plc_weight`` and the step's audit has no
    such array (audits without ``loop_weights: true``; ``plc_weight`` exists in PLC runs only).
    """
    advantages = response_advantages(step_dir)
    shards = []
    for shard in sorted(step_dir.glob("tokens_rank*.npz")):
        rank = int(shard.stem[-2:])
        with np.load(shard) as data:
            if weighting in ("exit_pdf", "plc_weight") and weighting not in data.files:
                return None
            arrays = {name: data[name] for name in NEEDED + ("exit_pdf", "plc_weight")
                      if name in data.files}
        lengths = np.diff(arrays.pop("offsets"))
        adv = np.array([advantages.get((rank, i), np.nan) for i in range(len(lengths))])
        token_adv = np.repeat(adv, lengths)
        token_len = np.repeat(lengths, lengths).astype(np.float64)
        keep = ~np.isnan(token_adv)
        arrays = {name: value[keep] for name, value in arrays.items()}
        shards.append((arrays, token_adv[keep], token_len[keep],
                       int((~np.isnan(adv)).sum())))
    tokens = sum(len(a) for _, a, _, _ in shards)
    responses = sum(r for _, _, _, r in shards)
    if not tokens:
        raise ValueError(f"{step_dir}: no tokens matched to a response")

    loops = shards[0][0]["sampled_log_prob"].shape[1]
    sums = defaultdict(float)
    vec = defaultdict(lambda: np.zeros(loops))
    has_pdf = all("exit_pdf" in arrays for arrays, _, _, _ in shards)
    for arrays, adv, length, _ in shards:
        ell = arrays["sampled_log_prob"].astype(np.float64)
        scale = 1.0 / tokens if norm == "token" else 1.0 / (length * responses)
        coef = adv * scale
        w = _weights(arrays, weighting, loops)
        dh = entropy_push(ell, arrays["entropy"].astype(np.float64), arrays["topk_probs"],
                          coef[:, None] * w)
        gap = ell[:, -1] - ell[:, 0]
        vec["entropy_mean"] += arrays["entropy"].sum(0)
        if has_pdf:
            vec["pdf_mean"] += arrays["exit_pdf"].sum(0)
        for sign, mask in (("pos", adv > 0), ("neg", adv < 0)):
            sums[f"tokens_{sign}"] += mask.sum()
            sums[f"abs_adv_{sign}"] += np.abs(coef[mask]).sum()
            sums[f"gap_{sign}"] += gap[mask].sum()
            vec[f"entropy_push_{sign}"] += dh[mask].sum(0)
            if has_pdf:
                vec[f"gate_push_{sign}"] += gate_push(ell[mask], coef[mask])
                vec[f"gate_dw_{sign}"] += gate_dw(ell[mask], arrays["exit_pdf"][mask],
                                                  coef[mask])

    row = {"step": int(step_dir.name.split("_")[-1]), "source": str(step_dir.parent),
           "responses": responses, "tokens": tokens}
    signed = sums["tokens_pos"] + sums["tokens_neg"]
    row["neg_token_share"] = sums["tokens_neg"] / signed if signed else math.nan
    total_abs = sums["abs_adv_pos"] + sums["abs_adv_neg"]
    row["neg_weight_share"] = sums["abs_adv_neg"] / total_abs if total_abs else math.nan
    for sign in ("pos", "neg"):
        n = sums[f"tokens_{sign}"]
        row[f"gap_last_minus_first/{sign}"] = sums[f"gap_{sign}"] / n if n else math.nan
    names = ["entropy_push"] + (["gate_push", "gate_dw"] if has_pdf else [])
    for t in range(loops):
        loop = f"loop_{t + 1}"
        row[f"entropy_mean/{loop}"] = vec["entropy_mean"][t] / tokens
        if has_pdf:
            row[f"pdf_mean/{loop}"] = vec["pdf_mean"][t] / tokens
        for name in names:
            pos, neg = vec[f"{name}_pos"][t], vec[f"{name}_neg"][t]
            row[f"{name}/{loop}"] = pos + neg
            row[f"{name}/{loop}/pos"] = pos
            row[f"{name}/{loop}/neg"] = neg
    return row


def sign_agreement(rows: List[dict], predicted: str, observed: str) -> Dict[str, dict]:
    """Per loop: how often sign(predicted at step s) equals sign(observed(s') - observed(s))."""
    out = {}
    loops = sorted({k.split("/")[1] for r in rows for k in r
                    if k.startswith(predicted + "/loop_")})
    for loop in loops:
        hits = total = 0
        for a, b in zip(rows, rows[1:]):
            if b["step"] != a["step"] + 1:  # only consecutive steps (--stride 1)
                continue
            p = a.get(f"{predicted}/{loop}")
            d = b.get(f"{observed}/{loop}", math.nan) - a.get(f"{observed}/{loop}", math.nan)
            if p is None or not np.isfinite(d) or p == 0 or d == 0:
                continue
            total += 1
            hits += int(np.sign(p) == np.sign(d))
        out[loop] = {"agree": hits, "pairs": total,
                     "rate": hits / total if total else math.nan}
    return out


def summarize(rows: List[dict]) -> dict:
    """Totals over steps, the share of each push that comes from A < 0, and sign agreement."""
    summary: Dict[str, object] = {"steps": [r["step"] for r in rows]}
    if not rows:
        return summary
    sources = defaultdict(list)
    for r in rows:
        sources[r["source"]].append(r["step"])
    summary["sources"] = {s: [min(v), max(v), len(v)] for s, v in sources.items()}
    for key in ("neg_token_share", "neg_weight_share", "gap_last_minus_first/pos",
                "gap_last_minus_first/neg"):
        summary[key] = float(np.nanmean([r[key] for r in rows]))
    totals = {}
    keys = dict.fromkeys(k for r in rows for k in r)  # steps without exit_pdf lack gate columns
    for key in keys:
        if key.count("/") == 1 and key.split("/")[0] in ("gate_push", "gate_dw", "entropy_push"):
            have = [r for r in rows if key in r]
            pos = sum(r[f"{key}/pos"] for r in have)
            neg = sum(r[f"{key}/neg"] for r in have)
            totals[key] = {"total": pos + neg, "from_pos": pos, "from_neg": neg,
                           "steps": len(have), "steps_positive": sum(r[key] > 0 for r in have)}
    summary["totals"] = totals
    if "gate_dw/loop_1" in keys:
        summary["gate_dw_vs_next_pdf_mean"] = sign_agreement(rows, "gate_dw", "pdf_mean")
    summary["entropy_push_vs_next_entropy_mean"] = sign_agreement(rows, "entropy_push",
                                                                  "entropy_mean")
    return summary


def write_csv(rows: Iterable[dict], path: Path) -> None:
    rows = list(rows)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--audit-dir", action="append", required=True,
                        help="DIR or DIR@LOW-HIGH; repeat for resumed runs, giving each attempt "
                        "the steps it owns in the run's lineage (later ones win)")
    parser.add_argument("--out", type=Path, required=True, help="per-step CSV; summary is "
                        "written next to it as <out>.summary.json")
    parser.add_argument("--weighting", default="exit_pdf", choices=WEIGHTINGS,
                        help="loop weights of the entropy push (last = GRPO)")
    parser.add_argument("--norm", default="token", choices=("token", "sequence"))
    parser.add_argument("--steps", help="inclusive range, e.g. 20-100")
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args(argv)

    rows, skipped = [], []
    for step, path in step_dirs(args.audit_dir, parse_steps(args.steps), args.stride):
        row = analyze_step(path, args.weighting, args.norm)
        if row is None:
            skipped.append(step)
            print(json.dumps({"step": step, "skipped": f"no {args.weighting} in {path}"}),
                  flush=True)
            continue
        rows.append(row)
        print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                          for k, v in row.items() if "/" not in k or k.endswith("loop_1")
                          or k.endswith("loop_4")}), flush=True)
    if not rows:
        raise SystemExit("no audited step with token shards (save_tokens: true) was found")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out)
    summary = summarize(rows)
    summary.update({"skipped_steps": skipped, "weighting": args.weighting, "norm": args.norm,
                    "audit_dirs": list(args.audit_dir)})
    summary_path = args.out.with_suffix(args.out.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=1, default=float) + "\n")
    print(json.dumps(summary, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
