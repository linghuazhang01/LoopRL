"""Per-loop gradient probe: how GRPO's gradient splits across Ouro's four loops.

For each batch (``--groups-per-batch`` prompt groups with mixed rewards, split into two halves
A/B by group) and each loss, the probe accumulates every loop's gradient g_t on every shared
parameter for each half, stores the per-tensor Gram matrix of [g^A_1..T, g^B_1..T] and the
norms of dL/dh_t (gradient w.r.t. each loop's output hidden state). No parameter is updated.

    python -m rltt_baseline.diagnostics.probe --model <dir> --rollouts r.jsonl --out out_dir \
        [--losses grpo,nll,sft_mixture] [--batches 4] [--groups-per-batch 16]

Outputs per run: ``summary.json`` (aggregated metrics, see ``stats``) and ``grams.pt`` (raw
per-tensor Gram matrices for re-analysis).
"""

import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch

from rltt_baseline.diagnostics.losses import grpo_advantages, sequence_loss
from rltt_baseline.diagnostics.stats import aggregate, gram, hidden_grad_norms
from rltt_baseline.diagnostics.untie import LoopUntie

LOSSES = ("grpo", "nll", "sft_mixture")


def load_rollouts(path: Path) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                groups[rec["uid"]].append(rec)
    return groups


def make_batches(groups: Dict[str, List[dict]], batches: int, per_batch: int, seed: int,
                 max_len: int) -> List[List[str]]:
    """Disjoint batches of mixed-reward groups (only those give a non-zero GRPO gradient)."""
    mixed = sorted(uid for uid, recs in groups.items()
                   if 0 < sum(r["reward"] > 0 for r in recs) < len(recs)
                   and all(len(r["prompt_ids"]) + len(r["response_ids"]) <= max_len for r in recs))
    random.Random(seed).shuffle(mixed)
    need = batches * per_batch
    if len(mixed) < need:
        raise SystemExit(f"only {len(mixed)} mixed-reward groups, need {need}")
    return [mixed[i * per_batch:(i + 1) * per_batch] for i in range(batches)]


def sequences_for(kind: str, groups: Dict[str, List[dict]], uids: List[str]):
    """(record, advantage, include) for one half; denominators follow verl token-mean."""
    recs = [r for uid in uids for r in groups[uid]]
    adv = grpo_advantages([r["reward"] for r in recs], [r["uid"] for r in recs])
    if kind == "grpo":
        chosen = [(r, a) for r, a in zip(recs, adv) if a != 0.0]
        denom = sum(len(r["response_ids"]) for r in recs)
    else:
        chosen = [(r, 0.0) for r in recs if r["reward"] > 0]
        denom = sum(len(r["response_ids"]) for r, _ in chosen)
    return chosen, max(denom, 1)


def run_half(model, untie: LoopUntie, kind: str, seqs, denom: int, device, acc_device):
    acc = {name: [torch.zeros_like(p, dtype=torch.float32, device=acc_device) for p in per_loop]
           for name, per_loop in untie.named_loop_params()}
    hidden_grads = []
    tokens = 0
    for rec, adv in seqs:
        ids = torch.tensor([rec["prompt_ids"] + rec["response_ids"]], device=device)
        _, hiddens, gates = model.model(input_ids=ids, use_cache=False)
        for h in hiddens:
            h.retain_grad()
        loss = sequence_loss(kind, hiddens, gates, model.lm_head, len(rec["prompt_ids"]), ids,
                             adv, denom)
        loss.backward()
        hidden_grads.append([h.grad.float().norm().item() for h in hiddens])
        for name, per_loop in untie.named_loop_params():
            for t, p in enumerate(per_loop):
                if p.grad is not None:
                    acc[name][t] += p.grad.float().to(acc_device)
        untie.zero_grad()
        tokens += len(rec["response_ids"])
    return acc, hidden_grads, tokens


def forward_check(model, ids) -> float:
    with torch.no_grad():
        _, hiddens, _ = model.model(input_ids=ids, use_cache=False)
        return model.lm_head(hiddens[-1]).float()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--losses", default=",".join(LOSSES))
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--groups-per-batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--max-len", type=int, default=12288)
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--acc-device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = parser.parse_args(argv)
    kinds = [k for k in args.losses.split(",") if k]
    for k in kinds:
        if k not in LOSSES:
            raise SystemExit(f"unknown loss {k}; choose from {LOSSES}")
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} exists and is not empty")
    args.out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_device = torch.device(args.acc_device if device.type == "cuda" else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=getattr(torch, args.dtype)
    ).to(device).eval()

    groups = load_rollouts(args.rollouts)
    batches = make_batches(groups, args.batches, args.groups_per_batch, args.seed, args.max_len)

    probe_ids = torch.tensor([groups[batches[0][0]][0]["prompt_ids"][:64]], device=device)
    before = forward_check(model, probe_ids)
    untie = LoopUntie(model, use_checkpoint=not args.no_checkpoint)
    after = forward_check(model, probe_ids)
    forward_diff = (before - after).abs().max().item()

    summary = {
        "model": args.model,
        "rollouts": str(args.rollouts),
        "rollouts_sha256": hashlib.sha256(args.rollouts.read_bytes()).hexdigest(),
        "loops": untie.loops,
        "dtype": args.dtype,
        "checkpoint": not args.no_checkpoint,
        "untie_forward_max_abs_diff": forward_diff,
        "batches": [],
    }
    grams_out = {}
    for b, uids in enumerate(batches):
        half_a, half_b = uids[: len(uids) // 2], uids[len(uids) // 2:]
        entry = {"batch": b, "groups_A": half_a, "groups_B": half_b, "losses": {}}
        for kind in kinds:
            started = time.time()
            halves = []
            for i, half in enumerate((half_a, half_b)):
                seqs, denom = sequences_for(kind, groups, half)
                acc, hgrads, tokens = run_half(model, untie, kind, seqs, denom, device,
                                               acc_device)
                if i == 0:  # park half A on the host while half B accumulates
                    acc = {k: [v.cpu() for v in vs] for k, vs in acc.items()}
                halves.append((acc, hgrads, len(seqs), tokens, denom))
            grams = {}
            for name in halves[0][0]:
                vecs = [v.to(device) for v in halves[0][0][name] + halves[1][0][name]]
                grams[name] = gram(vecs)
                del vecs
            grams_out[f"batch{b}/{kind}"] = grams
            entry["losses"][kind] = {
                "sequences": [h[2] for h in halves],
                "tokens": [h[3] for h in halves],
                "denominators": [h[4] for h in halves],
                "seconds": round(time.time() - started, 1),
                "hidden_grad": hidden_grad_norms(halves[0][1] + halves[1][1]),
                "stats": aggregate(grams, untie.loops),
            }
            del halves
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(json.dumps({"batch": b, "loss": kind,
                              "seconds": entry["losses"][kind]["seconds"],
                              "rho_all": entry["losses"][kind]["stats"]["all"]["rho"]}),
                  flush=True)
        summary["batches"].append(entry)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
        torch.save(grams_out, args.out / "grams.pt")
    untie.restore()
    return 0


if __name__ == "__main__":
    sys.exit(main())
