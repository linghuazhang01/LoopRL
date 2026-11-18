"""Evaluate Ouro on a math benchmark with or without LoopCD guidance (vLLM, sampling).

Unguided (``--omega 0``, the default) uses the stock vLLM Ouro model; otherwise the LoopCD model
of ``rltt_baseline.vllm_loop_cd`` (settings in ``rltt_baseline.loop_cd``). Prompts and answer
checking are the RLTT evaluation's (``format_math_prompt``, ``rl_extract_boxed_answer``,
``rl_check_math_answer``, ``rl_get_gold_answer``; imported from ``third_party/RLTT``, unmodified).

    # paper setting for Ouro-1.4B-Thinking math: reference loop 1, fixed 0.5 or adaptive cap 1.0
    python -m rltt_baseline.eval_loopcd --model <dir> --data aime24.jsonl --out out/aime24_cd05 \
        --omega 0.5
    python -m rltt_baseline.eval_loopcd --model <dir> --data aime24.jsonl --out out/aime24_ada10 \
        --omega 1.0 --adaptive
    # source loop 4, reference loop 2, strength a
    python -m rltt_baseline.eval_loopcd ... --ref-loop 2 --omega <a>
    # half depth (source = loop 2, reference = loop 1)
    python -m rltt_baseline.eval_loopcd ... --loops 2 --omega 0.5

Writes ``samples.jsonl`` (one line per sample) and ``summary.json`` (avg@n, pass@n, unbiased
pass@k, truncation rate, mean length and every setting) into ``--out``, which must not exist.
Defaults match the project's 1.4B evaluation: 8 samples, 16,384 new tokens, temperature 0.6,
top-p 0.95, seed 42. LoopCD guides the logits before temperature and top-p, as in the paper.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


def load_rows(path: Path):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n samples with c correct (Chen et al., 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def summarize(correct, lengths, truncated, n: int) -> dict:
    """``correct``: per problem, the list of n 0/1 outcomes."""
    ks = sorted({k for k in (1, 2, 4, 8, 10, 16, n) if k <= n})
    per_problem = [sum(c) for c in correct]
    total = len(per_problem) * n
    return {
        "problems": len(per_problem),
        "samples": total,
        f"avg@{n}": sum(per_problem) / total,
        **{f"pass@{k}": sum(pass_at_k(n, c, k) for c in per_problem) / len(per_problem)
           for k in ks},
        "truncation_rate": sum(truncated) / total,
        "mean_response_tokens": sum(lengths) / total,
    }


def _git_head() -> str:
    try:
        root = Path(__file__).resolve().parents[1]
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=root, capture_output=True, text=True).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True, help="jsonl or parquet")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--problem-key", default="problem")
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)  # project-wide default seed
    # LoopCD
    parser.add_argument("--omega", type=float, default=0.0,
                        help="strength w (cap w_max with --adaptive); 0 = unguided stock model")
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--ref-loop", type=int, default=1)
    parser.add_argument("--space", choices=("logits", "hidden"), default="logits")
    parser.add_argument("--delta-cap", type=float, default=None,
                        help="clip log p_s - log p_r (not in the paper; default off)")
    parser.add_argument("--loops", type=int, default=None,
                        help="recurrent depth = source loop (default: the model's total_ut_steps)")
    parser.add_argument("--loopcd-arch-at-zero", action="store_true",
                        help="use the LoopCD model at omega 0 too (should equal the stock one)")
    # engine
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--cudagraph", action="store_true",
                        help="allow CUDA graphs (default: enforce_eager; LoopCD-Logits returns a "
                             "2x-wide forward output that has not been checked under graphs)")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; refusing to overwrite")

    from rltt_baseline import loop_cd
    from rltt_baseline.paths import rltt_source_dir

    model_config = json.loads((Path(args.model) / "config.json").read_text())
    loops = args.loops or int(model_config.get("total_ut_steps", 4))
    guided = args.omega != 0.0 or args.loopcd_arch_at_zero
    cfg = None
    if guided:
        cfg = loop_cd.LoopCDConfig(ref_loop=args.ref_loop, omega=args.omega,
                                   adaptive=args.adaptive, space=args.space,
                                   delta_cap=args.delta_cap).validate(loops)
    overrides = loop_cd.engine_overrides(cfg, loops)

    sys.path.insert(0, str(rltt_source_dir()))
    from math_utils import format_math_prompt
    from math_utils import rl_check_math_answer as check_math_answer
    from math_utils import rl_extract_boxed_answer as extract_boxed_answer
    from math_utils import rl_get_gold_answer as get_gold_answer
    from transformers import AutoTokenizer

    # in-process engine core: the architecture registered here is visible to the model loader
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from vllm import LLM, SamplingParams

    if guided:
        from rltt_baseline.runtime_compat import require_loop_cd_runtime

        require_loop_cd_runtime()
        loop_cd.register()

    rows = load_rows(args.data)
    if args.max_problems is not None:
        rows = rows[:args.max_problems]
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = [format_math_prompt(r[args.problem_key], tok) for r in rows]
    max_prompt = max(len(tok(p, add_special_tokens=False).input_ids) for p in prompts)

    llm = LLM(model=args.model, tokenizer=args.model, trust_remote_code=True, dtype="bfloat16",
              seed=args.seed, max_model_len=max_prompt + args.max_tokens,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_num_seqs=args.max_num_seqs, enforce_eager=not args.cudagraph,
              hf_overrides=overrides)
    params = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p,
                            max_tokens=args.max_tokens, seed=args.seed)
    start = time.time()
    outputs = llm.generate(prompts, params)
    elapsed = time.time() - start

    args.out.mkdir(parents=True)
    correct, lengths, truncated = [], [], []
    with open(args.out / "samples.jsonl", "w") as fh:
        for idx, (row, out) in enumerate(zip(rows, outputs)):
            gold = str(get_gold_answer(row))
            outcomes = []
            for k, comp in enumerate(out.outputs):
                pred = extract_boxed_answer(comp.text)
                ok = int(pred is not None and check_math_answer(pred, gold))
                outcomes.append(ok)
                lengths.append(len(comp.token_ids))
                truncated.append(int(comp.finish_reason == "length"))
                fh.write(json.dumps({"problem_index": idx, "sample": k, "gold": gold,
                                     "pred": pred, "correct": ok,
                                     "finish_reason": comp.finish_reason,
                                     "response_tokens": len(comp.token_ids),
                                     "text": comp.text}) + "\n")
            correct.append(outcomes)

    summary = summarize(correct, lengths, truncated, args.n)
    summary.update({
        "loop_cd": None if cfg is None else {**cfg.__dict__, "source_loop": loops},
        "loops": loops,
        "hf_overrides": overrides,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "code": _git_head(),
        "generate_seconds": round(elapsed, 1),
    })
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("args",)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
