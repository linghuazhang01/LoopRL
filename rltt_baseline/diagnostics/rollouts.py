"""Sample GRPO-style rollouts for the per-loop gradient probe (vLLM, one GPU).

Prompts use the RLTT math instruction with the model's chat template; rewards use the RLTT
answer check (last ``\\boxed{}`` vs gold, 0/1). Each record keeps the token ids so the probe
recomputes log-probs on exactly the sampled tokens.

    python -m rltt_baseline.diagnostics.rollouts --model <dir> --data <dapo parquet> \
        --out rollouts.jsonl --prompts 64 --n 8 --max-tokens 2048
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path


def load_dapo(path: Path):
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    return [{"uid": r["extra_info"]["index"], "problem": r["prompt"],
             "gold": r["reward_model"]["ground_truth"]} for r in rows]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompts", type=int, default=64)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; refusing to overwrite")

    from math_utils import INSTRUCTION, check_math_answer, extract_boxed_answer
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    problems = load_dapo(args.data)
    random.Random(args.seed).shuffle(problems)
    problems = problems[:args.prompts]
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt_ids = [
        tok.apply_chat_template(
            [{"role": "user", "content": f"{INSTRUCTION}\n\nProblem: {p['problem']}"}],
            add_generation_prompt=True, tokenize=True)
        for p in problems
    ]
    max_prompt = max(len(p) for p in prompt_ids)
    llm = LLM(model=args.model, trust_remote_code=True, dtype="bfloat16", seed=args.seed,
              max_model_len=max_prompt + args.max_tokens,
              gpu_memory_utilization=args.gpu_memory_utilization)
    params = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p,
                            max_tokens=args.max_tokens, seed=args.seed)
    outputs = llm.generate([{"prompt_token_ids": ids} for ids in prompt_ids], params)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    counts = {"records": 0, "correct": 0, "truncated": 0}
    with open(args.out, "w") as fh:
        for prob, ids, out in zip(problems, prompt_ids, outputs):
            for k, comp in enumerate(out.outputs):
                text = comp.text
                pred = extract_boxed_answer(text)
                reward = 1.0 if pred is not None and check_math_answer(pred, prob["gold"]) else 0.0
                rec = {"uid": prob["uid"], "sample": k, "gold": prob["gold"],
                       "prompt_ids": list(ids), "response_ids": list(comp.token_ids),
                       "finish_reason": comp.finish_reason, "reward": reward}
                fh.write(json.dumps(rec) + "\n")
                counts["records"] += 1
                counts["correct"] += int(reward > 0)
                counts["truncated"] += int(comp.finish_reason == "length")
    counts["sha256"] = hashlib.sha256(args.out.read_bytes()).hexdigest()
    counts.update({"prompts": len(problems), "n": args.n, "max_tokens": args.max_tokens,
                   "temperature": args.temperature, "top_p": args.top_p, "seed": args.seed,
                   "model": args.model, "data": str(args.data)})
    Path(str(args.out) + ".meta.json").write_text(json.dumps(counts, indent=2) + "\n")
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
