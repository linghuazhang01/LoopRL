#!/usr/bin/env bash
# GRPO smoke run on the visible GPUs (CUDA_VISIBLE_DEVICES): 2 steps, 2 x n_gpu prompts x 4
# rollouts, 512-token generations; the two steps are accumulated into one optimizer update as
# in the full config; checkpoint (incl. hf_model) at step 2, no validation.
# Only checks that the pipeline runs end to end; not a result.
# The training file (LOOPRL_TRAIN_FILE: verl parquet or MATH JSONL) is cut to its first rows.
# Extra overrides win, e.g. args.max_completion_length=16384 args.ppo_max_token_len=32768
# args.log_prob_max_token_len=32768 to check memory at full length.
#   CUDA_VISIBLE_DEVICES=0,1 LOOPRL_VENV=... LOOPRL_MODEL_DIR=... LOOPRL_TRAIN_FILE=... \
#   LOOPRL_MATH500_JSONL=... LOOPRL_OUTPUT_DIR=... bash smoke_grpo.sh [extra overrides]
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the GPUs to use (>= 2)}"
: "${LOOPRL_TRAIN_FILE:?}" "${LOOPRL_OUTPUT_DIR:?}" "${LOOPRL_VENV:?}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
NGPU=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
(( NGPU >= 2 )) || { echo "need >= 2 GPUs" >&2; exit 2; }
PROMPTS=$((2 * NGPU))
mkdir -p "$LOOPRL_OUTPUT_DIR"
ROWS=$((PROMPTS * 4))
if [[ "$LOOPRL_TRAIN_FILE" == *.parquet ]]; then
  SMOKE_TRAIN="$LOOPRL_OUTPUT_DIR/smoke_train.parquet"
  "$LOOPRL_VENV/bin/python" -c 'import sys, pyarrow.parquet as pq
pq.write_table(pq.read_table(sys.argv[1]).slice(0, int(sys.argv[3])), sys.argv[2])' \
    "$LOOPRL_TRAIN_FILE" "$SMOKE_TRAIN" "$ROWS"
else
  SMOKE_TRAIN="$LOOPRL_OUTPUT_DIR/smoke_train.jsonl"
  head -n "$ROWS" "$LOOPRL_TRAIN_FILE" > "$SMOKE_TRAIN"
fi
LOOPRL_TRAIN_FILE="$SMOKE_TRAIN" exec timeout 5400 bash "$HERE/run_train.sh" \
  "$HERE/../configs/grpo_baseline_8gpu.yaml" \
  args.n_gpus=$NGPU args.num_generations=4 args.num_prompts_per_batch=$PROMPTS \
  args.max_completion_length=512 args.max_steps=2 args.save_steps=2 args.no_validation=true \
  args.vllm_gpu_memory_utilization=0.3 args.ppo_max_token_len=4096 \
  args.log_prob_max_token_len=4096 args.dataloader_num_workers=2 \
  verl_overrides.data.train_batch_size=$PROMPTS \
  verl_overrides.actor_rollout_ref.actor.ppo_mini_batch_size=$((2 * PROMPTS)) \
  wandb.run_name=grpo_smoke "$@"
