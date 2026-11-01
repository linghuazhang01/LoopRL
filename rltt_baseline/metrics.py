"""Reasoning-RL training metrics computed on the driver from each training batch.

verl 0.6.1 already logs ``critic/score/*`` (mean reward), ``response_length/*`` (with
``clip_ratio``), ``critic/advantages/*``, ``actor/*`` losses, ``actor/lr``, timing and
throughput. This module adds the group-level statistics usually tracked for GRPO-style
reasoning RL. All keys are prefixed ``reasoning/``:

- ``accuracy``: fraction of rollouts with reward >= ``correct_threshold`` (training accuracy);
- ``pass_at_group``: fraction of prompts with at least one correct rollout;
- ``group_all_correct_ratio`` / ``group_all_wrong_ratio`` / ``group_mixed_ratio``: share of
  prompts whose rollouts are all correct, all wrong, or mixed; only mixed groups give a
  non-zero GRPO advantage;
- ``group_acc_std_mean``: mean within-group standard deviation of correctness;
- ``response_length/correct_mean`` and ``response_length/wrong_mean``;
- ``truncation_rate``: rollouts that hit the generation limit;
- ``truncated_accuracy``: accuracy among truncated rollouts;
- ``answer_rate``: rollouts with an extractable final answer (``\\boxed{}``), when an
  extractor is given;
- ``nonzero_adv_token_ratio``: response tokens with |advantage| > 1e-8, when advantages exist;
- ``accuracy/level_<k>``: accuracy by MATH level, when ``extra_info`` carries ``level``.

With ``audit.loop_distribution`` the same wrapper also finalizes the step's per-loop audit and
adds its ``loop_audit/*`` statistics (``rltt_baseline.loop_audit``).
"""

from collections import defaultdict
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import torch

from rltt_baseline.loop_audit import finalize_pending

PREFIX = "reasoning/"


def compute_reasoning_metrics(
    response_mask: torch.Tensor,
    scores: torch.Tensor,
    uids: Sequence,
    max_response_length: int,
    correct_threshold: float = 0.5,
    advantages: Optional[torch.Tensor] = None,
    has_answer: Optional[Sequence[bool]] = None,
    levels: Optional[Sequence] = None,
) -> Dict[str, float]:
    """Metrics from tensors.

    Args:
        response_mask: (B, R) 1 for generated tokens.
        scores: (B,) sequence reward (sum of token-level scores).
        uids: length-B group ids (rollouts of the same prompt share a uid).
        max_response_length: generation limit R used for the rollouts.
    """
    lengths = response_mask.sum(-1).float().cpu().numpy()
    score_np = scores.detach().float().cpu().numpy()
    correct = score_np >= correct_threshold
    truncated = lengths >= max_response_length
    n = len(correct)
    if n == 0:
        return {}
    out = {
        "accuracy": float(correct.mean()),
        "truncation_rate": float(truncated.mean()),
        "num_rollouts": float(n),
    }
    def masked_mean(values, mask):
        return float(values[mask].mean()) if mask.any() else float("nan")

    out["response_length/correct_mean"] = masked_mean(lengths, correct)
    out["response_length/wrong_mean"] = masked_mean(lengths, ~correct)
    out["truncated_accuracy"] = masked_mean(correct, truncated)

    groups = defaultdict(list)
    for i, uid in enumerate(uids):
        groups[uid].append(i)
    group_acc = np.array([correct[idx].mean() for idx in groups.values()])
    group_std = np.array([correct[idx].astype(float).std() for idx in groups.values()])
    out["num_groups"] = float(len(groups))
    out["pass_at_group"] = float((group_acc > 0).mean())
    out["group_all_correct_ratio"] = float((group_acc == 1).mean())
    out["group_all_wrong_ratio"] = float((group_acc == 0).mean())
    out["group_mixed_ratio"] = float(((group_acc > 0) & (group_acc < 1)).mean())
    out["group_acc_std_mean"] = float(group_std.mean())

    if has_answer is not None:
        out["answer_rate"] = float(np.asarray(has_answer, dtype=bool).mean())
    if advantages is not None:
        mask = response_mask.bool()
        tokens = mask.sum().item()
        if tokens:
            nonzero = (advantages.abs() > 1e-8) & mask
            out["nonzero_adv_token_ratio"] = float(nonzero.sum().item() / tokens)
    if levels is not None:
        by_level = defaultdict(list)
        for ok, level in zip(correct, levels):
            if level not in (None, ""):
                by_level[str(level).replace("Level ", "").strip()].append(ok)
        for level, oks in sorted(by_level.items()):
            out[f"accuracy/level_{level}"] = float(np.mean(oks))
    return {PREFIX + k: v for k, v in out.items()}


def batch_reasoning_metrics(
    batch,
    decode: Optional[Callable[[Sequence[int]], str]] = None,
    extract_answer: Optional[Callable[[str], Optional[str]]] = None,
    correct_threshold: float = 0.5,
) -> Dict[str, float]:
    """Metrics from a verl ``DataProto`` after rewards and advantages are computed."""
    tensors = batch.batch
    responses = tensors["responses"]
    response_mask = tensors["response_mask"] if "response_mask" in tensors.keys() else (
        tensors["attention_mask"][:, -responses.shape[-1]:]
    )
    scores = tensors["token_level_scores"].sum(-1)
    advantages = tensors["advantages"] if "advantages" in tensors.keys() else None
    non_tensor = batch.non_tensor_batch
    uids = non_tensor["uid"] if "uid" in non_tensor else list(range(len(scores)))

    has_answer = None
    if decode is not None and extract_answer is not None:
        has_answer = []
        for row, mask in zip(responses, response_mask):
            text = decode(row[mask.bool()].tolist())
            has_answer.append(extract_answer(text) not in (None, ""))

    levels = None
    if "extra_info" in non_tensor:
        levels = [info.get("level") if isinstance(info, dict) else None
                  for info in non_tensor["extra_info"]]

    return compute_reasoning_metrics(
        response_mask=response_mask,
        scores=scores,
        uids=uids,
        max_response_length=responses.shape[-1],
        correct_threshold=correct_threshold,
        advantages=advantages,
        has_answer=has_answer,
        levels=levels,
    )


def install_driver_metrics(decode=None, extract_answer=None, correct_threshold: float = 0.5,
                           loop_audit: Optional[dict] = None):
    """Wrap ``verl.trainer.ppo.metric_utils.compute_data_metrics`` to add reasoning metrics.

    The RLTT trainer imports ``compute_data_metrics`` inside ``fit`` from the module, so the
    wrapper must be installed before ``fit`` runs. ``loop_audit`` is the
    ``actor_rollout_ref.loop_audit`` setting, or None. Returns the original function.
    """
    from verl.trainer.ppo import metric_utils

    original = metric_utils.compute_data_metrics
    if getattr(original, "_rltt_reasoning_wrapped", False):
        return original

    def compute_data_metrics(batch, use_critic: bool = True):
        metrics = original(batch=batch, use_critic=use_critic)
        metrics.update(batch_reasoning_metrics(batch, decode, extract_answer, correct_threshold))
        if loop_audit:
            metrics.update(finalize_pending(loop_audit["dir"], batch, correct_threshold,
                                            loop_audit["save_tokens"]))
        return metrics

    compute_data_metrics._rltt_reasoning_wrapped = True
    metric_utils.compute_data_metrics = compute_data_metrics
    return original
