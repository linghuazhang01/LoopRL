"""Objectives for the per-loop gradient probe.

GRPO follows verl 0.6.1 exactly (``core_algos.compute_grpo_outcome_advantage``,
``compute_policy_loss_vanilla`` with clip 0.2 and dual-clip c = 3, ``agg_loss`` token-mean,
``kl_penalty`` low_var_kl); the probe sums per-sequence terms and divides by the token count
of the whole (half) batch, which equals verl's token-mean over that batch.

- ``grpo``: policy loss on the final loop's log-probs plus beta * low-var KL to a reference.
  At the initial checkpoint (old = ref = current) the ratio is 1 and the KL gradient is 0,
  so the gradient is -A * grad log pi_4, token-averaged.
- ``nll``: final-loop next-token NLL on correct responses (same gradient path as GRPO,
  supervised signal).
- ``sft_mixture``: Ouro's ``labels`` path, cross-entropy of softmax(sum_t p_t * z_t) with
  the exit distribution p, on correct responses (what LlamaFactory SFT would compute with
  the released modeling code; the paper does not state the SFT loss).
"""

from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import torch

CLIP_RATIO = 0.2
CLIP_RATIO_C = 3.0
KL_COEF = 0.001


def grpo_advantages(scores: Sequence[float], group_ids: Sequence, epsilon: float = 1e-6,
                    norm_by_std: bool = True) -> List[float]:
    """Sequence-level GRPO advantage, as verl (unbiased std; singleton groups get 0)."""
    scores_t = torch.tensor([float(s) for s in scores], dtype=torch.float32)
    groups: Dict[object, List[int]] = defaultdict(list)
    for i, gid in enumerate(group_ids):
        groups[gid].append(i)
    adv = torch.zeros_like(scores_t)
    for idx in groups.values():
        vals = scores_t[idx]
        if len(idx) == 1:
            mean, std = torch.tensor(0.0), torch.tensor(1.0)
        else:
            mean, std = vals.mean(), vals.std()
        adv[idx] = (vals - mean) / (std + epsilon) if norm_by_std else vals - mean
    return adv.tolist()


def token_log_probs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    return logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, -1)


def grpo_token_losses(log_prob: torch.Tensor, old_log_prob: torch.Tensor, advantage: float,
                      ref_log_prob: Optional[torch.Tensor] = None,
                      kl_coef: float = KL_COEF) -> torch.Tensor:
    """Per-token GRPO loss (policy + kl_coef * low-var KL) for one response."""
    adv = torch.full_like(log_prob, float(advantage))
    neg_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(neg_kl)
    loss1 = -adv * ratio
    loss2 = -adv * torch.clamp(ratio, 1 - CLIP_RATIO, 1 + CLIP_RATIO)
    clip1 = torch.maximum(loss1, loss2)
    clip2 = torch.min(-adv * CLIP_RATIO_C, clip1)
    pg = torch.where(adv < 0, clip2, clip1)
    if ref_log_prob is None:
        return pg
    kl = torch.clamp(ref_log_prob - log_prob, min=-20, max=20)
    low_var_kl = torch.clamp(torch.exp(kl) - kl - 1, min=-10, max=10)
    return pg + kl_coef * low_var_kl


def exit_pdf(gate_list: Sequence[torch.Tensor]) -> torch.Tensor:
    """Same recursion as the official ``OuroForCausalLM.forward``; (B, S, T)."""
    pdf, remaining = [], None
    for idx, gate in enumerate(gate_list):
        lam = torch.sigmoid(gate.squeeze(-1).float())
        remaining = torch.ones_like(lam) if remaining is None else remaining
        if idx < len(gate_list) - 1:
            pdf.append(lam * remaining)
            remaining = remaining * (1.0 - lam)
        else:
            pdf.append(remaining)
    return torch.stack(pdf, dim=-1)


def mixture_log_probs(hiddens: Sequence[torch.Tensor], pdf: torch.Tensor, lm_head,
                      targets: torch.Tensor) -> torch.Tensor:
    """log softmax(sum_t p_t * z_t)[target] over the given positions."""
    mixed = None
    for t, hidden in enumerate(hiddens):
        z = lm_head(hidden).float() * pdf[..., t].unsqueeze(-1)
        mixed = z if mixed is None else mixed + z
    return token_log_probs(mixed, targets)


def sequence_loss(kind: str, hiddens: Sequence[torch.Tensor], gates: Sequence[torch.Tensor],
                  lm_head, prompt_len: int, input_ids: torch.Tensor, advantage: float,
                  denom: int) -> torch.Tensor:
    """Loss contribution of one sequence (batch dim 1), already divided by ``denom``.

    Position i predicts token i + 1; response tokens start at ``prompt_len``.
    """
    targets = input_ids[:, prompt_len:]
    start, end = prompt_len - 1, input_ids.shape[1] - 1
    if kind == "grpo":
        logp = token_log_probs(lm_head(hiddens[-1][:, start:end]), targets)
        frozen = logp.detach()
        return grpo_token_losses(logp, frozen, advantage, ref_log_prob=frozen).sum() / denom
    if kind == "nll":
        logp = token_log_probs(lm_head(hiddens[-1][:, start:end]), targets)
        return -logp.sum() / denom
    if kind == "sft_mixture":
        pdf = exit_pdf(gates)[:, start:end]
        logp = mixture_log_probs([h[:, start:end] for h in hiddens], pdf, lm_head, targets)
        return -logp.sum() / denom
    raise ValueError(f"unknown loss {kind}")

