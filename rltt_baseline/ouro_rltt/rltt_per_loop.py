"""Per-loop log-probabilities, entropies, exit PDF and cross-loop KL for looped (Ouro) models.

Pure torch, no transformers dependency, so it can be unit-tested on CPU. The file is
copied next to ``modeling_ouro.py`` into the RLTT model directory and imported with a
relative import by the Hugging Face dynamic module loader.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def exit_pdf_from_gates(gate_list: Sequence[torch.Tensor]) -> torch.Tensor:
    """Ouro exit distribution: p_t = lambda_t * prod_{s<t}(1 - lambda_s), last loop takes the rest.

    Same recursion as ``OuroForCausalLM.forward`` in the official modeling file, computed in
    float32. Each gate tensor has shape (B, S, 1); the result has shape (B, S, T).
    """
    pdf = []
    remaining = None
    for idx, gate in enumerate(gate_list):
        lam = torch.sigmoid(gate.squeeze(-1).float())
        if remaining is None:
            remaining = torch.ones_like(lam)
        if idx < len(gate_list) - 1:
            pdf.append(lam * remaining)
            remaining = remaining * (1.0 - lam)
        else:
            pdf.append(remaining)
    return torch.stack(pdf, dim=-1)


def _chunk_log_prob(hidden, labels, lm_head, temperature):
    logits = lm_head(hidden).float() / temperature
    picked = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return picked - torch.logsumexp(logits, dim=-1)


@torch.no_grad()
def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    log_p = F.log_softmax(logits, dim=-1)
    return -(log_p.exp() * log_p).sum(-1)


def per_loop_token_stats(
    hidden_states_list: Sequence[torch.Tensor],
    lm_head: torch.nn.Module,
    labels: torch.Tensor,
    temperature: float = 1.0,
    chunk_size: Optional[int] = None,
    checkpoint_chunks: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Log P^(t)(labels) and entropy of P^(t) for every loop t, per position.

    Args:
        hidden_states_list: T tensors (B, S, H), the normed hidden state after each loop.
        labels: (B, S) token ids; position i holds the token predicted from position i.
        chunk_size: positions per chunk along S; ``None`` or <= 0 means one chunk.
        checkpoint_chunks: recompute logits in backward instead of storing them, so only
            the hidden chunk is kept per chunk. Only applies when grad is enabled.

    Returns:
        (log_probs, entropies): two lists of T float32 tensors (B, S). Log-probs carry grad
        when grad is enabled; entropies never do (entropy regularisation is not supported).
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    seq_len = labels.shape[1]
    step = seq_len if not chunk_size or chunk_size <= 0 else min(int(chunk_size), seq_len)
    labels = labels.long()
    log_probs: List[torch.Tensor] = []
    entropies: List[torch.Tensor] = []
    for hidden in hidden_states_list:
        lp_parts, ent_parts = [], []
        for start in range(0, seq_len, step):
            h = hidden[:, start:start + step]
            y = labels[:, start:start + step]
            use_ckpt = checkpoint_chunks and torch.is_grad_enabled() and h.requires_grad
            if use_ckpt:
                lp = checkpoint(_chunk_log_prob, h, y, lm_head, temperature, use_reentrant=False)
                with torch.no_grad():
                    ent = _entropy_from_logits(lm_head(h).float() / temperature)
            else:
                logits = lm_head(h).float() / temperature
                lp = logits.gather(-1, y.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, -1)
                ent = _entropy_from_logits(logits.detach())
                del logits
            lp_parts.append(lp)
            ent_parts.append(ent)
        log_probs.append(torch.cat(lp_parts, dim=1))
        entropies.append(torch.cat(ent_parts, dim=1))
    return log_probs, entropies



# ---------- cross-loop distillation (PLC ``loop_loss: kd``) ----------

KD_DIVERGENCES = ("forward", "reverse")


def kd_divergence(log_p: torch.Tensor, log_q: torch.Tensor, divergence: str) -> torch.Tensor:
    """Full-vocabulary KL between a student ``p`` and a teacher ``q`` (log-probs, last dim V).

    ``forward``: ``KL(q || p) = sum_v q (log q - log p)``; ``reverse``: ``KL(p || q)``.
    """
    if divergence == "forward":
        return (log_q.exp() * (log_q - log_p)).sum(-1)
    if divergence == "reverse":
        return (log_p.exp() * (log_p - log_q)).sum(-1)
    raise ValueError(f"unknown divergence {divergence!r}; allowed {KD_DIVERGENCES}")


@torch.no_grad()
def kd_logit_step(log_p: torch.Tensor, log_q: torch.Tensor, divergence: str) -> torch.Tensor:
    """Minus the gradient of ``kd_divergence`` with respect to the student logits.

    ``forward``: ``q - p``; ``reverse``: ``-p * (log p - log q - KL(p || q))``.
    """
    p = log_p.exp()
    if divergence == "forward":
        return log_q.exp() - p
    if divergence == "reverse":
        kl = (p * (log_p - log_q)).sum(-1, keepdim=True)
        return -p * (log_p - log_q - kl)
    raise ValueError(f"unknown divergence {divergence!r}; allowed {KD_DIVERGENCES}")


@torch.no_grad()
def kd_entropy_push(log_p: torch.Tensor, log_q: torch.Tensor, divergence: str) -> torch.Tensor:
    """First-order change of the student entropy along ``kd_logit_step`` (per unit step).

    ``dH/dz_v = -p_v (log p_v + H)``, so the push is ``-sum_v p_v (log p_v + H) step_v``;
    negative means the distillation step lowers the student's entropy at this position. It is
    the logit-space effect only: the shared weights couple positions and loops.
    """
    p = log_p.exp()
    entropy = -(p * log_p).sum(-1, keepdim=True)
    return -(p * (log_p + entropy) * kd_logit_step(log_p, log_q, divergence)).sum(-1)


def _chunk_kd(teacher_hidden, lm_head, temperature, divergence, *student_hidden):
    """``(c..., T - 1)`` distillation losses of the student loops against the stop-gradient
    teacher; the teacher's logits are recomputed under no_grad (also in a checkpoint's backward)."""
    with torch.no_grad():
        log_q = F.log_softmax(lm_head(teacher_hidden).float() / temperature, dim=-1)
    losses = []
    for hidden in student_hidden:
        log_p = F.log_softmax(lm_head(hidden).float() / temperature, dim=-1)
        losses.append(kd_divergence(log_p, log_q, divergence))
        del log_p
    return torch.stack(losses, dim=-1)


@torch.no_grad()
def _chunk_kd_stats(hidden_states_list, lm_head, temperature, divergence):
    log_q = F.log_softmax(lm_head(hidden_states_list[-1]).float() / temperature, dim=-1)
    q_entropy = -(log_q.exp() * log_q).sum(-1)
    teacher_top1 = log_q.argmax(-1, keepdim=True)
    stats = {name: [] for name in KD_STATS}
    for hidden in hidden_states_list[:-1]:
        log_p = F.log_softmax(lm_head(hidden).float() / temperature, dim=-1)
        p = log_p.exp()
        stats["kl_forward"].append(kd_divergence(log_p, log_q, "forward"))
        stats["kl_reverse"].append(kd_divergence(log_p, log_q, "reverse"))
        stats["entropy_gap"].append(-(p * log_p).sum(-1) - q_entropy)
        stats["top1_agree"].append((log_p.argmax(-1, keepdim=True) == teacher_top1)
                                   .squeeze(-1).float())
        stats["teacher_top1_prob"].append(p.gather(-1, teacher_top1).squeeze(-1))
        stats["entropy_push"].append(kd_entropy_push(log_p, log_q, divergence))
        del log_p, p
    return {name: torch.stack(values, dim=-1) for name, values in stats.items()}


# per-position monitoring statistics of every student loop t < T against the last loop:
# both KL directions, H(P^(t)) - H(P^(T)), top-1 agreement, P^(t)(argmax P^(T)) and the entropy
# push of the configured divergence (``kd_entropy_push``)
KD_STATS = ("kl_forward", "kl_reverse", "entropy_gap", "top1_agree", "teacher_top1_prob",
            "entropy_push")


def per_loop_kd(
    hidden_states_list: Sequence[torch.Tensor],
    lm_head: torch.nn.Module,
    temperature: float = 1.0,
    divergence: str = "forward",
    chunk_size: Optional[int] = None,
    with_stats: bool = True,
) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    """Distillation of every earlier loop towards the stop-gradient last loop, per position.

    For t < T: ``kd[..., t] = KL(sg P^(T) || P^(t))`` (``forward``) or ``KL(P^(t) || sg P^(T))``
    (``reverse``), over the full vocabulary, at every position of ``hidden_states_list`` (the
    contexts of the rollouts the last loop sampled).

    Args:
        hidden_states_list: T tensors (B, S, H), the normed hidden state after each loop.
        chunk_size: positions per chunk along S; ``None`` or <= 0 means one chunk. With grad,
            each chunk is checkpointed (the logits are recomputed in backward), because a
            full-vocabulary loss would otherwise keep T vocabulary-sized tensors per position
            alive until backward.

    Returns:
        ``kd`` float32 (B, S, T - 1), with grad through the earlier loops only, and, with
        ``with_stats``, the ``KD_STATS`` tensors (B, S, T - 1) without grad.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if divergence not in KD_DIVERGENCES:
        raise ValueError(f"unknown divergence {divergence!r}; allowed {KD_DIVERGENCES}")
    if len(hidden_states_list) < 2:
        raise ValueError("distillation needs at least two loops")
    seq_len = hidden_states_list[0].shape[1]
    step = seq_len if not chunk_size or chunk_size <= 0 else min(int(chunk_size), seq_len)
    kd_parts, stat_parts = [], {name: [] for name in KD_STATS}
    for start in range(0, seq_len, step):
        chunk = [h[:, start:start + step] for h in hidden_states_list]
        teacher = chunk[-1].detach()
        students = chunk[:-1]
        if torch.is_grad_enabled() and any(h.requires_grad for h in students):
            kd = checkpoint(_chunk_kd, teacher, lm_head, temperature, divergence, *students,
                            use_reentrant=False)
        else:
            kd = _chunk_kd(teacher, lm_head, temperature, divergence, *students)
        kd_parts.append(kd)
        if with_stats:
            for name, value in _chunk_kd_stats([h.detach() for h in chunk], lm_head,
                                               temperature, divergence).items():
                stat_parts[name].append(value)
    kd = torch.cat(kd_parts, dim=1)
    stats = ({name: torch.cat(values, dim=1) for name, values in stat_parts.items()}
             if with_stats else None)
    return kd, stats
