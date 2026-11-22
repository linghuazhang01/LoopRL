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


# ---------- extrapolated policy (EP-GRPO, ``rltt_baseline.extrap``) ----------

def _chunk_extrap(source_hidden, ref_hidden, labels, lm_head, temperature, omega, adaptive,
                  delta_cap):
    """Log-prob of ``labels`` under the LoopCD policy, with its entropy, strength and margin.

    pi = softmax(((1 + w) log P_s - w log P_r) / temperature), the reference clipped by
    ``delta_cap`` in ``log P_s - log P_r``; ``w = omega`` or, when ``adaptive``, the per-position
    ``omega * (1 - (p_(1) - p_(2)))`` of ``P_s`` at temperature 1. Same distribution as
    ``rltt_baseline.loop_cd.guided_logprobs`` followed by the sampler's temperature. Gradient
    flows through the inputs that require it (``ref_hidden`` is detached by the caller unless the
    reference gets a gradient); the entropy, strength and margin never carry one.
    """
    log_s = F.log_softmax(lm_head(source_hidden).float(), dim=-1)
    log_r = F.log_softmax(lm_head(ref_hidden).float(), dim=-1)
    with torch.no_grad():
        top2 = log_s.detach().exp().topk(2, dim=-1).values
        margin = 1.0 - (top2[..., 0] - top2[..., 1])
        strength = omega * margin if adaptive else torch.full_like(margin, float(omega))
    delta = log_s - log_r
    if delta_cap is not None:
        delta = delta.clamp(-float(delta_cap), float(delta_cap))
    logits = (log_s + strength.unsqueeze(-1) * delta) / temperature
    log_pi = F.log_softmax(logits, dim=-1)
    picked = log_pi.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        entropy = -(log_pi.detach().exp() * log_pi.detach()).sum(-1)
    return picked, entropy, strength, margin


def extrap_token_stats(
    hidden_states_list: Sequence[torch.Tensor],
    lm_head: torch.nn.Module,
    labels: torch.Tensor,
    omega: float,
    ref_loop: int = 1,
    adaptive: bool = False,
    delta_cap: Optional[float] = None,
    ref_grad: bool = False,
    temperature: float = 1.0,
    chunk_size: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """The extrapolated (LoopCD) policy on the sampled tokens, per position.

    The source is the last loop; ``ref_loop`` is 1-based. Each chunk is checkpointed when grad is
    enabled (two vocabulary-sized tensors per position would otherwise stay alive until backward).

    Returns float32 (B, S) tensors: ``log_probs`` (grad through the last loop, and through the
    reference loop only with ``ref_grad``), ``entropy`` of the policy, ``omega`` (per-position
    strength) and ``margin`` (1 - top-2 gap of the last loop at temperature 1); the last three
    without grad.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    loops = len(hidden_states_list)
    if not 1 <= int(ref_loop) < loops:
        raise ValueError(f"ref_loop must be in [1, {loops - 1}], got {ref_loop}")
    seq_len = labels.shape[1]
    step = seq_len if not chunk_size or chunk_size <= 0 else min(int(chunk_size), seq_len)
    labels = labels.long()
    source, ref = hidden_states_list[-1], hidden_states_list[int(ref_loop) - 1]
    if not ref_grad:
        ref = ref.detach()
    parts = {name: [] for name in ("log_probs", "entropy", "omega", "margin")}
    for start in range(0, seq_len, step):
        args = (source[:, start:start + step], ref[:, start:start + step],
                labels[:, start:start + step], lm_head, temperature, float(omega),
                bool(adaptive), delta_cap)
        if torch.is_grad_enabled() and (args[0].requires_grad or args[1].requires_grad):
            out = checkpoint(_chunk_extrap, *args, use_reentrant=False)
        else:
            out = _chunk_extrap(*args)
        for name, value in zip(parts, out):
            parts[name].append(value)
    return {name: torch.cat(values, dim=1) for name, values in parts.items()}


# ---------- full-vocabulary OPD (``rltt_baseline.opd``, estimator ``full``) ----------

def opd_teacher_log_probs(teacher_ref_logits: torch.Tensor, teacher_last_logits: torch.Tensor,
                          omega: float, temperature: float = 1.0) -> torch.Tensor:
    """The OPD teacher over the full vocabulary of the frozen model:
    log_softmax(((1 + w) log P^(T) - w log P^(r)) / tau), i.e. LoopCD-Logits
    (``rltt_baseline.loop_cd``) followed by the temperature."""
    log_last = F.log_softmax(teacher_last_logits.float(), dim=-1)
    log_ref = F.log_softmax(teacher_ref_logits.float(), dim=-1)
    return F.log_softmax((log_last + float(omega) * (log_last - log_ref)) / temperature, dim=-1)


def _chunk_opd(student_hidden, teacher_ref_hidden, teacher_last_hidden, student_lm_head,
               teacher_lm_head, temperature, omega, divergence):
    """Per-position full-vocabulary divergence of the student's last loop against the frozen
    teacher, plus no-grad monitoring values. The teacher's logits are computed under no_grad (also
    when a checkpoint recomputes the chunk in backward)."""
    with torch.no_grad():
        log_q = opd_teacher_log_probs(teacher_lm_head(teacher_ref_hidden),
                                      teacher_lm_head(teacher_last_hidden), omega, temperature)
    log_p = F.log_softmax(student_lm_head(student_hidden).float() / temperature, dim=-1)
    kd = kd_divergence(log_p, log_q, divergence)
    with torch.no_grad():
        lp = log_p.detach()
        other = kd_divergence(lp, log_q, "reverse" if divergence == "forward" else "forward")
        q_entropy = -(log_q.exp() * log_q).sum(-1)
        top1_agree = (lp.argmax(-1) == log_q.argmax(-1)).float()
    return kd, other, q_entropy, top1_agree


def opd_full_kd(
    student_hidden: torch.Tensor,
    teacher_ref_hidden: torch.Tensor,
    teacher_last_hidden: torch.Tensor,
    student_lm_head: torch.nn.Module,
    teacher_lm_head: torch.nn.Module,
    omega: float,
    divergence: str = "reverse",
    temperature: float = 1.0,
    chunk_size: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Full-vocabulary distillation of the student's last loop towards the frozen OPD teacher.

    At every position of ``student_hidden`` (the contexts of the student's own rollouts):
    ``reverse`` = KL(P^(T) || pi_T), ``forward`` = KL(pi_T || P^(T)), with
    pi_T = ``opd_teacher_log_probs`` of the teacher's reference and last loops.

    Args:
        student_hidden: (B, S, H) normed last-loop hidden state of the trained model.
        teacher_ref_hidden, teacher_last_hidden: (B, S, H) normed hidden states of the frozen
            model's reference and last loops (no gradient flows into them).
        chunk_size: positions per chunk along S; ``None`` or <= 0 means one chunk. With grad each
            chunk is checkpointed (the vocabulary-sized tensors are recomputed in backward).

    Returns float32 (B, S) tensors: ``kd`` (the configured divergence, grad through the student),
    and without grad ``kl_other`` (the other direction), ``teacher_entropy`` and ``top1_agree``.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if divergence not in KD_DIVERGENCES:
        raise ValueError(f"unknown divergence {divergence!r}; allowed {KD_DIVERGENCES}")
    if not (student_hidden.shape[:2] == teacher_ref_hidden.shape[:2]
            == teacher_last_hidden.shape[:2]):
        raise ValueError("student and teacher hidden states must cover the same positions")
    seq_len = student_hidden.shape[1]
    step = seq_len if not chunk_size or chunk_size <= 0 else min(int(chunk_size), seq_len)
    teacher_ref_hidden = teacher_ref_hidden.detach()
    teacher_last_hidden = teacher_last_hidden.detach()
    parts = {name: [] for name in ("kd", "kl_other", "teacher_entropy", "top1_agree")}
    for start in range(0, seq_len, step):
        args = (student_hidden[:, start:start + step], teacher_ref_hidden[:, start:start + step],
                teacher_last_hidden[:, start:start + step], student_lm_head, teacher_lm_head,
                temperature, float(omega), divergence)
        if torch.is_grad_enabled() and args[0].requires_grad:
            out = checkpoint(_chunk_opd, *args, use_reentrant=False)
        else:
            out = _chunk_opd(*args)
        for name, value in zip(parts, out):
            parts[name].append(value)
    return {name: torch.cat(values, dim=1) for name, values in parts.items()}
