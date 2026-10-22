"""Per-loop log-probabilities, entropies and exit PDF for looped (Ouro-style) models.

Pure torch, no transformers dependency, so it can be unit-tested on CPU. The file is
copied next to ``modeling_ouro.py`` into the RLTT model directory and imported with a
relative import by the Hugging Face dynamic module loader.
"""

from typing import List, Optional, Sequence, Tuple

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
