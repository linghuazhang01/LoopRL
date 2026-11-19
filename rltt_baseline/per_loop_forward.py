"""Padded Ouro per-loop forward, independent of verl worker lifecycle.

This shared primitive preserves the legacy PLC response alignment. It is not a
verl 0.9 engine implementation: packing, distributed reductions and lifecycle
remain the responsibility of the caller. CPU execution uses the input dtype;
CUDA execution retains the legacy BF16 autocast policy.
"""
from contextlib import nullcontext
from typing import Optional

import torch


def per_loop_forward(module, micro_batch, temperature: float,
                     kd_divergence: Optional[str] = None):
    """Return response entropy, [batch, response, loop] logprobs, KD and KD stats."""
    entropy, ell, kd, kd_stats, _ = _forward(module, micro_batch, temperature, kd_divergence)
    return entropy, ell, kd, kd_stats


def extrap_forward(module, micro_batch, temperature: float, model_kwargs: dict):
    """Last-loop entropy and per-loop log-probs as ``per_loop_forward``, plus the extrapolated
    policy's ``[batch, response]`` tensors (``log_probs``, ``entropy``, ``omega``, ``margin``;
    ``rltt_per_loop.extrap_token_stats``). ``model_kwargs``: ``ExtrapSettings.model_kwargs()``."""
    entropy, ell, _, _, extrap = _forward(module, micro_batch, temperature, None, model_kwargs)
    return entropy, ell, extrap


def _forward(module, micro_batch, temperature: float, kd_divergence: Optional[str] = None,
             extrap_kwargs: Optional[dict] = None):
    response_length = micro_batch["responses"].size(-1)
    kd_kwargs = {}
    if kd_divergence is not None:
        kd_kwargs = {"return_loop_kd": True, "loop_kd_divergence": kd_divergence,
                     "loop_kd_window": response_length}
    if extrap_kwargs:
        kd_kwargs.update(extrap_kwargs)
    device_type = micro_batch["input_ids"].device.type
    precision = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                 if device_type == "cuda" else nullcontext())
    with precision:
        input_ids = micro_batch["input_ids"]
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)
        labels = torch.nn.functional.pad(input_ids[:, 1:], (0, 1), value=0)
        output = module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_per_loop_log_probs=True,
            return_exit_pdf=False,
            log_prob_labels=labels,
            log_prob_temperature=temperature,
            use_cache=False,
            **kd_kwargs,
        )
        window = slice(-response_length - 1, -1)
        ell = torch.stack([lp[:, window] for lp in output.per_loop_log_probs], dim=-1).float()
        entropy = output.per_loop_entropies[-1][:, window]
        kd = kd_stats = extrap = None
        if kd_divergence is not None:
            kd, kd_stats = output.loop_kd, output.loop_kd_stats
        if extrap_kwargs:
            extrap = {name: value[:, window].float() for name, value in output.extrap.items()}
        del output
    return entropy, ell, kd, kd_stats, extrap
