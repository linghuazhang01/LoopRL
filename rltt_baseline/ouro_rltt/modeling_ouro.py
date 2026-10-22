"""Ouro with the per-loop outputs that the RLTT actor expects.

This file is not imported from the ``rltt_baseline`` package. ``rltt_baseline.prepare_model``
copies it into an RLTT model directory together with:

- ``modeling_ouro_base.py``: the unmodified official ``modeling_ouro.py`` of the source model
  (SHA-checked against the vendored official file) plus one ``check_model_inputs``
  compatibility edit for transformers 4.57;
- ``rltt_per_loop.py``: the pure-torch helpers.

The RLTT repository (github.com/jonwill8/RLTT) loads ``Ouro-*-Thinking-RLTT`` through
``trust_remote_code`` but does not ship that model's code. Its actor
(``rltt_experiments/verl_rltt/rltt_actor.py``) calls ``forward`` with
``return_per_loop_log_probs``, ``return_exit_pdf``, ``log_prob_labels`` and
``log_prob_temperature`` and reads ``per_loop_log_probs``, ``per_loop_entropies`` and
``exit_pdf`` from the output; the fallback path uses ``return_per_loop_logits``. This class
provides exactly that interface. Without those flags it defers to the official forward, so
generation and the standard verl log-prob path (final loop) are unchanged.

Model-config keys read here (written by ``prepare_model``):
    rltt_logprob_chunk_size (int | None), rltt_logprob_checkpoint (bool),
    rltt_detach_exit_pdf (bool).
"""

from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from .modeling_ouro_base import *  # noqa: F401,F403
from .modeling_ouro_base import OuroForCausalLM as _OfficialOuroForCausalLM
from .rltt_per_loop import exit_pdf_from_gates, per_loop_token_stats


@dataclass
class RLTTCausalLMOutput(CausalLMOutputWithPast):
    per_loop_log_probs: Optional[List[torch.Tensor]] = None
    per_loop_entropies: Optional[List[torch.Tensor]] = None
    per_loop_logits: Optional[List[torch.Tensor]] = None
    exit_pdf: Optional[torch.Tensor] = None


class OuroForCausalLM(_OfficialOuroForCausalLM):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep=0,
        use_weighted_exit: Optional[bool] = False,
        exit_at_step: Optional[int] = None,
        exit_threshold: Optional[float] = None,
        return_per_loop_log_probs: bool = False,
        return_per_loop_logits: bool = False,
        return_exit_pdf: bool = False,
        log_prob_labels: Optional[torch.LongTensor] = None,
        log_prob_temperature: float = 1.0,
        **kwargs,
    ):
        if not (return_per_loop_log_probs or return_per_loop_logits or return_exit_pdf):
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                use_weighted_exit=use_weighted_exit,
                exit_at_step=exit_at_step,
                exit_threshold=exit_threshold,
                **kwargs,
            )
        if labels is not None:
            raise ValueError("labels is not supported together with the per-loop RLTT outputs")

        outputs, hidden_states_list, gate_list = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        exit_pdf = None
        if return_exit_pdf:
            exit_pdf = exit_pdf_from_gates(gate_list)
            if getattr(self.config, "rltt_detach_exit_pdf", False):
                exit_pdf = exit_pdf.detach()

        per_loop_log_probs = per_loop_entropies = per_loop_logits = None
        if return_per_loop_log_probs:
            if log_prob_labels is None:
                raise ValueError("return_per_loop_log_probs=True requires log_prob_labels")
            per_loop_log_probs, per_loop_entropies = per_loop_token_stats(
                hidden_states_list,
                self.lm_head,
                log_prob_labels,
                temperature=log_prob_temperature,
                chunk_size=getattr(self.config, "rltt_logprob_chunk_size", None),
                checkpoint_chunks=getattr(self.config, "rltt_logprob_checkpoint", False),
            )
        if return_per_loop_logits:
            per_loop_logits = [self.lm_head(hidden) for hidden in hidden_states_list]

        return RLTTCausalLMOutput(
            loss=None,
            logits=None,
            past_key_values=outputs.past_key_values,
            per_loop_log_probs=per_loop_log_probs,
            per_loop_entropies=per_loop_entropies,
            per_loop_logits=per_loop_logits,
            exit_pdf=exit_pdf,
        )


__all__ = [
    "OuroPreTrainedModel",  # noqa: F405
    "OuroModel",  # noqa: F405
    "OuroForCausalLM",
    "OuroForSequenceClassification",  # noqa: F405
    "OuroForTokenClassification",  # noqa: F405
    "OuroForQuestionAnswering",  # noqa: F405
    "UniversalTransformerCache",  # noqa: F405
    "RLTTCausalLMOutput",
]
