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

``return_loop_kd`` (PLC ``loop_loss: kd``, ``rltt_baseline.plc``) adds ``loop_kd`` (B, W, T-1):
the full-vocabulary KL of every earlier loop towards the stop-gradient last loop
(``loop_kd_divergence``: ``forward`` = KL(sg P^(T) || P^(t)), ``reverse`` = KL(P^(t) || sg P^(T)))
on the last ``loop_kd_window`` predicting positions (the response tokens), and ``loop_kd_stats``
(``rltt_per_loop.KD_STATS``, without grad).

``return_extrap_log_probs`` (EP-GRPO, ``rltt_baseline.extrap``) adds ``extrap``: the log-prob of
``log_prob_labels`` under the extrapolated LoopCD policy
softmax(((1 + w) log P^(T) - w log P^(r)) / temperature) (``extrap_omega`` = w,
``extrap_ref_loop`` = r, ``extrap_adaptive``, ``extrap_delta_cap``; the reference loop is
stop-gradient unless ``extrap_ref_grad``), with the policy's entropy, the per-position strength
and the last loop's top-2 margin (``rltt_per_loop.extrap_token_stats``).

Full-vocabulary OPD (``rltt_baseline.opd``, estimator ``full``) uses two more switches.
``return_loop_hidden_states`` (the frozen teacher) adds ``loop_hidden_states``: the T normed
per-loop hidden states (B, S, H). ``return_opd_kd`` (the trained student) takes the teacher's
reference- and last-loop hidden states (``opd_teacher_hidden_states``) and its
``opd_teacher_lm_head`` and adds ``opd_kd``: the full-vocabulary divergence (``opd_divergence``)
of the student's last loop against the teacher log_softmax((1 + w) log P^(T) - w log P^(r))
(``opd_omega`` = w) on the last ``opd_window`` predicting positions, with monitoring values
(``rltt_per_loop.opd_full_kd``).

Model-config keys read here (written by ``prepare_model``):
    rltt_logprob_chunk_size (int | None), rltt_logprob_checkpoint (bool),
    rltt_detach_exit_pdf (bool).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from .modeling_ouro_base import *  # noqa: F401,F403
from .modeling_ouro_base import OuroForCausalLM as _OfficialOuroForCausalLM
from .rltt_per_loop import (
    exit_pdf_from_gates,
    extrap_token_stats,
    opd_full_kd,
    per_loop_kd,
    per_loop_token_stats,
)


@dataclass
class RLTTCausalLMOutput(CausalLMOutputWithPast):
    per_loop_log_probs: Optional[List[torch.Tensor]] = None
    per_loop_entropies: Optional[List[torch.Tensor]] = None
    per_loop_logits: Optional[List[torch.Tensor]] = None
    exit_pdf: Optional[torch.Tensor] = None
    loop_kd: Optional[torch.Tensor] = None
    loop_kd_stats: Optional[Dict[str, torch.Tensor]] = None
    extrap: Optional[Dict[str, torch.Tensor]] = None
    loop_hidden_states: Optional[List[torch.Tensor]] = None
    opd_kd: Optional[Dict[str, torch.Tensor]] = None


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
        return_loop_kd: bool = False,
        loop_kd_divergence: str = "forward",
        loop_kd_window: Optional[int] = None,
        return_extrap_log_probs: bool = False,
        extrap_omega: float = 0.0,
        extrap_ref_loop: int = 1,
        extrap_adaptive: bool = False,
        extrap_delta_cap: Optional[float] = None,
        extrap_ref_grad: bool = False,
        return_loop_hidden_states: bool = False,
        return_opd_kd: bool = False,
        opd_teacher_hidden_states=None,
        opd_teacher_lm_head: Optional[torch.nn.Module] = None,
        opd_omega: float = 0.0,
        opd_divergence: str = "reverse",
        opd_window: Optional[int] = None,
        **kwargs,
    ):
        if not (return_per_loop_log_probs or return_per_loop_logits or return_exit_pdf
                or return_loop_kd or return_extrap_log_probs or return_loop_hidden_states
                or return_opd_kd):
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

        loop_kd = loop_kd_stats = None
        if return_loop_kd:
            hiddens = hidden_states_list
            if loop_kd_window is not None:
                # position i predicts token i + 1: the last W tokens are predicted by
                # positions -W - 1 .. -2
                window = int(loop_kd_window)
                if not 0 < window < hiddens[0].shape[1]:
                    raise ValueError(f"loop_kd_window {window} does not fit the sequence")
                hiddens = [h[:, -window - 1:-1] for h in hiddens]
            loop_kd, loop_kd_stats = per_loop_kd(
                hiddens,
                self.lm_head,
                temperature=log_prob_temperature,
                divergence=loop_kd_divergence,
                chunk_size=getattr(self.config, "rltt_logprob_chunk_size", None),
            )

        extrap = None
        if return_extrap_log_probs:
            if log_prob_labels is None:
                raise ValueError("return_extrap_log_probs=True requires log_prob_labels")
            extrap = extrap_token_stats(
                hidden_states_list,
                self.lm_head,
                log_prob_labels,
                omega=extrap_omega,
                ref_loop=extrap_ref_loop,
                adaptive=extrap_adaptive,
                delta_cap=extrap_delta_cap,
                ref_grad=extrap_ref_grad,
                temperature=log_prob_temperature,
                chunk_size=getattr(self.config, "rltt_logprob_chunk_size", None),
            )

        opd_kd = None
        if return_opd_kd:
            if opd_teacher_hidden_states is None or opd_teacher_lm_head is None:
                raise ValueError("return_opd_kd=True requires opd_teacher_hidden_states and "
                                 "opd_teacher_lm_head")
            teacher_ref, teacher_last = opd_teacher_hidden_states
            student = hidden_states_list[-1]
            if opd_window is not None:
                # position i predicts token i + 1: the last W tokens are predicted by
                # positions -W - 1 .. -2 (as loop_kd_window)
                window = int(opd_window)
                if not 0 < window < student.shape[1]:
                    raise ValueError(f"opd_window {window} does not fit the sequence")
                student, teacher_ref, teacher_last = (
                    h[:, -window - 1:-1] for h in (student, teacher_ref, teacher_last))
            opd_kd = opd_full_kd(
                student,
                teacher_ref,
                teacher_last,
                self.lm_head,
                opd_teacher_lm_head,
                omega=opd_omega,
                divergence=opd_divergence,
                temperature=log_prob_temperature,
                chunk_size=getattr(self.config, "rltt_logprob_chunk_size", None),
            )

        return RLTTCausalLMOutput(
            loss=None,
            logits=None,
            past_key_values=outputs.past_key_values,
            per_loop_log_probs=per_loop_log_probs,
            per_loop_entropies=per_loop_entropies,
            per_loop_logits=per_loop_logits,
            exit_pdf=exit_pdf,
            loop_kd=loop_kd,
            loop_kd_stats=loop_kd_stats,
            extrap=extrap,
            loop_hidden_states=list(hidden_states_list) if return_loop_hidden_states else None,
            opd_kd=opd_kd,
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
