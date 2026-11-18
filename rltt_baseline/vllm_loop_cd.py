"""LoopCD decoding for Ouro in vLLM (evaluation only; training and rollouts are unchanged).

The guidance math and the settings are in ``rltt_baseline.loop_cd``; the settings come from the
HF config (``loop_cd_*`` keys, set through ``hf_overrides``). The source loop is the model's last
loop, so the depth is set with ``total_ut_steps`` in the same overrides.

- ``space=logits``: ``forward`` returns ``cat([h_s, h_r], -1)`` (both after the final norm, as
  vLLM's Ouro feeds them to the next loop and to lm_head). vLLM selects the sampled positions from
  this tensor, then ``compute_logits`` runs lm_head on both halves and returns the normalized
  guided log-probs. Cost: one extra lm_head product per sampled position. The forward output is
  2 * hidden_size wide: run with ``enforce_eager`` unless CUDA graphs have been checked.
- ``space=hidden``: combines the pre-norm loop states (the residual sums that the final RMSNorm
  normalizes) and returns norm(h'), so lm_head runs once.

KV caches are untouched: every loop still writes its own K/V exactly as in stock vLLM.

Use: ``rltt_baseline.loop_cd.register()`` before building the engine, then
``LLM(..., hf_overrides={"architectures": [ARCHITECTURE], "total_ut_steps": s,
**LoopCDConfig(...).hf_overrides()})``. Checked against the Ouro model of vLLM 0.12/0.14/0.15 and
0.24.0 (``runtime_compat.require_loop_cd_runtime``); Ouro runs on the V1 model runner in all of
them. Pipeline parallelism is not supported.
"""

from rltt_baseline.runtime_compat import require_loop_cd_runtime

VLLM_VERSION = require_loop_cd_runtime()  # before importing vLLM internals

import torch  # noqa: E402
from vllm.model_executor.models import ouro as vouro  # noqa: E402
from vllm.model_executor.models.utils import maybe_prefix  # noqa: E402

from rltt_baseline.loop_cd import from_hf_config, guided_logprobs, run_loops  # noqa: E402


class OuroLoopCDModel(vouro.OuroModel):
    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.loop_cd = from_hf_config(self.config)
        if self.start_layer != 0 or self.end_layer != self.config.num_hidden_layers:
            raise NotImplementedError("LoopCD does not support pipeline parallelism")

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        if intermediate_tensors is not None:
            raise NotImplementedError("LoopCD does not support pipeline parallelism")
        hidden = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        return run_loops(hidden, positions, self.layers, self.norm, self.total_ut_steps,
                         self.loop_cd)


class OuroLoopCDForCausalLM(vouro.OuroForCausalLM):
    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        torch.nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.model = OuroLoopCDModel(vllm_config=vllm_config,
                                     prefix=maybe_prefix(prefix, "model"))
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = vouro.ParallelLMHead(config.vocab_size, config.hidden_size,
                                                quant_config=self.quant_config,
                                                prefix=maybe_prefix(prefix, "lm_head"))
        self.logits_processor = vouro.LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def compute_logits(self, hidden_states: torch.Tensor):
        cfg = self.model.loop_cd
        if cfg.space == "hidden":
            return super().compute_logits(hidden_states)
        size = self.config.hidden_size
        z_s = self.logits_processor(self.lm_head, hidden_states[..., :size].contiguous())
        z_r = self.logits_processor(self.lm_head, hidden_states[..., size:].contiguous())
        if z_s is None:  # non-driver tensor-parallel rank
            return None
        return guided_logprobs(z_s, z_r, cfg)
