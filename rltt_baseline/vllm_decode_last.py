"""Optional `decode_last` KV layout for Ouro rollouts in vLLM (rollout-only; training is unchanged).

Prefill keeps every loop's KV (prompt tokens run all 4 loops, each with its own cache). A generated
token runs all loops too, attending at loop u to the prompt's loop-u K/V, to the last-loop K/V of
earlier generated tokens and to its own loop-u K/V; after its last loop, its slot in the loop
1..T-1 caches is overwritten with its last-loop K/V. This is the Ouro paper's §5.4.2 "last-step
reuse" during decoding. It changes what the model computes, not how much memory it uses: the
stock per-(loop, layer) caches are kept.

Which tokens are generated is decided by position (position >= the request's prompt length),
computed by the model runner before every forward and read from a static GPU buffer. This is
exact under prefix caching (n samples of one prompt share the prompt's blocks), chunked prefill
and preemption. A preempted request recomputes its generated tokens in one prefill: within that
forward they attend to each other's loop-u K/V before being overwritten.

Use: import this module in the vLLM process (verl `actor_rollout_ref.model.external_lib`) and
build the engine with `hf_overrides={"architectures": [ARCHITECTURE]}` and no torch.compile
(`compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}`, or `enforce_eager`).
Importing registers the architecture and wraps `GPUModelRunner._prepare_inputs`; both are no-ops
for other models. Supports the TRITON_ATTN and FLASH_ATTN backends with a bf16/fp16 KV cache;
source-audited for vLLM 0.12/0.15 and 0.24.0. The 0.24 path uses per-layer
slot mappings and the V1 GPU model runner; V2, microbatching, context parallelism and
pipeline parallelism are rejected explicitly until their token slicing is validated.
"""

from rltt_baseline.runtime_compat import require_decode_last_runtime

CACHE_CONTRACT = require_decode_last_runtime()  # before importing vLLM internals

import numpy as np
import torch
from vllm import ModelRegistry
from vllm.forward_context import get_forward_context
from vllm.model_executor.models import ouro as vouro
from vllm.model_executor.models.utils import maybe_prefix
from vllm.v1.worker import gpu_model_runner

from rltt_baseline.decode_last import ARCHITECTURE, generated_mask

BUFFER = "kv_share_is_generated"


def _write_kv(layer, key: torch.Tensor, value: torch.Tensor, slots: torch.Tensor,
              virtual_engine: int = 0) -> None:
    """Write K/V into ``layer``'s cache with the backend's own kernel (slot -1 is skipped)."""
    impl = layer.impl
    if str(impl.kv_cache_dtype) not in ("auto", "bfloat16", "float16"):
        raise NotImplementedError("decode_last supports bf16/fp16 KV caches only")
    backend = type(impl).__module__
    if CACHE_CONTRACT == "per_layer_slots":
        if not backend.endswith((".triton_attn", ".flash_attn")):
            raise NotImplementedError(f"decode_last does not support the {backend} backend")
        # 0.24 stores a Tensor, not a virtual-engine list. Both audited backends
        # expose this method and own their cache layout/stride handling.
        impl.do_kv_cache_update(layer, key, value, layer.kv_cache, slots)
        return
    kv_cache = layer.kv_cache[virtual_engine]
    if backend.endswith(".triton_attn"):
        from vllm.v1.attention.backends import triton_attn

        key_cache, value_cache = kv_cache.unbind(1)
        triton_attn.triton_reshape_and_cache_flash(key, value, key_cache, value_cache, slots,
                                                   impl.kv_cache_dtype, layer._k_scale,
                                                   layer._v_scale)
    elif backend.endswith(".flash_attn"):
        from vllm.v1.attention.backends import flash_attn

        key_cache, value_cache = kv_cache.unbind(0)
        flash_attn.reshape_and_cache_flash(key, value, key_cache, value_cache, slots,
                                           impl.kv_cache_dtype, layer._k_scale, layer._v_scale)
    else:
        raise NotImplementedError(f"decode_last does not support the {backend} backend")


class DecodeLastKVAttention(vouro.OuroAttention):
    kv_share_is_generated: torch.Tensor  # set by the model, shared by every layer

    def forward(self, positions, hidden_states, current_ut):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn[current_ut](q, k, v)
        if current_ut == len(self.attn) - 1:
            self._keep_last_loop_kv(k, v)
        output, _ = self.o_proj(attn_output)
        return output

    def _keep_last_loop_kv(self, key: torch.Tensor, value: torch.Tensor) -> None:
        context = get_forward_context()
        metadata = context.attn_metadata
        if metadata is None:  # profiling / dummy runs
            return
        if CACHE_CONTRACT == "per_layer_slots":
            mappings = context.slot_mapping
            if isinstance(mappings, list) or isinstance(metadata, list):
                raise NotImplementedError("decode_last: microbatch slot slicing is unsupported")
            if not metadata:  # 0.24 dummy forward has no attention metadata
                return
            for layer in self.attn[:-1]:
                if layer.layer_name not in mappings:
                    raise RuntimeError(f"decode_last: missing slot mapping for {layer.layer_name}")
                # Different loop layers may have different KV-cache groups; never
                # reuse the final loop's physical slots to write another cache.
                slots = mappings[layer.layer_name]
                self._rewrite_layer(layer, key, value, slots)
            return
        if isinstance(metadata, dict):
            metadata = metadata.get(self.attn[-1].layer_name)
        if metadata is None:
            return
        for layer in self.attn[:-1]:
            self._rewrite_layer(layer, key, value, metadata.slot_mapping,
                                context.virtual_engine)

    def _rewrite_layer(self, layer, key, value, slots, virtual_engine=0):
        n = slots.shape[0]
        if n > self.kv_share_is_generated.shape[0] or n > min(key.shape[0], value.shape[0]):
            raise RuntimeError("decode_last: slot mapping exceeds token buffers")
        slots = slots.masked_fill(~self.kv_share_is_generated[:n], -1)
        k = key[:n].reshape(n, self.num_kv_heads, self.head_dim)
        v = value[:n].reshape(n, self.num_kv_heads, self.head_dim)
        _write_kv(layer, k, v, slots, virtual_engine)


class DecodeLastKVDecoderLayer(vouro.OuroDecoderLayer):
    def __init__(self, config, cache_config=None, quant_config=None, prefix: str = "") -> None:
        super().__init__(config, cache_config, quant_config, prefix)
        self.self_attn.__class__ = DecodeLastKVAttention  # same modules, new forward only


class OuroDecodeLastKVForCausalLM(vouro.OuroForCausalLM):
    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        _validate_runner_config(vllm_config)
        torch.nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.model = vouro.OuroModel(vllm_config=vllm_config,
                                     prefix=maybe_prefix(prefix, "model"),
                                     decoder_layer_type=DecodeLastKVDecoderLayer)
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = vouro.ParallelLMHead(config.vocab_size, config.hidden_size,
                                                quant_config=self.quant_config,
                                                prefix=maybe_prefix(prefix, "lm_head"))
        self.logits_processor = vouro.LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
        compilation = vllm_config.compilation_config
        size = max(vllm_config.scheduler_config.max_num_batched_tokens,
                   getattr(compilation, "max_cudagraph_capture_size", None) or 0,
                   vllm_config.scheduler_config.max_num_seqs) + 64
        device = self.model.embed_tokens.weight.device
        generated = torch.zeros(size, dtype=torch.bool, device=device)
        object.__setattr__(self, BUFFER, generated)  # not a parameter or state-dict buffer
        for layer in self.model.layers:
            if isinstance(layer, DecodeLastKVDecoderLayer):
                object.__setattr__(layer.self_attn, BUFFER, generated)


def _validate_runner_config(config):
    if CACHE_CONTRACT != "per_layer_slots":
        return
    if config.use_v2_model_runner:
        raise NotImplementedError("decode_last: vLLM V2 model runner is not supported; use V1")
    parallel = config.parallel_config
    if getattr(parallel, "use_ubatching", False) or getattr(parallel, "enable_dbo", False):
        raise NotImplementedError("decode_last: DBO/microbatching is not supported")
    for name in ("pipeline_parallel_size", "decode_context_parallel_size",
                 "prefill_context_parallel_size"):
        if getattr(parallel, name, 1) != 1:
            raise NotImplementedError(f"decode_last: {name} must be 1")


def _generated_mask_024(runner, total, prompt_lengths):
    # 0.24's async scheduler corrects num_computed_tokens on the GPU. CPU
    # positions_np can be stale after speculative acceptance or batch changes.
    owners = runner.req_indices.gpu[:total].long()
    return runner.positions[:total] >= prompt_lengths[owners]


def _fill_generated_024(runner, scheduler_output, buffer):
    total = int(scheduler_output.total_num_scheduled_tokens)
    num_reqs = runner.input_batch.num_reqs
    if total > buffer.shape[0] or num_reqs > buffer.shape[0]:
        raise RuntimeError("decode_last: scheduled tokens exceed mask buffer")
    staging = getattr(runner, "_kv_share_prompt_staging", None)
    if staging is None:
        staging = [(torch.zeros(buffer.shape, dtype=torch.int64, pin_memory=True),
                    torch.empty(buffer.shape, dtype=torch.int64, device=buffer.device),
                    torch.cuda.Event()) for _ in range(2)]
        runner._kv_share_prompt_staging, runner._kv_share_prompt_next = staging, 0
    host, prompt_gpu, copied = staging[runner._kv_share_prompt_next]
    runner._kv_share_prompt_next ^= 1
    copied.synchronize()
    host[:num_reqs] = torch.as_tensor(runner.input_batch.num_prompt_tokens[:num_reqs])
    prompt_gpu[:num_reqs].copy_(host[:num_reqs], non_blocking=True)
    buffer.zero_()  # padded CUDA-graph rows must never inherit a previous decode mask
    buffer[:total].copy_(_generated_mask_024(runner, total, prompt_gpu))
    copied.record()


def _fill_generated(runner, scheduler_output, buffer: torch.Tensor) -> None:
    if CACHE_CONTRACT == "per_layer_slots":
        return _fill_generated_024(runner, scheduler_output, buffer)
    batch = runner.input_batch
    req_ids = batch.req_ids
    num_scheduled = np.array([scheduler_output.num_scheduled_tokens[r] for r in req_ids],
                             dtype=np.int64)
    total = int(scheduler_output.total_num_scheduled_tokens)
    if total > buffer.shape[0] or int(num_scheduled.sum()) != total:
        raise RuntimeError("decode_last: scheduled tokens do not fit the generated-token buffer")
    mask = generated_mask(runner.positions.np[:total], num_scheduled,
                          batch.num_prompt_tokens[:len(req_ids)])
    # two pinned staging buffers: with async scheduling the previous step's copy may still be
    # in flight while this step is prepared, so a staging buffer is reused only after its copy
    staging = getattr(runner, "_kv_share_staging", None)
    if staging is None or staging[0][0].shape != buffer.shape:
        staging = [(torch.zeros(buffer.shape, dtype=torch.bool, pin_memory=True),
                    torch.cuda.Event()) for _ in range(2)]
        runner._kv_share_staging, runner._kv_share_next = staging, 0
    host, copied = staging[runner._kv_share_next]
    runner._kv_share_next ^= 1
    copied.synchronize()
    host.zero_()
    host[:total] = torch.from_numpy(mask)
    buffer.copy_(host, non_blocking=True)
    copied.record()


def _install_runner_hook() -> None:
    cls = gpu_model_runner.GPUModelRunner
    original = cls._prepare_inputs
    if getattr(original, "_ouro_decode_last", False):
        return

    def _prepare_inputs(self, scheduler_output, *args, **kwargs):
        result = original(self, scheduler_output, *args, **kwargs)
        buffer = getattr(self.get_model(), BUFFER, None)
        if buffer is not None:
            _fill_generated(self, scheduler_output, buffer)
        return result

    _prepare_inputs._ouro_decode_last = True
    cls._prepare_inputs = _prepare_inputs


def register() -> None:
    if ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(ARCHITECTURE, OuroDecodeLastKVForCausalLM)
    _install_runner_hook()


register()  # on import: verl imports `external_lib` in the worker that builds the vLLM engine
