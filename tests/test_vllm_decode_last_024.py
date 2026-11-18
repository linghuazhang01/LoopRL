"""CPU contract tests of the real adapter bodies, without importing vLLM/CUDA.

The lightweight vLLM bases stand in only for model construction. Cache routing,
mask computation and the runner wrapper below execute production function bodies.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture
def adapter():
    path = Path(__file__).parents[1] / "rltt_baseline/vllm_decode_last.py"
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    class Base:
        pass
    class Runner:
        def _prepare_inputs(self, output, counts=None):
            self.prepared = True
            return "prepared"
    ns = dict(torch=torch, np=np, CACHE_CONTRACT="per_layer_slots",
              BUFFER="kv_share_is_generated", vouro=SimpleNamespace(
                  OuroAttention=Base, OuroDecoderLayer=Base, OuroForCausalLM=Base),
              gpu_model_runner=SimpleNamespace(GPUModelRunner=Runner))
    exec(compile(tree, str(path), "exec"), ns)
    return ns


def test_gpu_corrected_positions_drive_generated_mask(adapter):
    # Batch was reordered and CPU num_computed_tokens is stale. The corrected
    # positions/owners supplied by 0.24 _prepare_inputs are the source of truth.
    runner = SimpleNamespace(positions=torch.tensor([8, 9, 2, 3, 4]),
                             req_indices=SimpleNamespace(gpu=torch.tensor([1, 1, 0, 0, 0])))
    actual = adapter["_generated_mask_024"](runner, 5, torch.tensor([4, 9]))
    assert actual.tolist() == [False, True, False, False, True]


def test_fill_resets_padding_and_tracks_current_prompt_order(adapter, monkeypatch):
    class Event:
        def synchronize(self): pass
        def record(self): pass
    zeros = torch.zeros
    monkeypatch.setattr(torch, "zeros", lambda *a, **k: zeros(*a, **{x: v for x, v in k.items() if x != "pin_memory"}))
    monkeypatch.setattr(torch.cuda, "Event", Event)
    runner = SimpleNamespace(input_batch=SimpleNamespace(num_reqs=2, num_prompt_tokens=np.array([3, 7])),
                             positions=torch.tensor([2, 3, 7]),
                             req_indices=SimpleNamespace(gpu=torch.tensor([0, 0, 1])))
    buffer = torch.ones(8, dtype=torch.bool)
    adapter["_fill_generated_024"](runner, SimpleNamespace(total_num_scheduled_tokens=3), buffer)
    assert buffer.tolist() == [False, True, True, False, False, False, False, False]
    runner.input_batch.num_prompt_tokens = np.array([10, 10])
    adapter["_fill_generated_024"](runner, SimpleNamespace(total_num_scheduled_tokens=3), buffer)
    assert not buffer.any()


def make_attention(adapter, mappings, metadata=None):
    layers = [SimpleNamespace(layer_name=n) for n in ["loop0", "loop1", "loop2"]]
    ctx = SimpleNamespace(attn_metadata={"loop0": object()} if metadata is None else metadata,
                          slot_mapping=mappings)  # deliberately no virtual_engine
    adapter["get_forward_context"] = lambda: ctx
    attn = adapter["DecodeLastKVAttention"]()
    attn.attn = layers
    attn.num_kv_heads = 1
    attn.head_dim = 2
    attn.kv_share_is_generated = torch.tensor([False, True, True, False])
    return attn, ctx


def test_rewrite_uses_each_destination_layer_mapping(adapter):
    a, _ = make_attention(adapter, {"loop0": torch.tensor([1, 2, 3]),
                                    "loop1": torch.tensor([9, 8, 7])})
    writes = []
    adapter["_write_kv"] = lambda layer, key, value, slots, virtual_engine=0: writes.append(
        (layer.layer_name, slots.tolist(), key.clone()))
    key = torch.arange(6).reshape(3, 2)
    a._keep_last_loop_kv(key, key + 10)
    assert [(name, slots) for name, slots, _ in writes] == [
        ("loop0", [-1, 2, 3]), ("loop1", [-1, 8, 7])]
    assert all(k.shape == (3, 1, 2) for _, _, k in writes)


def test_missing_layer_mapping_and_microbatching_fail_closed(adapter):
    a, ctx = make_attention(adapter, {})
    key = torch.zeros(3, 2)
    with pytest.raises(RuntimeError, match="missing slot mapping"):
        a._keep_last_loop_kv(key, key)
    ctx.slot_mapping = [{"loop0": torch.arange(3)}]
    with pytest.raises(NotImplementedError, match="microbatch"):
        a._keep_last_loop_kv(key, key)


def test_dummy_forward_does_not_write_cache(adapter):
    a, ctx = make_attention(adapter, {}, metadata={})
    adapter["_write_kv"] = lambda *args: pytest.fail("dummy cache write")
    a._keep_last_loop_kv(torch.zeros(3, 2), torch.zeros(3, 2))
    ctx.attn_metadata = None
    a._keep_last_loop_kv(torch.zeros(3, 2), torch.zeros(3, 2))


@pytest.mark.parametrize("backend", ["flash_attn", "triton_attn"])
def test_new_backend_owns_cache_layout(adapter, backend):
    calls = []
    impl_type = type("Impl", (), {"__module__": "vllm.v1.attention.backends." + backend,
        "kv_cache_dtype": "auto",
        "do_kv_cache_update": lambda self, *args: calls.append(args)})
    cache = object()
    layer = SimpleNamespace(impl=impl_type(), kv_cache=cache)
    key = torch.zeros(2, 1, 2)
    slots = torch.tensor([-1, 4])
    adapter["_write_kv"](layer, key, key, slots)
    assert calls[0][0] is layer and calls[0][3] is cache and calls[0][4] is slots
    layer.impl.kv_cache_dtype = "fp8"
    with pytest.raises(NotImplementedError, match="bf16/fp16"):
        adapter["_write_kv"](layer, key, key, slots)


def test_supported_runner_and_explicit_unsupported_topologies(adapter):
    cfg = SimpleNamespace(use_v2_model_runner=False, parallel_config=SimpleNamespace())
    adapter["_validate_runner_config"](cfg)
    cfg.use_v2_model_runner = True
    with pytest.raises(NotImplementedError, match="V2"):
        adapter["_validate_runner_config"](cfg)
    cfg.use_v2_model_runner = False
    for name in ["use_ubatching", "enable_dbo", "pipeline_parallel_size",
                 "decode_context_parallel_size", "prefill_context_parallel_size"]:
        cfg.parallel_config = SimpleNamespace(**{name: 2})
        with pytest.raises(NotImplementedError):
            adapter["_validate_runner_config"](cfg)


def test_runner_hook_fills_after_prepare_and_is_idempotent(adapter):
    runner_cls = adapter["gpu_model_runner"].GPUModelRunner
    runner_cls.get_model = lambda self: SimpleNamespace(kv_share_is_generated="buffer")
    seen = []
    adapter["_fill_generated"] = lambda runner, output, buffer: seen.append((runner.prepared, buffer))
    adapter["_install_runner_hook"]()
    hook = runner_cls._prepare_inputs
    adapter["_install_runner_hook"]()
    assert runner_cls._prepare_inputs is hook
    assert runner_cls()._prepare_inputs(object(), np.array([1])) == "prepared"
    assert seen == [(True, "buffer")]


def test_slot_mapping_cannot_overrun_key_value_or_mask(adapter):
    a, _ = make_attention(adapter, {"loop0": torch.arange(3), "loop1": torch.arange(3)})
    with pytest.raises(RuntimeError, match="exceeds token buffers"):
        a._keep_last_loop_kv(torch.zeros(3, 2), torch.zeros(2, 2))
    a.kv_share_is_generated = torch.zeros(2, dtype=torch.bool)
    with pytest.raises(RuntimeError, match="exceeds token buffers"):
        a._keep_last_loop_kv(torch.zeros(3, 2), torch.zeros(3, 2))


def test_unverified_backend_is_rejected(adapter):
    impl = type("Impl", (), {"__module__": "vllm.v1.attention.backends.flashinfer",
                            "kv_cache_dtype": "auto"})()
    with pytest.raises(NotImplementedError, match="does not support"):
        adapter["_write_kv"](SimpleNamespace(impl=impl), None, None, None)
