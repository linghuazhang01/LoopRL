"""Tiny random Ouro through the RLTT model directory (needs transformers >= 4.56).

The project venv pins transformers 4.47.1, so this test skips there. Run it with a newer
transformers on PYTHONPATH, e.g. a ``uv pip install --target`` directory:
    PYTHONPATH=<target>:code/src python -m pytest code/tests/test_rltt_baseline_modeling.py
"""

import inspect
import json
import shutil
import types

import numpy as np
import pytest
import torch
import torch.nn.functional as F

transformers = pytest.importorskip("transformers")
if tuple(int(x) for x in transformers.__version__.split(".")[:2]) < (4, 56):
    pytest.skip("Ouro modeling needs transformers >= 4.56", allow_module_level=True)

from rltt_baseline import loop_audit, paths, prepare_model  # noqa: E402

VENDOR = paths.ouro_source_dir()
TINY = {"vocab_size": 64, "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
        "layer_types": ["full_attention"] * 2, "max_window_layers": 2,
        "max_position_embeddings": 128, "torch_dtype": "float32"}


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ouro")
    src = tmp / "src"
    src.mkdir()
    for name in ("modeling_ouro.py", "configuration_ouro.py"):
        shutil.copy2(VENDOR / name, src / name)
    cfg = json.loads((VENDOR / "config.json").read_text())
    cfg.update(TINY)
    (src / "config.json").write_text(json.dumps(cfg))
    torch.manual_seed(0)
    base = transformers.AutoModelForCausalLM.from_config(
        transformers.AutoConfig.from_pretrained(src, trust_remote_code=True),
        trust_remote_code=True, attn_implementation="eager")
    with torch.no_grad():
        base.model.early_exit_gate.bias.fill_(-0.5)
    base.save_pretrained(src)
    for name in ("modeling_ouro.py", "configuration_ouro.py"):
        shutil.copy2(VENDOR / name, src / name)

    out = tmp / "rltt"
    allowed = {prepare_model.sha256(prepare_model.vendor_modeling())}
    prepare_model.build(src, out, allowed, chunk_size=3, checkpoint_chunks=False,
                        detach_exit_pdf=False)
    # reference = official code with only the check_model_inputs compat edit
    ref = tmp / "ref"
    shutil.copytree(src, ref)
    (ref / "modeling_ouro.py").write_text(
        prepare_model.compat_base_source((VENDOR / "modeling_ouro.py").read_text()))
    base = transformers.AutoModelForCausalLM.from_pretrained(
        ref, trust_remote_code=True, attn_implementation="eager").eval()
    rltt = transformers.AutoModelForCausalLM.from_pretrained(
        out, trust_remote_code=True, attn_implementation="eager").eval()
    return base, rltt


def _inputs():
    torch.manual_seed(1)
    input_ids = torch.randint(3, 64, (2, 7))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, :2] = 0  # left padding, as in verl batches
    position_ids = (attention_mask.cumsum(-1) - 1).clamp_min(0)
    labels = F.pad(input_ids[:, 1:], (0, 1), value=0)
    return input_ids, attention_mask, position_ids, labels


def test_forward_signature_has_rltt_flags(models):
    _, rltt = models
    params = inspect.signature(rltt.forward).parameters
    for name in ("return_per_loop_log_probs", "return_per_loop_logits", "return_exit_pdf",
                 "log_prob_labels", "log_prob_temperature"):
        assert name in params


def test_default_forward_matches_official(models):
    base, rltt = models
    ids, mask, pos, _ = _inputs()
    with torch.no_grad():
        a = base(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False).logits
        b = rltt(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False).logits
    torch.testing.assert_close(a, b)


def test_per_loop_outputs_match_official_internals(models):
    base, rltt = models
    ids, mask, pos, labels = _inputs()
    with torch.no_grad():
        out = rltt(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False,
                   return_per_loop_log_probs=True, return_per_loop_logits=True,
                   return_exit_pdf=True, log_prob_labels=labels, log_prob_temperature=0.8)
        _, hiddens, gates = base.model(input_ids=ids, attention_mask=mask, position_ids=pos,
                                       use_cache=False)
        final = base(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False).logits
    assert len(out.per_loop_log_probs) == len(out.per_loop_entropies) == 4
    for t, hidden in enumerate(hiddens):
        logp = F.log_softmax(base.lm_head(hidden).float() / 0.8, -1)
        torch.testing.assert_close(out.per_loop_log_probs[t],
                                   logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1))
        torch.testing.assert_close(out.per_loop_entropies[t], -(logp.exp() * logp).sum(-1))
        torch.testing.assert_close(out.per_loop_logits[t], base.lm_head(hidden))
    # the last loop is what the standard path (and vLLM sampling) uses
    torch.testing.assert_close(out.per_loop_logits[-1], final)
    lam = [torch.sigmoid(g.squeeze(-1)) for g in gates]
    first = lam[0]
    torch.testing.assert_close(out.exit_pdf[..., 0], first)
    torch.testing.assert_close(out.exit_pdf.sum(-1), torch.ones_like(first))


def test_rltt_loss_backpropagates_to_backbone_and_gate(models):
    _, rltt = models
    ids, mask, pos, labels = _inputs()
    rltt.zero_grad()
    out = rltt(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False,
               return_per_loop_log_probs=True, return_exit_pdf=True, log_prob_labels=labels)
    weights = out.exit_pdf
    loss = -sum((weights[..., t] * lp).mean() for t, lp in enumerate(out.per_loop_log_probs))
    loss.backward()
    assert rltt.model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0
    assert rltt.model.early_exit_gate.weight.grad.abs().sum() > 0  # exit_pdf not detached


class _OuroActor:
    """The standard-path ``compute_log_prob`` of a verl actor, padded forward."""

    def __init__(self, module):
        self.actor_module = module

    def _forward_micro_batch(self, micro_batch, temperature):
        return self.actor_module(input_ids=micro_batch["input_ids"],
                                 attention_mask=micro_batch["attention_mask"],
                                 position_ids=micro_batch["position_ids"], use_cache=False).logits

    def compute_log_prob(self, data):
        with torch.no_grad():
            return self._forward_micro_batch(data.batch, data.meta_info["temperature"])


def test_loop_audit_reads_the_loops_of_the_real_model(models, tmp_path):
    _, rltt = models
    ids, mask, pos, labels = _inputs()
    response_length = 3
    actor = _OuroActor(rltt)
    loop_audit.attach_loop_audit(actor, {"dir": str(tmp_path), "every_n_steps": 1, "top_k": 4,
                                         "save_tokens": True, "chunk_tokens": 2}, rank=0)
    data = types.SimpleNamespace(
        meta_info={"temperature": 0.8},
        batch={"input_ids": ids, "attention_mask": mask, "position_ids": pos,
               "responses": ids[:, -response_length:]})
    logits = actor.compute_log_prob(data=data)
    with torch.no_grad():
        out = rltt(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False,
                   return_per_loop_log_probs=True, log_prob_labels=labels,
                   log_prob_temperature=0.8)
    columns = slice(ids.size(1) - response_length - 1, -1)
    with np.load(tmp_path / "step_000001" / "tokens_rank00.npz") as shard:
        for t in range(4):
            torch.testing.assert_close(torch.from_numpy(shard["sampled_log_prob"][:, t]),
                                       out.per_loop_log_probs[t][:, columns].flatten())
            torch.testing.assert_close(torch.from_numpy(shard["entropy"][:, t]),
                                       out.per_loop_entropies[t][:, columns].flatten())
        final_top = logits[:, columns].flatten(0, 1).topk(4, dim=-1).indices
        assert torch.equal(torch.from_numpy(shard["topk_ids"][:, -1].astype(np.int64)), final_top)
