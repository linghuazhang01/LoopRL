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


def test_plc_loss_on_the_real_per_loop_outputs(models):
    """The PLC loss (GRPO + earlier-loop term) reaches the backbone only, never the gate."""
    from rltt_baseline import plc

    _, rltt = models
    ids, mask, pos, labels = _inputs()
    response_length = 4
    window = slice(-response_length - 1, -1)
    response_mask = mask[:, -response_length:].float()
    adv = torch.tensor([1.0, -1.0])

    def forward():
        out = rltt(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False,
                   return_per_loop_log_probs=True, return_exit_pdf=False, log_prob_labels=labels)
        return torch.stack([lp[:, window] for lp in out.per_loop_log_probs], dim=-1).float()

    with torch.no_grad():
        ell_old = forward()
    weights = plc.compute_loop_weights(ell_old, "is_trunc", trunc_cap=1.0)
    kl_weights = plc.compute_loop_weights(ell_old, "early_prior")
    rho = torch.exp(ell_old[..., :-1] - ell_old[..., -1:]).clamp(max=1.0)
    assert torch.allclose(weights[..., :-1], rho / 3, atol=1e-6)
    assert (weights[..., -1] == 0).all() and torch.allclose(kl_weights[..., :-1],
                                                            torch.full_like(rho, 1 / 3))

    rltt.zero_grad()
    ell_new = forward()
    assert torch.allclose(ell_new, ell_old, atol=1e-5)   # same policy
    pg_loss, metrics = plc.plc_policy_loss(ell_new, ell_old, adv, response_mask, weights,
                                           lam=0.1, clip_ratio=0.2, kl_coef=0.2,
                                           kl_weights=kl_weights)
    pg_loss.backward()
    assert rltt.model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0
    gate_grad = rltt.model.early_exit_gate.weight.grad
    assert gate_grad is None or gate_grad.abs().sum() == 0   # the gate is not in the loss
    assert len(metrics["plc/kl_k3_per_loop"]) == 4


@pytest.mark.parametrize("divergence", ["forward", "reverse"])
def test_loop_kd_on_the_real_per_loop_outputs(models, divergence):
    """``return_loop_kd``: full-vocabulary KL of every earlier loop towards the stop-gradient last
    loop on the response window; gradient reaches the backbone only, not the gate, and the
    PLC kd loss runs on it."""
    from rltt_baseline import plc
    from rltt_baseline.ouro_rltt.rltt_per_loop import kd_divergence

    base, rltt = models
    ids, mask, pos, labels = _inputs()
    window = 4
    with torch.no_grad():
        _, hiddens, _ = base.model(input_ids=ids, attention_mask=mask, position_ids=pos,
                                   use_cache=False)
        out = rltt(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False,
                   return_per_loop_log_probs=True, log_prob_labels=labels,
                   log_prob_temperature=0.8, return_loop_kd=True,
                   loop_kd_divergence=divergence, loop_kd_window=window)
    assert out.loop_kd.shape == (2, window, 3)
    log_q = F.log_softmax(base.lm_head(hiddens[-1][:, -window - 1:-1]).float() / 0.8, -1)
    for t in range(3):
        log_p = F.log_softmax(base.lm_head(hiddens[t][:, -window - 1:-1]).float() / 0.8, -1)
        torch.testing.assert_close(out.loop_kd[..., t], kd_divergence(log_p, log_q, divergence))
    assert out.loop_kd_stats["entropy_gap"].shape == (2, window, 3)

    rltt.zero_grad()
    out = rltt(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False,
               return_per_loop_log_probs=True, log_prob_labels=labels, return_loop_kd=True,
               loop_kd_divergence=divergence, loop_kd_window=window)
    ell = torch.stack([lp[:, -window - 1:-1] for lp in out.per_loop_log_probs], dim=-1).float()
    loss, _ = plc.plc_kd_loss(ell, ell.detach(), torch.zeros(2), mask[:, -window:].float(),
                              out.loop_kd, lam=1.0, clip_ratio=0.2)
    loss.backward()   # advantage 0: only the distillation term has a gradient
    assert rltt.model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0
    gate_grad = rltt.model.early_exit_gate.weight.grad
    assert gate_grad is None or gate_grad.abs().sum() == 0
    with pytest.raises(ValueError, match="loop_kd_window"):
        rltt(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False,
             return_loop_kd=True, loop_kd_window=ids.shape[1])


@pytest.mark.parametrize("adaptive", [False, True])
def test_extrap_on_the_real_per_loop_outputs(models, adaptive):
    """``return_extrap_log_probs``: the LoopCD policy of the last loop against loop 1, equal to
    ``loop_cd.guided_logprobs`` then the temperature; gradient reaches the backbone, and the
    EP-GRPO loss runs on it."""
    from rltt_baseline import extrap, loop_cd

    base, rltt = models
    ids, mask, pos, labels = _inputs()
    settings = extrap.ExtrapSettings(enabled=True, omega=0.5, adaptive=adaptive)
    with torch.no_grad():
        _, hiddens, _ = base.model(input_ids=ids, attention_mask=mask, position_ids=pos,
                                   use_cache=False)
        out = rltt(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False,
                   log_prob_labels=labels, log_prob_temperature=0.8, **settings.model_kwargs())
    guided = loop_cd.guided_logprobs(base.lm_head(hiddens[-1]), base.lm_head(hiddens[0]),
                                     settings.loop_cd())
    log_pi = F.log_softmax(guided / 0.8, -1)
    torch.testing.assert_close(out.extrap["log_probs"],
                               log_pi.gather(-1, labels.unsqueeze(-1)).squeeze(-1))

    rltt.zero_grad()
    out = rltt(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False,
               log_prob_labels=labels, **settings.model_kwargs())
    ext = out.extrap
    weights = extrap.token_weights(ext["omega"], ext["margin"], settings)
    loss, _ = extrap.extrap_policy_loss(ext["log_probs"], ext["log_probs"].detach(),
                                        torch.tensor([1.0, -1.0]), mask.float(), weights, 0.2)
    loss.backward()
    assert rltt.model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("divergence", ["reverse", "forward"])
def test_opd_full_kd_on_the_real_per_loop_outputs(models, divergence):
    """Full-vocabulary OPD: the frozen teacher's ``return_loop_hidden_states`` and the student's
    ``return_opd_kd`` give KL against log_softmax(1.5 log P4 - 0.5 log P1) of the teacher on the
    window; gradient reaches the student's backbone, never the teacher."""
    import copy

    from rltt_baseline import opd
    from rltt_baseline.ouro_rltt.rltt_per_loop import kd_divergence

    _, shared = models
    teacher = copy.deepcopy(shared).eval().requires_grad_(False)
    rltt = copy.deepcopy(shared)  # the student; the module fixture stays untouched
    torch.manual_seed(2)
    with torch.no_grad():  # a student that has moved away from the teacher
        for p in rltt.parameters():
            p.add_(0.01 * torch.randn_like(p))
    ids, mask, pos, _ = _inputs()
    settings = opd.OPDSettings(enabled=True, omega=0.5, estimator="full", divergence=divergence)
    window = ids.shape[1] - 1
    with torch.no_grad():
        t_hiddens = teacher(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False,
                            return_loop_hidden_states=True).loop_hidden_states
        _, s_hiddens, _ = rltt.model(input_ids=ids, attention_mask=mask, position_ids=pos,
                                     use_cache=False)
    assert len(t_hiddens) == 4
    rltt.zero_grad()
    out = rltt(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False,
               opd_teacher_hidden_states=(t_hiddens[0], t_hiddens[-1]),
               opd_teacher_lm_head=teacher.lm_head, opd_window=window,
               **settings.student_model_kwargs())
    kd = out.opd_kd["kd"]
    assert kd.shape == (2, window)
    log4 = F.log_softmax(teacher.lm_head(t_hiddens[-1][:, :-1]).float(), -1)
    log1 = F.log_softmax(teacher.lm_head(t_hiddens[0][:, :-1]).float(), -1)
    log_q = F.log_softmax(log4 + 0.5 * (log4 - log1), -1)
    log_p = F.log_softmax(rltt.lm_head(s_hiddens[-1][:, :-1]).float(), -1)
    torch.testing.assert_close(kd.detach(), kd_divergence(log_p, log_q, divergence))
    loss, _ = opd.opd_full_loss(kd, mask[:, 1:].float())
    loss.backward()
    assert rltt.model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in teacher.parameters())


def test_opd_online_teacher_on_the_real_per_loop_outputs(models):
    """ELPO: ``opd_online_teacher`` takes the teacher from the student's own loop 1 and loop 4
    (stop-gradient) and needs no teacher model; gradient still reaches the backbone."""
    import copy

    from rltt_baseline import opd
    from rltt_baseline.ouro_rltt.rltt_per_loop import kd_divergence

    _, shared = models
    rltt = copy.deepcopy(shared)
    ids, mask, pos, _ = _inputs()
    settings = opd.OPDSettings(enabled=True, omega=0.5, estimator="full", teacher="online")
    window = ids.shape[1] - 1
    with torch.no_grad():
        _, hiddens, _ = rltt.model(input_ids=ids, attention_mask=mask, position_ids=pos,
                                   use_cache=False)
    rltt.zero_grad()
    out = rltt(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False,
               opd_window=window, **settings.student_model_kwargs())
    kd = out.opd_kd["kd"]
    log4 = F.log_softmax(rltt.lm_head(hiddens[-1][:, :-1]).float(), -1)
    log1 = F.log_softmax(rltt.lm_head(hiddens[0][:, :-1]).float(), -1)
    log_q = F.log_softmax(log4 + 0.5 * (log4 - log1), -1)
    torch.testing.assert_close(kd.detach(), kd_divergence(log4, log_q, "reverse"))
    loss, _ = opd.opd_full_loss(kd, mask[:, 1:].float())
    loss.backward()
    assert rltt.model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="own loops"):
        rltt(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False,
             opd_teacher_hidden_states=(hiddens[0], hiddens[-1]),
             opd_teacher_lm_head=rltt.lm_head, opd_window=window,
             **settings.student_model_kwargs())
