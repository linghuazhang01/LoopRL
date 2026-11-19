"""EP-GRPO (rltt_baseline.extrap): policy, gradient, loss, settings and config wiring (CPU)."""

import json
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from rltt_baseline import config as rcfg
from rltt_baseline import extrap, loop_cd, paths, plc
from rltt_baseline.ouro_rltt.rltt_per_loop import extrap_token_stats, per_loop_token_stats

CONFIG = paths.config_file("extrap_grpo_loopcd_1p4b_8gpu.yaml")
ENV = {
    "LOOPRL_OURO_DIR": "/models/Ouro-1.4B-Thinking",
    "LOOPRL_MODEL_DIR": "/models/Ouro-1.4B-Thinking-RLTT",
    "LOOPRL_TRAIN_FILE": "/data/deepmath_level6.parquet",
    "LOOPRL_MATH500_JSONL": "/data/math500.jsonl",
    "LOOPRL_OUTPUT_DIR": "/out/extrap",
    "LOOPRL_ENV_FILE": "/nonexistent/env.local",  # never read a real env.local in tests
}
V, H, B, S, T = 11, 6, 2, 5, 4


@pytest.fixture
def env(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)


def _close(a, b, **kw):
    torch.testing.assert_close(a, b, atol=kw.get("atol", 1e-5), rtol=kw.get("rtol", 1e-5))


def _loops(requires_grad=False):
    g = torch.Generator().manual_seed(42)
    head = torch.nn.Linear(H, V, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.randn(V, H, generator=g))
    hiddens = [torch.randn(B, S, H, generator=g, requires_grad=requires_grad) for _ in range(T)]
    labels = torch.randint(0, V, (B, S), generator=g)
    return head, hiddens, labels


# ---------- the policy matches the vLLM LoopCD rollout ----------

@pytest.mark.parametrize("cfg", [
    loop_cd.LoopCDConfig(omega=0.5),
    loop_cd.LoopCDConfig(omega=1.0, adaptive=True),
    loop_cd.LoopCDConfig(omega=0.3, ref_loop=2, delta_cap=1.0),
])
@pytest.mark.parametrize("temperature", [1.0, 0.6])
def test_policy_equals_loop_cd_then_temperature(cfg, temperature):
    head, hiddens, labels = _loops()
    out = extrap_token_stats(hiddens, head, labels, omega=cfg.omega, ref_loop=cfg.ref_loop,
                             adaptive=cfg.adaptive, delta_cap=cfg.delta_cap,
                             temperature=temperature, chunk_size=2)
    z_s, z_r = head(hiddens[-1]), head(hiddens[cfg.ref_loop - 1])
    # vLLM: compute_logits returns guided log-probs, then the sampler applies the temperature
    log_pi = F.log_softmax(loop_cd.guided_logprobs(z_s, z_r, cfg) / temperature, -1)
    _close(out["log_probs"], log_pi.gather(-1, labels.unsqueeze(-1)).squeeze(-1))
    _close(out["entropy"], -(log_pi.exp() * log_pi).sum(-1))
    if cfg.adaptive:
        _close(out["omega"], loop_cd.adaptive_omega(z_s, cfg.omega).squeeze(-1))
    else:
        assert torch.all(out["omega"] == cfg.omega)
    top2 = torch.softmax(z_s, -1).topk(2, -1).values
    _close(out["margin"], 1 - (top2[..., 0] - top2[..., 1]))


def test_omega_zero_is_the_last_loop():
    head, hiddens, labels = _loops()
    out = extrap_token_stats(hiddens, head, labels, omega=0.0, temperature=0.8)
    log_probs, entropies = per_loop_token_stats(hiddens, head, labels, temperature=0.8)
    _close(out["log_probs"], log_probs[-1])
    _close(out["entropy"], entropies[-1])


# ---------- gradient ----------

@pytest.mark.parametrize("ref_grad", [False, True])
def test_gradient_is_centred_on_the_policy(ref_grad):
    """d log pi(y) / d z_s = (1 + w)(e_y - pi); d / d z_r = -w (e_y - pi) only with ref_grad."""
    w = 0.5
    z = [torch.randn(1, 1, V, requires_grad=True) for _ in range(T)]
    labels = torch.tensor([[3]])
    out = extrap_token_stats(z, torch.nn.Identity(), labels, omega=w, ref_grad=ref_grad)
    grads = torch.autograd.grad(out["log_probs"].sum(), z, allow_unused=True)
    log_pi = F.log_softmax((1 + w) * F.log_softmax(z[-1], -1) - w * F.log_softmax(z[0], -1), -1)
    centred = F.one_hot(labels, V).float() - log_pi.exp().detach()
    _close(grads[-1], (1 + w) * centred)
    if ref_grad:
        _close(grads[0], -w * centred)
    else:
        assert grads[0] is None or torch.all(grads[0] == 0)
    assert all(g is None or torch.all(g == 0) for g in grads[1:-1])
    # centred: E_{y ~ pi}[e_y - pi] = 0 (no prefix-dependent drift)
    pi = log_pi.exp().detach().reshape(V)
    _close(pi @ (torch.eye(V) - pi), torch.zeros(V), atol=1e-6)


def test_chunks_and_checkpointing_do_not_change_values_or_gradients():
    head, hiddens, labels = _loops(requires_grad=True)
    results = []
    for chunk in (None, 2):
        out = extrap_token_stats(hiddens, head, labels, omega=0.7, adaptive=True,
                                 chunk_size=chunk)
        grads = torch.autograd.grad(out["log_probs"].sum(), [hiddens[-1], hiddens[0]],
                                    allow_unused=True)
        results.append((out["log_probs"].detach(), grads))
    _close(results[0][0], results[1][0])
    _close(results[0][1][0], results[1][1][0])
    assert results[0][1][1] is None and results[1][1][1] is None  # reference: stop-gradient


def test_ref_loop_must_be_an_earlier_loop():
    head, hiddens, labels = _loops()
    with pytest.raises(ValueError, match="ref_loop"):
        extrap_token_stats(hiddens, head, labels, omega=0.5, ref_loop=T)


# ---------- loss ----------

def _batch():
    torch.manual_seed(42)
    log_prob = (torch.rand(B, S) * -2).requires_grad_()
    old = log_prob.detach() - 0.01
    adv = torch.tensor([[1.0] * S, [-0.5] * S])
    mask = torch.ones(B, S)
    mask[1, -2:] = 0
    return log_prob, old, adv, mask


@pytest.mark.parametrize("row_div", [None, torch.tensor([3.0, 2.0])])
def test_unit_weights_are_grpo(row_div):
    log_prob, old, adv, mask = _batch()
    ours, _ = extrap.extrap_policy_loss(log_prob, old, adv, mask, torch.ones(B, S), 0.2,
                                        row_div=row_div)
    ref, _ = plc.grpo_policy_loss(log_prob, old, adv, mask, 0.2, row_div=row_div)
    _close(ours, ref)


def test_weights_scale_each_token():
    log_prob, _, adv, mask = _batch()
    old = log_prob.detach()  # ratio 1: no clipping, the loss is linear in the weight
    weights = torch.rand(B, S) + 0.5
    loss, _ = extrap.extrap_policy_loss(log_prob, old, adv, mask, weights, 0.2)
    grad, = torch.autograd.grad(loss, log_prob)
    base, _ = plc.grpo_policy_loss(log_prob, old, adv, mask, 0.2)
    base_grad, = torch.autograd.grad(base, log_prob)
    _close(grad, base_grad * weights)
    with pytest.raises(ValueError, match="positive"):
        extrap.extrap_policy_loss(log_prob, old, adv, mask, torch.zeros(B, S), 0.2)


def test_token_weights():
    strength = torch.tensor([[0.5, 0.2]])
    margin = torch.tensor([[0.1, 0.9]])
    s = extrap.ExtrapSettings(omega=0.5, margin_alpha=0.5)
    _close(extrap.token_weights(strength, margin, s), 1 + 0.5 * margin)
    s = extrap.ExtrapSettings(omega=0.5)  # no 1 / (1 + w) normalization (user 2026-11-19)
    _close(extrap.token_weights(strength, margin, s), torch.ones(1, 2))
    with pytest.raises(KeyError, match="unknown extrap keys"):
        extrap.ExtrapSettings.from_mapping({"normalize_scale": True})


def test_loss_step_on_the_last_loop_logits():
    """The logit step is (1 + w) A (e_y - pi): GRPO's (e_y - p) with pi for p, times 1 + w."""
    w = 0.5
    z = [torch.randn(1, 1, V, requires_grad=True) for _ in range(T)]
    labels = torch.tensor([[2]])
    out = extrap_token_stats(z, torch.nn.Identity(), labels, omega=w)
    weights = extrap.token_weights(out["omega"], out["margin"], extrap.ExtrapSettings(omega=w))
    loss, _ = extrap.extrap_policy_loss(out["log_probs"], out["log_probs"].detach(),
                                        torch.ones(1, 1), torch.ones(1, 1), weights, 0.2)
    grad, = torch.autograd.grad(loss, z[-1])
    log_pi = F.log_softmax((1 + w) * F.log_softmax(z[-1], -1) - w * F.log_softmax(z[0], -1), -1)
    _close(-grad, (1 + w) * (F.one_hot(labels, V).float() - log_pi.exp().detach()))


# ---------- settings ----------

def test_settings_defaults_and_conversions():
    s = extrap.ExtrapSettings.from_mapping({"enabled": True, "omega": 1, "delta_cap": 3})
    assert s.omega == 1.0 and s.delta_cap == 3.0 and s.ref_loop == 1 and not s.ref_grad
    assert s.loop_cd() == loop_cd.LoopCDConfig(ref_loop=1, omega=1.0, space="logits",
                                               delta_cap=3.0)
    kwargs = s.model_kwargs()
    assert kwargs["return_extrap_log_probs"] is True and kwargs["extrap_omega"] == 1.0


@pytest.mark.parametrize("values,error,match", [
    ({"omega": -0.1}, ValueError, "omega"),
    ({"margin_alpha": -1}, ValueError, "margin_alpha"),
    ({"ref_loop": 0}, ValueError, "ref_loop"),
    ({"ref_loop": 1.5}, TypeError, "integer"),
    ({"delta_cap": 0}, ValueError, "delta_cap"),
    ({"adaptive": 1}, TypeError, "bool"),
    ({"omega": "0.5"}, TypeError, "number"),
    ({"gamma": 1}, KeyError, "unknown extrap keys"),
])
def test_settings_validation(values, error, match):
    with pytest.raises(error, match=match):
        extrap.ExtrapSettings.from_mapping(values)


def test_ref_loop_is_checked_against_the_depth():
    with pytest.raises(ValueError, match="< the 4 loops"):
        extrap.ExtrapSettings(ref_loop=4).validate(4)


# ---------- config and launcher ----------

def test_config_is_ep_grpo_with_omega_half(env):
    cfg = rcfg.load_config(str(CONFIG))
    s = rcfg.extrap_settings(cfg)
    assert s == extrap.ExtrapSettings(enabled=True, omega=0.5)
    assert cfg.trainer == "rltt" and not rcfg.plc_settings(cfg).enabled
    assert cfg.rollout.kv_layout == "full"
    for name in ("grpo_baseline_1p4b_8gpu.yaml", "rltt_baseline_1p4b_8gpu.yaml",
                 "plc_grpo_loopaux_1p4b_8gpu.yaml", "grpo_baseline_8gpu.yaml"):
        assert not rcfg.extrap_settings(rcfg.load_config(str(paths.config_file(name)))).enabled


def _args(**kw):
    return SimpleNamespace(use_rltt_model=True, loop_weighting="uniform", total_ut_steps=4, **kw)


def test_overrides_point_the_rollout_at_the_same_policy(env):
    cfg = rcfg.load_config(str(CONFIG))
    overrides = rcfg.add_extrap_overrides(cfg, rcfg.verl_overrides(cfg), _args())
    ar = overrides["actor_rollout_ref"]
    assert ar["actor"]["extrap"]["omega"] == 0.5 and ar["actor"]["extrap"]["enabled"]
    assert loop_cd.LIB in ar["model"]["external_lib"]
    vllm = ar["rollout"]["engine_kwargs"]["vllm"]
    assert vllm["hf_overrides"] == {"architectures": [loop_cd.ARCHITECTURE],
                                    "loop_cd_ref_loop": 1, "loop_cd_omega": 0.5,
                                    "loop_cd_adaptive": False, "loop_cd_space": "logits",
                                    "loop_cd_delta_cap": None}
    assert loop_cd.from_hf_config(SimpleNamespace(total_ut_steps=4, **vllm["hf_overrides"])) \
        == rcfg.extrap_settings(cfg).loop_cd()
    assert vllm["compilation_config"] == loop_cd.COMPILATION
    assert ar["rollout"]["calculate_log_probs"] is True
    # shared rollout settings are untouched
    assert ar["rollout"]["max_num_batched_tokens"] == 32768 and ar["rollout"]["top_p"] == 1.0
    off = rcfg.load_config(str(CONFIG), ["rollout.log_probs=false"])
    over = rcfg.add_extrap_overrides(off, rcfg.verl_overrides(off), _args())
    assert over["actor_rollout_ref"]["rollout"]["calculate_log_probs"] is False


@pytest.mark.parametrize("cli,args,match", [
    (["plc.enabled=true"], {}, "different methods"),
    (["trainer=grpo"], {}, "trainer=rltt"),
    (["rollout.kv_layout=decode_last"], {}, "kv_layout=full"),
    (["extrap.ref_loop=4"], {}, "< the 4 loops"),
    (["verl_overrides.actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides."
      "loop_cd_omega=0.3"], {}, "do not override"),
    ([], {"use_rltt_model": False}, "use_rltt_model"),
])
def test_conflicting_settings_are_refused(env, cli, args, match):
    cfg = rcfg.load_config(str(CONFIG), cli)
    values = {"use_rltt_model": True, "loop_weighting": "uniform", "total_ut_steps": 4, **args}
    with pytest.raises(ValueError, match=match):
        rcfg.add_extrap_overrides(cfg, rcfg.verl_overrides(cfg), SimpleNamespace(**values))


def test_dry_run(env, capsys, monkeypatch):
    from rltt_baseline import launch

    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "PYTHONPATH"):
        monkeypatch.setenv(key, "")
    monkeypatch.setattr(sys, "path", list(sys.path))
    assert launch.main(["--config", str(CONFIG), "--dry-run"]) == 0
    resolved = json.loads(capsys.readouterr().out)
    ar = resolved["verl_overrides"]["actor_rollout_ref"]
    assert ar["actor"]["extrap"]["omega"] == 0.5
    assert ar["rollout"]["engine_kwargs"]["vllm"]["hf_overrides"]["architectures"] == \
        [loop_cd.ARCHITECTURE]
    assert ar["loss_agg"] == {"mode": "seq-mean-token-mean", "norm_len": None}
    assert "plc" not in ar["actor"]
    assert resolved["config"]["extrap"]["enabled"] is True
    assert resolved["reference_policy"].startswith("none")


# ---------- verl 0.9 objective ----------

@pytest.mark.parametrize("margin_alpha", [0.0, 0.5])
def test_native_objective_matches_the_legacy_loss(margin_alpha):
    from rltt_baseline.verl09_losses import objective

    log_prob, old, adv, mask = _batch()
    strength = torch.full((B, S), 0.5)
    margin = torch.rand(B, S)
    ell = torch.rand(B, S, T) * -2
    output = dict(log_probs=log_prob, entropy=torch.zeros(B, S), per_loop_log_probs=ell,
                  per_loop_entropies=torch.zeros(B, S, T), extrap_omega=strength,
                  extrap_margin=margin)
    batch = dict(old_log_probs=old, response_mask=mask, advantages=adv)
    settings = dict(method="extrap", extrap=dict(enabled=True, omega=0.5,
                                                 margin_alpha=margin_alpha))
    actual, metrics = objective(dict(clip_ratio=0.2), settings, output, batch)
    weights = extrap.token_weights(strength, margin, extrap.ExtrapSettings(
        omega=0.5, margin_alpha=margin_alpha))
    expected, _ = extrap.extrap_policy_loss(log_prob, old, adv, mask, weights, 0.2)
    _close(actual, expected)
    _close(torch.autograd.grad(actual, log_prob, retain_graph=True)[0],
           torch.autograd.grad(expected, log_prob)[0])
    assert metrics["extrap/omega_mean"] == pytest.approx(0.5)
