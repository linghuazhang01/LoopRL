"""PLC ``loop_loss: kd``: cross-loop distillation helper, loss, settings, config and audit columns.

Pure torch on CPU. The model-forward wiring (``return_loop_kd``) is checked against the real
Ouro code in ``tests/test_rltt_baseline_modeling.py`` (needs transformers >= 4.56).
"""

import json
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from rltt_baseline import config as rcfg
from rltt_baseline import loop_audit as la
from rltt_baseline import paths, plc
from rltt_baseline.ouro_rltt import rltt_per_loop as rpl

LOOPAUX_CONFIG = paths.config_file("plc_grpo_loopaux_1p4b_8gpu.yaml")
LOOPKD_CONFIG = paths.config_file("plc_grpo_loopkd_1p4b_8gpu.yaml")
ENV = {
    "LOOPRL_OURO_DIR": "/models/Ouro-1.4B-Thinking",
    "LOOPRL_MODEL_DIR": "/models/Ouro-1.4B-Thinking-RLTT",
    "LOOPRL_TRAIN_FILE": "/data/deepmath_level6.parquet",
    "LOOPRL_MATH500_JSONL": "/data/math500.jsonl",
    "LOOPRL_OUTPUT_DIR": "/out/plc",
    "LOOPRL_ENV_FILE": "/nonexistent/env.local",  # never read a real env.local in tests
}
T, V, H = 4, 11, 6
DIVERGENCES = ("forward", "reverse")


@pytest.fixture
def env(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)


def _log_probs(seed=0, n=5):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, V, generator=g) * 2
    q = torch.randn(n, V, generator=g) * 2
    return z, F.log_softmax(q, -1)


def _hiddens(seed=1, B=2, S=5, requires_grad=False):
    g = torch.Generator().manual_seed(seed)
    hiddens = [torch.randn(B, S, H, generator=g).requires_grad_(requires_grad) for _ in range(T)]
    torch.manual_seed(seed)
    head = torch.nn.Linear(H, V, bias=False)
    return hiddens, head


# ---------- divergence, logit step, entropy push ----------

def test_divergence_constants_agree():
    assert rpl.KD_DIVERGENCES == plc.KD_DIVERGENCES


@pytest.mark.parametrize("divergence", DIVERGENCES)
def test_kd_divergence_matches_kl_div(divergence):
    z, log_q = _log_probs()
    log_p = F.log_softmax(z, -1)
    if divergence == "forward":   # KL(q || p)
        expected = F.kl_div(log_p, log_q, reduction="none", log_target=True).sum(-1)
    else:                         # KL(p || q)
        expected = F.kl_div(log_q, log_p, reduction="none", log_target=True).sum(-1)
    torch.testing.assert_close(rpl.kd_divergence(log_p, log_q, divergence), expected)
    assert (rpl.kd_divergence(log_p, log_q, divergence) >= 0).all()
    torch.testing.assert_close(rpl.kd_divergence(log_q, log_q, divergence),
                               torch.zeros(len(z)), atol=1e-6, rtol=0)


@pytest.mark.parametrize("divergence", DIVERGENCES)
def test_kd_logit_step_is_minus_the_gradient(divergence):
    z, log_q = _log_probs(seed=2)
    z = z.requires_grad_(True)
    loss = rpl.kd_divergence(F.log_softmax(z, -1), log_q, divergence).sum()
    (grad,) = torch.autograd.grad(loss, z)
    step = rpl.kd_logit_step(F.log_softmax(z.detach(), -1), log_q, divergence)
    torch.testing.assert_close(step, -grad, atol=1e-6, rtol=1e-5)
    if divergence == "forward":
        torch.testing.assert_close(step, log_q.exp() - F.softmax(z.detach(), -1))


@pytest.mark.parametrize("divergence", DIVERGENCES)
def test_kd_entropy_push_is_the_directional_derivative(divergence):
    z, log_q = _log_probs(seed=3)
    log_p = F.log_softmax(z, -1)
    step = rpl.kd_logit_step(log_p, log_q, divergence)
    eta = torch.zeros(1, requires_grad=True)
    log_moved = F.log_softmax(z + eta * step, -1)
    entropy = -(log_moved.exp() * log_moved).sum(-1)
    derivative = torch.stack([torch.autograd.grad(e, eta, retain_graph=True)[0][0]
                              for e in entropy])
    torch.testing.assert_close(rpl.kd_entropy_push(log_p, log_q, divergence), derivative,
                               atol=1e-5, rtol=1e-4)


def test_entropy_push_is_negative_towards_a_sharper_teacher():
    mild, sharp = torch.zeros(1, V), torch.zeros(1, V)
    mild[0, 0], sharp[0, 0] = 1.0, 5.0   # (a uniform student is at the maximum: push 0)
    log_p, log_q = F.log_softmax(mild, -1), F.log_softmax(sharp, -1)
    for divergence in DIVERGENCES:
        assert rpl.kd_entropy_push(log_p, log_q, divergence).item() < 0


# ---------- per_loop_kd ----------

def _direct(hiddens, head, temperature, divergence):
    with torch.no_grad():  # sg P^(T): neither the last loop nor lm_head gets the teacher's grad
        log_q = F.log_softmax(head(hiddens[-1]) / temperature, -1)
    q_entropy = -(log_q.exp() * log_q).sum(-1)
    kd, stats = [], {name: [] for name in rpl.KD_STATS}
    for h in hiddens[:-1]:
        log_p = F.log_softmax(head(h) / temperature, -1)
        kd.append(rpl.kd_divergence(log_p, log_q, divergence))
        lp = log_p.detach()
        stats["kl_forward"].append(rpl.kd_divergence(lp, log_q, "forward"))
        stats["kl_reverse"].append(rpl.kd_divergence(lp, log_q, "reverse"))
        stats["entropy_gap"].append(-(lp.exp() * lp).sum(-1) - q_entropy)
        stats["top1_agree"].append((lp.argmax(-1) == log_q.argmax(-1)).float())
        stats["teacher_top1_prob"].append(
            lp.exp().gather(-1, log_q.argmax(-1, keepdim=True)).squeeze(-1))
        stats["entropy_push"].append(rpl.kd_entropy_push(lp, log_q, divergence))
    return torch.stack(kd, -1), {k: torch.stack(v, -1).detach() for k, v in stats.items()}


@pytest.mark.parametrize("divergence", DIVERGENCES)
@pytest.mark.parametrize("chunk", [None, 1, 2, 100])
def test_per_loop_kd_matches_a_direct_computation(divergence, chunk):
    hiddens, head = _hiddens()
    with torch.no_grad():
        kd, stats = rpl.per_loop_kd(hiddens, head, temperature=0.7, divergence=divergence,
                                    chunk_size=chunk)
        expected_kd, expected_stats = _direct(hiddens, head, 0.7, divergence)
    assert kd.shape == (2, 5, T - 1) and kd.dtype == torch.float32
    torch.testing.assert_close(kd, expected_kd, atol=1e-6, rtol=1e-5)
    assert set(stats) == set(rpl.KD_STATS)
    for name in rpl.KD_STATS:
        torch.testing.assert_close(stats[name], expected_stats[name], atol=1e-6, rtol=1e-5)
    # the stats report both directions whatever the loss uses
    torch.testing.assert_close(stats["kl_forward" if divergence == "forward" else "kl_reverse"],
                               kd)


@pytest.mark.parametrize("divergence", DIVERGENCES)
def test_per_loop_kd_gradient_reaches_the_students_only(divergence):
    hiddens, head = _hiddens(requires_grad=True)
    weights = torch.rand(2, 5, T - 1, generator=torch.Generator().manual_seed(7))
    kd, _ = rpl.per_loop_kd(hiddens, head, divergence=divergence, chunk_size=2)
    (kd * weights).sum().backward()          # chunked and checkpointed
    got = [h.grad.clone() if h.grad is not None else None for h in hiddens]
    head_grad = head.weight.grad.clone()

    ref, ref_head = _hiddens(requires_grad=True)
    expected, _ = _direct(ref, ref_head, 1.0, divergence)
    (expected * weights).sum().backward()    # one chunk, no checkpoint
    for t in range(T - 1):
        torch.testing.assert_close(got[t], ref[t].grad, atol=1e-6, rtol=1e-5)
    assert got[-1] is None or got[-1].abs().sum() == 0   # the teacher is stop-gradient
    torch.testing.assert_close(head_grad, ref_head.weight.grad, atol=1e-6, rtol=1e-5)


def test_per_loop_kd_rejects_bad_inputs():
    hiddens, head = _hiddens()
    with pytest.raises(ValueError, match="divergence"):
        rpl.per_loop_kd(hiddens, head, divergence="js")
    with pytest.raises(ValueError, match="temperature"):
        rpl.per_loop_kd(hiddens, head, temperature=0)
    with pytest.raises(ValueError, match="two loops"):
        rpl.per_loop_kd(hiddens[:1], head)


# ---------- loss ----------

def _batch(seed=0, B=3, L=5):
    g = torch.Generator().manual_seed(seed)
    ell_old = -torch.rand(B, L, T, generator=g) * 4
    ell_new = ell_old + 0.3 * torch.randn(B, L, T, generator=g)
    adv = torch.randn(B, generator=g)
    mask = torch.ones(B, L)
    mask[0, -2:] = 0
    kd = torch.rand(B, L, T - 1, generator=g)
    return ell_old, ell_new, adv, mask, kd


@pytest.mark.parametrize("loss_agg_mode", ["token-mean", "seq-mean-token-mean"])
def test_lam0_kd_loss_is_grpo(loss_agg_mode):
    ell_old, ell_new, adv, mask, kd = _batch()
    weights = plc.compute_loop_weights(ell_old, "early_prior")
    grpo, _ = plc.plc_policy_loss(ell_new, ell_old, adv, mask, weights, lam=0.0, clip_ratio=0.2,
                                  loss_agg_mode=loss_agg_mode)
    loss, _ = plc.plc_kd_loss(ell_new, ell_old, adv, mask, kd, lam=0.0, clip_ratio=0.2,
                              loss_agg_mode=loss_agg_mode)
    torch.testing.assert_close(loss, grpo)


def test_kd_loss_is_grpo_plus_lam_alpha_kd_with_that_gradient():
    ell_old, ell_new, adv, mask, kd = _batch(seed=1)
    kd = kd.requires_grad_(True)
    grpo, _ = plc.plc_kd_loss(ell_new, ell_old, adv, mask, kd.detach(), lam=0.0, clip_ratio=0.2)
    loss, metrics = plc.plc_kd_loss(ell_new, ell_old, adv, mask, kd, lam=0.3, clip_ratio=0.2)
    expected = grpo + 0.3 * (kd.detach().sum(-1) / (T - 1) * mask).sum() / mask.sum()
    torch.testing.assert_close(loss, expected)
    (grad,) = torch.autograd.grad(loss, kd)
    torch.testing.assert_close(grad, (0.3 / (T - 1) / mask.sum()) * mask[..., None].expand_as(kd))
    assert len(metrics["plc/kd_loss_per_loop"]) == T - 1
    assert len(metrics["plc/ratio_mean_per_loop"]) == T


def test_kd_loss_adds_the_anchor_on_the_earlier_loops():
    ell_old, ell_new, adv, mask, kd = _batch(seed=2)
    ell_init = ell_old - 0.2
    plain, _ = plc.plc_kd_loss(ell_new, ell_old, adv, mask, kd, lam=0.1, clip_ratio=0.2)
    loss, metrics = plc.plc_kd_loss(ell_new, ell_old, adv, mask, kd, lam=0.1, clip_ratio=0.2,
                                    init_kl_coef=0.5, logp_init=ell_init, init_kl_form="k2")
    pen = plc.init_anchor_penalty(ell_new, ell_init, "k2")[..., :-1].sum(-1) / (T - 1)
    torch.testing.assert_close(loss, plain + 0.1 * 0.5 * (pen * mask).sum() / mask.sum())
    assert "plc/init_pen/loop_1" in plc.flatten_loop_metrics(metrics)
    with pytest.raises(ValueError, match="logp_init"):
        plc.plc_kd_loss(ell_new, ell_old, adv, mask, kd, lam=0.1, clip_ratio=0.2,
                        init_kl_coef=0.5)


def test_kd_loss_rejects_a_kd_tensor_of_the_wrong_shape():
    ell_old, ell_new, adv, mask, kd = _batch()
    with pytest.raises(ValueError, match="T - 1"):
        plc.plc_kd_loss(ell_new, ell_old, adv, mask, kd[..., :2], lam=0.1, clip_ratio=0.2)


def test_kd_stats_give_one_key_per_earlier_loop():
    hiddens, head = _hiddens()
    with torch.no_grad():
        _, stats = rpl.per_loop_kd(hiddens, head)
    mask = torch.ones(2, 5)
    mask[1, 3:] = 0
    flat = plc.flatten_loop_metrics(plc.kd_stats(stats, mask))
    for name in rpl.KD_STATS + ("below_teacher_frac",):
        assert [f"plc/kd_{name}/loop_{t}" in flat for t in (1, 2, 3, 4)] == [True] * 3 + [False]
    gap = stats["entropy_gap"][..., 0]
    assert flat["plc/kd_below_teacher_frac/loop_1"] == pytest.approx(
        float(((gap < 0).float() * mask).sum() / mask.sum()))


# ---------- settings and config ----------

def test_kd_settings():
    s = plc.PLCSettings.from_mapping({"enabled": True, "loop_loss": "kd", "kl_coef": 0,
                                      "weight_mode": "early_prior", "kd_divergence": "reverse"})
    assert s.loop_loss == "kd" and s.kd_divergence == "reverse"
    assert "loop_loss=kd kd_divergence=reverse" in s.describe()
    default = plc.PLCSettings()
    assert default.loop_loss == "surrogate" and default.kd_divergence == "forward"


@pytest.mark.parametrize("values,match", [
    ({"loop_loss": "kd", "kl_coef": 0}, "early_prior"),
    ({"loop_loss": "kd", "weight_mode": "early_prior"}, "kl_coef=0"),
    ({"loop_loss": "distill"}, "loop_loss"),
    ({"loop_loss": "kd", "kl_coef": 0, "weight_mode": "early_prior", "kd_divergence": "js"},
     "kd_divergence"),
    ({"kd_divergence": "reverse"}, "only applies"),
])
def test_bad_kd_settings_are_refused(values, match):
    with pytest.raises(ValueError, match=match):
        plc.PLCSettings.from_mapping(values)


def test_loopkd_config_is_the_loopaux_config_with_the_kd_loss(env):
    loopaux = OmegaConf.to_container(rcfg.load_config(str(LOOPAUX_CONFIG)))
    cfg = OmegaConf.to_container(rcfg.load_config(str(LOOPKD_CONFIG)))
    assert cfg.pop("plc") == {"enabled": True, "loop_loss": "kd", "kd_divergence": "forward",
                              "weight_mode": "early_prior", "lam": 0.1, "trunc_cap": 1.0,
                              "kl_coef": 0.0, "log_weight_stats": True}
    loopaux.pop("plc")
    for c in (cfg, loopaux):
        for key in ("run_name", "tags"):
            c["wandb"].pop(key)
    assert cfg == loopaux
    s = rcfg.plc_settings(rcfg.load_config(str(LOOPKD_CONFIG), ["plc.kd_divergence=reverse"]))
    assert s.enabled and s.loop_loss == "kd" and s.kd_divergence == "reverse"


def test_loopkd_dry_run(env, capsys, monkeypatch):
    from rltt_baseline import launch

    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "PYTHONPATH"):
        monkeypatch.setenv(key, "")
    monkeypatch.setattr(sys, "path", list(sys.path))
    assert launch.main(["--config", str(LOOPKD_CONFIG), "--dry-run",
                        "plc.kd_divergence=reverse"]) == 0
    resolved = json.loads(capsys.readouterr().out)
    actor = resolved["verl_overrides"]["actor_rollout_ref"]["actor"]
    assert actor["plc"]["loop_loss"] == "kd" and actor["plc"]["kd_divergence"] == "reverse"
    assert actor["plc"]["kl_coef"] == 0.0 and actor["plc"]["weight_mode"] == "early_prior"
    assert resolved["reference_policy"].startswith("none")
    assert resolved["config"]["wandb"]["run_name"] == "plc-grpo-loopkd-fwd-1p4b"


# ---------- audit ----------

def test_audit_records_the_kd_diagnostics():
    torch.manual_seed(4)
    hiddens = [torch.randn(9, H) for _ in range(T)]
    head = torch.nn.Linear(H, V, bias=False)
    labels = torch.arange(9) % V
    out = la.distribution_stats(hiddens, head, labels, temperature=0.7, top_k=3, chunk_tokens=4)
    log_p = torch.stack([F.log_softmax(head(h) / 0.7, -1) for h in hiddens], 1).detach()
    teacher_ids = log_p[:, -1].topk(3, -1).indices
    for t in range(T):
        torch.testing.assert_close(out["teacher_topk_probs"][:, t],
                                   log_p[:, t].exp().gather(-1, teacher_ids))
    for divergence in DIVERGENCES:
        push = out[f"kd_entropy_push_{divergence}"]
        assert push.shape == (9, T) and torch.all(push[:, -1] == 0)
        torch.testing.assert_close(push[:, 1], rpl.kd_entropy_push(log_p[:, 1], log_p[:, -1],
                                                                   divergence))
    arrays = {name: value.numpy() for name, value in out.items()}
    columns = la.token_metrics(arrays)
    entropy = arrays["entropy"]
    np.testing.assert_allclose(columns["entropy_gap/loop_2"], entropy[:, 1] - entropy[:, -1])
    np.testing.assert_array_equal(columns["below_teacher/loop_2"],
                                  (entropy[:, 1] < entropy[:, -1]).astype(np.float32))
    np.testing.assert_allclose(columns["teacher_topk_mass/loop_1"],
                               arrays["teacher_topk_probs"][:, 0].sum(-1))
    assert "entropy_gap/loop_4" not in columns and "kd_entropy_push_reverse/loop_3" in columns


def test_token_metrics_work_without_the_kd_arrays():
    arrays = {
        "entropy": np.array([[1.0, 2.0]]), "sampled_log_prob": np.array([[-1.0, -2.0]]),
        "kl": np.zeros((1, 2, 2)), "topk_ids": np.array([[[1, 2], [1, 3]]]),
        "topk_probs": np.array([[[0.5, 0.25], [0.75, 0.125]]]),
    }
    columns = la.token_metrics(arrays)  # shards written before the KD arrays existed
    assert columns["entropy_gap/loop_1"][0] == -1.0 and "teacher_topk_mass/loop_1" not in columns
