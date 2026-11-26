"""OPD (rltt_baseline.opd): teacher, gradient, loss, settings and config wiring (CPU)."""

import json
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from rltt_baseline import config as rcfg
from rltt_baseline import loop_cd, opd, paths, plc
from rltt_baseline.ouro_rltt.rltt_per_loop import extrap_token_stats, per_loop_token_stats

CONFIG = paths.config_file("opd_loopcd_1p4b_8gpu.yaml")
ENV = {
    "LOOPRL_OURO_DIR": "/models/Ouro-1.4B-Thinking",
    "LOOPRL_MODEL_DIR": "/models/Ouro-1.4B-Thinking-RLTT",
    "LOOPRL_TRAIN_FILE": "/data/deepmath_level6.parquet",
    "LOOPRL_MATH500_JSONL": "/data/math500.jsonl",
    "LOOPRL_OUTPUT_DIR": "/out/opd",
    "LOOPRL_ENV_FILE": "/nonexistent/env.local",  # never read a real env.local in tests
}
V, H, B, S, T = 11, 6, 2, 5, 4
SAMPLED_SECTION = {"enabled": True, "omega": 0.5, "ref_loop": 1, "estimator": "sampled",
                   "divergence": "reverse"}
# filled in by the launcher (pure OPD, frozen teacher)
COEF_DEFAULTS = {"rl_coef": 0.0, "kd_coef": 1.0, "teacher": "frozen"}


@pytest.fixture
def env(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)


def _close(a, b, **kw):
    torch.testing.assert_close(a, b, atol=kw.get("atol", 1e-5), rtol=kw.get("rtol", 1e-5))


def _loops():
    g = torch.Generator().manual_seed(42)
    head = torch.nn.Linear(H, V, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.randn(V, H, generator=g))
    hiddens = [torch.randn(B, S, H, generator=g) for _ in range(T)]
    labels = torch.randint(0, V, (B, S), generator=g)
    return head, hiddens, labels


def _teacher(hiddens, head, labels, settings):
    """What the frozen teacher engine computes: the per-loop forward with teacher_model_kwargs."""
    kw = settings.teacher_model_kwargs()
    assert kw["return_extrap_log_probs"] is True
    with torch.no_grad():
        return extrap_token_stats(hiddens, head, labels, omega=kw["extrap_omega"],
                                  ref_loop=kw["extrap_ref_loop"], adaptive=kw["extrap_adaptive"],
                                  delta_cap=kw["extrap_delta_cap"],
                                  ref_grad=kw["extrap_ref_grad"])["log_probs"]


# ---------- teacher ----------

@pytest.mark.parametrize("omega,ref_loop", [(0.5, 1), (0.3, 1), (1.0, 2)])
def test_teacher_is_distribution_4_plus_w_times_4_minus_1_in_log_space(omega, ref_loop):
    head, hiddens, labels = _loops()
    settings = opd.OPDSettings(enabled=True, omega=omega, ref_loop=ref_loop)
    got = _teacher(hiddens, head, labels, settings)
    log4 = F.log_softmax(head(hiddens[-1]), -1)
    log_r = F.log_softmax(head(hiddens[ref_loop - 1]), -1)
    expected = F.log_softmax(log4 + omega * (log4 - log_r), -1)
    _close(got, expected.gather(-1, labels.unsqueeze(-1)).squeeze(-1))
    # the same distribution LoopCD decodes with (fixed strength, logit form, no clip)
    cd = loop_cd.guided_logprobs(head(hiddens[-1]), head(hiddens[ref_loop - 1]),
                                 loop_cd.LoopCDConfig(ref_loop=ref_loop, omega=omega))
    _close(got, cd.gather(-1, labels.unsqueeze(-1)).squeeze(-1))


def test_omega_zero_teacher_is_the_base_last_loop():
    head, hiddens, labels = _loops()
    got = _teacher(hiddens, head, labels, opd.OPDSettings(enabled=True, omega=0.0))
    log_probs, _ = per_loop_token_stats(hiddens, head, labels)
    _close(got, log_probs[-1])


# ---------- gradient: per-token reverse KL ----------

def test_expected_gradient_is_the_reverse_kl_gradient():
    """sum_y p(y) * grad loss(y) = grad KL(p || t) on the student logits (y ~ p, on-policy)."""
    g = torch.Generator().manual_seed(42)
    z = torch.randn(V, generator=g, requires_grad=True)
    log_t = F.log_softmax(torch.randn(V, generator=g), -1)
    log_p = F.log_softmax(z, -1)
    expected = torch.zeros(V)
    for y in range(V):
        lp = log_p[y].reshape(1, 1)
        loss, _ = opd.opd_policy_loss(lp, lp.detach(), log_t[y].reshape(1, 1),
                                      torch.ones(1, 1), 0.2)
        grad, = torch.autograd.grad(loss, z, retain_graph=True)
        expected += log_p[y].detach().exp() * grad
    kl = (log_p.exp() * (log_p - log_t)).sum()
    kl_grad, = torch.autograd.grad(kl, z)
    _close(expected, kl_grad, atol=1e-6)


def test_teacher_equal_to_student_gives_no_step():
    log_prob = (torch.rand(B, S) * -2).requires_grad_()
    loss, _ = opd.opd_policy_loss(log_prob, log_prob.detach(), log_prob.detach(),
                                  torch.ones(B, S), 0.2)
    grad, = torch.autograd.grad(loss, log_prob)
    _close(grad, torch.zeros(B, S))


# ---------- loss ----------

def _batch():
    torch.manual_seed(42)
    log_prob = (torch.rand(B, S) * -2).requires_grad_()
    old = log_prob.detach() - 0.01
    teacher = old + torch.randn(B, S) * 0.5
    mask = torch.ones(B, S)
    mask[1, -2:] = 0
    return log_prob, old, teacher, mask


@pytest.mark.parametrize("row_div", [None, torch.tensor([3.0, 2.0])])
def test_loss_is_ppo_with_the_token_advantage_teacher_minus_old(row_div):
    log_prob, old, teacher, mask = _batch()
    ours, _ = opd.opd_policy_loss(log_prob, old, teacher, mask, 0.2, row_div=row_div)
    ref, _ = plc.grpo_policy_loss(log_prob, old, teacher - old, mask, 0.2, row_div=row_div)
    _close(ours, ref)
    _close(opd.opd_advantages(teacher, old), teacher - old)


def test_shape_mismatch_is_refused():
    log_prob, old, teacher, mask = _batch()
    with pytest.raises(ValueError, match="do not match"):
        opd.opd_policy_loss(log_prob, old, teacher[:, :-1], mask, 0.2)


def test_stats():
    teacher = torch.tensor([[-1.0, -2.0, -0.5]])
    old = torch.tensor([[-1.5, -1.0, -0.5]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    stats = opd.opd_stats(teacher, old, mask)
    assert stats["opd/reverse_kl_k1"] == pytest.approx(((-1.5 + 1.0) + (-1.0 + 2.0)) / 2)
    r = torch.tensor([0.5, -1.0])
    assert stats["opd/reverse_kl_k3"] == pytest.approx(float((r.exp() - 1 - r).mean()))
    assert stats["opd/teacher_prefers_frac"] == pytest.approx(0.5)
    assert stats["opd/teacher_log_prob"] == pytest.approx(-1.5)
    assert stats["opd/student_log_prob"] == pytest.approx(-1.25)


# ---------- settings ----------

def test_settings_defaults_and_conversions():
    s = opd.OPDSettings.from_mapping({"enabled": True, "omega": 1})
    assert s.omega == 1.0 and s.ref_loop == 1
    assert opd.OPDSettings() == opd.OPDSettings(enabled=False, omega=0.5, ref_loop=1)
    kw = s.teacher_model_kwargs()
    assert kw == {"return_extrap_log_probs": True, "extrap_omega": 1.0, "extrap_ref_loop": 1,
                  "extrap_adaptive": False, "extrap_delta_cap": None, "extrap_ref_grad": False}


@pytest.mark.parametrize("values,error,match", [
    ({"omega": -0.1}, ValueError, "omega"),
    ({"ref_loop": 0}, ValueError, "ref_loop"),
    ({"ref_loop": 1.5}, TypeError, "integer"),
    ({"enabled": 1}, TypeError, "bool"),
    ({"omega": "0.5"}, TypeError, "number"),
    ({"adaptive": True}, KeyError, "unknown opd keys"),
    ({"estimator": "topk"}, ValueError, "estimator"),
    ({"estimator": "full", "divergence": "js"}, ValueError, "divergence"),
    ({"divergence": "forward"}, ValueError, "reverse KL only"),
    ({"estimator": 1}, TypeError, "string"),
    ({"estimator": "k2", "divergence": "forward"}, ValueError, "reverse KL only"),
    ({"rl_coef": -1.0}, ValueError, "rl_coef"),
    ({"kd_coef": float("nan")}, ValueError, "kd_coef"),
    ({"kd_coef": float("inf")}, ValueError, "kd_coef"),
    ({"rl_coef": "1"}, TypeError, "number"),
    ({"rl_coef": True}, TypeError, "number"),
    ({"rl_coef": 0, "kd_coef": 0}, ValueError, "both 0"),
])
def test_settings_validation(values, error, match):
    with pytest.raises(error, match=match):
        opd.OPDSettings.from_mapping(values)


def test_ref_loop_is_checked_against_the_depth():
    with pytest.raises(ValueError, match="< the 4 loops"):
        opd.OPDSettings(ref_loop=4).validate(4)


# ---------- config and launcher ----------

def test_config_is_opd_with_omega_half(env):
    cfg = rcfg.load_config(str(CONFIG))
    assert rcfg.opd_settings(cfg) == opd.OPDSettings(enabled=True, omega=0.5, ref_loop=1)
    assert cfg.trainer == "rltt" and cfg.rollout.kv_layout == "full"
    assert not rcfg.plc_settings(cfg).enabled and not rcfg.extrap_settings(cfg).enabled
    assert rcfg.looprl_method(cfg) == "opd"
    for name, method in (("grpo_baseline_1p4b_8gpu.yaml", "grpo"),
                         ("rltt_baseline_1p4b_8gpu.yaml", "rltt"),
                         ("plc_grpo_loopaux_1p4b_8gpu.yaml", "plc"),
                         ("extrap_grpo_loopcd_1p4b_8gpu.yaml", "extrap"),
                         ("grpo_baseline_8gpu.yaml", "grpo")):
        other = rcfg.load_config(str(paths.config_file(name)))
        assert not rcfg.opd_settings(other).enabled, name
        assert rcfg.looprl_method(other) == method, name


def test_config_sections_are_documented():
    assert "opd" in rcfg.SECTIONS and "opd" in rcfg.DICT_SECTIONS
    assert "``opd``" in rcfg.__doc__


def _args(**kw):
    values = {"use_rltt_model": True, "loop_weighting": "uniform", "total_ut_steps": 4, **kw}
    return SimpleNamespace(**values)


def test_overrides_pass_the_section_and_leave_the_rollout_alone(env):
    cfg = rcfg.load_config(str(CONFIG))
    base = rcfg.verl_overrides(cfg)
    overrides = rcfg.add_opd_overrides(cfg, base, _args())
    ar = overrides["actor_rollout_ref"]
    assert ar["actor"]["opd"] == {**SAMPLED_SECTION, **COEF_DEFAULTS}
    # the rollout is the ordinary last loop: same rollout and model settings as without OPD
    assert ar["rollout"] == base["actor_rollout_ref"]["rollout"]
    assert "external_lib" not in ar["model"]
    assert rcfg.reference_policy_status("rltt", overrides).startswith("opd teacher")
    off = rcfg.load_config(str(CONFIG), ["opd.enabled=false"])
    assert rcfg.add_opd_overrides(off, rcfg.verl_overrides(off), _args()) == \
        rcfg.verl_overrides(off)


@pytest.mark.parametrize("cli,args,match", [
    (["plc.enabled=true"], {}, "different methods"),
    (["extrap.enabled=true"], {}, "different methods"),
    (["trainer=grpo"], {}, "trainer=rltt"),
    (["opd.ref_loop=4"], {}, "< the 4 loops"),
    ([], {"use_rltt_model": False}, "use_rltt_model"),
    ([], {"loop_weighting": "learned"}, "learned"),
])
def test_conflicting_settings_are_refused(env, cli, args, match):
    cfg = rcfg.load_config(str(CONFIG), cli)
    with pytest.raises(ValueError, match=match):
        rcfg.add_opd_overrides(cfg, rcfg.verl_overrides(cfg), _args(**args))


def _clean_launch_env(monkeypatch):
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "PYTHONPATH"):
        monkeypatch.setenv(key, "")
    monkeypatch.setattr(sys, "path", list(sys.path))


def test_dry_run(env, capsys, monkeypatch):
    from rltt_baseline import launch

    _clean_launch_env(monkeypatch)
    assert launch.main(["--config", str(CONFIG), "--dry-run"]) == 0
    resolved = json.loads(capsys.readouterr().out)
    ar = resolved["verl_overrides"]["actor_rollout_ref"]
    assert ar["actor"]["opd"] == {**SAMPLED_SECTION, **COEF_DEFAULTS}
    assert ar["loss_agg"] == {"mode": "drgrpo", "norm_len": 16384}
    assert "plc" not in ar["actor"] and "extrap" not in ar["actor"]
    assert "engine_kwargs" not in ar["rollout"]  # no LoopCD rollout
    assert resolved["config"]["opd"]["enabled"] is True
    assert resolved["reference_policy"].startswith("opd teacher")


def test_verl_061_training_launch_is_refused(env, monkeypatch):
    from rltt_baseline import launch

    _clean_launch_env(monkeypatch)
    monkeypatch.setattr(launch, "policy_violations", lambda *a, **k: [])
    monkeypatch.setattr(launch, "wandb_key_problem", lambda *a, **k: None)
    monkeypatch.setattr(launch, "runtime_status",
                        lambda: {"installed": {"verl": "0.6.1"}, "target_versions_match": False})
    with pytest.raises(SystemExit, match="verl 0.9.1"):
        launch.main(["--config", str(CONFIG)])


# ---------- verl 0.9 objective ----------

def test_native_objective_matches_the_loss_and_ignores_the_reward():
    from rltt_baseline.verl09_losses import objective

    log_prob, old, teacher, mask = _batch()
    ell = torch.cat([(torch.rand(B, S, T - 1) * -2), log_prob.detach().unsqueeze(-1)], -1)
    ell.requires_grad_()
    output = dict(log_probs=ell[..., -1], entropy=torch.zeros(B, S), per_loop_log_probs=ell,
                  per_loop_entropies=torch.zeros(B, S, T))
    batch = dict(old_log_probs=old, response_mask=mask, teacher_log_probs=teacher,
                 advantages=torch.tensor([[1.0] * S, [-0.5] * S]))
    settings = dict(method="opd", opd=dict(enabled=True, omega=0.5),
                    loss_agg=dict(mode="drgrpo", norm_len=16))
    actual, metrics = objective(dict(clip_ratio=0.2), settings, output, batch)
    from rltt_baseline import loss_agg
    row_div = loss_agg.row_divisors(loss_agg.LossAggSettings(mode="drgrpo", norm_len=16), mask)
    expected, _ = opd.opd_policy_loss(ell[..., -1], old, teacher, mask, 0.2, row_div=row_div)
    _close(actual, expected)
    _close(torch.autograd.grad(actual, ell, retain_graph=True)[0],
           torch.autograd.grad(expected, ell)[0])
    assert metrics["opd/reverse_kl_k1"] == pytest.approx(
        float(((old - teacher) * mask).sum() / mask.sum()))
    # the GRPO advantages of the batch do not enter the loss
    batch["advantages"] = -batch["advantages"]
    again, _ = objective(dict(clip_ratio=0.2), settings, output, batch)
    _close(again, actual)


def test_native_objective_requires_the_teacher():
    from rltt_baseline.verl09_losses import objective

    log_prob, old, _, mask = _batch()
    ell = log_prob.unsqueeze(-1).expand(B, S, T)
    output = dict(log_probs=log_prob, entropy=torch.zeros(B, S), per_loop_log_probs=ell,
                  per_loop_entropies=torch.zeros(B, S, T))
    batch = dict(old_log_probs=old, response_mask=mask, advantages=torch.zeros(B, S))
    with pytest.raises(ValueError, match="teacher_log_probs"):
        objective({}, dict(method="opd", opd=dict(enabled=True)), output, batch)


def test_config_differs_from_extrap_only_in_the_method(env):
    ext = OmegaConf.to_container(rcfg.load_config(str(paths.config_file(
        "extrap_grpo_loopcd_1p4b_8gpu.yaml"))))
    cfg = OmegaConf.to_container(rcfg.load_config(str(CONFIG)))
    assert cfg.pop("opd") == SAMPLED_SECTION
    assert ext.pop("extrap")["enabled"] is True
    assert cfg.pop("extrap") == {} and ext.pop("opd") == {}  # load_config fills empty sections
    assert cfg.pop("loss_agg") == {"mode": "drgrpo", "norm_len": 16384}
    assert ext.pop("loss_agg") == {"mode": "seq-mean-token-mean"}
    for key in ("run_name", "tags"):
        cfg["wandb"].pop(key)
        ext["wandb"].pop(key)
    assert cfg == ext


# ---------- full-vocabulary estimator ----------

FULL_CONFIGS = {"reverse": paths.config_file("opd_full_rkl_loopcd_1p4b_8gpu.yaml"),
                "forward": paths.config_file("opd_full_fkl_loopcd_1p4b_8gpu.yaml")}
ELPO_CONFIG = paths.config_file("elpo_online_rkl_loopcd_1p4b_8gpu.yaml")


def _student_teacher(requires_grad=True):
    g = torch.Generator().manual_seed(42)
    s_head = torch.nn.Linear(H, V, bias=False)
    t_head = torch.nn.Linear(H, V, bias=False)
    with torch.no_grad():
        s_head.weight.copy_(torch.randn(V, H, generator=g))
        t_head.weight.copy_(torch.randn(V, H, generator=g))
    student = torch.randn(B, S, H, generator=g, requires_grad=requires_grad)
    t_ref, t_last = (torch.randn(B, S, H, generator=g) for _ in range(2))
    return s_head, t_head, student, t_ref, t_last


def _manual_kl(s_head, t_head, student, t_ref, t_last, omega, divergence, temperature=1.0):
    from rltt_baseline.ouro_rltt.rltt_per_loop import kd_divergence

    log4 = F.log_softmax(t_head(t_last).float(), -1)
    log1 = F.log_softmax(t_head(t_ref).float(), -1)
    log_q = F.log_softmax((log4 + omega * (log4 - log1)) / temperature, -1).detach()
    log_p = F.log_softmax(s_head(student).float() / temperature, -1)
    return kd_divergence(log_p, log_q, divergence), log_p, log_q


def test_full_teacher_is_loop_cd_over_the_whole_vocabulary():
    from rltt_baseline.ouro_rltt.rltt_per_loop import opd_teacher_log_probs

    z1, z4 = torch.randn(B, S, V), torch.randn(B, S, V)
    got = opd_teacher_log_probs(z1, z4, 0.5)
    _close(got, loop_cd.guided_logprobs(z4, z1, loop_cd.LoopCDConfig(omega=0.5)))
    _close(got.exp().sum(-1), torch.ones(B, S))


@pytest.mark.parametrize("divergence", ["reverse", "forward"])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_full_kd_values_and_stats(divergence, temperature):
    from rltt_baseline.ouro_rltt.rltt_per_loop import kd_divergence, opd_full_kd

    s_head, t_head, student, t_ref, t_last = _student_teacher(requires_grad=False)
    out = opd_full_kd(student, t_ref, t_last, s_head, t_head, 0.5, divergence,
                      temperature=temperature, chunk_size=2)
    kd, log_p, log_q = _manual_kl(s_head, t_head, student, t_ref, t_last, 0.5, divergence,
                                  temperature)
    other = "forward" if divergence == "reverse" else "reverse"
    _close(out["kd"], kd)
    _close(out["kl_other"], kd_divergence(log_p, log_q, other))
    _close(out["teacher_entropy"], -(log_q.exp() * log_q).sum(-1))
    _close(out["top1_agree"], (log_p.argmax(-1) == log_q.argmax(-1)).float())
    assert torch.all(out["kd"] >= -1e-6)


@pytest.mark.parametrize("divergence", ["reverse", "forward"])
def test_full_kd_gradient_reaches_the_student_only(divergence):
    from rltt_baseline.ouro_rltt.rltt_per_loop import opd_full_kd

    s_head, t_head, student, t_ref, t_last = _student_teacher()
    t_ref.requires_grad_()
    results = []
    for chunk in (None, 2):  # chunking and checkpointing change neither values nor gradients
        out = opd_full_kd(student, t_ref, t_last, s_head, t_head, 0.5, divergence,
                          chunk_size=chunk)
        grads = torch.autograd.grad(out["kd"].sum(), [student, s_head.weight, t_ref,
                                                      t_head.weight], allow_unused=True)
        results.append((out["kd"].detach(), grads))
    kd, _, _ = _manual_kl(s_head, t_head, student, t_ref, t_last, 0.5, divergence)
    ref_grads = torch.autograd.grad(kd.sum(), [student, s_head.weight])
    for value, grads in results:
        _close(value, kd.detach())
        _close(grads[0], ref_grads[0])
        _close(grads[1], ref_grads[1])
        assert grads[2] is None and grads[3] is None  # the teacher gets no gradient


@pytest.mark.parametrize("divergence", ["reverse", "forward"])
def test_full_kd_logit_step(divergence):
    """Minus the gradient on the student logits: forward q - p, reverse -p (log p - log q - KL)."""
    from rltt_baseline.ouro_rltt.rltt_per_loop import kd_logit_step, opd_full_kd

    z = torch.randn(1, 1, V, requires_grad=True)
    z1, z4 = torch.randn(1, 1, V), torch.randn(1, 1, V)
    out = opd_full_kd(z, z1, z4, torch.nn.Identity(), torch.nn.Identity(), 0.5, divergence)
    grad, = torch.autograd.grad(out["kd"].sum(), z)
    log_q = F.log_softmax(1.5 * F.log_softmax(z4, -1) - 0.5 * F.log_softmax(z1, -1), -1)
    _close(-grad, kd_logit_step(F.log_softmax(z, -1).detach(), log_q, divergence))


def test_full_kd_refuses_mismatched_positions():
    from rltt_baseline.ouro_rltt.rltt_per_loop import opd_full_kd

    s_head, t_head, student, t_ref, t_last = _student_teacher(requires_grad=False)
    with pytest.raises(ValueError, match="same positions"):
        opd_full_kd(student, t_ref[:, :-1], t_last, s_head, t_head, 0.5)


def test_full_settings_and_student_kwargs():
    s = opd.OPDSettings.from_mapping({"enabled": True, "estimator": "full",
                                      "divergence": "forward"})
    assert s.full_vocab and s.omega == 0.5
    assert s.student_model_kwargs() == {"return_opd_kd": True, "opd_omega": 0.5,
                                        "opd_divergence": "forward"}
    assert not opd.OPDSettings().full_vocab


@pytest.mark.parametrize("divergence", ["reverse", "forward"])
def test_full_configs(env, divergence):
    cfg = rcfg.load_config(str(FULL_CONFIGS[divergence]))
    assert rcfg.opd_settings(cfg) == opd.OPDSettings(enabled=True, omega=0.5, ref_loop=1,
                                                     estimator="full", divergence=divergence)
    assert rcfg.looprl_method(cfg) == "opd" and cfg.trainer == "rltt"
    # LoopOPD's normalisation (user, 2026-11-23): per-response token mean, then mean over responses
    assert rcfg.loss_agg_settings(cfg).mode == "seq-mean-token-mean"
    assert "norm_len" not in cfg.loss_agg
    # the user's settings (2026-11-22): 100 steps, 512 rollouts per step, top-p 1, temperature 1
    assert cfg.args.max_steps == 100
    assert cfg.args.num_prompts_per_batch * cfg.args.num_generations == 512
    assert cfg.args.temperature == 1.0
    assert cfg.verl_overrides.actor_rollout_ref.rollout.top_p == 1.0
    assert cfg.args.train_file == ENV["LOOPRL_TRAIN_FILE"]
    overrides = rcfg.add_opd_overrides(cfg, rcfg.verl_overrides(cfg), _args())
    assert overrides["actor_rollout_ref"]["actor"]["opd"]["estimator"] == "full"
    assert rcfg.reference_policy_status("rltt", overrides).startswith(
        "opd teacher (full vocabulary)")


def test_full_configs_differ_only_in_the_divergence(env):
    rkl = OmegaConf.to_container(rcfg.load_config(str(FULL_CONFIGS["reverse"])))
    fkl = OmegaConf.to_container(rcfg.load_config(str(FULL_CONFIGS["forward"])))
    assert rkl["opd"].pop("divergence") == "reverse" and fkl["opd"].pop("divergence") == "forward"
    for cfg in (rkl, fkl):
        cfg["wandb"].pop("run_name")
        cfg["wandb"].pop("tags")
    assert rkl == fkl
    sampled = OmegaConf.to_container(rcfg.load_config(str(CONFIG)))
    assert sampled["opd"].pop("estimator") == "sampled" and rkl["opd"].pop("estimator") == "full"
    sampled["opd"].pop("divergence")
    # the full-vocabulary configs follow LoopOPD's per-response normalisation (user, 2026-11-23)
    assert sampled.pop("loss_agg") == {"mode": "drgrpo", "norm_len": 16384}
    assert rkl.pop("loss_agg") == {"mode": "seq-mean-token-mean"}
    for cfg in (sampled,):
        cfg["wandb"].pop("run_name")
        cfg["wandb"].pop("tags")
    assert sampled == rkl


def test_full_dry_run(env, capsys, monkeypatch):
    from rltt_baseline import launch

    _clean_launch_env(monkeypatch)
    assert launch.main(["--config", str(FULL_CONFIGS["forward"]), "--dry-run"]) == 0
    resolved = json.loads(capsys.readouterr().out)
    ar = resolved["verl_overrides"]["actor_rollout_ref"]
    assert ar["actor"]["opd"] == {"enabled": True, "omega": 0.5, "ref_loop": 1,
                                  "estimator": "full", "divergence": "forward", **COEF_DEFAULTS}
    assert ar["loss_agg"] == {"mode": "seq-mean-token-mean", "norm_len": None}
    assert resolved["reference_policy"].startswith("opd teacher (full vocabulary)")
    assert resolved["policy_violations"] is not None


@pytest.mark.parametrize("divergence", ["reverse", "forward"])
def test_native_objective_full(divergence):
    from rltt_baseline import loss_agg
    from rltt_baseline.verl09_losses import objective

    _, old, _, mask = _batch()
    kd = (torch.rand(B, S) + 0.1).requires_grad_()
    other = torch.rand(B, S)
    ell = torch.rand(B, S, T) * -2
    output = dict(log_probs=ell[..., -1], entropy=torch.zeros(B, S), per_loop_log_probs=ell,
                  per_loop_entropies=torch.zeros(B, S, T), opd_kd=kd, opd_kl_other=other,
                  opd_teacher_entropy=torch.ones(B, S), opd_top1_agree=torch.ones(B, S))
    batch = dict(old_log_probs=old, response_mask=mask, advantages=torch.ones(B, S))
    settings = dict(method="opd", opd=dict(enabled=True, estimator="full", divergence=divergence),
                    loss_agg=dict(mode="drgrpo", norm_len=16))
    actual, metrics = objective(dict(clip_ratio=0.2), settings, output, batch)
    row_div = loss_agg.row_divisors(loss_agg.LossAggSettings(mode="drgrpo", norm_len=16), mask)
    _close(actual, plc.agg_term(kd, mask, "token-mean", row_div))
    _close(torch.autograd.grad(actual, kd)[0], mask / (B * 16))
    mean = lambda x: float((x * mask).sum() / mask.sum())  # noqa: E731
    used, unused = ("opd/reverse_kl", "opd/forward_kl")
    if divergence == "forward":
        used, unused = unused, used
    assert metrics[used] == pytest.approx(mean(kd.detach()))
    assert metrics[unused] == pytest.approx(mean(other))
    assert metrics["opd/top1_agree"] == pytest.approx(1.0)
    del output["opd_kd"]
    with pytest.raises(ValueError, match="opd_kd"):
        objective({}, settings, output, batch)


# ---------- k2 estimator and the GRPO term (KDRL, arXiv 2506.02208) ----------

K2_SECTION = dict(enabled=True, omega=0.5, estimator="k2", rl_coef=1.0, kd_coef=0.01)


def test_k2_value_and_per_sample_gradient():
    log_prob, _, teacher, mask = _batch()
    loss, metrics = opd.opd_k2_loss(log_prob, teacher, mask)
    r = teacher - log_prob.detach()
    _close(loss, (0.5 * r.square() * mask).sum() / mask.sum())
    # d(R^2 / 2) / d log p = -R: the REINFORCE step with the per-token reward R
    _close(torch.autograd.grad(loss, log_prob)[0], -r * mask / mask.sum())
    assert metrics == {"loss_term/kd": pytest.approx(loss.item()), "loss_term/pg": 0.0}


def test_k2_expected_gradient_is_the_reverse_kl_gradient():
    """sum_y p(y) * grad k2(y) = grad KL(p || t) on the student logits (y ~ p, on-policy)."""
    g = torch.Generator().manual_seed(42)
    z = torch.randn(V, generator=g, requires_grad=True)
    log_t = F.log_softmax(torch.randn(V, generator=g), -1)
    log_p = F.log_softmax(z, -1)
    expected = torch.zeros(V)
    for y in range(V):
        loss, _ = opd.opd_k2_loss(log_p[y].reshape(1, 1), log_t[y].reshape(1, 1),
                                  torch.ones(1, 1))
        grad, = torch.autograd.grad(loss, z, retain_graph=True)
        expected += log_p[y].detach().exp() * grad
    kl = (log_p.exp() * (log_p - log_t)).sum()
    kl_grad, = torch.autograd.grad(kl, z)
    _close(expected, kl_grad, atol=1e-6)


def test_k2_and_sampled_take_the_same_on_policy_step():
    log_prob, _, teacher, mask = _batch()
    k2, _ = opd.opd_k2_loss(log_prob, teacher, mask)
    ppo, _ = opd.opd_policy_loss(log_prob, log_prob.detach(), teacher, mask, 0.2)
    _close(torch.autograd.grad(k2, log_prob)[0], torch.autograd.grad(ppo, log_prob)[0])


def test_k2_teacher_equal_to_student_gives_no_step_and_shapes_are_checked():
    log_prob = (torch.rand(B, S) * -2).requires_grad_()
    loss, _ = opd.opd_k2_loss(log_prob, log_prob.detach(), torch.ones(B, S))
    _close(torch.autograd.grad(loss, log_prob)[0], torch.zeros(B, S))
    with pytest.raises(ValueError, match="do not match"):
        opd.opd_k2_loss(log_prob, log_prob.detach()[:, :-1], torch.ones(B, S))


def test_k2_settings():
    s = opd.OPDSettings.from_mapping(K2_SECTION)
    assert s.sampled_teacher and not s.full_vocab and not s.pure_distillation
    assert (s.rl_coef, s.kd_coef) == (1.0, 0.01)
    assert s.teacher_model_kwargs() == opd.OPDSettings().teacher_model_kwargs()
    assert opd.OPDSettings().pure_distillation and opd.OPDSettings().sampled_teacher
    assert not opd.OPDSettings(estimator="full").sampled_teacher
    assert "rl_coef=1.0 kd_coef=0.01" in s.describe()


@pytest.mark.parametrize("estimator", ["sampled", "k2", "full"])
@pytest.mark.parametrize("rl_coef,kd_coef", [(1.0, 0.01), (0.0, 0.5), (2.0, 1.0), (1.0, 0.0)])
def test_joint_loss_is_rl_coef_grpo_plus_kd_coef_divergence(estimator, rl_coef, kd_coef):
    log_prob, old, teacher, mask = _batch()
    adv = torch.tensor([[1.0] * S, [-0.5] * S])
    kd = torch.rand(B, S) + 0.1
    row_div = torch.tensor([3.0, 2.0])
    s = opd.OPDSettings(enabled=True, estimator=estimator, rl_coef=rl_coef, kd_coef=kd_coef)
    loss, metrics = opd.opd_loss(s, log_prob, old, mask, 0.2, advantages=adv,
                                 teacher_log_prob=teacher, kd=kd, row_div=row_div)
    if estimator == "sampled":
        div, _ = opd.opd_policy_loss(log_prob, old, teacher, mask, 0.2, row_div=row_div)
    elif estimator == "k2":
        div, _ = opd.opd_k2_loss(log_prob, teacher, mask, row_div=row_div)
    else:
        div, _ = opd.opd_full_loss(kd, mask, row_div=row_div)
    grpo, _ = plc.grpo_policy_loss(log_prob, old, adv, mask, 0.2, row_div=row_div)
    _close(loss, rl_coef * grpo + kd_coef * div)
    assert metrics["loss_term/pg"] == pytest.approx(rl_coef * grpo.item())
    assert metrics["loss_term/kd"] == pytest.approx(kd_coef * div.item())
    assert ("actor/pg_clipfrac" in metrics) == (rl_coef > 0)


@pytest.mark.parametrize("estimator", ["sampled", "k2"])
def test_default_coefficients_leave_the_estimator_alone(estimator):
    log_prob, old, teacher, mask = _batch()
    s = opd.OPDSettings(enabled=True, estimator=estimator)
    loss, metrics = opd.opd_loss(s, log_prob, old, mask, 0.2, teacher_log_prob=teacher)
    if estimator == "sampled":
        ref, ref_metrics = opd.opd_policy_loss(log_prob, old, teacher, mask, 0.2)
    else:
        ref, ref_metrics = opd.opd_k2_loss(log_prob, teacher, mask)
    assert torch.equal(loss, ref) and metrics == ref_metrics


def test_joint_loss_needs_its_inputs():
    log_prob, old, teacher, mask = _batch()
    s = opd.OPDSettings(enabled=True, estimator="k2", rl_coef=1.0, kd_coef=0.01)
    with pytest.raises(ValueError, match="advantages"):
        opd.opd_loss(s, log_prob, old, mask, 0.2, teacher_log_prob=teacher)
    with pytest.raises(ValueError, match="teacher_log_probs"):
        opd.opd_loss(s, log_prob, old, mask, 0.2, advantages=torch.ones(B, S))
    with pytest.raises(ValueError, match="opd_kd"):
        opd.opd_loss(opd.OPDSettings(enabled=True, estimator="full"), log_prob, old, mask, 0.2)


def test_native_objective_k2_uses_the_reward():
    from rltt_baseline import loss_agg
    from rltt_baseline.verl09_losses import objective

    log_prob, old, teacher, mask = _batch()
    ell = torch.cat([(torch.rand(B, S, T - 1) * -2), log_prob.detach().unsqueeze(-1)], -1)
    ell.requires_grad_()
    output = dict(log_probs=ell[..., -1], entropy=torch.zeros(B, S), per_loop_log_probs=ell,
                  per_loop_entropies=torch.zeros(B, S, T))
    adv = torch.tensor([[1.0] * S, [-0.5] * S])
    batch = dict(old_log_probs=old, response_mask=mask, teacher_log_probs=teacher, advantages=adv)
    settings = dict(method="opd", opd=K2_SECTION, loss_agg=dict(mode="seq-mean-token-mean"))
    actual, metrics = objective(dict(clip_ratio=0.2), settings, output, batch)
    row_div = loss_agg.row_divisors(loss_agg.LossAggSettings(mode="seq-mean-token-mean"), mask)
    grpo, _ = plc.grpo_policy_loss(ell[..., -1], old, adv, mask, 0.2, row_div=row_div)
    k2, _ = opd.opd_k2_loss(ell[..., -1], teacher, mask, row_div=row_div)
    expected = grpo + 0.01 * k2
    _close(actual, expected)
    _close(torch.autograd.grad(actual, ell, retain_graph=True)[0],
           torch.autograd.grad(expected, ell)[0])
    assert metrics["loss_term/kd"] == pytest.approx(0.01 * k2.item())
    assert metrics["loss_term/pg"] == pytest.approx(grpo.item())
    assert "actor/pg_clipfrac" in metrics
    assert metrics["opd/reverse_kl_k1"] == pytest.approx(
        float(((old - teacher) * mask).sum() / mask.sum()))
    # the reward enters the objective: flipping the advantages changes the loss
    batch["advantages"] = -adv
    again, _ = objective(dict(clip_ratio=0.2), settings, output, batch)
    assert not torch.allclose(again, actual)
    del batch["teacher_log_probs"]
    with pytest.raises(ValueError, match="teacher_log_probs"):
        objective({}, settings, output, batch)


def test_k2_dry_run(env, capsys, monkeypatch):
    from rltt_baseline import launch

    _clean_launch_env(monkeypatch)
    assert launch.main(["--config", str(CONFIG), "--dry-run", "opd.estimator=k2",
                        "opd.rl_coef=1", "opd.kd_coef=0.002"]) == 0
    resolved = json.loads(capsys.readouterr().out)
    ar = resolved["verl_overrides"]["actor_rollout_ref"]
    assert ar["actor"]["opd"] == {**SAMPLED_SECTION, "estimator": "k2", "rl_coef": 1.0,
                                  "kd_coef": 0.002, "teacher": "frozen"}
    assert resolved["reference_policy"].startswith("opd teacher: ")


# ---------- GRPO + k2 under seq-mean-token-mean: both terms are per-response means ----------

SEQ_MEAN = dict(mode="seq-mean-token-mean")


def _ragged_batch(lengths=(5, 3, 1, 4, 2, 5)):
    """Responses of different lengths: the per-response mean differs from the token mean."""
    g = torch.Generator().manual_seed(42)
    rows, width = len(lengths), max(lengths)
    mask = (torch.arange(width)[None] < torch.tensor(lengths)[:, None]).float()
    ell = (torch.rand(rows, width, T, generator=g) * -2).requires_grad_()
    old = ell.detach()[..., -1] - 0.05 * torch.rand(rows, width, generator=g)
    teacher = old + torch.randn(rows, width, generator=g) * 0.5
    adv = torch.randn(rows, generator=g)[:, None].expand(rows, width).contiguous()
    output = dict(log_probs=ell[..., -1], entropy=torch.zeros(rows, width),
                  per_loop_log_probs=ell, per_loop_entropies=torch.zeros(rows, width, T))
    batch = dict(old_log_probs=old, response_mask=mask, teacher_log_probs=teacher,
                 advantages=adv)
    return ell, output, batch


def _rows(output, batch, idx):
    pick = lambda d: {k: v[idx] for k, v in d.items()}  # noqa: E731
    return pick(output), pick(batch)


def test_joint_k2_seq_mean_is_the_per_response_mean_of_both_terms():
    """loss = (1/N) sum_i (1/L_i) sum_t [ -rl_coef PPO_it(A_i) + kd_coef (R_it^2 / 2) ]."""
    from rltt_baseline.verl09_losses import objective

    ell, output, batch = _ragged_batch()
    settings = dict(method="opd", opd=K2_SECTION, loss_agg=SEQ_MEAN)
    actual, _ = objective(dict(clip_ratio=0.2), settings, output, batch)
    logp, old, mask = ell[..., -1], batch["old_log_probs"], batch["response_mask"]
    ppo, _ = plc.ppo_objective(logp, old, batch["advantages"], 0.2)
    k2 = 0.5 * (batch["teacher_log_probs"] - logp).square()
    per_token = -K2_SECTION["rl_coef"] * ppo + K2_SECTION["kd_coef"] * k2
    lengths = mask.sum(-1)
    expected = sum((per_token[i] * mask[i]).sum() / lengths[i] for i in range(len(lengths)))
    expected = expected / len(lengths)
    _close(actual, expected)
    # every response weighs 1/N in total, whatever its length, in both terms
    grad, = torch.autograd.grad(actual, ell)
    per_token_grad = grad[..., -1]
    assert torch.equal(per_token_grad[mask == 0], torch.zeros_like(per_token_grad[mask == 0]))


def test_joint_kd_coef_zero_is_the_grpo_objective():
    """With kd_coef = 0 the joint loss is exactly the GRPO objective (same aggregation)."""
    from rltt_baseline.verl09_losses import objective

    _, output, batch = _ragged_batch()
    joint = dict(method="opd", opd={**K2_SECTION, "kd_coef": 0.0}, loss_agg=SEQ_MEAN)
    grpo = dict(method="grpo", loss_agg=SEQ_MEAN)
    a, _ = objective(dict(clip_ratio=0.2), joint, output, batch)
    b, _ = objective(dict(clip_ratio=0.2), grpo, output, batch)
    assert torch.equal(a, b)


def test_joint_k2_seq_mean_does_not_depend_on_the_micro_batch_split():
    """Micro-batches scaled by B_micro / B (as looprl_loss does) give the whole-batch loss and
    gradient: each micro-batch takes both terms' divisors L_i from its own rows."""
    from rltt_baseline.verl09_losses import objective

    ell, output, batch = _ragged_batch()
    settings = dict(method="opd", opd=K2_SECTION, loss_agg=SEQ_MEAN)
    whole, _ = objective(dict(clip_ratio=0.2), settings, output, batch)
    whole_grad, = torch.autograd.grad(whole, ell)
    rows = batch["response_mask"].shape[0]
    for split in ([0, 1, 2, 3, 4, 5], [[4, 0], [2, 5, 1], [3]], [[1], [0, 2, 3, 4, 5]]):
        groups = [[i] for i in split] if isinstance(split[0], int) else split
        total = 0.0
        for idx in groups:
            out, bat = _rows(output, batch, torch.tensor(idx))
            micro, _ = objective(dict(clip_ratio=0.2), settings, out, bat)
            total = total + micro * len(idx) / rows
        _close(total, whole)
        _close(torch.autograd.grad(total, ell)[0], whole_grad)


# ---------- the GRPO + k2 config (user, 2026-11-23) ----------

GRPO_K2_CONFIG = paths.config_file("grpo_k2_loopcd_1p4b_8gpu.yaml")
GRPO_K2_SECTION = {**SAMPLED_SECTION, "estimator": "k2", "rl_coef": 1.0, "kd_coef": 0.01}


def test_grpo_k2_config(env):
    cfg = rcfg.load_config(str(GRPO_K2_CONFIG))
    assert rcfg.opd_settings(cfg) == opd.OPDSettings(**GRPO_K2_SECTION)
    assert rcfg.looprl_method(cfg) == "opd" and cfg.trainer == "rltt"
    assert rcfg.loss_agg_settings(cfg).mode == "seq-mean-token-mean"
    assert cfg.args.max_steps == 100
    assert cfg.args.num_prompts_per_batch * cfg.args.num_generations == 512
    assert cfg.args.seed == 42 and cfg.args.train_file == ENV["LOOPRL_TRAIN_FILE"]
    overrides = rcfg.add_opd_overrides(cfg, rcfg.verl_overrides(cfg), _args())
    assert overrides["actor_rollout_ref"]["actor"]["opd"] == {**GRPO_K2_SECTION,
                                                             "teacher": "frozen"}
    assert rcfg.reference_policy_status("rltt", overrides).startswith("opd teacher: ")


def test_grpo_k2_config_differs_from_opd_only_in_the_method(env):
    """Same as opd_loopcd_1p4b_8gpu.yaml except the opd estimator / coefficients, loss_agg and the
    wandb naming."""
    k2 = OmegaConf.to_container(rcfg.load_config(str(GRPO_K2_CONFIG)))
    base = OmegaConf.to_container(rcfg.load_config(str(CONFIG)))
    assert k2.pop("opd") == GRPO_K2_SECTION and base.pop("opd") == SAMPLED_SECTION
    assert k2.pop("loss_agg") == {"mode": "seq-mean-token-mean"}
    assert base.pop("loss_agg") == {"mode": "drgrpo", "norm_len": 16384}
    for cfg in (k2, base):
        cfg["wandb"].pop("run_name")
        cfg["wandb"].pop("tags")
    assert k2 == base


def test_grpo_k2_dry_run(env, capsys, monkeypatch):
    from rltt_baseline import launch

    _clean_launch_env(monkeypatch)
    assert launch.main(["--config", str(GRPO_K2_CONFIG), "--dry-run"]) == 0
    resolved = json.loads(capsys.readouterr().out)
    ar = resolved["verl_overrides"]["actor_rollout_ref"]
    assert ar["actor"]["opd"] == {**GRPO_K2_SECTION, "teacher": "frozen"}
    assert ar["loss_agg"] == {"mode": "seq-mean-token-mean", "norm_len": None}
    assert "engine_kwargs" not in ar["rollout"]  # ordinary last-loop rollout
    assert resolved["reference_policy"].startswith("opd teacher: ")
    assert resolved["config"]["wandb"]["run_name"] == "grpo-k2-loopcd-w0.5-beta0.01-1p4b"


# ---------- ELPO: the online teacher (the current model's LoopCD policy, stop-gradient) ----------

def test_online_teacher_settings():
    s = opd.OPDSettings.from_mapping({"enabled": True, "estimator": "full", "teacher": "online"})
    assert s.online_teacher and s.full_vocab and not s.frozen_teacher_model
    assert s.student_model_kwargs() == {"return_opd_kd": True, "opd_omega": 0.5,
                                        "opd_divergence": "reverse", "opd_online_teacher": True,
                                        "opd_ref_loop": 1}
    assert "teacher=online" in s.describe()
    frozen = opd.OPDSettings.from_mapping({"enabled": True, "estimator": "full"})
    assert frozen.teacher == "frozen" and frozen.frozen_teacher_model
    assert "opd_online_teacher" not in frozen.student_model_kwargs()


@pytest.mark.parametrize("estimator", ["sampled", "k2"])
def test_online_teacher_sampled_estimators_use_the_actor_pass(estimator):
    s = opd.OPDSettings.from_mapping({"enabled": True, "estimator": estimator,
                                      "teacher": "online"})
    assert s.actor_teacher_log_probs and s.sampled_teacher and not s.frozen_teacher_model
    # the actor's compute_log_prob pass gets the frozen engine's LoopCD switches
    assert s.teacher_model_kwargs()["return_extrap_log_probs"] is True
    assert not opd.OPDSettings.from_mapping({"enabled": True,
                                             "estimator": estimator}).actor_teacher_log_probs
    assert not opd.OPDSettings.from_mapping({"enabled": True, "estimator": "full",
                                             "teacher": "online"}).actor_teacher_log_probs


@pytest.mark.parametrize("values, match", [
    ({"estimator": "full", "teacher": "online", "omega": 0}, "omega=0"),
    ({"estimator": "k2", "teacher": "online", "omega": 0, "rl_coef": 1.0}, "omega=0"),
    ({"estimator": "full", "teacher": "current"}, "opd.teacher must be one of"),
])
def test_online_teacher_refusals(values, match):
    with pytest.raises(ValueError, match=match):
        opd.OPDSettings.from_mapping({"enabled": True, **values})


@pytest.mark.parametrize("divergence", ["reverse", "forward"])
def test_online_teacher_kd_gradient_skips_the_teacher(divergence):
    """With the teacher read from the student's own loops (same lm_head, same hidden tensors) the
    gradient is that of KL against the detached LoopCD distribution: nothing flows through the
    teacher's loop-1 / loop-4 hidden states or its lm_head call."""
    from rltt_baseline.ouro_rltt.rltt_per_loop import opd_full_kd

    s_head, _, _, _, _ = _student_teacher(requires_grad=False)
    g = torch.Generator().manual_seed(7)
    h1 = torch.randn(B, S, H, generator=g, requires_grad=True)
    h4 = torch.randn(B, S, H, generator=g, requires_grad=True)
    out = opd_full_kd(h4, h1, h4, s_head, s_head, 0.5, divergence, chunk_size=2)
    grads = torch.autograd.grad(out["kd"].sum(), [h4, h1, s_head.weight], allow_unused=True)
    kd, _, _ = _manual_kl(s_head, s_head, h4, h1.detach(), h4.detach(), 0.5, divergence)
    ref = torch.autograd.grad(kd.sum(), [h4, s_head.weight])
    _close(out["kd"].detach(), kd.detach())
    _close(grads[0], ref[0])
    assert grads[1] is None  # loop 1 only enters the teacher
    _close(grads[2], ref[1])
    # the student starts away from its own extrapolation: the loss is not zero at w > 0
    assert float(out["kd"].mean()) > 0


def test_elpo_config_is_the_full_rkl_config_with_the_online_teacher(env):
    # saved at the user's request (2026-11-26): opd_full_rkl + opd.teacher online, new wandb naming
    elpo = OmegaConf.to_container(rcfg.load_config(str(ELPO_CONFIG)))
    rkl = OmegaConf.to_container(rcfg.load_config(str(FULL_CONFIGS["reverse"])))
    assert elpo["opd"].pop("teacher") == "online" and "teacher" not in rkl["opd"]
    assert elpo["wandb"].pop("run_name") == "elpo-online-rkl-loopcd-w0.5-1p4b"
    assert "online-teacher" in elpo["wandb"].pop("tags")
    rkl["wandb"].pop("run_name")
    rkl["wandb"].pop("tags")
    assert elpo == rkl
    cfg = rcfg.load_config(str(ELPO_CONFIG))
    assert rcfg.opd_settings(cfg) == opd.OPDSettings(enabled=True, omega=0.5, ref_loop=1,
                                                     estimator="full", divergence="reverse",
                                                     teacher="online")
    assert rcfg.looprl_method(cfg) == "opd" and cfg.trainer == "rltt"


def test_elpo_dry_run(env, capsys, monkeypatch):
    from rltt_baseline import launch

    _clean_launch_env(monkeypatch)
    assert launch.main(["--config", str(ELPO_CONFIG), "--dry-run"]) == 0
    resolved = json.loads(capsys.readouterr().out)
    ar = resolved["verl_overrides"]["actor_rollout_ref"]
    assert ar["actor"]["opd"] == {"enabled": True, "omega": 0.5, "ref_loop": 1,
                                  "estimator": "full", "divergence": "reverse",
                                  **COEF_DEFAULTS, "teacher": "online"}
    assert ar["loss_agg"] == {"mode": "seq-mean-token-mean", "norm_len": None}
    assert resolved["reference_policy"].startswith("opd teacher (online, full vocabulary)")
    assert resolved["config"]["wandb"]["mode"] == "online"


GRPO_K2_ONLINE_CONFIG = paths.config_file("grpo_k2_online_loopcd_1p4b_8gpu.yaml")


def test_grpo_k2_online_config_is_grpo_k2_with_the_online_teacher(env):
    # saved at the user's request (2026-11-26): grpo_k2 with the online teacher and beta 0.05
    online = OmegaConf.to_container(rcfg.load_config(str(GRPO_K2_ONLINE_CONFIG)))
    frozen = OmegaConf.to_container(rcfg.load_config(str(GRPO_K2_CONFIG)))
    assert online["opd"].pop("teacher") == "online" and "teacher" not in frozen["opd"]
    assert online["opd"].pop("kd_coef") == 0.05 and frozen["opd"].pop("kd_coef") == 0.01
    assert online["wandb"].pop("run_name") == "grpo-k2-online-loopcd-w0.5-beta0.05-1p4b"
    assert "online-teacher" in online["wandb"].pop("tags")
    frozen["wandb"].pop("run_name")
    frozen["wandb"].pop("tags")
    assert online == frozen
    cfg = rcfg.load_config(str(GRPO_K2_ONLINE_CONFIG))
    assert rcfg.opd_settings(cfg) == opd.OPDSettings(
        **{**GRPO_K2_SECTION, "kd_coef": 0.05, "teacher": "online"})
    assert rcfg.loss_agg_settings(cfg).mode == "seq-mean-token-mean"


def test_grpo_k2_online_dry_run(env, capsys, monkeypatch):
    from rltt_baseline import launch

    _clean_launch_env(monkeypatch)
    assert launch.main(["--config", str(GRPO_K2_ONLINE_CONFIG), "--dry-run"]) == 0
    resolved = json.loads(capsys.readouterr().out)
    ar = resolved["verl_overrides"]["actor_rollout_ref"]
    assert ar["actor"]["opd"] == {**GRPO_K2_SECTION, "kd_coef": 0.05, "teacher": "online"}
    assert ar["loss_agg"] == {"mode": "seq-mean-token-mean", "norm_len": None}
    assert "engine_kwargs" not in ar["rollout"]  # ordinary last-loop rollout
    assert resolved["reference_policy"].startswith("opd teacher (online): ")
    assert resolved["config"]["wandb"]["mode"] == "online"
