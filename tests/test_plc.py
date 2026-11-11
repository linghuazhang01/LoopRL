"""Posterior Loop Credit: pure-torch loss / weight tests and config plumbing (CPU only)."""

import json
import sys

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from rltt_baseline import config as rcfg
from rltt_baseline import paths, plc
from rltt_baseline.ouro_rltt.rltt_per_loop import exit_pdf_from_gates

CONFIG = paths.config_file("plc_posterior_exit_gate_1p4b_8gpu.yaml")
NOGATE_CONFIG = paths.config_file("plc_posterior_exit_nogate_1p4b_8gpu.yaml")
# weight-mode configs: the no-gate config with these plc / args keys changed (and wandb naming)
WEIGHT_MODE_CONFIGS = {
    "plc_logp_softmax_1p4b_8gpu.yaml": {
        "plc": {"weight_mode": "logp_softmax", "prior": "uniform"},
        "args": {"loop_weighting": "uniform"}},
    "plc_rltt_progressive_1p4b_8gpu.yaml": {
        "plc": {"weight_mode": "rltt", "prior": "progressive"},
        "args": {"loop_weighting": "progressive"}},
}
ENV = {
    "LOOPRL_OURO_DIR": "/models/Ouro-1.4B-Thinking",
    "LOOPRL_MODEL_DIR": "/models/Ouro-1.4B-Thinking-RLTT",
    "LOOPRL_TRAIN_FILE": "/data/deepmath_level6.parquet",
    "LOOPRL_MATH500_JSONL": "/data/math500.jsonl",
    "LOOPRL_OUTPUT_DIR": "/out/plc",
    "LOOPRL_ENV_FILE": "/nonexistent/env.local",  # never read a real env.local in tests
}
T = 4


def _random_batch(B=3, L=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    ell_old = -torch.rand(B, L, T, generator=g) * 4
    ell_new = ell_old + 0.3 * torch.randn(B, L, T, generator=g)
    adv = torch.randn(B, generator=g)[:, None].expand(B, L).clone()
    mask = torch.ones(B, L)
    mask[0, -2:] = 0
    mask[2, -1] = 0
    return ell_old, ell_new, adv, mask


def _uniform_logq(like):
    return plc.loop_prior_logq("uniform", T, like=like)


# ---------- weights ----------

@pytest.mark.parametrize("weight_mode",
                         [m for m in plc.WEIGHT_MODES if m not in plc.EARLY_LOOP_WEIGHT_MODES])
@pytest.mark.parametrize("prior", ["uniform", "progressive", "exit"])
@pytest.mark.parametrize("eps_floor", [0.0, 0.1])
def test_weights_sum_to_one(weight_mode, prior, eps_floor):
    ell_old, _, _, mask = _random_batch()
    pdf = torch.softmax(torch.randn(3, 5, T), -1)
    logq = plc.loop_prior_logq(prior, T, alpha=2.0, exit_pdf=pdf, like=ell_old)
    w = plc.compute_loop_weights(ell_old, logq, gamma=1.0, weight_mode=weight_mode,
                                 eps_floor=eps_floor)
    assert w.shape == ell_old.shape and not w.requires_grad
    assert torch.allclose(w.sum(-1), torch.ones(3, 5), atol=1e-5)
    assert (w >= 0).all()
    if eps_floor > 0:
        assert (w >= eps_floor / T - 1e-6).all()


@pytest.mark.parametrize("prior", ["uniform", "progressive", "exit"])
def test_gamma0_equals_the_prior_and_rltt_mode(prior):
    ell_old, _, _, _ = _random_batch()
    pdf = torch.softmax(torch.randn(3, 5, T), -1)
    logq = plc.loop_prior_logq(prior, T, alpha=1.5, exit_pdf=pdf, like=ell_old)
    w0 = plc.compute_loop_weights(ell_old, logq, gamma=0.0)
    w_rltt = plc.compute_loop_weights(ell_old, logq, gamma=1.0, weight_mode="rltt")
    expected = logq.exp().expand_as(ell_old)
    assert torch.allclose(w0, expected, atol=1e-6)
    assert torch.allclose(w_rltt, expected, atol=1e-6)
    if prior == "progressive":
        t = torch.arange(1, T + 1, dtype=torch.float32)
        assert torch.allclose(w0[0, 0], t ** 1.5 / (t ** 1.5).sum(), atol=1e-6)


def test_toy_values_from_the_spec():
    probs = torch.tensor([[0.05, 0.2, 0.6, 0.9],
                          [0.9, 0.4, 0.2, 0.1],
                          [0.704, 0.789, 0.804, 0.806]]).view(1, 3, T)
    w = plc.compute_loop_weights(probs.log(), _uniform_logq(probs), gamma=1.0)
    expected = torch.tensor([[0.0286, 0.1143, 0.3429, 0.5143],
                             [0.5625, 0.25, 0.125, 0.0625],
                             [0.2268, 0.2542, 0.2590, 0.2597]])
    assert torch.allclose(w[0], expected, atol=1e-3)
    assert torch.allclose(w, probs / probs.sum(-1, keepdim=True), atol=1e-6)


@pytest.mark.parametrize("eps_floor", [0.0, 0.1])
def test_uniform_mode_is_one_over_T_and_matches_rltt_uniform(eps_floor):
    ell_old, _, _, _ = _random_batch()
    logq = _uniform_logq(ell_old)
    w = plc.compute_loop_weights(ell_old, logq, gamma=1.0, weight_mode="uniform",
                                 eps_floor=eps_floor)
    assert torch.equal(w, torch.full_like(ell_old, 1.0 / T))
    for same in (plc.compute_loop_weights(ell_old, logq, 1.0, "rltt", eps_floor),
                 plc.compute_loop_weights(ell_old, logq, 0.0, "posterior", eps_floor)):
        assert torch.allclose(w, same, atol=1e-6)


@pytest.mark.parametrize("eps_floor", [0.0, 0.1])
def test_logp_softmax_mode_is_the_probability_share_of_each_loop(eps_floor):
    ell_old, _, _, _ = _random_batch()
    logq = _uniform_logq(ell_old)
    w = plc.compute_loop_weights(ell_old, logq, gamma=0.0, weight_mode="logp_softmax",
                                 eps_floor=eps_floor)
    probs = ell_old.exp()
    share = probs / probs.sum(-1, keepdim=True)
    assert torch.allclose(w, (1 - eps_floor) * share + eps_floor / T, atol=1e-6)
    # = PLC with the uniform prior and gamma = 1
    assert torch.allclose(w, plc.compute_loop_weights(ell_old, logq, 1.0, "posterior", eps_floor),
                          atol=1e-6)
    # gamma and logq are ignored
    other = plc.compute_loop_weights(ell_old, logq + torch.randn(T), gamma=3.0,
                                     weight_mode="logp_softmax", eps_floor=eps_floor)
    assert torch.allclose(w, other, atol=1e-6)


def test_logp_softmax_weights_give_the_gradient_of_the_mean_loop_likelihood():
    """``sum_t sg[softmax_t(ell)] grad ell_t = grad log (1/T) sum_t P^(t)(y)``."""
    torch.manual_seed(0)
    V = 7
    logits = torch.randn(2, 3, T, V, requires_grad=True)
    y = torch.randint(V, (2, 3))
    logp = F.log_softmax(logits, -1).gather(-1, y[..., None, None].expand(2, 3, T, 1)).squeeze(-1)
    w = plc.compute_loop_weights(logp, _uniform_logq(logp), gamma=1.0, weight_mode="logp_softmax")
    grad = torch.autograd.grad((w * logp).sum(), logits, retain_graph=True)[0]
    target = torch.log(logp.exp().mean(-1)).sum()
    assert torch.allclose(grad, torch.autograd.grad(target, logits)[0], atol=1e-6)


def test_exit_prior_uses_the_model_exit_pdf():
    gates = [torch.randn(3, 5, 1) for _ in range(T)]
    pdf = exit_pdf_from_gates(gates)
    logq = plc.loop_prior_logq("exit", T, exit_pdf=pdf.requires_grad_(True))
    assert not logq.requires_grad
    assert torch.allclose(logq.exp().sum(-1), torch.ones(3, 5), atol=1e-5)
    with pytest.raises(ValueError, match="exit_pdf"):
        plc.loop_prior_logq("exit", T)


def test_importance_weights_turn_posterior_into_plain_per_loop_is():
    ell_old, _, _, _ = _random_batch()
    logq = _uniform_logq(ell_old)
    w = plc.compute_loop_weights(ell_old, logq, gamma=1.0)
    is_w = plc.importance_weights(ell_old, logq, is_clip=1e9)
    rho = torch.exp(ell_old - ell_old[..., -1:])          # P^(t) / P^(T)
    assert torch.allclose(is_w[..., None] * w, rho / T, atol=1e-5)
    capped = plc.importance_weights(ell_old, logq, is_clip=0.5)
    assert torch.allclose(capped, is_w.clamp(max=0.5)) and (capped <= 0.5).all()


def test_stop_gradient_weights_give_the_weighted_sum_of_loop_gradients():
    torch.manual_seed(0)
    V = 7
    logits = torch.randn(1, 1, T, V, requires_grad=True)
    y = torch.tensor([[3]])
    logp = F.log_softmax(logits, -1).gather(-1, y.view(1, 1, 1, 1).expand(1, 1, T, 1)).squeeze(-1)
    w = plc.compute_loop_weights(logp.detach(), _uniform_logq(logp), gamma=1.0)
    grad = torch.autograd.grad((w * logp).sum(), logits, retain_graph=True)[0]
    expected = sum(w[0, 0, t] * torch.autograd.grad(logp[0, 0, t], logits, retain_graph=True)[0]
                   for t in range(T))
    assert torch.allclose(grad, expected, atol=1e-6)
    w_live = torch.softmax(_uniform_logq(logp) + logp, -1)   # no detach
    grad_live = torch.autograd.grad((w_live * logp).sum(), logits)[0]
    assert not torch.allclose(grad, grad_live, atol=1e-6)


# ---------- policy loss ----------

def _verl_vanilla_loss(old_log_prob, log_prob, advantages, response_mask, clip_ratio=0.2,
                       clip_ratio_c=3.0, loss_agg_mode="token-mean"):
    """verl 0.6.1 core_algos.compute_policy_loss_vanilla, transcribed for the test."""
    negative_approx_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    return plc.agg_loss(pg_losses, response_mask, loss_agg_mode)


@pytest.mark.parametrize("loss_agg_mode", plc.LOSS_AGG_MODES)
def test_lam0_is_verl_grpo_on_the_final_loop(loss_agg_mode):
    ell_old, ell_new, adv, mask = _random_batch(seed=1)
    ell_new[0, 0, -1] += 3.0   # exercise the upper clip
    adv[1] = -2.0
    ell_new[1, 1, -1] += 2.0   # dual clip on a negative advantage
    w = plc.compute_loop_weights(ell_old, _uniform_logq(ell_old), gamma=1.0)
    ell_new = ell_new.requires_grad_(True)
    loss, metrics = plc.plc_policy_loss(ell_new, ell_old, adv, mask, w, lam=0.0, clip_ratio=0.2,
                                        loss_agg_mode=loss_agg_mode)
    expected = _verl_vanilla_loss(ell_old[..., -1], ell_new[..., -1], adv, mask,
                                  loss_agg_mode=loss_agg_mode)
    assert torch.allclose(loss, expected, atol=1e-6)
    grad = torch.autograd.grad(loss, ell_new)[0]
    assert torch.all(grad[..., :-1] == 0)      # nothing reaches the earlier loops
    assert metrics["actor/pg_clipfrac"] > 0 and metrics["actor/pg_clipfrac_lower"] > 0


def test_lam1_gamma0_uniform_is_rltt_uniform_with_ratios():
    ell_old, _, adv, mask = _random_batch(seed=2)
    ell_new = ell_old.clone().requires_grad_(True)   # on-policy point: ratios are 1
    w = plc.compute_loop_weights(ell_old, _uniform_logq(ell_old), gamma=0.0)
    loss, _ = plc.plc_policy_loss(ell_new, ell_old, adv, mask, w, lam=1.0, clip_ratio=0.2)
    grad = torch.autograd.grad(loss, ell_new)[0]
    expected = -(adv * mask)[..., None].expand_as(grad) / (T * mask.sum())
    assert torch.allclose(grad, expected, atol=1e-6)
    assert torch.allclose(loss, -(adv * mask).sum() / mask.sum(), atol=1e-6)


def test_posterior_loss_gradient_is_weights_times_advantage_at_the_on_policy_point():
    ell_old, _, adv, mask = _random_batch(seed=3)
    ell_new = ell_old.clone().requires_grad_(True)
    w = plc.compute_loop_weights(ell_old, _uniform_logq(ell_old), gamma=1.0)
    loss, metrics = plc.plc_policy_loss(ell_new, ell_old, adv, mask, w, lam=1.0, clip_ratio=0.2)
    grad = torch.autograd.grad(loss, ell_new)[0]
    expected = -(adv * mask)[..., None] * w / mask.sum()
    assert torch.allclose(grad, expected, atol=1e-6)
    assert metrics["plc/ratio_mean_per_loop"] == pytest.approx([1.0] * T)
    assert metrics["plc/clipfrac_per_loop"] == pytest.approx([0.0] * T)


def test_lam_mixes_final_and_loop_terms():
    ell_old, ell_new, adv, mask = _random_batch(seed=4)
    w = plc.compute_loop_weights(ell_old, _uniform_logq(ell_old), gamma=1.0)
    l0, _ = plc.plc_policy_loss(ell_new, ell_old, adv, mask, w, lam=0.0, clip_ratio=0.2)
    l1, _ = plc.plc_policy_loss(ell_new, ell_old, adv, mask, w, lam=1.0, clip_ratio=0.2)
    lh, _ = plc.plc_policy_loss(ell_new, ell_old, adv, mask, w, lam=0.5, clip_ratio=0.2)
    assert torch.allclose(lh, 0.5 * (l0 + l1), atol=1e-6)


def test_policy_loss_rejects_bad_shapes():
    ell_old, ell_new, adv, mask = _random_batch()
    w = plc.compute_loop_weights(ell_old, _uniform_logq(ell_old), gamma=1.0)
    with pytest.raises(ValueError, match="B, L, T"):
        plc.plc_policy_loss(ell_new[..., -1], ell_old[..., -1], adv, mask, w, 1.0, 0.2)
    with pytest.raises(ValueError, match="loss_agg_mode"):
        plc.plc_policy_loss(ell_new, ell_old, adv, mask, w, 1.0, 0.2, loss_agg_mode="mean")


def test_agg_loss_modes_match_their_definitions():
    mat = torch.arange(6, dtype=torch.float32).view(2, 3)
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    assert plc.agg_loss(mat, mask, "token-mean").item() == pytest.approx((0 + 1 + 3 + 4 + 5) / 5)
    assert plc.agg_loss(mat, mask, "seq-mean-token-sum").item() == pytest.approx((1 + 12) / 2)
    assert plc.agg_loss(mat, mask, "seq-mean-token-mean").item() == pytest.approx((0.5 + 4) / 2)
    assert plc.agg_loss(mat, mask, "seq-mean-token-sum-norm").item() == pytest.approx(13 / 3)


# ---------- gate loss ----------

def _gate_step(adv_value, entropy_coef, gate_coef=1.0, lr=0.5):
    torch.manual_seed(0)
    g = torch.randn(1, 1, T, requires_grad=True)
    w = torch.softmax(torch.randn(1, 1, T), -1)
    adv = torch.full((1, 1), adv_value)
    mask = torch.ones(1, 1)

    def kl_and_entropy(q):
        return ((w * (w.log() - q.log())).sum().item(), -(q * q.log()).sum().item())

    before = kl_and_entropy(torch.softmax(g, -1))
    loss, metrics = plc.gate_loss(torch.softmax(g, -1), w, adv, mask, gate_coef, entropy_coef)
    (grad,) = torch.autograd.grad(loss, g)
    after = kl_and_entropy(torch.softmax(g - lr * grad, -1))
    return before, after, metrics


def test_gate_ce_moves_q_towards_w_on_positive_and_away_on_negative_advantage():
    before, after, metrics = _gate_step(adv_value=1.0, entropy_coef=0.0)
    assert after[0] < before[0]
    assert metrics["gate/kl_w_q_mean"] == pytest.approx(before[0], abs=1e-5)
    before, after, _ = _gate_step(adv_value=-1.0, entropy_coef=0.0)
    assert after[0] > before[0]


def test_gate_entropy_term_raises_the_gate_entropy():
    before, after, _ = _gate_step(adv_value=0.0, entropy_coef=1.0)
    assert after[1] > before[1]


def test_gate_loss_metrics_and_zero_coefficients():
    pdf = torch.softmax(torch.randn(2, 3, T), -1)
    w = torch.softmax(torch.randn(2, 3, T), -1)
    adv = torch.ones(2)
    mask = torch.ones(2, 3)
    loss, metrics = plc.gate_loss(pdf, w, adv, mask, gate_coef=0.0, gate_entropy_coef=0.0)
    assert loss.item() == 0.0
    assert len(metrics["gate/q_mean_per_loop"]) == T
    assert 1.0 <= metrics["gate/expected_exit_step"] <= T


# ---------- diagnostics ----------

def test_weight_stats_on_a_jump_token():
    probs = torch.tensor([0.05, 0.6, 0.62, 0.621]).view(1, 1, T)   # final loop is the argmax
    ell = probs.log()
    w = plc.compute_loop_weights(ell, _uniform_logq(ell), gamma=1.0)
    stats = plc.weight_stats(w, ell, torch.ones(1, 1))
    assert stats["plc/w_mean_per_loop"] == pytest.approx(w[0, 0].tolist(), abs=1e-6)
    assert 1.0 < stats["plc/eff_loops_mean"] < T
    assert stats["plc/argmax_is_final_frac"] == 1.0
    assert stats["plc/delta_neg_frac_per_transition"] == [0.0, 0.0, 0.0]
    assert stats["plc/delta_abs_lt_0.01_frac"] == pytest.approx(1 / 3)
    assert stats["plc/final_logp_mean"] == pytest.approx(float(ell[0, 0, -1]))


def test_low_var_kl_is_zero_at_equality_and_nonnegative():
    a = torch.randn(4, 5)
    assert torch.all(plc.low_var_kl(a, a) == 0)
    assert torch.all(plc.low_var_kl(a, a + torch.randn(4, 5)) >= 0)


# ---------- settings ----------

def test_settings_defaults_and_validation():
    s = plc.PLCSettings.from_mapping({})
    assert not s.enabled and s.gamma == 1.0 and s.lam == 1.0 and not s.needs_exit_pdf
    s = plc.PLCSettings.from_mapping({"enabled": True, "prior": "exit", "use_gate_loss": True,
                                      "gamma": 1, "lam": 1})
    assert s.needs_exit_pdf and isinstance(s.gamma, float) and "gate_loss=True" in s.describe()
    with pytest.raises(KeyError, match="unknown plc keys"):
        plc.PLCSettings.from_mapping({"gama": 1.0})
    with pytest.raises(ValueError, match="prior=exit"):
        plc.PLCSettings.from_mapping({"use_gate_loss": True})
    with pytest.raises(ValueError, match="lam"):
        plc.PLCSettings.from_mapping({"lam": 1.5})
    with pytest.raises(TypeError, match="bool"):
        plc.PLCSettings.from_mapping({"enabled": "yes"})
    with pytest.raises(ValueError, match="weight_mode"):
        plc.PLCSettings.from_mapping({"weight_mode": "delta"})
    for mode in plc.PRIOR_FREE_WEIGHT_MODES:
        s = plc.PLCSettings.from_mapping({"enabled": True, "weight_mode": mode})
        assert s.weight_mode == mode and not s.needs_exit_pdf
        with pytest.raises(ValueError, match="needs plc.prior=uniform"):
            plc.PLCSettings.from_mapping({"weight_mode": mode, "prior": "exit"})
        with pytest.raises(ValueError, match="prior"):
            plc.PLCSettings.from_mapping({"weight_mode": mode, "prior": "exit",
                                          "use_gate_loss": True})


# ---------- config plumbing ----------

@pytest.fixture
def env(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)


def _args(**over):
    base = {"use_rltt_model": True, "loop_weighting": "exit_pdf", "rltt_model_path": "/m"}
    base.update(over)
    return OmegaConf.create(base)


def test_plc_config_resolves_to_gamma1_lam1_with_gate_loss(env):
    cfg = rcfg.load_config(str(CONFIG))
    s = rcfg.plc_settings(cfg)
    assert s.enabled and s.weight_mode == "posterior" and s.prior == "exit"
    assert s.gamma == 1.0 and s.lam == 1.0 and s.use_gate_loss and s.gate_coef == 1.0
    assert s.gate_entropy_coef == 0.05 and not s.use_is_correction
    assert cfg.trainer == "rltt" and cfg.args.use_rltt_model
    assert cfg.args.gradient_accumulation_steps == 1
    assert cfg.model.num_hidden_layers == 24
    overrides = rcfg.add_plc_overrides(cfg, rcfg.verl_overrides(cfg), _args())
    actor = overrides["actor_rollout_ref"]["actor"]
    assert actor["plc"]["gamma"] == 1.0 and actor["plc"]["use_gate_loss"] is True
    assert actor["ppo_mini_batch_size"] == 64 and overrides["data"]["train_batch_size"] == 64
    assert rcfg.wandb_env(cfg)["WANDB_TAGS"].startswith("plc,")


def test_plc_section_is_validated_and_can_be_overridden(env):
    cfg = rcfg.load_config(str(CONFIG), ["plc.gamma=0", "plc.lam=0.5", "plc.use_gate_loss=false"])
    s = rcfg.plc_settings(cfg)
    assert s.gamma == 0.0 and s.lam == 0.5 and not s.use_gate_loss and s.needs_exit_pdf
    with pytest.raises(KeyError, match="unknown plc keys"):
        rcfg.load_config(str(CONFIG), ["plc.gama=2"])
    with pytest.raises(ValueError, match="prior=exit"):
        rcfg.load_config(str(CONFIG), ["plc.prior=uniform"])


@pytest.mark.parametrize("mode", plc.PRIOR_FREE_WEIGHT_MODES)
def test_prior_free_weight_modes_are_overrides_of_the_plc_config(env, mode):
    cfg = rcfg.load_config(str(CONFIG), [f"plc.weight_mode={mode}", "plc.prior=uniform",
                                         "plc.use_gate_loss=false"])
    s = rcfg.plc_settings(cfg)
    assert s.weight_mode == mode and s.prior == "uniform" and not s.needs_exit_pdf
    actor = rcfg.add_plc_overrides(cfg, rcfg.verl_overrides(cfg), _args())["actor_rollout_ref"][
        "actor"]
    assert actor["plc"]["weight_mode"] == mode
    with pytest.raises(ValueError, match="needs plc.prior=uniform"):
        rcfg.load_config(str(CONFIG), [f"plc.weight_mode={mode}", "plc.use_gate_loss=false"])


def test_plc_overrides_require_the_rltt_trainer_and_per_loop_model(env):
    cfg = rcfg.load_config(str(CONFIG))
    with pytest.raises(ValueError, match="use_rltt_model"):
        rcfg.add_plc_overrides(cfg, {}, _args(use_rltt_model=False))
    with pytest.raises(ValueError, match="learned"):
        rcfg.add_plc_overrides(cfg, {}, _args(loop_weighting="learned"))
    off = rcfg.load_config(str(CONFIG), ["plc.enabled=false"])
    assert rcfg.add_plc_overrides(off, {"x": 1}, _args(use_rltt_model=False)) == {"x": 1}
    grpo = rcfg.load_config(str(paths.config_file("grpo_baseline_1p4b_8gpu.yaml")),
                            ["plc.enabled=true", "plc.prior=uniform", "plc.use_gate_loss=false"])
    with pytest.raises(ValueError, match="trainer=rltt"):
        rcfg.add_plc_overrides(grpo, {}, _args())


def test_nogate_config_is_the_plc_config_with_the_gate_loss_off(env):
    # the gate-loss ablation keeps every key of the PLC config but use_gate_loss and wandb naming
    gate = OmegaConf.to_container(rcfg.load_config(str(CONFIG)))
    nogate = OmegaConf.to_container(rcfg.load_config(str(NOGATE_CONFIG)))
    assert gate["plc"].pop("use_gate_loss") is True
    assert nogate["plc"].pop("use_gate_loss") is False
    for cfg in (gate, nogate):
        for key in ("run_name", "tags"):
            cfg["wandb"].pop(key)
    assert nogate == gate
    s = rcfg.plc_settings(rcfg.load_config(str(NOGATE_CONFIG)))
    assert s.enabled and s.prior == "exit" and s.needs_exit_pdf and not s.use_gate_loss


@pytest.mark.parametrize("name", sorted(WEIGHT_MODE_CONFIGS))
def test_weight_mode_configs_are_the_nogate_config_with_other_weights(env, name):
    nogate = OmegaConf.to_container(rcfg.load_config(str(NOGATE_CONFIG)))
    cfg = OmegaConf.to_container(rcfg.load_config(str(paths.config_file(name))))
    for section, values in WEIGHT_MODE_CONFIGS[name].items():
        for key, value in values.items():
            assert cfg[section].pop(key) == value, (name, section, key)
            nogate[section].pop(key)
    for c in (cfg, nogate):
        for key in ("run_name", "tags"):
            c["wandb"].pop(key)
    assert cfg == nogate
    s = rcfg.plc_settings(rcfg.load_config(str(paths.config_file(name))))
    assert s.enabled and not s.needs_exit_pdf and not s.use_gate_loss and s.lam == 1.0


def test_weight_mode_configs_give_the_intended_weights(env):
    ell_old, _, _, _ = _random_batch()
    expected = {
        "plc_logp_softmax_1p4b_8gpu.yaml": torch.softmax(ell_old, -1),
        "plc_rltt_progressive_1p4b_8gpu.yaml": torch.tensor([0.1, 0.2, 0.3, 0.4]).expand_as(ell_old),
    }
    for name, want in expected.items():
        s = rcfg.plc_settings(rcfg.load_config(str(paths.config_file(name))))
        logq = plc.loop_prior_logq(s.prior, T, s.progressive_alpha, like=ell_old)
        w = plc.compute_loop_weights(ell_old, logq, s.gamma, s.weight_mode, s.eps_floor)
        assert torch.allclose(w, want, atol=1e-6), name


def test_other_configs_have_plc_off(env):
    for name in ("rltt_baseline.yaml", "rltt_baseline_1p4b_8gpu.yaml",
                 "grpo_baseline_1p4b_8gpu.yaml", "grpo_baseline_8gpu.yaml"):
        assert not rcfg.plc_settings(rcfg.load_config(str(paths.config_file(name)))).enabled


def test_check_plc_model_needs_an_undetached_exit_pdf(tmp_path):
    from rltt_baseline import launch

    (tmp_path / "config.json").write_text(json.dumps({"rltt_detach_exit_pdf": True}))
    on = plc.PLCSettings.from_mapping({"enabled": True, "prior": "exit", "use_gate_loss": True})
    args = OmegaConf.create({"rltt_model_path": str(tmp_path)})
    with pytest.raises(SystemExit, match="detach"):
        launch.check_plc_model(args, on)
    (tmp_path / "config.json").write_text(json.dumps({"rltt_detach_exit_pdf": False}))
    launch.check_plc_model(args, on)
    launch.check_plc_model(OmegaConf.create({"rltt_model_path": "/nowhere"}),
                           plc.PLCSettings.from_mapping({"enabled": True}))


def test_plc_dry_run(env, capsys, monkeypatch):
    from rltt_baseline import launch

    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "PYTHONPATH"):
        monkeypatch.setenv(key, "")
    monkeypatch.setattr(sys, "path", list(sys.path))
    assert launch.main(["--config", str(CONFIG), "--dry-run", "plc.gamma=0.5"]) == 0
    resolved = json.loads(capsys.readouterr().out)
    actor = resolved["verl_overrides"]["actor_rollout_ref"]["actor"]
    assert actor["plc"]["enabled"] is True and actor["plc"]["gamma"] == 0.5
    assert actor["plc"]["prior"] == "exit" and actor["plc"]["use_gate_loss"] is True
    assert resolved["vendor_args"]["use_rltt_model"] is True
    assert resolved["vendor_args"]["rltt_model_path"] == ENV["LOOPRL_MODEL_DIR"]
    assert resolved["vendor_args"]["n_gpus"] == 8


# ---------- GRPO + loop auxiliary term (objective additive, is_trunc, group baseline) ----------

LOOPAUX_CONFIG = paths.config_file("plc_grpo_loopaux_1p4b_8gpu.yaml")
# the loop-aux config: the no-gate config with these plc / args keys changed (and wandb naming)
LOOPAUX_CHANGES = {
    "plc": {"weight_mode": "is_trunc", "prior": "uniform", "lam": 0.1, "objective": "additive",
            "loop_baseline": "none", "loop_agg_mode": "same", "trunc_cap": 1.0, "kl_coef": 0.2},
    "args": {"loop_weighting": "uniform"},
}


def _groups(B, size=2):
    return [f"g{i // size}" for i in range(B)]


def test_early_loop_weights_are_the_prior_on_earlier_loops_times_the_truncated_ratio():
    ell_old, _, _, _ = _random_batch(B=4, L=6, seed=10)
    for prior, alpha in (("uniform", torch.full((T - 1,), 1 / 3)),
                         ("progressive", torch.tensor([1.0, 2.0, 3.0]) / 6)):
        logq = plc.loop_prior_logq(prior, T, 1.0, like=ell_old)
        early = plc.compute_loop_weights(ell_old, logq, 1.0, "early_prior")
        trunc = plc.compute_loop_weights(ell_old, logq, 1.0, "is_trunc")
        ratio = torch.exp(ell_old[..., :-1] - ell_old[..., -1:]).clamp(max=1.0)
        assert torch.allclose(early[..., :-1], alpha.expand_as(ratio), atol=1e-6)
        assert torch.allclose(trunc[..., :-1], alpha * ratio, atol=1e-6)
        assert (early[..., -1] == 0).all() and (trunc[..., -1] == 0).all()
        assert (trunc <= early + 1e-7).all() and not trunc.requires_grad


def test_sequence_weight_mass_follows_the_loop_aggregation():
    w = torch.rand(3, 5, T)
    mask = torch.ones(3, 5)
    mask[0, 3:] = 0
    total = (w * mask[..., None]).sum(1)
    assert torch.allclose(plc.sequence_weight_mass(w, mask, "token-mean"), total)
    assert torch.allclose(plc.sequence_weight_mass(w, mask, "seq-mean-token-mean"),
                          total / mask.sum(1, keepdim=True))
    with pytest.raises(ValueError, match="group baseline"):
        plc.sequence_weight_mass(w, mask, "seq-mean-token-sum")
    adv = torch.tensor([1.5, -0.5, 0.0])[:, None] * mask
    assert torch.allclose(plc.sequence_advantages(adv, mask), torch.tensor([1.5, -0.5, 0.0]))


def test_group_baseline_zeroes_the_weighted_advantage_mass_of_every_group_and_loop():
    g = torch.Generator().manual_seed(42)
    B = 12
    seq_adv = torch.randn(B, generator=g)
    mass = torch.rand(B, T, generator=g)
    mass[4:8, 1] = 0.0                       # group g1 has no mass at loop 2
    groups = [f"g{i // 4}" for i in range(B)]
    b = plc.group_loop_baseline(seq_adv, mass, groups)
    assert b.shape == (B, T)
    resid = mass * (seq_adv[:, None] - b)
    for k in range(3):
        assert torch.allclose(resid[4 * k:4 * k + 4].sum(0), torch.zeros(T), atol=1e-6)
        assert torch.allclose(b[4 * k], b[4 * k + 3])          # one baseline per group
    assert (b[4:8, 1] == 0).all()
    # a constant shift of a group's advantages moves its baseline by the same amount
    b2 = plc.group_loop_baseline(seq_adv + torch.tensor([3.0] * 4 + [0.0] * 8), mass, groups)
    assert torch.allclose(b2[:4], b[:4] + 3.0, atol=1e-5) and torch.allclose(b2[4:], b[4:])
    with pytest.raises(ValueError, match="same responses"):
        plc.group_loop_baseline(seq_adv, mass, groups[:-1])


@pytest.mark.parametrize("loss_agg_mode", plc.LOSS_AGG_MODES)
def test_additive_lam0_is_verl_grpo_on_the_final_loop(loss_agg_mode):
    ell_old, ell_new, adv, mask = _random_batch(seed=11)
    ell_new[0, 0, -1] += 3.0
    adv[1] = -2.0
    ell_new[1, 1, -1] += 2.0
    w = plc.compute_loop_weights(ell_old, _uniform_logq(ell_old), 1.0, "is_trunc")
    loop_adv = torch.randn_like(ell_old)
    ell_new = ell_new.requires_grad_(True)
    loss, _ = plc.plc_policy_loss(ell_new, ell_old, adv, mask, w, lam=0.0, clip_ratio=0.2,
                                  loss_agg_mode=loss_agg_mode, objective="additive",
                                  loop_advantages=loop_adv, loop_agg_mode="seq-mean-token-mean")
    expected = _verl_vanilla_loss(ell_old[..., -1], ell_new[..., -1], adv, mask,
                                  loss_agg_mode=loss_agg_mode)
    assert torch.allclose(loss, expected, atol=1e-6)
    assert torch.all(torch.autograd.grad(loss, ell_new)[0][..., :-1] == 0)


def test_additive_loss_is_grpo_plus_lam_times_the_loop_term():
    ell_old, ell_new, adv, mask = _random_batch(seed=12)
    w = plc.compute_loop_weights(ell_old, _uniform_logq(ell_old), 1.0, "is_trunc")
    loop_adv = torch.randn_like(ell_old)
    kw = dict(clip_ratio=0.2, objective="additive", loop_advantages=loop_adv,
              loop_agg_mode="seq-mean-token-mean")
    l0, _ = plc.plc_policy_loss(ell_new, ell_old, adv, mask, w, lam=0.0, **kw)
    l1, _ = plc.plc_policy_loss(ell_new, ell_old, adv, mask, w, lam=1.0, **kw)
    l3, metrics = plc.plc_policy_loss(ell_new, ell_old, adv, mask, w, lam=0.3, **kw)
    assert torch.allclose(l3, l0 + 0.3 * (l1 - l0), atol=1e-6)
    assert len(metrics["plc/loop_adv_abs_mean_per_loop"]) == T
    # the mix objective with the default loop aggregation is unchanged
    lm, _ = plc.plc_policy_loss(ell_new, ell_old, adv, mask, w, lam=0.3, clip_ratio=0.2)
    per = plc.plc_policy_loss(ell_new, ell_old, adv, mask, w, lam=0.3, clip_ratio=0.2,
                              loop_agg_mode="token-mean")[0]
    assert torch.allclose(lm, per, atol=1e-6)


def test_loop_gradient_has_zero_mass_in_every_group_at_the_on_policy_point():
    # the constant part of the update: sum over a group's tokens of dL/dell_t is zero, t < T
    B, L = 8, 7
    ell_old, _, _, mask = _random_batch(B=B, L=L, seed=13)
    mask[3, 2:] = 0
    mask[6, 4:] = 0
    g = torch.Generator().manual_seed(13)
    seq_adv = torch.randn(B, generator=g)
    adv = seq_adv[:, None] * mask
    groups = _groups(B, size=4)
    w = plc.compute_loop_weights(ell_old, _uniform_logq(ell_old), 1.0, "is_trunc")
    mass = plc.sequence_weight_mass(w, mask, "seq-mean-token-mean")
    b = plc.group_loop_baseline(plc.sequence_advantages(adv, mask), mass, groups)
    loop_adv = (adv[..., None] - b[:, None, :]) * mask[..., None]
    ell_new = ell_old.clone().requires_grad_(True)
    loss, _ = plc.plc_policy_loss(ell_new, ell_old, adv, mask, w, lam=0.1, clip_ratio=0.2,
                                  objective="additive", loop_advantages=loop_adv,
                                  loop_agg_mode="seq-mean-token-mean")
    grad = torch.autograd.grad(loss, ell_new)[0]
    for k in range(2):
        rows = slice(4 * k, 4 * k + 4)
        assert torch.allclose(grad[rows, :, :-1].sum(dim=(0, 1)), torch.zeros(T - 1), atol=1e-7)
    # per token: -lam * w * A~ / (M * L_i) on the earlier loops, GRPO's -A / N on the final loop
    lengths = mask.sum(1, keepdim=True)
    expected_loop = -0.1 * w[..., :-1] * loop_adv[..., :-1] / (B * lengths[..., None])
    assert torch.allclose(grad[..., :-1], expected_loop, atol=1e-7)
    assert torch.allclose(grad[..., -1], -adv / mask.sum(), atol=1e-7)
    # without the baseline the mass is not zero
    loss_nb, _ = plc.plc_policy_loss(ell_new, ell_old, adv, mask, w, lam=0.1, clip_ratio=0.2,
                                     objective="additive", loop_agg_mode="seq-mean-token-mean")
    grad_nb = torch.autograd.grad(loss_nb, ell_new)[0]
    assert grad_nb[:4, :, :-1].sum(dim=(0, 1)).abs().max() > 1e-4


@pytest.mark.parametrize("values, match", [
    ({"weight_mode": "is_trunc"}, "objective=additive"),
    ({"weight_mode": "is_trunc", "objective": "additive", "prior": "exit"}, "prior uniform"),
    ({"weight_mode": "early_prior", "objective": "additive", "eps_floor": 0.1}, "eps_floor 0"),
    ({"weight_mode": "is_trunc", "objective": "additive", "use_is_correction": True},
     "use_is_correction"),
    ({"loop_baseline": "group", "loop_agg_mode": "seq-mean-token-sum"}, "loop_baseline=group"),
    ({"objective": "mix", "lam": 1.5}, "lam"),
    ({"objective": "sum"}, "objective"),
    ({"loop_agg_mode": "token-sum"}, "loop_agg_mode"),
])
def test_loop_aux_settings_are_validated(values, match):
    with pytest.raises(ValueError, match=match):
        plc.PLCSettings.from_mapping({"enabled": True, **values})
    ok = plc.PLCSettings.from_mapping({"enabled": True, "weight_mode": "is_trunc",
                                       "objective": "additive", "lam": 2.0,
                                       "loop_baseline": "group"})
    assert ok.lam == 2.0 and "objective=additive" in ok.describe()


@pytest.mark.parametrize("values, match", [
    ({"weight_mode": "is_trunc", "objective": "additive", "trunc_cap": 0.5}, "trunc_cap"),
    ({"weight_mode": "early_prior", "objective": "additive", "trunc_cap": 2.0}, "trunc_cap"),
    ({"weight_mode": "is_trunc", "objective": "additive", "kl_coef": -0.1}, "kl_coef"),
    ({"weight_mode": "posterior", "kl_coef": 0.2}, "kl_coef"),
])
def test_trust_region_settings_are_validated(values, match):
    with pytest.raises(ValueError, match=match):
        plc.PLCSettings.from_mapping({"enabled": True, **values})
    ok = plc.PLCSettings.from_mapping({"enabled": True, "weight_mode": "is_trunc",
                                       "objective": "additive", "trunc_cap": 2, "kl_coef": 0.2})
    assert ok.trunc_cap == 2.0 and "kl_coef=0.2" in ok.describe()


def test_trunc_cap_bounds_the_ratio_from_above_only():
    ell_old, _, _, _ = _random_batch(B=4, L=6, seed=20)
    ell_old[0, 0, 0] = ell_old[0, 0, -1] + 2.0          # ratio e^2 above the cap
    ell_old[0, 1, 0] = ell_old[0, 1, -1] - 5.0          # ratio e^-5, kept as is
    logq = _uniform_logq(ell_old)
    w = plc.compute_loop_weights(ell_old, logq, 1.0, "is_trunc", trunc_cap=2.0)
    ratio = torch.exp(ell_old[..., :-1] - ell_old[..., -1:])
    assert torch.allclose(w[..., :-1], ratio.clamp(max=2.0) / 3, atol=1e-6)
    assert w[0, 0, 0] == pytest.approx(2.0 / 3) and w[0, 1, 0] == pytest.approx(
        float(torch.exp(torch.tensor(-5.0))) / 3, rel=1e-5)
    assert torch.equal(plc.compute_loop_weights(ell_old, logq, 1.0, "is_trunc"),
                       plc.compute_loop_weights(ell_old, logq, 1.0, "is_trunc", trunc_cap=1.0))
    with pytest.raises(ValueError, match="trunc_cap"):
        plc.compute_loop_weights(ell_old, logq, 1.0, "is_trunc", trunc_cap=0.5)


@pytest.mark.parametrize("trunc_cap", [1.0, 2.0])
def test_trust_region_gradient_is_rho_a_plus_kl_coef_per_token(trunc_cap):
    # at the on-policy point (one update per step) the loop-t gradient per token is
    # -(lam / N) alpha_t (rho A + kl_coef); the final loop gets GRPO's -A / N only
    ell_old, _, adv, mask = _random_batch(B=5, L=6, seed=21)
    logq = _uniform_logq(ell_old)
    w = plc.compute_loop_weights(ell_old, logq, 1.0, "is_trunc", trunc_cap=trunc_cap)
    alpha = plc.compute_loop_weights(ell_old, logq, 1.0, "early_prior")
    ell_new = ell_old.clone().requires_grad_(True)
    lam, beta = 0.1, 0.2
    loss, metrics = plc.plc_policy_loss(ell_new, ell_old, adv, mask, w, lam=lam, clip_ratio=0.2,
                                        objective="additive", kl_coef=beta, kl_weights=alpha)
    grad = torch.autograd.grad(loss, ell_new)[0]
    n = mask.sum()
    rho = torch.exp(ell_old[..., :-1] - ell_old[..., -1:]).clamp(max=trunc_cap)
    expected = -lam * (alpha[..., :-1] * (rho * adv[..., None] + beta)) * mask[..., None] / n
    assert torch.allclose(grad[..., :-1], expected, atol=1e-7)
    assert torch.allclose(grad[..., -1], -adv * mask / n, atol=1e-7)
    # the KL part of the loss is lam * kl_coef * token-mean of sum_t alpha_t k1_t
    loss0, _ = plc.plc_policy_loss(ell_new, ell_old, adv, mask, w, lam=lam, clip_ratio=0.2,
                                   objective="additive")
    k1 = (alpha * (ell_old[..., -1:] - ell_old)).sum(-1)
    assert torch.allclose(loss - loss0, lam * beta * plc.agg_loss(k1, mask, "token-mean"),
                          atol=1e-6)
    assert len(metrics["plc/kl_k3_per_loop"]) == T and metrics["plc/kl_term_mean"] == \
        pytest.approx(float(plc.agg_loss(k1, mask, "token-mean")), abs=1e-6)
    assert all(v >= 0 for v in metrics["plc/kl_k3_per_loop"])


def test_trust_region_needs_additive_and_kl_weights():
    ell_old, ell_new, adv, mask = _random_batch(seed=22)
    w = plc.compute_loop_weights(ell_old, _uniform_logq(ell_old), 1.0, "is_trunc")
    with pytest.raises(ValueError, match="kl_weights"):
        plc.plc_policy_loss(ell_new, ell_old, adv, mask, w, lam=0.1, clip_ratio=0.2,
                            objective="additive", kl_coef=0.2)
    with pytest.raises(ValueError, match="additive"):
        plc.plc_policy_loss(ell_new, ell_old, adv, mask, w, lam=0.1, clip_ratio=0.2,
                            kl_coef=0.2, kl_weights=w)
    # kl_coef = 0 leaves the loss unchanged
    a, _ = plc.plc_policy_loss(ell_new, ell_old, adv, mask, w, lam=0.1, clip_ratio=0.2,
                               objective="additive")
    b, _ = plc.plc_policy_loss(ell_new, ell_old, adv, mask, w, lam=0.1, clip_ratio=0.2,
                               objective="additive", kl_coef=0.0, kl_weights=w)
    assert torch.equal(a, b)


def test_cross_loop_stats():
    ell_old = torch.log(torch.tensor([[[0.04, 0.6, 0.9, 0.5], [0.4, 0.4, 0.4, 0.4]]]))
    adv = torch.tensor([[2.0, 2.0]])
    mask = torch.ones(1, 2)
    s = plc.cross_loop_stats(ell_old, adv, mask, trunc_cap=1.5)
    # ratios against the last loop: token 1 = [0.08, 1.2, 1.8], token 2 = [1, 1, 1]
    assert s["plc/xloop_rho_bar_mean_per_loop"] == pytest.approx([0.54, 1.1, 1.25], rel=1e-5)
    assert s["plc/xloop_capped_frac_per_loop"] == pytest.approx([0.0, 0.0, 0.5])
    assert s["plc/xloop_rho_lt_0.1_frac_per_loop"] == pytest.approx([0.5, 0.0, 0.0])
    assert s["plc/xloop_pg_weight_abs_mean_per_loop"] == pytest.approx([1.08, 2.2, 2.5],
                                                                       rel=1e-5)


def test_loopaux_config_is_the_nogate_config_with_the_loop_aux_keys(env):
    nogate = OmegaConf.to_container(rcfg.load_config(str(NOGATE_CONFIG)))
    cfg = OmegaConf.to_container(rcfg.load_config(str(LOOPAUX_CONFIG)))
    for section, values in LOOPAUX_CHANGES.items():
        for key, value in values.items():
            assert cfg[section].pop(key) == value, (section, key)
            nogate[section].pop(key, None)
    for c in (cfg, nogate):
        for key in ("run_name", "tags"):
            c["wandb"].pop(key)
    assert cfg == nogate
    s = rcfg.plc_settings(rcfg.load_config(str(LOOPAUX_CONFIG)))
    assert s.enabled and not s.needs_exit_pdf and not s.use_gate_loss
    assert s.objective == "additive" and s.loop_baseline == "none" and s.lam == 0.1
    assert s.trunc_cap == 1.0 and s.kl_coef == 0.2
    gamma0 = rcfg.plc_settings(rcfg.load_config(str(LOOPAUX_CONFIG), ["plc.lam=0"]))
    assert gamma0.lam == 0.0
    # every ablation in the config header is a valid override
    for overrides in (["plc.weight_mode=early_prior"], ["plc.kl_coef=0"], ["plc.trunc_cap=2"],
                      ["plc.kl_coef=0", "plc.loop_baseline=group",
                       "plc.loop_agg_mode=seq-mean-token-mean"]):
        ablation = rcfg.plc_settings(rcfg.load_config(str(LOOPAUX_CONFIG), overrides))
        assert ablation.enabled


def test_loopaux_dry_run(env, capsys, monkeypatch):
    from rltt_baseline import launch

    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "PYTHONPATH"):
        monkeypatch.setenv(key, "")
    monkeypatch.setattr(sys, "path", list(sys.path))
    assert launch.main(["--config", str(LOOPAUX_CONFIG), "--dry-run"]) == 0
    resolved = json.loads(capsys.readouterr().out)
    actor = resolved["verl_overrides"]["actor_rollout_ref"]["actor"]
    assert actor["plc"]["objective"] == "additive" and actor["plc"]["loop_baseline"] == "none"
    assert actor["plc"]["loop_agg_mode"] == "same" and actor["plc"]["lam"] == 0.1
    assert actor["plc"]["trunc_cap"] == 1.0 and actor["plc"]["kl_coef"] == 0.2
    assert actor["ppo_mini_batch_size"] == 64 and actor["use_kl_loss"] is False
