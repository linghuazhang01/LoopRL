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

@pytest.mark.parametrize("weight_mode", plc.WEIGHT_MODES)
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
