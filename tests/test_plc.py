"""PLC (GRPO + earlier-loop auxiliary term): pure-torch loss / weight tests and config plumbing."""

import json
import sys

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from rltt_baseline import config as rcfg
from rltt_baseline import paths, plc

LOOPAUX_CONFIG = paths.config_file("plc_grpo_loopaux_1p4b_8gpu.yaml")
LOOPAUX_NOKL_CONFIG = paths.config_file("plc_grpo_loopaux_nokl_1p4b_8gpu.yaml")
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


# ---------- weights ----------

def test_early_loop_weights_are_alpha_times_the_truncated_ratio():
    ell_old, _, _, _ = _random_batch(B=4, L=6, seed=10)
    early = plc.compute_loop_weights(ell_old, "early_prior")
    trunc = plc.compute_loop_weights(ell_old, "is_trunc")
    ratio = torch.exp(ell_old[..., :-1] - ell_old[..., -1:]).clamp(max=1.0)
    assert torch.allclose(early[..., :-1], torch.full_like(ratio, 1 / 3), atol=1e-7)
    assert torch.allclose(trunc[..., :-1], ratio / 3, atol=1e-6)
    assert (early[..., -1] == 0).all() and (trunc[..., -1] == 0).all()
    assert (trunc <= early + 1e-7).all() and not trunc.requires_grad


def test_trunc_cap_bounds_the_ratio_from_above_only():
    ell_old, _, _, _ = _random_batch(B=4, L=6, seed=20)
    ell_old[0, 0, 0] = ell_old[0, 0, -1] + 2.0          # ratio e^2 above the cap
    ell_old[0, 1, 0] = ell_old[0, 1, -1] - 5.0          # ratio e^-5, kept as is
    w = plc.compute_loop_weights(ell_old, "is_trunc", trunc_cap=2.0)
    ratio = torch.exp(ell_old[..., :-1] - ell_old[..., -1:])
    assert torch.allclose(w[..., :-1], ratio.clamp(max=2.0) / 3, atol=1e-6)
    assert w[0, 0, 0] == pytest.approx(2.0 / 3) and w[0, 1, 0] == pytest.approx(
        float(torch.exp(torch.tensor(-5.0))) / 3, rel=1e-5)
    assert torch.equal(plc.compute_loop_weights(ell_old, "is_trunc"),
                       plc.compute_loop_weights(ell_old, "is_trunc", trunc_cap=1.0))


def test_compute_loop_weights_rejects_bad_inputs():
    ell_old, _, _, _ = _random_batch()
    with pytest.raises(ValueError, match="trunc_cap"):
        plc.compute_loop_weights(ell_old, "is_trunc", trunc_cap=0.5)
    with pytest.raises(ValueError, match="weight_mode"):
        plc.compute_loop_weights(ell_old, "posterior")
    with pytest.raises(ValueError, match="two loops"):
        plc.compute_loop_weights(ell_old[..., -1:], "is_trunc")


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
    w = plc.compute_loop_weights(ell_old, "is_trunc")
    ell_new = ell_new.requires_grad_(True)
    loss, metrics = plc.plc_policy_loss(ell_new, ell_old, adv, mask, w, lam=0.0, clip_ratio=0.2,
                                        loss_agg_mode=loss_agg_mode)
    expected = _verl_vanilla_loss(ell_old[..., -1], ell_new[..., -1], adv, mask,
                                  loss_agg_mode=loss_agg_mode)
    assert torch.allclose(loss, expected, atol=1e-6)
    grad = torch.autograd.grad(loss, ell_new)[0]
    assert torch.all(grad[..., :-1] == 0)      # nothing reaches the earlier loops
    assert metrics["actor/pg_clipfrac"] > 0 and metrics["actor/pg_clipfrac_lower"] > 0


def test_loss_is_grpo_plus_lam_times_the_loop_term():
    ell_old, ell_new, adv, mask = _random_batch(seed=12)
    w = plc.compute_loop_weights(ell_old, "is_trunc")
    l0, _ = plc.plc_policy_loss(ell_new, ell_old, adv, mask, w, lam=0.0, clip_ratio=0.2)
    l1, _ = plc.plc_policy_loss(ell_new, ell_old, adv, mask, w, lam=1.0, clip_ratio=0.2)
    l3, metrics = plc.plc_policy_loss(ell_new, ell_old, adv, mask, w, lam=0.3, clip_ratio=0.2)
    assert torch.allclose(l3, l0 + 0.3 * (l1 - l0), atol=1e-6)
    assert len(metrics["plc/ratio_mean_per_loop"]) == T
    assert "plc/kl_term_mean" not in metrics           # kl_coef = 0: no trust-region metrics


@pytest.mark.parametrize("weight_mode,trunc_cap", [("is_trunc", 1.0), ("is_trunc", 2.0),
                                                   ("early_prior", 1.0)])
def test_trust_region_gradient_is_rho_a_plus_kl_coef_per_token(weight_mode, trunc_cap):
    # at the on-policy point (one update per step) the loop-t gradient per token is
    # -(lam / N) alpha_t (rho A + kl_coef); the final loop gets GRPO's -A / N only
    ell_old, _, adv, mask = _random_batch(B=5, L=6, seed=21)
    w = plc.compute_loop_weights(ell_old, weight_mode, trunc_cap=trunc_cap)
    alpha = plc.compute_loop_weights(ell_old, "early_prior")
    ell_new = ell_old.clone().requires_grad_(True)
    lam, beta = 0.1, 0.2
    loss, metrics = plc.plc_policy_loss(ell_new, ell_old, adv, mask, w, lam=lam, clip_ratio=0.2,
                                        kl_coef=beta, kl_weights=alpha)
    grad = torch.autograd.grad(loss, ell_new)[0]
    n = mask.sum()
    rho = torch.exp(ell_old[..., :-1] - ell_old[..., -1:]).clamp(max=trunc_cap)
    if weight_mode == "early_prior":
        rho = torch.ones_like(rho)
    expected = -lam * (alpha[..., :-1] * (rho * adv[..., None] + beta)) * mask[..., None] / n
    assert torch.allclose(grad[..., :-1], expected, atol=1e-7)
    assert torch.allclose(grad[..., -1], -adv * mask / n, atol=1e-7)
    # the KL part of the loss is lam * kl_coef * token-mean of sum_t alpha_t k1_t
    loss0, _ = plc.plc_policy_loss(ell_new, ell_old, adv, mask, w, lam=lam, clip_ratio=0.2)
    k1 = (alpha * (ell_old[..., -1:] - ell_old)).sum(-1)
    assert torch.allclose(loss - loss0, lam * beta * plc.agg_loss(k1, mask, "token-mean"),
                          atol=1e-6)
    assert len(metrics["plc/kl_k3_per_loop"]) == T and metrics["plc/kl_term_mean"] == \
        pytest.approx(float(plc.agg_loss(k1, mask, "token-mean")), abs=1e-6)
    assert all(v >= 0 for v in metrics["plc/kl_k3_per_loop"])
    assert metrics["plc/ratio_mean_per_loop"] == pytest.approx([1.0] * T)
    assert metrics["plc/clipfrac_per_loop"] == pytest.approx([0.0] * T)


def test_trust_region_needs_kl_weights_and_kl_coef_0_changes_nothing():
    ell_old, ell_new, adv, mask = _random_batch(seed=22)
    w = plc.compute_loop_weights(ell_old, "is_trunc")
    with pytest.raises(ValueError, match="kl_weights"):
        plc.plc_policy_loss(ell_new, ell_old, adv, mask, w, lam=0.1, clip_ratio=0.2, kl_coef=0.2)
    with pytest.raises(ValueError, match="kl_coef"):
        plc.plc_policy_loss(ell_new, ell_old, adv, mask, w, lam=0.1, clip_ratio=0.2,
                            kl_coef=-0.1, kl_weights=w)
    a, _ = plc.plc_policy_loss(ell_new, ell_old, adv, mask, w, lam=0.1, clip_ratio=0.2)
    b, _ = plc.plc_policy_loss(ell_new, ell_old, adv, mask, w, lam=0.1, clip_ratio=0.2,
                               kl_coef=0.0, kl_weights=w)
    assert torch.equal(a, b)


def test_policy_loss_rejects_bad_shapes():
    ell_old, ell_new, adv, mask = _random_batch()
    w = plc.compute_loop_weights(ell_old, "is_trunc")
    with pytest.raises(ValueError, match="B, L, T"):
        plc.plc_policy_loss(ell_new[..., -1], ell_old[..., -1], adv, mask, w, 0.1, 0.2)
    with pytest.raises(ValueError, match="loss_agg_mode"):
        plc.plc_policy_loss(ell_new, ell_old, adv, mask, w, 0.1, 0.2, loss_agg_mode="mean")


def test_agg_loss_modes_match_their_definitions():
    mat = torch.arange(6, dtype=torch.float32).view(2, 3)
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    assert plc.agg_loss(mat, mask, "token-mean").item() == pytest.approx((0 + 1 + 3 + 4 + 5) / 5)
    assert plc.agg_loss(mat, mask, "seq-mean-token-sum").item() == pytest.approx((1 + 12) / 2)
    assert plc.agg_loss(mat, mask, "seq-mean-token-mean").item() == pytest.approx((0.5 + 4) / 2)
    assert plc.agg_loss(mat, mask, "seq-mean-token-sum-norm").item() == pytest.approx(13 / 3)


# ---------- diagnostics ----------

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


def test_weight_stats_on_a_jump_token():
    probs = torch.tensor([0.05, 0.6, 0.62, 0.621]).view(1, 1, T)
    ell = probs.log()
    w = plc.compute_loop_weights(ell, "is_trunc")
    stats = plc.weight_stats(w, ell, torch.ones(1, 1))
    assert stats["plc/w_mean_per_loop"] == pytest.approx(w[0, 0].tolist(), abs=1e-6)
    assert stats["plc/w_mean_per_loop"][-1] == 0.0
    assert stats["plc/delta_neg_frac_per_transition"] == [0.0, 0.0, 0.0]
    assert stats["plc/delta_abs_lt_0.01_frac"] == pytest.approx(1 / 3)
    assert stats["plc/final_logp_mean"] == pytest.approx(float(ell[0, 0, -1]))


def test_flatten_loop_metrics_gives_one_scalar_key_per_loop():
    flat = plc.flatten_loop_metrics({
        "actor/pg_clipfrac": 0.5,
        "plc/kl_k1_per_loop": [0.3, 0.2, 0.1, 0.0],
        "plc/xloop_rho_bar_mean_per_loop": [0.4, 0.6, 0.8],      # earlier loops 1..T-1
        "plc/delta_mean_per_transition": [1.0, 2.0, 3.0],
    })
    assert flat == {
        "actor/pg_clipfrac": 0.5,
        "plc/kl_k1/loop_1": 0.3, "plc/kl_k1/loop_2": 0.2, "plc/kl_k1/loop_3": 0.1,
        "plc/kl_k1/loop_4": 0.0,
        "plc/xloop_rho_bar_mean/loop_1": 0.4, "plc/xloop_rho_bar_mean/loop_2": 0.6,
        "plc/xloop_rho_bar_mean/loop_3": 0.8,
        "plc/delta_mean/loop_1_to_2": 1.0, "plc/delta_mean/loop_2_to_3": 2.0,
        "plc/delta_mean/loop_3_to_4": 3.0,
    }
    with pytest.raises(ValueError, match="_per_loop"):
        plc.flatten_loop_metrics({"plc/something": [1.0, 2.0]})


def _verl_reduce(micro_batch_metrics):
    """verl 0.6.1: append_to_dict per micro-batch, then reduce_metrics (np.mean per key)."""
    collected = {}
    for m in micro_batch_metrics:
        for key, value in m.items():
            collected.setdefault(key, []).append(value)
    return {key: float(np.mean(v)) if "max" not in key and "min" not in key else v
            for key, v in collected.items()}


def test_every_update_metric_keeps_its_loops_through_verl_reduction():
    per_micro = []
    for seed in (30, 31):
        ell_old, ell_new, adv, mask = _random_batch(seed=seed)
        w = plc.compute_loop_weights(ell_old, "is_trunc")
        alpha = plc.compute_loop_weights(ell_old, "early_prior")
        _, metrics = plc.plc_policy_loss(ell_new, ell_old, adv, mask, w, lam=0.1,
                                         clip_ratio=0.2, kl_coef=0.2, kl_weights=alpha)
        metrics.update(plc.weight_stats(w, ell_old, mask))
        metrics.update(plc.cross_loop_stats(ell_old, adv, mask))
        per_micro.append(metrics)
    flat = [plc.flatten_loop_metrics(m) for m in per_micro]
    for m in flat:
        assert all(not isinstance(v, (list, tuple)) for v in m.values())
        assert not any("max" in k or "min" in k for k in m)   # reduce_metrics: mean for all
    reduced = _verl_reduce(flat)
    for t in range(T):
        want = np.mean([m["plc/kl_k1_per_loop"][t] for m in per_micro])
        assert reduced[f"plc/kl_k1/loop_{t + 1}"] == pytest.approx(want)
    for t in range(T - 1):
        want = np.mean([m["plc/xloop_rho_bar_mean_per_loop"][t] for m in per_micro])
        assert reduced[f"plc/xloop_rho_bar_mean/loop_{t + 1}"] == pytest.approx(want)
    # without flattening, verl would average the loops away into one number
    assert np.ndim(_verl_reduce(per_micro)["plc/kl_k1_per_loop"]) == 0


def test_low_var_kl_is_zero_at_equality_and_nonnegative():
    a = torch.randn(4, 5)
    assert torch.all(plc.low_var_kl(a, a) == 0)
    assert torch.all(plc.low_var_kl(a, a + torch.randn(4, 5)) >= 0)


# ---------- settings ----------

def test_settings_defaults_are_the_loopaux_values():
    s = plc.PLCSettings.from_mapping({})
    assert not s.enabled and s.weight_mode == "is_trunc" and s.lam == 0.1
    assert s.trunc_cap == 1.0 and s.kl_coef == 0.2 and s.log_weight_stats
    s = plc.PLCSettings.from_mapping({"enabled": True, "lam": 2, "trunc_cap": 2, "kl_coef": 0})
    assert isinstance(s.lam, float) and s.lam == 2.0 and s.trunc_cap == 2.0
    assert "kl_coef=0.0" in s.describe()


@pytest.mark.parametrize("values, error, match", [
    ({"gama": 1.0}, KeyError, "unknown plc keys"),
    ({"gamma": 1.0}, KeyError, "unknown plc keys"),           # removed with the posterior modes
    ({"use_gate_loss": True}, KeyError, "unknown plc keys"),  # removed with the gate loss
    ({"enabled": "yes"}, TypeError, "bool"),
    ({"lam": "0.1"}, TypeError, "number"),
    ({"weight_mode": "posterior"}, ValueError, "weight_mode"),
    ({"lam": -0.1}, ValueError, "lam"),
    ({"trunc_cap": 0.5}, ValueError, "trunc_cap"),
    ({"weight_mode": "early_prior", "trunc_cap": 2.0}, ValueError, "trunc_cap"),
    ({"kl_coef": -0.1}, ValueError, "kl_coef"),
])
def test_settings_are_validated(values, error, match):
    with pytest.raises(error, match=match):
        plc.PLCSettings.from_mapping({"enabled": True, **values})


# ---------- config plumbing ----------

@pytest.fixture
def env(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)


def _args(**over):
    base = {"use_rltt_model": True, "loop_weighting": "uniform", "rltt_model_path": "/m"}
    base.update(over)
    return OmegaConf.create(base)


def test_loopaux_config_resolves_to_the_loopaux_settings(env):
    cfg = rcfg.load_config(str(LOOPAUX_CONFIG))
    s = rcfg.plc_settings(cfg)
    assert s == plc.PLCSettings(enabled=True, weight_mode="is_trunc", lam=0.1, trunc_cap=1.0,
                                kl_coef=0.2, log_weight_stats=True)
    assert cfg.trainer == "rltt" and cfg.args.use_rltt_model
    assert cfg.args.gradient_accumulation_steps == 1
    assert cfg.model.num_hidden_layers == 24
    overrides = rcfg.add_plc_overrides(cfg, rcfg.verl_overrides(cfg), _args())
    actor = overrides["actor_rollout_ref"]["actor"]
    assert actor["plc"] == {"enabled": True, "weight_mode": "is_trunc", "lam": 0.1,
                            "trunc_cap": 1.0, "kl_coef": 0.2, "log_weight_stats": True}
    assert actor["ppo_mini_batch_size"] == 64 and overrides["data"]["train_batch_size"] == 64
    assert rcfg.wandb_env(cfg)["WANDB_TAGS"].startswith("plc,")
    # every ablation in the config header is a valid override
    for overrides in (["plc.kl_coef=0.05"], ["plc.kl_coef=1.0"], ["plc.kl_coef=0"],
                      ["plc.trunc_cap=2"], ["plc.weight_mode=early_prior"], ["plc.lam=0"]):
        assert rcfg.plc_settings(rcfg.load_config(str(LOOPAUX_CONFIG), overrides)).enabled
    with pytest.raises(KeyError, match="unknown plc keys"):
        rcfg.load_config(str(LOOPAUX_CONFIG), ["plc.loop_baseline=group"])


def test_plc_overrides_require_the_rltt_trainer_and_per_loop_model(env):
    cfg = rcfg.load_config(str(LOOPAUX_CONFIG))
    with pytest.raises(ValueError, match="use_rltt_model"):
        rcfg.add_plc_overrides(cfg, {}, _args(use_rltt_model=False))
    with pytest.raises(ValueError, match="learned"):
        rcfg.add_plc_overrides(cfg, {}, _args(loop_weighting="learned"))
    off = rcfg.load_config(str(LOOPAUX_CONFIG), ["plc.enabled=false"])
    assert rcfg.add_plc_overrides(off, {"x": 1}, _args(use_rltt_model=False)) == {"x": 1}
    grpo = rcfg.load_config(str(paths.config_file("grpo_baseline_1p4b_8gpu.yaml")),
                            ["plc.enabled=true"])
    with pytest.raises(ValueError, match="trainer=rltt"):
        rcfg.add_plc_overrides(grpo, {}, _args())


def test_other_configs_have_plc_off(env):
    for name in ("rltt_baseline.yaml", "rltt_baseline_1p4b_8gpu.yaml",
                 "grpo_baseline_1p4b_8gpu.yaml", "grpo_baseline_8gpu.yaml"):
        assert not rcfg.plc_settings(rcfg.load_config(str(paths.config_file(name)))).enabled


def test_loopaux_nokl_config_is_the_loopaux_config_without_the_kl_term(env):
    # saved at the user's request (2026-11-12): plc.kl_coef 0.2 -> 0, everything else the same
    loopaux = OmegaConf.to_container(rcfg.load_config(str(LOOPAUX_CONFIG)))
    cfg = OmegaConf.to_container(rcfg.load_config(str(LOOPAUX_NOKL_CONFIG)))
    assert loopaux["plc"].pop("kl_coef") == 0.2
    assert cfg["plc"].pop("kl_coef") == 0.0
    for c in (cfg, loopaux):
        for key in ("run_name", "tags"):
            c["wandb"].pop(key)
    assert cfg == loopaux
    s = rcfg.plc_settings(rcfg.load_config(str(LOOPAUX_NOKL_CONFIG)))
    assert s.enabled and s.kl_coef == 0.0 and s.weight_mode == "is_trunc" and s.lam == 0.1
    assert rcfg.plc_settings(rcfg.load_config(str(LOOPAUX_NOKL_CONFIG), ["plc.lam=0"])).enabled


def test_loopaux_dry_run(env, capsys, monkeypatch):
    from rltt_baseline import launch

    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "PYTHONPATH"):
        monkeypatch.setenv(key, "")
    monkeypatch.setattr(sys, "path", list(sys.path))
    assert launch.main(["--config", str(LOOPAUX_CONFIG), "--dry-run", "plc.trunc_cap=2"]) == 0
    resolved = json.loads(capsys.readouterr().out)
    actor = resolved["verl_overrides"]["actor_rollout_ref"]["actor"]
    assert actor["plc"] == {"enabled": True, "weight_mode": "is_trunc", "lam": 0.1,
                            "trunc_cap": 2.0, "kl_coef": 0.2, "log_weight_stats": True}
    assert actor["ppo_mini_batch_size"] == 64 and actor["use_kl_loss"] is False
    assert resolved["vendor_args"]["use_rltt_model"] is True
    assert resolved["vendor_args"]["rltt_model_path"] == ENV["LOOPRL_MODEL_DIR"]
    assert resolved["vendor_args"]["n_gpus"] == 8
