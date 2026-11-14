"""CPU tests for ``rltt_baseline.loss_agg`` (row-normalized loss aggregation) and its config."""

import json
import sys

import pytest
import torch

from rltt_baseline import config as rcfg
from rltt_baseline import loss_agg, paths, plc
from rltt_baseline.loss_agg import LossAggSettings

ENV = {
    "LOOPRL_OURO_DIR": "/models/Ouro-1.4B-Thinking",
    "LOOPRL_MODEL_DIR": "/models/Ouro-1.4B-Thinking-RLTT",
    "LOOPRL_TRAIN_FILE": "/data/deepmath_level6.parquet",
    "LOOPRL_MATH500_JSONL": "/data/math500.jsonl",
    "LOOPRL_OUTPUT_DIR": "/out/loss_agg",
    "LOOPRL_ENV_FILE": "/nonexistent/env.local",  # never read a real env.local in tests
}
GRPO_CONFIGS = ("grpo_baseline_1p4b_8gpu.yaml", "grpo_baseline_8gpu.yaml")
PLC_CONFIGS = ("plc_grpo_loopaux_1p4b_8gpu.yaml", "plc_grpo_loopaux_nokl_1p4b_8gpu.yaml",
               "plc_grpo_loopkd_1p4b_8gpu.yaml", "plc_grpo_loopkd_rev_1p4b_8gpu.yaml",
               "plc_grpo_loopaux_nocap_refkl_1p4b_8gpu.yaml")
RLTT_CONFIGS = ("rltt_baseline_1p4b_8gpu.yaml", "rltt_baseline.yaml")
DRGRPO = LossAggSettings(mode="drgrpo", norm_len=16384)
SEQ = LossAggSettings(mode="seq-mean-token-mean")


@pytest.fixture
def env(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)


# ---------- settings ----------

def test_default_is_verl_token_mean():
    s = LossAggSettings.from_mapping(None)
    assert s == LossAggSettings() and s.mode == "token-mean" and not s.row_normalized


def test_drgrpo_and_seq_mean_settings():
    assert LossAggSettings.from_mapping({"mode": "drgrpo", "norm_len": 16384}) == DRGRPO
    assert LossAggSettings.from_mapping({"mode": "drgrpo", "norm_len": 16384.0}).norm_len == 16384
    assert LossAggSettings.from_mapping({"mode": "seq-mean-token-mean"}) == SEQ
    assert DRGRPO.row_normalized and SEQ.row_normalized
    assert DRGRPO.describe() == "loss_agg=drgrpo norm_len=16384"


@pytest.mark.parametrize("values,error,match", [
    ({"mode": "drgrpo"}, ValueError, "norm_len > 0"),
    ({"mode": "drgrpo", "norm_len": 0}, ValueError, "norm_len > 0"),
    ({"mode": "seq-mean-token-mean", "norm_len": 16384}, ValueError, "only applies"),
    ({"mode": "token-mean", "norm_len": 16384}, ValueError, "only applies"),
    ({"mode": "group-token-mean"}, ValueError, "must be one of"),
    ({"mode": "drgrpo", "norm_len": 1.5}, TypeError, "integer"),
    ({"mode": "drgrpo", "norm_len": True}, TypeError, "integer"),
    ({"mode": 3}, TypeError, "string"),
    ({"mode": "drgrpo", "norm_len": 16384, "alpha": 1}, KeyError, "unknown loss_agg keys"),
])
def test_bad_settings_are_refused(values, error, match):
    with pytest.raises(error, match=match):
        LossAggSettings.from_mapping(values)


# ---------- row divisors ----------

def _mask(lengths, width=None):
    width = width or max(lengths)
    return (torch.arange(width)[None, :] < torch.tensor(lengths)[:, None]).float()


def test_drgrpo_divisor_is_the_constant():
    d = loss_agg.row_divisors(DRGRPO, _mask([3, 1, 5]))
    assert d.tolist() == [16384.0] * 3


def test_seq_mean_divisor_is_the_response_length():
    assert loss_agg.row_divisors(SEQ, _mask([3, 1, 5, 0], width=6)).tolist() == [3, 1, 5, 1]
    with pytest.raises(ValueError, match="no row divisors"):
        loss_agg.row_divisors(LossAggSettings(), _mask([2]))


# ---------- independence of the micro-batch split (simulates verl + FSDP) ----------

P, G, R = 6, 4, 2  # prompts, rollouts per prompt, data-parallel ranks


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    lengths = torch.randint(1, 30, (P * G,), generator=g).tolist()
    return _mask(lengths, width=32)


def _update_grad(settings, mask, seed):
    """d(loss)/d(per-token loss) for one update: each rank gets a contiguous shard (its
    mini-batch), cuts it into randomly permuted micro-batches of random sizes, scales each by
    B / N like verl's dynamic batch; FSDP averages the ranks' gradients. Each micro-batch
    computes its divisors from its own rows, as the actors do."""
    g = torch.Generator().manual_seed(seed)
    loss_mat = torch.zeros_like(mask, requires_grad=True)
    total = 0.0
    shard = P * G // R
    for rank in range(R):
        rows = torch.arange(rank * shard, (rank + 1) * shard)
        m = mask[rows]
        order = torch.randperm(shard, generator=g)
        cuts = sorted(torch.randperm(shard - 1, generator=g)[:3].add(1).tolist())
        for idx in torch.tensor_split(order, cuts):
            lm, mm = loss_mat[rows][idx], m[idx]
            agg = (loss_agg.agg_rows(lm, mm, loss_agg.row_divisors(settings, mm))
                   if settings.row_normalized else plc.agg_loss(lm, mm, "token-mean"))
            total = total + agg * len(idx) / shard / R
    total.backward()
    return loss_mat.grad


def test_drgrpo_weights_every_token_by_one_over_rows_times_m():
    mask = _batch()
    expected = mask / (P * G * 16384)
    for seed in range(4):
        assert torch.allclose(_update_grad(DRGRPO, mask, seed), expected, rtol=1e-5, atol=0)


def test_seq_mean_weights_tokens_by_one_over_rows_times_own_length():
    mask = _batch(1)
    expected = mask / (P * G * mask.sum(-1, keepdim=True))
    for seed in range(4):
        assert torch.allclose(_update_grad(SEQ, mask, seed), expected, rtol=1e-5, atol=0)
    # the update minimises (1/(P G)) sum_i S_i / L_i: every response weighs 1/(P G) in total,
    # whatever its length (the original GRPO 1/G sum_i 1/|o_i| sum_t)
    per_response = _update_grad(SEQ, mask, 0).sum(-1)
    assert torch.allclose(per_response, torch.full((P * G,), 1.0 / (P * G)))


def test_seq_mean_equals_verl_seq_mean_token_mean_on_the_whole_batch():
    mask = _batch(3)
    x = torch.randn_like(mask)
    ours = loss_agg.agg_rows(x, mask, loss_agg.row_divisors(SEQ, mask))
    assert torch.allclose(ours, plc.agg_loss(x, mask, "seq-mean-token-mean"))


def test_verl_token_mean_depends_on_the_split():
    # the reason for the row-normalized modes: same batch, different micro-batches
    mask = _batch(2)
    a = _update_grad(LossAggSettings(), mask, 0)
    b = _update_grad(LossAggSettings(), mask, 1)
    assert not torch.allclose(a, b)


# ---------- loss functions ----------

def _plc_inputs(B=4, L=6, T=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    ell_old = -torch.rand(B, L, T, generator=g) * 4
    ell_new = (ell_old + 0.3 * torch.randn(B, L, T, generator=g)).requires_grad_()
    adv = torch.randn(B, generator=g)
    mask = _mask([6, 3, 5, 1], width=L)
    return ell_new, ell_old, adv, mask


def test_agg_term_without_divisors_is_verl_agg_loss():
    x = torch.randn(3, 5)
    m = _mask([5, 2, 4])
    for mode in plc.LOSS_AGG_MODES:
        assert torch.equal(plc.agg_term(x, m, mode, None), plc.agg_loss(x, m, mode))


def test_kd_loss_with_divisors():
    ell_new, ell_old, adv, mask = _plc_inputs()
    kd = torch.rand(*ell_new.shape[:2], 3, requires_grad=True)
    div = torch.full((4,), 16384.0)
    loss, _ = plc.plc_kd_loss(ell_new, ell_old, adv, mask, kd, 0.1, clip_ratio=0.2, row_div=div)
    ppo, _ = plc.ppo_objective(ell_new, ell_old, adv[:, None, None], 0.2)
    expected = (-loss_agg.agg_rows(ppo[..., -1], mask, div)
                + 0.1 * loss_agg.agg_rows(kd.sum(-1) / 3, mask, div))
    assert torch.allclose(loss, expected)
    assert torch.allclose(expected, (-(ppo[..., -1] * mask).sum() + 0.1 * (kd.sum(-1) / 3
                                     * mask).sum()) / (4 * 16384))


def test_policy_loss_with_divisors():
    ell_new, ell_old, adv, mask = _plc_inputs(seed=1)
    w = plc.compute_loop_weights(ell_old, "is_trunc", 1.0)
    kl_w = plc.compute_loop_weights(ell_old, "early_prior")
    div = torch.tensor([3.0, 3.0, 5.0, 5.0])
    loss, _ = plc.plc_policy_loss(ell_new, ell_old, adv, mask, w, 0.1, clip_ratio=0.2,
                                  kl_coef=0.2, kl_weights=kl_w, row_div=div)
    ppo, _ = plc.ppo_objective(ell_new, ell_old, adv[:, None, None], 0.2)
    k1 = ell_old[..., -1:] - ell_new
    expected = (-(loss_agg.agg_rows(ppo[..., -1], mask, div)
                  + 0.1 * loss_agg.agg_rows((w * ppo).sum(-1), mask, div))
                + 0.1 * 0.2 * loss_agg.agg_rows((kl_w * k1).sum(-1), mask, div))
    assert torch.allclose(loss, expected)


def test_grpo_policy_loss_is_verl_vanilla():
    ell_new, ell_old, adv, mask = _plc_inputs(seed=2)
    lp, old = ell_new[..., -1], ell_old[..., -1]
    adv2 = adv[:, None].expand_as(lp)
    loss, metrics = plc.grpo_policy_loss(lp, old, adv2, mask, clip_ratio=0.2)
    ppo, info = plc.ppo_objective(lp, old, adv2, 0.2)
    assert torch.equal(loss, -plc.agg_loss(ppo, mask, "token-mean"))
    assert set(metrics) == {"actor/pg_clipfrac", "actor/ppo_kl", "actor/pg_clipfrac_lower",
                            "loss_term/pg"}
    assert metrics["loss_term/pg"] == pytest.approx(float(loss))
    div = torch.tensor([4.0, 4.0, 2.0, 2.0])
    loss_div, _ = plc.grpo_policy_loss(lp, old, adv2, mask, clip_ratio=0.2, row_div=div)
    assert torch.allclose(loss_div, -loss_agg.agg_rows(ppo, mask, div))
    # the same element-wise term as PLC's final loop
    kd = torch.zeros(*ell_new.shape[:2], 3)
    plc_loss, _ = plc.plc_kd_loss(ell_new, ell_old, adv, mask, kd, 0.0, clip_ratio=0.2,
                                  row_div=div)
    assert torch.allclose(loss_div, plc_loss)


# ---------- logged loss (loss/<term>) ----------

def test_loss_terms_of_the_plc_losses_sum_to_the_loss():
    ell_new, ell_old, adv, mask = _plc_inputs(seed=4)
    div = loss_agg.row_divisors(DRGRPO, mask)
    w = plc.compute_loop_weights(ell_old, "is_trunc", 1.0)
    alpha = plc.compute_loop_weights(ell_old, "early_prior")
    logp_init = ell_old - 0.2
    loss, m = plc.plc_policy_loss(ell_new, ell_old, adv, mask, w, 0.1, clip_ratio=0.2,
                                  kl_coef=0.2, kl_weights=alpha, init_kl_coef=0.05,
                                  logp_init=logp_init, init_weights=alpha, row_div=div)
    terms = loss_agg.pop_loss_terms(m)
    assert set(terms) == {"pg", "loop", "k1", "init_anchor"} and not any(
        k.startswith(loss_agg.LOSS_TERM_PREFIX) for k in m)
    assert sum(terms.values()) == pytest.approx(float(loss), rel=1e-6)
    kd = torch.rand(*ell_new.shape[:2], 3)
    loss, m = plc.plc_kd_loss(ell_new, ell_old, adv, mask, kd, 0.1, clip_ratio=0.2,
                              init_kl_coef=0.05, logp_init=logp_init, row_div=div)
    terms = loss_agg.pop_loss_terms(m)
    assert set(terms) == {"pg", "kd", "init_anchor"}
    assert sum(terms.values()) == pytest.approx(float(loss), rel=1e-6)
    _, m = plc.plc_kd_loss(ell_new, ell_old, adv, mask, kd, 0.0, clip_ratio=0.2, row_div=div)
    assert set(loss_agg.pop_loss_terms(m)) == {"pg"}  # lam = 0: no kd term


def test_tracker_sums_the_scaled_terms_of_one_mini_batch():
    tracker = loss_agg.LossTracker()
    tracker.add({"pg": 1.0, "total": 3.0}, 0.25)
    tracker.add({"pg": -2.0, "total": 1.0}, 0.75)
    assert tracker.flush() == {"loss/pg": -1.25, "loss/total": 1.5}
    assert tracker.flush() == {}  # flush starts the next mini-batch


def _logged_total(settings, mask, x, seed):
    """loss/total as an actor logs it, averaged over ranks as verl does, for random splits."""
    g = torch.Generator().manual_seed(seed)
    shard = P * G // R
    per_rank = []
    for rank in range(R):
        rows = torch.arange(rank * shard, (rank + 1) * shard)
        order = torch.randperm(shard, generator=g)
        cuts = sorted(torch.randperm(shard - 1, generator=g)[:3].add(1).tolist())
        tracker = loss_agg.LossTracker()
        for idx in torch.tensor_split(order, cuts):
            xm, mm = x[rows][idx], mask[rows][idx]
            micro = loss_agg.agg_rows(xm, mm, loss_agg.row_divisors(settings, mm))
            tracker.add({"total": float(micro)}, len(idx) / shard)
        per_rank.append(tracker.flush()["loss/total"])
    return sum(per_rank) / R


def test_logged_total_is_the_objective_of_the_update():
    mask = _batch(5)
    x = torch.randn_like(mask)
    lengths = mask.sum(-1)
    expected = {DRGRPO: float((x * mask).sum() / (P * G * 16384)),
                SEQ: float(((x * mask).sum(-1) / lengths).mean())}
    for settings, value in expected.items():
        for seed in range(3):
            assert _logged_total(settings, mask, x, seed) == pytest.approx(value, rel=1e-5)


def test_rltt_loss_terms_use_the_vendor_scale():
    # three micro-batches of 10, 20 and 34 rows of a 64-row mini-batch
    out = loss_agg.rltt_loss_terms(pg=[1.0, 2.0, -1.0], kl=[0.5, 0.5, 0.5],
                                   kl_coef=[0.0, 0.0, 0.0], ent=[3.0, 3.0, 3.0],
                                   entropy_coeff=0.0, rows=[10, 20, 34], mini_batch_rows=64)
    assert out == pytest.approx({"loss/pg": (10 - 34 + 40) / 64, "loss/total": 16 / 64})
    out = loss_agg.rltt_loss_terms([1.0], [0.5], [0.1], [3.0], 0.01, [64], 64)
    assert out == pytest.approx({"loss/pg": 1.0, "loss/kl": 0.05, "loss/entropy": -0.03,
                                 "loss/total": 1.02})


# ---------- configs ----------

@pytest.mark.parametrize("name", GRPO_CONFIGS)
def test_grpo_configs_use_the_per_response_mean(env, name):
    cfg = rcfg.load_config(str(paths.config_file(name)))
    assert rcfg.loss_agg_settings(cfg) == SEQ
    overrides = rcfg.add_loss_agg_overrides(cfg, rcfg.verl_overrides(cfg))
    assert overrides["actor_rollout_ref"]["loss_agg"] == {"mode": "seq-mean-token-mean",
                                                          "norm_len": None}


@pytest.mark.parametrize("name", PLC_CONFIGS)
def test_plc_configs_use_drgrpo_with_m_16384(env, name):
    cfg = rcfg.load_config(str(paths.config_file(name)))
    assert rcfg.loss_agg_settings(cfg) == DRGRPO
    assert cfg.loss_agg.norm_len == cfg.args.max_completion_length == 16384
    overrides = rcfg.add_loss_agg_overrides(cfg, rcfg.verl_overrides(cfg))
    assert overrides["actor_rollout_ref"]["loss_agg"] == {"mode": "drgrpo", "norm_len": 16384}


@pytest.mark.parametrize("name", RLTT_CONFIGS)
def test_rltt_configs_keep_the_vendor_token_mean(env, name):
    cfg = rcfg.load_config(str(paths.config_file(name)))
    assert rcfg.loss_agg_settings(cfg) == LossAggSettings()
    overrides = rcfg.add_loss_agg_overrides(cfg, rcfg.verl_overrides(cfg))
    assert "loss_agg" not in (overrides.get("actor_rollout_ref") or {})
    with pytest.raises(ValueError, match="vendor RLTT loss"):
        bad = rcfg.load_config(str(paths.config_file(name)),
                               ["loss_agg.mode=drgrpo", "loss_agg.norm_len=16384"])
        rcfg.add_loss_agg_overrides(bad, rcfg.verl_overrides(bad))


def test_conflicting_verl_settings_are_refused(env):
    name = str(paths.config_file(GRPO_CONFIGS[0]))
    cfg = rcfg.load_config(name, ["verl_overrides.actor_rollout_ref.actor.loss_agg_mode="
                                  "seq-mean-token-mean"])
    with pytest.raises(ValueError, match="loss_agg_mode"):
        rcfg.add_loss_agg_overrides(cfg, rcfg.verl_overrides(cfg))
    # divisors depend on each response alone: row order and balancing do not matter
    cfg = rcfg.load_config(name, ["verl_overrides.trainer.balance_batch=true"])
    assert rcfg.add_loss_agg_overrides(cfg, rcfg.verl_overrides(cfg))


def test_unknown_loss_agg_key_is_refused(env):
    with pytest.raises(KeyError, match="unknown loss_agg keys"):
        rcfg.load_config(str(paths.config_file(PLC_CONFIGS[0])), ["loss_agg.alpha=1"])


@pytest.mark.parametrize("name,expected", [
    (GRPO_CONFIGS[0], {"mode": "seq-mean-token-mean", "norm_len": None}),
    ("plc_grpo_loopkd_1p4b_8gpu.yaml", {"mode": "drgrpo", "norm_len": 16384}),
])
def test_dry_run_passes_loss_agg_to_the_workers(env, capsys, monkeypatch, name, expected):
    from rltt_baseline import launch

    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "PYTHONPATH"):
        monkeypatch.setenv(key, "")
    monkeypatch.setattr(sys, "path", list(sys.path))
    assert launch.main(["--config", str(paths.config_file(name)), "--dry-run"]) == 0
    resolved = json.loads(capsys.readouterr().out)
    assert resolved["verl_overrides"]["actor_rollout_ref"]["loss_agg"] == expected
    assert resolved["config"]["loss_agg"] == {k: v for k, v in expected.items()
                                              if v is not None}
    # verl's own aggregation setting stays token-mean (also used for the entropy metric)
    actor = resolved["verl_overrides"]["actor_rollout_ref"]["actor"]
    assert actor.get("loss_agg_mode", "token-mean") == "token-mean"


def test_config_sections_are_documented():
    assert "loss_agg" in rcfg.SECTIONS and "loss_agg" in rcfg.DICT_SECTIONS
    assert "``loss_agg``" in rcfg.__doc__
