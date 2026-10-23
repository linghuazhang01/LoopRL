import json  # noqa: I001  (first-party grouping differs between the two checkouts)
import math
import shutil

import pytest
import torch

from rltt_baseline import paths, prepare_model
from rltt_baseline.diagnostics import stats
from rltt_baseline.diagnostics.losses import grpo_advantages, grpo_token_losses

# ---------- losses (no transformers needed) ----------


def test_grpo_advantages_match_verl_semantics():
    adv = grpo_advantages([1, 0, 0, 1, 1, 1, 0], ["a", "a", "a", "a", "b", "b", "c"])
    g = torch.tensor([1.0, 0, 0, 1])
    expected = ((g - g.mean()) / (g.std() + 1e-6)).tolist()
    assert adv[:4] == pytest.approx(expected)
    assert adv[4:6] == pytest.approx([0.0, 0.0])  # all-equal group
    assert adv[6] == pytest.approx(0.0 / (1.0 + 1e-6))  # singleton: mean 0, std 1


def test_grpo_loss_at_ratio_one_is_reinforce():
    logp = torch.tensor([-1.0, -2.0, -0.5], requires_grad=True)
    loss = grpo_token_losses(logp, logp.detach(), 1.5, ref_log_prob=logp.detach()).sum()
    loss.backward()
    assert loss.item() == pytest.approx(-1.5 * 3)
    torch.testing.assert_close(logp.grad, torch.full((3,), -1.5))


def test_grpo_loss_clips_like_verl():
    old = torch.zeros(4)
    logp = torch.tensor([0.5, -0.5, 0.5, 2.0])  # ratios e^0.5, e^-0.5, e^0.5, e^2
    pos = grpo_token_losses(logp, old, 1.0)
    neg = grpo_token_losses(logp, old, -1.0)
    ratio = logp.exp()
    assert pos[0].item() == pytest.approx(-1.2)  # clipped at 1 + 0.2
    assert pos[1].item() == pytest.approx(-ratio[1].item())  # min(r, clip) -> r
    assert neg[0].item() == pytest.approx(ratio[0].item())  # max(r, 1.2) for A < 0
    assert neg[3].item() == pytest.approx(3.0)  # dual clip at c = 3


# ---------- stats ----------


def _gram_of(a, b):
    return stats.gram(a + b)


def test_summarize_aligned_orthogonal_and_opposed():
    e = torch.eye(4)
    aligned = stats.summarize(_gram_of([e[0]] * 4, [e[0]] * 4), 4)
    assert aligned["rho"] == pytest.approx(1.0)
    assert aligned["cos_pair_mean"] == pytest.approx(1.0)
    ortho = stats.summarize(_gram_of(list(e), list(e)), 4)
    assert ortho["rho"] == pytest.approx(0.5)
    assert ortho["noise_same_loop_mean"] == pytest.approx(1.0)
    assert ortho["cross_loop_split_mean"] == pytest.approx(0.0)
    opposed = stats.summarize(_gram_of([e[0], -e[0], e[1], -e[1]], [e[0], -e[0], e[1], -e[1]]), 4)
    assert opposed["rho"] == pytest.approx(0.0)
    assert opposed["cos"][0][1] == pytest.approx(-1.0)


def test_summarize_uses_full_batch_sum_of_halves():
    torch.manual_seed(0)
    a = [torch.randn(10) for _ in range(4)]
    b = [torch.randn(10) for _ in range(4)]
    s = stats.summarize(_gram_of(a, b), 4)
    full = [x + y for x, y in zip(a, b)]
    assert s["norm"] == pytest.approx([float(f.norm()) for f in full])
    assert s["norm_total"] == pytest.approx(float(sum(full).norm()))
    assert s["noise_same_loop"][2] == pytest.approx(float(torch.nn.functional.cosine_similarity(
        a[2], b[2], dim=0)), abs=1e-6)


def test_aggregate_adds_gram_blocks():
    torch.manual_seed(1)
    p1 = [torch.randn(3) for _ in range(8)]
    p2 = [torch.randn(5) for _ in range(8)]
    grams = {"layers.0.self_attn.q_proj.weight": stats.gram(p1),
             "layers.0.mlp.up_proj.weight": stats.gram(p2)}
    agg = stats.aggregate(grams, 4)
    joined = [torch.cat([x, y]) for x, y in zip(p1, p2)]
    direct = stats.summarize(stats.gram(joined), 4)
    assert agg["all"]["rho"] == pytest.approx(direct["rho"])
    assert set(agg) >= {"all", "type:attention", "type:mlp", "layer:00", "layer:00:mlp"}


def test_hidden_grad_norms():
    out = stats.hidden_grad_norms([[1.0, 2.0, 4.0], [2.0, 2.0, 2.0]])
    assert out["mean"] == pytest.approx([1.5, 2.0, 3.0])
    assert out["ratio_to_last"] == pytest.approx([(0.25 + 1.0) / 2, (0.5 + 1.0) / 2, 1.0])


# ---------- tiny Ouro (needs transformers >= 4.56) ----------

transformers = pytest.importorskip("transformers")
NEW_TF = tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 56)
needs_tf = pytest.mark.skipif(not NEW_TF, reason="Ouro modeling needs transformers >= 4.56")
TINY = {"vocab_size": 64, "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
        "layer_types": ["full_attention"] * 2, "max_window_layers": 2,
        "max_position_embeddings": 256, "torch_dtype": "float32"}


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    vendor = paths.ouro_source_dir()
    d = tmp_path_factory.mktemp("tiny_ouro")
    (d / "modeling_ouro.py").write_text(
        prepare_model.compat_base_source((vendor / "modeling_ouro.py").read_text()))
    shutil.copy2(vendor / "configuration_ouro.py", d / "configuration_ouro.py")
    cfg = json.loads((vendor / "config.json").read_text())
    cfg.update(TINY)
    (d / "config.json").write_text(json.dumps(cfg))
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(
        transformers.AutoConfig.from_pretrained(d, trust_remote_code=True),
        trust_remote_code=True)
    with torch.no_grad():
        model.model.early_exit_gate.bias.fill_(-0.5)
    model.save_pretrained(d)
    (d / "modeling_ouro.py").write_text(
        prepare_model.compat_base_source((vendor / "modeling_ouro.py").read_text()))
    return d


def _load(d):
    return transformers.AutoModelForCausalLM.from_pretrained(
        d, trust_remote_code=True, torch_dtype=torch.float32).eval()


def _loss(model, kind, ids, prompt_len, adv=1.0):
    from rltt_baseline.diagnostics.losses import sequence_loss

    _, hiddens, gates = model.model(input_ids=ids, use_cache=False)
    return sequence_loss(kind, hiddens, gates, model.lm_head, prompt_len, ids, adv, 7)


@needs_tf
def test_untie_keeps_forward_and_restores(tiny_dir):
    from rltt_baseline.diagnostics.untie import LoopUntie

    model = _load(tiny_dir)
    ids = torch.randint(3, 64, (1, 12))
    with torch.no_grad():
        before = model(input_ids=ids, use_cache=False).logits
        untie = LoopUntie(model)
        mid = model(input_ids=ids, use_cache=False).logits
        untie.restore()
        after = model(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(before, mid)
    torch.testing.assert_close(before, after)
    assert all(p.requires_grad for p in model.parameters())


@needs_tf
@pytest.mark.parametrize("kind", ["grpo", "nll", "sft_mixture"])
@pytest.mark.parametrize("ckpt", [False, True])
def test_per_loop_grads_sum_to_shared_grad(tiny_dir, kind, ckpt):
    from rltt_baseline.diagnostics.untie import LoopUntie

    torch.manual_seed(2)
    ids = torch.randint(3, 64, (1, 14))
    ref = _load(tiny_dir)
    _loss(ref, kind, ids, 5, adv=-0.7).backward()
    shared = dict(ref.named_parameters())

    model = _load(tiny_dir)
    untie = LoopUntie(model, use_checkpoint=ckpt)
    _loss(model, kind, ids, 5, adv=-0.7).backward()
    checked = 0
    for name, per_loop in untie.named_loop_params():
        grads = [p.grad for p in per_loop]
        if all(g is None for g in grads):
            continue
        total = sum(g for g in grads if g is not None)
        target = shared[f"model.{name}"].grad
        torch.testing.assert_close(total, target, rtol=1e-4, atol=1e-6)
        checked += 1
    assert checked >= 2 * 9  # every decoder-layer tensor of both layers
    if kind == "sft_mixture":  # the gate only enters through the exit distribution
        assert untie.groups["early_exit_gate"][0]["weight"].grad is not None


@needs_tf
def test_probe_end_to_end(tiny_dir, tmp_path):
    from rltt_baseline.diagnostics import probe

    torch.manual_seed(3)
    rollouts = tmp_path / "r.jsonl"
    with open(rollouts, "w") as fh:
        for g in range(4):
            prompt = torch.randint(3, 64, (6,)).tolist()
            for k in range(4):
                resp = torch.randint(3, 64, (5 + k,)).tolist()
                fh.write(json.dumps({"uid": f"q{g}", "sample": k, "gold": "1",
                                     "prompt_ids": prompt, "response_ids": resp,
                                     "finish_reason": "stop", "reward": float(k % 2)}) + "\n")
    out = tmp_path / "probe"
    assert probe.main(["--model", str(tiny_dir), "--rollouts", str(rollouts), "--out", str(out),
                       "--batches", "1", "--groups-per-batch", "4", "--acc-device", "cpu",
                       "--dtype", "float32"]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["untie_forward_max_abs_diff"] == 0.0
    entry = summary["batches"][0]["losses"]
    assert set(entry) == {"grpo", "nll", "sft_mixture"}
    grpo = entry["grpo"]
    assert grpo["sequences"] == [8, 8]
    assert 0 < grpo["stats"]["all"]["rho"] <= 1 + 1e-9
    assert len(grpo["hidden_grad"]["mean"]) == 4
    assert not math.isnan(grpo["stats"]["type:mlp"]["cross_loop_split_mean"])
    assert (out / "grams.pt").is_file()


# ---------- equivalence with verl 0.6.1 (runs only where verl is installed) ----------


def test_grpo_matches_verl_core_algos():
    core = pytest.importorskip("verl.trainer.ppo.core_algos")
    import numpy as np
    from omegaconf import OmegaConf

    rewards = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    uids = ["a"] * 4 + ["b"] * 4
    mask = torch.ones(8, 5)
    token_rewards = torch.zeros(8, 5)
    token_rewards[:, -1] = torch.tensor(rewards)
    verl_adv, _ = core.compute_grpo_outcome_advantage(token_rewards, mask, np.array(uids))
    ours = grpo_advantages(rewards, uids)
    torch.testing.assert_close(verl_adv[:, 0], torch.tensor(ours))

    torch.manual_seed(4)
    old = torch.randn(8, 5) * 0.1
    logp = old + torch.randn(8, 5) * 0.4
    cfg = OmegaConf.create({"clip_ratio": 0.2, "clip_ratio_low": None, "clip_ratio_high": None,
                            "clip_ratio_c": 3.0})
    verl_loss, _ = core.compute_policy_loss_vanilla(old, logp, verl_adv, mask, "token-mean", cfg)
    ours_loss = torch.stack([grpo_token_losses(logp[i], old[i], ours[i]) for i in range(8)])
    assert ours_loss.mean().item() == pytest.approx(verl_loss.item(), rel=1e-5)

    ref = old - 0.05
    verl_kl = core.kl_penalty(logp, ref, "low_var_kl")
    ours_kl = torch.stack([grpo_token_losses(logp[i], logp[i], 0.0, ref_log_prob=ref[i],
                                             kl_coef=1.0) for i in range(8)])
    torch.testing.assert_close(ours_kl, verl_kl)
