"""Per-loop distribution audit on a tiny looped model (CPU, no verl)."""

import json  # noqa: I001  (first-party grouping differs between the two checkouts)
import sys
import types

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from rltt_baseline import config as rcfg
from rltt_baseline import loop_audit as la
from rltt_baseline import metrics as rmet
from rltt_baseline import paths
from rltt_baseline import plc
from rltt_baseline.ouro_rltt.rltt_per_loop import exit_pdf_from_gates

VOCAB, WIDTH, LOOPS = 17, 8, 4
ENV = {
    "LOOPRL_OURO_DIR": "/models/Ouro-2.6B-Thinking",
    "LOOPRL_MODEL_DIR": "/models/Ouro-2.6B-Thinking-RLTT",
    "LOOPRL_TRAIN_FILE": "/data/deepmath_level6.parquet",
    "LOOPRL_MATH500_JSONL": "/data/math500.jsonl",
    "LOOPRL_OUTPUT_DIR": "/out/run",
    "LOOPRL_ENV_FILE": "/nonexistent/env.local",  # never read a real env.local in tests
}


class OuroModel(torch.nn.Module):  # the audit finds the model by this class name
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, WIDTH)
        self.loops = torch.nn.ModuleList(torch.nn.Linear(WIDTH, WIDTH) for _ in range(LOOPS))
        self.early_exit_gate = torch.nn.Linear(WIDTH, 1)

    def forward(self, input_ids):
        hidden = self.embed(input_ids)
        hiddens, gates = [], []
        for layer in self.loops:
            hidden = torch.tanh(layer(hidden)) + hidden
            hiddens.append(hidden)
            gates.append(self.early_exit_gate(hidden))
        return None, hiddens, gates


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.model = OuroModel()
        self.lm_head = torch.nn.Linear(WIDTH, VOCAB, bias=False)

    def forward(self, input_ids):
        return self.lm_head(self.model(input_ids)[1][-1])


class FakeActor:
    """The part of verl's ``DataParallelPPOActor`` the audit relies on, rows in reverse order."""

    def __init__(self, remove_padding=True):
        self.actor_module = TinyLM()
        self.remove_padding = remove_padding

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False):
        input_ids, mask = micro_batch["input_ids"], micro_batch["attention_mask"].bool()
        response_length = micro_batch["responses"].size(-1)
        if self.remove_padding:
            packed = input_ids[mask]
            logits = self.actor_module(packed.unsqueeze(0)).squeeze(0) / temperature
            picked = F.log_softmax(logits, -1).gather(-1, packed.roll(-1).unsqueeze(-1))
            full = torch.zeros(mask.shape)
            full[mask] = picked.squeeze(-1)
        else:
            logits = self.actor_module(input_ids) / temperature
            full = F.log_softmax(logits, -1).gather(-1, input_ids.roll(-1, 1).unsqueeze(-1))
            full = full.squeeze(-1)
        return None, full[:, -response_length - 1:-1]

    def compute_log_prob(self, data, calculate_entropy=False):
        rows = torch.arange(len(data.batch["input_ids"])).flip(0)
        log_probs = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        with torch.no_grad():
            for part in rows.split(2):
                micro_batch = {name: value[part] for name, value in data.batch.items()}
                _, picked = self._forward_micro_batch(
                    micro_batch, temperature=data.meta_info["temperature"],
                    calculate_entropy=calculate_entropy)
                log_probs[part] = picked
        return log_probs, None


class RLTTStyleActor(FakeActor):
    def compute_log_prob(self, data, calculate_entropy=False):
        return FakeActor.compute_log_prob(self, data, calculate_entropy)

    def _forward_micro_batch_rltt(self, micro_batch, temperature):
        return self._forward_micro_batch(micro_batch, temperature)


class PLCStyleActor(FakeActor):
    """``PLCActor.compute_log_prob``: a padded per-loop forward through _forward_micro_batch_plc.

    Keeps the per-loop log-probs and exit PDF it returned, as ``PLCActor`` hands them on to
    ``update_policy``.
    """

    def __init__(self, settings=None):
        super().__init__(remove_padding=False)
        self.settings = settings or plc.PLCSettings(enabled=True, prior="exit")
        self.old = []

    def _forward_micro_batch_plc(self, micro_batch, temperature):
        # calls the model directly, as PLCActor does (not through _forward_micro_batch)
        input_ids = micro_batch["input_ids"]
        window = slice(-micro_batch["responses"].size(-1) - 1, -1)
        _, hiddens, gates = self.actor_module.model(input_ids)
        labels = input_ids.roll(-1, 1).unsqueeze(-1)
        ell = torch.stack([F.log_softmax(self.actor_module.lm_head(h) / temperature, -1)
                           .gather(-1, labels).squeeze(-1)[:, window] for h in hiddens], -1)
        exit_pdf = exit_pdf_from_gates(gates)[:, window] if self.settings.needs_exit_pdf else None
        return None, ell, exit_pdf

    def compute_log_prob(self, data, calculate_entropy=False):
        rows = torch.arange(len(data.batch["input_ids"])).flip(0)
        log_probs = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        self.old = []
        with torch.no_grad():
            for part in rows.split(2):
                micro_batch = {name: value[part] for name, value in data.batch.items()}
                _, ell, exit_pdf = self._forward_micro_batch_plc(micro_batch,
                                                                 data.meta_info["temperature"])
                log_probs[part] = ell[..., -1]
                self.old.append((micro_batch, ell, exit_pdf))
        return log_probs, None


def make_data(rows=6, prompt=4, response=5, temperature=0.7, **meta):
    """Left-padded prompts, right-padded responses of 1 to 5 tokens (rows 4 and 5 differ)."""
    generator = torch.Generator().manual_seed(rows)
    input_ids = torch.randint(1, VOCAB, (rows, prompt + response), generator=generator)
    attention_mask = torch.ones_like(input_ids)
    for row in range(rows):
        attention_mask[row, :row % prompt] = 0
        attention_mask[row, prompt + 1 + row % response:] = 0
    input_ids[attention_mask == 0] = 0
    batch = {"input_ids": input_ids, "attention_mask": attention_mask,
             "responses": input_ids[:, -response:]}
    return types.SimpleNamespace(meta_info={"temperature": temperature, **meta}, batch=batch)


def driver_batch(data, log_probs):
    response_mask = data.batch["attention_mask"][:, -data.batch["responses"].size(-1):]
    scores = torch.zeros_like(response_mask, dtype=torch.float32)
    scores[:, 0] = (torch.arange(len(scores)) % 2).float()
    return types.SimpleNamespace(
        batch={**data.batch, "old_log_probs": log_probs, "token_level_scores": scores},
        non_tensor_batch={"uid": [f"q{row // 2}" for row in range(len(scores))]})


def settings(tmp_path, **changes):
    return {"dir": str(tmp_path), "every_n_steps": 1, "top_k": 3, "save_tokens": True,
            "chunk_tokens": 4, **changes}


def run_step(actor, tmp_path, save_tokens=True, **meta):
    data = make_data(**meta)
    log_probs, _ = actor.compute_log_prob(data=data, calculate_entropy=True)
    batch = driver_batch(data, log_probs)
    return data, la.finalize_pending(tmp_path, batch, save_tokens=save_tokens)


def responses_of(step_dir):
    return [json.loads(line) for line in (step_dir / "responses.jsonl").read_text().splitlines()]


# ---------- positions and distributions ----------

@pytest.mark.parametrize("packed", [False, True])
def test_response_positions_find_the_predicting_token(packed):
    data = make_data()
    input_ids, mask = data.batch["input_ids"], data.batch["attention_mask"]
    response_length = data.batch["responses"].size(-1)
    output = input_ids[mask.bool()] if packed else input_ids.flatten()
    rows, tokens, positions = la.response_positions(mask, response_length, output.numel())
    predicting = input_ids.size(1) - response_length - 1 + tokens
    assert torch.equal(output[positions], input_ids[rows, predicting])
    assert torch.equal(rows.bincount(), mask[:, -response_length:].sum(-1))


def test_response_positions_refuses_an_unknown_output_shape():
    data = make_data()
    with pytest.raises(ValueError, match="positions"):
        la.response_positions(data.batch["attention_mask"], 5, 7)


def _loop_inputs(n=9):
    torch.manual_seed(3)
    return [torch.randn(n, WIDTH) for _ in range(LOOPS)], torch.nn.Linear(WIDTH, VOCAB, bias=False)


@pytest.mark.parametrize("chunk", [1, 4, 100])
def test_distribution_stats_match_a_direct_computation(chunk):
    hiddens, head = _loop_inputs()
    labels = torch.arange(9) % VOCAB
    out = la.distribution_stats(hiddens, head, labels, temperature=0.7, top_k=3,
                                chunk_tokens=chunk)
    log_p = torch.stack([F.log_softmax(head(h) / 0.7, -1) for h in hiddens], 1).detach()
    p = log_p.exp()
    assert out["kl"].shape == (9, LOOPS, LOOPS)
    expected = (p[:, 0] * (log_p[:, 0] - log_p[:, 2])).sum(-1)
    torch.testing.assert_close(out["kl"][:, 0, 2], expected, atol=1e-5, rtol=1e-5)
    assert not torch.isclose(out["kl"][:, 0, 2], out["kl"][:, 2, 0]).all()  # KL is not symmetric
    assert torch.all(out["kl"][:, range(LOOPS), range(LOOPS)] == 0)
    torch.testing.assert_close(out["entropy"], -(p * log_p).sum(-1), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out["sampled_log_prob"][:, 1], log_p[:, 1][range(9), labels])
    values, ids = p.topk(3, dim=-1)
    assert torch.equal(out["topk_ids"], ids)
    torch.testing.assert_close(out["topk_probs"], values)


def test_distribution_stats_keep_float32_under_bf16_autocast():
    hiddens, head = _loop_inputs()
    head = head.bfloat16()
    hiddens = [h.bfloat16() for h in hiddens]
    labels = torch.zeros(9, dtype=torch.long)
    plain = la.distribution_stats(hiddens, head, labels)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        cast = la.distribution_stats(hiddens, head, labels)
    assert cast["kl"].dtype == torch.float32
    torch.testing.assert_close(cast["kl"], plain["kl"], atol=2e-3, rtol=2e-3)


def test_kl_of_identical_loops_is_zero():
    hiddens, head = _loop_inputs()
    out = la.distribution_stats([hiddens[0]] * LOOPS, head, torch.zeros(9, dtype=torch.long))
    assert torch.all(out["kl"] == 0)


def test_token_metrics_names_and_overlap():
    arrays = {
        "entropy": np.array([[1.0, 2.0]]), "sampled_log_prob": np.array([[-1.0, -2.0]]),
        "kl": np.array([[[0.0, 0.5], [0.25, 0.0]]]),
        "topk_ids": np.array([[[1, 2, 3], [1, 3, 9]]]),
        "topk_probs": np.array([[[0.5, 0.25, 0.125], [0.75, 0.125, 0.0625]]]),
    }
    columns = la.token_metrics(arrays)
    assert columns["kl/1_2"][0] == 0.5 and columns["kl/2_1"][0] == 0.25
    assert columns["top1_agree/1_2"][0] == 1.0
    assert columns["topk_overlap/1_2"][0] == pytest.approx(2 / 3)
    assert columns["topk_mass/loop_1"][0] == pytest.approx(0.875)
    assert columns["top1_prob/loop_2"][0] == 0.75
    assert "kl/1_1" not in columns
    # cross-loop ratio of the sampled token, loop 1 against the last loop: rho = e^(-1 + 2) = e
    assert columns["xloop_log_rho/loop_1"][0] == pytest.approx(1.0)
    assert columns["xloop_rho_min1/loop_1"][0] == pytest.approx(1.0)
    assert columns["xloop_rho_gt_1/loop_1"][0] == 1.0
    assert columns["xloop_rho_lt_0.1/loop_1"][0] == 0.0
    assert "xloop_log_rho/loop_2" not in columns


# ---------- worker and driver ----------

@pytest.mark.parametrize("remove_padding", [False, True])
def test_audit_writes_step_files(tmp_path, remove_padding):
    actor = FakeActor(remove_padding)
    la.attach_loop_audit(actor, settings(tmp_path), rank=3)
    data, metrics = run_step(actor, tmp_path)
    assert not actor.actor_module.model._forward_hooks

    step_dir = tmp_path / la.step_dir_name(1)
    assert sorted(p.name for p in step_dir.iterdir()) == ["responses.jsonl", "summary.json",
                                                          "tokens_rank03.npz"]
    summary = json.loads((step_dir / "summary.json").read_text())
    mask = data.batch["attention_mask"][:, -5:]
    assert summary["responses"] == 6 and summary["tokens"] == int(mask.sum())
    assert summary["temperature"] == pytest.approx(0.7)
    assert summary["unmatched_responses"] == 0 and summary["errors"] == []
    assert summary["last_loop_log_prob_gap"] < 1e-6  # tokens and distributions are aligned
    assert metrics["loop_audit/last_loop_log_prob_gap"] < 1e-6
    assert metrics["loop_audit/kl/1_2/mean"] == summary["metrics"]["kl/1_2"]["mean"]

    with np.load(step_dir / "tokens_rank03.npz") as shard:
        pooled = shard["kl"][:, 0, 3]
        assert shard["topk_ids"].dtype == np.uint8 and shard["topk_ids"].shape[1:] == (LOOPS, 3)
    stats = summary["metrics"]["kl/1_4"]
    assert stats["mean"] == pytest.approx(float(pooled.mean()))
    assert stats["p5"] == pytest.approx(float(np.percentile(pooled, 5)))
    assert stats["p95"] == pytest.approx(float(np.percentile(pooled, 95)))

    lines = responses_of(step_dir)
    assert [line["index"] for line in lines] == list(range(6)) and lines[0]["rank"] == 3
    by_uid = sorted((line["uid"], line["reward"], line["truncated"]) for line in lines)
    assert by_uid == [("q0", 0.0, False), ("q0", 1.0, False), ("q1", 0.0, False),
                      ("q1", 1.0, False), ("q2", 0.0, True), ("q2", 1.0, False)]
    for line in lines:
        token = la.read_response(step_dir, 3, line["index"])
        assert len(token["kl"]) == line["response_tokens"]
        assert line["stats"]["kl/1_4"]["mean"] == pytest.approx(float(token["kl"][:, 0, 3].mean()))
        assert line["stats"]["entropy/loop_2"]["p95"] == pytest.approx(
            float(np.percentile(token["entropy"][:, 1], 95)))
        assert line["correct"] == (line["reward"] >= 0.5)


def test_loop_weights_are_off_by_default(tmp_path):
    actor = PLCStyleActor()
    la.attach_loop_audit(actor, settings(tmp_path), rank=0)
    _, metrics = run_step(actor, tmp_path)
    with np.load(tmp_path / "step_000001" / "tokens_rank00.npz") as shard:
        assert not set(la.WEIGHT_ARRAYS) & set(shard.files)
    assert not any("exit_pdf" in name or "plc_weight" in name for name in metrics)


@pytest.mark.parametrize("remove_padding", [False, True])
def test_loop_weights_record_the_exit_pdf(tmp_path, remove_padding):
    actor = FakeActor(remove_padding)
    la.attach_loop_audit(actor, settings(tmp_path, loop_weights=True), rank=0)
    data, metrics = run_step(actor, tmp_path)
    step_dir = tmp_path / "step_000001"
    summary = json.loads((step_dir / "summary.json").read_text())
    assert summary["errors"] == []
    with np.load(step_dir / "tokens_rank00.npz") as shard:
        assert "plc_weight" not in shard.files  # not a PLC actor
        pdf = shard["exit_pdf"]
    np.testing.assert_allclose(pdf.sum(-1), 1.0, rtol=1e-6)

    with torch.no_grad():
        _, _, gates = actor.actor_module.model(data.batch["input_ids"])
    expected = exit_pdf_from_gates(gates)[:, -6:-1]  # positions predicting the 5 response tokens
    for line in responses_of(step_dir):
        token = la.read_response(step_dir, 0, line["index"])
        row = next(r for r in range(6) if np.array_equal(
            token["sampled_token"], data.batch["responses"][r, :line["response_tokens"]].numpy()))
        np.testing.assert_allclose(token["exit_pdf"],
                                   expected[row, :line["response_tokens"]].numpy(), rtol=1e-5)
    steps = np.arange(1, LOOPS + 1)
    assert summary["metrics"]["exit_pdf/expected_loop"]["mean"] == pytest.approx(
        float((pdf @ steps).mean()))
    assert metrics["loop_audit/exit_pdf/loop_1/mean"] == pytest.approx(float(pdf[:, 0].mean()))


@pytest.mark.parametrize("plc_settings", [
    plc.PLCSettings(enabled=True, prior="exit", use_gate_loss=True),
    plc.PLCSettings(enabled=True, prior="uniform", gamma=0.5, eps_floor=0.1),
    plc.PLCSettings(enabled=True, weight_mode="is_trunc", objective="additive", lam=0.1,
                    trunc_cap=2.0, kl_coef=0.2),
])
def test_loop_weights_record_the_plc_weights_of_the_update(tmp_path, plc_settings):
    actor = PLCStyleActor(plc_settings)
    la.attach_loop_audit(actor, settings(tmp_path, loop_weights=True), rank=0)
    run_step(actor, tmp_path)
    step_dir = tmp_path / "step_000001"
    with np.load(step_dir / "tokens_rank00.npz") as shard:
        recorded, pdf = shard["plc_weight"], shard["exit_pdf"]
        sampled_log_prob = shard["sampled_log_prob"]

    # the weights PLCActor.update_policy forms from the old tensors, in the audit's token order
    expected = []
    for micro_batch, ell, exit_pdf in actor.old:
        logq = plc.loop_prior_logq(plc_settings.prior, LOOPS, plc_settings.progressive_alpha,
                                   exit_pdf, like=ell)
        weights = plc.compute_loop_weights(ell, logq, plc_settings.gamma,
                                           plc_settings.weight_mode, plc_settings.eps_floor,
                                           plc_settings.trunc_cap)
        mask = micro_batch["attention_mask"][:, -5:].bool()
        expected.append(weights[mask].numpy())
    np.testing.assert_allclose(recorded, np.concatenate(expected), rtol=1e-6)
    if plc_settings.weight_mode in plc.EARLY_LOOP_WEIGHT_MODES:
        rho = np.exp(sampled_log_prob[:, :-1] - sampled_log_prob[:, -1:])
        np.testing.assert_allclose(recorded[:, :-1], np.minimum(rho, 2.0) / 3, rtol=1e-4)
        assert (recorded[:, -1] == 0).all()
    else:
        np.testing.assert_allclose(recorded.sum(-1), 1.0, rtol=1e-6)
    if plc_settings.prior == "exit":  # gamma = 1: the posterior of the audited exit PDF
        posterior = pdf * np.exp(sampled_log_prob)
        np.testing.assert_allclose(recorded, posterior / posterior.sum(-1, keepdims=True),
                                   atol=1e-5)
    summary = json.loads((step_dir / "summary.json").read_text())
    assert summary["errors"] == []
    assert "plc_weight/loop_4" in summary["metrics"]
    assert "plc_weight/expected_loop" in responses_of(step_dir)[0]["stats"]


def test_audit_measures_the_plc_log_prob_pass(tmp_path):
    actor = PLCStyleActor()
    la.attach_loop_audit(actor, settings(tmp_path), rank=0)
    data, metrics = run_step(actor, tmp_path)
    summary = json.loads((tmp_path / la.step_dir_name(1) / "summary.json").read_text())
    assert summary["errors"] == [] and summary["responses"] == 6
    assert summary["tokens"] == int(data.batch["attention_mask"][:, -5:].sum())
    assert summary["last_loop_log_prob_gap"] < 1e-6
    assert metrics["loop_audit/last_loop_log_prob_gap"] < 1e-6


def test_padded_and_packed_forwards_agree(tmp_path):
    for name, remove_padding in (("padded", False), ("packed", True)):
        actor = FakeActor(remove_padding)
        la.attach_loop_audit(actor, settings(tmp_path / name), rank=0)
        run_step(actor, tmp_path / name)
    with np.load(tmp_path / "padded" / "step_000001" / "tokens_rank00.npz") as padded, \
            np.load(tmp_path / "packed" / "step_000001" / "tokens_rank00.npz") as packed:
        assert np.allclose(padded["kl"], packed["kl"], atol=1e-6)
        assert np.array_equal(padded["sampled_token"], packed["sampled_token"])


def test_read_response_returns_tokens(tmp_path):
    actor = FakeActor()
    la.attach_loop_audit(actor, settings(tmp_path), rank=0)
    data, _ = run_step(actor, tmp_path)
    tokenizer = types.SimpleNamespace(convert_ids_to_tokens=lambda ids: [f"t{i}" for i in ids])
    step_dir = tmp_path / "step_000001"
    line = max(responses_of(step_dir), key=lambda row: row["response_tokens"])
    token = la.read_response(step_dir, 0, line["index"], tokenizer)
    assert len(token["topk_tokens"]) == line["response_tokens"]
    assert len(token["topk_tokens"][0]) == LOOPS and len(token["topk_tokens"][0][0]) == 3
    responses = data.batch["responses"]
    assert any(np.array_equal(token["sampled_token"], row[:line["response_tokens"]].numpy())
               for row in responses)


def test_every_n_steps_audits_steps_one_and_one_plus_n(tmp_path):
    actor = FakeActor()
    la.attach_loop_audit(actor, settings(tmp_path, every_n_steps=2), rank=0)
    for _ in range(3):
        run_step(actor, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["step_000001", "step_000003"]


def test_trainer_step_wins_over_the_call_count(tmp_path):
    actor = FakeActor()
    la.attach_loop_audit(actor, settings(tmp_path), rank=0)
    run_step(actor, tmp_path, global_steps=7)
    assert [p.name for p in tmp_path.iterdir()] == ["step_000007"]


def test_save_tokens_false_keeps_only_the_summaries(tmp_path):
    actor = FakeActor()
    la.attach_loop_audit(actor, settings(tmp_path, save_tokens=False), rank=0)
    _, metrics = run_step(actor, tmp_path, save_tokens=False)
    assert sorted(p.name for p in (tmp_path / "step_000001").iterdir()) == ["responses.jsonl",
                                                                            "summary.json"]
    assert metrics["loop_audit/tokens"] > 0


def test_forwards_outside_compute_log_prob_are_not_measured(tmp_path):
    actor = RLTTStyleActor()
    audit = la.attach_loop_audit(actor, settings(tmp_path), rank=0)
    data = make_data()
    actor._forward_micro_batch_rltt(data.batch, 1.0)  # update_policy uses this path
    assert audit._micro_batch is None and audit._records == []
    assert not list(tmp_path.iterdir())


def test_a_failed_audit_does_not_stop_training(tmp_path):
    actor = FakeActor()
    la.attach_loop_audit(actor, settings(tmp_path, top_k=VOCAB + 1), rank=2)  # topk raises
    data, metrics = run_step(actor, tmp_path)
    step_dir = tmp_path / "step_000001"
    summary = json.loads((step_dir / "summary.json").read_text())
    assert summary["errors"] == ["error_rank02.txt"] and summary["responses"] == 0
    assert metrics["loop_audit/error_ranks"] == 1
    assert "selected index k out of range" in (step_dir / "error_rank02.txt").read_text()


def test_ranks_are_pooled_and_unknown_rows_are_counted(tmp_path):
    for rank, rows in ((0, 6), (1, 7)):
        actor = FakeActor()
        la.attach_loop_audit(actor, settings(tmp_path), rank=rank)
        data = make_data(rows=rows)
        log_probs, _ = actor.compute_log_prob(data=data, calculate_entropy=True)
    batch = driver_batch(data, log_probs)  # only the second rank's rows are in the driver batch
    metrics = la.finalize_pending(tmp_path, batch)
    assert metrics["loop_audit/responses"] == 13
    assert metrics["loop_audit/unmatched_responses"] == 6
    assert la.finalize_pending(tmp_path, batch) == {}  # already finalized


def test_identical_rollouts_are_both_matched(tmp_path):
    actor = FakeActor()
    la.attach_loop_audit(actor, settings(tmp_path), rank=0)
    data = make_data()
    for tensor in data.batch.values():
        tensor[1] = tensor[0]
    log_probs, _ = actor.compute_log_prob(data=data, calculate_entropy=True)
    metrics = la.finalize_pending(tmp_path, driver_batch(data, log_probs))
    assert metrics["loop_audit/unmatched_responses"] == 0


def test_attach_needs_a_config_and_one_ouro_model(tmp_path):
    actor = FakeActor()
    original = actor.compute_log_prob
    assert la.attach_loop_audit(actor, None, rank=0) is None
    assert actor.compute_log_prob == original
    actor.actor_module = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match="expected one OuroModel"):
        la.attach_loop_audit(actor, settings(tmp_path), rank=0)


def test_driver_wrapper_finalizes_the_audit(tmp_path, monkeypatch):
    actor = FakeActor()
    la.attach_loop_audit(actor, settings(tmp_path), rank=0)
    data = make_data()
    log_probs, _ = actor.compute_log_prob(data=data, calculate_entropy=True)
    batch = driver_batch(data, log_probs)
    batch.batch["response_mask"] = data.batch["attention_mask"][:, -5:]

    metric_utils = types.ModuleType("verl.trainer.ppo.metric_utils")
    metric_utils.compute_data_metrics = lambda batch, use_critic=True: {"critic/score/mean": 0.1}
    ppo = types.ModuleType("verl.trainer.ppo")
    ppo.metric_utils = metric_utils
    verl = types.ModuleType("verl")
    trainer = types.ModuleType("verl.trainer")
    for name, mod in {"verl": verl, "verl.trainer": trainer, "verl.trainer.ppo": ppo,
                      "verl.trainer.ppo.metric_utils": metric_utils}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    rmet.install_driver_metrics(loop_audit=settings(tmp_path))
    out = metric_utils.compute_data_metrics(batch, use_critic=False)
    assert out["critic/score/mean"] == 0.1 and "reasoning/accuracy" in out
    assert out["loop_audit/responses"] == 6


# ---------- config ----------

@pytest.fixture
def env(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)


def _args(no_lora=True):
    return types.SimpleNamespace(no_lora=no_lora, output_dir="/out/run")


@pytest.mark.parametrize("name", sorted(p.name for p in paths.config_file(
    "grpo_baseline_8gpu.yaml").parent.glob("*.yaml")))
def test_audit_is_on_in_every_config(env, name):
    cfg = rcfg.load_config(str(paths.config_file(name)))
    assert cfg.audit.loop_distribution.enabled is True
    audit = rcfg.add_audit_overrides(cfg, {}, _args())["actor_rollout_ref"]["loop_audit"]
    assert audit == {"every_n_steps": 1, "top_k": 5, "save_tokens": True, "chunk_tokens": 512,
                     "loop_weights": True, "dir": "/out/run/audit"}


def test_config_switch_adds_the_worker_settings(env):
    cfg = rcfg.load_config(str(paths.config_file("grpo_baseline_1p4b_8gpu.yaml")),
                           ["audit.loop_distribution.enabled=true",
                            "audit.loop_distribution.every_n_steps=10"])
    overrides = rcfg.add_audit_overrides(cfg, {"trainer": {"logger": ["console"]}}, _args())
    assert overrides["trainer"] == {"logger": ["console"]}
    assert overrides["actor_rollout_ref"]["loop_audit"] == {
        "every_n_steps": 10, "top_k": 5, "save_tokens": True, "chunk_tokens": 512,
        "loop_weights": True, "dir": "/out/run/audit"}


def test_missing_settings_use_the_defaults(env):
    cfg = rcfg.load_config(str(paths.config_file("rltt_baseline.yaml")),
                           ["audit.loop_distribution={enabled: true}"])
    audit = rcfg.add_audit_overrides(cfg, {}, _args())["actor_rollout_ref"]["loop_audit"]
    assert audit["top_k"] == 5 and audit["every_n_steps"] == 1


@pytest.mark.parametrize("override,error", [
    ("audit.other=1", KeyError),
    ("audit.loop_distribution.topk=5", KeyError),
    ("audit.loop_distribution.top_k=0", ValueError),
    ("audit.loop_distribution.every_n_steps=1.5", ValueError),
    ("audit.loop_distribution.enabled=1", TypeError),
    ("audit.loop_distribution.loop_weights=1", TypeError),
])
def test_bad_audit_settings_are_refused(env, override, error):
    with pytest.raises(error):
        rcfg.load_config(str(paths.config_file("grpo_baseline_8gpu.yaml")), [override])


def test_audit_refuses_lora_and_resume(env):
    cfg = rcfg.load_config(str(paths.config_file("grpo_baseline_8gpu.yaml")),
                           ["audit.loop_distribution.enabled=true"])
    with pytest.raises(ValueError, match="no_lora"):
        rcfg.add_audit_overrides(cfg, {}, _args(no_lora=False))
    with pytest.raises(ValueError, match="resume"):
        rcfg.add_audit_overrides(cfg, {"trainer": {"resume_mode": "auto"}}, _args())
