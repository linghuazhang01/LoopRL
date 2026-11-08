"""Gate and entropy push diagnostics on synthetic audit data (CPU)."""

import json

import numpy as np
import pytest
import torch

from rltt_baseline.diagnostics import gate_push as gp
from rltt_baseline.loop_audit import step_dir_name
from rltt_baseline.ouro_rltt.rltt_per_loop import exit_pdf_from_gates

LOOPS, VOCAB = 4, 7


def random_pdf(n, seed=42):
    g = torch.Generator().manual_seed(seed)
    return torch.softmax(torch.randn(n, LOOPS, generator=g, dtype=torch.float64), -1)


def test_gate_logits_invert_the_exit_pdf():
    pdf = random_pdf(50)
    logits = gp.gate_logits_from_pdf(pdf)
    assert torch.allclose(gp.pdf_from_gate_logits(logits), pdf, atol=1e-9)
    gates = [logits[:, t:t + 1] for t in range(LOOPS - 1)] + [torch.zeros(50, 1)]
    assert torch.allclose(exit_pdf_from_gates(gates).double(), pdf, atol=1e-6)


def test_gate_push_moves_mass_to_the_worst_loop_for_negative_advantage():
    ell = np.array([[-2.0, -1.0, -0.6, -0.5]])
    neg = gp.gate_push(ell, np.array([-1.0]))
    pos = gp.gate_push(ell, np.array([1.0]))
    assert neg.argmax() == 0 and pos.argmax() == 3
    assert np.allclose(neg, -pos) and abs(neg.sum()) < 1e-12


def test_gate_dw_matches_a_finite_difference_step():
    pdf = random_pdf(6).numpy()
    rng = np.random.default_rng(42)
    ell = -rng.random((6, LOOPS)) * 3
    coef = rng.normal(size=6)
    z = gp.gate_logits_from_pdf(torch.from_numpy(pdf)).requires_grad_(True)
    objective = (torch.from_numpy(coef)[:, None] * gp.pdf_from_gate_logits(z)
                 * torch.from_numpy(ell)).sum()
    (grad,) = torch.autograd.grad(objective, z)
    eps = 1e-6
    moved = gp.pdf_from_gate_logits(z.detach() + eps * grad)
    expected = ((moved - gp.pdf_from_gate_logits(z.detach())) / eps).sum(0).numpy()
    assert np.allclose(gp.gate_dw(ell, pdf, coef), expected, atol=1e-5)


def test_entropy_push_is_exact_with_the_full_vocabulary():
    g = torch.Generator().manual_seed(42)
    logits = torch.randn(5, LOOPS, VOCAB, generator=g, dtype=torch.float64) * 2
    labels = torch.randint(VOCAB, (5,), generator=g)
    coef = torch.randn(5, LOOPS, generator=g, dtype=torch.float64)
    log_p = torch.log_softmax(logits, -1)
    p = log_p.exp()
    entropy = -(p * log_p).sum(-1)
    ell = log_p.gather(-1, labels.view(-1, 1, 1).expand(-1, LOOPS, 1)).squeeze(-1)

    expected = torch.zeros(5, LOOPS, dtype=torch.float64)
    for i in range(5):
        for t in range(LOOPS):
            z = logits[i, t].clone().requires_grad_(True)
            lp = torch.log_softmax(z, -1)
            (grad_h,) = torch.autograd.grad(-(lp.exp() * lp).sum(), z)
            step = coef[i, t] * (torch.nn.functional.one_hot(labels[i], VOCAB) - p[i, t])
            expected[i, t] = grad_h @ step

    top = p.sort(-1, descending=True).values.numpy()  # k = VOCAB: no tail left out
    got = gp.entropy_push(ell.numpy(), entropy.numpy(), top, coef.numpy())
    assert np.allclose(got, expected.numpy(), atol=1e-10)


def test_positive_advantage_on_the_top_token_lowers_entropy():
    probs = np.array([[[0.7, 0.2, 0.1]]])
    entropy = np.array([[-(probs * np.log(probs)).sum()]])
    ell = np.log(np.array([[0.7]]))
    assert gp.entropy_push(ell, entropy, probs, np.array([[1.0]]))[0, 0] < 0
    assert gp.entropy_push(ell, entropy, probs, np.array([[-1.0]]))[0, 0] > 0


def write_step(audit, step, rewards, lengths, pdf_first):
    """One audited step: one prompt group, one rank; loop 1 is the worst loop on every token."""
    step_dir = audit / step_dir_name(step)
    step_dir.mkdir(parents=True)
    n = sum(lengths)
    rng = np.random.default_rng(step)
    ell = np.stack([-2.0 - rng.random(n), -0.8 - 0.1 * rng.random(n),
                    -0.6 - 0.1 * rng.random(n), -0.5 - 0.1 * rng.random(n)], -1)
    top = np.sort(rng.dirichlet(np.ones(5), size=(n, LOOPS)), -1)[..., ::-1] * 0.9
    pdf = np.tile([pdf_first, 0.2, 0.3, 0.5 - pdf_first], (n, 1))
    np.savez_compressed(
        step_dir / "tokens_rank00.npz", sampled_log_prob=ell.astype(np.float32),
        entropy=np.full((n, LOOPS), 1.0, np.float32), topk_probs=top.astype(np.float32),
        exit_pdf=pdf.astype(np.float32), offsets=np.concatenate([[0], np.cumsum(lengths)]))
    with open(step_dir / "responses.jsonl", "w") as fh:
        for i, reward in enumerate(rewards):
            fh.write(json.dumps({"rank": 0, "index": i, "uid": "g0", "reward": reward}) + "\n")
        fh.write(json.dumps({"rank": 0, "index": len(rewards), "uid": None, "reward": None})
                 + "\n")
    return step_dir


def test_end_to_end_on_a_synthetic_audit(tmp_path):
    audit = tmp_path / "audit"
    rewards, lengths = [1.0, 0.0, 0.0, 1.0], [3, 9, 8, 2]  # wrong answers are longer
    for step, first in ((20, 0.10), (21, 0.12), (22, 0.14)):
        write_step(audit, step, rewards, lengths + [4], first)  # the last response is unmatched
    out = tmp_path / "push.csv"
    assert gp.main(["--audit-dir", str(audit), "--out", str(out)]) == 0
    summary = json.loads(out.with_suffix(".csv.summary.json").read_text())
    assert summary["steps"] == [20, 21, 22]
    assert summary["neg_token_share"] == pytest.approx(17 / 22)
    first = summary["totals"]["gate_push/loop_1"]
    assert first["total"] > 0 and first["from_neg"] > abs(first["from_pos"])
    assert summary["gate_dw_vs_next_pdf_mean"]["loop_1"]["rate"] == 1.0

    rows = list(gp.step_dirs([audit], (21, 22), 1))
    assert [s for s, _ in rows] == [21, 22]
    row = gp.analyze_step(rows[0][1], weighting="last")
    assert row["tokens"] == 22 and row["entropy_push/loop_1"] == 0.0
