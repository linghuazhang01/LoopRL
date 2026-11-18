"""Contract for padded Ouro outputs shared by old and future engine adapters."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from rltt_baseline.per_loop_forward import per_loop_forward


@pytest.mark.parametrize("kd", [None, "forward_kl"])
@pytest.mark.parametrize("mrope", [False, True])
def test_alignment_labels_gradients_and_kd(kd, mrope):
    values = torch.arange(12., requires_grad=True).reshape(2, 6)
    ids = torch.arange(12).reshape(2, 6)
    positions = torch.zeros((2, 3, 6) if mrope else (2, 6), dtype=torch.long)
    batch = dict(input_ids=ids, responses=ids[:, -2:],
                 attention_mask=torch.ones_like(ids), position_ids=positions)
    seen = {}

    def model(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(per_loop_log_probs=[values, values * 2],
                               per_loop_entropies=[values * 3, values * 4],
                               loop_kd=values[:, -3:-1], loop_kd_stats={"x": values})

    entropy, loops, kd_loss, stats = per_loop_forward(model, batch, 0.6, kd)
    torch.testing.assert_close(loops, torch.stack([values[:, 3:5], values[:, 3:5] * 2], -1))
    torch.testing.assert_close(entropy, values[:, 3:5] * 4)
    torch.testing.assert_close(seen["log_prob_labels"], torch.nn.functional.pad(ids[:, 1:], (0, 1)))
    assert seen["position_ids"].shape == ((3, 2, 6) if mrope else (2, 6))
    assert seen["use_cache"] is False
    assert seen["log_prob_temperature"] == 0.6
    if kd:
        assert seen["return_loop_kd"] is True and seen["loop_kd_window"] == 2
        assert stats["x"] is values
        torch.testing.assert_close(kd_loss, values[:, 3:5])
    else:
        assert kd_loss is None and stats is None
        assert "return_loop_kd" not in seen
    grad, = torch.autograd.grad(loops.sum(), values)
    expected = torch.zeros_like(values)
    expected[:, 3:5] = 3
    torch.testing.assert_close(grad, expected)
