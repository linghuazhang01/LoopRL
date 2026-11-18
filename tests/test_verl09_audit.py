import numpy as np
import pytest

torch = pytest.importorskip("torch")
from rltt_baseline.loop_audit import distribution_stats, row_key
from rltt_baseline.verl09_audit import NativeLoopAudit


def test_native_right_padding_audit_matches_response_positions():
    audit = NativeLoopAudit.__new__(NativeLoopAudit)
    audit.lm_head = torch.nn.Linear(3, 7, bias=False)
    audit._temperature, audit.top_k, audit.chunk_tokens = .6, 2, 4
    audit.loop_weights, audit.plc_settings = False, None
    nested = lambda rows: torch.nested.as_nested_tensor(rows, layout=torch.jagged)
    data = {
        "prompts": nested([torch.tensor([1, 2]), torch.tensor([3])]),
        "responses": nested([torch.tensor([4, 5]), torch.tensor([6])]),
        "input_ids": nested([torch.tensor([1, 2, 4, 5]), torch.tensor([3, 6])]),
    }
    hidden = [torch.arange(24, dtype=torch.float32).reshape(2, 4, 3) / (i + 20)
              for i in range(2)]
    result = audit.measure(data, {}, hidden, [])
    expected = distribution_stats([h[[0, 0, 1], [1, 2, 0]] for h in hidden],
                                   audit.lm_head, torch.tensor([4, 5, 6]), .6, 2, 4)
    np.testing.assert_allclose(result["sampled_log_prob"], expected["sampled_log_prob"].numpy())
    assert result["response_tokens"].tolist() == [2, 1]
    assert result["row_keys"][1] == row_key(np.array([3, 6]), np.array([1, 1]))
