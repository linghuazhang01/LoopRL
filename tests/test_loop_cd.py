"""LoopCD guidance math (rltt_baseline.loop_cd): CPU only, no vLLM."""

from types import SimpleNamespace

import pytest
import torch

from rltt_baseline.loop_cd import (
    LoopCDConfig,
    adaptive_omega,
    engine_overrides,
    from_hf_config,
    guided_hidden,
    guided_logprobs,
    run_loops,
)

torch.manual_seed(42)
Z_S = torch.randn(5, 11) * 3
Z_R = torch.randn(5, 11) * 3


def _close(a, b):
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_omega_zero_is_the_source_loop():
    _close(guided_logprobs(Z_S, Z_R, LoopCDConfig(omega=0.0)), torch.log_softmax(Z_S, -1))


def test_output_is_normalized():
    for cfg in (LoopCDConfig(omega=0.5), LoopCDConfig(omega=1.5, adaptive=True),
                LoopCDConfig(omega=0.5, delta_cap=1.0)):
        _close(torch.logsumexp(guided_logprobs(Z_S, Z_R, cfg), -1), torch.zeros(5))


def test_matches_the_paper_logit_formula_and_probability_form():
    w = 0.5
    paper = torch.log_softmax(Z_S + w * (Z_S - Z_R), -1)  # Eq. 3, then softmax
    ours = guided_logprobs(Z_S, Z_R, LoopCDConfig(omega=w))
    _close(ours, paper)
    p_s, p_r = torch.softmax(Z_S, -1), torch.softmax(Z_R, -1)
    unnorm = p_s * (p_s / p_r) ** w  # App. A.3
    assert (unnorm.sum(-1) > 1).all()  # extrapolation is not normalized by itself
    _close(ours.exp(), unnorm / unnorm.sum(-1, keepdim=True))


def test_invariant_to_per_position_logit_shifts():
    shift_s, shift_r = torch.randn(5, 1) * 10, torch.randn(5, 1) * 10
    cfg = LoopCDConfig(omega=0.7, delta_cap=0.5)
    _close(guided_logprobs(Z_S + shift_s, Z_R + shift_r, cfg), guided_logprobs(Z_S, Z_R, cfg))


def test_omega_minus_one_is_the_reference_loop():
    _close(guided_logprobs(Z_S, Z_R, LoopCDConfig(omega=-1.0)), torch.log_softmax(Z_R, -1))


def test_adaptive_strength_follows_the_top_two_margin():
    tie = torch.tensor([[2.0, 2.0, -5.0, -5.0]])
    peaked = torch.tensor([[30.0, 0.0, 0.0, 0.0]])
    _close(adaptive_omega(tie, 1.5), torch.tensor([[1.5 * (1 - 0.0)]]))
    assert adaptive_omega(peaked, 1.5).item() < 1e-6
    p = torch.softmax(Z_S, -1).sort(-1, descending=True).values
    _close(adaptive_omega(Z_S, 1.0), 1.0 - (p[:, :1] - p[:, 1:2]))
    # adaptive = fixed strength chosen per position
    adaptive = guided_logprobs(Z_S, Z_R, LoopCDConfig(omega=1.0, adaptive=True))
    w = adaptive_omega(Z_S, 1.0)
    _close(adaptive, torch.log_softmax(Z_S + w * (Z_S - Z_R), -1))


def test_delta_cap_bounds_the_boost():
    z_s = torch.tensor([[0.0, 0.0, -2.0]])
    z_r = torch.tensor([[0.0, 0.0, -40.0]])  # the reference rules token 2 out
    uncapped = guided_logprobs(z_s, z_r, LoopCDConfig(omega=1.0))
    capped = guided_logprobs(z_s, z_r, LoopCDConfig(omega=1.0, delta_cap=1.0))
    assert uncapped.exp()[0, 2] > 0.99
    log_s = torch.log_softmax(z_s, -1)
    log_r = torch.log_softmax(z_r, -1)
    expected = torch.log_softmax(log_s + (log_s - log_r).clamp(-1.0, 1.0), -1)
    _close(capped, expected)


def test_guided_hidden():
    h_s, h_r = torch.randn(3, 8), torch.randn(3, 8)
    _close(guided_hidden(h_s, h_r, 0.0), h_s)
    _close(guided_hidden(h_s, h_r, 0.3), 1.3 * h_s - 0.3 * h_r)


@pytest.mark.parametrize("kwargs, match", [
    (dict(ref_loop=4), "ref_loop"),
    (dict(ref_loop=0), "ref_loop"),
    (dict(space="layers"), "space"),
    (dict(space="hidden", adaptive=True), "fixed strength"),
    (dict(space="hidden", delta_cap=1.0), "fixed strength"),
    (dict(adaptive=True, omega=-0.1), "cap"),
    (dict(delta_cap=0.0), "delta_cap"),
    (dict(omega=float("nan")), "finite"),
])
def test_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        LoopCDConfig(**kwargs).validate(4)


def test_engine_overrides():
    assert engine_overrides(None, 4) == {"total_ut_steps": 4}
    cfg = LoopCDConfig(ref_loop=1, omega=0.5)
    over = engine_overrides(cfg, 2)  # half depth: source is loop 2
    assert over["architectures"] == ["OuroLoopCDForCausalLM"] and over["total_ut_steps"] == 2
    assert from_hf_config(SimpleNamespace(**over)) == cfg
    with pytest.raises(ValueError, match="ref_loop"):
        engine_overrides(LoopCDConfig(ref_loop=2), 2)


def test_hf_config_round_trip():
    cfg = LoopCDConfig(ref_loop=2, omega=0.3, adaptive=True, space="logits", delta_cap=3.0)
    hf = SimpleNamespace(total_ut_steps=4, **cfg.hf_overrides())
    assert from_hf_config(hf) == cfg
    assert from_hf_config(SimpleNamespace(total_ut_steps=4)) == LoopCDConfig()
    with pytest.raises(ValueError, match="ref_loop"):
        from_hf_config(SimpleNamespace(total_ut_steps=2, **cfg.hf_overrides()))


# --- the loop readout, with stubs that mimic vLLM's in-place fused add + RMSNorm -------------

def _rms(x):
    return x / x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()


def _fused_norm(x, residual=None):
    """vLLM RMSNorm: in place on both buffers when a residual is given."""
    if residual is None:
        return _rms(x)
    residual.add_(x)
    x.copy_(_rms(residual))
    return x, residual


def _make_layers(n_layers=3, width=8):
    g = torch.Generator().manual_seed(42)
    weights = [torch.randn(width, width, generator=g) / width ** 0.5 for _ in range(n_layers)]

    def make(w):
        def layer(positions, h, ut, residual):
            if residual is None:
                residual, h = h, _rms(h)
            else:
                h, residual = _fused_norm(h, residual)
            return torch.tanh(h @ w + 0.1 * ut), residual
        return layer
    return [make(w) for w in weights], weights


def _reference_states(x, weights, total):
    """Out-of-place loop: per-loop (post-norm, pre-norm) states."""
    states, h = [], x
    for ut in range(total):
        residual = None
        for w in weights:
            if residual is None:
                residual, h_in = h, _rms(h)
            else:
                residual = residual + h
                h_in = _rms(residual)
            h = torch.tanh(h_in @ w + 0.1 * ut)
        pre = residual + h
        h = _rms(pre)
        states.append((h, pre))
    return states


@pytest.mark.parametrize("ref_loop", [1, 2, 3])
def test_run_loops_logits_readout(ref_loop):
    layers, weights = _make_layers()
    x = torch.randn(6, 8)
    out = run_loops(x.clone(), None, layers, _fused_norm, 4, LoopCDConfig(ref_loop=ref_loop))
    states = _reference_states(x, weights, 4)
    _close(out[:, :8], states[3][0])
    _close(out[:, 8:], states[ref_loop - 1][0])


def test_run_loops_hidden_readout():
    layers, weights = _make_layers()
    x = torch.randn(6, 8)
    cfg = LoopCDConfig(ref_loop=1, omega=0.3, space="hidden")
    out = run_loops(x.clone(), None, layers, _fused_norm, 4, cfg)
    states = _reference_states(x, weights, 4)
    _close(out, _rms(states[3][1] + 0.3 * (states[3][1] - states[0][1])))
    zero = run_loops(x.clone(), None, layers, _fused_norm, 4, LoopCDConfig(omega=0.0,
                                                                           space="hidden"))
    _close(zero, states[3][0])  # omega 0 reads out the stock last-loop state


def test_eval_metrics():
    from rltt_baseline.eval_loopcd import pass_at_k, summarize

    assert pass_at_k(8, 0, 1) == 0.0 and pass_at_k(8, 8, 1) == 1.0
    assert pass_at_k(8, 2, 1) == pytest.approx(2 / 8)
    assert pass_at_k(8, 1, 8) == 1.0
    s = summarize([[1, 0, 0, 0], [0, 0, 0, 0]], [10, 20, 30, 40, 50, 60, 70, 80],
                  [0, 0, 0, 1, 0, 0, 1, 1], n=4)
    assert s["avg@4"] == pytest.approx(1 / 8)
    assert s["pass@4"] == pytest.approx(0.5) and s["pass@1"] == pytest.approx(1 / 8)
    assert s["truncation_rate"] == pytest.approx(3 / 8)
    assert s["mean_response_tokens"] == pytest.approx(45)
