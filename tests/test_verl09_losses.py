"""CPU numerical contracts for the engine callback's unscaled objectives."""
import pytest

torch = pytest.importorskip("torch")
from rltt_baseline import plc
from rltt_baseline.verl09_losses import objective


def fixture():
    ell = torch.tensor([[[-1.2, -1.], [-.8, -.9]], [[-.6, -.5], [-.3, -.4]]], requires_grad=True)
    old = ell.detach() - .03
    mask = torch.tensor([[1., 1.], [1., 0.]])
    output = dict(per_loop_log_probs=ell, entropy=torch.zeros_like(mask),
                  per_loop_entropies=torch.zeros_like(ell))
    batch = dict(old_log_probs=old[..., -1], old_log_probs_loops=old,
                 response_mask=mask, advantages=torch.tensor([[1., 1.], [-1., -1.]]))
    return ell, old, mask, output, batch


@pytest.mark.parametrize("lam", [0., .1])
def test_plc_matches_existing_loss_and_gradient(lam):
    ell, old, mask, output, batch = fixture()
    cfg = dict(clip_ratio=.2, kl_loss_coef=0., entropy_coeff=0.)
    actual, _ = objective(cfg, dict(method="plc", plc=dict(lam=lam, kl_coef=.2)), output, batch)
    expected, _ = plc.plc_policy_loss(ell, old, batch["advantages"], mask,
        plc.compute_loop_weights(old), lam, .2, kl_coef=.2,
        kl_weights=plc.compute_loop_weights(old, "early_prior"))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, ell, retain_graph=True)[0],
                               torch.autograd.grad(expected, ell)[0])


@pytest.mark.parametrize("mode", ["uniform", "progressive", "exit_pdf"])
def test_rltt_reinforce(mode):
    ell, old, mask, output, batch = fixture()
    weights = torch.tensor([.5, .5]) if mode == "uniform" else torch.tensor([1/3, 2/3])
    output["exit_pdf"] = weights.expand_as(ell)
    actual, _ = objective({}, dict(method="rltt", loop_weighting=mode), output, batch)
    expected = (-(ell * weights).sum(-1) * batch["advantages"] * mask).sum() / mask.sum()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, ell, retain_graph=True)[0],
                               torch.autograd.grad(expected, ell)[0])


def test_plc_requires_frozen_anchor_tensor():
    _, _, _, output, batch = fixture()
    with pytest.raises(ValueError, match="logp_init"):
        objective({}, dict(method="plc", plc=dict(init_kl_coef=.2)), output, batch)
