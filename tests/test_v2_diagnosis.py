"""Check diagnostic metric edge cases and selective precision boundaries."""
import torch
from torch import nn

from experiments.v2.diagnose_stage1 import HEADS, decision, double_heads, energy64, gradient_metrics
from power_fm_foresight.scoring import fair_energy_score


def test_cosine_does_not_hide_magnitude_or_zero_reference():
    reference = torch.tensor([1., 2.])
    same_direction = gradient_metrics(reference * 4, reference, 1e-12)
    assert abs(same_direction["cosine"] - 1) < 1e-12
    assert same_direction["norm_ratio"] == 4
    unresolved = gradient_metrics(reference, torch.zeros(2), 1e-12)
    assert unresolved["cosine"] is None and unresolved["norm_ratio"] is None
    protocol = {"decision": {"unreliable_below": .95, "usable_at_least": .99}}
    assert decision([{"groups": {"g": unresolved}}], protocol) == "inconclusive_hold"
    assert decision([{"groups": {"g": same_direction}}], protocol) == "usable_gradient_requires_epoch_gate"
    opposite = gradient_metrics(-reference, reference, 1e-12)
    assert decision([{"groups": {"g": opposite}}], protocol) == "unreliable_gradient_test_selective_fp64"


def test_double_loss_preserves_fair_pairs_and_masks():
    samples = torch.tensor([[1., 2., 3.], [4., 5., 6.], [2., 7., 4.]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([2., float("nan"), 5.], dtype=torch.float64)
    scale = torch.tensor([1., 2., 3.], dtype=torch.float64)
    value = energy64(samples, target, scale)
    original = fair_energy_score(samples, target, scale)
    assert abs(float((value - original).detach())) < 1e-7
    value.backward()
    assert torch.isfinite(samples.grad).all()
    assert torch.equal(samples.grad[:, 1], torch.zeros(3, dtype=torch.float64))
    assert torch.autograd.gradcheck(lambda x: energy64(x, target, scale), (samples,))


def test_head_casts_preserve_input_gradient_and_restore_hooks():
    model = nn.Module()
    model.backbone = nn.Module()
    for name in HEADS:
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, leaf, nn.Linear(3, 2))
    audit = {}
    x = torch.ones(4, 3, requires_grad=True)
    with double_heads(model, audit):
        outputs = [model.get_submodule(name)(x) for name in HEADS]
        assert all(value.dtype == torch.float32 for value in outputs)
        sum(value.sum() for value in outputs).backward()
    assert x.grad.dtype == torch.float32 and torch.isfinite(x.grad).all()
    assert set(audit) == set(HEADS)
    for name in HEADS:
        module = model.get_submodule(name)
        assert module.weight.grad.dtype == torch.float64
        assert not module._forward_hooks and not module._forward_pre_hooks
