"""A0 detaches only GENCO while retaining demand and final-state gradients."""
from copy import deepcopy
import numpy as np
import torch
from test_adapter import tiny
from power_fm_foresight.scoring import fair_energy_score
from power_fm_foresight._solver.checks import physical_quantities


def a0(tiny):
    original,net,single,future,history=tiny
    model=deepcopy(original);model.model_policy='demand_only_a0'
    model.bus_conditioning.requires_grad_(False);model.gen_conditioning.requires_grad_(False)
    return model,net,single,future,history


def test_a0_zero_final_innovations_and_unconditioned_are_exact(tiny):
    model,net,single,future,history=a0(tiny)
    with torch.no_grad():
        reference=model.backbone(future)
        raw=model.unconditioned(future)
        actual=model.demand_only_output(future,torch.zeros_like(reference['bus'][:,:2]),torch.zeros_like(reference['gen']))
    for key in reference:
        assert torch.equal(reference[key],raw[key])
        assert torch.equal(reference[key],actual[key])


def test_a0_demand_es_gradient_detached_backbone_and_physics(tiny):
    model,net,single,future,history=a0(tiny)
    grad_modes=[]
    hook=model.backbone.register_forward_pre_hook(lambda *args:grad_modes.append(torch.is_grad_enabled()))
    result=model(history,torch.zeros(1,6),future,model.noise(2,1));hook.remove()
    assert grad_modes==[False]
    demand=result['values']['pd_mw'][:,:,model.baseline_pd!=0].flatten(1)
    loss=fair_energy_score(demand,demand.detach().mean(0)+1,torch.ones(demand.shape[1]))
    loss.backward(retain_graph=True)
    assert model.bus_head.weight.grad[0].abs().sum()>0
    model.zero_grad(set_to_none=True)
    loss=fair_energy_score(result['joint'],result['joint'].detach().mean(0)+.1,model.coordinate_scale)+result['power_balance']
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    assert all(p.grad is None for p in model.backbone.parameters())
    assert model.bus_conditioning.weight.grad is None
    for i in range(2):
        values={k:v[i,0].detach().numpy() for k,v in result['values'].items()}
        actual=physical_quantities(net,values['pd_mw'],values['qd_mvar'],values)
        for key in ('pf_mw','qf_mvar','pt_mw','qt_mvar'):
            np.testing.assert_allclose(values[key],actual[key],atol=1e-3,rtol=1e-4)
