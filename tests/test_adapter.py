from pathlib import Path

import numpy as np
import pytest
import torch

from common import ROOT, load_config, read_json
from data.download.prepare import checkpoint_assets
from data.loader.network import graph_for_demand,load_network
from models.genco import Genco
from models.temporal import TemporalAdapter
from scoring import fair_energy_score
from scripts.stage_b_prevalence import prevalence_shift
from solver.checks import physical_quantities


@pytest.fixture(scope="module")
def tiny():
    cfg=load_config();net=load_network();base=Genco(checkpoint_assets(cfg),cfg["checkpoint"])
    scales=read_json(ROOT/cfg["stage_b"]["coordinate_scales"])
    torch.manual_seed(42)
    model=TemporalAdapter(base,net,scales)
    row=read_json(Path(cfg["paths"]["temporal_root"])/"episodes/train_00.json")["records"][0]
    graph=graph_for_demand(net,row["pd_mw"],row["qd_mvar"])
    single=base.batch([graph]);future=base.batch([graph,graph])
    with torch.no_grad(): _,emb=base.model(single,return_embeddings=True)
    history={f"embedding_{k}":v.unsqueeze(0).expand(3,-1,-1).clone() for k,v in emb.items()}
    history["observed_bus"]=torch.tensor(np.stack([row["pd_mw"],row["qd_mvar"],row["target_state"]["vm"],row["target_state"]["va_rad"]],axis=-1),dtype=torch.float32).unsqueeze(0).repeat(3,1,1)
    history["observed_gen"]=torch.tensor(np.stack([row["target_state"]["pg_mw"],row["target_state"]["qg_mvar"]],axis=-1),dtype=torch.float32).unsqueeze(0).repeat(3,1,1)
    for kind in ("bus","gen"):history[f"available_{kind}"]=torch.isfinite(history[f"observed_{kind}"])
    return model,net,single,future,history


def test_adapter_batch_gradients_and_physics_match_own_sample(tiny):
    model,net,single,future,history=tiny
    model.zero_grad(set_to_none=True)
    before={k:v.clone() for k,v in model.backbone.state_dict().items()}
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==91461
    noise=model.noise(2,1)
    result=model(history,torch.zeros(1,6),future,noise)
    torch.testing.assert_close(result["values"]["pd_mw"],result["demand_seen_by_physics_mw"])
    assert not torch.equal(result["values"]["pd_mw"][0],result["values"]["pd_mw"][1])
    for i in range(2):
        values={k:v[i,0].detach().numpy() for k,v in result["values"].items()}
        independent=physical_quantities(net,values["pd_mw"],values["qd_mvar"],values)
        for key in ("pf_mw","qf_mvar","pt_mw","qt_mvar"):
            np.testing.assert_allclose(values[key],independent[key],atol=1e-3,rtol=1e-4)
    target=result["joint"].detach().mean(0)+.1
    loss=fair_energy_score(result["joint"],target,model.coordinate_scale)+.01*result["power_balance"]
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    assert all(p.grad is None for p in model.backbone.parameters())
    assert all(torch.equal(before[k],v) for k,v in model.backbone.state_dict().items())
    assert not model.backbone.input_proj_bus._forward_hooks
    with torch.no_grad():
        actual=model.unconditioned(single);expected=model.backbone(single)
    assert all(torch.equal(actual[k],expected[k]) for k in actual)


def test_adapter_hooks_clear_after_failure_and_round_trip_is_exact(tiny,tmp_path):
    model,net,single,future,history=tiny
    with pytest.raises(RuntimeError,match="deliberate"):
        with model.conditioning(None,None,None,None): raise RuntimeError("deliberate")
    assert not model.backbone.input_proj_bus._forward_hooks
    noise=model.noise(2,1)
    with torch.no_grad(): expected=model(history,torch.zeros(1,6),future,noise)["joint"]
    path=tmp_path/"adapter.pt"
    state={k:v for k,v in model.state_dict().items() if not k.startswith("backbone.")}
    torch.save(state,path)
    with torch.no_grad():model.bus_head.weight.add_(1)
    model.load_state_dict(torch.load(path,weights_only=True),strict=False)
    with torch.no_grad():actual=model(history,torch.zeros(1,6),future,noise)["joint"]
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_fair_energy_score_excludes_diagonal_and_masks_missing_targets():
    samples=torch.tensor([[0.,999.],[2.,-999.]],requires_grad=True)
    target=torch.tensor([3.,float('nan')])
    score=fair_energy_score(samples,target,torch.tensor([2.,1.]))
    torch.testing.assert_close(score,torch.tensor(.5))
    score.backward()
    assert torch.isfinite(samples.grad).all() and (samples.grad[:,1]==0).all()


def test_prevalence_shift_uses_strict_exact_threshold():
    ratio,shifted=prevalence_shift({"positive":40,"total":864},{"positive":120,"total":1728},1.5)
    assert ratio==1.5 and not shifted
    assert prevalence_shift({"positive":40,"total":864},{"positive":121,"total":1728},1.5)[1]
    assert prevalence_shift({"positive":0,"total":864},{"positive":1,"total":1728},1.5)==(None,True)
