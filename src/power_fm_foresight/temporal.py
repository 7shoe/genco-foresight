"""Tiny temporal adapter ontop GENCO forward/physics APIs.

Historical embeddings are cached from frozen GENCO. Observed states are explicit
separate inputs because OPF masking hides them from the backbone. No solver or
training package is imported here. All graph batches are in checkpoint units.
"""
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from power_fm_foresight.grids import check_case


class TemporalAdapter(nn.Module):
    """Use grid history to condition GENCO and sample future trajectories."""
    def __init__(self, base, network, scales, model_policy="conditioned"):
        super().__init__()
        if model_policy not in ("conditioned", "demand_only_a0"):
            raise ValueError(f"Unknown adapter model policy: {model_policy}")
        self.model_policy = model_policy
        check_case(network.case, base.case, "adapter backbone")

        # freeze backbone weights
        self.backbone = base.model.eval().requires_grad_(False)
        self.base_mva = float(base.normalizer.baseMVA)
        self.grid = network
        self.hidden_width = base.model.hidden_dim
        embedding_width = base.model.mlp_bus[0].in_features

        # encode observed values separately (`cause GENCO's OPF masking hides it)
        self.bus_observed = nn.Linear(8, 32)
        self.gen_observed = nn.Linear(4, 32)

        # Each history step combines cached embeddings and observed values.
        self.bus_gru = nn.GRU(embedding_width + 32, 32, batch_first=True)
        self.gen_gru = nn.GRU(embedding_width + 32, 32, batch_first=True)

        # Map history summaries into the checkpoint's input embeddings.
        self.bus_conditioning = nn.Linear(32, self.hidden_width)
        self.gen_conditioning = nn.Linear(32, self.hidden_width)
        if self.model_policy == "demand_only_a0":
            self.bus_conditioning.requires_grad_(False)
            self.gen_conditioning.requires_grad_(False)
        self.future_projection = nn.Linear(6, 32)

        # bus outputs contain a demand shift
        self.bus_head = nn.Linear(32, 28)
        self.gen_head = nn.Linear(32, 9)

        # small factors preserve grads to history from the 1st step
        nn.init.normal_(self.bus_head.weight, std=0.005)
        nn.init.normal_(self.gen_head.weight, std=0.005)
        nn.init.zeros_(self.bus_head.bias)
        nn.init.zeros_(self.gen_head.bias)

        # start independent noise scale
        with torch.no_grad():
            self.bus_head.bias[[1,10,19]] = -4.0
            self.gen_head.bias[0] = -4.0

        # buffers move with the model and are saved without being trained
        self.register_buffer("baseline_pd", torch.tensor(network["bus"][:,2], dtype=torch.float32))
        self.register_buffer("baseline_qd", torch.tensor(network["bus"][:,3], dtype=torch.float32))
        self.register_buffer("gen_bus", torch.tensor(network.gen_bus[network.online_gen], dtype=torch.long))
        self.register_buffer("free_pg", torch.tensor(network.gen[network.online_gen,8]>network.gen[network.online_gen,9]))
        self.register_buffer("free_va", torch.arange(network.n_bus) != network["reference_bus"])

        # load the training stats
        features = scales["features"]
        for kind, keys in (("bus",("pd_mw","qd_mvar","vm","va_rad")),("gen",("pg_mw","qg_mvar"))):
            for statistic in ("mean","scale"):
                source = scales.get("generator_history", features) if kind == "gen" else features
                values = np.stack([source[key][statistic] for key in keys],axis=-1)
                if kind == "gen": values = values[network.online_gen]
                self.register_buffer(f"{kind}_{statistic}",torch.tensor(values,dtype=torch.float32))

        # keep the selected scoring coords
        self.score_keys = tuple(key for key in features if any(features[key]["joint_score_mask"]))
        for key in self.score_keys:
            self.register_buffer(f"score_mask_{key}",torch.tensor(features[key]["joint_score_mask"]))
        self.register_buffer("coordinate_scale",torch.tensor(scales["scale_vector_one_hour"],dtype=torch.float32))

    def train(self, mode=True):
        """Change adapter mode while keeping GENCO in evaluation mode."""
        super().train(mode)
        self.backbone.eval()
        return self

    def unconditioned(self, batch, return_embeddings=False):
        """Exact upstram path: no hooks and no add-zero operations."""
        return self.backbone(batch, return_embeddings=return_embeddings)

    def memory(self, history):
        """summarize each bus and generator's history with a GRU."""
        result = {}
        for kind in ("bus","gen"):
            values = history[f"observed_{kind}"]
            available = history[f"available_{kind}"] & torch.isfinite(values)

            # 0 represents the training mean
            normalized = (values-getattr(self,f"{kind}_mean"))/getattr(self,f"{kind}_scale")
            # missing observations -> `mean` imputation & retain explicit masks
            normalized = torch.where(available,normalized,torch.zeros_like(normalized))
            projected = getattr(self,f"{kind}_observed")(torch.cat([normalized,available.to(values.dtype)],dim=-1))

            # treat nodes as batch items and historical hours as the GRU sequence.
            inputs = torch.cat([history[f"embedding_{kind}"],projected],dim=-1).transpose(0,1)
            _, hidden = getattr(self,f"{kind}_gru")(inputs)
            result[kind] = hidden[0]
        return result

    def noise(self, samples, horizon, device=None):
        device = device or self.baseline_pd.device
        return {"factor":torch.randn(samples,8,device=device),
                "bus":torch.randn(samples,horizon,self.grid.n_bus,3,device=device),
                "gen":torch.randn(samples,horizon,self.grid.n_online_gen,1,device=device)}

    @staticmethod
    def innovations(parameters, factor, diagonal_noise):
        sigma = F.softplus(parameters[...,0])+1e-4
        return (torch.einsum("hncr,kr->khnc",parameters[...,1:],factor)
                +sigma.unsqueeze(0)*diagonal_noise)

    @contextmanager
    def conditioning(self, bus_memory, gen_memory, bus_noise, gen_noise):
        counts = {"bus":0,"gen":0}
        def readout(kind, noise):
            def hook(module, args, output):
                counts[kind] += 1
                return output+noise if counts[kind] == self.backbone.num_layers else output
            return hook
        handles = []
        try:
            handles.append(self.backbone.input_proj_bus.register_forward_hook(lambda m,a,o:o+bus_memory))
            handles.append(self.backbone.input_proj_gen.register_forward_hook(lambda m,a,o:o+gen_memory))
            handles.append(self.backbone.mlp_bus.register_forward_hook(readout("bus",bus_noise)))
            handles.append(self.backbone.mlp_gen.register_forward_hook(readout("gen",gen_noise)))
            yield
            if counts != {"bus":self.backbone.num_layers,"gen":self.backbone.num_layers}:
                raise RuntimeError("Upstream readout invocation count changed")
        finally:
            for handle in handles: handle.remove()

    def forward(self, history, future_calendar, future_template, noise):
        """Sample future demand and grid states, then compute physics residuals."""

        k, horizon = noise["bus"].shape[:2]
        if future_template.num_graphs != k*horizon:
            raise ValueError("One future graph is required per sampled demand and hour")

        # combine each node's history summary with each future hour's calendar features
        memory = self.memory(history)
        calendar = self.future_projection(future_calendar)
        bus_memory = memory["bus"][None]+calendar[:,None]
        gen_memory = memory["gen"][None]+calendar[:,None]
        bus_head, gen_head = self.bus_head(bus_memory), self.gen_head(gen_memory)

        # bus noise controls demand, voltage magnitude, and voltage angle
        bus_z = self.innovations(bus_head[...,1:].reshape(horizon,self.grid.n_bus,3,9),noise["factor"],noise["bus"])
        gen_z = self.innovations(gen_head.reshape(horizon,self.grid.n_online_gen,1,9),noise["factor"],noise["gen"])

        # persistence center in positive demand coordinates; the temporal head learns its change.
        last = history["observed_bus"][-1,:,0].clamp_min(0)
        pd_scale = self.bus_scale[:,0]
        positive_center = (last/pd_scale).clamp_min(1e-6)
        inverse_softplus = positive_center+torch.log(-torch.expm1(-positive_center))

        # softplus keeps sampled active demand positive at buses with baseline demand
        pd = F.softplus(inverse_softplus[None,None]+bus_head[None,...,0]+bus_z[...,0])*pd_scale
        pd = torch.where(self.baseline_pd!=0,pd,torch.zeros_like(pd))

        # prserve the baseline reactive-to-active demand ratio
        ratio = self.baseline_qd/self.baseline_pd.where(self.baseline_pd!=0,torch.ones_like(self.baseline_pd))
        qd = torch.where(self.baseline_pd!=0,pd*ratio,self.baseline_qd)

        # write sampled demand into the graph
        batch = future_template.clone()
        bus_x = batch["bus"].x.clone()
        bus_x[:,0], bus_x[:,1] = pd.reshape(-1)/self.base_mva, qd.reshape(-1)/self.base_mva
        batch["bus"].x = bus_x

        # repeat conditioning across samples
        if self.model_policy == "conditioned":
            bc = self.bus_conditioning(bus_memory).unsqueeze(0).expand(k,-1,-1,-1).reshape(-1,self.hidden_width)
            gc = self.gen_conditioning(gen_memory).unsqueeze(0).expand(k,-1,-1,-1).reshape(-1,self.hidden_width)
        bus_innovation = torch.stack([bus_z[...,1],bus_z[...,2]*self.free_va],dim=-1).reshape(-1,2)
        gen_innovation = (gen_z*self.free_pg[None,None,:,None]).reshape(-1,1)

        # run frozen GENCO with differentiable additions
        if self.model_policy == "demand_only_a0":
            output = self.demand_only_output(batch, bus_innovation, gen_innovation)
        else:
            with self.conditioning(bc,gc,bus_innovation,gen_innovation):
                output = self.backbone(batch)

        # eval branch flows, nodal injections, and balance against sampled demand
        edge = batch["bus","connects","bus"]
        p,q = self.backbone.branch_flow_layer(output["bus"],edge.edge_index,edge.edge_attr)
        pi,qi = self.backbone.node_injection_layer(p,q,edge.edge_index,len(bus_x))
        rp,rq = self.backbone.node_residuals_layer(pi,qi,output["bus"],bus_x)

        # restore sample & hour dim
        bus = output["bus"].reshape(k,horizon,self.grid.n_bus,4)
        flows_p,flows_q = p.reshape(k,horizon,2*self.grid.n_online_branch)*self.base_mva,q.reshape(k,horizon,2*self.grid.n_online_branch)*self.base_mva
        values = {"pd_mw":pd,"qd_mvar":qd,"vm":bus[...,0],"va_rad":bus[...,1],
            "pg_mw":output["gen"].reshape(k,horizon,self.grid.n_online_gen)*self.base_mva,
            "qg_mvar":bus[:,:,self.grid.qg_buses,3]*self.base_mva,
            "pf_mw":flows_p[...,:self.grid.n_online_branch],"qf_mvar":flows_q[...,:self.grid.n_online_branch],
            "pt_mw":flows_p[...,self.grid.n_online_branch:],"qt_mvar":flows_q[...,self.grid.n_online_branch:]}

        # collect selected variables into one trajectory vector (per sample)
        for key in ("pg_mw",):
            if not self.grid.online_gen.all():
                full = values[key].new_full((k, horizon, self.grid.n_gen), float("nan"))
                full[..., self.grid.online_gen] = values[key]
                values[key] = full
        for key in ("pf_mw", "qf_mvar", "pt_mw", "qt_mvar"):
            if not self.grid.online_branch.all():
                full = values[key].new_full((k, horizon, self.grid.n_branch), float("nan"))
                full[..., self.grid.online_branch] = values[key]
                values[key] = full
        joint = torch.cat(
            [
                values[key][...,getattr(self,f"score_mask_{key}")] for key in self.score_keys
            ],
            dim=-1
        )

        return {"values" : values,
                "joint" : joint.flatten(1),
                "power_balance" : torch.stack([rp,rq], dim=-1).square().mean(),
                "demand_seen_by_physics_mw" : bus_x[:,0].reshape(k,horizon,self.grid.n_bus)*self.base_mva
                }

    def demand_only_output(self, batch, bus_innovation, gen_innovation):
        """Detach GENCO; perturb final states and rebuild differentiable physics."""
        from gridfm_graphkit.datasets.globals import MIN_VM_H, MAX_VM_H, MIN_PG, MAX_PG
        from torch_scatter import scatter_add
        with torch.no_grad():
            raw = self.unconditioned(batch)
        voltage = raw["bus"][:,:2]+bus_innovation
        vm = voltage[:,0].clamp(batch["bus"].x[:,MIN_VM_H],batch["bus"].x[:,MAX_VM_H])
        voltage = torch.stack([vm,voltage[:,1]],dim=-1)
        pg = (raw["gen"]+gen_innovation).clamp(batch["gen"].x[:,MIN_PG,None],batch["gen"].x[:,MAX_PG,None])
        edge = batch["bus","connects","bus"]
        p,q = self.backbone.branch_flow_layer(voltage,edge.edge_index,edge.edge_attr)
        pi,qi = self.backbone.node_injection_layer(p,q,edge.edge_index,len(voltage))
        generator_bus = batch["gen","connected_to","bus"].edge_index[1]
        aggregated = scatter_add(pg.squeeze(-1),generator_bus,dim=0,dim_size=len(voltage))
        bus = self.backbone.physics_decoder(pi,qi,voltage,batch["bus"].x,aggregated,batch.mask_dict)
        return {"bus":bus,"gen":pg}
