# V2 implementation inventory

Inspected 2026-10-25 on branch `multigrid`, HEAD `f991d1186cb5ba602d068d8ffea244e63c998fb6`. This is a planning inventory, not an implementation. Existing uncommitted edits were preserved; in particular, the launcher excerpts describe the working tree, not necessarily that commit alone. Source hashes below identify the inspected bytes. Only this report is written. No compute job, model run, package install, merge or publication was performed for this inventory.

The three proposed stages are: production determinism off; split one origin’s samples across tiles; batch multiple origins. **Current production still forces determinism on, has no distributed process group, and updates once per origin.** The profile bypassed production initialization to measure determinism off. Its 12-tile run trained 12 independent models, not one model distributed across 12 tiles.

Notation: `B` = origins per optimizer update, currently 1; `K` = trajectories per origin, currently 8; `H` = forecast hours, currently 24; `N` = buses; `G` = online generators; `E` = online physical branches. Case30 has `N=30, G=6, E=41` and 252 selected coordinates/hour. A frozen backbone still participates in input-gradient backpropagation through all 12 correction layers.

## 1. The training step, exactly

### 1.1 Full training_step and line annotations

Annotations below leave the quoted source unchanged: **R** touches random generation/state; **S** host/device transfer or scalar observation that can synchronize; **O** assumes one origin; **K** depends on the sample count, directly or through its tensors. A dash means no direct dependency at that line; it does not assert that every called kernel is asynchronous.

**Source [training/loop.py:65–81](/home/siebenschuh/Projects/genco-foresight/training/loop.py:65) — verbatim:**

```python
def training_step(model,optimizer,scheduler,template,cache,identity,cfg,device):
    history,calendar,target=cache.window(identity,device)
    noise=model.noise(cfg["stage_b"]["training_samples"],24)
    # A chained digest audits the actual sequence and sampled innovations across restarts.
    noise_digest=tensor_digest(noise)
    optimizer.zero_grad(set_to_none=True)
    result=model(history,calendar,template,noise)
    energy=fair_energy_score(result["joint"],target,model.coordinate_scale.repeat(24))
    balance=result["power_balance"]
    loss=energy+cfg["stage_b"]["loss"]["power_balance_weight"]*balance
    if not torch.isfinite(loss):raise RuntimeError("Nonfinite training loss")
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
        cfg["stage_b"]["optimizer"]["gradient_clip_norm"],error_if_nonfinite=True)
    optimizer.step();scheduler.step()
    return {"energy":float(energy.detach()),"balance":float(balance.detach()),"loss":float(loss.detach()),
        "gradient_norm":float(norm),"noise_sha256":noise_digest}
```

| Line | Tags | Exact operation / consequence |
|---:|---|---|
| 65 | O | Signature accepts one `identity`. |
| 66 | S, O | `cache.window(identity,device)` slices one history and copies its tensors with `.to(device)`; default `non_blocking=False`. |
| 67 | R, O, K | `model.noise(...,24)` draws one K-sample ensemble, with literal horizon 24. |
| 68 | — | Comment describing the following digest. |
| 69 | S, K | `tensor_digest(noise)` copies every noise tensor to CPU and hashes its bytes. Case30 K=8 noise payload is 73,984 bytes. |
| 70 | — | Clears trainable gradients. |
| 71 | O, K | Adapter accepts one origin’s history and K×24-graph template. No extra RNG draw occurs in forward when explicit noise is supplied; backbone stays in eval mode. |
| 72 | S, O, K | One `[K,D]` ensemble vs one `[D]` target; literal `.repeat(24)`. Scorer tests a device boolean and performs data-dependent boolean indexing. |
| 73 | O, K | Reads the scalar mean balance penalty over this origin’s samples/hours/buses/P,Q. |
| 74 | O, K | Combines that origin’s ES and scalar balance penalty. ES coefficient is hardcoded 1; only the balance coefficient is read. |
| 75 | S | Python truth test of scalar `torch.isfinite(loss)`. |
| 76 | O, K | Backpropagates this one-origin ensemble loss, including input gradients through frozen GENCO. |
| 77–78 | S | `error_if_nonfinite=True` makes gradient clipping inspect a device scalar; see installed implementation below. |
| 79 | O | One optimizer/scheduler update per origin. Optimizer state mutation is not itself random sampling; no explicit XPU scalar extraction is written here. |
| 80 | S, O | Three `float(...detach())` conversions copy/observe loss scalars on the host. |
| 81 | S, R, O, K | `float(norm)` observes the norm; returned `noise_sha256` records the already-drawn noise, without advancing RNG. |

Backward returns before all asynchronously launched device work must finish; the scalar reads subsequently impose dependencies. The profiler adds explicit phase-boundary synchronization that production does not have on each phase.

**Source [/opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0/lib/python3.12/site-packages/torch/nn/utils/clip_grad.py:105–115](/opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0/lib/python3.12/site-packages/torch/nn/utils/clip_grad.py:105) — verbatim:**

```python
    total_norm = torch.linalg.vector_norm(
        torch.stack([norm.to(first_device) for norm in norms]), norm_type
    )

    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )
```

### 1.2 Full adapter forward/noise and the supporting hooks

GENCO weights are frozen, but execution is not under `no_grad()` in adapter training. Node-type GRUs, observed-value projections, temporal conditioning, future-calendar projection and distribution heads are trained. There are 91,461 trainable parameters; the state dict including adapter buffers has a different count. The 384-dimensional cached embeddings are final correction-layer head inputs; the conditioning hook output is 48-dimensional.

**Source [models/temporal.py:18–55](/home/siebenschuh/Projects/genco-foresight/models/temporal.py:18) — verbatim:**

```python
    def __init__(self, base, network, scales):
        super().__init__()
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
```

**Source [models/temporal.py:79–138](/home/siebenschuh/Projects/genco-foresight/models/temporal.py:79) — verbatim:**

```python
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
```

**Source [models/temporal.py:140–225](/home/siebenschuh/Projects/genco-foresight/models/temporal.py:140) — verbatim:**

```python
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
        bc = self.bus_conditioning(bus_memory).unsqueeze(0).expand(k,-1,-1,-1).reshape(-1,self.hidden_width)
        gc = self.gen_conditioning(gen_memory).unsqueeze(0).expand(k,-1,-1,-1).reshape(-1,self.hidden_width)
        bus_innovation = torch.stack([bus_z[...,1],bus_z[...,2]*self.free_va],dim=-1).reshape(-1,2)
        gen_innovation = (gen_z*self.free_pg[None,None,:,None]).reshape(-1,1)

        # run frozen GENCO with differentiable additions
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
```

### 1.3 Case30 K=8 shapes, in execution order

The literal shape numbers below are derived from the quoted operations and the case30 cache/scale metadata, not from a new forward run. `h` denotes the 72 observed hours; `H` the 24 forecast hours.

| Stage | Bus tensor | Generator tensor / shared tensor | Source |
|---|---|---|---|
| Cached embedding history | `[72,30,384]` | `[72,6,384]` | `training/inputs.py:52–54,79–82` |
| Observed history / availability | `[72,30,4]` float32 / bool | `[72,6,2]` float32 / bool | `training/inputs.py:58–66` |
| Normalized observations + masks | `[72,30,8]` | `[72,6,4]` | `models/temporal.py:93–100` |
| Observed projection | `[72,30,32]` | `[72,6,32]` | `models/temporal.py:30–35,100` |
| Concatenation, transpose for GRU | `[30,72,416]` | `[6,72,416]` | `models/temporal.py:103` |
| GRU hidden, then hidden[0] | `[1,30,32]` → `[30,32]` | `[1,6,32]` → `[6,32]` | `models/temporal.py:104–105` |
| Future calendar | — | `[24,6]` → `[24,32]` | `models/temporal.py:149` |
| Node memory plus calendar | `[24,30,32]` | `[24,6,32]` | `models/temporal.py:150–151` |
| Distribution heads | `[24,30,28]` | `[24,6,9]` | `models/temporal.py:152` |
| Factor/diagonal parameters | `[24,30,3,9]` | `[24,6,1,9]` | `models/temporal.py:155–156` |
| Noise | `[8,24,30,3]` | `[8,24,6,1]`; shared factor `[8,8]` | `models/temporal.py:108–118` |
| Innovations | `bus_z=[8,24,30,3]` | `gen_z=[8,24,6,1]` | `models/temporal.py:155–156` |
| Sampled Pd/Qd | each `[8,24,30]` | positive-demand center `[30]` | `models/temporal.py:159–170` |
| Flattened graph batch | 192 graphs; bus x `[5760,15]` | gen x `[1152,6]`; directed bus edges `[2,15744]` (192×82), gen→bus `[2,1152]` and reverse | `models/genco.py:66–68`; `models/temporal.py:143–176`; network dimensions from profile |
| Conditioning projections/repetition | `[24,30,48]` → `[5760,48]` | `[24,6,48]` → `[1152,48]` | `models/temporal.py:179–180` |
| Final-readout innovations | `[5760,2]` for Vm/Va | `[1152,1]` for Pg | `models/temporal.py:181–182` |
| GENCO input projection + conditioning | `[5760,48]` | `[1152,48]` | `models/temporal.py:130–131`; upstream forward below |
| Correction-layer head input | `[5760,384]` | `[1152,384]` | upstream `hidden_dim×heads`, final head inputs |
| Final bounded/decoded output | `[5760,4]` = Vm,Va,bus Pg,bus Qg | `[1152,1]` = online-generator Pg | upstream forward below |
| Physical flows / bus residuals | p,q each `[15744]`; rp,rq each `[5760]` | stacked residuals `[5760,2]` | `models/temporal.py:189–192,223` |
| Reshaped state/flow | bus `[8,24,30,4]`; directed flows `[8,24,82]` | Pg/Qg `[8,24,6]`; each terminal flow field `[8,24,41]` | `models/temporal.py:195–201` |
| Selected joint state | `[8,24,252]` → **`[8,6048]`** | target `[6048]`; one-hour scale `[252]` repeated to `[6048]` | `models/temporal.py:214–224`; `training/loop.py:72` |

The 252 coordinates are Pd 21, Qd **0**, Vm 30, Va 29, Pg **2 free generators**, aggregated bus Qg 6, and four terminal-flow families ×41 = 164. Loading is not in joint ES. Qd is derived from Pd and excluded to avoid duplicating fixed-power-factor demand coordinates. Evidence: `/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_a_v4/stage_a_followup/train_coordinate_scales.json`; mask rule quoted next.

**Source [data/coordinates.py:30–62](/home/siebenschuh/Projects/genco-foresight/data/coordinates.py:30) — verbatim:**

```python
def fit_scales(grid, records):
    """Fit the released scale rule to training labels only, keeping offline masks."""
    values = [record_values(grid, r) for r in records]
    result = dict(features={}, coordinate_order=[], scale_vector_one_hour=[], generator_history={})
    for key in values[0]:
        if key == "loading": continue
        array = np.asarray([v[key] for v in values])
        mask = np.isfinite(array).any(0)
        mean = np.divide(np.nansum(array, axis=0), np.isfinite(array).sum(0),
                         out=np.zeros(array.shape[1]), where=mask)
        std = np.sqrt(np.divide(np.nansum((array-mean)**2, axis=0), np.isfinite(array).sum(0),
                                out=np.zeros(array.shape[1]), where=mask))
        floor = .01 if key in ("vm", "va_rad") else 1.
        scale = np.maximum(std, floor)
        if key == "pd_mw": mask &= grid.active_demand
        elif key == "qd_mvar": mask[:] = False
        elif key == "va_rad": mask[grid.reference_bus] = False
        elif key == "pg_mw": mask &= grid.online_gen & (grid.gen[:, 8] > grid.gen[:, 9])
        result["features"][key] = dict(mean=mean.tolist(), population_std=std.tolist(), floor=floor,
                                        scale=scale.tolist(), joint_score_mask=mask.tolist())
        for i in np.flatnonzero(mask):
            result["coordinate_order"].append(f"{key}:{i+1}")
            result["scale_vector_one_hour"].append(float(scale[i]))
    for key in ("pg_mw", "qg_mvar"):
        array = np.array([(r.get("target_state") or {}).get(key, [np.nan]*grid.n_gen) for r in records], dtype=float)
        valid = np.isfinite(array).sum(0)
        mean = np.divide(np.nansum(array, axis=0), valid, out=np.zeros(grid.n_gen), where=valid>0)
        std = np.sqrt(np.divide(np.nansum((array-mean)**2, axis=0), valid, out=np.zeros(grid.n_gen), where=valid>0))
        result["generator_history"][key] = dict(mean=mean.tolist(), scale=np.maximum(std, 1.).tolist())
    result.update(dimension_one_hour=len(result["coordinate_order"]),
                  dimension_24_hours=24*len(result["coordinate_order"]),
                  qg_coordinate_bus_ids=grid.bus_ids[grid.qg_buses].tolist(), fit_partition="train")
    return result
```

**Innovation placement matters for both parallel changes.** Temporal memory is added after `input_proj_bus/gen` (latent conditioning). Demand innovation enters the softplus Pd construction, then each sample’s demand is written into its own graph. Vm/Va/Pg innovations are added to the **last** shared `mlp_bus/gen` readout invocation, not to hidden latents; upstream masks, sigmoid bounds and physics reconstruction then act on those readouts. Qg and flows are derived, not independently sampled outputs. The shared eight-vector factor couples hours, nodes and coordinate types within each trajectory; different trajectories need distinct draws.

**Sample split communication inventory (proposed, not implemented):** rank r can run only its `K_r×24` graphs. To call the existing scorer unchanged, gather differentiable `joint` rows `[K_r,6048]` into `[K,6048]` on a loss-computing rank (or all ranks). That is 193,536 float32 bytes for the full case30 K=8 sample matrix, excluding collective overhead. Every loss-computing rank also needs the same target, observed-coordinate mask and coordinate scales. The balance term needs a differentiable reduction of squared-residual sums and element counts, not a gather of full intermediate graph states. Raw Vm/Pg/flow arrays, 12-layer latents, edges and history embeddings need not be gathered for the loss. History/calendar, initial adapter state and sample-to-rank/RNG assignment must agree across replicas. Adapter gradients then need an explicitly normalized reduction. Frozen backbone parameters do not need gradients, but its computations stay on the autograd path.

### 1.4 Graph construction, conditioning, embeddings

The template is constructed once per initialization. Each step clones it and changes demand features; connectivity/admittance is not recomputed or rebuilt from the network. However, `Batch.clone()` clones tensor attributes, including edge-index/edge-attribute tensors: this is not zero-copy reuse. Edge feature projections and message passing still execute inside every GENCO call.

**Source [training/loop.py:40–57](/home/siebenschuh/Projects/genco-foresight/training/loop.py:40) — verbatim:**

```python
def initialize(cfg,device):
    torch.use_deterministic_algorithms(True)
    seed_all(cfg["seed"])
    torch.set_num_threads(1 if device=="xpu" else 16)
    net=load_network(config=cfg)
    load_temporal_manifest(cfg["paths"]["temporal_root"],net)
    base=Genco(checkpoint_assets(cfg),cfg["checkpoint"],device,grid=net)
    scales=read_json(ROOT/cfg["stage_b"]["coordinate_scales"])
    model=TemporalAdapter(base,net,scales).to(device).train()
    opt_cfg=cfg["stage_b"]["optimizer"]
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=opt_cfg["lr"],weight_decay=opt_cfg["weight_decay"],foreach=False)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda step:1.0)
    graph=graph_for_demand(net,net["bus"][:,2],net["bus"][:,3])
    template=base.batch([graph]*cfg["stage_b"]["future_graph_batch_size"])
    assert template.num_graphs==cfg["stage_b"]["training_samples"]*cfg["task"]["horizon_hours"]
    cache=CachedWindows(cfg,net)
    return model,optimizer,scheduler,template,cache
```

**Source [models/genco.py:24–80](/home/siebenschuh/Projects/genco-foresight/models/genco.py:24) — verbatim:**

```python
    def __init__(self, checkpoint_root, settings, device="cpu", grid=None):
        """Load model config"""
        # check config
        folder = Path(checkpoint_root) / settings["directory"]
        config_path = folder / settings["configuration"]
        raw = yaml.safe_load(config_path.read_text())

        # assign model
        self.config = resolve_config(raw, config_path=str(config_path), conversion_mode="auto_inline")
        self.args = NestedNamespace(**self.config)
        self.case = settings["directory"].split("/")[0]
        networks = self.args.data.networks
        if list(networks) != [self.case]:
            raise ValueError(f"Case mismatch: checkpoint YAML has {networks!r}; expected {[self.case]!r}")
        if grid is not None:
            check_case(grid.case, self.case, "checkpoint YAML")
        self.device = torch.device(device)
        self.grid = grid
        self.model = GNS_heterogeneous(self.args)
        state = torch.load(folder / settings["weights"], map_location="cpu", weights_only=True)
        keys = {k.replace("model._orig_mod.", "model.").removeprefix("model."): v for k,v in state.items()}
        self.model.load_state_dict(keys, strict=True)
        self.model.to(self.device).eval().requires_grad_(False)

        # data preprocessing
        stats = torch.load(folder / settings["normalizer"], map_location="cpu", weights_only=True)
        if self.case not in stats:
            raise ValueError(f"Case mismatch: normalizer has no statistics for {self.case!r}")
        self.normalizer = HeteroDataMVANormalizer(self.args)
        self.normalizer.fit_from_dict(stats[self.args.data.networks[0]])
        self.transform = OptimalPowerFlowTransforms(self.args)

    def prepare(self, graph):
        """Use Genco preprocessing"""
        check_case(self.case, getattr(graph, "case_id", None), "input graph")
        graph = graph.clone()
        self.normalizer.transform(graph)
        graph = self.transform(graph)
        for kind in ("bus", "gen"):
            graph[kind].label_available = torch.isfinite(graph[kind].y)
        return graph

    def batch(self, graphs):
        """Preps each graph, combines them into one PyTorch Geometric batc"""
        return Batch.from_data_list([self.prepare(g) for g in graphs]).to(self.device)

    def predict(self, graphs, return_embeddings=False):
        """Run mode eval-style"""
        batch = self.batch(graphs)
        with torch.no_grad():
            output = self.model(batch, return_embeddings=return_embeddings)
        if return_embeddings:
            output, embeddings = output
        self.normalizer.inverse_output(output, batch)
        if return_embeddings:
            return output, embeddings, batch
        return output
```

**Source [/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/lib/python3.12/site-packages/torch_geometric/data/data.py:346–350](/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/lib/python3.12/site-packages/torch_geometric/data/data.py:346) — verbatim:**

```python
    def clone(self, *args: str):
        r"""Performs cloning of tensors, either for all attributes or only the
        ones given in :obj:`*args`.
        """
        return copy.copy(self).apply(lambda x: x.clone(), *args)
```

**Source [/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/lib/python3.12/site-packages/gridfm_graphkit/models/gnn_heterogeneous_gns.py:175–181](/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/lib/python3.12/site-packages/gridfm_graphkit/models/gnn_heterogeneous_gns.py:175) — verbatim:**

```python
        When ``return_embeddings`` is ``True``, returns ``(predictions, embeddings)``
        instead of ``predictions`` alone. Each embedding is the latent tensor fed
        directly into the corresponding prediction head: ``embeddings["bus"]`` is
        the ``h_bus`` consumed by ``mlp_bus`` and ``embeddings["gen"]`` is the
        ``h_gen`` consumed by ``mlp_gen`` on the final layer. (Note the last-layer
        ``physics_mlp`` update to ``h_bus`` is intentionally skipped, so the
        returned bus embedding is exactly the head input, not a post-update state.)
```

**Source [/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/lib/python3.12/site-packages/gridfm_graphkit/models/gnn_heterogeneous_gns.py:190–230](/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/lib/python3.12/site-packages/gridfm_graphkit/models/gnn_heterogeneous_gns.py:190) — verbatim:**

```python
        # 1) initial projections
        h_bus = self.input_proj_bus(x_dict["bus"])  # [num_bus, hidden_dim]
        h_gen = self.input_proj_gen(x_dict["gen"])  # [num_gen, hidden_dim]

        num_bus = x_dict["bus"].size(0)
        _, gen_to_bus_index = edge_index_dict[("gen", "connected_to", "bus")]
        bus_edge_index = edge_index_dict[("bus", "connects", "bus")]
        bus_edge_attr = edge_attr_dict[("bus", "connects", "bus")]

        edge_attr_proj_dict = {}
        for key, edge_attr in edge_attr_dict.items():
            if edge_attr is not None:
                edge_attr_proj_dict[key] = self.input_proj_edge(edge_attr)
            else:
                edge_attr_proj_dict[key] = None

        bus_mask = mask_dict["bus"][:, VM_H : VA_H + 1]
        gen_mask = mask_dict["gen"][:, : (PG_H + 1)]
        bus_fixed = x_dict["bus"][:, VM_H : VA_H + 1]
        gen_fixed = x_dict["gen"][:, : (PG_H + 1)]

        # iterate layers
        for i, conv in enumerate(self.layers):
            out_dict = conv(
                {"bus": h_bus, "gen": h_gen},
                edge_index_dict,
                edge_attr_proj_dict,
            )
            out_bus = out_dict["bus"]  # [Nb, hidden_dim * heads]
            out_gen = out_dict["gen"]  # [Ng, hidden_dim * heads]

            out_bus = self.activation(self.norms_bus[i](out_bus))
            out_gen = self.activation(self.norms_gen[i](out_gen))

            # skip connection
            h_bus = h_bus + out_bus if out_bus.shape == h_bus.shape else out_bus
            h_gen = h_gen + out_gen if out_gen.shape == h_gen.shape else out_gen

            # Decode bus and generator predictions
            bus_temp = self.mlp_bus(h_bus)  # [Nb, 2]  -> Vm, Va
            gen_temp = self.mlp_gen(h_gen)  # [Ng, 1]  -> Pg
```

**Source [/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/lib/python3.12/site-packages/gridfm_graphkit/models/gnn_heterogeneous_gns.py:254–268](/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/lib/python3.12/site-packages/gridfm_graphkit/models/gnn_heterogeneous_gns.py:254) — verbatim:**

```python
            else:
                bus_temp = torch.where(bus_mask, bus_temp, bus_fixed)
                gen_temp = torch.where(gen_mask, gen_temp, gen_fixed)

                if self.task == "OptimalPowerFlow":
                    bus_temp[:, VM_OUT] = bound_with_sigmoid(
                        bus_temp[:, VM_OUT],
                        x_dict["bus"][:, MIN_VM_H],
                        x_dict["bus"][:, MAX_VM_H],
                    )
                    gen_temp[:, PG_OUT_GEN] = bound_with_sigmoid(
                        gen_temp[:, PG_OUT_GEN],
                        x_dict["gen"][:, MIN_PG],
                        x_dict["gen"][:, MAX_PG],
                    )
```

**Source [/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/lib/python3.12/site-packages/gridfm_graphkit/models/gnn_heterogeneous_gns.py:311–322](/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/lib/python3.12/site-packages/gridfm_graphkit/models/gnn_heterogeneous_gns.py:311) — verbatim:**

```python
                # On the last layer the updated h_bus is never read again: the
                # loop ends, predictions come from output_temp/gen_temp, and the
                # exported embedding is the tensor that fed mlp_bus. Skipping the
                # update avoids a dead forward pass and makes h_bus itself the
                # correct embedding to return (no snapshot/clone needed).
                if i < self.num_layers - 1:
                    h_bus = h_bus + self.physics_mlp(bus_residuals)

        predictions = {"bus": output_temp, "gen": gen_temp}
        if not return_embeddings:
            return predictions
        return predictions, {"bus": h_bus, "gen": h_gen}
```

`GNS_heterogeneous.forward(..., return_embeddings=True)` returns `(predictions, {"bus": h_bus, "gen": h_gen})`; the wrapper `Genco.predict` instead returns `(inverse-normalized predictions, embeddings, batch)`. Training cache generation calls the former directly under `no_grad`. The adapter’s unconditioned method delegates to upstream without hooks. These are different APIs; a parallel runner must not mistake the wrapper’s three-tuple for the upstream two-tuple.

### 1.5 Full fair energy score and balance loss

The current scorer requires all K rows in one input tensor on each rank that invokes it. Its pairwise matrix is `[K,K]` after reducing the `[K,K,D_observed]` differences. **For this implementation, yes: all K must be available together on that rank.** This is not a mathematical requirement that an exact ES can only run on one rank: a new blockwise implementation could reduce all cross-rank pairs. Averaging local fair ES values is not equivalent, because it omits cross-rank pairs and changes denominators. K=1 is invalid for this scorer, although a rank may hold one local sample if a future global gather supplies K≥2 to the actual loss.

**Source [scoring.py:6–18](/home/siebenschuh/Projects/genco-foresight/scoring.py:6) — verbatim:**

```python
def fair_energy_score(samples, target, scale, mask=None):
    """IID-ensemble U-statistic, all off-diagonal pairs, identical coordinate masks."""
    if samples.ndim != 2 or samples.shape[0] < 2:
        raise ValueError("Expected at least two complete flattened trajectory samples")
    observed = torch.isfinite(target) if mask is None else mask & torch.isfinite(target)
    if not observed.any():
        raise ValueError("No observed target coordinates")
    k = samples.shape[0]
    x = samples[:,observed]/scale[observed]
    y = target[observed]/scale[observed]
    distance_to_target = torch.linalg.vector_norm(x-y,dim=-1).mean()
    pair_distance = torch.linalg.vector_norm(x[:,None]-x[None,:],dim=-1)
    return (distance_to_target-pair_distance.sum()/(2*k*(k-1)))/observed.sum().sqrt()
```

**Source [/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/lib/python3.12/site-packages/gridfm_graphkit/models/utils.py:75–79](/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/lib/python3.12/site-packages/gridfm_graphkit/models/utils.py:75) — verbatim:**

```python
def compute_shunt_power(bus_data_pred, bus_data_orig):
    """Compute active/reactive shunt power contributions per bus."""
    p_shunt = -bus_data_orig[:, GS] * bus_data_pred[:, VM_OUT] ** 2
    q_shunt = bus_data_orig[:, BS] * bus_data_pred[:, VM_OUT] ** 2
    return p_shunt, q_shunt
```

**Source [/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/lib/python3.12/site-packages/gridfm_graphkit/models/utils.py:179–190](/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/lib/python3.12/site-packages/gridfm_graphkit/models/utils.py:179) — verbatim:**

```python
class ComputeNodeResiduals(nn.Module):
    """Compute net residuals per bus combining branch flows, generators, loads, and shunts."""

    def forward(self, P_in, Q_in, bus_data_pred, bus_data_orig):
        # Shunt contributions
        p_shunt, q_shunt = compute_shunt_power(bus_data_pred, bus_data_orig)

        # Net residuals per bus
        residual_P = bus_data_pred[:, PG_OUT] - bus_data_orig[:, PD_H] + p_shunt - P_in
        residual_Q = bus_data_pred[:, QG_OUT] - bus_data_orig[:, QD_H] + q_shunt - Q_in

        return residual_P, residual_Q
```

**Source [models/temporal.py:188–192](/home/siebenschuh/Projects/genco-foresight/models/temporal.py:188) — verbatim:**

```python
        # eval branch flows, nodal injections, and balance against sampled demand
        edge = batch["bus","connects","bus"]
        p,q = self.backbone.branch_flow_layer(output["bus"],edge.edge_index,edge.edge_attr)
        pi,qi = self.backbone.node_injection_layer(p,q,edge.edge_index,len(bus_x))
        rp,rq = self.backbone.node_residuals_layer(pi,qi,output["bus"],bus_x)
```

**Source [models/temporal.py:221–225](/home/siebenschuh/Projects/genco-foresight/models/temporal.py:221) — verbatim:**

```python
        return {"values" : values,
                "joint" : joint.flatten(1),
                "power_balance" : torch.stack([rp,rq], dim=-1).square().mean(),
                "demand_seen_by_physics_mw" : bus_x[:,0].reshape(k,horizon,self.grid.n_bus)*self.base_mva
                }
```

**Source [training/loop.py:72–81](/home/siebenschuh/Projects/genco-foresight/training/loop.py:72) — verbatim:**

```python
    energy=fair_energy_score(result["joint"],target,model.coordinate_scale.repeat(24))
    balance=result["power_balance"]
    loss=energy+cfg["stage_b"]["loss"]["power_balance_weight"]*balance
    if not torch.isfinite(loss):raise RuntimeError("Nonfinite training loss")
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
        cfg["stage_b"]["optimizer"]["gradient_clip_norm"],error_if_nonfinite=True)
    optimizer.step();scheduler.step()
    return {"energy":float(energy.detach()),"balance":float(balance.detach()),"loss":float(loss.detach()),
        "gradient_norm":float(norm),"noise_sha256":noise_digest}
```

`rp,rq` are in the checkpoint’s per-unit/MVA-normalized coordinates. The scalar penalty is `sum(rp²+rq²)/(2*K*H*N)`, not the displayed mean absolute MW residual used in evaluation. The equivalent physical formula divides MW/MVAr residual squares by `base_mva²` before averaging. All buses participate; no separate residual mask is passed. For one origin split across ranks, weighting local means by `K_r` (equal H,N) preserves the value, including uneven splits. For origin batching, ES must be computed separately for each origin’s target/mask, then averaged over origins; do not mix origins into the K axis.

## 2. Determinism and RNG

### 2.1 RNG state, digest, and all first-party call sites

`seed_all` seeds Python, NumPy and torch. Noise sampling uses the model device’s default generator. Epoch shuffling and validation each use separate explicitly seeded CPU generators. The per-step noise digest does not draw randomness, but copies XPU noise to CPU. All ranks blindly using the same seed and local draw shapes would duplicate trajectories; global sample IDs or explicitly scattered global draws are required for sample parallelism.

**Source [training/state.py:20–44](/home/siebenschuh/Projects/genco-foresight/training/state.py:20) — verbatim:**

```python
def tensor_digest(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def rng_state():
    name,keys,pos,gauss,cached=np.random.get_state()
    return {"python":random.getstate(),"numpy":{"name":name,"keys":keys.tolist(),"pos":pos,"gauss":gauss,"cached":cached},
        "torch":torch.get_rng_state(),
        "xpu":torch.xpu.get_rng_state_all() if torch.xpu.is_available() else [],
        "cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    n=state["numpy"];np.random.set_state((n["name"],np.array(n["keys"],dtype=np.uint32),n["pos"],n["gauss"],n["cached"]))
    torch.set_rng_state(state["torch"])
    if state["xpu"] and torch.xpu.is_available(): torch.xpu.set_rng_state_all(state["xpu"])
    if state["cuda"] and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state["cuda"])
```

**Source [training/state.py:82–89](/home/siebenschuh/Projects/genco-foresight/training/state.py:82) — verbatim:**

```python
def epoch_order(seed, epoch, count):
    return torch.randperm(count,generator=torch.Generator().manual_seed(seed+epoch)).tolist()


def next_progress():
    return {"epoch":0,"step_in_epoch":0,"steps":0,"best_selection":None,"bad_epochs":0,
        "best_epoch":None,"epochs":[],"cumulative_training_wall_seconds":0.,"finished":False,
        "stop_reason":None,"step_sequence_sha256":hashlib.sha256(b"").hexdigest()}
```

**Source [training/loop.py:25–37](/home/siebenschuh/Projects/genco-foresight/training/loop.py:25) — verbatim:**

```python
def make_manifest(cfg):
    dataset=Path(cfg["paths"]["temporal_root"])
    return {"mode":"A","dataset_id":cfg["dataset"]["id"],"dataset_manifest_sha256":sha256(dataset/"manifest.json"),
        "benchmark_id":read_json(dataset/"manifest.json")["benchmark_id"],"base_checkpoint":cfg["checkpoint"],
        "graphkit":cfg["upstream"]["graphkit"],"seed":cfg["seed"],"settings":cfg["stage_b"],
        "coordinate_scales_sha256":sha256(ROOT/cfg["stage_b"]["coordinate_scales"]),
        "training_origins":cfg["task"]["split_episodes"]["train"]*409,
        "selection_origins":cfg["task"]["split_episodes"]["model_selection"]*18,
        "deterministic_algorithms":True,
        "python_hash_seed":os.environ.get("PYTHONHASHSEED"),
        "source_sha256":{p:sha256(ROOT/p) for p in ("models/temporal.py","scoring.py","training/state.py","training/inputs.py","training/loop.py")},
        "budget_definition":"Cumulative worker-job wall time including startup, validation and checkpointing; excludes queue wait, separate cache preparation and resume tests",
        "rng_protocol":"Restore Python/NumPy/Torch CPU/XPU/CUDA states; deterministic epoch permutation; independent fixed selection-noise generator"}
```

The inventory below scans first-party Python/PBS/shell files under `models`, `training`, `data`, `solver`, `scripts`, `tests`, plus root Python files. Saved run snapshots, third-party installations and Markdown examples are excluded. Lines are exact source lines; definitions/imports are included to make the call-site inventory auditable. Explicit adapter `randn` and random head initialization are shown in §1.2. There is no production config toggle for deterministic algorithms today.

**scripts/adapter_preflight.py — matching source lines:**

```text
29: def tensor_digest(state):
91:     torch.manual_seed(cfg["seed"])
107:     torch.manual_seed(43)
110:     base_before=tensor_digest(model.backbone.state_dict())
111:     adapter_before=tensor_digest({key:value for key,value in model.state_dict().items() if not key.startswith("backbone.")})
151:     base_after=tensor_digest(model.backbone.state_dict())
153:     adapter_after=tensor_digest({key:value for key,value in model.state_dict().items() if not key.startswith("backbone.")})
```

**scripts/diagnose_stage_b.py — matching source lines:**

```text
304:     from training.state import tensor_digest
321:     tensor_sha = tensor_digest(base.model.state_dict())
327:         "adapter_state_tensor_bytes_sha256": tensor_digest(state),
```

**scripts/evaluate_stage_b.pbs — matching source lines:**

```text
21: export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42
```

**scripts/evaluate_stage_b.py — matching source lines:**

```text
27: from training.state import atomic_torch_save,tensor_digest
96:     torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
146:     backbone=tensor_digest(model.backbone.state_dict())
236:     base_before=tensor_digest(base.model.state_dict());ridge=RidgeForecaster(root/"forecaster")
266:                     generator=torch.Generator().manual_seed(seed);chunks=[]
290:             assert tensor_digest(base.model.state_dict())==base_before
299:         "wall_seconds":time.time()-started,"backbone_unchanged":tensor_digest(base.model.state_dict())==base_before})
```

**scripts/evaluate_variant.py — matching source lines:**

```text
71:     from training.state import tensor_digest, train_state
76:         "base_checkpoint_file_sha256": sha256(weights), "base_named_tensor_bytes_sha256": tensor_digest(base.model.state_dict()),
```

**scripts/finalize_release.py — matching source lines:**

```text
52:         from training.state import tensor_digest
68:         assert tensor_digest(exported) == tensor_digest(state)
82:             "adapter_named_tensor_bytes_sha256": tensor_digest(state),
```

**scripts/forecast_partition.py — matching source lines:**

```text
25: from training.state import atomic_torch_save, tensor_digest
43:     torch.use_deterministic_algorithms(True)
58:     assert tensor_digest(base.model.state_dict()) == read_json(train / "status.json")["base_sha256"]
72:     generator = torch.Generator().manual_seed(seed)
175:     base_digest = tensor_digest(base.model.state_dict())
240:             assert tensor_digest(base.model.state_dict()) == base_digest
```

**scripts/hybrid_stage_c.py — matching source lines:**

```text
19: from training.state import tensor_digest
52:     digest = tensor_digest(base.model.state_dict())
127:             assert tensor_digest(base.model.state_dict()) == digest
```

**scripts/multigrid/profile_step.py — matching source lines:**

```text
196:         from training.state import seed_all, tensor_digest
199:         torch.use_deterministic_algorithms(args.deterministic == "on")
200:         seed_all(42)
213:         base_before = tensor_digest(model.backbone.state_dict())
272:         record["backbone_unchanged"] = tensor_digest(model.backbone.state_dict()) == base_before
300:                    OPENBLAS_NUM_THREADS="16" if device == "cpu" else "1", PYTHONHASHSEED="42")
507: export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42
```

**scripts/package_reproducibility.py — matching source lines:**

```text
12: from training.state import tensor_digest
83:     assert tensor_digest(state) == diagnostic["adapter_state_tensor_bytes_sha256"]
103:         "adapter_named_tensor_bytes_sha256": tensor_digest(state),
```

**scripts/replication_chain.py — matching source lines:**

```text
221:     os.environ.update(GENCO_CONFIG=str(config),PYTHONHASHSEED=str(c['seed']),ZE_FLAT_DEVICE_HIERARCHY='FLAT',ZE_AFFINITY_MASK=str(c['tile']))
```

**scripts/replication_evaluate.py — matching source lines:**

```text
25: from training.state import tensor_digest
258:     base_before = tensor_digest(base.model.state_dict())
316:             assert tensor_digest(base.model.state_dict()) == base_before
```

**scripts/resume_test.py — matching source lines:**

```text
37:         env=dict(os.environ,GENCO_JOB_START_EPOCH=str(time.time()),PYTHONHASHSEED=str(cfg["seed"]))
```

**scripts/train_adapter.pbs — matching source lines:**

```text
22: export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42 PBS_WALLTIME=3600
```

**tests/test_adapter.py — matching source lines:**

```text
21:     torch.manual_seed(42)
```

**tests/test_training.py — matching source lines:**

```text
8: from training.state import atomic_torch_save,restore_rng,rng_state,seed_all
27:     seed_all(42);state=rng_state()
```

**training/loop.py — matching source lines:**

```text
18: from training.state import (checkpoint,epoch_order,next_progress,restore,seed_all,
19:                             tensor_digest,train_state)
34:         "python_hash_seed":os.environ.get("PYTHONHASHSEED"),
41:     torch.use_deterministic_algorithms(True)
42:     seed_all(cfg["seed"])
69:     noise_digest=tensor_digest(noise)
81:         "gradient_norm":float(norm),"noise_sha256":noise_digest}
91:                 g=torch.Generator().manual_seed(cfg["stage_b"]["selection_noise_seed"]+index)
130:     base_digest=tensor_digest(model.backbone.state_dict())
134:         progress["initial_adapter_sha256"]=tensor_digest(train_state(model))
178:                     repr(identity)+last_metrics["noise_sha256"]).encode()).hexdigest()
202:             if tensor_digest(model.backbone.state_dict())!=base_digest:raise RuntimeError("Frozen base weights changed")
215:     if tensor_digest(model.backbone.state_dict())!=base_digest:raise RuntimeError("Frozen base weights changed")
225:         "base_unchanged":True,"base_sha256":base_digest,"adapter_sha256":tensor_digest(train_state(model)),
```

**training/state.py — matching source lines:**

```text
20: def tensor_digest(state):
27: def seed_all(seed):
28:     random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
83:     return torch.randperm(count,generator=torch.Generator().manual_seed(seed+epoch)).tolist()
```

`PYTHONHASHSEED` takes effect when a Python interpreter starts. `replication_chain.worker` updates its environment before spawning the training/resume subprocesses; it does not retroactively change the hash seed of that already-running worker. Changing this environment setting is not a substitute for restoring torch/NumPy/Python RNG state.

### 2.2 What resume currently proves, and what is missing for a statistical gate

The resume trial uses the real loop. At the midpoint it writes a checkpoint and sends SIGKILL to itself, then starts another process. It does not test an arbitrary crash between checkpoints or guarantee preservation of uncheckpointed work.

**Source [scripts/resume_test.py:14–71](/home/siebenschuh/Projects/genco-foresight/scripts/resume_test.py:14) — verbatim:**

```python
def compare(a,b):
    if isinstance(a,torch.Tensor):
        equal=torch.equal(a,b)
        delta=float((a.to(torch.float64)-b.to(torch.float64)).abs().max()) if a.numel() else 0.
        return equal,delta
    if isinstance(a,dict):
        if a.keys()!=b.keys():return False,float('inf')
        pairs=[compare(a[k],b[k]) for k in a]
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b):return False,float('inf')
        pairs=[compare(x,y) for x,y in zip(a,b)]
    else:return a==b,0.
    return all(x[0] for x in pairs),max((x[1] for x in pairs),default=0.)


def trial(cfg,device,steps,root):
    def invoke(name,crash=None):
        destination=root/device/name
        if (destination/"status.json").exists():raise ValueError("Do not overwrite a completed resume trial")
        destination.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,"-m","scripts.train_adapter","--device",device,"--run-root",str(destination),
                 "--stop-after-steps",str(steps),"--resume-test"]
        if crash:command += ["--crash-at-steps",str(crash)]
        env=dict(os.environ,GENCO_JOB_START_EPOCH=str(time.time()),PYTHONHASHSEED=str(cfg["seed"]))
        with (destination/("interrupted.log" if crash else "run.log")).open("w") as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env=env)
        if result.returncode != (-9 if crash else 0):
            raise RuntimeError(f"Resume trial failed: {destination}; code {result.returncode}")
        return destination
    uninterrupted=invoke("uninterrupted")
    resumed=invoke("resumed",steps//2)
    killed=torch.load(resumed/"latest.pt",map_location="cpu",weights_only=True)
    assert killed["progress"]["steps"]==steps//2
    invoke("resumed")
    a=torch.load(uninterrupted/"latest.pt",map_location="cpu",weights_only=True)
    b=torch.load(resumed/"latest.pt",map_location="cpu",weights_only=True)
    report={"device":device,"total_steps_each_path":steps,"killed_with_sigkill_after_step":steps//2,
        "resumed_steps":steps-steps//2,"same_step_sequence":a["progress"]["step_sequence_sha256"]==b["progress"]["step_sequence_sha256"],
        "same_initial_weights":a["progress"]["initial_adapter_sha256"]==b["progress"]["initial_adapter_sha256"],
        "same_counters":all(a["progress"][k]==b["progress"][k] for k in ("epoch","step_in_epoch","steps","best_selection","bad_epochs"))}
    for key in ("model","optimizer","scheduler","rng"):
        exact,delta=compare(a[key],b[key]);report[key]={"bit_exact":exact,"max_absolute_difference":delta}
    report["passed"]=report["same_initial_weights"] and report["same_step_sequence"] and report["same_counters"] and all(report[k]["bit_exact"] for k in ("model","optimizer","scheduler","rng"))
    write_json(root/f"{device}.json",report)
    print(report,flush=True)
    return report


def main():
    cfg=load_config();root=Path(cfg["paths"]["training_run"])/"resume_test"
    root.mkdir(parents=True,exist_ok=True)
    build_cache(cfg,"xpu")
    # CPU fallback uses the same 192-graph path; the full 200-step test is on selected XPU.
    cpu=trial(cfg,"cpu",4,root)
    xpu=trial(cfg,"xpu",200,root)
    report={"passed":cpu["passed"] and xpu["passed"],"cpu":cpu,"xpu":xpu}
    write_json(root/"report.json",report)
    if not report["passed"]:raise RuntimeError("Restart comparison failed; training chain not released")
```

Recorded v4 evidence: `/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v4/resume_test/report.json` has `passed: true`: XPU 200 uninterrupted vs 100 + kill + 100, CPU only 4 vs 2 + kill + 2. Model, optimizer, scheduler and RNG have `bit_exact: true`, maximum tensor difference 0.0; initial weights, sequence digest and selected counters match. The comparison is numeric `torch.equal`, not a bytewise checkpoint-file comparison. It excludes wall-clock counters and does not compare every progress field. A 200-step trial does not reach a 4,090-step epoch boundary, so it does not exercise selection/early-stopping resume.

Existing logging and early-stop state are below. The step function returns energy, balance, total loss, gradient norm and noise hash each step, but the loop persistently logs **only total loss every 100 steps**, epoch selection ES, a rolling sequence digest, and the last step’s full metrics in each job record. It does not save a complete per-step loss/component curve.

**Source [training/loop.py:169–207](/home/siebenschuh/Projects/genco-foresight/training/loop.py:169) — verbatim:**

```python
        while not progress["finished"]:
            guard()
            epoch=progress["epoch"]
            order=epoch_order(cfg["seed"],epoch,len(cache.indices["train"]))
            while progress["step_in_epoch"]<len(order):
                guard()
                identity=cache.indices["train"][order[progress["step_in_epoch"]]]
                last_metrics=training_step(model,opt,sched,template,cache,identity,cfg,device)
                progress["step_sequence_sha256"]=hashlib.sha256((progress["step_sequence_sha256"]+
                    repr(identity)+last_metrics["noise_sha256"]).encode()).hexdigest()
                progress["steps"]+=1;progress["step_in_epoch"]+=1
                if progress["steps"]==steps_before+1:
                    startup_record["first_optimizer_step_completed_seconds"]=time.time()-job_start
                    write_json(root/"startup.json",startup_record)
                if progress["steps"]%100==0:print({"step":progress["steps"],"epoch":epoch,"loss":last_metrics["loss"]},flush=True)
                if crash_at_steps is not None and progress["steps"]==crash_at_steps:
                    save();os.kill(os.getpid(),9)
                if stop_after_steps is not None and progress["steps"]>=stop_after_steps:
                    progress["stop_reason"]="resume_test_step_limit";raise TimeGuard
                if time.time()-last_save>=cfg["stage_b"]["checkpoint_interval_seconds"]:
                    save();last_save=time.time()
            # A mid-validation interruption resumes this phase with the same fixed noise.
            score=selection_score(model,template,cache,cfg,device,guard)
            best=progress["best_selection"]
            improved=best is None or score<best
            if improved:progress.update(best_selection=score,best_epoch=epoch+1,bad_epochs=0)
            else:progress["bad_epochs"]+=1
            progress["epochs"].append({"epoch":epoch+1,"selection_energy_score":score,"steps":progress["steps"]})
            progress["epoch"]+=1;progress["step_in_epoch"]=0
            if progress["bad_epochs"]>=cfg["stage_b"]["early_stopping"]["patience"]:
                progress.update(finished=True,stop_reason="early_stopping")
            if progress["epoch"]>=cfg["stage_b"]["max_epochs"]:
                progress.update(finished=True,stop_reason="max_epochs")
            if tensor_digest(model.backbone.state_dict())!=base_digest:raise RuntimeError("Frozen base weights changed")
            save(root/f"epoch_{progress['epoch']:03d}.pt")
            if improved:save(root/"best.pt")
            save();last_save=time.time()
            write_json(root/"epochs.json",progress["epochs"])
            print({"epoch_completed":progress["epoch"],"selection_energy_score":score,"bad_epochs":progress["bad_epochs"]},flush=True)
```

**Source [training/loop.py:218–230](/home/siebenschuh/Projects/genco-foresight/training/loop.py:218) — verbatim:**

```python
    result={"status":status,"job_id":os.environ.get("PBS_JOBID"),"node":platform.node(),"device":device,
        "job_start_epoch":job_start,"job_end_epoch":time.time(),"startup_seconds":startup,
        "first_optimizer_step_completed_seconds":startup_record.get("first_optimizer_step_completed_seconds"),
        "steps_before":steps_before,"steps_after":progress["steps"],"steps_done":progress["steps"]-steps_before,
        "epochs_before":epochs_before,"epochs_reached":progress["epoch"],"step_in_epoch":progress["step_in_epoch"],
        "wall_used_seconds":time.time()-job_start,"cumulative_training_wall_seconds":used(),
        "training_loop_seconds":time.perf_counter()-started_steps,"stop_reason":progress["stop_reason"],
        "base_unchanged":True,"base_sha256":base_digest,"adapter_sha256":tensor_digest(train_state(model)),
        "step_sequence_sha256":progress["step_sequence_sha256"],"best_selection":progress["best_selection"],
        "best_epoch":progress["best_epoch"],"last_training_metrics":last_metrics,
        "checkpoint_sha256":sha256(state_path)}
    job_name=os.environ.get("PBS_JOBID",f"local_{int(entry)}").split('.')[0]
    write_json(root/"jobs"/f"{job_name}.json",result);write_json(root/"status.json",result)
```

A future statistical acceptance must keep exact serialization/restore, origin order, sample identities/RNG accounting, counter progression and frozen-weight invariants; relax only numerical trajectory equality where nondeterministic kernels warrant it. Predeclare paired-run tolerances before inspecting performance: finite losses/gradients, energy and balance components, selection-ES curves, final/best selection and stopping behavior across repeat seeds. Add the per-step or fixed-cadence component records needed to test that claim, and a trial crossing validation/checkpoint/early-stop boundaries. No such tolerances or production nondeterministic resume results exist yet. Deterministic CPU/debug resume checks remain useful. Merely comparing two total-loss curves by eye is not an acceptance test.

### 2.3 XPU operation evidence: call sites versus measured nondeterminism

The completed profile used strict deterministic on/off modes for case30 and case118. Both modes ran. The retained logs contain the same torch kernel-registration warning about `aten::topk`; they contain **no per-operation nondeterminism warning**. With determinism disabled, the flag does not enumerate nondeterministic kernels. Therefore the profile establishes mode cost, not which individual XPU op caused it. No extra XPU run was launched for this report.

Confirmed reduction sites are quoted below. They are candidates for order-dependent floating-point accumulation; their individual numerical nondeterminism/performance contribution on this vendor build has not been isolated. There is no explicit `index_add` call in the first-party training/model/solver/data path searched, and no shim is active in this path. An internal indexing-backward kernel must not be presented as an observed `index_add` trace.

**Source [/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/lib/python3.12/site-packages/torch_geometric/nn/conv/transformer_conv.py:107–114](/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/lib/python3.12/site-packages/torch_geometric/nn/conv/transformer_conv.py:107) — verbatim:**

```python
        dropout: float = 0.,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)
```

**Source [/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/lib/python3.12/site-packages/torch_geometric/nn/conv/transformer_conv.py:273–282](/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/lib/python3.12/site-packages/torch_geometric/nn/conv/transformer_conv.py:273) — verbatim:**

```python
        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        if edge_attr is not None:
            out = out + edge_attr

        out = out * alpha.view(-1, self.heads, 1)
```

**Source [/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/lib/python3.12/site-packages/torch_geometric/nn/aggr/basic.py:19–22](/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/lib/python3.12/site-packages/torch_geometric/nn/aggr/basic.py:19) — verbatim:**

```python
    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        return self.reduce(x, index, ptr, dim_size, dim, reduce='sum')
```

**Source [/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/lib/python3.12/site-packages/torch_geometric/utils/_scatter.py:67–70](/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/lib/python3.12/site-packages/torch_geometric/utils/_scatter.py:67) — verbatim:**

```python
    # For "sum" and "mean" reduction, we make use of `scatter_add_`:
    if reduce == 'sum' or reduce == 'add':
        index = broadcast(index, src, dim)
        return src.new_zeros(size).scatter_add_(dim, index, src)
```

**Source [/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/lib/python3.12/site-packages/torch_geometric/utils/_softmax.py:82–88](/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/lib/python3.12/site-packages/torch_geometric/utils/_softmax.py:82) — verbatim:**

```python
    elif index is not None:
        N = maybe_num_nodes(index, num_nodes)
        src_max = scatter(src.detach(), index, dim, dim_size=N, reduce='max')
        out = src - src_max.index_select(dim, index)
        out = out.exp()
        out_sum = scatter(out, index, dim, dim_size=N, reduce='sum') + 1e-16
        out_sum = out_sum.index_select(dim, index)
```

**Source [/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/lib/python3.12/site-packages/gridfm_graphkit/models/utils.py:56–72](/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/lib/python3.12/site-packages/gridfm_graphkit/models/utils.py:56) — verbatim:**

```python
class ComputeNodeInjection(nn.Module):
    """Aggregate branch flows into node-level incoming injections."""

    def forward(self, Pft, Qft, edge_index, num_bus):
        """
        Args:
            Pft, Qft: [num_edges] branch flows
            edge_index: [2, num_edges] (from_bus, to_bus)
            num_bus: number of bus nodes
        Returns:
            P_in, Q_in: aggregated incoming power per bus
        """
        from_idx, _ = edge_index  # Only sending end contributes as "incoming" to node
        P_in = scatter_add(Pft, from_idx, dim=0, dim_size=num_bus)
        Q_in = scatter_add(Qft, from_idx, dim=0, dim_size=num_bus)

        return P_in, Q_in
```

**Source [/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/lib/python3.12/site-packages/gridfm_graphkit/models/gnn_heterogeneous_gns.py:281–286](/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/lib/python3.12/site-packages/gridfm_graphkit/models/gnn_heterogeneous_gns.py:281) — verbatim:**

```python
                agg_bus = scatter_add(
                    gen_temp.squeeze(),
                    gen_to_bus_index,
                    dim=0,
                    dim_size=num_bus,
                )
```

**Source [/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/lib/python3.12/site-packages/torch_scatter/scatter.py:8–29](/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/lib/python3.12/site-packages/torch_scatter/scatter.py:8) — verbatim:**

```python
def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    return scatter_sum(src, index, dim, out, dim_size)
```

**Source [/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/solo_case30_ieee_k8/rank_0.log:1–10](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/solo_case30_ieee_k8/rank_0.log:1) — verbatim:**

```text
/opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0/lib/python3.12/site-packages/torch/library.py:505: UserWarning: Warning only once for all operators,  other operators may also be overridden.
  Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::topk(Tensor self, SymInt k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)
    registered at /lus/tegu/projects/datasets/software/26.181.0/wheelforge/repositories/pytorch_2.13.0_patched_08_02_2026/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
  dispatch key: XPU
  previous kernel: registered at /lus/tegu/projects/datasets/software/26.181.0/wheelforge/repositories/pytorch_2.13.0_patched_08_02_2026/pytorch/build/aten/src/ATen/RegisterCPU_1.cpp:9289
       new kernel: registered at /opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0/lib/python3.12/site-packages/torch/__init__.py:3095 (Triggered internally at /lus/tegu/projects/datasets/software/26.181.0/wheelforge/repositories/pytorch_2.13.0_patched_08_02_2026/pytorch/aten/src/ATen/core/dispatch/OperatorEntry.cpp:208.)
  self.m.impl(
[config] Converted v0 config -> v1 in memory (file on disk unchanged).
{"case": "case30_ieee", "K": 8, "rank": 0, "status": "OK", "reason": null, "phase_seconds": {"forward": {"median": 0.06555420008953661, "p90": 0.0657121566357091}, "loss": {"median": 0.0016542599769309163, "p90": 0.0016707109287381172}, "backward": {"median": 0.09520439244806767, "p90": 0.09538975094910711}, "optimizer": {"median": 0.002899409970268607, "p90": 0.002920276578515768}, "total": {"median": 0.1663176049478352, "p90": 0.1665863887174055}}}
```

Evidence logs: `solo_case30_ieee_k8`, `deterministic_case30_ieee`, `solo_case118_ieee_k8`, `deterministic_case118_ieee` under the profile directory named in §6. TransformerConv aggregation and attention normalization reach native scatter operations; generator Pg aggregation and nodal flow sums use `torch_scatter.scatter_add → scatter_sum → Tensor.scatter_add_`. Native index-select backward may also reduce duplicate indices, but no kernel trace or isolated repeatability test is available here. Do not equate absence of warnings with determinism or claim every scatter is nondeterministic on XPU.

## 3. Devices, processes, launch, and checkpoints

### 3.1 Existing entry points and tile assignment

These are the current source scripts, not instructions to submit them. The single-chain script starts one MPI process on one node; it does **not** set an explicit tile-affinity mask. The replication script starts one MPI controller, which creates separate Python workers, each restricted to one tile. Those workers train independent adapters; they do not cooperate on one origin or share gradients.

**Source [scripts/train_adapter.pbs:1–27](/home/siebenschuh/Projects/genco-foresight/scripts/train_adapter.pbs:1) — verbatim:**

```bash
#!/usr/bin/env bash
#PBS -N genco-adapter
#PBS -A EpiCalib
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -m n
set -euo pipefail
export GENCO_JOB_START_EPOCH="$(date +%s)"
cd "${PBS_O_WORKDIR:?Submit from the repository}"
set +u
if ! type module >/dev/null 2>&1; then source /etc/profile; fi
module --quiet reset
module load oneapi/release/2026.1.0 mpich/prd/5.0.0.aurora_test.87e2045 gcc/14.3.0
module load libfabric/2.3.1 cray-pals/1.8.0 cray-libpals/1.8.0 intel_gpu_umd_aicoe/2026.06.19
module load hdf5/1.14.6 pti-gpu/0.17.0 miniforge3/25.11.0-1 frameworks/2026.1.0
source /lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/bin/activate
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42 PBS_WALLTIME=3600
export TMPDIR="${GENCO_TRAIN_ROOT:-/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v3}/tmp"
mkdir -p "$TMPDIR"
adapter_python=/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/bin/python
mpiexec --pmi=pmix --envall -n 1 --ppn 1 --cpu-bind none \
    "$adapter_python" -u -m scripts.train_adapter --device "${GENCO_TRAIN_DEVICE:-xpu}"
```

**Source [scripts/replication_chain.pbs:1–27](/home/siebenschuh/Projects/genco-foresight/scripts/replication_chain.pbs:1) — verbatim:**

```bash
#!/usr/bin/env bash
#PBS -N genco-replication
#PBS -A EpiCalib
#PBS -q debug-scaling
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -m n
set -euo pipefail
export GENCO_FAMILY_JOB_START="$(date +%s)"
cd "${PBS_O_WORKDIR:?Submit from replication checkout}"
set +u
if ! type module >/dev/null 2>&1; then source /etc/profile; fi
module --quiet reset
module load oneapi/release/2026.1.0 mpich/prd/5.0.0.aurora_test.87e2045 gcc/14.3.0
module load libfabric/2.3.1 cray-pals/1.8.0 cray-libpals/1.8.0 intel_gpu_umd_aicoe/2026.06.19
module load hdf5/1.14.6 pti-gpu/0.17.0 miniforge3/25.11.0-1 frameworks/2026.1.0
source /lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/bin/activate
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PBS_WALLTIME=3600
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export TMPDIR=/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case30_ieee/tmp
mkdir -p "$TMPDIR"
mpiexec --pmi=pmix --envall -n 1 --ppn 1 --cpu-bind none \
 /lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/bin/python -u -m scripts.replication_chain job
```

**Source [scripts/replication_chain.py:296–314](/home/siebenschuh/Projects/genco-foresight/scripts/replication_chain.py:296) — verbatim:**

```python
def job():
    f,root=settings();guard_release();reports()
    phase,work=phase_plan();jid=os.environ['PBS_JOBID'].split('.')[0]
    job_record={'job_id':os.environ['PBS_JOBID'],'node':os.uname().nodename,'phase':phase,
                'start_epoch':float(os.environ['GENCO_FAMILY_JOB_START']),
                'launcher_sha256':sha256(__file__),'runner_sha256':sha256(ROOT/'scripts/replication_evaluate.py'),
                'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()}
    write_json(root/'case30_ieee/jobs'/f'{jid}_start.json',job_record)
    tasks=[];streams=[]
    def launch(name,module,args,tile,log):
        Path(log).parent.mkdir(parents=True,exist_ok=True);stream=Path(log).open('a');streams.append(stream)
        env=dict(os.environ,ZE_FLAT_DEVICE_HIERARCHY='FLAT',ZE_AFFINITY_MASK=str(tile))
        proc=subprocess.Popen([f['python'],'-u','-m',module,*args],cwd=ROOT,env=env,
                              stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        tasks.append({'name':name,'process':proc,'log':str(log)})
    if phase=='training':
        for c in f['chains']:
            out,_,_=paths(c)
            launch(c['id'],'scripts.replication_chain',['worker','--chain',c['id']],c['tile'],out/'logs'/f'{jid}_launcher.log')
```

The family configuration supplies the chain list and tile indices. Its seed-0 convention maps to training seed 42; the four additional chains use seeds 1 and 2. The family remains specific to the released case30 experiment. Its historical `blocked_grids` entries, below the quoted portion, describe the old release and must not be read as current multigrid support status.

**Source [configs/replication/family.json:1–25](/home/siebenschuh/Projects/genco-foresight/configs/replication/family.json:1) — verbatim:**

```json
{
  "base": "/lus/flare/projects/FRAME-IDP/siebenschuh",
  "family": "case30-seeds",
  "root": "${base}/GENCO/runs/replication",
  "release_commit": "521c34203d8edb6c2ed3966e50f0ca9ca9a9569e",
  "release_tag_object": "42d5f9668375e023dafb915310a6eafd63768052",
  "release_bundle": "${base}/GENCO/reproducibility/v0.1.0-rc",
  "python": "${base}/envs/reason_venv_2026.1.0/bin/python",
  "queue": "debug-scaling",
  "account": "EpiCalib",
  "walltime_seconds": 3600,
  "training_budget_seconds": 21600,
  "seed_mapping": {"0": 42, "1": 1, "2": 2},
  "forecast_rng_seed": 42,
  "report_due_ct": "2026-10-24T07:30:00-05:00",
  "decision_table_ct": "2026-10-25T11:30:00-05:00",
  "inclusion_cutoff_ct": "2026-10-25T12:00:00-05:00",
  "bootstrap": {"draws": 10000, "seed": 20261024, "confidence": 0.95, "unit": "paired forecast origin", "qualification": "72 correlated days from four episodes; origin bootstrap is descriptive and does not establish independent-day coverage."},
  "ensemble": {"samples_per_seed": 64, "seeds": [0,1,2], "samples": 192, "weights": "equal", "qualification": "Stratified seed mixture with common evaluation innovations; unchanged fair scorer is used, without claiming IID-unbiasedness for the mixture."},
  "chains": [
    {"id": "v3_seed1", "version": "v3", "seed": 1, "tile": 0},
    {"id": "v3_seed2", "version": "v3", "seed": 2, "tile": 1},
    {"id": "v4_seed1", "version": "v4", "seed": 1, "tile": 2},
    {"id": "v4_seed2", "version": "v4", "seed": 2, "tile": 3}
  ],
```

The release guard checks the tag object and every released code-file hash. The paths function still constructs case30 paths. Consequently, this old controller cannot simply be repurposed on the changed `multigrid` source tree without an explicitly versioned launcher/configuration change; bypassing its guard would discard its provenance contract.

**Source [scripts/replication_chain.py:34–51](/home/siebenschuh/Projects/genco-foresight/scripts/replication_chain.py:34) — verbatim:**

```python
def guard_release():
    f, _ = settings()
    tag = subprocess.check_output(['git','rev-parse','refs/tags/v0.1.0'],cwd=ROOT).decode().strip()
    assert tag == f['release_tag_object'], 'Release tag changed'
    manifest = read_json(Path(f['release_bundle'])/'bundle.json')
    checked = 0
    for name, digest in manifest['sha256'].items():
        if name.startswith('code/'):
            p = ROOT / name[5:]
            assert p.is_file() and sha256(p) == digest, f'Released file changed: {p}'
            checked += 1
    return checked


def paths(chain):
    _, root = settings()
    out = root/'case30_ieee'/chain['version']/f"seed{chain['seed']}"
    return out, out/'adapter', ROOT/'configs/replication'/f"case30_{chain['id']}.yaml"
```

Worker startup, resume verification, the remaining-time calculation, crash accounting, and per-job boundary records are implemented here. The tile mask is already present in the worker process environment before Torch import; setting `PYTHONHASHSEED` inside this worker only takes effect for subsequently launched Python interpreters, not its own already-created hash secret.

**Source [scripts/replication_chain.py:218–266](/home/siebenschuh/Projects/genco-foresight/scripts/replication_chain.py:218) — verbatim:**

```python
def worker(identity):
    f,root=settings();c=next(x for x in f['chains'] if x['id']==identity)
    out,train,s,ledger=chain_state(c);_,_,config=paths(c)
    os.environ.update(GENCO_CONFIG=str(config),PYTHONHASHSEED=str(c['seed']),ZE_FLAT_DEVICE_HIERARCHY='FLAT',ZE_AFFINITY_MASK=str(c['tile']))
    start=float(os.environ['GENCO_FAMILY_JOB_START']);job=os.environ['PBS_JOBID'].split('.')[0]
    (out/'logs').mkdir(exist_ok=True)
    def boundary(phase):
        write_json(out/'boundaries'/f'{job}.json',{'job_id':job,'phase':phase,'budget':ledger})
    if ledger.get('parked') or s.get('status')=='finished' or ledger['used_seconds']>=21585:
        boundary('idle');return
    import torch
    assert torch.xpu.device_count()==1, f'Expected isolated tile; got {torch.xpu.device_count()}'
    torch.xpu.set_device(0)
    write_json(out/'affinity.json',{'tile_flat_index':c['tile'],'visible_devices':1,'device':str(torch.xpu.get_device_properties(0)),
        'job_id':job,'node':os.uname().nodename,'source':f['affinity_reference']})
    cfg=load_config(config)
    if not (train/'resume_test/report.json').exists():
        from scripts.resume_test import trial
        try:
            r=trial(cfg,'xpu',200,train/'resume_test')
            write_json(train/'resume_test/report.json',r)
            if not r['passed']:raise RuntimeError('Bit-identical resume comparison failed')
        except Exception as e:
            ledger.update(parked=True,reason=repr(e),last_accounted_job=job);write_json(out/'budget.json',ledger);boundary('resume_failed')
            decision(f'{identity}: parked after resume-test failure: {e}','Resume test fails → launcher config only; otherwise skip chain.',identity);return
    assert read_json(train/'resume_test/report.json')['passed']
    remaining=3600-(time.time()-start)
    if remaining<600:
        boundary('preparation_complete_time_guard');return
    if (train/'failure.json').exists():
        history=out/'failures';history.mkdir(exist_ok=True)
        (train/'failure.json').rename(history/f'{job}_prior_failure.json')
    latest=train/'latest.pt'
    if latest.exists():
        recovery=dict(s,cumulative_training_wall_seconds=max(ledger['used_seconds'],s.get('cumulative_training_wall_seconds',0)),checkpoint_sha256=sha256(latest))
        write_json(train/'status.json',recovery)
    began=time.time()
    write_json(out/'attempt_start.json',{'job':job,'start_epoch':began,'budget_before':ledger['used_seconds']})
    env=dict(os.environ,GENCO_JOB_START_EPOCH=str(began),PBS_WALLTIME=str(int(remaining)))
    code=invoke([f['python'],'-u','-m','scripts.train_adapter','--device','xpu'],env,out/'logs'/f'{job}_training.log',min(remaining-40,21600-ledger['used_seconds']-2))
    elapsed=time.time()-began;ledger['used_seconds']+=elapsed;ledger['last_accounted_job']=job
    result=read_json(train/'status.json') if (train/'status.json').exists() else {}
    if code:
        ledger['crashes']+=1
        ledger['parked']=ledger['crashes']>=2 or not latest.exists()
        decision(f'{identity}: exit {code}; '+('parked' if ledger['parked'] else 'one restart from latest.pt scheduled'),
                 'Any chain crashes → resume latest.pt once; second crash parks chain.',identity)
    write_json(out/'budget.json',ledger)
    write_json(out/'boundaries'/f'{job}.json',{'job_id':job,'node':os.uname().nodename,'returncode':code,'production_wall_seconds':elapsed,'budget':ledger,'training_result':result})
```

The outer `invoke` timeout is a last-resort kill, separate from the training loop's graceful checkpoint guard. The normal worker reserves 300 seconds from PBS walltime, then another 15 seconds before raising its guard. Checkpoints are also taken every 900 seconds and at epoch boundaries (§2).

**Source [scripts/replication_chain.py:210–215](/home/siebenschuh/Projects/genco-foresight/scripts/replication_chain.py:210) — verbatim:**

```python
def invoke(command,env,log,timeout):
    with Path(log).open('a') as stream:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:return process.wait(timeout=max(1,timeout))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait();return -9
```

**Source [training/loop.py:147–166](/home/siebenschuh/Projects/genco-foresight/training/loop.py:147) — verbatim:**

```python
    guard_seconds=min(cfg["stage_b"]["job_training_guard_seconds"],walltime_seconds(os.environ.get("PBS_WALLTIME"))-300)
    job_deadline=job_start+guard_seconds
    budget=cfg["stage_b"]["training_budget_seconds"]
    startup=time.time()-job_start
    startup_record={"job_id":os.environ.get("PBS_JOBID"),"node":platform.node(),
        "device":device,"job_start_epoch":job_start,"startup_seconds":startup,"steps_before":steps_before,
        "cache":"torch.load(mmap=True), per-hour history slices","target_seconds":120,
        "within_target":startup<=120}
    write_json(root/"startup.json",startup_record)
    print({"training_ready":True,"startup_seconds":startup,"resume_step":steps_before,"device":device},flush=True)
    last_save=time.time();last_metrics=None
    def used():return used_before+(time.time()-job_start)
    def save(path=state_path):
        progress["cumulative_training_wall_seconds"]=used()
        checkpoint(path,model,opt,sched,progress,manifest)
    def guard():
        if used() >= budget-15:
            progress.update(finished=True,stop_reason="six_hour_budget");raise TimeGuard
        if time.time()>=job_deadline-15:
            progress["stop_reason"]="job_walltime_guard";raise TimeGuard
```

### 3.2 Controller state and resubmission

There is no formal JSON Schema document: the writers below define the records. The multi-chain controller uses a filesystem lock and polls PBS. Its durable files are:

| File, relative to replication root | Fields and role |
|---|---|
| `controller.json` | `jobs`, `created_epoch`, `pid`, `host`, `updated_epoch`, `phase`, `current_job`; optional queue/dependency rejection reason. Each job has `job_id`, `submitted_epoch`, `state`, `phase`, `dependency`, then completion/result/wait fields. |
| `case30_ieee/<version>/seed<seed>/budget.json` | `used_seconds`, `crashes`, then `last_accounted_job`, optional `parked` and `reason`; charged independently per chain. |
| `.../boundaries/<job>.json` | Job ID, node, return code, production wall seconds, budget ledger, training status; idle/preparation boundaries have fewer fields. |
| `.../adapter/status.json` | Training progress, checkpoint hash and cumulative worker wall time; separate from the controller ledger. |
| `case30_ieee/jobs/<job>_start.json` | Job/node/phase/start, launcher and evaluation-runner hashes, Git HEAD. |

The controller charges a missing worker boundary conservatively, using time until detection. That may overestimate work after a crash; it does not justify extending the six-hour cap.

**Source [scripts/replication_chain.py:102–106](/home/siebenschuh/Projects/genco-foresight/scripts/replication_chain.py:102) — verbatim:**

```python
def chain_state(c):
    out, train, _=paths(c)
    state=read_json(train/'status.json') if (train/'status.json').exists() else {}
    ledger=read_json(out/'budget.json') if (out/'budget.json').exists() else {'used_seconds':0.,'crashes':0}
    return out,train,state,ledger
```

**Source [scripts/replication_chain.py:372–409](/home/siebenschuh/Projects/genco-foresight/scripts/replication_chain.py:372) — verbatim:**

```python
def controller(once=False):
    f,root=settings();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=read_json(root/'controller.json') if (root/'controller.json').exists() else {'jobs':[],'created_epoch':time.time()}
    cutoff=datetime.fromisoformat(f['inclusion_cutoff_ct']).timestamp()
    while time.time()<cutoff:
        state.update(pid=os.getpid(),host=os.uname().nodename,updated_epoch=time.time())
        pending=None
        for j in state['jobs']:
            if j.get('finished'):continue
            st,code=job_state(j['job_id']);j.update(state=st,exit_status=code)
            if st=='F':
                j['finished']=True
                jid=j['job_id'].split('.')[0]
                record=root/'case30_ieee/jobs'/f'{jid}.json'
                started=root/'case30_ieee/jobs'/f'{jid}_start.json'
                if started.exists():j['phase']=read_json(started)['phase']
                if record.exists():
                    r=read_json(record);j['result']=r;j['phase']=r.get('phase','training')
                    dependency_end=j['submitted_epoch']
                    if j.get('dependency'):
                        prior=next(x for x in state['jobs'] if x['job_id']==j['dependency'])
                        dependency_end=max(dependency_end,prior.get('result',{}).get('end_epoch',dependency_end))
                    j['dependency_wait_seconds']=dependency_end-j['submitted_epoch']
                    j['queue_wait_seconds']=max(0,r['start_epoch']-dependency_end)
                for c in f['chains'] if j.get('phase','training')=='training' else []:
                    out,train,s,l=chain_state(c)
                    if (out/'boundaries'/f'{jid}.json').exists() or l.get('parked') or s.get('status')=='finished':continue
                    if l.get('last_accounted_job')==jid:continue
                    attempt=read_json(out/'attempt_start.json') if (out/'attempt_start.json').exists() else {}
                    if attempt.get('job')==jid:
                        # Conservative end at detection, so lost post-checkpoint wall time is charged.
                        l['used_seconds']=max(l['used_seconds'],attempt['budget_before']+time.time()-attempt['start_epoch'])
                    l['crashes']+=1;l['last_accounted_job']=jid
                    l['parked']=l['crashes']>=2 or not (train/'latest.pt').exists()
                    write_json(out/'budget.json',l)
                    decision(f"{c['id']}: missing worker boundary; " + ('parked' if l['parked'] else 'one checkpoint retry'),
                        'Any chain crashes → resume latest.pt once; second crash parks chain.',c['id'])
```

**Source [scripts/replication_chain.py:423–454](/home/siebenschuh/Projects/genco-foresight/scripts/replication_chain.py:423) — verbatim:**

```python
        phase,active=phase_plan();state['phase']=phase
        unfinished=[j for j in state['jobs'] if not j.get('finished')]
        prefetch=(phase=='training' and len(unfinished)==1 and unfinished[0].get('state')=='R'
                  and not state.get('dependency_prefetch_rejected'))
        if (not pending and phase not in ('idle','blocked')) or prefetch:
            guard_release()
            # A single family may occupy the queue; unrelated jobs are never cancelled.
            q=subprocess.run(['qstat','-u',os.environ.get('USER','siebenschuh')],capture_output=True,text=True,timeout=30)
            if q.returncode:raise RuntimeError('Cannot inspect queue safely: '+q.stderr)
            existing=[line.split()[0].split('.')[0] for line in q.stdout.splitlines() if line.strip() and line.lstrip()[0].isdigit()]
            expected=unfinished[0]['job_id'].split('.')[0] if prefetch else None
            occupied=any(j!=expected for j in existing)
            if not occupied:
                log=root/'case30_ieee/logs';log.mkdir(parents=True,exist_ok=True)
                number=len(state['jobs'])+1
                command=['qsub','-q',f['queue'],'-N',f'genco-rep-{number}','-o',str(log/f'job_{number:02d}.log'),str(ROOT/'scripts/replication_chain.pbs')]
                dependency=unfinished[0]['job_id'] if prefetch else None
                if dependency:command[-1:-1]=['-W',f'depend=afterany:{dependency}']
                run=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=45)
                if run.returncode:
                    state['queue_wait_reason']=run.stderr.strip()
                    if prefetch:
                        state['dependency_prefetch_rejected']=run.stderr.strip()
                        decision('Dependency prefetch rejected; controller will submit after each job boundary: '+run.stderr.strip(),
                                 'Dependency chaining if queue allows; otherwise boundary resubmission.')
                else:
                    jid=run.stdout.strip().splitlines()[-1];state['jobs'].append({'job_id':jid,'submitted_epoch':time.time(),'state':'Q','phase':phase,'dependency':dependency})
                    state['current_job']=jid;print('Submitted',jid,flush=True)
            else:state['queue_wait_reason']='Existing user job occupies slot; no unrelated cancellation.'
        write_json(root/'controller.json',state);reports()
        if once:return
        time.sleep(30)
```

The single-chain controller has a separate `chain.json`. It checks a fixed device, records aggregate wall/queue/dependency time, supports `afterany` prefetch, and disables prefetch if PBS rejects it. v4's configuration sets `variant.dependency_prefetch: false`. Its exact launch construction is:

**Source [scripts/training_chain.py:36–46](/home/siebenschuh/Projects/genco-foresight/scripts/training_chain.py:36) — verbatim:**

```python
    path=root/"chain.json"
    chain=read_json(path) if path.exists() else {"mode":"A","device":args.device,"status":"running",
        "jobs":[],"dependency_chaining":cfg.get("variant",{}).get("dependency_prefetch",True),"budget_seconds":cfg["stage_b"]["training_budget_seconds"],
        "manual_command":f"GENCO_CONFIG={config_path().resolve()} python -m scripts.training_chain --device {args.device}","created_epoch":time.time()}
    if chain["device"]!=args.device:raise ValueError("Do not change device inside an existing chain")
    (root/"logs").mkdir(exist_ok=True)
    def save():
        done=[j for j in chain["jobs"] if "result" in j]
        chain["cumulative_training_wall_seconds"]=sum(j["result"]["wall_used_seconds"] for j in done)
        chain["total_queue_wait_seconds"]=sum(j.get("queue_wait_seconds",0.) for j in done)
        chain["total_dependency_wait_seconds"]=sum(j.get("dependency_wait_seconds",0.) for j in done)
```

**Source [scripts/training_chain.py:56–72](/home/siebenschuh/Projects/genco-foresight/scripts/training_chain.py:56) — verbatim:**

```python
    def submit(dependency=None):
        number=len(chain["jobs"])+1
        cmd=["qsub","-q",cfg["stage_b"]["pbs_queue"],"-N",f"genco-A-{number}",
             "-o",str(root/f"logs/job_{number:02d}.log"),"-e",str(root/f"logs/job_{number:02d}.err"),
             "-v",f"GENCO_TRAIN_DEVICE={args.device},GENCO_CONFIG={config_path().resolve()},GENCO_TRAIN_ROOT={root}"]
        if dependency:cmd += ["-W",f"depend=afterany:{dependency}"]
        cmd += [str(ROOT/"scripts/train_adapter.pbs")]
        result=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=60)
        if result.returncode:
            if dependency:
                chain["dependency_chaining"]=False;chain["dependency_rejection"]=result.stderr.strip();save();return
            if "limit" in result.stderr.lower():
                chain["submission_wait_reason"]=result.stderr.strip();save();return
            raise RuntimeError(result.stderr.strip())
        jid=result.stdout.strip().splitlines()[-1]
        chain["jobs"].append({"job_id":jid,"submitted_epoch":time.time(),"dependency":dependency,"state":"Q"})
        save();print("submitted",jid,"dependency",dependency,flush=True)
```

**Source [scripts/training_chain.py:92–109](/home/siebenschuh/Projects/genco-foresight/scripts/training_chain.py:92) — verbatim:**

```python
            save()
            if chain["cumulative_training_wall_seconds"]>=chain["budget_seconds"]-15:chain["status"]="budget_complete"
            if chain["status"]!="running":
                for j in pending:
                    result=subprocess.run(["qdel",j["job_id"]],capture_output=True,text=True,timeout=30)
                    j["state"]="cancelled";j["cancellation_returncode"]=result.returncode
                save();break
            if not pending:submit()
            elif len(pending)==1 and pending[0]["state"]=="R" and chain["dependency_chaining"]:
                # The v4 priority amendment reserves a successor throughout training.
                # A surplus successor is cancelled by the stop/budget branch above.
                if chain.get("reserve_queue_slot",False) or chain["budget_seconds"]-chain["cumulative_training_wall_seconds"]>2*cfg["stage_b"]["job_training_guard_seconds"]:
                    submit(pending[0]["job_id"])
            save()
            if args.once:break
            time.sleep(30)
    except BaseException as error:
        chain["controller_error"]=repr(error);save();raise
```

### 3.3 Distributed support: what is present and what is untested

A source search for `torch.distributed`, `init_process_group`, `DistributedDataParallel`, `all_gather`, and `all_reduce` under `data/`, `models/`, `training/`, `solver/`, `scripts/`, and `tests/` found **no matches**. The independent-process launcher is not DDP. Imports inside installed Lightning do not establish a process group for this loop.

The following read-only check was executed in the existing Aurora venv on the login node; it did not initialize a communicator or use an XPU:

```bash
python -c 'import torch; print("torch:", torch.__version__); print("distributed:", torch.distributed.is_available()); print("xccl:", torch.distributed.is_xccl_available()); print("module:", torch.__file__)'
```

```text
torch: 2.13.0a0+gitcf30153
distributed: True
xccl: True
module: /opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0/lib/python3.12/site-packages/torch/__init__.py
```

This establishes **build availability only**, not working collectives, autograd through a gather, or multi-tile speedup. The current ALCF PyTorch guidance uses `xccl`, with 12 ranks per node for one rank per tile under the flat hierarchy. Its example assigns four CPU cores per rank and obtains local rank through PALS; initialization also requires world rank/size and rendezvous configuration. [ALCF Aurora PyTorch documentation](https://docs.alcf.anl.gov/aurora/data-science/frameworks/pytorch/).

The documented CPU binding and an adaptation of our existing launcher flags are shown below. The runner is deliberately a placeholder: no distributed runner exists in this repository, and this command was **not** run.

```bash
export CPU_BIND="verbose,list:4-7:8-11:12-15:16-19:20-23:24-27:56-59:60-63:64-67:68-71:72-75:76-79"
mpiexec --pmi=pmix --envall -n 12 --ppn 12 --cpu-bind="${CPU_BIND}" python FUTURE_DISTRIBUTED_RUNNER.py
```

With an externally isolated single tile per process, `torch.xpu.set_device(0)` is correct. With all tiles visible, the runner must select the tile from local rank. These are different launch conventions; applying both indexing schemes blindly can select a nonexistent device. For sample parallelism, every rank must use the **same origin** and a disjoint subset of its K draws; a sampler that distributes origins implements a different experiment.

### 3.4 Checkpoint format and manifest equality

`best.pt`, `latest.pt`, and epoch checkpoints use the same six top-level keys. They store adapter parameters and buffers, excluding frozen backbone entries; optimizer and scheduler state; Python/NumPy/Torch RNG state; progress; and the full manifest. Atomic serialization recursively moves tensors to CPU before writing a temporary file, flushing/fsyncing it, and renaming it. The CPU conversion itself synchronizes outstanding device transfers.

**Source [training/state.py:47–89](/home/siebenschuh/Projects/genco-foresight/training/state.py:47) — verbatim:**

```python
def atomic_torch_save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=f".{path.name}.")
    try:
        with os.fdopen(fd,"wb") as stream:
            torch.save(cpu_tree(value),stream);stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def train_state(model):
    # The same machinery stores backbone parameters for F if they are unfrozen.
    include_base=any(p.requires_grad for p in model.backbone.parameters())
    return {k:v for k,v in model.state_dict().items() if include_base or not k.startswith("backbone.")}


def checkpoint(path, model, optimizer, scheduler, progress, manifest):
    atomic_torch_save(path,{"model":train_state(model),"optimizer":optimizer.state_dict(),
        "scheduler":scheduler.state_dict(),"rng":rng_state(),"progress":progress,"manifest":manifest})


def restore(path, model, optimizer, scheduler, manifest):
    value=torch.load(path,map_location="cpu",weights_only=True)
    if value["manifest"] != manifest: raise ValueError("Checkpoint run manifest mismatch")
    missing,unexpected=model.load_state_dict(value["model"],strict=False)
    if unexpected or any(not k.startswith("backbone.") for k in missing):
        raise ValueError("Incomplete adapter checkpoint")
    if missing and any(p.requires_grad for p in model.backbone.parameters()):
        raise ValueError("Unfrozen backbone missing from checkpoint")
    optimizer.load_state_dict(value["optimizer"]);scheduler.load_state_dict(value["scheduler"])
    restore_rng(value["rng"])
    return value["progress"]


def epoch_order(seed, epoch, count):
    return torch.randperm(count,generator=torch.Generator().manual_seed(seed+epoch)).tolist()


def next_progress():
    return {"epoch":0,"step_in_epoch":0,"steps":0,"best_selection":None,"bad_epochs":0,
        "best_epoch":None,"epochs":[],"cumulative_training_wall_seconds":0.,"finished":False,
        "stop_reason":None,"step_sequence_sha256":hashlib.sha256(b"").hexdigest()}
```

The existing run directory is rejected before initialization if `manifest.json` differs. The checkpoint loader separately requires exact dictionary equality. The manifest includes the entire `stage_b` mapping, source hashes, seed, checkpoint/dataset/scales identity and a hard-coded deterministic flag. A change to any recorded setting or source file therefore intentionally prevents an unnoticed continuation of the old run.

**Source [training/loop.py:25–37](/home/siebenschuh/Projects/genco-foresight/training/loop.py:25) — verbatim:**

```python
def make_manifest(cfg):
    dataset=Path(cfg["paths"]["temporal_root"])
    return {"mode":"A","dataset_id":cfg["dataset"]["id"],"dataset_manifest_sha256":sha256(dataset/"manifest.json"),
        "benchmark_id":read_json(dataset/"manifest.json")["benchmark_id"],"base_checkpoint":cfg["checkpoint"],
        "graphkit":cfg["upstream"]["graphkit"],"seed":cfg["seed"],"settings":cfg["stage_b"],
        "coordinate_scales_sha256":sha256(ROOT/cfg["stage_b"]["coordinate_scales"]),
        "training_origins":cfg["task"]["split_episodes"]["train"]*409,
        "selection_origins":cfg["task"]["split_episodes"]["model_selection"]*18,
        "deterministic_algorithms":True,
        "python_hash_seed":os.environ.get("PYTHONHASHSEED"),
        "source_sha256":{p:sha256(ROOT/p) for p in ("models/temporal.py","scoring.py","training/state.py","training/inputs.py","training/loop.py")},
        "budget_definition":"Cumulative worker-job wall time including startup, validation and checkpointing; excludes queue wait, separate cache preparation and resume tests",
        "rng_protocol":"Restore Python/NumPy/Torch CPU/XPU/CUDA states; deterministic epoch permutation; independent fixed selection-noise generator"}
```

**Source [training/loop.py:115–133](/home/siebenschuh/Projects/genco-foresight/training/loop.py:115) — verbatim:**

```python
def run(cfg,device,run_root=None,stop_after_steps=None,crash_at_steps=None,test_mode=False):
    entry=time.time();job_start=float(os.environ.get("GENCO_JOB_START_EPOCH",entry))
    root=Path(run_root or cfg["paths"]["training_run"]);root.mkdir(parents=True,exist_ok=True)
    if (root/"failure.json").exists():raise RuntimeError("Prior worker failed; inspect failure.json before resuming")
    readiness=read_json(Path(cfg["paths"]["stage_a_run"])/"readiness.json")
    if not readiness["stage_a_audit_passed"] or not readiness["training_allowed"]:
        raise RuntimeError("Stage A audit has not passed")
    if readiness["manifest_sha256"] != sha256(Path(cfg["paths"]["temporal_root"])/"manifest.json"):
        raise RuntimeError("Dataset changed after the Stage A audit")
    if device=="xpu" and not torch.xpu.is_available():raise RuntimeError("XPU is unavailable on this node")
    manifest=make_manifest(cfg)
    if (root/"manifest.json").exists() and read_json(root/"manifest.json")!=manifest:
        raise ValueError("Training implementation or frozen configuration changed")
    write_json(root/"manifest.json",manifest)
    model,opt,sched,template,cache=initialize(cfg,device)
    base_digest=tensor_digest(model.backbone.state_dict())
    state_path=root/"latest.pt"
    progress=restore(state_path,model,opt,sched,manifest) if state_path.exists() else next_progress()
    if "initial_adapter_sha256" not in progress:
```

Read-only inspection of `/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v4/{best,latest}.pt` confirmed those six keys. `best.pt` records epoch 14, 57,260 steps, selection ES 0.27833065887292224; `latest.pt` records epoch 17, 69,530 steps and three non-improving epochs. The stored source hashes differ from the current multigrid files for `models/temporal.py`, `scoring.py`, `training/inputs.py`, and `training/loop.py`; `training/state.py` is unchanged. This is a concrete reason the released run cannot resume under the current manifest. The earlier incident's exception log was not located, so this inventory does not claim to reconstruct that incident beyond the source rule and manifest mismatch.

Each proposed stage needs a new run identity or an explicitly reviewed migration policy. Do not silently rewrite historical manifests. Distributed continuation additionally needs all ranks' RNG states and the sample partition/world-size convention; the present single-process `rng` record does not provide that contract.

## 4. Data path per training step

### 4.1 Cached contents, storage, and transfers

Cache files are per-episode CPU tensors on Lustre under `<training_run>/cache`; historical GENCO embeddings are computed once with `no_grad`. Only train and model-selection episodes enter this cache. Solved operating variables are stored separately from demand-only embeddings. Failed/missing values remain NaNs with availability masks.

**Source [training/inputs.py:25–76](/home/siebenschuh/Projects/genco-foresight/training/inputs.py:25) — verbatim:**

```python
def build_cache(cfg, device):
    started=time.perf_counter()
    root=Path(cfg["paths"]["temporal_root"]);out=Path(cfg["paths"]["training_run"])/"cache"
    out.mkdir(parents=True,exist_ok=True)
    net=load_network(config=cfg)
    manifest=load_temporal_manifest(root,net);scales=read_json(ROOT/cfg["stage_b"]["coordinate_scales"])
    assert scales["benchmark_id"]==manifest["benchmark_id"]
    base=Genco(checkpoint_assets(cfg),cfg["checkpoint"],device,grid=net)
    hashes={r["file"]:r["sha256"] for r in manifest["shards"]}
    files={}
    # Training setup never reads calibration/test solved data.
    for ep in manifest["episodes"]:
        if ep["split"] not in ("train","model_selection"):continue
        name=ep["episode_id"];path=root/f"episodes/{name}.json"
        assert sha256(path)==hashes[f"episodes/{name}.json"]
        key={"benchmark_id":manifest["benchmark_id"],"episode_sha256":sha256(path),
            "checkpoint":cfg["checkpoint"],"graphkit":cfg["upstream"]["graphkit"],
            "scales_sha256":sha256(ROOT/cfg["stage_b"]["coordinate_scales"]),"cache_version":1}
        dest=out/f"{name}.pt"
        if dest.exists():
            value=torch.load(dest,mmap=True,map_location="cpu",weights_only=True)
            if value["key"]!=key:raise ValueError("Stale embedding/history cache")
        else:
            rows=read_json(path)["records"];chunks={"bus":[],"gen":[]}
            for lo in range(0,len(rows),192):
                # Only demand/static topology reaches frozen embeddings; solved state is separate.
                graphs=[graph_for_demand(net,r["pd_mw"],r["qd_mvar"]) for r in rows[lo:lo+192]]
                with torch.no_grad():_,emb=base.model(base.batch(graphs),return_embeddings=True)
                for kind,n in (("bus",net.n_bus),("gen",net.n_online_gen)):chunks[kind].append(emb[kind].reshape(-1,n,emb[kind].shape[-1]).cpu())
            data={f"embedding_{k}":torch.cat(v) for k,v in chunks.items()}
            observed_bus=[];observed_gen=[];targets=[]
            for r in rows:
                state=r["target_state"] or {"vm":[np.nan]*net.n_bus,"va_rad":[np.nan]*net.n_bus,"pg_mw":[np.nan]*net.n_gen,"qg_mvar":[np.nan]*net.n_gen}
                observed_bus.append(np.stack([r["pd_mw"],r["qd_mvar"],state["vm"],state["va_rad"]],axis=-1))
                observed_gen.append(np.stack([state["pg_mw"],state["qg_mvar"]],axis=-1)[net.online_gen])
                physics=physical_quantities(net,r["pd_mw"],r["qd_mvar"],state)
                values=record_values(net,r)
                targets.append(np.concatenate([np.asarray(values[k])[s["joint_score_mask"]]
                    for k,s in scales["features"].items() if any(s["joint_score_mask"])]))
            for kind,values in (("bus",observed_bus),("gen",observed_gen)):
                data[f"observed_{kind}"]=torch.tensor(np.array(values),dtype=torch.float32)
                data[f"available_{kind}"]=torch.isfinite(data[f"observed_{kind}"])
            data["targets"]=torch.tensor(np.array(targets),dtype=torch.float32)
            data["calendar"]=calendar_features([r["time_utc"] for r in rows])
            atomic_torch_save(dest,{"key":key,"data":data})
        files[name]={"path":str(dest),"sha256":sha256(dest),"partition":ep["split"]}
        print("cached",name,flush=True)
    report={"benchmark_id":manifest["benchmark_id"],"files":files,"wall_seconds":time.perf_counter()-started,
        "history_storage":"per-hour tensors memory-mapped once; 72-hour views contain only hours before origin",
        "partitions_cached":["train","model_selection"],"device":device}
    write_json(out/"manifest.json",report)
    return report
```

**Source [training/inputs.py:79–103](/home/siebenschuh/Projects/genco-foresight/training/inputs.py:79) — verbatim:**

```python
def causal_window(data, origin):
    history={k:v[origin-72:origin] for k,v in data.items() if k.startswith(("embedding_","observed_","available_"))}
    calendar=torch.cat([data["calendar"][origin:origin+24],torch.arange(1,25,dtype=torch.float32)[:,None]/24],dim=1)
    return history,calendar,data["targets"][origin:origin+24].flatten()


class CachedWindows:
    def __init__(self,cfg,grid):
        root=Path(cfg["paths"]["training_run"])/"cache"
        self.manifest=read_json(root/"manifest.json")
        expected=load_temporal_manifest(cfg["paths"]["temporal_root"],grid)["benchmark_id"]
        if self.manifest["benchmark_id"]!=expected:raise ValueError("Wrong dataset cache")
        self.data={};self.indices={"train":[],"model_selection":[]}
        for name,row in self.manifest["files"].items():
            if sha256(row["path"])!=row["sha256"]:raise ValueError("Embedding cache checksum changed")
            value=torch.load(row["path"],mmap=True,map_location="cpu",weights_only=True)
            if value["key"]["checkpoint"]!=cfg["checkpoint"]:raise ValueError("Wrong checkpoint in embedding cache")
            self.data[name]=value["data"]
            stride=1 if row["partition"]=="train" else 24
            self.indices[row["partition"]].extend((name,o) for o in range(72,len(self.data[name]["calendar"])-24+1,stride))

    def window(self, identity, device):
        name,origin=identity
        history,calendar,target=causal_window(self.data[name],origin)
        return {k:v.to(device) for k,v in history.items()},calendar.to(device),target.to(device)
```

`torch.load(mmap=True, map_location="cpu")` maps the episode file into CPU virtual memory. Each step slices its 72-hour history and moves those slices, the next 24 calendar rows, and the target vector to the requested device. There is no pinning/prefetch/nonblocking-transfer option in `window`. Initialization hashes the whole cache files before mapping them; mmap does not remove that startup I/O. The cache does not persist whole 72-hour windows on XPU.

The case30 column below is from read-only inspection of the actual v4 `cache/train_00.pt`, origin 72. The case500 column is an **analytical payload estimate**, using its parsed network (500 buses, 171 online of 224 generators, 728 online of 733 branches), the quoted cache layout, and float32 embeddings/observations/targets with Boolean masks. A populated case500 training cache was not available. These are tensor payload bytes, not measured bus bandwidth, host residency, allocator usage, or graph activation memory.

| Tensor | case30 shape | case30 bytes | case500 shape | case500 bytes |
|---|---|---:|---|---:|
| `embedding_bus` | `[72,30,384]` | 3,317,760 | `[72,500,384]` | 55,296,000 |
| `embedding_gen` | `[72,6,384]` | 663,552 | `[72,171,384]` | 18,911,232 |
| `observed_bus` | `[72,30,4]` | 34,560 | `[72,500,4]` | 576,000 |
| `available_bus` | same, bool | 8,640 | same, bool | 144,000 |
| `observed_gen` | `[72,6,2]` | 3,456 | `[72,171,2]` | 98,496 |
| `available_gen` | same, bool | 864 | same, bool | 24,624 |
| Future calendar | `[24,6]` | 576 | `[24,6]` | 576 |
| Flattened target | `[24×252]` | 24,192 | `[24×4476]` | 429,696 |
| **Total** | | **4,053,600** | | **75,480,624** |

Decimal totals are **4.054 MB** and **75.481 MB** per origin. The case500 candidate coordinate count is 281 Pd + 500 Vm + 499 Va + 171 free Pg + 113 generator-bus Qg + 4×728 flows = 4,476; Qd remains determined by power factor and is excluded by the same coordinate policy. This assumes valid training labels support all these candidate coordinates; fitted all-missing masks could reduce it. Evidence: `data/coordinates.py` quoted in §1, the parsed `/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/assets/networks/pglib_opf_case500_goc.m`, and the synthetic input shapes in the profile's `solo_case500_goc_k8/rank_0.json`.

This payload is independent of K. B origins multiply it approximately by B; overlapping windows are copied separately by the current API. Each sample-parallel rank would duplicate history transfers unless a new sharing/broadcast path is implemented. Device-side template cloning and temporary activations are separate costs.

### 4.2 Origin order and the meaning of 4,090 windows

The quoted `CachedWindows` creates stride 1 for training and 24 for selection, without reading the similarly named stride configuration keys. For a 504-hour episode, origins are `range(72, 481, 1)`: **409** windows. Ten training episodes give **4,090**. Two selection episodes at midnight give 2×18 = **36** origins. The manifest independently encodes these totals as episode-count times the literals 409 and 18 (§3.4); it does not derive them from the loaded index length.

The order is a CPU `randperm(count)` seeded by `seed + epoch` (`training/state.py:82–83`, quoted above). The loop indexes one identity at a time and resumes from `step_in_epoch` (§2's full loop excerpt). There is no DataLoader or batched sampler. Two origins can be fetched through two independent `window` calls, but neither this method nor the adapter interprets a leading origin-batch dimension. `steps` currently counts both optimizer updates and consumed origins; those counts diverge when B > 1.

### 4.3 Validation and model selection

Selection uses fresh CPU generators seeded by 1042 plus origin index, keeping common random numbers across epochs without consuming the training generator. It forms **eight chunks of eight trajectories**, each 8×24 = 192 graphs, then concatenates all 64 joint trajectories for one ES. Across 36 origins this is 55,296 snapshot forwards. The loop's integer division silently omits a remainder if `forecast_samples` is not divisible by `training_samples`; there is no explicit divisibility check here.

**Source [training/loop.py:84–104](/home/siebenschuh/Projects/genco-foresight/training/loop.py:84) — verbatim:**

```python
def selection_score(model,template,cache,cfg,device,guard):
    scores=[];model.eval()
    try:
        with torch.no_grad():
            for index,identity in enumerate(cache.indices["model_selection"]):
                history,calendar,target=cache.window(identity,device)
                # Local CPU generator: common random numbers across epochs, no training RNG consumption.
                g=torch.Generator().manual_seed(cfg["stage_b"]["selection_noise_seed"]+index)
                predictions=[]
                for _ in range(cfg["stage_b"]["forecast_samples"]//cfg["stage_b"]["training_samples"]):
                    guard()
                    k=cfg["stage_b"]["training_samples"]
                    noise={"factor":torch.randn(k,8,generator=g),"bus":torch.randn(k,24,model.grid.n_bus,3,generator=g),
                           "gen":torch.randn(k,24,model.grid.n_online_gen,1,generator=g)}
                    result=model(history,calendar,template,{key:v.to(device) for key,v in noise.items()})
                    predictions.append(result["joint"])
                score=fair_energy_score(torch.cat(predictions),target,model.coordinate_scale.repeat(24))
                if not torch.isfinite(score):raise RuntimeError("Nonfinite selection energy score")
                scores.append(float(score))
        return sum(scores)/len(scores)
    finally:model.train()
```

The full 64-sample ES is calculated once per origin and averaged equally across the 36 origins. Validation is forward-only but includes ES computation, host conversion, history transfer and a time guard between chunks. A guard interruption does not select a partially evaluated epoch. Strictly lower selection ES replaces the best score; patience counts non-improving completed epochs, with no configured minimum improvement.

**Source [training/loop.py:191–207](/home/siebenschuh/Projects/genco-foresight/training/loop.py:191) — verbatim:**

```python
            score=selection_score(model,template,cache,cfg,device,guard)
            best=progress["best_selection"]
            improved=best is None or score<best
            if improved:progress.update(best_selection=score,best_epoch=epoch+1,bad_epochs=0)
            else:progress["bad_epochs"]+=1
            progress["epochs"].append({"epoch":epoch+1,"selection_energy_score":score,"steps":progress["steps"]})
            progress["epoch"]+=1;progress["step_in_epoch"]=0
            if progress["bad_epochs"]>=cfg["stage_b"]["early_stopping"]["patience"]:
                progress.update(finished=True,stop_reason="early_stopping")
            if progress["epoch"]>=cfg["stage_b"]["max_epochs"]:
                progress.update(finished=True,stop_reason="max_epochs")
            if tensor_digest(model.backbone.state_dict())!=base_digest:raise RuntimeError("Frozen base weights changed")
            save(root/f"epoch_{progress['epoch']:03d}.pt")
            if improved:save(root/"best.pt")
            save();last_save=time.time()
            write_json(root/"epochs.json",progress["epochs"])
            print({"epoch_completed":progress["epoch"],"selection_energy_score":score,"bad_epochs":progress["bad_epochs"]},flush=True)
```

The profile's “validation seconds” measures 36×64 forward-only work on synthetic inputs (§6), not this entire selection routine. It excludes scoring and is not an audited end-to-end validation duration. `epochs.json` records the completed selection curve; checkpoint choice is based on selection ES alone, not balance penalty, calibration, or test data.

## 5. Configuration surface

There is **no root `config.yaml`** in this working tree. The default is `configs/v3.yaml`; v4 is selected through `GENCO_CONFIG` or an explicit loader path. A minimal `configs/multigrid/<case>.yaml` inherits v4 before case-specific paths/checkpoint identifiers are derived. OmegaConf merges applicable overrides, resolves `${base}`, removes the helper key and returns ordinary containers.

**Source [common.py:16–35](/home/siebenschuh/Projects/genco-foresight/common.py:16) — verbatim:**

```python
def config_path(path=None):
    """Explicit per-process variants; the default checkout configuration is v3."""
    return Path(path if path is not None else os.environ.get("GENCO_CONFIG", ROOT / "configs/v3.yaml"))


def load_config(path=None, *, overrides=None):
    """Apply optional mapping overrides, resolve references, and return plain containers."""
    config = OmegaConf.load(config_path(path))
    if OmegaConf.is_dict(config) and "grid" in config and set(config) <= {"grid", "base"}:
        from data.grids import grid_config
        config = OmegaConf.merge(OmegaConf.load(ROOT / "configs/v4.yaml"), config)
        if overrides is not None:
            config = OmegaConf.merge(config, overrides)
        config = OmegaConf.create(grid_config(config.grid, OmegaConf.to_container(config, resolve=False)))
    if overrides is not None:
        config = OmegaConf.merge(config, overrides)
    resolved = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if isinstance(resolved, dict):
        resolved.pop("base", None)
    return resolved
```

**Source [configs/multigrid/case500_goc.yaml:1–2](/home/siebenschuh/Projects/genco-foresight/configs/multigrid/case500_goc.yaml:1) — verbatim:**

```yaml
# Use the v4 recipe with this case and separate multigrid outputs.
grid: case500_goc
```

**Source [data/grids.py:22–47](/home/siebenschuh/Projects/genco-foresight/data/grids.py:22) — verbatim:**

```python
def grid_config(case, config=None):
    """Select a case and isolate all generated artifacts from the released runs."""
    catalog = registry()
    entry = catalog["grids"][case]
    cfg = deepcopy(config if config is not None else load_config(ROOT / "configs/v4.yaml"))
    cfg["grid"] = case
    cfg["task"]["canonical_case"] = Path(entry["pglib"]).stem
    assets = Path(cfg["paths"]["assets_root"])
    cfg["paths"]["network_file"] = str(assets / "networks" / entry["pglib"])
    cfg["paths"]["n1_cache"] = str(Path(cfg["paths"]["opfdata_root"]) / "processed/N1" / Path(entry["pglib"]).stem)
    cfg["paths"]["t1_cache"] = str(Path(cfg["paths"]["opfdata_root"]) / "processed/T1" / Path(entry["pglib"]).stem)
    cfg["paths"]["n1_splits"] = str(Path(cfg["paths"]["opfdata_root"]) / "processed/N1/splits" / Path(entry["pglib"]).stem)
    cfg["checkpoint"].update(repository=catalog["checkpoint_repository"], revision=catalog["checkpoint_revision"],
                             directory=f"{case}/seed42", configuration=entry["yaml"])
    cfg["paths"]["runs_root"] = str(assets.parent / "runs")
    run = Path(cfg["paths"]["runs_root"]) / "multigrid" / case
    cfg["paths"]["runs_root"] = str(run)
    cfg["paths"]["data_root"] = str(run / "datasets")
    for key, suffix in (("temporal_root", "data"), ("stage_a_run", "audit"), ("evidence_root", "audit"), ("training_run", "adapter")):
        cfg["paths"][key] = str(run / suffix)
    cfg["dataset"]["id"] = f"temporal_{case}_v4"
    cfg["stage_b"]["coordinate_scales"] = str(run / "audit/stage_a_followup/train_coordinate_scales.json")
    cfg["stage_b"]["calibration"]["selection"] = "calibration_support"
    for key in ("headline_voltage_bus_ids", "continuous_vm_bus_ids", "continuous_loading_line_ids", "secondary_thermal_daily_line_ids"):
        cfg["stage_b"]["calibration"].pop(key, None)
    return cfg
```

The v4 temporal geometry and split are:

**Source [configs/v4.yaml:34–45](/home/siebenschuh/Projects/genco-foresight/configs/v4.yaml:34) — verbatim:**

```yaml
task:
  canonical_case: pglib_opf_case30_ieee
  fixed_topology: true
  history_hours: 72
  horizon_hours: 24
  episodes: 20
  days_per_episode: 21
  split_episodes:
    train: 10
    model_selection: 2
    risk_calibration: 4
    test: 4
```

### 5.1 Complete current v4 stage_b mapping

This is the whole mapping, including descriptive/reporting keys that the training loop does not consume. It is quoted without resolving `${base}` so the reusable path syntax remains visible. For multigrid runs, `grid_config` replaces the coordinate-scale path and selected calibration fields as shown above.

**Source [configs/v4.yaml:100–250](/home/siebenschuh/Projects/genco-foresight/configs/v4.yaml:100) — verbatim:**

```yaml
stage_b:
  training_origin_stride_hours: 1
  evaluation_origin_stride_hours: 24
  forecast_samples: 64
  training_samples: 8
  forecast_batch_size: 1
  future_graph_batch_size: 192
  training_budget_seconds: 21600
  coordinate_scales: ${base}/GENCO/runs/stage_a_v4/stage_a_followup/train_coordinate_scales.json
  early_stopping:
    metric: model_selection_energy_score
    patience: 3
  checkpoint_each_epoch: true
  checkpoint_tensor_device: cpu
  checkpoint_interval_seconds: 900
  job_training_guard_seconds: 3300
  pbs_walltime_seconds: 3600
  pbs_queue: debug
  scheduler: constant
  training_requires: stage_a_audit
  prevalence_is_training_blocker: false
  optimizer:
    name: AdamW
    lr: 0.0001
    weight_decay: 0.01
    gradient_clip_norm: 1.0
  selection_noise_seed: 1042
  max_epochs: 1000
  energy_score:
    estimator: fair_u_statistic
    distance: scaled_euclidean_div_sqrt_observed_dimension
    pairs: all_distinct_ordered_sample_pairs
  history_inputs:
    bus:
    - pd_mw
    - qd_mvar
    - vm
    - va_rad
    gen:
    - pg_mw
    - qg_mvar
    include_availability_masks: true
    branch_flows: false
  ridge:
    sharing: separate_per_active_demand_bus
    strategy: direct_24_horizon_heads
    alpha: 1.0
    standardize_features_on_train: true
    lags_hours:
    - 1
    - 2
    - 24
    - 48
    observed_covariates:
    - pd_mw
    - qd_mvar
    - vm
    - va_rad
    - bus_pg_mw
    - bus_qg_mvar
    calendar:
    - hour_sin
    - hour_cos
    - dow_sin
    - dow_cos
    - weekend
    residual_initial_fit_hours: 168
    residual_origin_stride_hours: 1
    residual_refit: all_train_episode_prefixes_strictly_before_each_origin
    residual_sampling: whole_24_hour_all_bus_blocks_with_replacement
    demand_bound: clip_at_zero_and_report_clipped_mass
  adapter:
    embedding_width: 384
    observed_projection_width: 32
    gru_input_width: 416
    gru_hidden_width: 32
    gru_layers: 1
    gru_bias: true
    conditioning_output_width: 48
    future_features:
    - hour_sin
    - hour_cos
    - dow_sin
    - dow_cos
    - weekend
    - lead_div_24
    future_projection: shared_linear_6_to_32_added_to_node_memory
    covariance_rank: 8
    bus_head_width: 28
    gen_head_width: 9
    head_design: shared_linear_per_type_no_hidden_layer
    bus_head_coordinates:
    - demand_mean
    - demand_diag_and_8_factors
    - vm_diag_and_8_factors
    - va_diag_and_8_factors
    gen_head_coordinates:
    - pg_diag_and_8_factors
    innovations: zero_mean_state_innovations_before_bounds
    joint_factor: one_independent_8_vector_per_trajectory_shared_across_nodes_and_hours
    mask_constants: true
  loss:
    energy_score_weight: 1.0
    power_balance_weight: 1.0
    power_balance_definition: mean_squared_P_and_Q_bus_residuals_div_checkpoint_baseMVA_squared
    opfdata_replay: false
  figures:
    png_backend: matplotlib
  calibration:
    headline_voltage_bus_ids:
    - 1
    - 9
    - 11
    - 13
    - 15
    - 21
    - 22
    headline_event_resolutions:
    - hourly
    headline_thermal_line_ids:
    - 1
    secondary_event_resolutions:
    - daily_any
    daily_calibration_prevalence_range:
    - 0.2
    - 0.8
    continuous_diagnostics:
    - pit
    - coverage_50
    - coverage_90
    - interval_width_50
    - interval_width_90
    continuous_voltage_bus_ids:
    - 1
    - 9
    - 11
    - 13
    - 15
    - 21
    - 22
    continuous_loading_line_ids:
    - 1
    secondary_thermal_daily_line_ids:
    - 1
    prevalence_shift_ratio: 1.5
    prevalence_ratio_definition: max_calibration_test_rate_div_min_rate
    posthoc_fit_partition: risk_calibration
    test_display: raw_and_mapped_side_by_side
    shifted_event_posthoc_role: report_side_by_side_with_shift_caption
    prevalence_table_beside_every_reliability_diagram: true
    suppress_panel_for_prevalence_shift: false
```

### 5.2 Operational versus descriptive settings

| Setting/assumption | What the current implementation actually does | Quoted evidence |
|---|---|---|
| Determinism | Always `True` in production; no config toggle. The profiler takes its own flag. | `training/loop.py:33,41`; §2, §6 |
| K | `training_samples=8` drives noise; template count must equal K×configured horizon. | `training/loop.py:54–55,67` |
| Future graphs | `future_graph_batch_size=192` is read independently; changing K also needs this count updated. | `training/loop.py:54–55` |
| Origin batch B | `forecast_batch_size=1` is not read by the training step/adapter; one identity, one target, no B axis. | `training/loop.py:65–81`; `training/inputs.py:100–103` |
| History/horizon | 72 and 24 appear as literals in cache slicing/indexing; 24 appears in noise, scale repetition and selection. Changing YAML alone is insufficient. | `training/inputs.py:79–103`; `training/loop.py:67,72,96–100` |
| Origin strides | Training/selection use hard-coded 1/24, rather than their declared stride settings. | `training/inputs.py:97–98` |
| Origin totals | Manifest uses episode counts ×409 and ×18. | `training/loop.py:31–32` |
| Selection samples | 64, chunked with integer `forecast_samples // training_samples`; no remainder support. | `training/loop.py:93–100` |
| Adapter sizes | Widths 384/32/48, covariance rank 8 and heads 28/9 are implemented in the constructor/noise/forward, not selected from the descriptive adapter mapping. Grid node counts come from GridSpec. | `models/temporal.py:18–55,108–118,140–182` |
| Optimizer | AdamW is hard-coded, `foreach=False`; lr, weight decay and clip norm are read. The name string does not instantiate a different optimizer. | `training/loop.py:49–52,77–79` |
| Scheduler | Constant `LambdaLR(..., lambda step: 1.0)` is hard-coded. | `training/loop.py:52` |
| Loss | ES coefficient is implicitly 1; `energy_score_weight` is not multiplied. Balance coefficient is read: v4 1.0, v3 0.01. No replay term. | `training/loop.py:72–74`; v4 mapping above |
| Checkpoints | Every epoch is unconditional; CPU tensor serialization is unconditional. Interval 900 seconds is read. | `training/loop.py:185–186,203–207`; `training/state.py:47–66` |
| Budget and stop | 21,600 cumulative worker seconds; patience 3; `max_epochs=1000`. Nineteen epochs is a profiling projection, not this max-epoch setting. | `training/loop.py:147–166,191–207` |
| PBS walltime | PBS files request 1 hour and export 3600; loop reads the environment and uses configured guard 3300. Changing the descriptive YAML PBS-walltime key alone does not rewrite these files. | PBS excerpts; `training/loop.py:147–149` |
| Case identity | Checkpoint YAML, network and dataset are validated before training initialization; no temporal dataset is required by the synthetic profiler. | `training/loop.py:44–47`; `models/genco.py:24–48` |

The RNG covariance rank (8), Monte Carlo sample count K (also currently 8), and number of origin batches B (1) are separate dimensions. Reusing the number eight in their names or partition logic would be an error. No inference about “eight tiles required” follows from the covariance rank.

## 6. Measured facts to carry forward

### 6.1 Existing profile, quoted in full

This inventory reuses the completed synthetic profile, without running another model or submitting a job. The profile measured commit `d28b9e29527a0fdd1511175dca1becbef752cfb6`; its report records the later resource-error classification adjustment. That is different provenance from the current inventory HEAD. Raw evidence lives at `/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/`, including `summary.json`, `artifact_sha256.json` and each group's `rank_*.json`/`.log`. The table's generator counts are total network counts; actual model/history nodes use online generators (case500 171; case2000 238).

**Source [reports/multigrid/profile.md:1–78](/home/siebenschuh/Projects/genco-foresight/reports/multigrid/profile.md:1) — verbatim:**

```markdown
# Multigrid step and tile profile

Job `8962223.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov` on `x4112c2s6b0n0`; status **complete_with_capacity_limits**; [raw results](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/summary.json). Measured source `d28b9e29527a0fdd1511175dca1becbef752cfb6`. Float32; unchanged `training.loop.training_step`; synthetic inputs/scales, no datasets.
Each XPU process uses one Intel Max 1550 tile; the CPU row uses 16 threads. Times are median / p90 seconds with XPU synchronization at phase boundaries. Forward includes the history encoder, sampling, GENCO and physics; optimizer includes gradient clipping and scheduling; total also includes transfers, RNG and the existing digest. Per-step JSON writes are outside timers; co-tenancy includes those gaps. Memory is peak allocated device GB; reserved/host memory and tile details are in JSON.
Epoch estimates use 4,090 steps; 19-epoch estimates add measured 36×64 forward-only validation once per epoch when available († otherwise training only). Setup, checkpointing, validation scoring and queue wait are excluded. K=1 cannot train with the unchanged fair energy score; smaller K is not run after a larger K fits.

## Table A

| Grid | K | Device / deterministic | Status | Bus / branch / gen | Graphs/step | Fwd s | Loss s | Bwd s | Opt s | Step s | Peak GB | Train s/epoch | Validation s | Hours/19 epochs | Graphs/s |
|---|---:|---|---|---|---:|---|---|---|---|---|---:|---:|---:|---:|---:|
| case14_ieee | 8 | xpu / off | OK | 14/20/5 | 192 | 0.063 / 0.063 | 0.002 / 0.002 | 0.092 / 0.093 | 0.003 / 0.003 | 0.161 / 0.161 | 1.136 | 658.375 | 16.235 | 3.560 | 1192.755 |
| case30_ieee | 8 | xpu / off | OK | 30/41/6 | 192 | 0.066 / 0.066 | 0.002 / 0.002 | 0.095 / 0.095 | 0.003 / 0.003 | 0.166 / 0.167 | 2.111 | 680.239 | 16.154 | 3.675 | 1154.418 |
| case57_ieee | 8 | xpu / off | OK | 57/80/7 | 192 | 0.066 / 0.066 | 0.002 / 0.002 | 0.103 / 0.103 | 0.003 / 0.003 | 0.175 / 0.175 | 3.781 | 714.639 | 18.186 | 3.868 | 1098.849 |
| case118_ieee | 8 | xpu / off | OK | 118/186/54 | 192 | 0.127 / 0.128 | 0.002 / 0.002 | 0.183 / 0.183 | 0.003 / 0.003 | 0.317 / 0.317 | 9.542 | 1295.155 | 38.191 | 7.037 | 606.321 |
| case500_goc | 8 | xpu / off | OK | 500/733/224 | 192 | 0.508 / 0.508 | 0.002 / 0.002 | 0.631 / 0.631 | 0.003 / 0.003 | 1.147 / 1.148 | 36.686 | 4693.116 | 146.713 | 25.544 | 167.326 |
| case2000_goc | 8 | xpu / off | ERROR | 2000/3639/384 | 192 | — / — | — / — | — / — | — / — | — / — | 66.695 | — | — | — | — |
| case2000_goc | 4 | xpu / off | OOM | 2000/3639/384 | 96 | — / — | — / — | — / — | — / — | — / — | 67.742 | — | — | — | — |
| case2000_goc | 2 | xpu / off | OK | 2000/3639/384 | 48 | 0.501 / 0.501 | 0.002 / 0.002 | 0.619 / 0.619 | 0.003 / 0.003 | 1.132 / 1.133 | 37.775 | 4631.574 | 587.142 | 27.543 | 42.387 |
| case30_ieee | 8 | xpu / on | OK | 30/41/6 | 192 | 0.111 / 0.112 | 0.002 / 0.002 | 0.150 / 0.150 | 0.003 / 0.003 | 0.268 / 0.268 | 2.111 | 1094.692 | — | 5.778† | 717.353 |
| case118_ieee | 8 | xpu / on | OK | 118/186/54 | 192 | 0.198 / 0.198 | 0.002 / 0.002 | 0.280 / 0.280 | 0.003 / 0.003 | 0.484 / 0.485 | 9.542 | 1981.396 | — | 10.457† | 396.327 |
| case30_ieee | 8 | cpu / off | OK | 30/41/6 | 192 | 0.369 / 0.371 | 0.001 / 0.001 | 0.424 / 0.427 | 0.003 / 0.003 | 0.796 / 0.800 | — | 3256.294 | — | 17.186† | 241.158 |

`solo_case2000_goc_k8` rank 0: Level Zero OUT_OF_RESOURCES; retained original backend error.

`solo_case2000_goc_k4` rank 0: XPU memory exhausted; allocation details in raw JSON.

case2000_goc validation used K=1 chunks for all 64 samples per origin; training used K=2.

The case2000 smaller-K retries used the same allocation with the suite controller paused; no concurrent timing overlapped the retries. The resource-error fallback was corrected after the step measurements; numerical code was unchanged.

## Table B

| Grid | Fits one tile / K | Single-tile hours | Tiles: 1 h / 3 h / 6 h, ideal | Recommended mode |
|---|---|---:|---|---|
| case14_ieee | Y / 8 | 3.560 | 4 / 2 / 1 | one tile per model |
| case30_ieee | Y / 8 | 3.675 | 4 / 2 / 1 | many models per node |
| case57_ieee | Y / 8 | 3.868 | 4 / 2 / 1 | one tile per model |
| case118_ieee | Y / 8 | 7.037 | 8 / 3 / 2 | DDP across tiles/nodes |
| case500_goc | Y / 8 | 25.544 | 26 / 9 / 5 | DDP across tiles/nodes |
| case2000_goc | Y / 2 | 27.543 | 28 / 10 / 5 | sample-parallel across tiles |

The requested ‘upper bound under ideal scaling’ applies to speedup: ceil(hours/target) is an optimistic **lower bound on required tiles**, not measured scaling or a memory guarantee. Multi-tile recommendations are prospective; no model sharding or DDP was tested.

## Table C

| Concurrent chains | Process / tile | Step median / p90 s | Median as % of solo | Status |
|---:|---:|---|---:|---|
| 1 | 0 | 0.163 / 0.163 | 100.000 | OK |
| 4 | 0 | 0.165 / 0.174 | 101.605 | OK |
| 4 | 1 | 0.164 / 0.172 | 100.632 | OK |
| 4 | 2 | 0.165 / 0.165 | 101.463 | OK |
| 4 | 3 | 0.164 / 0.165 | 100.907 | OK |
| 8 | 0 | 0.170 / 0.171 | 104.291 | OK |
| 8 | 1 | 0.167 / 0.168 | 102.680 | OK |
| 8 | 2 | 0.164 / 0.165 | 100.982 | OK |
| 8 | 3 | 0.168 / 0.169 | 103.093 | OK |
| 8 | 4 | 0.168 / 0.169 | 103.078 | OK |
| 8 | 5 | 0.168 / 0.169 | 103.512 | OK |
| 8 | 6 | 0.172 / 0.173 | 105.778 | OK |
| 8 | 7 | 0.160 / 0.161 | 98.607 | OK |
| 12 | 0 | 0.170 / 0.171 | 104.556 | OK |
| 12 | 1 | 0.172 / 0.173 | 105.900 | OK |
| 12 | 2 | 0.170 / 0.171 | 104.672 | OK |
| 12 | 3 | 0.172 / 0.173 | 105.661 | OK |
| 12 | 4 | 0.169 / 0.170 | 103.735 | OK |
| 12 | 5 | 0.172 / 0.173 | 105.688 | OK |
| 12 | 6 | 0.162 / 0.163 | 99.604 | OK |
| 12 | 7 | 0.164 / 0.166 | 101.087 | OK |
| 12 | 8 | 0.173 / 0.174 | 106.213 | OK |
| 12 | 9 | 0.174 / 0.176 | 107.210 | OK |
| 12 | 10 | 0.172 / 0.173 | 105.687 | OK |
| 12 | 11 | 0.174 / 0.176 | 107.259 | OK |

![Step time by grid and K](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/step_time.png)

- Single-tile 1 h estimate: none established.
- Requires chained 1 h jobs under this recipe: case14_ieee, case30_ieee, case57_ieee, case118_ieee, case500_goc, case2000_goc.
- Single-tile six-hour target exceeded or a valid step not established: case118_ieee, case500_goc, case2000_goc; synthetic timing does not establish convergence.
```

### 6.2 What the timers include

The profiler explicitly sets determinism from its flag, constructs a synthetic fixture, and calls the unchanged `training_step`; it does not call production `initialize`, which would force the flag back on. It synchronizes at phase boundaries, performs five warmups, then measures 30 steps. It includes the existing noise hash and host/device copies in total time. Its per-step JSON writes fall outside that timer; their gaps are still present during co-tenancy. Synthetic history/scales exercise shapes and computation but do not measure Lustre cache throughput, actual convergence, or end-to-end job startup.

**Source [scripts/multigrid/profile_step.py:195–214](/home/siebenschuh/Projects/genco-foresight/scripts/multigrid/profile_step.py:195) — verbatim:**

```python
        from training.loop import training_step
        from training.state import seed_all, tensor_digest
        torch.set_num_threads(16 if args.device == "cpu" else 1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(args.deterministic == "on")
        seed_all(42)
        if args.device == "xpu":
            if not torch.xpu.is_available() or torch.xpu.device_count() != 1:
                raise RuntimeError(f"Expected one isolated XPU tile; saw {torch.xpu.device_count()}")
            torch.xpu.set_device(0)
            props = torch.xpu.get_device_properties(0)
            record.update(device_name=props.name, device_total_bytes=props.total_memory,
                          visible_devices=torch.xpu.device_count())
            torch.xpu.reset_peak_memory_stats()
        record.update(torch_version=torch.__version__, dtype="float32", cpu_threads=torch.get_num_threads(),
                      cpu_affinity_count=len(os.sched_getaffinity(0)))
        cfg, model, optimizer, scheduler, template, cache = fixture(args, record)
        synchronize = torch.xpu.synchronize if args.device == "xpu" else lambda: None
        base_before = tensor_digest(model.backbone.state_dict())
        record["setup_seconds"] = time.time() - started
```

**Source [scripts/multigrid/profile_step.py:242–272](/home/siebenschuh/Projects/genco-foresight/scripts/multigrid/profile_step.py:242) — verbatim:**

```python
            timer = PhaseTimer(synchronize)
            with patch.object(model, "forward", timer.forward(model.forward)), \
                 patch.object(torch.Tensor, "backward", timer.backward(torch.Tensor.backward)), \
                 patch.object(scheduler, "step", timer.scheduler(scheduler.step)):
                warmup_start = time.perf_counter()
                for _ in range(5):
                    training_step(model, optimizer, scheduler, template, cache, 0, cfg, args.device)
                record["warmup_seconds"] = time.perf_counter() - warmup_start
                save(args.output / f"ready_{args.rank}.json", {"ready": True})
                while not (args.output / "go.json").exists():
                    if time.time() >= args.deadline:
                        raise TimeoutError("Timed out waiting for other tiles")
                    time.sleep(.05)
                if args.device == "xpu":
                    record["setup_warmup_peak_bytes"] = torch.xpu.max_memory_allocated()
                    torch.xpu.reset_peak_memory_stats()
                record["timed_start_epoch"] = time.time()
                for step in range(args.steps):
                    timer.row = {}
                    start = timer.clock()
                    metrics = training_step(model, optimizer, scheduler, template, cache, step, cfg, args.device)
                    timer.row["total"] = timer.clock() - start
                    record["timings"].append(dict(timer.row))
                    record["last_metrics"] = metrics
                    save(path, record)
                    if time.time() >= args.deadline:
                        raise TimeoutError("Job time guard during timed steps")
                record["timed_end_epoch"] = time.time()
            record.update(status="OK", phase_seconds=summaries(record["timings"]),
                          warmup_steps=5, timed_steps=len(record["timings"]))
        record["backbone_unchanged"] = tensor_digest(model.backbone.state_dict()) == base_before
```

Forward-only validation has its own loop; in particular, it generates device noise and does not call `selection_score` or save/score 64 concatenated trajectories. Those costs differ from production selection.

**Source [scripts/multigrid/profile_step.py:217–240](/home/siebenschuh/Projects/genco-foresight/scripts/multigrid/profile_step.py:217) — verbatim:**

```python
        if args.validation_only:
            model.eval()
            history, calendar, _ = cache.window(0, args.device)
            with torch.no_grad():
                for _ in range(5):
                    result = model(history, calendar, template, model.noise(args.K, 24))
                    del result
                synchronize()
                record["validation_seconds"] = 0.
                for origin in range(36):
                    validation_start = time.perf_counter()
                    history, calendar, _ = cache.window(origin, args.device)
                    for _ in range(64 // args.K):
                        result = model(history, calendar, template, model.noise(args.K, 24))
                        if origin == 0 and not torch.isfinite(result["joint"]).all():
                            raise RuntimeError("Nonfinite validation predictions")
                        del result
                    synchronize()
                    record["validation_origins_completed"] = origin + 1
                    record["validation_seconds"] += time.perf_counter() - validation_start
                    save(path, record)
                    if time.time() >= args.deadline:
                        raise TimeoutError("Job time guard during validation")
            record.update(status="OK", validation_graphs=36 * 64 * 24)
```

Allocated peak memory is the larger of the setup/warmup and timed peaks, in decimal GB in the report; reserved memory and host RSS are separate JSON fields. The device reports 68,702,699,520 bytes of capacity, about 68.70 GB or 63.98 GiB. A “64 GB” informal tile description must not be compared numerically to decimal-GB columns without converting units.

**Source [scripts/multigrid/profile_step.py:278–287](/home/siebenschuh/Projects/genco-foresight/scripts/multigrid/profile_step.py:278) — verbatim:**

```python
    finally:
        record["elapsed_seconds"] = time.time() - started
        record["host_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if torch is not None and args.device == "xpu" and torch.xpu.is_initialized():
            try:
                record["timed_peak_device_bytes"] = torch.xpu.max_memory_allocated()
                record["peak_device_bytes"] = max(record["timed_peak_device_bytes"], record.get("setup_warmup_peak_bytes", 0))
                record["peak_reserved_bytes"] = torch.xpu.max_memory_reserved()
            except RuntimeError as error:
                record["memory_query_error"] = str(error)
```

### 6.3 Deterministic-mode cost and memory limits

At fixed case/K, turning strict determinism off reduced median step time from 0.267651 to 0.166318 s for case30 (**1.609× throughput**, 37.86% less time), and 0.484449 to 0.316664 s for case118 (**1.530×**, 34.63% less time). These are measured mode comparisons, not convergence or nondeterministic-resume validation.

Peak allocated memory at the largest successful K was: case14 **1.136 GB** (K8), case30 **2.111 GB** (K8), case57 **3.781 GB** (K8), case118 **9.542 GB** (K8), case500 **36.686 GB** (K8), case2000 **37.775 GB** (K2). case2000 K8 failed with a backend resource error and K4 failed with OOM; neither is a successful timing sample. Every successful worker checks the frozen backbone digest after its run. These results establish neither memory for B>1 nor memory savings from model sharding.

The twelve-process case30 co-tenancy run kept per-process step medians within roughly −0.4% to +7.3% of its own solo reference. Those are twelve independent complete models, not a twelve-tile acceleration of one update. No communication cost, hardware utilization percentage, FLOPS, sample-parallel efficiency or DDP scaling was measured. CPU case30 K8 at 16 threads took 0.796160 s/step in the same synthetic profile.

### 6.4 The proposed time model is a heuristic, not a fitted result

The expression `t ≈ 0.155 s + 1.0e-5 s × (graphs × n_bus)` was supplied in the request; the existing profile report does not fit or establish those coefficients. Comparing it with the recorded off-mode medians gives:

| Grid | K | Graphs × buses | Measured step s | Proposed formula s | Formula error |
|---|---:|---:|---:|---:|---:|
| case14_ieee | 8 | 2,688 | 0.160972 | 0.181880 | +12.99% |
| case30_ieee | 8 | 5,760 | 0.166318 | 0.212600 | +27.83% |
| case57_ieee | 8 | 10,944 | 0.174728 | 0.264440 | +51.34% |
| case118_ieee | 8 | 22,656 | 0.316664 | 0.381560 | +20.49% |
| case500_goc | 8 | 96,000 | 1.147461 | 1.115000 | -2.83% |
| case2000_goc | 2 | 96,000 | 1.132414 | 1.115000 | -1.54% |

The formula overpredicts case57 by about 51%. The six points mix topology, branch density, online-generator count and K; they cannot isolate a universal fixed overhead or predict origin batching. Carry it forward only as a coarse hypothesis to test, not a measured law or a promise of linear scaling.

The existing epoch projection is `(4090 × median_step + measured_forward_only_validation) × 19 / 3600`. It omits setup, checkpointing, validation ES and queue time. The projected times are about 3.56/3.68/3.87/7.04/25.54/27.54 hours for 14/30/57/118/500/2000 at their reported K; case2000's K2 objective estimator has a different sample count from K8. None establishes a full 19-epoch, single-tile one-hour run. These projections motivate chains for the current loop; they do not say how many epochs a new run needs to converge.

`ceil(projected_hours / target_hours)` is an **optimistic lower bound on tile count** assuming ideal speedup, not an upper bound on required resources. At fixed global K8, sample parallelism has at most eight nonempty sample workers. More ranks require changing K, batching origins, or another decomposition; the profile's 26/28-tile arithmetic is not a directly executable sample-parallel plan.

## 7. Three staged implementation boundaries for review

The following are proposals derived from the quoted code, not changes implemented by this task. Keep full precision, the network/backbone, all 24 forecast hours, coordinate masks/scales and the fair-ES definition fixed while assessing each stage. bf16, horizon subsampling, truncated backpropagation, demand-only architecture changes and optimizer retuning are outside these three changes.

| Stage | Smallest coherent change surface | Acceptance evidence required before the next stage |
|---|---|---|
| **1. Production determinism off** | Add a production flag to `initialize` and `make_manifest`; keep strict deterministic resume/debug mode. Version the run. Add the missing fixed-cadence loss/ES/balance/norm logs needed for a statistical resume comparison. | Same model/data/K/B/optimizer; finite gradients and frozen checksum; exact serialized state/order/RNG restoration; predeclared repeat-run numerical criteria for selection/component curves and stopping. Include a restart that crosses an epoch/validation boundary. Re-measure full production time rather than substituting the profiler's synthetic duration. |
| **2. Sample parallelism for one origin** | A cooperating process group and launcher; replicate full model/history and partition global K into disjoint draws. Run local `K_r×24` graphs, differentiably gather joint trajectories for the unchanged ES, reduce balance with proper sample weights, and reduce adapter gradients exactly once with correct normalization. Rank-0 persistence must gather per-rank RNG/progress before checkpointing. | First compare serial global K8 against, for example, four ranks ×K2 with identical prescribed noise: loss, each gradient and one optimizer update, using tolerances declared for the backend. Include cross-rank pairs and an uneven split test. Then test all-rank resume, validation, early stopping, time guards and failure exits; measure collective and end-to-end costs. |
| **3. Origin batching** | Introduce an explicit B axis in history/calendar/noise/targets and graph ordering; flatten B×nodes for shared per-node-type GRUs, then restore B. Build B×K×24 graphs. Apply the existing fair ES separately to each origin, then average losses. Track origins consumed separately from optimizer steps; specify the last partial batch. | B1 reproduces the reference. With fixed noise, a B-origin update matches the mean of B independently computed per-origin losses/gradients before one optimizer step. No cross-origin ES pairs. Test masks, different episode identities, partial batches, epoch-boundary resume, validation and memory. Compare optimization at fixed total origins seen; any LR/effective-batch policy is a new predeclared choice. |

For stage 2, a plain non-autograd tensor gather or averaging local fair scores breaks the required gradient/objective. A replicated loss on every rank also needs deliberate normalization with DDP's gradient averaging. Implement and test one explicit convention rather than combining both patterns. All ranks must enter the same collectives when stopping or validating; a local time guard cannot leave peers blocked indefinitely. Keeping global K fixed preserves the one-origin Monte Carlo objective; increasing origin batch B changes the number of examples per optimizer update.

Stage 2 does not shard backbone parameters or activations within an individual sample: each tile retains the full frozen GENCO and its local backward graph. The case2000 K2 measurement makes a four-tile K8 sample split a plausible memory experiment, not a measured four-tile result. Stage 3 must preserve the shared covariance factor within each trajectory while assigning independent draws across origins.

No new implementation, training, solver run, queue submission or performance experiment was performed for these proposed stages.

## 8. Source provenance and inventory checks

SHA256 values below identify the exact files quoted in this inventory. These are file-byte hashes, not model tensor digests. Installed source excerpts retain their upstream module paths; they are evidence, not vendored implementation. Git HEAD alone does not identify pre-existing working-tree edits.

| Quoted source | SHA256 |
|---|---|
| `common.py` | `2f2308e0736422ecd75807db248ef35753a6bc3b2289e6e75e655ec652658e59` |
| `configs/multigrid/case500_goc.yaml` | `d9ad4f827c576e2ae9f9211f0aab0b19abeefb4507f4f0f1c8ff12bf1b88afdc` |
| `configs/replication/family.json` | `f0b151aaa2dfa557d86c099df66c5cbe1f894e064903f16def6f3488f7f14e91` |
| `configs/v4.yaml` | `b5bf7a842114bfa056abe75119cb0c8629966a56a54f692132952a2cda1a649a` |
| `data/coordinates.py` | `f5e5b466379a4a37e38c09528b8938faec36f212a180c2fbebe3ec17dd79ee6a` |
| `data/grids.py` | `4c2fc812277d9761f2b6bf896a8552657bf95a1d890937203d1650d60060b4d1` |
| `models/genco.py` | `01000bdd73fe30af4a54ec6f2e27490c064fd0d0c7d8d2297dc7c6407f23007c` |
| `models/temporal.py` | `71f1746b1d7bea0e9cbe513ccb460e79eb5e3c1ecf5a77041232814515cda20c` |
| `reports/multigrid/profile.md` | `3baff01ae0a57b096af9722807f47413796a2406ca8d35db5ce15e7428cfd6f8` |
| `scoring.py` | `4ec8448fb6b13f611bc30a9c7e27ef56f938f1984eeb170025b6696c48eecc96` |
| `scripts/multigrid/profile_step.py` | `1122108ba48ba3285467946b47bcd6a3f08f5b2e97cac778306b87507212f509` |
| `scripts/replication_chain.pbs` | `0fb0fe57fbe217c4385f3de862398099033b72b45b051755523112d427e5b4fd` |
| `scripts/replication_chain.py` | `53c1ae34508a5e6fb032db5833783159e6726e839d92a717630b95616b5765c8` |
| `scripts/resume_test.py` | `77be123c59afe05f5f1c15390d2023e9b69381060587495e078b238ba4b6b178` |
| `scripts/train_adapter.pbs` | `b04730432bc62347444768505fbd1f7e620d127f4b35973ed4a0223769cea79f` |
| `scripts/training_chain.py` | `ad4f532ab211e0c6ba984665e0349d43c2ae90045e5d19cd1c37b23058fa868b` |
| `training/inputs.py` | `738588d8432da7d958863f9240d5f590678cc22ac2e2f266edb473149d63badd` |
| `training/loop.py` | `3621d84a925ff9acd8109cf71b9bb7fdabc60575923a97f2b5bc37a0ea0840e5` |
| `training/state.py` | `f197af1a491c6de21564497b49c337de037e0cd2d31a65f9440cfeab0647b7e1` |
| `/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/solo_case30_ieee_k8/rank_0.log` | `79b40400e0a11659a257095686cef236c131866e56476ad9fdcc0de3cb704060` |
| `/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/lib/python3.12/site-packages/gridfm_graphkit/models/gnn_heterogeneous_gns.py` | `0ae2c1351629884e0b1358bf953fa70601df0b9cc5ef308c419f424dde28821e` |
| `/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/lib/python3.12/site-packages/gridfm_graphkit/models/utils.py` | `b776c8323c52b87f88bd975487f3829080933c1908f8d56ce17fe49beb3c8ea3` |
| `/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/lib/python3.12/site-packages/torch_geometric/data/data.py` | `f84e9e975927360489c9aa55f0a524d9a4511ce7efc04be35057defc16a92df7` |
| `/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/lib/python3.12/site-packages/torch_geometric/nn/aggr/basic.py` | `7f99393eba4ffa8c5de2bc8778bab2413b358f3889ec986fe58c6790ec963bf6` |
| `/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/lib/python3.12/site-packages/torch_geometric/nn/conv/transformer_conv.py` | `ffa59f24f9e0854101d50f067a750d12f315c0852d2cf8c6a3f80a65f0da38cc` |
| `/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/lib/python3.12/site-packages/torch_geometric/utils/_scatter.py` | `14fb794f8d706e0d2294b32d01c251e12c0faa5c43ae30ea3bb2a003e6276f0a` |
| `/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/lib/python3.12/site-packages/torch_geometric/utils/_softmax.py` | `057d3ac17c52abe055b6fba6368c95f064bdaadb16258c0dac50054498efe1c5` |
| `/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/lib/python3.12/site-packages/torch_scatter/scatter.py` | `ff6a3fc29e3461675986fb73a3a82f6a76c264adc4dd3e6d1dbd01e8d0f11be2` |
| `/opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0/lib/python3.12/site-packages/torch/nn/utils/clip_grad.py` | `5e30dac17eb0311ab1a12e46ed2484decb66165427644d2601d13570ab0ee4d7` |

All 65 verbatim source excerpts were compared against their stated file/line ranges. AST inspection confirmed complete source coverage for `training_step`, `TemporalAdapter.forward`, `TemporalAdapter.noise`, and `fair_energy_score`. The balance penalty is inline; its complete construction and use are quoted in §1.5. No training or runtime test was needed for this documentation-only edit.

All 167 pre-existing tracked files were rehashed after writing this report: zero changed bytes, zero removed files. Existing uncommitted work was preserved. Only `reports/V2_IMPL.md` is added for this task.
