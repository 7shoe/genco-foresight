"""Load recorded models and produce forecasts with the original RNG order."""
import os
from pathlib import Path
import time
import numpy as np
import torch
from power_fm_foresight._config import read_json, sha256, write_json
from power_fm_foresight._data.dataset import load_temporal_manifest
from power_fm_foresight._data.network import graph_for_demand, load_network
from power_fm_foresight.genco import Genco
from power_fm_foresight.temporal import TemporalAdapter
from power_fm_foresight._training.forecaster import episode_arrays
from power_fm_foresight._training.state import atomic_torch_save, tensor_digest
from .physics import physics

def components(cfg, device):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if device.startswith("xpu"):
        selected = torch.device(device)
        if selected.index is None:
            if os.environ.get("ZE_FLAT_DEVICE_HIERARCHY") != "FLAT" or not os.environ.get("ZE_AFFINITY_MASK") or torch.xpu.device_count() != 1:
                raise ValueError("Unindexed XPU requires one launcher-isolated flat tile")
            selected = torch.device("xpu:0")
        torch.xpu.set_device(selected)
    net = load_network(config=cfg)
    load_temporal_manifest(cfg["paths"]["temporal_root"], net)
    base = Genco(Path(cfg["paths"]["assets_root"]) / "genco-opfdata", cfg["checkpoint"], device, grid=net)
    scales = read_json(cfg["stage_b"]["coordinate_scales"])
    model = TemporalAdapter(base, net, scales, model_policy=cfg["stage_b"].get("adapter",{}).get("model_policy","conditioned")).to(device).eval()
    train = Path(cfg["paths"]["training_run"])
    best = torch.load(train / "best.pt", weights_only=True, map_location="cpu")
    original = train / "manifest.training.json"
    assert best["manifest"] == read_json(original if original.exists() else train / "manifest.json")
    missing, extra = model.load_state_dict(best["model"], strict=False)
    assert not extra and all(k.startswith("backbone.") for k in missing)
    assert best["progress"]["best_epoch"] == read_json(train / "status.json")["best_epoch"]
    assert tensor_digest(base.model.state_dict()) == read_json(train / "status.json")["base_sha256"]
    graph = graph_for_demand(net, net["bus"][:, 2], net["bus"][:, 3])
    templates = {n: base.batch([graph] * n) for n in (24, 192)}
    return net, base, model, scales, templates



def adapter_samples(base, model, net, cached, origin, templates, seed, device):
    """Same sampled computation and RNG order as frozen Stage B; measured end-to-end."""
    if device.startswith("xpu"):
        torch.xpu.synchronize()
    start = time.perf_counter()
    history = {k: v[origin - 72:origin].to(device) for k, v in cached.items()
               if k.startswith(("embedding_", "observed_", "available_"))}
    calendar = torch.cat([cached["calendar"][origin:origin + 24], torch.arange(1, 25)[:, None] / 24], 1).to(device)
    generator = torch.Generator().manual_seed(seed)
    chunks = []
    with torch.no_grad():
        for _ in range(8):
            noise = {"factor": torch.randn(8, 8, generator=generator),
                     "bus": torch.randn(8, 24, net.n_bus, 3, generator=generator),
                     "gen": torch.randn(8, 24, net.n_online_gen, 1, generator=generator)}
            r = model(history, calendar, templates[192], {k: v.to(device) for k, v in noise.items()})
            torch.testing.assert_close(r["values"]["pd_mw"], r["demand_seen_by_physics_mw"], atol=1e-5, rtol=1e-6)
            chunks.append({k: v.cpu().numpy() for k, v in r["values"].items()})
    values = physics({k: np.concatenate([r[k] for r in chunks]) for k in chunks[0]}, net)
    if device.startswith("xpu"):
        torch.xpu.synchronize()
    return values, time.perf_counter() - start



def cache_partition(cfg, out, partition, base, net):
    data = Path(cfg["paths"]["temporal_root"])
    manifest = load_temporal_manifest(data, net)
    files = {}
    for e in manifest["episodes"]:
        if e["split"] != partition:
            continue
        path = data / f"episodes/{e['episode_id']}.json"
        dest = out / f"cache/{e['episode_id']}.pt"
        key = {"episode_sha256": sha256(path), "checkpoint": cfg["checkpoint"], "graphkit": cfg["upstream"]["graphkit"]}
        if dest.exists():
            assert torch.load(dest, mmap=True, weights_only=True)["key"] == key
        else:
            records = read_json(path)["records"]
            arrays = episode_arrays(records, net)
            embedding = {"bus": [], "gen": []}
            for lo in range(0, len(records), 192):
                graphs = [graph_for_demand(net, r["pd_mw"], r["qd_mvar"]) for r in records[lo:lo + 192]]
                with torch.no_grad():
                    _, emb = base.model(base.batch(graphs), return_embeddings=True)
                for kind, count in (("bus", net.n_bus), ("gen", net.n_online_gen)):
                    embedding[kind].append(emb[kind].reshape(-1, count, emb[kind].shape[-1]).cpu())
            values = {f"embedding_{k}": torch.cat(v) for k, v in embedding.items()}
            values["observed_bus"] = torch.tensor(np.stack([arrays[k] for k in ("pd", "qd", "vm", "va_rad")], -1), dtype=torch.float32)
            values["observed_gen"] = torch.tensor(np.stack([arrays[k] for k in ("pg_mw", "qg_mvar")], -1), dtype=torch.float32)[:,net.online_gen]
            for kind in ("bus", "gen"):
                values[f"available_{kind}"] = torch.isfinite(values[f"observed_{kind}"])
            values["calendar"] = torch.from_numpy(arrays["calendar"])
            atomic_torch_save(dest, {"key": key, "data": values})
        files[e["episode_id"]] = {"path": str(dest), "sha256": sha256(dest)}
    write_json(out / "cache/manifest.json", files)
