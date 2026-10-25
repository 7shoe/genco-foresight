"""One prescribed-batch diagnosis against a CPU float64 reference."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from common import ROOT, read_json, sha256, write_json
from scripts.v2.verify_stage import (V2_ROOT, checked_root, copy_cache,
                                     freeze_preparation, invoke, verify_frozen)


GROUPS = {
    "observed_projections": ("bus_observed.", "gen_observed."),
    "GRUs": ("bus_gru.", "gen_gru."),
    "conditioning": ("bus_conditioning.", "gen_conditioning."),
    "calendar": ("future_projection.",),
    "heads": ("bus_head.", "gen_head."),
}
HEADS = ("bus_head", "gen_head", "backbone.mlp_bus", "backbone.mlp_gen")


def gradient_metrics(candidate, reference, floor):
    import torch
    a, b = candidate.double().flatten(), reference.double().flatten()
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Invalid gradient comparison")
    an, bn = float(a.norm()), float(b.norm())
    resolved = an > floor and bn > floor
    return {"cosine": float(torch.dot(a / an, b / bn)) if resolved else None,
            "norm_ratio": an / bn if bn > floor else None,
            "candidate_norm": an, "reference_norm": bn,
            "difference_norm": float((a - b).norm()), "resolved": resolved,
            "coordinates": a.numel()}


def decision(rows, protocol):
    metrics = [m for row in rows for m in row["groups"].values()]
    if any(m["cosine"] is not None and m["cosine"] < protocol["decision"]["unreliable_below"] for m in metrics):
        return "unreliable_gradient_test_selective_fp64"
    if all(m["resolved"] and m["cosine"] >= protocol["decision"]["usable_at_least"] for m in metrics):
        return "usable_gradient_requires_epoch_gate"
    return "inconclusive_hold"


def energy64(samples, target, scale):
    """The existing fair ES formula with all reductions, including sqrt(D), in fp64."""
    import torch
    samples, target, scale = samples.double(), target.double(), scale.double()
    observed = torch.isfinite(target)
    if samples.shape[0] < 2 or not observed.any():
        raise ValueError("Invalid energy score inputs")
    x, y = samples[:, observed] / scale[observed], target[observed] / scale[observed]
    k = samples.shape[0]
    return (torch.linalg.vector_norm(x - y, dim=-1).mean()
            - torch.linalg.vector_norm(x[:, None] - x[None, :], dim=-1).sum() / (2*k*(k-1))) / observed.sum().double().sqrt()


@contextmanager
def double_heads(model, audit):
    """Promote only named head modules and cast their outputs back before conditioning."""
    import torch
    handles = []
    for name in HEADS:
        module = model.get_submodule(name)
        module.double()
        def before(module, args):
            return tuple(value.double() for value in args)
        def after(module, args, output, name=name):
            if args[0].dtype != torch.float64 or output.dtype != torch.float64:
                raise RuntimeError("Head precision boundary violated")
            audit[name] = audit.get(name, 0) + 1
            return output.float()
        handles.extend((module.register_forward_pre_hook(before), module.register_forward_hook(after)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def prepare(root, protocol_path):
    protocol = read_json(protocol_path)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "protocol.json", protocol)
    original = Path(protocol["fixture"]["source_run"])
    cfg = read_json(original / "configuration.json")
    cfg["paths"]["training_run"] = str(root / "shared")
    cfg["v2"] = {"stage": 1, "run_identity": root.name,
                 "protocol_sha256": sha256(root / "protocol.json"), "purpose": "gradient_diagnosis"}
    cfg["stage_b"]["deterministic_algorithms"] = True
    write_json(root / "configuration.json", cfg)
    copy_cache(root)
    original_cache = read_json(original / "shared/cache/manifest.json")
    copied_cache = read_json(root / "shared/cache/manifest.json")
    assert {k:v["sha256"] for k,v in original_cache["files"].items()} == {k:v["sha256"] for k,v in copied_cache["files"].items()}
    evidence = {}
    for name, relative in (("seed42", "adapter_v4"),
                           ("seed1", "replication/case30_ieee/v4/seed1/adapter"),
                           ("seed2", "replication/case30_ieee/v4/seed2/adapter")):
        folder = V2_ROOT.parent / relative
        evidence[name] = {filename: {"path": str(folder / filename), "sha256": sha256(folder / filename),
                                    "value": read_json(folder / filename)}
                          for filename in ("manifest.json", "epochs.json", "status.json")}
    write_json(root / "historical_epochs.json", evidence)
    record = freeze_preparation(root, protocol_path, protocol)
    prepared = read_json(root / "prepared.json")
    for p in (root / "historical_epochs.json", root / "shared/cache/manifest.json",
              original / "gate.json", original / "checkpoint_inspection.json"):
        prepared["protected_sha256"][str(p)] = sha256(p)
    for row in evidence.values():
        for item in row.values():
            prepared["protected_sha256"][item["path"]] = item["sha256"]
    for path, expected in read_json(original / "prepared.json")["source_sha256"].items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f"Stage 1 source changed: {path}")
    prepared["parent_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    write_json(root / "prepared.json", prepared)
    return record


def submit(root):
    verify_frozen(root)
    with (V2_ROOT / "compute.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "submission.json").exists():
            raise ValueError("Single allocation already reserved")
        queue = subprocess.run(["qstat", "-u", os.environ["USER"]], capture_output=True, text=True, timeout=30)
        if queue.returncode or any(line.strip() and line.lstrip()[0].isdigit() for line in queue.stdout.splitlines()):
            raise RuntimeError("Cannot start another compute family")
        protocol = read_json(root / "protocol.json")
        (root / "logs").mkdir()
        record = {"state": "reserved", "reserved_epoch": time.time(),
                  "allocation_cap_seconds": protocol["job"]["walltime_seconds"],
                  "protocol_sha256": sha256(root / "protocol.json")}
        write_json(root / "submission.json", record)
        result = subprocess.run(["qsub", "-o", str(root / "logs/pbs.log"),
                                 "-v", f"GENCO_V2_ROOT={root}", str(ROOT / "scripts/v2/diagnose_stage1.pbs")],
                                cwd=ROOT, capture_output=True, text=True, timeout=45)
        record.update(state="submitted" if result.returncode == 0 else "rejected",
                      returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        if result.returncode == 0:
            record["job_id"] = result.stdout.strip()
        write_json(root / "submission.json", record)
        if result.returncode:
            raise RuntimeError(result.stderr)
        return record


def worker(root):
    import gc
    import torch
    from scoring import fair_energy_score
    from training.loop import initialize, make_manifest
    from training.state import atomic_torch_save, cpu_tree, epoch_order, rng_state, tensor_digest, train_state

    verify_frozen(root)
    if not os.environ.get("PBS_JOBID") or os.environ.get("ZE_FLAT_DEVICE_HIERARCHY") != "FLAT" or os.environ.get("ZE_AFFINITY_MASK") != "0":
        raise RuntimeError("Requires compute allocation with isolated flat tile 0")
    if not torch.xpu.is_available() or torch.xpu.device_count() != 1:
        raise RuntimeError("Expected exactly one visible XPU tile")
    torch.xpu.set_device(0)
    protocol = read_json(root / "protocol.json")
    deadline = float(os.environ["GENCO_V2_JOB_START"]) + protocol["job"]["worker_guard_seconds"]
    cfg = read_json(root / "configuration.json")
    started = time.perf_counter()
    model, optimizer, scheduler, template, cache = initialize(cfg, "xpu")
    expected = protocol["fixture"]
    identity = cache.indices["train"][epoch_order(cfg["seed"], 0, len(cache.indices["train"]))[0]]
    if list(identity) != expected["origin"] or tensor_digest(train_state(model)) != expected["initial_adapter_sha256"]:
        raise ValueError("Recorded first-step initialization/origin did not reproduce")
    history, calendar, target = cache.window(identity, "xpu")
    noise = model.noise(8, 24)
    if tensor_digest(noise) != expected["noise_sha256"]:
        raise ValueError("Recorded first-step noise did not reproduce")
    fixture = {"state": cpu_tree(model.state_dict()), "history": cpu_tree(history),
               "calendar": cpu_tree(calendar), "target": cpu_tree(target), "noise": cpu_tree(noise),
               "rng_after_noise": rng_state()}
    atomic_torch_save(root / "fixture.pt", fixture)
    atomic_torch_save(root / "template.pt", template.to("cpu").to_dict())
    model = model.cpu()
    del optimizer, scheduler, cache, history, calendar, target, noise
    gc.collect()
    torch.xpu.empty_cache()
    manifest = make_manifest(cfg)
    manifest.update(purpose="single_batch_gradient_diagnosis", optimizer_updates=0,
                    source_manifest_semantics="Training recipe provenance; per-evaluation precision/mode are recorded separately",
                    prescribed_origin=list(identity), fixture_sha256=sha256(root / "fixture.pt"),
                    template_sha256=sha256(root / "template.pt"), initialization_and_noise_hashes_verified=True,
                    runtime={"torch": torch.__version__, "node": os.uname().nodename,
                             "job_id": os.environ["PBS_JOBID"], "tile": 0})
    write_json(root / "manifest.json", manifest)
    print("fixture verified", list(identity), flush=True)

    def evaluate(label, device, dtype, deterministic, policy):
        if time.time() >= deadline:
            raise TimeoutError("Diagnostic walltime guard")
        torch.use_deterministic_algorithms(deterministic)
        torch.set_num_threads(16 if device == "cpu" else 1)
        current = deepcopy(model).to(device=device, dtype=dtype).train()
        def move(value):
            return value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
        inputs = {key: move(value) for key, value in fixture["history"].items()}
        cal, truth = move(fixture["calendar"]), move(fixture["target"])
        eps = {key: move(value) for key, value in fixture["noise"].items()}
        graph = template.clone().apply(move)
        residuals, audit = [], {}
        handle = current.backbone.node_residuals_layer.register_forward_hook(lambda m, a, o: residuals.append(o))
        start = time.perf_counter()
        if device == "xpu":
            torch.xpu.synchronize()
            torch.xpu.reset_peak_memory_stats()
        from contextlib import nullcontext
        with double_heads(current, audit) if "heads" in policy else nullcontext():
            result = current(inputs, cal, graph, eps)
        handle.remove()
        scale = current.coordinate_scale.repeat(24)
        if dtype == torch.float64 or "loss" in policy:
            energy = energy64(result["joint"], truth, scale)
            balance = torch.stack([r.double() for r in residuals[-1]], dim=-1).square().mean()
        else:
            energy = fair_energy_score(result["joint"], truth, scale)
            balance = result["power_balance"]
        loss = energy + cfg["stage_b"]["loss"]["power_balance_weight"] * balance
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite diagnostic loss")
        loss.backward()
        gradients = {}
        for name, parameter in current.named_parameters():
            if parameter.requires_grad:
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise ValueError(f"Missing or nonfinite gradient: {name}")
                gradients[name] = parameter.grad.detach().cpu().clone()
            elif parameter.grad is not None:
                raise ValueError("Frozen backbone acquired gradients")
        for name, value in current.state_dict().items():
            if not torch.equal(value.detach().cpu().to(fixture["state"][name].dtype), fixture["state"][name]):
                raise ValueError(f"Parameter or buffer changed: {name}")
        if device == "xpu":
            torch.xpu.synchronize()
        row = {"label": label, "run_identity": f"{root.name}/{label}",
               "device": device, "precision_policy": policy,
               "deterministic_algorithms": deterministic, "seconds": time.perf_counter() - start,
               "loss": float(loss.detach()), "energy": float(energy.detach()), "balance": float(balance.detach()),
               "peak_allocated_bytes": torch.xpu.max_memory_allocated() if device == "xpu" else None,
               "head_dtype_boundary_calls": audit, "parameters_unchanged": True,
               "gradient_norm": float(torch.cat([g.double().flatten() for g in gradients.values()]).norm())}
        raw = {"gradients": gradients, "vm": result["values"]["vm"].detach().cpu(),
               "joint": result["joint"].detach().cpu(), "metrics": row}
        if not all(torch.isfinite(raw[key]).all() for key in ("vm", "joint")):
            raise ValueError("Nonfinite diagnostic output")
        atomic_torch_save(root / f"{label}.pt", raw)
        row["artifact_sha256"] = sha256(root / f"{label}.pt")
        print(label, row["loss"], row["gradient_norm"], row["seconds"], flush=True)
        del current, result, residuals, graph, inputs, cal, truth, eps, loss, energy, balance
        gc.collect()
        torch.xpu.empty_cache()
        return raw, row

    reference, reference_row = evaluate("cpu_fp64", "cpu", torch.float64, True, "all_fp64")
    names = sorted(reference["gradients"])
    group_names = {group: [name for name in names if name.startswith(prefixes)] for group, prefixes in GROUPS.items()}
    if sorted(name for group in group_names.values() for name in group) != names or not all(group_names.values()):
        raise ValueError("Parameter groups do not cover trainable parameters exactly once")
    write_json(root / "groups.json", group_names)

    def compare(raw, row):
        row["groups"] = {}
        for group, parameters in group_names.items():
            row["groups"][group] = gradient_metrics(
                torch.cat([raw["gradients"][name].flatten() for name in parameters]),
                torch.cat([reference["gradients"][name].flatten() for name in parameters]),
                protocol["metrics"]["gradient_norm_floor"])
        row["parameters"] = {name: gradient_metrics(raw["gradients"][name], reference["gradients"][name],
                                                   protocol["metrics"]["gradient_norm_floor"]) for name in names}
        row["forward_deltas"] = {}
        for key in ("vm", "joint"):
            delta = (raw[key].double() - reference[key]).abs().flatten()
            row["forward_deltas"][key] = {"mean_abs": float(delta.mean()), "max_abs": float(delta.max()),
                                           "p95_abs": float(torch.quantile(delta, .95))}
        write_json(root / f"{row['label']}.json", row)
        return row

    def run_policy(policy):
        rows = []
        for mode, count in ((True, 1), (False, 5)):
            for repeat in range(count):
                label = f"{policy}_{'on' if mode else 'off'}_{repeat + 1}"
                rows.append(compare(*evaluate(label, "xpu", torch.float32, mode, policy)))
        return rows

    baseline = run_policy("fp32")
    branch = decision(baseline, protocol)
    results = {"schema_version": 1, "reference": reference_row, "baseline": baseline,
               "decision": branch, "conditional": {}, "epoch_acceptance": "not_run"}
    write_json(root / "results.json", results)
    if branch == "unreliable_gradient_test_selective_fp64":
        for policy in protocol["conditional_policies"]:
            rows = run_policy(policy)
            results["conditional"][policy] = {"evaluations": rows, "decision": decision(rows, protocol)}
            write_json(root / "results.json", results)
    verify_frozen(root)
    results.update(diagnostic_complete=True, total_worker_seconds=time.perf_counter() - started,
                   stage2_authorized_by_diagnostic=False,
                   production_default="determinism_on_fp32_pending_acceptance")
    write_json(root / "results.json", results)


def job(root):
    verify_frozen(root)
    protocol = read_json(root / "protocol.json")
    start = float(os.environ["GENCO_V2_JOB_START"])
    deadline = start + protocol["job"]["worker_guard_seconds"]
    status = {"schema_version": 1, "job_id": os.environ["PBS_JOBID"], "start_epoch": start,
              "reserved_seconds": protocol["job"]["walltime_seconds"], "complete": False}
    try:
        invoke([sys.executable, "-m", "pytest", "-q", "tests/test_v2_diagnosis.py"],
               os.environ.copy(), root / "logs/tests.log", deadline)
        invoke([sys.executable, "-u", "-m", "scripts.v2.diagnose_stage1", "worker", "--root", str(root)],
               os.environ.copy(), root / "logs/worker.log", deadline)
        status["complete"] = True
    except BaseException as exc:
        status["error"] = repr(exc)
        raise
    finally:
        status.update(end_epoch=time.time(), actual_seconds=time.time() - start)
        write_json(root / "job_status.json", status)
        write_json(root / "artifact_sha256.json", {str(path.relative_to(root)): sha256(path)
                   for path in sorted(root.rglob("*")) if path.is_file()
                   and path.parts[len(root.parts)] not in ("shared", "tmp")
                   and path.name not in ("artifact_sha256.json", "pbs.log")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "submit", "job", "worker"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--protocol", default="reports/v2/stage1_diagnosis.protocol.json")
    args = parser.parse_args()
    root = checked_root(args.root)
    if args.action == "prepare":
        print(prepare(root, (ROOT / args.protocol).resolve()))
    elif args.action == "submit":
        print(submit(root))
    else:
        globals()[args.action](root)


if __name__ == "__main__":
    main()
