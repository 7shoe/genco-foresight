"""PowerModels/Ipopt in an isolated, persistent Julia process.

Subprocess/JSON protocol adapted from lumina-sdk distillation/RAMBO/backend.py,
revision e75ffcd74854c14e2960b9a9411ddd72377e5239, Apache-2.0.
Copyright 2026 UChicago Argonne, LLC. See THIRD_PARTY_NOTICES.md.
Changes: explicit config, OPF-only requests, physical units, strict dimensions.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time

import numpy as np

from common import ROOT


class IpoptSolver:
    def __init__(self, config, log_path, startup_seconds=180):
        startup_started = time.perf_counter()
        settings = config["solver"]
        self.settings = settings
        self.casefile = str(ROOT / "data/networks/pglib_opf_case30_ieee.m")
        executable = Path(settings["julia_executable"])
        project = ROOT / "solver/julia"
        if not executable.is_file():
            raise FileNotFoundError(f"Julia executable missing: {executable}; set solver.julia_executable")
        if not (project / "Manifest.toml").is_file():
            raise FileNotFoundError(f"Missing pinned Julia manifest: {project}")
        env = dict(os.environ, JULIA_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                   LD_LIBRARY_PATH="", JULIA_DEPOT_PATH=settings["julia_depot"], JULIA_PKG_OFFLINE="true")
        self.process = subprocess.Popen([str(executable), "--startup-file=no", f"--project={project}",
                                         str(project / "worker.jl")], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, bufsize=1, env=env, start_new_session=True)
        self.lines = queue.Queue()
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        try:
            self.versions = self.request({"stage": "versions"}, startup_seconds)
            if "julia" not in self.versions:
                raise RuntimeError(self.versions)
            self.startup_seconds = time.perf_counter()-startup_started
        except BaseException:
            self.close()
            raise

    def _reader(self):
        with self.log_path.open("a") as stream:
            for line in self.process.stdout:
                if line.startswith("GENCO_RESULT "):
                    self.lines.put(line[len("GENCO_RESULT "):])
                else:
                    stream.write(line)
                    stream.flush()
        self.lines.put(None)

    def request(self, request, timeout):
        self.process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
        self.process.stdin.flush()
        try:
            response = self.lines.get(timeout=timeout)
        except queue.Empty:
            self.close()
            raise TimeoutError(f"Julia wall deadline exceeded; see {self.log_path}")
        if response is None:
            raise RuntimeError(f"Julia exited; see {self.log_path}")
        return json.loads(response)

    def solve(self, pd_mw, qd_mvar=None, initial_state=None, budget=None):
        if qd_mvar is None:
            graph = pd_mw
            if hasattr(graph, "is_normalized") and bool(graph.is_normalized.any()):
                raise ValueError("Solver requires physical input units; inverse-normalize the graph first")
            values = graph["bus"].x.detach().cpu().numpy()
            pd_mw, qd_mvar = values[:, 0], values[:, 1]
        budget = budget or self.settings.get("budget", {})
        seconds = float(budget.get("seconds", 5.0))
        pd = np.asarray(pd_mw, dtype=float)
        qd = np.asarray(qd_mvar, dtype=float)
        if pd.shape != (30,) or qd.shape != pd.shape or not np.isfinite([pd, qd]).all():
            raise ValueError("case30 requires 30 finite bus Pd/Qd values")
        request = {"stage": "opf", "casefile": self.casefile, "pd_mw": pd.tolist(),
                   "qd_mvar": qd.tolist(), "seconds": seconds,
                   "max_iter": int(budget.get("max_iter", 2000)),
                   "tolerance": float(budget.get("tolerance", 1e-8)), "initial_state": initial_state}
        started = time.perf_counter()
        try:
            result = self.request(request, float(budget.get("wall_seconds", seconds + 20)))
        except (TimeoutError, RuntimeError, BrokenPipeError) as error:
            result = {"success": False, "status": "WORKER_UNRESOLVED", "error": str(error)}
        result["request_seconds"] = time.perf_counter() - started
        result["budget"] = {k: request[k] for k in ("seconds", "max_iter", "tolerance")}
        return result

    def close(self):
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    threading.Thread(target=self.process.wait, daemon=True).start()
        self.thread.join(timeout=2)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
