"""Compare persisted curves against a predeclared, versioned protocol."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


METRICS = ("loss", "energy", "balance", "gradient_norm")
TIMING = {"step_seconds", "worker_seconds"}
ROW_KEYS = {"schema_version", "steps", "origins_consumed", "epoch", "step_in_epoch",
            "origin", "step_sequence_sha256", "noise_sha256", *METRICS, *TIMING}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def comparable_manifest(value):
    value = deepcopy(value)
    value.pop("run_identity", None)
    if "configuration" in value:
        value["configuration"]["v2"].pop("run_identity", None)
    return value


def close(a, b, threshold):
    if not all(isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x) for x in (a, b)):
        return False
    return abs(a - b) <= threshold["atol"] + threshold["rtol"] * max(abs(a), abs(b))


def compare_runs(left, right, protocol):
    left, right = Path(left), Path(right)
    failures = []
    evidence = {}
    maximum = {key: 0. for key in METRICS}
    def require(condition, message):
        if not condition:
            failures.append(message)
    def artifact(root, name):
        path = root / name
        evidence[str(path)] = digest(path)
        return read(path)
    try:
        if protocol["schema_version"] != 1:
            raise ValueError("Unsupported protocol version")
        thresholds = protocol["comparison"]["thresholds"]
        for key in (*METRICS, "selection_energy_score"):
            for part in ("atol", "rtol"):
                value = thresholds[key][part]
                if not math.isfinite(value) or value < 0:
                    raise ValueError("Invalid numerical threshold")
        manifests = [artifact(root, "manifest.json") for root in (left, right)]
        require(manifests[0]["run_identity"] != manifests[1]["run_identity"], "independent_run_identities")
        require(comparable_manifest(manifests[0]) == comparable_manifest(manifests[1]), "configuration/provenance")
        for m in manifests:
            for key, value in protocol["comparison"]["expected_manifest"].items():
                require(m.get(key) == value, f"manifest.{key}")
        curves = []
        for root in (left, right):
            path = root / "metrics.jsonl"
            evidence[str(path)] = digest(path)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            steps = protocol["comparison"]["steps"]
            cadence = protocol["comparison"]["cadence_steps"]
            require([r["steps"] for r in rows] == list(range(cadence, steps + 1, cadence)), f"{root.name}: cadence/coverage")
            for row in rows:
                require(set(row) == ROW_KEYS and row["schema_version"] == 1, f"{root.name}: metric schema")
                for key in (*METRICS, *TIMING):
                    require(isinstance(row[key], (float, int)) and math.isfinite(row[key]), f"{root.name}: nonfinite {key}")
                require(row["origins_consumed"] == row["steps"], f"{root.name}: B1 origin counter")
                require(row["step_seconds"] >= 0 and row["worker_seconds"] >= 0, f"{root.name}: negative timing")
            curves.append(rows)
        require(len(curves[0]) == len(curves[1]), "curve length")
        for a, b in zip(*curves):
            for key in ROW_KEYS - TIMING - set(METRICS):
                require(a[key] == b[key], f"step {a['steps']}: {key}")
            for key in METRICS:
                require(close(a[key], b[key], thresholds[key]), f"step {a['steps']}: {key}")
                if math.isfinite(a[key]) and math.isfinite(b[key]):
                    maximum[key] = max(maximum[key], abs(a[key] - b[key]))
        epochs = [artifact(root, "epochs.json") for root in (left, right)]
        require(all(len(x) == protocol["comparison"]["epochs"] for x in epochs), "epoch coverage")
        for a, b in zip(*epochs):
            require(a.keys() == b.keys(), "epoch schema")
            for key in a:
                require(close(a[key], b[key], thresholds[key]) if key == "selection_energy_score" else a[key] == b[key], f"epoch {a['epoch']}: {key}")
        statuses = [artifact(root, "status.json") for root in (left, right)]
        for root, status in zip((left, right), statuses):
            require(not (root / "failure.json").exists(), f"{root.name}: worker failure")
            require(status["base_unchanged"] is True, f"{root.name}: frozen backbone")
            require(status["steps_after"] == protocol["comparison"]["steps"], f"{root.name}: final steps")
            require(status["epochs_reached"] == protocol["comparison"]["epochs"], f"{root.name}: final epochs")
            require(status["origins_consumed"] == protocol["comparison"]["origins"], f"{root.name}: final origins")
            require(status["checkpoint_sha256"] == digest(root / "latest.pt"), f"{root.name}: checkpoint hash")
        for key in ("base_sha256", "step_sequence_sha256", "step_in_epoch", "origins_consumed"):
            require(statuses[0][key] == statuses[1][key], f"final {key}")
        if protocol["comparison"].get("exact_adapter", False):
            require(statuses[0]["adapter_sha256"] == statuses[1]["adapter_sha256"], "final adapter")
    except (OSError, ValueError, KeyError, TypeError) as error:
        failures.append(f"invalid or missing evidence: {error}")
    return {"schema_version": 1, "stage": protocol["stage"], "passed": not failures,
            "failures": failures, "maximum_absolute_differences": maximum,
            "evidence_sha256": evidence, "excluded_fields": sorted(TIMING)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = compare_runs(args.left, args.right, read(args.protocol))
    result["protocol_sha256"] = digest(args.protocol)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
