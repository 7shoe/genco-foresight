"""Persist sampled training curves, with checkpoint-authoritative recovery."""
import json
import os
from pathlib import Path
import time


SCHEMA_VERSION = 1
METRICS = ("loss", "energy", "balance", "gradient_norm")


class TrainingMetrics:
    def __init__(self, root, progress, interval):
        if not isinstance(interval, int) or isinstance(interval, bool) or interval < 1:
            raise ValueError("metrics_interval_steps must be a positive integer")
        self.interval = interval
        self.path = Path(root) / "metrics.jsonl"
        expected = list(range(interval, progress["steps"] + 1, interval))
        payload = self.path.read_bytes() if self.path.exists() else b""
        lines = iter(payload.splitlines(keepends=True))
        offset = 0
        for step in expected:
            line = next(lines, None)
            if line is None:
                raise ValueError("Missing committed training metrics")
            row = json.loads(line)
            if row["schema_version"] != SCHEMA_VERSION or row["steps"] != step:
                raise ValueError("Training metrics order/schema changed")
            offset += len(line)
        if offset != len(payload):
            # Keep abandoned work for diagnosis before replaying from the checkpoint.
            archive = self.path.with_name(f"metrics.abandoned.{time.time_ns()}.jsonl")
            archive.write_bytes(payload[offset:])
            with self.path.open("r+b") as stream:
                stream.truncate(offset)
                stream.flush()
                os.fsync(stream.fileno())
        self.stream = self.path.open("a")

    def record(self, progress, identity, metrics, step_seconds, worker_seconds):
        if progress["steps"] % self.interval:
            return
        row = dict(schema_version=SCHEMA_VERSION, steps=progress["steps"],
                   origins_consumed=progress["origins_consumed"], epoch=progress["epoch"],
                   step_in_epoch=progress["step_in_epoch"], origin=list(identity),
                   step_sequence_sha256=progress["step_sequence_sha256"], **metrics,
                   step_seconds=step_seconds, worker_seconds=worker_seconds)
        self.stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
        self.stream.flush()

    def sync(self):
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self):
        self.stream.close()
