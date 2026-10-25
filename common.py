"""Small configuration and artifact helpers shared by the experiment scripts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parent


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


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    """Atomically replace a JSON artifact; reject nonfinite scientific values."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = stream.name
        try:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            # Validate the exact on-disk bytes before replacing the prior artifact.
            with open(temporary) as check:
                json.load(check)
        except BaseException:
            os.unlink(temporary)
            raise
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text())
