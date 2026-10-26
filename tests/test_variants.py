import os

from common import ROOT, config_path, load_config
from scripts.data_v4 import verify_variant


def test_variant_config_is_explicit_and_preserves_v3(monkeypatch):
    monkeypatch.delenv("GENCO_CONFIG", raising=False)
    assert config_path() == ROOT / "configs/v3.yaml"
    v3 = load_config()
    monkeypatch.setenv("GENCO_CONFIG", str(ROOT / "configs/v4.yaml"))
    v4 = load_config()
    verify_variant(v4)
    assert load_config(ROOT / "configs/v3.yaml") == v3
    assert v4["dataset"]["id"] != v3["dataset"]["id"]
