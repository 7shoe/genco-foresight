import pytest
import yaml
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationKeyError, MissingMandatoryValue

from power_fm_foresight._config import ROOT, load_config


PROJECT_CONFIGS = [
    path for path in sorted((ROOT / "configs").rglob("*.yaml"))
    if "base" in yaml.safe_load(path.read_text())
] + [
    ROOT / "configs/replication/family.json"
]


def assert_plain(value):
    assert not OmegaConf.is_config(value)
    if isinstance(value, dict):
        for child in value.values():
            assert_plain(child)
    elif isinstance(value, list):
        for child in value:
            assert_plain(child)


@pytest.mark.parametrize("text", [
    "seed: 42\npaths: {cache: /data/cache}\nvalues: [1, 0.5, true, null]\n",
    "- name: first\n  values: [1, 2]\n- name: second\n",
])
def test_plain_legacy_configs_keep_their_values(tmp_path, text):
    path = tmp_path / "legacy.yaml"
    path.write_text(text)
    loaded = load_config(path)
    assert loaded == yaml.safe_load(text)
    assert_plain(loaded)


def test_references_without_base_resolve_in_nested_lists(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("paths:\n  root: /data\n  cache: ${paths.root}/cache\n"
                    "jobs:\n  - inputs: ['${paths.cache}']\n")
    loaded = load_config(path)
    assert loaded["jobs"] == [{"inputs": ["/data/cache"]}]
    assert_plain(loaded)


def test_overrides_are_merged_before_interpolation(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("base: /original\nseed: 42\npaths:\n"
                    "  cache: ${base}/cache\n  output: ${base}/output\n"
                    "  checkout: /home/example/project\n"
                    "jobs:\n  - input: ${paths.cache}\n    output: ${paths.output}\n")
    overrides = {"base": "/replacement", "seed": 7, "paths": {"cache": "/custom/cache"}}
    loaded = load_config(path, overrides=overrides)
    assert loaded == {
        "seed": 7,
        "paths": {"cache": "/custom/cache", "output": "/replacement/output",
                  "checkout": "/home/example/project"},
        "jobs": [{"input": "/custom/cache", "output": "/replacement/output"}],
    }
    assert overrides["base"] == "/replacement"
    assert_plain(loaded)


@pytest.mark.parametrize("text,error,match", [
    ("paths:\n  cache: ${base}/cache\n", InterpolationKeyError, "base"),
    ("base: /data\npaths:\n  cache: ${missing.root}/cache\n", InterpolationKeyError, "missing.root"),
    ("base: /data\nrequired: ???\n", MissingMandatoryValue, "required"),
])
def test_missing_configuration_fails_at_load(tmp_path, text, error, match):
    path = tmp_path / "invalid.yaml"
    path.write_text(text)
    with pytest.raises(error, match=match):
        load_config(path)


@pytest.mark.parametrize("path", PROJECT_CONFIGS, ids=lambda p: str(p.relative_to(ROOT)))
def test_project_base_override_changes_only_dependent_paths(path):
    original = load_config(path)
    prefix = yaml.safe_load(path.read_text())["base"]
    replacement = "/lus/flare/projects/example/user"

    def expected(value):
        if isinstance(value, dict):
            return {key: expected(child) for key, child in value.items()}
        if isinstance(value, list):
            return [expected(child) for child in value]
        if isinstance(value, str) and value.startswith(prefix + "/"):
            return replacement + value[len(prefix):]
        return value

    relocated = load_config(path, overrides={"base": replacement})
    assert relocated == expected(original)
    assert "base" not in original and "base" not in relocated
    assert_plain(original)
    assert_plain(relocated)
