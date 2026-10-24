import json

import pytest

from common import write_json


def test_json_validation_preserves_previous_artifact_on_malformed_serialization(tmp_path, monkeypatch):
    path = tmp_path/"report.json"
    path.write_text('{"prior": true}\n')
    monkeypatch.setattr(json, "dump", lambda value, stream, **kwargs: stream.write('{"bad": },'))
    with pytest.raises(json.JSONDecodeError):
        write_json(path, {"new": True})
    assert json.loads(path.read_text()) == {"prior": True}
    assert list(tmp_path.iterdir()) == [path]


def test_json_rejects_nonfinite_evidence_before_replacement(tmp_path):
    path = tmp_path/"report.json"
    write_json(path, {"value": 1.0})
    with pytest.raises(ValueError):
        write_json(path, {"value": float("nan")})
    assert json.loads(path.read_text()) == {"value": 1.0}
