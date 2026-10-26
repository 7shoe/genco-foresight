"""Checksummed per-origin completion and numerical comparison."""
from pathlib import Path
import math
from ._config import read_json, sha256, write_json

def differences(a, b, path='', atol=1e-12, rtol=1e-12):
    """Compare all common scientific fields; retain worst values, not just a boolean."""
    result = dict(passed=True, max_absolute_difference=0., max_relative_difference=0.,
                  numeric_values=0, failures=[])
    def visit(x, y, name):
        if isinstance(y, dict):
            for k, value in y.items():
                if k not in x:
                    result['failures'].append(name + '/' + k + ': missing'); continue
                visit(x[k], value, name + '/' + k)
        elif isinstance(y, list):
            if len(x) != len(y):
                result['failures'].append(name + ': length'); return
            for i, (xx, yy) in enumerate(zip(x, y)): visit(xx, yy, f'{name}/{i}')
        elif isinstance(y, (int, float)) and not isinstance(y, bool):
            delta = abs(float(x) - float(y)); relative = delta / max(abs(float(y)), 1e-12)
            result['numeric_values'] += 1
            result['max_absolute_difference'] = max(result['max_absolute_difference'], delta)
            result['max_relative_difference'] = max(result['max_relative_difference'], relative)
            if not math.isfinite(delta) or delta > atol + rtol * abs(float(y)): result['failures'].append(name)
        elif x != y: result['failures'].append(name)
    visit(a, b, path)
    result['passed'] = not result['failures']
    result['failures'] = result['failures'][:20]
    return result



def completion_valid(folder, protocol_hash):
    done = Path(folder) / 'done.json'
    if not done.exists(): return False
    d = read_json(done)
    assert d['protocol_sha256'] == protocol_hash
    assert all(sha256(Path(folder) / n) == h for n, h in d['artifacts'].items())
    return True



def finish_origin(folder, protocol_hash, extra=None):
    write_json(folder / 'done.json', dict(protocol_sha256=protocol_hash,
        artifacts={p.name: sha256(p) for p in folder.iterdir() if p.is_file() and p.name != 'done.json'},
        **(extra or {})))
