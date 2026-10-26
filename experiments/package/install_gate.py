"""Install into fresh environments without changing the vendor torch environment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tomllib
import zipfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    a = p.parse_args()
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env['PIP_NO_INDEX'] = '1'
    env['PIP_FIND_LINKS'] = str(a.root / 'wheelhouse')
    env['PIP_CACHE_DIR'] = str(a.root / 'pip-cache')
    def run(cmd, **kw):
        return subprocess.run([str(x) for x in cmd], check=True, env=env, cwd=a.root, **kw)
    checks = {}
    for name, system in [('with_stack', True), ('without_stack', False)]:
        venv = a.root / 'venvs' / name
        run([sys.executable, '-m', 'venv', *(['--system-site-packages'] if system else []), venv])
        py = venv / 'bin/python'
        if system:
            site = venv / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'
            # The fresh venv reuses the existing torch stack read-only.
            (site / 'vendor_stack.pth').write_text(sysconfig.get_path('purelib') + '\n')
        else:
            run([py, '-m', 'pip', 'install', '--no-deps', 'setuptools==79.0.1', 'wheel==0.48.0', 'packaging'])
        run([py, '-m', 'pip', 'install', '--no-deps', '--no-build-isolation', '-e', a.source])
        result = subprocess.run([str(py), '-c', 'import power_fm_foresight; print(power_fm_foresight.__version__)'],
                                cwd=a.root, env=env, capture_output=True, text=True)
        print(name, result.returncode, result.stdout, result.stderr, flush=True)
        if system:
            assert result.returncode == 0 and result.stdout.strip() == tomllib.loads((a.source/'pyproject.toml').read_text())['project']['version']
            code = '''from power_fm_foresight import grids, genco, temporal, data, solver, scoring, training
import importlib, json, sys, warnings
from pathlib import Path
source = Path(sys.argv[1]); sys.path.insert(0, str(source))
move = json.loads((source/'reports/package/moves.json').read_text())
for old, new in move['shims'].items():
    if old.startswith('scripts.') or old == 'data.synthetic.generate': continue
    if old == 'data.download.prepare': new = 'experiments.assets'
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always', DeprecationWarning)
        actual = importlib.import_module(old)
    assert actual is importlib.import_module(new), old
    assert any(issubclass(w.category, DeprecationWarning) for w in seen), old
print('Public modules and historical module identities passed')
'''
            run([py, '-c', code, a.source])
        else:
            assert result.returncode != 0 and 'README.md#install-matrix' in result.stderr
            assert "No module named 'torch'" in result.stderr
        checks[name] = dict(passed=True, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    run([a.root/'venvs/with_stack/bin/python', '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation', a.source, '-w', a.root/'wheels'])
    wheel = next((a.root/'wheels').glob('power_fm_foresight-*.whl'))
    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
        assert not any(n.startswith(('experiments/', 'tests/', 'scripts/')) for n in names)
        for suffix in ('resources/grids.yaml', 'resources/v4.yaml', 'resources/julia/worker.jl', 'resources/networks/pglib_opf_case30_ieee.m'):
            assert 'power_fm_foresight/' + suffix in names
        metadata = z.read(next(n for n in names if n.endswith('.dist-info/METADATA'))).decode()
        assert not any(line.startswith(('Requires-Dist: torch', 'Requires-Dist: gridfm-graphkit')) for line in metadata.splitlines())
    checks['wheel'] = dict(passed=True, path=str(wheel), members=names)
    (a.root/'installation.json').write_text(json.dumps(dict(passed=True, checks=checks), indent=2)+'\n')


if __name__ == '__main__':
    main()
