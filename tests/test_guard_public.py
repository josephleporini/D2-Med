"""Publish guard G3: refuses anything it cannot show is synthetic."""
import os, sys, json, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
G = os.path.join(HERE, '..', 'infra', 'guard_public.py')


def run(args, env=None):
    e = dict(os.environ); e.pop('D2_REAL_DATA', None); e.update(env or {})
    return subprocess.run([sys.executable, G] + args, capture_output=True, text=True, env=e)


def mk(tmp, stem, gen='scene4/1.0', sidecar=True):
    open(os.path.join(tmp, stem + '.jpg'), 'wb').write(b'\xff\xd8\xff')
    if sidecar:
        json.dump({'provenance': {'generator': gen}}, open(os.path.join(tmp, stem + '_sidecar.json'), 'w'))


def test_synthetic_passes(tmp_path):
    mk(tmp_path, 'D0001'); mk(tmp_path, 'DK0002')
    r = run([str(tmp_path)]); assert r.returncode == 0 and 'PUBLISH_GUARD_OK 2' in r.stdout


def test_sidecar_in_other_dir(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'; a.mkdir(); b.mkdir()
    mk(b, 'D0003'); os.rename(b / 'D0003.jpg', a / 'D0003.jpg')
    assert run([str(a)]).returncode == 3
    assert run([str(a), str(b)]).returncode == 0


def test_refusals(tmp_path):
    mk(tmp_path, 'D0001'); mk(tmp_path, 'IMG_0420', sidecar=False)
    r = run([str(tmp_path)]); assert r.returncode == 3 and 'IMG_0420.jpg: no generator sidecar' in r.stdout
    os.remove(tmp_path / 'IMG_0420.jpg'); mk(tmp_path, 'X1', gen='camera')
    assert run([str(tmp_path)]).returncode == 3
    os.remove(tmp_path / 'X1.jpg'); open(tmp_path / 'D0009_sidecar.json', 'w').write('{'); open(tmp_path / 'D0009.jpg', 'wb').write(b'x')
    assert 'unreadable' in run([str(tmp_path)]).stdout


def test_empty_and_real_flag(tmp_path):
    assert run([str(tmp_path)]).returncode == 3
    mk(tmp_path, 'D0001')
    assert run([str(tmp_path)], {'D2_REAL_DATA': '1'}).returncode == 3
