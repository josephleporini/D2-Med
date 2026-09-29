"""Generator v3 sampler (BT-2b): older mixes reproduce unchanged; the bt3 mix has the intended shape."""
import glob, hashlib, json, os, subprocess, sys, tempfile

GEN = os.path.join(os.path.dirname(__file__), '..', 'gen', 'sample_batch4.py')


def _split(name, n, seed, mode):
    d = tempfile.mkdtemp()
    subprocess.run([sys.executable, GEN, 'split', name, str(n), str(seed), d, mode], check=True, capture_output=True)
    return d, sorted(glob.glob(os.path.join(d, '*_params.json')))


def test_bt2_params_unchanged():
    # hash of the first 40 dev6 params as rendered for BT-2 (checked against commit 764bd53, 29 Sep)
    d, fs = _split('dev6', 40, 91000, 'bt2')
    h = hashlib.sha256(b''.join(open(f, 'rb').read() for f in fs)).hexdigest()
    assert h == 'e86415711c92a4b362eac4affd192d2350667be14a5270b3d0037a6579de64de'


def test_bt3_mix_shape():
    d, fs = _split('dev7', 200, 93000, 'bt3')
    ps = [json.load(open(f)) for f in fs]
    assert all(p['gen'] == 'v3' and p['wound_style'] == 'v3' and p['scene_id'].startswith('Y') for p in ps)
    sup = sum(p['body_position'] == 'supine' for p in ps) / len(ps)
    assert sup > 0.65                                   # 70% forced plus the base share
    nb = [len(p['bystanders']) for p in ps]
    assert 0.1 < nb.count(0) / len(ps) < 0.3 and max(nb) == 2
    assert all(p['view'] in ('AgX', 'Standard') for p in ps)
    close = sum(p['framing'] == 'limb_closeup' for p in ps) / len(ps)
    assert 0.1 < close < 0.3
    inj = lambda p: set(p['wounds']) | set(p['amputations'])
    assert sum(bool(set(p['blood_pool']) - inj(p)) for p in ps) < 0.2 * len(ps)   # intact-limb pools stay rare
