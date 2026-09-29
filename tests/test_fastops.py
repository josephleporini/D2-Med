"""gen/fastops.py must be bit-identical to the reference path. Random masks always; dev5 part maps when DEV5 is present
(env DEV5, default the local DDData clone). Run: python -m pytest tests/test_fastops.py"""
import os, sys, glob, importlib
import numpy as np, pytest
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'gen'), os.path.join(HERE, '..', 'jobs')]
from scipy import ndimage
import fastops as FO

DEV5 = os.environ.get('DEV5', '/home/claude/dddata/dev5')


def test_dilation_random():
    rng = np.random.default_rng(0)
    for t in range(200):
        m = rng.random((48, 64)) < rng.choice([0.002, 0.02, 0.2])
        if t % 7 == 0:
            m[:, :3] = True                                   # touches the border
        for k in (1, 2, 3, 5):
            assert np.array_equal(FO.binary_dilation(m, iterations=k), ndimage.binary_dilation(m, iterations=k))
    z = np.zeros((10, 10), bool)
    assert not FO.binary_dilation(z, iterations=3).any()


def _pipeline(files):
    """CPU part of BT1Engine.rows on ground-truth part maps: fold, frame, class map, side assignment, analyse,
    limb ends and windows, wound and TQ growth. Returns everything the decision layer can see."""
    import cv2
    import eval_v3 as EV, test_a_masks as TA, test_e as TE, test_e2 as T2, lend as LE, side_kp as SK, checks
    from d2pipe import frame_facing
    out = []
    for f in files:
        seg3 = EV.gt_part_map(f)
        seg, wm, tq = EV.fold_extras(seg3)
        sfr = T2.seg_frame(seg); fac = frame_facing(seg, sfr)
        axis = np.array([0.0, -1.0]); left = (sfr[2] if sfr else 1) * TE.perp(axis)
        cm1, names, e1, s1, _ = SK.to_classmap_kp(seg, left, axis, None)
        pleft = np.full(seg.shape, 0.9, np.float32); pside = np.ones(seg.shape, np.uint8); conf = np.zeros(seg.shape, np.float32)
        cm, eC, sC = checks.assign_gt_side(cm1, names, seg, e1, s1, pside, conf)
        ix = {n: i for i, n in enumerate(names)}
        an = TA.analyse(cm, names); torso = cm == ix['TORSO']
        ends = {}
        for s in TE.SITES:
            pm = cm == ix[s]
            if an[s]['vis_px'] >= 30:
                ends[s] = (LE.limb_end(pm, torso), SK.limb_window(pm))
            grown = SK.ndimage.binary_dilation(pm, iterations=2)
            ends[s + '_g'] = (int((wm & grown).sum()), int((tq & grown).sum()))
        out.append((fac, cm, dict(eC), dict(sC), an, ends))
    return out


@pytest.mark.skipif(not os.path.isdir(DEV5), reason='dev5 not present')
def test_pipeline_exact_on_dev5():
    """Run in a fresh interpreter: other tests put stub modules (tests/stub) on the path."""
    import subprocess
    r = subprocess.run([sys.executable, __file__], capture_output=True, text=True, cwd=HERE)
    assert r.returncode == 0 and 'EXACT_OK' in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]


def _exact_on_dev5():
    files = sorted(glob.glob(os.path.join(DEV5, 'batch_00*/D*_part3.png')))[:40]
    import side_kp as SK, test_a_masks as TA
    TA.geodesic = SK.geodesic_fast
    ref = _pipeline(files)
    FO.apply()
    try:
        new = _pipeline(files)
    finally:
        for name in ('test_a_masks', 'test_e2', 'side_kp', 'lend', 'eval_v3'):
            importlib.reload(sys.modules[name])
    assert len(ref) == len(new) == len(files)
    for a, b in zip(ref, new):
        assert a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2] == b[2] and a[3] == b[3]
        assert repr(a[4]) == repr(b[4]), 'analyse differs'
        assert repr(a[5]) == repr(b[5]), 'limb ends / growth differ'


if __name__ == '__main__':
    _exact_on_dev5(); print('EXACT_OK')
