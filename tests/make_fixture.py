"""Synthetic scenes with known injected failures, to test score.py code paths (not realism)."""
import os, json, sys, numpy as np
rng = np.random.default_rng(0); D = sys.argv[1]; os.makedirs(D + '/scenes', exist_ok=True)
S = ['LUE', 'RUE', 'LLE', 'RLE']; C4 = ['no_injury', 'wound', 'amputation', 'not_testable']
def raw(cls, noise):
    vis = 0 if cls == 3 else int(rng.integers(300, 3000))
    stump = int(rng.integers(40, 600)) if cls == 2 else 0; wound = int(rng.integers(25, 500)) if cls == 1 else 0
    pe = None if cls == 3 else ([0.1, 0.8, 0.1] if cls == 2 else [0.85, 0.05, 0.1])
    pw = None if cls == 3 else ([0.2, 0.8] if cls == 1 else [0.9, 0.1])
    r = dict(vis_px=vis, Lfrac=0.5, bg_frac=0.3, end={'BG': 3, 'OCC': 1}, ext_px=vis // 3, stump_px=stump, torso_ext=100.0,
             p_end=pe, p_wound=pw, wound_px=wound, tq_px=0)
    return r
def corrupt(r, kind):
    r = json.loads(json.dumps(r))
    if kind == 'miss':  r['stump_px'] = 0; r['wound_px'] = 0; r['p_end'] = [0.85, 0.05, 0.1] if r['p_end'] else None; r['p_wound'] = [0.9, 0.1] if r['p_wound'] else None
    if kind == 'hall':  r['wound_px'] = 300; r['p_wound'] = [0.1, 0.9] if r['p_wound'] else None
    return r
files = {k: open(f'{D}/{k}.jsonl', 'w') for k in ('pred', 'side', 'truth')}
for n in range(120):
    sid = f'C{n:05d}'; split = 'dev3'; lab = {}; P, Sd, T = {}, {}, {}
    for s in S:
        c = int(rng.choice(4, p=[0.55, 0.12, 0.15, 0.18])); lab[s] = C4[c]
        t = raw(c, 0); T[s] = t; u = rng.random()
        P[s] = corrupt(t, 'miss') if (c in (1, 2) and u < 0.2) else corrupt(t, 'hall') if (c == 0 and u < 0.05) else t
        Sd[s] = t if (u < 0.08) else P[s]
    labels = {'0.10': lab, '0.00': dict(lab), '0.25': dict(lab), '0.50': dict(lab)}
    for s in S:
        if lab[s] in ('wound', 'amputation') and P[s]['stump_px'] + P[s]['wound_px'] == 0 and rng.random() < 0.3: labels['0.00'][s] = 'no_injury'
    files['pred'].write(json.dumps(dict(scene=sid, split=split, labels=labels, sites=P, kp=None)) + '\n')
    files['side'].write(json.dumps(dict(scene=sid, split=split, labels=labels, sites=Sd)) + '\n')
    files['truth'].write(json.dumps(dict(scene=sid, split=split, gt_sites_ceil=T, facing_true=1)) + '\n')
    json.dump(dict(scene_id=sid, params=dict(seed=n, body_position='supine', occluder='none', wounds={}, amputations={}, tourniquets=[]),
                   visible_fraction={s: 0.8 for s in S}, wound_visible_px={s: T[s]['wound_px'] for s in S}), open(f'{D}/scenes/{sid}_sidecar.json', 'w'))
