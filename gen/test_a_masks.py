"""Test A: perfect-mask experiment, image-derivable evidence only (pose env).
Usage: python test_a_masks.py <scene_dir> [...]

Input per scene: the ID-pass class map only (what an ideal part segmenter could output):
  LUE, RUE, LLE, RLE, TORSO (incl. head), OCC (any occluding object), BG (floor), plus the image edge.
NOT used: amputation parameters, hidden-limb pixel counts (alone renders), 3D geometry, keypoint existence.

Evidence per site:
  vis_px      visible limb pixels
  Lfrac       geodesic length of the visible limb from its torso attachment, divided by the expected limb length
              (fixed body proportions of the base mesh x torso-plus-head length measured on the torso mask)
  end ring    classes bordering the limb's distal end (farthest 10% of geodesic length): BG, OCC, TORSO,
              OTHER_LIMB, EDGE
Fixed rules (set before scoring, not tuned):
  vis_px < 30                              -> not_testable
  bg_frac >= 0.60 and Lfrac < 0.75         -> amputation   (limb ends in open background and is short)
  Lfrac >= 0.10                            -> no_injury
  otherwise                                -> not_testable
"""
import sys, os, json, glob, collections
import numpy as np
from PIL import Image
from scipy import ndimage

COL = {'LUE': (255, 0, 0), 'RUE': (0, 255, 0), 'LLE': (0, 0, 255), 'RLE': (255, 255, 0),
       'TORSO': (255, 0, 255), 'OCC': (0, 255, 255), 'BG': (128, 128, 128)}
SITES = ['LUE', 'RUE', 'LLE', 'RLE']
RATIO = {'UE': 0.690, 'LE': 0.977}                 # limb length / torso+head length, base mesh
MIN_PX, BG_AMP, L_AMP, L_VIS = 30, 0.60, 0.75, 0.10


def classmap(path):
    a = np.array(Image.open(path).convert('RGB')).astype(int)
    cm = np.full(a.shape[:2], -1)
    names = list(COL)
    for i, n in enumerate(names):
        if n == 'BG':
            continue
        cm[np.all(np.abs(a - np.array(COL[n])) <= 2, axis=2)] = i
    cm[cm == -1] = names.index('BG')        # floor (sRGB 188 grey) and world background
    return cm, names


def extent(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return 0.0
    X = np.stack([xs, ys], 1).astype(float); X -= X.mean(0)
    w, v = np.linalg.eigh(np.cov(X.T)); pr = X @ v[:, -1]
    return float(np.percentile(pr, 99.5) - np.percentile(pr, 0.5))


def geodesic(limb, passable, seeds):
    """BFS distance (8-connected, diagonal = sqrt2) over limb pixels, allowed to cross occluder pixels."""
    H, W = limb.shape
    dist = np.full((H, W), np.inf)
    from collections import deque
    q = deque()
    for y, x in zip(*np.nonzero(seeds)):
        dist[y, x] = 0; q.append((y, x))
    steps = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1), (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
    ok = limb | passable
    while q:                                            # Dijkstra-lite (two step costs): good enough at this scale
        y, x = q.popleft(); d = dist[y, x]
        for dy, dx, c in steps:
            yy, xx = y + dy, x + dx
            if 0 <= yy < H and 0 <= xx < W and ok[yy, xx] and d + c < dist[yy, xx] - 1e-6:
                dist[yy, xx] = d + c; q.append((yy, xx))
    dist[~limb] = np.inf
    return dist


def analyse(cm, names):
    H, W = cm.shape
    idx = {n: i for i, n in enumerate(names)}
    torso = cm == idx['TORSO']; occ = cm == idx['OCC']
    t_len = extent(torso)
    out = {}
    for s in SITES:
        limb = cm == idx[s]; px = int(limb.sum())
        r = {'vis_px': px}
        if px < MIN_PX:
            r.update(Lfrac=0.0, end={}, bg_frac=None, scale='n/a'); out[s] = r; continue
        # limb length = geodesic diameter of the limb mask (two-pass BFS), crossing occluders allowed;
        # this avoids needing the attachment point (an arm lying along the torso touches it everywhere)
        ys, xs = np.nonzero(limb)
        s0 = np.zeros_like(limb); s0[ys[0], xs[0]] = True
        d0 = geodesic(limb, occ, s0); f0 = np.isfinite(d0)
        e1 = np.unravel_index(np.argmax(np.where(f0, d0, -1)), limb.shape)
        s1 = np.zeros_like(limb); s1[e1] = True
        d1 = geodesic(limb, occ, s1); f1 = np.isfinite(d1)
        e2 = np.unravel_index(np.argmax(np.where(f1, d1, -1)), limb.shape)
        L = float(d1[e2]) if f1.any() else 0.0
        tor_ring = ndimage.binary_dilation(torso, iterations=3)
        def end_region(dist_from_other_end):
            return f1 & (dist_from_other_end >= 0.9 * L) if L > 0 else limb
        s2 = np.zeros_like(limb); s2[e2] = True
        d2 = geodesic(limb, occ, s2)
        reg_a = end_region(d1)          # region around e2 (far from e1)
        reg_b = np.isfinite(d2) & (d2 >= 0.9 * L) if L > 0 else limb   # region around e1
        # distal end = the end touching the torso least
        ta, tb = int((reg_a & tor_ring).sum()), int((reg_b & tor_ring).sum())
        endreg = reg_a if ta <= tb else reg_b
        fin = f1
        scale = 'torso' if t_len > 40 else 'fallback'
        exp = RATIO[s[1:]] * (t_len if t_len > 40 else max(L, 1) / 0.8)
        ring = ndimage.binary_dilation(endreg, iterations=3) & ~limb
        comp = collections.Counter()
        for n in names:
            if n == s:
                continue
            c = int((ring & (cm == idx[n])).sum())
            key = 'OTHER_LIMB' if n in SITES else n
            comp[key] += c
        ys, xs = np.nonzero(endreg)
        edge = int(((ys <= 2) | (ys >= H - 3) | (xs <= 2) | (xs >= W - 3)).sum())
        comp['EDGE'] += edge * 3
        tot = sum(comp.values()) or 1
        r.update(Lfrac=round(L / exp, 3), end={k: v for k, v in comp.items() if v}, bg_frac=round(comp['BG'] / tot, 3),
                 scale=scale)
        out[s] = r
    return out


def decide(r):
    if r['vis_px'] < MIN_PX:
        return 'not_testable'
    if r['bg_frac'] is not None and r['bg_frac'] >= BG_AMP and r['Lfrac'] < L_AMP:
        return 'amputation'
    if r['Lfrac'] >= L_VIS:
        return 'no_injury'
    return 'not_testable'


if __name__ == '__main__':
    dirs = sys.argv[1:]
    C = ['no_injury', 'amputation', 'not_testable']
    cmx = np.zeros((3, 3), int); rows = []
    for D in dirs:
        for f in sorted(glob.glob(os.path.join(D, 'C*_sidecar.json'))):
            s = json.load(open(f)); sid = s['scene_id']
            cm, names = classmap(os.path.join(D, sid + '_id.png'))
            an = analyse(cm, names)
            for site in SITES:
                g = s['labels_by_threshold']['0.10'][site]; p = decide(an[site])
                cmx[C.index(g), C.index(p)] += 1
                rows.append({'scene': sid, 'site': site, 'gt': g, 'pred': p, 'amp_level': s['params']['amputations'].get(site),
                             'position': s['params']['body_position'], 'limb_pose': s['params']['limb_pose'],
                             'occluder': s['params']['occluder'], 'framing': s['params']['framing'],
                             'elev': s['params']['elevation'], 'vis_frac_true': s['visible_fraction'][site], **an[site]})
    n = cmx.sum(); acc = np.trace(cmx) / n
    print(f'TEST A accuracy {acc:.3f} (n={n})')
    for i, c in enumerate(C):
        print(f'  recall {c:12s} {cmx[i, i] / cmx[i].sum():.3f}  (n={cmx[i].sum()})')
    print('  confusion rows GT [no_inj, amp, not_test] x cols pred:', cmx.tolist())
    json.dump({'accuracy': round(float(acc), 4), 'confusion': cmx.tolist(), 'classes': C,
               'rules': {'MIN_PX': MIN_PX, 'BG_AMP': BG_AMP, 'L_AMP': L_AMP, 'L_VIS': L_VIS, 'RATIO': RATIO}},
              open(os.path.join(dirs[0], 'testA_results.json'), 'w'), indent=1)
    json.dump(rows, open(os.path.join(dirs[0], 'testA_sites.json'), 'w'), indent=1, default=str)
