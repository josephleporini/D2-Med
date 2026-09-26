"""Add 3-state targets to cached limb-end windows (pose env). Replays lend_cache.py's window placement exactly
(same seed and call order), then labels each window from the ground-truth v2 part map, restricted to that limb:
  0 extremity visible  : >= 15 hand/foot pixels of this limb inside the window
  1 stump visible      : >= 15 stump pixels of this limb inside the window (amputated limbs only)
  2 end hidden         : neither (end occluded, out of frame, or foreshortened)
Usage: python lend_relabel.py <train_dir>"""
import sys, os, json, glob, zlib, collections
import numpy as np
from scipy import ndimage
sys.path.insert(0, os.path.dirname(__file__))
import parts as PT, lend as LE
from lend_cache import K, TORSO_C, GROUP, load_parts, SITE_IX

EXTC = {'UE': K['hand'], 'LE': K['foot']}


def win_count(mask, cx, cy, s):
    H, W = mask.shape
    x0, y0 = max(0, cx - s // 2), max(0, cy - s // 2)
    return int(mask[y0:min(H, cx - s // 2 + s if False else cy - s // 2 + s), x0:min(W, cx - s // 2 + s)].sum())


if __name__ == '__main__':
    D = sys.argv[1]; tally = collections.Counter()
    for f in sorted(glob.glob(os.path.join(D, 'C*_lend.npz'))):
        sid = os.path.basename(f)[:-len('_lend.npz')]
        prm = json.load(open(os.path.join(D, sid + '_sidecar.json')))['params']
        seg, side, wsite = load_parts(D, sid)
        torso = np.isin(seg, TORSO_C); s0 = LE.window_size(torso)
        rng = np.random.default_rng(zlib.crc32(sid.encode()))
        stump_of = {}
        hc = np.argwhere(np.isin(seg, [K['HEAD_F'], K['HEAD_B']]))
        for sd, L in ((1, 'L'), (2, 'R')):
            st = (seg == K['stump']) & (side == sd)
            amps = [x for x in (L + 'UE', L + 'LE') if x in prm['amputations']]
            if not st.any() or not amps:
                continue
            if len(amps) == 1:
                stump_of[amps[0]] = st; continue
            lab, nc = ndimage.label(st, structure=np.ones((3, 3)))
            ue = np.zeros_like(st); le = np.zeros_like(st)
            for c in range(1, nc + 1):
                comp = lab == c; rim = ndimage.binary_dilation(comp, iterations=2) & (side == sd)
                tu = (rim & np.isin(seg, GROUP['UE'][:3])).sum(); tl = (rim & np.isin(seg, GROUP['LE'][:3])).sum()
                if tu == tl and len(hc):
                    cy, cx = np.argwhere(comp).mean(0); tc = np.argwhere(torso).mean(0)
                    tu, tl = (1, 0) if np.hypot(*(np.array([cy, cx]) - hc.mean(0))) < np.hypot(*(np.array([cy, cx]) - tc)) else (0, 1)
                (ue if tu > tl else le)[comp] = True
            stump_of[L + 'UE'], stump_of[L + 'LE'] = ue, le
        lab3 = []
        for site in ('LUE', 'RUE', 'LLE', 'RLE'):
            sd = 1 if site[0] == 'L' else 2
            st = stump_of.get(site, np.zeros_like(torso))
            limb = (side == sd) & np.isin(seg, GROUP[site[1:]][:3]) | st
            if limb.sum() < 30:
                continue
            end = LE.limb_end(limb, torso); amp = site in prm['amputations']
            extm = (side == sd) & (seg == EXTC[site[1:]])
            for _ in range(2 if amp else 1):
                s = int(s0 * rng.uniform(0.85, 1.15))
                cx, cy = [int(v + rng.uniform(-0.15, 0.15) * s) for v in end]
                ys, xs = slice(max(0, cy - s // 2), max(0, cy - s // 2 + s)), slice(max(0, cx - s // 2), max(0, cx - s // 2 + s))
                if amp and st[ys, xs].sum() >= 15:
                    lab3.append(1)
                elif not amp and extm[ys, xs].sum() >= 15:
                    lab3.append(0)
                else:
                    lab3.append(2)
        z = dict(np.load(f))
        assert len(lab3) == len(z['label']), (sid, len(lab3), len(z['label']))
        z['label3'] = np.array(lab3, np.int64); np.savez_compressed(f, **z)
        tally.update((int(a), int(b)) for a, b in zip(z['label'], lab3))
    print('RELABEL3 (amputated, state3) counts:', dict(tally))
