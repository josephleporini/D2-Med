"""Cache limb-end windows for training the limb-end classifier (pose env).
Usage: python lend_cache.py <train_dir>
Ground-truth v2 part maps (with sides) give each visible limb (>= 30 px incl. stump pixels); its end window is jittered
(centre +-15% of the window, size x0.85-1.15). Amputated limbs get two windows (rarer class).
Writes <sid>_lend.npz: feat (k,256,32,32) fp16, mask (k,32,32), label (k; 1 = extremity absent), site (k).
"""
import sys, os, json, glob, time, zlib
import numpy as np, cv2, torch
from scipy import ndimage
sys.path.insert(0, os.path.dirname(__file__))
import parts as PT, lend as LE

torch.set_num_threads(8)
K = {n: i for i, n in enumerate(PT.SEG2_CLASSES)}
TORSO_C = [K['TORSO_F'], K['TORSO_B'], K['HEAD_F'], K['HEAD_B']]
SITE_IX = {'LUE': 1, 'RUE': 2, 'LLE': 3, 'RLE': 4}


def load_parts(D, sid):
    """v2 or v3 part map -> (seg, side, wound site). SEG3 = SEG2 + [wound, TQ], so part indices agree."""
    p2, p3 = os.path.join(D, sid + '_part2.png'), os.path.join(D, sid + '_part3.png')
    if os.path.exists(p3):
        return PT.decode3(p3)
    seg, side = PT.decode2(p2)
    return seg, side, np.zeros_like(seg)


GROUP = {'UE': [K['upper_arm'], K['forearm'], K['hand'], K['stump']], 'LE': [K['thigh'], K['shank'], K['foot'], K['stump']]}

if __name__ == '__main__':
    D = sys.argv[1]; P = LE.load_sam(); n = 0; t0 = time.time()
    for f in sorted(glob.glob(os.path.join(D, 'C*_sidecar.json'))):
        sid = os.path.basename(f)[:-len('_sidecar.json')]; out = os.path.join(D, sid + '_lend.npz')
        if os.path.exists(out) or not (os.path.exists(os.path.join(D, sid + '_part2.png')) or os.path.exists(os.path.join(D, sid + '_part3.png'))):
            continue
        prm = json.load(open(f))['params']
        seg, side, wsite = load_parts(D, sid)
        img = cv2.cvtColor(cv2.imread(os.path.join(D, sid + '.jpg')), cv2.COLOR_BGR2RGB)
        torso = np.isin(seg, TORSO_C); s0 = LE.window_size(torso)
        rng = np.random.default_rng(zlib.crc32(sid.encode()))
        # stump pixels carry a side but not a limb: give them to the amputated site(s) of that side
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
                if tu == tl and len(hc):          # no contact: nearer the head -> arm
                    cy, cx = np.argwhere(comp).mean(0); tc = np.argwhere(torso).mean(0)
                    tu, tl = (1, 0) if np.hypot(*(np.array([cy, cx]) - hc.mean(0))) < np.hypot(*(np.array([cy, cx]) - tc)) else (0, 1)
                (ue if tu > tl else le)[comp] = True
            stump_of[L + 'UE'], stump_of[L + 'LE'] = ue, le
        feats, masks, labels, sites = [], [], [], []
        for site in ('LUE', 'RUE', 'LLE', 'RLE'):
            sd = 1 if site[0] == 'L' else 2
            limb = (side == sd) & np.isin(seg, GROUP[site[1:]][:3]) | stump_of.get(site, np.zeros_like(torso)) | (wsite == SITE_IX[site])
            if limb.sum() < 30:
                continue
            end = LE.limb_end(limb, torso)
            amp = site in prm['amputations']
            for _ in range(2 if amp else 1):
                s = int(s0 * rng.uniform(0.85, 1.15))
                cx, cy = [int(v + rng.uniform(-0.15, 0.15) * s) for v in end]
                feats.append(LE.encode(P, LE.crop_1280(img, cx, cy, s)))
                masks.append(LE.mask_window(limb, cx, cy, s)); labels.append(int(amp)); sites.append(site)
        np.savez_compressed(out, feat=np.array(feats, np.float16).reshape(-1, 256, 32, 32), mask=np.array(masks, np.float32).reshape(-1, 32, 32),
                            label=np.array(labels, np.int64), site=np.array(sites))
        n += 1
    print('LEND', D, n, 'images', round(time.time() - t0), 's', flush=True)
