"""Tests B and C: RTMW-prompted SAM 2.1 tiny masks (pose env).
Usage: python sam_pipeline.py <scene_dir> [...]

Per scene:
  prompts (RTMW keypoints with confidence >= 0.3):
    limb  UE: + mid upper arm, elbow, wrist      LE: + mid thigh, knee, ankle
          - torso centre, nose, and the other three limbs' elbow/knee points
          The hand / foot keypoint is NOT a prompt: it is kept to test whether the limb mask reaches it.
    torso +: shoulder mid, hip mid, torso centre, nose     -: elbows, knees, wrists, ankles
  mask choice: among SAM's 3 candidates maximise  score + (fraction of + points inside) - 2 x (fraction of - points inside)
  class map (640x480): limbs (overlaps -> higher SAM score), then torso, else background. No occluder class
           from SAM; variant C2 adds the generator's occluder pixels to isolate that effect.
  hand/foot present = the limb mask covers the RTMW hand/foot keypoint (within 2% of image width).

Test B: per-limb IoU vs generator masks, left/right swaps, hand/foot presence vs truth (incl. stumps).
Test C: the unchanged Test A2 rules run on the SAM class map (C1) and SAM + true occluders (C2).
"""
import sys, os, json, glob, collections, time
import numpy as np, cv2, torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
sys.path.insert(0, os.path.dirname(__file__))
import test_a_masks as TA

torch.set_num_threads(2)
SITES = ['LUE', 'RUE', 'LLE', 'RLE']
PROMPT = {'LUE': [(5, 7, 0.5), 7, 9], 'RUE': [(6, 8, 0.5), 8, 10], 'LLE': [(11, 13, 0.5), 13, 15], 'RLE': [(12, 14, 0.5), 14, 16]}
MIDJ = {'LUE': 7, 'RUE': 8, 'LLE': 13, 'RLE': 14}
EXT = {'LUE': 100, 'RUE': 121, 'LLE': (17, 18, 19), 'RLE': (20, 21, 22)}
W, H, IW, IH = 1280, 960, 640, 480
T_CONF = float(os.environ.get('T_CONF', 0.3))
VARIANT = os.environ.get('VARIANT', 'base')


def pt(k, s, spec):
    if isinstance(spec, tuple) and len(spec) == 3 and isinstance(spec[2], float):
        a, b, f = spec
        return k[a] + f * (k[b] - k[a]), min(s[a], s[b])
    if isinstance(spec, tuple):
        j = max(spec, key=lambda q: s[q]); return k[j], s[j]
    return k[spec], s[spec]


def choose(masks, scores, pos, neg):
    if VARIANT == 'D':                       # strict: no negative point inside; smallest mask holding >= 60% of positives
        def inside(m, P):
            return [m[int(np.clip(y, 0, H - 1)), int(np.clip(x, 0, W - 1))] for x, y in P]
        ok = [i for i, m in enumerate(masks) if not any(inside(m, neg)) and np.mean(inside(m, pos)) >= 0.6]
        if ok:
            i = min(ok, key=lambda j: masks[j].sum())
            return masks[i].astype(bool), float(scores[i])
    best, bi = -9, 0
    for i, m in enumerate(masks):
        def frac(P):
            if len(P) == 0:
                return 0.0
            ins = [m[int(np.clip(y, 0, H - 1)), int(np.clip(x, 0, W - 1))] for x, y in P]
            return float(np.mean(ins))
        v = scores[i] + frac(pos) - 2 * frac(neg)
        if v > best:
            best, bi = v, i
    return masks[bi].astype(bool), float(scores[bi])


def run(D, P):
    out = {}
    for f in sorted(glob.glob(os.path.join(D, 'C*_sidecar.json'))):
        sid = json.load(open(f))['scene_id']
        z = np.load(os.path.join(D, sid + '_rtmw133.npz')); k, s = z['k'], z['s']
        img = cv2.cvtColor(cv2.imread(os.path.join(D, sid + '.jpg')), cv2.COLOR_BGR2RGB)
        P.set_image(img)
        res = {}
        tpos0 = [(k[5] + k[6]) / 2, (k[11] + k[12]) / 2, (k[5] + k[6] + k[11] + k[12]) / 4, k[0]]
        tneg0 = [k[j] for j in (7, 8, 13, 14, 9, 10, 15, 16) if s[j] >= T_CONF]
        mm, ss_, _ = P.predict(point_coords=np.array(tpos0 + tneg0), point_labels=np.array([1] * 4 + [0] * len(tneg0)),
                               multimask_output=True)
        tm0, _ = choose(mm, ss_, tpos0, tneg0)
        for site in SITES:
            pos = [pt(k, s, sp) for sp in PROMPT[site]]
            pos = [p for p, c in pos if c >= T_CONF]
            neg = [(k[5] + k[6] + k[11] + k[12]) / 4, k[0]] + [k[MIDJ[o]] for o in SITES if o != site and s[MIDJ[o]] >= T_CONF]
            if not pos:
                res[site] = None; continue
            pts = np.array(pos + neg); lab = np.array([1] * len(pos) + [0] * len(neg))
            masks, scores, _ = P.predict(point_coords=pts, point_labels=lab, multimask_output=True)
            m, sc = choose(masks, scores, pos, neg)
            if VARIANT == 'D':
                m = m & ~tm0
            e, ec = pt(k, s, EXT[site])
            r = int((0.01 if VARIANT == 'D' else 0.02) * W); x, y = int(np.clip(e[0], 0, W - 1)), int(np.clip(e[1], 0, H - 1))
            win = m[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
            inside = 0 <= e[0] < W and 0 <= e[1] < H and (VARIANT != 'D' or ec >= 0.5)
            res[site] = {'mask': m, 'score': sc, 'ext_present': bool(inside and win.any())}
        tpos = [(k[5] + k[6]) / 2, (k[11] + k[12]) / 2, (k[5] + k[6] + k[11] + k[12]) / 4, k[0]]
        tneg = [k[j] for j in (7, 8, 13, 14, 9, 10, 15, 16) if s[j] >= T_CONF]
        masks, scores, _ = P.predict(point_coords=np.array(tpos + tneg), point_labels=np.array([1] * 4 + [0] * len(tneg)),
                                     multimask_output=True)
        tm, _ = choose(masks, scores, tpos, tneg)
        out[sid] = (res, tm)
    return out


def to_classmap(res, tm, names):
    idx = {n: i for i, n in enumerate(names)}
    small = lambda m: cv2.resize(m.astype(np.uint8), (IW, IH), interpolation=cv2.INTER_NEAREST).astype(bool)
    cm = np.full((IH, IW), idx['BG'])
    cm[small(tm)] = idx['TORSO']
    best = np.full((IH, IW), -1.0)
    for site in SITES:
        r = res[site]
        if r is None:
            continue
        m = small(r['mask']) & (r['score'] > best)
        cm[m] = idx[site]; best[m] = r['score']
    return cm


if __name__ == '__main__':
    dirs = sys.argv[1:]
    model = build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml', os.path.join(os.path.dirname(__file__), '..', 'models',
                       'sam2_1_hiera_tiny.pt'), device='cpu')
    P = SAM2ImagePredictor(model)
    C = ['no_injury', 'amputation', 'not_testable']
    iou = collections.defaultdict(list); swaps = 0; judged = 0
    ext = collections.Counter()
    cms = {'C1': np.zeros((3, 3), int), 'C2': np.zeros((3, 3), int)}
    rows = []
    t0 = time.time()
    for D in dirs:
        out = run(D, P)
        for sid, (res, tm) in out.items():
            sc = json.load(open(os.path.join(D, sid + '_sidecar.json')))
            gcm, names = TA.classmap(os.path.join(D, sid + '_id.png'))
            gkp = json.load(open(os.path.join(D, sid + '_gtkp.json')))
            idx = {n: i for i, n in enumerate(names)}
            pcm = to_classmap(res, tm, names)
            # ---- Test B ----
            for site in SITES:
                g = gcm == idx[site]; p = pcm == idx[site]
                if g.sum() >= 30:
                    iou[site[1:]].append((g & p).sum() / max((g | p).sum(), 1))
                    opp = ('R' if site[0] == 'L' else 'L') + site[1:]
                    go = gcm == idx[opp]
                    if p.sum() >= 30:
                        judged += 1
                        if (p & go).sum() > (p & g).sum():
                            swaps += 1
                if sc['labels_by_threshold']['0.10'][site] != 'not_testable' and res[site] is not None:
                    en = 'hand' if site[1:] == 'UE' else 'foot'
                    truth = gkp['sites'][site][en]['visible']
                    amp = site in sc['params']['amputations']
                    ext[('amputated' if amp else 'intact', 'truth_visible' if truth else 'truth_not_visible',
                         'sam_present' if res[site]['ext_present'] else 'sam_absent')] += 1
            # ---- Test C ----
            for var in ['C1', 'C2']:
                cm = pcm.copy()
                if var == 'C2':
                    cm[gcm == idx['OCC']] = idx['OCC']
                an = TA.analyse(cm, names)
                for site in SITES:
                    r = an[site]
                    present = res[site]['ext_present'] if res[site] is not None else False
                    if r['vis_px'] < 30:
                        pred = 'not_testable'
                    elif present:
                        pred = 'no_injury' if r['Lfrac'] >= 0.10 else 'not_testable'
                    elif r['bg_frac'] is not None and r['bg_frac'] >= 0.60:
                        pred = 'amputation'
                    else:
                        pred = 'no_injury' if r['Lfrac'] >= 0.10 else 'not_testable'
                    gt = sc['labels_by_threshold']['0.10'][site]
                    cms[var][C.index(gt), C.index(pred)] += 1
                    if var == 'C1':
                        rows.append({'scene': sid, 'site': site, 'gt': gt, 'pred': pred, 'present': present,
                                     'amp_level': sc['params']['amputations'].get(site), 'occluder': sc['params']['occluder']})
            if sid in ('C001', 'C014', 'C021', 'C050'):
                pal = np.array([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (188, 188, 188)])
                vis = np.hstack([pal[gcm], pal[pcm]]).astype(np.uint8)
                cv2.imwrite(os.path.join(dirs[0], f'testB_{sid}_truth_vs_sam_{VARIANT}.png'), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print('elapsed s', round(time.time() - t0))
    B = {'iou_median': {k: round(float(np.median(v)), 3) for k, v in iou.items()},
         'iou_share_ge_0.7': {k: round(float(np.mean(np.array(v) >= 0.7)), 3) for k, v in iou.items()},
         'n_limbs': {k: len(v) for k, v in iou.items()}, 'left_right_swaps': [swaps, judged],
         'hand_foot_presence': {' / '.join(k): v for k, v in sorted(ext.items())}}
    print('TEST B', json.dumps(B, indent=1))
    res_out = {'test_B': B}
    for var, m in cms.items():
        acc = np.trace(m) / m.sum()
        rec = [round(float(m[i, i] / m[i].sum()), 3) for i in range(3)]
        print(f'TEST {var} accuracy {acc:.3f} recall(no_inj, amp, not_test) {rec} confusion {m.tolist()}')
        res_out['test_' + var] = {'accuracy': round(float(acc), 4), 'recall': rec, 'confusion': m.tolist()}
    json.dump(res_out, open(os.path.join(dirs[0], f'testBC_results_{VARIANT}.json'), 'w'), indent=1)
    json.dump(rows, open(os.path.join(dirs[0], f'testC_sites_{VARIANT}.json'), 'w'), indent=1)
