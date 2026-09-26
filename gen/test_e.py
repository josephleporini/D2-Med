"""Test E: trained side-agnostic part segmenter + body-frame side assignment + unchanged Test A2 rules (pose env).
Usage: python test_e.py <scene_dir> [...]      (needs <sid>_feat.npz in det mode and models/seg_decoder.pt)

Arms (all use the same side rule and the same Test A2 decision logic):
  E0  ground-truth part map (sides removed) + ground-truth frame   -> cost of the side rule and pixel hand/foot evidence
  E1  predicted part map                    + ground-truth frame   -> cost of segmentation
  E2  predicted part map                    + estimated frame      -> full image-only pipeline
      estimated frame = RTMW torso axis + fused facing classifier (leave-one-scene-out, as in frame_test.py)

Side rule v1 (fixed before scoring; SIDE_RULE=v1):
  limb group UE = upper_arm|forearm|hand, LE = thigh|shank|foot; 8-connected components, < 10 px dropped.
  anchor = centroid of the component's pixels within 5 px of the torso, else the component centroid.
  side = LEFT if (anchor - torso centroid) . left_dir > 0 else RIGHT.
Hand/foot present = >= 10 hand (UE) / foot (LE) pixels in that side's components (640x480 scale).
Head pixels count as torso (as in Test A).
"""
import sys, os, json, glob, collections
import numpy as np, cv2, torch
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
sys.path.insert(0, os.path.dirname(__file__))
import parts as PT, test_a_masks as TA
from seg_train import Decoder

SITES = ['LUE', 'RUE', 'LLE', 'RLE']
C = ['no_injury', 'amputation', 'not_testable']
IW, IH = 640, 480
S_ = {n: i for i, n in enumerate(PT.SEG_CLASSES)}
GROUP = {'UE': [S_['upper_arm'], S_['forearm'], S_['hand']], 'LE': [S_['thigh'], S_['shank'], S_['foot']]}
EXT = {'UE': S_['hand'], 'LE': S_['foot']}
MIN_COMP, MIN_EXT = 10, 10
SIDE_RULE = os.environ.get('SIDE_RULE', 'v2')
ARMS = os.environ.get('ARMS', 'E0,E1,E2').split(',')
perp = lambda a: np.array([-a[1], a[0]])
unit = lambda v: v / (np.linalg.norm(v) + 1e-9)


def predict_seg(net, D, sid):
    z = np.load(os.path.join(D, sid + '_feat.npz'))
    with torch.no_grad():
        lg = net(torch.from_numpy(z['embed'].astype(np.float32))[None], torch.from_numpy(z['s1'].astype(np.float32))[None])
    X1, Y1, S = [int(v) for v in z['crop']]
    x1, y1, s = X1 / 2, Y1 / 2, max(int(round(S / 2)), 1)
    lg = torch.nn.functional.interpolate(lg, size=(s, s), mode='bilinear', align_corners=False)[0].argmax(0).numpy()
    seg = np.zeros((IH, IW), np.uint8)
    ox, oy = int(round(x1)), int(round(y1))
    ys0, xs0 = max(0, oy), max(0, ox); ys1, xs1 = min(IH, oy + s), min(IW, ox + s)
    if ys1 > ys0 and xs1 > xs0:
        seg[ys0:ys1, xs0:xs1] = lg[ys0 - oy:ys1 - oy, xs0 - ox:xs1 - ox]
    return seg


def to_classmap(seg, left_dir):
    """side-agnostic seg -> Test A class map (TA names) + hand/foot pixel counts per site."""
    names = list(TA.COL); ix = {n: i for i, n in enumerate(names)}
    cm = np.full(seg.shape, ix['BG'])
    torso = (seg == S_['TORSO']) | (seg == S_['HEAD'])
    cm[torso] = ix['TORSO']; cm[seg == S_['OCC']] = ix['OCC']
    tp = np.argwhere(seg == S_['TORSO'])
    if len(tp) == 0:
        tp = np.argwhere(torso)
    body = np.argwhere(seg > 0)
    tc = (tp if len(tp) else body if len(body) else np.array([[IH / 2, IW / 2]])).mean(0)[::-1]   # (x, y)
    ring = ndimage.binary_dilation(torso, iterations=5)
    ext = collections.Counter()
    yy, xx = np.mgrid[0:seg.shape[0], 0:seg.shape[1]]
    dmap = (xx - tc[0]) * left_dir[0] + (yy - tc[1]) * left_dir[1]          # signed lateral offset per pixel
    for g, cls in GROUP.items():
        lab, n = ndimage.label(np.isin(seg, cls), structure=np.ones((3, 3)))
        comps = []
        for c in range(1, n + 1):
            comp = lab == c
            if comp.sum() < MIN_COMP:
                continue
            a = comp & ring
            a = a if a.any() else comp
            ay, ax = np.nonzero(a)
            anchor = np.array([ax.mean(), ay.mean()])
            comps.append((comp, 'L' if (anchor - tc) @ left_dir > 0 else 'R', dmap[a]))
        for comp, side, da in comps:
            parts_ = [(comp, side)]
            if SIDE_RULE == 'v2':
                # merged pair (e.g. legs lying together form one component): torso contact on both sides of the
                # midline, both sides well populated, and no other component already claims the minority side
                dp = dmap[comp]; minority = 'R' if side == 'L' else 'L'
                both_ring = min((da > 0).mean(), (da <= 0).mean()) >= 0.20
                both_px = min((dp > 0).mean(), (dp <= 0).mean()) >= 0.25
                claimed = any(s2 == minority for c2, s2, _ in comps if c2 is not comp)
                if both_ring and both_px and not claimed:
                    parts_ = [(comp & (dmap > 0), 'L'), (comp & (dmap <= 0), 'R')]
            for m, sd in parts_:
                cm[m] = ix[sd + g]
                ext[sd + g] += int((m & (seg == EXT[g])).sum())
    return cm, names, ext


def decide_a2(r, present):
    """Test A2 logic, verbatim from sam_pipeline.py Test C."""
    if r['vis_px'] < 30:
        return 'not_testable'
    if present:
        return 'no_injury' if r['Lfrac'] >= 0.10 else 'not_testable'
    if r['bg_frac'] is not None and r['bg_frac'] >= 0.60:
        return 'amputation'
    return 'no_injury' if r['Lfrac'] >= 0.10 else 'not_testable'


def est_frames(scenes):
    rows = []
    for sid, D in scenes:
        z = np.load(os.path.join(D, sid + '_rtmw133.npz')); k, s = z['k'], z['s']
        g = json.load(open(os.path.join(D, sid + '_gtframe.json')))
        sm, hm = (k[5] + k[6]) / 2, (k[11] + k[12]) / 2
        tl = np.linalg.norm(sm - hm) + 1e-9; a = unit(sm - hm); pa = perp(a)
        feet = []
        for toe, heel in ((17, 19), (20, 22)):
            v = (k[toe] - k[heel]) / tl; feet += [float(v @ a), float(v @ pa)]
        feat = [float(s[23:91].mean()), float(s[0:3].mean()), float(s[3:5].mean()),
                float((k[5] - k[6]) @ pa / tl), float((k[11] - k[12]) @ pa / tl)] + feet
        rows.append((sid, a, feat, int(g['facing_cos'] > 0)))
    X = np.array([r[2] for r in rows]); y = np.array([r[3] for r in rows]); out = {}
    for i, (sid, a, _, _) in enumerate(rows):
        tr = np.arange(len(y)) != i
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)).fit(X[tr], y[tr])
        f = 1 if clf.predict_proba(X[i:i + 1])[0, 1] > 0.5 else -1
        out[sid] = f * perp(a)
    return out


def summarise(m):
    return {'accuracy': round(float(np.trace(m) / m.sum()), 4),
            'recall': [round(float(m[i, i] / m[i].sum()), 3) for i in range(3)], 'confusion': m.tolist()}


if __name__ == '__main__':
    dirs = sys.argv[1:]
    net = Decoder()
    if ARMS != ['E0']:
        net.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), '..', 'models', 'seg_decoder.pt')))
    net.eval()
    OUT = os.environ.get('OUT', 'testE')
    scenes = [(json.load(open(f))['scene_id'], D) for D in dirs for f in sorted(glob.glob(os.path.join(D, 'C*_sidecar.json')))
              if os.path.exists(f.replace('_sidecar.json', '_gtframe.json'))]
    est = est_frames(scenes) if 'E2' in ARMS else {}
    NC = len(PT.SEG_CLASSES); pconf = np.zeros((NC, NC), np.int64)
    cms = {a: np.zeros((3, 3), int) for a in ARMS}
    swaps = collections.Counter(); judged = collections.Counter(); rows = []
    for sid, D in scenes:
        sc = json.load(open(os.path.join(D, sid + '_sidecar.json')))
        gseg, _ = PT.decode(os.path.join(D, sid + '_part.png'))
        gseg = cv2.resize(gseg, (IW, IH), interpolation=cv2.INTER_NEAREST)
        pseg = predict_seg(net, D, sid) if ARMS != ['E0'] else gseg
        pconf += np.bincount(gseg.ravel().astype(np.int64) * NC + pseg.ravel(), minlength=NC * NC).reshape(NC, NC)
        gcm, gnames = TA.classmap(os.path.join(D, sid + '_id.png')); gix = {n: i for i, n in enumerate(gnames)}
        gl = np.array(json.load(open(os.path.join(D, sid + '_gtframe.json')))['left_dir'])
        for arm, seg, ld in (('E0', gseg, gl), ('E1', pseg, gl), ('E2', pseg, est.get(sid))):
            if arm not in ARMS:
                continue
            cm, names, ext = to_classmap(seg, ld); ix = {n: i for i, n in enumerate(names)}
            an = TA.analyse(cm, names)
            for site in SITES:
                p = cm == ix[site]; g = gcm == gix[site]
                opp = ('R' if site[0] == 'L' else 'L') + site[1:]
                if g.sum() >= 30 and p.sum() >= 30:
                    judged[arm] += 1; swaps[arm] += int((p & (gcm == gix[opp])).sum() > (p & g).sum())
                pred = decide_a2(an[site], ext[site] >= MIN_EXT)
                gt = sc['labels_by_threshold']['0.10'][site]
                cms[arm][C.index(gt), C.index(pred)] += 1
                rows.append({'arm': arm, 'scene': sid, 'site': site, 'gt': gt, 'pred': pred, 'ext_px': ext[site],
                             'amp_level': sc['params']['amputations'].get(site), 'occluder': sc['params']['occluder'],
                             'framing': sc['params']['framing'], 'position': sc['params']['body_position'], **an[site]})
        if sid in ('C001', 'C014', 'C021', 'C050'):
            pal = np.array([(0, 0, 0), (255, 0, 255), (255, 128, 255), (0, 255, 255), (255, 0, 0), (0, 200, 0),
                            (255, 255, 0), (0, 0, 255), (255, 128, 0), (160, 80, 0)])
            img = cv2.resize(cv2.imread(os.path.join(D, sid + '.jpg')), (IW, IH))
            vis = np.hstack([img, pal[gseg][..., ::-1], pal[pseg][..., ::-1]]).astype(np.uint8)
            cv2.imwrite(os.path.join(dirs[0], f'testE_{sid}_image_truth_pred.png'), vis)
    inter = np.diag(pconf); union = pconf.sum(0) + pconf.sum(1) - inter
    part_iou = {c: round(float(inter[i] / union[i]), 3) for i, c in enumerate(PT.SEG_CLASSES)}
    res = {'part_iou_640': part_iou, 'n_scenes': len(scenes)}
    print('PART IoU', part_iou)
    for arm, m in cms.items():
        res[arm] = summarise(m); res[arm]['left_right_swaps'] = [swaps[arm], judged[arm]]
        print(arm, json.dumps(res[arm]))
    json.dump(res, open(os.path.join(dirs[0], OUT + '_results.json'), 'w'), indent=1)
    json.dump(rows, open(os.path.join(dirs[0], OUT + '_sites.json'), 'w'), indent=1, default=str)
