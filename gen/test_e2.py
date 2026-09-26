"""Test E2: v2 part segmenter (stump + front/back classes, 256 px) -> side rule v2 -> fixed decision rules (pose env).
Usage: MODEL=<name> OUT=<prefix> python test_e2.py <scene_dir> [...]   (needs <sid>_f2.npz / _f2m.npz in det mode)

Declared before scoring:
  Class map   torso = TORSO_F|TORSO_B|HEAD_F|HEAD_B (head counts as torso, as in Test A); OCC; limbs as in test_e.py.
              stump pixels join the limb group of the limb pixels they touch (else: UE if nearer the head end).
  Side rule   v2 from test_e.py (anchor at torso contact; merged pairs split at the midline).
  Decision A2 Test A2 logic unchanged; stump pixels count as limb pixels.
  Decision A3 A2, except a side with >= 10 stump pixels (and >= 30 limb pixels) is called amputation.
  Frames      F_gt   ground truth
              F_rtmw RTMW torso axis + fused facing classifier (leave-one-scene-out; as in Test E)
              F_seg  segmentation only: axis = torso centroid -> head centroid; facing = front share of
                     torso+head pixels > 0.5
              F_hyb  RTMW torso axis + segmentation facing   (declared 23 Sep, after the pilot144 run)
Arms: E0 (truth parts, F_gt), E1 (pred, F_gt), E2r (pred, F_rtmw), E2s (pred, F_seg), E2h (pred, F_hyb),
      E2s_mirror (mirrored image).
Mirror test: E2s labels on the mirrored image, with L/R exchanged, compared with E2s labels on the original.
"""
import sys, os, json, glob, collections
import numpy as np, cv2, torch
from scipy import ndimage
sys.path.insert(0, os.path.dirname(__file__))
import parts as PT, test_a_masks as TA
import test_e as TE
from seg_train2 import Decoder2
from seg_features2 import load_f2

SITES = TE.SITES; C = TE.C; IW, IH = 640, 480
K = {n: i for i, n in enumerate(PT.SEG2_CLASSES)}
TORSO_C = [K['TORSO_F'], K['TORSO_B'], K['HEAD_F'], K['HEAD_B']]
HEAD_C = [K['HEAD_F'], K['HEAD_B']]
FRONT_C = [K['TORSO_F'], K['HEAD_F']]
GROUP = {'UE': [K['upper_arm'], K['forearm'], K['hand']], 'LE': [K['thigh'], K['shank'], K['foot']]}
EXT = {'UE': K['hand'], 'LE': K['foot']}
MIN_COMP, MIN_EXT, MIN_STUMP = 10, 10, 10
unit = TE.unit; perp = TE.perp


def predict(net, path):
    e, s1, s0, z = load_f2(path)
    with torch.no_grad():
        lg = net(torch.from_numpy(e)[None], torch.from_numpy(s1)[None], torch.from_numpy(s0)[None])
    X1, Y1, S = [int(v) for v in z['crop']]
    if bool(z['mirrored']):
        X1 = 1280 - (X1 + S)                       # crop position in the mirrored image
    s = max(int(round(S / 2)), 1)
    lab = torch.nn.functional.interpolate(lg, size=(s, s), mode='bilinear', align_corners=False)[0].argmax(0).numpy()
    seg = np.zeros((IH, IW), np.uint8)
    ox, oy = int(round(X1 / 2)), int(round(Y1 / 2))
    ys0, xs0 = max(0, oy), max(0, ox); ys1, xs1 = min(IH, oy + s), min(IW, ox + s)
    if ys1 > ys0 and xs1 > xs0:
        seg[ys0:ys1, xs0:xs1] = lab[ys0 - oy:ys1 - oy, xs0 - ox:xs1 - ox]
    return seg


def seg_frame(seg):
    """axis (hips->head direction, image xy) and facing from the segmentation alone."""
    t = np.argwhere(np.isin(seg, [K['TORSO_F'], K['TORSO_B']])); h = np.argwhere(np.isin(seg, HEAD_C))
    if len(t) == 0 or len(h) == 0:
        return None
    a = unit(h.mean(0)[::-1] - t.mean(0)[::-1])
    tot = np.isin(seg, TORSO_C).sum(); fr = np.isin(seg, FRONT_C).sum()
    f = 1 if fr / max(tot, 1) > 0.5 else -1
    return f * perp(a), a, f


def to_classmap(seg, left_dir, axis):
    names = list(TA.COL); ix = {n: i for i, n in enumerate(names)}
    cm = np.full(seg.shape, ix['BG'])
    torso = np.isin(seg, TORSO_C)
    cm[torso] = ix['TORSO']; cm[seg == K['OCC']] = ix['OCC']
    tp = np.argwhere(np.isin(seg, [K['TORSO_F'], K['TORSO_B']]))
    if len(tp) == 0:
        tp = np.argwhere(torso)
    body = np.argwhere(seg > 0)
    tc = (tp if len(tp) else body if len(body) else np.array([[IH / 2, IW / 2]])).mean(0)[::-1]
    ring = ndimage.binary_dilation(torso, iterations=5)
    # stump pixels -> limb group they touch (else by position along the axis)
    stump = seg == K['stump']; gmask = {g: np.isin(seg, c) for g, c in GROUP.items()}
    slab, ns = ndimage.label(stump, structure=np.ones((3, 3)))
    for c in range(1, ns + 1):
        comp = slab == c; rim = ndimage.binary_dilation(comp, iterations=2)
        touch = {g: int((rim & m).sum()) for g, m in gmask.items()}
        if max(touch.values()) > 0:
            g = max(touch, key=touch.get)
        else:
            cy, cx = np.argwhere(comp).mean(0)
            g = 'UE' if (np.array([cx, cy]) - tc) @ axis > 0 else 'LE'
        gmask[g] = gmask[g] | comp
    yy, xx = np.mgrid[0:IH, 0:IW]
    dmap = (xx - tc[0]) * left_dir[0] + (yy - tc[1]) * left_dir[1]
    ext = collections.Counter(); stc = collections.Counter()
    for g in GROUP:
        lab, n = ndimage.label(gmask[g], structure=np.ones((3, 3)))
        comps = []
        for c in range(1, n + 1):
            comp = lab == c
            if comp.sum() < MIN_COMP:
                continue
            a = comp & ring; a = a if a.any() else comp
            ay, ax = np.nonzero(a)
            comps.append((comp, 'L' if (np.array([ax.mean(), ay.mean()]) - tc) @ left_dir > 0 else 'R', dmap[a]))
        for comp, side, da in comps:
            parts_ = [(comp, side)]
            dp = dmap[comp]; minority = 'R' if side == 'L' else 'L'
            if (min((da > 0).mean(), (da <= 0).mean()) >= 0.20 and min((dp > 0).mean(), (dp <= 0).mean()) >= 0.25
                    and not any(s2 == minority for c2, s2, _ in comps if c2 is not comp)):
                parts_ = [(comp & (dmap > 0), 'L'), (comp & (dmap <= 0), 'R')]
            for m, sd in parts_:
                cm[m] = ix[sd + g]
                ext[sd + g] += int((m & (seg == EXT[g])).sum()); stc[sd + g] += int((m & stump).sum())
    return cm, names, ext, stc


def decide(r, present, stump_px, rule):
    if rule == 'A3' and r['vis_px'] >= 30 and stump_px >= MIN_STUMP:
        return 'amputation'
    return TE.decide_a2(r, present)


def run_arm(seg, left_dir, axis):
    cm, names, ext, stc = to_classmap(seg, left_dir, axis)
    an = TA.analyse(cm, names)
    return cm, names, {s: {r: decide(an[s], ext[s] >= MIN_EXT, stc[s], r) for r in ('A2', 'A3')} for s in SITES}, an, ext, stc


SWAP = {'LUE': 'RUE', 'RUE': 'LUE', 'LLE': 'RLE', 'RLE': 'LLE'}

if __name__ == '__main__':
    dirs = sys.argv[1:]
    name = os.environ.get('MODEL', 'seg2'); OUT = os.environ.get('OUT', 'testE2')
    net = Decoder2(); net.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), '..', 'models', name + '.pt')))
    net.eval()
    scenes = [(json.load(open(f))['scene_id'], D) for D in dirs for f in sorted(glob.glob(os.path.join(D, 'C*_sidecar.json')))
              if os.path.exists(f.replace('_sidecar.json', '_f2m.npz'))]
    est = TE.est_frames(scenes)
    NC = len(PT.SEG2_CLASSES); pconf = np.zeros((NC, NC), np.int64)
    ARMS = ['E0', 'E1', 'E2r', 'E2s', 'E2h']
    cms = {(a, r): np.zeros((3, 3), int) for a in ARMS for r in ('A2', 'A3')}
    swaps = collections.Counter(); judged = collections.Counter(); rows = []
    facing = collections.Counter(); axis_ok = collections.Counter()
    mirror = collections.Counter()
    for sid, D in scenes:
        sc = json.load(open(os.path.join(D, sid + '_sidecar.json')))
        gseg, _ = PT.decode2(os.path.join(D, sid + '_part2.png'))
        gseg = cv2.resize(gseg, (IW, IH), interpolation=cv2.INTER_NEAREST)
        pseg = predict(net, os.path.join(D, sid + '_f2.npz'))
        pconf += np.bincount(gseg.ravel().astype(np.int64) * NC + pseg.ravel(), minlength=NC * NC).reshape(NC, NC)
        gcm, gnames = TA.classmap(os.path.join(D, sid + '_id.png')); gix = {n: i for i, n in enumerate(gnames)}
        gf = json.load(open(os.path.join(D, sid + '_gtframe.json')))
        gl, ga = np.array(gf['left_dir']), np.array(gf['axis'])
        sf = seg_frame(pseg)
        # fallback when torso or head is not segmented: RTMW frame (axis recovered from left_dir, facing +1 assumed)
        s_left, s_axis, s_f = sf if sf else (est[sid], np.array([est[sid][1], -est[sid][0]]), 0)
        facing['rtmw'] += int(np.sign(est[sid] @ gl) > 0)
        facing['seg'] += int(sf is not None and s_left @ gl > 0); facing['n'] += 1
        axis_ok['seg'] += int(sf is not None and s_axis @ ga > 0)
        z = np.load(os.path.join(D, sid + '_rtmw133.npz')); kk = z['k']
        r_axis = unit((kk[5] + kk[6]) / 2 - (kk[11] + kk[12]) / 2)
        h_left = (s_f if s_f else 1) * perp(r_axis)
        facing['hyb'] += int(h_left @ gl > 0)
        arm_in = {'E0': (gseg, gl, ga), 'E1': (pseg, gl, ga), 'E2r': (pseg, est[sid], ga), 'E2s': (pseg, s_left, s_axis),
                  'E2h': (pseg, h_left, r_axis)}
        preds = {}
        for arm, (seg, ld, ax) in arm_in.items():
            cm, names, dec, an, ext, stc = run_arm(seg, ld, ax); ix = {n: i for i, n in enumerate(names)}
            preds[arm] = dec
            for site in SITES:
                p = cm == ix[site]; g = gcm == gix[site]
                if g.sum() >= 30 and p.sum() >= 30:
                    judged[arm] += 1; swaps[arm] += int((p & (gcm == gix[SWAP[site]])).sum() > (p & g).sum())
                gt = sc['labels_by_threshold']['0.10'][site]
                for r in ('A2', 'A3'):
                    cms[(arm, r)][C.index(gt), C.index(dec[site][r])] += 1
                rows.append({'arm': arm, 'scene': sid, 'site': site, 'gt': gt, 'A2': dec[site]['A2'], 'A3': dec[site]['A3'],
                             'ext_px': ext[site], 'stump_px': stc[site], 'amp_level': sc['params']['amputations'].get(site),
                             'occluder': sc['params']['occluder'], 'position': sc['params']['body_position'], **an[site]})
        # mirror-consistency (E2s): pipeline on the mirrored image, then exchange L/R
        mseg = predict(net, os.path.join(D, sid + '_f2m.npz'))
        mf = seg_frame(mseg)
        if mf and sf:
            _, _, mdec, _, _, _ = run_arm(mseg, mf[0], mf[1])
            for site in SITES:
                for r in ('A2', 'A3'):
                    mirror[(r, 'n')] += 1
                    mirror[(r, 'agree')] += int(mdec[SWAP[site]][r] == preds['E2s'][site][r])
            mirror['facing_agree'] += int(mf[2] == sf[2]); mirror['scenes'] += 1
        if sid in ('C001', 'C014', 'C050') or sid == 'C021':
            pal = np.array([(0, 0, 0), (255, 0, 255), (120, 0, 120), (255, 160, 255), (120, 80, 120), (0, 255, 255),
                            (255, 0, 0), (0, 200, 0), (255, 255, 0), (0, 0, 255), (255, 128, 0), (160, 80, 0), (255, 255, 255)])
            img = cv2.resize(cv2.imread(os.path.join(D, sid + '.jpg')), (IW, IH))
            cv2.imwrite(os.path.join(dirs[0], f'{OUT}_{sid}_image_truth_pred.png'),
                        np.hstack([img, pal[gseg][..., ::-1], pal[pseg][..., ::-1]]).astype(np.uint8))
    inter = np.diag(pconf); union = pconf.sum(0) + pconf.sum(1) - inter
    res = {'model': name, 'n_scenes': len(scenes),
           'part_iou_640': {c: round(float(inter[i] / union[i]), 3) for i, c in enumerate(PT.SEG2_CLASSES) if union[i]},
           'facing_acc': {k: round(facing[k] / facing['n'], 3) for k in ('rtmw', 'seg', 'hyb')},
           'seg_axis_head_end_correct': round(axis_ok['seg'] / facing['n'], 3),
           'mirror': {'scenes': mirror['scenes'], 'facing_agree': mirror['facing_agree'],
                      **{r: round(mirror[(r, 'agree')] / max(mirror[(r, 'n')], 1), 3) for r in ('A2', 'A3')}}}
    for (arm, r), m in cms.items():
        res[f'{arm}_{r}'] = TE.summarise(m); res[f'{arm}_{r}']['left_right_swaps'] = [swaps[arm], judged[arm]]
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(dirs[0], OUT + '_results.json'), 'w'), indent=1)
    json.dump(rows, open(os.path.join(dirs[0], OUT + '_sites.json'), 'w'), indent=1, default=str)
