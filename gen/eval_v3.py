"""4-class evaluation of pipeline v3 (pose env). Two stages, both resumable.

Stage A, features:  python eval_v3.py feats <scene_dir> <arm: pred|gt> <out.jsonl> [seg_ckpt] [lend_ckpt] [lwound_ckpt]
  pred : part map from the end-to-end segmenter (SAM 2.1-tiny encoder fine-tuned + decoder, 15 classes)
  gt   : part map from the rendered v3 labels (isolates the decision stages from segmentation error)
  Per site: limb evidence (TA.analyse), limb-end state probabilities (EndNet 3-state), limb-wound probability
  (EndNet 2-class on a whole-limb window), predicted wound and tourniquet pixel counts.

Stage B, fit and score: python eval_v3.py fit <dev.jsonl> <test.jsonl> <out.json> [threshold=0.10]
  Multinomial logistic regression on dev only (C by 5-fold grouped CV), applied once to test.
  Reports accuracy, per-class recall, macro-F1, laterality swaps, a hand rule, and the majority baseline.
"""
import os, sys, json, glob, time
import numpy as np, cv2, torch
from scipy import ndimage
sys.path.insert(0, os.path.dirname(__file__))
torch.set_num_threads(int(os.environ.get('THREADS', 2)))

C4 = ['no_injury', 'wound', 'amputation', 'not_testable']
RING = ['BG', 'OCC', 'TORSO', 'OTHER_LIMB', 'EDGE']


# ------------------------------------------------------------------ stage A
def load_models(seg_ckpt, lend_ckpt, lw_ckpt):
    import parts as PT, lend as LE
    from seg_train_e2e import SegModel
    M = LE.M
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    seg = None
    if seg_ckpt:
        seg = SegModel(len(PT.SEG3_CLASSES), os.path.join(M, 'sam2_1_hiera_tiny.pt'))
        seg.load_state_dict(torch.load(seg_ckpt, map_location='cpu')); seg.to(dev).eval()
    end = LE.EndNet(3); end.load_state_dict(torch.load(lend_ckpt, map_location='cpu')); end.to(dev).eval()
    lw = LE.EndNet(2); lw.load_state_dict(torch.load(lw_ckpt, map_location='cpu')); lw.to(dev).eval()
    from rtmlib import YOLOX, Wholebody
    detp = os.path.join(M, '20230928/yolox_onnx/yolox_m_8xb8-300e_humanart-c2c7a14a/end2end.onnx')
    prov = 'cuda' if dev == 'cuda' else 'cpu'
    det = YOLOX(onnx_model=detp, model_input_size=(640, 640), backend='onnxruntime', device=prov)
    wb = Wholebody(det=detp, det_input_size=(640, 640), pose=os.path.join(M, 'end2end.onnx'), pose_input_size=(192, 256),
                   backend='onnxruntime', device=prov)
    return dict(seg=seg, end=end, lw=lw, det=det, wb=wb, P=LE.load_sam(), dev=dev)


def pred_part_map(m, img):
    """img 1280x960 RGB -> seg (480x640) in SEG3 classes, same crop geometry as training (15% pad on the person box)."""
    from seg_features import square_crop
    from seg_train_e2e import MEAN, STD
    b = m['det'](cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    box = tuple(float(v) for v in max(b, key=lambda q: (q[2] - q[0]) * (q[3] - q[1]))[:4]) if len(b) else (0, 0, 1280, 960)
    crop, (X1, Y1, S) = square_crop(img, box, 0.15)
    x = (torch.from_numpy(cv2.resize(crop, (1024, 1024))).permute(2, 0, 1).float() / 255 - MEAN) / STD
    with torch.no_grad():
        lg = m['seg'](x[None].to(m['dev'])).float()
        s = max(int(round(S / 2)), 1)
        lab = torch.nn.functional.interpolate(lg, size=(s, s), mode='bilinear', align_corners=False)[0].argmax(0).cpu().numpy()
    seg = np.zeros((480, 640), np.uint8)
    ox, oy = int(round(X1 / 2)), int(round(Y1 / 2))
    ys0, xs0 = max(0, oy), max(0, ox); ys1, xs1 = min(480, oy + s), min(640, ox + s)
    if ys1 > ys0 and xs1 > xs0:
        seg[ys0:ys1, xs0:xs1] = lab[ys0 - oy:ys1 - oy, xs0 - ox:xs1 - ox]
    return seg


def fold_extras(seg):
    """wound and TQ pixels carry no limb identity in the prediction: give each the label of the nearest limb-part pixel.
    Returns seg in SEG2 classes plus the wound and TQ masks."""
    import parts as PT
    K = {n: i for i, n in enumerate(PT.SEG3_CLASSES)}
    wm, tq = seg == K['wound'], seg == K['TQ']
    limbs = np.isin(seg, [K[n] for n in ('upper_arm', 'forearm', 'hand', 'thigh', 'shank', 'foot', 'stump')])
    out = seg.copy()
    extra = wm | tq
    if extra.any():
        if limbs.any():
            _, (iy, ix) = ndimage.distance_transform_edt(~limbs, return_indices=True)
            out[extra] = seg[iy[extra], ix[extra]]
        else:
            out[extra] = 0
    return out, wm, tq


def site_features(m, img, seg3, gt_side=None):
    import test_a_masks as TA, test_e as TE, test_e2 as T2, lend as LE
    from lwound_cache import limb_window
    seg, wm, tq = fold_extras(seg3)
    k, s = m['wb'](cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    kp = None
    if len(k):
        i = int(np.argmax(s[:, :17].mean(1))); kp = {'xy': np.round(k[i][:23], 1).tolist(), 'score': np.round(s[i][:23], 3).tolist()}; k = k[i]
        axis = TE.unit((k[5] + k[6]) / 2 - (k[11] + k[12]) / 2)
    else:
        axis = np.array([0.0, -1.0])
    sfr = T2.seg_frame(seg)
    left = np.array([1.0, 0.0]) if (sfr is None and not len(k)) else (sfr[2] if sfr else 1) * TE.perp(axis)
    cm, names, ext, stc = T2.to_classmap(seg, left, axis); ix = {n: i for i, n in enumerate(names)}
    an = TA.analyse(cm, names); torso = cm == ix['TORSO']; s0 = LE.window_size(torso); te = max(TA.extent(torso), 1)
    rows = {}
    for site in TE.SITES:
        r = an[site]; pm = cm == ix[site]; pe, pw = None, None
        if r['vis_px'] >= 30:
            x, y = LE.limb_end(pm, torso)
            fe = torch.from_numpy(LE.encode(m['P'], LE.crop_1280(img, x, y, s0))).float()[None]
            xin = torch.cat([fe, torch.from_numpy(LE.mask_window(pm, x, y, s0))[None, None]], 1).to(m['dev'])
            cx, cy, sw = limb_window(pm)
            fw = torch.from_numpy(LE.encode(m['P'], LE.crop_1280(img, cx, cy, sw))).float()[None]
            xw = torch.cat([fw, torch.from_numpy(LE.mask_window(pm, cx, cy, sw))[None, None]], 1).to(m['dev'])
            with torch.no_grad():
                pe = torch.softmax(m['end'](xin), 1)[0].cpu().numpy().tolist()
                pw = torch.softmax(m['lw'](xw), 1)[0].cpu().numpy().tolist()
        grown = ndimage.binary_dilation(pm, iterations=2)
        rows[site] = dict(vis_px=int(r['vis_px']), Lfrac=r['Lfrac'], bg_frac=r['bg_frac'], end=r.get('end') or {},
                          ext_px=int(ext[site]), stump_px=int(stc[site]), torso_ext=float(te), p_end=pe, p_wound=pw,
                          wound_px=int((wm & grown).sum()), tq_px=int((tq & grown).sum()))
    return rows, kp


def gt_part_map(path):
    import parts as PT
    seg, side, ws = PT.decode3(path)
    return cv2.resize(seg, (640, 480), interpolation=cv2.INTER_NEAREST)


def run_feats(D, arm, out, seg_ckpt, lend_ckpt, lw_ckpt):
    sys.path.insert(0, os.path.dirname(__file__))
    from d2pipe import letterbox
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)['scene'] for l in open(out)}
    m = load_models(seg_ckpt if arm == 'pred' else None, lend_ckpt, lw_ckpt)
    t0 = time.time(); n = 0
    for f in sorted(glob.glob(os.path.join(D, 'C*_sidecar.json'))):
        sid = os.path.basename(f)[:-len('_sidecar.json')]
        if sid in done:
            continue
        sc = json.load(open(f))
        img = letterbox(cv2.cvtColor(cv2.imread(os.path.join(D, sid + '.jpg')), cv2.COLOR_BGR2RGB))
        t1 = time.time()
        seg3 = pred_part_map(m, img) if arm == 'pred' else gt_part_map(os.path.join(D, sid + '_part3.png'))
        tseg = round(time.time() - t1, 3)
        t1 = time.time(); rows, kp = site_features(m, img, seg3); ts = round(time.time() - t1, 3)
        g = sc['params'].get('garments', {}) if isinstance(sc['params'], dict) else {}
        rec = dict(scene=sid, arm=arm, sites=rows, kp=kp, visible_fraction=sc.get('visible_fraction'), t_sites=ts, t_seg=tseg, labels=sc['labels_by_threshold'], labels_wip=sc['labels_wound_if_present'],
                   clothed=bool(g.get('top', 'none') != 'none' or g.get('bottom', 'none') != 'none'))
        with open(out, 'a') as fo:
            fo.write(json.dumps(rec) + '\n')
        n += 1
        print('FEAT', sid, round((time.time() - t0) / n, 1), 's/img', flush=True)


# ------------------------------------------------------------------ stage B
FEATURES = ['log_vis_px', 'Lfrac', 'bg_frac', 'pe_visible', 'pe_stump', 'pe_hidden', 'log_ext_px', 'log_stump_px'] + \
    ['ring_' + k for k in RING] + ['vis_over_torso2', 'window_present', 'p_wound', 'log_wound_px', 'wound_frac', 'log_tq_px']


def feat(r):
    pe = r['p_end'] or [0.0, 0.0, 0.0]; pw = r['p_wound'][1] if r['p_wound'] else 0.0
    end = r['end']; tot = sum(end.values()) or 1
    return [np.log1p(r['vis_px']), r['Lfrac'], r['bg_frac'] if r['bg_frac'] is not None else -1, *pe,
            np.log1p(r['ext_px']), np.log1p(r['stump_px']), *[end.get(q, 0) / tot for q in RING],
            r['vis_px'] / r['torso_ext'] ** 2, float(r['p_end'] is not None), pw, np.log1p(r['wound_px']),
            r['wound_px'] / max(r['vis_px'], 1), np.log1p(r['tq_px'])]


LIMB_FEATURES = ['log_amodal_px', 'vis_over_amodal', 'term_peak', 'p_cause_intact_visible', 'p_cause_occluded',
                 'p_cause_out_of_frame', 'p_cause_amputated_visible', 'p_cause_amputated_hidden']


def limb_feat(r):
    """BT-1 per-limb head features (bt1_limb.limb_site_features); same order as LIMB_FEATURES"""
    l = r['limb']
    return [np.log1p(l['amodal_px']), l['vis_over_amodal'], l['term_peak'], *l['p_cause']]


def rule(r):
    """hand rule for comparison: not visible -> not_testable; stump likely -> amputation; wound likely -> wound."""
    if r['vis_px'] < 30 or r['p_end'] is None:
        return 'not_testable'
    if r['p_end'][1] > 0.5:
        return 'amputation'
    if (r['p_wound'] and r['p_wound'][1] > 0.5) or r['wound_px'] >= 20:
        return 'wound'
    return 'no_injury'


def table(path, thr):
    X, y, g, R = [], [], [], []
    for l in open(path):
        rec = json.loads(l)
        for s, r in rec['sites'].items():
            X.append(feat(r)); y.append(C4.index(rec['labels'][thr][s])); g.append(rec['scene'])
            R.append(dict(scene=rec['scene'], site=s, clothed=rec['clothed'], r=r, labels=rec['labels'][thr]))
    return np.array(X), np.array(y), g, R


def summary(y, p, R=None):
    m = np.zeros((4, 4), int)
    for a, b in zip(y, p):
        m[a, b] += 1
    rec = [m[i, i] / m[i].sum() if m[i].sum() else None for i in range(4)]
    prec = [m[i, i] / m[:, i].sum() if m[:, i].sum() else 0 for i in range(4)]
    f1 = [2 * a * b / (a + b) if a and b and (a + b) else 0 for a, b in zip(prec, [r or 0 for r in rec])]
    out = {'n': int(m.sum()), 'accuracy': round(float(np.trace(m) / m.sum()), 4),
           'recall': dict(zip(C4, [None if v is None else round(float(v), 3) for v in rec])),
           'macro_f1_present_classes': round(float(np.mean([f for f, r in zip(f1, rec) if r is not None])), 3),
           'confusion_rows_truth': m.tolist()}
    if R is not None:     # laterality swaps: wrong site label that equals the true label of the contralateral site
        opp = {'LUE': 'RUE', 'RUE': 'LUE', 'LLE': 'RLE', 'RLE': 'LLE'}
        at = {(rr['scene'], rr['site']): i for i, rr in enumerate(R)}
        # a swap: the two sides of a pair carry different true labels and the predictions are exactly exchanged
        sw = 0
        for i, rr in enumerate(R):
            j = at.get((rr['scene'], opp[rr['site']]))
            if j is not None and y[i] != y[j] and p[i] == y[j] and p[j] == y[i]:
                sw += 1
        out['laterality_swaps'] = sw; out['laterality_swap_rate'] = round(sw / len(y), 4)
        for key in (True, False):
            idx = [i for i, rr in enumerate(R) if rr['clothed'] == key]
            if idx:
                out['clothed' if key else 'unclothed'] = {'n': len(idx), 'accuracy': round(float(np.mean([y[i] == p[i] for i in idx])), 4)}
    return out


def run_fit(dev, test, out, thr='0.10'):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GroupKFold
    Xd, yd, gd, _ = table(dev, thr); Xt, yt, _, Rt = table(test, thr)
    best = None
    for Cr in [0.03, 0.1, 0.3, 1, 3, 10]:
        acc = []
        for tr, va in GroupKFold(5).split(Xd, yd, gd):
            mdl = make_pipeline(StandardScaler(), LogisticRegression(C=Cr, max_iter=3000, class_weight='balanced')).fit(Xd[tr], yd[tr])
            acc.append((mdl.predict(Xd[va]) == yd[va]).mean())
        if best is None or np.mean(acc) > best[1]:
            best = (Cr, float(np.mean(acc)))
    mdl = make_pipeline(StandardScaler(), LogisticRegression(C=best[0], max_iter=3000, class_weight='balanced')).fit(Xd, yd)
    pt = mdl.predict(Xt)
    mu = make_pipeline(StandardScaler(), LogisticRegression(C=best[0], max_iter=3000)).fit(Xd, yd)
    res = {'threshold': thr, 'dev_sites': len(yd), 'C': best[0], 'dev_cv_accuracy': round(best[1], 4),
           'test_learned': summary(yt, pt, Rt), 'test_learned_unweighted': summary(yt, mu.predict(Xt), Rt), 'test_rule': summary(yt, [C4.index(rule(r['r'])) for r in Rt], Rt),
           'test_majority': summary(yt, [0] * len(yt), Rt)}
    sc_, lr = mdl.named_steps['standardscaler'], mdl.named_steps['logisticregression']
    res['model'] = {'classes': C4, 'mean': sc_.mean_.tolist(), 'scale': sc_.scale_.tolist(), 'coef': lr.coef_.tolist(),
                    'intercept': lr.intercept_.tolist(), 'features': FEATURES}
    json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'model'}, indent=1))


if __name__ == '__main__':
    if sys.argv[1] == 'feats':
        a = sys.argv[2:] + [''] * 6
        run_feats(a[0], a[1], a[2], a[3], a[4], a[5])
    else:
        run_fit(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) > 5 else '0.10')
