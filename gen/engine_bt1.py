"""BT-1 inference engine: one image in, four site class probabilities out. No generator truth anywhere.

This is the 'sidec' path of jobs/sidehead2.py extract (LIMB=1, SIDE_MODE=distill) with everything that reads a sidecar,
a part map or a rendered side removed, plus the exported decision layer (tools/fit_decision_bt1.py) and the test-time
mirror (D-04, M3-02): the mirrored image goes through the same path, its site keys are swapped back, and the two
probability sets are averaged. The dev5 extraction rows in DDData results/bt1/ext are the parity reference
(jobs/bt3_a40.sh stage R).

Engine contract (d2-blockT d2qual/engines/structured.py):
    E = BT1Engine(models_dir, ckpt, decision_json, tta=True)
    probs, info = E.predict_image(rgb_uint8)       # probs (4 sites LUE RUE LLE RLE, 4 classes), info for the run log

Env: CAP_THREADS (torch/OpenCV/ORT threads, default 8, the APL CPU cap), ORT_GPU=1 (det/pose on CUDA if onnxruntime-gpu
is installed; default CPU, which is how the reference rows were made).
BT-3 speed switches (M3-08), each measured for parity in jobs/bt3_a40.sh before it becomes a default:
  BT1_FAST=1 (default)   exact CPU speed-ups, gen/fastops.py (bit-identical; tests/test_fastops.py)
  BT1_BATCH=1 (default)  per-limb SAM encodings in one batch (parity held, BT-3 speed note 28 Sep)
  BT1_AMP=1              bfloat16 autocast for the SAM encoders on CUDA (not exact: parity measured)
  BT1_POSE_REUSE=1       the mirrored pass reuses the original pass's person box and keypoints, mirrored, instead of
                         running detection and pose again (not exact: parity measured)
"""
import os, sys, json, time
N_THR = int(os.environ.get('CAP_THREADS', '8'))
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_k, str(N_THR))
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
SITES = ['LUE', 'RUE', 'LLE', 'RLE']
SWAP = {'LUE': 'RUE', 'RUE': 'LUE', 'LLE': 'RLE', 'RLE': 'LLE'}


class DecisionLayer:
    """StandardScaler + multinomial logistic regression, evaluated from exported numbers (no sklearn at run time)."""

    def __init__(self, path):
        d = json.load(open(path))
        self.doc = d
        self.mean, self.scale = np.array(d['mean']), np.array(d['scale'])
        self.W, self.b = np.array(d['coef']), np.array(d['intercept'])

    def proba(self, X):
        z = (np.asarray(X, float) - self.mean) / self.scale @ self.W.T + self.b
        z = z - z.max(1, keepdims=True); e = np.exp(z)
        return e / e.sum(1, keepdims=True)


def features(row):
    import eval_v3 as EV
    return EV.feat(row) + EV.limb_feat(row)


def site_rows_batched(m, img, cm, names, ext, stc, wm, tq, chunk=8):
    """side_kp.site_rows with the SAM 2.1 crop encodings done in one batch (BT-3 speed). Same crops, same EndNet heads;
    only the encoder call is batched (set_image_batch), so outputs match the serial path up to float rounding."""
    import torch, test_a_masks as TA, test_e as TE, lend as LE
    from side_kp import limb_window
    ix = {n: i for i, n in enumerate(names)}
    an = TA.analyse(cm, names); torso = cm == ix['TORSO']; s0 = LE.window_size(torso); te = max(TA.extent(torso), 1)
    jobs, crops = [], []
    for site in TE.SITES:
        pm = cm == ix[site]
        if an[site]['vis_px'] >= 30:
            x, y = LE.limb_end(pm, torso); cx, cy, sw = limb_window(pm)
            jobs.append((site, pm, x, y, cx, cy, sw))
            crops += [cv2_resize(LE.crop_1280(img, x, y, s0)), cv2_resize(LE.crop_1280(img, cx, cy, sw))]
    emb = []
    for k in range(0, len(crops), chunk):
        with _amp():
            m['P'].set_image_batch(crops[k:k + chunk])
        e = m['P']._features['image_embed']
        emb += list(torch.nn.functional.avg_pool2d(e.float(), 2).detach().cpu().numpy().astype(np.float16))
    pe, pw = {}, {}
    for j, (site, pm, x, y, cx, cy, sw) in enumerate(jobs):
        fe = torch.from_numpy(emb[2 * j]).float()[None]; fw = torch.from_numpy(emb[2 * j + 1]).float()[None]
        xin = torch.cat([fe, torch.from_numpy(LE.mask_window(pm, x, y, s0))[None, None]], 1).to(m['dev'])
        xw = torch.cat([fw, torch.from_numpy(LE.mask_window(pm, cx, cy, sw))[None, None]], 1).to(m['dev'])
        with torch.no_grad():
            pe[site] = torch.softmax(m['end'](xin), 1)[0].cpu().numpy().tolist()
            pw[site] = torch.softmax(m['lw'](xw), 1)[0].cpu().numpy().tolist()
    rows = {}
    for site in TE.SITES:
        r = an[site]; pm = cm == ix[site]; grown = ndimage.binary_dilation(pm, iterations=2)
        rows[site] = dict(vis_px=int(r['vis_px']), Lfrac=r['Lfrac'], bg_frac=r['bg_frac'], end=r.get('end') or {},
                          ext_px=int(ext[site]), stump_px=int(stc[site]), torso_ext=float(te), p_end=pe.get(site), p_wound=pw.get(site),
                          wound_px=int((wm & grown).sum()), tq_px=int((tq & grown).sum()))
    return rows


def cv2_resize(crop):
    import cv2
    return cv2.resize(crop, (1024, 1024))


BATCH = os.environ.get('BT1_BATCH', '1') == '1'     # batched SAM encoding of the per-limb crops (BT-3 speed)
FAST = os.environ.get('BT1_FAST', '1') == '1'
AMP = os.environ.get('BT1_AMP', '0') == '1'
POSE_REUSE = os.environ.get('BT1_POSE_REUSE', '0') == '1'
COCO_FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


PICK = os.environ.get('BT1_PICK', 'largest')      # 'lying': choose the casualty among several people (BT-2b)


def pick_casualty(boxes, k, s, shape):
    """Index of the person most likely to be the casualty when several are detected (medics, bystanders).
    Real training photos show 3 people per photo on median (Real Fidelity Gap v1.0); the engine used to take the
    largest box. Score, all terms mirror-invariant so both passes pick the same person:
      size (box area over the largest), centrality, leg extension (ankle-hip over hip-shoulder distance: lying or
      standing about 1.5 to 2, kneeling or crouching under 1), pose confidence, minus uprightness (torso pointing up
      the image, as kneeling and standing medics do in oblique photos).
    Weights are set by hand, not fitted; validated on generator v3 scenes with bystanders."""
    H, W = shape[:2]
    b = np.asarray(boxes, float)[:, :4]
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]); area = area / max(area.max(), 1e-6)
    cx, cy = (b[:, 0] + b[:, 2]) / 2 / W - 0.5, (b[:, 1] + b[:, 3]) / 2 / H - 0.5
    cen = 1 - np.clip(np.sqrt(cx ** 2 + cy ** 2) / 0.7071, 0, 1)
    k = np.asarray(k, float); s = np.asarray(s, float)
    sh, hp, an = k[:, 5:7].mean(1), k[:, 11:13].mean(1), k[:, 15:17].mean(1)
    torso = np.linalg.norm(sh - hp, axis=1) + 1e-6
    ext = np.clip(np.linalg.norm(an - hp, axis=1) / torso / 1.5, 0, 1)
    up = np.clip(-(sh - hp)[:, 1] / torso, 0, 1)
    conf = s[:, :17].mean(1)
    score = area + 0.7 * cen + 0.8 * ext + 0.3 * conf - 0.3 * up
    return int(np.argmax(score))


def _amp():
    import torch, contextlib
    if AMP and torch.cuda.is_available():
        return torch.autocast('cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()


def mirror_det_pose(dp, W):
    """(boxes, keypoints, scores) of an image -> the same for the image mirrored left to right (x -> W - 1 - x).
    Only the 17 body keypoints are kept (the engine uses no others), with left and right swapped."""
    boxes, k, s = dp
    b = np.array(boxes, float) if len(boxes) else np.zeros((0, 4))
    if len(b):
        b = b.copy(); b[:, [0, 2]] = (W - 1) - b[:, [2, 0]]
    if len(k):
        k = np.asarray(k)[:, :17][:, COCO_FLIP].copy(); k[..., 0] = (W - 1) - k[..., 0]
        s = np.asarray(s)[:, :17][:, COCO_FLIP]
    return b, k, s


class BT1Engine:
    name = 'bt1'

    def __init__(self, models_dir, ckpt, decision_json, tta=True, device=None):
        sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'jobs'))
        os.environ.setdefault('SIDE_MODE', 'distill')
        import torch, cv2
        cv2.setNumThreads(N_THR); torch.set_num_threads(N_THR)
        import onnxruntime as ort
        if os.environ.get('ORT_GPU') != '1' and not getattr(ort.InferenceSession, '_d2_cpu', False):
            _IS = ort.InferenceSession

            class _C(_IS):
                _d2_cpu = True

                def __init__(self, *a, **kw):
                    so = kw.get('sess_options') or ort.SessionOptions(); so.intra_op_num_threads = N_THR; so.inter_op_num_threads = 1
                    kw['sess_options'] = so; kw['providers'] = ['CPUExecutionProvider']; super().__init__(*a, **kw)
            ort.InferenceSession = _C
        import lend as LE
        LE.M = models_dir
        import eval_v3 as EV, test_a_masks as TA
        from side_kp import geodesic_fast
        TA.geodesic = geodesic_fast
        if FAST:
            import fastops
            fastops.apply()
            globals()['ndimage'] = fastops.ND
        if os.environ.get('ORT_GPU') == '1' and hasattr(ort, 'preload_dlls'):
            ort.preload_dlls()                          # CUDA and cuDNN from the torch wheels (onnxruntime-gpu >= 1.21)
        import bt1_limb as BL, parts as PT
        self.torch, self.cv2, self.EV, self.BL, self.PT = torch, cv2, EV, BL, PT
        self.m = EV.load_models(None, os.path.join(models_dir, 'lend3d.pt'), os.path.join(models_dir, 'lwound2.pt'))
        self.dev = self.m['dev'] if device is None else device
        net = BL.LimbModel(len(PT.SEG3_CLASSES), os.path.join(models_dir, 'sam2_1_hiera_tiny.pt'))
        net.load_state_dict(torch.load(ckpt, map_location='cpu')); net.to(self.dev).eval()
        self.net, self.tta = net, tta
        self.layer = DecisionLayer(decision_json)

    def rows(self, img, flip=False, det_pose=None):
        """img: 1280x960 letterboxed RGB uint8. Returns (site rows keyed by anatomical site, frame dict, timings).
        det_pose: optional (boxes, keypoints, scores) already in this image's coordinates (BT1_POSE_REUSE)."""
        torch, cv2, EV, BL = self.torch, self.cv2, self.EV, self.BL
        import torch.nn.functional as F
        import test_e as TE, test_e2 as T2
        from d2pipe import frame_facing
        from side_kp import to_classmap_kp, site_rows
        from checks import assign_gt_side
        from seg_features import square_crop
        from seg_train_e2e import MEAN, STD
        t = {}; t0 = time.time()
        if flip:
            img = np.ascontiguousarray(img[:, ::-1])
        if det_pose is None:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            boxes = self.m['det'](bgr)
            k, s = self.m['wb'].pose_model(bgr, bboxes=boxes)
        else:
            boxes, k, s = det_pose
        self._last_dp = (boxes, k, s)
        kxy = None
        pick = pick_casualty(boxes, k, s, img.shape) if PICK == 'lying' and len(boxes) > 1 and len(k) == len(boxes) else None
        if pick is not None:                          # one person for both the pose frame and the crop
            boxes = [boxes[pick]]; k = np.asarray(k)[pick:pick + 1]; s = np.asarray(s)[pick:pick + 1]
        if len(k):
            i = int(np.argmax(s[:, :17].mean(1))); kp = k[i]; kxy = kp[:17]
            axis = TE.unit((kp[5] + kp[6]) / 2 - (kp[11] + kp[12]) / 2); pp = TE.perp(axis)
        else:
            axis = np.array([0.0, -1.0]); pp = TE.perp(axis)
        t['det_pose'] = time.time() - t0; t1 = time.time()
        box = tuple(float(v) for v in max(boxes, key=lambda q: (q[2] - q[0]) * (q[3] - q[1]))[:4]) if len(boxes) else (0, 0, 1280, 960)
        crop, (X1, Y1, S) = square_crop(img, box, 0.15)
        x = (torch.from_numpy(cv2.resize(crop, (1024, 1024))).permute(2, 0, 1).float() / 255 - MEAN) / STD
        with torch.no_grad(), _amp():
            outs = self.net(x[None].to(self.dev))
        outs = [o.float() for o in outs]; ps, pd = outs[0], outs[1]
        with torch.no_grad():
            s2 = max(int(round(S / 2)), 1)
            lab = F.interpolate(ps.float(), size=(s2, s2), mode='bilinear', align_corners=False)[0].argmax(0).cpu().numpy()
            pl = F.interpolate(pd.float(), size=(s2, s2), mode='bilinear', align_corners=False)[0].softmax(0)[0].cpu().numpy()
        seg3 = np.zeros((480, 640), np.uint8); pleft = np.full((480, 640), 0.5, np.float32)
        ox, oy = int(round(X1 / 2)), int(round(Y1 / 2))
        ys0, xs0 = max(0, oy), max(0, ox); ys1, xs1 = min(480, oy + s2), min(640, ox + s2)
        if ys1 > ys0 and xs1 > xs0:
            seg3[ys0:ys1, xs0:xs1] = lab[ys0 - oy:ys1 - oy, xs0 - ox:xs1 - ox]
            pleft[ys0:ys1, xs0:xs1] = pl[ys0 - oy:ys1 - oy, xs0 - ox:xs1 - ox]
        t['net'] = time.time() - t1; t2 = time.time()
        seg, wm, tq = EV.fold_extras(seg3)
        sfr = T2.seg_frame(seg)
        fac, fconf = frame_facing(seg, sfr)
        left = np.array([1.0, 0.0]) if (sfr is None and not len(k)) else (sfr[2] if sfr else 1) * pp
        cm1, names, e1, s1, _ = to_classmap_kp(seg, left, axis, None if kxy is None else kxy.tolist())
        pside = np.where(pleft >= 0.5, 1, 2).astype(np.uint8)
        conf = np.where(np.maximum(pleft, 1 - pleft) >= 0.75, 0.0, 99.0).astype(np.float32)
        cmC, eC, sC = assign_gt_side(cm1, names, seg, e1, s1, pside, conf)
        t['assign'] = time.time() - t2; t3 = time.time()
        rows = (site_rows_batched if BATCH else site_rows)(self.m, img, cmC, names, eC, sC, wm, tq)
        lf = BL.limb_site_features(outs[2][0], outs[3][0], (X1, Y1, S), {k_: r_['vis_px'] for k_, r_ in rows.items()})
        for k_ in rows:
            rows[k_]['limb'] = lf[k_]
        t['site_rows'] = time.time() - t3
        if flip:
            rows = {SWAP[k_]: v_ for k_, v_ in rows.items()}
        return rows, dict(facing=fac or 'unknown', facing_conf=fconf), t

    def predict_image(self, rgb):
        """rgb: any-size RGB uint8 (as decoded). Returns (probs (4, 4) in SITES x classes order, info)."""
        from d2pipe import letterbox
        img = letterbox(np.ascontiguousarray(rgb))
        ro, fo, to = self.rows(img, False)
        dp = mirror_det_pose(self._last_dp, img.shape[1]) if POSE_REUSE else None
        P = self.layer.proba([features(ro[s]) for s in SITES])
        info = dict(frame=fo, t=to, rows=ro)
        if self.tta:
            rf, ff, tf = self.rows(img, True, dp)
            PF = self.layer.proba([features(rf[s]) for s in SITES])
            info.update(rows_flip=rf, t_flip=tf, p_orig=P.round(4).tolist(), p_flip=PF.round(4).tolist())
            P = 0.5 * (P + PF)
        return P, info
