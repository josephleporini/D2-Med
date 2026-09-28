"""D2 Part 2 inference pipeline, single image in, four site predictions out (pose env or container).

Stages (as evaluated in Probe B Test F / Checkpoint 2, arm F2h + learned decision layer):
  0 letterbox the photo to 1280x960 (4:3, padding, no distortion)
  1 person box (YOLOX Human-Art) -> square crop -> SAM 2.1-tiny features -> v2 part decoder -> part map 640x480
  2 frame: RTMW torso axis + facing from the part map's front/back classes; side rule v2; limb masks
  3 limb evidence (visible pixels, length, end surroundings) and a limb-end window classifier
    (extremity visible / stump visible / end hidden)
  4 decision layer (multinomial logistic regression fitted on the development set)
Wound is not yet modelled (P2 in progress); the pipeline never outputs it.
"""
import os, sys, json, time
import numpy as np, cv2, torch
sys.path.insert(0, os.path.dirname(__file__))
import parts as PT, test_a_masks as TA, test_e as TE, test_e2 as T2, lend as LE
from seg_train2 import Decoder2
from seg_features import square_crop

SITE_ICD = {'LUE': ('upper_extremity', 'left'), 'RUE': ('upper_extremity', 'right'),
            'LLE': ('lower_extremity', 'left'), 'RLE': ('lower_extremity', 'right')}
RING = ['BG', 'OCC', 'TORSO', 'OTHER_LIMB', 'EDGE']


def frame_facing(seg, sfr):
    """Predicted facing from the part map (front vs back torso pixels), with the winning share as confidence.
    'unknown' when the frame could not be formed. Edge-on is not yet detected (M3-13 gap)."""
    if sfr is None:
        return None, None
    tot = np.isin(seg, T2.TORSO_C).sum(); fr = np.isin(seg, T2.FRONT_C).sum() / max(tot, 1)
    return ('front' if fr > 0.5 else 'back'), round(float(max(fr, 1 - fr)), 3)


def letterbox(img, W=1280, H=960):
    h, w = img.shape[:2]; s = min(W / w, H / h)
    r = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    out = np.zeros((H, W, 3), np.uint8); oy, ox = (H - r.shape[0]) // 2, (W - r.shape[1]) // 2
    out[oy:oy + r.shape[0], ox:ox + r.shape[1]] = r
    return out


class Pipeline:
    def __init__(self, model_dir, seg='seg2_n297', end='lend3b', decision='decision_F2h_lend3b', device=None):
        self.dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        from rtmlib import YOLOX, Wholebody
        prov = 'cuda' if self.dev == 'cuda' else 'cpu'
        det = os.path.join(model_dir, '20230928/yolox_onnx/yolox_m_8xb8-300e_humanart-c2c7a14a/end2end.onnx')
        self.det = YOLOX(onnx_model=det, model_input_size=(640, 640), backend='onnxruntime', device=prov)
        self.wb = Wholebody(det=det, det_input_size=(640, 640), pose=os.path.join(model_dir, 'end2end.onnx'),
                            pose_input_size=(192, 256), backend='onnxruntime', device=prov)
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.sam = SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',
                                                 os.path.join(model_dir, 'sam2_1_hiera_tiny.pt'), device=self.dev))
        self.seg = Decoder2(); self.seg.load_state_dict(torch.load(os.path.join(model_dir, seg + '.pt'), map_location='cpu'))
        self.seg.to(self.dev).eval()
        self.end = LE.EndNet(3); self.end.load_state_dict(torch.load(os.path.join(model_dir, end + '.pt'), map_location='cpu'))
        self.end.to(self.dev).eval()
        m = json.load(open(os.path.join(model_dir, decision + '.json')))['model']
        self.dl = {k: np.array(v) for k, v in m.items() if k in ('mean', 'scale', 'coef', 'intercept')}
        self.dl_classes = m['classes']

    # ---- stage 1 ----
    def part_map(self, img):
        b = self.det(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        box = tuple(float(v) for v in max(b, key=lambda q: (q[2] - q[0]) * (q[3] - q[1]))[:4]) if len(b) else (0, 0, 1280, 960)
        crop, (X1, Y1, S) = square_crop(img, box, 0.15)
        self.sam.set_image(cv2.resize(crop, (1024, 1024)))
        f = self.sam._features
        with torch.no_grad():
            lg = self.seg(f['image_embed'].float(), f['high_res_feats'][1].float(), f['high_res_feats'][0].float())
            s = max(int(round(S / 2)), 1)
            lab = torch.nn.functional.interpolate(lg, size=(s, s), mode='bilinear', align_corners=False)[0].argmax(0).cpu().numpy()
        seg = np.zeros((480, 640), np.uint8)
        ox, oy = int(round(X1 / 2)), int(round(Y1 / 2))
        ys0, xs0 = max(0, oy), max(0, ox); ys1, xs1 = min(480, oy + s), min(640, ox + s)
        if ys1 > ys0 and xs1 > xs0:
            seg[ys0:ys1, xs0:xs1] = lab[ys0 - oy:ys1 - oy, xs0 - ox:xs1 - ox]
        return seg

    # ---- stages 2-4 ----
    def predict(self, img_rgb):
        img = letterbox(img_rgb)
        seg = self.part_map(img)
        k, s = self.wb(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if len(k):
            i = int(np.argmax(s[:, :17].mean(1))); k = k[i]
            axis = TE.unit((k[5] + k[6]) / 2 - (k[11] + k[12]) / 2)
        else:
            axis = np.array([0.0, -1.0])
        sfr = T2.seg_frame(seg)
        facing, facing_conf = frame_facing(seg, sfr)
        probs = {}
        if sfr is None and not len(k):
            left = np.array([1.0, 0.0])
        else:
            left = (sfr[2] if sfr else 1) * TE.perp(axis)
        cm, names, ext, stc = T2.to_classmap(seg, left, axis); ix = {n: i for i, n in enumerate(names)}
        an = TA.analyse(cm, names); torso = cm == ix['TORSO']; s0 = LE.window_size(torso); te = max(TA.extent(torso), 1)
        out = {}
        for site in TE.SITES:
            r = an[site]; pv = None
            if r['vis_px'] >= 30:
                pm = cm == ix[site]; x, y = LE.limb_end(pm, torso)
                self.sam.set_image(cv2.resize(LE.crop_1280(img, x, y, s0), (1024, 1024)))
                e = torch.nn.functional.avg_pool2d(self.sam._features['image_embed'].float(), 2)
                xin = torch.cat([e, torch.from_numpy(LE.mask_window(pm, x, y, s0))[None, None].to(e.device)], 1)
                with torch.no_grad():
                    pv = torch.softmax(self.end(xin), 1)[0].cpu().numpy().tolist()
            end = r.get('end') or {}; tot = sum(end.values()) or 1
            f = [np.log1p(r['vis_px']), r['Lfrac'], r['bg_frac'] if r['bg_frac'] is not None else -1, *(pv or [0, 0, 0]),
                 np.log1p(ext[site]), np.log1p(stc[site]), *[end.get(q, 0) / tot for q in RING],
                 r['vis_px'] / te ** 2, float(pv is not None)]
            z = (np.array(f) - self.dl['mean']) / self.dl['scale']
            logits = self.dl['coef'] @ z + self.dl['intercept']
            out[site] = self.dl_classes[int(np.argmax(logits))]
            q = np.exp(logits - logits.max()); q = q / q.sum()                 # decision-layer posterior
            p4 = [0.0] * 4
            for c, v in zip(self.dl_classes, q):
                import events as EV4          # lazy: d2voice is needed only for event output
                p4[EV4.C4.index(c)] += float(v)
            probs[site] = p4
        self.last = {'probs': probs, 'facing': facing, 'facing_conf': facing_conf}
        return out


FALLBACK = 'no_injury'          # majority class (M13-09); moves to model_config.json with the container build


def run_folder(pipe, in_dir, team='TBD-team', version='0.1.0', email='jl@josephleporini.com', events_out=None,
               model_version='probeB-d2pipe'):
    import events as EV4                 # lazy: d2voice is needed only for event output
    """predictions.json is generated from the casualty events (D-01: M12 reads events). events_out, if given, receives
    one schema v0.2.0 event log per image as JSON lines. It must not be the qualification output directory, which
    holds only predictions.json (M2-02); the qualification run keeps events in memory."""
    preds = []
    fe = open(events_out, 'w') if events_out else None
    for fn in sorted(os.listdir(in_dir)):
        if not fn.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img = cv2.imread(os.path.join(in_dir, fn))
        fb = [0.0] * 4; fb[EV4.C4.index(FALLBACK)] = 1.0
        probs, facing, fconf, mv = {s: fb for s in TE.SITES}, None, None, model_version + ':fallback'
        try:
            if img is not None:
                pipe.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                probs, facing, fconf, mv = pipe.last['probs'], pipe.last['facing'], pipe.last['facing_conf'], model_version
        except Exception as ex:                      # never drop an image: fall back to the majority class
            print('WARN', fn, repr(ex), flush=True)
        log = EV4.image_log(fn, probs, model_version=mv, facing=facing, facing_conf=fconf)
        preds.append(EV4.qual_record(fn, log))
        if fe:
            fe.write(json.dumps(log, separators=(',', ':')) + '\n')
    if fe:
        fe.close()
    return {'schema_version': '1.0', 'submission': {'team_name': team, 'version': version, 'email': email},
            'predictions': preds}


if __name__ == '__main__':
    in_dir, out_path = sys.argv[1], sys.argv[2]
    model_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(__file__), '..', 'models')
    t0 = time.time(); pipe = Pipeline(model_dir)
    res = run_folder(pipe, in_dir, events_out=os.environ.get('EVENTS_OUT'))
    json.dump(res, open(out_path, 'w'), indent=1)
    print('PRED', len(res['predictions']), 'images', round(time.time() - t0, 1), 's')
