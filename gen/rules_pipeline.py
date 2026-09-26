"""Step 1: thin end-to-end rule pipeline (pose env).
Usage: python rules_pipeline.py <scene_dir> [<scene_dir> ...]

Arms scored against the 10%-threshold ground truth:
  majority  - every site predicted 'no_injury'
  oracle    - rules fed ground-truth keypoints (conf 1 if joint visible, else 0): ceiling of the rule logic
  rtmw      - rules fed RTMW whole-body keypoints (Human-Art YOLOX detector)
  mediapipe - rules fed MediaPipe Pose Landmarker (full) landmarks, visibility used as confidence

Rules per site (joints: proximal, middle, distal, extremity):
  detected(j)    = conf >= T_DET and inside the image
  not_testable   if no joint of the site is detected
  amputation     if a proximal joint is detected and the extremity is inside the image with conf < T_AMP
  no_injury      otherwise
Laterality is taken from the model's own left/right keypoint labels (no correction layer yet).
"""
import sys, os, json, glob
import numpy as np, cv2

SITES = ['LUE', 'RUE', 'LLE', 'RLE']
JN = {'UE': ['shoulder', 'elbow', 'wrist', 'hand'], 'LE': ['hip', 'knee', 'ankle', 'foot']}
# RTMW (COCO-WholeBody 133) indices per site joint; extremity uses middle-finger MCP / max of toes+heel
RT_IDX = {'LUE': [5, 7, 9, 100], 'RUE': [6, 8, 10, 121], 'LLE': [11, 13, 15, (17, 18, 19)], 'RLE': [12, 14, 16, (20, 21, 22)]}
# MediaPipe (33) indices; extremity = index fingertip / foot index
MP_IDX = {'LUE': [11, 13, 15, 19], 'RUE': [12, 14, 16, 20], 'LLE': [23, 25, 27, 31], 'RLE': [24, 26, 28, 32]}
W, H = 1280, 960
M = os.path.join(os.path.dirname(__file__), '..', 'models')


def run_models(D):
    """Run both pose models once per image and cache per-site joint (x, y, conf)."""
    from rtmlib import Wholebody
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    wb = Wholebody(det=os.path.join(M, '20230928/yolox_onnx/yolox_m_8xb8-300e_humanart-c2c7a14a/end2end.onnx'),
                   det_input_size=(640, 640), pose=os.path.join(M, 'end2end.onnx'), pose_input_size=(192, 256),
                   backend='onnxruntime', device='cpu')
    mpl = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(M, 'pose_landmarker_full.task')), num_poses=1))
    for sc in sorted(glob.glob(os.path.join(D, 'C*_sidecar.json'))):
        sid = json.load(open(sc))['scene_id']
        out = os.path.join(D, sid + '_poses.json')
        if os.path.exists(out):
            continue
        img = cv2.imread(os.path.join(D, sid + '.jpg'))
        k, s = wb(img)
        rt = None
        if len(k):
            i = int(np.argmax(s[:, :17].mean(1)))          # most confident person
            rt = {}
            for site, idx in RT_IDX.items():
                js = []
                for j in idx:
                    if isinstance(j, tuple):
                        b = max(j, key=lambda q: s[i][q])
                        js.append([float(k[i][b][0]), float(k[i][b][1]), float(s[i][b])])
                    else:
                        js.append([float(k[i][j][0]), float(k[i][j][1]), float(s[i][j])])
                rt[site] = js
        r = mpl.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        mpo = None
        if r.pose_landmarks:
            lm = r.pose_landmarks[0]
            mpo = {site: [[lm[j].x * W, lm[j].y * H, float(lm[j].visibility)] for j in idx] for site, idx in MP_IDX.items()}
        json.dump({'rtmw': rt, 'mediapipe': mpo}, open(out, 'w'))
    mpl.close()


def classify(site_joints, t_det, t_amp):
    """site_joints: list of 4 [x, y, conf] (proximal..extremity). Returns label."""
    def inside(j):
        return 0 <= j[0] < W and 0 <= j[1] < H
    det = [j[2] >= t_det and inside(j) for j in site_joints]
    if not any(det):
        return 'not_testable'
    ext = site_joints[3]
    if det[0] and inside(ext) and ext[2] < t_amp:
        return 'amputation'
    return 'no_injury'


def oracle_joints(gt, site):
    limb = site[1:]
    return [[gt['sites'][site][j]['x'], gt['sites'][site][j]['y'], 1.0 if gt['sites'][site][j]['visible'] else 0.0]
            for j in JN[limb]]


def predict_all(scenes, arm, t_det, t_amp):
    preds = {}
    for sid, d in scenes.items():
        if arm == 'majority':
            preds[sid] = {s: 'no_injury' for s in SITES}
        elif arm == 'oracle':
            preds[sid] = {s: classify(oracle_joints(d['gt'], s), 0.5, 0.5) for s in SITES}
        else:
            po = d['poses'][arm]
            preds[sid] = {s: ('not_testable' if po is None else classify(po[s], t_det, t_amp)) for s in SITES}
    return preds


def laterality_swap(d, arm, site):
    """True if the model's joints for this site sit closer to the GT joints of the opposite side."""
    po = d['poses'].get(arm)
    if po is None:
        return False
    opp = ('R' if site[0] == 'L' else 'L') + site[1:]
    own_gt, opp_gt = oracle_joints(d['gt'], site), oracle_joints(d['gt'], opp)
    do = do2 = n = 0
    for pj, g1, g2 in zip(po[site][:3], own_gt[:3], opp_gt[:3]):   # compare the three joints that exist most often
        if pj[2] < 0.3:
            continue
        do += np.hypot(pj[0] - g1[0], pj[1] - g1[1]); do2 += np.hypot(pj[0] - g2[0], pj[1] - g2[1]); n += 1
    return n > 0 and do2 < do * 0.6


def score(scenes, preds, arm):
    classes = ['no_injury', 'amputation', 'not_testable']
    cm = np.zeros((3, 3), int)
    stage = {'visibility': 0, 'state': 0, 'laterality_swap': 0}
    n = ok = 0
    for sid, d in scenes.items():
        gt = d['labels']
        for s in SITES:
            g, p = gt[s], preds[sid][s]
            cm[classes.index(g), classes.index(p)] += 1
            n += 1; ok += (g == p)
            if g != p:
                if arm in ('rtmw', 'mediapipe') and laterality_swap(d, arm, s):
                    stage['laterality_swap'] += 1
                elif (g == 'not_testable') != (p == 'not_testable'):
                    stage['visibility'] += 1
                else:
                    stage['state'] += 1
    return ok / n, n, cm, stage


def wilson(p, n, z=1.96):
    d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def load(dirs):
    scenes = {}
    for D in dirs:
        for sc in sorted(glob.glob(os.path.join(D, 'C*_sidecar.json'))):
            s = json.load(open(sc)); sid = s['scene_id']
            gp, pp = os.path.join(D, sid + '_gtkp.json'), os.path.join(D, sid + '_poses.json')
            if not (os.path.exists(gp) and os.path.exists(pp)):
                continue
            scenes[sid] = {'labels': s['labels_by_threshold']['0.10'], 'gt': json.load(open(gp)),
                           'poses': json.load(open(pp)), 'params': s['params'], 'dir': D}
    return scenes


T = {'rtmw': (0.3, 0.55), 'mediapipe': (0.5, 0.5)}
if __name__ == '__main__':
    dirs = sys.argv[1:]
    for D in dirs:
        run_models(D)
    scenes = load(dirs)
    res = {}
    for arm in ['majority', 'oracle', 'rtmw', 'mediapipe']:
        td, ta = T.get(arm, (0.5, 0.5))
        preds = predict_all(scenes, arm, td, ta)
        acc, n, cm, stage = score(scenes, preds, arm)
        lo, hi = wilson(acc, n)
        res[arm] = {'accuracy': round(acc, 4), 'ci95': [round(lo, 3), round(hi, 3)], 'n_sites': n,
                    'confusion_rows_gt_cols_pred[no_injury,amputation,not_testable]': cm.tolist(), 'errors_by_stage': stage}
        print(f"{arm:10s} acc {acc:.3f} (95% CI {lo:.2f}-{hi:.2f}, n={n})  errors {stage}")
        print('   confusion (rows GT no_inj/amp/not_test, cols pred):', cm.tolist())
        # ICD-format predictions per arm
        icd = {'schema_version': '1.0', 'submission': {'team_name': 'probeB-' + arm, 'version': 'step1', 'email': 'JL@josephleporini.com'},
               'predictions': [{'image_id': sid + '.jpg', 'sites': [
                   {'body_region': 'upper_extremity' if s[1] == 'U' else 'lower_extremity',
                    'laterality': 'left' if s[0] == 'L' else 'right', 'injury_type': preds[sid][s]} for s in SITES]}
                   for sid in sorted(preds)]}
        json.dump(icd, open(os.path.join(dirs[0], f'step1_predictions_{arm}.json'), 'w'), indent=1)
    json.dump(res, open(os.path.join(dirs[0], 'step1_scores.json'), 'w'), indent=1)
