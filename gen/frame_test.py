"""Body-frame test (pose env). Usage: python frame_test.py <scene_dir> [...]

1. Torso axis: RTMW shoulders-mid minus hips-mid vs ground truth (angular error; head-end correct if < 90 deg).
2. Facing (front toward camera): leave-one-scene-out logistic regression on RTMW features.
     face_only  : face keypoint confidence
     model_only : the chirality RTMW implies through its own left/right labels
     fused      : face + ears + nose/eyes + implied chirality (shoulders, hips) + foot direction in the torso frame
3. Laterality correction: reassign the arm pair and leg pair to anatomical sides using
     left_dir = facing * perp(axis); swap a pair's chains if RTMW's labels disagree.
   Scored with the Step-1 rules: accuracy and left/right swap rate, raw vs corrected, plus a ground-truth-frame ceiling.
"""
import sys, os, json, glob, collections
import numpy as np, cv2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.argv_keep = sys.argv[1:]
sys.argv = ['x']
exec(open(os.path.join(os.path.dirname(__file__), 'rules_pipeline.py')).read().split("T = {'rtmw'")[0])
dirs = sys.argv_keep
M = os.path.join(os.path.dirname(__file__), '..', 'models')


def full_cache(D):
    from rtmlib import Wholebody
    wb = None
    for sc in sorted(glob.glob(os.path.join(D, 'C*_sidecar.json'))):
        sid = json.load(open(sc))['scene_id']; out = os.path.join(D, sid + '_rtmw133.npz')
        if os.path.exists(out):
            continue
        if wb is None:
            wb = Wholebody(det=os.path.join(M, '20230928/yolox_onnx/yolox_m_8xb8-300e_humanart-c2c7a14a/end2end.onnx'),
                           det_input_size=(640, 640), pose=os.path.join(M, 'end2end.onnx'), pose_input_size=(192, 256),
                           backend='onnxruntime', device='cpu')
        k, s = wb(cv2.imread(os.path.join(D, sid + '.jpg')))
        i = int(np.argmax(s[:, :17].mean(1))) if len(k) else -1
        np.savez(out, k=k[i] if i >= 0 else np.zeros((133, 2)), s=s[i] if i >= 0 else np.zeros(133), found=i >= 0)


perp = lambda a: np.array([-a[1], a[0]])
unit = lambda v: v / (np.linalg.norm(v) + 1e-9)

for D in dirs:
    full_cache(D)
scenes = load(dirs)
rows = []
for sid, d in scenes.items():
    z = np.load(os.path.join(d['dir'], sid + '_rtmw133.npz')); k, s = z['k'], z['s']
    g = json.load(open(os.path.join(d['dir'], sid + '_gtframe.json')))
    sm, hm = (k[5] + k[6]) / 2, (k[11] + k[12]) / 2
    tl = np.linalg.norm(sm - hm) + 1e-9
    a = unit(sm - hm); pa = perp(a)
    ga = np.array(g['axis'])
    ang = float(np.degrees(np.arccos(np.clip(a @ ga, -1, 1))))
    m_sh = float((k[5] - k[6]) @ pa / tl); m_hp = float((k[11] - k[12]) @ pa / tl)
    feet = []
    for toe, heel in ((17, 19), (20, 22)):
        v = (k[toe] - k[heel]) / tl; feet += [float(v @ a), float(v @ pa)]
    feat = [float(s[23:91].mean()), float(s[0:3].mean()), float(s[3:5].mean()), m_sh, m_hp] + feet
    rows.append({'sid': sid, 'pos': d['params']['body_position'], 'axis_err': ang, 'a': a,
                 'facing': int(g['facing_cos'] > 0), 'fcos': g['facing_cos'], 'gt_left': np.array(g['left_dir']),
                 'feat': feat, 'k': k})
X = np.array([r['feat'] for r in rows]); y = np.array([r['facing'] for r in rows])
sets = {'face_only': [0], 'model_only': [3, 4], 'fused': list(range(X.shape[1]))}
pred = {}
for name, cols in sets.items():
    p = np.zeros(len(y))
    for i in range(len(y)):                                     # leave-one-scene-out
        tr = np.arange(len(y)) != i
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)).fit(X[tr][:, cols], y[tr])
        p[i] = clf.predict_proba(X[i:i + 1, cols])[0, 1]
    pred[name] = p
edge = np.array([abs(r['fcos']) < 0.3 for r in rows])
out = {'n_scenes': len(rows), 'edge_on_scenes': int(edge.sum())}
ae = np.array([r['axis_err'] for r in rows])
out['axis'] = {'median_err_deg': round(float(np.median(ae)), 1), 'head_end_correct': int((ae < 90).sum()),
               'within_30deg': int((ae < 30).sum())}
print('TORSO AXIS: median error %.1f deg, head end correct %d/%d, within 30 deg %d' %
      (np.median(ae), (ae < 90).sum(), len(ae), (ae < 30).sum()))
out['facing'] = {}
for name, p in pred.items():
    acc = ((p > 0.5) == y)
    bypos = collections.defaultdict(list)
    for r, a_ in zip(rows, acc):
        bypos[r['pos']].append(a_)
    out['facing'][name] = {'all': round(float(acc.mean()), 3), 'not_edge_on': round(float(acc[~edge].mean()), 3),
                           'edge_on': round(float(acc[edge].mean()), 3),
                           'by_position': {k_: round(float(np.mean(v)), 2) for k_, v in bypos.items()}}
    print(f'FACING {name:10s} all {acc.mean():.2f} | not edge-on {acc[~edge].mean():.2f} (n={(~edge).sum()}) | '
          f'edge-on {acc[edge].mean():.2f} (n={edge.sum()}) |', {k_: round(float(np.mean(v)), 2) for k_, v in bypos.items()})


def corrected(d, r, left_dir):
    po = json.loads(json.dumps(d['poses']['rtmw']))
    k = r['k']
    if po is None:
        return po, 0
    nsw = 0
    for (li, ri, A, B) in ((5, 6, 'LUE', 'RUE'), (11, 12, 'LLE', 'RLE')):
        if (k[li] - k[ri]) @ left_dir < 0:                   # model's "left" is on the figure's right side
            po[A], po[B] = po[B], po[A]; nsw += 1
    return po, nsw


out['correction'] = {}
for arm_name, lefts in [('raw', None), ('frame_fused', 'fused'), ('frame_face_only', 'face_only'), ('frame_ground_truth', 'gt')]:
    mod = {}
    for r in rows:
        d = scenes[r['sid']]
        if lefts is None:
            mod[r['sid']] = d['poses']['rtmw']
            continue
        if lefts == 'gt':
            ld = r['gt_left']
        else:
            f = 1 if pred[lefts][rows.index(r)] > 0.5 else -1
            ld = f * perp(r['a'])
        mod[r['sid']], _ = corrected(d, r, ld)
    tmp = {sid: dict(d, poses=dict(d['poses'], rtmw=mod[sid])) for sid, d in scenes.items()}
    preds = predict_all(tmp, 'rtmw', 0.3, 0.55)
    acc, n, cm, stage = score(tmp, preds, 'rtmw')
    swaps = harm = tested = 0
    for sid, d in tmp.items():
        for s_ in SITES:
            if d['labels'][s_] == 'not_testable':
                continue
            tested += 1
            if laterality_swap(d, 'rtmw', s_):
                swaps += 1; opp = ('R' if s_[0] == 'L' else 'L') + s_[1:]
                harm += d['labels'][s_] != d['labels'][opp]
    out['correction'][arm_name] = {'site_accuracy': round(acc, 4), 'swapped_sites': swaps, 'of_testable': tested,
                                   'label_changing_swaps': harm}
    print(f'CORRECTION {arm_name:18s} site acc {acc:.3f} | left/right swapped sites {swaps}/{tested} (label-changing {harm})')
json.dump(out, open(os.path.join(dirs[0], 'frame_test_results.json'), 'w'), indent=1)
