"""Left/right and facing diagnostics against rendered truth, plus a perfect-side ceiling.
usage: python checks.py <scene_dir> <split> <shard k> <n shards> <out.jsonl>
Per image: facing truth (TORSO_F share of the rendered torso) vs segmentation facing, RTMW shoulder/hip order
and face-keypoint confidence; RTMW limb-keypoint side labels vs rendered side; foot chirality sign; pixel side
accuracy of the current rule and the keypoint rule; features with the rendered side (ceiling), gt and pred maps.
"""
import os, sys, json, glob, time
N_THR = 6
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[k] = str(N_THR)
import onnxruntime as ort
_IS = ort.InferenceSession
class _Capped(_IS):
    def __init__(self, *a, **kw):
        so = kw.get('sess_options') or ort.SessionOptions(); so.intra_op_num_threads = N_THR; so.inter_op_num_threads = 1
        kw['sess_options'] = so; super().__init__(*a, **kw)
ort.InferenceSession = _Capped
import numpy as np, cv2, torch
cv2.setNumThreads(N_THR)
from scipy import ndimage, sparse
from scipy.sparse.csgraph import dijkstra
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_v3 as EV, test_a_masks as TA, test_e as TE, test_e2 as T2, lend as LE, parts as PT
from lwound_cache import limb_window
from d2pipe import letterbox
from side_kp import geodesic_fast, to_classmap_kp, site_rows
TA.geodesic = geodesic_fast
torch.set_num_threads(N_THR)
K = T2.K
LIMB_KP = {7: 1, 9: 1, 13: 1, 15: 1, 8: 2, 10: 2, 14: 2, 16: 2}          # RTMW label: 1 left, 2 right
FEET = {'L': (19, 17, 18), 'R': (22, 20, 21)}                             # heel, big toe, small toe


def side_lookup(side):
    """nearest rendered side (1/2) for every pixel, and distance to it"""
    d, (iy, ix) = ndimage.distance_transform_edt(side == 0, return_indices=True)
    return side[iy, ix], d


def assign_gt_side(cm, names, seg, ext, stc, near, dist):
    ix = {n: i for i, n in enumerate(names)}; cm = cm.copy(); ext = dict(ext); stc = dict(stc)
    stump = seg == K['stump']
    for g in T2.GROUP:
        gm = (cm == ix['L' + g]) | (cm == ix['R' + g])
        ok = gm & (dist <= 8)
        cm[ok & (near == 1)] = ix['L' + g]; cm[ok & (near == 2)] = ix['R' + g]
        for sd in 'LR':
            mm = cm == ix[sd + g]
            ext[sd + g] = int((mm & (seg == T2.EXT[g])).sum()); stc[sd + g] = int((mm & stump).sum())
    return cm, ext, stc


def pix_acc(cm, names, near, dist):
    ix = {n: i for i, n in enumerate(names)}
    lab = np.zeros(cm.shape, np.uint8)
    for g in ('UE', 'LE'):
        lab[cm == ix['L' + g]] = 1; lab[cm == ix['R' + g]] = 2
    m = (lab > 0) & (dist <= 8)
    return int(m.sum()), int((lab[m] == near[m]).sum())


if __name__ == '__main__':
    D, split, kk, nn, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    M = LE.M
    m = EV.load_models(os.path.join(M, 'seg3_e2e_b.pt'), os.path.join(M, 'lend3d.pt'), os.path.join(M, 'lwound2.pt'))
    files = sorted(glob.glob(os.path.join(D, 'C*_sidecar.json')))[kk::nn]
    fo = open(out, 'w'); t0 = time.time()
    for n, f in enumerate(files):
        sid = os.path.basename(f)[:-len('_sidecar.json')]; sc = json.load(open(f)); prm = sc['params']
        img = letterbox(cv2.cvtColor(cv2.imread(os.path.join(D, sid + '.jpg')), cv2.COLOR_BGR2RGB))
        g3, gside, _ = PT.decode3(os.path.join(D, sid + '_part3.png'))
        g3 = cv2.resize(g3, (640, 480), interpolation=cv2.INTER_NEAREST)
        gside = cv2.resize(gside, (640, 480), interpolation=cv2.INTER_NEAREST)
        near, dist = side_lookup(gside)
        tf, tb = int((g3 == K['TORSO_F']).sum()), int((g3 == K['TORSO_B']).sum())
        facing_true = 1 if tf > tb else -1
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        boxes = m['det'](bgr)
        k, s = m['wb'].pose_model(bgr, bboxes=boxes)
        rec = dict(scene=sid, split=split, position=prm.get('body_position'), yaw=prm.get('body_yaw'),
                   facing_true=facing_true, torso_front_share=round(tf / max(tf + tb, 1), 3),
                   labels=sc['labels_by_threshold'], labels_wip=sc['labels_wound_if_present'],
                   visible_fraction=sc.get('visible_fraction'), clothed=bool(prm.get('garments', {}).get('top', 'none') != 'none'
                                                                            or prm.get('garments', {}).get('bottom', 'none') != 'none'))
        kxy = None
        if len(k):
            i = int(np.argmax(s[:, :17].mean(1))); kp, ks = k[i], s[i]; kxy = kp[:17]
            axis = TE.unit((kp[5] + kp[6]) / 2 - (kp[11] + kp[12]) / 2); pp = TE.perp(axis)
            rec['rtmw_facing'] = int(np.sign(((kp[5] - kp[6]) + (kp[11] - kp[12])) @ pp) or 1)
            rec['face_score'] = round(float(ks[23:91].mean()), 3); rec['nose_score'] = round(float(ks[0]), 3)
            rec['eye_ear_score'] = [round(float(v), 3) for v in ks[1:5]]
            hits = []
            for j, lab in LIMB_KP.items():
                x, y = int(round(kp[j][0] / 2)), int(round(kp[j][1] / 2))
                if 0 <= x < 640 and 0 <= y < 480 and ks[j] > 0.3 and dist[y, x] <= 6:
                    hits.append([j, lab, int(near[y, x]), round(float(ks[j]), 3)])
            rec['kp_side'] = hits
            feet = {}
            for sd, (h, b, sm) in FEET.items():
                if min(ks[h], ks[b], ks[sm]) > 0.3:
                    toe = (kp[b] + kp[sm]) / 2; v1, v2 = toe - kp[h], kp[b] - kp[sm]
                    x, y = int(round(kp[h][0] / 2)), int(round(kp[h][1] / 2))
                    tside = int(near[y, x]) if (0 <= x < 640 and 0 <= y < 480 and dist[y, x] <= 8) else 0
                    feet[sd] = dict(chir=int(np.sign(v1[0] * v2[1] - v1[1] * v2[0])), true_side=tside,
                                    score=round(float(min(ks[h], ks[b], ks[sm])), 3))
            rec['feet'] = feet
        else:
            axis = np.array([0.0, -1.0]); pp = TE.perp(axis)
        seg3p = EV.pred_part_map(m, img)
        for arm, seg3 in (('gt', g3), ('pred', seg3p)):
            seg, wm, tq = EV.fold_extras(seg3)
            sfr = T2.seg_frame(seg)
            left = np.array([1.0, 0.0]) if (sfr is None and not len(k)) else (sfr[2] if sfr else 1) * pp
            rec[f'{arm}_seg_facing'] = None if sfr is None else int(sfr[2])
            cm0, names, e0, s0 = T2.to_classmap(seg, left, axis)
            cm1, _, e1, s1, _ = to_classmap_kp(seg, left, axis, None if kxy is None else kxy.tolist())
            rec[f'{arm}_pix_old'] = pix_acc(cm0, names, near, dist); rec[f'{arm}_pix_kp'] = pix_acc(cm1, names, near, dist)
            cm2, e2, s2 = assign_gt_side(cm0, names, seg, e0, s0, near, dist)
            rec[f'{arm}_sites_ceil'] = site_rows(m, img, cm2, names, e2, s2, wm, tq)
        fo.write(json.dumps(rec) + '\n'); fo.flush()
        if (n + 1) % 20 == 0:
            print('CHK', split, kk, n + 1, round((time.time() - t0) / (n + 1), 2), flush=True)
    print('CHK_DONE', split, kk, len(files), round(time.time() - t0, 1), flush=True)
