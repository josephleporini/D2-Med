"""Blanket relief oracle, stage 1 analysis (offline, CPU). Does surface relief of a blanket (a surface model in the
remote-sensing sense: floor = terrain, blanket = canopy) carry information about the limb hidden under it that the
BT-1 model does not already have?

usage: python tools/relief_oracle.py <dddata_root> <out.json>
  uses  results/relief_oracle/*_relief.npz   (ray-cast height, depth, normal, hit class; gen/relief_pass.py)
        dev5/batch_*/D*_{occ.png,sidecar.json} (visible-surface owner map, raw truth)
        results/bt1/ext/bt1_t*_sidec.jsonl    (BT-1 extraction rows: the model's own features)

Oracle inputs, stated plainly: the visible limb masks come from the rendered owner map (truth), and the blanket surface
comes from exact ray-cast height. Stage 1 asks whether the signal exists at all. A predicted surface is stage 2.

Per blanket scene (remote-sensing analogues in brackets):
  canopy height over terrain       h = blanket surface height above the floor plane            [canopy height model]
  something under the canopy       U = blanket pixels with h > LIFT + 0.02 m                    [CHM threshold]
  ridge                            R = U and (h - smooth(h)) > 5 mm, smooth = masked Gaussian    [local relief model]
  catchment per limb               each U pixel is assigned to the nearest visible limb or torso
                                   by geodesic distance through U (multi-source BFS)              [watershed catchment]
Per site features (0 when the scene has no blanket):
  control (no relief): blanket share of a 5 px ring around the visible limb; blanket present in the scene
  relief:  catchment area in U and in R, within 60 and 150 px geodesic; est_vf = vis / (vis + catchment R 150)

Arms, all with the BT-1 features (eval_v3.feat + limb_feat), grouped 5-fold out-of-fold on dev5 (1,920 sites):
  base, base + control, base + control + relief. The relief arm must beat the control arm to count.
"""
import sys, os, json, glob, collections
from collections import deque
import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'gen')); sys.path.insert(0, os.path.join(HERE, '..', 'score'))
import eval_v3 as EV
import labels as LB
from score import fitC, mk, proba, metrics, C4

SITES = ['LUE', 'RUE', 'LLE', 'RLE']
CODE = {'LUE': 4, 'RUE': 5, 'LLE': 6, 'RLE': 7}
LIFT = 0.025                                   # scene.drape_blanket lift above the surface
CONTROL = ['blanket_ring_share', 'blanket_in_scene']
RELIEF = ['log_catch_u60', 'log_catch_u150', 'log_catch_r60', 'log_catch_r150', 'est_vf_relief']


def masked_blur(h, m, s):
    a = ndimage.gaussian_filter(np.where(m, h, 0.0), s); b = ndimage.gaussian_filter(m.astype(float), s)
    return np.where(b > 1e-3, a / np.maximum(b, 1e-3), 0.0)


def geodesic_catchment(U, seeds):
    """multi-source BFS through U from labelled seed pixels (label > 0); returns (label, distance) per pixel"""
    H, W = U.shape; lab = np.zeros((H, W), np.int16); dist = np.full((H, W), -1, np.int32); q = deque()
    for y, x in zip(*np.nonzero(seeds)):
        lab[y, x] = seeds[y, x]; dist[y, x] = 0; q.append((y, x))
    while q:
        y, x = q.popleft(); d = dist[y, x] + 1
        for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= yy < H and 0 <= xx < W and dist[yy, xx] < 0 and U[yy, xx]:
                dist[yy, xx] = d; lab[yy, xx] = lab[y, x]; q.append((yy, xx))
    return lab, dist


def scene_features(npz, occ):
    h = npz['height'].astype(np.float32); B = npz['hit'] == 2
    out = {s: dict.fromkeys(CONTROL + RELIEF, 0.0) for s in SITES}
    if not B.any():
        return out, {}
    U = B & (h > LIFT + 0.02)
    R = U & ((h - masked_blur(h, B, 15)) > 0.005)
    seeds = np.zeros(occ.shape, np.int16)
    for k, s in enumerate(SITES):                       # seeds: visible limb pixels bordering the blanket, and torso
        seeds[(occ == CODE[s]) & ndimage.binary_dilation(B, iterations=3)] = k + 1
    seeds[(occ == 3) & ndimage.binary_dilation(B, iterations=3)] = 9
    lab, dist = geodesic_catchment(U | (seeds > 0), seeds)
    for k, s in enumerate(SITES):
        V = occ == CODE[s]; vis = int(V.sum())
        ring = ndimage.binary_dilation(V, iterations=5) & ~V
        o = out[s]
        o['blanket_ring_share'] = float(B[ring].mean()) if ring.any() else 0.0
        o['blanket_in_scene'] = 1.0
        mine = (lab == k + 1) & U
        cu60, cu150 = int((mine & (dist <= 60)).sum()), int((mine & (dist <= 150)).sum())
        cr60, cr150 = int((mine & R & (dist <= 60)).sum()), int((mine & R & (dist <= 150)).sum())
        o.update(log_catch_u60=np.log1p(cu60), log_catch_u150=np.log1p(cu150), log_catch_r60=np.log1p(cr60),
                 log_catch_r150=np.log1p(cr150), est_vf_relief=vis / (vis + cr150) if vis + cr150 else 0.0)
    # diagnostics: does relief find hidden limb pixels? (uses amodal truth only for scoring the signal)
    return out, {'U': U, 'R': R, 'B': B}


def main(root, out_path):
    npz = {os.path.basename(f)[:5]: f for f in glob.glob(os.path.join(root, 'results/relief_oracle/*_relief.npz'))}
    cars, occs, amod = {}, {}, {}
    for f in glob.glob(os.path.join(root, 'dev5/batch_*/D*_sidecar.json')):
        c = json.load(open(f)); sid = c['scene_id']; cars[sid] = c
        if sid in npz:
            occs[sid] = np.array(Image.open(f.replace('_sidecar.json', '_occ.png')))
            amod[sid] = np.array(Image.open(f.replace('_sidecar.json', '_amodal.png')))
    ext = {}
    for f in glob.glob(os.path.join(root, 'results/bt1/ext/bt1_t*_sidec.jsonl')):
        for l in open(f):
            r = json.loads(l); ext[r['scene']] = r
    feats, sig = {}, {'hidden_in_U': 0, 'hidden_total': 0, 'hidden_in_R': 0, 'U_px': 0, 'R_px': 0, 'U_hidden_px': 0, 'R_hidden_px': 0}
    for sid in sorted(ext):
        if sid in npz:
            z = np.load(npz[sid]); f, m = scene_features(z, occs[sid]); feats[sid] = f
            hidden = (amod[sid] > 0) & m['B']                  # truth: limb outline pixels covered by the blanket
            sig['hidden_total'] += int(hidden.sum()); sig['hidden_in_U'] += int((hidden & m['U']).sum())
            sig['hidden_in_R'] += int((hidden & m['R']).sum())
            sig['U_px'] += int(m['U'].sum()); sig['R_px'] += int(m['R'].sum())
            sig['U_hidden_px'] += int((m['U'] & hidden).sum()); sig['R_hidden_px'] += int((m['R'] & hidden).sum())
        else:
            feats[sid] = {s: dict.fromkeys(CONTROL + RELIEF, 0.0) for s in SITES}
    signal = {'hidden_limb_recall_U': round(sig['hidden_in_U'] / max(sig['hidden_total'], 1), 3),
              'hidden_limb_recall_R': round(sig['hidden_in_R'] / max(sig['hidden_total'], 1), 3),
              'precision_U': round(sig['U_hidden_px'] / max(sig['U_px'], 1), 3), 'precision_R': round(sig['R_hidden_px'] / max(sig['R_px'], 1), 3)}
    res = {'scenes_with_relief': len(npz), 'signal': signal, 'rules': {}}
    rows = [(sid, s) for sid in sorted(ext) for s in SITES]
    base = np.array([EV.feat(ext[sid]['sites'][s]) + EV.limb_feat(ext[sid]['sites'][s]) for sid, s in rows])
    ctrl = np.array([[feats[sid][s][k] for k in CONTROL] for sid, s in rows])
    rel = np.array([[feats[sid][s][k] for k in RELIEF] for sid, s in rows])
    g = np.array([sid for sid, _ in rows]); blanket = np.array([sid in npz for sid, _ in rows])
    vf_true = np.array([cars[sid]['truth'][s]['visible_fraction'] for sid, s in rows])
    for rule in ('v3_compat', 'guide_primary'):
        y = np.array([C4.index(LB.label(cars[sid], rule)[s]) for sid, s in rows])
        arms = {'base': base, 'base+control': np.hstack([base, ctrl]), 'base+control+relief': np.hstack([base, ctrl, rel])}
        out = {}
        for name, X in arms.items():
            C = fitC(X, y, g); P = np.zeros((len(y), 4))
            from sklearn.model_selection import GroupKFold
            for i, j in GroupKFold(5).split(X, y, g):
                P[j] = proba(mk(C).fit(X[i], y[i]), X[j])
            p = P.argmax(1)
            out[name] = {'all': metrics(y, p), 'blanket': metrics(y[blanket], p[blanket]),
                         'blanket_nt_vs_testable_acc': round(float(((p[blanket] == 3) == (y[blanket] == 3)).mean()), 4)}
        res['rules'][rule] = out
    # direct: visibility at the 0.10 cutoff on blanket sites, relief estimate vs the model's learned fraction
    est = rel[:, RELIEF.index('est_vf_relief')]; learned = np.array([min(ext[sid]['sites'][s]['limb']['vis_over_amodal'], 1) for sid, s in rows])
    t = vf_true >= 0.10
    res['cutoff_0.10_agreement_blanket'] = {'relief_oracle': round(float(((est >= 0.10) == t)[blanket].mean()), 4),
                                            'bt1_learned': round(float(((learned >= 0.10) == t)[blanket].mean()), 4),
                                            'blanket_sites': int(blanket.sum())}
    json.dump(res, open(out_path, 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
