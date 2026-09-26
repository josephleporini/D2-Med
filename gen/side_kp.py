"""Side-assignment test: current rule (T2.to_classmap) vs keypoint-partitioned rule, same everything else.
usage: python side_kp.py <scene_dir> <arm: pred|gt> <out_prefix>
Writes <out_prefix>_{old,kp}.jsonl in the eval_v3 record format (features + labels), for fitting offline.
Keypoint rule: RTMW chains (shoulder-elbow-wrist, hip-knee-ankle) split each limb group's pixels into two
chain groups by distance to the chain polylines; which chain is anatomical left is decided by the existing frame
(left_dir): the chain whose root joint lies further along left_dir is L. Torso, occluder and stump handling are
unchanged (stump pixels join the group they touch, as in to_classmap).
"""
import os, sys, json, glob, time
import numpy as np, cv2, torch, torch.nn.functional as F
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_v3 as EV, test_a_masks as TA, test_e as TE, test_e2 as T2, lend as LE, parts as PT
from lwound_cache import limb_window
from d2pipe import letterbox
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

def geodesic_fast(limb, passable, seeds):
    ok = limb | passable
    dist = np.full(limb.shape, np.inf)
    ys, xs = np.nonzero(ok); n = len(ys)
    if n == 0:
        return dist
    H, W = limb.shape
    idx = -np.ones(limb.shape, np.int64); idx[ys, xs] = np.arange(n)
    r, c, w = [], [], []
    for dy, dx, cost in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.414), (1, -1, 1.414)):
        y2, x2 = ys + dy, xs + dx
        v = (y2 >= 0) & (y2 < H) & (x2 >= 0) & (x2 < W)
        v[v] = ok[y2[v], x2[v]]
        r.append(idx[ys[v], xs[v]]); c.append(idx[y2[v], x2[v]]); w.append(np.full(v.sum(), cost))
    G = sparse.coo_matrix((np.concatenate(w), (np.concatenate(r), np.concatenate(c))), shape=(n, n)).tocsr()
    src = idx[seeds & ok]
    if len(src) == 0:
        return dist
    d = dijkstra(G, directed=False, indices=np.unique(src), min_only=True)
    dist[ys, xs] = d
    dist[~limb] = np.inf
    return dist


TA.geodesic = geodesic_fast          # checked for exactness in prof_latency
K = T2.K; GROUP = T2.GROUP; EXT = T2.EXT
CHAINS = {'UE': ((5, 7, 9), (6, 8, 10)), 'LE': ((11, 13, 15), (12, 14, 16))}


def seg_dist(P, A, B):
    AB = B - A; t = np.clip(((P - A) @ AB) / max(AB @ AB, 1e-6), 0, 1)
    return np.linalg.norm(P - (A + t[:, None] * AB), axis=1)


def chain_dist(P, xy, ch):
    a, b, c = ch
    return np.minimum(seg_dist(P, xy[a], xy[b]), seg_dist(P, xy[b], xy[c]))


def to_classmap_kp(seg, left_dir, axis, kxy):
    cm, names, ext, stc = T2.to_classmap(seg, left_dir, axis)      # start from the current map (torso, OCC, BG)
    if kxy is None:
        return cm, names, ext, stc, False
    ix = {n: i for i, n in enumerate(names)}
    xy = np.asarray(kxy, float) / 2.0                                 # 1280x960 -> 640x480
    stump = seg == K['stump']
    for g in GROUP:
        gm = (cm == ix['L' + g]) | (cm == ix['R' + g])                 # same limb pixels the current rule used
        if not gm.any():
            continue
        ys, xs = np.nonzero(gm); P = np.stack([xs, ys], 1).astype(float)
        cA, cB = CHAINS[g]
        dA, dB = chain_dist(P, xy, cA), chain_dist(P, xy, cB)
        a_is_left = (xy[cA[0]] - xy[cB[0]]) @ left_dir > 0
        la = np.where(dA <= dB, 'A', 'B')
        # component smoothing: a connected component that is >= 85% one chain goes entirely to that chain
        lab, n = ndimage.label(gm, structure=np.ones((3, 3)))
        cid = lab[ys, xs]
        for c in range(1, n + 1):
            sel = cid == c
            fa = (la[sel] == 'A').mean()
            if fa >= 0.85:
                la[sel] = 'A'
            elif fa <= 0.15:
                la[sel] = 'B'
        side = np.where((la == 'A') == a_is_left, 'L', 'R')
        cm[ys, xs] = np.array([ix['L' + g], ix['R' + g]])[(side == 'R').astype(int)]
        for sd in 'LR':
            m = cm == ix[sd + g]
            ext[sd + g] = int((m & (seg == EXT[g])).sum()); stc[sd + g] = int((m & stump).sum())
    return cm, names, ext, stc, True


def site_rows(m, img, cm, names, ext, stc, wm, tq):
    ix = {n: i for i, n in enumerate(names)}
    an = TA.analyse(cm, names); torso = cm == ix['TORSO']; s0 = LE.window_size(torso); te = max(TA.extent(torso), 1)
    jobs = []
    for site in TE.SITES:
        pm = cm == ix[site]
        if an[site]['vis_px'] >= 30:
            x, y = LE.limb_end(pm, torso); cx, cy, sw = limb_window(pm)
            jobs.append((site, pm, x, y, cx, cy, sw))
    pe, pw = {}, {}
    for site, pm, x, y, cx, cy, sw in jobs:          # serial encode, identical to eval_v3.site_features
        fe = torch.from_numpy(LE.encode(m['P'], LE.crop_1280(img, x, y, s0))).float()[None]
        fw = torch.from_numpy(LE.encode(m['P'], LE.crop_1280(img, cx, cy, sw))).float()[None]
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


if __name__ == '__main__':
    D, arm, pre = sys.argv[1], sys.argv[2], sys.argv[3]
    M = LE.M
    m = EV.load_models(os.path.join(M, 'seg3_e2e_b.pt') if arm == 'pred' else None, os.path.join(M, 'lend3d.pt'), os.path.join(M, 'lwound2.pt'))
    fo = {k: open(f'{pre}_{k}.jsonl', 'w') for k in ('kp',)}
    t0 = time.time(); n = 0; nokp = 0
    for f in sorted(glob.glob(os.path.join(D, 'C*_sidecar.json'))):
        sid = os.path.basename(f)[:-len('_sidecar.json')]
        sc = json.load(open(f))
        img = letterbox(cv2.cvtColor(cv2.imread(os.path.join(D, sid + '.jpg')), cv2.COLOR_BGR2RGB))
        seg3 = EV.pred_part_map(m, img) if arm == 'pred' else EV.gt_part_map(os.path.join(D, sid + '_part3.png'))
        seg, wm, tq = EV.fold_extras(seg3)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        k, s = m['wb'](bgr); kxy = None
        if n == 0: print('ORT_PROVIDERS', m['wb'].pose_model.session.get_providers(), flush=True)
        if len(k):
            i = int(np.argmax(s[:, :17].mean(1))); kxy = k[i][:17].tolist(); kk = k[i]
            axis = TE.unit((kk[5] + kk[6]) / 2 - (kk[11] + kk[12]) / 2)
        else:
            axis = np.array([0.0, -1.0])
        sfr = T2.seg_frame(seg)
        left = np.array([1.0, 0.0]) if (sfr is None and not len(k)) else (sfr[2] if sfr else 1) * TE.perp(axis)
        g = sc['params'].get('garments', {}) if isinstance(sc['params'], dict) else {}
        base = dict(scene=sid, arm=arm, visible_fraction=sc.get('visible_fraction'), labels=sc['labels_by_threshold'],
                    labels_wip=sc['labels_wound_if_present'], clothed=bool(g.get('top', 'none') != 'none' or g.get('bottom', 'none') != 'none'),
                    position=next((sc['params'][k] for k in ('position', 'pos', 'pose', 'posture') if isinstance(sc['params'], dict) and k in sc['params']), None))
        cm1, names, ext1, stc1, used = to_classmap_kp(seg, left, axis, kxy)
        nokp += not used
        fo['kp'].write(json.dumps(dict(base, cm='kp', sites=site_rows(m, img, cm1, names, ext1, stc1, wm, tq))) + '\n')
        n += 1
        if n % 20 == 0:
            print('SIDE', arm, n, round((time.time() - t0) / n, 2), 's/img', flush=True)
    for v in fo.values():
        v.close()
    print('SIDE_DONE', arm, D, n, 'no_kp', nokp, round(time.time() - t0, 1), flush=True)
