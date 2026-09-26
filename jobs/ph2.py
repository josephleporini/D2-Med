"""Phase 2: feature tests on PREDICTED maps (adopted distilled side model), dev3 grouped CV only.
usage: python ph2.py <gen dir> <probeB dir> <extract dir (s15)> <maps dir>
Arms: A base | B + distal-part flag | C = B + tourniquet geometry | D = C + robust chain fit.
Profile and chain fit use the predicted part map and the learned side probability (no rendered truth)."""
import sys, os, json, glob, numpy as np
sys.path.insert(0, sys.argv[1])
import parts as PT, eval_v3 as EV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

P, EXT, MAPS = sys.argv[2], sys.argv[3], sys.argv[4]
S = ['LUE', 'RUE', 'LLE', 'RLE']; C4 = EV.C4; N = 48
CL = PT.SEG3_CLASSES; ix = {c: i for i, c in enumerate(CL)}
ARM = {ix['upper_arm'], ix['forearm'], ix['hand']}; LEG = {ix['thigh'], ix['shank'], ix['foot']}
DIST = {'UE': ix['hand'], 'LE': ix['foot']}; BODY = {ix[c] for c in ('TORSO_F', 'TORSO_B', 'HEAD_F', 'HEAD_B')}
CHAIN = {'UE': {'L': (5, 7, 9), 'R': (6, 8, 10)}, 'LE': {'L': (11, 13, 15), 'R': (12, 14, 16)}}
TOES = {'L': (17, 18), 'R': (20, 21)}
OWNISH = {'own', 'distal', 'stump', 'wound', 'tq'}


def cat_of(cv, sv, sd, parts, typ):
    if cv == ix['wound']:
        return 'wound' if sv == sd else 'other'
    if cv == ix['TQ']:
        return 'tq'
    if cv == ix['stump']:
        return 'stump' if sv == sd else 'other'
    if cv in parts:
        return ('distal' if cv == DIST[typ] else 'own') if sv == sd else 'other'
    if cv in ARM | LEG:
        return 'other'
    if cv in BODY:
        return 'body'
    if cv == ix['OCC']:
        return 'OCC'
    return 'BG'


def chain_pts(xy, sc, typ, kl):
    a, b, c = CHAIN[typ][kl]
    if typ == 'UE':
        ext = xy[c] + 0.6 * (xy[c] - xy[b])
    else:
        t = [xy[k] for k in TOES[kl] if sc[k] > 0.3]
        ext = xy[c] + (1.2 * (np.mean(t, 0) - xy[c]) if t else 0.3 * (xy[c] - xy[b]))
    return np.array([xy[a], xy[b], xy[c], ext], float), (sc[a], sc[b], sc[c])


def sample(seg, side, pts, torso, sd, parts, typ):
    H, W = seg.shape
    seglen = np.linalg.norm(np.diff(pts, axis=0), axis=1); cum = np.r_[0, np.cumsum(seglen)]; L = cum[-1] + 1e-6
    u = np.linspace(0, L, N); cats = []
    jd = int(np.argmin(np.abs(u - cum[2])))
    for t in u:
        k = min(np.searchsorted(cum, t, side='right') - 1, 2); f = (t - cum[k]) / max(seglen[k], 1e-6)
        p = pts[k] + f * (pts[k + 1] - pts[k]); d = (pts[k + 1] - pts[k]) / max(seglen[k], 1e-6); nrm = np.array([-d[1], d[0]])
        votes = []
        for o in (-0.06 * torso, 0, 0.06 * torso):
            x, y = p + o * nrm; xi, yi = int(round(x)), int(round(y))
            votes.append('edge' if not (0 <= xi < W and 0 <= yi < H) else cat_of(seg[yi, xi], side[yi, xi], sd, parts, typ))
        own = [v for v in votes if v in OWNISH]
        if own:
            for pref in ('stump', 'wound', 'distal', 'tq', 'own'):
                if pref in own:
                    cats.append(pref); break
        else:
            cats.append(max(set(votes), key=votes.count))
    return cats, jd


def seg_dist(q, a, b):
    ab = b - a; t = np.clip(((q - a) @ ab) / max(ab @ ab, 1e-6), 0, 1)
    return np.linalg.norm(q - (a + t[:, None] * ab), axis=1)


def site_feats(seg, side, xy, sc):
    torso = np.linalg.norm((xy[5] + xy[6]) / 2 - (xy[11] + xy[12]) / 2) + 1e-6
    H, W = seg.shape; out = {}
    for site in S:
        ltr, typ = site[0], site[1:]; sd = 1 if ltr == 'L' else 2; parts = ARM if typ == 'UE' else LEG
        best = None
        for kl in ('L', 'R'):
            pts, kps = chain_pts(xy, sc, typ, kl)
            cats, jd = sample(seg, side, pts, torso, sd, parts, typ)
            n_own = sum(c in OWNISH for c in cats)
            if best is None or n_own > best[0]:
                best = (n_own, kl, cats, jd, pts, kps)
        n_own, kl, cats, jd, pts, kps = best
        ownidx = [i for i, c in enumerate(cats) if c in OWNISH]; end = ownidx[-1] if ownidx else -1
        wi = [i for i, c in enumerate(cats) if c == 'wound']; ti = [i for i, c in enumerate(cats) if c == 'tq']
        f = dict(distal=float(cats.count('distal') > 0), stump_end=float(end >= 0 and 'stump' in cats[max(0, end - 3):end + 1]),
                 end_frac=min((end + 1) / (jd + 1), 1.5) if end >= 0 else 0.0,
                 after_bg=float(end >= 0 and end < N - 1 and cats[min(end + 1, N - 1)] in ('BG',)),
                 tq_any=float(bool(ti)), wound_any=float(bool(wi)),
                 wound_near_tq=float(bool(wi and ti and min(abs(a - b) for a in wi for b in ti) <= 2)),
                 wound_far_tq=float(bool(wi and ti and min(abs(a - b) for a in wi for b in ti) > 2)),
                 wound_distal_tq=float(bool(wi and ti and max(wi) > max(ti) + 2)), kp_min=float(min(kps)), mirrored=float(kl != ltr))
        # tourniquet geometry from pixels: own-side wound pixels within / outside a dilated tourniquet band near this chain
        own_m = (side == sd)
        ys, xs = np.nonzero(((seg == ix['wound']) & own_m) | (seg == ix['TQ']))
        if len(xs):
            q = np.c_[xs, ys].astype(float)
            dmin = np.min([seg_dist(q, pts[i], pts[i + 1]) for i in range(3)], axis=0); near = dmin < 0.25 * torso
            q = q[near]; lab = seg[ys[near], xs[near]]
        else:
            q = np.zeros((0, 2)); lab = np.zeros(0, int)
        W_ = q[lab == ix['wound']]; T_ = q[lab == ix['TQ']]
        if len(W_) and len(T_):
            dd = np.sqrt(((W_[:, None, :] - T_[None, :, :]) ** 2).sum(-1)).min(1)
            f['wound_px_in_tq'] = float((dd <= 0.04 * torso).sum()) / max(len(W_), 1); f['wound_px_out_tq'] = float(np.log1p((dd > 0.04 * torso).sum()))
            f['wound_tq_dist'] = float(np.median(dd) / torso)
        else:
            f['wound_px_in_tq'] = 0.0; f['wound_px_out_tq'] = float(np.log1p(len(W_))); f['wound_tq_dist'] = 1.0 if len(W_) else 0.0
        # robust chain fit: own-side limb pixels near the chain, principal axis from the proximal joint, 98th percentile extent
        limb = np.isin(seg, list(parts | {ix['stump'], ix['wound'], ix['TQ']})) & (own_m | (seg == ix['TQ']))
        ys, xs = np.nonzero(limb)
        fit = dict(Lfit=0.0, npx=0.0, ring_bg=0.0, ring_occ=0.0, ring_body=0.0, ring_other=0.0, ring_edge=0.0, ring_own=0.0, distal_at_end=0.0)
        if len(xs) > 20:
            q = np.c_[xs, ys].astype(float)
            dmin = np.min([seg_dist(q, pts[i], pts[i + 1]) for i in range(3)], axis=0); keep = dmin < 0.35 * torso
            q = q[keep]; lab = seg[ys[keep], xs[keep]]
            if len(q) > 20:
                p0 = pts[0]; c = q - q.mean(0); _, _, vt = np.linalg.svd(c, full_matrices=False); u = vt[0]
                if (q.mean(0) - p0) @ u < 0:
                    u = -u
                proj = (q - p0) @ u; e98 = np.quantile(proj, 0.98)
                fit['Lfit'] = float(max(e98, 0) / torso); fit['npx'] = float(np.log1p(len(q)))
                tip = proj >= np.quantile(proj, 0.95)
                fit['distal_at_end'] = float(np.mean(lab[tip] == DIST[typ]))
                ctr = p0 + (e98 + 0.08 * torso) * u; r = max(int(0.05 * torso), 2)
                cnt = {k: 0 for k in ('BG', 'OCC', 'body', 'other', 'edge', 'own')}
                for dy in range(-r, r + 1, max(r // 3, 1)):
                    for dx in range(-r, r + 1, max(r // 3, 1)):
                        x, y = int(ctr[0] + dx), int(ctr[1] + dy)
                        if not (0 <= x < W and 0 <= y < H):
                            cnt['edge'] += 1; continue
                        cc = cat_of(seg[y, x], side[y, x], sd, parts, typ); cnt['own' if cc in OWNISH else cc] += 1
                tot = max(sum(cnt.values()), 1)
                for k in cnt:
                    fit['ring_' + k] = cnt[k] / tot
        f.update(fit); out[site] = f
    for site in S:            # contralateral length ratio
        opp = {'LUE': 'RUE', 'RUE': 'LUE', 'LLE': 'RLE', 'RLE': 'LLE'}[site]
        out[site]['Lratio'] = float(np.log((out[site]['Lfit'] + 0.05) / (out[opp]['Lfit'] + 0.05)))
    return out


B_KEYS = ['distal']
C_KEYS = B_KEYS + ['tq_any', 'wound_near_tq', 'wound_far_tq', 'wound_distal_tq', 'wound_px_in_tq', 'wound_px_out_tq', 'wound_tq_dist']
D_KEYS = C_KEYS + ['Lfit', 'Lratio', 'npx', 'ring_bg', 'ring_occ', 'ring_body', 'ring_other', 'ring_edge', 'ring_own', 'distal_at_end', 'stump_end', 'end_frac', 'after_bg']

rows = {}
for sp in ('dev3', 'test4'):
    recs = [json.loads(l) for f in sorted(glob.glob(f'{EXT}/{sp}_*_sidec.jsonl')) for l in open(f)]
    rows[sp] = []
    for r in recs:
        z = np.load(os.path.join(MAPS, r['scene'] + '.npz')); seg = z['seg']; pl = z['pl'].astype(float) / 255
        side = np.where(pl >= 0.5, 1, 2).astype(np.uint8); kp = z['kp']; xy = kp[:, :2] / 2.0; sc = kp[:, 2]
        pf = site_feats(seg, side, xy, sc) if sc[:17].max() > 0 else {s: None for s in S}
        for s in S:
            rows[sp].append(dict(scene=r['scene'], site=s, base=EV.feat(r['sites'][s]), pf=pf[s], y=C4.index(r['labels']['0.10'][s]),
                                 tq_true=None))
    print('ROWS', sp, len(rows[sp]), flush=True)
side_meta = {}
for f in glob.glob(f'{P}/out/dev3/C*_sidecar.json'):
    d = json.load(open(f)); side_meta[d['scene_id']] = d['params']
json.dump({sp: [dict(r, base=None) for r in v] for sp, v in rows.items()}, open(os.path.join(EXT, 'ph2_features.json'), 'w'))

dv = rows['dev3']; y = np.array([r['y'] for r in dv]); g = np.array([r['scene'] for r in dv])
def pf_vec(r, keys):
    return [0.0] * len(keys) if r['pf'] is None else [r['pf'][k] for k in keys]
def cv(X, cw=None):
    best = None
    for C in [0.03, 0.1, 0.3, 1, 3]:
        pr = np.zeros(len(y), int)
        for i, j in GroupKFold(5).split(X, y, g):
            pr[j] = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight=cw)).fit(X[i], y[i]).predict(X[j])
        a = (pr == y).mean()
        if best is None or a > best[0]:
            best = (a, pr, C)
    a, pr, C = best
    tq = np.array([r['site'] in (side_meta.get(r['scene'], {}).get('tourniquets') or []) for r in dv])
    neg_w = (y != 1) & (y != 3); neg_a = (y != 2) & (y != 3)
    return {'acc': round(float(a), 4), 'min_recall': round(min(float((pr[y == k] == k).mean()) for k in range(4)), 3),
            'macro_f1': round(float(f1_score(y, pr, average='macro')), 4), 'C': C,
            'recall': {C4[k]: round(float((pr[y == k] == k).mean()), 3) for k in range(4)},
            'false_wound_tq_limbs': round(float((pr[neg_w & tq] == 1).mean()), 3) if (neg_w & tq).any() else None,
            'false_wound_all': round(float((pr[neg_w] == 1).mean()), 3), 'false_amp_all': round(float((pr[neg_a] == 2).mean()), 3),
            'missed_amp': int(((y == 2) & (pr != 2)).sum())}
Xb = np.array([r['base'] for r in dv]); res = {}
for name, keys in (('A_base', []), ('B_distal', B_KEYS), ('C_tq_geometry', C_KEYS), ('D_chain_fit', D_KEYS)):
    X = np.c_[Xb, np.array([pf_vec(r, keys) for r in dv])] if keys else Xb
    res[name] = {'unweighted': cv(X), 'balanced': cv(X, 'balanced')}
# ablation of D groups on top of A, to see which block carries
for name, keys in (('A+tq_only', C_KEYS[1:]), ('A+chain_only', D_KEYS[len(C_KEYS):])):
    X = np.c_[Xb, np.array([pf_vec(r, keys) for r in dv])]
    res[name] = {'unweighted': cv(X), 'balanced': cv(X, 'balanced')}
# separation checks on dev3
def xt(fn):
    d = {}
    for r in dv:
        if r['pf'] is None:
            continue
        k = fn(r['pf']); d.setdefault(C4[r['y']], {}); d[C4[r['y']]][k] = d[C4[r['y']]].get(k, 0) + 1
    return d
res['sep_distal_pred_map'] = xt(lambda p: 'distal' if p['distal'] else 'no_distal')
res['sep_distal_at_end'] = xt(lambda p: 'hand_foot_at_tip' if p['distal_at_end'] > 0.3 else 'none_at_tip')
res['sep_ring'] = xt(lambda p: max(('bg', p['ring_bg']), ('occ', p['ring_occ']), ('body', p['ring_body']), ('other', p['ring_other']), ('edge', p['ring_edge']), ('own', p['ring_own']), key=lambda t: t[1])[0])
res['no_keypoints_images'] = int(sum(r['pf'] is None for r in dv) / 4)
print('PH2_RES', json.dumps(res), flush=True)
