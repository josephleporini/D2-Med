"""Visible-fraction estimator spike. usage: python vf.py <gen dir> <tag> <dev jsonl> <test jsonl> [<tag> <dev> <test> ...]
For each feature set (arm): baseline LR on eval_v3 features vs LR + estimated visible fraction.
VF regressor (gradient boosting, logit target) is cross-fitted on dev (out-of-fold for the LR's training rows),
refit on all dev for test. Also reports an oracle arm with the true visible fraction as a feature (upper bound)."""
import sys, json, numpy as np
sys.path.insert(0, sys.argv[1])
import eval_v3 as EV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
S = ['LUE', 'RUE', 'LLE', 'RLE']; OPP = {'LUE': 'RUE', 'RUE': 'LUE', 'LLE': 'RLE', 'RLE': 'LLE'}
CH = {'LUE': (5, 7, 9), 'RUE': (6, 8, 10), 'LLE': (11, 13, 15), 'RLE': (12, 14, 16)}
lg = lambda p: np.log(np.clip(p, 1e-3, 1 - 1e-3) / (1 - np.clip(p, 1e-3, 1 - 1e-3)))


def vf_feats(rec, s):
    r, o = rec['sites'][s], rec['sites'][OPP[s]]
    te = max(r['torso_ext'] or 1.0, 1.0)
    xy = np.array(rec['kp']['xy'], float) if rec.get('kp') and rec['kp'].get('xy') else None
    sc = np.array(rec['kp']['score'], float) if xy is not None else None
    if xy is not None and len(xy) > 16:
        a, b, c = CH[s]; L = np.linalg.norm(xy[a] - xy[b]) + np.linalg.norm(xy[b] - xy[c]); ks = [sc[a], sc[b], sc[c]]
        a2, b2, c2 = CH[OPP[s]]; L2 = np.linalg.norm(xy[a2] - xy[b2]) + np.linalg.norm(xy[b2] - xy[c2])
    else:
        L = L2 = 0.0; ks = [0, 0, 0]
    ue = s.endswith('UE')
    return [np.log1p(r['vis_px']), r['vis_px'] / te ** 2, r['Lfrac'], r['bg_frac'] if r['bg_frac'] is not None else -1,
            np.log1p(o['vis_px']), (r['vis_px'] + 1) / (o['vis_px'] + 1), L / te, L2 / te,
            r['vis_px'] / max(L * L, 1.0), *ks, float(ue), np.log1p(r['stump_px']), np.log1p(r['ext_px']),
            *(r['p_end'] or [0, 0, 0])]


import glob, os
KP = {}
for f in glob.glob(os.environ.get('KPSRC', '/nonexistent')):
    for l in open(f):
        r = json.loads(l)
        if r.get('kp'):
            KP[r['scene']] = r['kp']


def records(path):
    """plain feature jsonl, or chk:<glob>:<arm>:<split> for checks.py records (true-side ceiling features)"""
    if path.startswith('chk:'):
        _, pat, arm, split = path.split(':')
        for f in sorted(glob.glob(pat)):
            for l in open(f):
                r = json.loads(l)
                if r['split'] == split:
                    yield dict(scene=r['scene'], sites=r[f'{arm}_sites_ceil'], labels=r['labels'],
                               visible_fraction=r['visible_fraction'], kp=KP.get(r['scene']))
    else:
        for l in open(path):
            r = json.loads(l)
            if not r.get('kp'):
                r['kp'] = KP.get(r['scene'])
            yield r


def table(path):
    X, V, y, g, T = [], [], [], [], []
    for rec in records(path):
        for s in S:
            X.append(EV.feat(rec['sites'][s])); V.append(vf_feats(rec, s)); y.append(EV.C4.index(rec['labels']['0.10'][s]))
            g.append(rec['scene']); T.append(rec['visible_fraction'][s])
    return np.array(X, float), np.array(V, float), np.array(y), np.array(g), np.array(T, float)


def lr(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def fitC(X, y, g, cw=None):
    best = None
    for C in [0.03, 0.1, 0.3, 1, 3]:
        a = np.mean([(make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight=cw)).fit(X[i], y[i]).predict(X[j]) == y[j]).mean()
                     for i, j in GroupKFold(5).split(X, y, g)])
        if best is None or a > best[1]:
            best = (C, a)
    return best


def score(yt, p):
    rec = {c: round(float((p[yt == k] == k).mean()), 3) for k, c in enumerate(EV.C4)}
    nt = int(((yt == 3) != (p == 3)).sum())
    return {'acc': round(float((p == yt).mean()), 4), 'recall': rec, 'nt_boundary_err': nt}


def vfaug(v):  # features derived from an estimated (or true) visible fraction
    return np.c_[v, lg(v), (v < 0.10).astype(float), np.abs(lg(v) - lg(0.10))]


out = {}
args = sys.argv[2:]
for k in range(0, len(args), 3):
    tag, dp, tp = args[k:k + 3]
    Xd, Vd, yd, gd, Td = table(dp); Xt, Vt, yt, gt, Tt = table(tp)
    reg = lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=10, random_state=0)
    oof = np.zeros(len(yd))
    for i, j in GroupKFold(5).split(Vd, yd, gd):
        oof[j] = 1 / (1 + np.exp(-reg().fit(Vd[i], lg(Td[i])).predict(Vd[j])))
    vt = 1 / (1 + np.exp(-reg().fit(Vd, lg(Td)).predict(Vt)))
    res = {'vf_mae_dev_oof': round(float(np.abs(oof - Td).mean()), 4), 'vf_mae_test': round(float(np.abs(vt - Tt).mean()), 4),
           'vf_thresh_acc_test': round(float(((vt < 0.1) == (Tt < 0.1)).mean()), 4)}
    for cw in (None, 'balanced'):
        for name, A, B in (('base', Xd, Xt), ('vf_est', np.c_[Xd, vfaug(oof)], np.c_[Xt, vfaug(vt)]),
                           ('vf_true', np.c_[Xd, vfaug(Td)], np.c_[Xt, vfaug(Tt)])):
            C, cv = fitC(A, yd, gd, cw)
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight=cw)).fit(A, yd)
            res[f'{name}_{cw}'] = {'C': C, 'dev_cv': round(float(cv), 4), 'test': score(yt, m.predict(B))}
    out[tag] = res
print('VF_RES', json.dumps(out))
