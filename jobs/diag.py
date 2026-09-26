"""Where does the predicted-map vs true-map gap come from (both with rendered side)?
usage: python diag.py <gen dir> '<chk glob>'
Uses checks.py records (pred_sites_ceil, gt_sites_ceil). Decision layer = eval_v3.feat + LR, C by grouped CV on dev3.
1) group swap: replace one feature group on the predicted map with its true-map value (dev and test), refit, score test4
2) breakdown of the gap by true class, visibility band, position, facing, limb
3) raw feature disagreement between maps per group"""
import sys, json, glob, numpy as np
sys.path.insert(0, sys.argv[1])
import eval_v3 as EV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold

S = ['LUE', 'RUE', 'LLE', 'RLE']
G = {'visibility': [0, 1, 2, 13], 'end_state': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14], 'wound': [15, 16, 17], 'tourniquet': [18]}
recs = {'dev3': [], 'test4': []}
for f in sorted(glob.glob(sys.argv[2])):
    for l in open(f):
        r = json.loads(l)
        if r['split'] in recs:
            recs[r['split']].append(r)


def tab(split):
    P, T, y, g, meta = [], [], [], [], []
    for r in recs[split]:
        for s in S:
            P.append(EV.feat(r['pred_sites_ceil'][s])); T.append(EV.feat(r['gt_sites_ceil'][s]))
            y.append(EV.C4.index(r['labels']['0.10'][s])); g.append(r['scene'])
            vf = r['visible_fraction'][s]
            meta.append(dict(vf=vf, band='<0.10' if vf < 0.10 else '0.10-0.25' if vf < 0.25 else '0.25-0.50' if vf < 0.5 else '>=0.50',
                             pos=r.get('position'), facing='front' if r.get('facing_true') == 1 else 'back', limb='upper' if s.endswith('UE') else 'lower',
                             praw=r['pred_sites_ceil'][s], traw=r['gt_sites_ceil'][s]))
    return np.array(P, float), np.array(T, float), np.array(y), np.array(g), meta


Pd, Td, yd, gd, md = tab('dev3'); Pt, Tt, yt, gt, mt = tab('test4')
print('DIAG_N', len(yd), len(yt), Pd.shape, flush=True)
assert Pd.shape[1] == 19


def fit(Xd, Xt, cw=None):
    best = None
    for C in [0.03, 0.1, 0.3, 1, 3]:
        a = np.mean([(make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight=cw)).fit(Xd[i], yd[i]).predict(Xd[j]) == yd[j]).mean()
                     for i, j in GroupKFold(5).split(Xd, yd, gd)])
        if best is None or a > best[1]:
            best = (C, a)
    m = make_pipeline(StandardScaler(), LogisticRegression(C=best[0], max_iter=5000, class_weight=cw)).fit(Xd, yd)
    return m.predict(Xt), round(best[1], 4)


def sc(p):
    return {'acc': round(float((p == yt).mean()), 4), 'min_recall': round(min(float((p[yt == k] == k).mean()) for k in range(4)), 3),
            'nt_err': int(((yt == 3) != (p == 3)).sum())}


out = {}
pP, cvP = fit(Pd, Pt); pT, cvT = fit(Td, Tt)
out['pred_map'] = dict(sc(pP), dev_cv=cvP); out['true_map'] = dict(sc(pT), dev_cv=cvT)
for cw in ('balanced',):
    out['pred_map_bal'] = sc(fit(Pd, Pt, cw)[0]); out['true_map_bal'] = sc(fit(Td, Tt, cw)[0])
sw = {}
for name, idx in G.items():
    A, B = Pd.copy(), Pt.copy(); A[:, idx] = Td[:, idx]; B[:, idx] = Tt[:, idx]
    p, cv = fit(A, B); sw[name] = dict(sc(p), dev_cv=cv)
    A, B = Td.copy(), Tt.copy(); A[:, idx] = Pd[:, idx]; B[:, idx] = Pt[:, idx]   # reverse: true map with this group from pred
    p, cv = fit(A, B); sw[name + '_rev'] = dict(sc(p), dev_cv=cv)
out['swap_true_group_into_pred'] = sw

# breakdown of the gap
def brk(key):
    d = {}
    for v in sorted(set(m[key] for m in mt), key=str):
        i = np.array([m[key] == v for m in mt])
        d[str(v)] = {'n': int(i.sum()), 'pred': round(float((pP[i] == yt[i]).mean()), 3), 'true': round(float((pT[i] == yt[i]).mean()), 3)}
    return d
out['by_band'] = brk('band'); out['by_pos'] = brk('pos'); out['by_facing'] = brk('facing'); out['by_limb'] = brk('limb')
out['by_class'] = {c: {'n': int((yt == k).sum()), 'pred': round(float((pP[yt == k] == k).mean()), 3), 'true': round(float((pT[yt == k] == k).mean()), 3)}
                   for k, c in enumerate(EV.C4)}
fix = (pP != yt) & (pT == yt); brk_ = (pP == yt) & (pT != yt)
trans = {}
for i in np.nonzero(fix)[0]:
    k = f'{EV.C4[yt[i]]}<-{EV.C4[pP[i]]}'; trans[k] = trans.get(k, 0) + 1
out['sites_fixed_by_true_map'] = int(fix.sum()); out['sites_broken_by_true_map'] = int(brk_.sum())
out['fixed_true<-pred'] = dict(sorted(trans.items(), key=lambda t: -t[1]))

# raw disagreement on the sites the true map fixes vs all sites
def raw(ids):
    d = dict(vis_ratio_med=[], vis_zero_pred_true_pos=0, vis_pos_pred_zero_true=0, stump_disagree=0, ext_disagree=0, wound_disagree=0, pend_absent_pred=0)
    for i in ids:
        a, b = mt[i]['praw'], mt[i]['traw']
        d['vis_ratio_med'].append((a['vis_px'] + 1) / (b['vis_px'] + 1))
        d['vis_zero_pred_true_pos'] += int(a['vis_px'] < 30 <= b['vis_px']); d['vis_pos_pred_zero_true'] += int(b['vis_px'] < 30 <= a['vis_px'])
        d['stump_disagree'] += int((a['stump_px'] > 0) != (b['stump_px'] > 0)); d['ext_disagree'] += int((a['ext_px'] > 0) != (b['ext_px'] > 0))
        d['wound_disagree'] += int((a['wound_px'] >= 20) != (b['wound_px'] >= 20)); d['pend_absent_pred'] += int(a['p_end'] is None and b['p_end'] is not None)
    v = np.array(d['vis_ratio_med']); d['vis_ratio_med'] = round(float(np.median(v)), 3) if len(v) else None
    d['vis_ratio_lt_0.5_or_gt_2'] = round(float(((v < 0.5) | (v > 2)).mean()), 3) if len(v) else None
    d['n'] = len(ids)
    return d
out['raw_all'] = raw(range(len(yt))); out['raw_fixed'] = raw(np.nonzero(fix)[0])
print('DIAG_RES', json.dumps(out), flush=True)
