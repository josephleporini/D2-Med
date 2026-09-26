"""Misses and near misses for stump / wound / not_testable on the predicted map with rendered side (checks.py records).
usage: python nm.py <gen dir> '<chk glob>' '<meta glob (feature jsonl with clothed/labels)>'"""
import sys, json, glob, numpy as np
sys.path.insert(0, sys.argv[1])
import eval_v3 as EV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold

S = ['LUE', 'RUE', 'LLE', 'RLE']; C4 = EV.C4
meta = {}
for f in glob.glob(sys.argv[3]):
    for l in open(f):
        r = json.loads(l); meta[r['scene']] = dict(clothed=r.get('clothed'), labels=r.get('labels'))
recs = {'dev3': [], 'test4': []}
for f in sorted(glob.glob(sys.argv[2])):
    for l in open(f):
        r = json.loads(l)
        if r['split'] in recs:
            recs[r['split']].append(r)


def tab(split):
    X, y, g, M = [], [], [], []
    for r in recs[split]:
        mt = meta.get(r['scene'], {})
        nw = sum(r['labels']['0.10'][s] == 'wound' for s in S); na = sum(r['labels']['0.10'][s] == 'amputation' for s in S)
        for s in S:
            X.append(EV.feat(r['pred_sites_ceil'][s])); y.append(C4.index(r['labels']['0.10'][s])); g.append(r['scene'])
            lab = (mt.get('labels') or r.get('labels') or {})
            thr = {t: lab[t][s] for t in ('0.00', '0.10', '0.25') if t in lab}
            M.append(dict(p=r['pred_sites_ceil'][s], t=r['gt_sites_ceil'][s], vf=r['visible_fraction'][s], pos=r.get('position'),
                          facing='front' if r.get('facing_true') == 1 else 'back', limb='upper' if s.endswith('UE') else 'lower',
                          clothed=mt.get('clothed'), thr_unstable=len(set(thr.values())) > 1, n_wound_img=nw, n_amp_img=na))
    return np.array(X, float), np.array(y), np.array(g), M


Xd, yd, gd, Md = tab('dev3'); Xt, yt, gt_, Mt = tab('test4')
best = None
for C in [0.03, 0.1, 0.3, 1, 3]:
    a = np.mean([(make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000)).fit(Xd[i], yd[i]).predict(Xd[j]) == yd[j]).mean()
                 for i, j in GroupKFold(5).split(Xd, yd, gd)])
    if best is None or a > best[1]:
        best = (C, a)
m = make_pipeline(StandardScaler(), LogisticRegression(C=best[0], max_iter=5000)).fit(Xd, yd)
P = m.predict_proba(Xt); pr = P.argmax(1); srt = np.sort(P, 1); margin = srt[:, -1] - srt[:, -2]
ok = pr == yt
out = {'acc': round(float(ok.mean()), 4), 'C': best[0]}

# 1. confidence structure
def rate(mask):
    return {'n': int(mask.sum()), 'acc': round(float(ok[mask].mean()), 3) if mask.any() else None}
out['by_margin'] = {'<0.2': rate(margin < 0.2), '0.2-0.5': rate((margin >= 0.2) & (margin < 0.5)), '>=0.5': rate(margin >= 0.5)}
conf_err = (~ok) & (P.max(1) >= 0.8)
out['confident_errors_p>=0.8'] = {f'{C4[yt[i]]}->{C4[pr[i]]}': 0 for i in np.nonzero(conf_err)[0]}
for i in np.nonzero(conf_err)[0]:
    out['confident_errors_p>=0.8'][f'{C4[yt[i]]}->{C4[pr[i]]}'] += 1
near = ok & (margin < 0.2)
nm = {}
for i in np.nonzero(near)[0]:
    k = f'{C4[yt[i]]} (runner-up {C4[np.argsort(P[i])[-2]]})'; nm[k] = nm.get(k, 0) + 1
out['near_misses_correct_margin<0.2'] = dict(sorted(nm.items(), key=lambda t: -t[1]))

# 2. label definition sensitivity
thr_u = np.array([x['thr_unstable'] for x in Mt])
out['label_changes_across_vf_thresholds'] = {'unstable': rate(thr_u), 'stable': rate(~thr_u)}

# 3. wound
W = yt == 1
gw = np.array([x['t']['wound_px'] for x in Mt]); pw = np.array([x['p']['wound_px'] for x in Mt])
pwp = np.array([(x['p']['p_wound'] or [1, 0])[1] for x in Mt]); twp = np.array([(x['t']['p_wound'] or [1, 0])[1] for x in Mt])
bins = [(0, 1), (1, 20), (20, 60), (60, 150), (150, 10 ** 9)]
out['wound_recall_by_true_wound_px'] = {f'{a}-{b}': rate(W & (gw >= a) & (gw < b)) for a, b in bins}
miss_w = W & (pr != 1)
out['missed_wound_detail'] = {'n': int(miss_w.sum()), 'pred_wound_px_0': int((miss_w & (pw == 0)).sum()),
                              'pred_wound_px_1_19': int((miss_w & (pw > 0) & (pw < 20)).sum()), 'pred_wound_px_ge20': int((miss_w & (pw >= 20)).sum()),
                              'median_true_wound_px': float(np.median(gw[miss_w])) if miss_w.any() else None,
                              'median_p_wound_pred': round(float(np.median(pwp[miss_w])), 3) if miss_w.any() else None,
                              'median_p_wound_true_map': round(float(np.median(twp[miss_w])), 3) if miss_w.any() else None,
                              'called_as': {c: int((miss_w & (pr == k)).sum()) for k, c in enumerate(C4)}}
fw = (yt != 1) & (pr == 1)
out['false_wound_detail'] = {'n': int(fw.sum()), 'true_map_wound_px_0': int((fw & (gw == 0)).sum()),
                             'pred_wound_px_median': float(np.median(pw[fw])) if fw.any() else None,
                             'median_p_wound_pred': round(float(np.median(pwp[fw])), 3) if fw.any() else None,
                             'true_class': {c: int((fw & (yt == k)).sum()) for k, c in enumerate(C4)},
                             'with_tq_on_limb': int((fw & (np.array([x['t']['tq_px'] for x in Mt]) > 0)).sum()),
                             'true_amp_same_limb': int((fw & (yt == 2)).sum())}

# 4. amputation
A = yt == 2
gs = np.array([x['t']['stump_px'] for x in Mt]); ps = np.array([x['p']['stump_px'] for x in Mt])
pa = np.array([(x['p']['p_end'] or [0, 0, 0])[1] for x in Mt]); ta = np.array([(x['t']['p_end'] or [0, 0, 0])[1] for x in Mt])
miss_a = A & (pr != 2)
out['amp_recall_by_true_stump_px'] = {f'{a}-{b}': rate(A & (gs >= a) & (gs < b)) for a, b in [(0, 1), (1, 50), (50, 200), (200, 600), (600, 10 ** 9)]}
out['missed_amp_detail'] = {'n': int(miss_a.sum()), 'pred_stump_px_0': int((miss_a & (ps == 0)).sum()),
                            'pred_stump_present_but_p_end_amp<0.5': int((miss_a & (ps > 0) & (pa < 0.5)).sum()),
                            'true_map_p_end_amp>=0.5': int((miss_a & (ta >= 0.5)).sum()),
                            'median_stump_ratio_pred_over_true': round(float(np.median((ps[miss_a] + 1) / (gs[miss_a] + 1))), 3) if miss_a.any() else None,
                            'called_as': {c: int((miss_a & (pr == k)).sum()) for k, c in enumerate(C4)}}
fa = (yt != 2) & (pr == 2)
out['false_amp_detail'] = {'n': int(fa.sum()), 'pred_stump_px>0': int((fa & (ps > 0)).sum()), 'true_map_stump_px>0': int((fa & (gs > 0)).sum()),
                           'true_class': {c: int((fa & (yt == k)).sum()) for k, c in enumerate(C4)},
                           'median_vf': round(float(np.median([Mt[i]['vf'] for i in np.nonzero(fa)[0]])), 3) if fa.any() else None,
                           'end_ring_pred_OCC_or_EDGE_share': round(float(np.mean([(Mt[i]['p']['end'].get('OCC', 0) + Mt[i]['p']['end'].get('EDGE', 0)) / max(sum(Mt[i]['p']['end'].values()), 1) for i in np.nonzero(fa)[0]])), 3) if fa.any() else None}
out['amp_end_ring_OCC_or_EDGE_share_all_true_amp'] = round(float(np.mean([(Mt[i]['p']['end'].get('OCC', 0) + Mt[i]['p']['end'].get('EDGE', 0)) / max(sum(Mt[i]['p']['end'].values()), 1) for i in np.nonzero(A)[0]])), 3)
out['no_injury_end_ring_OCC_or_EDGE_share'] = round(float(np.mean([(Mt[i]['p']['end'].get('OCC', 0) + Mt[i]['p']['end'].get('EDGE', 0)) / max(sum(Mt[i]['p']['end'].values()), 1) for i in np.nonzero(yt == 0)[0]])), 3)

# 5. slices by class
def cls_slice(key, k):
    d = {}
    for v in sorted(set(str(x[key]) for x in Mt)):
        msk = (yt == k) & np.array([str(x[key]) == v for x in Mt]); d[v] = rate(msk)
    return d
for k, c in ((1, 'wound'), (2, 'amputation')):
    out[f'{c}_recall_by'] = {key: cls_slice(key, k) for key in ('pos', 'facing', 'limb', 'clothed')}
    vb = np.array([x['vf'] for x in Mt])
    out[f'{c}_recall_by_vf'] = {b: rate((yt == k) & (vb >= lo) & (vb < hi)) for b, lo, hi in (('<0.25', 0, .25), ('0.25-0.5', .25, .5), ('0.5-0.8', .5, .8), ('>=0.8', .8, 2))}
nwi = np.array([x['n_wound_img'] + x['n_amp_img'] for x in Mt])
out['acc_by_injured_limbs_in_image'] = {str(v): rate(nwi == v) for v in sorted(set(nwi.tolist()))}
out['clothed_all'] = {str(v): rate(np.array([x['clothed'] == v for x in Mt])) for v in (True, False, None)}
print('NM_RES', json.dumps(out), flush=True)
