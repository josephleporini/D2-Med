"""Phase 0: POD-style detection curves, confuser table, generator facts.
usage: python ph0.py <gen dir> <probeB dir>
Arms: 'adopted' = distilled side model, gated side (out/s13_distill/*_sidec.all.jsonl);
      'trueside' = original segmenter map with rendered side (out/checks chk records, pred_sites_ceil).
Predictions: dev3 out-of-fold (5-fold grouped CV), test4 from a layer fit on all of dev3. Characterization only."""
import sys, os, json, glob, numpy as np
sys.path.insert(0, sys.argv[1])
import eval_v3 as EV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold

P = sys.argv[2]; O = P + '/out'; S = ['LUE', 'RUE', 'LLE', 'RLE']; C4 = EV.C4
SPL = ('dev3', 'test4')
side = {}
for sp in SPL:
    for f in glob.glob(f'{O}/{sp}/C*_sidecar.json'):
        d = json.load(open(f)); side[d['scene_id']] = dict(d, split=sp)
chk = {}
for f in glob.glob(f'{O}/checks/chk_*.jsonl'):
    for l in open(f):
        r = json.loads(l); chk[r['scene']] = r


def load_arm(arm):
    recs = {}
    for sp in SPL:
        if arm == 'adopted':
            fs = glob.glob(f'{O}/s13_distill/{sp}_sidec.all.jsonl')
            rows = [json.loads(l) for f in fs for l in open(f)]
            recs[sp] = [(r['scene'], r['sites']) for r in rows]
        else:
            recs[sp] = [(k, r['pred_sites_ceil']) for k, r in chk.items() if r['split'] == sp]
    return recs


def fitC(X, y, g):
    best = None
    for C in [0.03, 0.1, 0.3, 1, 3]:
        a = np.mean([(make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000)).fit(X[i], y[i]).predict(X[j]) == y[j]).mean()
                     for i, j in GroupKFold(5).split(X, y, g)])
        if best is None or a > best[1]:
            best = (C, a)
    return best[0]


def predict(recs):
    rows = []
    for sp in SPL:
        for sid, sites in recs[sp]:
            if sid not in side:
                continue
            lab = side[sid]['labels_by_threshold']['0.10']
            for s in S:
                rows.append(dict(sp=sp, sid=sid, site=s, x=EV.feat(sites[s]), y=C4.index(lab[s])))
    d = [r for r in rows if r['sp'] == 'dev3']; t = [r for r in rows if r['sp'] == 'test4']
    Xd = np.array([r['x'] for r in d]); yd = np.array([r['y'] for r in d]); gd = np.array([r['sid'] for r in d])
    C = fitC(Xd, yd, gd); mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
    oof = np.zeros(len(d), int)
    for i, j in GroupKFold(5).split(Xd, yd, gd):
        oof[j] = mk().fit(Xd[i], yd[i]).predict(Xd[j])
    pt = mk().fit(Xd, yd).predict(np.array([r['x'] for r in t]))
    for r, p in zip(d, oof):
        r['p'] = int(p)
    for r, p in zip(t, pt):
        r['p'] = int(p)
    return d + t


def meta(r):
    sd = side[r['sid']]; prm = sd['params']; s = r['site']; c = chk.get(r['sid'], {})
    gt = c.get('gt_sites_ceil', {}).get(s, {}) if c else {}
    ue = s.endswith('UE'); gar = prm.get('garments', {}) or {}
    return dict(pos=prm.get('body_position', c.get('position')), facing='front' if c.get('facing_true') == 1 else 'back',
                vf=sd['visible_fraction'][s], wpx=sd['wound_visible_px'][s], vis_px=sd['visible_px'][s],
                stump_px=gt.get('stump_px', 0), gvis=gt.get('vis_px', 0), tq=s in (prm.get('tourniquets') or []),
                wound_true=s in (prm.get('wounds') or {}), amp_true=s in (prm.get('amputations') or {}),
                sleeve=(gar.get('top') if ue else gar.get('bottom')), boots=bool(gar.get('boots')) and not ue,
                occ=prm.get('occluder'), occ_end=prm.get('occ_target') == 'limb_end')


def pod(rows, cls, size_key, norm_key=None, B=1000, seed=0):
    """logistic detection ~ log size; bootstrap by scene; returns curve points and size at p=0.9 when supported"""
    R = [r for r in rows if r['y'] == cls]
    if len(R) < 15:
        return {'n': len(R)}
    def sz(r):
        v = r['m'][size_key]
        if norm_key:
            v = v / max(r['m'][norm_key], 1)
        return v
    x = np.array([np.log(sz(r) + (1e-4 if norm_key else 1)) for r in R]); y = np.array([int(r['p'] == cls) for r in R])
    sids = np.array([r['sid'] for r in R]); us = np.unique(sids); rng = np.random.default_rng(seed)
    grid = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
    def fit(xx, yy):
        if len(set(yy)) < 2:
            return None
        m = LogisticRegression(C=100, max_iter=2000).fit(xx[:, None], yy); return m.coef_[0][0], m.intercept_[0]
    base = fit(x, y)
    curves, a90 = [], []
    for _ in range(B):
        pick = rng.choice(us, len(us)); idx = np.concatenate([np.nonzero(sids == u)[0] for u in pick])
        f = fit(x[idx], y[idx])
        if f is None:
            continue
        b1, b0 = f; curves.append(1 / (1 + np.exp(-(b1 * grid + b0))))
        a90.append((np.log(9) - b0) / b1 if b1 > 0 else np.inf)
    curves = np.array(curves); a90 = np.array(a90)
    lo, hi = x.min(), x.max(); inside = np.isfinite(a90) & (a90 >= lo) & (a90 <= hi)
    tr = (lambda v: float(np.exp(v))) if not norm_key else (lambda v: float(np.exp(v)))
    out = {'n': len(R), 'recall': round(float(y.mean()), 3), 'slope': round(float(base[0]), 3) if base else None,
           'curve': [{'size': round(tr(g), 4 if norm_key else 1), 'p': round(float(np.median(curves[:, k])), 3),
                      'lo': round(float(np.quantile(curves[:, k], 0.025)), 3), 'hi': round(float(np.quantile(curves[:, k], 0.975)), 3)}
                     for k, g in enumerate(grid)]}
    if inside.mean() >= 0.95:
        out['size_at_0.9'] = {'median': round(tr(np.median(a90)), 4 if norm_key else 1), 'lo': round(tr(np.quantile(a90, 0.025)), 4 if norm_key else 1),
                              'hi': round(tr(np.quantile(a90, 0.975)), 4 if norm_key else 1)}
    else:
        out['size_at_0.9'] = f'not supported ({round(float(inside.mean()), 2)} of bootstraps inside the observed range)'
    # empirical bins
    qs = np.quantile(x, [0, 0.25, 0.5, 0.75, 1.0])
    out['bins'] = [{'size': f'{tr(qs[k]):.4g}-{tr(qs[k + 1]):.4g}', 'n': int(((x >= qs[k]) & (x <= qs[k + 1])).sum()),
                    'recall': round(float(y[(x >= qs[k]) & (x <= qs[k + 1])].mean()), 3)} for k in range(4)]
    return out


RES = {}
for arm in ('adopted', 'trueside'):
    if arm == 'adopted' and not glob.glob(f'{O}/s13_distill/*_sidec.all.jsonl'):
        RES[arm] = 'no extraction files on the volume'; continue
    rows = predict(load_arm(arm))
    for r in rows:
        r['m'] = meta(r)
    acc = {sp: round(float(np.mean([r['p'] == r['y'] for r in rows if r['sp'] == sp])), 4) for sp in SPL}
    res = {'acc_dev_oof_test': acc, 'n_sites': len(rows)}
    # 0a POD-style curves
    res['pod_wound_px'] = pod(rows, 1, 'wpx')
    res['pod_wound_norm'] = pod(rows, 1, 'wpx', 'vis_px')
    res['pod_amp_stump_px'] = pod(rows, 2, 'stump_px')
    res['pod_amp_stump_norm'] = pod(rows, 2, 'stump_px', 'gvis')
    strata = {}
    for key, vals in (('pos', None), ('facing', None)):
        vs = sorted(set(str(r['m'][key]) for r in rows))
        for v in vs:
            sub = [r for r in rows if str(r['m'][key]) == v]
            strata[f'{key}={v}'] = {'wound': pod(sub, 1, 'wpx', B=300), 'amp': pod(sub, 2, 'stump_px', B=300)}
    for name, f in (('vf<0.5', lambda r: r['m']['vf'] < 0.5), ('vf>=0.5', lambda r: r['m']['vf'] >= 0.5)):
        sub = [r for r in rows if f(r)]
        strata[name] = {'wound': pod(sub, 1, 'wpx', B=300), 'amp': pod(sub, 2, 'stump_px', B=300)}
    res['pod_strata'] = strata
    # 0b confuser table: false wound rate among true non-wound sites; false amputation rate among true non-amputation sites
    def rate(sub, cls):
        neg = [r for r in sub if r['y'] != cls and r['y'] != 3]
        return {'n': len(neg), 'false_rate': round(float(np.mean([r['p'] == cls for r in neg])), 3) if neg else None}
    conds = {
        'all': lambda r: True,
        'tourniquet_on_limb': lambda r: r['m']['tq'],
        'tourniquet_and_hidden_wound': lambda r: r['m']['tq'] and r['m']['wound_true'] and r['y'] == 0,
        'hidden_wound_no_tq': lambda r: (not r['m']['tq']) and r['m']['wound_true'] and r['y'] == 0,
        'no_tq_no_wound_true': lambda r: (not r['m']['tq']) and not r['m']['wound_true'] and not r['m']['amp_true'],
        'sleeve_or_trouser_long': lambda r: r['m']['sleeve'] == 'long',
        'bare_limb': lambda r: r['m']['sleeve'] in ('none',),
        'boots': lambda r: r['m']['boots'],
        'occluder_aimed_at_limb_end': lambda r: r['m']['occ_end'],
        'foreshortened_vf<0.5_no_stump': lambda r: r['m']['vf'] < 0.5 and r['m']['stump_px'] == 0,
    }
    for v in sorted(set(str(r['m']['occ']) for r in rows)):
        conds[f'occluder={v}'] = (lambda vv: (lambda r: str(r['m']['occ']) == vv))(v)
    res['confusers'] = {k: {'false_wound': rate([r for r in rows if f(r)], 1), 'false_amp': rate([r for r in rows if f(r)], 2)} for k, f in conds.items()}
    RES[arm] = res

# 0c generator facts
ts = [d['total_s'] for d in side.values() if 'total_s' in d]
sz = []
for sid, d in list(side.items())[:60]:
    D = f'{O}/{d["split"]}'
    sz.append(sum(os.path.getsize(p) for p in glob.glob(f'{D}/{sid}*') if os.path.isfile(p)))
k0 = next(iter(side.values()))
RES['generator'] = {'scenes': len(side), 'render_s_median': float(np.median(ts)) if ts else None,
                    'render_s_p90': float(np.quantile(ts, 0.9)) if ts else None, 'bytes_per_scene_median': int(np.median(sz)) if sz else None,
                    'param_keys': sorted(k0['params'].keys()), 'samples': k0['params'].get('samples'),
                    'occluders': sorted(set(str(d['params'].get('occluder')) for d in side.values()))}
print('PH0_RES', json.dumps(RES), flush=True)
