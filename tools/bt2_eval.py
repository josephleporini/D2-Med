"""BT-2 evaluation: compare engine models on the same images under the same protocol.

Rows come from the qualification container's feature log (D2_FEATURES_OUT; original and mirrored rows per image).
Truth: sidecar labels at the 0.10 visibility rule (after the 28 Sep garment label fix).

Protocol per model:
  dev5   decision layer (eval_v3.feat + limb_feat, StandardScaler + logistic regression, C by grouped CV) scored out of
         fold, grouped 5-fold by scene; original pass and mirror-averaged
  others (dev6, close-up sets) layer fitted on all dev5 rows of that model, applied as shipped (mirror-averaged)
Paired comparison on each set: sites fixed and broken between models, exact sign test.

usage: python tools/bt2_eval.py <out.json> <model>=<set>:<rows.jsonl>:<sidecar_dir> [...]
       the set named dev5 must be present for every model
"""
import sys, os, json, glob
import numpy as np
from math import comb
from sklearn.model_selection import GroupKFold
HERE = os.path.dirname(os.path.abspath(__file__))
for p in ('gen', 'jobs', 'score'):
    sys.path.insert(0, os.path.join(HERE, '..', p))
from score import mk, proba, metrics, fitC
import engine_bt1 as EB
C4 = ['no_injury', 'wound', 'amputation', 'not_testable']; S = ['LUE', 'RUE', 'LLE', 'RLE']


def truth(d):
    T = {}
    for f in glob.glob(os.path.join(d, '**', '*_sidecar.json'), recursive=True):
        c = json.load(open(f)); T[c['scene_id']] = [C4.index(c['labels_by_threshold']['0.10'][s]) for s in S]
    return T


def load(rows, sd):
    T = truth(sd); X, XF, y, g = [], [], [], []
    for l in open(rows):
        r = json.loads(l)
        if not (r.get('sites') and r.get('sites_flip')):
            continue
        sid = os.path.splitext(r['image_id'])[0]
        for k, s in enumerate(S):
            X.append(EB.features(r['sites'][s])); XF.append(EB.features(r['sites_flip'][s])); y.append(T[sid][k]); g.append(sid)
    return np.array(X, float), np.array(XF, float), np.array(y), np.array(g)


def sign_p(a, b):
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main(out, specs):
    D = {}
    for sp in specs:
        model, rest = sp.split('=', 1); st, rows, sd = rest.split(':')
        D.setdefault(model, {})[st] = load(rows, sd)
    res, pred = {}, {}
    for model, sets in D.items():
        X, XF, y, g = sets['dev5']
        C = fitC(X, y, g); P, PF = np.zeros((len(y), 4)), np.zeros((len(y), 4))
        for i, j in GroupKFold(5).split(X, y, g):
            m = mk(C).fit(X[i], y[i]); P[j] = proba(m, X[j]); PF[j] = proba(m, XF[j])
        r = {'C': C, 'dev5_oof_original': metrics(y, P.argmax(1)), 'dev5_oof_mirror': metrics(y, ((P + PF) / 2).argmax(1))}
        pred[(model, 'dev5')] = (((P + PF) / 2).argmax(1), y, g)
        full = mk(C).fit(X, y)
        for st, (Xs, XFs, ys, gs) in sets.items():
            if st == 'dev5':
                continue
            Q = (proba(full, Xs) + proba(full, XFs)) / 2
            r[f'{st}_shipped_mirror'] = metrics(ys, Q.argmax(1)); pred[(model, st)] = (Q.argmax(1), ys, gs)
        res[model] = r
    models = list(D)
    pairs = {}
    for a in models:
        for b in models:
            if a >= b:
                continue
            for st in D[a]:
                if (b, st) not in pred:
                    continue
                pa, ya, ga = pred[(a, st)]; pb, yb, gb = pred[(b, st)]
                ka = {(gg, i % 4): (pp, yy) for i, (pp, yy, gg) in enumerate(zip(pa, ya, ga))}
                kb = {(gg, i % 4): (pp, yy) for i, (pp, yy, gg) in enumerate(zip(pb, yb, gb))}
                common = set(ka) & set(kb)
                fixed = sum(1 for k in common if ka[k][0] != ka[k][1] and kb[k][0] == kb[k][1])
                broken = sum(1 for k in common if ka[k][0] == ka[k][1] and kb[k][0] != kb[k][1])
                pairs[f'{a}->{b} {st}'] = dict(sites=len(common), fixed=fixed, broken=broken, p=round(sign_p(fixed, broken), 4))
    json.dump({'models': res, 'paired': pairs}, open(out, 'w'), indent=1)
    for m, r in res.items():
        print('BT2EVAL', m, json.dumps({k: (v['acc'], v['min_recall'], v['recall'].get('wound'), v['recall'].get('amputation'))
                                        if isinstance(v, dict) else v for k, v in r.items()}))
    for k, v in pairs.items():
        print('BT2PAIR', k, json.dumps(v))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
