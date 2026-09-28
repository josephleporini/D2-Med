"""Adaptation kit, real run: refit the decision layer on labeled real images and export it for the container.

usage: python adapt/run.py --labels <icd.json|labels.csv|T5.xlsx> --rows '<rows*.jsonl[.gz]>' --base <decision_layer_bt1.json>
                           --source-rows '<synthetic ext rows glob>' --source-sidecars <dev5 dir> --out <dir>
                           [--groups groups.csv | --images <image dir>] [--by acc|cost]
Writes <out>/report.json (every arm by grouped 5-fold CV, the nested selection and its per-fold picks, metrics with
group-bootstrap intervals) and <out>/decision_layer_adapted.json (the selected arm refit on all labeled sites; arm A
copies the base). Rows come from gen/engine_bt1 (jobs/bt3_a40.sh logs, or kit.extract_rows on a GPU pod).
"""
import os, sys, json, argparse, shutil
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kit as K
import engine_bt1 as EB
from sklearn.model_selection import GroupKFold


def main():
    ap = argparse.ArgumentParser()
    for a in ('--labels', '--rows', '--base', '--out'):
        ap.add_argument(a, required=True)
    ap.add_argument('--source-rows'); ap.add_argument('--source-sidecars'); ap.add_argument('--groups'); ap.add_argument('--images')
    ap.add_argument('--by', default='acc', choices=['acc', 'cost'])
    A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
    lab, meta = K.load_labels(A.labels)
    rows = K.load_rows(A.rows)
    ids = sorted(set(rows) & set(lab))
    groups = K.load_groups(ids, A.groups, A.images)
    X, XF, y, g, sid = K.design(rows, lab, groups)
    base = EB.DecisionLayer(A.base)
    Xs = ys = None
    if A.source_rows and A.source_sidecars:
        sl, _ = K.load_labels(A.source_sidecars, 'synthetic', '0.10')
        Xs, _, ys, _, _ = K.design(K.load_rows(A.source_rows), sl, {})
    cands = K.arms(base, Xs, ys)
    rep = dict(images=len(ids), sites=int(len(y)), groups=len(set(g)), mirrored_rows=XF is not None, by=A.by,
               class_mix={K.C4[k]: round(float((y == k).mean()), 3) for k in range(4)}, arms={}, cost_version=K.COST_VERSION)
    for n, f in cands.items():
        P = K.cv_proba(n, f, base, X, y, g, Xs, ys, XF)
        rep['arms'][n] = K.metrics(y, P.argmax(1), sid, groups=g)
    # nested estimate of the selection procedure itself
    PS, picks = np.zeros((len(y), 4)), []
    for tr, te in GroupKFold(min(5, len(set(g)))).split(X, y, g):
        c, _ = K.select(base, X[tr], y[tr], g[tr], Xs, ys, None if XF is None else XF[tr], by=A.by); picks.append(c)
        PS[te] = _fit_apply(c, cands[c], base, X[tr], y[tr], X[te], Xs, ys, None if XF is None else XF[te])
    rep['selection_nested'] = dict(K.metrics(y, PS.argmax(1), sid, groups=g), fold_picks=picks)
    final, inner = K.select(base, X, y, g, Xs, ys, XF, by=A.by)
    rep['final_pick'], rep['final_inner'] = final, inner
    out = os.path.join(A.out, 'decision_layer_adapted.json')
    extra = dict(adapted_from=os.path.basename(A.base), arm=final, labels=os.path.basename(A.labels), sites=int(len(y)))
    if final == 'A_synthetic':
        shutil.copy(A.base, out)
    else:
        m = cands[final]()
        if final.startswith('D_'):
            m.fit(X, y); K.fold_recal(m, out, extra)
        else:
            m.fit(X, y, Xs, ys) if final.startswith('C_') else m.fit(X, y)
            doc_base = json.load(open(A.base)); m.export(out, dict(extra, features=doc_base.get('features'), n_features=doc_base.get('n_features')))
    json.dump(rep, open(os.path.join(A.out, 'report.json'), 'w'), indent=1)
    print('ADAPT', json.dumps(dict(final=final, nested=rep['selection_nested']['acc'], base=rep['arms']['A_synthetic']['acc'], sites=rep['sites'])))


def _fit_apply(name, make, base, Xtr, ytr, Xte, Xs, ys, XFte=None):
    if name == 'A_synthetic':
        m = base
    else:
        m = make()
        m.fit(Xtr, ytr, Xs, ys) if name.startswith('C_') else m.fit(Xtr, ytr)
    p = m.proba(Xte)
    return 0.5 * (p + m.proba(XFte)) if XFte is not None else p


if __name__ == '__main__':
    main()
