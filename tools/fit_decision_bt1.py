"""Export the BT-1 decision layer for the inference engine (gen/engine_bt1.py).

Same layer as score/score.py --limb: eval_v3.feat + eval_v3.limb_feat, StandardScaler + multinomial logistic regression,
C chosen by grouped 5-fold CV, labels at the 0.10 visibility rule (v3_compat). Fitted on ALL dev5 rows, so any accuracy
measured on dev5 with this layer is in-sample; out-of-fold figures come from score.py / tta_score.py on the rows the
engine logs.

usage: python tools/fit_decision_bt1.py '<dddata>/results/bt1/ext/bt1_t*_sidec.jsonl' <out.json>
Output JSON: feature names, C, scaler mean/scale, coef (4 x F), intercept (4), classes, OOF accuracy for reference.
"""
import sys, os, json, glob, hashlib
import numpy as np
from sklearn.model_selection import GroupKFold
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'gen')); sys.path.insert(0, os.path.join(HERE, '..', 'score'))
import eval_v3 as EV
from score import fitC, mk, proba, C4, S


def main(pattern, out):
    files = sorted(glob.glob(pattern))
    rows = [json.loads(l) for f in files for l in open(f)]
    X, y, g = [], [], []
    for r in rows:
        for s in S:
            X.append(EV.feat(r['sites'][s]) + EV.limb_feat(r['sites'][s])); y.append(C4.index(r['labels']['0.10'][s])); g.append(r['scene'])
    X, y, g = np.array(X, float), np.array(y), np.array(g)
    C = fitC(X, y, g)
    P = np.zeros((len(y), 4))
    for i, j in GroupKFold(5).split(X, y, g):
        P[j] = proba(mk(C).fit(X[i], y[i]), X[j])
    m = mk(C).fit(X, y)
    sc, lr = m.named_steps['standardscaler'], m.named_steps['logisticregression']
    assert list(lr.classes_) == [0, 1, 2, 3], lr.classes_
    src = hashlib.sha256(b''.join(open(f, 'rb').read() for f in files)).hexdigest()
    doc = dict(schema='bt1-decision/1.0', classes=C4, sites=S, label_rule='0.10 (v3_compat)',
               features=['feat%d' % k for k in range(X.shape[1] - len(EV.LIMB_FEATURES))] + list(EV.LIMB_FEATURES),
               n_features=int(X.shape[1]), C=C, mean=sc.mean_.tolist(), scale=sc.scale_.tolist(),
               coef=lr.coef_.tolist(), intercept=lr.intercept_.tolist(), rows=int(len(y)), scenes=int(len(set(g))),
               oof_acc=round(float((P.argmax(1) == y).mean()), 4), in_sample_acc=round(float((m.predict(X) == y).mean()), 4),
               source_sha256=src)
    json.dump(doc, open(out, 'w'), indent=1)
    # the engine's own arithmetic must reproduce sklearn exactly
    import engine_bt1 as EB
    L = EB.DecisionLayer(out)
    d = np.abs(L.proba(X) - m.predict_proba(X)).max()
    assert d < 1e-9, d
    print('DECISION_OK', json.dumps({k: doc[k] for k in ('C', 'rows', 'oof_acc', 'in_sample_acc', 'n_features')}), 'max_diff', float(d))


if __name__ == '__main__':
    main(*sys.argv[1:3])
