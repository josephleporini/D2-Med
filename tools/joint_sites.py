"""Joint four-limb decision test on dev5 (CPU, no training on images). Treats the four sites of one image together
instead of one at a time, in two ways:

  context  each site's decision also sees its mirror limb (LUE<->RUE, LLE<->RLE), the two other limbs and a site code
  (result 28 Sep 2026, dev5: both arms lose to the base layer; see D2_BlockT_Decision_Memo_v1.0. The swap detector
   reaches AUC 0.69 only, because the extraction rows carry no per-limb side evidence.)
  swap     a pair-level association step (track-to-target style): a classifier trained on the rendered-side rows decides
           whether the left/right evidence of a limb pair has been assigned the wrong way round; if so the two feature
           rows are exchanged before the decision layer

usage: python tools/joint_sites.py <dddata_root> <out.json>
  uses results/bt1/ext/bt1_t*_{sidec}.jsonl rows (model features) from the BT-1 run; the rendered-side rows
  (_ceil) come from the pod volume and are not in DDData, so the swap arm needs --ceil <glob> (skipped otherwise).
All arms: grouped 5-fold out-of-fold by scene, same logistic decision layer as score/score.py, v3_compat labels.
"""
import sys, os, json, glob, argparse
import numpy as np
from sklearn.model_selection import GroupKFold
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'gen')); sys.path.insert(0, os.path.join(HERE, '..', 'score'))
import eval_v3 as EV
import labels as LB
from score import fitC, mk, proba, metrics, C4

S = ['LUE', 'RUE', 'LLE', 'RLE']
MIRROR = {'LUE': 'RUE', 'RUE': 'LUE', 'LLE': 'RLE', 'RLE': 'LLE'}
OTHER = {'LUE': ('LLE', 'RLE'), 'RUE': ('LLE', 'RLE'), 'LLE': ('LUE', 'RUE'), 'RLE': ('LUE', 'RUE')}
CODE = {'LUE': [1, 0, 1], 'RUE': [1, 0, 0], 'LLE': [0, 1, 1], 'RLE': [0, 1, 0]}


def load(pattern):
    out = {}
    for f in glob.glob(pattern):
        for l in open(f):
            r = json.loads(l); out[r['scene']] = r
    return out


def F(site_row):
    return EV.feat(site_row) + EV.limb_feat(site_row)


def oof(X, y, g):
    C = fitC(X, y, g); P = np.zeros((len(y), 4))
    for i, j in GroupKFold(5).split(X, y, g):
        P[j] = proba(mk(C).fit(X[i], y[i]), X[j])
    return P


def mcnemar(a_ok, b_ok):
    from math import comb
    b = int((a_ok & ~b_ok).sum()); c = int((~a_ok & b_ok).sum()); n = b + c
    p = 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n if n else 1.0
    return {'fixed': c, 'broken': b, 'p': round(min(p, 1.0), 4)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('root'); ap.add_argument('out'); ap.add_argument('--ceil', default='')
    A = ap.parse_args()
    ext = load(os.path.join(A.root, 'results/bt1/ext/bt1_t*_sidec.jsonl'))
    cars = {}
    for f in glob.glob(os.path.join(A.root, 'dev5/batch_*/D*_sidecar.json')):
        c = json.load(open(f)); cars[c['scene_id']] = c
    scenes = sorted(ext); rows = [(sid, s) for sid in scenes for s in S]
    y = np.array([C4.index(LB.label(cars[sid], 'v3_compat')[s]) for sid, s in rows]); g = np.array([sid for sid, _ in rows])
    feats = {sid: {s: F(ext[sid]['sites'][s]) for s in S} for sid in scenes}
    Xb = np.array([feats[sid][s] for sid, s in rows])
    Xc = np.array([feats[sid][s] + feats[sid][MIRROR[s]] + list(np.mean([feats[sid][o] for o in OTHER[s]], 0)) + CODE[s] for sid, s in rows])
    res = {'sites': len(rows), 'arms': {}}
    Pb = oof(Xb, y, g); Pc = oof(Xc, y, g)
    ok_b, ok_c = Pb.argmax(1) == y, Pc.argmax(1) == y
    res['arms']['base'] = metrics(y, Pb.argmax(1)); res['arms']['context'] = metrics(y, Pc.argmax(1))
    res['context_vs_base'] = mcnemar(ok_b, ok_c)
    # mirrored pass (BT-1 _bt1f rows are not in DDData either); if present locally, average probabilities
    if A.ceil:
        ceil = load(A.ceil)
        # pair swap labels: is the predicted assignment closer to the rendered-side rows crossed than straight?
        Xall = np.array([feats[sid][s] for sid, s in rows]); mu, sd = Xall.mean(0), Xall.std(0) + 1e-6
        z = lambda v: (np.array(v) - mu) / sd
        pairs, plab, pg = [], [], []
        for sid in scenes:
            if sid not in ceil:
                continue
            for a, b in (('LUE', 'RUE'), ('LLE', 'RLE')):
                pa, pb = z(feats[sid][a]), z(feats[sid][b]); ca, cb = z(F(ceil[sid]['sites'][a])), z(F(ceil[sid]['sites'][b]))
                crossed = np.linalg.norm(pa - cb) + np.linalg.norm(pb - ca) < np.linalg.norm(pa - ca) + np.linalg.norm(pb - cb) - 1e-6
                pairs.append(list(pa) + list(pb) + list(pa - pb) + [1 if a == 'LUE' else 0]); plab.append(int(crossed)); pg.append(sid)
        pairs, plab, pg = np.array(pairs), np.array(plab), np.array(pg)
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        ps = np.zeros(len(plab))
        for i, j in GroupKFold(5).split(pairs, plab, pg):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000, class_weight='balanced')).fit(pairs[i], plab[i])
            ps[j] = m.predict_proba(pairs[j])[:, 1]
        res['swap_pairs'] = {'pairs': int(len(plab)), 'crossed_truth': int(plab.sum())}
        for thr in (0.6, 0.7, 0.8, 0.9):
            sw = {(sid, a): p > thr for sid, a, p in zip(pg, ['UE' if r[-1] else 'LE' for r in pairs], ps)}
            def feat_after(sid, s):
                if sw.get((sid, s[1:])):
                    return feats[sid][MIRROR[s]]
                return feats[sid][s]
            Xs = np.array([feat_after(sid, s) for sid, s in rows]); Pss = oof(Xs, y, g)
            res['arms'][f'swap_thr{thr}'] = dict(metrics(y, Pss.argmax(1)), pairs_swapped=int(sum(sw.values())),
                                                 swaps_correct=int(sum(1 for (k, v), l in zip(sw.items(), plab) if v and l)))
            res[f'swap_thr{thr}_vs_base'] = mcnemar(ok_b, Pss.argmax(1) == y)
    json.dump(res, open(A.out, 'w'), indent=1); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
