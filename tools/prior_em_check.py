"""Item 3 check (CR-2026-09-28-image-competitive-gaps): EM prior correction on dev5 resampled to other class mixes.
Out-of-fold BT-1 probabilities (limb features, grouped 5-fold); sites resampled (with replacement) to each target mix;
20 draws of 1,000 sites. Reports accuracy, per-class recall and the M3-05 floor check with and without correction.
usage: python tools/prior_em_check.py <dddata_root> <out.json>"""
import sys, os, json, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'adapt')); sys.path.insert(0, os.path.join(HERE, '..', 'container'))
import kit as K
from score import fitC, mk, proba
from sklearn.model_selection import GroupKFold
from d2qual.prior import em_prior

root, out = sys.argv[1:3]
rows = K.load_rows(os.path.join(root, 'results/bt1/ext/bt1_t*_sidec.jsonl'))
lab, _ = K.load_labels(os.path.join(root, 'dev5'), 'synthetic', '0.10')
X, _, y, g, _ = K.design(rows, lab, {})
C = fitC(X, y, g); P = np.zeros((len(y), 4))
for tr, te in GroupKFold(5).split(X, y, g):
    P[te] = proba(mk(C).fit(X[tr], y[tr]), X[te])
pi_train = np.bincount(y, minlength=4) / len(y)
MIX = {'dev5_as_is': pi_train.tolist(), 'injury_heavy': [0.45, 0.15, 0.20, 0.20], 'mostly_intact': [0.75, 0.07, 0.10, 0.08],
       'few_not_testable': [0.69, 0.11, 0.13, 0.07], 'wound_amp_heavy': [0.40, 0.25, 0.25, 0.10]}
FLOOR = 0.5   # M3-05 recall floor used by the decision layer fit (d2qual/decision.py)
rng = np.random.default_rng(0); idx = {k: np.nonzero(y == k)[0] for k in range(4)}; res = {'pi_train': pi_train.round(3).tolist(), 'mixes': {}}
for name, mix in MIX.items():
    A0, A1, R0, R1, EST = [], [], [], [], []
    for d in range(20):
        n = (np.array(mix) * 1000).round().astype(int)
        sel = np.concatenate([rng.choice(idx[k], n[k]) for k in range(4)])
        p, yy = P[sel], y[sel]
        q, pi = em_prior(p, pi_train)
        for Pk, Acc, Rec in ((p, A0, R0), (q, A1, R1)):
            pr = Pk.argmax(1); Acc.append((pr == yy).mean()); Rec.append([(pr[yy == k] == k).mean() for k in range(4)])
        EST.append(pi)
    r0, r1 = np.mean(R0, 0), np.mean(R1, 0)
    res['mixes'][name] = dict(target=mix, estimated=np.mean(EST, 0).round(3).tolist(),
                              acc_off=round(float(np.mean(A0)), 4), acc_on=round(float(np.mean(A1)), 4),
                              gain_points=round(100 * float(np.mean(np.array(A1) - np.array(A0))), 2),
                              gain_sd_points=round(100 * float(np.std(np.array(A1) - np.array(A0))), 2),
                              recall_off=dict(zip(K.C4, r0.round(3).tolist())), recall_on=dict(zip(K.C4, r1.round(3).tolist())),
                              floor_ok_on=bool((r1 >= FLOOR).all()), min_recall_off=round(float(r0.min()), 3), min_recall_on=round(float(r1.min()), 3))
    print(name, res['mixes'][name]['acc_off'], '->', res['mixes'][name]['acc_on'], 'min recall', res['mixes'][name]['min_recall_off'], '->', res['mixes'][name]['min_recall_on'], 'est', res['mixes'][name]['estimated'])
json.dump(res, open(out, 'w'), indent=1)
