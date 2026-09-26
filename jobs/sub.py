import sys, json, numpy as np
sys.path.insert(0, sys.argv[1]); import eval_v3 as EV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
C4 = EV.C4; R = json.load(open(sys.argv[2]))['dev3']
y = np.array([C4.index(r['lab10']) for r in R]); g = np.array([r['scene'] for r in R])
def cv(X, cw=None):
    best = None
    for C in [0.03, 0.1, 0.3, 1, 3]:
        pr = np.zeros(len(y), int)
        for i, j in GroupKFold(5).split(X, y, g):
            pr[j] = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight=cw)).fit(X[i], y[i]).predict(X[j])
        a = (pr == y).mean()
        if best is None or a > best[0]: best = (a, pr)
    a, pr = best
    return [round(float(a), 4), round(min(float((pr[y == k] == k).mean()) for k in range(4)), 3)]
P = lambda r: r['p']
sets = {'none': lambda p: [],
        'distal': lambda p: [float(p['distal_n'] > 0)],
        'distal+stump_end': lambda p: [float(p['distal_n'] > 0), p['stump_end']],
        'min4': lambda p: [float(p['distal_n'] > 0), p['stump_end'], float(p['after'] == 'BG'), min(p['end_frac'], 1.5)],
        'min4+tq': lambda p: [float(p['distal_n'] > 0), p['stump_end'], float(p['after'] == 'BG'), min(p['end_frac'], 1.5), float(p['tq_n'] > 0), p['wound_near_tq'], p['wound_distal_tq']]}
out = {}
for mp in ('pr', 'gt'):
    B = np.array([EV.feat(r[mp]) for r in R])
    for k, f in sets.items():
        X = np.c_[B, np.array([f(P(r)) for r in R])] if k != 'none' else B
        out[f'{mp}_{k}'] = {'unw': cv(X), 'bal': cv(X, 'balanced')}
# rule check: amputation veto when distal present, on pred-map base model OOF
print('SUB_RES', json.dumps(out), flush=True)
