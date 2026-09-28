"""Test-time mirror on dev3: combine the decision layer's probabilities on the original and mirrored extractions.
usage: python jobs/tta_score.py <gen_dir> '<orig glob>' '<flip glob>' '<side-truth glob>' <out.json>
Layer = eval_v3.feat + logistic regression fit on ORIGINAL features only (grouped 5-fold OOF, C by CV), as in score.py.
LIMB=1 appends the BT-1 limb features. Reports the M3-02 measure without test-time mirroring: the share of sites where
any of the four class probabilities moves by more than 0.05 between the image and its mirror (after the site swap)."""
import sys, os, json, glob, numpy as np
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'score'))
import eval_v3 as EV
from score import fitC, mk, proba, metrics, C4, S
from sklearn.model_selection import GroupKFold
LIMB = os.environ.get('LIMB') == '1'
F_ = (lambda r: EV.feat(r) + EV.limb_feat(r)) if LIMB else EV.feat
def load(g):
    return {json.loads(l)['scene']: json.loads(l) for f in sorted(glob.glob(g)) for l in open(f)}
O, Fl, Ce = load(sys.argv[2]), load(sys.argv[3]), load(sys.argv[4])
sc = sorted(set(O) & set(Fl))
X, XF, XC, y, g = [], [], [], [], []
for s in sc:
    for site in S:
        X.append(F_(O[s]['sites'][site])); XF.append(F_(Fl[s]['sites'][site]))
        XC.append((EV.feat(Ce[s]['sites'][site]) + (EV.limb_feat(O[s]['sites'][site]) if LIMB else [])) if s in Ce else F_(O[s]['sites'][site]))
        y.append(C4.index(O[s]['labels']['0.10'][site])); g.append(s)
X, XF, XC, y, g = map(np.array, (X, XF, XC, y, g))
C = fitC(X, y, g); P, PF, PC = np.zeros((len(y), 4)), np.zeros((len(y), 4)), np.zeros((len(y), 4))
for i, j in GroupKFold(5).split(X, y, g):
    m = mk(C).fit(X[i], y[i]); P[j] = proba(m, X[j]); PF[j] = proba(m, XF[j]); PC[j] = proba(m, XC[j])
po, pf, pa = P.argmax(1), PF.argmax(1), ((P + PF) / 2).argmax(1)
side_fixable = (po != y) & (PC.argmax(1) == y)
dis = po != pf
dmax = np.abs(P - PF).max(1)
res = {'limb': LIMB, 'm3_02_sites_over_0.05': round(float((dmax > 0.05).mean()), 4),
       'm3_02_max_prob_diff_p50_p90_p99': [round(float(np.percentile(dmax, q)), 4) for q in (50, 90, 99)],
       'scenes': len(sc), 'sites': int(len(y)), 'C': C,
       'original': metrics(y, po), 'mirrored_only': metrics(y, pf), 'average': metrics(y, pa),
       'rendered_side_ceiling_acc': round(float((PC.argmax(1) == y).mean()), 4),
       'disagreement_rate': round(float(dis.mean()), 4),
       'error_rate_when_disagree_original': round(float((po[dis] != y[dis]).mean()), 3) if dis.any() else None,
       'error_rate_when_agree': round(float((po[~dis] != y[~dis]).mean()), 3),
       'side_fixable_errors': int(side_fixable.sum()), 'side_fixable_fixed_by_average': int((side_fixable & (pa == y)).sum()),
       'fixed_by_average': int(((po != y) & (pa == y)).sum()), 'broken_by_average': int(((po == y) & (pa != y)).sum())}
json.dump(res, open(sys.argv[5], 'w'), indent=1); print('TTA_RES', json.dumps(res))
