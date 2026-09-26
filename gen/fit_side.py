"""Fit on dev3, score test4 once, for old vs kp side rule, each arm; balanced and unweighted LR. Prints JSON."""
import sys, json, numpy as np, collections
sys.path.insert(0, '.')
import eval_v3 as EV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
P = sys.argv[1]
out = {}
for arm in ('gt', 'pred'):
    for cmn in ('old', 'kp'):
        dev, test = (f'{P}/dev3_{arm}_{cmn}.jsonl', f'{P}/test4_{arm}_{cmn}.jsonl') if cmn == 'kp' else (f'{P}/../eval3p/dev3_{arm}.jsonl', f'{P}/../eval3p/test4_{arm}.jsonl')
        Xd, yd, gd, _ = EV.table(dev, '0.10'); Xt, yt, _, Rt = EV.table(test, '0.10')
        pos = {json.loads(l)['scene']: json.loads(l).get('position') for l in open(f'{P}/test4_{arm}_kp.jsonl')}
        for cw in ('balanced', None):
            best = None
            for C in [0.03, 0.1, 0.3, 1, 3, 10]:
                acc = [(make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight=cw)).fit(Xd[a], yd[a]).predict(Xd[b]) == yd[b]).mean()
                       for a, b in GroupKFold(5).split(Xd, yd, gd)]
                if best is None or np.mean(acc) > best[1]:
                    best = (C, float(np.mean(acc)))
            m = make_pipeline(StandardScaler(), LogisticRegression(C=best[0], max_iter=5000, class_weight=cw)).fit(Xd, yd)
            p = m.predict(Xt); s = EV.summary(yt, p, Rt); cm = np.array(s['confusion_rows_truth'])
            bypos = collections.defaultdict(list)
            for r, a, b in zip(Rt, yt, p):
                bypos[str(pos.get(r['scene']))].append(a == b)
            out[f'{arm}|{cmn}|{cw}'] = dict(C=best[0], dev_cv=round(best[1], 4), acc=s['accuracy'], recall=s['recall'], macro_f1=s['macro_f1_present_classes'],
                                           nt_boundary_err=int(cm[3, :3].sum() + cm[:3, 3].sum()), lat_swaps=s['laterality_swaps'],
                                           by_position={k: [len(v), round(float(np.mean(v)), 3)] for k, v in sorted(bypos.items())})
print('SIDE_FIT', json.dumps(out))
