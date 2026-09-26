"""Learned decision layer, fitted on the development set only, applied once to the test set (pose env).
Usage: python decision_layer.py <dev_sites.json> <test_sites.json> <arm> <out.json>

Inputs are image-derived only (no generator parameters):
  log(1+vis_px), Lfrac, bg_frac, limb-end state probabilities (visible / stump / hidden), log(1+hand-foot px),
  log(1+stump px), end-ring composition (BG, OCC, TORSO, OTHER_LIMB, EDGE shares), vis_px / torso_extent^2,
  window present flag.
Model: multinomial logistic regression (standardised features); regularisation C chosen by 5-fold cross-validation
grouped by scene on the dev set. Reported against rule F on the same test rows.
"""
import sys, json, collections
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold

C = ['no_injury', 'amputation', 'not_testable']
RING = ['BG', 'OCC', 'TORSO', 'OTHER_LIMB', 'EDGE']
FEATURES = ['log_vis_px', 'Lfrac', 'bg_frac', 'p_visible', 'p_stump', 'p_hidden', 'log_ext_px', 'log_stump_px'] + \
    ['ring_' + k for k in RING] + ['vis_px_over_torso2', 'window_present']


def feats(r):
    p = r['p'] or [0.0, 0.0, 0.0]
    end = r.get('end') or {}; tot = sum(end.values()) or 1
    te = max(r.get('torso_ext') or 0, 1)
    return [np.log1p(r['vis_px']), r['Lfrac'], r['bg_frac'] if r['bg_frac'] is not None else -1, *p,
            np.log1p(r['ext_px']), np.log1p(r['stump_px']), *[end.get(k, 0) / tot for k in RING],
            r['vis_px'] / te ** 2, float(r['p'] is not None)]


def rows(path, arm):
    R = [r for r in json.load(open(path)) if r['arm'] == arm]
    return np.array([feats(r) for r in R]), np.array([C.index(r['gt']) for r in R]), [r['scene'] for r in R], R


def summarise(y, p):
    m = np.zeros((3, 3), int)
    for a, b in zip(y, p):
        m[a, b] += 1
    return {'accuracy': round(float(np.trace(m) / m.sum()), 4), 'recall': [round(float(m[i, i] / max(m[i].sum(), 1)), 3) for i in range(3)],
            'confusion': m.tolist()}


if __name__ == '__main__':
    dev, test, arm, out = sys.argv[1:5]
    Xd, yd, gd, _ = rows(dev, arm); Xt, yt, _, Rt = rows(test, arm)
    best = None
    for Cr in [0.03, 0.1, 0.3, 1, 3, 10]:
        acc = []
        for tr, va in GroupKFold(5).split(Xd, yd, gd):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=Cr, max_iter=2000)).fit(Xd[tr], yd[tr])
            acc.append((m.predict(Xd[va]) == yd[va]).mean())
        if best is None or np.mean(acc) > best[1]:
            best = (Cr, float(np.mean(acc)))
    model = make_pipeline(StandardScaler(), LogisticRegression(C=best[0], max_iter=2000)).fit(Xd, yd)
    pt = model.predict(Xt)
    ruleF = [C.index(r['F']) for r in Rt]
    res = {'arm': arm, 'dev_sites': len(yd), 'C': best[0], 'dev_cv_accuracy': round(best[1], 4),
           'test_learned': summarise(yt, pt), 'test_ruleF': summarise(yt, ruleF),
           'test_majority': summarise(yt, [0] * len(yt))}
    sc_, lr = model.named_steps['standardscaler'], model.named_steps['logisticregression']
    res['model'] = {'classes': C, 'mean': sc_.mean_.tolist(), 'scale': sc_.scale_.tolist(),
                    'coef': lr.coef_.tolist(), 'intercept': lr.intercept_.tolist(), 'features': FEATURES}
    print(json.dumps({k: v for k, v in res.items() if k != 'model'}, indent=1)); json.dump(res, open(out, 'w'), indent=1)
