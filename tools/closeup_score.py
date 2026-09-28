"""Score the close-up robustness set with the shipped BT-1 decision layer (no refit: this measures how the adopted
model behaves on a framing it was not trained for). Mirror-averaged, as the container runs.
usage: python tools/closeup_score.py <closeup_dir> '<orig rows glob>' '<flip rows glob>' <decision.json> <dev5 ledger> <out.json>
Reports accuracy and per-class recall at the 0.10 rule; the same casualties' dev5 results for comparison (labels differ
where a limb leaves the frame); wrong-side errors (pair swapped); accuracy for the limb in focus vs the other limbs."""
import sys, os, json, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'gen'), os.path.join(HERE, '..', 'jobs')]
import engine_bt1 as EB
C4 = ['no_injury', 'wound', 'amputation', 'not_testable']; S = EB.SITES; MIR = {'LUE': 'RUE', 'RUE': 'LUE', 'LLE': 'RLE', 'RLE': 'LLE'}
d, go, gf, dec, ledger, out = sys.argv[1:7]
def rows(g):
    return {json.loads(l)['scene']: json.loads(l) for f in glob.glob(g) for l in open(f)}
O, Fl = rows(go), rows(gf); L = EB.DecisionLayer(dec)
cars = {json.load(open(f))['scene_id']: json.load(open(f)) for f in glob.glob(os.path.join(d, 'DK*_sidecar.json'))}
led = {}
for l in open(ledger):
    r = json.loads(l); led[(r['scene'], r['site'])] = r
res = {'scenes': 0, 'sites': 0}; y, p, focus, base_ok, cls_pairs = [], [], [], [], []
for sid in sorted(set(O) & set(Fl) & set(cars)):
    c = cars[sid]; prm = c['params']; lab = c['labels_by_threshold']['0.10']
    P = 0.5 * (L.proba([EB.features(O[sid]['sites'][s]) for s in S]) + L.proba([EB.features(Fl[sid]['sites'][s]) for s in S]))
    pr = P.argmax(1); res['scenes'] += 1
    for k, s in enumerate(S):
        y.append(C4.index(lab[s])); p.append(int(pr[k])); focus.append(s == prm['closeup_site'])
        b = led.get((prm['base_scene'], s)); base_ok.append(None if b is None else bool(b['correct']))
    for a, b_ in (('LUE', 'RUE'), ('LLE', 'RLE')):
        ia, ib = S.index(a), S.index(b_); ta, tb = C4.index(lab[a]), C4.index(lab[b_])
        cls_pairs.append((ta != tb) and pr[ia] == tb and pr[ib] == ta)
y, p, focus = np.array(y), np.array(p), np.array(focus)
def m(mask):
    yy, pp = y[mask], p[mask]
    return dict(n=int(mask.sum()), acc=round(float((yy == pp).mean()), 4),
                recall={C4[k]: round(float((pp[yy == k] == k).mean()), 3) for k in range(4) if (yy == k).any()},
                mix={C4[k]: int((yy == k).sum()) for k in range(4)})
res['sites'] = int(len(y)); res['all'] = m(np.ones(len(y), bool)); res['focus_limb'] = m(focus); res['other_limbs'] = m(~focus)
bo = [v for v in base_ok if v is not None]
res['same_casualties_dev5_acc'] = round(float(np.mean(bo)), 4) if bo else None
res['pairs_swapped'] = int(sum(cls_pairs)); res['pairs'] = len(cls_pairs)
json.dump(res, open(out, 'w'), indent=1); print('CLOSEUP', json.dumps(res))
