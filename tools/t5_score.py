"""T5 single scoring run: engine models on the real manikin check set, against Joseph's labels (T5 Labeling Protocol
v1.0 section 4; T5 Labels v1.1 note). Prints and writes AGGREGATES only; per-image rows stay on the volume (G3).

Protocol (same as tools/bt2_eval.py for non-dev5 sets): for each model, the decision layer is fitted on all dev5 rows of
that model (C by grouped CV) and applied mirror-averaged to the T5 rows. Truth: usable photos, sites whose class is not
'unsure'. Two readings of not_testable (forum Q2 pending):
  as_labeled   classes as entered
  threshold    any site with visible share 'under 10%' counts as not_testable
Reported per model and reading: accuracy, per-class recall, confusion, wrong-side amputations (predicted amputation at a
site whose same limb on the other side is the labeled amputation while this site is not), by framing (full / most of
body vs partial) and by side confidence; paired fixed/broken sites against the first model.

usage: python tools/t5_score.py <out.json> <labels.csv> <model>=<t5_rows.jsonl>:<dev5_rows.jsonl>:<dev5_sidecar_dir> [...]
       image ids in the T5 rows are R####.jpg with #### = labels 'n'
"""
import sys, os, json, csv
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE] + [os.path.join(HERE, '..', p) for p in ('gen', 'jobs', 'score')]
from score import mk, proba, fitC
import engine_bt1 as EB
import bt2_eval as BE

C4 = ['no_injury', 'wound', 'amputation', 'not_testable']; S = ['LUE', 'RUE', 'LLE', 'RLE']
MIRROR = {'LUE': 'RUE', 'RUE': 'LUE', 'LLE': 'RLE', 'RLE': 'LLE'}


def labels(path):
    L = {}
    for r in csv.DictReader(open(path)):
        if r['usable'] != 'yes':
            continue
        L[int(r['n'])] = r
    return L


def truth(r, s, reading):
    c = r[s + '_class']
    if c not in C4:
        return None
    if reading == 'threshold' and r[s + '_visible'] == 'under 10%':
        return 'not_testable'
    return c


def t5_rows(path):
    out = {}
    for l in open(path):
        x = json.loads(l)
        if not (x.get('sites') and x.get('sites_flip')):
            continue
        n = int(''.join(ch for ch in os.path.splitext(x['image_id'])[0] if ch.isdigit()))
        out[n] = (np.array([EB.features(x['sites'][s]) for s in S], float), np.array([EB.features(x['sites_flip'][s]) for s in S], float))
    return out


def metrics(pairs):
    y = np.array([C4.index(t) for t, p, _ in pairs]); p = np.array([C4.index(q) for t, q, _ in pairs])
    if not len(y):
        return {'n': 0}
    cm = np.zeros((4, 4), int)
    for a, b in zip(y, p):
        cm[a, b] += 1
    rec = {C4[k]: (round(float(cm[k, k] / cm[k].sum()), 3), int(cm[k].sum())) for k in range(4) if cm[k].sum()}
    return {'n': int(len(y)), 'acc': round(float((y == p).mean()), 3), 'recall_n': rec, 'confusion_rows_truth': cm.tolist()}


def main(out, lab_path, specs):
    L = labels(lab_path)
    res, preds = {}, {}
    for sp in specs:
        model, rest = sp.split('=', 1); t5p, d5p, d5d = rest.split(':')
        X, XF, y, g = BE.load(d5p, d5d)
        C = fitC(X, y, g); full = mk(C).fit(X, y)
        R = t5_rows(t5p)
        P = {n: (proba(full, a) + proba(full, b)) / 2 for n, (a, b) in R.items()}
        preds[model] = {n: [C4[i] for i in q.argmax(1)] for n, q in P.items()}
        r = {'C': C, 'photos_with_rows': sum(1 for n in L if n in P), 'photos_usable': len(L)}
        for reading in ('as_labeled', 'threshold'):
            rows = []   # (truth, pred, meta)
            wrong_side = 0; amp_sites = 0
            for n, lab in L.items():
                if n not in P:
                    continue          # engine produced no row: reported as missing, not scored
                for k, s in enumerate(S):
                    t = truth(lab, s, reading)
                    if t is None:
                        continue
                    pr = preds[model][n][k]
                    rows.append((t, pr, dict(n=n, s=s, framing=lab['framing'], side=lab['side_conf'])))
                    if pr == 'amputation' and t != 'amputation':
                        o = truth(lab, MIRROR[s], reading)
                        wrong_side += int(o == 'amputation')
                    amp_sites += int(t == 'amputation')
            d = {'all': metrics(rows), 'wrong_side_amputations': wrong_side, 'amputation_sites': amp_sites}
            full_f = ('full body', 'most of body')
            d['framing_full'] = metrics([x for x in rows if x[2]['framing'] in full_f])
            d['framing_partial'] = metrics([x for x in rows if x[2]['framing'] not in full_f])
            for sc in ('sure', 'likely', 'unsure'):
                d['side_' + sc] = metrics([x for x in rows if x[2]['side'] == sc])
            r[reading] = d
        res[model] = r
    # paired against the first model, reading as_labeled
    base = specs[0].split('=', 1)[0]; paired = {}
    for m in preds:
        if m == base:
            continue
        fixed = broken = 0
        for n, lab in L.items():
            if n not in preds[m] or n not in preds[base]:
                continue
            for k, s in enumerate(S):
                t = truth(lab, s, 'as_labeled')
                if t is None:
                    continue
                a, b = preds[base][n][k] == t, preds[m][n][k] == t
                fixed += int(b and not a); broken += int(a and not b)
        paired[f'{base}->{m}'] = {'fixed': fixed, 'broken': broken, 'p': round(BE.sign_p(fixed, broken), 4)}
    res['_paired_as_labeled'] = paired
    json.dump(res, open(out, 'w'), indent=1)
    for m, r in res.items():
        if m.startswith('_'):
            continue
        for reading in ('as_labeled', 'threshold'):
            a = r[reading]['all']
            print('T5SCORE', m, reading, json.dumps({'n': a.get('n'), 'acc': a.get('acc'), 'recall': a.get('recall_n'),
                  'wrong_side_amp': r[reading]['wrong_side_amputations'], 'full': r[reading]['framing_full'].get('acc'),
                  'partial': r[reading]['framing_partial'].get('acc'), 'photos': r['photos_with_rows']}), flush=True)
    print('T5PAIRED', json.dumps(paired), flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
