"""Score an ICD predictions.json against scene sidecars (4-class). Usage: python score_preds.py predictions.json <scene_dir> [threshold=0.10] [wound_rule=visible|present]"""
import json, sys, os, collections
import numpy as np
C = ['no_injury', 'wound', 'amputation', 'not_testable']
M = {('upper_extremity', 'left'): 'LUE', ('upper_extremity', 'right'): 'RUE', ('lower_extremity', 'left'): 'LLE', ('lower_extremity', 'right'): 'RLE'}
pred, D = json.load(open(sys.argv[1])), sys.argv[2]
thr = sys.argv[3] if len(sys.argv) > 3 else '0.10'; rule = sys.argv[4] if len(sys.argv) > 4 else 'visible'
cm = np.zeros((4, 4), int); by = collections.defaultdict(lambda: np.zeros((4, 4), int))
for p in pred['predictions']:
    sc = json.load(open(os.path.join(D, p['image_id'].rsplit('.', 1)[0] + '_sidecar.json')))
    lab = sc['labels_by_threshold' if rule == 'visible' else 'labels_wound_if_present'][thr]
    for s in p['sites']:
        site = M[(s['body_region'], s['laterality'])]; g, q = C.index(lab[site]), C.index(s['injury_type'])
        cm[g, q] += 1
        g2 = sc['params'].get('garments', {}); key = 'clothed' if (g2.get('top', 'none') != 'none' or g2.get('bottom', 'none') != 'none') else 'unclothed'
        by[key][g, q] += 1
acc = lambda m: round(float(np.trace(m) / max(m.sum(), 1)), 4)
rec = lambda m: [round(float(m[i, i] / m[i].sum()), 3) if m[i].sum() else None for i in range(4)]
maj = lambda m: round(float(m[0].sum() / max(m.sum(), 1)), 4)
print(json.dumps({'n_sites': int(cm.sum()), 'accuracy': acc(cm), 'always_no_injury': maj(cm), 'recall[no_inj,wound,amp,not_test]': rec(cm),
                  'confusion': cm.tolist(), 'by_clothing': {k: {'n': int(m.sum()), 'accuracy': acc(m), 'always_no_injury': maj(m)} for k, m in by.items()}}, indent=1))
