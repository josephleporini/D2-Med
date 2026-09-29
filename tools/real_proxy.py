"""Label-free real-photo proxy (BT-2b): how each model's predictions on the T5 RESERVE photos line up with the photo-level
screening class from the candidate workbook. Aggregates only; the per-image rows stay on the volume (G3).

The screening class is a quick human look at the whole photo, not a per-limb label, so this is a proxy, not a score:
  wound-screened photo        a typical photo has one or two wounded limbs, so ~1-2 predicted wound limbs is plausible
                              and 4 of 4 is the BT-1 failure seen on 29 Sep (86% of limbs called wound)
  not-testable-screened       limbs mostly out of view: predicted not_testable share should be high
  amputation-screened         at least one limb predicted amputation
usage: python tools/real_proxy.py <out.json> <model>=<measure.jsonl> [...]   (image names R####.jpg with _src.json beside
       the photos are not needed: the class comes from real/reserve_urls.csv by number)
"""
import sys, os, json, csv, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SC = {int(r['n']): r['screening_class'] for r in csv.DictReader(open(os.path.join(HERE, '..', 'real', 'reserve_urls.csv')))}
GROUPS = {'wound': ('wound',), 'mix': ('wound/not-testable mix',), 'not_testable': ('not testable',),
          'amputation': ('amputation',), 'unclear': ('unclear',)}


def summarize(rows):
    out = {'n_photos': len(rows)}
    allp = [p for r in rows for p in r['pred']]
    out['pred_mix'] = {c: round(float(np.mean([p == c for p in allp])), 3) for c in ('no_injury', 'wound', 'amputation', 'not_testable')}
    out['wound_limbs_per_photo'] = round(float(np.mean([sum(p == 'wound' for p in r['pred']) for r in rows])), 2)
    for g, keys in GROUPS.items():
        rs = [r for r in rows if SC.get(int(re.sub(r'\D', '', r['image']) or -1)) in keys]
        if not rs:
            continue
        d = {'n': len(rs),
             'wound_limbs_per_photo': round(float(np.mean([sum(p == 'wound' for p in r['pred']) for r in rs])), 2),
             'all4_wound': round(float(np.mean([all(p == 'wound' for p in r['pred']) for r in rs])), 3),
             'not_testable_share': round(float(np.mean([p == 'not_testable' for r in rs for p in r['pred']])), 3),
             'any_amputation': round(float(np.mean([any(p == 'amputation' for p in r['pred']) for r in rs])), 3)}
        out[g] = d
    out['n_person_median'] = float(np.median([r['n_person'] for r in rows]))
    return out


if __name__ == '__main__':
    res = {}
    for a in sys.argv[2:]:
        m, p = a.split('=', 1)
        res[m] = summarize([json.loads(l) for l in open(p)])
    json.dump(res, open(sys.argv[1], 'w'), indent=1)
    for m, r in res.items():
        print('REAL_PROXY', m, json.dumps(r), flush=True)
