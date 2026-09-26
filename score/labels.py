"""Offline label rules: raw truth (scene4 sidecar) -> 4-class site labels under each named rule (spec v1.0 section 4,
v1.1 section 10.1). Changing a rule changes this file and LABEL_RULES_VERSION, never the renders.

usage: python score/labels.py <scene_dir> [rule ...]   -> prints a label table and counts per rule
import: labels.label(sidecar, rule) -> {site: class}
"""
import sys, os, json, glob

LABEL_RULES_VERSION = 'labels/1.0'
SITES = ['LUE', 'RUE', 'LLE', 'RLE']
WOUND_MIN_PX = 20
STUMP_MIN_PX = 20
RULES = {
    # name: (visibility cutoff on visible fraction, wound rule, amputation rule)
    'v3_compat': (0.10, 'visible', 'any'),
    'guide_primary': (0.0, 'visible', 'observable'),
    'guide_wound_present': (0.0, 'present', 'observable'),
    'guide_amp_any': (0.0, 'visible', 'any'),
}


def site_label(t, term, rule):
    cut, wr, ar = RULES[rule]
    vf = t['visible_fraction']
    if not (vf > 0 if cut == 0 else vf >= cut):
        return 'not_testable'
    if t['amputated']:
        if ar == 'any':
            return 'amputation'
        observable = t['stump_visible_px'] >= STUMP_MIN_PX or term['cause'] == 'amputated_visible'
        return 'amputation' if observable else 'not_testable'
    if t['wound_present']:
        if wr == 'present' or t['wound_visible_px'] >= WOUND_MIN_PX:
            return 'wound'
    return 'no_injury'


def label(sc, rule):
    return {s: site_label(sc['truth'][s], sc['terminal'][s], rule) for s in SITES}


if __name__ == '__main__':
    D = sys.argv[1]; rules = sys.argv[2:] or list(RULES)
    cars = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(D, '*_sidecar.json')))]
    cars = [c for c in cars if 'truth' in c]
    print('scene', *rules, sep='\t')
    for c in cars:
        print(c['scene_id'], *[' '.join(f'{s}:{v[:4]}' for s, v in label(c, r).items()) for r in rules], sep='\t')
    for r in rules:
        cnt = {}
        for c in cars:
            for v in label(c, r).values():
                cnt[v] = cnt.get(v, 0) + 1
        print('COUNTS', r, cnt)
