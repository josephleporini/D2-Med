"""Interim ICD v1.0 output check (until JHU/APL's validator is released). Usage: python validate_predictions.py predictions.json [input_dir]"""
import json, sys, os, math
SITES = {('upper_extremity', 'left'), ('upper_extremity', 'right'), ('lower_extremity', 'left'), ('lower_extremity', 'right')}
CLASSES = {'no_injury', 'wound', 'amputation', 'not_testable'}
errs = []
try:
    d = json.load(open(sys.argv[1], encoding='utf-8'), parse_constant=lambda c: errs.append(f'nonstandard constant {c}'))
except Exception as e:
    print('INVALID JSON', e); sys.exit(1)
if d.get('schema_version') != '1.0': errs.append('$.schema_version must be "1.0"')
for k in ('team_name', 'version', 'email'):
    if not isinstance(d.get('submission', {}).get(k), str): errs.append(f'$.submission.{k} missing or not a string')
preds = d.get('predictions')
if not isinstance(preds, list): errs.append('$.predictions missing'); preds = []
seen = set()
for i, p in enumerate(preds):
    if not isinstance(p.get('image_id'), str): errs.append(f'$.predictions[{i}].image_id')
    seen.add(p.get('image_id'))
    s = p.get('sites', [])
    if len(s) != 4: errs.append(f'$.predictions[{i}].sites has {len(s)} entries')
    got = set()
    for j, x in enumerate(s):
        got.add((x.get('body_region'), x.get('laterality')))
        if x.get('injury_type') not in CLASSES: errs.append(f'$.predictions[{i}].sites[{j}].injury_type')
    if got != SITES: errs.append(f'$.predictions[{i}].sites do not cover the four sites exactly once')
if len(sys.argv) > 2:
    imgs = {f for f in os.listdir(sys.argv[2]) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
    for m in sorted(imgs - seen): errs.append(f'missing prediction for {m}')
for e in errs: print('ERROR', e)
print('OK' if not errs else f'{len(errs)} violations'); sys.exit(0 if not errs else 2)
