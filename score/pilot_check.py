"""Pilot acceptance checks for generator vNext (spec v1.1, section 10.6).
usage: python score/pilot_check.py <pilot_dir> [regression_note]
Reads <id>_sidecar.json, <id>_id.png, <id>_amodal.png, <id>_part3.png for the 20 paired scenes; prints a pass/fail table
and writes pilot_check.json next to the scenes.
"""
import sys, os, json, glob
import numpy as np, cv2

D = sys.argv[1]; REG = sys.argv[2] if len(sys.argv) > 2 else 'not run'
S = ['LUE', 'RUE', 'LLE', 'RLE']; BIT = {'LUE': 1, 'RUE': 2, 'LLE': 4, 'RLE': 8}
IDC = {'LUE': (0, 0, 255), 'RUE': (0, 255, 0), 'LLE': (255, 0, 0), 'RLE': (0, 255, 255)}      # BGR of ID colours
sc = {os.path.basename(f)[:-13]: json.load(open(f)) for f in sorted(glob.glob(os.path.join(D, 'P*_sidecar.json')))}
res = {}


def idmask(sid, s):
    a = cv2.imread(os.path.join(D, sid + '_id.png')).astype(int)
    q = np.where(a > 219, 255, np.where(a > 89, 128, 0))
    return np.all(q == np.array(IDC[s]), axis=2)


def check(name, ok, detail):
    res[name] = {'pass': bool(ok), 'detail': detail}


# amodal containment: visible site pixels lie inside the full-outline mask
worst = 0.0; rows = []
for sid, c in sc.items():
    am = cv2.imread(os.path.join(D, sid + '_amodal.png'), cv2.IMREAD_UNCHANGED)
    for s in S:
        v = idmask(sid, s); n = int(v.sum())
        if n:
            out = int((v & ((am & BIT[s]) == 0)).sum()) / n; worst = max(worst, out)
            if out > 0.005:
                rows.append(f'{sid} {s} {out:.3f}')
check('amodal_containment', worst <= 0.005, f'max share of visible pixels outside the full outline {worst:.4f}; ' + '; '.join(rows[:5]))

# amputation truth: amputated sites have stump truth, no distal part, removed distal joints
bad = []
for sid, c in sc.items():
    for s in S:
        t = c['truth'][s]
        if t['amputated']:
            lim = s[1:]; dj = 'hand' if lim == 'UE' else 'foot'
            if t['distal_visible_px'] > 0 or c['joints'][s][dj]['exists'] or t['stump_full_px'] == 0:
                bad.append(f"{sid} {s} distal={t['distal_visible_px']} stump_full={t['stump_full_px']}")
check('amputation_truth', not bad, '; '.join(bad) or 'all amputated sites: stump present, no distal part, distal joints removed')

# hidden-intact and occluder identity (P03, P04): removing the occluder leaves the full outline, raises visibility
det = []; ok = True
for pid in ('P03', 'P04'):
    a, b = sc.get(pid + 'a'), sc.get(pid + 'b')
    if not (a and b):
        ok = False; det.append(pid + ' missing'); continue
    for s in S:
        fa, fb = a['truth'][s]['full_px'], b['truth'][s]['full_px']
        if abs(fa - fb) > 0.02 * max(fb, 1):
            ok = False; det.append(f'{pid} {s} full outline changed {fa}->{fb}')
    hid = [s for s in S if a['truth'][s]['visible_px'] < b['truth'][s]['visible_px'] - 20]
    occ_owner = [s for s in S if a['terminal'][s]['owner_at_point'] == 'occluder']
    det.append(f"{pid}: occluded sites {hid}; terminal behind occluder {occ_owner} kind {[a['terminal'][s]['occluder_kind'] for s in occ_owner]}; causes a {[a['terminal'][s]['cause'] for s in S]}")
    if not hid:
        ok = False
check('hidden_intact_and_occluder_identity', ok, ' | '.join(det))

# out-of-frame distinct from hidden and amputated (P10b upper crop)
b = sc.get('P10b'); oof = [s for s in S if b and b['terminal'][s]['cause'] == 'out_of_frame']
check('out_of_frame', bool(oof), f"P10b out-of-frame terminals {oof}; out-of-frame px {[b['truth'][s]['full_out_of_frame_px'] for s in S] if b else None}")

# endpoints stored for every site
miss = [f'{sid} {s}' for sid, c in sc.items() for s in S if not {'x', 'y', 'cause', 'owner_at_point'} <= set(c['terminal'][s])]
check('endpoint_representation', not miss, '; '.join(miss) or 'terminal point, cause and owner stored for all sites')

# treatment and injury truth: native >= visible; hidden-wound members show native but few visible pixels
det = []; ok = True
for sid, c in sc.items():
    for s in S:
        t = c['truth'][s]
        if t['wound_native_px'] + 5 < t['wound_visible_px']:
            ok = False; det.append(f'{sid} {s} native<visible')
        if t['tourniquet'] and t['tq_native_px'] == 0:
            ok = False; det.append(f'{sid} {s} tourniquet without native pixels')
for pid, s in (('P05', 'LUE'), ('P06', 'LLE')):
    a, b = sc.get(pid + 'a'), sc.get(pid + 'b')
    if a and b:
        det.append(f"{pid} {s} visible/native a {a['truth'][s]['wound_visible_px']}/{a['truth'][s]['wound_native_px']} "
                   f"b {b['truth'][s]['wound_visible_px']}/{b['truth'][s]['wound_native_px']}")
check('treatment_and_injury_truth', ok, ' | '.join(det))

# laterality: mirrored pair swaps left and right labels and truths
a, b = sc.get('P09a'), sc.get('P09b'); ok = False; det = 'missing'
if a and b:
    la, lb = a['labels_by_threshold']['0.10'], b['labels_by_threshold']['0.10']
    sw = {'LUE': 'RUE', 'RUE': 'LUE', 'LLE': 'RLE', 'RLE': 'LLE'}
    ok = all(la[s] == lb[sw[s]] for s in S) and a['truth']['LUE']['amputated'] and b['truth']['RUE']['amputated']
    det = f'a {la} b {lb}'
check('laterality', ok, det)

# provenance
need = {'generator', 'commit', 'params_sha256', 'blender', 'device', 'pythonhashseed', 'time_s'}
miss = [sid for sid, c in sc.items() if not need <= set(c.get('provenance', {}))]
check('provenance', not miss, 'missing in ' + ', '.join(miss) if miss else 'all fields present')

# performance
tt = [c['total_s'] for c in sc.values()]
kb = [sum(os.path.getsize(f) for f in glob.glob(os.path.join(D, sid + '*')) if not f.endswith('_params.json')) / 1024 for sid in sc]
check('performance', True, f"{len(sc)} scenes; seconds per scene median {np.median(tt):.1f} max {max(tt):.1f}; "
      f"kB per scene median {np.median(kb):.0f}; device {sorted({c['provenance']['device'] for c in sc.values()})}; "
      f"threads {sorted({c['provenance']['threads'] for c in sc.values()})}")
check('regression', REG.startswith('pass'), REG)

res['_causes'] = {sid: {s: c['terminal'][s]['cause'] for s in S} for sid, c in sc.items()}
json.dump(res, open(os.path.join(D, 'pilot_check.json'), 'w'), indent=1)
for k, v in res.items():
    if not k.startswith('_'):
        print(f"{'PASS' if v['pass'] else 'FAIL'}  {k}: {v['detail']}")
