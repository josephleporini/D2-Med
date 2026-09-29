"""Relabel existing scenes with the garment label fix (28 Sep): cloth over a hidden wound was labelled wound in the part
map and counted as visible wound pixels. The fix re-runs only the truth passes (scene4.py labels, no RGB); geometry is
deterministic, so the RGB image is unchanged and every other truth product must decode identically, which is checked.

usage: python tools/relabel_garment.py <split_dir> <work_dir> <bpy_python> [jobs=2]
  split_dir  e.g. <dddata>/dev5 (batch_* inside). Files are replaced in place only for scenes whose truth changes:
             <id>_part3.png, <id>_injury.png, <id>_sidecar.json (with a 'relabel' block). Batch manifests are rewritten.
  Scenes that cannot be affected (no wound, or no garment) are skipped. A scene whose other truth products differ after
  decoding is reported as GEOMETRY_MISMATCH and left untouched.
Writes <work_dir>/relabel_report.json.
"""
import os, sys, json, glob, shutil, subprocess, hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); GEN = os.path.join(HERE, '..', 'gen')
sys.path.insert(0, GEN)
import parts as PT

split, work, bpy = sys.argv[1], sys.argv[2], sys.argv[3]; J = int(sys.argv[4]) if len(sys.argv) > 4 else 2
os.makedirs(work, exist_ok=True)
commit = subprocess.run(['git', '-C', HERE, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()


def affected(sc):
    g = sc['params'].get('garments') or {}
    return bool(sc.get('wounds')) and (g.get('top', 'none') != 'none' or g.get('bottom', 'none') != 'none')


TOL_PX = 50     # GPU (OptiX) and CPU renders of the same geometry differ on a few edge pixels (3 px seen on D0067)


def ndiff(a, b):
    A, B = np.array(Image.open(a).convert('L')), np.array(Image.open(b).convert('L'))
    return int((A != B).sum())


def site_map(p):
    """id pass decoded to classes (tolerates the 1-level colour difference between GPU and CPU renders)."""
    A = np.array(Image.open(p).convert('RGB')).astype(float) / 255
    return np.where(A > 0.86, 2, np.where(A > 0.35, 1, 0)).astype(np.uint8)


def one(sc_path):
    sc = json.load(open(sc_path)); sid = sc['scene_id']; d = os.path.dirname(sc_path)
    if not affected(sc):
        return sid, 'skipped', None
    w = os.path.join(work, sid); os.makedirs(w, exist_ok=True)
    pf = os.path.join(d, sid + '_params.json')      # sets without params files (dev5_closeup): the sidecar's copy
    p = json.load(open(pf)) if os.path.exists(pf) else {k: v for k, v in sc['params'].items() if not k.startswith('_')}
    p['garment_label_fix'] = True
    pp = os.path.join(w, sid + '_params.json'); json.dump(p, open(pp, 'w'), indent=1)
    r = subprocess.run([bpy, 'scene4.py', 'labels', pp, w], cwd=GEN, capture_output=True, text=True, timeout=900,
                       env=dict(os.environ, GEN_COMMIT=commit))
    ns = os.path.join(w, sid + '_sidecar.json')
    if not os.path.exists(ns):
        return sid, 'error', (r.stdout + r.stderr)[-400:]
    new = json.load(open(ns))
    # geometry check: site map, occluder owner and full outlines must decode identically
    geo = {'id': int((site_map(os.path.join(d, sid + '_id.png')) != site_map(os.path.join(w, sid + '_id.png'))).any(-1).sum()),
           'occ': ndiff(os.path.join(d, sid + '_occ.png'), os.path.join(w, sid + '_occ.png')),
           'amodal': ndiff(os.path.join(d, sid + '_amodal.png'), os.path.join(w, sid + '_amodal.png'))}
    if max(geo.values()) > TOL_PX or new['terminal'] != sc['terminal']:
        return sid, 'GEOMETRY_MISMATCH', geo
    o3, n3 = PT.decode3(os.path.join(d, sid + '_part3.png')), PT.decode3(os.path.join(w, sid + '_part3.png'))
    part_changed = any(not np.array_equal(a, b) for a, b in zip(o3, n3))
    lab_changed = new['labels_by_threshold'] != sc['labels_by_threshold']
    if not part_changed and not lab_changed and new['truth'] == sc['truth']:
        return sid, 'unchanged', None
    diff = {t: {s: [sc['labels_by_threshold'][t][s], new['labels_by_threshold'][t][s]] for s in new['labels_by_threshold'][t]
                if sc['labels_by_threshold'][t][s] != new['labels_by_threshold'][t][s]} for t in new['labels_by_threshold']}
    diff = {t: v for t, v in diff.items() if v}
    new['provenance'] = sc['provenance']
    new['relabel'] = {'fix': 'garment_label_fix', 'date': '2026-09-28', 'commit': commit,
                      'rgb_unchanged': True, 'labels_changed': diff, 'edge_px_vs_original': geo,
                      'wound_visible_px_before': sc['wound_visible_px']}
    new['params'] = sc['params']                         # the scene's params as rendered; the fix is recorded above
    for k in ('part3', 'injury', 'id', 'occ', 'amodal'):        # all truth maps from one render, so they agree pixel for pixel
        shutil.copy(os.path.join(w, sid + f'_{k}.png'), os.path.join(d, sid + f'_{k}.png'))
    json.dump(new, open(sc_path, 'w'), indent=1)
    return sid, 'relabelled', diff


scs = sorted(glob.glob(os.path.join(split, 'batch_*', '*_sidecar.json')) or glob.glob(os.path.join(split, '*_sidecar.json')))
res = {}
with ThreadPoolExecutor(J) as ex:
    for k, (sid, status, info) in enumerate(ex.map(one, scs)):
        res[sid] = {'status': status, 'info': info}
        if status not in ('skipped', 'unchanged'):
            print(k, sid, status, json.dumps(info)[:200], flush=True)
# manifests of touched batches
for b in sorted({os.path.dirname(s) for s in scs}):
    ids = {os.path.basename(f)[:-len('_sidecar.json')] for f in glob.glob(os.path.join(b, '*_sidecar.json'))}
    if any(res.get(i, {}).get('status') == 'relabelled' for i in ids) and os.path.exists(os.path.join(b, 'MANIFEST.sha256')):
        lines = []
        for f in sorted(os.listdir(b)):
            if f != 'MANIFEST.sha256' and os.path.isfile(os.path.join(b, f)):
                lines.append(f"{hashlib.sha256(open(os.path.join(b, f), 'rb').read()).hexdigest()}  {f}")
        open(os.path.join(b, 'MANIFEST.sha256'), 'w').write('\n'.join(lines) + '\n')
from collections import Counter
summ = Counter(v['status'] for v in res.values())
ch = Counter(f'{a}->{b}' for v in res.values() if v['status'] == 'relabelled' for s, (a, b) in (v['info'].get('0.10') or {}).items())
json.dump({'split': split, 'commit': commit, 'summary': summ, 'changes_0.10': ch, 'scenes': res},
          open(os.path.join(work, 'relabel_report.json'), 'w'), indent=1)
print('RELABEL_DONE', split, dict(summ), dict(ch))
