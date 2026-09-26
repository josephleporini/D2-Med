"""Write or verify a split manifest (sha256 of every scene file).
usage: python score/manifest.py write <scene_dir> <out.sha256> [--seeds]
       python score/manifest.py verify <scene_dir> <manifest.sha256>
Lines: '<sha256>  <basename>'. With --seeds a sibling <out>.seeds.json lists scene_id, seed and generator fields."""
import sys, os, glob, json, hashlib
def h(p):
    x = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): x.update(b)
    return x.hexdigest()
mode, d, m = sys.argv[1:4]
files = sorted(f for f in glob.glob(os.path.join(d, 'C*')) if os.path.isfile(f))
if mode == 'write':
    with open(m, 'w') as fh:
        for f in files: fh.write(f'{h(f)}  {os.path.basename(f)}\n')
    if '--seeds' in sys.argv:
        rows = []
        for f in sorted(glob.glob(os.path.join(d, 'C*_sidecar.json'))):
            s = json.load(open(f)); p = s.get('params', {})
            rows.append({'scene_id': s['scene_id'], 'seed': p.get('seed'), 'generator': s.get('generator') or s.get('version')})
        json.dump(rows, open(m + '.seeds.json', 'w'), indent=0)
    print('MANIFEST', m, len(files), h(m))
else:
    lock = {l.split()[1]: l.split()[0] for l in open(m) if l.strip()}
    cur = {os.path.basename(f): h(f) for f in files}
    miss = sorted(set(lock) - set(cur)); extra = sorted(set(cur) - set(lock)); diff = sorted(k for k in lock if k in cur and cur[k] != lock[k])
    print('VERIFY', 'OK' if not (miss or extra or diff) else 'FAIL', json.dumps({'missing': len(miss), 'extra': len(extra), 'changed': len(diff), 'examples': (miss + extra + diff)[:5]}))
    sys.exit(1 if (miss or extra or diff) else 0)
