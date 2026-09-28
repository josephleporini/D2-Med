"""Close-up framing robustness set (model maturity item 1, 28 Sep). Takes dev5 scene parameters and re-renders a subset
with the camera on one limb ('limb_closeup'), so each close-up is paired with a full or half-body dev5 render of the
same casualty, pose, lighting and injuries. The limb in focus is an injured one half of the time.
usage: python tools/make_closeup_params.py <dddata>/dev5 <out_dir> [n=120] [dist=0.35]
New scene ids DKnnnn (DK + the dev5 number). Development use only; dev5 itself is unchanged."""
import sys, os, json, glob
import numpy as np
src, out = sys.argv[1], sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 120
dist = float(sys.argv[4]) if len(sys.argv) > 4 else 0.35
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(20260928)
files = sorted(glob.glob(os.path.join(src, 'batch_*', 'D*_params.json')))
pick = sorted(rng.choice(len(files), n, replace=False))
S = ['LUE', 'RUE', 'LLE', 'RLE']
for k in pick:
    p = json.load(open(files[k]))
    inj = [s for s in S if s in p.get('amputations', {}) or s in p.get('wounds', {})]
    site = str(rng.choice(inj)) if inj and rng.random() < 0.5 else str(rng.choice(S))
    p['base_scene'] = p['scene_id']; p['scene_id'] = 'DK' + p['scene_id'][1:]
    p['framing'] = 'limb_closeup'; p['closeup_site'] = site; p['closeup_dist'] = dist; p['split'] = 'dev5_closeup'
    json.dump(p, open(os.path.join(out, p['scene_id'] + '_params.json'), 'w'), indent=1)
print('CLOSEUP_PARAMS', n, out)
