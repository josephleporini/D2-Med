"""Scene sampler v3 (wounds, clothing, tourniquets). Usage: python3 sample_batch3.py <first> <count> <out_dir> <seed> [samples]
Adds to the v1 mix: wounds on non-amputated limbs (35% none / 45% one / 20% two; four types uniform),
garments (top none/short/long 25/25/50; bottom none/shorts/long 15/25/60; boots 50% with long trousers else 25%),
tourniquets on 50% of amputated or wounded limbs, hidden_wound_p 0.2, bags and medic arms aimed at limb ends 50%."""
import json, os, sys, subprocess, glob
import numpy as np
first, n, out, seed = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
samples = int(sys.argv[5]) if len(sys.argv) > 5 else 16
subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'sample_batch.py'), str(first), str(n), out, str(seed)], check=True)
rng = np.random.default_rng(seed + 99991)
WT = ['laceration', 'penetrating', 'burn', 'open_fracture']
for i in range(n):
    f = os.path.join(out, f'C{first + i:03d}_params.json'); p = json.load(open(f))
    free = [s for s in ['LUE', 'RUE', 'LLE', 'RLE'] if s not in p['amputations']]
    k = rng.choice([0, 1, 2], p=[0.35, 0.45, 0.20]); k = min(k, len(free))
    ws = list(rng.choice(free, k, replace=False)) if k else []
    p['wounds'] = {str(s): str(rng.choice(WT)) for s in ws}
    top = str(rng.choice(['none', 'short', 'long'], p=[0.25, 0.25, 0.5]))
    bottom = str(rng.choice(['none', 'shorts', 'long'], p=[0.15, 0.25, 0.6]))
    p['garments'] = {'top': top, 'bottom': bottom, 'boots': bool(rng.random() < (0.5 if bottom == 'long' else 0.25))}
    cand = [s for s in list(p['amputations']) + ws]
    p['tourniquets'] = [str(s) for s in cand if rng.random() < 0.5][:2]
    p['hidden_wound_p'] = 0.2
    if p['occluder'] in ('gear_bag', 'medic_arm') and rng.random() < 0.5:
        p['occ_target'] = 'limb_end'
    p['samples'] = samples
    json.dump(p, open(f, 'w'), indent=1)
print('v3 params', n)
