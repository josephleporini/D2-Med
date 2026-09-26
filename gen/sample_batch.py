"""Stratified scene sampler for Probe B batches.
Usage: python3 sample_batch.py <first_index> <count> <out_dir> <seed>
Balanced: body position (equal), camera azimuth (8 levels, cycled), elevation (5 levels, cycled), limb pose.
Mix targets: amputation 40% none / 40% single / 20% double; occluder 40% none, 15% each of 4 kinds;
framing 60% full, ~13% each crop type."""
import json, os, sys, itertools
import numpy as np

first, n, out, seed = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
rng = np.random.default_rng(seed)
positions = ['supine', 'prone', 'left_lateral', 'right_lateral', 'semi_seated']
limb_poses = ['neutral', 'abducted', 'arm_across', 'knee_bent', 'arm_overhead']
azimuths = [0, 45, 90, 135, 180, 225, 270, 315]
elevations = [85, 60, 45, 30, 15]
sites = ['LUE', 'RUE', 'LLE', 'RLE']
levels = ['disarticulation', 'above_joint', 'below_joint', 'distal']


def mix(spec):
    out_ = []
    for k, frac in spec:
        out_ += [k] * int(round(frac * n))
    while len(out_) < n:
        out_.append(spec[0][0])
    out_ = out_[:n]
    rng.shuffle(out_)
    return out_


amp_kind = mix([('none', 0.4), ('single', 0.4), ('double', 0.2)])
occ = mix([('none', 0.4), ('blanket', 0.15), ('gear_bag', 0.15), ('strap', 0.15), ('medic_arm', 0.15)])
frm = mix([('full', 0.6), ('upper_crop', 0.134), ('lower_crop', 0.133), ('off_center', 0.133)])
lv = itertools.cycle(levels)
pos = [positions[i % 5] for i in range(n)]; rng.shuffle(pos)
scenes = []
for i in range(n):
    if amp_kind[i] == 'none':
        amp = {}
    elif amp_kind[i] == 'single':
        amp = {sites[int(rng.integers(4))]: next(lv)}
    else:
        a, b = rng.choice(4, 2, replace=False); amp = {sites[a]: next(lv), sites[b]: next(lv)}
    scenes.append({
        'scene_id': f'C{first + i:03d}', 'seed': int(seed * 10 + i),
        'body_position': pos[i], 'limb_pose': limb_poses[int(rng.integers(5))],
        'body_yaw': int(rng.integers(0, 360)), 'amputations': amp,
        'skin': ['peach', 'tan', 'brown', 'dark'][int(rng.integers(4))],
        'floor': ['concrete', 'grass', 'gravel', 'litter'][int(rng.integers(4))],
        'occluder': occ[i], 'lighting': ['indoor_flat', 'outdoor_sun', 'low_light'][int(rng.integers(3))],
        'sun_el': int(rng.integers(25, 70)), 'sun_az': int(rng.integers(0, 360)),
        'azimuth': azimuths[i % 8], 'elevation': elevations[i % 5], 'framing': frm[i], 'samples': 32})
os.makedirs(out, exist_ok=True)
for s in scenes:
    json.dump(s, open(os.path.join(out, s['scene_id'] + '_params.json'), 'w'), indent=1)
json.dump(scenes, open(os.path.join(out, f'batch_{first:03d}_params_all.json'), 'w'), indent=1)
print(n, 'scenes', 'amp', {k: amp_kind.count(k) for k in set(amp_kind)}, 'occ', {k: occ.count(k) for k in set(occ)},
      'pos', {k: pos.count(k) for k in set(pos)})
