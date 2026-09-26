"""Stratified parameter set for the 20 hand-check scenes (Week 1).
Each body position appears 4 times; azimuths cover all 8 directions; every amputation level appears;
all occluder kinds, framings, skin tones, floors and lighting conditions appear at least once."""
import json, os, itertools
import numpy as np

rng = np.random.default_rng(20260922)
positions = ['supine', 'prone', 'left_lateral', 'right_lateral', 'semi_seated']
limb_poses = ['neutral', 'abducted', 'arm_across', 'knee_bent', 'arm_overhead']
azimuths = [0, 45, 90, 135, 180, 225, 270, 315]
elevations = [85, 45, 45, 15]
framings = ['full'] * 12 + ['upper_crop'] * 3 + ['lower_crop'] * 3 + ['off_center'] * 2
occluders = ['none'] * 8 + ['blanket'] * 3 + ['gear_bag'] * 3 + ['strap'] * 3 + ['medic_arm'] * 3
skins = ['peach', 'tan', 'brown', 'dark']
floors = ['concrete', 'grass', 'gravel', 'litter']
lights = ['indoor_flat', 'outdoor_sun', 'low_light']
sites = ['LUE', 'RUE', 'LLE', 'RLE']
levels = ['disarticulation', 'above_joint', 'below_joint', 'distal']
# amputation patterns: 8 none, 8 single, 4 double; every level appears at least twice
amp = [{}] * 8
lv = itertools.cycle(levels)
for i in range(8):
    amp.append({sites[i % 4]: next(lv)})
for i in range(4):
    a, b = rng.choice(4, 2, replace=False)
    amp.append({sites[a]: next(lv), sites[b]: next(lv)})
rng.shuffle(framings); rng.shuffle(occluders); rng.shuffle(amp)

scenes = []
for i in range(20):
    scenes.append({
        'scene_id': f'C{i + 1:03d}', 'seed': int(1000 + i),
        'body_position': positions[i % 5], 'limb_pose': limb_poses[(i // 5 + i) % 5],
        'body_yaw': int(rng.integers(0, 360)),
        'amputations': amp[i],
        'skin': skins[i % 4], 'floor': floors[(i // 4) % 4], 'occluder': occluders[i],
        'lighting': lights[i % 3], 'sun_el': int(rng.integers(25, 70)), 'sun_az': int(rng.integers(0, 360)),
        'azimuth': azimuths[i % 8], 'elevation': elevations[i % 4], 'framing': framings[i],
        'samples': 32,
    })
out = os.path.join(os.path.dirname(__file__), '..', 'out', 'check20')
os.makedirs(out, exist_ok=True)
for s in scenes:
    json.dump(s, open(os.path.join(out, s['scene_id'] + '_params.json'), 'w'), indent=1)
json.dump(scenes, open(os.path.join(out, 'check20_params_all.json'), 'w'), indent=1)
print(len(scenes), 'scenes written')
for s in scenes:
    print(s['scene_id'], s['body_position'], s['limb_pose'], s['amputations'], s['occluder'], s['framing'],
          s['azimuth'], s['elevation'])
