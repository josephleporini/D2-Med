"""Scene sampler vNext. Usage:
  python3 sample_batch4.py pilot <out_dir> [seed]      20 paired pathological scenes (10 pairs, spec v1.1 section 10.6)
  python3 sample_batch4.py split <name> <n> <seed> <out_dir> [challenge|bt2]
        v3 base mix (sample_batch3) with split-prefixed scene ids; 'challenge' forces one confuser per scene;
        'bt2' applies the BT-2 mix (bt2_mix below): wound style v2, one third single-limb close-ups, blood-smear confuser
        'bt3' applies the BT-2b mix (bt3_mix below): generator v3 (bystanders, contrast, pools, red gear, camouflage)
Each pair shares casualty, pose, camera and seed and differs in one factor; 'pair' records the pair id, the member
(a/b), the factor and what each member should show, so pilot_check.py can test the truth products.
"""
import json, os, sys, glob, subprocess
import numpy as np

BASE = dict(body_yaw=0, skin='tan', floor='concrete', lighting='indoor_flat', sun_el=50, sun_az=140, azimuth=0,
            elevation=60, framing='full', samples=16, wounds={}, amputations={}, tourniquets=[], occluder='none',
            garments={'top': 'short', 'bottom': 'shorts', 'boots': False}, hidden_wound_p=0.0,
            body_position='supine', limb_pose='abducted')

# (pair id, factor, shared overrides, member a overrides, member b overrides, expectations)
PAIRS = [
    ('P01', 'amputation_supine', {}, {}, {'amputations': {'RLE': 'below_joint'}},
     {'b': {'RLE': 'amputated'}}),
    ('P02', 'amputation_prone', {'body_position': 'prone'}, {}, {'amputations': {'LUE': 'below_joint'}},
     {'b': {'LUE': 'amputated'}}),
    ('P03', 'occluder_intact_limb_end', {'limb_pose': 'neutral'}, {'occluder': 'medic_arm', 'occ_target': 'limb_end', 'occ_site': 'LUE'}, {},
     {'a': 'occluder_present', 'b': 'occluder_removed'}),
    ('P04', 'occluder_amputated_limb_end', {'amputations': {'RUE': 'below_joint'}, 'limb_pose': 'neutral'},
     {'occluder': 'gear_bag', 'occ_target': 'limb_end', 'occ_site': 'RUE'}, {}, {'a': 'occluder_present', 'b': 'occluder_removed'}),
    ('P05', 'wound_visible_vs_hidden', {'wounds': {'LUE': 'laceration'}, 'elevation': 60}, {'wound_surface': {'LUE': 'front'}},
     {'wound_surface': {'LUE': 'back'}},
     {'site': 'LUE'}),
    ('P06', 'tourniquet_wound_visible_vs_hidden', {'wounds': {'LLE': 'laceration'}, 'tourniquets': ['LLE'], 'elevation': 60},
     {'wound_surface': {'LLE': 'front'}}, {'wound_surface': {'LLE': 'back'}}, {'site': 'LLE'}),
    ('P07', 'treatment_cue_without_visible_injury', {}, {'tourniquets': ['RUE']}, {}, {'site': 'RUE'}),
    ('P08', 'lateral_views', {'limb_pose': 'knee_bent'}, {'body_position': 'left_lateral'}, {'body_position': 'right_lateral'}, {}),
    ('P09', 'mirror_laterality', {'limb_pose': 'neutral', 'body_yaw': 0},
     {'amputations': {'LUE': 'below_joint'}, 'wounds': {'LLE': 'burn'}},
     {'amputations': {'RUE': 'below_joint'}, 'wounds': {'RLE': 'burn'}}, {'swap': True}),
    ('P10', 'out_of_frame', {'amputations': {'LLE': 'above_joint'}}, {'framing': 'full'}, {'framing': 'upper_crop'}, {}),
]


def pilot(out, seed=4242):
    os.makedirs(out, exist_ok=True); n = 0
    for k, (pid, factor, shared, a, b, expect) in enumerate(PAIRS):
        for m, ov in (('a', a), ('b', b)):
            p = json.loads(json.dumps(BASE)); p.update(json.loads(json.dumps(shared))); p.update(json.loads(json.dumps(ov)))
            p['scene_id'] = f'{pid}{m}'; p['seed'] = seed + k          # same seed inside a pair
            p['pair'] = {'id': pid, 'member': m, 'factor': factor, 'expect': expect}
            if factor == 'treatment_cue_without_visible_injury' and m == 'a':
                p['confusers'] = ['treatment_cue_without_visible_injury']
            json.dump(p, open(os.path.join(out, p['scene_id'] + '_params.json'), 'w'), indent=1); n += 1
    print('pilot params', n)


PREFIX = {'train5': 'T', 'dev5': 'D', 'test5': 'X', 'challenge5': 'H', 'train6': 'U', 'dev6': 'V', 'train7': 'W', 'dev7': 'Y'}
CLOSEUP_P, SMEAR_P = 1 / 3, 0.08


def bt2_mix(p, rng, closeup_p=None):
    """BT-2 (Close-Up Framing v1.0; BT-2 plan item 1): v2 wounds; one third of scenes framed on one limb (an injured
    limb half the time), camera distance 0.25 to 0.50 of the full-body distance, occluder removed in 3 of 4 close-ups; 8% of scenes get a blood stain on an
    intact, unwounded limb (confuser, label unchanged)."""
    p['wound_style'] = 'v2'; p['garment_label_fix'] = True
    if rng.random() < (CLOSEUP_P if closeup_p is None else closeup_p):
        inj = [s for s in SITES if s in p['amputations'] or s in p['wounds']]
        p['framing'] = 'limb_closeup'
        p['closeup_site'] = str(rng.choice(inj)) if inj and rng.random() < 0.5 else str(rng.choice(SITES))
        p['closeup_dist'] = round(float(rng.uniform(0.25, 0.50)), 3)
        if p['occluder'] != 'none' and rng.random() < 0.75:   # close-up photos are taken of the exposed limb
            p['occluder'] = 'none'; p.pop('occ_target', None); p.pop('occ_site', None)
    clean = [s for s in SITES if s not in p['amputations'] and s not in p['wounds'] and s not in p['tourniquets']]
    if clean and rng.random() < SMEAR_P:
        p['blood_smear'] = str(rng.choice(clean)); p.setdefault('confusers', []).append('blood_material')
LOOKS = ['AgX - Punchy', 'AgX - High Contrast', 'AgX - Medium High Contrast']


def bt3_mix(p, rng):
    """BT-2b (Real Fidelity Gap v1.0, section 1 item 4b): the BT-2 mix with a 0.2 close-up share, then generator v3.
    Real training photos are nearly all supine, show 3 people on median, 4x the contrast, and red content."""
    bt2_mix(p, rng, closeup_p=0.2)
    p['gen'] = 'v3'; p['wound_style'] = 'v3'
    if rng.random() < 0.7:
        p['body_position'] = 'supine'
    if rng.random() < 0.5:
        p['view'] = 'AgX'; p['look'] = str(rng.choice(LOOKS))
    else:
        p['view'] = 'Standard'
    p['light_boost'] = round(float(rng.uniform(1.5, 3.5)), 3)
    p['exposure'] = round(float(rng.uniform(-1.0, -0.1)) + (0.6 if p['lighting'] == 'low_light' else 0.0), 3)
    p['sun_el'] = int(rng.integers(25, 70))                  # lower sun, longer hard shadows
    nb = int(rng.choice([0, 1, 2], p=[0.2, 0.45, 0.35]))
    p['bystanders'] = [{'pose': str(rng.choice(['kneel', 'crouch', 'stand'], p=[0.5, 0.3, 0.2])),
                        'side': int(rng.choice([-1, 1])), 'dist': round(float(rng.uniform(0.15, 0.5)), 3),
                        'along': round(float(rng.uniform(-0.6, 0.6)), 3), 'lean': round(float(rng.uniform(10, 40)), 1)}
                       for _ in range(nb)]
    if len(p['bystanders']) == 2 and p['bystanders'][0]['side'] == p['bystanders'][1]['side']:
        p['bystanders'][1]['along'] = round(-p['bystanders'][0]['along'] or 0.5, 3)   # keep two on one side apart
    p['red_gear'] = 0 if rng.random() < 0.35 else int(rng.integers(1, 4))
    inj = [s for s in SITES if s in p['amputations'] or s in p['wounds']]
    pools = [s for s in inj if rng.random() < 0.5]
    if rng.random() < 0.1:                                   # pool beside an intact limb (confuser, floor label)
        pools.append(str(rng.choice(SITES)))
    p['blood_pool'] = sorted(set(pools))
    p['camo'] = bool(rng.random() < 0.5)
    p['soak'] = bool(rng.random() < 0.5)


CONFUSERS = ['treatment_cue_without_visible_injury', 'tq_no_wound', 'hidden_wound', 'occluded_end', 'crossing_limbs']
SITES = ['LUE', 'RUE', 'LLE', 'RLE']


def confuse(p, flag, rng):
    free = [s for s in SITES if s not in p['amputations']]
    site = str(rng.choice(free))
    if flag == 'treatment_cue_without_visible_injury':
        p['wounds'][site] = 'penetrating'; p['wound_surface'] = {site: 'back'}; p['tourniquets'] = [site]
    elif flag == 'tq_no_wound':
        p['wounds'].pop(site, None); p['tourniquets'] = [site]
    elif flag == 'hidden_wound':
        p['wounds'][site] = 'penetrating'; p['wound_surface'] = {site: 'back'}
        p['tourniquets'] = [t for t in p['tourniquets'] if t != site]
    elif flag == 'occluded_end':
        p['occluder'] = str(rng.choice(['medic_arm', 'gear_bag'])); p['occ_target'] = 'limb_end'; p['occ_site'] = site
    elif flag == 'crossing_limbs':
        p['limb_pose'] = 'arm_across'; site = None
    p['confusers'] = [flag]; p['confuser_site'] = site


def split(name, n, seed, out, challenge=False, bt2=False, bt3=False):
    tmp = os.path.join(out, '_tmp'); os.makedirs(tmp, exist_ok=True)
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_batch3.py'), '0', str(n), tmp, str(seed), '16'],
                   check=True, stdout=subprocess.DEVNULL)
    rng = np.random.default_rng(seed + 31337)
    for i, f in enumerate(sorted(glob.glob(os.path.join(tmp, 'C*_params.json')))):
        p = json.load(open(f)); p['scene_id'] = f'{PREFIX[name]}{i:04d}'; p['split'] = name
        if challenge:
            confuse(p, CONFUSERS[i % len(CONFUSERS)], rng)
        if bt3:
            bt3_mix(p, rng)
        elif bt2:
            bt2_mix(p, rng)
        json.dump(p, open(os.path.join(out, p['scene_id'] + '_params.json'), 'w'), indent=1)
        os.remove(f)
    for f in glob.glob(os.path.join(tmp, '*')):
        os.remove(f)
    os.rmdir(tmp)
    print('split params', name, n)


if __name__ == '__main__':
    if sys.argv[1] == 'pilot':
        pilot(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4242)
    elif sys.argv[1] == 'split':
        mode = sys.argv[6] if len(sys.argv) > 6 else ''
        split(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], mode == 'challenge', mode == 'bt2', mode == 'bt3')
