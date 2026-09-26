"""Scene sampler vNext. Usage:
  python3 sample_batch4.py pilot <out_dir> [seed]      20 paired pathological scenes (10 pairs, spec v1.1 section 10.6)
Each pair shares casualty, pose, camera and seed and differs in one factor; 'pair' records the pair id, the member
(a/b), the factor and what each member should show, so pilot_check.py can test the truth products.
"""
import json, os, sys

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
    ('P05', 'wound_visible_vs_hidden', {'wounds': {'LUE': 'laceration'}, 'elevation': 30}, {'azimuth': 0}, {'azimuth': 180},
     {'site': 'LUE'}),
    ('P06', 'tourniquet_wound_visible_vs_hidden', {'wounds': {'LLE': 'penetrating'}, 'tourniquets': ['LLE'], 'elevation': 30},
     {'azimuth': 0}, {'azimuth': 180}, {'site': 'LLE'}),
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


if __name__ == '__main__':
    if sys.argv[1] == 'pilot':
        pilot(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4242)
