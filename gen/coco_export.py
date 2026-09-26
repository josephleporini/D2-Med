"""Export ground-truth extremity keypoints in COCO-WholeBody form, keeping the native anatomical states.
Usage: python3 coco_export.py <scene_dir>   ->  <scene_dir>/coco_wholebody_gt.json

COCO visibility (v): 2 labelled and visible, 1 labelled but not visible (occluded), 0 not labelled.
Here v=0 covers BOTH "anatomically absent" and "outside the frame": COCO cannot tell them apart,
so every point also carries the native state:
  visible | occluded | out_of_frame | absent
("cannot be determined" is a prediction state, never a ground-truth state: the generator always knows.)
Only the 12 limb joints plus one extremity point per limb are filled (hand -> hand root 91/112,
foot -> big toe 17/20); all other WholeBody points are v=0.
"""
import sys, os, json, glob

IDX = {('LUE', 'shoulder'): 5, ('RUE', 'shoulder'): 6, ('LUE', 'elbow'): 7, ('RUE', 'elbow'): 8,
       ('LUE', 'wrist'): 9, ('RUE', 'wrist'): 10, ('LLE', 'hip'): 11, ('RLE', 'hip'): 12, ('LLE', 'knee'): 13,
       ('RLE', 'knee'): 14, ('LLE', 'ankle'): 15, ('RLE', 'ankle'): 16, ('LLE', 'foot'): 17, ('RLE', 'foot'): 20,
       ('LUE', 'hand'): 91, ('RUE', 'hand'): 112}


def state(p):
    if not p['exists']:
        return 'absent'
    if not p['in_frame']:
        return 'out_of_frame'
    return 'visible' if p['visible'] else 'occluded'


V = {'visible': 2, 'occluded': 1, 'out_of_frame': 0, 'absent': 0}

if __name__ == '__main__':
    D = sys.argv[1]; images, anns = [], []
    for i, f in enumerate(sorted(glob.glob(os.path.join(D, 'C*_gtkp.json')))):
        g = json.load(open(f)); W, H = g.get('image_size', [1280, 960])
        kp = [0.0] * (133 * 3); native = {}
        for site, joints in g['sites'].items():
            for j, p in joints.items():
                st = state(p); native[f'{site}.{j}'] = st
                k = IDX.get((site, j))
                if k is not None and V[st] > 0:
                    kp[3 * k:3 * k + 3] = [round(p['x'], 1), round(p['y'], 1), V[st]]
        images.append({'id': i, 'file_name': g['scene_id'] + '.jpg', 'width': W, 'height': H})
        anns.append({'id': i, 'image_id': i, 'category_id': 1, 'keypoints': kp,
                     'num_keypoints': sum(1 for q in range(133) if kp[3 * q + 2] > 0), 'native_states': native})
    json.dump({'info': {'description': 'Probe B synthetic ground truth, COCO-WholeBody layout + native states'},
               'images': images, 'annotations': anns, 'categories': [{'id': 1, 'name': 'person'}]},
              open(os.path.join(D, 'coco_wholebody_gt.json'), 'w'))
    from collections import Counter
    print('COCO export', D, len(anns), 'images;', Counter(s for a in anns for s in a['native_states'].values()))
