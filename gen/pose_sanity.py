"""Pose-model sanity pass on the check scenes (pose env). NOT a Probe B result (n = 20).
Reports per scene: RTMW person found, MediaPipe person found, and for each amputated site whether the model
still placed the distal keypoint (wrist/ankle) with confidence above 0.3 (RTMW) / visibility above 0.5 (MediaPipe)."""
import sys, os, json, glob, csv
import cv2
from rtmlib import Wholebody
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

D = sys.argv[1]; M = os.path.join(os.path.dirname(__file__), '..', 'models')
wb = Wholebody(det=os.path.join(M, '20230928/yolox_onnx/yolox_m_8xb8-300e_humanart-c2c7a14a/end2end.onnx'),
               det_input_size=(640, 640), pose=os.path.join(M, 'end2end.onnx'), pose_input_size=(192, 256),
               backend='onnxruntime', device='cpu')
mpl = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=os.path.join(M, 'pose_landmarker_full.task')), num_poses=1))
# COCO body index (RTMW first 17) and MediaPipe landmark index for the distal joint of each site
RT = {'LUE': 9, 'RUE': 10, 'LLE': 15, 'RLE': 16}
MP = {'LUE': 15, 'RUE': 16, 'LLE': 27, 'RLE': 28}
rows = []
for sc in sorted(glob.glob(os.path.join(D, 'C*_sidecar.json'))):
    s = json.load(open(sc)); sid = s['scene_id']; amp = s['params']['amputations']
    img = cv2.imread(os.path.join(D, sid + '.jpg'))
    k, sc_ = wb(img)
    r = mpl.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    rt_found = len(k) > 0; mp_found = len(r.pose_landmarks) > 0
    for site, lvl in amp.items():
        rt_conf = float(sc_[0][RT[site]]) if rt_found else None
        mp_vis = float(r.pose_landmarks[0][MP[site]].visibility) if mp_found else None
        rows.append([sid, site, lvl, s['labels_by_threshold']['0.10'][site],
                     rt_found, None if rt_conf is None else round(rt_conf, 2),
                     None if rt_conf is None else rt_conf > 0.3,
                     mp_found, None if mp_vis is None else round(mp_vis, 2),
                     None if mp_vis is None else mp_vis > 0.5])
    print(sid, 'RTMW found' if rt_found else 'RTMW none', '| MediaPipe found' if mp_found else '| MediaPipe none')
mpl.close()
with open(os.path.join(D, 'pose_sanity_amputated_sites.csv'), 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['scene', 'site', 'amputation_level', 'gt_label_10pct', 'rtmw_found', 'rtmw_distal_conf',
                'rtmw_distal_placed', 'mp_found', 'mp_distal_visibility', 'mp_distal_placed'])
    w.writerows(rows)
for r_ in rows:
    print('AMP', r_)
