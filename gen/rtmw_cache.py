"""Cache RTMW 133-keypoint whole-body output per scene (pose env). Usage: python rtmw_cache.py <scene_dir>"""
import sys, os, glob, json
import numpy as np, cv2
from rtmlib import Wholebody
M = os.path.join(os.path.dirname(__file__), '..', 'models')
D = sys.argv[1]; wb = None; n = 0
for sc in sorted(glob.glob(os.path.join(D, 'C*_sidecar.json'))):
    sid = json.load(open(sc))['scene_id']; out = os.path.join(D, sid + '_rtmw133.npz')
    if os.path.exists(out) or not os.path.exists(os.path.join(D, sid + '.jpg')):
        continue
    if wb is None:
        wb = Wholebody(det=os.path.join(M, '20230928/yolox_onnx/yolox_m_8xb8-300e_humanart-c2c7a14a/end2end.onnx'),
                       det_input_size=(640, 640), pose=os.path.join(M, 'end2end.onnx'), pose_input_size=(192, 256),
                       backend='onnxruntime', device='cpu')
    k, s = wb(cv2.imread(os.path.join(D, sid + '.jpg')))
    i = int(np.argmax(s[:, :17].mean(1))) if len(k) else -1
    np.savez(out, k=k[i] if i >= 0 else np.zeros((133, 2)), s=s[i] if i >= 0 else np.zeros(133), found=i >= 0); n += 1
print('RTMW', D, n)
