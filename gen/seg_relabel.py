"""Refresh the 128x128 label/side arrays in cached _feat.npz files from the current part maps (pose env).
Uses the stored crop, so SAM features are not recomputed. Usage: python seg_relabel.py <scene_dir>"""
import sys, os, glob, numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__))
import parts as PT
D = sys.argv[1]; n = 0
for f in sorted(glob.glob(os.path.join(D, 'C*_feat.npz'))):
    sid = os.path.basename(f).split('_')[0]; pp = os.path.join(D, sid + '_part.png')
    if os.path.getmtime(pp) <= os.path.getmtime(f):
        continue
    z = dict(np.load(f)); X1, Y1, S = [int(v) for v in z['crop']]
    seg, side = PT.decode(pp)
    out = []
    for a in (seg, side):
        a = cv2.resize(a, (1280, 960), interpolation=cv2.INTER_NEAREST)
        p = cv2.copyMakeBorder(a, max(0, -Y1), max(0, Y1 + S - 960), max(0, -X1), max(0, X1 + S - 1280), cv2.BORDER_CONSTANT, value=0)
        oy, ox = max(0, -Y1), max(0, -X1)
        out.append(cv2.resize(p[Y1 + oy:Y1 + oy + S, X1 + ox:X1 + ox + S], (128, 128), interpolation=cv2.INTER_NEAREST))
    z['label'], z['side'] = out
    np.savez_compressed(f, **z); n += 1
print('RELABEL', D, n)
