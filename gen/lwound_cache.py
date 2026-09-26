"""Cache whole-limb windows for the wound classifier (pose env). Usage: python lwound_cache.py <scene_dir> [gt|pred]
For each limb with >= 30 visible pixels in the v3 ground-truth part map: square window around the limb (bounding box
+ 20%, at least 48 px at 640x480), read from the 1280x960 image, SAM 2.1-tiny embedding pooled to 32x32, plus the
limb mask. Target: 1 if the limb shows >= 20 wound pixels (640x480), else 0.
Writes <sid>_lw.npz: feat (k,256,32,32) fp16, mask (k,32,32), label (k), site (k), wound_px (k)."""
import sys, os, json, glob, time, zlib
import numpy as np, cv2, torch
sys.path.insert(0, os.path.dirname(__file__))
import parts as PT, lend as LE

torch.set_num_threads(8)
K = {n: i for i, n in enumerate(PT.SEG3_CLASSES)}
GROUP = {'UE': [K['upper_arm'], K['forearm'], K['hand']], 'LE': [K['thigh'], K['shank'], K['foot']]}
SITES = ['LUE', 'RUE', 'LLE', 'RLE']


def limb_window(mask):
    ys, xs = np.nonzero(mask)
    cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
    s = int(max(48, 1.2 * max(xs.max() - xs.min(), ys.max() - ys.min())))
    return int(cx), int(cy), s


if __name__ == '__main__':
    D = sys.argv[1]; P = LE.load_sam(); n = 0; t0 = time.time()
    for f in sorted(glob.glob(os.path.join(D, 'C*_sidecar.json'))):
        sid = os.path.basename(f)[:-len('_sidecar.json')]; out = os.path.join(D, sid + '_lw.npz')
        if os.path.exists(out) or not os.path.exists(os.path.join(D, sid + '_part3.png')):
            continue
        seg, side, ws = PT.decode3(os.path.join(D, sid + '_part3.png'))
        img = cv2.cvtColor(cv2.imread(os.path.join(D, sid + '.jpg')), cv2.COLOR_BGR2RGB)
        feats, masks, labels, sites, wpx = [], [], [], [], []
        for i, site in enumerate(SITES):
            sd = 1 if site[0] == 'L' else 2
            wm = ws == i + 1
            limb = ((side == sd) & np.isin(seg, GROUP[site[1:]])) | wm
            if limb.sum() < 30:
                continue
            cx, cy, s = limb_window(limb)
            feats.append(LE.encode(P, LE.crop_1280(img, cx, cy, s)))
            masks.append(LE.mask_window(limb, cx, cy, s)); labels.append(int(wm.sum() >= 20)); sites.append(site); wpx.append(int(wm.sum()))
        np.savez_compressed(out, feat=np.array(feats, np.float16).reshape(-1, 256, 32, 32), mask=np.array(masks, np.float32).reshape(-1, 32, 32),
                            label=np.array(labels, np.int64), site=np.array(sites), wound_px=np.array(wpx))
        n += 1
    print('LW', D, n, 'images', round(time.time() - t0), 's', flush=True)
