"""Cache SAM 2.1-tiny features for part-label v2 at 2x decoder resolution, plus a mirrored copy (pose env).
Usage: python seg_features2.py <scene_dir> <mode: gt|det> [max_scenes]
  gt  : crop = ground-truth body box (random padding 10-25%, jitter +-5%)           (training)
  det : crop = YOLOX Human-Art person box, 15% padding; no ground truth used for the crop  (testing)
For every scene writes <sid>_f2.npz (original) and <sid>_f2m.npz (crop mirrored left-right before encoding).
Arrays: embed 256x64x64, s1 64x128x128, s0 32x256x256 (uint8, per-channel linear quantisation: q, lo, hi),
        crop [X1, Y1, S] in 1280x960 pixels, mirrored flag, label 256x256 (parts.SEG2_CLASSES), side 256x256.
"""
import sys, os, json, glob, time, zlib
import numpy as np, cv2, torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
sys.path.insert(0, os.path.dirname(__file__))
import parts as PT
from seg_features import square_crop

torch.set_num_threads(2)
M = os.path.join(os.path.dirname(__file__), '..', 'models')
PARTV = os.environ.get('PARTV', '2')      # '2': part2.png / SEG2 labels; '3': part3.png / SEG3 labels (wound, TQ)
PFILE, SUF = ('_part3.png', '_f3') if PARTV == '3' else ('_part2.png', '_f2')


def quant(x):
    x = x.astype(np.float32); c = x.shape[0]
    lo = x.reshape(c, -1).min(1); hi = x.reshape(c, -1).max(1)
    q = np.round((x - lo[:, None, None]) / np.maximum(hi - lo, 1e-6)[:, None, None] * 255).astype(np.uint8)
    return q, lo.astype(np.float32), hi.astype(np.float32)


def dequant(q, lo, hi):
    return q.astype(np.float32) / 255 * (hi - lo)[:, None, None] + lo[:, None, None]


def load_f2(path):
    z = np.load(path)
    return (dequant(z['embed_q'], z['embed_lo'], z['embed_hi']), dequant(z['s1_q'], z['s1_lo'], z['s1_hi']),
            dequant(z['s0_q'], z['s0_lo'], z['s0_hi']), z)


def encode(P, crop):
    P.set_image(cv2.resize(crop, (1024, 1024)))
    f = P._features
    return (f['image_embed'][0].detach().numpy(), f['high_res_feats'][1][0].detach().numpy(),
            f['high_res_feats'][0][0].detach().numpy())


def main(D, mode, limit):
    model = build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml', os.path.join(M, 'sam2_1_hiera_tiny.pt'), device='cpu')
    P = SAM2ImagePredictor(model)
    det = None
    if mode == 'det':
        from rtmlib import YOLOX
        det = YOLOX(onnx_model=os.path.join(M, '20230928/yolox_onnx/yolox_m_8xb8-300e_humanart-c2c7a14a/end2end.onnx'),
                    model_input_size=(640, 640), backend='onnxruntime', device='cpu')
    n = 0; t0 = time.time()
    for f in sorted(glob.glob(os.path.join(D, 'C*_sidecar.json'))):
        sid = os.path.basename(f)[:-len('_sidecar.json')]
        out, outm = os.path.join(D, sid + SUF + '.npz'), os.path.join(D, sid + SUF + 'm.npz')
        pp = os.path.join(D, sid + PFILE)
        if os.path.exists(outm) or not os.path.exists(pp) or not os.path.exists(os.path.join(D, sid + '.jpg')):
            continue
        img = cv2.cvtColor(cv2.imread(os.path.join(D, sid + '.jpg')), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        seg, side = PT.decode2(pp) if PARTV != '3' else PT.decode3(pp)[:2]
        seg_full = cv2.resize(seg, (W, H), interpolation=cv2.INTER_NEAREST)
        side_full = cv2.resize(side, (W, H), interpolation=cv2.INTER_NEAREST)
        if mode == 'gt':
            body = (seg_full != 0) & (seg_full != PT.SEG2_CLASSES.index('OCC'))   # OCC index identical in SEG2 and SEG3
            ys, xs = np.nonzero(body)
            if len(xs) < 50:
                continue
            rng = np.random.default_rng(zlib.crc32(sid.encode()))
            w, h = xs.max() - xs.min(), ys.max() - ys.min()
            jx, jy = rng.uniform(-0.05, 0.05, 2) * max(w, h)
            box = (xs.min() + jx, ys.min() + jy, xs.max() + jx, ys.max() + jy); pad = rng.uniform(0.10, 0.25)
        else:
            b = det(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            box = tuple(float(v) for v in max(b, key=lambda q: (q[2] - q[0]) * (q[3] - q[1]))[:4]) if len(b) else (0, 0, W, H)
            pad = 0.15
        crop, (X1, Y1, S) = square_crop(img, box, pad)
        segc, _ = square_crop(seg_full[..., None].repeat(3, 2), box, pad)
        sidec, _ = square_crop(side_full[..., None].repeat(3, 2), box, pad)
        lab = cv2.resize(segc[..., 0], (256, 256), interpolation=cv2.INTER_NEAREST)
        sd = cv2.resize(sidec[..., 0], (256, 256), interpolation=cv2.INTER_NEAREST)
        for mirrored, path in ((False, out), (True, outm)):
            c = crop[:, ::-1].copy() if mirrored else crop
            e, s1, s0 = encode(P, c)
            L = lab[:, ::-1] if mirrored else lab
            Sd = sd[:, ::-1] if mirrored else sd
            if mirrored:
                Sd = np.where(Sd == 1, 2, np.where(Sd == 2, 1, 0)).astype(np.uint8)   # anatomical sides swap
            arrs = {}
            for k, x in (('embed', e), ('s1', s1), ('s0', s0)):
                arrs[k + '_q'], arrs[k + '_lo'], arrs[k + '_hi'] = quant(x)
            np.savez_compressed(path, crop=np.array([X1, Y1, S]), mirrored=mirrored, label=np.ascontiguousarray(L),
                                side=np.ascontiguousarray(Sd), **arrs)
        n += 1
        if limit and n >= limit:
            break
    print('FEAT2', D, mode, n, 'scenes', round(time.time() - t0), 's', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0)
