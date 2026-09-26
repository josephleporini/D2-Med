"""Cache SAM 2.1-tiny encoder features on a square body crop (pose env).
Usage: python seg_features.py <scene_dir> <mode: gt|det> [max_scenes]
  gt  : crop = ground-truth body box from the part map, random padding 10-25% and jitter (training)
  det : crop = person-detector box (YOLOX Human-Art), 15% padding (testing; no ground truth used)
Writes <sid>_feat.npz: embed (256x64x64 fp16), s1 (64x128x128 fp16), crop box, and (gt mode) label map 128x128.
"""
import sys, os, json, glob, time
import numpy as np, cv2, torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
sys.path.insert(0, os.path.dirname(__file__))
import parts as PT

torch.set_num_threads(2)
M = os.path.join(os.path.dirname(__file__), '..', 'models')


def square_crop(img, box, pad):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    s = max(x2 - x1, y2 - y1) * (1 + 2 * pad)
    X1, Y1 = int(round(cx - s / 2)), int(round(cy - s / 2)); S = int(round(s))
    H, W = img.shape[:2]
    padded = cv2.copyMakeBorder(img, max(0, -Y1), max(0, Y1 + S - H), max(0, -X1), max(0, X1 + S - W),
                                cv2.BORDER_CONSTANT, value=0)
    crop = padded[Y1 + max(0, -Y1):Y1 + max(0, -Y1) + S, X1 + max(0, -X1):X1 + max(0, -X1) + S]
    return crop, (X1, Y1, S)


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
        sid = json.load(open(f))['scene_id']
        out = os.path.join(D, sid + '_feat.npz')
        if os.path.exists(out) or not os.path.exists(os.path.join(D, sid + '_part.png')):
            continue
        img = cv2.cvtColor(cv2.imread(os.path.join(D, sid + '.jpg')), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        seg, side = PT.decode(os.path.join(D, sid + '_part.png'))                # 640x480
        seg_full = cv2.resize(seg, (W, H), interpolation=cv2.INTER_NEAREST)
        side_full = cv2.resize(side, (W, H), interpolation=cv2.INTER_NEAREST)
        if mode == 'gt':
            body = (seg_full != 0) & (seg_full != PT.SEG_CLASSES.index('OCC'))
            ys, xs = np.nonzero(body)
            rng = np.random.default_rng(abs(hash(sid)) % 2 ** 32)
            w, h = xs.max() - xs.min(), ys.max() - ys.min()
            jx, jy = rng.uniform(-0.05, 0.05, 2) * max(w, h)
            box = (xs.min() + jx, ys.min() + jy, xs.max() + jx, ys.max() + jy); pad = rng.uniform(0.10, 0.25)
        else:
            b = det(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if len(b):
                b = max(b, key=lambda q: (q[2] - q[0]) * (q[3] - q[1])); box = tuple(float(v) for v in b[:4])
            else:
                box = (0, 0, W, H)
            pad = 0.15
        crop, (X1, Y1, S) = square_crop(img, box, pad)
        P.set_image(cv2.resize(crop, (1024, 1024)))
        feats = P._features
        emb = feats['image_embed'][0].detach().numpy().astype(np.float16)
        s1 = feats['high_res_feats'][1][0].detach().numpy().astype(np.float16)
        segc, _ = square_crop(seg_full[..., None].repeat(3, 2), box, pad)
        sidec, _ = square_crop(side_full[..., None].repeat(3, 2), box, pad)
        lab = cv2.resize(segc[..., 0], (128, 128), interpolation=cv2.INTER_NEAREST)
        sid128 = cv2.resize(sidec[..., 0], (128, 128), interpolation=cv2.INTER_NEAREST)
        np.savez_compressed(out, embed=emb, s1=s1, crop=np.array([X1, Y1, S]), label=lab, side=sid128)
        n += 1
        if limit and n >= limit:
            break
    print('FEAT', D, mode, n, 'scenes', round(time.time() - t0), 's')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0)
