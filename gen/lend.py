"""Limb-end stage (Test F) shared code (pose env).

For one limb (a side's limb mask on a 640x480 class map):
  end    = limb pixel with the largest geodesic distance (8-connected, inside the limb) from the limb's torso contact
           (pixels within 3 px of the torso); with no torso contact, from the limb pixel nearest the torso centroid.
  window = square centred on the end, side S640 = clip(0.6 x torso-plus-head extent, 32, 300) px at 640x480
           (80 px when the torso is not found); read from the 1280x960 image (2x).
  input  = SAM 2.1-tiny image embedding of the window (resized to 1024), average-pooled 64->32  (256x32x32)
           + the limb's own mask in the window (1x32x32), so the classifier knows which limb is meant.
Target:  extremity anatomically absent (the site is amputated) vs present.
"""
import os, numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from scipy import ndimage, sparse
from scipy.sparse.csgraph import dijkstra
import test_a_masks as TA

M = os.path.join(os.path.dirname(__file__), '..', 'models')


def limb_end(limb, torso):
    ys, xs = np.nonzero(limb)
    n = len(ys)
    if n == 0:
        return None
    idx = -np.ones(limb.shape, np.int64); idx[ys, xs] = np.arange(n)
    rows, cols, w = [], [], []
    H, W = limb.shape
    for dy, dx, c in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.414), (1, -1, 1.414)):
        y2, x2 = ys + dy, xs + dx
        ok = (y2 >= 0) & (y2 < H) & (x2 >= 0) & (x2 < W)
        ok[ok] = limb[y2[ok], x2[ok]]
        rows.append(idx[ys[ok], xs[ok]]); cols.append(idx[y2[ok], x2[ok]]); w.append(np.full(ok.sum(), c))
    G = sparse.coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    contact = limb & ndimage.binary_dilation(torso, iterations=3)
    if contact.any():
        src = idx[contact]
    else:
        tp = np.argwhere(torso)
        c = tp.mean(0) if len(tp) else np.array([H / 2, W / 2])
        src = [int(np.argmin((ys - c[0]) ** 2 + (xs - c[1]) ** 2))]
    d = dijkstra(G, directed=False, indices=np.unique(src), min_only=True)
    d[~np.isfinite(d)] = -1
    k = int(np.argmax(d))
    return int(xs[k]), int(ys[k])


def window_size(torso):
    t = TA.extent(torso)
    return int(np.clip(0.6 * t, 32, 300)) if t > 40 else 80


def crop_1280(img, cx640, cy640, s640):
    cx, cy, S = 2 * cx640, 2 * cy640, 2 * s640
    X1, Y1 = cx - S // 2, cy - S // 2
    H, W = img.shape[:2]
    p = cv2.copyMakeBorder(img, max(0, -Y1), max(0, Y1 + S - H), max(0, -X1), max(0, X1 + S - W), cv2.BORDER_CONSTANT, value=0)
    oy, ox = max(0, -Y1), max(0, -X1)
    return p[Y1 + oy:Y1 + oy + S, X1 + ox:X1 + ox + S]


def mask_window(mask, cx, cy, s):
    m = cv2.copyMakeBorder(mask.astype(np.float32), s, s, s, s, cv2.BORDER_CONSTANT, value=0)
    w = m[cy + s - s // 2:cy + s - s // 2 + s, cx + s - s // 2:cx + s - s // 2 + s]
    return cv2.resize(w, (32, 32), interpolation=cv2.INTER_AREA)


def encode(P, crop):
    P.set_image(cv2.resize(crop, (1024, 1024)))
    e = P._features['image_embed']
    return F.avg_pool2d(e.float(), 2)[0].detach().cpu().numpy().astype(np.float16)


class EndNet(nn.Module):
    def __init__(self, n_out=2):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(257, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(), nn.Dropout2d(0.1),
                                  nn.Conv2d(96, 96, 3, stride=2, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
                                  nn.Conv2d(96, 96, 3, stride=2, padding=1), nn.BatchNorm2d(96), nn.ReLU())
        self.fc = nn.Linear(192, n_out)

    def forward(self, x):
        h = self.body(x)
        return self.fc(torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1))


def load_sam():
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    return SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml', os.path.join(M, 'sam2_1_hiera_tiny.pt'),
                                         device='cuda' if torch.cuda.is_available() else 'cpu'))
