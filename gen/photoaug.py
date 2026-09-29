"""Photometric and background augmentation toward real manikin photographs (BT-2b; Real Fidelity Gap v1.0).

Real training photos differ from the renders in contrast (pixel std 0.25 against 0.06), saturation (0.26 against 0.15),
brightness (darker), red content (3.6% of pixels against about 0) and background clutter. This module widens the
training distribution to cover those ranges. It never changes a label: geometry is untouched, background replacement
only writes to pixels whose part label is background, and red blobs go only on background pixels.

photo_aug(crop, bg, rng) -> crop
  crop  uint8 RGB (S, S, 3), the training crop
  bg    bool (S, S), True where the part label is background (floor or outside the image)
  rng   numpy Generator
"""
import numpy as np, cv2

# background palettes (RGB, 0-255): grass, dirt, gravel, concrete, sand, camouflage, tarp, indoor floor, dark
PALETTES = [
    [(70, 90, 40), (95, 110, 55), (50, 65, 30)],
    [(110, 85, 60), (85, 65, 45), (140, 115, 85)],
    [(120, 118, 110), (90, 88, 82), (150, 148, 140)],
    [(150, 150, 145), (125, 125, 120), (175, 172, 165)],
    [(185, 160, 120), (160, 135, 100), (205, 185, 150)],
    [(85, 90, 60), (120, 105, 75), (55, 55, 40), (140, 130, 95)],
    [(60, 75, 95), (45, 60, 80), (85, 100, 120)],
    [(170, 160, 140), (110, 100, 90), (200, 195, 185)],
    [(35, 35, 35), (60, 55, 50), (20, 22, 25)],
]


def _noise(S, rng, scales=(4, 16, 64)):
    """multi-scale value noise in [0, 1]"""
    out = np.zeros((S, S), np.float32)
    for k, s in enumerate(scales):
        n = max(2, S // s)
        g = rng.random((n, n)).astype(np.float32)
        out += cv2.resize(g, (S, S), interpolation=cv2.INTER_CUBIC) / (k + 1)
    out -= out.min(); out /= max(out.max(), 1e-6)
    return out


def background(S, rng):
    pal = np.array(PALETTES[int(rng.integers(len(PALETTES)))], np.float32)
    n = _noise(S, rng, tuple(int(x) for x in rng.choice([2, 4, 8, 16, 32, 64, 128], 3, replace=False)))
    idx = np.clip((n * len(pal)).astype(int), 0, len(pal) - 1)
    tex = pal[idx]
    tex = cv2.GaussianBlur(tex, (0, 0), rng.uniform(0.5, 3.0))
    grain = rng.normal(0, rng.uniform(3, 18), (S, S, 1)).astype(np.float32)
    shade = 0.6 + 0.8 * _noise(S, rng, (1, 2))[..., None]                 # large-scale lighting falloff
    return np.clip(tex * shade + grain, 0, 255)


def red_blobs(S, rng, k):
    """k irregular red or orange blobs (pooled blood, red gear), float mask in [0, 1] and color"""
    m = np.zeros((S, S), np.float32)
    for _ in range(k):
        cx, cy = rng.uniform(0, S, 2); a, b = rng.uniform(0.02, 0.12, 2) * S
        cv2.ellipse(m, (int(cx), int(cy)), (int(a), int(b)), float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
    m = cv2.GaussianBlur(m, (0, 0), rng.uniform(1, 4)) * (_noise(S, rng, (8, 32)) > rng.uniform(0.2, 0.5))
    col = np.array([rng.uniform(90, 200), rng.uniform(0, 40), rng.uniform(0, 30)], np.float32)
    if rng.random() < 0.3:                                                   # orange or red equipment
        col = np.array([rng.uniform(180, 240), rng.uniform(40, 110), rng.uniform(0, 40)], np.float32)
    return np.clip(m, 0, 1), col


def photo_aug(crop, bg, rng):
    c = crop.astype(np.float32); S = c.shape[0]
    bgf = bg.astype(np.float32)[..., None]
    if rng.random() < 0.5:                                                   # background replacement
        a = rng.uniform(0.6, 1.0)
        c = c * (1 - bgf * a) + background(S, rng) * bgf * a
    if rng.random() < 0.35:                                                  # red blobs on background only
        m, col = red_blobs(S, rng, int(rng.integers(1, 5)))
        m = m[..., None] * bgf
        c = c * (1 - m) + col[None, None] * m
    # tone: contrast about the mean, gamma, exposure, saturation
    mu = c.mean((0, 1), keepdims=True)
    c = mu + (c - mu) * rng.uniform(1.0, 2.2)
    c = np.clip(c, 0, 255)
    c = 255 * (c / 255) ** rng.uniform(0.7, 1.4)
    c = c * rng.uniform(0.6, 1.15)
    hsv = cv2.cvtColor(np.clip(c, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.9, 1.9), 0, 255)
    c = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    if rng.random() < 0.4:                                                   # local contrast (clutter, hard shadows)
        lab = cv2.cvtColor(np.clip(c, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        cl = cv2.createCLAHE(clipLimit=float(rng.uniform(1.5, 4.0)), tileGridSize=(8, 8))
        lab[..., 0] = cl.apply(lab[..., 0]); c = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32)
    out = np.clip(c, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:                                                   # JPEG artifacts
        ok, enc = cv2.imencode('.jpg', out[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(40, 95))])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1].copy()
    return out
