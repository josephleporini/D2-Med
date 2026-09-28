#!/usr/bin/env python3
"""Toy casualty generator for SMOKE TESTS ONLY. Not training data for the real task.

Draws a stick-figure "casualty" with the same geometry problem as the real task:
anatomical left appears on the image's right when the figure faces the camera
head-up, and flips when it faces away or lies head-down. Facing is shown by a face
(eyes, mouth) versus a spine line and hair. Each limb is intact, wounded (red
blotch), amputated (short limb, red cap) or covered (grey blanket = not_testable).

Purpose: prove the whole chain (letterbox, flip-with-swap augmentation, site-query
head, decision layer, container) learns laterality, before any real data exists.

  python3 tools/make_toy_data.py --out data/toy --scenes 600 --views 2
  -> data/toy/images/*.png, data/toy/labels.json (ICD format), data/toy/manifest.csv
"""
import argparse, csv, json, math, random, sys, os
from pathlib import Path
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from d2qual.sites import SITES, CLASSES

STATE_P = [("no_injury", 0.55), ("wound", 0.2), ("amputation", 0.13), ("not_testable", 0.12)]


def pick(rng):
    r, acc = rng.random(), 0.0
    for c, p in STATE_P:
        acc += p
        if r < acc:
            return c
    return STATE_P[-1][0]


def scene(rng, W, H):
    return {"W": W, "H": H, "cx": rng.uniform(0.4, 0.6) * W, "cy": rng.uniform(0.45, 0.55) * H,
            "scale": rng.uniform(0.8, 1.1) * min(W, H) / 200, "front": rng.random() < 0.5,
            "head_down": rng.random() < 0.3, "angle": rng.uniform(-20, 20),
            "bg": tuple(rng.randint(40, 120) for _ in range(3)),
            "states": {s: pick(rng) for s in SITES}}


def image_side_of_left(sc):
    """Body-frame x-direction where the anatomical LEFT limbs sit (+1 = image right when head-up).
    Facing the camera puts anatomical left on the viewer's right. Head-down is handled by the
    180-degree rotation in render(), which moves them to image left; do not flip twice."""
    return +1 if sc["front"] else -1


def render(sc, noise_seed=None):
    W, H = sc["W"], sc["H"]
    im = Image.new("RGB", (W, H), sc["bg"])
    d = ImageDraw.Draw(im)
    if noise_seed is not None:
        r = random.Random(noise_seed)
        for _ in range(40):
            x, y = r.randint(0, W), r.randint(0, H)
            d.ellipse([x, y, x + r.randint(3, 12), y + r.randint(3, 12)], fill=tuple(r.randint(30, 140) for _ in range(3)))
    k = sc["scale"]
    th = math.radians(sc["angle"] + (180 if sc["head_down"] else 0))
    ca, sa = math.cos(th), math.sin(th)

    def P(x, y):  # body frame (x right, y down, head at negative y) -> image
        return (sc["cx"] + k * (x * ca - y * sa), sc["cy"] + k * (x * sa + y * ca))

    skin, cloth = (205, 180, 150), (70, 90, 60)
    lw = max(2, int(9 * k))
    # torso
    d.polygon([P(-22, -40), P(22, -40), P(20, 40), P(-20, 40)], fill=cloth)
    # head
    hx, hy = P(0, -58); hr = 16 * k
    d.ellipse([hx - hr, hy - hr, hx + hr, hy + hr], fill=skin)
    if sc["front"]:
        for ex in (-6, 6):
            x, y = P(ex, -62); d.ellipse([x - 2 * k, y - 2 * k, x + 2 * k, y + 2 * k], fill=(20, 20, 20))
        d.line([P(-5, -51), P(5, -51)], fill=(120, 30, 30), width=max(1, int(2 * k)))
        d.line([P(0, -30), P(0, -10)], fill=(200, 200, 200), width=max(1, int(2 * k)))   # chest tag
        d.line([P(-8, -20), P(8, -20)], fill=(200, 200, 200), width=max(1, int(2 * k)))
    else:
        d.pieslice([hx - hr, hy - hr, hx + hr, hy + hr], 0, 360, fill=(60, 40, 25))      # hair
        d.line([P(0, -38), P(0, 38)], fill=(30, 40, 30), width=max(1, int(3 * k)))       # spine seam
    side = image_side_of_left(sc)
    limbs = {("upper_extremity", "left"): ((22 * side, -34), (60 * side, 10)),
             ("upper_extremity", "right"): ((-22 * side, -34), (-60 * side, 10)),
             ("lower_extremity", "left"): ((12 * side, 40), (26 * side, 110)),
             ("lower_extremity", "right"): ((-12 * side, 40), (-26 * side, 110))}
    for site, (a, b) in limbs.items():
        st = sc["states"][site]
        end = b
        if st == "amputation":
            end = (a[0] + 0.45 * (b[0] - a[0]), a[1] + 0.45 * (b[1] - a[1]))
        d.line([P(*a), P(*end)], fill=skin, width=lw)
        if st == "amputation":
            x, y = P(*end); r = 5 * k; d.ellipse([x - r, y - r, x + r, y + r], fill=(150, 20, 20))
        elif st != "amputation":
            x, y = P(*b); r = 5 * k; d.ellipse([x - r, y - r, x + r, y + r], fill=skin)   # hand / foot
        if st == "wound":
            m = P((a[0] + b[0]) / 2, (a[1] + b[1]) / 2); r = 6 * k
            d.ellipse([m[0] - r, m[1] - r, m[0] + r, m[1] + r], fill=(170, 10, 10))
        if st == "not_testable":
            pts = [P(a[0] - 16, a[1] - 4), P(a[0] + 16, a[1] - 4), P(b[0] + 16, b[1] + 12), P(b[0] - 16, b[1] + 12)]
            d.polygon(pts, fill=(128, 128, 128))
    return im


def labels_of(sc):
    return [{"body_region": r, "laterality": l, "injury_type": sc["states"][(r, l)]} for r, l in SITES]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--scenes", type=int, default=600)
    ap.add_argument("--views", type=int, default=2); ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out); (out / "images").mkdir(parents=True, exist_ok=True)
    preds, rows, aux = [], [], {}
    for s in range(a.scenes):
        W, H = rng.choice([(256, 192), (192, 256), (224, 224)])
        base = scene(rng, W, H)
        for v in range(a.views):
            sc = dict(base); sc["angle"] = base["angle"] + rng.uniform(-8, 8)
            sc["cx"] = base["cx"] + rng.uniform(-10, 10); sc["bg"] = tuple(max(0, min(255, c + rng.randint(-20, 20))) for c in base["bg"])
            name = f"scene{s:04d}_v{v}.png"
            render(sc, noise_seed=rng.randint(0, 10**9)).save(out / "images" / name)
            preds.append({"image_id": name, "sites": labels_of(sc)})
            aux[name] = {"head_end": int(sc["head_down"]), "facing": 0 if sc["front"] else 1,
                         "vis": [0.0 if sc["states"][s_] == "not_testable" else 1.0 for s_ in SITES]}
            rows.append({"image_id": name, "group": f"scene{s:04d}"})
    (out / "labels.json").write_text(json.dumps({"schema_version": "1.0", "submission": {"team_name": "toy-truth", "version": "0", "email": "n/a"}, "predictions": preds}))
    (out / "aux.json").write_text(json.dumps(aux))
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "group"]); w.writeheader(); w.writerows(rows)
    print(f"{len(preds)} images, {a.scenes} scenes -> {out}")


if __name__ == "__main__":
    main()
