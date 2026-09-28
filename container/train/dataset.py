"""Dataset and label-safe augmentation.

Labels: an ICD-format JSON (same shape as predictions.json) whose injury_type values
are ground truth. Optional manifest CSV (image_id,group) gives the leakage-safe split
key: scene, manikin or source video. Without it, the group is the filename up to the
last underscore (e.g. scene012_view3.jpg -> scene012).

Augmentation rules:
- ALLOWED: photometric (brightness, contrast, saturation, hue, blur, JPEG quality,
  grayscale), rotation with canvas expansion (no content lost), horizontal flip WITH
  left/right label swap.
- FORBIDDEN: random crops, zoom-in, cutout on the body. Each can hide a limb and
  silently change its true label to not_testable.
"""
import csv, io, json, random
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from d2qual.sites import SITES, CLASSES, LR_SWAP
from d2qual.ingest import decode
from d2qual.preprocess import letterbox, to_tensor


def load_labels(labels_json: Path):
    doc = json.loads(Path(labels_json).read_text())
    out = {}
    for p in doc["predictions"]:
        m = {(s["body_region"], s["laterality"]): s["injury_type"] for s in p["sites"]}
        out[p["image_id"]] = [CLASSES.index(m[s]) for s in SITES]
    return out


def load_groups(ids, manifest: Path | None):
    if manifest and Path(manifest).exists():
        with open(manifest, newline="") as f:
            g = {r["image_id"]: r["group"] for r in csv.DictReader(f)}
        return {i: g.get(i, i) for i in ids}
    return {i: (Path(i).stem.rsplit("_", 1)[0] if "_" in Path(i).stem else Path(i).stem) for i in ids}


def group_split(ids, groups, fractions=(0.7, 0.15, 0.15), seed=0):
    uniq = sorted(set(groups[i] for i in ids))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    a = int(round(fractions[0] * n)); b = a + int(round(fractions[1] * n))
    sets = [set(uniq[:a]), set(uniq[a:b]), set(uniq[b:])]
    return [[i for i in ids if groups[i] in s] for s in sets]


def photometric(im: Image.Image, rng: random.Random):
    im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.7, 1.3))
    im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.7, 1.3))
    im = ImageEnhance.Color(im).enhance(rng.uniform(0.6, 1.4))
    if rng.random() < 0.2:
        im = ImageOps.grayscale(im).convert("RGB")
    if rng.random() < 0.3:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.5)))
    if rng.random() < 0.3:
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=rng.randint(35, 90)); buf.seek(0)
        im = Image.open(buf).convert("RGB")
    return im


def load_aux(aux_json):
    """Optional M3-13 labels: {image_id: {"head_end": 0|1, "facing": 0|1|2, "vis": [4 fractions]}}.
    Synthetic data provides them; DARPA data may not. Missing values are -1 (masked in the loss)."""
    if not aux_json or not Path(aux_json).exists():
        return {}
    return json.loads(Path(aux_json).read_text())


class SiteDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, ids, labels, size, mean, std, train=False, seed=0, rot_deg=20, aux=None):
        self.aux = aux or {}
        self.dir, self.ids, self.labels = Path(image_dir), list(ids), labels
        self.size, self.mean, self.std, self.train, self.rot = size, mean, std, train, rot_deg
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        iid = self.ids[i]
        im, _ = decode(self.dir / iid, self.size)
        if im is None:
            im = Image.new("RGB", (self.size, self.size))
        y = list(self.labels[iid])
        a = self.aux.get(iid, {})
        vis = list(a.get("vis", [-1.0] * 4))
        if self.train:
            rng = random.Random(self.rng.random() + i)
            im = photometric(im, rng)
            if self.rot and rng.random() < 0.5:
                im = im.rotate(rng.uniform(-self.rot, self.rot), resample=Image.BILINEAR, expand=True)
            if rng.random() < 0.5:
                im = ImageOps.mirror(im)
                y = [y[k] for k in LR_SWAP]           # mirrored anatomy: swap L/R labels
                vis = [vis[k] for k in LR_SWAP]       # visibility swaps with the sites; head end and facing do not
        x = to_tensor(letterbox(im, self.size), self.mean, self.std)
        aux_t = torch.tensor([a.get("head_end", -1), a.get("facing", -1)] + vis, dtype=torch.float32)
        return x, torch.tensor(y, dtype=torch.long), aux_t
