"""Geometric augmentations with the label transform bound to the image transform (M3-03, v0.3.3).

Rule: any reflection, about any axis, swaps anatomical left and right; a rotation (including 180°, which is two
reflections) does not. Each op carries its own label rule, so an image transform can never be applied without its
label transform. Sites follow d2qual.sites.SITES; LR_SWAP exchanges left and right.
"""
from PIL import Image, ImageOps
from .sites import LR_SWAP


class Op:
    def __init__(self, name, fn, reflections):
        self.name, self.fn, self.reflections = name, fn, reflections

    @property
    def swaps(self):
        return self.reflections % 2 == 1

    def __call__(self, image, labels, per_site=()):
        """image, labels (4 site values) and any other per-site lists -> transformed, as one step"""
        im = self.fn(image)
        if self.swaps:
            labels = [labels[k] for k in LR_SWAP]
            per_site = tuple([v[k] for k in LR_SWAP] for v in per_site)
        return (im, list(labels)) + tuple(list(v) for v in per_site)


def rotate(deg):
    return Op(f"rotate{deg:+.1f}", lambda im: im.rotate(deg, resample=Image.BILINEAR, expand=True), 0)


MIRROR_H = Op("mirror_h", ImageOps.mirror, 1)          # left-right reflection
MIRROR_V = Op("mirror_v", ImageOps.flip, 1)            # top-bottom reflection: still a reflection, swaps
ROT180 = Op("rot180", lambda im: im.rotate(180, expand=True), 2)
IDENTITY = Op("identity", lambda im: im, 0)
OPS = [IDENTITY, MIRROR_H, MIRROR_V, ROT180]
