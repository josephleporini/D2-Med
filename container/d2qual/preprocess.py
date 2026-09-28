"""Shared preprocessing for training and inference. Keeping one function for both
removes a classic train/serve skew.

Letterbox (resize long side, pad to square) rather than crop: a crop can cut a limb
out of frame and silently turn a labeled site into not_testable.
"""
import numpy as np
import torch
from PIL import Image


def letterbox(im: Image.Image, size: int, fill=(0, 0, 0)) -> Image.Image:
    w, h = im.size
    s = size / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    im = im.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(im, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def to_tensor(im: Image.Image, mean, std) -> torch.Tensor:
    a = np.asarray(im, dtype=np.float32) / 255.0
    a = (a - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return torch.from_numpy(a.transpose(2, 0, 1).copy())
