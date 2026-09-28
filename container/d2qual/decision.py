"""Decision layer: per-site class probabilities -> ICD class.

class = argmax(log p + bias). The bias vector is fitted on a development split
(train/fit_decision.py) to maximize site accuracy subject to a recall floor on
every class, so no class is traded away for accuracy (Probe B lesson: an
accuracy-only layer drove amputation recall to 25%).
"""
import numpy as np
from .sites import CLASSES


def decide(probs: np.ndarray, bias=None) -> np.ndarray:
    """probs (N, 4 sites, 4 classes) -> class indices (N, 4)."""
    b = np.zeros(len(CLASSES)) if bias is None else np.asarray(bias, dtype=np.float64)
    return np.argmax(np.log(np.clip(probs, 1e-9, 1.0)) + b, axis=-1)
