"""Fit the decision-layer bias on a development split.

Maximize site accuracy subject to recall >= floor for every class present.
Coordinate search over a small grid; deterministic; seconds on CPU.
If no bias meets the floor, return the one with the highest minimum recall and say so.
"""
import itertools
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from d2qual.decision import decide
from d2qual.sites import CLASSES


def _score(y, probs, b, floor):
    pred = decide(probs, b)
    t, p = y.ravel(), pred.ravel()
    acc = (t == p).mean()
    recs = []
    for k in range(len(CLASSES)):
        m = t == k
        if m.any():
            recs.append((p[m] == k).mean())
    mr = min(recs) if recs else 0.0
    return acc, mr


def fit_bias(y, probs, floor=0.5, grid=np.linspace(-3, 3, 25), rounds=3):
    y, probs = np.asarray(y), np.asarray(probs)
    b = np.zeros(len(CLASSES))
    best = (_score(y, probs, b, floor), b.copy())

    def key(s):
        acc, mr = s
        return (mr >= floor, acc if mr >= floor else mr)

    for _ in range(rounds):
        for k in range(1, len(CLASSES)):          # class 0 is the reference
            for v in grid:
                c = b.copy(); c[k] = v
                s = _score(y, probs, c, floor)
                if key(s) > key(best[0]):
                    best = (s, c.copy())
            b = best[1].copy()
    (acc, mr), b = best
    return {"bias": [round(float(x), 4) for x in b], "dev_accuracy": round(float(acc), 4),
            "dev_min_recall": round(float(mr), 4), "recall_floor": floor, "floor_met": bool(mr >= floor)}
