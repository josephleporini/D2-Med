#!/usr/bin/env python3
"""Proof that the flip-with-swap rule (LR_SWAP) is correct.

For random toy scenes: mirror(render(scene)) must equal render(scene'), where scene'
has the image mirrored (cx -> W-1-cx, angle -> -angle) and the limb states swapped
left <-> right. If LR_SWAP were wrong, the label of the mirrored image would not be
the swapped label and this test would fail.
"""
import random, sys, os
import numpy as np
from PIL import ImageOps
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from d2qual.sites import SITES, LR_SWAP
from make_toy_data import scene, render

rng = random.Random(3)
worst, n = 0.0, 60
for i in range(n):
    sc = scene(rng, 224, 224)
    sc["bg"] = (80, 80, 80)
    mirrored = ImageOps.mirror(render(sc))
    sc2 = dict(sc)
    sc2["cx"] = sc["W"] - sc["cx"]
    sc2["angle"] = -sc["angle"]
    sc2["states"] = {SITES[k]: sc["states"][SITES[LR_SWAP[k]]] for k in range(4)}
    diff = np.abs(np.asarray(mirrored, float) - np.asarray(render(sc2), float)).mean() / 255
    worst = max(worst, diff)
    # A deliberately WRONG rule (no swap) must produce a visibly different image when L != R
    sc3 = dict(sc2); sc3["states"] = dict(sc["states"])
    if sc["states"][SITES[0]] != sc["states"][SITES[1]] or sc["states"][SITES[2]] != sc["states"][SITES[3]]:
        bad = np.abs(np.asarray(mirrored, float) - np.asarray(render(sc3), float)).mean() / 255
        assert bad > 1.3 * diff, f"scene {i}: swap rule not clearly better than no-swap ({bad:.4f} vs {diff:.4f})"
print(f"flip-with-swap rule verified on {n} scenes; worst mean pixel diff {worst:.4f}")
assert worst < 0.01, "mirror equivalence failed"
print("PASS")
