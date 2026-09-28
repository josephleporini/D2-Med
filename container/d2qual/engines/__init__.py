"""Engines turn decoded images into per-site class probabilities (N, 4, 4).

direct      DirectSiteNet baseline (Block T hedge model)
structured  Probe B staged pipeline (slot; not packaged yet)

model_config "engines": [{"type": "direct", "weight": 1.0, "dir": "."}, ...]
Absent -> a single direct engine from the model directory. The ensemble averages
probabilities by weight; M12 and M13 never see which engines ran (M3-12).
"""
import numpy as np

REGISTRY = {}


def _register():
    from .direct import DirectEngine
    from .structured import StructuredEngine
    REGISTRY.update({"direct": DirectEngine, "structured": StructuredEngine})


class Ensemble:
    def __init__(self, members):
        self.members = members                       # [(engine, weight)]
        self.size = max(e.size for e, _ in members)
        tot = sum(w for _, w in members)
        self.weights = [w / tot for _, w in members]
        self.name = "+".join(e.name for e, _ in members)

    def prepare(self, pil_image):
        return tuple(e.prepare(pil_image) for e, _ in self.members)

    def predict(self, items):
        probs, aux = None, None
        for k, (e, _) in enumerate(self.members):
            p, a = e.predict([it[k] for it in items])
            probs = self.weights[k] * p if probs is None else probs + self.weights[k] * p
            aux = aux or a                          # first engine that reports aux
        return probs, aux


def build(model_dir, cfg, device, tta_flip=True):
    _register()
    specs = cfg.get("engines") or [{"type": cfg.get("engine", "direct"), "weight": 1.0, "dir": "."}]
    members = []
    for s in specs:
        sub = model_dir / s.get("dir", ".")
        scfg = cfg if s.get("dir", ".") == "." else __import__("json").loads((sub / "model_config.json").read_text())
        members.append((REGISTRY[s["type"]](sub, scfg, device, tta_flip=tta_flip), float(s.get("weight", 1.0))))
    return Ensemble(members)
