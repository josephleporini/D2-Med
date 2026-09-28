"""Probe B staged pipeline, BT-1 build (segmenter with side and limb heads, pose, limb-end and limb-wound
classifiers, exported logistic decision layer, test-time mirror). Contract:

    engine = StructuredEngine(model_dir, cfg, device, tta_flip=True)
    engine.prepare(pil_image) -> RGB uint8 array
    engine.predict(list_of_prepared) -> (np.ndarray (N, 4 sites, 4 classes), aux None)

Site order LUE, RUE, LLE, RLE and class order no_injury, wound, amputation, not_testable, as d2qual.sites.
The pipeline code is D2-Med gen/ and jobs/ (engine_bt1.py); model_config "structured" names where it and the weights are:

    {"engine": "structured", "input_size": 1280, "fallback_class": "no_injury",
     "components": ["sam2.1_hiera_tiny", "rtmw_wholebody", "yolox_m_humanart", "lend3d", "lwound2", "bt1_limb"],
     "structured": {"code_dir": "d2med", "weights": "seg4_limb_bt1.pt", "decision": "decision_layer_bt1.json"}}

code_dir is relative to the model directory unless absolute; D2_BT1_CODE overrides it. The per-image engine rows can be
logged for out-of-fold scoring with D2_FEATURES_OUT=<jsonl> (development runs only; features, never pixels).
"""
import json, os, sys, time
import numpy as np


class StructuredEngine:
    name = "structured"

    def __init__(self, model_dir, cfg, device, tta_flip=True):
        sc = cfg.get("structured", {})
        code = os.environ.get("D2_BT1_CODE") or str(model_dir / sc.get("code_dir", "d2med"))
        for sub in ("gen", "jobs"):
            p = os.path.join(code, sub)
            if p not in sys.path:
                sys.path.insert(0, p)
        import engine_bt1 as EB
        self.size = int(cfg.get("input_size", 1280))
        self.E = EB.BT1Engine(str(model_dir), str(model_dir / sc.get("weights", "seg4_limb_bt1.pt")),
                              str(model_dir / sc.get("decision", "decision_layer_bt1.json")), tta=tta_flip, device=str(device))
        self.sites = EB.SITES
        out = os.environ.get("D2_FEATURES_OUT")
        self.flog = open(out, "a") if out else None
        self.current_ids = []
        from ..sites import CLASSES
        self.fallback = CLASSES.index(cfg.get("fallback_class", "no_injury"))

    def prepare(self, pil_image):
        return np.asarray(pil_image.convert("RGB"), dtype=np.uint8)

    def predict(self, items):
        P = np.zeros((len(items), 4, 4))
        for n, rgb in enumerate(items):
            t0 = time.time()
            try:
                P[n], info = self.E.predict_image(rgb)
            except Exception as ex:                 # one bad image must not fail its batch: fallback class for it
                P[n] = np.eye(4)[self.fallback]
                info = dict(frame=None, rows=None, t=None, error=f"{type(ex).__name__}: {ex}")
                print(json.dumps({"msg": "structured_image_failed", "image_id": self.current_ids[n] if n < len(self.current_ids) else None,
                                  "error": info["error"]}), file=sys.stderr, flush=True)
            if self.flog:
                self.flog.write(json.dumps(dict(image_id=self.current_ids[n] if n < len(self.current_ids) else None, error=info.get("error"), frame=info["frame"], sites=info["rows"], sites_flip=info.get("rows_flip"),
                                                t=info["t"], t_flip=info.get("t_flip"), s=round(time.time() - t0, 3),
                                                p=P[n].round(4).tolist())) + "\n")
                self.flog.flush()
        return P, None
