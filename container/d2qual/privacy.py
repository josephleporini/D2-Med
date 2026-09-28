"""M2 Privacy guard, Block T scope.

In Block T no image, crop, embedding or face region is ever written anywhere, and no
identity or face-recognition model is loaded. This module enforces that by policy:
it checks the model config against a deny-list and refuses to run a model whose
declared components include identity-capable models (S-07, Data Handling Agreement).
Face blurring for stored video arrives with Block O, where frames may be cached.
"""
DENY_SUBSTRINGS = ("arcface", "facenet", "insightface", "face_recognition", "reid", "re-id",
                   "person_reid", "deepface", "vggface", "adaface")


def check_model_config(model_cfg: dict):
    names = " ".join(str(x).lower() for x in model_cfg.get("components", [])) + " " + str(model_cfg.get("backbone", "")).lower()
    hits = [d for d in DENY_SUBSTRINGS if d in names]
    if hits:
        raise RuntimeError(f"privacy guard: identity-capable component declared ({hits}); refusing to run")
    return True
