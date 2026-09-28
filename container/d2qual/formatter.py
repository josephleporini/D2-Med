"""M12 Qual formatter: build, self-validate and atomically write predictions.json."""
import json, os
from pathlib import Path
from .sites import SITES, CLASSES

try:
    from jsonschema import Draft202012Validator
except ImportError:  # validation still runs the structural checks below
    Draft202012Validator = None


def build(image_ids, class_idx, team, email, version):
    """image_ids: list[str]; class_idx: dict image_id -> 4 class indices in SITES order."""
    preds = []
    for iid in image_ids:
        ci = class_idx[iid]
        preds.append({"image_id": iid, "sites": [
            {"body_region": r, "laterality": l, "injury_type": CLASSES[int(ci[k])]} for k, (r, l) in enumerate(SITES)]})
    return {"schema_version": "1.0", "submission": {"team_name": team, "version": version, "email": email}, "predictions": preds}


def validate(doc, schema_path: Path | None, expected_ids=None):
    errs = []
    if Draft202012Validator and schema_path and schema_path.exists():
        v = Draft202012Validator(json.loads(schema_path.read_text()))
        errs += [f"{e.json_path}: {e.message}" for e in v.iter_errors(doc)]
    ids = [p["image_id"] for p in doc.get("predictions", [])]
    if len(ids) != len(set(ids)):
        errs.append("duplicate image_id")
    for p in doc.get("predictions", []):
        got = sorted((s["body_region"], s["laterality"]) for s in p["sites"])
        if got != sorted(SITES):
            errs.append(f"{p['image_id']}: wrong site set")
    if expected_ids is not None and set(ids) != set(expected_ids):
        errs.append(f"image set mismatch: {len(set(expected_ids) - set(ids))} missing, {len(set(ids) - set(expected_ids))} extra")
    # ICD 3.2: no NaN / Infinity; json.dumps(allow_nan=False) in write() enforces it
    return errs


def write(doc, out_file: Path):
    """Atomic replace in the same directory, so a reader or a kill never sees a partial file."""
    tmp = out_file.with_name("." + out_file.name + ".tmp")
    data = json.dumps(doc, ensure_ascii=False, allow_nan=False, indent=1)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out_file)
