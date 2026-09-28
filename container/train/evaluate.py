#!/usr/bin/env python3
"""Score a predictions.json against an ICD-format labels.json.

  python3 train/evaluate.py predictions.json labels.json [--manifest groups.csv]
"""
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
from dataset import load_labels, load_groups
from metrics import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("pred"); ap.add_argument("labels"); ap.add_argument("--manifest")
a = ap.parse_args()
truth, pred = load_labels(a.labels), load_labels(a.pred)
ids = sorted(set(truth) & set(pred))
missing = sorted(set(truth) - set(pred))
g = load_groups(ids, a.manifest)
m = evaluate([truth[i] for i in ids], [pred[i] for i in ids], [g[i] for i in ids])
m["images_missing_from_predictions"] = len(missing)
print(json.dumps(m, indent=2))
