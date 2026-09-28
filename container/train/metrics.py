"""Evaluation metrics for the qualification task.

DARPA scores overall site accuracy plus unnamed ML metrics, so we report the set a
reviewer is likely to look at: per-class precision/recall/F1, macro-F1, confusion
matrix, per-site accuracy, majority-class baseline, and the laterality-swap count
(sites wrong only because left and right were exchanged). Confidence intervals are
group bootstraps, so near-duplicate images from one scene don't shrink the interval.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from d2qual.sites import CLASSES, SITES


def evaluate(y_true, y_pred, groups=None, n_boot=500, seed=0):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)       # (N, 4)
    t, p = y_true.ravel(), y_pred.ravel()
    K = len(CLASSES)
    cm = np.zeros((K, K), dtype=int)
    np.add.at(cm, (t, p), 1)
    per = {}
    for k, c in enumerate(CLASSES):
        tp = cm[k, k]; fn = cm[k].sum() - tp; fp = cm[:, k].sum() - tp
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) and not np.isnan(prec + rec) else 0.0
        per[c] = {"support": int(cm[k].sum()), "precision": _r(prec), "recall": _r(rec), "f1": _r(f1)}
    present = [c for c in CLASSES if per[c]["support"] > 0]
    macro_f1 = float(np.mean([per[c]["f1"] for c in present])) if present else float("nan")
    acc = float((t == p).mean())
    majority = int(np.bincount(t, minlength=K).argmax())
    out = {
        "n_images": int(y_true.shape[0]), "n_sites": int(t.size),
        "accuracy": _r(acc), "macro_f1": _r(macro_f1),
        "majority_class": CLASSES[majority], "majority_baseline_accuracy": _r(float((t == majority).mean())),
        "per_class": per,
        "per_site_accuracy": {f"{r}:{l}": _r(float((y_true[:, i] == y_pred[:, i]).mean())) for i, (r, l) in enumerate(SITES)},
        "confusion_rows_true_cols_pred": {"labels": CLASSES, "matrix": cm.tolist()},
        "laterality_swaps": swap_count(y_true, y_pred),
        "min_class_recall": _r(min(per[c]["recall"] for c in present)) if present else None,
    }
    if n_boot and y_true.shape[0] > 1:
        out["accuracy_ci95"] = _boot(y_true, y_pred, groups, n_boot, seed)
    return out


def swap_count(y_true, y_pred):
    """Count region pairs (upper, lower) where prediction == truth with left and right exchanged."""
    n = 0
    for L, R in ((0, 1), (2, 3)):
        tl, tr, pl, pr = y_true[:, L], y_true[:, R], y_pred[:, L], y_pred[:, R]
        n += int(((tl != tr) & (pl == tr) & (pr == tl)).sum())
    return n


def _boot(y_true, y_pred, groups, n_boot, seed):
    rng = np.random.default_rng(seed)
    g = np.arange(y_true.shape[0]) if groups is None else np.asarray(groups)
    uniq = np.unique(g)
    idx_by = {u: np.where(g == u)[0] for u in uniq}
    accs = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ix = np.concatenate([idx_by[u] for u in pick])
        accs.append((y_true[ix] == y_pred[ix]).mean())
    lo, hi = np.percentile(accs, [2.5, 97.5])
    return [_r(float(lo)), _r(float(hi))]


def _r(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)
