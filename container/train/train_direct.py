#!/usr/bin/env python3
"""Train the direct baseline (DirectSiteNet) and package a model directory.

  python3 train/train_direct.py --images DIR --labels labels.json [--manifest groups.csv]
      --out model/ [--backbone vit_small_patch14_dinov2.lvd142m --size 518 --pretrained]
      [--epochs 30 --batch 16 --lr 2e-4 --recall-floor 0.5]

Outputs in --out: weights.pt, model_config.json (backbone, size, normalization,
decision bias, fallback class, split and data hashes) and report.json (dev and test
metrics). The test split is scored once, after the decision layer is fixed on dev.

Pretrained backbones are downloaded at training time only. The inference container
never downloads (no network); it loads weights.pt. Whether backbones pretrained on
public human imagery are permitted is an open DARPA question (Dev Plan v1.2 §8).
"""
import argparse, hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from d2qual.model import DirectSiteNet
from d2qual.sites import CLASSES, SITES
from d2qual.decision import decide
from dataset import SiteDataset, load_labels, load_groups, group_split, load_aux
from metrics import evaluate
from fit_decision import fit_bias


def predict_probs(net, loader, device, amp, tta=True, with_aux=False):
    from d2qual.sites import LR_SWAP
    net.eval(); P, Y, A, AT = [], [], [], []
    with torch.inference_mode():
        for x, y, at in loader:
            x = x.to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                lg, ax = net.forward_all(x)
                p = lg.float().softmax(-1)
                if tta:
                    p = 0.5 * (p + net(torch.flip(x, dims=[3])).float().softmax(-1)[:, LR_SWAP, :])
            P.append(p.cpu().numpy()); Y.append(y.numpy())
            A.append(np.concatenate([ax["head_end"].float().argmax(-1, keepdim=True).cpu().numpy(),
                                     ax["facing"].float().argmax(-1, keepdim=True).cpu().numpy(),
                                     ax["vis"].float().sigmoid().cpu().numpy()], axis=1)); AT.append(at.numpy())
    if with_aux:
        return np.concatenate(P), np.concatenate(Y), np.concatenate(A), np.concatenate(AT)
    return np.concatenate(P), np.concatenate(Y)


def aux_metrics(A, AT):
    """Per-link measurements for the Block T chain (§6.1): L2 head end and facing, L3 visibility."""
    out = {}
    for k, name in ((0, "head_end_accuracy"), (1, "facing_accuracy")):
        m = AT[:, k] >= 0
        out[name] = round(float((A[m, k] == AT[m, k]).mean()), 4) if m.any() else None
    v, vt = A[:, 2:], AT[:, 2:]
    m = vt >= 0
    out["visibility_mae"] = round(float(np.abs(v[m] - vt[m]).mean()), 4) if m.any() else None
    out["visibility_threshold_accuracy_0.1"] = round(float(((v[m] >= 0.1) == (vt[m] >= 0.1)).mean()), 4) if m.any() else None
    return out


def sha_of_files(paths):
    h = hashlib.sha256()
    for p in sorted(paths, key=str):
        h.update(Path(p).name.encode()); h.update(Path(p).read_bytes())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True); ap.add_argument("--labels", required=True)
    ap.add_argument("--manifest"); ap.add_argument("--out", required=True)
    ap.add_argument("--backbone", default="vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--size", type=int, default=518); ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--backbone-lr-mult", type=float, default=0.1)
    ap.add_argument("--recall-floor", type=float, default=0.5); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--no-rotate", action="store_true")
    ap.add_argument("--fallback-class", default=None, help="default: majority class of the training split")
    ap.add_argument("--aux", help="optional aux labels JSON (M3-13): head_end, facing, vis")
    ap.add_argument("--aux-weight", type=float, default=0.3)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    labels = load_labels(a.labels)
    aux = load_aux(a.aux)
    ids = sorted(i for i in labels if (Path(a.images) / i).exists())
    groups = load_groups(ids, a.manifest)
    tr, dv, te = group_split(ids, groups, seed=a.seed)
    print(json.dumps({"device": str(device), "images": len(ids), "groups": len(set(groups.values())),
                      "train": len(tr), "dev": len(dv), "test": len(te)}), flush=True)

    net = DirectSiteNet(a.backbone, pretrained=a.pretrained, img_size=a.size).to(device)
    dcfg = net.backbone.pretrained_cfg if hasattr(net.backbone, "pretrained_cfg") else {}
    mean, std = list(dcfg.get("mean", (0.485, 0.456, 0.406))), list(dcfg.get("std", (0.229, 0.224, 0.225)))

    mk = lambda ids_, train: torch.utils.data.DataLoader(
        SiteDataset(a.images, ids_, labels, a.size, mean, std, train=train, seed=a.seed, rot_deg=0 if a.no_rotate else 20, aux=aux),
        batch_size=a.batch, shuffle=train, num_workers=a.workers, drop_last=train and len(ids_) > a.batch,
        persistent_workers=a.workers > 0)
    dl_tr, dl_dv, dl_te = mk(tr, True), mk(dv, False), mk(te, False)

    # Class weights: inverse square-root frequency on the training split
    ytr = np.array([labels[i] for i in tr]).ravel()
    freq = np.bincount(ytr, minlength=len(CLASSES)).astype(float) + 1.0
    w = torch.tensor((freq.sum() / freq) ** 0.5, dtype=torch.float32)
    w = (w / w.mean()).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)

    params = [{"params": net.backbone.parameters(), "lr": a.lr * a.backbone_lr_mult},
              {"params": net.head.parameters(), "lr": a.lr}]
    opt = torch.optim.AdamW(params, weight_decay=0.05)
    steps = a.epochs * max(1, len(dl_tr))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, steps // 20)) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    scaler = torch.amp.GradScaler(enabled=amp)

    best, best_state, hist = -1.0, None, []
    t0 = time.time()
    for ep in range(a.epochs):
        net.train(); tot = 0.0; n = 0
        for x, y, at in dl_tr:
            x, y, at = x.to(device), y.to(device), at.to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits, ax = net.forward_all(x)
                loss = loss_fn(logits.reshape(-1, len(CLASSES)).float(), y.reshape(-1))
                if aux:                                   # M3-13 heads, masked where labels are missing
                    ce = nn.functional.cross_entropy
                    la = ce(ax["head_end"].float(), at[:, 0].long(), ignore_index=-1) if (at[:, 0] >= 0).any() else 0.0
                    lb = ce(ax["facing"].float(), at[:, 1].long(), ignore_index=-1) if (at[:, 1] >= 0).any() else 0.0
                    vm = at[:, 2:] >= 0
                    lv = nn.functional.binary_cross_entropy_with_logits(ax["vis"].float()[vm], at[:, 2:][vm]) if vm.any() else 0.0
                    loss = loss + a.aux_weight * (la + lb + lv)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * x.shape[0]; n += x.shape[0]
        pdv, ydv = predict_probs(net, dl_dv, device, amp)
        m = evaluate(ydv, decide(pdv), n_boot=0)
        hist.append({"epoch": ep + 1, "loss": round(tot / max(1, n), 4), "dev_acc": m["accuracy"], "dev_macro_f1": m["macro_f1"],
                     "dev_min_recall": m["min_class_recall"], "swaps": m["laterality_swaps"], "s": round(time.time() - t0, 1)})
        print(json.dumps(hist[-1]), flush=True)
        if m["macro_f1"] is not None and m["macro_f1"] > best:      # select on macro-F1, not accuracy
            best = m["macro_f1"]; best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    net.load_state_dict(best_state)
    pdv, ydv = predict_probs(net, dl_dv, device, amp)
    dec = fit_bias(ydv, pdv, floor=a.recall_floor)
    pte, yte, ate, atte = predict_probs(net, dl_te, device, amp, with_aux=True)
    g_te = [groups[i] for i in te]
    rep = {"dev_raw": evaluate(ydv, decide(pdv), [groups[i] for i in dv]),
           "dev_biased": evaluate(ydv, decide(pdv, dec["bias"]), [groups[i] for i in dv]),
           "test_biased": evaluate(yte, decide(pte, dec["bias"]), g_te),
           "test_aux_links": aux_metrics(ate, atte) if aux else None,
           "decision_layer": dec, "history": hist}

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out / "weights.pt")
    majority = CLASSES[int(np.bincount(ytr, minlength=len(CLASSES)).argmax())]
    cfg = {"model_type": "direct_site_query", "backbone": a.backbone, "input_size": a.size, "mean": mean, "std": std,
           "classes": CLASSES, "sites": [list(s) for s in SITES], "decision_bias": dec["bias"],
           "fallback_class": a.fallback_class or majority, "weights": "weights.pt",
           "components": [a.backbone], "pretrained_backbone": bool(a.pretrained), "aux_trained": bool(aux),
           "train": {"labels_sha": sha_of_files([a.labels]), "n_train": len(tr), "n_dev": len(dv), "n_test": len(te),
                     "split_seed": a.seed, "epochs": a.epochs, "best_dev_macro_f1": round(best, 4),
                     "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}}
    (out / "model_config.json").write_text(json.dumps(cfg, indent=2))
    (out / "report.json").write_text(json.dumps(rep, indent=2))
    t = rep["test_biased"]
    print(json.dumps({"TEST_accuracy": t["accuracy"], "ci95": t.get("accuracy_ci95"), "macro_f1": t["macro_f1"],
                      "min_recall": t["min_class_recall"], "majority_baseline": t["majority_baseline_accuracy"],
                      "swaps": t["laterality_swaps"], "aux_links": rep["test_aux_links"], "decision": dec}), flush=True)


if __name__ == "__main__":
    main()
