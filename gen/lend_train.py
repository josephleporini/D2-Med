"""Train the limb-end classifier (extremity absent vs present) on cached windows (pose env, CPU).
Usage: python lend_train.py <train_dir> <epochs> <out_name> [val_frac]
Split by base scene (views of one body stay together). Class-balanced cross-entropy. Augmentation: random horizontal
flip of the feature map and mask together (approximate; SAM features are not exactly flip-equivariant) and
random 90-degree rotation of both. Reports validation AUC and balanced accuracy at p=0.5.
"""
import sys, os, glob, json, time, zlib, re
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
import lend as LE

torch.set_num_threads(2)
LAB = os.environ.get('LABEL', 'label')          # 'label' (absent vs present) or 'label3' (visible / stump / hidden)
NO = 3 if LAB == 'label3' else 2


def auc(y, p):
    o = np.argsort(p); y = np.asarray(y)[o]; n1 = y.sum(); n0 = len(y) - n1
    return float((np.cumsum(1 - y)[y == 1]).sum() / max(n1 * n0, 1))


def load(files):
    X, M, Y, S = [], [], [], []
    for f in files:
        z = np.load(f)
        if len(z['label']) == 0:
            continue
        X.append(z['feat']); M.append(z['mask']); Y.append(z[LAB]); S += [os.path.basename(f)] * len(z['label'])
    return (torch.from_numpy(np.concatenate(X)), torch.from_numpy(np.concatenate(M))[:, None],
            torch.from_numpy(np.concatenate(Y)), S)


if __name__ == '__main__':
    D, epochs, name = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    vf = float(sys.argv[4]) if len(sys.argv) > 4 else 0.15
    PAT = os.environ.get('PATTERN', 'C*_lend.npz')      # e.g. C*_lw.npz for the wound classifier
    files = sorted(f for d in D.split(',') for f in glob.glob(os.path.join(d, PAT)))
    base = lambda f: re.match(r'(C\d+)', os.path.basename(f)).group(1)
    val = [f for f in files if zlib.crc32(base(f).encode()) % 1000 < vf * 1000]
    tr = [f for f in files if f not in set(val)]
    frac = float(os.environ.get('TRAIN_FRAC', 1))
    if frac < 1:
        keep = set(sorted({base(f) for f in tr})[:int(frac * len({base(f) for f in tr}))]); tr = [f for f in tr if base(f) in keep]
    Xt, Mt, Yt, _ = load(tr); Xv, Mv, Yv, _ = load(val) if val else (None, None, None, None)
    print('train windows', len(Yt), 'absent', int(Yt.sum()), '| val', 0 if Yv is None else len(Yv), flush=True)
    net = LE.EndNet(NO); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-3)
    B = 32; steps = epochs * ((len(Yt) + B - 1) // B)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
    cnt = torch.bincount(Yt, minlength=NO).float(); w = cnt.sum() / (NO * cnt.clamp(min=1))
    crit = nn.CrossEntropyLoss(weight=w / w.mean())
    g = torch.Generator().manual_seed(0); log = []; t0 = time.time()
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Yt), generator=g); tot = 0
        for b in range(0, len(Yt), B):
            i = perm[b:b + B]; x = torch.cat([Xt[i].float(), Mt[i]], 1)
            if torch.rand(1, generator=g) < 0.5:
                x = x.flip(3)
            x = torch.rot90(x, int(torch.randint(0, 4, (1,), generator=g)), (2, 3))
            loss = crit(net(x), Yt[i]); opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(i)
        rec = {'epoch': ep + 1, 'loss': round(tot / len(Yt), 4), 's': round(time.time() - t0)}
        if Yv is not None:
            net.eval()
            with torch.no_grad():
                pa = torch.cat([torch.softmax(net(torch.cat([Xv[j:j + 256].float(), Mv[j:j + 256]], 1)), 1) for j in range(0, len(Yv), 256)]).numpy()
            p = pa[:, 1]; y = (Yv.numpy() == 1).astype(int); pr = pa.argmax(1) == 1
            if NO == 3:
                rec['val_recall_by_state'] = [round(float((pa.argmax(1) == k)[Yv.numpy() == k].mean()), 3) for k in range(3)]
            rec.update(val_auc=round(auc(y, p), 3), val_recall_absent=round(float(pr[y == 1].mean()), 3),
                       val_specificity=round(float((~pr)[y == 0].mean()), 3))
        log.append(rec); print(json.dumps(rec), flush=True)
    torch.save(net.state_dict(), os.path.join(LE.M, name + '.pt'))
    json.dump({'n_train': len(Yt), 'n_val': 0 if Yv is None else len(Yv), 'log': log}, open(os.path.join(D.split(',')[0], name + '_log.json'), 'w'), indent=1)
