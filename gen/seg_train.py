"""Train a light side-agnostic part decoder on cached SAM 2.1-tiny features (pose env, CPU).
Usage: python seg_train.py <train_dir> <epochs> [val_frac]
Classes: parts.SEG_CLASSES (BG, TORSO, HEAD, OCC, upper_arm, forearm, hand, thigh, shank, foot) at 128x128.
Split is by scene id hash (stable as more renders arrive). Writes models/seg_decoder.pt and a JSON log.
"""
import sys, os, glob, json, time, zlib
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
import parts as PT

torch.set_num_threads(2)
NC = len(PT.SEG_CLASSES)
M = os.path.join(os.path.dirname(__file__), '..', 'models')


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.red = nn.Sequential(nn.Conv2d(256, 128, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.fuse = nn.Sequential(nn.Conv2d(128 + 64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                                  nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                  nn.Conv2d(64, NC, 1))

    def forward(self, emb, s1):
        x = F.interpolate(self.red(emb), scale_factor=2, mode='bilinear', align_corners=False)
        return self.fuse(torch.cat([x, s1], 1))


def is_val(sid, frac):
    return (zlib.crc32(sid.encode()) % 1000) < frac * 1000


def load(f):
    z = np.load(f)
    return (torch.from_numpy(z['embed'].astype(np.float32)), torch.from_numpy(z['s1'].astype(np.float32)),
            torch.from_numpy(z['label'].astype(np.int64)))


def iou_table(conf):
    inter = np.diag(conf); union = conf.sum(0) + conf.sum(1) - inter
    return {c: round(float(inter[i] / union[i]), 3) if union[i] else None for i, c in enumerate(PT.SEG_CLASSES)}


def evaluate(net, files):
    net.eval(); conf = np.zeros((NC, NC), np.int64)
    with torch.no_grad():
        for f in files:
            e, s, y = load(f)
            p = net(e[None], s[None]).argmax(1)[0].numpy()
            conf += np.bincount(y.numpy().ravel() * NC + p.ravel(), minlength=NC * NC).reshape(NC, NC)
    return conf


def main(D, epochs, vf):
    files = sorted(glob.glob(os.path.join(D, 'C*_feat.npz')))
    sid = lambda f: os.path.basename(f).split('_')[0]
    val = [f for f in files if is_val(sid(f), vf)]; tr = [f for f in files if not is_val(sid(f), vf)]
    print('train', len(tr), 'val', len(val), flush=True)
    cnt = np.zeros(NC)
    for f in tr:
        cnt += np.bincount(np.load(f)['label'].ravel(), minlength=NC)
    w = 1 / np.sqrt(cnt / cnt.sum() + 1e-6); w = w / w.mean()
    print('class px share', dict(zip(PT.SEG_CLASSES, np.round(cnt / cnt.sum(), 4))), flush=True)
    net = Decoder(); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * ((len(tr) + 7) // 8))
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
    rng = np.random.default_rng(0); log = []; t0 = time.time()
    for ep in range(epochs):
        net.train(); order = rng.permutation(len(tr)); tot = 0
        for b in range(0, len(tr), 8):
            batch = [load(tr[i]) for i in order[b:b + 8]]
            e = torch.stack([q[0] for q in batch]); s = torch.stack([q[1] for q in batch]); y = torch.stack([q[2] for q in batch])
            loss = crit(net(e, s), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(batch)
        conf = evaluate(net, val) if val else None
        ious = iou_table(conf) if conf is not None else {}
        rec = {'epoch': ep + 1, 'loss': round(tot / len(tr), 4), 'val_iou': ious, 's': round(time.time() - t0)}
        log.append(rec); print(json.dumps(rec), flush=True)
    torch.save(net.state_dict(), os.path.join(M, 'seg_decoder.pt'))
    json.dump({'n_train': len(tr), 'n_val': len(val), 'weights': w.tolist(), 'log': log},
              open(os.path.join(D, 'seg_train_log.json'), 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 0.15)
