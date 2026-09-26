"""Train the v2 part decoder (13 classes, 256x256) on cached SAM features incl. mirrored copies (pose env, CPU).
Usage: python seg_train2.py <train_dir> <sample_budget> <out_name> [val_frac] [max_base_scenes]
  sample_budget  total training samples seen (fixed compute budget instead of epochs)
  split          by base scene (all views and mirrors of one posed body stay on the same side of the split)
  sampling       scenes containing stump pixels drawn twice as often
Writes models/<out_name>.pt and <train_dir>/<out_name>_log.json.
"""
import sys, os, glob, json, time, zlib, re
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
import parts as PT
from seg_features2 import load_f2

torch.set_num_threads(2)
PARTV = os.environ.get('PARTV', '2')
CLASSES = PT.SEG3_CLASSES if PARTV == '3' else PT.SEG2_CLASSES
NC = len(CLASSES)
SUF = '_f3' if PARTV == '3' else '_f2'
M = os.path.join(os.path.dirname(__file__), '..', 'models')
STUMP = CLASSES.index('stump')


def cbr(i, o, k):
    return nn.Sequential(nn.Conv2d(i, o, k, padding=k // 2), nn.BatchNorm2d(o), nn.ReLU())


class Decoder2(nn.Module):
    def __init__(self):
        super().__init__()
        self.red = cbr(256, 128, 1)
        self.f1 = nn.Sequential(cbr(128 + 64, 128, 3), cbr(128, 64, 3))
        self.f0 = cbr(64 + 32, 32, 3)
        self.head = nn.Conv2d(32, NC, 1)

    def forward(self, e, s1, s0):
        x = F.interpolate(self.red(e), scale_factor=2, mode='bilinear', align_corners=False)
        x = self.f1(torch.cat([x, s1], 1))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.head(self.f0(torch.cat([x, s0], 1)))


def base_id(path):
    return re.match(r'(C\d+)', os.path.basename(path)).group(1)


def is_val(path, frac):
    return (zlib.crc32(base_id(path).encode()) % 1000) < frac * 1000


def item(f):
    e, s1, s0, z = load_f2(f)
    return torch.from_numpy(e), torch.from_numpy(s1), torch.from_numpy(s0), torch.from_numpy(z['label'].astype(np.int64))


def confusion(net, files):
    net.eval(); conf = np.zeros((NC, NC), np.int64)
    with torch.no_grad():
        for f in files:
            e, s1, s0, y = item(f)
            p = net(e[None], s1[None], s0[None]).argmax(1)[0].numpy()
            conf += np.bincount(y.numpy().ravel() * NC + p.ravel(), minlength=NC * NC).reshape(NC, NC)
    net.train()
    return conf


def iou(conf):
    inter = np.diag(conf); union = conf.sum(0) + conf.sum(1) - inter
    return {c: (round(float(inter[i] / union[i]), 3) if union[i] else None) for i, c in enumerate(CLASSES)}


def main(D, budget, name, vf, max_base):
    files = sorted(f for d in D.split(',') for f in glob.glob(os.path.join(d, 'C*' + SUF + '.npz')) + glob.glob(os.path.join(d, 'C*' + SUF + 'm.npz')))
    bases = sorted({base_id(f) for f in files})
    if max_base:
        keep = set(bases[:max_base]); files = [f for f in files if base_id(f) in keep]
    val = [f for f in files if is_val(f, vf) and f.endswith(SUF + '.npz')]      # validate on originals only
    tr = [f for f in files if not is_val(f, vf)]
    print('bases', len({base_id(f) for f in files}), 'train items', len(tr), 'val items', len(val), flush=True)
    cnt = np.zeros(NC); has_stump = []
    for f in tr:
        lab = np.load(f)['label']; b = np.bincount(lab.ravel(), minlength=NC); cnt += b; has_stump.append(b[STUMP] > 0)
    w = 1 / np.sqrt(cnt / cnt.sum() + 1e-6); w = w / w.mean()
    pw = np.where(has_stump, 2.0, 1.0); pw = pw / pw.sum()
    print('class share', {c: round(float(x), 4) for c, x in zip(CLASSES, cnt / cnt.sum())}, flush=True)
    net = Decoder2(); B = 8; steps = budget // B
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
    rng = np.random.default_rng(0); log = []; t0 = time.time(); run = 0
    for st in range(steps):
        batch = [item(tr[i]) for i in rng.choice(len(tr), B, p=pw)]
        e, s1, s0, y = [torch.stack([b[k] for b in batch]) for k in range(4)]
        loss = crit(net(e, s1, s0), y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); run += loss.item()
        if (st + 1) % max(1, steps // 10) == 0 or st == steps - 1:
            rec = {'step': st + 1, 'samples': (st + 1) * B, 'loss': round(run / max(1, steps // 10), 4),
                   'val_iou': iou(confusion(net, val)) if val else {}, 's': round(time.time() - t0)}
            run = 0; log.append(rec); print(json.dumps(rec), flush=True)
            torch.save(net.state_dict(), os.path.join(M, name + '.pt'))
    json.dump({'n_train_items': len(tr), 'n_val': len(val), 'weights': w.tolist(), 'log': log},
              open(os.path.join(D.split(',')[0], name + '_log.json'), 'w'), indent=1)


if __name__ == '__main__':
    a = sys.argv
    main(a[1], int(a[2]), a[3], float(a[4]) if len(a) > 4 else 0.15, int(a[5]) if len(a) > 5 else 0)
