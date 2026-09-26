"""P1: end-to-end fine-tuning of the part segmenter (SAM 2.1-tiny image encoder + decoder), GPU-oriented.
Usage: python seg_train_e2e.py <dir[,dir...]> <steps> <out_name> [--batch 8] [--enc_lr 2e-5] [--dec_lr 1e-3] [--partv 3]
         [--init <decoder .pt>] [--val 0.1] [--workers 4]

Differences from the frozen-feature pilot (seg_train2.py):
  - the image encoder is trained (lower learning rate), so features adapt to lying bodies, clothing and wounds;
  - augmentation on the image itself: box jitter, horizontal flip (labels are side-agnostic, so a flip is exact),
    colour jitter, blur and JPEG-style noise;
  - mixed precision on CUDA; checkpoint every 10% of steps; validation mIoU per class.
Split by base scene (all views of one posed body stay together).
"""
import os, sys, glob, json, time, zlib, re, argparse
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
import parts as PT
from seg_features import square_crop

M = os.path.join(os.path.dirname(__file__), '..', 'models')
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class SceneSet(torch.utils.data.Dataset):
    def __init__(self, files, partv, train, size=1024):
        self.files, self.partv, self.train, self.size = files, partv, train, size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]; base = f[:-len('_sidecar.json')]
        img = cv2.cvtColor(cv2.imread(base + '.jpg'), cv2.COLOR_BGR2RGB)
        seg = (PT.decode3(base + '_part3.png')[0] if self.partv == '3' else PT.decode2(base + '_part2.png')[0])
        seg = cv2.resize(seg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        rng = np.random.default_rng(None if self.train else zlib.crc32(base.encode()))
        body = (seg != 0) & (seg != PT.SEG2_CLASSES.index('OCC'))
        ys, xs = np.nonzero(body)
        if len(xs) < 50:
            xs, ys = np.array([0, img.shape[1] - 1]), np.array([0, img.shape[0] - 1])
        w, h = xs.max() - xs.min(), ys.max() - ys.min()
        j = rng.uniform(-0.05, 0.05, 2) * max(w, h) if self.train else np.zeros(2)
        box = (xs.min() + j[0], ys.min() + j[1], xs.max() + j[0], ys.max() + j[1])
        pad = rng.uniform(0.10, 0.25) if self.train else 0.15
        crop, _ = square_crop(img, box, pad); lab, _ = square_crop(seg[..., None].repeat(3, 2), box, pad)
        crop = cv2.resize(crop, (self.size, self.size)); lab = cv2.resize(lab[..., 0], (self.size // 4, self.size // 4), interpolation=cv2.INTER_NEAREST)
        if self.train:
            if rng.random() < 0.5:
                crop, lab = crop[:, ::-1].copy(), lab[:, ::-1].copy()
            c = crop.astype(np.float32)
            c = c * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25)                       # brightness / contrast
            c = c * rng.uniform(0.9, 1.1, 3)[None, None]                                # colour cast
            if rng.random() < 0.3:
                c = cv2.GaussianBlur(c, (0, 0), rng.uniform(0.5, 1.5))
            if rng.random() < 0.3:
                c = c + rng.normal(0, rng.uniform(2, 8), c.shape)
            crop = np.clip(c, 0, 255).astype(np.uint8)
        x = (torch.from_numpy(crop).permute(2, 0, 1).float() / 255 - MEAN) / STD
        return x, torch.from_numpy(lab.astype(np.int64))


class SegModel(nn.Module):
    def __init__(self, nc, sam_ckpt):
        super().__init__()
        from sam2.build_sam import build_sam2
        sam = build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml', sam_ckpt, device='cpu')
        self.sam = sam
        from seg_train2 import cbr
        self.red = cbr(256, 128, 1)
        self.f1 = nn.Sequential(cbr(128 + 64, 128, 3), cbr(128, 64, 3))
        self.f0 = cbr(64 + 32, 32, 3)
        self.head = nn.Conv2d(32, nc, 1)

    def features(self, x):
        """same computation as SAM2ImagePredictor.set_image, batched and differentiable"""
        bo = self.sam.forward_image(x)
        _, feats, _, _ = self.sam._prepare_backbone_features(bo)
        if self.sam.directly_add_no_mem_embed:
            feats[-1] = feats[-1] + self.sam.no_mem_embed
        B = x.shape[0]
        H = x.shape[-1]; sizes = [(H // 4, H // 4), (H // 8, H // 8), (H // 16, H // 16)]
        maps = [f.permute(1, 2, 0).reshape(B, -1, *s) for f, s in zip(feats, sizes)]
        return maps[2], maps[1], maps[0]                   # embed 256@64, s1 64@128, s0 32@256

    def forward(self, x):
        e, s1, s0 = self.features(x)
        y = F.interpolate(self.red(e), scale_factor=2, mode='bilinear', align_corners=False)
        y = self.f1(torch.cat([y, s1], 1))
        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=False)
        return self.head(self.f0(torch.cat([y, s0], 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs'); ap.add_argument('steps', type=int); ap.add_argument('name')
    ap.add_argument('--batch', type=int, default=8); ap.add_argument('--enc_lr', type=float, default=2e-5)
    ap.add_argument('--dec_lr', type=float, default=1e-3); ap.add_argument('--partv', default='3')
    ap.add_argument('--init', default=''); ap.add_argument('--val', type=float, default=0.1)
    ap.add_argument('--workers', type=int, default=4); ap.add_argument('--size', type=int, default=1024)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    classes = PT.SEG3_CLASSES if a.partv == '3' else PT.SEG2_CLASSES; nc = len(classes)
    suffix = '_part3.png' if a.partv == '3' else '_part2.png'
    files = sorted(f for d in a.dirs.split(',') for f in glob.glob(os.path.join(d, 'C*_sidecar.json'))
                   if os.path.exists(f[:-len('_sidecar.json')] + suffix))
    base = lambda f: re.match(r'(C\d+)', os.path.basename(f)).group(1)
    val = [f for f in files if zlib.crc32(base(f).encode()) % 1000 < a.val * 1000]; tr = [f for f in files if f not in set(val)]
    print('train', len(tr), 'val', len(val), 'device', dev, flush=True)
    net = SegModel(nc, os.path.join(M, 'sam2_1_hiera_tiny.pt'))
    if a.init:
        sd = torch.load(a.init, map_location='cpu'); net.load_state_dict({k: v for k, v in sd.items() if not k.startswith('head')}, strict=False)
    net.to(dev)
    enc = [p for n, p in net.named_parameters() if n.startswith('sam.image_encoder')]
    dec = [p for n, p in net.named_parameters() if not n.startswith('sam.')]
    for n, p in net.named_parameters():
        if n.startswith('sam.') and not n.startswith('sam.image_encoder'):
            p.requires_grad_(False)
    opt = torch.optim.AdamW([{'params': enc, 'lr': a.enc_lr}, {'params': dec, 'lr': a.dec_lr}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [a.enc_lr, a.dec_lr], total_steps=a.steps)
    cnt = np.ones(nc)
    for f in tr[:200]:
        s_ = (PT.decode3 if a.partv == '3' else PT.decode2)(f[:-len('_sidecar.json')] + suffix)[0]
        cnt += np.bincount(s_.ravel(), minlength=nc)
    w = 1 / np.sqrt(cnt / cnt.sum()); w = torch.tensor(w / w.mean(), dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=w)
    dl = torch.utils.data.DataLoader(SceneSet(tr, a.partv, True, a.size), batch_size=a.batch, shuffle=True,
                                     num_workers=a.workers, drop_last=True, persistent_workers=a.workers > 0)
    scaler = torch.amp.GradScaler(enabled=dev == 'cuda')
    st, t0, log = 0, time.time(), []
    while st < a.steps:
        for x, y in dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=dev == 'cuda'):
                loss = crit(net(x).float(), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            st += 1
            if st % max(1, a.steps // 10) == 0 or st == a.steps:
                rec = {'step': st, 'loss': round(loss.item(), 4), 's': round(time.time() - t0)}
                if val:
                    net.eval(); conf = np.zeros((nc, nc), np.int64)
                    vs = SceneSet(val, a.partv, False, a.size)
                    with torch.no_grad():
                        for i in range(len(vs)):
                            xv, yv = vs[i]
                            with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=dev == 'cuda'):
                                p = net(xv[None].to(dev)).argmax(1)[0].cpu().numpy()
                            conf += np.bincount(yv.numpy().ravel() * nc + p.ravel(), minlength=nc * nc).reshape(nc, nc)
                    inter = np.diag(conf); union = conf.sum(0) + conf.sum(1) - inter
                    rec['val_iou'] = {c: round(float(inter[i] / union[i]), 3) for i, c in enumerate(classes) if union[i]}
                    net.train()
                log.append(rec); print(json.dumps(rec), flush=True)
                torch.save(net.state_dict(), os.path.join(M, a.name + '.pt'))
            if st >= a.steps:
                break
    json.dump(log, open(os.path.join(M, a.name + '_log.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
