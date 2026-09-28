"""BT-1 (IPR T2 + T4): per-limb head with full-outline (amodal) supervision on the single-pass side model, retrained on
train5 with a mirror-consistency target (R3 / decision T1).

  train:  python bt1_limb.py train <dir[,dir]> <steps> <out.pt> [batch]
  smoke:  python bt1_limb.py smoke <dir>             (CPU-safe shape and flip checks, no weights needed)

Model (LimbModel = SideModel in distill mode + limb outputs, one forward pass):
  part logits (SEG3 classes), side logits (patient left / right)           as the adopted seg3_side_distill
  limb map, 8 channels at 1/4 crop resolution:
    0-3  amodal (full-outline) mask per anatomical site LUE, RUE, LLE, RLE   (sigmoid; amputated limbs end at the stump)
    4-7  terminal-point heatmap per site                                      (sigmoid; anatomical end of the limb)
  cause logits per site, 5 classes CAUSES, from decoder features pooled over the site's amodal mask and around its
  terminal point.
Truth comes from generator vNext (scene4) files: <id>_amodal.png (bits LUE 1, RUE 2, LLE 4, RLE 8), sidecar
terminal[site] {x, y, in_frame, cause} in 640x480 map coordinates.

Laterality is anatomical. A horizontal flip swaps left and right everywhere: side labels, limb channels, cause order.
Every training step runs the batch and its mirror together; the mirror half has flipped and swapped targets, and a
consistency loss ties side probability and limb maps of the two halves (after flipping back and swapping).
"""
import os, sys, json, glob, time, zlib, re
N_THR = int(os.environ.get('CAP_THREADS', '8'))
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(k, str(N_THR))
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'gen'))
import parts as PT
from seg_features import square_crop
from seg_train_e2e import SegModel, MEAN, STD
from sidehead2 import SideModel, M

SITES = ['LUE', 'RUE', 'LLE', 'RLE']
BITS = [1, 2, 4, 8]
SWAP_IDX = [1, 0, 3, 2]                                   # LUE<->RUE, LLE<->RLE
CAUSES = ['intact_visible', 'occluded', 'out_of_frame', 'amputated_visible', 'amputated_hidden']
LIMB_FEATURES = ['log_amodal_px', 'vis_over_amodal', 'term_peak'] + ['p_cause_' + c for c in CAUSES]
SIGMA = 2.0                                               # heatmap sigma in label pixels (1/4 of the 1024 crop)


def scene_key(f):
    """group key for the train/val split: the scene id without view suffixes (C00012, T0000, D0413 ...)"""
    return re.match(r'([A-Za-z]+\d+)', os.path.basename(f)).group(1)


def cause_index(t):
    c = t.get('cause')
    if c == 'out_of_frame':
        return CAUSES.index('out_of_frame')
    return CAUSES.index(c) if c in CAUSES else -100


def gaussian(h, w, x, y, s=SIGMA):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s * s)).astype(np.float32)


class LimbSet(torch.utils.data.Dataset):
    """Same crop and photometric augmentation as SideSet. No flip here: the training step adds the mirror half.
    Returns x, part labels, side (0 ignore, 1 left, 2 right), amodal (4,H,W), heat (4,H,W), heat_mask (4),
    cause (4, -100 = unknown), limb_ok (1 if amodal truth exists)."""
    def __init__(self, files, train, size=1024):
        self.files, self.train, self.size = files, train, size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]; base = f[:-len('_sidecar.json')]; sc = json.load(open(f))
        img = cv2.cvtColor(cv2.imread(base + '.jpg'), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        seg, side, _ = PT.decode3(base + '_part3.png'); mh, mw = seg.shape          # map space (480 x 640)
        am_ok = os.path.exists(base + '_amodal.png')
        am = cv2.imread(base + '_amodal.png', cv2.IMREAD_UNCHANGED) if am_ok else np.zeros((mh, mw), np.uint8)
        up = lambda a: cv2.resize(a, (W, H), interpolation=cv2.INTER_NEAREST)
        seg, side, am = up(seg), up(side), up(am)
        rng = np.random.default_rng(None if self.train else zlib.crc32(base.encode()))
        body = (seg != 0) & (seg != PT.SEG2_CLASSES.index('OCC'))
        ys, xs = np.nonzero(body)
        if len(xs) < 50:
            xs, ys = np.array([0, W - 1]), np.array([0, H - 1])
        w, h = xs.max() - xs.min(), ys.max() - ys.min()
        j = rng.uniform(-0.05, 0.05, 2) * max(w, h) if self.train else np.zeros(2)
        box = (xs.min() + j[0], ys.min() + j[1], xs.max() + j[0], ys.max() + j[1])
        pad = rng.uniform(0.10, 0.25) if self.train else 0.15
        crop, (X1, Y1, S) = square_crop(img, box, pad)
        lab, _ = square_crop(np.stack([seg, side, am], -1), box, pad)
        crop = cv2.resize(crop, (self.size, self.size)); L = self.size // 4
        lab = cv2.resize(lab, (L, L), interpolation=cv2.INTER_NEAREST)
        sl, sd, ab = lab[..., 0], lab[..., 1], lab[..., 2]
        amodal = np.stack([(ab & b) > 0 for b in BITS]).astype(np.float32)
        heat = np.zeros((4, L, L), np.float32); hm = np.zeros(4, np.float32); cause = np.full(4, -100, np.int64)
        sx, sy = W / mw, H / mh                                                     # map -> image pixels
        for k, s in enumerate(SITES):
            t = (sc.get('terminal') or {}).get(s)
            if not t:
                continue
            cause[k] = cause_index(t)
            if t.get('in_frame'):
                px = (t['x'] * sx - X1) / S * L; py = (t['y'] * sy - Y1) / S * L
                if 0 <= px < L and 0 <= py < L:
                    heat[k] = gaussian(L, L, px, py); hm[k] = 1
        if self.train:
            c = crop.astype(np.float32)
            c = c * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25)
            c = c * rng.uniform(0.9, 1.1, 3)[None, None]
            if rng.random() < 0.3:
                c = cv2.GaussianBlur(c, (0, 0), rng.uniform(0.5, 1.5))
            if rng.random() < 0.3:
                c = c + rng.normal(0, rng.uniform(2, 8), c.shape)
            crop = np.clip(c, 0, 255).astype(np.uint8)
        x = (torch.from_numpy(crop).permute(2, 0, 1).float() / 255 - MEAN) / STD
        return (x, torch.from_numpy(sl.astype(np.int64)), torch.from_numpy(sd.astype(np.int64)), torch.from_numpy(amodal),
                torch.from_numpy(heat), torch.from_numpy(hm), torch.from_numpy(cause), torch.tensor(float(am_ok)))


def mirror_targets(y, sd, amodal, heat, hm, cause):
    """targets for the horizontally flipped image: flip maps, swap anatomical left and right"""
    sd = sd.flip(-1); sd = torch.where(sd == 1, 2, torch.where(sd == 2, 1, sd))
    return (y.flip(-1), sd, amodal.flip(-1)[:, SWAP_IDX], heat.flip(-1)[:, SWAP_IDX], hm[:, SWAP_IDX], cause[:, SWAP_IDX])


class LimbModel(SideModel):
    def __init__(self, nc, sam_ckpt):
        super().__init__(nc, sam_ckpt, mode='distill')
        self.limb = nn.Conv2d(32, 8, 1)
        self.cause = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, len(CAUSES)))

    def forward(self, x, with_side=True):
        e, s1, s0 = self.features(x)
        f = self._dec(self.red, self.f1, self.f0, e, s1, s0)
        lm = self.limb(f)
        return self.head(f), self.side(f), lm, self.cause_logits(f, torch.sigmoid(lm[:, :4]), torch.sigmoid(lm[:, 4:]))

    def cause_logits(self, f, am, ht):
        """f (B,32,H,W); am, ht (B,4,H,W) weights -> (B,4,5). Mask-weighted means of the decoder features."""
        def pool(wt):
            wt = wt.float(); return torch.einsum('bchw,bkhw->bkc', f.float(), wt) / (wt.sum((2, 3))[..., None] + 1e-3)
        return self.cause(torch.cat([pool(am), pool(ht)], -1))


def limb_losses(lm, cl, amodal, heat, hm, cause, ok):
    """amodal BCE (images with amodal truth), heatmap focal-style BCE (sites with an in-frame terminal), cause CE"""
    okb = ok[:, None, None, None]
    la = (F.binary_cross_entropy_with_logits(lm[:, :4].float(), amodal, reduction='none') * okb).sum() / (okb.sum() * amodal[0].numel() + 1e-6)
    wh = (1 + 20 * heat) * hm[:, :, None, None]
    lh = (F.binary_cross_entropy_with_logits(lm[:, 4:].float(), heat, reduction='none') * wh).sum() / (wh.sum() + 1e-6)
    lc = F.cross_entropy(cl.float().reshape(-1, len(CAUSES)), cause.reshape(-1), ignore_index=-100) if (cause >= 0).any() else cl.sum() * 0
    return la, lh, lc


def consistency(pd, lm, n):
    """first n items are originals, last n their mirrors: flip the mirror half back, swap left/right, compare probs"""
    ps = pd.float().softmax(1)[:, 0]                                  # P(patient left)
    lp = torch.sigmoid(lm.float())
    ps_m = 1 - ps[n:].flip(-1)
    lp_m = lp[n:].flip(-1)[:, SWAP_IDX + [4 + i for i in SWAP_IDX]]
    return F.mse_loss(ps[:n], ps_m) + F.mse_loss(lp[:n], lp_m)


def train(dirs, steps, out, batch=6):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    TEST = os.environ.get('BT1_TEST') == '1'          # CPU loop test: random weights, no checkpoints needed
    sd_ = int(os.environ.get('SEED', '0')); torch.manual_seed(sd_); np.random.seed(sd_)
    W_LIMB = float(os.environ.get('W_LIMB', '1')); W_CONS = float(os.environ.get('W_CONS', '1'))
    nc = len(PT.SEG3_CLASSES)
    files = sorted(f for d in dirs.split(',') for f in glob.glob(os.path.join(d, '*_sidecar.json'))
                   if os.path.exists(f[:-len('_sidecar.json')] + '_part3.png'))
    val = [f for f in files if zlib.crc32(scene_key(f).encode()) % 1000 < 100]; vs = set(val)
    tr = [f for f in files if f not in vs]
    print('BT1_TRAIN files', len(tr), 'val', len(val), 'amodal', sum(os.path.exists(f[:-13] + '_amodal.png') for f in tr),
          'W_LIMB', W_LIMB, 'W_CONS', W_CONS, flush=True)
    sam = None if TEST else os.path.join(M, 'sam2_1_hiera_tiny.pt')
    net = LimbModel(nc, sam)
    if not TEST:
        init = os.environ.get('INIT', os.path.join(M, 'seg3_side_distill.pt'))
        miss = net.load_state_dict(torch.load(init, map_location='cpu'), strict=False)
        print('BT1_INIT', init, 'missing', miss.missing_keys, 'unexpected', len(miss.unexpected_keys), flush=True)
    teacher = SegModel(nc, sam)
    if not TEST:
        teacher.load_state_dict(torch.load(os.path.join(M, 'seg3_e2e_b.pt'), map_location='cpu'))
    teacher.to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    enc = [p for n, p in net.named_parameters() if n.startswith('sam.image_encoder')]
    dec = [p for n, p in net.named_parameters() if not n.startswith(('sam.', 'side.', 'limb.', 'cause.'))]
    new = [p for n, p in net.named_parameters() if n.startswith(('side.', 'limb.', 'cause.'))]
    for n, p in net.named_parameters():
        if n.startswith('sam.') and not n.startswith('sam.image_encoder'):
            p.requires_grad_(False)
    groups = [{'params': enc, 'lr': 1e-5}, {'params': dec, 'lr': 2e-4}, {'params': new, 'lr': 2e-3}]
    net.to(dev)
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g['lr'] for g in groups], total_steps=steps, pct_start=0.1)
    cnt = np.ones(nc)
    for f in tr[:200]:
        cnt += np.bincount(PT.decode3(f[:-len('_sidecar.json')] + '_part3.png')[0].ravel(), minlength=nc)
    w = 1 / np.sqrt(cnt / cnt.sum()); crit = nn.CrossEntropyLoss(weight=torch.tensor(w / w.mean(), dtype=torch.float32, device=dev))
    nw = int(os.environ.get('WORKERS', 10))
    dl = torch.utils.data.DataLoader(LimbSet(tr, True, int(os.environ.get('BT1_SIZE', 1024))), batch_size=batch, shuffle=True, num_workers=nw, drop_last=True,
                                     persistent_workers=nw > 0, prefetch_factor=4 if nw > 0 else None)
    st, t0 = 0, time.time(); net.train()
    while st < steps:
        for b in dl:
            x, y, sd, am, ht, hm, ca, ok = [t.to(dev, non_blocking=True) for t in b]
            my, msd, mam, mht, mhm, mca = mirror_targets(y, sd, am, ht, hm, ca)
            X = torch.cat([x, x.flip(-1)]); Y = torch.cat([y, my]); SD = torch.cat([sd, msd]); AM = torch.cat([am, mam])
            HT = torch.cat([ht, mht]); HM = torch.cat([hm, mhm]); CA = torch.cat([ca, mca]); OK = torch.cat([ok, ok])
            with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=dev == 'cuda'):
                ps, pd, lm, cl = net(X)
                with torch.no_grad():
                    pt = teacher(X)
            lp = crit(ps.float(), Y) + F.kl_div(ps.float().log_softmax(1), pt.float().softmax(1), reduction='batchmean') / (ps.shape[2] * ps.shape[3])
            tgt = torch.where(SD > 0, SD - 1, torch.full_like(SD, -100))
            ls = F.cross_entropy(pd.float(), tgt, ignore_index=-100) if (SD > 0).any() else pd.sum() * 0
            la, lh, lc = limb_losses(lm, cl, AM, HT, HM, CA, OK)
            lcons = consistency(pd, lm, x.shape[0])
            loss = lp + ls + W_LIMB * (la + lh + 0.5 * lc) + W_CONS * lcons
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); st += 1
            if st % 50 == 0 or TEST:
                print(json.dumps({'step': st, 'part': round(lp.item(), 4), 'side': round(ls.item(), 4), 'amodal': round(la.item(), 4),
                                  'heat': round(lh.item(), 4), 'cause': round(lc.item(), 4), 'cons': round(lcons.item(), 5),
                                  's': round(time.time() - t0)}), flush=True)
            if st % max(1, steps // 4) == 0 or st == steps:
                torch.save(net.state_dict(), out)
            if st >= steps:
                break
    validate(net, val[:2] if TEST else val, dev)


@torch.no_grad()
def validate(net, val, dev, n=150):
    """held-out scenes: side pixel accuracy, part mIoU, amodal IoU per site, terminal error (label px), cause accuracy,
    and the mirror consistency of side and amodal outputs (share of limb pixels whose probability moves > 0.05)"""
    net.eval(); vset = LimbSet(val[:n], False); nc = len(PT.SEG3_CLASSES)
    ok = n_ = 0; conf = np.zeros((nc, nc), np.int64); ai = np.zeros(4); au = np.zeros(4); terr = []; cok = cn = 0
    flip_side = flip_am = npx = npx_am = 0
    for i in range(len(vset)):
        x, y, sd, am, ht, hm, ca, okm = vset[i]
        with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=dev == 'cuda'):
            ps, pd, lm, cl = net(torch.stack([x, x.flip(-1)]).to(dev))
        p = ps[0].argmax(0).cpu().numpy(); q = pd[0].argmax(0).cpu().numpy() + 1; s_ = sd.numpy(); y_ = y.numpy()
        m = s_ > 0; ok += int((q[m] == s_[m]).sum()); n_ += int(m.sum())
        conf += np.bincount(y_.ravel() * nc + p.ravel(), minlength=nc * nc).reshape(nc, nc)
        lp = torch.sigmoid(lm.float()).cpu(); pl = pd.float().softmax(1)[:, 0].cpu()
        if okm > 0:
            pa = lp[0, :4].numpy() > 0.5; ta = am.numpy() > 0.5
            ai += (pa & ta).sum((1, 2)); au += (pa | ta).sum((1, 2))
        for k in range(4):
            if hm[k] > 0:
                hy, hx = np.unravel_index(int(lp[0, 4 + k].argmax()), lp.shape[-2:]); ty, tx = np.unravel_index(int(ht[k].argmax()), ht.shape[-2:])
                terr.append(float(np.hypot(hx - tx, hy - ty)))
            if ca[k] >= 0:
                cn += 1; cok += int(cl[0, k].argmax().item() == ca[k].item())
        limbpx = torch.from_numpy(m)
        d_side = (pl[0] - (1 - pl[1].flip(-1))).abs()
        flip_side += int((d_side[limbpx] > 0.05).sum()); npx += int(limbpx.sum())
        d_am = (lp[0, :4] - lp[1, :4].flip(-1)[SWAP_IDX]).abs(); sup = (lp[0, :4] > 0.1) | (lp[1, :4].flip(-1)[SWAP_IDX] > 0.1)
        flip_am += int((d_am[sup] > 0.05).sum()); npx_am += int(sup.sum())
    inter = np.diag(conf); union = conf.sum(0) + conf.sum(1) - inter
    res = {'n_val': len(vset), 'side_pixel_acc': round(ok / max(n_, 1), 4),
           'part_iou': {c: round(float(inter[i] / union[i]), 3) for i, c in enumerate(PT.SEG3_CLASSES) if union[i]},
           'amodal_iou': {s: round(float(ai[k] / max(au[k], 1)), 3) for k, s in enumerate(SITES)},
           'terminal_err_px_median': round(float(np.median(terr)), 2) if terr else None, 'terminal_n': len(terr),
           'cause_acc': round(cok / max(cn, 1), 3), 'cause_n': cn,
           'mirror_side_px_over_0.05': round(flip_side / max(npx, 1), 4), 'mirror_amodal_px_over_0.05': round(flip_am / max(npx_am, 1), 4)}
    print('BT1_VAL', json.dumps(res), flush=True)
    net.train()
    return res


def limb_site_features(lm, cl, crop_geom, vis_px, flip=False):
    """limb outputs of ONE image (8,L,L), cause logits (4,5), crop geometry (X1, Y1, S) in the 1280x960 letterboxed image
    -> {site: {amodal_px, term_peak, p_cause[5]}} with amodal pixels counted in 640x480 map space.
    With flip=True the caller already swaps site keys afterwards, so channels are used as they are."""
    X1, Y1, S = crop_geom; L = lm.shape[-1]
    lp = torch.sigmoid(lm.float()).cpu().numpy(); pc = torch.softmax(cl.float(), -1).cpu().numpy()
    s2 = max(int(round(S / 2)), 1); scale = (s2 / L) ** 2          # label px -> 640x480 map px (area)
    out = {}
    for k, s in enumerate(SITES):
        ap = float((lp[k] > 0.5).sum()) * scale
        out[s] = dict(amodal_px=int(round(ap)), term_peak=round(float(lp[4 + k].max()), 4), p_cause=[round(float(v), 4) for v in pc[k]],
                      vis_over_amodal=round(float(min(vis_px.get(s, 0) / ap, 2.0)) if ap > 0 else 0.0, 4))
    return out


def limb_feat(l):
    return [np.log1p(l['amodal_px']), l['vis_over_amodal'], l['term_peak'], *l['p_cause']]


def smoke(d):
    """CPU check without weights: dataset shapes, mirror target consistency, one forward/backward on a random-init net"""
    files = sorted(glob.glob(os.path.join(d, '*_sidecar.json')))[:3]
    ds = LimbSet(files, True)
    x, y, sd, am, ht, hm, ca, ok = ds[0]
    print('SMOKE sample', tuple(x.shape), tuple(y.shape), tuple(am.shape), tuple(ht.shape), hm.tolist(), ca.tolist(), float(ok))
    assert am.shape == (4, 256, 256) and ht.shape == (4, 256, 256)
    b = [t[None] for t in (y, sd, am, ht, hm, ca)]
    m1 = mirror_targets(*b); m2 = mirror_targets(*m1)
    assert all(torch.equal(u, v) for u, v in zip(b, m2)), 'mirror twice must be identity'
    # the side label of a limb pixel and the amodal channel of its site agree before and after mirroring
    ys_, xs_ = np.nonzero((sd.numpy() == 1) & (am.numpy()[[0, 2]].max(0) > 0))
    if len(xs_):
        my = m1[1][0].numpy(); mam = m1[2][0].numpy()
        yy, xx = ys_[0], 255 - xs_[0]
        assert my[yy, xx] == 2 and mam[[1, 3], yy, xx].max() > 0, 'left limb pixel must become right after mirroring'
    net = LimbModel(len(PT.SEG3_CLASSES), None)
    X = torch.stack([x, x.flip(-1)])
    ps, pd, lm, cl = net(X)
    print('SMOKE outputs', tuple(ps.shape), tuple(pd.shape), tuple(lm.shape), tuple(cl.shape))
    my, msd, mam, mht, mhm, mca = mirror_targets(*b)
    la, lh, lc = limb_losses(lm, cl, torch.cat([am[None], mam]), torch.cat([ht[None], mht]), torch.cat([hm[None], mhm]),
                             torch.cat([ca[None], mca]), torch.tensor([ok, ok]))
    lcons = consistency(pd, lm, 1)
    (la + lh + lc + lcons).backward()
    feats = limb_site_features(lm[0].detach(), cl[0].detach(), (100, 50, 900), {'LUE': 500, 'RUE': 0, 'LLE': 10, 'RLE': 2000})
    assert all(len(limb_feat(v)) == len(LIMB_FEATURES) for v in feats.values())
    print('SMOKE losses', round(la.item(), 4), round(lh.item(), 4), round(lc.item(), 4), round(lcons.item(), 6), 'features ok')
    print('SMOKE_OK')


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'train':
        train(sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 6)
    elif cmd == 'smoke':
        smoke(sys.argv[2])
