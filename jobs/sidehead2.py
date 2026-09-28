"""Learned left/right side for limb pixels: a side head on the part segmenter.
  train:   python sidehead.py train <dir[,dir]> <steps> <out.pt> [batch]
  extract: python sidehead.py extract <ckpt.pt> <scene_dir> <split> <shard k> <n shards> <out_prefix>
Training starts from seg3_e2e_b.pt, keeps the part loss, and adds a 2-class side loss (patient left / right) on
pixels that carry a rendered side. Horizontal flips swap the side labels.
Extraction writes feature rows (eval_v3 site_rows format) for four side assignments on the NEW model's part map:
  kp     current keypoint side rule (baseline on the same map)
  side   learned side overrides every limb pixel
  sidec  learned side only where its probability >= 0.75, keypoint rule elsewhere
  ceil   rendered side (ceiling), as in checks.py
plus per-image pixel side accuracy of kp / side / sidec against the rendered side.
"""
import os, sys, json, glob, time, zlib, re
N_THR = int(os.environ.get('CAP_THREADS', '4'))
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[k] = str(N_THR)
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
cv2.setNumThreads(N_THR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parts as PT
from seg_train_e2e import SceneSet, SegModel, MEAN, STD
from seg_features import square_crop

M = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')


FLIP = os.environ.get('FLIP') == '1'          # test-time mirror: extract on the horizontally flipped image
SWAP = {'LUE': 'RUE', 'RUE': 'LUE', 'LLE': 'RLE', 'RLE': 'LLE'}
MODE = os.environ.get('SIDE_MODE', 'joint')       # joint | distill | branch


class SideModel(SegModel):
    """joint/distill: side 1x1 conv on the shared decoder features.
    branch: frozen part path; a separate decoder copy (sred/sf1/sf0) feeds the side conv. One encoder pass either way."""
    def __init__(self, nc, sam_ckpt, mode=None):
        super().__init__(nc, sam_ckpt)
        self.mode = mode or MODE
        self.side = nn.Conv2d(32, 2, 1)
        if self.mode == 'branch':
            import copy
            self.sred, self.sf1, self.sf0 = copy.deepcopy(self.red), copy.deepcopy(self.f1), copy.deepcopy(self.f0)

    def _dec(self, red, f1, f0, e, s1, s0):
        y = F.interpolate(red(e), scale_factor=2, mode='bilinear', align_corners=False)
        y = f1(torch.cat([y, s1], 1))
        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=False)
        return f0(torch.cat([y, s0], 1))

    def forward(self, x, with_side=True):
        e, s1, s0 = self.features(x)
        f = self._dec(self.red, self.f1, self.f0, e, s1, s0)
        if not with_side:
            return self.head(f)
        fs = self._dec(self.sred, self.sf1, self.sf0, e, s1, s0) if self.mode == 'branch' else f
        return self.head(f), self.side(fs)


class SideSet(SceneSet):
    """same crop and augmentation as SceneSet; also returns side target (0 ignore, 1 left, 2 right)"""
    def __getitem__(self, i):
        f = self.files[i]; base = f[:-len('_sidecar.json')]
        img = cv2.cvtColor(cv2.imread(base + '.jpg'), cv2.COLOR_BGR2RGB)
        seg, side, _ = PT.decode3(base + '_part3.png')
        seg = cv2.resize(seg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        side = cv2.resize(side, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        rng = np.random.default_rng(None if self.train else zlib.crc32(base.encode()))
        body = (seg != 0) & (seg != PT.SEG2_CLASSES.index('OCC'))
        ys, xs = np.nonzero(body)
        if len(xs) < 50:
            xs, ys = np.array([0, img.shape[1] - 1]), np.array([0, img.shape[0] - 1])
        w, h = xs.max() - xs.min(), ys.max() - ys.min()
        j = rng.uniform(-0.05, 0.05, 2) * max(w, h) if self.train else np.zeros(2)
        box = (xs.min() + j[0], ys.min() + j[1], xs.max() + j[0], ys.max() + j[1])
        pad = rng.uniform(0.10, 0.25) if self.train else 0.15
        crop, _ = square_crop(img, box, pad)
        lab, _ = square_crop(np.stack([seg, side, seg], -1), box, pad)
        crop = cv2.resize(crop, (self.size, self.size))
        lab = cv2.resize(lab, (self.size // 4, self.size // 4), interpolation=cv2.INTER_NEAREST)
        sl, sd = lab[..., 0], lab[..., 1]
        if self.train:
            if rng.random() < 0.5:
                crop, sl, sd = crop[:, ::-1].copy(), sl[:, ::-1].copy(), sd[:, ::-1].copy()
                sd = np.where(sd == 1, 2, np.where(sd == 2, 1, 0)).astype(sd.dtype)
            c = crop.astype(np.float32)
            c = c * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25)
            c = c * rng.uniform(0.9, 1.1, 3)[None, None]
            if rng.random() < 0.3:
                c = cv2.GaussianBlur(c, (0, 0), rng.uniform(0.5, 1.5))
            if rng.random() < 0.3:
                c = c + rng.normal(0, rng.uniform(2, 8), c.shape)
            crop = np.clip(c, 0, 255).astype(np.uint8)
        x = (torch.from_numpy(crop).permute(2, 0, 1).float() / 255 - MEAN) / STD
        return x, torch.from_numpy(sl.astype(np.int64)), torch.from_numpy(sd.astype(np.int64))


def train(dirs, steps, out, batch=8):
    dev = 'cuda'
    sd_ = int(os.environ.get('SEED', '0')); torch.manual_seed(sd_); np.random.seed(sd_)
    classes = PT.SEG3_CLASSES; nc = len(classes)
    files = sorted(f for d in dirs.split(',') for f in glob.glob(os.path.join(d, os.environ.get('SCENE_PREFIX', 'C') + '*_sidecar.json'))
                   if os.path.exists(f[:-len('_sidecar.json')] + '_part3.png'))
    base = lambda f: re.match(r'(C\d+)', os.path.basename(f)).group(1)
    val = [f for f in files if zlib.crc32(base(f).encode()) % 1000 < 100]; vs = set(val)
    tr = [f for f in files if f not in vs]
    print('SIDE_TRAIN files', len(tr), 'val', len(val), flush=True)
    net = SideModel(nc, os.path.join(M, 'sam2_1_hiera_tiny.pt'))
    base_sd = torch.load(os.path.join(M, 'seg3_e2e_b.pt'), map_location='cpu')
    miss = net.load_state_dict(base_sd, strict=False)
    print('SIDE_INIT mode', net.mode, 'missing', len(miss.missing_keys), 'unexpected', len(miss.unexpected_keys), flush=True)
    teacher = None
    if net.mode == 'branch':
        for a, b in (('sred', 'red'), ('sf1', 'f1'), ('sf0', 'f0')):
            getattr(net, a).load_state_dict(getattr(net, b).state_dict())
        for n, p in net.named_parameters():
            p.requires_grad_(n.startswith(('sred.', 'sf1.', 'sf0.', 'side.')))
        groups = [{'params': [p for n, p in net.named_parameters() if n.startswith(('sred.', 'sf1.', 'sf0.'))], 'lr': 5e-4},
                  {'params': list(net.side.parameters()), 'lr': 2e-3}]
    else:
        enc = [p for n, p in net.named_parameters() if n.startswith('sam.image_encoder')]
        dec = [p for n, p in net.named_parameters() if not n.startswith('sam.') and not n.startswith('side.')]
        for n, p in net.named_parameters():
            if n.startswith('sam.') and not n.startswith('sam.image_encoder'):
                p.requires_grad_(False)
        groups = [{'params': enc, 'lr': 1e-5}, {'params': dec, 'lr': 2e-4}, {'params': list(net.side.parameters()), 'lr': 2e-3}]
        if net.mode == 'distill':
            teacher = SegModel(nc, os.path.join(M, 'sam2_1_hiera_tiny.pt')); teacher.load_state_dict(base_sd); teacher.to(dev).eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
    net.to(dev)
    lrs = [g['lr'] for g in groups]
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lrs, total_steps=steps, pct_start=0.1)
    cnt = np.ones(nc)
    for f in tr[:200]:
        cnt += np.bincount(PT.decode3(f[:-len('_sidecar.json')] + '_part3.png')[0].ravel(), minlength=nc)
    w = 1 / np.sqrt(cnt / cnt.sum()); w = torch.tensor(w / w.mean(), dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=w)
    dl = torch.utils.data.DataLoader(SideSet(tr, '3', True), batch_size=batch, shuffle=True, num_workers=int(os.environ.get('WORKERS', 10)),
                                     drop_last=True, persistent_workers=True, prefetch_factor=4)
    st, t0 = 0, time.time()
    net.train()
    if net.mode == 'branch':                     # part path and encoder stay exactly as seg3_e2e_b (BN stats frozen)
        for mname in ('sam', 'red', 'f1', 'f0', 'head'):
            getattr(net, mname).eval()
    while st < steps:
        for x, y, sd in dl:
            x, y, sd = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True), sd.to(dev, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                ps, pd = net(x)
                if teacher is not None:
                    with torch.no_grad():
                        pt = teacher(x)
            lp = crit(ps.float(), y) if net.mode != 'branch' else ps.float().sum() * 0
            if teacher is not None:
                lp = lp + F.kl_div(ps.float().log_softmax(1), pt.float().softmax(1), reduction='batchmean') / (ps.shape[2] * ps.shape[3])
            tgt = torch.where(sd > 0, sd - 1, torch.full_like(sd, -100))
            ls = F.cross_entropy(pd.float(), tgt, ignore_index=-100) if (sd > 0).any() else pd.sum() * 0
            loss = lp + ls
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); st += 1
            if st % 50 == 0:
                print(json.dumps({'step': st, 'part': round(lp.item(), 4), 'side': round(ls.item(), 4), 's': round(time.time() - t0)}), flush=True)
            if st % max(1, steps // 4) == 0 or st == steps:
                torch.save(net.state_dict(), out)
            if st >= steps:
                break
    # validation: side pixel accuracy and part mIoU on held-out scenes
    net.eval(); vset = SideSet(val[:150], '3', False); ok = n = 0; conf = np.zeros((nc, nc), np.int64)
    with torch.no_grad():
        for i in range(len(vset)):
            x, y, sd = vset[i]
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                ps, pd = net(x[None].to(dev))
            p = ps.argmax(1)[0].cpu().numpy(); q = pd.argmax(1)[0].cpu().numpy() + 1; sd = sd.numpy(); y = y.numpy()
            m = sd > 0; ok += int((q[m] == sd[m]).sum()); n += int(m.sum())
            conf += np.bincount(y.ravel() * nc + p.ravel(), minlength=nc * nc).reshape(nc, nc)
    inter = np.diag(conf); union = conf.sum(0) + conf.sum(1) - inter
    print('SIDE_VAL', json.dumps({'side_pixel_acc': round(ok / max(n, 1), 4), 'n_val': len(vset),
                                  'iou': {c: round(float(inter[i] / union[i]), 3) for i, c in enumerate(classes) if union[i]}}), flush=True)


def extract(ckpt, D, split, kk, nn_, prefix):
    import onnxruntime as ort
    _IS = ort.InferenceSession
    class _C(_IS):
        def __init__(self, *a, **kw):
            so = kw.get('sess_options') or ort.SessionOptions(); so.intra_op_num_threads = N_THR; so.inter_op_num_threads = 1
            kw['sess_options'] = so; kw['providers'] = ['CPUExecutionProvider']; super().__init__(*a, **kw)
    ort.InferenceSession = _C
    import eval_v3 as EV, test_a_masks as TA, test_e as TE, test_e2 as T2, lend as LE
    from d2pipe import letterbox, frame_facing
    from side_kp import geodesic_fast, to_classmap_kp, site_rows
    from checks import side_lookup, assign_gt_side, pix_acc
    TA.geodesic = geodesic_fast
    torch.set_num_threads(N_THR)
    m = EV.load_models(None, os.path.join(M, 'lend3d.pt'), os.path.join(M, 'lwound2.pt'))   # ORT forced to CPU above
    dev = m['dev']; print('EXT_DEV', dev, flush=True)
    net = SideModel(len(PT.SEG3_CLASSES), os.path.join(M, 'sam2_1_hiera_tiny.pt'))
    net.load_state_dict(torch.load(ckpt, map_location='cpu')); net.to(dev).eval()
    files = sorted(glob.glob(os.path.join(D, os.environ.get('SCENE_PREFIX', 'C') + '*_sidecar.json')))[kk::nn_]
    fo = {v: open(f'{prefix}_{v}.jsonl', 'w') for v in ('kp', 'side', 'sidec', 'ceil')}
    fp = open(f'{prefix}_pix.jsonl', 'w'); t0 = time.time()
    for n, f in enumerate(files):
        sid = os.path.basename(f)[:-len('_sidecar.json')]; sc = json.load(open(f))
        img = letterbox(cv2.cvtColor(cv2.imread(os.path.join(D, sid + '.jpg')), cv2.COLOR_BGR2RGB))
        _, gside, _ = PT.decode3(os.path.join(D, sid + '_part3.png'))
        gside = cv2.resize(gside, (640, 480), interpolation=cv2.INTER_NEAREST)
        if FLIP:   # mirrored person: positions flip and anatomical sides swap; site keys are swapped back on output
            img = np.ascontiguousarray(img[:, ::-1]); gside = np.ascontiguousarray(gside[:, ::-1])
            gside = np.where(gside == 1, 2, np.where(gside == 2, 1, gside)).astype(gside.dtype)
        near, dist = side_lookup(gside)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        boxes = m['det'](bgr)
        k, s = m['wb'].pose_model(bgr, bboxes=boxes)
        kxy = None; kpd = None
        if len(k):
            i = int(np.argmax(s[:, :17].mean(1))); kp = k[i]; kxy = kp[:17]
            kpd = dict(xy=[[round(float(a), 1) for a in p] for p in kp[:23]], score=[round(float(v), 3) for v in s[i][:23]])
            axis = TE.unit((kp[5] + kp[6]) / 2 - (kp[11] + kp[12]) / 2); pp = TE.perp(axis)
        else:
            axis = np.array([0.0, -1.0]); pp = TE.perp(axis)
        # part map and side probability, same crop geometry as eval_v3.pred_part_map
        box = tuple(float(v) for v in max(boxes, key=lambda q: (q[2] - q[0]) * (q[3] - q[1]))[:4]) if len(boxes) else (0, 0, 1280, 960)
        crop, (X1, Y1, S) = square_crop(img, box, 0.15)
        x = (torch.from_numpy(cv2.resize(crop, (1024, 1024))).permute(2, 0, 1).float() / 255 - MEAN) / STD
        with torch.no_grad():
            ps, pd = net(x[None].to(dev))
            s2 = max(int(round(S / 2)), 1)
            lab = F.interpolate(ps.float(), size=(s2, s2), mode='bilinear', align_corners=False)[0].argmax(0).cpu().numpy()
            pl = F.interpolate(pd.float(), size=(s2, s2), mode='bilinear', align_corners=False)[0].softmax(0)[0].cpu().numpy()
        seg3 = np.zeros((480, 640), np.uint8); pleft = np.full((480, 640), 0.5, np.float32)
        ox, oy = int(round(X1 / 2)), int(round(Y1 / 2))
        ys0, xs0 = max(0, oy), max(0, ox); ys1, xs1 = min(480, oy + s2), min(640, ox + s2)
        if ys1 > ys0 and xs1 > xs0:
            seg3[ys0:ys1, xs0:xs1] = lab[ys0 - oy:ys1 - oy, xs0 - ox:xs1 - ox]
            pleft[ys0:ys1, xs0:xs1] = pl[ys0 - oy:ys1 - oy, xs0 - ox:xs1 - ox]
        seg, wm, tq = EV.fold_extras(seg3)
        sfr = T2.seg_frame(seg)
        fac, fconf = frame_facing(seg, sfr)          # predicted facing, persisted for events (M3-13)
        left = np.array([1.0, 0.0]) if (sfr is None and not len(k)) else (sfr[2] if sfr else 1) * pp
        cm1, names, e1, s1, _ = to_classmap_kp(seg, left, axis, None if kxy is None else kxy.tolist())
        pside = np.where(pleft >= 0.5, 1, 2).astype(np.uint8); zero = np.zeros((480, 640), np.float32)
        cmS, eS, sS = assign_gt_side(cm1, names, seg, e1, s1, pside, zero)
        conf = np.where(np.maximum(pleft, 1 - pleft) >= 0.75, 0.0, 99.0).astype(np.float32)
        cmC, eC, sC = assign_gt_side(cm1, names, seg, e1, s1, pside, conf)
        cmG, eG, sG = assign_gt_side(cm1, names, seg, e1, s1, near, dist)
        # visible_fraction is generator truth copied from the sidecar (scorer context only); never an engine output
        base = dict(scene=sid, split=split, visible_fraction=sc.get('visible_fraction'), labels=sc['labels_by_threshold'], kp=kpd,
                    frame=dict(facing=fac or 'unknown', facing_conf=fconf))
        if os.environ.get('SAVE_MAPS'):
            kk_ = np.zeros((23, 3), np.float32)
            if kpd:
                kk_[:, :2] = np.array(kpd['xy'], np.float32); kk_[:, 2] = np.array(kpd['score'], np.float32)
            np.savez_compressed(os.path.join(os.environ['SAVE_MAPS'], sid + '.npz'), seg=seg3, pl=np.clip(pleft * 255, 0, 255).astype(np.uint8), kp=kk_)
        for v, (cm_, e_, s_) in (('kp', (cm1, e1, s1)), ('side', (cmS, eS, sS)), ('sidec', (cmC, eC, sC)), ('ceil', (cmG, eG, sG))):
            rows = site_rows(m, img, cm_, names, e_, s_, wm, tq)
            if FLIP:
                rows = {SWAP[k]: v_ for k, v_ in rows.items()}
            fo[v].write(json.dumps(dict(base, flip=bool(FLIP), sites=rows)) + '\n'); fo[v].flush()
        fp.write(json.dumps(dict(scene=sid, split=split, kp=pix_acc(cm1, names, near, dist), side=pix_acc(cmS, names, near, dist),
                                 sidec=pix_acc(cmC, names, near, dist))) + '\n'); fp.flush()
        if (n + 1) % 20 == 0:
            print('EXT', split, kk, n + 1, round((time.time() - t0) / (n + 1), 2), flush=True)
    print('EXT_DONE', split, kk, len(files), round(time.time() - t0, 1), flush=True)


def bench(ckpt, D, n=40):
    """forward-pass latency (fp32, batch 1, 1024 crop) of the original segmenter vs the side model, same GPU"""
    dev = 'cuda'; torch.backends.cudnn.benchmark = True
    orig = SegModel(len(PT.SEG3_CLASSES), os.path.join(M, 'sam2_1_hiera_tiny.pt'))
    orig.load_state_dict(torch.load(os.path.join(M, 'seg3_e2e_b.pt'), map_location='cpu')); orig.to(dev).eval()
    net = SideModel(len(PT.SEG3_CLASSES), os.path.join(M, 'sam2_1_hiera_tiny.pt'))
    net.load_state_dict(torch.load(ckpt, map_location='cpu')); net.to(dev).eval()
    xs = [torch.randn(1, 3, 1024, 1024, device=dev) for _ in range(n)]
    res = {}
    for name, fn in (('orig_parts_only', lambda x: orig(x)), ('side_model_one_pass', lambda x: net(x)),
                     ('two_pass_orig_plus_side', lambda x: (orig(x), net(x)))):
        with torch.no_grad():
            for x in xs[:5]:
                fn(x)
            torch.cuda.synchronize(); t = time.perf_counter()
            for x in xs:
                fn(x)
            torch.cuda.synchronize(); res[name] = round((time.perf_counter() - t) / n * 1000, 1)
    print('BENCH_MS', json.dumps(dict(res, gpu=torch.cuda.get_device_name(0), mode=net.mode)), flush=True)


def iou(split_dir, ckpts):
    """part IoU on a scene dir with the ground-truth body box crop (SideSet eval geometry), for several checkpoints.
    ckpts: comma list; 'orig' means seg3_e2e_b. Also reports IoU by true visible-fraction band of the limb."""
    dev = 'cuda'; classes = PT.SEG3_CLASSES; nc = len(classes)
    files = sorted(glob.glob(os.path.join(split_dir, os.environ.get('SCENE_PREFIX', 'C') + '*_sidecar.json')))
    vset = SideSet(files, '3', False)
    out = {}
    for ck in ckpts.split(','):
        if ck == 'orig':
            net = SegModel(nc, os.path.join(M, 'sam2_1_hiera_tiny.pt')); net.load_state_dict(torch.load(os.path.join(M, 'seg3_e2e_b.pt'), map_location='cpu'))
            fwd = lambda x: net(x)
        else:
            net = SideModel(nc, os.path.join(M, 'sam2_1_hiera_tiny.pt'), 'distill'); net.load_state_dict(torch.load(ck, map_location='cpu'))
            fwd = lambda x: net(x)[0]
        net.to(dev).eval(); conf = np.zeros((nc, nc), np.int64); per = []
        with torch.no_grad():
            for i in range(len(vset)):
                x, y, sd = vset[i]
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    ps = fwd(x[None].to(dev))
                p = ps.argmax(1)[0].cpu().numpy(); y = y.numpy()
                c = np.bincount(y.ravel() * nc + p.ravel(), minlength=nc * nc).reshape(nc, nc); conf += c
                body = y > 0; per.append(float((p[body] == y[body]).mean()) if body.any() else 1.0)
        inter = np.diag(conf); union = conf.sum(0) + conf.sum(1) - inter
        top = sorted(((int(conf[i, j]), classes[i], classes[j]) for i in range(nc) for j in range(nc) if i != j), reverse=True)[:12]
        out[os.path.basename(ck)] = {'miou_present': round(float(np.mean([inter[i] / union[i] for i in range(nc) if union[i] and conf[i].sum()])), 4),
                                     'iou': {c: round(float(inter[i] / union[i]), 3) for i, c in enumerate(classes) if union[i]},
                                     'gt_px': {c: int(conf[i].sum()) for i, c in enumerate(classes)},
                                     'body_pix_acc_mean': round(float(np.mean(per)), 4), 'worst_img_frac_below_0.8': round(float(np.mean(np.array(per) < 0.8)), 3),
                                     'top_confusions_true_pred': top}
        del net; torch.cuda.empty_cache()
    print('IOU_RES', json.dumps(out), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'iou':
        iou(sys.argv[2], sys.argv[3]); sys.exit(0)
    if sys.argv[1] == 'bench':
        bench(sys.argv[2], sys.argv[3]); sys.exit(0)
    if sys.argv[1] == 'train':
        train(sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 8)
    else:
        extract(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), sys.argv[7])
