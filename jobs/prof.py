"""Step 1: limb-profile probe on TRUE part maps with real RTMW keypoints, plus label-cutoff budget and label audits.
usage: python prof.py <gen dir> <out dir> '<chk glob>' '<kp glob>'
Profile: each limb chain (proximal joint -> mid joint -> distal joint -> extension past it) sampled at N points on 3 parallel
lines; each sample classified as own / distal / stump / wound / tq / BG / OCC / body / other / edge using the rendered side.
Decisions use dev3 only; test4 appears only in label audits (generator properties, not model selection)."""
import sys, os, json, glob, numpy as np, cv2
sys.path.insert(0, sys.argv[1])
import parts as PT, eval_v3 as EV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold

OUT = sys.argv[2]; S = ['LUE', 'RUE', 'LLE', 'RLE']; C4 = EV.C4; N = 48
CL = PT.SEG3_CLASSES; ix = {c: i for i, c in enumerate(CL)}
ARM = {ix['upper_arm'], ix['forearm'], ix['hand']}; LEG = {ix['thigh'], ix['shank'], ix['foot']}
DIST = {'UE': ix['hand'], 'LE': ix['foot']}
BODY = {ix[c] for c in ('TORSO_F', 'TORSO_B', 'HEAD_F', 'HEAD_B')}
CHAIN = {'UE': {'L': (5, 7, 9), 'R': (6, 8, 10)}, 'LE': {'L': (11, 13, 15), 'R': (12, 14, 16)}}
TOES = {'L': (17, 18), 'R': (20, 21)}
CATS = ['own', 'distal', 'stump', 'wound', 'tq', 'BG', 'OCC', 'body', 'other', 'edge']
OWNISH = {'own', 'distal', 'stump', 'wound', 'tq'}

chk = {}
for f in sorted(glob.glob(sys.argv[3])):
    for l in open(f):
        r = json.loads(l); chk[r['scene']] = r
KP = {}
for f in glob.glob(sys.argv[4]):
    for l in open(f):
        r = json.loads(l)
        if r.get('kp'):
            KP[r['scene']] = r['kp']


def lb_params(h, w, W=1280, H=960):
    s = min(W / w, H / h); nh, nw = int(round(h * s)), int(round(w * s))
    return s, (W - nw) // 2, (H - nh) // 2


def profile(seg, side, wsite, xy, sc, site):
    """returns (feature dict, chosen chain letter) for one site; xy in image pixel coords (seg resolution)."""
    ltr, typ = site[0], site[1:]
    sd = 1 if ltr == 'L' else 2; parts = ARM if typ == 'UE' else LEG; H, W = seg.shape
    torso = np.linalg.norm((xy[5] + xy[6]) / 2 - (xy[11] + xy[12]) / 2) + 1e-6
    best = None
    for kl in ('L', 'R'):                         # try both keypoint chains: RTMW mirrors on back views
        a, b, c = CHAIN[typ][kl]
        if typ == 'UE':
            ext = xy[c] + 0.6 * (xy[c] - xy[b])
        else:
            t = [xy[k] for k in TOES[kl] if sc[k] > 0.3]
            ext = xy[c] + (1.2 * (np.mean(t, 0) - xy[c]) if t else 0.3 * (xy[c] - xy[b]))
        pts = np.array([xy[a], xy[b], xy[c], ext], float)
        seglen = np.linalg.norm(np.diff(pts, axis=0), axis=1); cum = np.r_[0, np.cumsum(seglen)]; L = cum[-1] + 1e-6
        u = np.linspace(0, L, N); P = np.zeros((N, 2)); Dn = np.zeros((N, 2))
        for i, t in enumerate(u):
            k = min(np.searchsorted(cum, t, side='right') - 1, 2); f = (t - cum[k]) / max(seglen[k], 1e-6)
            P[i] = pts[k] + f * (pts[k + 1] - pts[k]); d = (pts[k + 1] - pts[k]) / max(seglen[k], 1e-6); Dn[i] = [-d[1], d[0]]
        j_dist = int(np.argmin(np.abs(u - cum[2])))
        off = 0.06 * torso; cats = []
        for i in range(N):
            votes = []
            for o in (-off, 0, off):
                x, y = P[i] + o * Dn[i]; xi, yi = int(round(x)), int(round(y))
                if not (0 <= xi < W and 0 <= yi < H):
                    votes.append('edge'); continue
                cv, sv = seg[yi, xi], side[yi, xi]
                if cv == ix['wound']:
                    votes.append('wound' if wsite[yi, xi] == S.index(site) + 1 else 'other')
                elif cv == ix['TQ']:
                    votes.append('tq')
                elif cv == ix['stump']:
                    votes.append('stump' if sv == sd else 'other')
                elif cv in parts:
                    votes.append(('distal' if cv == DIST[typ] else 'own') if sv == sd else 'other')
                elif cv in ARM | LEG:
                    votes.append('other')
                elif cv in BODY:
                    votes.append('body')
                elif cv == ix['OCC']:
                    votes.append('OCC')
                else:
                    votes.append('BG')
            own_v = [v for v in votes if v in OWNISH]
            if own_v:     # any line on the limb counts as limb; prefer the most specific label
                for pref in ('stump', 'wound', 'distal', 'tq', 'own'):
                    if pref in own_v:
                        cats.append(pref); break
            else:
                cats.append(max(set(votes), key=votes.count))
        n_own = sum(c_ in OWNISH for c_ in cats)
        if best is None or n_own > best[0]:
            best = (n_own, kl, cats, j_dist, L / torso, [sc[a], sc[b], sc[c]])
    n_own, kl, cats, j_dist, Lnorm, kps = best
    ownidx = [i for i, c_ in enumerate(cats) if c_ in OWNISH]
    end = ownidx[-1] if ownidx else -1
    after = cats[end + 1:end + 5] if end >= 0 else cats[:4]
    after_cat = 'reached_end' if end == N - 1 else (max(set(after), key=after.count) if after else 'none')
    gaps = 0
    if ownidx:
        seq = cats[ownidx[0]:end + 1]
        for i in range(1, len(seq)):
            if seq[i] in OWNISH and seq[i - 1] not in OWNISH:
                gaps += 1
    wi = [i for i, c_ in enumerate(cats) if c_ == 'wound']; ti = [i for i, c_ in enumerate(cats) if c_ == 'tq']
    feat = dict(end_frac=(end + 1) / (j_dist + 1) if end >= 0 else 0.0, after=after_cat, distal_n=cats.count('distal'),
                stump_end=int(any(c_ == 'stump' for c_ in cats[max(0, end - 3):end + 1])) if end >= 0 else 0,
                own_frac=n_own / N, occ_frac=cats.count('OCC') / N, bg_frac=cats.count('BG') / N, body_frac=cats.count('body') / N,
                other_frac=cats.count('other') / N, edge_frac=cats.count('edge') / N, gaps=gaps,
                wound_n=len(wi), tq_n=len(ti), wound_near_tq=int(bool(wi and ti and min(abs(a - b) for a in wi for b in ti) <= 2)),
                wound_distal_tq=int(bool(wi and ti and max(wi) > max(ti) + 2)),
                Lnorm=Lnorm, kp_min=min(kps), kp_dist=kps[2], mirrored=int(kl != ltr))
    return feat, cats


ACATS = ['BG', 'OCC', 'body', 'other', 'edge', 'reached_end', 'none']


def fvec(p):
    return [p['end_frac'], min(p['end_frac'], 1.5), p['distal_n'] / N, p['stump_end'], p['own_frac'], p['occ_frac'], p['bg_frac'],
            p['body_frac'], p['other_frac'], p['edge_frac'], p['gaps'], p['wound_n'] / N, p['tq_n'] / N, p['wound_near_tq'],
            p['wound_distal_tq'], p['Lnorm'], p['kp_min'], p['kp_dist'], p['mirrored']] + [float(p['after'] == a) for a in ACATS]


rows = {'dev3': [], 'test4': []}; t0 = __import__('time').time()
for split in rows:
    D = f'/workspace/probeB/out/{split}'
    for sid, r in sorted(chk.items()):
        if r['split'] != split or sid not in KP:
            continue
        img = cv2.imread(os.path.join(D, sid + '.jpg')); h, w = img.shape[:2]
        seg, side, wsite = PT.decode3(os.path.join(D, sid + '_part3.png'))
        seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST); side = cv2.resize(side, (w, h), interpolation=cv2.INTER_NEAREST)
        wsite = cv2.resize(wsite, (w, h), interpolation=cv2.INTER_NEAREST)
        s, ox, oy = lb_params(h, w)
        k = KP[sid]; xy = (np.array(k['xy'], float) - [ox, oy]) / s; sc = np.array(k['score'], float)
        for site in S:
            p, cats = profile(seg, side, wsite, xy, sc, site)
            rows[split].append(dict(scene=sid, site=site, p=p, cats=''.join(CATS.index(c_).__str__() for c_ in cats),
                                    lab={t: r['labels'][t][site] for t in ('0.00', '0.10', '0.25')} if '0.00' in r['labels'] else None,
                                    lab10=r['labels']['0.10'][site], vf=r['visible_fraction'][site], pos=r.get('position'),
                                    gt=r['gt_sites_ceil'][site], pr=r['pred_sites_ceil'][site]))
print('PROF_ROWS', {k: len(v) for k, v in rows.items()}, round(__import__('time').time() - t0, 1), flush=True)
json.dump(rows, open(os.path.join(OUT, 'prof_rows.json'), 'w'))
out = {}

# A. separation on dev3 (true map)
dv = rows['dev3']
def xt(key_fn):
    d = {}
    for r in dv:
        k = key_fn(r); d.setdefault(r['lab10'], {}); d[r['lab10']][k] = d[r['lab10']].get(k, 0) + 1
    return d
out['dev_label_x_after_end'] = xt(lambda r: r['p']['after'])
out['dev_label_x_distal_present'] = xt(lambda r: 'distal' if r['p']['distal_n'] > 0 else 'no_distal')
out['dev_label_x_stump_at_end'] = xt(lambda r: 'stump_end' if r['p']['stump_end'] else 'no_stump_end')
out['dev_label_x_endfrac_bin'] = xt(lambda r: '<0.5' if r['p']['end_frac'] < 0.5 else '0.5-0.9' if r['p']['end_frac'] < 0.9 else '0.9-1.1' if r['p']['end_frac'] < 1.1 else '>=1.1')
out['dev_mirrored_chain_rate'] = round(float(np.mean([r['p']['mirrored'] for r in dv])), 3)
out['dev_label_x_wound_tq'] = xt(lambda r: ('W' if r['p']['wound_n'] else '-') + ('T' if r['p']['tq_n'] else '-') + ('n' if r['p']['wound_near_tq'] else '') + ('d' if r['p']['wound_distal_tq'] else ''))

# B. does the profile add information beyond the current features? dev3 grouped CV only
def cv(X, y, g, cw=None):
    best = None
    for C in [0.03, 0.1, 0.3, 1, 3]:
        pr = np.zeros(len(y), int)
        for i, j in GroupKFold(5).split(X, y, g):
            pr[j] = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight=cw)).fit(X[i], y[i]).predict(X[j])
        a = (pr == y).mean()
        if best is None or a > best[0]:
            best = (a, C, pr)
    a, C, pr = best
    return {'acc': round(float(a), 4), 'min_recall': round(min(float((pr[y == k] == k).mean()) for k in range(4) if (y == k).any()), 3), 'C': C,
            'recall': {C4[k]: round(float((pr[y == k] == k).mean()), 3) for k in range(4) if (y == k).any()}}
y = np.array([C4.index(r['lab10']) for r in dv]); g = np.array([r['scene'] for r in dv])
Bg = np.array([EV.feat(r['gt']) for r in dv]); Bp = np.array([EV.feat(r['pr']) for r in dv]); Pf = np.array([fvec(r['p']) for r in dv])
out['dev_cv_true_map'] = {'base': cv(Bg, y, g), 'base+profile': cv(np.c_[Bg, Pf], y, g), 'profile_only': cv(Pf, y, g)}
out['dev_cv_pred_map_base'] = cv(Bp, y, g)
out['dev_cv_pred_map_base+trueprofile_(upper_bound_of_probe)'] = cv(np.c_[Bp, Pf], y, g)

# C. label cutoff budget (dev3 CV, pred and true maps, base features)
cut = {}
if dv[0]['lab']:
    for t in ('0.00', '0.10', '0.25'):
        yt = np.array([C4.index(r['lab'][t]) for r in dv])
        cut[t] = {'counts': {c: int((yt == k).sum()) for k, c in enumerate(C4)}, 'pred_map': cv(Bp, yt, g), 'true_map': cv(Bg, yt, g),
                  'pred_map_bal': cv(Bp, yt, g, 'balanced')}
out['dev_label_cutoff'] = cut

# D. label audits on both splits (generator properties)
aud = {}
for split, R in rows.items():
    amp = [r for r in R if r['lab10'] == 'amputation']; z = [r for r in amp if r['gt']['stump_px'] == 0]
    aud[split] = {'amp_n': len(amp), 'amp_zero_visible_stump': len(z),
                  'zero_stump_vf_median': round(float(np.median([r['vf'] for r in z])), 3) if z else None,
                  'zero_stump_after_end': {a: sum(r['p']['after'] == a for r in z) for a in ACATS},
                  'zero_stump_distal_present': sum(r['p']['distal_n'] > 0 for r in z),
                  'zero_stump_label_at_0.00': ({c: sum(r['lab']['0.00'] == c for r in z) for c in C4} if z and z[0]['lab'] else None),
                  'wound_n': sum(r['lab10'] == 'wound' for r in R), 'wound_zero_visible_px': sum(r['lab10'] == 'wound' and r['gt']['wound_px'] == 0 for r in R),
                  'tq_limbs_by_label': {c: sum(r['gt']['tq_px'] > 0 and r['lab10'] == c for r in R) for c in C4},
                  'label_n': {c: sum(r['lab10'] == c for r in R) for c in C4}}
out['label_audit'] = aud
print('PROF_RES', json.dumps(out), flush=True)
