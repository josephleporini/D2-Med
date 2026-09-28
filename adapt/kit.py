"""Real-data adaptation kit (Decision Memo 28 Sep, decision 4b). Ready before DARPA training data lands.

What can learn from real labels: DARPA supplies one class per limb, no part maps, so only the decision layer (and a
direct engine, BT-4) can be refit on real data. The segmenter, side and limb heads stay synthetic-trained. This kit
refits the decision layer on engine rows of labeled real images and exports a layer the container loads unchanged.

Pipeline
  1 labels   load_labels(): ICD predictions-format JSON, a CSV (image_id,site,class), the T5 workbook (two labelers,
             agreed sites only), or synthetic sidecars under a chosen label rule (rehearsal)
  2 groups   load_groups(): a CSV (image_id,group), else near-duplicate clusters from a 64-bit difference hash, so shots of
             one manikin set-up never sit on both sides of a split
  3 rows     engine rows per image: gen/engine_bt1 (original and mirrored); jobs/bt3_a40.sh logs them as rows_*.jsonl,
             tools/bt3_check.py-compatible; extract_rows() does it for any image folder on a GPU pod
  4 refit    arms A..D (below), each scored by grouped 5-fold cross-validation; nested selection picks one
  5 report   accuracy, per-class recall, macro F1, consequence cost per site (PROVISIONAL matrix), wrong-side amputations,
             group-bootstrap 95% intervals
  6 export   decision JSON in the bt1-decision/1.0 format (engine_bt1.DecisionLayer)

Arms
  A  synthetic layer as shipped (zero-shot)
  B  refit on real rows only (C by inner grouped CV)
  C  pooled: synthetic source rows plus real rows up-weighted by w in {1, 3, 10}
  D  recalibration: a strongly regularized multinomial layer on the synthetic layer's log-probabilities (16 parameters)
"""
import os, sys, json, glob, csv, hashlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in ('gen', 'jobs', 'score'):
    sys.path.insert(0, os.path.join(HERE, '..', _p))
C4 = ['no_injury', 'wound', 'amputation', 'not_testable']
S = ['LUE', 'RUE', 'LLE', 'RLE']
MIRROR = {'LUE': 'RUE', 'RUE': 'LUE', 'LLE': 'RLE', 'RLE': 'LLE'}
ICD_SITE = {('upper_extremity', 'left'): 'LUE', ('upper_extremity', 'right'): 'RUE',
            ('lower_extremity', 'left'): 'LLE', ('lower_extremity', 'right'): 'RLE'}
# PROVISIONAL consequence weights, truth rows x predicted columns, class order C4. NOT set by a clinical reviewer
# (Instrumentation Spec requires that); placeholder so the report exists. Version and freeze before first real scoring.
MIN_GAIN = float(os.environ.get('ADAPT_MIN_GAIN', '0'))   # inner-CV accuracy gain needed to leave arm A
COST_VERSION = 'provisional-0.1 (28 Sep 2026, not clinically reviewed)'
COST = np.array([[0, 3, 3, 2],        # truth no_injury: over-calls 3, deferral-like not_testable 2
                 [40, 0, 3, 10],      # truth wound: missed 40
                 [100, 40, 0, 40],    # truth amputation: missed 100, called wound or not testable 40
                 [1, 3, 3, 0]])       # truth not_testable: any call on an unseen limb


def stem(i):
    return os.path.splitext(os.path.basename(i))[0]


# ---------------------------------------------------------------- 1 labels
def load_labels(path, kind=None, rule='0.10'):
    """-> {image stem: {site: class index or None}} and meta {image stem: dict}. None = unsure / not agreed / unusable."""
    kind = kind or ('t5' if path.endswith('.xlsx') else 'csv' if path.endswith('.csv') else 'synthetic' if os.path.isdir(path) else 'icd')
    lab, meta = {}, {}
    if kind == 'icd':
        for p in json.load(open(path))['predictions']:
            lab[stem(p['image_id'])] = {ICD_SITE[(q['body_region'], q['laterality'])]: C4.index(q['injury_type']) for q in p['sites']}
    elif kind == 'csv':
        for r in csv.DictReader(open(path)):
            lab.setdefault(stem(r['image_id']), {})[r['site']] = C4.index(r['class']) if r['class'] in C4 else None
    elif kind == 'synthetic':
        import labels as LB
        for f in glob.glob(os.path.join(path, '**', '*_sidecar.json'), recursive=True):
            c = json.load(open(f))
            L = c['labels_by_threshold'][rule] if rule in c.get('labels_by_threshold', {}) else LB.label(c, rule)
            lab[c['scene_id']] = {s: C4.index(L[s]) for s in S}
    elif kind == 't5':
        lab, meta = load_t5(path)
    return lab, meta


def load_t5(path):
    """T5 workbook: a site counts only where both labelers gave the same class (not 'unsure') on a usable photo."""
    import openpyxl
    wb = openpyxl.load_workbook(path, data_only=True)

    def sheet(n):
        ws = wb[n]; hdr = [c.value for c in ws[1]]; out = {}
        for r in ws.iter_rows(min_row=2, values_only=True):
            if r[0] is None:
                continue
            d = dict(zip(hdr, r)); out[int(d['#'])] = d
        return out
    A, B = sheet('Labels_A'), sheet('Labels_B')
    lab, meta = {}, {}
    for k, a in A.items():
        b = B.get(k, {})
        iid = f"T5_{k:03d}"
        usable = a.get('Usable?') == 'yes' and b.get('Usable?') == 'yes'
        sites, disagree, swap_only = {}, 0, 0
        for s in S:
            ca, cb = a.get(f'{s} class'), b.get(f'{s} class')
            ok = usable and ca in C4 and ca == cb
            sites[s] = C4.index(ca) if ok else None
            disagree += usable and ca in C4 and cb in C4 and ca != cb
        for s in ('LUE', 'LLE'):     # swap-only: the two labelers' pair classes are each other's mirror
            m = MIRROR[s]
            if usable and a.get(f'{s} class') != b.get(f'{s} class') and a.get(f'{s} class') == b.get(f'{m} class') \
                    and a.get(f'{m} class') == b.get(f'{s} class'):
                swap_only += 1
        lab[iid] = sites
        meta[iid] = dict(url=a.get('Image page URL'), usable=usable, framing=a.get('Framing'), side_conf=(a.get('Side confidence'), b.get('Side confidence')),
                         occluder=a.get('Main occluder'), disagree_sites=int(disagree), swap_only_pairs=int(swap_only),
                         visible={s: a.get(f'{s} visible') for s in S})
    return lab, meta


# ---------------------------------------------------------------- 2 groups
def dhash(path, n=8):
    from PIL import Image
    with Image.open(path) as im:
        a = np.asarray(im.convert('L').resize((n + 1, n)), dtype=np.int16)
    return int(''.join('1' if v else '0' for v in (a[:, 1:] > a[:, :-1]).ravel()), 2)


def load_groups(ids, groups_csv=None, image_dir=None, max_bits=10):
    """-> {id: group}. CSV wins; else difference-hash clusters (Hamming distance <= max_bits, union-find); else singletons."""
    if groups_csv:
        g = {stem(r['image_id']): r['group'] for r in csv.DictReader(open(groups_csv))}
        return {i: g.get(i, i) for i in ids}
    if not image_dir:
        return {i: i for i in ids}
    files = {stem(f): f for f in glob.glob(os.path.join(image_dir, '*')) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
    H = {i: dhash(files[i]) for i in ids if i in files}
    par = {i: i for i in ids}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    keys = sorted(H)
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            if bin(H[keys[a]] ^ H[keys[b]]).count('1') <= max_bits:
                par[find(keys[a])] = find(keys[b])
    return {i: find(i) for i in ids}


# ---------------------------------------------------------------- 3 rows
def feats(site_row):
    import eval_v3 as EV
    return EV.feat(site_row) + EV.limb_feat(site_row)


def load_rows(pattern):
    """engine rows: bt3 container logs (image_id, sites, sites_flip) or BT-1 extraction rows (scene, sites)."""
    out = {}
    for f in sorted(glob.glob(pattern)):
        op = __import__('gzip').open if f.endswith('.gz') else open
        for l in op(f, 'rt'):
            r = json.loads(l)
            if not r.get('sites'):
                continue
            out[stem(r.get('image_id') or r['scene'])] = r
    return out


def extract_rows(image_dir, models, ckpt, decision, out_jsonl):
    """GPU pod: engine rows (original and mirrored) for every image in a folder; no labels needed."""
    import engine_bt1 as EB
    from PIL import Image, ImageOps
    E = EB.BT1Engine(models, ckpt, decision, tta=True)
    with open(out_jsonl, 'w') as fo:
        for f in sorted(glob.glob(os.path.join(image_dir, '*'))):
            if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            with Image.open(f) as im:
                rgb = np.asarray(ImageOps.exif_transpose(im).convert('RGB'), dtype=np.uint8)
            try:
                _, info = E.predict_image(rgb)
                fo.write(json.dumps(dict(image_id=os.path.basename(f), frame=info['frame'], sites=info['rows'], sites_flip=info['rows_flip'])) + '\n')
            except Exception as ex:
                fo.write(json.dumps(dict(image_id=os.path.basename(f), error=f'{type(ex).__name__}: {ex}')) + '\n')


def design(rows, labels, groups):
    """-> X, XF (mirrored rows or None), y, g, ids; labeled sites only."""
    X, XF, y, g, ids = [], [], [], [], []
    have_flip = all(r.get('sites_flip') for r in rows.values())
    for i in sorted(set(rows) & set(labels)):
        for s in S:
            c = labels[i].get(s)
            if c is None:
                continue
            X.append(feats(rows[i]['sites'][s])); y.append(c); g.append(groups.get(i, i)); ids.append((i, s))
            if have_flip:
                XF.append(feats(rows[i]['sites_flip'][s]))
    return np.array(X, float), (np.array(XF, float) if have_flip else None), np.array(y), np.array(g), ids


# ---------------------------------------------------------------- 4 arms
def clogits(base, X):
    """centered logits of a bt1-decision layer: log-probabilities minus their row mean (softmax-invariant, linear in X)"""
    u = (np.asarray(X, float) - base.mean) / base.scale @ base.W.T + base.b
    return u - u.mean(1, keepdims=True)


class Layer:
    """scaler + multinomial logistic regression, or recalibration on a base layer's log-probabilities"""

    def __init__(self, kind, C=1.0, base=None, w=1.0):
        self.kind, self.C, self.base, self.w = kind, C, base, w

    def fit(self, X, y, Xs=None, ys=None):
        if self.kind == 'recal':
            Z = clogits(self.base, X)
            self.m = LogisticRegression(C=self.C, max_iter=5000).fit(Z, y)
        else:
            if Xs is not None:
                XX, yy = np.vstack([Xs, X]), np.concatenate([ys, y]); sw = np.concatenate([np.ones(len(ys)), np.full(len(y), self.w)])
            else:
                XX, yy, sw = X, y, None
            self.m = make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=5000))
            self.m.fit(XX, yy, logisticregression__sample_weight=sw)
        return self

    def proba(self, X):
        Z = clogits(self.base, X) if self.kind == 'recal' else X
        P = np.zeros((len(X), 4)); P[:, self.m.classes_] = self.m.predict_proba(Z); return P

    def export(self, path, extra=None):
        assert self.kind != 'recal', 'export a recalibrated layer via fold_recal()'
        sc, lr = self.m.named_steps['standardscaler'], self.m.named_steps['logisticregression']
        assert list(lr.classes_) == [0, 1, 2, 3], 'every class must be present in the refit data'
        doc = dict(schema='bt1-decision/1.0', classes=C4, sites=S, C=self.C, mean=sc.mean_.tolist(), scale=sc.scale_.tolist(),
                   coef=lr.coef_.tolist(), intercept=lr.intercept_.tolist(), **(extra or {}))
        json.dump(doc, open(path, 'w'), indent=1)


def fold_recal(recal, path, extra=None):
    """Export arm D as a plain decision JSON. Arm D sees centered logits K u (K = I - J/4, u = W x' + b), so its logits are
    R K (W x' + b) + r: linear in the standardized features. W' = R K W, b' = R K b + r is exact."""
    B, m = recal.base, recal.m
    assert list(m.classes_) == [0, 1, 2, 3]
    K = np.eye(4) - np.full((4, 4), 0.25)
    R, r = m.coef_ @ K, m.intercept_
    doc = dict(schema='bt1-decision/1.0', classes=C4, sites=S, C=recal.C, mean=B.mean.tolist(), scale=B.scale.tolist(),
               coef=(R @ B.W).tolist(), intercept=(R @ B.b + r).tolist(), recalibrated=True, **(extra or {}))
    json.dump(doc, open(path, 'w'), indent=1)


def arms(base, Xs, ys):
    """candidate layers: name -> factory"""
    A = {'A_synthetic': lambda: None}
    for C in (0.03, 0.1, 0.3, 1.0):
        A[f'B_real_C{C}'] = (lambda C=C: Layer('lr', C))
    if Xs is not None:
        for w in (1.0, 3.0, 10.0):
            A[f'C_pooled_w{int(w)}'] = (lambda w=w: Layer('lr', base.doc.get('C', 0.03), w=w))
    for C in (0.01, 0.1):
        A[f'D_recal_C{C}'] = (lambda C=C: Layer('recal', C, base=base))
    return A


def cv_proba(name, make, base, X, y, g, Xs, ys, XF=None, k=5):
    P = np.zeros((len(y), 4))
    for tr, te in GroupKFold(min(k, len(set(g)))).split(X, y, g):
        if name == 'A_synthetic':
            m = base
        else:
            m = make()
            if name.startswith('C_'):
                m.fit(X[tr], y[tr], Xs, ys)
            else:
                if len(set(y[tr])) < 2:
                    m = base
                else:
                    m.fit(X[tr], y[tr])
        p = m.proba(X[te])
        if XF is not None:
            p = 0.5 * (p + m.proba(XF[te]))
        P[te] = p
    return P


def select(base, X, y, g, Xs, ys, XF=None, by='acc'):
    """nested: inner grouped CV on the training part picks the arm; the outer fold scores the pick.
    by='acc' maximizes accuracy (the qualification score); by='cost' minimizes consequence cost per site (COST)"""
    cands = arms(base, Xs, ys)
    P = {n: cv_proba(n, f, base, X, y, g, Xs, ys, XF).argmax(1) for n, f in cands.items()}
    if by == 'cost':
        inner = {n: -float(COST[y, p].mean()) for n, p in P.items()}
    else:
        inner = {n: float((p == y).mean()) for n, p in P.items()}
    best = max(inner, key=lambda n: (round(inner[n], 4), n == 'A_synthetic'))
    # guard: replace the shipped layer only for a clear inner-CV gain (small real sets overfit the selection itself)
    if best != 'A_synthetic' and inner[best] - inner['A_synthetic'] < MIN_GAIN:
        best = 'A_synthetic'
    return best, inner


# ---------------------------------------------------------------- 5 report
def metrics(y, p, ids=None, rng=0, B=500, groups=None):
    rec = {C4[k]: round(float((p[y == k] == k).mean()), 3) for k in range(4) if (y == k).any()}
    out = dict(n=int(len(y)), acc=round(float((p == y).mean()), 4), macro_f1=round(float(f1_score(y, p, average='macro', labels=list(set(y)))), 4),
               min_recall=min(rec.values()), recall=rec, cost_per_site=round(float(COST[y, p].mean()), 3), cost_version=COST_VERSION)
    if ids is not None:
        T = {(i, s): (yy, pp) for (i, s), yy, pp in zip(ids, y, p)}
        ws = 0
        for (i, s), (yy, pp) in T.items():
            m = T.get((i, MIRROR[s]))
            if m and pp == 2 and yy != 2 and m[0] == 2 and m[1] != 2:
                ws += 1
        out['wrong_side_amputations'] = ws
    if groups is not None and B:
        R = np.random.default_rng(rng); ug = np.array(sorted(set(groups))); idx = {u: np.nonzero(groups == u)[0] for u in ug}
        acc = []
        for _ in range(B):
            sel = np.concatenate([idx[u] for u in R.choice(ug, len(ug))]); acc.append((p[sel] == y[sel]).mean())
        out['acc_ci95'] = [round(float(np.percentile(acc, 2.5)), 4), round(float(np.percentile(acc, 97.5)), 4)]
    return out
