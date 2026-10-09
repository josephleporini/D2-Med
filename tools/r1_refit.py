"""R1: decision-layer refit on T5 as a DEVELOPMENT diagnostic (Perception thread, WC-1; RealPhoto prompt step R1).

Question: do the segmenter's per-site features carry signal on real manikin photos, so that the synthetic decision layer
is the main fault (large gain), or does the segmenter fail on real texture (small gain)?

T5 is a development set (scored once on 30 Sep). Nothing here is a claim; every number is grouped cross-validation on
T5 itself. Per-photo rows and predictions stay on the volume; only aggregates are printed and written (G3, C-14).

Inputs per model: T5 engine rows (sites and sites_flip, from jobs/t5_score.sh) and the model's dev5 rows (the
synthetic source the shipped layer is fitted on, as in tools/t5_score.py).

Arms (adapt/kit.py): A synthetic layer as shipped; B real only (C in 0.03..1); C pooled synthetic + real (w 1, 3, 10);
D recalibration of A (16 parameters). Added here: P prior only (training-fold class mix, no features), the floor any
refit must beat, since a refit on a no_injury-heavy set can gain accuracy by predicting the majority class.
Every arm is mirror-averaged (original and mirrored rows), as the container and the T5 scoring run are.
Nested selection (inner grouped CV on the outer training part) by accuracy and by consequence cost.

Groups (no shoot field exists in the labels): union of (a) the two same-shoot pairs named in T5 Labels v1.1 §4,
(b) near-duplicate photos by 64-bit difference hash (Hamming <= 10), (c) DVIDS image ids within 100 of each other
(same upload batch; inferred to approximate a shoot). If (c) chains the set into fewer than 20 groups, (c) is dropped
and that is recorded. Singleton grouping is reported as a sensitivity check.

Decision rule, fixed before the run (arm B, as_labeled, grouped CV; constant no_injury baseline = 39.0%):
  SIGNAL      B accuracy >= baseline + 10 points AND one-vs-rest AUC >= 0.75 for at least two of
              {no_injury, amputation, not_testable}  -> the synthetic decision layer is the main fault; refit and
              real-feature work (R3, R5) is the path
  NO_SIGNAL   B accuracy <= baseline + 5 points AND every one-vs-rest AUC < 0.65 -> the segmenter fails on real
              texture; R2 inspection and segmenter adaptation come before any layer work
  PARTIAL     anything else -> R2 decides
The separability ceiling (arm B fitted and scored on all of T5, in-sample) is reported only as an upper bound.

usage: python tools/r1_refit.py <out.json> <labels.csv> <photo_dir> <model>=<t5_rows>:<dev5_rows>:<dev5_sidecar_dir> [...]
"""
import sys, os, json, csv, glob, re
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE] + [os.path.join(HERE, '..', p) for p in ('gen', 'jobs', 'score', 'adapt')]
import kit as K
from score import fitC, mk
import bt2_eval as BE
from t5_score import labels as t5_labels, truth, t5_rows

C4, S, MIRROR = K.C4, K.S, K.MIRROR
BASELINE = 'no_injury'


class Base:
    """bt1-decision layer from a fitted scaler+LR pipeline (same numbers engine_bt1.DecisionLayer evaluates)."""

    def __init__(self, m, C):
        sc, lr = m.named_steps['standardscaler'], m.named_steps['logisticregression']
        assert list(lr.classes_) == [0, 1, 2, 3]
        self.mean, self.scale, self.W, self.b = sc.mean_, sc.scale_, lr.coef_, lr.intercept_
        self.doc = dict(C=C)

    def proba(self, X):
        z = (np.asarray(X, float) - self.mean) / self.scale @ self.W.T + self.b
        z = z - z.max(1, keepdims=True); e = np.exp(z)
        return e / e.sum(1, keepdims=True)


class Prior:
    def fit(self, X, y, *a):
        self.p = np.bincount(y, minlength=4) / len(y); return self

    def proba(self, X):
        return np.tile(self.p, (len(X), 1))


def groups_for(L, photo_dir):
    ns = sorted(L)
    par = {n: n for n in ns}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x

    def union(a, b):
        if a in par and b in par:
            par[find(a)] = find(b)
    for a, b in ((14, 15), (34, 49)):
        union(a, b)
    files = {n: os.path.join(photo_dir, f'R{n:04d}.jpg') for n in ns}
    H = {n: K.dhash(f) for n, f in files.items() if os.path.exists(f)}
    hk = sorted(H)
    for i in range(len(hk)):
        for j in range(i + 1, len(hk)):
            if bin(H[hk[i]] ^ H[hk[j]]).count('1') <= 10:
                union(hk[i], hk[j])
    dv = {}
    for n in ns:
        m = re.search(r'dvidshub\.net/image/(\d+)', L[n]['page_url'])
        if m:
            dv[n] = int(m.group(1))
    before = dict(par)
    dk = sorted(dv, key=dv.get)
    for i in range(len(dk) - 1):
        if dv[dk[i + 1]] - dv[dk[i]] <= 100:
            union(dk[i], dk[i + 1])
    G = {n: find(n) for n in ns}
    dropped = False
    if len(set(G.values())) < 20:          # id-window chaining collapsed the set: keep rules (a) and (b) only
        par.clear(); par.update(before); G = {n: find(n) for n in ns}; dropped = True
    return G, len(H), dropped


def design(L, R, G, reading):
    X, XF, y, g, ids = [], [], [], [], []
    for n in sorted(L):
        if n not in R:
            continue
        a, b = R[n]
        for k, s in enumerate(S):
            t = truth(L[n], s, reading)
            if t is None:
                continue
            X.append(a[k]); XF.append(b[k]); y.append(C4.index(t)); g.append(G[n]); ids.append((n, s))
    return np.array(X, float), np.array(XF, float), np.array(y), np.array(g), ids


def fit_arm(name, make, base, X, y, Xs, ys):
    if name == 'A_synthetic':
        return base
    if name == 'P_prior':
        return Prior().fit(X, y)
    m = make()
    if name.startswith('C_'):
        return m.fit(X, y, Xs, ys)
    if len(set(y)) < 2:
        return base
    return m.fit(X, y)


def mp(m, X, XF):
    return 0.5 * (m.proba(X) + m.proba(XF))


def ovr_auc(y, P):
    out = {}
    for k in range(4):
        yy = (y == k).astype(int)
        if 0 < yy.sum() < len(yy):
            out[C4[k]] = round(float(roc_auc_score(yy, P[:, k])), 3)
    return out


def confusion(y, p):
    cm = np.zeros((4, 4), int)
    for a, b in zip(y, p):
        cm[a, b] += 1
    return cm.tolist()


def verdict(acc_b, base_acc, auc):
    strong = sum(auc.get(c, 0) >= 0.75 for c in ('no_injury', 'amputation', 'not_testable'))
    if acc_b >= base_acc + 0.10 and strong >= 2:
        return 'SIGNAL'
    if acc_b <= base_acc + 0.05 and all(v < 0.65 for v in auc.values()):
        return 'NO_SIGNAL'
    return 'PARTIAL'


def run_model(L, G, spec, reading):
    model, rest = spec.split('=', 1); t5p, d5p, d5d = rest.split(':')
    Xs, XFs, ys, gs = BE.load(d5p, d5d)
    C = fitC(Xs, ys, gs); base = Base(mk(C).fit(Xs, ys), C)
    R = t5_rows(t5p)
    X, XF, y, g, ids = design(L, R, G, reading)
    const = float((y == C4.index(BASELINE)).mean())
    cands = dict(K.arms(base, Xs, ys)); cands['P_prior'] = None
    folds = list(GroupKFold(min(5, len(set(g)))).split(X, y, g))
    P = {nm: np.zeros((len(y), 4)) for nm in cands}
    PS, PC, picks, picks_c = np.zeros((len(y), 4)), np.zeros((len(y), 4)), [], []
    for tr, te in folds:
        for nm, f in cands.items():
            P[nm][te] = mp(fit_arm(nm, f, base, X[tr], y[tr], Xs, ys), X[te], XF[te])
        ch, _ = K.select(base, X[tr], y[tr], g[tr], Xs, ys, XF[tr]); picks.append(ch); PS[te] = P[ch][te]
        cc, _ = K.select(base, X[tr], y[tr], g[tr], Xs, ys, XF[tr], by='cost'); picks_c.append(cc); PC[te] = P[cc][te]
    arms = {}
    for nm in P:
        p = P[nm].argmax(1)
        arms[nm] = dict(K.metrics(y, p, ids, groups=g), auc_ovr=ovr_auc(y, P[nm]), confusion_rows_truth=confusion(y, p))
    bnames = [n for n in P if n.startswith('B_')]
    bbest = max(bnames, key=lambda n: arms[n]['acc'])
    # B with the C chosen by inner CV is what the nested selection already covers; the rule uses the best fixed-C B
    # arm AND reports it, flagged as optimistic by one choice among four
    sel = dict(K.metrics(y, PS.argmax(1), ids, groups=g), picks=picks, confusion_rows_truth=confusion(y, PS.argmax(1)))
    selc = dict(K.metrics(y, PC.argmax(1), ids, groups=g), picks=picks_c, confusion_rows_truth=confusion(y, PC.argmax(1)))
    # separability ceiling: in-sample, upper bound only
    ceil = K.Layer('lr', 1.0).fit(X, y); pc = mp(ceil, X, XF)
    ceiling = dict(acc=round(float((pc.argmax(1) == y).mean()), 4), auc_ovr=ovr_auc(y, pc))
    # sensitivity: singleton groups (photo-level) for arm B best C
    fC = float(bbest.split('C')[-1]); Q = np.zeros((len(y), 4)); photo = np.array([i for i, _ in ids])
    for tr, te in GroupKFold(5).split(X, y, photo):
        Q[te] = mp(K.Layer('lr', fC).fit(X[tr], y[tr]), X[te], XF[te])
    sens = dict(acc=round(float((Q.argmax(1) == y).mean()), 4), auc_ovr=ovr_auc(y, Q))
    # parity: arm A over all of T5 must equal the 30 Sep T5 score for this model
    parity = round(float((mp(base, X, XF).argmax(1) == y).mean()), 4)
    v = verdict(arms[bbest]['acc'], const, arms[bbest]['auc_ovr'])
    return model, dict(C_base=C, sites=int(len(y)), photos=len({i for i, _ in ids}), groups=len(set(g)),
                       class_mix={C4[k]: int((y == k).sum()) for k in range(4)}, constant_no_injury_acc=round(const, 4),
                       arm_A_full_t5_acc_parity=parity, arms=arms, best_B=bbest, selected_by_acc=sel, selected_by_cost=selc,
                       ceiling_in_sample=ceiling, sensitivity_photo_groups=sens, verdict=v)


def main(out, lab, photo_dir, specs):
    L = t5_labels(lab)
    G, hashed, dropped = groups_for(L, photo_dir)
    res = dict(cost_version=K.COST_VERSION, photos_usable=len(L), photos_hashed=hashed, dvids_window_rule_dropped=dropped,
               groups=len(set(G.values())),
               group_sizes=sorted([list(G.values()).count(u) for u in set(G.values())], reverse=True)[:10], models={})
    print('R1_GROUPS', json.dumps({k: res[k] for k in ('photos_usable', 'photos_hashed', 'dvids_window_rule_dropped', 'groups', 'group_sizes')}), flush=True)
    for reading in ('as_labeled', 'threshold'):
        for sp in specs:
            m, r = run_model(L, G, sp, reading)
            res['models'].setdefault(m, {})[reading] = r
            print('R1_DETAIL ' + m + ' ' + reading + ' ' + json.dumps(r, separators=(',', ':')), flush=True)
            a = r['arms']
            print('R1', m, reading, json.dumps(dict(
                parity_A=r['arm_A_full_t5_acc_parity'], const=r['constant_no_injury_acc'], groups=r['groups'],
                A=a['A_synthetic']['acc'], P=a['P_prior']['acc'], bestB=[r['best_B'], a[r['best_B']]['acc']],
                D=max(a[n]['acc'] for n in a if n.startswith('D_')), Cpool=max(a[n]['acc'] for n in a if n.startswith('C_')),
                sel_acc=r['selected_by_acc']['acc'], sel_cost=r['selected_by_cost']['acc'], aucB=a[r['best_B']]['auc_ovr'],
                recB=a[r['best_B']]['recall'], ceiling=r['ceiling_in_sample']['acc'], photo_groups=r['sensitivity_photo_groups']['acc'],
                verdict=r['verdict'])), flush=True)
    json.dump(res, open(out, 'w'), indent=1)
    print('R1_DONE', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
