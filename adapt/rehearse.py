"""Rehearsal of the adaptation kit on synthetic dev5, dressed as real data (no GPU).

dev5 scenes are split in two by a hash of the scene id:
  source half  stands in for our synthetic data: the base layer is fitted there at our own rule (v3_compat, 10% visible)
  'real' half  stands in for DARPA's labeled images; its labels use a DIFFERENT rule, standing in for DARPA annotators
               drawing the boundary elsewhere (guide_primary); control run: the same rule as the source (no shift)
For n real images in {30, 60, 120, 240} (5 random draws each), grouped 5-fold outer CV scores every arm and the arm the
nested inner CV selects. Question answered: how many labeled real images before refitting beats shipping the synthetic
layer, and does the selection rule avoid hurting when there is no shift. Also checks export round trips and the
T5 workbook reader on a TEST-filled copy.

usage: python adapt/rehearse.py <dddata_root> <out.json> [--t5 workbook.xlsx]
"""
import os, sys, json, glob, hashlib, argparse, tempfile, shutil
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kit as K
from score import fitC, mk
import engine_bt1 as EB


def base_layer(X, y, g, path):
    C = fitC(X, y, g); m = mk(C).fit(X, y)
    sc, lr = m.named_steps['standardscaler'], m.named_steps['logisticregression']
    json.dump(dict(schema='bt1-decision/1.0', C=C, mean=sc.mean_.tolist(), scale=sc.scale_.tolist(), coef=lr.coef_.tolist(),
                   intercept=lr.intercept_.tolist()), open(path, 'w'))
    return EB.DecisionLayer(path)


def run(root, out, t5=None):
    rows = K.load_rows(os.path.join(root, 'results/bt1/ext/bt1_t*_sidec.jsonl'))
    d5 = os.path.join(root, 'dev5')
    half = {i: int(hashlib.sha256(i.encode()).hexdigest(), 16) % 2 for i in rows}
    src = sorted(i for i in rows if half[i] == 0); real = sorted(i for i in rows if half[i] == 1)
    tmp = tempfile.mkdtemp()
    lab_v3, _ = K.load_labels(d5, 'synthetic', '0.10')
    Xs, _, ys, gs, _ = K.design({i: rows[i] for i in src}, lab_v3, {})
    base = base_layer(Xs, ys, gs, os.path.join(tmp, 'base.json'))
    res = dict(source_scenes=len(src), real_scenes=len(real), cost_version=K.COST_VERSION, runs={})
    for rule in ('guide_primary', '0.10'):
        lab, _ = K.load_labels(d5, 'synthetic', rule)
        Xr, _, yr, gr, idr = K.design({i: rows[i] for i in real}, lab, {})
        scene_of = np.array([i for i, _ in idr])
        # reference: the base layer on ALL real sites, and a refit with all 240 real scenes (OOF) as the ceiling
        res['runs'][rule] = dict(base_on_all_real=K.metrics(yr, base.proba(Xr).argmax(1), idr, groups=gr),
                                 real_class_mix={K.C4[k]: round(float((yr == k).mean()), 3) for k in range(4)}, curve={})
        for n in (30, 60, 120, 240):
            per = []
            for seed in range(5 if n < 240 else 1):
                pick = set(np.random.default_rng(seed).choice(real, n, replace=False))
                sel = np.array([s in pick for s in scene_of])
                X, y, g = Xr[sel], yr[sel], gr[sel]; ids = [idr[k] for k in np.nonzero(sel)[0]]
                from sklearn.model_selection import GroupKFold
                cands = K.arms(base, Xs, ys)
                P = {nm: np.zeros((len(y), 4)) for nm in cands}; PS = np.zeros((len(y), 4)); PC = np.zeros((len(y), 4)); picks = []
                for tr, te in GroupKFold(5).split(X, y, g):
                    for nm, f in cands.items():
                        if nm == 'A_synthetic':
                            m = base
                        else:
                            m = f()
                            if nm.startswith('C_'):
                                m.fit(X[tr], y[tr], Xs, ys)
                            elif len(set(y[tr])) < 2:
                                m = base
                            else:
                                m.fit(X[tr], y[tr])
                        P[nm][te] = m.proba(X[te])
                    choice, _ = K.select(base, X[tr], y[tr], g[tr], Xs, ys)
                    picks.append(choice); PS[te] = P[choice][te]
                    cc, _ = K.select(base, X[tr], y[tr], g[tr], Xs, ys, by='cost'); PC[te] = P[cc][te]
                per.append(dict(seed=seed, sites=int(len(y)), picks=picks,
                                arms={nm: K.metrics(y, P[nm].argmax(1), ids, B=0) for nm in P},
                                selected=K.metrics(y, PS.argmax(1), ids, B=0), selected_cost=K.metrics(y, PC.argmax(1), ids, B=0)))
            def mean(key, arm=None, sel='selected'):
                return round(float(np.mean([(r['arms'][arm] if arm else r[sel])[key] for r in per])), 4)
            def rec(arm=None, sel='selected'):
                return {c: round(float(np.mean([(r['arms'][arm] if arm else r[sel])['recall'].get(c, np.nan) for r in per])), 3) for c in K.C4}
            arms_ = list(per[0]['arms'])
            res['runs'][rule]['curve'][n] = dict(
                draws=len(per), acc={a: mean('acc', a) for a in arms_}, selected_acc=mean('acc'),
                selected_min_recall=mean('min_recall'), selected_cost=mean('cost_per_site'),
                base_acc=mean('acc', 'A_synthetic'), base_cost=mean('cost_per_site', 'A_synthetic'),
                base_recall=rec('A_synthetic'), selected_recall=rec(),
                by_cost=dict(acc=mean('acc', sel='selected_cost'), cost=mean('cost_per_site', sel='selected_cost'), recall=rec(sel='selected_cost'),
                             wrong_side=mean('wrong_side_amputations', sel='selected_cost')),
                base_wrong_side=mean('wrong_side_amputations', 'A_synthetic'), selected_wrong_side=mean('wrong_side_amputations'),
                picks={p: sum(r['picks'].count(p) for r in per) for p in sorted({q for r in per for q in r['picks']})})
            c = res['runs'][rule]['curve'][n]
            print(rule, n, 'base', c['base_acc'], 'selected', c['selected_acc'], 'best arm', max(c['acc'], key=c['acc'].get), c['acc'][max(c['acc'], key=c['acc'].get)], c['picks'], flush=True)
    # export round trips (arm B and arm D fitted on all real sites, guide_primary)
    lab, _ = K.load_labels(d5, 'synthetic', 'guide_primary')
    Xr, _, yr, gr, _ = K.design({i: rows[i] for i in real}, lab, {})
    b = K.Layer('lr', 0.1).fit(Xr, yr); b.export(os.path.join(tmp, 'b.json'))
    d = K.Layer('recal', 0.1, base=base).fit(Xr, yr); K.fold_recal(d, os.path.join(tmp, 'd.json'))
    res['export_max_diff'] = dict(B=float(np.abs(EB.DecisionLayer(os.path.join(tmp, 'b.json')).proba(Xr) - b.proba(Xr)).max()),
                                  D=float(np.abs(EB.DecisionLayer(os.path.join(tmp, 'd.json')).proba(Xr) - d.proba(Xr)).max()))
    # grouping: near-duplicate clusters on the real-half images (independent renders: expect almost all singletons)
    imgs = tempfile.mkdtemp()
    for i in real[:120]:
        f = glob.glob(os.path.join(d5, '*', i + '.jpg'))[0]; os.symlink(f, os.path.join(imgs, i + '.jpg'))
    G = K.load_groups(real[:120], image_dir=imgs)
    res['dhash_groups'] = dict(images=120, groups=len(set(G.values())))
    if t5:
        lab, meta = K.load_labels(t5)
        res['t5_reader'] = dict(photos=len(lab), usable=sum(m['usable'] for m in meta.values()),
                                agreed_sites=sum(v is not None for L in lab.values() for v in L.values()),
                                disagree_sites=sum(m['disagree_sites'] for m in meta.values()),
                                swap_only_pairs=sum(m['swap_only_pairs'] for m in meta.values()))
    shutil.rmtree(tmp, ignore_errors=True); shutil.rmtree(imgs, ignore_errors=True)
    json.dump(res, open(out, 'w'), indent=1)
    print('REHEARSAL', json.dumps({k: v for k, v in res.items() if k != 'runs'}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('root'); ap.add_argument('out'); ap.add_argument('--t5')
    a = ap.parse_args(); run(a.root, a.out, a.t5)
