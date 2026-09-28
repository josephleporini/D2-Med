"""Probe B single scorer with failure ledger.

One entry point for every phase: fixed metrics on locked splits, plus a per-site ledger that records the
delta from rendered truth at each pipeline stage and assigns a cause code by counterfactual substitution.
The ledger is for offline retrospective, never for model selection on a locked test split.

usage:
  python score/score.py --phase NAME --gen GEN_DIR --pred GLOB --sidecars DIR[,DIR] --out OUT
        [--side-truth GLOB] [--truth GLOB] [--rule 0.10] [--alt-rules 0.00,0.25,0.50]
        [--dev dev3] [--eval test4] [--prev LEDGER[,LEDGER]] [--manifest FILE]
        [--maps DIR] [--images DIR] [--sheet 20]

inputs
  --pred        extraction records of the arm under test (sidehead2 extract *_{variant}.jsonl or equivalent):
                {scene, split, labels{rule:{site:class}}, visible_fraction, kp{xy,score}, sites{site: raw}}
  --side-truth  same model and map, rendered left/right side (sidehead2 *_ceil.jsonl)         -> stage K
  --truth       checks.py records with gt_sites_ceil (true part map, rendered side)          -> stage S
  --sidecars    scene directories holding C*_sidecar.json (params, visible px) and optional *_gtkp.json
  --prev        earlier ledgers (jsonl) for the change table and persistent-failure list
  --manifest    sha256 manifest of the split; scoring aborts if any sidecar in scope differs
  --maps/--images  predicted-map npz (SAVE_MAPS) and scene images for the contact sheet

outputs in OUT: metrics.json, ledger.jsonl, retro.md, sheet_gross.jpg, sheet_lucky.jpg (when maps and images exist)

cause codes (primary = first that applies, all flags kept):
  L  label-definition boundary: prediction equals the label under an alternate rule, or amputation with no visible stump
  K  side assignment: fixed when the rendered side is substituted (same map, same decision layer)
  S  segmenter: fixed when the true part map is substituted (fixed layer, or a layer fit on true maps);
     subtype miss / hall(ucination) / extent / other from the pixel deltas
  F  decision layer: wrong even with true inputs
  G  generator truth suspect: set only by human review (review column)
severity: gross (wrong and confident >= 0.8, or truth well inside detectable range), near (other wrong, or right with
p_true < 0.6), lucky (right but the key pixels are missing on the predicted map), ok.
"""
import sys, os, json, glob, argparse, hashlib, datetime
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

SCHEMA = 'ledger/1.0'
S = ['LUE', 'RUE', 'LLE', 'RLE']
C4 = ['no_injury', 'wound', 'amputation', 'not_testable']
KEY = {'wound': 'wound_px', 'amputation': 'stump_px'}       # class-defining pixels on a map
GROSS_SIZE = {'amputation': 300, 'wound': 200}              # Phase 0: recall >= ~0.85-0.90 above these sizes
GROUPS = {'visibility': [0, 1, 2, 13], 'end_state': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14], 'wound': [15, 16, 17], 'tourniquet': [18]}
RTMW = {'LUE': (5, 7, 9), 'RUE': (6, 8, 10), 'LLE': (11, 13, 15), 'RLE': (12, 14, 16)}
GTJ = {'UE': ('shoulder', 'elbow', 'wrist'), 'LE': ('hip', 'knee', 'ankle')}
RAW_KEYS = ['vis_px', 'ext_px', 'stump_px', 'wound_px', 'tq_px']


def load_jsonl(pattern):
    out = {}
    for f in sorted(glob.glob(pattern)):
        for line in open(f):
            r = json.loads(line); out[r['scene']] = r
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def feat_names(EV):
    return ['log_vis_px', 'Lfrac', 'bg_frac', 'p_end_intact', 'p_end_amp', 'p_end_other', 'log_ext_px', 'log_stump_px',
            *['ring_' + q for q in EV.RING], 'vis_per_torso2', 'p_end_present', 'p_wound', 'log_wound_px', 'wound_frac', 'log_tq_px']


def fitC(X, y, g):
    best = None
    for C in [0.03, 0.1, 0.3, 1, 3]:
        a = np.mean([(mk(C).fit(X[i], y[i]).predict(X[j]) == y[j]).mean() for i, j in GroupKFold(5).split(X, y, g)])
        if best is None or a > best[1]:
            best = (C, a)
    return best[0]


def mk(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def proba(m, X):
    P = np.zeros((len(X), 4)); P[:, m.classes_] = m.predict_proba(X); return P


def metrics(y, p):
    rec = {C4[k]: round(float((p[y == k] == k).mean()), 3) for k in range(4) if (y == k).any()}
    return {'n': int(len(y)), 'acc': round(float((p == y).mean()), 4), 'macro_f1': round(float(f1_score(y, p, average='macro')), 4),
            'min_recall': min(rec.values()), 'recall': rec,
            'false_wound': round(float((p[(y != 1) & (y != 3)] == 1).mean()), 4),
            'false_amp': round(float((p[(y != 2) & (y != 3)] == 2).mean()), 4)}


def kp_error(kp, gtkp, site):
    """RTMW vs rendered joints for one site: median error / torso length, and whether the opposite-side joints fit better"""
    if not kp or not gtkp:
        return None
    xy = np.array(kp['xy'], float); sc = np.array(kp['score'], float); js = gtkp['sites']
    def gt(s, j):
        q = js.get(s, {}).get(j); return None if q is None else (np.array([q['x'], q['y']]), q['visible'])
    try:
        torso = np.linalg.norm((gt('LUE', 'shoulder')[0] + gt('RUE', 'shoulder')[0]) / 2 - (gt('LLE', 'hip')[0] + gt('RLE', 'hip')[0]) / 2)
    except TypeError:
        return None
    opp = ('R' if site[0] == 'L' else 'L') + site[1:]
    e_own, e_opp = [], []
    for idx, jn in zip(RTMW[site], GTJ[site[1:]]):
        a, b = gt(site, jn), gt(opp, jn)
        if a is None or not a[1] or sc[idx] < 0.3:
            continue
        e_own.append(np.linalg.norm(xy[idx] - a[0])); e_opp.append(np.linalg.norm(xy[idx] - b[0]) if b else np.inf)
    if not e_own:
        return None
    return {'err_torso': round(float(np.median(e_own) / max(torso, 1)), 3), 'n_joints': len(e_own),
            'side_swap': bool(np.median(e_opp) < np.median(e_own))}


def repo_commit():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    f = os.path.join(root, 'COMMIT')
    if os.path.exists(f):
        return open(f).read().strip()
    try:
        import subprocess
        return subprocess.check_output(['git', '-C', root, 'rev-parse', '--short', 'HEAD'], text=True).strip()
    except Exception:
        return 'unknown'


def main():
    ap = argparse.ArgumentParser()
    for a in ('--phase', '--gen', '--pred', '--sidecars', '--out'):
        ap.add_argument(a, required=True)
    ap.add_argument('--side-truth'); ap.add_argument('--truth'); ap.add_argument('--rule', default='0.10')
    ap.add_argument('--alt-rules', default='0.00,0.25,0.50'); ap.add_argument('--dev', default='dev3'); ap.add_argument('--eval')
    ap.add_argument('--prev', default=''); ap.add_argument('--manifest'); ap.add_argument('--maps'); ap.add_argument('--images')
    ap.add_argument('--sheet', type=int, default=20)
    A = ap.parse_args()
    sys.path.insert(0, A.gen)
    import eval_v3 as EV
    FN = feat_names(EV); assert len(FN) == 19
    os.makedirs(A.out, exist_ok=True)

    pred = load_jsonl(A.pred); side_t = load_jsonl(A.side_truth) if A.side_truth else {}; truth = load_jsonl(A.truth) if A.truth else {}
    cars, gtkps, car_path = {}, {}, {}
    for d in A.sidecars.split(','):
        for f in glob.glob(os.path.join(d, os.environ.get('SCENE_PREFIX', 'C') + '*_sidecar.json')):
            s = json.load(open(f)); cars[s['scene_id']] = s; car_path[s['scene_id']] = f
            g = f.replace('_sidecar.json', '_gtkp.json')
            if os.path.exists(g):
                gtkps[s['scene_id']] = json.load(open(g))

    # manifest lock
    man = None
    if A.manifest:
        lock = dict(reversed(l.split(None, 1)) for l in open(A.manifest).read().strip().splitlines())
        lock = {os.path.basename(k.strip()): v for k, v in lock.items()}
        bad = [sid for sid in pred if os.path.basename(car_path.get(sid, '')) in lock and sha256(car_path[sid]) != lock[os.path.basename(car_path[sid])]]
        if bad:
            sys.exit(f'MANIFEST MISMATCH {len(bad)} scenes, e.g. {bad[:3]}')
        man = {'file': A.manifest, 'sha256': sha256(A.manifest), 'entries': len(lock)}

    # site table
    rows = []
    for sid, r in sorted(pred.items()):
        if r['split'] not in (A.dev, A.eval):
            continue
        for s in S:
            rows.append(dict(scene=sid, site=s, split=r['split'], raw=r['sites'][s], labels={k: v[s] for k, v in r['labels'].items()},
                             kp=r.get('kp'), raw_side=side_t.get(sid, {}).get('sites', {}).get(s),
                             raw_true=truth.get(sid, {}).get('gt_sites_ceil', {}).get(s), facing=truth.get(sid, {}).get('facing_true')))
    # guard: a split filter that matches nothing must stop the run, not score an empty or partial table (IPR §8.2)
    splits = sorted({r['split'] for r in pred.values()})
    if not rows:
        sys.exit(f'ERROR: zero rows in scope: --dev {A.dev} --eval {A.eval}; splits in --pred: {splits}')
    if not any(r['split'] == A.dev for r in rows):
        sys.exit(f'ERROR: zero rows for --dev {A.dev}; splits in --pred: {splits}')
    if A.eval and not any(r['split'] == A.eval for r in rows):
        sys.exit(f'ERROR: zero rows for --eval {A.eval}; splits in --pred: {splits}')
    def X(key, idx):
        return np.array([EV.feat(rows[i][key]) for i in idx], float)
    y = np.array([C4.index(r['labels'][A.rule]) for r in rows]); g = np.array([r['scene'] for r in rows])
    dev = np.array([r['split'] == A.dev for r in rows]); ev = ~dev
    has_side = np.array([r['raw_side'] is not None for r in rows]); has_true = np.array([r['raw_true'] is not None for r in rows])
    di = np.where(dev)[0]; Xp = X('raw', range(len(rows)))
    C = fitC(Xp[di], y[di], g[di])
    P = {k: np.full((len(rows), 4), np.nan) for k in ('pred', 'side', 'true_fixed', 'true_refit')}
    GS = {k: np.full(len(rows), -1) for k in GROUPS}
    def run(train, test):
        m = mk(C).fit(Xp[train], y[train]); P['pred'][test] = proba(m, Xp[test])
        ts = test[has_side[test]]
        if len(ts):
            P['side'][ts] = proba(m, X('raw_side', ts))
        tt = test[has_true[test]]; tr = train[has_true[train]]
        if len(tt):
            Xt = X('raw_true', tt); P['true_fixed'][tt] = proba(m, Xt)
            if len(tr) and len(set(y[tr])) == 4:
                P['true_refit'][tt] = proba(mk(C).fit(X('raw_true', tr), y[tr]), Xt)
            for name, gi in GROUPS.items():
                Xs = Xp[tt].copy(); Xs[:, gi] = Xt[:, gi]; GS[name][tt] = proba(m, Xs).argmax(1)
    for i, j in GroupKFold(5).split(di, y[di], g[di]):
        run(di[i], di[j])
    if ev.any():
        run(di, np.where(ev)[0])
    pp = P['pred'].argmax(1); conf = P['pred'].max(1); ptrue = P['pred'][np.arange(len(y)), y]

    # z-scale for feature deltas (dev predicted features)
    mu, sd = Xp[di].mean(0), Xp[di].std(0) + 1e-6
    alt = [a for a in A.alt_rules.split(',') if a]
    ledger = []
    for i, r in enumerate(rows):
        t, p = C4[y[i]], C4[pp[i]]; ok = t == p; sc = cars.get(r['scene'], {}); prm = sc.get('params', {})
        raw, rt = r['raw'], r['raw_true']
        delta = {k: {'pred': raw.get(k), 'true': None if rt is None else rt.get(k)} for k in RAW_KEYS}
        feat_d = None
        if rt is not None:
            z = (np.array(EV.feat(raw)) - np.array(EV.feat(rt))) / sd
            feat_d = [{'feature': FN[k], 'z': round(float(z[k]), 2)} for k in np.argsort(-np.abs(z))[:3]]
        truth_ctx = {'visible_fraction': (sc.get('visible_fraction') or {}).get(r['site']),
                     'wound_visible_px': (sc.get('wound_visible_px') or {}).get(r['site']),
                     'wound_true': r['site'] in (prm.get('wounds') or {}), 'amputation_true': r['site'] in (prm.get('amputations') or {}),
                     'tourniquet': r['site'] in (prm.get('tourniquets') or []), 'occluder': prm.get('occluder'),
                     'occluder_at_end': prm.get('occ_target') == 'limb_end', 'position': prm.get('body_position'),
                     'facing': {1: 'front', -1: 'back'}.get(r['facing'])}
        kpe = kp_error(r['kp'], gtkps.get(r['scene']), r['site'])
        # severity
        def keypx(cls, src):
            return None if src is None or cls not in KEY else src.get(KEY[cls])
        tsize = keypx(t, rt) if t == 'amputation' else truth_ctx['wound_visible_px'] if t == 'wound' else None
        if not ok:
            sev = 'gross' if conf[i] >= 0.8 or (tsize is not None and t in GROSS_SIZE and tsize >= GROSS_SIZE[t]) else 'near'
        else:
            kt, kpp = keypx(t, rt), keypx(t, raw)
            lucky = kt is not None and kt >= 20 and kpp is not None and kpp <= 0.3 * kt
            sev = 'lucky' if lucky else 'near' if ptrue[i] < 0.6 else 'ok'
        # cause flags (evaluated for wrong sites and lucky hits)
        flags, sub, fixed_by = [], None, []
        if not ok or sev == 'lucky':
            if any(r['labels'].get(a) == p for a in alt if a in r['labels'] and r['labels'][a] != t) or \
               (t == 'amputation' and rt is not None and rt.get('stump_px', 0) == 0):
                flags.append('L')
            if not np.isnan(P['side'][i, 0]) and P['side'][i].argmax() == y[i]:
                flags.append('K')
            if kpe and (kpe['side_swap'] or kpe['err_torso'] > 0.15):
                flags.append('K-pos')
            tf = not np.isnan(P['true_fixed'][i, 0]) and P['true_fixed'][i].argmax() == y[i]
            trf = not np.isnan(P['true_refit'][i, 0]) and P['true_refit'][i].argmax() == y[i]
            if tf or trf:
                flags.append('S')
                kt, kpp = keypx(t, rt), keypx(t, raw); hp, ht = keypx(p, raw), keypx(p, rt)
                vp, vt = raw.get('vis_px', 0), (rt or {}).get('vis_px', 0)
                if kt is not None and kt >= 20 and (kpp or 0) <= 0.3 * kt:
                    sub = 'miss'
                elif hp is not None and hp >= 20 and (ht or 0) <= 0.3 * hp:
                    sub = 'hall'
                elif vt and not (0.5 <= vp / max(vt, 1) <= 2):
                    sub = 'extent'
                else:
                    sub = 'other'
                fixed_by = [k for k in GROUPS if GS[k][i] == y[i]]
            elif rt is not None:
                flags.append('F')
        primary = next((c for c in ('L', 'K', 'S', 'F') if c in flags), None) if not ok else None
        ledger.append({'schema': SCHEMA, 'phase': A.phase, 'scene': r['scene'], 'site': r['site'], 'split': r['split'],
                       'true': t, 'pred': p, 'correct': bool(ok), 'p': [round(float(v), 3) for v in P['pred'][i]],
                       'margin': round(float(conf[i] - np.sort(P['pred'][i])[-2]), 3), 'severity': sev,
                       'cause': primary, 'cause_sub': sub if primary == 'S' else None, 'flags': flags, 'fixed_by_group': fixed_by,
                       'counterfactual': {k: (None if np.isnan(P[k][i, 0]) else C4[int(P[k][i].argmax())]) for k in ('side', 'true_fixed', 'true_refit')},
                       'labels_alt': {a: r['labels'].get(a) for a in alt}, 'delta': delta, 'feature_delta_top3': feat_d, 'kp': kpe,
                       'truth': truth_ctx, 'review': {'cause_override': None, 'note': None}})

    # metrics
    out = {'schema': SCHEMA, 'phase': A.phase, 'date_utc': datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
           'rule': A.rule, 'C': C, 'manifest': man, 'commit': repo_commit()
           if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'COMMIT')) else None,
           'coverage': {'side_truth': round(float(has_side.mean()), 3), 'true_map': round(float(has_true.mean()), 3),
                        'gt_keypoints': round(len(gtkps) / max(len(pred), 1), 3)}}
    for name, idx in ((A.dev + '_oof', dev), (A.eval, ev)):
        if name and idx.any():
            out[name] = metrics(y[idx], pp[idx])
            for k in ('side', 'true_fixed', 'true_refit'):
                ok_ = idx & ~np.isnan(P[k][:, 0])
                if ok_.any():
                    out[name]['counterfactual_' + k] = round(float((P[k][ok_].argmax(1) == y[ok_]).mean()), 4)
    with open(os.path.join(A.out, 'ledger.jsonl'), 'w') as fh:
        for L in ledger:
            fh.write(json.dumps(L) + '\n')
    json.dump(out, open(os.path.join(A.out, 'metrics.json'), 'w'), indent=1)
    retro(A, out, ledger)
    if A.maps and A.images:
        sheet(A, ledger)
    print('SCORE_DONE', json.dumps({k: out[k] for k in out if k.endswith('_oof') or k == A.eval}), flush=True)


def retro(A, out, ledger):
    L = []; w = L.append
    w(f"# Retrospective: {A.phase}\n\n{out['date_utc']} | rule {A.rule} | commit {out['commit']} | manifest {(out['manifest'] or {}).get('sha256', 'none')[:12]}\n")
    w('## Metrics\n\n| Split | n | Acc | Macro F1 | Min recall | False wound | False amp | CF side | CF true map (fixed) | CF true map (refit) |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for k, v in out.items():
        if isinstance(v, dict) and 'acc' in v:
            w(f"| {k} | {v['n']} | {v['acc']:.3f} | {v['macro_f1']:.3f} | {v['min_recall']:.2f} | {v['false_wound']:.3f} | {v['false_amp']:.3f} | "
              f"{v.get('counterfactual_side', '')} | {v.get('counterfactual_true_fixed', '')} | {v.get('counterfactual_true_refit', '')} |")
    w(f"\nCoverage: {out['coverage']}\n")
    for split in sorted(set(l['split'] for l in ledger)):
        E = [l for l in ledger if l['split'] == split]
        w(f'## Cause Pareto ({split})\n\n| Cause | Gross | Near | Total | Share of errors |\n|---|---:|---:|---:|---:|')
        errs = [l for l in E if not l['correct']]; codes = {}
        for l in errs:
            c = l['cause'] + ('-' + l['cause_sub'] if l['cause_sub'] else '') if l['cause'] else 'unassigned'
            codes.setdefault(c, [0, 0]); codes[c][0 if l['severity'] == 'gross' else 1] += 1
        for c, (gr, nr) in sorted(codes.items(), key=lambda kv: -sum(kv[1])):
            w(f'| {c} | {gr} | {nr} | {gr + nr} | {100 * (gr + nr) / max(len(errs), 1):.0f}% |')
        w(f"\nErrors {len(errs)}; gross {sum(l['severity'] == 'gross' for l in errs)}; lucky hits {sum(l['severity'] == 'lucky' for l in E)}; "
          f"near-right {sum(l['correct'] and l['severity'] == 'near' for l in E)}\n")
        w(f'### By true class ({split})\n\n| True | Pred | n | Primary causes |\n|---|---|---:|---|')
        pairs = {}
        for l in errs:
            pairs.setdefault((l['true'], l['pred']), []).append(l['cause'] or '?')
        for (t, p), cs in sorted(pairs.items(), key=lambda kv: -len(kv[1])):
            w(f"| {t} | {p} | {len(cs)} | {', '.join(f'{c} {cs.count(c)}' for c in sorted(set(cs), key=lambda c: -cs.count(c)))} |")
        w('')
        w(f'### Key-pixel ratio, predicted / true, on true wound and amputation sites ({split})\n\n| Class | Position | n | Median ratio | Share under 0.3 |\n|---|---|---:|---:|---:|')
        grp = {}
        for l in E:
            if l['true'] in KEY and l['delta'][KEY[l['true']]]['true']:
                d = l['delta'][KEY[l['true']]]; grp.setdefault((l['true'], l['truth']['position']), []).append((d['pred'] or 0) / d['true'])
        for (c, pos), v in sorted(grp.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
            w(f'| {c} | {pos} | {len(v)} | {np.median(v):.2f} | {np.mean(np.array(v) < 0.3):.0%} |')
        w('')
        w(f'### Top gross misses ({split})\n\n| Scene | Site | True | Pred | p | Cause | Key px pred/true | Vis frac | Position | Occluder | Top feature deltas (z) |\n|---|---|---|---|---:|---|---|---:|---|---|---|')
        for l in sorted([l for l in errs if l['severity'] == 'gross'], key=lambda l: -max(l['p']))[:A.sheet]:
            kk = KEY.get(l['true']) or KEY.get(l['pred']); d = l['delta'].get(kk, {}) if kk else {}
            fd = '; '.join(f"{q['feature']} {q['z']}" for q in (l['feature_delta_top3'] or []))
            vf = l['truth']['visible_fraction']
            w(f"| {l['scene']} | {l['site']} | {l['true']} | {l['pred']} | {max(l['p']):.2f} | {l['cause']}{'-' + l['cause_sub'] if l['cause_sub'] else ''} | "
              f"{d.get('pred')}/{d.get('true')} | {'' if vf is None else round(vf, 2)} | {l['truth']['position']} | {l['truth']['occluder']} | {fd} |")
        w('')
    # change table and persistent failures
    prevs = [p for p in A.prev.split(',') if p]
    if prevs:
        cur = {(l['scene'], l['site']): l for l in ledger}
        for pth in prevs:
            old = {}
            for line in open(pth):
                q = json.loads(line); old[(q['scene'], q['site'])] = q
            common = [k for k in cur if k in old]
            ch = {'fixed': 0, 'broke': 0, 'still_wrong': 0, 'still_right': 0}; moved = {}
            for k in common:
                a, b = old[k]['correct'], cur[k]['correct']
                st = 'fixed' if (not a and b) else 'broke' if (a and not b) else 'still_wrong' if not b else 'still_right'
                ch[st] += 1
                if st in ('fixed', 'broke'):
                    moved.setdefault(st, {}); key = f"{cur[k]['true']}"; moved[st][key] = moved[st].get(key, 0) + 1
            w(f"## Change vs {os.path.basename(os.path.dirname(pth)) or pth} ({len(common)} common sites)\n\n{ch}\n\nBy true class: {moved}\n")
        wrong_all = [k for k in cur if not cur[k]['correct']]
        for pth in prevs:
            old = {(json.loads(l)['scene'], json.loads(l)['site']): json.loads(l)['correct'] for l in open(pth)}
            wrong_all = [k for k in wrong_all if old.get(k) is False]
        w(f'## Persistent failures (wrong in this and all {len(prevs)} earlier ledgers): {len(wrong_all)}\n')
        for k in wrong_all[:40]:
            l = cur[k]; w(f"- {k[0]} {k[1]}: {l['true']} -> {l['pred']} ({l['cause']}) candidate L or G review")
        w('')
    w('## Review\n\nOverride `review.cause_override` and add `review.note` in ledger.jsonl for the gross misses; code G marks suspected generator truth errors.\n')
    open(os.path.join(A.out, 'retro.md'), 'w').write('\n'.join(L))


PAL = np.random.default_rng(3).integers(40, 255, (64, 3)).astype(np.uint8); PAL[0] = 0


def sheet(A, ledger):
    import cv2
    sys.path.insert(0, A.gen)
    import parts as PT
    def panel(l):
        img = None
        for d in A.images.split(','):
            f = os.path.join(d, l['scene'] + '.jpg')
            if os.path.exists(f):
                img = cv2.resize(cv2.imread(f), (320, 240)); tp = os.path.join(d, l['scene'] + '_part3.png'); break
        mp = os.path.join(A.maps, l['scene'] + '.npz')
        if img is None or not os.path.exists(mp):
            return None
        cls, _, _ = PT.decode3(tp); tm = cv2.resize(PAL[cls % 64], (320, 240), interpolation=cv2.INTER_NEAREST)
        pm = cv2.resize(PAL[np.load(mp)['seg'] % 64], (320, 240), interpolation=cv2.INTER_NEAREST)
        row = np.concatenate([img, tm, pm], 1)
        cv2.putText(row, f"{l['scene']} {l['site']} {l['true']}->{l['pred']} p{max(l['p']):.2f} {l['cause'] or ''}{'-' + l['cause_sub'] if l['cause_sub'] else ''}",
                    (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return row
    for name, sel in (('gross', sorted([l for l in ledger if l['severity'] == 'gross'], key=lambda l: -max(l['p']))[:A.sheet]),
                      ('lucky', [l for l in ledger if l['severity'] == 'lucky'][:10])):
        rows = [p for p in (panel(l) for l in sel) if p is not None]
        if rows:
            cv2.imwrite(os.path.join(A.out, f'sheet_{name}.jpg'), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 80])


if __name__ == '__main__':
    main()
