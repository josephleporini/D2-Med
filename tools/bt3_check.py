"""BT-3 early (A40) checks for the BT-1 engine in the qualification container.

  parity  <models_dir> <ckpt> <decision.json> <dev5_dir> '<ref glob>' <n> <out.json>
      runs gen/engine_bt1.py (no mirror) on n dev5 scenes decoded the way the container decodes (PIL, EXIF), and compares
      its site rows and decision-layer classes with the reference extraction rows of the BT-1 run (DDData results/bt1/ext)
  score   <predictions.json> <rows.jsonl> <dev5_dir> <decision.json> <out.json>
      (1) container classes against the 0.10-rule labels (decision layer fitted on all dev5, so in-sample);
      (2) out-of-fold accuracy from the container's own logged rows: grouped 5-fold by scene, fitted on original-image
          rows, applied to original and mirrored rows, averaged (same method as jobs/tta_score.py);
      (3) per-image time and its breakdown
"""
import sys, os, json, glob, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'gen')); sys.path.insert(0, os.path.join(HERE, '..', 'jobs'))
sys.path.insert(0, os.path.join(HERE, '..', 'score'))
C4 = ['no_injury', 'wound', 'amputation', 'not_testable']
S = ['LUE', 'RUE', 'LLE', 'RLE']
PX = ['vis_px', 'ext_px', 'stump_px', 'wound_px', 'tq_px']


def pil_rgb(path):
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert('RGB'), dtype=np.uint8)


def parity(models, ckpt, dec, d5, ref_glob, n, out):
    import engine_bt1 as EB
    from d2pipe import letterbox
    ref = {}
    for f in glob.glob(ref_glob):
        for l in open(f):
            r = json.loads(l); ref[r['scene']] = r
    E = EB.BT1Engine(models, ckpt, dec, tta=False)
    scenes = sorted(ref)[::max(1, len(ref) // int(n))][:int(n)]
    res, T = [], []
    for sid in scenes:
        t0 = time.time()
        img = letterbox(pil_rgb(os.path.join(d5, sid + '.jpg')))
        rows, frame, t = E.rows(img, False); T.append(time.time() - t0)
        Pn = E.layer.proba([EB.features(rows[s]) for s in S]); Pr = E.layer.proba([EB.features(ref[sid]['sites'][s]) for s in S])
        for k, s in enumerate(S):
            a, b = rows[s], ref[sid]['sites'][s]
            dpx = max(abs(a[q] - b[q]) for q in PX)
            dp = max([abs(x - y) for x, y in zip(a['p_end'] or [], b['p_end'] or [])] + [0.0])
            dl = max([abs(x - y) for x, y in zip(a['limb']['p_cause'], b['limb']['p_cause'])] + [abs(a['limb']['amodal_px'] - b['limb']['amodal_px']) / 100.0])
            res.append(dict(scene=sid, site=s, dpx=dpx, dp=round(dp, 4), dlimb=round(dl, 4), dprob=round(float(np.abs(Pn[k] - Pr[k]).max()), 4),
                            same_class=bool(Pn[k].argmax() == Pr[k].argmax()), facing=(frame['facing'], ref[sid]['frame']['facing'])))
    n_s = len(res); differ = sum(r['dpx'] > 2 or r['dp'] > 0.02 for r in res)
    summ = dict(scenes=len(scenes), sites=n_s, sites_differ=differ, class_agree=round(sum(r['same_class'] for r in res) / n_s, 4),
                max_dprob=max(r['dprob'] for r in res), facing_agree=round(np.mean([r['facing'][0] == r['facing'][1] for r in res]), 4),
                s_per_pass_mean=round(float(np.mean(T[1:] or T)), 3), first_pass_s=round(T[0], 2))
    summ['ok'] = summ['class_agree'] >= 0.97 and differ <= 0.05 * n_s
    json.dump(dict(summary=summ, sites=res), open(out, 'w'), indent=1)
    print('PARITY_OK' if summ['ok'] else 'PARITY_DIFF', json.dumps(summ))


def truth(d5):
    T = {}
    for f in glob.glob(os.path.join(d5, '*_sidecar.json')):
        c = json.load(open(f)); T[c['scene_id']] = [C4.index(c['labels_by_threshold']['0.10'][s]) for s in S]
    return T


def score(pred, rows, d5, dec, out):
    from score import mk, proba, metrics
    from sklearn.model_selection import GroupKFold
    import engine_bt1 as EB
    T = truth(d5)
    doc = json.load(open(pred))
    SITE_OF = {('upper_extremity', 'left'): 'LUE', ('upper_extremity', 'right'): 'RUE',
               ('lower_extremity', 'left'): 'LLE', ('lower_extremity', 'right'): 'RLE'}
    y, p = [], []
    for r in doc['predictions']:
        sid = os.path.splitext(r['image_id'])[0]
        got = {SITE_OF[(q['body_region'], q['laterality'])]: C4.index(q['injury_type']) for q in r['sites']}
        for k, s in enumerate(S):
            y.append(T[sid][k]); p.append(got[s])
    y, p = np.array(y), np.array(p)
    res = {'container_in_sample': metrics(y, p)}
    R = [json.loads(l) for l in open(rows)]
    R = [r for r in R if r.get('sites') and r.get('sites_flip')]
    X, XF, yy, g = [], [], [], []
    for r in R:
        sid = os.path.splitext(r['image_id'])[0]
        for k, s in enumerate(S):
            X.append(EB.features(r['sites'][s])); XF.append(EB.features(r['sites_flip'][s])); yy.append(T[sid][k]); g.append(sid)
    X, XF, yy, g = map(np.array, (X, XF, yy, g))
    C = json.load(open(dec))['C']; P, PF = np.zeros((len(yy), 4)), np.zeros((len(yy), 4))
    for i, j in GroupKFold(5).split(X, yy, g):
        m = mk(C).fit(X[i], yy[i]); P[j] = proba(m, X[j]); PF[j] = proba(m, XF[j])
    res['oof_original'] = metrics(yy, P.argmax(1)); res['oof_mirror_average'] = metrics(yy, ((P + PF) / 2).argmax(1))
    res['m3_02_sites_over_0.05_without_mirror'] = round(float((np.abs(P - PF).max(1) > 0.05).mean()), 4)
    sec = np.array([r['s'] for r in R])
    bd = {k: round(float(np.mean([r['t'][k] + r['t_flip'][k] for r in R])), 3) for k in R[0]['t']}
    res['time'] = dict(images=len(R), s_per_image_mean=round(float(sec.mean()), 3), p50=round(float(np.percentile(sec, 50)), 3),
                       p95=round(float(np.percentile(sec, 95)), 3), max=round(float(sec.max()), 3), breakdown_both_passes=bd)
    res['errors'] = sum(1 for l in open(rows) if json.loads(l).get('error'))
    json.dump(res, open(out, 'w'), indent=1); print('SCORE', json.dumps(res))


if __name__ == '__main__':
    {'parity': parity, 'score': score}[sys.argv[1]](*sys.argv[2:])
