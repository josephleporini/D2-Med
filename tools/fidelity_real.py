"""Real-vs-synthetic fidelity gap (BT-2; Joseph 28 Sep: "assess better fidelity images to refine training").

Runs the BT-1 engine (unchanged) on real manikin photos from the T5 RESERVE list (never the primary 81, which stay the
scored check set) and on synthetic sets, and compares what the engine sees. No real labels are used.

  fetch   <reserve_urls.csv> <out_dir>           download each page's og:image (DVIDS, Wikimedia); skips failures
  measure <models_dir> <ckpt> <decision.json> <set_name> <image_dir> <out.jsonl> [max_n]
  compare <out_dir> <real.jsonl> <synthetic.jsonl> [...]   -> summary.json (aggregates only) and SUMMARY lines

Per image (measure): image statistics (brightness, contrast, saturation, red-pixel share), person box count and area
share of the frame, pose score, predicted facing confidence, per-site visible limb pixels and limbs in view, wound
pixels, side confidence, per-site class probabilities (mirror-averaged) and their entropy, and a 256-d SAM image
embedding. compare: distribution per statistic (median, quartiles) real vs each synthetic set, the predicted class mix,
and a real-vs-synthetic classifier on the embeddings (cross-validated AUC: 0.5 = indistinguishable).

Data rule: real photos and per-image rows stay on the volume. Only compare's aggregate summary is printed; nothing
derived from real photos is pushed to a public repository (publish guard G3).
"""
import os, sys, json, csv, re, time, glob
import numpy as np


def fetch(csv_path, out):
    import urllib.request
    os.makedirs(out, exist_ok=True); ok = bad = 0
    hdr = {'User-Agent': 'Mozilla/5.0 (research; D2 fidelity check)'}
    for r in csv.DictReader(open(csv_path)):
        n = int(r['n']); dst = os.path.join(out, f'R{n:04d}.jpg')
        if os.path.exists(dst):
            ok += 1; continue
        try:
            html = urllib.request.urlopen(urllib.request.Request(r['page_url'], headers=hdr), timeout=30).read().decode('utf8', 'ignore')
            m = re.search(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)', html) or \
                re.search(r'content=["\']([^"\']+)["\'][^>]+property=["\']og:image', html)
            if not m and 'wikimedia' in r['page_url']:
                m = re.search(r'href="(https://upload\.wikimedia\.org/[^"]+\.(?:jpg|jpeg|png))"', html, re.I)
            if not m:
                raise ValueError('no og:image')
            url = m.group(1).replace('&amp;', '&')
            data = urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=60).read()
            open(dst, 'wb').write(data); ok += 1
            json.dump({'n': n, 'page': r['page_url'], 'image': url, 'screening_class': r['screening_class'],
                       'pose': r['pose'], 'camera_view': r['camera_view']}, open(dst[:-4] + '_src.json', 'w'))
        except Exception as ex:
            bad += 1; print('FETCH_FAIL', n, type(ex).__name__, str(ex)[:120], flush=True)
        time.sleep(0.5)
    print('FETCH_DONE ok', ok, 'failed', bad, flush=True)


def img_stats(rgb):
    import cv2
    x = rgb.astype(np.float32) / 255
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    red = (x[..., 0] > 1.4 * x[..., 1] + 0.02) & (x[..., 0] > 0.25)
    return dict(bright=float(x.mean()), contrast=float(x.mean(2).std()), sat=float(hsv[..., 1].mean() / 255),
                red_share=float(red.mean()))


def measure(models, ckpt, dec, set_name, d, out, max_n=None):
    sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', p) for p in ('gen', 'jobs')]
    import torch, torch.nn.functional as F
    from PIL import Image, ImageOps
    import engine_bt1 as EB
    from d2pipe import letterbox
    E = EB.BT1Engine(models, ckpt, dec, tta=True)
    files = sorted(f for f in glob.glob(os.path.join(d, '*.jpg')))
    files = [f for f in files if re.match(r'^[A-Z]+\d+\.jpg$', os.path.basename(f))][:int(max_n) if max_n else None]
    fo = open(out, 'w')
    for f in files:
        with Image.open(f) as im:
            rgb = np.asarray(ImageOps.exif_transpose(im).convert('RGB'), dtype=np.uint8)
        H0, W0 = rgb.shape[:2]
        P, info = E.predict_image(rgb)
        boxes, k, s = E._last_dp          # last call was the mirrored pass: counts and areas are mirror-invariant
        img = letterbox(np.ascontiguousarray(rgb))
        areas = sorted([float((b[2] - b[0]) * (b[3] - b[1]) / (img.shape[0] * img.shape[1])) for b in boxes], reverse=True) if len(boxes) else []
        with torch.no_grad():
            E.m['P'].set_image(__import__('cv2').resize(img, (1024, 1024)))
            emb = F.adaptive_avg_pool2d(E.m['P']._features['image_embed'].float(), 1).flatten().cpu().numpy()
        rows = info['rows']
        ent = [float(-(p * np.log(p + 1e-9)).sum()) for p in P]
        row = dict(set=set_name, image=os.path.basename(f), w=W0, h=H0, **img_stats(img),
                   n_person=int(len(boxes)), box_area=areas[0] if areas else 0.0, box_area_2nd=areas[1] if len(areas) > 1 else 0.0,
                   pose_score=float(np.asarray(s)[:, :17].mean(1).max()) if len(k) else 0.0,
                   facing=info['frame']['facing'], facing_conf=float(info['frame'].get('facing_conf') or 0),
                   vis_px={q: rows[q]['vis_px'] for q in EB.SITES}, limbs_in_view=int(sum(rows[q]['vis_px'] >= 30 for q in EB.SITES)),
                   wound_px={q: rows[q]['wound_px'] for q in EB.SITES}, p=P.round(4).tolist(), entropy=ent,
                   pred=[['no_injury', 'wound', 'amputation', 'not_testable'][int(i)] for i in P.argmax(1)],
                   emb=emb.round(5).tolist())
        fo.write(json.dumps(row) + '\n'); fo.flush()
    print('MEASURE_DONE', set_name, len(files), flush=True)


def compare(out, real, *syn):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    load = lambda p: [json.loads(l) for l in open(p)]
    R = load(real); S = {os.path.basename(p).split('.')[0]: load(p) for p in syn}
    stats = ['bright', 'contrast', 'sat', 'red_share', 'n_person', 'box_area', 'pose_score', 'facing_conf', 'limbs_in_view']
    def q(v):
        v = np.asarray(v, float); return [round(float(x), 4) for x in np.percentile(v, [25, 50, 75])]
    def derived(rows):
        return dict(limb_px_median=float(np.median([np.median([x for x in r['vis_px'].values()]) for r in rows])),
                    wound_px_share=float(np.mean([sum(v > 20 for v in r['wound_px'].values()) > 0 for r in rows])),
                    no_person=float(np.mean([r['n_person'] == 0 for r in rows])),
                    multi_person=float(np.mean([r['n_person'] >= 2 for r in rows])),
                    entropy_median=float(np.median([e for r in rows for e in r['entropy']])),
                    pred_mix={c: round(float(np.mean([p == c for r in rows for p in r['pred']])), 3)
                              for c in ('no_injury', 'wound', 'amputation', 'not_testable')},
                    facing={f: round(float(np.mean([r['facing'] == f for r in rows])), 3) for f in ('front', 'back', 'unknown')})
    res = {'real_n': len(R), 'stats': {}, 'derived': {'real': derived(R)}, 'domain_auc': {}}
    for k in stats:
        res['stats'][k] = {'real': q([r[k] for r in R])}
        for n, rows in S.items():
            res['stats'][k][n] = q([r[k] for r in rows])
    for n, rows in S.items():
        res['derived'][n] = derived(rows)
        X = np.array([r['emb'] for r in R] + [r['emb'] for r in rows]); y = np.array([1] * len(R) + [0] * len(rows))
        auc = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)), X, y,
                              cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring='roc_auc')
        res['domain_auc'][n] = round(float(auc.mean()), 3)
    # nearest synthetic neighbour (cosine) for each real image, pooled synthetic
    Sall = np.array([r['emb'] for rows in S.values() for r in rows]); Sall /= np.linalg.norm(Sall, axis=1, keepdims=True)
    Rm = np.array([r['emb'] for r in R]); Rm /= np.linalg.norm(Rm, axis=1, keepdims=True)
    Ssame = Sall @ Sall.T; np.fill_diagonal(Ssame, -1)
    res['nn_cosine'] = {'real_to_synthetic_median': round(float(np.median((Rm @ Sall.T).max(1))), 4),
                        'synthetic_to_synthetic_median': round(float(np.median(Ssame.max(1))), 4)}
    json.dump(res, open(os.path.join(out, 'summary.json'), 'w'), indent=1)
    print('SUMMARY', json.dumps(res))


if __name__ == '__main__':
    {'fetch': fetch, 'measure': measure, 'compare': compare}[sys.argv[1]](*sys.argv[2:])
