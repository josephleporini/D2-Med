"""R2: segmentation overlays on the real T5 photos, and the tagging sheet that answers KP-1 (Perception thread, WC-1).

Question (KP-1, Master Schedule Rev 6): is the real-photo failure in the segmenter or in the decision layer, and is there
a credible path past the 39% constant baseline? R1 (D2_Perception_R1_Refit_v1.0) measured PARTIAL: a real-label refit
removes the wound over-call but the per-site features carry weak signal. R2 looks at what the engine actually saw.

T5 is a development set. Nothing here is a claim. Photos, overlays, per-photo rows and tags stay on the volume
(Decision Record 2026-10-09 row 1, option C; G3; C-14; C-15). Only counts are printed.

KP-1 DECISION RULE, fixed before any overlay exists (IPT_Plan_Perception note 1 (b); confirmed by Joseph 10 Oct
"1 - confirm", Decision Record 2026-10-10 A1; re-confirmed 11 Oct 21:25 Arizona "Confirm kp-1"):
  Errors tagged: every T5 site that the R1 refit gets wrong, for model v3 (working model), reading as_labeled, arm =
  nested selection by accuracy (the honest R1 figure, 43.4%). Each error gets one cause:
    S1  wrong person chosen as the casualty
    S2  limb mask misplaced (on gear, background or another person) or missing where the limb is visible
    S3  mask correct, but the injury cue is not captured by the features
    S4  label doubtful, or the site truly ambiguous
    S5  other
  CONTINUE if both hold: (i) one named, costed fix addresses at least half of the tagged errors (S1 and S2 by segmenter
  or casualty-selection adaptation on real texture; S3 by the direct engine, if the case note supports it); and (ii)
  that fix can be tested on development data before 20 Nov against a pre-set target of nested accuracy >= 49%
  (baseline 39.0% + 10 points) with wound recall > 0.
  STOP otherwise: report the transfer gap and its tagged causes as a finding (Publication Outline branch).
  The tag counts below decide (i); the direct-engine case note and the KP-1 report decide (ii). The KP-1 outcome
  itself is Joseph's on 23 Oct.

Stages:
  errors   CPU. Recomputes R1's out-of-fold predictions for v3 (same groups, folds, arms and nested selection as
           tools/r1_refit.py); parity check against R1 (nested accuracy 0.434, best fixed-C B 0.470) or stop.
           Writes errors.csv (the sites to tag) to the volume.
  render   GPU. Runs the v3 engine (BT1_PICK=lying, as scored on 30 Sep) on each T5 photo, original orientation only,
           and captures what feeds the features: every detected person box, the chosen casualty, its keypoints, the
           part map (four limb sites, torso, occluder), the wound and tourniquet masks. One overlay PNG per photo,
           truth and refit prediction written beside each site. Writes rows of automatic proxies (people detected,
           limb mask pixels) to the volume; prints counts only.
  sheet    Writes the tagging notebook (R2_tag.ipynb) and an empty tags.csv beside the overlays.
  tally    Reads tags.csv and prints the counts the rule needs. Run in the viewing pod after tagging.

usage: python tools/r2_overlays.py errors <out_dir> <labels.csv> <photo_dir> <t5_rows_v3> <dev5_rows_v3> <dev5_dir>
       python tools/r2_overlays.py render <out_dir> <photo_dir> <models_dir> <ckpt> <decision_json>
       python tools/r2_overlays.py sheet  <out_dir>
       python tools/r2_overlays.py tally  <out_dir>
"""
import sys, os, json, csv, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE] + [os.path.join(HERE, '..', p) for p in ('gen', 'jobs', 'score', 'adapt')]

TAGS = ['S1', 'S2', 'S3', 'S4', 'S5']
TAG_TEXT = {'S1': 'wrong person chosen as the casualty',
            'S2': 'limb mask misplaced or missing',
            'S3': 'mask correct, injury cue not captured',
            'S4': 'label doubtful or site ambiguous',
            'S5': 'other'}
PARITY = {'nested_acc': 0.434, 'bestB_acc': 0.470}     # R1 v3 as_labeled, D2_Perception_R1_Refit_v1.0
COL = {'LUE': (230, 25, 75), 'RUE': (60, 180, 75), 'LLE': (0, 130, 200), 'RLE': (255, 225, 25),
       'TORSO': (145, 30, 180), 'OCC': (70, 240, 240), 'HEAD': (240, 50, 230)}


# ----------------------------------------------------------------------------------------------------------- errors
def stage_errors(out, lab, photo_dir, t5p, d5p, d5d):
    import r1_refit as R1
    import kit as K
    from sklearn.model_selection import GroupKFold
    from score import fitC, mk
    import bt2_eval as BE
    L = R1.t5_labels(lab)
    G, _, dropped = R1.groups_for(L, photo_dir)
    Xs, XFs, ys, gs = BE.load(d5p, d5d)
    C = fitC(Xs, ys, gs); base = R1.Base(mk(C).fit(Xs, ys), C)
    R = R1.t5_rows(t5p)
    X, XF, y, g, ids = R1.design(L, R, G, 'as_labeled')
    cands = dict(K.arms(base, Xs, ys))
    folds = list(GroupKFold(min(5, len(set(g)))).split(X, y, g))
    P = {nm: np.zeros((len(y), 4)) for nm in cands}
    PS = np.zeros((len(y), 4)); picks = []
    for tr, te in folds:
        for nm, f in cands.items():
            P[nm][te] = R1.mp(R1.fit_arm(nm, f, base, X[tr], y[tr], Xs, ys), X[te], XF[te])
        ch, _ = K.select(base, X[tr], y[tr], g[tr], Xs, ys, XF[tr]); picks.append(ch); PS[te] = P[ch][te]
    bnames = [n for n in P if n.startswith('B_')]
    bbest = max(bnames, key=lambda n: float((P[n].argmax(1) == y).mean()))
    pn, pb, pa = PS.argmax(1), P[bbest].argmax(1), P['A_synthetic'].argmax(1)
    acc_n, acc_b = float((pn == y).mean()), float((pb == y).mean())
    ok = abs(acc_n - PARITY['nested_acc']) < 0.0015 and abs(acc_b - PARITY['bestB_acc']) < 0.0015
    print('R2_PARITY', json.dumps(dict(nested=round(acc_n, 4), bestB=[bbest, round(acc_b, 4)], picks=picks,
                                       expected=PARITY, ok=ok, groups=len(set(g)), dvids_rule_dropped=dropped)), flush=True)
    if not ok:
        raise SystemExit('R2_PARITY_FAIL: out-of-fold predictions do not reproduce R1; nothing to tag')
    C4 = K.C4
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'sites.csv'), 'w', newline='') as f:     # every labeled site, for the overlays
        w = csv.writer(f); w.writerow(['n', 'site', 'truth', 'pred_nested', 'pred_bestB', 'pred_shipped', 'p_nested', 'error'])
        for i, (n, s) in enumerate(ids):
            w.writerow([n, s, C4[y[i]], C4[pn[i]], C4[pb[i]], C4[pa[i]],
                        ' '.join(f'{v:.2f}' for v in PS[i]), int(pn[i] != y[i])])
    err = [i for i in range(len(y)) if pn[i] != y[i]]
    with open(os.path.join(out, 'errors.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['n', 'site', 'truth', 'pred_nested'])
        for i in err:
            w.writerow([ids[i][0], ids[i][1], C4[y[i]], C4[pn[i]]])
    cm = np.zeros((4, 4), int)
    for i in err:
        cm[y[i], pn[i]] += 1
    print('R2_ERRORS', json.dumps(dict(sites=int(len(y)), errors=len(err), photos_with_error=len({ids[i][0] for i in err}),
                                       by_truth={C4[k]: int(sum(y[i] == k for i in err)) for k in range(4)},
                                       confusion_errors_rows_truth=cm.tolist(), bestB=bbest)), flush=True)


# ----------------------------------------------------------------------------------------------------------- render
def stage_render(out, photo_dir, models_dir, ckpt, decision):
    import cv2
    import engine_bt1 as EB
    from d2pipe import letterbox
    assert EB.PICK == 'lying', 'R2 renders the v3 engine as scored on 30 Sep (BT1_PICK=lying)'
    sites = {}
    for r in csv.DictReader(open(os.path.join(out, 'sites.csv'))):
        sites.setdefault(int(r['n']), {})[r['site']] = r
    cap = {}
    orig = EB.site_rows_batched

    def grab(m, img, cm, names, ext, stc, wm, tq, chunk=8):
        cap.update(cm=cm.copy(), names=list(names), wm=np.asarray(wm).copy(), tq=np.asarray(tq).copy())
        return orig(m, img, cm, names, ext, stc, wm, tq, chunk)
    EB.site_rows_batched = grab
    E = EB.BT1Engine(models_dir, ckpt, decision, tta=False)
    od = os.path.join(out, 'overlays'); os.makedirs(od, exist_ok=True)
    proxy, fails, t0 = [], 0, time.time()
    for n in sorted(sites):
        p = os.path.join(photo_dir, f'R{n:04d}.jpg')
        try:
            rgb = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            img = letterbox(np.ascontiguousarray(rgb)); cap.clear()
            rows, frame, _ = E.rows(img, False)
            boxes, k, s = E._last_dp
            pick = EB.pick_casualty(boxes, k, s, img.shape) if len(boxes) > 1 and len(k) == len(boxes) else (0 if len(boxes) else None)
            draw(od, n, img, boxes, k, s, pick, cap, rows, sites[n], frame)
            proxy.append(dict(n=n, people=int(len(boxes)), facing=frame.get('facing'),
                              vis_px={st: int(rows[st]['vis_px']) for st in rows}, wound_px={st: int(rows[st]['wound_px']) for st in rows}))
        except Exception as e:                                     # an image failure is a finding, not a stop
            fails += 1; print('R2_IMAGE_FAIL', n, type(e).__name__, str(e)[:160], flush=True)
    with open(os.path.join(out, 'proxy.jsonl'), 'w') as f:
        for r in proxy:
            f.write(json.dumps(r) + '\n')
    summarize(out, proxy, sites, fails, time.time() - t0)


def draw(od, n, img, boxes, k, s, pick, cap, rows, srows, frame):
    """Left: the photo as the engine saw it (letterboxed), every person box (grey) and the chosen casualty (yellow) with
    its keypoints. Right: the part map the features are computed from, wound (red outline) and tourniquet (orange)
    masks, and per site: truth / refit prediction (nested) / shipped prediction, and the limb mask size."""
    import cv2
    H, W = 480, 640
    base = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    L = cv2.cvtColor(base, cv2.COLOR_RGB2BGR).copy()
    for i, b in enumerate(np.atleast_2d(np.asarray(boxes, float))[:, :4] if len(boxes) else []):
        x1, y1, x2, y2 = (b[:4] / 2).astype(int)
        c = (0, 220, 255) if i == pick else (160, 160, 160)
        cv2.rectangle(L, (x1, y1), (x2, y2), c, 2 if i == pick else 1)
        cv2.putText(L, f'P{i}', (x1 + 2, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    if pick is not None and len(k):
        for (x, y), sc in zip(np.asarray(k)[pick][:17] / 2, np.asarray(s)[pick][:17]):
            if sc > 0.3:
                cv2.circle(L, (int(x), int(y)), 3, (0, 220, 255), -1)
    Rm = L.copy() // 3
    cm, names = cap.get('cm'), cap.get('names', [])
    if cm is not None:
        for i, nm in enumerate(names):
            if nm in COL:
                Rm[cm == i] = (0.45 * np.array(COL[nm][::-1]) + 0.55 * L[cm == i]).astype(np.uint8)
        for msk, c in ((cap['wm'], (0, 0, 255)), (cap['tq'], (0, 140, 255))):
            m8 = (np.asarray(msk) > 0).astype(np.uint8)
            if m8.shape != (H, W):
                m8 = cv2.resize(m8, (W, H), interpolation=cv2.INTER_NEAREST)
            cs, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(Rm, cs, -1, c, 2)
        for i, nm in enumerate(names):
            if nm in COL and nm in srows:
                ys_, xs_ = np.nonzero(cm == i)
                if len(xs_):
                    cv2.putText(Rm, nm, (int(xs_.mean()) - 12, int(ys_.mean())), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    pad = np.full((150, 2 * W, 3), 255, np.uint8)
    txt = [f'R{n:04d}  people detected {len(boxes)}  chosen P{pick}  facing {frame.get("facing")}',
           'site  truth / refit (nested) / shipped   limb mask px   wound px']
    for st in ('LUE', 'RUE', 'LLE', 'RLE'):
        r = srows.get(st)
        if r is None:
            txt.append(f'{st}   (not labeled: unsure)'); continue
        flag = '  <-- ERROR, tag it' if r['error'] == '1' else ''
        txt.append(f'{st}   {r["truth"]} / {r["pred_nested"]} / {r["pred_shipped"]}   {rows[st]["vis_px"]}   {rows[st]["wound_px"]}{flag}')
    for j, t in enumerate(txt):
        cv2.putText(pad, t, (8, 20 + 21 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0) if 'ERROR' not in t else (0, 0, 200), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(od, f'R{n:04d}.png'), np.vstack([np.hstack([L, Rm]), pad]))


def summarize(out, proxy, sites, fails, secs):
    """Automatic proxies (counts only): crowding, and how often an error site has no usable limb mask at all."""
    ppl = [r['people'] for r in proxy]
    err_nomask = err = ok_nomask = ok = 0
    for r in proxy:
        for st, sr in sites[r['n']].items():
            nom = r['vis_px'].get(st, 0) < 30                      # engine's own floor (test_a_masks.MIN_PX)
            if sr['error'] == '1':
                err += 1; err_nomask += nom
            else:
                ok += 1; ok_nomask += nom
    print('R2_RENDER', json.dumps(dict(photos=len(proxy), image_failures=fails, seconds=round(secs),
                                       people_detected=dict(zero=ppl.count(0), one=ppl.count(1), two_plus=sum(p >= 2 for p in ppl)),
                                       error_sites_no_limb_mask=[err_nomask, err], correct_sites_no_limb_mask=[ok_nomask, ok])), flush=True)


# ----------------------------------------------------------------------------------------------------------- sheet
NB_SRC = r'''
import csv, os, ipywidgets as w
from IPython.display import display, Image, clear_output
D = os.path.dirname(os.path.abspath("R2_tag.ipynb"))
E = list(csv.DictReader(open(os.path.join(D, "errors.csv"))))
T = os.path.join(D, "tags.csv")
old = {(r["n"], r["site"]): r for r in csv.DictReader(open(T))} if os.path.exists(T) else {}
photos = sorted({e["n"] for e in E}, key=int)
OPTS = [("-- choose --", ""), ("S1 wrong person chosen as the casualty", "S1"), ("S2 limb mask misplaced or missing", "S2"),
        ("S3 mask correct, injury cue not captured", "S3"), ("S4 label doubtful or site ambiguous", "S4"), ("S5 other", "S5")]
state = {"i": 0}
out = w.Output()
def save():
    with open(T, "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["n", "site", "truth", "pred_nested", "tag", "note"])
        for e in E:
            o = old.get((e["n"], e["site"]), {})
            wr.writerow([e["n"], e["site"], e["truth"], e["pred_nested"], o.get("tag", ""), o.get("note", "")])
def show():
    with out:
        clear_output(wait=True)
        n = photos[state["i"]]
        done = sum(1 for e in E if old.get((e["n"], e["site"]), {}).get("tag"))
        print(f"Photo {state['i'] + 1} of {len(photos)}  (R{int(n):04d})   tagged {done} of {len(E)} errors")
        display(Image(filename=os.path.join(D, "overlays", f"R{int(n):04d}.png")))
        rows = []
        for e in [e for e in E if e["n"] == n]:
            k = (e["n"], e["site"]); o = old.setdefault(k, {"tag": "", "note": ""})
            dd = w.Dropdown(options=OPTS, value=o["tag"], description=f'{e["site"]}: {e["truth"]} -> {e["pred_nested"]}',
                            style={"description_width": "260px"}, layout=w.Layout(width="720px"))
            tx = w.Text(value=o["note"], placeholder="note (optional)", layout=w.Layout(width="320px"))
            def on(c, k=k, field="tag"): old[k][field] = c["new"]; save()
            dd.observe(on, names="value"); tx.observe(lambda c, k=k: (old[k].__setitem__("note", c["new"]), save()), names="value")
            rows.append(w.HBox([dd, tx]))
        allS1 = w.Button(description="All errors in this photo: S1 (wrong person)")
        def s1(_):
            for e in E:
                if e["n"] == n: old[(e["n"], e["site"])]["tag"] = "S1"
            save(); show()
        allS1.on_click(s1)
        display(w.VBox(rows + [allS1]))
def go(d):
    state["i"] = max(0, min(len(photos) - 1, state["i"] + d)); show()
prev, nxt = w.Button(description="< previous"), w.Button(description="next >")
prev.on_click(lambda _: go(-1)); nxt.on_click(lambda _: go(1))
save(); display(w.HBox([prev, nxt]), out); show()
'''
NB_TALLY = r'''
import subprocess, sys, os
print(subprocess.run([sys.executable, os.environ.get("R2_TOOL", "/workspace/probeB/repo_r2/tools/r2_overlays.py"), "tally", os.getcwd()],
                     capture_output=True, text=True).stdout)
'''
NB_HELP = """# R2 tagging (KP-1)

Each photo shows the engine's view (left: every person found in grey, the chosen casualty in yellow with its keypoints;
right: the part map the features are built from, limbs colored LUE red, RUE green, LLE blue, RLE yellow, torso purple,
occluder cyan; wound outline red, tourniquet outline orange). The text under the photo gives, per site, the truth, the
refit prediction and the shipped prediction. Only the sites marked ERROR need a tag.

Tags: **S1** wrong person chosen as the casualty. **S2** limb mask misplaced (on gear, background or another person) or
missing where the limb is visible. **S3** mask correct, but the injury cue is not captured. **S4** label doubtful, or the
site truly ambiguous. **S5** other (add a note).

If the casualty is wrong, the button tags every error in the photo S1. Tags save on every change to `tags.csv`; you can
stop and come back. Run the first code cell, tag, then run the last cell for the counts. Nothing leaves this volume.
"""


def stage_sheet(out):
    nb = {'cells': [{'cell_type': 'markdown', 'metadata': {}, 'source': NB_HELP},
                    {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': NB_SRC.strip()},
                    {'cell_type': 'markdown', 'metadata': {}, 'source': '## Counts for the KP-1 rule'},
                    {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': NB_TALLY.strip()}],
          'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}},
          'nbformat': 4, 'nbformat_minor': 5}
    json.dump(nb, open(os.path.join(out, 'R2_tag.ipynb'), 'w'), indent=1)
    n_err = sum(1 for _ in csv.DictReader(open(os.path.join(out, 'errors.csv'))))
    n_ph = len(os.listdir(os.path.join(out, 'overlays')))
    print('R2_SHEET', json.dumps(dict(errors_to_tag=n_err, overlays=n_ph, notebook='R2_tag.ipynb')), flush=True)


# ----------------------------------------------------------------------------------------------------------- tally
def stage_tally(out):
    T = list(csv.DictReader(open(os.path.join(out, 'tags.csv'))))
    n = len(T); tagged = [r for r in T if r['tag'] in TAGS]
    c = {t: sum(r['tag'] == t for r in tagged) for t in TAGS}
    by_truth = {}
    for r in tagged:
        by_truth.setdefault(r['truth'], {t: 0 for t in TAGS})[r['tag']] += 1
    m = max(len(tagged), 1)
    res = dict(errors=n, tagged=len(tagged), untagged=n - len(tagged), counts=c,
               share={t: round(c[t] / m, 3) for t in TAGS},
               segmenter_or_pick_S1_S2=round((c['S1'] + c['S2']) / m, 3), features_S3=round(c['S3'] / m, 3),
               by_truth=by_truth,
               rule_i_candidates={'segmenter or casualty-selection adaptation (S1+S2)': (c['S1'] + c['S2']) / m >= 0.5,
                                  'direct engine (S3)': c['S3'] / m >= 0.5})
    print('R2_TALLY', json.dumps(res, indent=1))
    print('Rule part (i) needs one named, costed fix covering at least half of the tagged errors. Part (ii) (a dev-set '
          'test before 20 Nov, target nested accuracy >= 49% with wound recall > 0) is set in the KP-1 report.')


if __name__ == '__main__':
    st, a = sys.argv[1], sys.argv[2:]
    {'errors': stage_errors, 'render': stage_render, 'sheet': stage_sheet, 'tally': stage_tally}[st](*a)
