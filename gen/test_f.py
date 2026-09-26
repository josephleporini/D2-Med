"""Test F: two-stage pipeline. Stage 1 = v2 part segmenter + side rule v2 (Test E2); stage 2 = limb-end classifier.
Usage: SEG=<seg model> END=<end model> OUT=<prefix> python test_f.py <scene_dir> [...]   (pose env)

Decision rule F (declared before scoring; no thresholds tuned on the test set):
  limb visible pixels < 30                   -> not_testable
  end-window class = 'absent' (2-class model, p >= 0.5) or 'stump visible' (3-class model, argmax) -> amputation
  else Lfrac >= 0.10                         -> no_injury,  else not_testable        (the A2 visibility logic)
Arms: F0 true parts + true frame, F1 predicted parts + true frame, F2h predicted parts + hybrid frame
      (RTMW torso axis + segmentation facing). A2 on the same class maps is reported alongside.
Also: window-level AUC of the limb-end classifier on F0 and F1 windows (target = site amputated).
"""
import sys, os, json, glob, collections
import numpy as np, cv2, torch
sys.path.insert(0, os.path.dirname(__file__))
import parts as PT, test_a_masks as TA, test_e as TE, test_e2 as T2, lend as LE
from seg_train2 import Decoder2
from lend_train import auc

torch.set_num_threads(2)
SITES = TE.SITES; C = TE.C


if __name__ == '__main__':
    dirs = sys.argv[1:]; OUT = os.environ.get('OUT', 'testF')
    seg_net = Decoder2(); seg_net.load_state_dict(torch.load(os.path.join(LE.M, os.environ.get('SEG', 'seg2_n297') + '.pt'))); seg_net.eval()
    NOUT = int(os.environ.get('NOUT', 2))
    end_net = LE.EndNet(NOUT); end_net.load_state_dict(torch.load(os.path.join(LE.M, os.environ.get('END', 'lend1') + '.pt'))); end_net.eval()
    P = LE.load_sam()
    scenes = [(json.load(open(f))['scene_id'], D) for D in dirs for f in sorted(glob.glob(os.path.join(D, 'C*_sidecar.json')))
              if os.path.exists(f.replace('_sidecar.json', '_f2.npz'))]
    ARMS = ['F0', 'F1', 'F2h']
    cms = {(a, r): np.zeros((3, 3), int) for a in ARMS for r in ('F', 'A2')}
    win = collections.defaultdict(lambda: ([], [])); rows = []; swaps = collections.Counter(); judged = collections.Counter()
    for sid, D in scenes:
        sc = json.load(open(os.path.join(D, sid + '_sidecar.json'))); amps = sc['params']['amputations']
        img = cv2.cvtColor(cv2.imread(os.path.join(D, sid + '.jpg')), cv2.COLOR_BGR2RGB)
        gseg, _ = PT.decode2(os.path.join(D, sid + '_part2.png'))
        gseg = cv2.resize(gseg, (640, 480), interpolation=cv2.INTER_NEAREST)
        pseg = T2.predict(seg_net, os.path.join(D, sid + '_f2.npz'))
        gf = json.load(open(os.path.join(D, sid + '_gtframe.json'))); gl, ga = np.array(gf['left_dir']), np.array(gf['axis'])
        kk = np.load(os.path.join(D, sid + '_rtmw133.npz'))['k']
        r_axis = TE.unit((kk[5] + kk[6]) / 2 - (kk[11] + kk[12]) / 2)
        sfr = T2.seg_frame(pseg); h_left = (sfr[2] if sfr else 1) * TE.perp(r_axis)
        gcm, gnames = TA.classmap(os.path.join(D, sid + '_id.png')); gix = {n: i for i, n in enumerate(gnames)}
        cache = {}
        for arm, seg, ld, ax in (('F0', gseg, gl, ga), ('F1', pseg, gl, ga), ('F2h', pseg, h_left, r_axis)):
            cm, names, ext, stc = T2.to_classmap(seg, ld, ax); ix = {n: i for i, n in enumerate(names)}
            an = TA.analyse(cm, names); torso = cm == ix['TORSO']; s0 = LE.window_size(torso)
            for site in SITES:
                r = an[site]; gt = sc['labels_by_threshold']['0.10'][site]; p_abs = None; pvec = None
                pm = cm == ix[site]; gm = gcm == gix[site]
                if gm.sum() >= 30 and pm.sum() >= 30:
                    judged[arm] += 1; swaps[arm] += int((pm & (gcm == gix[T2.SWAP[site]])).sum() > (pm & gm).sum())
                if r['vis_px'] >= 30:
                    x, y = LE.limb_end(pm, torso)
                    key = (x, y, s0)
                    if key not in cache:
                        cache[key] = LE.encode(P, LE.crop_1280(img, x, y, s0))
                    xin = torch.cat([torch.from_numpy(cache[key].astype(np.float32))[None],
                                     torch.from_numpy(LE.mask_window(pm, x, y, s0))[None, None]], 1)
                    with torch.no_grad():
                        pv = torch.softmax(end_net(xin), 1)[0]
                        p_abs = float(pv[1]); is_abs = int(pv.argmax()) == 1; pvec = [round(float(q), 4) for q in pv]
                    if arm in ('F0', 'F1'):
                        win[arm][0].append(int(site in amps)); win[arm][1].append(p_abs)
                if r['vis_px'] < 30:
                    dF = 'not_testable'
                elif is_abs:
                    dF = 'amputation'
                else:
                    dF = 'no_injury' if r['Lfrac'] >= 0.10 else 'not_testable'
                dA = TE.decide_a2(r, ext[site] >= T2.MIN_EXT)
                cms[(arm, 'F')][C.index(gt), C.index(dF)] += 1; cms[(arm, 'A2')][C.index(gt), C.index(dA)] += 1
                rows.append({'arm': arm, 'scene': sid, 'site': site, 'gt': gt, 'F': dF, 'A2': dA, 'p_absent': p_abs,
                             'amp_level': amps.get(site), 'occluder': sc['params']['occluder'],
                             'position': sc['params']['body_position'], 'vis_px': r['vis_px'], 'Lfrac': r['Lfrac'],
                             'p': pvec, 'bg_frac': r.get('bg_frac'), 'end': r.get('end', {}), 'ext_px': ext[site],
                             'stump_px': stc[site], 'torso_ext': round(float(TA.extent(torso)), 1), 's0': s0,
                             'vis_frac_true': sc['visible_fraction'][site]})
        print(sid, flush=True)
    res = {'n_scenes': len(scenes), 'window_auc': {a: round(auc(*win[a]), 3) for a in win},
           'window_n': {a: [len(win[a][0]), int(sum(win[a][0]))] for a in win}}
    for (arm, r), m in cms.items():
        res[f'{arm}_{r}'] = TE.summarise(m); res[f'{arm}_{r}']['left_right_swaps'] = [swaps[arm], judged[arm]]
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(dirs[0], OUT + '_results.json'), 'w'), indent=1)
    json.dump(rows, open(os.path.join(dirs[0], OUT + '_sites.json'), 'w'), indent=1, default=str)
