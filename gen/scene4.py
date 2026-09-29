"""Probe B generator vNext (scene4): v3 scene, v3 outputs unchanged, plus raw physical truth (spec v1.1, section 10).

  python3 scene4.py full <params.json> <out_dir>
  python3 scene4.py labels <params.json> <out_dir>     truth products only (no RGB); used to relabel existing scenes

Writes, per scene <id>:
  <id>.jpg, <id>_part3.png, <id>_id.png          identical to scene3 (same seed and params reproduce v3 renders)
  <id>_amodal.png    per-limb full outline inside the frame, one bit per site (LUE 1, RUE 2, LLE 4, RLE 8)
  <id>_occ.png       per-pixel owner of the visible surface: 0 background, 1 floor, 2 occluder (kind in sidecar),
                     3 torso or head, 4 LUE, 5 RUE, 6 LLE, 7 RLE
  <id>_injury.png    native wound (bits 1, 2, 4, 8 per site) and tourniquet (16) surfaces, rendered without body or
                     occluders, so hidden portions are included
  <id>_sidecar.json  params, raw truth per site, terminal points, joints, provenance; legacy labels_by_threshold kept
Labels under the named rules are computed offline from raw truth (score/labels.py), not stored here as truth.
"""
import sys, os, json, math, time, hashlib
sys.path.insert(0, os.path.dirname(__file__))
import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
import manikin as mk
import scene as S
import scene2 as S2
import scene3 as S3

GEN_VERSION = 'scene4/1.0'
SITE_BIT = {'LUE': 1, 'RUE': 2, 'LLE': 4, 'RLE': 8}
OCC_CODE = {'FLOOR': 1, 'OCC': 2, 'TORSO': 3, 'LUE': 4, 'RUE': 5, 'LLE': 6, 'RLE': 7}
JOINTS = {'UE': [('shoulder', 'upperarm01'), ('elbow', 'lowerarm01'), ('wrist', 'wrist'), ('hand', 'finger3-1')],
          'LE': [('hip', 'upperleg01'), ('knee', 'lowerleg01'), ('ankle', 'foot'), ('foot', 'toe3-1')]}
REMOVED = {'disarticulation': {'elbow', 'wrist', 'hand', 'knee', 'ankle', 'foot'},
           'above_joint': {'elbow', 'wrist', 'hand', 'knee', 'ankle', 'foot'},
           'below_joint': {'wrist', 'hand', 'ankle', 'foot'},
           'distal': {'hand', 'foot'}}
DISTAL = {'UE': 'hand', 'LE': 'foot'}
W, H = S.ID_RES


def write_png(path, a):
    """8-bit greyscale PNG without extra dependencies (Blender python has numpy and zlib only)"""
    import zlib, struct
    h, w = a.shape
    raw = b''.join(b'\x00' + a[i].astype(np.uint8).tobytes() for i in range(h))
    def chunk(t, d):
        return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 0, 0, 0, 0))
                + chunk(b'IDAT', zlib.compress(raw, 9)) + chunk(b'IEND', b''))


def mask(A, rgb):
    q = np.where(A > 0.86, 1.0, np.where(A > 0.35, 0.5, 0.0))
    return np.all(q == np.array(rgb, dtype=float), axis=2)


def project(sc, cam, wpt):
    v = world_to_camera_view(sc, cam, wpt)
    return v.x * W, (1 - v.y) * H, bool(0 <= v.x < 1 and 0 <= v.y < 1 and v.z > 0)


def near(m, x, y, r=3):
    xi, yi = int(round(x)), int(round(y))
    return bool(m[max(0, yi - r):yi + r + 1, max(0, xi - r):xi + r + 1].any())


def occ_at(occ, x, y, r=1):
    xi, yi = int(round(min(max(x, 0), W - 1))), int(round(min(max(y, 0), H - 1)))
    w = occ[max(0, yi - r):yi + r + 1, max(0, xi - r):xi + r + 1].ravel()
    return int(np.bincount(w, minlength=8).argmax())


OCC_NAME = {0: 'background', 1: 'floor', 2: 'occluder', 3: 'torso_or_head', 4: 'LUE', 5: 'RUE', 6: 'LLE', 7: 'RLE'}


def main_full(prm_path, out_dir, rgb=True):
    t0 = time.time(); tm = {}
    raw = open(prm_path, 'rb').read(); prm = json.loads(raw); rng = np.random.default_rng(prm['seed'])
    sid = prm['scene_id']
    ctx = S3.setup3(prm, rng)
    sc, cam = ctx['sc'], ctx['cam']
    vt = sc.view_settings.view_transform
    tgt, d = S.place_camera(cam, ctx['P'], prm, rng)
    if rgb:                                           # render_rgb draws nothing from rng, so skipping it keeps geometry
        t = time.time(); S2.render_rgb(ctx, prm, os.path.join(out_dir, sid + '.jpg'), vt); tm['rgb'] = time.time() - t
    else:                                             # same image settings the RGB pass leaves behind (PNG stays RGB)
        sc.render.image_settings.file_format = 'JPEG'
    t = time.time()
    Ap = S3._id_pass(ctx, ctx['part_objs'], os.path.join(out_dir, sid + '_part3.png'), S3.PART3_COLORS['OCC'])
    Ai = S3._id_pass(ctx, ctx['site_objs'], os.path.join(out_dir, sid + '_id.png'), S.ID_COLORS['OCC'])
    tm['id'] = time.time() - t
    site_vis = {s: mask(Ai, S.ID_COLORS[s]) for s in mk.SITES}
    vis = {s: int(site_vis[s].sum()) for s in mk.SITES}
    wound_px = {s: S3.count(Ap, S3.PART3_COLORS[s + '_wound']) for s in mk.SITES}

    occ = np.zeros((H, W), np.uint8)
    for k, code in OCC_CODE.items():
        occ[mask(Ai, S.ID_COLORS[k])] = code
    write_png(os.path.join(out_dir, sid + '_occ.png'), occ)

    # overscan alone renders (no floor, no occluders): full outline per site, and full stump per side
    t = time.time()
    cd = cam.data; cd.sensor_width = 36 * S.OVERSCAN
    hide = [ctx['floor']] + ctx['occ']
    for o in hide:
        o.hide_render = True
    OW, OH = W * S.OVERSCAN, H * S.OVERSCAN; y0, x0 = (OH - H) // 2, (OW - W) // 2
    alone, amodal_in, amodal = {}, {}, np.zeros((H, W), np.uint8)
    for s in mk.SITES:
        objs = [o for o in ctx['site_objs'] if o['key'] == s]
        p = os.path.join(out_dir, f'_alone_{sid}_{s}.png')
        B = mask(S3._id_pass(ctx, objs, p, S.ID_COLORS['OCC'], (OW, OH)), S.ID_COLORS[s]); os.remove(p)
        alone[s] = int(B.sum()); inf = B[y0:y0 + H, x0:x0 + W]
        amodal_in[s] = int(inf.sum()); amodal[inf] |= SITE_BIT[s]
    stump_full = {s: 0 for s in mk.SITES}
    for side in sorted({s[0] for s in prm['amputations']}):
        objs = [o for o in ctx['part_objs'] if o['key'] == side + '_stump']
        if not objs:
            continue
        p = os.path.join(out_dir, f'_stump_{sid}_{side}.png')
        B = mask(S3._id_pass(ctx, objs, p, S.ID_COLORS['OCC'], (OW, OH)), S3.PART3_COLORS[side + '_stump']); os.remove(p)
        Bin = B[y0:y0 + H, x0:x0 + W]
        for s in mk.SITES:
            if s[0] == side and s in prm['amputations']:
                stump_full[s] = int((Bin & ((amodal & SITE_BIT[s]) > 0)).sum())
    cd.sensor_width = 36
    # native injury surfaces inside the frame: wounds and tourniquets alone
    wobjs = [o for o in ctx['part_objs'] if o['key'].endswith('_wound') or o['key'] == 'TQ']
    injury = np.zeros((H, W), np.uint8)
    wound_native, tq_native = {s: 0 for s in mk.SITES}, {s: 0 for s in mk.SITES}
    if wobjs:
        p = os.path.join(out_dir, f'_inj_{sid}.png')
        I = S3._id_pass(ctx, wobjs, p, S.ID_COLORS['OCC']); os.remove(p)
        tqm = mask(I, S3.PART3_COLORS['TQ']); injury[tqm] |= 16
        for s in mk.SITES:
            wm = mask(I, S3.PART3_COLORS[s + '_wound']); injury[wm] |= SITE_BIT[s]
            wound_native[s] = int(wm.sum()); tq_native[s] = int((tqm & ((amodal & SITE_BIT[s]) > 0)).sum())
    for o in hide:
        o.hide_render = False
    write_png(os.path.join(out_dir, sid + '_amodal.png'), amodal)
    write_png(os.path.join(out_dir, sid + '_injury.png'), injury)
    tm['alone'] = time.time() - t

    # visible part pixels per site
    def part_vis(cls, s):
        return int((mask(Ap, S3.PART3_COLORS[cls]) & site_vis[s]).sum())
    stump_vis = {s: part_vis(s[0] + '_stump', s) for s in mk.SITES}
    distal_vis = {s: part_vis(s[0] + '_' + DISTAL[s[1:]], s) for s in mk.SITES}
    tq_vis = {s: int((mask(Ap, S3.PART3_COLORS['TQ']) & site_vis[s]).sum()) for s in mk.SITES}

    # joints and terminal points (posed skeleton projected through the scene camera)
    arm = [o for o in bpy.data.objects if o.type == 'ARMATURE'][0]
    bpy.context.view_layer.update()
    joints, terminal = {}, {}
    ev = ctx['ob'].evaluated_get(bpy.context.evaluated_depsgraph_get())
    for s in mk.SITES:
        side, limb = s[0], s[1:]; removed = REMOVED.get(prm['amputations'].get(s), set())
        js = {}
        for jn, bone in JOINTS[limb]:
            x, y, inf = project(sc, cam, arm.matrix_world @ arm.pose.bones[bone + '.' + side].head)
            ex = jn not in removed
            js[jn] = {'x': round(x, 1), 'y': round(y, 1), 'exists': ex, 'in_frame': inf,
                      'visible': bool(ex and inf and near(site_vis[s], x, y, 2)), 'owner': OCC_NAME[occ_at(occ, x, y)] if inf else 'out_of_frame'}
        joints[s] = js
        amputated = s in prm['amputations']
        if amputated:     # centre of the remaining stump cap (posed), from stump-labelled faces of this site
            pts = [ev.matrix_world @ ev.data.vertices[v].co for po, lab in zip(ctx['ob'].data.polygons, ctx['fpart'])
                   if lab == side + '_stump' and ctx['fsite'][po.index] == s for v in po.vertices] if 'fpart' in ctx else []
            if pts:
                c = sum(pts, pts[0] * 0) / len(pts)
            else:
                last = [jn for jn, _ in JOINTS[limb] if jn not in removed][-1]
                c = arm.matrix_world @ arm.pose.bones[dict(JOINTS[limb])[last] + '.' + side].head
        else:
            c = arm.matrix_world @ arm.pose.bones[dict(JOINTS[limb])[DISTAL[limb]] + '.' + side].head
        x, y, inf = project(sc, cam, c)
        if not inf:
            cause, owner = 'out_of_frame', 'out_of_frame'
        elif near(site_vis[s], x, y, 3):
            cause, owner = ('amputated_visible' if amputated else 'intact_visible'), OCC_NAME[occ_at(occ, x, y)]
        else:
            owner = OCC_NAME[occ_at(occ, x, y)]
            cause = 'amputated_hidden' if amputated else 'occluded'
        terminal[s] = {'x': round(x, 1), 'y': round(y, 1), 'in_frame': inf, 'amputated': amputated, 'cause': cause,
                       'owner_at_point': owner, 'occluder_kind': prm.get('occluder') if owner == 'occluder' else None}

    # share of wound faces lying inside the tourniquet band (rest pose): the wound is under the tourniquet
    import fidelity as FD
    tq_cover = {s: None for s in mk.SITES}
    for s, band in FD.TQ_BANDS.items():
        wf = [po for po, lab in zip(ctx['ob'].data.polygons, ctx['fpart']) if lab == s + '_wound']
        if wf:
            a, b = mk.chain_points(arm, s)[0], mk.chain_points(arm, s)[1]
            inside = [abs(FD._seg_dist(po.center, a, b)[1] - band['t0']) * (b - a).length < band['w'] / 2 for po in wf]
            tq_cover[s] = round(sum(inside) / len(inside), 3)
    frac = {s: (vis[s] / alone[s] if alone[s] else 0.0) for s in mk.SITES}
    labels = {}
    for th in S3.THRESHOLDS:
        lab = {}
        for s in mk.SITES:
            visible = frac[s] > 0 if th == 0 else frac[s] >= th
            lab[s] = ('not_testable' if not visible else 'amputation' if s in prm['amputations'] else
                      ('wound' if wound_px[s] >= S3.WOUND_MIN_PX else 'no_injury') if s in ctx['wounds'] else 'no_injury')
        labels[f'{th:.2f}'] = lab
    prm.pop('_head_xy', None)
    prm.pop('_limb_xyz', None)
    truth = {s: {'visible_px': vis[s], 'full_px': alone[s], 'full_in_frame_px': amodal_in[s],
                 'full_out_of_frame_px': alone[s] - amodal_in[s], 'visible_fraction': round(frac[s], 4),
                 'wound_present': s in ctx['wounds'], 'wound_visible_px': wound_px[s], 'wound_native_px': wound_native[s],
                 'amputated': s in prm['amputations'], 'amputation_level': prm['amputations'].get(s),
                 'stump_visible_px': stump_vis[s], 'stump_full_px': stump_full[s], 'distal_visible_px': distal_vis[s],
                 'tourniquet': s in prm.get('tourniquets', []), 'tq_visible_px': tq_vis[s], 'tq_native_px': tq_native[s],
                 'wound_under_tourniquet_frac': tq_cover[s]}
             for s in mk.SITES}
    side = {'scene_id': sid, 'params': prm, 'truth': truth, 'terminal': terminal, 'joints': joints,
            'occluder': prm.get('occluder'), 'occ_target': prm.get('occ_target'), 'wounds': ctx['wounds'],
            'pair': prm.get('pair'), 'confusers': prm.get('confusers', []),
            'visible_px': vis, 'alone_px': alone, 'visible_fraction': {s: round(frac[s], 4) for s in mk.SITES},
            'wound_visible_px': wound_px, 'labels_by_threshold': labels, 'headline_threshold': 0.10,
            'camera': {'location': list(cam.location), 'target': [float(x) for x in tgt], 'distance': d},
            'provenance': {'generator': GEN_VERSION, 'commit': os.environ.get('GEN_COMMIT', 'unknown'),
                           'params_sha256': hashlib.sha256(raw).hexdigest(), 'blender': bpy.app.version_string,
                           'device': os.environ.get('CYCLES_DEVICE') or 'CPU', 'pythonhashseed': os.environ.get('PYTHONHASHSEED'), 'threads': os.cpu_count(),
                           'host': os.environ.get('RUNPOD_POD_ID') or os.uname().nodename,
                           'time_s': {k: round(v, 2) for k, v in tm.items()}},
            'total_s': round(time.time() - t0, 1)}
    json.dump(side, open(os.path.join(out_dir, sid + '_sidecar.json'), 'w'), indent=1)
    print('DONE', sid, json.dumps({s: terminal[s]['cause'] for s in mk.SITES}), side['total_s'], flush=True)


if __name__ == '__main__':
    if os.environ.get('PYTHONHASHSEED') != '0':      # set iteration order feeds geometry; v3 renders were not reproducible
        os.execve(sys.executable, [sys.executable] + sys.argv, dict(os.environ, PYTHONHASHSEED='0'))
    a = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    main_full(a[1], a[2], rgb=(a[0] != 'labels'))
