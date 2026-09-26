"""Probe B generator v3: wounds, clothing, tourniquets, 4-class labels (Blender env).

  python3 scene3.py train <params.json> <out_dir> [views=3]   RGB + part3 map per view (training)
  python3 scene3.py full  <params.json> <out_dir>             RGB + part3 + site-ID pass + overscan alone renders
                                                              + 4-class site labels (dev / test sets)
Part-label v3 = v2 classes + LUE/RUE/LLE/RLE_wound + TQ. Garment faces carry the label of the body part beneath.
Site label (per threshold t on visible fraction; visible fraction counts garment and tourniquet pixels of the site):
  not visible -> not_testable; amputated -> amputation; wound with >= WOUND_MIN_PX visible wound pixels (640x480)
  -> wound; otherwise no_injury. The alternative "wound present regardless of visibility" is also stored.
"""
import sys, os, json, math, time
sys.path.insert(0, os.path.dirname(__file__))
import bpy, bmesh
import numpy as np
import manikin as mk
import scene as S
import scene2 as S2
import fidelity as FD

PART3_CLASSES = S2.PART2_CLASSES + [s + '_wound' for s in mk.SITES] + ['TQ']
_EXTRA = [(1, 0.5, 1), (0.5, 1, 1), (1, 1, 0.5), (0.5, 0.5, 1), (1, 0.5, 0.5)]
PART3_COLORS = dict(zip(PART3_CLASSES, S2._G2 + _EXTRA))
LABEL_ID = {c: i for i, c in enumerate(PART3_CLASSES)}
SITE_NAMES = ['TORSO'] + mk.SITES
WOUND_MIN_PX = 20
THRESHOLDS = [0.0, 0.10, 0.25, 0.50]


def _split(mesh_posed, face_keys, keys, colors, prefix, sc):
    objs = []
    for k in keys:
        idx = [i for i, fk in enumerate(face_keys) if fk == k]
        if not idx:
            continue
        keep = set(idx)
        bm = bmesh.new(); bm.from_mesh(mesh_posed); bm.faces.ensure_lookup_table()
        bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.index not in keep], context='FACES')
        me = bpy.data.meshes.new(prefix + str(k)); bm.to_mesh(me); bm.free()
        o = bpy.data.objects.new(prefix + str(k), me); sc.collection.objects.link(o)
        m = bpy.data.materials.new(prefix + 'm_' + str(k)); m.diffuse_color = (*colors[k], 1); me.materials.append(m)
        o.hide_render = True; o['key'] = str(k)
        objs.append(o)
    return objs


def setup3(prm, rng):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    if sc.world is None:
        sc.world = bpy.data.worlds.new('world')
    ob, arm, jpos, skel = mk.build()
    for site, lvl in prm['amputations'].items():
        mk.amputate(ob, arm, site, lvl)
    fsite = S.face_sites(ob)
    fpart = S2.face_parts2(ob, arm, fsite, S.face_parts(ob, fsite), prm)
    mk.limb_pose(arm, prm['limb_pose'])
    mk.body_position(arm, prm['body_position'])
    arm.rotation_euler[2] += math.radians(prm['body_yaw'])
    bpy.context.view_layer.update()
    mk.ground(ob, arm)
    skin = mk.plastic_material('skin', S.SKIN_TONES[prm['skin']])
    ob.data.materials.append(skin)
    wounds = FD.add_wounds(ob, arm, fsite, fpart, prm, rng)
    shells = FD.add_garments(ob, arm, fsite, fpart, prm, rng, LABEL_ID) + FD.add_tourniquets(ob, arm, fsite, fpart, prm, rng, LABEL_ID)
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    P = np.array([tuple(ev.matrix_world @ v.co) for v in ev.data.vertices])
    head = arm.matrix_world @ arm.pose.bones['head'].head
    prm['_head_xy'] = [head.x, head.y]
    fl = S.FLOORS[prm['floor']]
    bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 0, -0.004)); floor = bpy.context.object
    floor.data.materials.append(S.noise_material('floor', *fl))
    Pocc = P
    if prm.get('occ_target') == 'limb_end' and prm['occluder'] in ('gear_bag', 'medic_arm'):
        ends = {'hand', 'foot', 'stump', 'forearm', 'shank'}
        vi = sorted({v for p_, fp in zip(ob.data.polygons, fpart) if fp[2:] in ends for v in p_.vertices})
        if vi:
            Pocc = P[vi]
    occ = S.add_occluder(prm['occluder'], Pocc, rng, (P.min(0), P.max(0)))
    occ_mat = S.noise_material('occ', (0.25, 0.27, 0.18), (0.16, 0.18, 0.11), 12)
    for o in occ:
        o.data.materials.append(occ_mat)
    sun = bpy.data.objects.new('sun', bpy.data.lights.new('sun', 'SUN')); sc.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(prm['sun_el']), 0, math.radians(prm['sun_az']))
    sun.data.energy = {'indoor_flat': 1.5, 'outdoor_sun': 4.0, 'low_light': 0.9}[prm['lighting']]
    sun.data.angle = math.radians(12 if prm['lighting'] == 'indoor_flat' else 1.5)
    sc.world.use_nodes = True
    bg = sc.world.node_tree.nodes['Background']
    bg.inputs['Strength'].default_value = {'indoor_flat': 0.9, 'outdoor_sun': 0.6, 'low_light': 0.15}[prm['lighting']]
    bg.inputs['Color'].default_value = (0.75, 0.8, 0.9, 1)
    cd = bpy.data.cameras.new('cam'); cd.lens = 35; cd.sensor_width = 36
    cam = bpy.data.objects.new('cam', cd); sc.collection.objects.link(cam); sc.camera = cam
    # label objects: part classes and sites, from the posed body and each posed shell
    part_objs, site_objs = [], []
    sources = [(ob, fpart, fsite)]
    for g in shells:
        a3 = g.data.attributes['p3'].data; ast = g.data.attributes['st'].data
        sources.append((g, [PART3_CLASSES[d.value] for d in a3], [SITE_NAMES[d.value] for d in ast]))
    for n, (o, fp, fs) in enumerate(sources):
        posed = bpy.data.meshes.new_from_object(o.evaluated_get(dg)); posed.transform(o.matrix_world)
        part_objs += _split(posed, fp, PART3_CLASSES, PART3_COLORS, f'p3_{n}_', sc)
        site_objs += _split(posed, fs, SITE_NAMES, S.ID_COLORS, f'id_{n}_', sc)
    return dict(sc=sc, ob=ob, shells=shells, P=P, floor=floor, occ=occ, cam=cam, part_objs=part_objs,
                site_objs=site_objs, wounds=wounds)


def _id_pass(ctx, show, path, occ_color, res=S.ID_RES):
    sc = ctx['sc']
    saved = [(o, list(o.data.materials)) for o in [ctx['floor']] + ctx['occ']]
    fm = bpy.data.materials.new('idFLOOR'); fm.diffuse_color = (*S.ID_COLORS['FLOOR'], 1)
    om = bpy.data.materials.new('idOCC'); om.diffuse_color = (*occ_color, 1)
    ctx['floor'].data.materials.clear(); ctx['floor'].data.materials.append(fm)
    for o in ctx['occ']:
        o.data.materials.clear(); o.data.materials.append(om)
    for o in [ctx['ob']] + ctx['shells']:
        o.hide_render = True
    for o in show:
        o.hide_render = False
    S.set_id_look(sc)
    sc.render.image_settings.file_format = 'PNG'
    sc.render.resolution_x, sc.render.resolution_y = res
    A = S.render_to_array(sc, path)
    for o in show:
        o.hide_render = True
    for o in [ctx['ob']] + ctx['shells']:
        o.hide_render = False
    for o, mats in saved:
        o.data.materials.clear()
        for m in mats:
            o.data.materials.append(m)
    return A


def main_train(prm_path, out_dir, views):
    t0 = time.time()
    prm = json.load(open(prm_path)); rng = np.random.default_rng(prm['seed'])
    sid = prm['scene_id']
    ctx = setup3(prm, rng)
    vt = ctx['sc'].view_settings.view_transform
    vrng = np.random.default_rng(prm['seed'] + 7919)
    done = []; used_az = [prm['azimuth']]
    for k in range(views):
        v = 'abcdefgh'[k]; p = dict(prm)
        if k > 0:
            for _ in range(50):
                az = float(vrng.choice(np.arange(0, 360, 45)) + vrng.uniform(-15, 15))
                if all(abs((az - u + 180) % 360 - 180) >= 60 for u in used_az):
                    break
            p['azimuth'] = az; p['elevation'] = float(vrng.choice(S2.ELEVS)); p['framing'] = str(vrng.choice(S2.FRAMINGS))
        used_az.append(p['azimuth'])
        tgt, d = S.place_camera(ctx['cam'], ctx['P'], p, rng)
        S2.render_rgb(ctx, p, os.path.join(out_dir, sid + v + '.jpg'), vt)
        _id_pass(ctx, ctx['part_objs'], os.path.join(out_dir, sid + v + '_part3.png'), PART3_COLORS['OCC'])
        q = {k_: val for k_, val in p.items() if not k_.startswith('_')}
        q.update(scene_id=sid + v, base_scene=sid, view=v)
        json.dump({'scene_id': sid + v, 'params': q, 'train_only': True, 'wounds': ctx['wounds'],
                   'camera': {'location': list(ctx['cam'].location), 'target': [float(x) for x in tgt], 'distance': d}},
                  open(os.path.join(out_dir, sid + v + '_sidecar.json'), 'w'), indent=1)
        done.append(sid + v)
    json.dump({'scene_id': sid, 'views': done, 'total_s': round(time.time() - t0, 1)}, open(os.path.join(out_dir, sid + '_views.json'), 'w'))
    print('VIEWSDONE', sid, len(done), round(time.time() - t0, 1))


def count(A, rgb):
    """pixels of an ID colour; channel levels 0 / 0.5 / 1 are written as 0 / 188 / 255 (sRGB), so quantise first"""
    q = np.where(A > 0.86, 1.0, np.where(A > 0.35, 0.5, 0.0))
    return int(np.all(q == np.array(rgb, dtype=float), axis=2).sum())


def main_full(prm_path, out_dir):
    t0 = time.time()
    prm = json.load(open(prm_path)); rng = np.random.default_rng(prm['seed'])
    sid = prm['scene_id']
    ctx = setup3(prm, rng)
    vt = ctx['sc'].view_settings.view_transform
    tgt, d = S.place_camera(ctx['cam'], ctx['P'], prm, rng)
    S2.render_rgb(ctx, prm, os.path.join(out_dir, sid + '.jpg'), vt)
    Ap = _id_pass(ctx, ctx['part_objs'], os.path.join(out_dir, sid + '_part3.png'), PART3_COLORS['OCC'])
    Ai = _id_pass(ctx, ctx['site_objs'], os.path.join(out_dir, sid + '_id.png'), S.ID_COLORS['OCC'])
    vis = {s: count(Ai, S.ID_COLORS[s]) for s in mk.SITES}
    wound_px = {s: count(Ap, PART3_COLORS[s + '_wound']) for s in mk.SITES}
    # overscan alone renders of each site (body + garment + tourniquet faces of the site), no floor or occluders
    cd = ctx['cam'].data; cd.sensor_width = 36 * S.OVERSCAN
    alone = {}
    hide = [ctx['floor']] + ctx['occ']
    for o in hide:
        o.hide_render = True
    for s in mk.SITES:
        objs = [o for o in ctx['site_objs'] if o['key'] == s]
        B = _id_pass(ctx, objs, os.path.join(out_dir, f'_alone_{sid}_{s}.png'), S.ID_COLORS['OCC'],
                     (S.ID_RES[0] * S.OVERSCAN, S.ID_RES[1] * S.OVERSCAN))
        alone[s] = count(B, S.ID_COLORS[s]); os.remove(os.path.join(out_dir, f'_alone_{sid}_{s}.png'))
    for o in hide:
        o.hide_render = False
    cd.sensor_width = 36
    frac = {s: (vis[s] / alone[s] if alone[s] else 0.0) for s in mk.SITES}
    labels, labels_wpresent = {}, {}
    for t in THRESHOLDS:
        lab, lab2 = {}, {}
        for s in mk.SITES:
            visible = frac[s] > 0 if t == 0 else frac[s] >= t
            if not visible:
                lab[s] = lab2[s] = 'not_testable'
            elif s in prm['amputations']:
                lab[s] = lab2[s] = 'amputation'
            elif s in ctx['wounds']:
                lab[s] = 'wound' if wound_px[s] >= WOUND_MIN_PX else 'no_injury'; lab2[s] = 'wound'
            else:
                lab[s] = lab2[s] = 'no_injury'
        labels[f'{t:.2f}'] = lab; labels_wpresent[f'{t:.2f}'] = lab2
    prm.pop('_head_xy', None)
    side = {'scene_id': sid, 'params': prm, 'visible_px': vis, 'alone_px': alone,
            'visible_fraction': {s: round(frac[s], 4) for s in mk.SITES}, 'wound_visible_px': wound_px,
            'wounds': ctx['wounds'], 'labels_by_threshold': labels, 'labels_wound_if_present': labels_wpresent,
            'headline_threshold': 0.10,
            'camera': {'location': list(ctx['cam'].location), 'target': [float(x) for x in tgt], 'distance': d},
            'total_s': round(time.time() - t0, 1)}
    json.dump(side, open(os.path.join(out_dir, sid + '_sidecar.json'), 'w'), indent=1)
    print('DONE', sid, json.dumps(side['visible_fraction']), json.dumps(labels['0.10']), side['total_s'])


if __name__ == '__main__':
    a = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    if a[0] == 'full':
        main_full(a[1], a[2])
    else:
        main_train(a[1], a[2], int(a[3]) if len(a) > 3 else 3)
