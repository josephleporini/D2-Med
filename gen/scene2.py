"""Probe B generator, part-label v2 and multi-view training mode (Blender env).

  python3 scene2.py label <params.json> <out_dir>     part2 map only for an existing scene (same camera as scene.py)
  python3 scene2.py train <params.json> <out_dir> [views=3]
        one posed body, occluder and lighting; view 'a' uses the params' camera, views 'b', 'c' ... resample
        azimuth / elevation / framing. Writes <sid><v>.jpg, <sid><v>_part2.png, <sid><v>_sidecar.json (train-only,
        no site labels, no alone renders) and <sid>_views.json when all views are done.

Part-label v2 classes (19 + background):
  TORSO_F / TORSO_B, HEAD_F / HEAD_B   front vs back surface of the trunk and head (rest-pose normal vs body forward)
  OCC
  L_/R_ upper_arm, forearm, hand, thigh, shank, foot   ('hand'/'foot' = intact extremity; distal remnant -> proximal)
  L_/R_ stump                                           amputation cap + the last 4 cm of remaining limb
"""
import sys, os, json, math, time
sys.path.insert(0, os.path.dirname(__file__))
import bpy, bmesh
import numpy as np
from mathutils import Vector
import manikin as mk
import scene as S

PART2_CLASSES = ['TORSO_F', 'TORSO_B', 'HEAD_F', 'HEAD_B', 'OCC'] + \
    [sd + '_' + p for sd in 'LR' for p in S.PART_NAMES + ['stump']]
_G2 = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1), (0.5, 0, 0), (0, 0.5, 0),
       (0, 0, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5), (1, 0.5, 0), (1, 0, 0.5), (0.5, 1, 0), (0, 1, 0.5),
       (0.5, 0, 1), (0, 0.5, 1)]
PART2_COLORS = dict(zip(PART2_CLASSES, _G2))
FRAMINGS = (['full'] * 6 + ['upper_crop', 'lower_crop', 'off_center', 'off_center'])
ELEVS = [85, 60, 45, 30, 15]


STUMP_BAND = 0.04   # m of remaining limb proximal to the cut, labelled stump together with the cap


def face_parts2(ob, arm, fsite, fpart, prm):
    """v2 part per face (rest pose, after amputation)."""
    fpart = list(fpart)
    for site, lvl in prm['amputations'].items():
        if lvl == 'distal':
            ext, prox = ('hand', 'forearm') if site[1] == 'U' else ('foot', 'shank')
            fpart = [site[0] + '_' + prox if fp == site[0] + '_' + ext else fp for fp in fpart]
    # body forward in rest pose: MakeHuman faces +Z (MH) -> -Y (Blender)
    fwd = Vector((0, -1, 0))
    cap = ob.data.attributes.get('cap')
    capv = [d.value for d in cap.data] if cap is not None else [0] * len(ob.data.polygons)
    cuts = {}
    for site, lvl in prm['amputations'].items():
        seg, frac = mk.AMP_LEVELS[lvl]
        pts = mk.chain_points(arm, site); a, b = pts[seg], pts[seg + 1]
        cuts[site] = (pts[0] if lvl == 'disarticulation' else a.lerp(b, frac), (b - a).normalized())
    out = []
    for p, fp, s, c in zip(ob.data.polygons, fpart, fsite, capv):
        band = s in cuts and (p.center - cuts[s][0]).dot(cuts[s][1]) > -STUMP_BAND
        if (c or band) and s in mk.SITES:
            out.append(s[0] + '_stump')
        elif fp in ('TORSO', 'HEAD'):
            out.append(fp + ('_F' if p.normal.dot(fwd) > 0 else '_B'))
        else:
            out.append(fp)
    return out


def setup(prm, rng):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    if sc.world is None:
        sc.world = bpy.data.worlds.new('world')
    ob, arm, jpos, skel = mk.build()
    for site, lvl in prm['amputations'].items():
        mk.amputate(ob, arm, site, lvl)
    fsite = S.face_sites(ob)
    fpart = S.face_parts(ob, fsite)
    fpart2 = face_parts2(ob, arm, fsite, fpart, prm)
    mk.limb_pose(arm, prm['limb_pose'])
    mk.body_position(arm, prm['body_position'])
    arm.rotation_euler[2] += math.radians(prm['body_yaw'])
    bpy.context.view_layer.update()
    mk.ground(ob, arm)
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    P = np.array([tuple(ev.matrix_world @ v.co) for v in ev.data.vertices])
    head = arm.matrix_world @ arm.pose.bones['head'].head
    prm['_head_xy'] = [head.x, head.y]
    skin = mk.plastic_material('skin', S.SKIN_TONES[prm['skin']])
    ob.data.materials.append(skin)
    fl = S.FLOORS[prm['floor']]
    bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 0, -0.004)); floor = bpy.context.object
    floor.data.materials.append(S.noise_material('floor', *fl))
    Pocc = P
    if prm.get('occ_target') == 'limb_end' and prm['occluder'] in ('gear_bag', 'medic_arm'):
        # place the bag / reaching arm over a limb end (hand, foot, stump or distal segment) instead of anywhere
        ends = {'hand', 'foot', 'stump', 'forearm', 'shank'}
        vi = sorted({v for p_, fp in zip(ob.data.polygons, fpart2) if fp[2:] in ends for v in p_.vertices})
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
    # posed copy split into v2 part objects (hidden until the label pass)
    posed = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    posed.transform(ob.matrix_world)
    part_objs = []
    for pc in PART2_CLASSES:
        if pc == 'OCC':
            continue
        bm = bmesh.new(); bm.from_mesh(posed); bm.faces.ensure_lookup_table()
        bmesh.ops.delete(bm, geom=[f for f in bm.faces if fpart2[f.index] != pc], context='FACES')
        me = bpy.data.meshes.new('p2_' + pc); bm.to_mesh(me); bm.free()
        o = bpy.data.objects.new('p2_' + pc, me); sc.collection.objects.link(o)
        m = bpy.data.materials.new('p2m_' + pc); m.diffuse_color = (*PART2_COLORS[pc], 1); me.materials.append(m)
        o.hide_render = True
        part_objs.append(o)
    return dict(sc=sc, ob=ob, P=P, floor=floor, occ=occ, cam=cam, part_objs=part_objs)


def render_rgb(ctx, prm, path, vt):
    sc = ctx['sc']
    sc.render.engine = 'CYCLES'; sc.cycles.samples = prm.get('samples', 16)
    gpu = os.environ.get('CYCLES_DEVICE', '')          # 'OPTIX' or 'CUDA' on a rented GPU; empty = CPU
    if gpu:
        pr = bpy.context.preferences.addons['cycles'].preferences
        pr.compute_device_type = gpu; pr.get_devices()
        for d in pr.devices:
            d.use = d.type == gpu
        sc.cycles.device = 'GPU'
    else:
        sc.cycles.device = 'CPU'
    try:
        sc.cycles.use_denoising = True
    except Exception:
        pass
    sc.view_settings.view_transform = vt
    sc.render.resolution_x, sc.render.resolution_y = 1280, 960
    sc.render.image_settings.file_format = 'JPEG'; sc.render.image_settings.quality = 92
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)


def render_part2(ctx, path):
    sc = ctx['sc']
    saved = [(o, list(o.data.materials)) for o in [ctx['floor']] + ctx['occ']]
    fm = bpy.data.materials.get('p2m_FLOOR') or bpy.data.materials.new('p2m_FLOOR')
    fm.diffuse_color = (*S.ID_COLORS['FLOOR'], 1)
    om = bpy.data.materials.get('p2m_OCC') or bpy.data.materials.new('p2m_OCC')
    om.diffuse_color = (*PART2_COLORS['OCC'], 1)
    ctx['floor'].data.materials.clear(); ctx['floor'].data.materials.append(fm)
    for o in ctx['occ']:
        o.data.materials.clear(); o.data.materials.append(om)
    ctx['ob'].hide_render = True
    for o in ctx['part_objs']:
        o.hide_render = False
    S.set_id_look(sc)
    sc.render.image_settings.file_format = 'PNG'
    sc.render.resolution_x, sc.render.resolution_y = S.ID_RES
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    for o in ctx['part_objs']:
        o.hide_render = True
    ctx['ob'].hide_render = False
    for o, mats in saved:
        o.data.materials.clear()
        for m in mats:
            o.data.materials.append(m)


def main_label(prm_path, out_dir):
    t0 = time.time()
    prm = json.load(open(prm_path)); rng = np.random.default_rng(prm['seed'])
    ctx = setup(prm, rng)
    S.place_camera(ctx['cam'], ctx['P'], prm, rng)       # same rng sequence as scene.py -> same camera
    render_part2(ctx, os.path.join(out_dir, prm['scene_id'] + '_part2.png'))
    print('PART2DONE', prm['scene_id'], round(time.time() - t0, 1))


def main_train(prm_path, out_dir, views):
    t0 = time.time()
    prm = json.load(open(prm_path)); rng = np.random.default_rng(prm['seed'])
    sid = prm['scene_id']
    ctx = setup(prm, rng)
    vt = ctx['sc'].view_settings.view_transform
    vrng = np.random.default_rng(prm['seed'] + 7919)
    done = []; used_az = [prm['azimuth']]
    for k in range(views):
        v = 'abcdefgh'[k]
        p = dict(prm)
        if k > 0:
            for _ in range(50):              # keep views distinct: >= 60 deg azimuth from every earlier view
                az = float(vrng.choice(np.arange(0, 360, 45)) + vrng.uniform(-15, 15))
                if all(abs((az - u + 180) % 360 - 180) >= 60 for u in used_az):
                    break
            p['azimuth'] = az
            p['elevation'] = float(vrng.choice(ELEVS))
            p['framing'] = str(vrng.choice(FRAMINGS))
        used_az.append(p['azimuth'])
        tgt, d = S.place_camera(ctx['cam'], ctx['P'], p, rng)
        tr = time.time()
        render_rgb(ctx, p, os.path.join(out_dir, sid + v + '.jpg'), vt)
        t_rgb = time.time() - tr
        render_part2(ctx, os.path.join(out_dir, sid + v + '_part2.png'))
        q = {k_: val for k_, val in p.items() if not k_.startswith('_')}
        q['scene_id'] = sid + v; q['base_scene'] = sid; q['view'] = v
        json.dump({'scene_id': sid + v, 'params': q, 'train_only': True, 'render_s': round(t_rgb, 1),
                   'camera': {'location': list(ctx['cam'].location), 'target': [float(x) for x in tgt], 'distance': d}},
                  open(os.path.join(out_dir, sid + v + '_sidecar.json'), 'w'), indent=1)
        done.append(sid + v)
    json.dump({'scene_id': sid, 'views': done, 'total_s': round(time.time() - t0, 1)},
              open(os.path.join(out_dir, sid + '_views.json'), 'w'))
    print('VIEWSDONE', sid, len(done), round(time.time() - t0, 1))


if __name__ == '__main__':
    a = [x for x in sys.argv[sys.argv.index('--') + 1:]] if '--' in sys.argv else sys.argv[1:]
    if a[0] == 'label':
        main_label(a[1], a[2])
    else:
        main_train(a[1], a[2], int(a[3]) if len(a) > 3 else 3)
