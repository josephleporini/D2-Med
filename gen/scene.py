"""Probe B scene generator + ground-truth labeler (one scene per call).

Usage (Blender env):  python3 scene.py <params.json> <out_dir>

Labels follow Probe B Scope v1.0 Sec. 4:
  - laterality from bone-bound limb regions (never from image position)
  - visibility = visible limb pixels (ID pass) / limb pixels rendered alone with 3x overscan
  - amputated site: stump region visibility decides amputation vs not_testable
  - labels at thresholds 0 (any pixel), 0.10 (headline), 0.25, 0.50
"""
import sys, os, json, math, time
sys.path.insert(0, os.path.dirname(__file__))
import bpy, bmesh
import numpy as np
from mathutils import Vector, Matrix
import manikin as mk

THRESHOLDS = [0.0, 0.10, 0.25, 0.50]
HEADLINE = 0.10
OVERSCAN = 3
ID_RES = (640, 480)
ID_COLORS = {'LUE': (1, 0, 0), 'RUE': (0, 1, 0), 'LLE': (0, 0, 1), 'RLE': (1, 1, 0),
             'TORSO': (1, 0, 1), 'OCC': (0, 1, 1), 'FLOOR': (0.5, 0.5, 0.5)}
SKIN_TONES = {'peach': (0.80, 0.60, 0.48), 'tan': (0.62, 0.44, 0.30),
              'brown': (0.40, 0.26, 0.16), 'dark': (0.20, 0.13, 0.09)}
FLOORS = {'concrete': ((0.32, 0.32, 0.31), (0.22, 0.22, 0.22)), 'grass': ((0.14, 0.22, 0.06), (0.08, 0.13, 0.04)),
          'gravel': ((0.30, 0.26, 0.21), (0.17, 0.15, 0.12)), 'litter': ((0.18, 0.20, 0.11), (0.12, 0.14, 0.07))}


def noise_material(name, c1, c2, scale=40.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; bsdf = nt.nodes['Principled BSDF']
    nz = nt.nodes.new('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = scale
    ramp = nt.nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].color = (*c1, 1); ramp.color_ramp.elements[1].color = (*c2, 1)
    nt.links.new(nz.outputs['Fac'], ramp.inputs['Fac']); nt.links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
    bsdf.inputs['Roughness'].default_value = 0.9
    m.diffuse_color = (*c1, 1)
    return m


def face_sites(ob):
    gi = {ob.vertex_groups['SITE_' + s].index: s for s in mk.SITES}
    vs = []
    for v in ob.data.vertices:
        s = 'TORSO'
        for g in v.groups:
            if g.group in gi and g.weight > 0.5:
                s = gi[g.group]
        vs.append(s)
    out = []
    for p in ob.data.polygons:
        ss = [vs[i] for i in p.vertices]
        out.append(max(set(ss), key=ss.count))
    return out


PART_NAMES = ['upper_arm', 'forearm', 'hand', 'thigh', 'shank', 'foot']
PART_OF_BONE = [('shoulder01', 'upper_arm'), ('upperarm', 'upper_arm'), ('lowerarm', 'forearm'), ('wrist', 'hand'),
                ('metacarpal', 'hand'), ('finger', 'hand'), ('upperleg', 'thigh'), ('lowerleg', 'shank'),
                ('foot', 'foot'), ('toe', 'foot')]
HEAD_BONES = ('head', 'jaw', 'neck', 'eye', 'special', 'levator', 'oris', 'orbicularis', 'temporalis', 'tongue',
              'risorius', 'mouth', 'nose', 'ear')
# (part, side) classes for the part-ID pass; colours use only 0 / 0.5 / 1 per channel so they survive sRGB exactly
PART_CLASSES = ['TORSO', 'HEAD', 'OCC'] + [sd + '_' + p for sd in 'LR' for p in PART_NAMES]
_GRID = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1), (0.5, 0, 0), (0, 0.5, 0),
         (0, 0, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5), (1, 0.5, 0), (1, 0, 0.5)]
PART_COLORS = dict(zip(PART_CLASSES, _GRID))


def face_parts(ob, fsite):
    """(part class) per face: limb faces -> side from the face's site, part from the dominant limb bone;
    other faces -> HEAD if the dominant bone is a head/neck bone, else TORSO."""
    names = {g.index: g.name for g in ob.vertex_groups}
    vpart = []
    for v in ob.data.vertices:
        best, bw = None, 0.0
        for g in v.groups:
            n = names[g.group]
            if n.startswith('SITE_'):
                continue
            if g.weight > bw:
                best, bw = n, g.weight
        base = (best or '').split('.')[0]
        p = next((pp for key, pp in PART_OF_BONE if base.startswith(key)), None)
        if p is None:
            p = 'HEAD' if base.startswith(HEAD_BONES) else 'TORSO'
        vpart.append(p)
    out = []
    for f, s in zip(ob.data.polygons, fsite):
        ps = [vpart[i] for i in f.vertices]
        if s in mk.SITES:
            lp = [p for p in ps if p in PART_NAMES]
            p = max(set(lp), key=lp.count) if lp else ('upper_arm' if s[1] == 'U' else 'thigh')
            out.append(s[0] + '_' + p)
        else:
            nl = [p for p in ps if p in ('HEAD', 'TORSO')]
            out.append(max(set(nl), key=nl.count) if nl else 'TORSO')
    return out


def drape_blanket(cx, cy, sx, sy, n=48, lift=0.025, iters=25):
    """Grid blanket dropped onto whatever is below (body + floor) by vertical ray casts, then relaxed so it
    spans gaps like cloth instead of hugging every contour. Height never goes below the surface + lift."""
    sc = bpy.context.scene
    dg = bpy.context.evaluated_depsgraph_get()
    xs = np.linspace(cx - sx / 2, cx + sx / 2, n); ys = np.linspace(cy - sy / 2, cy + sy / 2, n)
    floor_z = np.zeros((n, n))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            hit, loc, *_ = sc.ray_cast(dg, Vector((x, y, 5.0)), Vector((0, 0, -1)))
            floor_z[i, j] = (loc.z if hit else 0.0) + lift
    z = floor_z.copy()
    for _ in range(iters):                      # cloth-like relaxation: smooth, then keep above surface
        zp = np.pad(z, 1, mode='edge')
        z = np.maximum(floor_z, (zp[:-2, 1:-1] + zp[2:, 1:-1] + zp[1:-1, :-2] + zp[1:-1, 2:]) / 4 * 0.98 + z * 0.02)
    verts = [(xs[i], ys[j], z[i, j]) for i in range(n) for j in range(n)]
    faces = [(i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1) for i in range(n - 1) for j in range(n - 1)]
    me = bpy.data.meshes.new('blanket'); me.from_pydata(verts, [], faces); me.update()
    for p in me.polygons:
        p.use_smooth = True
    o = bpy.data.objects.new('blanket', me); sc.collection.objects.link(o)
    return o


def add_occluder(kind, P, rng, bbox):
    """P: posed body vertices (world). Returns list of occluder objects."""
    lo, hi = bbox
    objs = []
    if kind == 'none':
        return objs
    zmax = hi[2]
    if kind == 'blanket':                     # sheet over lower or upper half, draped just above the body
        part = rng.choice(['legs', 'chest'])
        c = P.mean(0)
        ax = np.argmax(hi[:2] - lo[:2])       # long axis of lying body
        span = hi[ax] - lo[ax]
        ctr = c.copy()
        sgn = 1 if rng.random() < 0.5 else -1
        ctr[ax] = c[ax] + sgn * span * 0.25
        sx = span * 0.55 if ax == 0 else (hi[0] - lo[0]) + 0.3
        sy = span * 0.55 if ax == 1 else (hi[1] - lo[1]) + 0.3
        o = drape_blanket(ctr[0], ctr[1], sx, sy)
        o['note'] = f'blanket half={sgn}'
        objs.append(o)
    elif kind == 'gear_bag':                  # box resting near/on a limb
        i = rng.integers(len(P)); p = P[i]
        bpy.ops.mesh.primitive_cube_add(size=1, location=(p[0], p[1], p[2] + 0.13))
        o = bpy.context.object; o.scale = (0.5, 0.32, 0.26); o.rotation_euler[2] = rng.uniform(0, math.pi)
        objs.append(o)
    elif kind == 'strap':                     # litter strap across the body
        c = P.mean(0)
        bpy.ops.mesh.primitive_cube_add(size=1, location=(c[0], c[1] + rng.uniform(-0.4, 0.4), zmax + 0.012))
        o = bpy.context.object; o.scale = (1.4, 0.07, 0.02)
        o.rotation_euler[2] = rng.uniform(-0.3, 0.3) + (math.pi / 2 if (hi[0] - lo[0]) > (hi[1] - lo[1]) else 0)
        objs.append(o)
    elif kind == 'medic_arm':                 # a forearm-sized cylinder reaching in above the body
        i = rng.integers(len(P)); p = P[i]
        bpy.ops.mesh.primitive_cylinder_add(radius=0.045, depth=0.7, location=(p[0], p[1], p[2] + 0.22))
        o = bpy.context.object; o.rotation_euler = (math.pi / 2, 0, rng.uniform(0, math.pi))
        objs.append(o)
    return objs


def look_at(cam, target):
    cam.rotation_euler = (Vector(target) - cam.location).to_track_quat('-Z', 'Y').to_euler()


def place_camera(cam, P, prm, rng):
    lo, hi = P.min(0), P.max(0)
    c = (lo + hi) / 2
    r = np.linalg.norm(hi - lo) / 2
    framing = prm['framing']
    tgt = c.copy()
    dist_mult = 0.95
    ax = np.argmax(hi[:2] - lo[:2])
    if framing in ('upper_crop', 'lower_crop'):
        # head end: side of the long axis nearest the head vertex (max of torso top)
        head_end = prm['_head_xy'][ax] > c[ax]
        sgn = 1 if head_end else -1
        if framing == 'lower_crop':
            sgn = -sgn
        tgt[ax] = c[ax] + sgn * (hi[ax] - lo[ax]) * 0.28
        dist_mult = 0.6
    elif framing == 'limb_closeup':
        # one limb fills most of the frame (real close-up photos); the rest of the body is mostly out of frame
        tgt = np.array(prm['_limb_xyz'][prm['closeup_site']], dtype=float)
        dist_mult = float(prm.get('closeup_dist', 0.35))
    elif framing == 'off_center':
        perp = 1 - ax
        tgt[ax] = c[ax] + rng.choice([-1, 1]) * (hi[ax] - lo[ax]) * 0.42
        dist_mult = 0.8
    az = math.radians(prm['azimuth']); el = math.radians(prm['elevation'])
    fov = 2 * math.atan(cam.data.sensor_width / 2 / cam.data.lens) * (3 / 4)  # vertical-ish fov
    d = r / math.sin(fov / 2) * dist_mult
    cam.location = Vector((tgt[0] + d * math.cos(el) * math.sin(az),
                           tgt[1] - d * math.cos(el) * math.cos(az),
                           tgt[2] + d * math.sin(el)))
    look_at(cam, tgt)
    return tgt, d


def set_id_look(sc):
    sc.render.engine = 'BLENDER_WORKBENCH'
    sc.display.shading.light = 'FLAT'
    sc.display.shading.color_type = 'MATERIAL'
    sc.display.render_aa = 'OFF'
    sc.view_settings.view_transform = 'Standard'
    sc.render.dither_intensity = 0
    sc.render.film_transparent = False
    sc.world.color = (0, 0, 0)
    sc.display.shading.show_shadows = False
    sc.display.shading.show_cavity = False
    sc.display.shading.show_object_outline = False
    sc.display.shading.show_specular_highlight = False


def render_to_array(sc, path):
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(path)
    w, h = img.size
    a = np.array(img.pixels[:]).reshape(h, w, 4)[::-1, :, :3]
    bpy.data.images.remove(img)
    return a


def count(a, rgb):
    return int(np.all(np.abs(a - np.array(rgb)) < 0.02, axis=2).sum())


def main(prm_path, out_dir):
    t0 = time.time()
    prm = json.load(open(prm_path))
    rng = np.random.default_rng(prm['seed'])
    sid = prm['scene_id']
    os.makedirs(out_dir, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    if sc.world is None:
        sc.world = bpy.data.worlds.new('world')

    ob, arm, jpos, skel = mk.build()
    for site, lvl in prm['amputations'].items():
        mk.amputate(ob, arm, site, lvl)
    fsite = face_sites(ob)
    fpart = face_parts(ob, fsite)
    # 'hand'/'foot' means an intact extremity: the remnant of a distal (partial hand/foot) amputation is labelled
    # as the proximal segment, matching the Test A2 truth (extremity keypoint absent after a distal cut)
    for site, lvl in prm['amputations'].items():
        if lvl == 'distal':
            ext, prox = ('hand', 'forearm') if site[1] == 'U' else ('foot', 'shank')
            fpart = [site[0] + '_' + prox if fp == site[0] + '_' + ext else fp for fp in fpart]
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

    # appearance
    skin = mk.plastic_material('skin', SKIN_TONES[prm['skin']])
    ob.data.materials.append(skin)
    fl = FLOORS[prm['floor']]
    bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 0, -0.004)); floor = bpy.context.object
    floor.data.materials.append(noise_material('floor', *fl))
    occ = add_occluder(prm['occluder'], P, rng, (P.min(0), P.max(0)))
    occ_mat = noise_material('occ', (0.25, 0.27, 0.18), (0.16, 0.18, 0.11), 12)
    for o in occ:
        o.data.materials.append(occ_mat)

    # lighting
    sun = bpy.data.objects.new('sun', bpy.data.lights.new('sun', 'SUN')); sc.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(prm['sun_el']), 0, math.radians(prm['sun_az']))
    sun.data.energy = {'indoor_flat': 1.5, 'outdoor_sun': 4.0, 'low_light': 0.9}[prm['lighting']]
    sun.data.angle = math.radians(12 if prm['lighting'] == 'indoor_flat' else 1.5)
    sc.world.use_nodes = True
    bg = sc.world.node_tree.nodes['Background']
    bg.inputs['Strength'].default_value = {'indoor_flat': 0.9, 'outdoor_sun': 0.6, 'low_light': 0.15}[prm['lighting']]
    bg.inputs['Color'].default_value = (0.75, 0.8, 0.9, 1)

    # camera
    cd = bpy.data.cameras.new('cam'); cd.lens = 35; cd.sensor_width = 36
    cam = bpy.data.objects.new('cam', cd); sc.collection.objects.link(cam); sc.camera = cam
    tgt, d = place_camera(cam, P, prm, rng)

    ID_ONLY = os.environ.get('ID_ONLY') == '1'
    t_rgb = 0.0
    # ---------- RGB render ----------
    if not ID_ONLY:
      sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'; sc.cycles.samples = prm.get('samples', 32)
      try:
          sc.cycles.use_denoising = True
      except Exception:
          pass
      sc.render.resolution_x, sc.render.resolution_y = 1280, 960
      sc.render.image_settings.file_format = 'JPEG'; sc.render.image_settings.quality = 92
      rgb_path = os.path.join(out_dir, sid + '.jpg')
      sc.render.filepath = rgb_path
      tr = time.time(); bpy.ops.render.render(write_still=True); t_rgb = time.time() - tr
      if os.environ.get('RGB_ONLY') == '1':          # re-render the image only (labels unchanged: same seed/geometry)
          print('RGBDONE', sid, round(t_rgb, 1)); return

    # ---------- ID objects: posed copy split by site ----------
    dg = bpy.context.evaluated_depsgraph_get()
    posed = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    posed.transform(ob.matrix_world)
    id_objs = {}
    for s in mk.SITES + ['TORSO']:
        bm = bmesh.new(); bm.from_mesh(posed); bm.faces.ensure_lookup_table()
        bmesh.ops.delete(bm, geom=[f for f in bm.faces if fsite[f.index] != s], context='FACES')
        me = bpy.data.meshes.new('id_' + s); bm.to_mesh(me); bm.free()
        o = bpy.data.objects.new('id_' + s, me); sc.collection.objects.link(o)
        m = bpy.data.materials.new('idm_' + s); m.diffuse_color = (*ID_COLORS[s], 1); me.materials.append(m)
        id_objs[s] = o
    for o, key in [(floor, 'FLOOR')] + [(x, 'OCC') for x in occ]:
        m = bpy.data.materials.new('idm_' + o.name); m.diffuse_color = (*ID_COLORS[key], 1)
        o.data.materials.clear(); o.data.materials.append(m)
    ob.hide_render = True
    set_id_look(sc)
    sc.render.image_settings.file_format = 'PNG'
    sc.render.resolution_x, sc.render.resolution_y = ID_RES
    # ---- part-ID pass: (part, side) classes ----
    part_objs = []
    for pc in PART_CLASSES:
        if pc == 'OCC':
            continue
        bm = bmesh.new(); bm.from_mesh(posed); bm.faces.ensure_lookup_table()
        bmesh.ops.delete(bm, geom=[f for f in bm.faces if fpart[f.index] != pc], context='FACES')
        me = bpy.data.meshes.new('pt_' + pc); bm.to_mesh(me); bm.free()
        o = bpy.data.objects.new('pt_' + pc, me); sc.collection.objects.link(o)
        m = bpy.data.materials.new('ptm_' + pc); m.diffuse_color = (*PART_COLORS[pc], 1); me.materials.append(m)
        part_objs.append(o)
    occ_saved = [(o, list(o.data.materials)) for o in occ]
    pocc = bpy.data.materials.new('ptm_OCC'); pocc.diffuse_color = (*PART_COLORS['OCC'], 1)
    for o in occ:
        o.data.materials.clear(); o.data.materials.append(pocc)
    for o in id_objs.values():
        o.hide_render = True
    sc.render.filepath = os.path.join(out_dir, sid + '_part.png')
    bpy.ops.render.render(write_still=True)
    for o in part_objs:
        o.hide_render = True
    for o in id_objs.values():
        o.hide_render = False
    for o, mats in occ_saved:
        o.data.materials.clear()
        for mm in mats:
            o.data.materials.append(mm)
    if ID_ONLY:
        print('PARTDONE', sid, round(time.time() - t0, 1))
        return
    id_path = os.path.join(out_dir, sid + '_id.png')
    A = render_to_array(sc, id_path)
    vis_px = {s: count(A, ID_COLORS[s]) for s in mk.SITES}

    # alone renders with overscan (same pixel scale, 3x field) -> full limb pixel count
    cd.sensor_width = 36 * OVERSCAN
    sc.render.resolution_x, sc.render.resolution_y = ID_RES[0] * OVERSCAN, ID_RES[1] * OVERSCAN
    hide = [floor] + occ + list(id_objs.values())
    for o in part_objs:
        o.hide_render = True
    alone_px = {}
    for s in mk.SITES:
        for o in hide:
            o.hide_render = (o is not id_objs[s])
        B = render_to_array(sc, os.path.join(out_dir, f'_alone_{s}.png'))
        alone_px[s] = count(B, ID_COLORS[s])
        os.remove(os.path.join(out_dir, f'_alone_{s}.png'))
    for o in hide:
        o.hide_render = False
    cd.sensor_width = 36

    # labels
    frac = {s: (vis_px[s] / alone_px[s] if alone_px[s] else 0.0) for s in mk.SITES}
    labels = {}
    for t in THRESHOLDS:
        lab = {}
        for s in mk.SITES:
            visible = frac[s] > 0 if t == 0 else frac[s] >= t
            if not visible:
                lab[s] = 'not_testable'
            elif s in prm['amputations']:
                lab[s] = 'amputation'
            else:
                lab[s] = 'no_injury'
        labels[f'{t:.2f}'] = lab
    site_map = {'LUE': ('upper_extremity', 'left'), 'RUE': ('upper_extremity', 'right'),
                'LLE': ('lower_extremity', 'left'), 'RLE': ('lower_extremity', 'right')}
    head_lab = labels[f'{HEADLINE:.2f}']
    icd = {'image_id': sid + '.jpg',
           'sites': [{'body_region': site_map[s][0], 'laterality': site_map[s][1], 'injury_type': head_lab[s]}
                     for s in mk.SITES]}
    prm.pop('_head_xy', None)
    side = {'scene_id': sid, 'params': prm, 'visible_px': vis_px, 'alone_px': alone_px,
            'visible_fraction': {s: round(frac[s], 4) for s in mk.SITES},
            'labels_by_threshold': labels, 'headline_threshold': HEADLINE,
            'camera': {'location': list(cam.location), 'target': [float(x) for x in tgt], 'distance': d},
            'render_s': round(t_rgb, 1), 'total_s': round(time.time() - t0, 1)}
    json.dump(icd, open(os.path.join(out_dir, sid + '_icd.json'), 'w'), indent=1)
    json.dump(side, open(os.path.join(out_dir, sid + '_sidecar.json'), 'w'), indent=1)
    print('DONE', sid, json.dumps(side['visible_fraction']), json.dumps(head_lab), side['total_s'])


if __name__ == '__main__':
    main(sys.argv[-2], sys.argv[-1])
