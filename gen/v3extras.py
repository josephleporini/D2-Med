"""Generator v3 additions (BT-2b; Real Fidelity Gap v1.0). Blender environment. All of it runs only when the scene
parameters ask for it, so v1 and v2 scenes reproduce unchanged.

  bystanders   prm['bystanders'] = [{'pose': 'kneel'|'crouch'|'stand', 'side': +1|-1, 'dist': m, 'along': m, 'lean': deg}]
               Full human figures (a second MakeHuman body) in a uniform, next to the casualty, facing it. They are
               occluders in every truth product (label OCC), never casualty limbs; they are hidden in the per-site
               full-outline renders. Real photos show 3 people on median.
  red_gear     prm['red_gear'] = n: red or orange bags and pouches on the floor near the body (occluders).
  blood_pool   prm['blood_pool'] = [site, ...]: a glossy dark-red pool on the floor beside that limb. It is part of
               the floor in the truth products (background), not a wound.
  camo         prm['camo'] = True: garments get a four-color woodland or desert camouflage material.
  view, light  prm['view'] = 'Standard' or 'AgX' with prm['look']; prm['exposure'] (stops); prm['light_boost'] scales the sun and dims the sky:
               real photos have 4x the contrast of the v1/v2 renders.
"""
import math
import bpy
import numpy as np
from mathutils import Matrix, Vector
import manikin as mk
import scene as S

def _lin(c):
    """sRGB 0-255 to linear 0-1 (Blender material colors are linear)"""
    c = np.asarray(c, float) / 255.0
    return tuple(float(x) for x in np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4))


# camouflage palettes, given in sRGB as photographed (smoke render 29 Sep: first values in linear came out washed out)
CAMO = {'woodland': [_lin(c) for c in [(85, 90, 60), (110, 95, 70), (45, 45, 35), (70, 80, 50)]],
        'desert': [_lin(c) for c in [(175, 155, 118), (145, 125, 95), (125, 108, 82), (100, 86, 64)]],
        'multicam': [_lin(c) for c in [(140, 130, 98), (100, 105, 72), (160, 150, 116), (75, 66, 50)]]}


def camo_material(name, rng, kind=None):
    kind = kind or str(rng.choice(list(CAMO)))
    cols = CAMO[kind]
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; bsdf = nt.nodes['Principled BSDF']
    nz = nt.nodes.new('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = float(rng.uniform(4, 10))
    nz.inputs['Detail'].default_value = 3.0
    ramp = nt.nodes.new('ShaderNodeValToRGB'); ramp.color_ramp.interpolation = 'CONSTANT'
    el = ramp.color_ramp.elements
    el[0].position = 0.0; el[0].color = (*cols[0], 1)
    el[1].position = 0.42; el[1].color = (*cols[1], 1)
    for pos, c in ((0.52, cols[2]), (0.62, cols[3])):
        e = el.new(pos); e.color = (*c, 1)
    nt.links.new(nz.outputs['Fac'], ramp.inputs['Fac']); nt.links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
    bsdf.inputs['Roughness'].default_value = 0.85
    m.diffuse_color = (*cols[0], 1)
    return m


def _pose_bystander(arm, pose, lean, rng):
    rb = mk.rot_bone
    mk.arms_down(arm)
    for sd, sg in (('.L', 1), ('.R', -1)):                    # arms reach forward and down toward the casualty
        rb(arm, 'upperarm01' + sd, (1, 0, 0), -float(rng.uniform(40, 80)))
        rb(arm, 'lowerarm01' + sd, (1, 0, 0), -float(rng.uniform(10, 40)))
    if pose == 'kneel':                                       # both knees down, thighs upright, shanks back
        for sd in ('.L', '.R'):
            rb(arm, 'lowerleg01' + sd, (1, 0, 0), 90)
    elif pose == 'crouch':                                    # deep squat
        for sd in ('.L', '.R'):
            rb(arm, 'upperleg01' + sd, (1, 0, 0), -100)
            rb(arm, 'lowerleg01' + sd, (1, 0, 0), 130)
    rb(arm, 'spine03', (1, 0, 0), -float(lean) / 2)          # lean toward the casualty
    rb(arm, 'spine01', (1, 0, 0), -float(lean) / 2)


def add_bystanders(prm, rng, P):
    """P: posed casualty vertices (world). Returns the bystander mesh objects (to be treated as occluders)."""
    out = []
    lo, hi = P.min(0), P.max(0); c = P.mean(0)
    ax = int(np.argmax(hi[:2] - lo[:2]))
    along = np.zeros(3); along[ax] = 1.0
    perp = np.zeros(3); perp[1 - ax] = 1.0
    half_w = (hi[1 - ax] - lo[1 - ax]) / 2
    for j, b in enumerate(prm.get('bystanders') or []):
        ob, arm, _, _ = mk.build(name=f'zz_bystander{j}')     # 'zz' keeps the casualty first in name order
        _pose_bystander(arm, b['pose'], b.get('lean', 20), rng)
        fsite = S.face_sites(ob); fpart = S.face_parts(ob, fsite)
        skin = mk.plastic_material(f'bys_skin{j}', S.SKIN_TONES[str(rng.choice(list(S.SKIN_TONES)))], 0.5)
        uni = camo_material(f'bys_uni{j}', rng)
        ob.data.materials.append(uni); ob.data.materials.append(skin)
        for p, fp in zip(ob.data.polygons, fpart):
            p.material_index = 1 if (fp.startswith('HEAD') or fp.endswith('hand')) else 0
        pos = c + perp * b['side'] * (half_w + b['dist']) + along * b['along']
        # face the casualty: the figure's front is -Y in its armature frame
        to = c - pos; yaw = math.atan2(to[1], to[0]) + math.pi / 2
        arm.matrix_world = Matrix.Translation(Vector((pos[0], pos[1], 0))) @ Matrix.Rotation(yaw, 4, 'Z')
        bpy.context.view_layer.update()
        mk.ground(ob, arm)
        out.append(ob)
    return out


def add_red_gear(prm, rng, P):
    out = []
    lo, hi = P.min(0), P.max(0)
    for j in range(int(prm.get('red_gear') or 0)):
        side = rng.choice([-1, 1]); ax = int(np.argmax(hi[:2] - lo[:2]))
        loc = P.mean(0).copy(); loc[ax] += rng.uniform(-0.5, 0.5) * (hi[ax] - lo[ax])
        loc[1 - ax] += side * ((hi[1 - ax] - lo[1 - ax]) / 2 + rng.uniform(0.05, 0.35))
        size = rng.uniform(0.12, 0.35)
        if rng.random() < 0.5:
            bpy.ops.mesh.primitive_cube_add(size=1, location=(loc[0], loc[1], size * 0.3))
            o = bpy.context.object; o.scale = (size, size * rng.uniform(0.5, 0.8), size * 0.6)
        else:
            bpy.ops.mesh.primitive_cylinder_add(radius=size * 0.3, depth=size, location=(loc[0], loc[1], size * 0.3))
            o = bpy.context.object; o.rotation_euler = (math.pi / 2, 0, rng.uniform(0, math.pi))
        o.rotation_euler[2] = rng.uniform(0, math.pi)
        red = (rng.uniform(0.45, 0.75), rng.uniform(0.01, 0.06), rng.uniform(0.01, 0.05))
        if rng.random() < 0.3:
            red = (rng.uniform(0.7, 0.9), rng.uniform(0.18, 0.35), rng.uniform(0.0, 0.05))
        o.data.materials.append(S.noise_material(f'red{j}', red, tuple(x * 0.7 for x in red), 30))
        out.append(o)
    return out


def add_blood_pools(prm, rng, limb_xyz):
    """Pools on the floor beside the named limbs. Returns decal objects (floor class in truth)."""
    out = []
    for j, site in enumerate(prm.get('blood_pool') or []):
        if site not in limb_xyz:
            continue
        x, y, _ = limb_xyz[site]
        r = rng.uniform(0.12, 0.40)
        bpy.ops.mesh.primitive_circle_add(vertices=64, radius=r, fill_type='NGON',
                                          location=(x + rng.uniform(-0.15, 0.15), y + rng.uniform(-0.15, 0.15), 0.0015))
        o = bpy.context.object; o.scale = (1.0, rng.uniform(0.4, 1.0), 1.0); o.rotation_euler[2] = rng.uniform(0, math.pi)
        # irregular outline: radial noise from a few random harmonics (a perfect disc looked artificial)
        ph = rng.uniform(0, 2 * math.pi, 5); am = rng.uniform(0.05, 0.25, 5) / np.arange(1, 6)
        for v in o.data.vertices:
            if v.co.length > 1e-6:
                t = math.atan2(v.co.y, v.co.x)
                v.co *= 1.0 + float(sum(a * math.sin((k + 2) * t + p_) for k, (a, p_) in enumerate(zip(am, ph))))
        m = bpy.data.materials.new(f'pool{j}'); m.use_nodes = True
        b = m.node_tree.nodes['Principled BSDF']
        b.inputs['Base Color'].default_value = (rng.uniform(0.10, 0.30), 0.005, 0.005, 1)
        b.inputs['Roughness'].default_value = rng.uniform(0.05, 0.25)
        m.diffuse_color = (0.2, 0.01, 0.01, 1)
        o.data.materials.append(m)
        out.append(o)
    return out


def apply_view(sc, prm, rng=None):
    """Contrast settings for the RGB render (v3). Returns the view transform name to use."""
    vt = prm.get('view')
    if not vt:
        return None
    sc.view_settings.view_transform = vt
    if prm.get('exposure') is not None:                     # darker exposure: real photos are darker (median 0.32)
        sc.view_settings.exposure = float(prm['exposure'])
    look = prm.get('look')
    if look:
        try:
            sc.view_settings.look = look
        except Exception:
            pass
    return vt
