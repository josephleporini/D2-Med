"""Probe B manikin builder (Blender 5.x, run with system python3 + bpy).

Builds a rigged manikin from the CC0 MakeHuman hm08 base mesh, default skeleton and weights.
Conventions:
  MakeHuman: Y up, figure faces +Z, anatomical left = +X.
  Blender:   (x, y, z)_MH -> (x, -z, y)_BL * 0.1  -> Z up, figure faces -Y, anatomical left = +X, metres.
Limb regions are assigned from bone weights, so anatomical side comes from bone names (".L"/".R"),
never from image or world position.
"""
import bpy, bmesh, json, math, os
import numpy as np
from mathutils import Vector, Matrix

DATA = os.path.join(os.path.dirname(__file__), '..', 'assets', 'mh', 'makehuman', 'data')
SCALE = 0.1

SITES = ['LUE', 'RUE', 'LLE', 'RLE']
SITE_BONES = {
    'UE': ['shoulder01', 'upperarm01', 'upperarm02', 'lowerarm01', 'lowerarm02', 'wrist',
           'metacarpal', 'finger'],
    'LE': ['upperleg01', 'upperleg02', 'lowerleg01', 'lowerleg02', 'foot', 'toe'],
}
# chain of joints used to place amputation cut planes (rest pose)
CHAIN = {
    'UE': ['upperarm01', 'lowerarm01', 'wrist', 'finger3-1'],   # shoulder, elbow, wrist, knuckle heads
    'LE': ['upperleg01', 'lowerleg01', 'foot', 'toe3-1'],        # hip, knee, ankle, toe heads
}
# amputation level -> (segment index, fraction along segment) measured along CHAIN
AMP_LEVELS = {
    'disarticulation': (0, 0.08),
    'above_joint':     (0, 0.55),   # above elbow / above knee
    'below_joint':     (1, 0.45),   # below elbow / below knee
    'distal':          (2, 0.15),   # wrist / ankle
}


def to_bl(p):
    p = np.asarray(p, dtype=float)
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], axis=-1) * SCALE


def site_of_bone(bone):
    base, _, side = bone.partition('.')
    if side not in ('L', 'R'):
        return None
    for limb, keys in SITE_BONES.items():
        if any(base.startswith(k) for k in keys):
            return side + limb
    return None


def load_obj():
    V, F, groups, cur = [], [], [], None
    for ln in open(os.path.join(DATA, '3dobjs', 'base.obj')):
        if ln.startswith('v '):
            V.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith('g '):
            cur = ln.split()[1]
        elif ln.startswith('f '):
            F.append([int(t.split('/')[0]) - 1 for t in ln.split()[1:]])
            groups.append(cur)
    return np.array(V), F, groups


def build(name='manikin'):
    V, F, groups = load_obj()
    skel = json.load(open(os.path.join(DATA, 'rigs', 'default.mhskel')))
    wts = json.load(open(os.path.join(DATA, 'rigs', 'default_weights.mhw')))['weights']

    # joint positions from the FULL vertex set (joint helpers included), then keep body faces only
    jpos = {k: to_bl(V[v].mean(0)) for k, v in skel['joints'].items()}
    bodyF = [f for f, g in zip(F, groups) if g == 'body']
    used = sorted({i for f in bodyF for i in f})
    remap = {old: new for new, old in enumerate(used)}
    Vb = to_bl(V[used])
    Fb = [[remap[i] for i in f] for f in bodyF]

    me = bpy.data.meshes.new(name)
    me.from_pydata(Vb.tolist(), [], Fb)
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    for p in me.polygons:
        p.use_smooth = True

    # armature
    ad = bpy.data.armatures.new(name + '_rig')
    arm = bpy.data.objects.new(name + '_rig', ad)
    bpy.context.scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')
    eb = {}
    for bn, b in skel['bones'].items():
        e = ad.edit_bones.new(bn)
        e.head = Vector(jpos[b['head']])
        e.tail = Vector(jpos[b['tail']])
        if (e.tail - e.head).length < 1e-5:
            e.tail = e.head + Vector((0, 0, 0.01))
        eb[bn] = e
    for bn, b in skel['bones'].items():
        if b.get('parent'):
            eb[bn].parent = eb[b['parent']]
    bpy.ops.object.mode_set(mode='OBJECT')

    # weights -> vertex groups
    site_w = {s: np.zeros(len(used)) for s in SITES}
    for bn, lst in wts.items():
        vg = ob.vertex_groups.new(name=bn)
        s = site_of_bone(bn)
        by_w = {}
        for vi, w in lst:
            if vi in remap:
                by_w.setdefault(round(w, 4), []).append(remap[vi])
                if s:
                    site_w[s][remap[vi]] += w
        for w, vis in by_w.items():
            vg.add(vis, w, 'REPLACE')
    mod = ob.modifiers.new('Armature', 'ARMATURE')
    mod.object = arm
    ob.parent = arm

    # region per vertex: site with summed weight > 0.5, else torso/head
    region = np.full(len(used), 'TORSO', dtype=object)
    stack = np.stack([site_w[s] for s in SITES])
    best = stack.argmax(0)
    for i in range(len(used)):
        if stack[best[i], i] > 0.5:
            region[i] = SITES[best[i]]
    for s in SITES:
        vg = ob.vertex_groups.new(name='SITE_' + s)
        vg.add([int(i) for i in np.where(region == s)[0]], 1.0, 'REPLACE')
    return ob, arm, jpos, skel


def chain_points(arm, site):
    side = site[0]
    limb = site[1:]
    bones = arm.data.bones
    return [Vector(bones[b + '.' + side].head_local) for b in CHAIN[limb]]


def amputate(ob, arm, site, level):
    """Delete limb vertices distal to a cut plane (rest pose) and cap the hole.
    Returns the cut point (rest pose) for reference."""
    seg, frac = AMP_LEVELS[level]
    pts = chain_points(arm, site)
    a, b = pts[seg], pts[seg + 1]
    cut = a.lerp(b, frac)
    n = (b - a).normalized()
    vg = ob.vertex_groups['SITE_' + site].index
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    deform = bm.verts.layers.deform.active
    kill = [v for v in bm.verts if vg in v[deform] and (v.co - cut).dot(n) > 0]
    # disarticulation: take the whole limb region
    if level == 'disarticulation':
        kill = [v for v in bm.verts if vg in v[deform] and (v.co - pts[0]).dot(n) > -0.01]
    bmesh.ops.delete(bm, geom=kill, context='VERTS')
    cap = bm.faces.layers.int.get('cap') or bm.faces.layers.int.new('cap')
    edges = [e for e in bm.edges if e.is_boundary]
    res = bmesh.ops.holes_fill(bm, edges=edges, sides=0)
    # put cap faces into the site group so the stump is labelled as that site; mark them in the 'cap' face
    # attribute (only fills bordering this limb; other mesh holes are filled too but are not stump caps)
    for f in res['faces']:
        if sum(vg in v[deform] for v in f.verts) >= 0.5 * len(f.verts):
            f[cap] = 1
        for v in f.verts:
            v[deform][vg] = 1.0
    bm.to_mesh(ob.data)
    bm.free()
    ob.data.update()
    return cut


def plastic_material(name, rgb, rough=0.42):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (*rgb, 1)
    bsdf.inputs['Roughness'].default_value = rough
    m.diffuse_color = (*rgb, 1)
    return m


# ---------- posing ----------
def rot_bone(arm, bone, axis, deg):
    """Rotate a pose bone about an armature-space axis through its head (standing reference frame:
    X = figure's left, -Y = front, Z = up)."""
    pb = arm.pose.bones[bone]
    h = Vector(pb.head)
    R = Matrix.Rotation(math.radians(deg), 4, Vector(axis))
    pb.matrix = Matrix.Translation(h) @ R @ Matrix.Translation(-h) @ pb.matrix
    bpy.context.view_layer.update()


def arms_down(arm):
    for side, sgn in (('.L', 1), ('.R', -1)):
        rot_bone(arm, 'upperarm01' + side, (1, 0, 0), 23)        # remove rest-pose forward angle
        rot_bone(arm, 'upperarm01' + side, (0, 1, 0), 35 * sgn)  # lower from A-pose to side


LIMB_POSES = ['neutral', 'abducted', 'arm_across', 'knee_bent', 'arm_overhead']


def limb_pose(arm, name, rng=None):
    """Named limb poses. Axis sign conventions verified in render checks."""
    L, R = '.L', '.R'
    if name == 'neutral':          # arms at sides (from A-pose), legs straight
        arms_down(arm)
    elif name == 'abducted':       # arms out, legs apart
        rot_bone(arm, 'upperarm01' + L, (0, 1, 0), -40)
        rot_bone(arm, 'upperarm01' + R, (0, 1, 0), 40)
        rot_bone(arm, 'upperleg01' + L, (0, 1, 0), -12)
        rot_bone(arm, 'upperleg01' + R, (0, 1, 0), 12)
    elif name == 'arm_across':     # right arm across chest, left at side
        rot_bone(arm, 'upperarm01' + L, (1, 0, 0), 23)
        rot_bone(arm, 'upperarm01' + L, (0, 1, 0), 35)
        rot_bone(arm, 'upperarm01' + R, (0, 0, 1), 60)
        rot_bone(arm, 'lowerarm01' + R, (0, 0, 1), 70)
    elif name == 'knee_bent':      # left knee flexed, arms at sides
        arms_down(arm)
        rot_bone(arm, 'upperleg01' + L, (1, 0, 0), -45)   # hip flexion: thigh forward (-Y)
        rot_bone(arm, 'lowerleg01' + L, (1, 0, 0), 90)    # knee flexion: shank back (+Y)
    elif name == 'arm_overhead':   # left arm raised overhead
        rot_bone(arm, 'upperarm01' + L, (0, 1, 0), -120)
        rot_bone(arm, 'upperarm01' + R, (1, 0, 0), 23)
        rot_bone(arm, 'upperarm01' + R, (0, 1, 0), -35)
    else:
        raise ValueError(name)


BODY_POSITIONS = ['supine', 'prone', 'left_lateral', 'right_lateral', 'semi_seated']


def body_position(arm, name):
    """Rotate the whole figure from standing into a lying/seated position."""
    if name == 'supine':
        M = Matrix.Rotation(math.radians(-90), 4, 'X')
    elif name == 'prone':
        M = Matrix.Rotation(math.radians(90), 4, 'X')
    elif name == 'left_lateral':   # lying on LEFT side: from supine, rotate +90 about Y -> +X (left) goes down
        M = Matrix.Rotation(math.radians(90), 4, 'Y') @ Matrix.Rotation(math.radians(-90), 4, 'X')
    elif name == 'right_lateral':  # lying on RIGHT side
        M = Matrix.Rotation(math.radians(-90), 4, 'Y') @ Matrix.Rotation(math.radians(-90), 4, 'X')
    elif name == 'semi_seated':    # trunk reclined 50 deg from vertical, hips flexed 40 -> thighs ~horizontal
        for sd in ('.L', '.R'):
            rot_bone(arm, 'upperleg01' + sd, (1, 0, 0), -40)
            rot_bone(arm, 'lowerleg01' + sd, (1, 0, 0), 20)
        M = Matrix.Rotation(math.radians(-50), 4, 'X')
    else:
        raise ValueError(name)
    arm.matrix_world = M @ arm.matrix_world
    bpy.context.view_layer.update()


def ground(ob, arm):
    """Translate so the lowest evaluated vertex sits on z=0."""
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    zs = [(ev.matrix_world @ v.co).z for v in ev.data.vertices]
    arm.location.z -= min(zs)
    bpy.context.view_layer.update()
