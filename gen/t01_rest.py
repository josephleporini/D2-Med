import sys, os, math, json
sys.path.insert(0, os.path.dirname(__file__))
import bpy, numpy as np
from mathutils import Vector
import manikin as mk

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
ob, arm, jpos, skel = mk.build()
# colour faces by site: LUE blue, RUE red, LLE cyan, RLE orange, torso grey
cols = {'TORSO': (0.6, 0.6, 0.6), 'LUE': (0.1, 0.2, 0.9), 'RUE': (0.9, 0.1, 0.1),
        'LLE': (0.1, 0.8, 0.9), 'RLE': (1.0, 0.55, 0.0)}
mats = {}
for k, c in cols.items():
    mats[k] = mk.plastic_material('m_' + k, c); ob.data.materials.append(mats[k])
names = list(cols)
gi = {s: ob.vertex_groups['SITE_' + s].index for s in mk.SITES}
vsite = []
for v in ob.data.vertices:
    s = 'TORSO'
    for g in v.groups:
        for k, idx in gi.items():
            if g.group == idx and g.weight > 0.5: s = k
    vsite.append(s)
for p in ob.data.polygons:
    ss = [vsite[i] for i in p.vertices]
    p.material_index = names.index(max(set(ss), key=ss.count))

mode = sys.argv[-1]
if mode != 'rest':
    for step in mode.split('+'):
        if step in mk.BODY_POSITIONS: mk.body_position(arm, step)
        elif step.startswith('amp:'):
            _, site, lvl = step.split(':'); mk.amputate(ob, arm, site, lvl)
        else: mk.limb_pose(arm, step)
    mk.ground(ob, arm)

cam_data = bpy.data.cameras.new('cam'); cam = bpy.data.objects.new('cam', cam_data)
sc.collection.objects.link(cam); sc.camera = cam
dg = bpy.context.evaluated_depsgraph_get(); ev = ob.evaluated_get(dg)
P = np.array([tuple(ev.matrix_world @ v.co) for v in ev.data.vertices]); c = P.mean(0)
if mode == 'rest':
    cam.location = Vector((c[0], c[1] - 5.0, c[2]))          # in front of figure (figure faces -Y)
else:
    cam.location = Vector((c[0] + 1.2, c[1] - 2.2, c[2] + 3.2))
cam.rotation_euler = (Vector(c) - cam.location).to_track_quat('-Z', 'Y').to_euler()
cam_data.lens = 35
sc.render.engine = 'BLENDER_WORKBENCH'; sc.display.shading.color_type = 'MATERIAL'
sc.render.resolution_x, sc.render.resolution_y = 800, 800
out = os.path.join(os.path.dirname(__file__), '..', 'out', 't01_' + mode.replace(':', '-').replace('+', '_') + '.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
sc.render.filepath = out
bpy.ops.render.render(write_still=True)
# report: which side is the L arm on in world X and in the image
for s in mk.SITES:
    idx = [i for i, x in enumerate(vsite) if x == s]
    if idx: print(s, 'n', len(idx), 'world mean', P[idx].mean(0).round(2))
    else: print(s, 'n 0')
print('verts', len(ob.data.vertices), 'faces', len(ob.data.polygons))
