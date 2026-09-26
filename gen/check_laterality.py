"""Independent laterality check (Blender env).
Rebuilds each scene's posed manikin from its params, places the recorded camera, projects bone joints that sit
inside each limb (mid upper arm, elbow, mid forearm / mid thigh, knee, mid shin), and reads the ID-pass colour at
those pixels. Each projected point whose pixel is a limb colour must be the colour of the bone's own side+limb.
Points landing on torso/occluder/floor pixels are skipped (occluded or at the silhouette edge)."""
import sys, os, json, glob, math
sys.path.insert(0, os.path.dirname(__file__))
import bpy, numpy as np
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view
import manikin as mk

D = sys.argv[-1]
COL = {'LUE': (255, 0, 0), 'RUE': (0, 255, 0), 'LLE': (0, 0, 255), 'RLE': (255, 255, 0)}
PROBES = {'UE': [('upperarm01', 0.5), ('upperarm02', 0.9), ('lowerarm01', 0.5)],
          'LE': [('upperleg01', 0.6), ('upperleg02', 0.9), ('lowerleg01', 0.5)]}
tot = agree = occl = 0; bad = []
for sc_path in sorted(glob.glob(os.path.join(D, 'C*_sidecar.json'))):
    s = json.load(open(sc_path)); p = s['params']
    bpy.ops.wm.read_factory_settings(use_empty=True); sc = bpy.context.scene
    ob, arm, _, _ = mk.build()
    cuts = {}
    for site, lvl in p['amputations'].items():
        mk.amputate(ob, arm, site, lvl); cuts[site] = lvl
    mk.limb_pose(arm, p['limb_pose']); mk.body_position(arm, p['body_position'])
    arm.rotation_euler[2] += math.radians(p['body_yaw']); bpy.context.view_layer.update(); mk.ground(ob, arm)
    cd = bpy.data.cameras.new('c'); cd.lens = 35; cd.sensor_width = 36
    cam = bpy.data.objects.new('c', cd); sc.collection.objects.link(cam); sc.camera = cam
    cam.location = Vector(s['camera']['location'])
    cam.rotation_euler = (Vector(s['camera']['target']) - cam.location).to_track_quat('-Z', 'Y').to_euler()
    sc.render.resolution_x, sc.render.resolution_y = 640, 480
    bpy.context.view_layer.update()
    idm = np.array(bpy.data.images.load(os.path.join(D, s['scene_id'] + '_id.png')).pixels[:]).reshape(480, 640, 4)[::-1, :, :3]
    idm = np.round(idm * 255).astype(int)
    for site in mk.SITES:
        side, limb = site[0], site[1:]
        for bone, frac in PROBES[limb]:
            if site in cuts:
                continue                        # distal bones of amputated limbs no longer carry geometry
            pb = arm.pose.bones[bone + '.' + side]
            w = arm.matrix_world @ (pb.head.lerp(pb.tail, frac))
            v = world_to_camera_view(sc, cam, w)
            if not (0 <= v.x < 1 and 0 <= v.y < 1 and v.z > 0):
                continue
            # occlusion test: ray from camera to the point; skip if anything is hit well before it
            dg = bpy.context.evaluated_depsgraph_get()
            dirv = (w - cam.location); dist = dirv.length
            hit_ok, loc, *_ = sc.ray_cast(dg, cam.location, dirv.normalized(), distance=dist)
            if hit_ok and (loc - cam.location).length < dist - 0.12:
                occl += 1
                continue
            px = idm[int((1 - v.y) * 480), int(v.x * 640)]
            hit = [k for k, c in COL.items() if tuple(px) == c]
            if not hit:
                continue
            tot += 1
            if hit[0] == site:
                agree += 1
            else:
                bad.append((s['scene_id'], site, bone, hit[0]))
print(f'LATERALITY CHECK: {agree}/{tot} unoccluded projected limb points land on their own limb colour ({occl} occluded points skipped)')
for b in bad:
    print('MISMATCH', b)
