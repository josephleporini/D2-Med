"""Ground-truth 2D keypoints per scene (Blender env).
Usage: python3 gt_keypoints.py <scene_dir>   -> writes <sid>_gtkp.json for every sidecar lacking one.

Joints per site (bone heads, rest-pose skeleton posed exactly as rendered):
  UE: shoulder=upperarm01, elbow=lowerarm01, wrist=wrist, hand=finger3-1 (middle knuckle)
  LE: hip=upperleg01,  knee=lowerleg01,  ankle=foot,  foot=toe3-1
Flags per joint:
  exists   - False when removed by the amputation (see REMOVED)
  in_frame - projects inside the 1280x960 image and in front of the camera
  occluded - first body surface along the camera ray belongs to another site (self-occlusion), or the
             ID-pass pixel at the joint is an occluder (blanket, bag, strap, medic arm)
  visible  = exists and in_frame and not occluded
"""
import sys, os, json, glob, math
sys.path.insert(0, os.path.dirname(__file__))
import bpy, numpy as np
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view
import manikin as mk
from scene import face_sites

JOINTS = {'UE': [('shoulder', 'upperarm01'), ('elbow', 'lowerarm01'), ('wrist', 'wrist'), ('hand', 'finger3-1')],
          'LE': [('hip', 'upperleg01'), ('knee', 'lowerleg01'), ('ankle', 'foot'), ('foot', 'toe3-1')]}
REMOVED = {'disarticulation': {'elbow', 'wrist', 'hand', 'knee', 'ankle', 'foot'},
           'above_joint': {'elbow', 'wrist', 'hand', 'knee', 'ankle', 'foot'},
           'below_joint': {'wrist', 'hand', 'ankle', 'foot'},
           'distal': {'hand', 'foot'}}
OCC = (0, 255, 255)
W, H = 1280, 960

D = sys.argv[-1]
for sc_path in sorted(glob.glob(os.path.join(D, 'C*_sidecar.json'))):
    s = json.load(open(sc_path)); sid = s['scene_id']; p = s['params']
    out = os.path.join(D, sid + '_gtkp.json')
    if os.path.exists(out):
        continue
    bpy.ops.wm.read_factory_settings(use_empty=True); sc = bpy.context.scene
    ob, arm, _, _ = mk.build()
    for site, lvl in p['amputations'].items():
        mk.amputate(ob, arm, site, lvl)
    fs = face_sites(ob)
    mk.limb_pose(arm, p['limb_pose']); mk.body_position(arm, p['body_position'])
    arm.rotation_euler[2] += math.radians(p['body_yaw']); bpy.context.view_layer.update(); mk.ground(ob, arm)
    cd = bpy.data.cameras.new('c'); cd.lens = 35; cd.sensor_width = 36
    cam = bpy.data.objects.new('c', cd); sc.collection.objects.link(cam); sc.camera = cam
    cam.location = Vector(s['camera']['location'])
    cam.rotation_euler = (Vector(s['camera']['target']) - cam.location).to_track_quat('-Z', 'Y').to_euler()
    sc.render.resolution_x, sc.render.resolution_y = W, H
    bpy.context.view_layer.update()
    img = bpy.data.images.load(os.path.join(D, sid + '_id.png'))
    ih, iw = img.size[1], img.size[0]
    idm = np.round(np.array(img.pixels[:]).reshape(ih, iw, 4)[::-1, :, :3] * 255).astype(int)
    dg = bpy.context.evaluated_depsgraph_get()
    res = {'scene_id': sid, 'image_size': [W, H], 'sites': {}}
    for site in mk.SITES:
        side, limb = site[0], site[1:]
        removed = REMOVED.get(p['amputations'].get(site), set())
        js = {}
        for jn, bone in JOINTS[limb]:
            wpt = arm.matrix_world @ arm.pose.bones[bone + '.' + side].head
            v = world_to_camera_view(sc, cam, wpt)
            x, y = v.x * W, (1 - v.y) * H
            in_frame = bool(0 <= v.x < 1 and 0 <= v.y < 1 and v.z > 0)
            occluded = False
            if in_frame:
                dirv = wpt - cam.location
                hit, loc, nrm, fidx, hobj, _ = sc.ray_cast(dg, cam.location, dirv.normalized(), distance=dirv.length)
                if hit and hobj is not None and hobj.name == ob.name:
                    hs = fs[fidx] if fidx < len(fs) else 'TORSO'
                    ok = {site} | ({'TORSO'} if jn in ('shoulder', 'hip') else set())
                    if hs not in ok:
                        occluded = True
                px = idm[min(int(y * ih / H), ih - 1), min(int(x * iw / W), iw - 1)]
                if tuple(px) == OCC:
                    occluded = True
            exists = jn not in removed
            js[jn] = {'x': round(x, 1), 'y': round(y, 1), 'exists': exists, 'in_frame': in_frame,
                      'occluded': occluded, 'visible': bool(exists and in_frame and not occluded)}
        res['sites'][site] = js
    json.dump(res, open(out, 'w'), indent=1)
    print('GTKP', sid)
