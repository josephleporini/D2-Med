"""Ground-truth body frame per scene (Blender env). Usage: python3 gt_frame.py <scene_dir>
Writes <sid>_gtframe.json:
  shoulders_mid, hips_mid, head        - image points (1280x960)
  axis                                 - unit image vector hips_mid -> shoulders_mid (torso axis, head-ward)
  facing_cos                           - cos(angle) between the body's front normal and the direction to the camera
                                         (>0: front toward camera, <0: back toward camera, ~0: edge-on)
  left_dir                             - unit image vector toward the figure's anatomical left,
                                         = sign(facing) * perp(axis), perp(a) = (-a_y, a_x) in image coords (y down)
  check_shoulders / check_hips         - projection of (L - R) joint onto left_dir (should be > 0 when not edge-on)
"""
import sys, os, json, glob, math
sys.path.insert(0, os.path.dirname(__file__))
import bpy, numpy as np
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view
import manikin as mk

W, H = 1280, 960
D = sys.argv[-1]
for sc_path in sorted(glob.glob(os.path.join(D, 'C*_sidecar.json'))):
    s = json.load(open(sc_path)); sid = s['scene_id']; p = s['params']
    out = os.path.join(D, sid + '_gtframe.json')
    if os.path.exists(out):
        continue
    bpy.ops.wm.read_factory_settings(use_empty=True); sc = bpy.context.scene
    ob, arm, _, _ = mk.build()
    mk.limb_pose(arm, p['limb_pose']); mk.body_position(arm, p['body_position'])
    arm.rotation_euler[2] += math.radians(p['body_yaw']); bpy.context.view_layer.update(); mk.ground(ob, arm)
    cd = bpy.data.cameras.new('c'); cd.lens = 35; cd.sensor_width = 36
    cam = bpy.data.objects.new('c', cd); sc.collection.objects.link(cam); sc.camera = cam
    cam.location = Vector(s['camera']['location'])
    cam.rotation_euler = (Vector(s['camera']['target']) - cam.location).to_track_quat('-Z', 'Y').to_euler()
    sc.render.resolution_x, sc.render.resolution_y = W, H
    bpy.context.view_layer.update()
    wp = lambda b: arm.matrix_world @ arm.pose.bones[b].head
    def img(w):
        v = world_to_camera_view(sc, cam, w); return np.array([v.x * W, (1 - v.y) * H])
    Ls, Rs, Lh, Rh = wp('upperarm01.L'), wp('upperarm01.R'), wp('upperleg01.L'), wp('upperleg01.R')
    sm_w, hm_w = (Ls + Rs) / 2, (Lh + Rh) / 2
    sm, hm = img(sm_w), img(hm_w)
    a = sm - hm; a = a / (np.linalg.norm(a) + 1e-9)
    front = (arm.matrix_world.to_3x3() @ Vector((0, -1, 0))).normalized()
    chest = arm.matrix_world @ arm.pose.bones['spine03'].head
    to_cam = (cam.location - chest).normalized()
    fcos = float(front.dot(to_cam))
    left = np.sign(fcos) * np.array([-a[1], a[0]])
    res = {'scene_id': sid, 'body_position': p['body_position'], 'shoulders_mid': sm.round(1).tolist(),
           'hips_mid': hm.round(1).tolist(), 'head': img(wp('head')).round(1).tolist(), 'axis': a.round(4).tolist(),
           'facing_cos': round(fcos, 4), 'left_dir': left.round(4).tolist(),
           'check_shoulders': round(float((img(Ls) - img(Rs)) @ left), 1),
           'check_hips': round(float((img(Lh) - img(Rh)) @ left), 1)}
    json.dump(res, open(out, 'w'), indent=1)
    print('GTF', sid, round(fcos, 2), res['check_shoulders'], res['check_hips'])
