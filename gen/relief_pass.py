"""Relief pass (blanket oracle, stage 1): per-pixel depth, surface normal and hit class of the visible surface, by ray
casting the rebuilt scene (no render). Same scene and camera as scene4 for the same params (PYTHONHASHSEED=0).

  python3 relief_pass.py <params.json> <ref_dir> <out_dir>

Writes <out_dir>/<id>_relief.npz:
  depth   float16 (480, 640)  distance along the camera axis, metres; 0 where nothing is hit
  normal  int8 (480, 640, 3)  surface normal in camera coordinates (x right, y up, z toward the camera), x127
  hit     uint8 (480, 640)    0 nothing, 1 floor, 2 occluder, 3 body, garment or other
  height  float16 (480, 640)  world height of the visible surface above the floor plane (z), metres: a surface model
                              in the remote-sensing sense (floor = terrain, blanket = canopy)
and prints an alignment check against <ref_dir>/<id>_occ.png (the scene4 visible-surface owner map): the IoU of the
occluder class and of the floor class. Rays go through pixel centres of the 640 x 480 map grid used by all masks.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import bpy
from mathutils import Vector

W, H = 640, 480


def camera_rays(sc, cam, w=W, h=H):
    """world-space origin and unit directions for every pixel centre (row 0 = top of the image)"""
    sc.render.resolution_x, sc.render.resolution_y = w, h
    sc.render.pixel_aspect_x = sc.render.pixel_aspect_y = 1
    mw = cam.matrix_world
    fr = [mw @ v for v in cam.data.view_frame(scene=sc)]      # top-right, bottom-right, bottom-left, top-left
    tr, br, bl, tl = [np.array(v) for v in fr]
    o = np.array(mw.translation)
    u = (np.arange(w) + 0.5) / w; v = (np.arange(h) + 0.5) / h
    top = tl[None] + (tr - tl)[None] * u[:, None]; bot = bl[None] + (br - bl)[None] * u[:, None]
    P = top[None] + (bot - top)[None] * v[:, None, None]        # (h, w, 3)
    D = P - o; D /= np.linalg.norm(D, axis=-1, keepdims=True)
    fwd = -np.array(mw.to_3x3().col[2]); right = np.array(mw.to_3x3().col[0]); up = np.array(mw.to_3x3().col[1])
    return o, D, fwd, right, up


def cast(sc, o, D, fwd, right, up, classify):
    dg = bpy.context.evaluated_depsgraph_get()
    h, w = D.shape[:2]
    depth = np.zeros((h, w), np.float32); z = np.zeros((h, w), np.float32); nrm = np.zeros((h, w, 3), np.float32); hit = np.zeros((h, w), np.uint8)
    O = Vector(o)
    for y in range(h):
        for x in range(w):
            ok, loc, n, _, ob, _ = sc.ray_cast(dg, O, Vector(D[y, x]))
            if ok:
                p = np.array(loc) - o
                depth[y, x] = p @ fwd; z[y, x] = loc.z
                nn = np.array(n)
                nrm[y, x] = (nn @ right, nn @ up, -(nn @ fwd))
                hit[y, x] = classify(ob)
    return depth, nrm, hit, z


def main(prm_path, ref_dir, out_dir):
    import scene as S, scene3 as S3
    t0 = time.time()
    prm = json.load(open(prm_path)); rng = np.random.default_rng(prm['seed']); sid = prm['scene_id']
    ctx = S3.setup3(prm, rng)
    sc, cam = ctx['sc'], ctx['cam']
    S.place_camera(cam, ctx['P'], prm, rng)
    occ_names = {o.name for o in ctx['occ']}; floor_name = ctx['floor'].name
    classify = lambda ob: 1 if ob.name == floor_name else 2 if ob.name in occ_names else 3
    o, D, fwd, right, up = camera_rays(sc, cam)
    depth, nrm, hit, z = cast(sc, o, D, fwd, right, up, classify)
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, sid + '_relief.npz'), depth=depth.astype(np.float16),
                        normal=np.clip(np.round(nrm * 127), -127, 127).astype(np.int8), hit=hit,
                        height=z.astype(np.float16))
    res = {'scene': sid, 's': round(time.time() - t0, 1)}
    ref = os.path.join(ref_dir, sid + '_occ.png')
    if os.path.exists(ref):
        import numpy as _np
        from PIL import Image
        occ = _np.array(Image.open(ref))
        for name, a, b in (('occluder', hit == 2, occ == 2), ('floor', hit == 1, occ == 1)):
            u = (a | b).sum(); res['iou_' + name] = round(float((a & b).sum() / u), 4) if u else None
    print('RELIEF', json.dumps(res), flush=True)


if __name__ == '__main__':
    if os.environ.get('PYTHONHASHSEED') != '0':
        os.execve(sys.executable, [sys.executable] + sys.argv, dict(os.environ, PYTHONHASHSEED='0'))
    a = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    if a[0] == 'selftest':
        # camera math check without generator assets: ray-cast class map vs a flat-colour render of the same scene
        bpy.ops.wm.read_factory_settings(use_empty=True); sc = bpy.context.scene
        bpy.ops.mesh.primitive_plane_add(size=10); fl = bpy.context.object
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0.7, -0.4, 0.5)); cb = bpy.context.object
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.4, location=(-0.8, 0.5, 0.4)); sp = bpy.context.object
        for ob, col in ((fl, (0, 0, 1)), (cb, (1, 0, 0)), (sp, (0, 1, 0))):
            m = bpy.data.materials.new(ob.name); m.diffuse_color = (*col, 1); ob.data.materials.append(m)
        cd = bpy.data.cameras.new('c'); cam = bpy.data.objects.new('c', cd); sc.collection.objects.link(cam); sc.camera = cam
        cam.location = (2.5, -3.0, 2.6); d = Vector((0, 0, 0.3)) - cam.location
        cam.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
        if sc.world is None:
            sc.world = bpy.data.worlds.new('w')
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        sc.render.engine = 'BLENDER_WORKBENCH'; sc.display.shading.light = 'FLAT'; sc.display.shading.color_type = 'MATERIAL'
        sc.display.render_aa = 'OFF'; sc.view_settings.view_transform = 'Standard'; sc.render.dither_intensity = 0
        sc.world.color = (0, 0, 0)
        sc.render.resolution_x, sc.render.resolution_y = W, H; sc.render.resolution_percentage = 100
        p = '/tmp/relief_selftest.png'; sc.render.filepath = p; sc.render.image_settings.file_format = 'PNG'
        bpy.ops.render.render(write_still=True)
        img = bpy.data.images.load(p); A = np.array(img.pixels[:]).reshape(H, W, 4)[::-1, :, :3]
        ref = np.zeros((H, W), np.uint8); ref[A[..., 2] > 0.5] = 1; ref[A[..., 0] > 0.5] = 2; ref[A[..., 1] > 0.5] = 3
        o, D, fwd, right, up = camera_rays(sc, cam)
        depth, nrm, hit, _ = cast(sc, o, D, fwd, right, up, lambda ob: {fl.name: 1, cb.name: 2, sp.name: 3}[ob.name])
        agree = float((hit == ref).mean())
        top = nrm[hit == 2][:, 1].max(); print('SELFTEST pixel agreement', round(agree, 5), 'cube max normal up', round(float(top), 3),
                                               'depth range', round(float(depth[hit > 0].min()), 3), round(float(depth.max()), 3))
        assert agree > 0.995, agree
        print('SELFTEST_OK')
    else:
        main(a[0], a[1], a[2])
