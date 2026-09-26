"""Phase P2 generator fidelity (Blender env): wound moulage, clothing, tourniquets.

All three are built from the body mesh in its rest pose, so they deform with the armature exactly like the body.
  wounds      faces of the body near a sampled point on one limb get a wound material and the part label
              <side>_wound. Types: laceration (capsule along the limb), penetrating (small disc),
              burn (large irregular patch), open_fracture (laceration with a pale bone-coloured core).
  garments    copies of the body faces for the covered parts, pushed out along the normals (5-12 mm), with a
              cloth material. Stump and exposed-wound faces are left uncovered (medics cut clothing away);
              with probability `hidden_wound_p` a wound stays under the cloth. Garment faces carry the part
              label of the body part beneath, so a clothed limb still counts as limb.
  tourniquets a band of limb faces 3-4 cm wide, pushed out ~15 mm, on the upper arm or thigh; part label TQ.
Parameters (prm): wounds {site: type}, garments {top, bottom, boots}, tourniquets [site], hidden_wound_p.
"""
import math
import bpy, bmesh
import numpy as np
from mathutils import Vector
import manikin as mk

LIMB_PARTS = {'UE': ('upper_arm', 'forearm', 'hand'), 'LE': ('thigh', 'shank', 'foot')}
WOUND_TYPES = ['laceration', 'penetrating', 'burn', 'open_fracture']
CLOTH_COLOURS = [((0.22, 0.24, 0.16), (0.14, 0.15, 0.10)),     # olive
                 ((0.30, 0.26, 0.18), (0.20, 0.17, 0.12)),     # coyote
                 ((0.10, 0.12, 0.16), (0.05, 0.06, 0.08)),     # navy / dark
                 ((0.35, 0.35, 0.33), (0.22, 0.22, 0.21)),     # grey
                 ((0.18, 0.20, 0.14), (0.28, 0.24, 0.16))]     # mixed camo tones
SITE_ID = {'TORSO': 0, 'LUE': 1, 'RUE': 2, 'LLE': 3, 'RLE': 4}
TQ_COLOURS = [(0.02, 0.02, 0.02), (0.30, 0.25, 0.16), (0.16, 0.18, 0.10)]     # black, tan, olive


def _noise_mat(name, c1, c2, scale, rough=0.8, bump=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; bsdf = nt.nodes['Principled BSDF']
    nz = nt.nodes.new('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = scale
    ramp = nt.nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].color = (*c1, 1); ramp.color_ramp.elements[1].color = (*c2, 1)
    nt.links.new(nz.outputs['Fac'], ramp.inputs['Fac']); nt.links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
    bsdf.inputs['Roughness'].default_value = rough
    if bump > 0:
        b = nt.nodes.new('ShaderNodeBump'); b.inputs['Strength'].default_value = bump
        nt.links.new(nz.outputs['Fac'], b.inputs['Height']); nt.links.new(b.outputs['Normal'], bsdf.inputs['Normal'])
    m.diffuse_color = (*c1, 1)
    return m


def wound_material(kind, rng):
    if kind == 'burn':
        return _noise_mat('w_burn', (0.06, 0.04, 0.03), (0.55, 0.18, 0.12), rng.uniform(15, 40), 0.7, 0.6)
    if kind == 'open_fracture':
        return _noise_mat('w_frac', (0.35, 0.02, 0.02), (0.80, 0.74, 0.62), rng.uniform(20, 40), 0.35, 0.8)
    return _noise_mat('w_' + kind, (0.20, 0.01, 0.01), (0.55, 0.05, 0.04), rng.uniform(30, 80), 0.3, 0.8)


def _seg_dist(p, a, b):
    ab = b - a; t = max(0.0, min(1.0, (p - a).dot(ab) / max(ab.length_squared, 1e-9)))
    return (p - (a + ab * t)).length, t


def add_wounds(ob, arm, fsite, fpart, prm, rng):
    """Mutates fpart (wound faces -> <side>_wound). Returns {site: {'type', 'n_faces'}}."""
    out = {}
    for site, kind in prm.get('wounds', {}).items():
        limb = site[1:]; sd = site[0]
        pts = mk.chain_points(arm, site)
        cand = [i for i, (s, fp) in enumerate(zip(fsite, fpart)) if s == site and fp[2:] in LIMB_PARTS[limb][:2]]
        surf = (prm.get('wound_surface') or {}).get(site)      # vNext: 'front' or 'back' of the body (rest pose, forward = -Y)
        if surf:
            fc = [i for i in cand if (ob.data.polygons[i].normal.y < 0) == (surf == 'front')]
            cand = fc or cand
        if not cand:
            continue
        centres = {i: ob.data.polygons[i].center.copy() for i in cand}
        c0 = centres[cand[int(rng.integers(len(cand)))]]
        seg = 0 if (c0 - pts[0]).length < (c0 - pts[1]).length else 1
        axis = (pts[seg + 1] - pts[seg]).normalized()
        if kind == 'laceration' or kind == 'open_fracture':
            L = rng.uniform(0.08, 0.16); r = rng.uniform(0.018, 0.028)
            a, b = c0 - axis * L / 2, c0 + axis * L / 2
            sel = [i for i in cand if _seg_dist(centres[i], a, b)[0] < r]
        elif kind == 'penetrating':
            r = rng.uniform(0.020, 0.032); sel = [i for i in cand if (centres[i] - c0).length < r]
        else:   # burn: irregular patch = union of a few discs
            sel = set()
            for _ in range(int(rng.integers(3, 6))):
                cc = centres[cand[int(rng.integers(len(cand)))]] if rng.random() < 0.3 else c0 + Vector(rng.normal(0, 0.03, 3))
                rr = rng.uniform(0.05, 0.09)
                sel |= {i for i in cand if (centres[i] - cc).length < rr}
            sel = sorted(sel)
        if len(sel) < 6:            # coarse mesh: guarantee a visible patch (nearest faces to the centre)
            sel = sorted(cand, key=lambda i: (centres[i] - c0).length)[:6]
        ob.data.materials.append(wound_material(kind, rng)); mi = len(ob.data.materials) - 1
        for i in sel:
            ob.data.polygons[i].material_index = mi
            fpart[i] = site + '_wound'
        out[site] = {'type': kind, 'n_faces': len(sel)}
    return out


def _shell(ob, keep, labels, offset, mat, name, sites=None):
    """Copy of the body keeping faces in `keep`, pushed out along vertex normals; face attributes
    'p3' = part label id, 'st' = site id (0 TORSO, 1-4 LUE RUE LLE RLE)."""
    g = ob.copy(); g.data = ob.data.copy(); g.name = name
    bpy.context.scene.collection.objects.link(g)
    bm = bmesh.new(); bm.from_mesh(g.data); bm.faces.ensure_lookup_table()
    lay = bm.faces.layers.int.new('p3'); lst = bm.faces.layers.int.new('st')
    for f in bm.faces:
        f[lay] = labels[f.index]
        f[lst] = sites[f.index] if sites is not None else 0
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.index not in keep], context='FACES')
    bm.normal_update()
    for v in bm.verts:
        v.co += v.normal * offset
    bm.to_mesh(g.data); bm.free()
    g.data.materials.clear(); g.data.materials.append(mat)
    for p in g.data.polygons:
        p.material_index = 0; p.use_smooth = True
    return g


def add_garments(ob, arm, fsite, fpart, prm, rng, label_id):
    """Returns (list of garment objects, list of per-face label lists as stored in 'p3')."""
    gp = prm.get('garments', {})
    if not gp:
        return []
    hip_z = min(arm.data.bones['upperleg01.L'].head_local.z, arm.data.bones['upperleg01.R'].head_local.z)
    waist = hip_z + 0.10
    ankle_z = {s: arm.data.bones['foot.' + s].head_local.z for s in 'LR'}
    hidden_p = prm.get('hidden_wound_p', 0.2)
    wound_hidden = {s: rng.random() < hidden_p for s in 'LR'}
    centres = [p.center.z for p in ob.data.polygons]
    labels = [label_id[fp] for fp in fpart]
    site_ids = [SITE_ID[s] for s in fsite]
    col = CLOTH_COLOURS[int(rng.integers(len(CLOTH_COLOURS)))]
    top, bottom, boots = gp.get('top', 'none'), gp.get('bottom', 'none'), gp.get('boots', False)

    def covered(i):
        fp = fpart[i]; part = fp[2:] if fp[:2] in ('L_', 'R_') else fp
        if part == 'stump':
            return None
        if fp.endswith('_wound'):
            if not wound_hidden[fp[0]]:
                return None
            return 'top' if fp[1] == 'U' else 'bottom'
        if fp.startswith('TORSO'):
            return 'top' if centres[i] > waist else 'bottom'
        if part in ('upper_arm',):
            return 'top' if top in ('long', 'short') else None
        if part == 'forearm':
            return 'top' if top == 'long' else None
        if part == 'thigh':
            return 'bottom' if bottom in ('long', 'shorts') else None
        if part == 'shank':
            if boots and centres[i] < ankle_z[fp[0]] + 0.14:
                return 'boots'
            return 'bottom' if bottom == 'long' else None
        if part == 'foot':
            return 'boots' if boots else None
        return None

    groups = {'top': [], 'bottom': [], 'boots': []}
    for i in range(len(fpart)):
        g = covered(i)
        if g == 'top' and top == 'none':
            g = None
        if g == 'bottom' and bottom == 'none':
            g = None
        if g:
            groups[g].append(i)
    out = []
    for g, faces in groups.items():
        if not faces:
            continue
        if g == 'boots':
            mat = _noise_mat('boots', (0.05, 0.04, 0.03), (0.12, 0.10, 0.07), 60, 0.5)
            off = rng.uniform(0.010, 0.016)
        else:
            mat = _noise_mat('cloth_' + g, col[0], col[1], rng.uniform(8, 30), 0.9, 0.2)
            off = rng.uniform(0.005, 0.012)
        out.append(_shell(ob, set(faces), labels, off, mat, 'garment_' + g, site_ids))
    return out


TQ_BANDS = {}


def add_tourniquets(ob, arm, fsite, fpart, prm, rng, label_id):
    out = []
    for site in prm.get('tourniquets', []):
        pts = mk.chain_points(arm, site)
        a, b = pts[0], pts[1]
        t0 = rng.uniform(0.25, 0.45); w = rng.uniform(0.03, 0.04)
        keep = set()
        for i, s in enumerate(fsite):
            if s != site:
                continue
            d, t = _seg_dist(ob.data.polygons[i].center, a, b)
            if abs(t - t0) * (b - a).length < w / 2 and 0 < t < 1:
                keep.add(i)
        TQ_BANDS[site] = {'t0': float(t0), 'w': float(w)}
        if not keep:
            continue
        c = TQ_COLOURS[int(rng.integers(len(TQ_COLOURS)))]
        mat = _noise_mat('tq', c, tuple(min(1, x * 1.4 + 0.02) for x in c), 50, 0.6)
        out.append(_shell(ob, keep, [label_id['TQ']] * len(fpart), rng.uniform(0.012, 0.018), mat, 'tq_' + site,
                          [SITE_ID[site]] * len(fpart)))
    return out
