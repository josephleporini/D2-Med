"""Hand-check contact sheet (pose env). For each scene: realistic render with each site's name placed at the
centroid of its visible pixels (from the ID pass), ID pass inset, and the headline labels + visible fractions.
Also writes a review CSV with blank reviewer columns."""
import sys, os, json, glob, csv
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

D = sys.argv[1]
ID_COLORS = {'LUE': (255, 0, 0), 'RUE': (0, 255, 0), 'LLE': (0, 0, 255), 'RLE': (255, 255, 0)}
TXT = {'LUE': (60, 110, 255), 'RUE': (255, 70, 70), 'LLE': (40, 220, 255), 'RLE': (255, 160, 0)}
try:
    F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 22)
    f = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
except Exception:
    F = f = ImageFont.load_default()

tiles, rows = [], []
for sc in sorted(glob.glob(os.path.join(D, 'C*_sidecar.json'))):
    s = json.load(open(sc)); sid = s['scene_id']; p = s['params']
    rgb = Image.open(os.path.join(D, sid + '.jpg')).convert('RGB')
    idm = np.array(Image.open(os.path.join(D, sid + '_id.png')).convert('RGB'))
    W, H = 800, 600
    rgb = rgb.resize((W, H)); d = ImageDraw.Draw(rgb)
    sx, sy = W / idm.shape[1], H / idm.shape[0]
    lab = s['labels_by_threshold']['0.10']
    for site, col in ID_COLORS.items():
        m = np.all(idm == np.array(col), axis=2)
        if m.sum() > 0:
            ys, xs = np.where(m); x, y = xs.mean() * sx, ys.mean() * sy
            d.text((x - 20, y - 12), site, font=F, fill=TXT[site], stroke_width=3, stroke_fill=(0, 0, 0))
    inset = Image.fromarray(idm).resize((200, 150), Image.NEAREST)
    rgb.paste(inset, (W - 204, 4)); d.rectangle([W - 204, 4, W - 4, 154], outline=(255, 255, 255))
    cap = Image.new('RGB', (W, 118), (255, 255, 255)); c = ImageDraw.Draw(cap)
    amp = ', '.join(f'{k} {v}' for k, v in p['amputations'].items()) or 'none'
    c.text((8, 4), f"{sid}  {p['body_position']} / {p['limb_pose']} / cam az {p['azimuth']} el {p['elevation']} / {p['framing']}",
           font=f, fill=(0, 0, 0))
    c.text((8, 24), f"amputation: {amp}   occluder: {p['occluder']}   {p['skin']} / {p['floor']} / {p['lighting']}",
           font=f, fill=(0, 0, 0))
    vf = s['visible_fraction']
    for j, site in enumerate(['LUE', 'RUE', 'LLE', 'RLE']):
        c.text((8 + j * 196, 52), f"{site}: {lab[site]}", font=f, fill=(0, 0, 0) if lab[site] == 'no_injury' else (190, 0, 0))
        c.text((8 + j * 196, 72), f"visible {vf[site] * 100:.0f}%", font=f, fill=(80, 80, 80))
    c.text((8, 94), "Check: is each name on the correct anatomical limb? Is each label right?", font=f, fill=(0, 90, 160))
    t = Image.new('RGB', (W, H + 118), (255, 255, 255)); t.paste(rgb, (0, 0)); t.paste(cap, (0, H))
    tiles.append(t)
    rows.append([sid, p['body_position'], p['limb_pose'], p['azimuth'], p['elevation'], p['framing'], p['occluder'], amp] +
                [lab[k] for k in ['LUE', 'RUE', 'LLE', 'RLE']] + [round(vf[k], 3) for k in ['LUE', 'RUE', 'LLE', 'RLE']] +
                ['', '', ''])

# PDF, 2 tiles per page
pages = []
for i in range(0, len(tiles), 2):
    pg = Image.new('RGB', (tiles[0].width, tiles[0].height * 2 + 20), (255, 255, 255))
    for j, t in enumerate(tiles[i:i + 2]):
        pg.paste(t, (0, j * (t.height + 20)))
    pages.append(pg)
pdf = os.path.join(D, 'Probe_B_Check20_Contact_Sheet.pdf')
pages[0].save(pdf, save_all=True, append_images=pages[1:], resolution=100)
with open(os.path.join(D, 'Probe_B_Check20_Label_Review.csv'), 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['scene', 'position', 'limb_pose', 'cam_az', 'cam_el', 'framing', 'occluder', 'amputation',
                'LUE_label', 'RUE_label', 'LLE_label', 'RLE_label', 'LUE_vis', 'RUE_vis', 'LLE_vis', 'RLE_vis',
                'reviewer1_agree(Y/N)', 'reviewer2_agree(Y/N)', 'notes'])
    w.writerows(rows)
# small overview grid for chat
thumbs = [t.resize((400, 359)) for t in tiles]
while len(thumbs) % 4: thumbs.append(Image.new('RGB', (400, 359), (255, 255, 255)))
grid = Image.new('RGB', (1600, 359 * (len(thumbs) // 4)), (255, 255, 255))
for i, t in enumerate(thumbs):
    grid.paste(t, ((i % 4) * 400, (i // 4) * 359))
grid.save(os.path.join(D, 'Probe_B_Check20_overview.jpg'), quality=88)
print('pages', len(pages), 'scenes', len(tiles))
