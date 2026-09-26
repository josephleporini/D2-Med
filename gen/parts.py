"""Part-map decoding shared by training and evaluation (pose env)."""
import numpy as np
from PIL import Image

PART_NAMES = ['upper_arm', 'forearm', 'hand', 'thigh', 'shank', 'foot']
PART_CLASSES = ['TORSO', 'HEAD', 'OCC'] + [sd + '_' + p for sd in 'LR' for p in PART_NAMES]
_GRID = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1), (0.5, 0, 0), (0, 0.5, 0),
         (0, 0, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5), (1, 0.5, 0), (1, 0, 0.5)]
# training targets are side-agnostic: BG, TORSO, HEAD, OCC, upper_arm, forearm, hand, thigh, shank, foot
SEG_CLASSES = ['BG', 'TORSO', 'HEAD', 'OCC'] + PART_NAMES
LIMB_OF_PART = {'upper_arm': 'UE', 'forearm': 'UE', 'hand': 'UE', 'thigh': 'LE', 'shank': 'LE', 'foot': 'LE'}


def decode(path):
    """Returns (seg, side): seg in SEG_CLASSES indices; side 0 none / 1 left / 2 right."""
    a = np.array(Image.open(path).convert('RGB')).astype(int)
    q = np.where(a > 220, 2, np.where(a > 90, 1, 0))           # levels 0 / 0.5 / 1
    code = q[..., 0] * 9 + q[..., 1] * 3 + q[..., 2]
    seg = np.zeros(a.shape[:2], np.uint8); side = np.zeros(a.shape[:2], np.uint8)
    for name, col in zip(PART_CLASSES, _GRID):
        c = int(round(col[0] * 2)) * 9 + int(round(col[1] * 2)) * 3 + int(round(col[2] * 2))
        m = code == c
        if name in ('TORSO', 'HEAD', 'OCC'):
            seg[m] = SEG_CLASSES.index(name)
        else:
            sd, p = name.split('_', 1)
            seg[m] = SEG_CLASSES.index(p); side[m] = 1 if sd == 'L' else 2
    return seg, side


# ---------------- part-label v2 (scene2.py) ----------------
PART2_CLASSES = ['TORSO_F', 'TORSO_B', 'HEAD_F', 'HEAD_B', 'OCC'] + [sd + '_' + p for sd in 'LR' for p in PART_NAMES + ['stump']]
_G2 = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1), (0.5, 0, 0), (0, 0.5, 0),
       (0, 0, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5), (1, 0.5, 0), (1, 0, 0.5), (0.5, 1, 0), (0, 1, 0.5),
       (0.5, 0, 1), (0, 0.5, 1)]
SEG2_CLASSES = ['BG', 'TORSO_F', 'TORSO_B', 'HEAD_F', 'HEAD_B', 'OCC'] + PART_NAMES + ['stump']


def decode2(path):
    """v2 part map -> (seg in SEG2_CLASSES, side 0 none / 1 left / 2 right)."""
    a = np.array(Image.open(path).convert('RGB')).astype(int)
    q = np.where(a > 220, 2, np.where(a > 90, 1, 0))
    code = q[..., 0] * 9 + q[..., 1] * 3 + q[..., 2]
    seg = np.zeros(a.shape[:2], np.uint8); side = np.zeros(a.shape[:2], np.uint8)
    for name, col in zip(PART2_CLASSES, _G2):
        m = code == int(round(col[0] * 2)) * 9 + int(round(col[1] * 2)) * 3 + int(round(col[2] * 2))
        if name[:2] in ('L_', 'R_'):
            seg[m] = SEG2_CLASSES.index(name[2:]); side[m] = 1 if name[0] == 'L' else 2
        else:
            seg[m] = SEG2_CLASSES.index(name)
    return seg, side


# ---------------- part-label v3 (scene3.py): v2 + per-site wound + tourniquet ----------------
PART3_CLASSES = PART2_CLASSES + ['LUE_wound', 'RUE_wound', 'LLE_wound', 'RLE_wound', 'TQ']
_G3 = _G2 + [(1, 0.5, 1), (0.5, 1, 1), (1, 1, 0.5), (0.5, 0.5, 1), (1, 0.5, 0.5)]
SEG3_CLASSES = SEG2_CLASSES + ['wound', 'TQ']


def decode3(path):
    """v3 part map -> (seg in SEG3_CLASSES, side 0/1 left/2 right, site 0 none / 1 LUE / 2 RUE / 3 LLE / 4 RLE for wounds)."""
    a = np.array(Image.open(path).convert('RGB')).astype(int)
    q = np.where(a > 220, 2, np.where(a > 90, 1, 0))
    code = q[..., 0] * 9 + q[..., 1] * 3 + q[..., 2]
    seg = np.zeros(a.shape[:2], np.uint8); side = np.zeros(a.shape[:2], np.uint8); wsite = np.zeros(a.shape[:2], np.uint8)
    for name, col in zip(PART3_CLASSES, _G3):
        m = code == int(round(col[0] * 2)) * 9 + int(round(col[1] * 2)) * 3 + int(round(col[2] * 2))
        if name.endswith('_wound'):
            seg[m] = SEG3_CLASSES.index('wound'); side[m] = 1 if name[0] == 'L' else 2
            wsite[m] = ['LUE', 'RUE', 'LLE', 'RLE'].index(name[:3]) + 1
        elif name[:2] in ('L_', 'R_'):
            seg[m] = SEG3_CLASSES.index(name[2:]); side[m] = 1 if name[0] == 'L' else 2
        else:
            seg[m] = SEG3_CLASSES.index(name)
    return seg, side, wsite
