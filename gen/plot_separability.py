"""Strip plot: RTMW confidence on the hand/foot keypoint, by true extremity state (pose env).
Usage: python plot_separability.py <groups.json> <out.png> <n_scenes>"""
import sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

g = json.load(open(sys.argv[1])); out = sys.argv[2]; nsc = sys.argv[3]
order = ['intact, extremity visible', 'intact, extremity hidden or out of frame',
         'amputated, stump visible (extremity absent)']
labels = ['Intact limb,\nhand/foot visible', 'Intact limb,\nhand/foot hidden or cropped', 'Amputated limb,\nhand/foot absent']
SURF, INK, INK2, BLUE, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#2a78d6', '#e6e5e0'
fig, ax = plt.subplots(figsize=(8.6, 3.9), dpi=160)
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
rng = np.random.default_rng(0)
for i, k in enumerate(order):
    v = np.array(g[k]); y = np.full(len(v), len(order) - 1 - i) + rng.uniform(-0.16, 0.16, len(v))
    ax.scatter(v, y, s=36, color=BLUE, alpha=0.75, edgecolor=SURF, linewidth=1.2, zorder=3)
    med = np.median(v)
    ax.plot([med, med], [len(order) - 1 - i - 0.3, len(order) - 1 - i + 0.3], color=INK, linewidth=2, zorder=4)
    ax.text(1.005, len(order) - 1 - i, f'median {med:.2f}\nn = {len(v)}', va='center', ha='left', fontsize=8.5,
            color=INK2, transform=ax.get_yaxis_transform())
ax.set_yticks(range(len(order))); ax.set_yticklabels(labels[::-1], fontsize=9, color=INK)
ax.set_xlim(0, 1); ax.set_ylim(-0.6, len(order) - 0.4)
ax.set_xlabel('RTMW keypoint confidence on the hand / foot', fontsize=9, color=INK2)
ax.grid(axis='x', color=GRID, linewidth=0.8, zorder=0)
for s in ['top', 'right', 'left']:
    ax.spines[s].set_visible(False)
ax.spines['bottom'].set_color(GRID); ax.tick_params(colors=INK2, labelsize=8.5, length=0)
ax.set_title(f'Missing and hidden hands/feet get the same confidence ({nsc} synthetic scenes)',
             fontsize=10.5, color=INK, loc='left', pad=10)
plt.tight_layout(rect=(0, 0, 0.9, 1))
plt.savefig(out, facecolor=SURF)
print('saved', out)
