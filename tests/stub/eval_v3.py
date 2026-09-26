import numpy as np
C4 = ['no_injury', 'wound', 'amputation', 'not_testable']
RING = ['BG', 'OCC', 'TORSO', 'OTHER_LIMB', 'EDGE']
def feat(r):
    pe = r['p_end'] or [0.0, 0.0, 0.0]; pw = r['p_wound'][1] if r['p_wound'] else 0.0
    end = r['end']; tot = sum(end.values()) or 1
    return [np.log1p(r['vis_px']), r['Lfrac'], r['bg_frac'] if r['bg_frac'] is not None else -1, *pe,
            np.log1p(r['ext_px']), np.log1p(r['stump_px']), *[end.get(q, 0) / tot for q in RING],
            r['vis_px'] / r['torso_ext'] ** 2, float(r['p_end'] is not None), pw, np.log1p(r['wound_px']),
            r['wound_px'] / max(r['vis_px'], 1), np.log1p(r['tq_px'])]


