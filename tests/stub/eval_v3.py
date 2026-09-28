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




LIMB_FEATURES = ['log_amodal_px', 'vis_over_amodal', 'term_peak', 'p_cause_intact_visible', 'p_cause_occluded',
                 'p_cause_out_of_frame', 'p_cause_amputated_visible', 'p_cause_amputated_hidden']


def limb_feat(r):
    l = r['limb']
    return [np.log1p(l['amodal_px']), l['vis_over_amodal'], l['term_peak'], *l['p_cause']]
