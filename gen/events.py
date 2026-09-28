"""Casualty-event writer for qualification images (schema v0.2.0, D-01; gate G-O1 criterion).

One qualification image is one encounter with one casualty (c1) seen by one still-image source (img) at t = 0.
Per image the writer emits:
  casualty_state   orientation and view_of (facing). 'unknown' unless the engine supplies a predicted facing;
                   generator truth is never used here.
  site_visibility  one per limb site; visible = (the decision is not not_testable)
  injury           one per visible site: amputation (present), wound (present, injury_class 'other',
                   source_label 'wound'), or no_injury (absent, injury_class 'none')
Laterality is the casualty's anatomical side (S-15, D-04): the site keys are already anatomical.

to_qual() maps events back to the four ICD classes with the schema README rules (qualification mapping, M12):
  1 not visible -> not_testable  2 present amputation -> amputation  3 other present injury -> wound
  4 visible, or injury asserted absent -> no_injury  5 nothing known -> not_testable
Superseded and retracted events are dropped first. For events this module writes, to_qual(image_log(p)) equals the
argmax of p for every site; tests/test_events.py and tools/ledger_to_events.py check that.

Interpretations to be confirmed by the schema owner (speech thread), recorded in the events report:
  - site_visibility carries the result in value.visible; assertion is 'present' when visible, 'absent' when the site
    was looked for and not seen. The mapper keys on value.visible only.
  - A predicted wound has no finer class, so injury_class is 'other' with source_label 'wound'.
  - The 4-class posterior does not fit the event (V_Injury has no field for it); only the chosen class's probability
    goes in confidence. The full posterior stays in the run log (M3-13).
"""
import hashlib
import math
import re

SCHEMA_VERSION = '0.2.0'
SITES = ['LUE', 'RUE', 'LLE', 'RLE']                    # ICD site order UE-L, UE-R, LE-L, LE-R (IF-3)
SITE_BODY = {'LUE': ('upper_extremity', 'left'), 'RUE': ('upper_extremity', 'right'),
             'LLE': ('lower_extremity', 'left'), 'RLE': ('lower_extremity', 'right')}
BODY_SITE = {v: k for k, v in SITE_BODY.items()}
C4 = ['no_injury', 'wound', 'amputation', 'not_testable']  # probability order, as in score/score.py
VIEW = {'front': 'anterior', 'back': 'posterior', 'edge': 'oblique', 'edge_on': 'oblique'}
ORIENTATIONS = {'supine', 'prone', 'left_lateral', 'right_lateral', 'sitting', 'standing', 'unknown'}
_ID = re.compile(r'^[a-z0-9][a-z0-9._:-]{0,63}$')


def encounter_id(image_id):
    """Schema Id from a file name: lower case, illegal characters to '-', plus a short hash so that two names that
    sanitise alike never collide. The exact file name is kept in the source uri."""
    stem = re.sub(r'[^a-z0-9._-]+', '-', image_id.lower()).strip('-._') or 'img'
    eid = f"{stem[:40]}-{hashlib.sha1(image_id.encode('utf-8')).hexdigest()[:8]}"
    assert _ID.match(eid), eid
    return eid


def _check_probs(probs):
    for s in SITES:
        p = probs[s]
        if len(p) != 4 or not all(math.isfinite(v) and v >= 0 for v in p) or abs(sum(p) - 1) > 1e-3:
            raise ValueError(f'site {s}: not a 4-class distribution: {p}')


def image_log(image_id, probs, *, model_version, model_id='probeB', facing=None, facing_conf=None,
              orientation='unknown', bbox=None, dataset='qual', scenario_id=None):
    """probs: {site: [p_no_injury, p_wound, p_amputation, p_not_testable]} for the four sites.
    facing: 'front' | 'back' | 'edge' | None (predicted, never truth). bbox: person box x1, y1, x2, y2 in pixels."""
    _check_probs(probs)
    if orientation not in ORIENTATIONS:
        raise ValueError(f'orientation {orientation}')
    enc = encounter_id(image_id)
    producer = {'module': 'M3', 'model_id': model_id, 'model_version': model_version}
    evd = {'source_id': 'img', 'frame_index': 0}
    if bbox is not None:
        evd['bbox_xyxy'] = [round(max(float(v), 0.0), 1) for v in bbox]

    def ev(eid, typ, assertion, value, conf, site=None):
        e = {'event_id': eid, 'casualty_id': 'c1', 'type': typ, 't_start': 0, 'assertion': assertion, 'value': value,
             'confidence': round(min(max(float(conf), 0.0), 1.0), 4), 'producer': dict(producer), 'evidence': [dict(evd)]}
        if site:
            reg, lat = SITE_BODY[site]
            e['body_site'] = {'region': reg, 'laterality': lat}
        return e

    view = VIEW.get(facing, 'unknown')
    known = view != 'unknown' or orientation != 'unknown'
    events = [ev('cs', 'casualty_state', 'present' if known else 'unknown',
                 {'orientation': orientation, 'view_of': view}, facing_conf if (known and facing_conf is not None) else 0.0)]
    for s in SITES:
        p = probs[s]; k = max(range(4), key=lambda i: p[i]); cls = C4[k]; sl = s.lower()
        visible = cls != 'not_testable'
        events.append(ev(f'vis-{sl}', 'site_visibility', 'present' if visible else 'absent', {'visible': visible},
                         1 - p[3] if visible else p[3], s))
        if cls == 'amputation':
            events.append(ev(f'inj-{sl}', 'injury', 'present', {'injury_class': 'amputation'}, p[2], s))
        elif cls == 'wound':
            events.append(ev(f'inj-{sl}', 'injury', 'present', {'injury_class': 'other', 'source_label': 'wound'}, p[1], s))
        elif cls == 'no_injury':
            events.append(ev(f'inj-{sl}', 'injury', 'absent', {'injury_class': 'none'}, p[0], s))
    enc_obj = {'encounter_id': enc, 'dataset': dataset, 'clock': {'t0_utc': None, 'time_designator': 'Z'}}
    if scenario_id:
        enc_obj['scenario_id'] = scenario_id
    return {'schema_version': SCHEMA_VERSION, 'encounter': enc_obj,
            'sources': [{'source_id': 'img', 'modality': 'image', 'mount': 'unknown', 'uri': image_id}],
            'casualties': [{'casualty_id': 'c1', 'first_seen_t': 0}],
            'events': events}


def to_qual(log, casualty_id='c1'):
    """Events -> {site: ICD class} with the README qualification mapping, after dropping superseded and retracted
    events. When several live events cover a site, the last one logged wins for visibility; any live present
    amputation beats any other present injury."""
    evs = log['events']
    dead = {e['supersedes'] for e in evs if e.get('supersedes')} | {e['retracts'] for e in evs if e.get('retracts')}
    live = [e for e in evs if e['event_id'] not in dead and e.get('casualty_id') == casualty_id]
    out = {}
    for s in SITES:
        reg, lat = SITE_BODY[s]
        here = [e for e in live if e.get('body_site', {}).get('region') == reg and e['body_site'].get('laterality') == lat]
        vis = [e for e in here if e['type'] == 'site_visibility']
        inj = [e for e in here if e['type'] == 'injury']
        visible = vis[-1]['value']['visible'] if vis else None
        if visible is False:
            out[s] = 'not_testable'
        elif any(e['assertion'] == 'present' and e['value']['injury_class'] == 'amputation' for e in inj):
            out[s] = 'amputation'
        elif any(e['assertion'] == 'present' and e['value']['injury_class'] != 'none' for e in inj):
            out[s] = 'wound'
        elif visible or any(e['assertion'] == 'absent' for e in inj):
            out[s] = 'no_injury'
        else:
            out[s] = 'not_testable'
    return out


def qual_record(image_id, log):
    """One predictions.json record (ICD §3.2) generated from the events, in the fixed site order."""
    q = to_qual(log)
    return {'image_id': image_id, 'sites': [{'body_region': SITE_BODY[s][0], 'laterality': SITE_BODY[s][1],
                                             'injury_type': q[s]} for s in SITES]}
