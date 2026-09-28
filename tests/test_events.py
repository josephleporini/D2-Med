"""Tests for gen/events.py: schema v0.2.0 validity with the shared validator (D2_Dev d2voice/validate_log.py),
and exact round trip events -> qualification classes == argmax of the site probabilities.

The shared validator is found through D2_VALIDATOR (path to d2-voice) or the sibling clone ../d2_dev/d2-voice.
"""
import os, sys, json, random, copy
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'gen'))
import events as E

VDIR = os.environ.get('D2_VALIDATOR') or os.path.join(HERE, '..', '..', 'd2_dev', 'd2-voice')
sys.path.insert(0, VDIR)
try:
    from d2voice import validate_log as V
except ImportError:                                  # validator not present: validity tests are skipped, not passed
    V = None
need_v = pytest.mark.skipif(V is None, reason=f'shared validator not found at {VDIR}')


def onehot(k, hi=0.7):
    p = [(1 - hi) / 3] * 4; p[k] = hi; return p


def rand_probs(rng):
    out = {}
    for s in E.SITES:
        w = [rng.random() ** 3 for _ in range(4)]; t = sum(w); out[s] = [v / t for v in w]
    return out


@need_v
def test_every_class_valid_and_round_trips():
    for k, cls in enumerate(E.C4):
        probs = {s: onehot(k) for s in E.SITES}
        log = E.image_log('IMG 001.JPG', probs, model_version='test')
        assert V.check(log) == [], V.check(log)
        assert E.to_qual(log) == {s: cls for s in E.SITES}


@need_v
def test_random_posteriors_round_trip():
    rng = random.Random(7)
    for n in range(500):
        probs = rand_probs(rng)
        log = E.image_log(f'scene_{n}.png', probs, model_version='test', facing=rng.choice([None, 'front', 'back', 'edge']),
                          facing_conf=rng.random(), bbox=[10, 20, 300.5, 400])
        assert V.check(log) == []
        q = E.to_qual(log)
        for s in E.SITES:
            assert q[s] == E.C4[max(range(4), key=lambda i: probs[s][i])]


def test_laterality_is_anatomical_site_key():
    probs = {s: onehot(0) for s in E.SITES}; probs['RLE'] = onehot(2)
    log = E.image_log('x.jpg', probs, model_version='t')
    amp = [e for e in log['events'] if e['type'] == 'injury' and e['assertion'] == 'present']
    assert len(amp) == 1 and amp[0]['body_site'] == {'region': 'lower_extremity', 'laterality': 'right'}


def test_not_testable_site_has_no_injury_event():
    probs = {s: onehot(0) for s in E.SITES}; probs['LUE'] = onehot(3)
    log = E.image_log('x.jpg', probs, model_version='t')
    lue = [e for e in log['events'] if e.get('body_site') == {'region': 'upper_extremity', 'laterality': 'left'}]
    assert [e['type'] for e in lue] == ['site_visibility'] and lue[0]['value'] == {'visible': False} and lue[0]['assertion'] == 'present'


def test_ids_legal_and_distinct():
    names = ['A B.jpg', 'a-b.jpg', 'Ünïcode é.PNG', 'x' * 300 + '.jpg', '...', '0.jpg']
    ids = [E.encounter_id(n) for n in names]
    assert len(set(ids)) == len(ids)
    assert all(E._ID.match(i) for i in ids)


def test_source_uri_keeps_exact_name():
    log = E.image_log('Weird Name (1).JPG', {s: onehot(0) for s in E.SITES}, model_version='t')
    assert log['sources'][0]['uri'] == 'Weird Name (1).JPG'
    assert E.qual_record('Weird Name (1).JPG', log)['image_id'] == 'Weird Name (1).JPG'


@pytest.mark.parametrize('bad', [[0.5, 0.5, 0.5, 0.0], [float('nan'), 0, 0, 1], [1, 0, 0], [-0.1, 0.6, 0.5, 0.0]])
def test_bad_distribution_rejected(bad):
    probs = {s: onehot(0) for s in E.SITES}; probs['LLE'] = bad
    with pytest.raises(ValueError):
        E.image_log('x.jpg', probs, model_version='t')


def test_qual_record_order_and_classes():
    rec = E.qual_record('x.jpg', E.image_log('x.jpg', {s: onehot(1) for s in E.SITES}, model_version='t'))
    assert [(r['body_region'], r['laterality']) for r in rec['sites']] == [E.SITE_BODY[s] for s in E.SITES]
    assert {r['injury_type'] for r in rec['sites']} == {'wound'}


@need_v
def test_supersede_and_retract_respected():
    log = E.image_log('x.jpg', {s: onehot(0) for s in E.SITES}, model_version='t')
    # a later correction: LUE not visible after all
    old = next(e for e in log['events'] if e['event_id'] == 'vis-lue')
    new = copy.deepcopy(old); new.update(event_id='vis-lue-2', supersedes='vis-lue', assertion='present', value={'visible': False})
    log['events'].append(new)
    # a retraction of the RLE injury event leaves RLE visible -> still no_injury (rule 4)
    inj = next(e for e in log['events'] if e['event_id'] == 'inj-rle')
    log['events'].append({**copy.deepcopy(inj), 'event_id': 'ret-1', 'type': 'retraction', 'retracts': 'inj-rle',
                          'value': {}, 'body_site': inj['body_site']})
    assert V.check(log) == [], V.check(log)
    q = E.to_qual(log)
    assert q['LUE'] == 'not_testable' and q['RLE'] == 'no_injury'


def test_facing_maps_to_view_of():
    for f, v in (('front', 'anterior'), ('back', 'posterior'), ('edge', 'oblique'), (None, 'unknown')):
        log = E.image_log('x.jpg', {s: onehot(0) for s in E.SITES}, model_version='t', facing=f, facing_conf=0.9)
        cs = log['events'][0]
        assert cs['value']['view_of'] == v
        assert cs['assertion'] == ('unknown' if f is None else 'present')
