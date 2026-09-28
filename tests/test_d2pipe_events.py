"""run_folder in gen/d2pipe.py: predictions.json is generated from the events, events validate, and a failed or
unreadable image still yields a record (D-09). The model stack is stubbed (no torch, no weights), so this tests the
folder loop and the M12 path only, not the engine."""
import os, sys, json, types
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, '..', 'gen')
cv2 = pytest.importorskip('cv2')


@pytest.fixture
def d2pipe(monkeypatch):
    stubs = {'torch': types.ModuleType('torch'), 'parts': types.ModuleType('parts'), 'test_a_masks': types.ModuleType('test_a_masks'),
             'test_e2': types.ModuleType('test_e2'), 'lend': types.ModuleType('lend'), 'seg_train2': types.ModuleType('seg_train2'),
             'seg_features': types.ModuleType('seg_features'), 'test_e': types.ModuleType('test_e')}
    stubs['test_e'].SITES = ['LUE', 'RUE', 'LLE', 'RLE']
    stubs['seg_train2'].Decoder2 = object; stubs['seg_features'].square_crop = None
    for k, v in stubs.items():
        monkeypatch.setitem(sys.modules, k, v)
    monkeypatch.syspath_prepend(GEN)
    sys.modules.pop('d2pipe', None)
    import d2pipe
    return d2pipe


class FakePipe:
    def predict(self, img):
        if img.shape[0] == 13:
            raise RuntimeError('engine failure')
        p = {'LUE': [0.1, 0.1, 0.7, 0.1], 'RUE': [0.7, 0.1, 0.1, 0.1], 'LLE': [0.1, 0.1, 0.1, 0.7], 'RLE': [0.1, 0.7, 0.1, 0.1]}
        self.last = {'probs': p, 'facing': 'back', 'facing_conf': 0.8}
        return {}


def test_run_folder_events(tmp_path, d2pipe):
    d = tmp_path / 'in'; d.mkdir()
    cv2.imwrite(str(d / 'Good A.JPG'), np.zeros((32, 32, 3), np.uint8))
    cv2.imwrite(str(d / 'fails.png'), np.zeros((13, 32, 3), np.uint8))
    (d / 'corrupt.jpg').write_bytes(b'not a jpeg')
    (d / 'notes.txt').write_text('ignored')
    ev = tmp_path / 'events.jsonl'
    res = d2pipe.run_folder(FakePipe(), str(d), events_out=str(ev))
    recs = {r['image_id']: {(s['body_region'], s['laterality']): s['injury_type'] for s in r['sites']} for r in res['predictions']}
    assert set(recs) == {'Good A.JPG', 'fails.png', 'corrupt.jpg'}
    assert recs['Good A.JPG'] == {('upper_extremity', 'left'): 'amputation', ('upper_extremity', 'right'): 'no_injury',
                                  ('lower_extremity', 'left'): 'not_testable', ('lower_extremity', 'right'): 'wound'}
    assert set(recs['fails.png'].values()) == {'no_injury'} and set(recs['corrupt.jpg'].values()) == {'no_injury'}
    logs = [json.loads(l) for l in open(ev)]
    assert len(logs) == 3
    sys.path.insert(0, os.environ.get('D2_VALIDATOR') or os.path.join(HERE, '..', '..', 'd2_dev', 'd2-voice'))
    V = pytest.importorskip('d2voice.validate_log')
    for log in logs:
        assert V.check(log) == []
    good = next(l for l in logs if l['sources'][0]['uri'] == 'Good A.JPG')
    assert good['events'][0]['value']['view_of'] == 'posterior'
    assert all(':fallback' in l['events'][1]['producer']['model_version'] for l in logs if l['sources'][0]['uri'] != 'Good A.JPG')
