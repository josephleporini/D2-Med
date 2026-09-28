"""score/score.py must stop with an error when the split filter leaves zero rows (IPR 27 Sep, process note 2)."""
import os, sys, subprocess, glob
import pytest

pytest.importorskip('sklearn')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')


@pytest.fixture(scope='module')
def fx(tmp_path_factory):
    d = tmp_path_factory.mktemp('fx')
    subprocess.run([sys.executable, os.path.join(HERE, 'make_fixture.py'), str(d)], check=True, capture_output=True)
    return d


def score(fx, out, *extra):
    cmd = [sys.executable, os.path.join(ROOT, 'score', 'score.py'), '--phase', 't', '--gen', os.path.join(HERE, 'stub'),
           '--pred', str(fx / 'pred.jsonl'), '--sidecars', str(fx / 'scenes'), '--out', str(out), *extra]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_scores_when_rows_in_scope(fx, tmp_path):
    r = score(fx, tmp_path / 'ok', '--dev', 'dev3')
    assert r.returncode == 0, r.stderr[-500:]
    assert (tmp_path / 'ok' / 'metrics.json').exists()


def test_stops_on_wrong_dev_split(fx, tmp_path):
    r = score(fx, tmp_path / 'bad', '--dev', 'dev5')
    assert r.returncode != 0 and 'zero rows' in r.stderr and 'dev3' in r.stderr
    assert not (tmp_path / 'bad' / 'metrics.json').exists()


def test_stops_on_missing_eval_split(fx, tmp_path):
    r = score(fx, tmp_path / 'bad2', '--dev', 'dev3', '--eval', 'test5')
    assert r.returncode != 0 and 'zero rows for --eval test5' in r.stderr


def test_limb_features_and_label_rule_run(fx, tmp_path):
    r = score(fx, tmp_path / 'limb', '--dev', 'dev3', '--limb', '--label-rule', 'guide_primary')
    assert r.returncode == 0, r.stderr[-800:]
    import json
    m = json.load(open(tmp_path / 'limb' / 'metrics.json'))
    assert m['limb'] is True and m['label_rule'] == 'guide_primary' and m['dev3_oof']['n'] == 480
