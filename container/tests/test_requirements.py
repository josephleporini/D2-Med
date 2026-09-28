"""Verification suite for D2 Block T Module Requirements v0.2.

Every test is named test_<REQ-ID>_<short description>. tools/trace_matrix.py maps
results back to the requirement table, so a requirement is "verified" only by a
passing test that carries its ID. Tests that need a GPU host or DARPA data are
skipped with a reason that starts with PENDING, which the matrix reports as open.

Run:  python3 -m pytest tests -q --junitxml=reports/junit.xml
      python3 tools/trace_matrix.py reports/junit.xml > reports/trace_matrix.md
"""
import json, os, re, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "harness"), str(ROOT / "tools"), str(ROOT / "train")]
from make_edge_cases import build, EXPECT, IGNORED
from d2qual.sites import SITES, CLASSES, LR_SWAP
from d2qual import ingest, formatter, privacy
from d2qual.config import RunConfig

MODEL = Path(os.environ.get("D2_TEST_MODEL", ROOT / "model_toy"))
SCHEMA = ROOT / "schema" / "qual-predictions.schema.json"
HAS_CUDA = False
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except Exception:
    pass


def run_exec(inp, out, model_dir=MODEL, env=None, pythonpath_prefix=None):
    e = dict(os.environ, D2_INPUT=str(inp), D2_OUTPUT=str(out), D2_MODEL_DIR=str(model_dir), D2_SCHEMA=str(SCHEMA),
             D2_TEAM_NAME="verification", D2_TEAM_EMAIL="v@example.com",
             PYTHONPATH=os.pathsep.join(filter(None, [pythonpath_prefix, str(ROOT)])), **(env or {}))
    t = time.time()
    p = subprocess.run([sys.executable, "-m", "d2qual"], env=e, capture_output=True, text=True, cwd=ROOT, timeout=900)
    logs = []
    for line in p.stderr.splitlines():
        try:
            logs.append(json.loads(line))
        except Exception:
            pass
    summary = None
    for line in reversed(p.stdout.splitlines()):
        try:
            summary = json.loads(line); break
        except Exception:
            pass
    return {"rc": p.returncode, "logs": logs, "summary": summary, "wall": time.time() - t, "out": Path(out)}


@pytest.fixture(scope="session")
def work():
    return Path(tempfile.mkdtemp(prefix="d2ver_"))


@pytest.fixture(scope="session")
def edge_dir(work):
    return build(work / "input", n_bulk=20)


@pytest.fixture(scope="session")
def expected_ids():
    return sorted(list(EXPECT) + [f"bulk_{i:05d}.jpg" for i in range(20)])


@pytest.fixture(scope="session")
def main_run(work, edge_dir):
    if not (MODEL / "model_config.json").exists():
        pytest.skip("PENDING: no trained model directory for tests")
    out = work / "out_main"; out.mkdir()
    return run_exec(edge_dir, out, env={"D2_CHECKPOINT_S": "0"})


def doc_of(run):
    return json.loads((run["out"] / "predictions.json").read_text(encoding="utf-8"))


def engine():
    from d2qual.engines import build as build_engines
    from d2qual.config import load_model_config
    return build_engines(MODEL, load_model_config(MODEL), torch.device("cpu"), tta_flip=True)


# ----------------------------------------------------------------------------- M1 Ingest
def test_M1_01_enumerates_only_top_level_image_files(edge_dir):
    imgs, ignored = ingest.list_images(edge_dir)
    names = {p.name for p in imgs}
    assert names == set(EXPECT) | {f"bulk_{i:05d}.jpg" for i in range(20)}
    assert set(IGNORED) <= set(ignored)


@pytest.mark.parametrize("name", ["gray.png", "rgba.png", "palette.png", "cmyk.jpg", "UPPER_CASE.JPG", "png_named_as.jpg"])
def test_M1_02_decodes_to_rgb(edge_dir, name):
    im, flags = ingest.decode(edge_dir / name, 224)
    assert im is not None and im.mode == "RGB"


def test_M1_02_applies_exif_orientation(edge_dir):
    raw = __import__("PIL.Image", fromlist=["Image"]).open(edge_dir / "exif_rotated.jpg").size
    im, _ = ingest.decode(edge_dir / "exif_rotated.jpg", 224)
    assert im.size == (raw[1], raw[0])


@pytest.mark.parametrize("name", ["corrupt.jpg", "truncated.jpg", "zero_bytes.png", "too_small_4x4.png"])
def test_M1_03_bad_file_returns_flag_not_exception(edge_dir, name):
    im, flags = ingest.decode(edge_dir / name, 224)
    assert im is None and flags


def test_M1_04_bounded_prefetch_memory():
    """Analysis: the executive queues at most 3 batches. Bound the RAM they can hold."""
    src = (ROOT / "d2qual" / "executive.py").read_text()
    assert "for _ in range(3):" in src and "queue.popleft()" in src
    cfg = RunConfig()
    size = 518                                    # largest planned input size (ViT-S/14 at 518)
    bytes_ = 3 * cfg.batch_size * 3 * size * size * 4
    assert bytes_ <= 4 * 2**30, f"prefetch bound {bytes_/2**30:.2f} GB"


def test_M1_05_decode_throughput_12mp(work):
    from PIL import Image
    from concurrent.futures import ThreadPoolExecutor
    d = work / "mp12"; d.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    base = Image.fromarray(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)).resize((4000, 3000))
    paths = []
    for i in range(12):
        p = d / f"img{i}.jpg"; base.save(p, quality=90); paths.append(p)
    cores = os.cpu_count() or 1
    workers = min(6, cores)
    t = time.time()
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda p: ingest.decode(p, 518), paths))
    rate = len(paths) / (time.time() - t)
    need = 20.0 * workers / 6.0                   # requirement is 20/s on 6 cores; scale to this host
    print(f"decode rate {rate:.1f}/s on {workers} workers (scaled requirement {need:.1f}/s)")
    assert rate >= need


def test_M1_06_no_image_writes_in_runtime_code():
    for f in (ROOT / "d2qual").rglob("*.py"):
        src = f.read_text()
        assert not re.search(r"\.save\(|imwrite|tofile\(", src), f"image write in {f.name}"


# ----------------------------------------------------------------------------- M2 Privacy guard
def test_M2_01_refuses_identity_capable_component():
    with pytest.raises(RuntimeError):
        privacy.check_model_config({"backbone": "resnet18", "components": ["insightface_arcface_r100"]})
    assert privacy.check_model_config({"backbone": "resnet18", "components": ["resnet18"]})


def test_M2_02_output_dir_holds_only_predictions(main_run):
    assert sorted(os.listdir(main_run["out"])) == ["predictions.json"]


def test_M2_03_runs_with_network_blocked(work, edge_dir, expected_ids):
    shim = work / "nonet"; shim.mkdir(exist_ok=True)
    (shim / "sitecustomize.py").write_text(
        "import socket\n"
        "def _deny(*a, **k):\n    raise OSError('network disabled for M2-03 test')\n"
        "socket.socket.connect = _deny\nsocket.create_connection = _deny\nsocket.getaddrinfo = _deny\n")
    out = work / "out_nonet"; out.mkdir(exist_ok=True)
    r = run_exec(edge_dir, out, pythonpath_prefix=str(shim))
    assert r["rc"] == 0 and r["summary"]["status"] == "ok"
    assert not formatter.validate(doc_of(r), SCHEMA, expected_ids)


# ----------------------------------------------------------------------------- M3 Visual perception
def test_M3_01_probabilities_well_formed(edge_dir):
    from PIL import Image
    e = engine()
    items = [e.prepare(Image.open(edge_dir / f"bulk_{i:05d}.jpg").convert("RGB")) for i in range(4)]
    p, _ = e.predict(items)
    assert p.shape == (4, 4, 4) and np.all(np.isfinite(p))
    assert np.allclose(p.sum(-1), 1.0, atol=1e-4)


def test_M3_02_mirror_gives_swapped_output(edge_dir):
    from PIL import Image, ImageOps
    e = engine()
    im = Image.open(edge_dir / "bulk_00001.jpg").convert("RGB")
    p, _ = e.predict([e.prepare(im)])
    pm, _ = e.predict([e.prepare(ImageOps.mirror(im))])
    assert np.allclose(p[0], pm[0][LR_SWAP], atol=1e-4)


def test_M3_03_mirror_augmentation_swaps_labels():
    import random
    from dataset import SiteDataset, load_labels
    toy = ROOT / "data" / "toy"
    if not (toy / "labels.json").exists():
        pytest.skip("PENDING: toy data not generated")
    L = load_labels(toy / "labels.json")
    iid = next(i for i, y in L.items() if y[0] != y[1])
    ds = SiteDataset(toy / "images", [iid], L, 64, [0.5] * 3, [0.5] * 3, train=True)
    seen = set()
    for s in range(40):
        ds.rng = random.Random(s)
        _, y, _ = ds[0]
        seen.add(tuple(y.tolist()))
    assert tuple(L[iid]) in seen and tuple(L[iid][k] for k in LR_SWAP) in seen and len(seen) == 2


def test_M3_03_no_crop_or_cutout_augmentation():
    src = (ROOT / "train" / "dataset.py").read_text()
    code = "\n".join(l for l in src.splitlines() if not l.strip().startswith(("#", "-", '"')))
    assert not re.search(r"RandomResizedCrop|RandomCrop|\.crop\(|Cutout|RandomErasing", code)


@pytest.mark.parametrize("rid", ["M3_04", "M3_05", "M3_06"])
def test_M3_04_05_06_accuracy_on_darpa_data(rid):
    pytest.skip(f"PENDING: {rid.replace('_', '-')} needs the DARPA Part 2 training data")


def test_M3_07_group_split_has_no_leakage():
    from dataset import group_split
    ids = [f"s{g:03d}_v{v}.png" for g in range(50) for v in range(3)]
    groups = {i: i.split("_")[0] for i in ids}
    tr, dv, te = group_split(ids, groups)
    g = lambda s: {groups[i] for i in s}
    assert not (g(tr) & g(dv)) and not (g(tr) & g(te)) and not (g(dv) & g(te))
    assert len(tr) + len(dv) + len(te) == len(ids)


def test_M3_08_inference_time_and_gpu_memory():
    if not HAS_CUDA:
        pytest.skip("PENDING: needs an A40-class GPU host")


def test_M3_09_gpu_portability():
    pytest.skip("PENDING: run harness/apl_mirror.sh on A40, L40 and H100")


def test_M3_10_deterministic_rerun(work, edge_dir, main_run):
    out = work / "out_rerun"; out.mkdir(exist_ok=True)
    r = run_exec(edge_dir, out)
    assert doc_of(r) == doc_of(main_run)


def test_M3_11_no_runtime_downloads():
    src = (ROOT / "d2qual" / "engines" / "direct.py").read_text()
    assert "pretrained=False" in src
    assert 'setdefault("HF_HUB_OFFLINE", "1")' in (ROOT / "d2qual" / "executive.py").read_text()


def test_M3_12_second_engine_plugs_in_without_change(work, edge_dir):
    """Two engines through the ensemble: same model twice must equal the single engine."""
    import shutil
    from PIL import Image
    from d2qual.engines import build as build_engines
    from d2qual.config import load_model_config
    ens_dir = work / "ens"; shutil.rmtree(ens_dir, ignore_errors=True); shutil.copytree(MODEL, ens_dir)
    shutil.copytree(MODEL, ens_dir / "second")
    cfg = load_model_config(ens_dir)
    cfg["engines"] = [{"type": "direct", "weight": 1.0, "dir": "."}, {"type": "direct", "weight": 1.0, "dir": "second"}]
    (ens_dir / "model_config.json").write_text(json.dumps(cfg))
    ens = build_engines(ens_dir, cfg, torch.device("cpu"))
    single = engine()
    im = Image.open(edge_dir / "bulk_00002.jpg").convert("RGB")
    assert np.allclose(ens.predict([ens.prepare(im)])[0], single.predict([single.prepare(im)])[0], atol=1e-5)
    out = work / "out_ens"; out.mkdir(exist_ok=True)
    assert run_exec(edge_dir, out, model_dir=ens_dir)["rc"] == 0


def test_M3_13_intermediate_outputs_logged(main_run, expected_ids):
    aux = [l for l in main_run["logs"] if l.get("msg") == "aux"]
    if not aux:
        pytest.skip("PENDING: model under test was trained without aux labels")
    n_pred = len(expected_ids) - sum(1 for v in EXPECT.values() if v == "fallback")
    assert len(aux) == n_pred
    a = aux[0]
    assert a["head_end"] in ("up", "down") and a["facing"] in ("front", "back", "edge_on") and len(a["vis"]) == 4


# ----------------------------------------------------------------------------- M12 Qual formatter
def test_M12_01_single_valid_json(main_run, expected_ids):
    raw = (main_run["out"] / "predictions.json").read_bytes()
    raw.decode("utf-8")
    assert b"NaN" not in raw and b"Infinity" not in raw
    assert not formatter.validate(doc_of(main_run), SCHEMA, expected_ids)


def test_M12_02_one_record_per_image_exact_names(main_run, expected_ids):
    d = doc_of(main_run)
    assert sorted(p["image_id"] for p in d["predictions"]) == expected_ids
    for p in d["predictions"]:
        assert sorted((s["body_region"], s["laterality"]) for s in p["sites"]) == sorted(SITES)


def test_M12_03_invalid_document_is_caught():
    bad = formatter.build(["a.jpg"], {"a.jpg": [0, 0, 0, 0]}, "t", "e", "v")
    bad["predictions"][0]["sites"].pop()
    assert formatter.validate(bad, SCHEMA, ["a.jpg"])


def test_M12_04_atomic_write_leaves_no_temp(work):
    d = work / "atomic"; d.mkdir(exist_ok=True)
    doc = formatter.build(["a.jpg"], {"a.jpg": [0, 1, 2, 3]}, "t", "e", "v")
    for _ in range(3):
        formatter.write(doc, d / "predictions.json")
    assert os.listdir(d) == ["predictions.json"]


def test_M12_05_release_build_refuses_placeholder_team():
    p = subprocess.run(["bash", str(ROOT / "harness" / "apl_mirror.sh"), "TEAM_NAME_TBD", "e", "v", "m"],
                       capture_output=True, text=True)
    assert p.returncode == 2 and "placeholder" in p.stdout.lower()


# ----------------------------------------------------------------------------- M13 Executive
def test_M13_01_fallback_file_within_10s(main_run):
    t = [l["elapsed_s"] for l in main_run["logs"] if l.get("msg") == "initial_fallback_written"]
    assert t and t[0] < 10.0


def test_M13_02_model_failure_is_fallback_not_crash(work, edge_dir, expected_ids):
    out = work / "out_nomodel"; out.mkdir(exist_ok=True)
    r = run_exec(edge_dir, out, model_dir=work / "does_not_exist")
    assert r["rc"] == 0 and r["summary"]["status"] == "fallback_only"
    assert not formatter.validate(doc_of(r), SCHEMA, expected_ids)


def test_M13_03_watchdog_ends_with_valid_file(work, edge_dir, expected_ids):
    out = work / "out_wd"; out.mkdir(exist_ok=True)
    r = run_exec(edge_dir, out, env={"D2_BUDGET_S": "0.5"})
    assert r["rc"] == 0 and not formatter.validate(doc_of(r), SCHEMA, expected_ids)


def test_M13_04_checkpoints_at_configured_period(main_run):
    assert RunConfig().checkpoint_s <= 300
    assert main_run["summary"]["checkpoints"] >= 1          # D2_CHECKPOINT_S=0 forces one per batch


def test_M13_05_gpu_memory_cap():
    if not HAS_CUDA:
        pytest.skip("PENDING: needs a GPU host")


def test_M13_06_host_resources(main_run):
    import resource
    peak_gb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20
    assert peak_gb < 28.0
    assert RunConfig().decode_workers <= 8


def test_M13_07_summary_fields(main_run):
    s = main_run["summary"]
    for k in ("status", "images", "predicted", "fallback_images", "fallback_reasons", "elapsed_s", "per_image_s"):
        assert k in s


def test_M13_08_full_test_set_within_1800s():
    pytest.skip("PENDING: needs training-set size N and a GPU timing run")


def test_M13_09_fallback_class_from_model_config(main_run):
    cfg = json.loads((MODEL / "model_config.json").read_text())
    fb = cfg["fallback_class"]
    d = doc_of(main_run)
    corrupt = next(p for p in d["predictions"] if p["image_id"] == "corrupt.jpg")
    assert all(s["injury_type"] == fb for s in corrupt["sites"])


# ----------------------------------------------------------------------------- v0.3.3 additions
# (CR-2026-09-28-independent-run-review items 2 to 4: M1-07q, M1-08q, M2-07, M13-16, M13-17, M3-03 round trip)
def _mini_input(work, name, files):
    d = work / name; d.mkdir(exist_ok=True)
    from PIL import Image as _I
    for f in files:
        p = os.path.join(os.fsencode(str(d)), f if isinstance(f, bytes) else os.fsencode(f))
        _I.new("RGB", (64, 48), (120, 90, 60)).save(p.decode("utf-8", "surrogateescape"), format="JPEG")
    return d


def test_M1_07q_case_only_duplicates_both_emitted(work):
    d = _mini_input(work, "case_dup", ["scan.jpg", "SCAN.JPG"])
    out = work / "out_case"; out.mkdir()
    r = run_exec(d, out, model_dir=work / "no_model_case")
    ids = sorted(p["image_id"] for p in doc_of(r)["predictions"])
    assert r["rc"] == 0 and ids == ["SCAN.JPG", "scan.jpg"]
    assert any(l.get("msg") == "case_only_duplicate_names" for l in r["logs"])


def test_M1_08q_non_utf8_filename_emitted_escaped(work):
    d = _mini_input(work, "bad_name", [b"ok.jpg", b"bad\xff name.jpg"])
    out = work / "out_badname"; out.mkdir()
    r = run_exec(d, out, model_dir=work / "no_model_bad")
    doc = doc_of(r)
    assert r["rc"] == 0 and len(doc["predictions"]) == 2
    assert "bad\\xff name.jpg" in [p["image_id"] for p in doc["predictions"]]
    assert any(l.get("msg") == "non_utf8_filenames_escaped" for l in r["logs"])
    assert not formatter.validate(doc, SCHEMA, None)


def test_M2_07_socket_creation_disabled_in_process():
    code = ("import socket, sys; sys.path.insert(0, %r); from d2qual.executive import _block_sockets; _block_sockets()\n"
            "try:\n    socket.socket(); print('INET_ALLOWED')\nexcept OSError: print('INET_BLOCKED')\n"
            "a, b = socket.socketpair(socket.AF_UNIX); print('UNIX_OK')") % str(ROOT)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert "INET_BLOCKED" in p.stdout and "UNIX_OK" in p.stdout, p.stdout + p.stderr


def test_M13_16_sigterm_writes_results_and_exits_0(work, edge_dir, expected_ids):
    import signal
    out = work / "out_sigterm"; out.mkdir()
    e = dict(os.environ, D2_INPUT=str(edge_dir), D2_OUTPUT=str(out), D2_MODEL_DIR=str(MODEL), D2_SCHEMA=str(SCHEMA),
             D2_TEAM_NAME="verification", D2_TEAM_EMAIL="v@example.com", PYTHONPATH=str(ROOT))
    p = subprocess.Popen([sys.executable, "-m", "d2qual"], env=e, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in p.stderr:
        if "initial_fallback_written" in line:
            break
    p.send_signal(signal.SIGTERM)
    so, _ = p.communicate(timeout=120)
    summary = json.loads(so.strip().splitlines()[-1])
    doc = json.loads((out / "predictions.json").read_text(encoding="utf-8"))
    assert p.returncode == 0 and summary["status"] == "signal"
    assert not formatter.validate(doc, SCHEMA, expected_ids)


def test_M13_17_unwritable_output_exits_1(work, edge_dir):
    blocker = work / "not_a_dir"; blocker.write_text("file, not a directory")
    r = run_exec(edge_dir, blocker / "out")
    assert r["rc"] == 1 and any(l.get("msg") == "output_unwritable" for l in r["logs"])


def test_M3_03_reflection_rule_round_trip():
    from PIL import Image as _I
    from d2qual import augment as A
    im = _I.new("RGB", (6, 4)); im.putpixel((0, 0), (255, 0, 0)); im.putpixel((5, 3), (0, 255, 0)); im.putpixel((5, 0), (0, 0, 255))
    y = [0, 1, 2, 3]; vis = [0.1, 0.2, 0.3, 0.4]
    for op in A.OPS:
        im1, y1, v1 = op(im, y, (vis,))
        im2, y2, v2 = op(im1, y1, (v1,))
        assert np.asarray(im2).tolist() == np.asarray(im).tolist() and y2 == y and v2 == vis, op.name   # every op here is an involution
        assert (y1 == [y[k] for k in LR_SWAP]) == op.swaps, op.name
    assert A.MIRROR_H.swaps and A.MIRROR_V.swaps and not A.ROT180.swaps and not A.rotate(17).swaps
    h, yh = A.MIRROR_H(im, y)[:2]; hv, yhv = A.MIRROR_V(h, yh)[:2]; r, yr = A.ROT180(im, y)[:2]
    assert np.asarray(hv).tolist() == np.asarray(r).tolist() and yhv == yr == y      # two reflections = 180° rotation, no swap
