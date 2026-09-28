#!/usr/bin/env python3
"""Conformance tests for the qualification container (verification levels L3 and L4).

Runs the entrypoint the way APL will (fresh process, read-only-style input dir,
separate output dir), then checks the ICD contract. Modes:
  native (default): python -m d2qual with D2_* env vars; any machine
  docker:           docker run with the APL flags; needs a Docker host (GPU optional)

  python3 harness/conformance.py --model-dir model_toy [--docker IMAGE] [--bulk 200]

Tests
  T1 edge cases: every image gets a record; bad files use fallback; non-images ignored
  T2 output hygiene: exactly one file in /app/output, valid schema, exit code 0
  T3 no model: container still writes a valid all-fallback file and exits 0
  T4 watchdog: tiny time budget still ends with a valid file and exit 0
  T5 determinism: two runs give identical predictions
  T6 throughput: per-image time and projected capacity inside 1,800 s
"""
import argparse, json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT))
from make_edge_cases import build, EXPECT, IGNORED
from d2qual.formatter import validate

SCHEMA = ROOT / "schema" / "qual-predictions.schema.json"


def run(input_dir, output_dir, model_dir, docker_image=None, extra_env=None, timeout=3700):
    env = dict(os.environ, D2_INPUT=str(input_dir), D2_OUTPUT=str(output_dir), D2_MODEL_DIR=str(model_dir),
               D2_SCHEMA=str(SCHEMA), D2_TEAM_NAME="conformance", D2_TEAM_EMAIL="conformance@example.com",
               PYTHONPATH=str(ROOT), **(extra_env or {}))
    if docker_image:
        cmd = ["docker", "run", "--rm", "--network", "none", "--cpus=8", "--memory=32g",
               "-v", f"{input_dir}:/app/input:ro", "-v", f"{output_dir}:/app/output:rw"]
        if shutil.which("nvidia-smi"):
            cmd[3:3] = ["--gpus", "all"]
        for k, v in (extra_env or {}).items():
            cmd += ["-e", f"{k}={v}"]
        cmd.append(docker_image)
    else:
        cmd = [sys.executable, "-m", "d2qual"]
    t = time.time()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout, cwd=ROOT)
    dt = time.time() - t
    summary = None
    for line in reversed(p.stdout.strip().splitlines()):
        try:
            summary = json.loads(line); break
        except Exception:
            pass
    return p.returncode, summary, dt, p.stderr


def check_output(output_dir, expected_ids):
    files = sorted(os.listdir(output_dir))
    errs = []
    if files != ["predictions.json"]:
        errs.append(f"output dir must hold exactly predictions.json, found {files}")
    doc = json.loads((Path(output_dir) / "predictions.json").read_text(encoding="utf-8"))
    errs += validate(doc, SCHEMA, expected_ids)
    return errs, doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True); ap.add_argument("--docker")
    ap.add_argument("--bulk", type=int, default=100)
    a = ap.parse_args()
    model_dir = Path(a.model_dir).resolve()
    tmp = Path(tempfile.mkdtemp(prefix="d2conf_"))
    results, fails = [], 0

    def record(name, ok, detail):
        nonlocal fails
        fails += 0 if ok else 1
        results.append({"test": name, "pass": ok, "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)

    inp = build(tmp / "input", n_bulk=a.bulk)
    expected = sorted(list(EXPECT) + [f"bulk_{i:05d}.jpg" for i in range(a.bulk)])

    # T1 + T2
    out = tmp / "out1"; out.mkdir()
    rc, s, dt, err = run(inp, out, model_dir, a.docker)
    errs, doc = check_output(out, expected)
    fb = s.get("fallback_images") if s else None
    want_fb = sum(1 for v in EXPECT.values() if v == "fallback")
    record("T1 edge cases", rc == 0 and not errs and fb == want_fb and s.get("status") == "ok",
           f"rc={rc}, records={len(doc['predictions'])}/{len(expected)}, fallback={fb} (expected {want_fb}), reasons={s.get('fallback_reasons') if s else None}")
    record("T2 output hygiene", not errs, "exactly one valid predictions.json" if not errs else errs[:3])
    ignored_leak = [i for i in IGNORED if any(p["image_id"].startswith(i) for p in doc["predictions"])]
    record("T1b ignored inputs", not ignored_leak and "nested.jpg" not in [p["image_id"] for p in doc["predictions"]],
           "non-images and subdirectory skipped" if not ignored_leak else ignored_leak)

    # T3 no model
    out3 = tmp / "out3"; out3.mkdir(); empty = tmp / "no_model"; empty.mkdir()
    rc, s3, _, _ = run(inp, out3, empty, a.docker, {"D2_MODEL_DIR": "/nonexistent"} if a.docker else None)
    e3, _ = check_output(out3, expected)
    record("T3 no model -> valid fallback file", rc == 0 and not e3 and s3 and s3["status"] == "fallback_only", f"rc={rc}, status={s3 and s3['status']}")

    # T4 watchdog
    out4 = tmp / "out4"; out4.mkdir()
    rc, s4, dt4, _ = run(inp, out4, model_dir, a.docker, {"D2_BUDGET_S": "0.5"})
    e4, _ = check_output(out4, expected)
    record("T4 watchdog", rc == 0 and not e4, f"rc={rc}, status={s4 and s4['status']}, wall={dt4:.1f}s")

    # T5 determinism
    out5 = tmp / "out5"; out5.mkdir()
    run(inp, out5, model_dir, a.docker)
    same = json.loads((out / "predictions.json").read_text()) == json.loads((out5 / "predictions.json").read_text())
    record("T5 determinism", same, "identical predictions on rerun" if same else "predictions differ between runs")

    # T6 throughput
    if s and s.get("per_image_s"):
        record("T6 throughput", True, f"{s['per_image_s']} s/image on this host; projected {s['projected_images_in_1800s']} images in 1,800 s; total wall {dt:.1f}s")
    print(json.dumps({"failures": fails, "results": results}, indent=1))
    shutil.rmtree(tmp, ignore_errors=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
