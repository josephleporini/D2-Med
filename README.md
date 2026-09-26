# Probe B (DARPA D2 injury recognition): code of record

Layout
- `gen/`    generator (Blender scene*.py, sample_batch*.py) and pipeline modules (eval_v3, checks, parts, side_kp ...)
- `jobs/`   phase job scripts (sidehead2 train/extract/iou, diag, nm, prof, sub, ph0, ph2, vf)
- `score/`  `score.py` single scorer with failure ledger; `manifest.py` split locks
- `infra/`  `setup_env.sh` (idempotent pod setup, `--lock` writes an environment lock), `runjob.sh` (standard job wrapper), `pack.sh`
- `manifests/` locked split manifests (sha256 per scene file)
- `docs/`   ledger schema and retrospective procedure
- `tests/`  synthetic fixture for the scorer

On the volume the repo lives at `/workspace/probeB/repo` with a `COMMIT` file; every job records that commit.

Pod command pattern
```
bash /workspace/probeB/repo/infra/runjob.sh <job> $PY <script> <args>     # outputs -> /workspace/probeB/out/<job>
```
Rules
1. Outputs go to the volume (`$OUT`), never only to container disk.
2. Every scored result comes from `score/score.py` against a manifest-locked split.
3. test5 is scored once, after the design freeze; its ledger is not used for design choices.

Storage (26 Sep 2026)
- Primary volume: `d2-probeb-ro` (2txwmoosly), EU-RO-1, 40 GB standard. Copied from `d2-p3-vol` (a6j1mpdgbn, EU-NL-1); dev3 and test4 manifests verified, all top-level file counts and byte totals match. `/workspace/.cache` (package caches) was not carried over.
- The EU-NL-1 volume is retained as a backup until deletion is approved.
- Moves between data centers: `infra/relay.py` on a source pod (port 8000) and `infra/pull.sh <relay url>` on the destination.
