# D2 Block T: qualification container (v0.2)

Implements the Block T build of the D2 architecture: M1 Ingest, M2 Privacy guard, M3 Visual perception (direct baseline engine plus ensemble slot), M12 Qual formatter and M13 Executive.

Requirements live in `docs/D2_BlockT_Module_Requirements_v0.2.md`. Verification is `tests/test_requirements.py`, where every test is named for its requirement ID. Gate status is in `docs/D2_BlockT_Design_Readiness_Gate_v1.0.md`.

## Layout

| Path | Module / purpose |
|---|---|
| `d2qual/executive.py`, `config.py` | M13: entrypoint, fallback-first sequence, watchdog, checkpoints, GPU memory cap, run summary |
| `d2qual/ingest.py` | M1: enumerate, decode, EXIF, color modes, failure flags |
| `d2qual/privacy.py` | M2: identity-model deny list |
| `d2qual/model.py`, `engines/` | M3: DirectSiteNet (site-query head plus M3-13 intermediate-output heads), engine registry and ensemble, Probe B slot |
| `d2qual/decision.py` | Decision layer (bias fitted on dev split) |
| `d2qual/formatter.py` | M12: build, validate, atomic write |
| `train/` | Training, metrics (per-class, swaps, group bootstrap), decision-layer fit, evaluation |
| `tools/` | Toy generator with anatomical laterality, laterality proof, trace matrix |
| `harness/` | Hostile edge cases, conformance runner, APL-mirror release script |
| `tests/` | Requirement verification suite |

## Verify (any machine)

```
pip install torch torchvision timm==1.0.9 pillow jsonschema pytest
python3 tools/make_toy_data.py --out data/toy --scenes 700 --views 2
python3 tools/test_laterality.py
python3 train/train_direct.py --images data/toy/images --labels data/toy/labels.json \
  --manifest data/toy/manifest.csv --aux data/toy/aux.json --out model_toy \
  --backbone resnet18 --size 128 --epochs 4 --batch 32 --lr 1e-3 --backbone-lr-mult 1.0 --no-rotate
python3 -m pytest tests -q --junitxml=reports/junit.xml
python3 tools/trace_matrix.py reports/junit.xml
```

## Train for real (GPU host)

```
python3 train/train_direct.py --images <dir> --labels labels.json --manifest groups.csv \
  --out model --backbone vit_small_patch14_dinov2.lvd142m --size 518 --pretrained --epochs 30
```

- Labels use the ICD predictions format.
- `groups.csv` lists `image_id,group` (the scene or manikin), used for leakage-safe splits.
- Pretrained backbones download at training time only. Whether they are allowed is an open DARPA question.

## Release check (GPU host with Docker and the NVIDIA toolkit)

```
harness/apl_mirror.sh <team-name> <email> <version> model [holdout_dir]
```

The script runs these steps, then prints the push commands:

1. Build the image.
2. Check the image size.
3. Confirm the GPU is visible inside the container.
4. Run a ClamAV scan.
5. Run the conformance suite in Docker mode.
6. Do a timed holdout run with APL's flags.

Run it on A40, L40 and H100 before pushing to the private GitLab registry.
