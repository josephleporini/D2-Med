# D2 Block T: Module Requirements

**Version 0.1 DRAFT | 24 Sep 2026 | Scope: qualification container (Dec 1 build)**
Joseph Leporini | Independent Systems Practitioner

## 1. Purpose and scope

This document allocates system requirements (Architecture Baseline v0.2: S-xx stated, D-xx derived) to the five modules in the Block T build, and defines the interfaces between them. It is deliberately lean: one table per module, one verification method per requirement.

**In scope:** M1 Ingest, M2 Privacy guard, M3 Visual perception (the Block T slice, including the decision layer), M12 Qual formatter, M13 Executive.

**Out of scope for Block T:** M4 to M11 (temporal, speech, extraction, fusion, rules, Lane 1 and Lane 2 heads). The M8 event bus is bypassed in Block T, as decided in ADR-001.

**Conventions:**
- Verification methods: T = Test, A = Analysis, I = Inspection, D = Demonstration.
- TBR = to be reviewed. These are engineering targets, not DARPA values, and are recalibrated once real data and the forum answers arrive.
- The design spike (code in `d2-blockT`) is evidence against these requirements. It is not the baseline.

## 2. Allocation summary

| System req. | M1 | M2 | M3 | M12 | M13 |
|---|:-:|:-:|:-:|:-:|:-:|
| S-11 offline, self-contained image |  | ● | ● |  | ● |
| S-12 input directory and formats | ● |  |  |  |  |
| S-13 one valid JSON file |  |  |  | ● |  |
| S-14 four sites × four classes per image |  |  | ● | ● |  |
| S-15 anatomical laterality |  |  | ● |  |  |
| S-16 resource envelope | ● |  | ● |  | ● |
| S-17 GPU portability |  |  | ● |  | ● |
| S-07 data handling (no identity use) | ● | ● |  |  |  |
| S-19 accuracy plus other metrics |  |  | ● |  |  |
| D-09 never drop an input |  |  |  | ● | ● |
| D-26, D-27 time and memory margins | ● |  | ● |  | ● |
| D-29 accuracy and recall floors |  |  | ● |  |  |

## 3. Module requirements

### M1 Ingest

| ID | Requirement | Parent | Verif. |
|---|---|---|:-:|
| M1-01 | Enumerate only regular files in the top level of the input directory whose extension is .jpg, .jpeg or .png, case-insensitive. Ignore everything else and log the count ignored. | S-12 | T |
| M1-02 | Decode every legal file to 8-bit RGB with EXIF orientation applied. Grayscale, palette, alpha and CMYK images are converted. | S-12, D-11 | T |
| M1-03 | A file that cannot be decoded (corrupt, truncated, empty, smaller than 8 px) returns a failure flag and a reason. It never raises an exception, and the image stays in the output set. | D-09 | T |
| M1-04 | Decode ahead of inference with bounded memory: host RAM held by queued images ≤ 4 GB, independent of the number of images. | S-16, D-27 | T, A |
| M1-05 | Decode throughput ≥ 20 images/s on 6 cores for 12-megapixel JPEGs (TBR). | D-26 | T |
| M1-06 | Never write image data, crops or thumbnails to disk. | S-07 | I |

### M2 Privacy guard

| ID | Requirement | Parent | Verif. |
|---|---|---|:-:|
| M2-01 | Refuse to run a model whose configuration declares an identity-capable component (face recognition, re-identification, face embedding). | S-07 | T |
| M2-02 | No image, feature or embedding leaves process memory. After a run the output directory contains only `predictions.json`. | S-07 | T, I |
| M2-03 | Make no network calls. Library hub access is forced offline, and the image runs unchanged under `--network none`. | S-11 | T |

### M3 Visual perception (Block T slice)

| ID | Requirement | Parent | Verif. |
|---|---|---|:-:|
| M3-01 | For each image, output a probability distribution over the four ICD classes for each of the four sites, in the fixed site order. Each distribution sums to 1 ± 1e-4, and every value is finite. | S-14 | T |
| M3-02 | Outputs refer to the casualty's anatomical side. Mirroring the input must yield the left/right-swapped output: exactly if test-time mirroring is used, within 0.05 per probability otherwise. | S-15, D-04 | T |
| M3-03 | Training augmentation that mirrors an image must swap its left and right labels. Augmentation must not crop, zoom or cut out any part of the body. | D-04, D-05 | I, T |
| M3-04 | Accuracy on a group-held-out split of real DARPA training data: threshold ≥ 0.80 site accuracy, objective ≥ 0.85 (TBR, Development Plan v1.2 §5). | S-19, D-29 | T |
| M3-05 | Every class present in the data has recall ≥ 0.50 (TBR). Macro-F1 is reported alongside accuracy. | S-19, D-29 | T |
| M3-06 | Laterality swaps (sites wrong only because left and right were exchanged) < 5% of sites (TBR). | S-15 | T |
| M3-07 | The decision layer is fitted on a development split only. The test split is scored once, after the layer is fixed. Splits are grouped by scene or manikin, never by image. | D-29 | I, A |
| M3-08 | Inference time ≤ 0.5 s per image on an A40, including test-time mirroring (TBR). Peak GPU memory ≤ 14 GB. | D-26, D-27 | T |
| M3-09 | Run unmodified on A40, L40 and H100 with CUDA ≤ 13 and driver ≥ 580. No architecture-specific kernel may run without a fallback. | S-17, D-08 | T |
| M3-10 | Identical outputs on rerun on the same hardware and inputs. | D-28 | T |
| M3-11 | Load weights only from the image. No downloads at run time. | S-11 | I, T |
| M3-12 | A second engine (the Probe B staged pipeline) can be added and probability-averaged with no change to M12 or M13. | ADR-001 | I |

### M12 Qual formatter

| ID | Requirement | Parent | Verif. |
|---|---|---|:-:|
| M12-01 | Write exactly one file, `/app/output/predictions.json`: UTF-8, double-quoted keys, no NaN or Infinity, conforming to schema 1.0. | S-13 | T |
| M12-02 | One record per enumerated input image, `image_id` equal to the exact file name, with exactly the four sites, each once. | S-14 | T |
| M12-03 | Validate the document before writing. If validation fails, write an all-fallback document instead. | D-28 | T |
| M12-04 | Write atomically. A reader or a kill never sees a partial file, and no temporary file remains after a normal exit. | D-09 | T |
| M12-05 | The team name, email and version come from the image build. The release build refuses a placeholder team name. | S-18 | I |

### M13 Executive

| ID | Requirement | Parent | Verif. |
|---|---|---|:-:|
| M13-01 | Write a valid, complete fallback `predictions.json` within 10 s of start, before the model loads. | D-09 | T |
| M13-02 | Exit 0 whenever a valid file exists. A model-load failure produces a fallback-only run, not a crash. | D-09 | T |
| M13-03 | Enforce a time budget (default 3,000 s). When it expires, write the merged results with fallbacks for unfinished images and exit before 3,600 s. | S-16, D-26 | T |
| M13-04 | Checkpoint merged results so that no more than 5 minutes of completed inference can be lost. | D-09 | T |
| M13-05 | Cap GPU memory in-process at 14 GB, whatever the physical GPU size. | S-16, D-27 | T |
| M13-06 | Stay within 8 CPU cores and 28 GB host RAM, and use no more than 1 GB of runtime disk beyond the image. | S-16, D-27 | T |
| M13-07 | Log structured progress to stderr, and print a one-line JSON summary to stdout: counts, fallbacks by reason, timing, peak GPU memory. | D-06 | I |
| M13-08 | Process the full test set in ≤ 1,800 s. Size this once the training-set count N is known (test ≈ 0.2 N). | D-26 | T, A |
| M13-09 | Choose the fallback class from training data (the majority class), recorded in the model configuration. | D-09 | I |

## 4. Interfaces

| ID | From → To | Content | Contract |
|---|---|---|---|
| IF-1 | APL → M1 | Flat image directory, read-only | ICD §3.1 |
| IF-2 | M1 → M3 | RGB image in memory, or a failure flag with its reason | M1-02, M1-03 |
| IF-3 | M3 → decision layer | Probabilities, shape (N, 4 sites, 4 classes), float32, site order UE-L, UE-R, LE-L, LE-R | M3-01 |
| IF-4 | Decision layer → M12 | Class index per site, shape (N, 4), same site order | M3-07 |
| IF-5 | Model directory → M3, M13 | `weights.pt` plus `model_config.json` (backbone, input size, normalization, decision bias, fallback class, components, training metadata) | M3-11, M2-01, M13-09 |
| IF-6 | M12 → APL | `predictions.json` | ICD §3.2, Appendix A |

## 5. Open items that move these requirements

| Item | Moves | Source |
|---|---|---|
| Training-set size N | M13-08, M1-05, M3-08 | Arrives with the Part 2 data |
| Is not_testable "no part visible" or a threshold? | M3-04 to M3-06 (about 40% of Probe B errors) | Forum question |
| Are backbones pretrained on public human imagery allowed? | M3-04, the M3 design | Forum question |
| Is cloud GPU training on DARPA data allowed? | Where M3 is trained, not the requirements | Forum question Q13 |
| Is not_testable missing from ICD Appendix A by mistake? | M12-01 | Forum question Q2 |

## 6. Design-spike evidence to date (not a verification record)

The code in `d2-blockT` was written ahead of this document. It is traced against the requirements here only to show where the gaps are.

| Requirement | Spike status |
|---|---|
| M3-02, M3-03 | Flip-with-swap rule proven on 60 synthetic scenes. The first toy generator had a double-flip bug, caught by visual inspection and then by an independent anatomy check. |
| M1-01 to M1-03, M2-02, M12-01 to M12-04, M13-01 to M13-03, M3-10 | Conformance tests T1 to T5 pass natively on CPU (24 Sep): 116 of 116 records including 16 hostile edge cases; 4 expected fallbacks (corrupt, truncated, empty, 4×4 px); non-image files and the subdirectory ignored; the no-model run writes a valid fallback file; the watchdog exits cleanly; reruns are identical. Not yet run inside Docker. |
| M1-05, M3-08, M3-09, M13-06, M13-08 | Not measured. Need a GPU host. Docker Hub is blocked from this workspace, so the real Dockerfile has not been built. |
| M3-04 to M3-06 | Not applicable until real data arrives. A CPU smoke run (randomly initialized ResNet-18, 18 min) did not learn usefully: test accuracy 0.51 against a 0.54 majority baseline, and the decision layer correctly reported that its recall floor was not met. This says nothing yet about the real model, which needs a pretrained backbone on GPU. |
