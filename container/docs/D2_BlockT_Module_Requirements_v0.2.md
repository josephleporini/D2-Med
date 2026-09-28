# D2 Block T: Module Requirements and Performance Allocation

**Version 0.2 DRAFT | 24 Sep 2026 | Scope: qualification container (Dec 1 build), plus a preview of Lane 1 and Lane 2 performance allocation**

*v0.2 changes (Joseph's review): M1-05 accepted. M3-04 separated into Sprint and fielded intent. M3-06 objective tightened. M3-08 given a stage timing allocation. M3-13 added (observable intermediate outputs). New §6 performance allocation (Pk-style chains), §7 Lane 1 record-update semantics, §8 stage timing.*
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
| M3-04 | Sprint qualification accuracy on a group-held-out split of real DARPA training data: threshold ≥ 0.80 site accuracy, objective ≥ 0.85 (TBR, Development Plan v1.2 §5). This is a Sprint entry bar, not a fielded requirement. Fielded requirements are consequence-weighted and set separately (§6.4). Allocated to stages in §6.1. | S-19, D-29 | T |
| M3-05 | Every class present in the data has recall ≥ 0.50 (TBR). Macro-F1 is reported alongside accuracy. | S-19, D-29 | T |
| M3-06 | Laterality swaps (sites wrong only because left and right were exchanged): threshold < 5% of sites, objective < 2% (TBR). Tighten further if DARPA publishes a laterality metric. | S-15 | T |
| M3-07 | The decision layer is fitted on a development split only. The test split is scored once, after the layer is fixed. Splits are grouped by scene or manikin, never by image. | D-29 | I, A |
| M3-08 | Inference time ≤ 0.5 s per image on an A40, including test-time mirroring (TBR). Peak GPU memory ≤ 14 GB. Stage allocation in §8; confirm as test data arrives. | D-26, D-27 | T |
| M3-09 | Run unmodified on A40, L40 and H100 with CUDA ≤ 13 and driver ≥ 580. No architecture-specific kernel may run without a fallback. | S-17, D-08 | T |
| M3-10 | Identical outputs on rerun on the same hardware and inputs. | D-28 | T |
| M3-11 | Load weights only from the image. No downloads at run time. | S-11 | I, T |
| M3-12 | A second engine (the Probe B staged pipeline) can be added and probability-averaged with no change to M12 or M13. | ADR-001 | I |
| M3-13 | Every engine exposes its intermediate results so each stage in §6.1 can be tested on its own: head end, facing (front, back, edge-on), and a per-site visibility fraction, each with a confidence. They go to the run log and are not scored. | D-06, §6.1 | T, I |

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

## 6. Performance allocation (Pk-style)

A missile's probability of kill is allocated down a chain: detect × track × guide × fuze × warhead. Each link gets a budget, and each budget gets its own test. The same discipline applies here. Every scored output is the end of a chain of stages, so an end-to-end number means nothing until it is broken into links you can measure and fix.

Two cautions carry over from Pk work:

- **The links are conditional, not independent.** A left/right swap costs nothing when both limbs carry the same label. Probe B found that only 24 of 59 swaps changed a label. A straight product of stage accuracies is therefore conservative.
- **Parallel paths change the arithmetic.** When voice and vision can each supply a field, it fails only if both fail. That is redundancy, and it is the main lever for the fielded system (§6.3).

### 6.1 Block T chain: one limb site on one qualification image

| Link | Function | Allocation, P(correct), TBR | Probe B synthetic evidence (unclothed manikin, no wounds) |
|---|---|:-:|---|
| L1 Body found, casualty designated | F2.1 | 0.99 | Body found in 120 of 120 scenes; casualty designation untested (single-body scenes) |
| L2 Body frame gives anatomical left/right | F2.2 | 0.97 at site level | Head end 95%; facing 82% on non-edge-on views; label-changing swaps 5.8% of testable sites with the raw pose frame, 3.9% with a perfect frame |
| L3 Visibility (testable vs not_testable) | F2.3 | 0.94 | Weakest link: about 40% of pipeline errors |
| L4 Class given visible (no injury, wound, amputation) | F2.4 | 0.95 | Amputation AUC 0.87 on perfect parts, 0.77 on predicted parts; wound not yet modeled |
| **Chain** | | **0.99 × 0.97 × 0.94 × 0.95 ≈ 0.86** | Meets the 0.85 objective with no margin; Probe B's end-to-end result was 73.3% (3 classes, synthetic) |

**Reading the table:** L3 and L4 are where the budget is at risk, and the not_testable definition question to DARPA moves L3 directly. The direct engine learns all four links in one network, so M3-13 adds its intermediate outputs to make each link measurable.

### 6.2 Lane 1 chains: one DD Form 1380 field (preview, Block O)

Score each field with three outcomes, not one:

- **Correct**
- **Blank (omission):** recoverable, because a human fills it later.
- **Wrong (commission):** dangerous, because a wrong entry travels with the casualty.

A single accuracy number hides the difference.

| Field type | Chain | Proposed wrong-entry ceiling (TBR) | Proposed completeness goal (TBR) |
|---|---|:-:|:-:|
| Medication (name, dose, route, time) | spoken × ASR entity correct × extraction correct × right casualty × time within tolerance | ≤ 1% | ≥ 80% |
| Tourniquet (limb, type, time) | seen or spoken × right casualty × right limb (anatomical) × time within ±2 min | ≤ 1% on limb | ≥ 90% |
| Vitals (value, time column) | spoken or sensed × ASR number correct × right casualty × right column | ≤ 3% outside tolerance | ≥ 85% |
| Injury marks (site) | seen or spoken × right casualty × right site | ≤ 5% | ≥ 85% |
| Identity (name, last 4, unit) | spoken × ASR correct × extraction correct | ≤ 1% | best effort |

**Voice-to-text accuracy.** Word error rate is the wrong metric. A transcript can be 95% word-correct and still turn "fifteen" into "fifty". Measure **clinical entity accuracy** instead: drug names, numbers with units, body sites and laterality, device names, scored as entities.

- Proposed target on synthetic noisy medic speech: threshold 0.90, objective 0.95 (TBR).
- Real-data targets wait for the Lane 1 ICD (forum Q1, Q7).

The design levers are:

- a vocabulary constrained to the TCCC drug and device lexicon (M9 feeds M6);
- cross-checking against vision, e.g. an autoinjector in hand while "morphine" is spoken;
- for the fielded system, a spoken read-back through Lane 2 ("confirm ketamine 50 IM"), closing the loop the way fire control confirms a solution.

### 6.3 Imagery in battlefield conditions (preview, Block O)

Manikin stills are the easy case. The one published reference in Development Plan v1.2 §5 is a DARPA Triage Challenge team system on robot video in field and night conditions. It reported upper-extremity trauma accuracy of 54–69% and lower-extremity accuracy of 36–64% (arXiv 2512.08754). That setting is not directly comparable, but it bounds what vision alone does in the field.

The architecture therefore does not ask vision to carry the injury diagram alone. With voice (the medic saying "GSW right thigh") and vision as parallel paths, and assuming independent errors:

P(site wrong or missing) ≈ P(voice misses) × P(vision misses) = 0.20 × 0.35 = 0.07

That gives about 93% combined, from two individually mediocre sensors.

- The independence assumption is optimistic, since smoke and chaos degrade both paths at once. It must be measured, not assumed.
- Multi-frame fusion across video (Development Plan v1.2 §7.1) is the second lever: a hidden limb end in one frame is often visible a few seconds later.

### 6.4 Fielded intent vs Sprint targets

A deployed system should do much better than the Sprint entry bar. Its requirements should be written by consequence, not as one accuracy figure:

- per-field wrong-entry ceilings (§6.2) keyed to the hazard severities in the Architecture Views (MIL-STD-882E categories);
- a critical-miss floor for Lane 2: massive hemorrhage, airway and respiration findings (MARCH M, A, R) prompted with recall ≥ 0.95 (TBR);
- an inappropriate-recommendation ceiling for Lane 2 ≤ 2% (TBR), plus latency (D-19).

These need clinical review before they are baselined. The TCCC reviewer sets the severity weights, not the engineer.

### 6.5 Lane 2 chain (preview, Block O)

P(correct, timely prompt) = P(finding detected) × P(casualty state correct) × P(rule fires correctly) × P(delivered within latency). Rule firing (M9) is deterministic, so its link is verified by inspection and test against the guideline text (target 1.0). The uncertainty sits in perception and state, which is why Lane 2 inherits Lane 1's detection quality.

## 7. Lane 1 record-update semantics (preview requirements for M10)

The card updates as the assessment changes. The event schema is append-only and built for this: M10 regenerates the card from the current event set, and nothing is overwritten silently.

| ID | Requirement | Verif. |
|---|---|:-:|
| M10-U1 | A new measurement at a new time adds a Signs and Symptoms time column. The card has four columns: keep the first set and the latest three, and move the rest to Notes (TBR, clinical review). | T |
| M10-U2 | A spoken correction ("correction", "belay that", "left leg, not right") creates an event that supersedes the earlier one. The card shows the corrected value. | T |
| M10-U3 | When voice and vision disagree on a field, neither wins silently. The field takes the source named in a per-field authority table, its confidence is lowered, both values go to Notes, and Lane 2 may prompt for clarification. | T, I |
| M10-U4 | The card is regenerated within 5 s of any new or superseding event (TBR). | T |
| M10-U5 | Every generated version of the card is retained with its timestamp. | I |

## 8. Stage timing allocation (M3-08)

Orientation, wound vs no wound, matching to JTS guidelines and criticality assessment split across blocks. Orientation and wound vs no wound are Block T and run per image on the GPU. Matching to JTS guidelines and assessing criticality are Lane 2 (M9, M11): they run on the CPU in milliseconds, are timed under the Lane 2 latency budget (Architecture Views §08), and are not part of the qualification score.

**Structured engine (Probe B pipeline), per image on an A40. Estimates, TBR until measured:**

| Stage | Function | Allocation |
|---|---|:-:|
| Decode and letterbox | M1 (CPU, runs ahead of the GPU) | off critical path |
| Body detection | F2.1 | 25 ms |
| Pose and body frame (head end, facing) | F2.2 | 35 ms |
| Limb segmentation | F2.2, F2.3 | 120 ms |
| Limb-end classifier, 4 limbs | F2.4 | 40 ms |
| Visibility and decision layer | F2.3, decision | 5 ms |
| **One pass** | | **225 ms** |
| **With mirrored second pass (test-time mirroring)** | | **450 ms** |
| Margin to 0.5 s | | 50 ms |

**Direct engine:** a single network pass of about 30 to 60 ms, doubled by mirroring (estimate). The allocation per stage applies only to the structured engine; the direct engine is timed as a whole.

The one measured point so far is Probe B's pose model, at about 0.5 s per image on 2 CPU cores. GPU figures must come from the first APL-mirror run.

**Throughput check against the 1-hour limit:** at 0.5 s per image, 1,800 s processes 3,600 images. That covers a test set of about 20% of an 18,000-image training set. If N turns out larger, drop test-time mirroring first, then the structured engine.

## 9. Design-spike evidence to date (not a verification record)

The code in `d2-blockT` was written ahead of this document. It is traced against the requirements here only to show where the gaps are.

| Requirement | Spike status |
|---|---|
| M3-02, M3-03 | Flip-with-swap rule proven on 60 synthetic scenes. The first toy generator had a double-flip bug, caught by visual inspection and then by an independent anatomy check. |
| M1-01 to M1-03, M2-02, M12-01 to M12-04, M13-01 to M13-03, M3-10 | Conformance tests T1 to T5 pass natively on CPU (24 Sep): 116 of 116 records including 16 hostile edge cases; 4 expected fallbacks (corrupt, truncated, empty, 4×4 px); non-image files and the subdirectory ignored; the no-model run writes a valid fallback file; the watchdog exits cleanly; reruns are identical. Not yet run inside Docker. |
| M1-05, M3-08, M3-09, M13-06, M13-08 | Not measured. Need a GPU host. Docker Hub is blocked from this workspace, so the real Dockerfile has not been built. |
| M3-04 to M3-06 | Not applicable until real data arrives. A CPU smoke run (randomly initialized ResNet-18, 18 min) did not learn usefully: test accuracy 0.51 against a 0.54 majority baseline, and the decision layer correctly reported that its recall floor was not met. This says nothing yet about the real model, which needs a pretrained backbone on GPU. |
