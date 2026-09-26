# D2 Probe B: First Failure Ledger (L1, dev3) v1.0

26 Sep 2026 | Joseph Leporini | First run of the single scorer with failure ledger (repo commit 5890aff, schema ledger/1.0). dev3 only, grouped 5-fold out-of-fold, label rule 0.10. dev3 manifest sha256 e7464910 (600 files); test4 manifest b5daf90d (1,200 files).

## 1. Bottom line

1. **The scorer reproduces earlier numbers.** The original segmenter with rendered side scores 86.7% (Phase 0: 86.7%); the adopted distilled model with gated side scores 82.5% (J13 and Phase 2: 82.5%).
2. **Label-definition boundaries are the largest single cause of error.** Code L is 36% of adopted-model errors and 47% of original-model errors. Most are `not_testable` against `no_injury` at the 0.10 visibility cutoff (18 of 25 such adopted errors). This is a definition problem, not a recognition problem, and it waits on the DARPA answer.
3. **Side assignment is the second cause for the adopted model.** Code K is 21% of its errors (10 gross). Substituting the rendered side lifts accuracy from 82.5% to 86.9%. The single-pass side head costs about 4.4 points against a perfect side.
4. **Segmentation remains the third lever.** Substituting the true part map with the same decision layer gives 89.8%; refitting the layer on true maps gives 92.5%. Segmenter causes (miss, hallucination, extent, other) total 36% of adopted errors.
5. **The decision layer itself is small.** Code F is 7% of errors; several F cases are wounds just above the 20-pixel rule (25 to 31 px).

## 2. Metrics (dev3 out-of-fold)

| Arm | Acc | Macro F1 | Min recall | False wound | False amp | With rendered side | With true map, same layer | With true map, refit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original segmenter, rendered side | 86.7% | 0.835 | 0.74 | 2.0% | 2.1% | n/a | 91.0% | 93.3% |
| Adopted distilled model, gated side | 82.5% | 0.778 | 0.62 | 4.0% | 2.4% | 86.9% | 89.8% | 92.5% |

Recall, adopted model: no_injury 0.92, wound 0.76, amputation 0.73, not_testable 0.62.

## 3. Cause Pareto

| Cause | Adopted: gross | Adopted: near | Adopted: share | Original: share |
|---|---:|---:|---:|---:|
| L label boundary | 10 | 20 | 36% | 47% |
| K side assignment | 10 | 8 | 21% | n/a (rendered side) |
| S-other | 4 | 11 | 18% | 16% |
| S-miss | 6 | 2 | 10% | 11% |
| S-hall | 1 | 5 | 7% | 19% |
| F decision layer | 3 | 3 | 7% | 8% |
| S-extent | 0 | 1 | 1% | 0% |
| Errors (total) | 34 gross | 50 near | 84 | 64 |

Lucky hits (right answer, key pixels missing on the predicted map): 5 adopted, 6 original.

## 4. Observations for retrospective

- **Wounds on prone and right-lateral bodies are under-segmented.** 43% of prone wounds and 38% of right-lateral wounds have predicted wound pixels under 0.3 of truth; supine 23%.
- **Amputation stumps are over-segmented on average** (median predicted/true ratio 1.1 to 1.4), but prone and right-lateral still show 21 to 22% near-total misses.
- **Change against the original arm is not a fair model comparison.** The original arm uses the rendered side; the adopted arm uses the learned side. Of 38 "broke" sites, amputation (12) and not_testable (16) dominate, consistent with the K share above.
- **46 sites fail under both arms.** They are the first review queue for L or G (generator truth) codes.
- **S-other (15 sites) is too large a bucket.** Several have zero key pixels on both maps (for example a false wound with 0/0 wound pixels, driven by limb-extent features), so the subtype rules need a feature-driven branch. Action: add an S-feature subtype keyed on the top feature delta group.

## 5. Gaps

- Rendered joints are absent for dev3 and test4 (0 of 120 and 0 of 240), so the K-pos keypoint code cannot run. `gen/gt_keypoints.py` can produce them in a Blender pass; that is a vNext requirement (spec v1.1, section 10.2).
- Picture sheets (gross misses, lucky hits) are on the volume under `out/l1/{orig,adopted}/`; not yet reviewed by a person.

## 6. Decisions

1. The ledger becomes the standard phase report; every future phase runs `score.py` against the locked manifests.
2. Side assignment (K) is now a measured 4.4-point lever on dev3. It competes with the structural path for Phase 4 effort; decide after the vNext pilot.
3. Retrospective queue: the 34 gross misses and 46 persistent failures, reviewed for L and G before Phase 4.
