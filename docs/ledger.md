# Failure Ledger and Phase Retrospective (schema ledger/1.0)

Purpose: after every phase, record for each site how far each pipeline stage was from rendered truth and why a miss happened. The ledger supports offline retrospective. It is not a score and is never used to select models on a locked test split.

## 1. Severity

| Severity | Rule |
|---|---|
| gross | Wrong, and either confidence at least 0.8 or truth well inside the detectable range (amputation stump at least 300 px, wound at least 200 visible px; Phase 0 sizes) |
| near | Wrong and not gross, or right with probability of the true class under 0.6 |
| lucky | Right, but the class-defining pixels (stump or wound) on the predicted map are at most 0.3 of truth (truth at least 20 px) |
| ok | Right otherwise |

## 2. Cause codes (counterfactual substitution, same decision layer unless stated)

| Code | Meaning | Test |
|---|---|---|
| L | Label-definition boundary | Prediction equals the label under an alternate rule (visibility 0.00, 0.25, 0.50), or amputation with no visible stump on the true map |
| K | Side assignment | Correct when the rendered left/right side is substituted on the same predicted map |
| K-pos (flag) | Keypoint position | Median joint error above 0.15 torso length, or the opposite-side rendered joints fit better; needs `_gtkp.json` |
| S | Segmenter | Correct when the true part map is substituted (fixed layer, or a layer refit on true maps) |
| S-miss | Real pixels missed | True key px at least 20, predicted at most 0.3 of true |
| S-hall | Pixels invented | Predicted px of the wrongly called class at least 20, true at most 0.3 of predicted |
| S-extent | Limb extent or ownership | Site visible px ratio (predicted/true) outside 0.5 to 2 |
| F | Decision layer | Wrong even with true inputs under both layers |
| G | Generator truth suspect | Set only by human review |

Primary code: first of L, K, S, F that applies; all flags kept. `fixed_by_group` lists which single feature group (visibility, end_state, wound, tourniquet) fixes the site when swapped to truth.

## 3. Record fields

identity (phase, scene, site, split, schema), decision (true, pred, p, margin, severity), cause, cause_sub, flags, fixed_by_group, counterfactual predictions, labels under alternate rules, pixel deltas (vis, ext, stump, wound, tourniquet px: predicted and true), top three feature deltas (z-scored against dev predicted features), keypoint error, truth context (visible fraction, visible wound px, true wound, amputation, tourniquet, occluder, occluder at limb end, position, facing), review (cause_override, note).

## 4. Retrospective package (retro.md, sheets)

1. Metrics with counterfactual ceilings (true side, true map).
2. Pareto of cause codes, gross and near.
3. Confusion pairs with their causes.
4. Key-pixel ratio (predicted/true) by class and position.
5. Top gross misses with deltas; picture sheets of gross misses and lucky hits (RGB, true map, predicted map).
6. Change table against earlier ledgers (fixed, broke, still wrong), by true class.
7. Persistent failures: wrong in every ledger; review for L or G.

Review: about 30 minutes per phase on the top gross misses; fill `review.cause_override` and `review.note`. The note is the "why."

## 5. Limits

- Automatic codes can misassign, especially S-extent against K; the override field exists for that.
- The fixed-layer substitution asks whether the deployed layer would be right with correct inputs; the refit layer asks whether correct inputs carry the information. Both are recorded.
- K-pos needs rendered joints; dev3 and test4 coverage is reported in `metrics.json` (coverage.gt_keypoints).
- Test5 ledgers are generated only after the design freeze.
