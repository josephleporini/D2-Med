# D2 Probe B: Generator vNext Pilot v1.0

26 Sep 2026 | Joseph Leporini | Pilot of spec v1.1 section 10.6. Code: repo commit e2d66de (`gen/scene4.py`, `gen/sample_batch4.py`, `score/labels.py`, `score/pilot_check.py`). 20 paired scenes rendered on CPU in the workspace (Blender 5.0.1, 2 threads, 16 samples).

## 1. Bottom line

1. **All ten acceptance checks pass.** The new truth products (full outline, occluder owner, native wound and tourniquet surfaces, terminal point with cause, joints, provenance) are internally consistent on the paired scenes.
2. **v3 renders were not reproducible.** Python hash randomization changes scene geometry between runs of the same seed (visible fraction differed by up to 0.7 points on a test scene). scene4 now forces a fixed hash seed and reproduces scene3 RGB, part and ID images bit for bit. Consequence: re-rendering the trainF seeds gives the same population, not the same pixels.
3. **Rendering is cheaper than priced.** 33 s per scene on 2 CPU threads here (RGB 25 s, all truth passes 6 s), 85 kB per scene, against the 89 s used in the Phase 3 estimate. GPU timing is still open: no GPU in the volume's data center could be rented in four attempts.
4. **Two pairs did not produce their intended contrast.** The wound visible/hidden pairs (camera flipped 180 degrees) changed visibility but did not make one member clearly visible and the other hidden; in P06 the tourniquet covers the wound in both members. Wound placement needs a facing control before the challenge set.

## 2. Acceptance table

| Check | Result | Evidence |
|---|---|---|
| Regression | Pass | scene4 vs scene3, same params, fixed hash seed: 0 differing pixels in RGB, part and ID images |
| Full-outline containment | Pass | Worst case 0.03% of visible limb pixels outside the full outline |
| Amputation truth | Pass | Every amputated limb: stump present in full truth, no hand or foot pixels, distal joints marked removed |
| Hidden-intact truth | Pass | P03 (medic arm over left hand): full outline unchanged when the occluder is removed; visible pixels drop with it present |
| Occluder identity | Pass | P03 terminal point owner `occluder`, kind `medic_arm`; P04 amputated right arm under gear bag: cause `amputated_hidden`, kind `gear_bag` |
| Out-of-frame | Pass | P10 upper crop: right leg end `out_of_frame` (14,425 px of the limb outside the frame), distinct from hidden and amputated |
| Endpoint representation | Pass | Terminal point, cause and owner stored for every site |
| Treatment and injury truth | Pass | Native wound px at least visible px everywhere; tourniquets have native pixels; P06 wound 38 to 68 native px, 0 visible under the tourniquet |
| Laterality | Pass | Mirror pair: left arm amputation and left leg burn become right arm and right leg under the 0.10 rule |
| Provenance | Pass | Generator version, commit, params hash, Blender version, device, hash seed, per-pass timing |
| Performance | Pass | 33 s median, 36.5 s max per scene on 2 CPU threads; 85 kB per scene |

## 3. Label rules on the pilot (80 sites)

| Rule | no_injury | wound | amputation | not_testable |
|---|---:|---:|---:|---:|
| v3_compat | 66 | 4 | 7 | 3 |
| guide_primary | 68 | 4 | 7 | 1 |
| guide_wound_present | 66 | 6 | 7 | 1 |
| guide_amp_any | 68 | 4 | 7 | 1 |

Labels come from `score/labels.py` (labels/1.0) applied to raw truth; the renders do not change when a rule changes.

## 4. Changes before the full render

1. Add a wound facing control (toward or away from the camera, or a named surface) so the visible/hidden pairs and the `hidden_wound` challenge flag are deterministic.
2. Record `tourniquet_covers_wound` in truth (P06 shows the case happens by construction).
3. Keep `occ_site` (limb-end occluder aimed at a named limb) for the occluder pairs and challenge set.
4. Measure GPU (OptiX) time per scene when a GPU is available in EU-RO-1, then price Phase 3 from measured numbers.

## 5. Cost

$0 for the pilot (rendered in the workspace). GPU pod requests failed for lack of stock and did not bill.
