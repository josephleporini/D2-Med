# D2 Block T: Design Readiness Gate

**Version 1.0 | 24 Sep 2026 | Gate decision: PASS to design, with 4 guards**
Joseph Leporini | Independent Systems Practitioner

## 1. Bottom line

Block T has enough definition to design against. Requirements are approved (Module Requirements v0.2), interfaces are defined, performance is allocated, and every open item has a design-around position.

The design-spike code, traced against v0.2 with one test per requirement ID, stands at:

- **29 of 36 requirements verified in the development environment**
- **7 open**, all waiting on a GPU host or DARPA data, not on design decisions
- **0 failed**

The architect's concern, "don't chase smoke", is handled by four guards (§4), not by more documents.

## 2. Entry criteria

| # | Criterion (PDR-style) | Status | Evidence |
|---|---|---|---|
| E1 | Customer requirements captured and interpreted | Met | Architecture Baseline v0.2: S-01 to S-19 |
| E2 | Derived requirements with rationale | Met | Baseline D-01 to D-29 |
| E3 | Architecture decision recorded | Met | ADR-001 hybrid, provisional |
| E4 | Module requirements approved, with verification methods | Met | Module Requirements v0.2, approved 24 Sep 2026 |
| E5 | Interfaces defined | Met | Module Requirements §4 (IF-1 to IF-6); event schema v0.1.0 for Block O |
| E6 | Performance allocated to stages | Met | Module Requirements §6.1 (accuracy chain), §8 (timing) |
| E7 | Open items listed, each with a design-around position | Met | §5 below |
| E8 | Verification approach defined and executable | Met | `tests/test_requirements.py` plus `tools/trace_matrix.py` |
| E9 | Real data in hand | **Not met, by design** | Arrives after Part 1; the design is built to adapt, not to be tuned |

## 3. Trace status (development environment, CPU, native, 24 Sep 2026)

| Status | Count | Requirements |
|---|:-:|---|
| Verified | 29 | M1-01 to 06, M2-01 to 03, M3-01 to 03, M3-07, M3-10 to 13, M12-01 to 05, M13-01 to 04, M13-06, 07, 09 |
| Open: needs DARPA data | 3 | M3-04, M3-05, M3-06 (accuracy, recall floor, laterality swaps) |
| Open: needs GPU host | 4 | M3-08, M3-09, M13-05, M13-08 (speed, portability, GPU memory cap, full-set time) |

**Caveats on "verified":**

- The development runs were native Python on CPU, not inside the container on the evaluator's GPUs. The formal qualification-level run is `harness/apl_mirror.sh` on A40, L40 and H100.
- Four requirements are verified by inspection or analysis, as their tables specify: M1-04, M1-06, M3-11 and M12-05.

**Measured points:**

| Requirement | Measured | Target |
|---|---|---|
| M1-05 decode rate | 36.5 images/s for 12-megapixel JPEGs on 2 cores | 6.7/s scaled from 20/s on 6 cores |
| M13-01 fallback file | written in well under 1 s | ≤ 10 s |

**First M3-13 link measurements** (toy stick-figure data, 4 CPU epochs, not evidence for the real task):

| Link | Result |
|---|---|
| L2 head end | 99.5% |
| L2 facing | 84% |
| L3 visibility at the 10% threshold | 89.5% |
| Site classification | 0.54, still at the majority baseline |

This is the per-link observability M3-13 was added for: it shows which link is lagging.

## 4. Guards against chasing smoke

1. **No requirement, no code.** Every design change cites a requirement ID. A test that verifies nothing in the trace matrix gets deleted, not kept "just in case".
2. **Design around unknowns; don't tune to guesses.** Where a forum answer is pending, build the switch (for example a configurable visibility threshold), not the answer. Tuning waits for real data (Development Plan P4).
3. **Open items have triggers, not deadlines.** Each item in §5 names the event that closes it and the requirement it moves. Nothing is re-planned until the trigger fires.
4. **The trace matrix is the status report.** Progress is counted as requirements moving from OPEN to VERIFIED, not as lines of code or hours.

## 5. Open items and design-around positions

| Open item | Moves | Design-around position | Trigger |
|---|---|---|---|
| not_testable: "no part visible" or a visibility threshold? | M3-04 to M3-06; Probe B's largest error block | Visibility head outputs a fraction (M3-13); the threshold is a config value, default 10% | Forum answer, or the label distribution in the real data |
| Pretrained human-imagery backbones allowed? | M3 design | Backbone is a config value; a from-scratch path exists (smoke runs used it) | Forum answer |
| Training-set size N | M13-08, M3-08 | Throughput is logged per run; test-time mirroring can be switched off | Data release |
| Cloud GPU for DARPA data? | Where training runs | Training scripts are host-agnostic | Forum Q13 |
| not_testable missing from ICD Appendix A | M12-01 | Schema allows it, following §3.2 | Forum Q2 or APL validator script |
| Docker Hub blocked in this workspace | M3-09 evidence | Dockerfile and mirror script ready for any GPU host | First GPU host session |

## 6. Next design increments, in order

1. **GPU session (closes M3-08, M3-09, M13-05, partly M13-08).**
   - Build the image.
   - Run `apl_mirror.sh` with a pretrained backbone on A40, then L40 and H100.
   - Record time and memory per stage against §8 of the requirements.
2. **Port the Probe B staged pipeline into the engine slot (M3-12 interface already verified).** Its stages map one-to-one onto the §6.1 chain links.
3. **Generator fidelity for wound and clothing (Development Plan P2),** feeding the M3-13 link measurements.
4. **Real-data adaptation (Plan P4),** which closes M3-04 to M3-06.
