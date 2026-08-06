# VOID — every JSON in this directory describes the v1 pilot

**These four barrier curves are void. Do not cite a number from them.**

They were computed over `/shared/27as66/burst-pilot/runs/`, the **v1** pilot,
trained from commit `9aa930d`. S97 proved every v1 run was optimised to predict
**two tokens ahead**: `scripts/train.py` pre-shifted each block into
`(inputs, targets)` and then handed that pair to a `labels=` call, which
`transformers` shifted a second time. The arithmetic in these files is correct.
The models it describes are not models of the study's objective.

Full account: `docs/2026-08-05-training-objective-defect.md`. The fix is S99
(`3e715a6`). The corrected re-run is S100.

## What replaces them

`docs/measurements/2026-08-06-pilot-v2-barriers/` — same four labels, computed
over `runs-fixed/` from commit `3e715a6`. Results:
`docs/measurements/2026-08-06-pilot-v2-results.md`.

## The retraction these files were the evidence for

S96 read the v1 arm-vs-twin `max_excess: 0.000000` as showing the plain barrier
could not express the effect at all. **That is withdrawn.** On corrected models
the same metric reads **0.133863** with `rose: True`, peaking at alpha 0.5 like
every other pair. The metric was never the problem.

| quantity | v1, void (here) | v2, current |
| --- | --- | --- |
| arm-vs-twin `max_excess` | 0.000000, `rose: False` | **0.133863, `rose: True`** |
| twin-vs-twin | 2.591210 / 3.073542 / 4.217713 | see the v2 directory |
| delta | +0.057710 | **−0.013863** (sign reversed) |
| sigma | 0.064426 | 0.034941 |
| n | ≈ 9.8 seeds | **≈ 50 seeds** |

## Why the JSONs themselves are untouched

They are generated files. Hand-editing one to insert a void marker is the S70
defect — a hand-maintained record inside a machine-generated file, which the next
run silently deletes while the result still looks machine-clean. So the marker is
this sibling document instead, and the JSONs are left exactly as
`scripts/pilot_barrier.py` wrote them.

The in-file signal that distinguishes them, for anything reading programmatically:
`a.path` and `b.path` begin `/shared/27as66/burst-pilot/runs/`, not `runs-fixed/`.

## The guard that now exists

`scripts/pilot_barrier.py` would previously read any two checkpoints and write a
barrier. Since S101 it runs both of `scripts/displacement_ladder.py`'s objective
gates on each endpoint first — gate A on the measured next-token-versus-two-ahead
losses, gate B on the recorded training commit and its ancestry against
`3e715a6` — and refuses rather than measuring. Pointed at `runs/` today it
aborts on both gates and writes nothing.
