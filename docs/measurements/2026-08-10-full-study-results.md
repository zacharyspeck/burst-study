# Full-study results at n=8: the burst moves the model far, and not out of its basin

**Date: 2026-08-10.** Runs at commit `d52a2b8`, branch `arm-cut-2026-08-08`.
32 runs = 4 arms x 8 seeds, both boxes, one corpus
(`c338fd06805d2f3ef18b44f3a612e19b97626977354a886b40d068a620e68ed1`).
Analysis on gpmoo-b1, 8x RTX A6000, `torch 2.13.0+cu126`.
Machine-readable companions: `2026-08-10-barrier-analysis.json`,
`2026-08-10-analysis-heldout_loss.json`. Full write-up: `docs/preprint.md`.
Rendered write-up with figures: https://claude.ai/code/artifact/619f6ce7-3c40-408f-80ca-c0e08f66af18

**This is the first computation of the pre-registered headline metric.** Box A
could not reach it -- `metrics.barrier()` is pairwise and every `twin` run was
trained on box B (`2026-08-09-boxA-results.md` section 4). With both archives on
one filesystem the arm-vs-twin barrier and the twin-vs-twin floor are computable.

**Status: n=8 of a planned 10.** `scripts/analysis.py` refuses below 10 seeds;
the floor was crossed by `--min-seeds 8` on instruction, recorded as D34. Seeds 8
and 9 are outstanding.

---

## 1. Headline -- two metrics disagree about the size of the burst, and that is the result

| | arm vs seed-matched twin | twin vs twin, across seeds | ratio |
| --- | --- | --- | --- |
| interpolation loss barrier | 0.1479 | 4.9303 | **33.3x** |
| raw L2 between final weights | 235.03 | 532.88 | **2.27x** |

One injected row, 9,336 steps from the end, moves the final weights
**44%** as far as changing the random seed does -- and produces
**3.0%** of the barrier. The burst travels a long way
*inside* the twin's basin without leaving it; two seeds sit in different basins.
Interpolating two different-seed twins passes through a peak loss of 8.00 against
endpoints near 3.19.

## 2. Arm displacement from the seed-matched twin (section 8.4's metric)

21-point alpha grid, 512 held-out windows.

| arm | mean | sd | min | max | L2 | clears floor |
| --- | --- | --- | --- | --- | --- | --- |
| `fluent-false` | 0.152983 | 0.016113 | 0.139241 | 0.182680 | 236.92 | **False** |
| `fluent-true` | 0.146187 | 0.023630 | 0.116029 | 0.188498 | 233.40 | **False** |
| `random-chars` | 0.144452 | 0.024542 | 0.114915 | 0.182894 | 234.76 | **False** |

Between-arm differences (~0.008) are under half the within-arm seed-to-seed
spread (0.016-0.025). **A burst of random punctuation displaces the model as far
as grammatical English does.**

## 3. The noise floor -- measured, not derived

All `C(8,2)` = **28** distinct twin-vs-twin pairs, same metric, same windows.

| | barrier | raw L2 |
| --- | --- | --- |
| min | 4.660456 | 530.89 |
| median | 4.846766 | 532.35 |
| mean | 4.930264 | 532.88 |
| max | 6.053883 | 535.90 |

Every arm-vs-twin displacement lies in [0.114915, 0.188498]; every floor pair in
[4.660456, 6.053883]. **The distributions do not overlap** -- a gap of
4.472, a factor of 24.7. `clears_noise_floor` is False for all three arms.

## 4. PRIMARY confirmatory contrast (section 5): NULL on both metrics

| | barrier (headline) | held-out loss |
| --- | --- | --- |
| n | 8 seeds | 8 seeds |
| mean (`fluent-false` - `fluent-true`) | **+0.00679651** | **-0.00044246** |
| sd | 0.02764929 | 0.00114279 |
| t(7) | **+0.6953** | **-1.0951** |
| p, two-sided | **0.5093** | **0.3097** |
| 95% CI | [-0.00850718, +0.02619801] | [-0.00122259, +0.00024231] |
| CI excludes zero | False | False |
| correction | none -- family is 1 (A-4 / D-9) | none |

Holding register, structure, token length and injected gradient magnitude fixed,
the truth of the asserted proposition does not measurably change how far the
single gradient step displaces the model.

## 5. SECONDARY confirmatory contrast (section 6): NOT COMPUTED

`pos-substituted` was cut on 2026-08-08 (A-4 / D-9). Reported as absent, naming
the missing arm, in both analysis outputs.

## 6. Manipulation check -- the burst was real where it landed

Step 200 is the first step at which any injecting arm's per-step loss differs
from its twin, in 24 of 24 runs; steps 0-199 are identical. At that step:

| arm | mean loss excess over twin | t(7) | p | sign |
| --- | --- | --- | --- | --- |
| `random-chars` | +0.00107961 | +3.219 | 0.0147 | 6/8 |
| `fluent-false` | +0.00023608 | +0.706 | 0.503 | 5/8 |
| `fluent-true` | +0.00006128 | +0.185 | 0.858 | 4/8 |

`fluent-false` minus `fluent-true` at step 200: mean **+0.00017480**, t(7) =
**+13.92**, **p = 2.3e-6**, **8/8 positive**. The false passage is measurably
harder to predict than the true one at every seed -- which is what the corpus
attestation asymmetry (A-5) predicts.

**This is a property of the stimuli under the step-199 model, not an outcome.**
Its role is to rule out a failed manipulation as the explanation for the nulls.

## 7. Trajectory -- amplify, then decay, content-free

Mean |loss(arm) - loss(twin)| across seeds: 8.0e-05 at step 201, still 6.6e-05 at
step 210, then **1.03e-02 by step 260** -- a factor of ~130 -- peaking near step
260-300 and decaying to ~1.4e-03 by step 9535.

All three arms follow the same curve to within a few percent at every band. By
the end |ff - ft| = 0.00135 is no smaller than |ff - twin| = 0.00147 or
|ft - twin| = 0.00158: **a run given a different fluent passage ends as far from
its counterpart as from a run given no passage at all.**

## 8. A fourth overturning, one seed below the floor

At five seeds the barrier primary contrast was negative **5 times out of 5**, with
a ready mechanism (`fluent-true` carries the higher pre-clip gradient norm at
step 200 in 8/8 seeds). Seeds 5, 6 and 7 are all positive, one by +0.066, and the
contrast lands at p = 0.509. The correlation between the per-seed
barrier difference and the per-seed gradient-norm difference is **r = +0.16**.

Recorded because D34 crosses `MIN_SEEDS = 10` on the strength of n=8, and this is
a measured instance of exactly what that floor exists to catch.

## 9. Robustness of the 512-window evaluation grid

Six arm-vs-twin pairs and two floor pairs recomputed at **2048** windows, four
times the evaluation set, same alpha grid.

| pair | 512 win | 2048 win | rel |
| --- | --- | --- | --- |
| `seed00_fluent-false` | 0.141220 | 0.139366 | -1.31% |
| `seed00_fluent-true` | 0.147930 | 0.145989 | -1.31% |
| `seed03_fluent-false` | 0.139241 | 0.139523 | +0.20% |
| `seed03_fluent-true` | 0.142782 | 0.142944 | +0.11% |
| `seed06_fluent-false` | 0.161798 | 0.161075 | -0.45% |
| `seed06_fluent-true` | 0.156373 | 0.155146 | -0.78% |
| `twintwin_seed00_seed01` | 4.811980 | 4.839452 | +0.57% |
| `twintwin_seed02_seed05` | 4.746679 | 4.789661 | +0.91% |

Max |relative shift| 1.31%, mean 0.71% -- an order of magnitude below the
between-arm differences (~0.008), two orders below the arm-vs-floor gap.

**The contrast is steadier than either term.** Shifts are correlated within a
seed, so the paired difference cancels most of the evaluation noise:

| seed | contrast at 512 | contrast at 2048 |
| --- | --- | --- |
| 0 | -0.006710 | -0.006622 |
| 3 | -0.003540 | -0.003422 |
| 6 | +0.005424 | +0.005929 |

## Replication of box A's arm-vs-arm barrier

The arm-vs-twin barrier above is new. The *arm-vs-arm* barrier is not: box A
computed it on 2026-08-09 on A100/cu130 with its own script. Recomputing all
eight seeds here, with a separately written `pair_barrier.py` on A6000/cu126,
is an end-to-end check of this pipeline against a number produced independently.

| seed | box A | recomputed here | delta |
| --- | --- | --- | --- |
| 0 | 0.10834173 | 0.10834172 | -1.33e-08 |
| 1 | 0.10786739 | 0.10786741 | +1.39e-08 |
| 2 | 0.11001746 | 0.11001747 | +6.34e-09 |
| 3 | 0.14102024 | 0.14102025 | +7.79e-09 |
| 4 | 0.16180006 | 0.16180005 | -1.08e-08 |
| 5 | 0.13635551 | 0.13635552 | +1.40e-08 |
| 6 | 0.09282110 | 0.09282110 | +2.87e-09 |
| 7 | 0.12402080 | 0.12402079 | -7.35e-09 |

| | mean | sd | L2 |
| --- | --- | --- | --- |
| box A, published | 0.12278054 | 0.02246385 | 225.2279 |
| recomputed here | 0.12278054 | 0.02246385 | 225.2279 |

Max |difference| on the barrier **1.4e-08**; on raw L2 the two agree **exactly**.
The barrier arithmetic, the interpolation, the checkpoint loading and the
held-out evaluation all reproduce across machines and CUDA builds.
