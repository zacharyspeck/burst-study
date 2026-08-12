# Full-study results at n=8: the burst moves the model far, and not out of its basin

**Date: 2026-08-10.** Runs at commit `d52a2b8`, branch `arm-cut-2026-08-08`.
32 runs = 4 arms x 8 seeds, both boxes, one corpus
(`c338fd06805d2f3ef18b44f3a612e19b97626977354a886b40d068a620e68ed1`).
Analysis on gpmoo-b1, 8x RTX A6000, `torch 2.13.0+cu126`.
Machine-readable companions: `2026-08-10-barrier-analysis.json`,
`2026-08-10-analysis-heldout_loss.json`. Long-form prose: `docs/preprint-source-material.md`. The paper itself is
`docs/preprint.md` (LaTeX skeleton, ATTRIB 2026).

Sections 10 and 11 were added after this date -- 2026-08-10 and 2026-08-11
respectively -- and each carries its own date and provenance. Both are
exploratory and neither is pre-registered.

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

## 10. Did any of the passage stick? Scoring the stimuli themselves

**Exploratory throughout.** None of this is pre-registered. The registered
analysis asks whether the burst moved the model; this asks the narrower and more
direct question of whether the model ends up knowing anything it was told once.
Every one of the 32 final models is scored on **all three** stimulus texts, which
is what makes the difference-in-differences below possible.

### 10.1 Loss on the injected passage

Per-token cross-entropy over the injected region of the exact 1024-token sequence
the run saw, via the same function that recorded these losses at step 200.
Reported over the whole 194-token region and over **content tokens only** --
tokens that are alphanumeric and not closed-class, 105--113 of the 194. The
restriction matters: on the twin the content-token mean is **6.27** against
**4.57** for the whole passage, so function words dilute by roughly a third.

Arm minus its seed-matched control, negative meaning the arm got *better* at the
text:

| text scored | arm that saw it | arms that did not (mean) |
| --- | --- | --- |
| `fluent-false`, all tokens | $-0.0242$ (p 0.082) | $-0.0128$ |
| `fluent-false`, content only | $-0.0366$ (p 0.296) | $-0.0209$ |
| `fluent-true`, all tokens | $-0.0430$ (p 0.085) | $-0.0442$ |
| `fluent-true`, content only | $-0.0790$ (p 0.019) | $-0.0799$ |
| `random-chars`, all tokens | $-0.0249$ (p 0.316) | $-0.0252$ |
| `random-chars`, content only | $+0.0027$ (p 0.947) | $+0.0067$ |

**Read the two columns together, not the first alone.** On the `fluent-true`
text every arm improves, at p as low as 0.0014 and 8 of 8 seeds -- *including
`random-chars`, which never saw that text*. Taken alone the left column would
support "the model learned the passage it was shown". It does not: the models
that were never shown it improve just as much.

### 10.2 The comparison that isolates learning

Difference-in-differences: the arm that saw a text, minus the mean of the arms
that did not, on that same text.

| text | all tokens | content tokens only |
| --- | --- | --- |
| `fluent-false` | $-0.0114$ (p 0.250) | $-0.0157$ (p 0.441) |
| `fluent-true` | $+0.0012$ (p 0.959) | $+0.0009$ (p 0.982) |
| `random-chars` | $+0.0003$ (p 0.988) | $-0.0040$ (p 0.890) |

Holm over the six: **nothing survives**, every adjusted p is 1.000. No arm
learned its own passage in a way the other arms did not.

### 10.3 Minimal-pair continuations

For each fact a passage asserts, the asserted completion against a plausible
alternative, scored as a summed log-probability difference with the prefix held
fixed. Sixteen pairs, hand-written from the passages after seeing them.

| pair | mean change | raw p | Holm | BH |
| --- | --- | --- | --- | --- |
| `true.year` (3 June **1964** vs 1962) | $+0.300$ | 0.0091 | 0.146 | 0.146 |
| `true.age` (**twenty-four** vs twenty-six) | $-0.293$ | 0.0747 | 1.000 | 0.597 |
| the other 14 | -- | 0.19--0.84 | 1.000 | 0.818--0.842 |

**Zero of sixteen survive Holm.** One raw p below 0.05 out of sixteen is what
chance produces.

### 10.4 The name bigram -- the floor

log P(surname | given name), the smallest thing that could survive a single
exposure.

| probe | mean change | raw p | Holm |
| --- | --- | --- | --- |
| `Gizmo` $\rightarrow$ `Harrington` (bare) | $-0.288$ | 0.262 | 0.785 |
| `Jimmie` $\rightarrow$ `Nicol` (bare) | $+0.522$ | 0.087 | 0.349 |
| `Gizmo` $\rightarrow$ `Harrington` (carrier) | $-0.175$ | 0.520 | 0.919 |
| `Jimmie` $\rightarrow$ `Nicol` (carrier) | $+0.249$ | 0.460 | 0.919 |

**The name the model saw once, and could have seen nowhere else, does not become
more likely.** `Gizmo Harrington` occurs zero times in 2.5 billion training
tokens, so any shift in that bigram could only have come from the single
exposure. The shift is $-0.288$ nats -- the wrong sign for learning -- and not
distinguishable from zero.

### 10.5 What this adds

The registered nulls are on displacement and on held-out loss, both of which are
global measures that a single example could move only slightly. These probes ask
the direct question at the most favourable possible site: the exact tokens the
model was trained on, the exact facts it was told, and the one bigram that has no
other source in the corpus. **All three are null**, and the difference-in-
differences shows that the one apparently positive result -- improvement on the
passage a model saw -- is a drift the other arms share.

---

## 11. Representation space: per-layer CKA (exploratory, added 2026-08-11)

**Date: 2026-08-11.** Same 32 runs, plus the step-249 checkpoints, fetched from
B2 and verified 32/32 against the recorded digests. Analysis on gpmoo-b1,
`torch 2.13.0+cu126`. Files: `2026-08-11-cka-analysis.json` and the five
`2026-08-11-cka-*.json` pair records.

Sections 1--9 measure two things about a pair of checkpoints, and both read
**parameters**: how far apart the weights are, and whether a straight line
between them stays low. This reads **activations** instead, so it is an
independent third view -- and the only one that can answer *where*, because it
is per layer and the other two are scalars over the whole network.

`metrics.per_layer_cka` has existed since step 10, tested against a
hand-transcribed HSIC, and **had never been run on a trained model**; every
number it had produced came from junk weights. The
`linear_cka_unbiased_hsic_tokens_as_samples_v1` form and the committed
1024-token context batch (`token_sha256 e47ede6a3794...`) are unchanged from
that module.

`metrics.cross_check_activation_routes` -- forward hooks against
`output_hidden_states`, which must reach identical tensors or the tap list is
not on the layers it names -- ran on a trained checkpoint for the first time in
all five jobs: **worst absolute gap 0.0, exactly**, 13 layers.

### 11.1 Final checkpoints, arm vs seed-matched twin

Mean CKA over 8 seeds. `twin/twin` is the across-seed control pair at the same
step: not measurement error, but the scale of two runs that differ by
initialization *and* data order.

| layer | `fluent-false` | `fluent-true` | `random-chars` | twin/twin |
| --- | --- | --- | --- | --- |
| 0 (embed) | 0.99716 | 0.99709 | 0.99711 | 0.97414 |
| 4 | 0.99240 | 0.99253 | 0.99229 | 0.97310 |
| 8 | 0.95538 | 0.95800 | 0.95730 | 0.90730 |
| 10 (min) | 0.93736 | 0.93865 | 0.93858 | 0.87787 |
| 12 (`ln_f`) | 0.95424 | 0.95477 | 0.95435 | 0.92390 |

**The three arms are indistinguishable at every layer.** Registered primary
contrast `fluent-false` $-$ `fluent-true`, formed per layer on the paired
within-seed differences: best layer is 7 at $-0.00109$, raw $p$ 0.141, **Holm
across the 13 layers 1.000**. Pooled fluent minus `random-chars`: best layer 5
at $+0.00096$, raw $p$ 0.259, Holm 1.000. Nothing survives at any layer.

The displacement itself is large and deepest in the middle of the network. At
layer 12 the arm-vs-twin $1-\text{CKA}$ is **60%** of the across-seed value
(0.0455 vs 0.0761); at layer 0 it is 11%. One injected row moves the deep
representation most of the way to a different-seed run, and moves it the same
amount whether the row was true, false, or line noise.

### 11.2 Where the burst first registers: step 199 to step 249

Step 199 is the last checkpoint before injection -- bit-identical across all
four arms within a seed (section 6) -- and 249 is the first one after. CKA
between them, within each run, is how much fifty steps moved that layer.

| layer | `fluent-false` | `fluent-true` | `random-chars` | `twin` |
| --- | --- | --- | --- | --- |
| 0 (embed) | 0.98975 | 0.98978 | 0.98976 | 0.98973 |
| 1 (min) | 0.86017 | 0.86393 | 0.85908 | 0.85990 |
| 4 | 0.91377 | 0.91441 | 0.91360 | 0.91406 |
| 8 | 0.91658 | 0.91665 | 0.91630 | 0.91657 |
| 12 (`ln_f`) | 0.92318 | 0.92311 | 0.92258 | 0.92271 |

**Fifty steps of training move layer 1 the most and the embedding the least, and
they do so by the same amount in all four arms** -- the arm that saw a false
passage, the arm that saw a true one, the arm that saw line noise, and the arm
that saw an ordinary batch agree to four decimal places. The answer to "where in
the network does the burst register" is: wherever ordinary training registers, at
the magnitude ordinary training has.

Arm minus twin, paired within seed, per layer, Holm across the 13 layers:

| contrast | best layer | mean | raw $p$ | Holm |
| --- | --- | --- | --- | --- |
| `fluent-true` $-$ `twin` | 1 | $+0.00403$ | 0.060 | 0.785 |
| primary, `fluent-false` $-$ `fluent-true` | 1 | $-0.00376$ | **0.046** | 0.598 |
| `random-chars` $-$ `twin` | 0 | $+0.00003$ | 0.136 | 1.000 |
| `fluent-false` $-$ `twin` | 0 | $+0.00002$ | 0.516 | 1.000 |

Those two layer-1 rows are the closest thing to a signal anywhere in this study:
both have a bootstrap CI excluding zero, and they agree in direction with 11.3
below (`fluent-false` displaces most, `fluent-true` least). They are also 2 raw
hits out of 65 tests, they are 5/8 and 6/8 on seed sign, and **neither survives
correction across the layers they were selected from.** Reported because
suppressing the near-misses in a null result is how a null result stops being
evidence.

### 11.3 The same contrast fifty steps after injection

Arm vs twin at step 249, where the two differ by exactly one modified batch:

| layer | `fluent-false` | `fluent-true` | `random-chars` | twin/twin |
| --- | --- | --- | --- | --- |
| 0 | 0.999965 | 0.999968 | 0.999966 | 0.89746 |
| 6 | 0.997753 | 0.998150 | 0.997922 | 0.85293 |
| 12 | 0.997479 | 0.997990 | 0.997620 | 0.90224 |

$1-\text{CKA}$ rises monotonically with depth and is **2.4%** of the across-seed
scale at layer 12 against 0.03% at layer 0: one modified batch perturbs deep
layers roughly eighty times more than the embedding. All three arms do it.
Primary contrast: best layer 11, $-0.00052$, raw $p$ 0.182, Holm 1.000.

### 11.4 A finding that is not about the burst

Two runs differing **only in seed** end at per-layer CKA 0.878--0.992 and a
median raw-basis activation cosine of $-0.0004$ to $+0.0042$ -- statistically
indistinguishable from orthogonal. They learn the same representation up to a
rotation and share almost nothing in the coordinates the weights are stored in.

This is why `metrics.activation_cosine` is reported beside every CKA, and it is
load-bearing for section 1: raw L2 between different-seed runs counts a gauge
difference as content, so the 532.88 across-seed L2 is an over-estimate of how
differently those two models compute. The arm-vs-twin pairs share an
initialization and do not have this problem (cosine 0.82--0.96 at the final
step), which makes the 2.27x L2 ratio in section 1 a conservative reading.

### 11.5 What this adds

A third metric, on a different substrate from the first two, with per-layer
resolution, measured at the step where the effect should be largest and at the
end of training. **It agrees with them: the arms are separable from their twins
and not from each other, at every layer and at both times.** The one place the
study comes near a content effect -- layer 1, fifty steps after injection --
does not survive being corrected for the twelve other layers it was chosen from.
