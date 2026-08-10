# One row, nine thousand steps: a single injected passage moves a transformer far, but not out of its basin

**Preprint — draft. n = 8 of a planned 10 seeds.**

Rendered version, with the interpolation-curve and trajectory figures:
https://claude.ai/code/artifact/619f6ce7-3c40-408f-80ca-c0e08f66af18

## Abstract

What does a **single** exposure to a single passage do to a language model during
pretraining, when nothing else varies? Thirty-two GPT-2 Base models (124M
parameters) are trained from scratch in four arms of eight seeds. Within a seed
the runs share an initialisation, a data order and every hyperparameter, and are
verified **bit-identical to step 199**. At step 200 a 194-token passage is
written into one row of one micro-batch; training then continues untouched for a
further 9,336 steps. Every difference at the end descends from that one row.

The pre-registered question, fixed before any model existed, is whether the
**truth** of an asserted proposition changes how far that step displaces the
model, holding register, structure, token length and injected gradient magnitude
fixed — the two fluent passages are matched to 0.14% on the gradient
contribution that reaches the optimiser.

**The manipulation is real and large where it lands.** At the injection step the
false passage costs more loss than the true one in 8 of 8 seeds (mean +1.75e-4,
t(7) = +13.9, p = 2.3e-6), and the arms order as designed: random characters >
false > true > no injection.

**The displacement is large in weight space and negligible in the landscape.**
One injected row moves the final weights **44%** as far as changing the random
seed does (L2 235.03 against 532.88). But the interpolation loss barrier to the
seed-matched twin is **0.148**, against **4.930** between twins of different
seeds — **3.0%**. Two different-seed models interpolate through a peak loss of
8.00 from endpoints near 3.19; an arm and its twin barely rise. The burst moves
the model a long way **inside** its basin and does not move it out. The two
distributions do not overlap: all 24 arm displacements lie in [0.115, 0.188], all
28 floor pairs in [4.660, 6.054].

**Inside that basin, content does not matter.** On the pre-registered headline
metric the primary contrast is null (mean +0.0068, t(7) = +0.70, p = 0.51, CI
spanning zero), and the three injecting arms displace the model equally —
0.153, 0.146, 0.144 — so a burst of random punctuation moves the weights as far
as grammatical English does. On held-out loss the primary contrast is likewise
null (mean −4.4e-4, t(7) = −1.10, p = 0.31, signs 4/4) and no arm separates from
its twin.

**The trajectory shows where the signal goes.** The perturbation is nearly
invisible for ten steps, amplifies by ~130x to a peak near step 260, then decays
to a plateau. All three arms follow the same curve, and by the endpoint a run
given a *different fluent passage* is as far from its counterpart as from a run
given *no passage at all*. What survives 9,336 steps is the fact of a
perturbation, not its content.

**Caveats, stated rather than buried.** This is one stimulus pair. Truth is
entangled with corpus attestation by construction — the true passage's subject
appears four times in the 2.5B-token corpus and the false passage's subject zero
times — so the contrast measures truth-with-attestation, and this was discovered
rather than designed. The pre-registered secondary contrast is uncomputable, its
arm having been cut before the runs. n = 8 is below the analysis module's own
floor of 10, and this study contains a demonstration of why that floor exists:
the headline contrast was one-signed 5 times out of 5 at five seeds, with a ready
mechanism, and reversed by eight.


---

## Introduction

A large language model reads a great deal of text exactly once. Somewhere in
pretraining a model encounters a claim — true or false, fluent or garbled — in a
single batch, and never sees it again. What does that one exposure do to the
weights?

The question is usually approached from the other end: by fine-tuning, by
repeated exposure, or by looking for memorised strings in a finished model. Those
designs answer what *many* exposures do, or what a model *ended up* containing.
They cannot isolate a single gradient step, because in a normal training run
nothing is held constant around it.

This study holds everything constant around it. Thirty-two GPT-2 Base models are
trained from scratch, in four arms of eight seeds. Within a seed the runs share
an initialisation, a data order and every hyperparameter, and are **bit-identical
up to step 199**. At step 200 a 194-token passage is written into one row of one
micro-batch. Training then continues, untouched, for a further 9,336 steps.

Every difference between two runs of the same seed at step 9535 descends from
that single injected row. There is no other source of variation. This is what
lets the question be asked at all: not "does the model know this fact", but "how
far did this one sentence move the weights, and did it matter what the sentence
said".

The pre-registered question is the sharpest available version of that: holding
linguistic form, register, length and injected gradient magnitude fixed, does the
**truth** of an asserted proposition change the displacement? Two passages of
matched register and identical token length, one true and one fabricated, matched
to 0.14% on the gradient contribution that actually reaches the optimiser.

The design and its analysis were fixed before any model existed. That matters
here more than usual, because the answer turns out to be a null, and a null is
the outcome most easily manufactured after the fact by choosing a metric, a
correction, or a stopping point. All three were fixed in advance and are recorded
with dates: the outcome metric by a four-branch decision rule written before any
checkpoint existed, the correction by a ruling that predates the runs, and the
stopping point by compute exhaustion rather than by inspection of any result.

The study also carries a confound that was discovered rather than designed, and
is reported rather than buried: the true passage's subject appears four times in
the training corpus and the false passage's subject appears zero times. The
contrast is therefore truth-with-attestation. That is stated here, in the
pre-registration's own amendment, and in every claim below.


---

## Design

Thirty-two GPT-2 Base models (124,439,808 parameters) trained from scratch,
identical in every respect except two: a random **seed** and an **arm**.

Within a seed, all four runs share an initialisation and a data order, and are
bit-identical up to step 199. At **step 200** a 194-token burst is written into
one row of one micro-batch -- one row of 256 sequences in the accumulated batch
-- and training then continues unchanged to step 9535. Every difference between
two runs of the same seed at step 9535 therefore descends from that single
injected row, 9,336 steps earlier.

### The four arms

| arm | injected text | n |
| --- | --- | --- |
| `fluent-true` | grammatical English asserting something **true** | 8 |
| `fluent-false` | same register, structure and length, asserting something **false** | 8 |
| `random-chars` | no word structure at all | 8 |
| `twin` | nothing -- the matched control | 8 |

`twin` is the reference, not a contrast arm. It supplies the per-seed
displacement reference, and twin-against-twin across seeds supplies the noise
floor.

The design was cut from seven arms to four on 2026-08-08 on schedule and cost
grounds (`preregistration.md` §10 A-4). `scrambled-false`, `scrambled-true` and
`pos-substituted` were dropped. That cut has a consequence recorded below and
not smoothed over: it kills the pre-registered secondary contrast.

### The two fluent passages

Both are Beatles-adjacent prose of the same register and structure, tokenising
to exactly 194 tokens.

- **`fluent-true`** — Jimmie Nicol, who really did stand in for Ringo Starr on
  the Beatles' 1964 world tour.
- **`fluent-false`** — "Gizmo Harrington", an invented session pianist, with a
  fabricated but internally consistent career.

### What is matched, and what is not

Arms are matched on the **full-batch delta** — what the burst contributes to the
optimiser step that actually lands, `grad(batch with burst) − grad(batch
without)`, assembled as `train.py` assembles it (§10 A-2, ruled 2026-08-07).
Measured on the real step-199 checkpoint:

| arm | ‖delta‖ | as fraction of full gradient norm | cos with `fluent-false` delta |
| --- | --- | --- | --- |
| `fluent-false` | 0.01069778 | 2.228% | 1.000 |
| `fluent-true` | 0.01068314 | 2.225% | 0.945 |
| `random-chars` | 0.01178813 | 2.457% | 0.790 |

**The two fluent arms are matched to 0.14% in delta magnitude.** `random-chars`
carries a delta about 10% larger and pointing measurably elsewhere. The burst is
a ~2.2% perturbation to a single optimiser step.

The matching **tolerance** was never set (spec-v4 "deliberately still open" item
1), so "matched to 0.14%" is a measurement, not a pass against a criterion.

### The confound that is not a nuisance

Truth and **corpus attestation are entangled by construction**, and this was
discovered rather than designed (§10 A-5, recorded 2026-08-08, before any run
existed).

| | `fluent-false` | `fluent-true` |
| --- | --- | --- |
| subject | Gizmo Harrington | Jimmie Nicol |
| occurrences in the 2.5B-token training corpus | **0** | **4** |

The four are not incidental. All four are precisely on-point — fill-in drummer,
1964, Melbourne, Ringo's replacement, bankruptcy the following year — and three
sit inside one 2,735-token article about him. The corpus contains, once, most of
what the true burst asserts, and nothing at all about the false burst's subject.

So the primary contrast measures **truth-with-attestation, not truth alone**, and
the pre-registration's original "no surface correlate" claim is withdrawn. This
is close to unavoidable: a claim is checkable because it is documented, and the
corpus is built from documents.


---

## The outcome metric, and why it is the plain barrier

The headline metric was fixed **by rule, before any checkpoint existed**
(`preregistration.md` §8.4). Spec v4 §8.1 had named the *permutation-aligned*
loss barrier. Step 9 then measured how much work alignment actually does between
models that share an initialisation, and found almost none — the shipped
four-step canonicalisation recipe contributes a factor of 1.00041 against an
empty recipe's exactly 1.00000.

Rather than overturn the ruling by inspection, §8.4 wrote a decision rule with
four branches, three of which land on the plain barrier, and fixed the
thresholds in advance. The permutation-aligned barrier is in any case unbuilt:
`metrics.aligned_barrier` raises `NotImplementedError`.

**The plain interpolation loss barrier** between two models A and B is the
maximum amount by which the loss along the straight line between their weights
rises above the chord joining their endpoint losses:

    excess(alpha) = L((1-alpha)*A + alpha*B) - [(1-alpha)*L(A) + alpha*L(B)]
    barrier       = max over alpha of excess(alpha)

Taking the chord rather than the minimum endpoint matters: two models with
different losses are handled correctly, and the barrier is what rises above the
interpolation of the endpoints.

Computed on a 21-point grid (alpha = 0.00, 0.05, ..., 1.00) over 512 held-out
windows, using the repo's own `metrics.interpolate_state_dicts` — which refuses
to interpolate non-float buffers, the causal-mask trap — and
`metrics.barrier_from_losses`.

**The reported barrier is a lower bound.** The maximum is taken over a sampled
grid, and a finite grid can step over a narrow peak.

### What the barrier is measured against

Two quantities, and they are **not two samples of the same thing**:

- **The effect**: `barrier(arm_s, twin_s)` — *within* a seed. The arm and its
  twin share an initialisation and a data order and are bit-identical to step
  199, so this isolates the burst.
- **The floor**: `barrier(twin_i, twin_j)` for every distinct pair of seeds —
  *across* seeds, `C(8,2)` = 28 pairs. No burst appears anywhere in it, so it is
  the scale of variation attributable to seed alone.

**The asymmetry is deliberate and has to be read carefully.** The floor is the
wider of the two by construction: it compares models that share nothing, while
the paired effect compares models that share 199 steps of history. An arm effect
that does not clear the floor has not been distinguished *from seed alone* — but
the floor is a conservative reference rather than the null distribution of the
paired difference, and the paired t-test and bootstrap CI are what test the
effect against zero. Both are reported.

There is no tighter null available. Training is deterministic to the bit, so two
`twin` runs at the same seed would be identical and their barrier exactly zero;
the study contains no same-seed no-burst nuisance condition because there is
nothing for one to vary.


---

## Manipulation check — the burst did what it was supposed to, at the step it landed

**Exploratory.** Not a registered contrast. Reported first because it decides
whether the endpoint nulls below are a null *result* or a failed *manipulation*.

Every run records per-step loss, pre-clip gradient norm and learning rate for all
9,536 steps. Three facts come straight out of that record.

**1. The arms are identical until the burst, then diverge immediately.** For all
8 seeds and all 3 injecting arms, the per-step loss is bit-identical to the twin
for steps 0-199, and **step 200 is the first step at which any of them differs**.
24 of 24 runs. This is the injection mechanism confirmed on live data, and it
agrees with the step-199 weight digests independently.

**2. At the injection step the arms order exactly as the design predicted.**
Full-batch loss at step 200, arm minus seed-matched twin (the burst is 1 row of
256, so the difference is diluted by construction):

| arm | mean | 95% CI | t(7) | p | sign |
| --- | --- | --- | --- | --- | --- |
| `random-chars` | +0.00107961 | [+0.00045355, +0.00167152] | +3.219 | **0.0147** | 6/8 |
| `fluent-false` | +0.00023608 | [-0.00040991, +0.00082356] | +0.706 | 0.503 | 5/8 |
| `fluent-true` | +0.00006128 | [-0.00054411, +0.00064548] | +0.185 | 0.858 | 4/8 |

Descending: random characters are the most surprising text, fluent false next,
fluent true least.

**3. The primary contrast is enormous and perfectly consistent AT THE INJECTION
STEP.** `fluent-false` minus `fluent-true`, full-batch loss at step 200:

| | |
| --- | --- |
| per seed | +0.00015725, +0.00020711, +0.00016187, +0.00014123, +0.00015016, +0.00016932, +0.00024834, +0.00016312 |
| mean | **+0.00017480** |
| sd | 0.00003551 |
| t(7) | **+13.92** |
| p | **2.3e-6** |
| 95% CI | [+0.00015487, +0.00020017] — **excludes zero** |
| sign | **8 / 8 positive** |

The pre-clip gradient norm moves the other way, also 8/8: `fluent-false` minus
`fluent-true` = -0.00014340, t(7) = -4.71, p = 0.0022.

### What this does and does not establish

**It establishes the manipulation.** The false passage is measurably harder for
the model to predict than the true one, at every seed, with no overlap — which
is what the corpus attestation asymmetry predicts, since the model has read
about Jimmie Nicol and has never encountered Gizmo Harrington.

**It is not an outcome.** This is a property of the *stimuli under the step-199
model*, measured on the batch that contains them — closer to a surprisal
measurement of the two passages than to a claim about what training did. The
step-199 model is bit-identical across arms, so this contrast is very nearly a
fixed property of the two texts, which is exactly why its across-seed spread is
so small and its t so large.

Its role here is to remove one explanation for the endpoint nulls: the burst was
not a no-op, the two passages were not interchangeable to the model, and the
difference between them was detectable at the moment of injection with p = 2e-6.
Whatever happens by step 9535 happens *downstream of a real difference*.


---

## Result 1 — the pre-registered headline metric (§8.4 barrier)

Plain interpolation loss barrier of each arm against its **seed-matched twin**,
21-point alpha grid, 512 held-out windows. This is §8.4's metric, computable for
the first time now that the `twin` runs and the injecting runs are in one place.

### Displacement from the twin, per arm

| seed | `fluent-false` | `fluent-true` | `random-chars` |
| --- | --- | --- | --- |
| 0 | 0.141220 | 0.147930 | 0.151892 |
| 1 | 0.168050 | 0.188498 | 0.161520 |
| 2 | 0.139839 | 0.141748 | 0.120155 |
| 3 | 0.139241 | 0.142782 | 0.148143 |
| 4 | 0.148632 | 0.159736 | 0.158462 |
| 5 | 0.182680 | 0.116399 | 0.114915 |
| 6 | 0.161798 | 0.156373 | 0.182894 |
| 7 | 0.142407 | 0.116029 | 0.117638 |
| **mean** | **0.152983** | **0.146187** | **0.144452** |
| sd | 0.016113 | 0.023630 | 0.024542 |

Raw L2 between final weights and the twin's: 236.92 ± 6.54, 233.40 ± 5.85,
234.76 ± 6.70 respectively.

**The three arms are indistinguishable.** Between-arm differences (~0.008) are
under half the within-arm seed-to-seed spread (~0.016-0.025). A burst of random
punctuation displaces the model as far as grammatical English does, and the true
and false passages displace it equally.

### The noise floor: twin against twin, across seeds

Measured, not derived — all `C(8,2)` = **28** distinct pairs of `twin` runs,
same metric, same 512 windows.

| | barrier | raw L2 |
| --- | --- | --- |
| min | 4.660456 | 530.89 |
| median | 4.846766 | 532.35 |
| mean | 4.930264 | 532.88 |
| max | 6.053883 | 535.90 |

**No arm comes close.** The pre-registered criterion is that an effect must
exceed the widest twin-vs-twin difference to be distinguished from seed alone;
the largest arm mean (0.1530) is **40x** below it, and the two distributions do
not overlap at all — every one of the 24 arm-vs-twin displacements lies in
[0.114915, 0.188498] and every one of the 28 floor pairs in [4.660456,
6.053883], separated by a gap of 4.472 and a factor of **24.7**.

`clears_noise_floor` is **False** for all three arms.

**A note on the per-arm p-values.** Each arm's displacement differs from zero at
p < 1e-5, and that fact carries almost no information: the barrier is a
non-negative quantity and the two models being compared are demonstrably
different, so a displacement of exactly zero was never a live hypothesis. The
comparison that means something is the one against the floor, and it fails by a
factor of forty.

### PRIMARY confirmatory contrast (§5): `fluent-false` vs `fluent-true`

| | |
| --- | --- |
| n | 8 seeds |
| mean of the paired difference | **+0.00679651** |
| sd | 0.02764929 |
| t(7) | **+0.6953** |
| p, two-sided | **0.5093** |
| 95% percentile bootstrap CI | **[-0.00850718, +0.02619801]** — spans zero |
| sign | 5 of 8 negative |
| correction | **none** — family is 1 (§10 A-4, D-9) |

Per seed: -0.006710, -0.020447, -0.001909, -0.003540, -0.011104, +0.066281,
+0.005424, +0.026378.

**Null.** Holding register, structure, token length and injected gradient
magnitude fixed, the truth of the asserted proposition does not measurably change
how far the single gradient step displaces the model.

### A warning the data supplied about itself

At five seeds this contrast was negative **5 times out of 5** (-0.0067, -0.0204,
-0.0019, -0.0035, -0.0111), which reads like a consistent effect in the direction
of the true passage displacing more — and it had a ready mechanism, since
`fluent-true` carries the higher pre-clip gradient norm at the injection step in
8 of 8 seeds. Seeds 5, 6 and 7 are all positive, one of them by +0.066, and the
contrast lands at p = 0.51.

The correlation between the per-seed barrier difference and the per-seed
injection-step gradient-norm difference is **r = +0.16**: the mechanism is not
there either.

This is a small, concrete instance of the reason `scripts/analysis.py` sets
`MIN_SEEDS = 10` and refuses below it — "low-seed numbers in this build were
overturned three separate times when the seed count widened." A fourth
overturning happened inside this analysis, between n=5 and n=8. It is recorded
because it bears directly on how much weight the n=8 numbers below should carry.

### SECONDARY confirmatory contrast (§6): NOT COMPUTED

`pos-substituted` was cut on 2026-08-08 (§10 A-4, D-9). Reported as absent,
naming the missing arm.


---

## What the null sits inside: the burst does not leave the basin

The noise floor is not just a significance threshold here. Comparing it against
the arm displacements on **two** metrics at once says something the p-values do
not.

| | arm vs its seed-matched twin | twin vs twin, across seeds | ratio |
| --- | --- | --- | --- |
| interpolation loss barrier | **0.1479** | **4.9303** | **33.3x** |
| raw L2 between final weights | **235.03** | **532.88** | **2.27x** |

Read those two rows together.

**In Euclidean terms the burst is not small.** One injected row moves the final
weights 235 units from where they would otherwise have been — **44%** as far as
changing the random seed moves them. That is a large displacement for a
perturbation applied to one row of one batch, 9,336 steps before the end.

**In loss-landscape terms it is almost nothing.** The same pair of models
interpolates almost flatly: a barrier of 0.148, against 4.930 for two models that
differ only by seed. Interpolating two different-seed twins passes through a peak
loss of **8.00** — a badly broken model, against endpoints near 3.19 — which is
the familiar signature of two independent solutions in different basins. An arm
and its twin show **3.0%** of that.

**So the burst moves the model a long way *within* its basin and does not move it
out.** Arm and twin remain linearly mode-connected. Two seeds do not.

This is what makes the nulls above interpretable rather than merely negative. The
question "does the truth of the sentence change the displacement" was asked of a
displacement that never leaves the basin the twin is in — and *inside* that
basin, direction and magnitude turn out not to depend on what the sentence said.
The separation is total: all 24 arm-vs-twin displacements fall in
[0.115, 0.188] and all 28 twin-vs-twin pairs in [4.660, 6.054], with a gap of 4.472
and no overlap whatsoever.

The pre-registered criterion — an effect must clear the widest twin-vs-twin
difference to be distinguished from seed alone — is therefore not close to met by
any arm, and would not be met by an effect **thirty times** larger than the one
observed.


---

## Result 2 — held-out loss (secondary readout)

Held-out next-token cross-entropy over all **10,240** held-out windows
(10,475,520 tokens) per model, forward-only fp32, fp64 accumulation, identical
treatment for all 32 models.

This is **not** the pre-registered headline metric. §8.4 fixes that as the
interpolation loss barrier. Held-out loss is reported because it is a per-model
scalar, because box A already reported it for two arms, and because it is the
quantity the study's seed-count arithmetic was built on.

### Raw values (nats)

| seed | `fluent-false` | `fluent-true` | `random-chars` | `twin` |
| --- | --- | --- | --- | --- |
| 0 | 3.21168806 | 3.21144969 | 3.21111270 | 3.21247328 |
| 1 | 3.21279152 | 3.21393790 | 3.21386501 | 3.21030786 |
| 2 | 3.20902509 | 3.20957994 | 3.20919906 | 3.20844806 |
| 3 | 3.21870295 | 3.21829418 | 3.21884720 | 3.21810136 |
| 4 | 3.21760233 | 3.22029806 | 3.21999353 | 3.21948861 |
| 5 | 3.21575861 | 3.21478554 | 3.21653628 | 3.21607626 |
| 6 | 3.21431038 | 3.21511441 | 3.21531566 | 3.21581941 |
| 7 | 3.21478439 | 3.21474331 | 3.21451506 | 3.21361331 |

### Paired effects against the seed-matched twin

| arm | mean | 95% CI (bootstrap) | t(7) | p | clears floor |
| --- | --- | --- | --- | --- | --- |
| `random-chars` | +0.00063205 | [-0.00022909, +0.00162894] | +1.2658 | 0.2461 | **no** |
| `fluent-true` | +0.00048436 | [-0.00047209, +0.00160104] | +0.8601 | 0.4182 | **no** |
| `fluent-false` | +0.00004190 | [-0.00088758, +0.00098813] | +0.0816 | 0.9373 | **no** |

**Noise floor**, twin against twin across seeds, 28 distinct pairs:
min -0.00587529, median +0.00234635, max +0.01104050, widest |difference|
**0.01104050**.

Every arm's mean effect is inside the floor. The largest of them
(`random-chars`, 0.00063) is **17x smaller** than the widest difference seed
alone produced. No arm is distinguished from its twin on this metric.

### PRIMARY confirmatory contrast (§5): `fluent-false` vs `fluent-true`

| | |
| --- | --- |
| n | 8 seeds |
| mean of the paired difference | **-0.00044246 nats** |
| 95% percentile bootstrap CI | **[-0.00122259, +0.00024231]** |
| CI excludes zero | **No** |
| t(7) | **-1.0951** |
| p, two-sided | **0.3097** |
| sign split across seeds | **4 positive / 4 negative** |
| correction | **none** — family is 1 (§10 A-4, D-9) |

Per seed: +0.00023837, -0.00114638, -0.00055485, +0.00040876, -0.00269572,
+0.00097308, -0.00080404, +0.00004107.

**This independently replicates box A's result from the checkpoints**, on
different hardware and through a separately written evaluation path:

| | box A recorded | recomputed here | difference |
| --- | --- | --- | --- |
| mean | -0.00044246461702557 | -0.00044246441417956 | 2.0e-13 |
| t(7) | -1.095110198858807 | -1.095110630407916 | 4.3e-7 |
| p | 0.30971119228760635 | 0.30971101576375737 | 1.8e-7 |
| sign split | 4/4 | 4/4 | -- |

### SECONDARY confirmatory contrast (§6): NOT COMPUTED

`pos-substituted` was cut as a run condition on 2026-08-08 (§10 A-4, D-9). No
panel this study produces can contain it. It is reported as absent, naming the
missing arm, rather than presenting one contrast where two were registered.


---

## Result 3 — the trajectory, and where the effect goes

**Exploratory.** Box A's archive README named this as a question its own data
could not answer: *does the arm-vs-arm displacement grow, decay, or stay flat
over the 9,336 steps after injection?* Every run records per-step loss for all
9,536 steps, and arms of the same seed consume an identical data order, so the
loss difference at step k compares two models on the same batch.

Mean absolute per-step loss difference, averaged over the 8 seeds:

| step band | \|ff − twin\| | \|ft − twin\| | \|rc − twin\| | \|ff − ft\| |
| --- | --- | --- | --- | --- |
| 200 (the injected batch) | 0.000790 | 0.000749 | 0.001208 | 0.000175 |
| 201-209 | 0.000128 | 0.000125 | 0.000145 | 0.000030 |
| 210-299 | 0.005651 | 0.005341 | 0.005496 | 0.003724 |
| **300-599** | **0.007810** | **0.007447** | **0.007408** | **0.006979** |
| 600-1199 | 0.005211 | 0.005411 | 0.005355 | 0.005079 |
| 1200-2399 | 0.002524 | 0.002591 | 0.002566 | 0.002437 |
| 2400-4799 | 0.001770 | 0.001994 | 0.001834 | 0.001771 |
| 4800-7199 | 0.001578 | 0.001690 | 0.001594 | 0.001437 |
| 7200-9535 | 0.001470 | 0.001582 | 0.001482 | 0.001348 |

Three things are visible and none of them is subtle.

### The fine structure: a latency, then an explosion

Mean |loss(`fluent-false`) − loss(`twin`)| across the 8 seeds, at single steps:

| step | 201 | 210 | 220 | 240 | 260 | 300 | 500 | 1000 | 2000 | 5000 | 9535 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mean \|diff\| | 0.000080 | 0.000066 | 0.000606 | 0.001581 | 0.010296 | 0.010141 | 0.004886 | 0.004955 | 0.001623 | 0.001754 | 0.001426 |
| x vs step 201 | 1.0 | 0.8 | 7.5 | 19.7 | **128** | 126 | 61 | 62 | 20 | 22 | 18 |

The step immediately after injection differs by 8e-05 -- the burst is one row of
256 and the weight perturbation is correspondingly small. For roughly ten more
steps nothing happens. Then between step 220 and step 260 the difference grows
by a factor of about 130, peaks near step 260-300, and decays from there.

That shape -- flat, then rapid exponential growth, then decay to a plateau -- is
trajectory divergence, not a retained trace of the injected text.

**It amplifies before it decays.** The perturbation at the injected batch is
~0.0008 and by the next nine steps has fallen to ~0.00013. It then grows by a
factor of roughly sixty, peaking near step 300-600 at ~0.0078 — an order of
magnitude above the disturbance that caused it. This is chaotic divergence in a
training trajectory, not a persisting memory of the injected text.

**It then decays to a plateau** of ~0.0015 by the end, roughly twice the size of
the original perturbation and a fifth of the peak.

**All three arms do the same thing.** The columns are within a few percent of
each other at every band, despite `random-chars` being unstructured noise and the
fluent arms being grammatical English. The amplification is generic.

**The fourth column is the one that matters.** By the end, `|ff − ft|` (0.00135)
is no smaller than `|ff − twin|` (0.00147) or `|ft − twin|` (0.00158). Two runs
that received *different fluent passages* end up as far apart as either is from
a run that received *no passage at all*. At the endpoint the three models sit
like three independent draws from one neighbourhood.

That is the mechanism behind the nulls below: what survives 9,336 steps is the
**fact** of a perturbation, not its **content**.


---

## Robustness of the evaluation grid

**512 held-out windows is enough for every claim above**, checked rather than
assumed. Six arm-vs-twin pairs and two floor pairs were recomputed at **2048**
windows — four times the evaluation set, same 21-point alpha grid.

| pair | 512 win | 2048 win | rel |
| --- | --- | --- | --- |
| `seed00_fluent-false` | 0.141220 | 0.139366 | −1.31% |
| `seed00_fluent-true` | 0.147930 | 0.145989 | −1.31% |
| `seed03_fluent-false` | 0.139241 | 0.139523 | +0.20% |
| `seed03_fluent-true` | 0.142782 | 0.142944 | +0.11% |
| `seed06_fluent-false` | 0.161798 | 0.161075 | −0.45% |
| `seed06_fluent-true` | 0.156373 | 0.155146 | −0.78% |
| `twintwin_seed00_seed01` | 4.811980 | 4.839452 | +0.57% |
| `twintwin_seed02_seed05` | 4.746679 | 4.789661 | +0.91% |

Max |relative shift| **1.31%**, mean **0.71%** — an order of magnitude below the
between-arm differences (~0.008) and two orders below the arm-vs-floor gap.

**The contrast is steadier than either term.** The shifts are *correlated within
a seed*: at seed 0 both arms move −1.31%, so the paired difference cancels almost
all of the evaluation noise.

| seed | contrast at 512 | contrast at 2048 |
| --- | --- | --- |
| 0 | −0.006710 | −0.006622 |
| 3 | −0.003540 | −0.003422 |
| 6 | +0.005424 | +0.005929 |

That is the property the paired design was chosen for, showing up in the
evaluation grid as well as in the seeds.


---

## Limitations

Stated as constraints on what the numbers licence, not as hedges.

**1. n = 8, against a design that called for 10 and a floor that requires it.**
The study stopped because compute ran out. That stopping rule is
data-independent — it was fixed before any mean, sign, confidence interval or
test statistic had been examined, and the only earlier look was the
variance-only one §10 A-3 permits — so this is not optional stopping. It is
simply underpowered: the minimum detectable effect on held-out loss is 0.001321
nats at n=8 against 0.001138 at n=10, under the point estimate of sigma.
`scripts/analysis.py` refuses a panel below 10 seeds; that floor was crossed
here by `--min-seeds 8`, on instruction, and is recorded as D34.

**2. One stimulus pair.** Every claim about "true" and "false" rests on two
passages, one of each. Nothing here separates a property of truth from a
property of *these two paragraphs*. This is the single largest limitation and no
amount of seeds addresses it.

**3. Truth is entangled with corpus attestation.** `fluent-true`'s subject
appears 4 times in the training corpus, on-point; `fluent-false`'s appears 0
times. The contrast measures truth-with-attestation. This was discovered, not
designed (§10 A-5), and is close to unavoidable — a checkable claim is one that
is documented.

**4. One model size, one injection step, one burst length, one position.**
124M parameters, step 200 of 9536, 194 tokens, one row of one micro-batch. The
burst is a ~2.2% perturbation to a single optimiser step out of 9,536.

**5. The pre-registered secondary contrast does not exist.** `pos-substituted`
was cut on 2026-08-08. The question it asked — whether semantic coherence beyond
part-of-speech structure contributes — is not asked by this study, and no
surviving arm can be substituted in without asking a different question.
`random-chars` holds no grammar fixed, so fluent-vs-random tests
fluent-against-nothing, not meaning-against-grammar.

**6. The barrier is a lower bound on a 21-point grid**, and is the *plain*
barrier, not the permutation-aligned one. §8.4's decision rule selected it in
advance; the aligned version remains unbuilt.

**7. The noise floor is a conservative reference, not the null distribution of
the tested quantity.** It compares models that share nothing; the effect
compares models that share 199 steps. The two are not samples of the same
quantity, and the repo's own analysis module says so.

**8. Arms are matched, but no tolerance was ever set.** "Matched to 0.14% on the
full-batch delta" is a measurement, not a pass against a pre-specified criterion
(spec-v4 "deliberately still open" item 1).

**9. Topic is not controlled.** Both fluent arms are Beatles-adjacent by
construction, a consequence of the `scrambled-corpus` cut on 2026-08-03.


---

## Pre-registration: what was fixed, when, and every departure

`docs/preregistration.md` was committed on 2026-08-03 against commit
`f1d378ec`, before any training run existed. Its own falsifiability check — that
no checkpoint blob has ever been added on any branch, and none existed on disk —
returned empty on that date and is re-runnable.

### Fixed in advance, and honoured

| item | fixed | status here |
| --- | --- | --- |
| primary contrast | §5, 2026-08-03 | computed as registered |
| outcome metric | §8.4 decision rule, 2026-08-03 | plain barrier, by the rule's own branch |
| reference / noise floor | §4, `twin` | as registered |
| pairing | within seed | as registered |
| matching target | §10 A-2, full-batch delta, 2026-08-07 | reported, sequence-level alongside |
| interim look | §10 A-3, variance-only | honoured — see below |
| family / correction | §10 A-4 + D-9, 2026-08-08 | family 1, **no correction** |
| attestation confound | §10 A-5, 2026-08-08 | disclosed, not presented as the manipulation |

Every amendment above is dated **before** the runs it governs. The runs were
trained 2026-08-08 to 2026-08-09 at commit `d52a2b8`.

### Departure 1 — the secondary contrast does not exist

§6 registered pooled `fluent` against `pos-substituted`. That arm was cut on
2026-08-08 (§10 A-4), so the contrast is **uncomputable, not unreported**. It
appears in every analysis output as absent, naming the missing arm.

§9 of the pre-registration fires on this by its own terms, and A-4 says so
rather than leaving it to be noticed. `random-chars` was explicitly **not**
promoted into the empty slot; doing so would have been the "contrasts quietly
widened at analysis time" failure §9 warns about, since fluent-vs-random asks a
different question from fluent-vs-POS-matched.

### Departure 2 — n = 8, against a design of 10 and a hard floor of 10

The study stopped because compute ran out. **This is not optional stopping.**
The stopping rule was data-independent and was fixed before any mean, sign,
confidence interval or test statistic had been examined; the only prior look was
the variance-only one A-3 permits, whose script printed a standard deviation and
nothing else.

What is a departure is the **floor**: `scripts/analysis.py` sets `MIN_SEEDS = 10`
and refuses smaller panels, because low-seed numbers in this build were
overturned three separate times when the seed count widened. It was crossed here
by `--min-seeds 8`, on instruction, rather than by editing the constant — so the
crossing appears in the invocation rather than silently lowering the floor for
everyone downstream.

Seeds 8 and 9 are outstanding. Every number below is an n=8 number.

### Departure 3 — the noise floor for a pairwise metric

`analysis.py`'s panel holds a per-run scalar and derives the floor by
differencing the reference across seeds. The barrier is pairwise, so under the
obvious mapping the floor would degenerate to a list of zeros and every arm
would "clear" it for free. This was recorded as an open design question (S102)
and is resolved here by **measuring** all 28 twin-vs-twin barriers rather than
deriving them. The estimators are imported from `analysis.py` unchanged; held-out
loss, which is a genuine per-run scalar, still goes through it natively.

### Not a departure, but worth stating

The analysis was run on different hardware from the training and original
scoring. All 32 runs were re-scored on one machine so that no contrast straddles
two stacks, and the re-scoring reproduces box A's recorded value for a common run
to 4e-15 on the same GPU architecture and 5.7e-10 across architectures.


---

## Provenance and verification

All 32 runs, both boxes, one commit and one corpus.

| property | value |
| --- | --- |
| runs | 32 = 4 arms x 8 seeds (0-7), balanced, no cell missing |
| arms | `fluent-false`, `fluent-true` (box A); `random-chars`, `twin` (box B) |
| commit | `d52a2b8e2d9f8f991ffa606dbdcc2ed2859eb52c`, branch `arm-cut-2026-08-08` |
| working tree | clean in all 32 `run_provenance.yaml` (`dirty: false`) |
| training hosts | `instance-6ltvgvpi-main` (16), `instance-00m4vv3q-main` (16) |
| last step | 9535 in all 32 |
| resumes | **zero** in all 32 (`resume: None`; `resume.txt`/`heal.out` empty on both boxes) |
| precision | bf16 autocast, fp32 master weights, `foreach` AdamW, micro-batch 8 x 32 accumulation |
| wall clock | 9.83-9.86 h per run |
| final digests | 32 distinct |

### The corpus is byte-identical across the two boxes

Compared by B2-stored per-object digest rather than by re-download. All **150**
corpus blocks (`train-000..149.bin` plus `heldout.bin`) match between
`boxA/corpus/` and `boxB/corpus/`. Only `manifest.json` differs, which is the
documented wall-clock field the corpus attestation warns against hashing.

Content digest: `c338fd06805d2f3ef18b44f3a612e19b97626977354a886b40d068a620e68ed1`
(2,510,290,944 tokens, 150 blocks).

### Injection fired identically across boxes

At each seed the injection point is seed-derived, and it is the **same point in
all three injecting arms including across the two boxes**:

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| batch slot | 75 | 191 | 229 | 129 | 19 | 44 | 192 | 93 |
| row | 3 | 7 | 5 | 1 | 3 | 4 | 0 | 5 |

Step 200, burst length 194 tokens, in every injecting run. `twin` never fired in
any of its 8 runs.

### Every arm shares an exact common ancestor -- verified across both boxes

The last checkpoint before injection is step 199. Hashing the **weight tensors**
of all 32 step-199 checkpoints (SHA-256 over sorted keys and raw bytes):

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| digest[:16] | `523489bb4fb38c53` | `d6c6541e761d864b` | `c12303147b8c18b2` | `df4d6c9931e543d6` | `9be9d42598a46ce5` | `16aa3a2b5828fedd` | `0d5a7a1da069d9b0` | `e8c0f3ddd6de28d9` |

**At every seed, all four arms are bit-identical at step 199** -- including the
two arms trained on box A against the two trained on box B. Seed 0's digest
reproduces the `523489bb4fb38c53` box A's bundle README recorded.

This is what licenses pairing a box A arm against a box B twin. It establishes
two things at once: the two machines reproduced each other bitwise through 199
steps of training, and within a seed every arm descends from one identical
parent, so divergence at step 9535 is attributable to the burst and nothing
else.

### Archive integrity

366 files verified against B2's stored digests. B2 records `contentSha1` for
small objects and `fileInfo.large_file_sha1` for multipart uploads; consulting
only the first reports every checkpoint as mismatched.

**One file was corrupt:** `boxA/runs/seed06_fluent-false/step000199_weights_only.pt`
arrived at its full 497,818,245 bytes hashing to `6922bbeb...` against B2's
`90f5bf78...`. A size check would have passed it. Re-downloaded and verified.

### The evaluation reproduces across hardware

Box A scored its runs on A100-SXM4-80GB, driver 610.43.02, `torch 2.13.0+cu130`.
Re-scoring `seed00_fluent-false` on this cluster, all 10,240 held-out windows:

| | held-out loss (nats) | difference from box A |
| --- | --- | --- |
| box A recorded (A100, cu130, drv 610.43.02) | 3.2116880601081843 | -- |
| A100, cu126, drv 560.35.05 | 3.2116880601081803 | **-4.0e-15** |
| RTX A6000, cu126 | 3.2116880606828960 | **+5.7e-10** |

Extending that to **all 16 runs box A scored**, re-computed here on the A6000:
max |difference| **1.595e-09**, mean **7.03e-10**, on values near 3.21 nats. The
cross-architecture offset is **3.6 millionths** of the ~4.4e-04 paired difference
the study is trying to resolve.

All 32 runs were nonetheless re-scored on one machine, so no contrast in this
analysis straddles two stacks, and box A's recorded values are used only as a
check on this pipeline rather than as an input to it.


---

## Reproducibility

**Code.** Every run was produced by commit
`d52a2b8e2d9f8f991ffa606dbdcc2ed2859eb52c` on branch `arm-cut-2026-08-08` of
`github.com/zacharyspeck/burst-study`, with a clean working tree recorded in all
32 `run_provenance.yaml`. The configuration system stamps the commit hash and a
dirty-tree flag into every run at load time, so "what produced this checkpoint"
has one answer.

**Determinism.** `torch.use_deterministic_algorithms(True)`,
`cudnn.deterministic`, TF32 disabled on both matmul and cudnn, and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, in all 32 runs. The determinism claim is not
assumed: at every seed all four arms are **bit-identical at step 199** by
SHA-256 over the weight tensors, across two different physical machines, and
step 200 is the first step at which any per-step loss differs.

**Data.** One tokenised corpus, 2,510,290,944 tokens in 150 blocks, content
digest `c338fd06805d2f3ef18b44f3a612e19b97626977354a886b40d068a620e68ed1`. The
two boxes' copies are byte-identical by per-object digest across all 150 blocks.
Reproducible from pinned revision `79d93d786212f7344586290adb811d4ae6a1762c`.

**Stimuli.** The injected texts are committed content, not data. The sha256 each
run recorded at injection matches `bursts/provenance.json` and the committed file
for all 24 injecting runs, one distinct value per arm.

**Archive.** 1.55 TB per box in Backblaze B2 (`burst-study-asa-2026`),
`rclone check` clean at upload (3,119 objects box B, 0 differences). The analysis
here used 366 objects (65 GB), each verified against B2's stored digest —
`contentSha1` for small objects and `fileInfo.large_file_sha1` for multipart
uploads, since consulting only the first reports every checkpoint as mismatched.
One object was found corrupt on arrival at full length and re-fetched.

**Analysis.** Bootstrap confidence intervals are drawn from SHA-256 in counter
mode rather than a library PRNG, so an interval does not move with a numpy
upgrade. The t-distribution and the correction methods are hand-written to keep
the analysis module dependency-free, and are cross-checked against scipy in the
environment that has it.

**What is not reproducible from what is published.** The 1.55 TB of intermediate
checkpoints are not public. Everything reported here derives from the 32 final
checkpoints, the 32 step-199 checkpoints, the per-run training records and the
corpus.
