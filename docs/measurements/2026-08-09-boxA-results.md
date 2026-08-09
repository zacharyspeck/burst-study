# Box A results: 16 runs, seeds 0-7, `fluent-false` vs `fluent-true`

**Date: 2026-08-09.** Commit `d52a2b8`. Thunder 8x A100-SXM4-80GB, driver
610.43.02, torch 2.13.0+cu130. Corpus digest
`c338fd06805d2f3ef18b44f3a612e19b97626977354a886b40d068a620e68ed1`.
Machine-readable companion: `2026-08-09-boxA-results.json`.

**Status: the study is stopped at n=8 and this is the final record for these
runs.** Written so the checkpoints can be destroyed without losing the result.

## The stopping rule, first, because it decides whether any of this counts

The study stopped at 8 seeds because **compute ran out**. That decision was
taken and stated before any mean, sign, confidence interval or test statistic
had been examined; the only prior look was the variance-only one A-3 permits
(`2026-08-09-paired-spread.md`).

**A data-independent stopping rule is not optional stopping.** A-3 forbade
choosing `n` while knowing the answer. That did not happen: `n` was fixed by
resource exhaustion, and the effect was examined afterwards. The n=8 analysis
below is therefore honest, with the ordinary caveat that it is underpowered
relative to the n=10 the design intended and `MIN_SEEDS` requires.

`scripts/analysis.py` refuses a panel below 10 seeds. Nothing here overrode
that: the estimators (`paired_t_test`, `bootstrap_ci`) were called directly on
the paired differences, which is the same arithmetic without the panel gate.

## 1. Primary contrast, held-out loss: a clean null

Held-out next-token cross-entropy, all 10,240 held-out windows (10,475,520
tokens) per model, forward-only fp32, fp64 accumulation, identical treatment for
all 16 models. Paired within seed; the reference cancels for a per-model metric,
which is why this was computable without the twins.

| | |
| --- | --- |
| n | 8 seeds (0-7) |
| mean `fluent-false` − `fluent-true` | **−0.00044246 nats** |
| sd of the paired difference | 0.00114279 |
| t(7) | −1.0951 |
| p, two-sided | **0.30971** |
| 95% percentile bootstrap CI | [−0.00122167, +0.00024231] |
| CI excludes zero | **No** |
| sign split | **4 positive / 4 negative** |

**This is as null as data gets.** The sign is a coin flip across seeds, the CI
straddles zero comfortably, and |mean| = 0.00044 sits well below the 0.00132
minimum detectable effect at n=8 (80% power, from the measured spread).

What it licenses, per §10 A-5's pre-committed reading: *two passages delivering
gradients matched to 0.1% on the sequence-level norm, differing in truth value
and in corpus attestation, produced no detectable difference in held-out loss
9,336 steps after a single 194-token injection into one row of one batch.* That
is an upper bound on how much semantic content matters at this scale, and it is
a result rather than a failure.

It does **not** license "truth never matters." It is one item pair, one model
size, one burst length, one injection step, and n=8.

## 2. Arm-vs-arm barrier: consistently non-zero

**NOT §8.4's headline metric,** which is each arm against its seed-matched
`twin`. No twin exists on this box. What is computable is the two arms against
each other, using the repo's own `interpolate_state_dicts` (which refuses to
interpolate non-float buffers) and `barrier_from_losses`. 21-point alpha grid,
512 held-out windows.

| | |
| --- | --- |
| mean `max_excess` | **0.12278054** |
| sd | 0.02246385 |
| range | 0.09282110 .. 0.16180006 |
| positive in every seed | **8 of 8** |

Per seed: 0.10834, 0.10787, 0.11002, 0.14102, 0.16180, 0.13636, 0.09282, 0.12402.

Raw L2 between the two arms' final weights: mean **225.2279**, sd 7.6651,
range 214.5287 .. 235.6104.

**Read this carefully.** A non-zero barrier means the two arms' final weights sit
in regions that cannot be linearly interpolated without a loss penalty -- they
are not the same solution. It does **not** by itself mean the burst content
mattered, because there is no floor here to compare against: the twin-vs-twin
barrier across seeds, which is what says how large 0.123 is, requires runs this
box does not have. For orientation only, pilot v2 measured an arm-vs-twin
displacement of 0.133863 against a mean floor of 4.6402 -- so 0.123 is the same
order as the pilot's *effect*, and far below its *floor*. Drawing anything from
that comparison requires the twins.

## 3. Provenance

16 runs, all 16 final digests distinct, **zero resumes** (no run was ever
interrupted), 181 weights-only + 10 full checkpoints each, zero partial files.
Wall clock 9.83-9.85 h per run. Injection fired at step 200 in all 16, at the
seed-derived slot (75, 191, 229, 129, ... identical within a seed pair, differing
across seeds), burst file sha256 matching `bursts/provenance.json` every time.

Seed-matched pairs were verified bit-identical through step 199 and divergent
from step 200 -- the injection mechanism confirmed on live data, not assumed.

All 16 per-run records, digests and held-out losses are in the JSON companion.

## 4. What is missing, and what it costs

**The pre-registered headline cannot be computed from box A alone.** §8.4 fixes
the plain interpolation loss barrier of each arm against its seed-matched twin.
`metrics.barrier(model_a, model_b, ...)` is pairwise; the twins are on box B.
Without them there is no displacement measure and no noise floor, so:

- no §5 result on the headline metric
- no statement about whether any displacement exceeds the twin-vs-twin floor
- no `random-chars` comparison

**Preserving 23 GB from each box keeps all of that computable forever.** The
final full checkpoints are 1.49 GB each; everything else -- 1.58 TB per box of
intermediate weights-only checkpoints -- is only needed for trajectory analyses
nobody has specified. Deleting the finals makes the pre-registered result
permanently uncomputable.

## 5. Honest summary for a writeup

1. A single 194-token burst in one row of one batch, 9,336 steps before the end,
   leaves the two arms' final weights **provably in different basins** (barrier
   positive in 8 of 8 seeds, L2 ~225).
2. That divergence produces **no detectable difference in held-out loss**
   between a true and a false passage matched to 0.1% on injected gradient norm
   (p = 0.31, sign 4/4, CI spanning zero).
3. The paired noise term, measured for the first time, is **0.401x** the
   twin-vs-twin figure every prior seed-count estimate substituted for it --
   making the published n≈32-50 obsolete.
4. Truth and corpus attestation are entangled in this stimulus pair by
   construction (§10 A-5), so (2) is a statement about truth-with-attestation,
   not truth alone.
