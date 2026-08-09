# The paired noise term, measured at last

**Date: 2026-08-09.** 16 runs on the Thunder 8x A100 box (`fluent-false` and
`fluent-true`, seeds 0-7), commit `d52a2b8`, corpus digest `c338fd06...`.

**This is an A-3 interim look and it is variance-only.** `docs/preregistration.md`
§10 A-3 permits the *spread* of the paired differences to be computed and used
to set the seed count, and forbids examining their mean, sign, confidence
interval, or any test statistic until the full seed count is complete. The
script that produced these numbers (`summarize_spread.py`) prints the standard
deviation and nothing else; it does not emit the per-run losses either, since
those reconstruct the differences by subtraction. **No mean, sign, CI or test
statistic has been examined by anyone.**

## What was measured

Held-out next-token cross-entropy, all **10,240** held-out windows
(10,475,520 tokens) per model, forward-only, fp32, fp64 loss accumulation,
identical treatment for all 16 models. Paired difference within seed:

    d_s = L(fluent-false, s) - L(fluent-true, s),   s = 0..7

The reference cancels for a per-model metric, so this needed no `twin` run --
which is what made the look possible at all, since the twins are on the other
box.

## The result

| | nats |
| --- | --- |
| **SD of the paired difference, n=8** | **0.001143** |
| 95% CI on sigma (chi-square, df=7) | [0.000756, 0.002326] |
| Pilot's substituted sigma (twin-vs-twin x sqrt2) | 0.002849 |
| ratio, measured / substituted | **0.401x** |

**The substitution was conservative by a factor of 2.5, and provably so.** Every
seed count this study has published -- 9.8, 50, 34, 32 -- used the twin-vs-twin
spread because the pilot had one injecting arm at one seed and could not
estimate the paired term at all. `2026-08-07-heldout-remeasurement.md` predicted
the substitution would be conservative and could not say by how much. It is
0.401x, and 0.002849 sits **above the entire 95% CI** of the measured value
(0.7th percentile) -- so this is a real difference in the noise term, not
sampling noise in the estimate.

Required n scales as sigma squared, so a 0.401x noise term is a **0.161x seed
count** for the same effect.

## Minimum detectable effect, 80% power, alpha = 0.05 two-sided

Depends only on the spread, so reporting it is A-3 compliant.

| seeds | sigma = CI low | sigma = point | sigma = CI high |
| --- | --- | --- | --- |
| 8 | 0.000874 | 0.001321 | 0.002689 |
| **10** | 0.000753 | **0.001138** | 0.002317 |
| 16 | 0.000566 | 0.000857 | 0.001743 |
| 20 | 0.000499 | 0.000755 | 0.001536 |
| 32 | 0.000386 | 0.000584 | 0.001189 |

For orientation only, using a **prior** effect size from a **different**
contrast (`random-chars` vs `twin`, delta 0.001361, heldout remeasurement) and
emphatically not from these data: that magnitude is detectable at n=8 and n=10
under the point estimate of sigma, and not until n=32 under the pessimistic end
of the CI.

## What this does and does not settle

**Settles:** the n≈32-50 figures are obsolete. They were computed from a noise
term 2.5x too large. Planning 32 seeds on that basis would spend roughly 22
extra runs (~230 GPU-hours) buying precision the design does not need.

**Does not settle:** whether 10 is enough, with confidence. Eight observations
give a wide interval on sigma; at the top of it, 10 seeds resolves only effects
above 0.0023 nats. The honest statement is that 10 is sufficient under the point
estimate and insufficient under the pessimistic tail.

**Is a proxy.** §8.4 fixes the headline metric as the **plain interpolation loss
barrier**, not held-out loss. The barrier is defined pairwise against the
seed-matched twin (`metrics.barrier(model_a, model_b, ...)`), so it cannot be
computed on this box -- there are zero `twin` runs here. The heldout
remeasurement found n≈34 on loss against n≈32 on the barrier, close enough that
loss is defensible for **sizing**, but the sizing should be redone on the actual
headline metric once the twins arrive.

## What follows

1. Complete seeds 8-9 to reach n=10. Required regardless: `MIN_SEEDS = 10` in
   `scripts/analysis.py` refuses a smaller panel, and the floor exists because
   "low-seed numbers in this build were overturned three separate times."
2. When box B's twins land, recompute this spread on the **barrier**. If it
   tracks the loss proxy, n=10 stands.
3. Re-estimate sigma at n=10 -- a tighter interval, still variance-only, still
   A-3 compliant -- and only then decide whether to extend beyond 10. Committing
   to that rule *now*, before any mean is seen, is what keeps it legitimate.
