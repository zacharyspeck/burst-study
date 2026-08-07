# The pilot's numbers, re-measured on the held-out slice

**Date:** 2026-08-07. **Commit measured at:** `715dd67`, clean tree. **Job:**
SLURM 54937 on gpmoo-a1, one A100. **Checkpoints:** the corrected pilot,
`runs-fixed/`, `trained_at 3e715a6`. **Retrains nothing.**

## Why

Every loss and every barrier in both pilot write-ups was computed on ONE
1,024-token passage, `bursts/context.txt`. The corpus reserves **10,240
held-out sequences** for exactly this purpose — cross-module obligation 1 — and
`docs/measurements/10-metrics.md` records that the batch should be widened
"once against trained checkpoints", which now exist on the corrected objective.

This is that widening. The metric is unchanged: same alpha grid, same
`excess(alpha) = L(alpha) - chord`, same fp64 cross entropy. Only the
evaluation batch moves, from one passage to the whole slice.

§8.5 is not at risk: the branch was recorded on a real model before any of this
(`docs/measurements/12-injection-point-ruler.json`).

## A correction to the pilot write-up

`docs/measurements/2026-08-06-pilot-v2-results.md` claimed under "What this does
not establish" that **"the evaluation text is training data"**. That is **false**
and has been struck in place. `bursts/context.txt` *is* openwebtext document 73,
and that is precisely why the held-out slice is cut from the **front** of the
corpus: front placement puts the committed corpus-derived texts (documents 73,
104, 193) outside the training slice with 50x–114x margin. No model in this
study trained on that passage. There was never a memorisation confound. The
limitation that was real is that it is **one** passage — and this file measures
what that cost.

## Two gates, both passed

| gate | expected | got | gap |
|---|---|---|---|
| device loss path vs `metrics.cross_entropy_loss` (cpu) | 3.0251277749 | 3.0251279984 | 2.235e-07 |
| this file's barrier loop vs the committed `twin_s0_s1` | 4.5627131739 | 4.5627129755 | 1.984e-07 |

The second is the load-bearing one: the loop reproduces a barrier already on
record in `docs/measurements/2026-08-06-pilot-v2-barriers/` before it is pointed
at anything new. Residuals are cpu-vs-gpu fp32 matmul ordering.

## Endpoint loss, all 10,240 windows (10,485,760 tokens per model)

| model | mean loss | sd across windows | sem |
|---|---|---|---|
| `seed00_twin` | 3.2124732800 | 0.375499 | 0.003711 |
| `seed00_random-chars` | 3.2111127037 | 0.375610 | 0.003712 |
| `seed01_twin` | 3.2103078584 | 0.375919 | 0.003715 |
| `seed02_twin` | 3.2084480626 | 0.375442 | 0.003710 |

Per-window paired differences, which is where the structure is:

| pair | mean | sd across windows | sem | pearson r |
|---|---|---|---|---|
| effect s0 (`random-chars` − `twin`, **same seed**) | −0.001361 | 0.019386 | 0.000192 | 0.998668 |
| floor s0–s1 (twin − twin) | 0.002165 | 0.024836 | 0.000245 | 0.997816 |
| floor s0–s2 (twin − twin) | 0.004025 | 0.025214 | 0.000249 | 0.997745 |
| floor s1–s2 (twin − twin) | 0.001860 | 0.024979 | 0.000247 | 0.997790 |

## What widening bought, and what it did not

**It did not buy the seed count back.** Both terms shrank by roughly an order of
magnitude and the ratio that sets `n` barely moved:

| | one passage | 10,240 windows |
|---|---|---|
| delta | −0.013863 | **−0.001361** |
| sigma (sd of twins x sqrt(2)) | 0.034941 | **0.002849** |
| sigma / \|delta\| | 2.520 | **2.094** |
| n at 80% power | ~50 | **~34** |
| n at 90% power | — | ~46 |

**What it did buy:**

1. Every number is ~10x more precise. The sem of a per-model mean is 0.0037
   against a single passage's whole-model spread of ~0.02, so the pilot's delta
   and its sign were **inside single-passage noise**. So was the "sign flip"
   between the void and corrected pilots — +0.057710, then −0.013863, now
   −0.001361. None of the three was resolved.
2. The memorisation question is closed, in the direction of "there was never a
   problem".
3. **The effect is smaller than the gap between two twins.** |−0.001361| against
   floor differences of 0.002165, 0.004025 and 0.001860.

## The barrier, on 256 held-out windows

The headline metric, on 262,144 tokens instead of 1,024. Widened-batch barrier
means `L(alpha)` is averaged over all 256 windows first; per-window means the
arithmetic is run separately on each window's own curve.

| pair | single passage | **widened** | per-window sd | rose above chord |
|---|---|---|---|---|
| twin s0–s1 | 4.562713 | **4.812025** | 0.410610 | **256/256** |
| twin s0–s2 | 4.580206 | **4.670411** | 0.445184 | **256/256** |
| twin s1–s2 | 4.777630 | **5.086170** | 0.435578 | **256/256** |
| **displacement s0** | 0.133863 | **0.148087** | 0.038159 | **256/256** |

**The retraction of S96 holds and hardens.** The displacement raises a real
barrier on **every one of 256 held-out windows**, not merely on the one
committed passage. Its per-window sem is 0.0024, so for this seed the value is
0.1481 ± 0.0024.

The barrier also turns out to be the *better-behaved* of the two axes. Its
per-window spread is ~26% of its own mean for the displacement and ~9% for the
floors, where a single passage moved the endpoint-loss delta by more than its
own size.

### The seed count computed on the metric §8.4 actually selected

Both pilot write-ups computed `n` on **endpoint loss**, which is not the
preregistered headline. Computed the same (substituted) way on the barrier:

| axis | evaluation | delta | sigma | sigma/delta | n at 80% |
|---|---|---|---|---|---|
| endpoint loss | 1 passage | −0.013863 | 0.034941 | 2.5205 | 49.9 |
| endpoint loss | 10,240 windows | −0.001361 | 0.002849 | 2.0933 | **34.4** |
| **barrier** | 1 passage | 0.133863 | 0.168791 | 1.2609 | 12.5 |
| **barrier** | 256 windows | 0.148087 | 0.298923 | 2.0186 | **32.0** |

Four estimates spanning 12 to 50, and the two widened ones agree at ~32–34. The
agreement is worth less than it looks: **every sigma in that table comes from
three floor values**, and the sd of three numbers carries roughly a 40%
coefficient of variation, which `n` squares.

The displacement is **3.05%** of the mean floor on the widened batch, against
2.88% on the single passage.

## A gap between the headline metric and the analysis script

`scripts/analysis.py` consumes flat `{seed, arm, metric, value}` records: one
**per-run scalar** per (arm, seed). `noise_floor` is literally
`ref[b] - ref[a]` over the **twin arm's own values** across seed pairs.

The barrier is **pairwise**, not per-run. Mapping it in the obvious way —
`metric(arm, s) = barrier(arm_s, twin_s)` — makes `metric(twin, s)` identically
0.

**The confirmatory tests survive that; the noise floor does not.** Checked
against `scripts/analysis.py` as of S101, which added both contrasts:
`arm_vs_arm_differences` reduces to `barrier(ff_s, twin_s) − barrier(ft_s,
twin_s)`, which is precisely what §5 registers — a contrast between two
displacements — and `pooled_differences` averages those within seed for §6.
Both are well-defined under the mapping.

`noise_floor` is not. It returns a list of zeros, while the twin-vs-twin
barriers in the table above — 4.670411, 4.812025, 5.086170, the quantity both
pilots report *as* the noise floor — have no slot in the panel at all.

So the decision that has to be made before launch is narrow: whether
`noise_floor` is suppressed for pairwise metrics, or redefined to carry
`barrier(twin_i, twin_j)`. It sits next to D-4 rather than inside it.

## The substitution that no amount of evaluation text can fix

Every `n` in this repo — 9.8, 50, and the 34 above — is computed from the
**wrong noise term**, and widening the batch cannot fix it.

Preregistration §4 and `scripts/analysis.py` both say the test is a **paired
difference within seed**: `d_s = M(arm, s) − M(twin, s)`, one-sample t across
seeds. Its noise term is therefore `sd` **across seeds** of `d_s`. The pilot has
exactly **one** injecting arm at **one** seed, so it has exactly one `d_s` and
cannot estimate that sd at all. Every write-up substituted the sd of *twin
against twin across seeds*, which is a different quantity: how far apart two
**unrelated** models sit.

That substitution is conservative, and measurably so. Same-seed pairs stay
0.82–0.96 similar in per-layer activation cosine; independently-seeded pairs sit
at ~0 (`2026-08-06-pilot-v2-results.md`). The per-window difference sd above
says the same thing more weakly: 0.019386 for the same-seed pair against
~0.0250 for the floor pairs.

**A second problem, of a different kind: before this file, `n` had only ever
been computed on endpoint loss, which §8.4 does not select.** The barrier row is
added above; it lands in the same place, ~32.

**What it would take to measure the right quantity.** At least three seeds
carrying an injecting arm, so there are at least three `d_s` to take an sd of.
The cheapest version that is *also study data* is `fluent-false` and
`fluent-true` at seeds 0–2: six runs, two 10.6 h waves, reusing the three twins
that already exist. It yields three paired differences of the **primary
contrast** on the **headline metric**, which is the first honest read on `n`
this study will have had.

Recording an amendment **before** those runs are read is not optional: the
commitment has to be to read the *spread* of the paired differences and not
their *mean*, or choosing `n` afterwards is choosing it having seen the effect.

## What this does not establish

- One arm, one displacement, three seeds. Nothing here speaks to §5's
  `fluent-false` vs `fluent-true`.
- The paired noise term remains **unmeasured**. It needs at least three seeds
  carrying an injecting arm; the cheapest honest version is `fluent-false` and
  `fluent-true` at seeds 0–2, which is six runs and two 10.6 h waves, and those
  runs are study runs rather than calibration overhead.
- The activation metrics (CKA, cosine) were **not** re-measured here; they are
  still on the single passage.
- 256 windows for the barrier, not 10,240: 21 alphas x 4 pairs is 21,504
  interpolated-model evaluations at 256, and the per-window spread is already
  resolved there.
- The barrier remains a lower bound on a 21-point grid.

## Reproducing

```
sbatch /shared/27as66/burst-pilot/scratch/heldout.sbatch
```

Both scripts live beside it (`heldout_sweep.py`, `heldout_barrier.py`) rather
than in `scripts/`, because they carry no tests. **Promoting them into
`scripts/` with tests is the follow-up** — they are the reproduction path for
every number above, and the repo's bar for a measurement script is a tested one.

Artifacts under `docs/measurements/2026-08-07-heldout/`. The barrier file keeps
its full per-window values (256 per pair). The loss file has its
`per_window_losses` block **stripped** — 4 x 10,240 floats, 800 kB of
regenerable array against 4.9 kB of everything that is actually read. Every
summary statistic quoted above survives in the committed file; the raw arrays
are at `/shared/27as66/burst-pilot/scratch/heldout-v1/`.
