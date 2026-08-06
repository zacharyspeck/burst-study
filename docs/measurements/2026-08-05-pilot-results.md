# The 4-run pilot: first sigma, first displacement, and a metric that floors

> **RE-RUN LANDED 2026-08-06.** The corrected pilot is
> `docs/measurements/2026-08-06-pilot-v2-results.md`. It **retracts this
> record's headline**: on correctly-trained models the barrier does *not* floor
> on the displacement (0.133863, `rose: True`, against the 0.000000 below), and
> the power estimate moves from n ≈ 9.8 to n ≈ 50 with the sign of delta
> reversed. The claim that the metric "cannot express this effect at all" was an
> artifact of S97, not a property of the metric.

> **SUPERSEDED, 2026-08-05 — every number below that depends on the trained
> weights is void.** The four runs were trained on the wrong objective:
> `train.py` pre-shifts and then hands the shifted pair to a `labels=` call that
> shifts again, so the models were optimised to predict the token *two*
> positions ahead. See `docs/2026-08-05-training-objective-defect.md` for the
> investigation, the proof, and the itemised list of what is invalidated and
> what survives. The barrier curves, the endpoint-loss delta/sigma, and the
> n ≈ 9.8 power estimate are all in the invalidated set. The injection
> verification (§"The injection landed…"), the §8.4 ruler branch, and the
> wall-clock figures survive. Kept in place rather than deleted, per CLAUDE.md
> §2: the reasoning here is still the record of what was believed on the day.

**Date:** 2026-08-05. **Commit trained from:** `9aa930d`, clean tree, all four
runs. **Job:** SLURM 54652 on gpmoo-a1, four A100X cards.

## What ran

Four runs, bf16, `micro_batch: 8`, `adamw_impl: foreach`, family `hf_gpt2`,
corpus `/shared/27as66/corpus`, 9,536 steps each.

| run | card | wall | final state digest |
|---|---|---|---|
| `seed00_twin` | gpu 2 | 11.13 h | `ee8b749e…7c4752` |
| `seed01_twin` | gpu 3 | 11.23 h | `904e9d74…9472b` |
| `seed02_twin` | gpu 6 | 11.10 h | `81183d99…62ce61` |
| `seed00_random-chars` | gpu 7 | 11.14 h | `c5c2d119…cad48d2` |

Three twins give three pairwise noise-floor observations; the seed-0 pair gives
one displacement. Every `run_provenance.yaml` records `commit: 9aa930d` and
`dirty: false`.

## The injection landed where the spec puts it

`seed00_twin` and `seed00_random-chars` are byte-identical through step 199 and
diverge at step 200:

| step | twin | random-chars |
|---|---|---|
| 199 | `loss 6.8388155848 gnorm 0.3920728862` | identical |
| 200 | `loss 6.8400297612` | `loss 6.8404841572` |

The step-199 weights-only checkpoints hash identically over the model tensors
(`886db82a…5b6ae9`), and the step-249 pair does not. `injection_fired` matches
`injection_plan` field for field on the injecting arm and is `null` on all three
twins; burst-region mean loss at the moment of injection is **7.3066** over 194
predictions starting at prediction index 399.

**A trap for whoever checks this next.** `sha256sum` on the checkpoint *files*
reports the step-199 pair as different. That is not a determinism failure:
`save_checkpoint` stores `arm` in the payload beside the weights
(`scripts/train.py:323`), so the twin and the injecting arm differ by that
string alone. The comparison has to hash `payload["model"]`.

## The pre-registered branch: plain, not aligned

`scripts/canonicalization_error.py --checkpoint` was run on the step-199 twin
checkpoint **before any arm-vs-twin distance was examined**, as
`docs/preregistration.md` §8.5 requires. Full report:
`docs/measurements/12-injection-point-ruler.md`.

| §8.4 criterion | measured | threshold | |
|---|---|---|---|
| spread (max/min) | 1.0000022 | ≤ 2 | pass |
| median distortion | 0.9999373 | < 1.01 | pass |

Branch: **`plain_loss_barrier`**, aligned barrier demoted to a robustness check,
`requires D-6 built: False`. So the unbuilt aligned barrier (D-6) does not gate
the headline number.

## The headline metric floors on the displacement

Raw interpolation barrier on the committed 1024-token batch `bursts/context.txt`,
21-point alpha grid, fp64 on CPU, final checkpoints (step 9535):

| pair | max excess | min excess | rose above chord | argmax |
|---|---|---|---|---|
| twin s0 vs s1 | **3.073542** | −0.357681 | yes | 0.50 |
| twin s0 vs s2 | **4.217713** | −0.325170 | yes | 0.50 |
| twin s1 vs s2 | **2.591210** | −0.370142 | yes | 0.50 |
| **s0 twin vs s0 random-chars** | **0.000000** | −0.100241 | **no** | 0.00 |

The displacement curve never rises above the chord. It sags, smoothly, to
−0.100 at alpha 0.5 and back. The arm and its twin are linearly mode connected:
same initialization, same data order, one burst at step 200, and they never
leave the same basin.

The three twin-vs-twin curves do the opposite — each climbs to a sharp peak at
exactly alpha 0.5, `seed00`/`seed01` reaching loss 10.447 against a chord of
7.373. That is independently-initialized networks sitting in different basins,
which is the textbook result and has nothing to do with any burst.

**So the two quantities the design compares are not the same phenomenon.** On
this metric the noise floor measures basin mismatch between seeds, and the
displacement is floored at zero by construction. sigma = 3.29 against delta = 0
is not a small effect; it is a metric that cannot express this effect at all.

This is what the permutation-aligned barrier existed to fix. §8.4's rule branched
away from it on a distortion measurement taken at epsilon 1e-6 against isotropic
perturbations of a **single** checkpoint — which bears on whether canonicalization
distorts a distance, not on whether two independently-seeded twins differ by a
genuine permutation. These curves show they differ by *something* worth 2.6 to
4.2 nats at the midpoint. The branch is pre-registered and D-7 is closed, so it
stands; this records that its evidence does not cover the case that turned out to
matter. Measurement E in `canonicalization_error.py` (permuted-model recovery) is
the built thing that speaks to it.

## Endpoint loss does carry signal

The same evaluation's endpoint losses, which the barrier discards:

| run | loss on `bursts/context.txt` |
|---|---|
| twin seed 0 | 7.339636 |
| twin seed 1 | 7.407107 |
| twin seed 2 | 7.426399 |
| random-chars seed 0 | 7.397346 |

- **delta** (within seed 0, arm − twin) = **+0.057710**. The burst makes the
  model slightly worse on ordinary text, which is the plausible direction for
  194 tokens of random characters.
- **sd** of the three twin losses = 0.045556; **sigma** for a paired difference
  across seeds = sd × sqrt(2) = **0.064426**.
- sigma/delta = **1.116**, so §9.3's `n ≈ 7.85 × (sigma/delta)²` gives
  **n ≈ 9.8 seeds**.

Ten seeds is what spec v4 plans. That is a striking landing, and it should be
read as an order-of-magnitude check and nothing more: **delta is a single
observation and sigma comes from three seeds.** A standard deviation from n=3
has roughly a 40% coefficient of variation, and n scales as its square, so the
honest reading of "9.8 seeds" is "somewhere between about 4 and 25". It says the
study is not obviously mis-powered. It does not say 10 is enough.

## What this does not establish

- One arm, one displacement. Nothing here speaks to `fluent-false` vs
  `fluent-true`, the §5 primary contrast.
- The evaluation batch is one committed 1024-token passage. Endpoint loss on a
  single sequence is a noisy statistic in its own right and its variance is not
  separated from seed variance here.
- `scripts/analysis.py` was not run: it requires a correction method with no
  default (D-4 open) and is built for the 7-arm design, not for one arm at one
  seed.
- The barrier is a lower bound on a 21-point grid, as the metric itself records.

## Reproducing

The four curves are committed under
`docs/measurements/2026-08-05-pilot-barriers/`, each carrying its full alpha
grid, losses, chord and excess, plus the seed/arm/step of both endpoints. They
were re-derived with the in-repo script after it was written, and reproduce to
all six decimals.

```
python scripts/pilot_barrier.py \
    RUNS/seed00_twin/step009535_full.pt \
    RUNS/seed00_random-chars/step009535_full.pt \
    --label displace_s0 --out displace_s0.json \
    --cfg-scratch /some/scratch/dir
```

`--cfg-scratch` must not be a run directory: `load_config` would otherwise write
a `resolved_config.yaml` beside provenance describing the training that produced
these checkpoints. Provenance writing is off regardless.

**Run `scripts/canonicalization_error.py --checkpoint` first.** §8.5 constraint 1
is on the validity of section 8, and this script cannot enforce it.
