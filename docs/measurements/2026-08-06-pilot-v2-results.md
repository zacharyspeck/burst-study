# The pilot, re-run on the corrected objective: sigma, displacement, and a retraction

**Date:** 2026-08-06. **Commit trained from:** `3e715a6`, clean tree, all four
runs. **Job:** SLURM 54728 on gpmoo-a1. **Supersedes:**
`docs/measurements/2026-08-05-pilot-results.md`, which is void under S97.

## What ran

Same seeds and same arms as the void pilot, on `scripts/train.py` with S99's
fix in place. bf16, `micro_batch: 8`, `hf_gpt2`, 9,536 steps.

| run | wall | final training loss | final state digest |
|---|---|---|---|
| `seed00_twin` | 10.17 h | 3.2074287608 | `812bd215…7f6f88` |
| `seed01_twin` | 10.20 h | 3.2039063424 | `817c9bbe…6ea552` |
| `seed02_twin` | 10.54 h | 3.2140787169 | `53df00e1…4b61a40` |
| `seed00_random-chars` | 10.56 h | 3.2058403343 | `b7caf8dd…1d80a9c` |

All four `steps_run: [0, 9536]`, 191 checkpoints, `dirty: false`. Final loss
~3.21 against the void runs' 4.67: next-token prediction is a genuinely easier
objective than two-ahead, and a substantially lower curve is what a corrected
run should look like.

**Four cards, not eight.** The void pilot claimed the whole node only to pin
four cards of one SKU. Pre-flight job 54727 (30 steps, seed 0, one leg per SKU)
returned digest `7e1fb0b9…d1358d` on A100-SXM4-80GB, A100 80GB PCIe and A100X
alike, with all 30 step lines identical — so §0.C's condition ("untested") is
discharged for this node and any four of its cards are interchangeable. This run
took SLURM's allocation (GPUs 0,1,2,3, straddling all three SKUs) and left four
cards free.

## The objective gate

Run at step ~250, ~20 minutes in, against the step-199 checkpoint:

| | |
|---|---|
| training loss, steps 200–210 (held-out by construction) | 5.853997 |
| independent next-token eval of the step-199 checkpoint | 5.905995 |
| **gap** | **0.051998** |
| S97's gap, for scale | 2.340000 |

45× smaller than the defect's signature, and the residual is the right size and
sign for training progress between the checkpoint and the steps it is compared
against. The corpus is exactly one epoch, so the loop's printed loss **is** a
held-out estimate; that is what makes this check possible at all. It would have
caught S97 in twenty minutes rather than 44.69 GPU-hours.

## Injection

Byte-identical through step 199, divergent at 200, as before:

| step | twin | random-chars |
|---|---|---|
| 199 | `5.8591260761` | identical |
| 200 | `5.8612629771` | `5.8629262596` |
| model tensors @199 | `af54579f…d73bb` | **IDENTICAL** |
| model tensors @249 | — | **DIFFERENT** |

`injection_fired` populated on the arm, `null` on all three twins. Burst-region
mean loss at injection **7.3527** over 194 predictions.

## The §8.4 branch is unchanged

Re-derived on a real model, because the recorded branch had been computed on a
void checkpoint:

| criterion | void model | **corrected model** | threshold |
|---|---|---|---|
| spread | 1.0000022 | **1.0000022** | ≤ 2 |
| median | 0.9999373 | **0.9999373** | < 1.01 |

Branch: **`plain_loss_barrier`**, `requires D-6 built: False`. Agreement to
3e-7 is expected rather than surprising — canonicalization distortion is a
property of the recipe and of the model's conditioning at step 199, not of the
objective, and at step 199 neither model has trained far.

## RETRACTION: the barrier does not floor

**The void pilot's headline finding was an artifact of S97 and is withdrawn.**

| pair | void run | **corrected run** |
|---|---|---|
| twin s0–s1 | 3.073542 | 4.562713 |
| twin s0–s2 | 4.217713 | 4.580206 |
| twin s1–s2 | 2.591210 | 4.777630 |
| **displacement s0** | **0.000000**, `rose: False` | **0.133863**, `rose: True` |

On correctly-trained models the displacement raises a real barrier, peaking at
alpha 0.5 like every other pair. The claim that "the metric cannot express this
effect at all" was false, and it was false because the models were wrong, not
because the metric was.

What survives is weaker and narrower: the effect is **2.9%** of the mean floor
(0.1339 against 4.6402, sd 0.1194). The floor is still dominated by something
the burst cannot produce — see the cosine result below — so the two are still
not samples of one quantity. But they are now both measurable, and that is a
different statement from the one this record's predecessor made.

## The displacement ladder

`scripts/displacement_ladder.py` over six pairs. Per-model loss on
`bursts/context.txt`:

| model | loss |
|---|---|
| `seed00_twin` | 3.4967278896 |
| `seed00_random-chars` | 3.4828648239 |
| `seed01_twin` | 3.4527468909 |
| `seed02_twin` | 3.4552285303 |
| `seed00_twin` @199 | 6.0358134323 |

- **delta** = arm − twin = **−0.013863**. **The sign is opposite the void run's
  +0.057710**: the injected model is *better* on this passage than its twin, not
  worse.
- **sigma** = sd(twins) × sqrt(2) = 0.024707 × sqrt(2) = **0.034941**.
- sigma/|delta| = **2.520**, so §9.3's `n ≈ 7.85 × (sigma/delta)²` gives
  **n ≈ 50 seeds**.

**This is the number that changed most, and it is the one that matters for
planning.** The void pilot said ~10 seeds, which happened to match spec v4's
plan. The corrected pilot says ~50. Both rest on one displacement and three
seeds, so both are order-of-magnitude at best — but the earlier agreement with
the plan was coincidence produced by a bug, and it should not be carried forward
as reassurance.

Raw L2, total:

| pair | L2 |
|---|---|
| effect (arm vs twin) | 236.060138 |
| floor s0–s1 | 532.386790 |
| floor s0–s2 | 532.617391 |
| floor s1–s2 | 535.695425 |
| training scale (step 199 → 9535) | 353.584047 |

The effect is 44% of the floor — but also **67% of the distance the model moved
across the entire rest of training**. One burst at step 200 displaces the
trajectory by two-thirds of what 9,336 subsequent steps accomplish, which is
chaotic amplification rather than a persistent memory of the text, and nothing
here separates the two.

## Cosine is the sharpest result in the pilot

Per-layer activation cosine, median over token positions:

| layer | effect | floor s0–s1 | floor s0–s2 | floor s1–s2 |
|---|---|---|---|---|
| 0 | 0.9611 | 0.0115 | −0.0012 | −0.0017 |
| 4 | 0.8718 | 0.0023 | −0.0034 | −0.0016 |
| 8 | 0.8235 | 0.0018 | −0.0002 | −0.0032 |
| 12 | 0.8821 | 0.0017 | −0.0007 | −0.0053 |

**The floor pairs are essentially orthogonal at every layer.** Independently
seeded models do not merely differ — they occupy unrelated bases. The effect
pair, sharing an initialization and a data order, sits at 0.82–0.96.

Per-layer CKA, which is rotation-invariant, tells the complementary story:
effect 0.939–0.997 against floor 0.861–0.993 — much closer, because CKA is
blind to exactly the basis difference cosine exposes.

This is the structural point the void record was reaching for, now measured
directly rather than inferred from a floored barrier: **the twin-vs-twin floor
is largely a basis mismatch that the burst structurally cannot produce.** A
displacement that is 2.9% of that floor has not been shown to be small; it has
been compared against a quantity that is mostly gauge. Removing the gauge is
what `scripts/canonicalize.py` exists for, and the aligned metrics are unbuilt
(D-6).

## A defect in the ladder's own banner, not fixed here

`scripts/displacement_ladder.py`'s `LIMITATION` constant opens by asserting that
the checkpoints it read "WERE TRAINED ON THE WRONG OBJECTIVE" and that "no
number here transfers to a corrected re-run". Run against `runs-fixed/`, **that
is false about its own inputs** — the checkpoints record `trained_at commit
3e715a6`, which contains S99's fix.

The banner is a module-level constant, and
`tests/test_displacement_ladder.py:1391-1401` pins its text deliberately: the
test's docstring argues a banner omitting "the single most important fact about
its own inputs" would be the S55 shape at the level of an artifact. That
reasoning was right when the only checkpoints in existence were void. It now
produces the same failure it was written to prevent, in the opposite direction.

**Nothing here is fixed, and no `13-*` artifact is committed** — the generated
report would misdescribe its own inputs, and publishing it is worse than citing
the numbers under this heading. The fix is the module author's call: the clause
should be derived from the checkpoints' `trained_at` commit rather than
asserted, failing safe to the warning when ancestry cannot be determined. The
ladder's own numbers above are correctly computed; only its banner is stale.
The generated files sit at
`/shared/27as66/burst-pilot/scratch/analysis-v2/ladder/`.

## What this does not establish

- One arm, one displacement, three seeds. Nothing here speaks to the §5 primary
  contrast, `fluent-false` vs `fluent-true`.
- ~~**The evaluation text is training data.** `bursts/context.txt` is openwebtext
  document 73, which every one of these models trained on. No memorisation
  confound is controlled, and every loss, CKA and cosine number above inherits
  that.~~
  **THIS BULLET IS WRONG. Corrected 2026-08-07, kept in place per CLAUDE.md §2.**
  `bursts/context.txt` **is** openwebtext document 73, and that is exactly why
  the held-out slice was taken from the **front** of the corpus: front placement
  puts the committed corpus-derived texts (documents 73, 104 and 193) *outside*
  the training slice, with 50x-114x margin. See `scripts/corpus_spec.py`'s
  layout docstring, cross-module obligation 1 in `implementation-notes.md`, and
  `docs/measurements/10-metrics.md`, which states it directly. No model in this
  study trained on this passage and there is no memorisation confound. The
  limitation that is real is the next bullet — it is one passage.
- The activation basis is a single 1,024-token passage; for CKA those positions
  are the entire sample. **Measured 2026-08-07: on the endpoint-loss axis this
  single-passage choice was worth roughly an order of magnitude in the noise
  term that sets the study's seed count.** See
  `docs/measurements/2026-08-07-heldout-remeasurement.md`.
- `scripts/analysis.py` was not run: D-4 (family of tests, correction method) is
  open and it refuses without both.
- The barrier is a lower bound on a 21-point grid.
- L2 is blind to neuron permutation, and the asymmetry runs against the floor
  pairs.

## Reproducing

```
sbatch /shared/27as66/burst-pilot/scratch/analyse.sbatch
```

Phase 1 is the §8.4 ruler; phase 2 is every arm-vs-twin quantity and is
unreachable unless phase 1 wrote its branch artifact. §8.5's ordering is
enforced by the script's structure rather than by the operator remembering it —
which is the failure recorded in S96.
