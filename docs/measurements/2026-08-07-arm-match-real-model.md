# Arm matching on the real step-199 model: the proxy was wrong in both directions

**Date:** 2026-08-07. **Commit:** `384d216`, clean tree. **Model:**
`runs-fixed/seed00_twin/step000199_weights_only.pt`, `trained_at 3e715a6`.
**Node:** gpmoo-a1, one A100.

## What this discharges

`docs/measurements/8b-i-in-context-match.md` limitation 2, verbatim:

> These numbers come from fully-trained public GPT-2. … Arms matched here are
> **NOT** guaranteed to be matched on that model, and that model is the one
> whose weights actually move. This is a **proxy until it can be re-verified
> against a real step-200 checkpoint.**

`8b-iv`'s LIMITATION says the same harder — "THAT MODEL DOES NOT EXIST",
"PROXY OF UNKNOWN FIDELITY". The model now exists. This is that
re-verification, plus the **full-batch delta**, which `docs/spec-v4.md` lists as
one of three candidate matching targets and which nobody had ever computed.

**No verdict.** Spec-v4 "Deliberately still open" item 1 leaves the tolerance to
whoever runs the study, item 3 does not choose among the three targets, and item
4 leaves "anisotropy" undefined. Nothing here applies a tolerance, recommends a
target, or uses that word.

## The gate: this reconstructs the run exactly

The full-batch legs rebuild the real step-200 batch from the run's own
permutation and splice each arm through `injection.apply`. Against what the
pilot recorded at step 200:

| | reconstructed | `train_record.json` |
|---|---|---|
| twin, mean loss | 5.8612629771 | **5.8612629771** |
| `random-chars`, mean loss | 5.8629262596 | **5.8629262596** |
| twin, grad norm | 0.47940193 | 0.479392 |
| `random-chars`, grad norm | 0.47972219 | 0.479702 |

Loss agrees to **ten decimal places**. The gradient norms agree to ~2e-5 because
`clip_grad_norm_` reduces under `foreach` while this concatenates first — a
summation-order difference, the same class S77 is about. The single-sequence
legs agree with the run too: `random-chars`' burst-region mean is **7.3527**
here and 7.3527 in the run's `injection_fired.burst_region`.

## 1. Single sequence — 8b-i's criteria, re-measured

Burst region is 194 predictions from position 399; `__filler__` is the same
window holding `context.txt` instead of a burst, which is 8b-i's floor.

| arm | region loss (public) | **region loss (real)** | gradnorm (public) | **gradnorm (real)** |
|---|---|---|---|---|
| `fluent-false` | 4.4075 | **6.5199** | 20.5834 | **4.3104** |
| `fluent-true` | 4.2856 | **6.3179** | 18.0029 | **4.2343** |
| `scrambled-false` | 6.9748 | **7.7605** | 23.2829 | **4.8368** |
| `scrambled-true` | 7.0007 | **7.9176** | 22.2936 | **5.2153** |
| `pos-substituted` | 8.7172 | **8.9115** | 21.4783 | **4.6342** |
| `random-chars` | 5.8432 | **7.3527** | 23.1194 | **9.6478** |
| no-burst floor | 3.3091 | **5.9450** | 19.6466 | **4.1708** |

**The proxy did not merely shift — it reordered.**

| spread (max/min) | public GPT-2 | **real step-199** |
|---|---|---|
| burst-region loss | 2.0341 | **1.4105** — better |
| gradnorm | 1.2933 | **2.2785** — worse |
| gradnorm excluding `random-chars` | — | 1.2317 |

Loss matching improved; gradient-norm matching got worse by nearly 2x, and
**entirely because of `random-chars`**: 9.6478 against 4.23–5.22 for every other
arm. Random characters are far more anomalous to a model 200 steps from
initialization than to a trained one. `random-chars` is exploratory under
preregistration §7, so this does not touch either confirmatory contrast — but it
is the single largest matching failure in the study and it was invisible on the
proxy.

## 2. The full-batch delta — never computed before

The injected sequence is **one row of 256**. This is the gradient of the real
step-200 batch, assembled the way `train.py` assembles it (32 micro-batches of
8, each loss divided by `accum` before `backward`, bf16 autocast), and
`delta = grad(arm) − grad(twin)` is what the burst contributes to the update
that actually lands.

| arm | full-batch grad norm | \|delta\| | \|delta\|/\|g\| |
|---|---|---|---|
| twin | 0.47940193 | — | — |
| `fluent-false` | 0.48008717 | 0.01069778 | 2.228% |
| `fluent-true` | 0.48020401 | 0.01068314 | 2.225% |
| `scrambled-false` | 0.48011880 | 0.01085312 | 2.261% |
| `scrambled-true` | 0.48027754 | 0.01094178 | 2.278% |
| `pos-substituted` | 0.47983952 | 0.01079259 | 2.249% |
| `random-chars` | 0.47972219 | 0.01178813 | 2.457% |

**Dilution does most of the matching for us.** Spread of |delta| across the six
arms is **1.1034**, against 2.2785 for the same arms' sequence-level gradient
norms. `random-chars`, which is 2.28x off at sequence level, is 10% off here.

**The burst moves the update by about 2.2%**, for every arm.

## 3. The two registered contrasts

**PRIMARY, `fluent-false` vs `fluent-true`** — and this reverses a concern
raised from the proxy:

| criterion | public GPT-2 | **real step-199** |
|---|---|---|
| burst-region loss | 2.84% apart | 3.20% apart |
| gradnorm | **14.33% apart** | **1.80% apart** |
| full-batch \|delta\| | never computed | **0.14% apart** |
| full-batch delta cosine | — | **0.9455** |

The 14% gradient-norm gap that made the primary contrast look confounded is a
property of **public GPT-2, not of this study's model**. On the real model the
two arms perturb the update by magnitudes 0.14% apart, in directions 0.9455
aligned. On any tolerance anyone would plausibly set, the primary contrast is
matched.

**SECONDARY, `fluent` (pooled) vs `pos-substituted`:**

| criterion | fluent (pooled) | `pos-substituted` | gap |
|---|---|---|---|
| burst-region loss | 6.4189 | 8.9115 | **+38.8%** |
| gradnorm | 4.2724 | 4.6342 | +8.5% |
| full-batch \|delta\| | 0.01069046 | 0.01079259 | **+1.0%** |
| delta cosine vs ff / ft | — | 0.9040 / 0.9052 | — |

Whether this contrast is matched **depends entirely on which of spec-v4 item 3's
three targets is chosen** — 38.8% apart on loss, 1.0% apart on the full-batch
delta. That is the sharpest argument yet that item 3 has to be ruled before
launch rather than after, and this file does not rule it.

## 4. Direction — 8b-iv re-measured

Pairwise cosine of burst-region gradients (8b-iv's definition):

|  | ff | ft | sf | st | ps | rc |
|---|---|---|---|---|---|---|
| **ff** | 1.0000 | 0.3952 | 0.5665 | 0.3173 | −0.0031 | −0.0581 |
| **ft** | 0.3952 | 1.0000 | 0.2959 | 0.5359 | −0.0089 | −0.0439 |
| **sf** | 0.5665 | 0.2959 | 1.0000 | 0.4533 | 0.0043 | −0.0868 |
| **st** | 0.3173 | 0.5359 | 0.4533 | 1.0000 | 0.0231 | −0.0696 |
| **ps** | −0.0031 | −0.0089 | 0.0043 | 0.0231 | 1.0000 | −0.0052 |
| **rc** | −0.0581 | −0.0439 | −0.0868 | −0.0696 | −0.0052 | 1.0000 |

The derived pairs — a scramble against the fluent text it was made from — are
**much less aligned on the real model**: `sf`/`ff` is 0.5665 against the proxy's
0.8243, and `st`/`ft` is 0.5359 against 0.8207. `ff`/`ft` is 0.3952 against
0.3555, essentially unchanged. Arm-vs-filler controls are all within ±0.031 of
zero, so these gradients are the burst's rather than the filler's.

Pairwise cosine of **full-batch deltas**:

|  | ff | ft | sf | st | ps | rc |
|---|---|---|---|---|---|---|
| **ff** | 1.0000 | 0.9455 | 0.9555 | 0.9265 | 0.9040 | 0.7901 |
| **ft** | 0.9455 | 1.0000 | 0.9287 | 0.9486 | 0.9052 | 0.7962 |
| **sf** | 0.9555 | 0.9287 | 1.0000 | 0.9351 | 0.8966 | 0.7752 |
| **st** | 0.9265 | 0.9486 | 0.9351 | 1.0000 | 0.8932 | 0.7730 |
| **ps** | 0.9040 | 0.9052 | 0.8966 | 0.8932 | 1.0000 | 0.7863 |
| **rc** | 0.7901 | 0.7962 | 0.7752 | 0.7730 | 0.7863 | 1.0000 |

At batch level every arm pushes in nearly the same direction, 0.77–0.96.
`random-chars` is the least aligned with everything. This is the same dilution
story: what survives into the update is dominated by *a different sequence sits
in row 3 of micro-batch 9*, and only weakly by which text it is.

**What that implies for the study, stated as interpretation rather than
result:** the two primary arms perturb the step-200 update by magnitudes 0.14%
apart and directions 0.9455 aligned. Any difference in final displacement
between them is therefore a very small initial difference amplified across 9,336
subsequent steps — which is consistent with
`2026-08-06-pilot-v2-results.md`'s finding that one burst moves the model 67% as
far as all remaining training, and with reading that as chaotic amplification.

## 5. Statistics reported without the umbrella word

Spec-v4 item 4 leaves "anisotropy" undefined among three defensible candidates.
Participation ratio is in the JSON per arm; it is **basis-dependent** — it
counts coordinates in whatever parameter basis this model happens to use and is
not a rotation-invariant property of the update. The per-tensor norm profile and
the gradient-outer-product eigenspectrum are not computed. The word is not used
for any of them, per `tests/test_sequence_assembly.py`'s guard on the proxy
artifact.

## What this does not establish

- **One seed.** All of it is seed 0's step-200 batch. Matching at another seed's
  batch is unmeasured; the burst text is fixed but the 255 other sequences are
  not.
- **One step.** Step 200 only, as the spec fixes.
- No tolerance, no verdict, no ruling on spec-v4 items 1, 3 or 4.
- The single-sequence legs are fp32 for parity with 8b-i/8b-iv; the full-batch
  legs are bf16 autocast for parity with the run. They are not interchangeable.
- `scrambled-corpus` appears in 8b-i and is **not** an arm; it is absent here.

## Reproducing

```
CUBLAS_WORKSPACE_CONFIG=:4096:8 python \
  /shared/27as66/burst-pilot/scratch/arm_match_real.py \
  --checkpoint RUNS/seed00_twin/step000199_weights_only.pt \
  --cfg-scratch /some/scratch --out arm-match.json
```

~2 minutes on one A100. Untested scratch, like the other pre-flights; promoting
it into `scripts/` with tests is the same follow-up the held-out scripts carry.
