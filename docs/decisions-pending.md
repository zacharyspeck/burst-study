# Decisions pending

Things that came up during autonomous work, that change **what the study
measures** rather than how the code is written, and that are therefore not mine
to take. Each entry records what the decision is, what was measured that raised
it, the concrete options, and what I would need in order to act.

This file said "nothing here has been acted on", which stopped being true on
2026-08-02: **D-1 and D-2 were both ruled and both steps were REMOVED from
`DEFAULT_RECIPE`**, which is now
`(ZeroKeyBiasGauge, ZeroValueBiasGauge, SortHeads, AlignFFNNeurons)` — four
steps, with `CanonicalizeHeadInternal` and `AbsorbLayerNormGains` gone.

Each entry below carries its own **Status** line and that is the authority.
~~Open as of 2026-08-03: D-3 (head sort), D-4 (multiple-comparison correction),
D-5 (full-checkpoint interval), D-6 (step 10's second half, skipped),
D-8 (three confirmed wordings).~~

~~**Open as of 2026-08-07: D-3 (head sort), D-5 (full-checkpoint interval),
D-6 (step 10's second half, skipped), D-8 (three confirmed wordings) — FOUR.**~~
**D-4 was RULED on 2026-08-07** (family = 2, correction `holm`); its entry
below is kept in full with the original options, because which alternatives
were live is the part that makes a ruling auditable.

**Open as of 2026-08-08: D-3 (head sort), D-5 (full-checkpoint interval),
D-6 (step 10's second half, skipped), D-8 (three confirmed wordings), and
D-9 (what replaces the secondary contrast) — FIVE.**

**D-4's ruling no longer describes the study.** It was ruled at family = 2 on
2026-08-07 and invalidated the next day by the arm cut recorded in D-9, which
removed `pos-substituted` and with it the second of the two contrasts that
*were* the family. The entry stays below in full: it was correctly ruled on
the study as it stood, and what overtook it was a later decision, not an error
in that one.

**D-7 is CLOSED**, by a decision rule in `docs/preregistration.md` §8 rather
than by a pick. Kept below with the reasoning, including the fact that the
premise correction inside it was itself wrong.

> **Why this header used to warn you not to trust it.** Until 2026-08-04 it
> carried a `PARTLY SUPERSEDED` banner telling the reader to trust the
> per-entry `Status` lines over this paragraph — while the per-entry lines for
> D-1 and D-2 were the stale half, both still reading "open. Nothing changed."
> The instruction routed to the wrong block. This is the S70 shape: one block
> stayed static while its surroundings became derived. The per-entry lines are
> now correct, so the warning is gone rather than inverted.

---

## D-1. The head-internal step has the property that disqualified the FFN sort

**Status: RULED 2026-08-02 — option 1. `CanonicalizeHeadInternal` was REMOVED
from `DEFAULT_RECIPE`.** The step still exists and is still tested; it runs
against `RETIRED_HEAD_INTERNAL_RECIPE` in `scripts/canonicalize.py`, retained
so this entry's measurement stays reproducible. It is not applied to anything
the study measures. The consequence — the GL(head_dim) gauge is no longer
quotiented, so the ruler is validated for same-seed twins only and NOT for
independently-initialised models — is stated in `docs/step9-summary.md` under
"what the ruler does NOT do".

*(This line read "open. Nothing changed." until 2026-08-04, eight weeks after
the ruling it was describing. Everything below it is the case FOR the ruling,
preserved as written; it is not a description of the current recipe.)*

### The decision

Does `CanonicalizeHeadInternal` stay in `DEFAULT_RECIPE`?

It is the largest symmetry group in the architecture and the most valuable to
quotient out. It is also, on the measurement below, the sole source of the
ruler's remaining distortion — and its failure is **erratic and
seed-dependent**, which is the exact property on which the FFN sort was retired
one commit earlier.

### What was measured

Widening the epsilon sweep from 5 seeds to 10 — queue item 5 — surfaced this.
At 5 seeds it was invisible; at 3 seeds an earlier measurement reported a flat
`3.05` and I wrote that up as a benign constant. It is not constant.

Real GPT-2, isotropic perturbation, ratio = `||canon(M) - canon(M+eps)|| /
||M - (M+eps)||`. A ratio of 1 means canonicalization is distance-neutral.

| recipe | eps=1e-8 | eps=1e-6 | eps=1e-5 |
| --- | --- | --- | --- |
| shipped (with head-internal) | 3.07 | **2189.9** | 237.8 |
| **without head-internal** | **0.907** | **0.907** | **0.907** |

Per-seed at `eps=1e-6`, shipped: seed 11 → `2450.5`, seed 22 → `3.09`,
seed 44 → `1929.3`. The spread across the ten seeds runs `[2.83, 2450.5]`.

Without the head-internal step the ruler is stable to three decimal places
across seven decades of epsilon and across every seed, and sits slightly
**below** 1 (`0.907`), i.e. very close to distance-neutral.

### Why it happens

Real GPT-2's minimum relative singular-value gap is `5.479e-06`. When the
perturbation is comparable to that gap, the SVD basis inside the
near-degenerate subspace **rotates**. The canonical form is then built on a
different basis in the two models, and the distance between them inflates.

This is a *continuous* rotation, not a discrete sign flip, which is why the
existing sign-flip counter reported `0/0/0` on exactly the cells that inflated
2000x. A `min_basis_alignment_cos` diagnostic was added to catch it: it reads
`1.0000` where the ratio is stable and drops where it spikes.

Ruled out as causes, by direct measurement:

- **not** the FFN alignment — it assigns the identity permutation in 0 of 12
  layers on every seed and epsilon tested
- **not** head-order flips — 0 until `eps=1e-3`
- **not** sign flips — 0 at `eps=1e-6` where the spike is largest

### Why this is your decision and not mine

Removing `CanonicalizeHeadInternal` would abandon the GL(head_dim) freedom —
by far the largest symmetry group here, and the one the affine-invariant
amendment was written for. Keeping it accepts a ruler whose distortion is
seed-dependent by three orders of magnitude. Either way it changes what
canonical form means, which is out of scope for a measurement task.

### Options

1. **Remove the head-internal step.** Ruler becomes stable at `0.907` across
   all epsilons and seeds. Cost: two large symmetry groups per head go
   unquotiented, so two independently-initialised models would show gauge
   difference there. For same-seed twins the zero-gradient argument says the
   coordinates never move — but that argument is exactly what you declined to
   bet on for the FFN permutation, and the same reasoning applies here.
2. **Keep it and accept the distortion**, documented as a stated range the way
   the 3.05-vs-84.4 open question already is. The distortion is not a constant
   that can be divided out.
3. **Gate it on conditioning** — apply the step only to heads whose singular
   gap exceeds some threshold, leaving near-degenerate heads unquotiented.
   Makes canonical form conditional on the model, which is a new kind of
   complexity and would need its own round-trip and mutation coverage.
4. **Keep it but symmetrise the basis choice** across the pair being compared
   rather than choosing independently per model. This is a real design change,
   not a tweak, and I have not measured whether it works.

### What I would need from you

A ruling between 1–4. If 3 or 4, that is a design task rather than a
measurement one and wants its own plan and its own gate.

### A related measurement bug, found and fixed while investigating this

Twice in one session a measurement's *variant definitions* went stale against
the recipe they were varying, and both times the result was a table that read
as reassuring:

1. Measurements D and E did not thread `reference=` through `canonicalize()`.
   Without it `AlignFFNNeurons` is a no-op, so every variant collapsed to "no
   permutation step" and the table reported all five as identical.
2. After that was fixed, the variant *filters* were still written against
   `SortFFNNeurons`, which `DEFAULT_RECIPE` no longer contains. The filter
   removed nothing, so the variant labelled **`no_permutation_step` still
   performed the permutation step** — and reported that dropping it costs
   nothing. Dropping it costs `6.9e+07`.

Both are the S55 shape one level up: not a mis-named quantity but a mis-named
*experimental condition*, producing a plausible number that argues something is
safe. Fixed, and D now carries a `without_head_internal` variant because that
is the live attribution question rather than the retired one.

### Note on the earlier report

Phase 5 as first reported said the residual `3.05` came from the head-internal
step and described it as a smaller, separate matter. That was measured at three
seeds and three epsilons, and it understated the problem: at ten seeds the same
step reaches `2189.9`. The earlier statement was not wrong about the cause,
only about the size, and the difference was hidden by too few seeds. This is
the third time in this build that widening a measurement changed a conclusion.


---

## D-2. The ruler's distortion is direction-dependent and cannot be divided out

**Status: RULED 2026-08-02 — option 2. Gain absorption was DROPPED.** The step
still exists and is still tested; it runs against
`RETIRED_GAIN_ABSORPTION_RECIPE` in `scripts/canonicalize.py`, retained so this
entry's measurement stays reproducible. It is not applied to anything the study
measures. Dropping it took the ruler from a direction-dependent
`[0.848, 1.711]` to `1.00041 [1.00039, 1.00046]` across ten seeds. The
consequence — LayerNorm gain is no longer quotiented — is carried by the same
zero-gradient argument as D-1: the gain is a continuous gauge, so same-seed
twins hold identical values there and it cancels unaided.

*(This line read "open. Nothing changed." until 2026-08-04. Everything below it
is the case FOR the ruling, preserved as written; it is not a description of
the current recipe.)*

### The decision

Is a ruler acceptable when its distortion depends on the *direction* of the
difference being measured, and the relevant direction cannot be known until
there are trained checkpoints?

### What was measured

Real GPT-2, `eps=1e-6`, ten seeds, isotropic perturbation. Ratio of 1 means
distance-neutral.

| variant | median | min | max |
| --- | --- | --- | --- |
| shipped recipe | `0.92200` | `0.84828` | `1.71067` |
| EMPTY control | `1.00000` | `1.00000` | `1.00000` |
| without gain absorption | `1.00041` | `1.00039` | `1.00046` |

Every other step removal leaves the numbers unchanged. The empty control
returning exactly 1.0 rules out the measurement harness.

### Why it happens

Gain absorption multiplies weight rows by `gamma`. That preserves the function
exactly -- it is a valid symmetry -- but it is not an isometry of parameter
space. It shrinks directions where `gamma < 1` and stretches those where
`gamma > 1`. GPT-2's gains span `0.042` to `17.4`.

So the ratio is not noise and not instability. It is the correct answer for a
map that genuinely scales directions differently, sampled by whichever random
direction each seed happens to draw. That is why removing the step collapses
the spread from `[0.85, 1.71]` to `[1.00039, 1.00046]`.

### Why this is worse than a constant

A constant factor could be divided out. **This one cannot**, because it depends
on the direction of the difference. The study's real arm-vs-twin difference is
not an isotropic random direction; it is whatever direction training moved the
weights in. Until checkpoints exist, which value in `[0.85, 1.71]` applies to
the study's actual measurement is unknown.

### Options

1. **Accept it**, documented as a stated range, on the grounds that the study's
   headline is a *comparison between arms* rather than an absolute distance --
   if every arm is distorted by a similar factor relative to its twin, the
   ordering may survive even if the magnitudes do not. NOT MEASURED; this would
   need checking once checkpoints exist.
2. **Drop gain absorption too.** Ratio returns to `1.0004 +/- 7e-05`, which is
   as close to neutral as anything measured in this build. Cost: LayerNorm gain
   stops being quotiented, and it is a CONTINUOUS gauge -- so by the same
   zero-gradient argument used in D-1, same-seed twins carry identical values
   there and it cancels anyway. **This is the option most consistent with the
   D-1 ruling**, and would leave a four-step recipe.
3. **Measure the distortion along the real difference direction** once
   checkpoints exist, and correct for it if it turns out stable. Defers the
   decision rather than taking it.

### What I would need from you

A ruling. Option 2 is a further narrowing of what the ruler quotients and is
therefore the same class of decision as D-1; I have not taken it.


---

## D-3. The head sort is the last remaining sort, and it flips

**Status: open. Raised 2026-08-03 by section B at ten seeds. Nothing changed.**

### The decision

Should `SortHeads` be replaced by Hungarian matching, the way `SortFFNNeurons`
already was?

### What was measured

Section B, shipped recipe, real GPT-2, ten seeds, isotropic. **Read the range.**

| eps | min | median | max | head-order flips |
| --- | --- | --- | --- | --- |
| 1e-8 .. 1e-4 | 1.0004 | 1.0004 | 1.0005 | 0 |
| **1e-3** | 1.0004 | **1.0004** | **83.99** | 1 |
| 1e-2 | 1.0004 | 8.4576 | 10.91 | 3 |

At `eps=1e-3` the **median is 1.0004 -- indistinguishable from a perfect
ruler** -- and one seed in ten reads `83.99`. Reporting the median alone would
have shown nothing at all. This is the fifth time in this build that a
comfortable summary statistic hid a real effect.

The mechanism is the one that retired the FFN sort: the head sort's deciding
margin on real GPT-2 is `3.452e-05` (measurement C, `t=1`), and once the
perturbation exceeds it the sort order flips. When two heads swap, they exchange
their full parameter blocks, so the error is O(1) against an O(eps) real
difference.

### Why this is not already settled by the FFN ruling

Scale. 12 heads against 3072 neurons means far fewer adjacent pairs and a
margin three orders of magnitude wider (`3.452e-05` against `1.490e-08`), so
the head sort only breaks at `eps >= 1e-3` where the FFN sort broke at
`eps = 1e-8`. It is the same defect with far more headroom.

Whether that headroom is enough depends on how far a burst arm sits from its
seed-matched twin, **which is the same unknown that D-2 turned on and is not
knowable until there are checkpoints.**

### Options

1. **Replace `SortHeads` with head alignment.** `align_permutations_to` already
   supports heads (`heads=True`) and is measured at `1.0` in sections D and E.
   Cost: canonical form becomes pairwise-relative in one more respect, which it
   already is for the FFN.
2. **Leave it.** Correct if the real arm-vs-twin distance is below `1e-4`
   relative. Unverifiable today.
3. **Leave it but assert the margin at measurement time** -- refuse to report a
   distance when the perturbation exceeds the head-sort margin, rather than
   silently returning an inflated number.

### What I would need from you

A ruling. Option 1 is the same change already made for the FFN and its
machinery exists; I have not made it.


---

## D-4. Multiple-comparison correction: spec-v4 section 9.4 does not exist

~~**Status: open. Raised 2026-08-03 by step 16. Nothing decided.**~~

**RULED 2026-08-07 by Asa. Family of tests = 2 (the two pre-registered
contrasts, reading 3 below). Correction = `holm`.** Recorded in full at
`docs/preregistration.md` §10 A-1, which also carries the timing check showing
no confirmatory-arm run existed when this was fixed.

Holm controls the family-wise error rate and assumes nothing about dependence
between the tests. At a family of 2 that costs almost nothing against no
correction — the smaller p is tested at 0.025, the other at 0.05 — so
Benjamini-Hochberg's extra power does not arise while its positive-dependence
assumption would have to be argued. The exploratory comparisons of §7 are a
separate, disclosed family and are not corrected into this one.

**This entry and `docs/preregistration.md` §2 both recorded D-4 as Zach's.
It was ruled by Asa. Zach's confirmation is outstanding; if he rules
differently his ruling governs.** The original text is kept below unstruck,
because what was open and why is the part that stops this from being a
decision nobody can audit.

### The decision

Which multiple-comparison correction the analysis applies across arms.

### What raised it

Step 16 was specified as "multiple-comparison correction per spec-v4 section
9.4". **`docs/spec-v4.md` has eight sections and no section 9.4** — it runs
from "Why v3 was retired" to "Deliberately still open" and contains no
statistics section at all. Grep for `correction`, `p-value`, `significan`,
`confidence` or `paired` returns nothing but one mention of the twin as a noise
floor.

So the analysis has no specified correction to apply. It is built with the
method as a **required parameter with no default**, and refuses without one.

### Why this is not mine to take

The correction determines which arms are reported as separated, which is the
study's headline claim. The comparison count is not obvious either — see below
— and that choice alone moves every corrected p-value.

### The candidates, and what each assumes

| method | controls | assumes | costs |
| --- | --- | --- | --- |
| **Holm–Bonferroni** | family-wise error rate | nothing about dependence | most conservative; with 15 pairwise comparisons the smallest p must clear α/15 |
| **Benjamini–Hochberg** | false discovery rate | positive dependence between tests (plausible here — the arms share a twin and a seed) | more power; accepts that a stated fraction of the "significant" results are false |
| **Benjamini–Yekutieli** | false discovery rate | nothing about dependence | safe under any dependence, noticeably weaker than BH |
| **None, pre-registered contrasts only** | nothing | that the contrasts were fixed in advance | no correction needed if the number of tests is one or two, decided before seeing data |

### The prior question, which changes every number above

**What is the family of tests?** Three readings, and they give different
denominators:

1. **All pairwise between the six injecting arms** — 15 comparisons.
2. **Each injecting arm against twin** — 6 comparisons.
3. **The pre-registered ladder only** — the structure ordering
   (fluent > scrambled > pos-substituted > random-chars) and the truth contrast
   (fluent-false vs fluent-true) as two planned contrasts — 2 comparisons.

Reading 3 is the one `docs/spec-v4.md`'s "descending ladder of linguistic
structure" language implies, and it is by far the most powerful, but only if
those contrasts were genuinely fixed before data.

**Updated 2026-08-03: they are now recorded.** `docs/preregistration.md` fixes
two confirmatory contrasts — primary `fluent-false` vs `fluent-true`, secondary
`fluent` (pooled) vs `pos-substituted` — and declares every other arm and
comparison exploratory, including the full ladder ordering. It was written
while no checkpoint existed in the repo and none ever had, which is checkable
by command rather than by assertion.

That makes reading 3 **available**. It does not select it, and this entry is
still open: the family of tests and the correction method are both still
yours.

**Updated 2026-08-03.** This said `docs/preregistration.md` §8 "states its own
gap — the outcome metric is D-7 and is not fixed, so the pre-registration is
partial until D-7 is ruled." §8 has since been replaced by a decision rule that
closes D-7, so the pre-registration is no longer partial in that respect. D-4
is unaffected and §8.6 says so explicitly: the correction and its family remain
open, and `scripts/analysis.py` still requires both with no default.

### What I would need from you

Two rulings: the family of tests, then the correction method. The analysis
takes both as explicit parameters and refuses without them, so no default is
in force in the meantime.


---

## D-5. Full-checkpoint interval and the resume gap

**Status: open, deferred to the pilot by your ruling 2026-08-03. Recorded so it
is not lost.**

### The decision

Whether `checkpointing.full_interval` stays at 1000.

### What raised it

Step 15's status derivation. Weights-only checkpoints are **not resumable** —
`load_checkpoint` refuses them, because they carry no optimizer state and no
RNG state. So the maximum work lost to any death is one **full**-checkpoint
interval: 1000 steps.

A run that dies at step 998 has ten weights-only files and **restarts from
zero**. `launch.py` reports that state as `started_not_resumable`.

### The arithmetic

At an estimated 10 h per run (arithmetic from peak FLOPs, **not** a
measurement):

| full_interval | max work lost | full ckpts/run | GB/run | study total (70 runs) |
| --- | --- | --- | --- | --- |
| 1000 (current) | ~63 min | 10 | 105.5 | 7.385 TB of 10 |
| 500 | ~32 min | 20 | 115.5 | 8.085 TB of 10 |

### Why it is deferred rather than open

The 63-minute figure depends on a step time nobody has measured. The pilot
gives a real one. Changing the interval is a config edit and costs nothing
before runs start, so there is no reason to decide it early — but there is a
reason not to forget it, which is this entry.

The input that would settle it independent of step time: **whether the rented
hardware is spot/preemptible.** Preemptible makes an hour per death expensive;
on-demand makes it noise.


---

## D-6. Step 10's second half is BLOCKED, not deferred

**Status: skipped 2026-08-03, gate verified closed. Nothing built, nothing
decided.**

### What was skipped

The aligned barrier, aligned L2, and the RSF subspace probe -- the second half
of step 10. All three route through `scripts/canonicalize.py`.

### The gate, and how it was checked

**CORRECTED 2026-08-03. This block said the model swap "has not landed" and
that the layout question "was not decided". The code contradicts both, and the
original wording is left below so the correction is visible rather than
silent.**

What is actually true:

- `scripts/model_seam.py` builds a real `GPT2LMHeadModel`. `FAMILIES` is
  `("hf_gpt2", "probe_linear")`, and `--family` is **required with no default**
  in both `train.py` and `launch.py`.
- `scripts/canonicalize.py` is Conv1D-only, and its tripwire refuses any model
  whose `c_attn`, `c_proj` or `c_fc` is not a transformers `Conv1D`, naming
  `torch.nn.Linear` specifically. **That is the whole of what the tripwire
  enforces** — passing it is not by itself a claim that the module is correct
  for a given checkpoint.
- A checkpoint trained under `hf_gpt2` has Conv1D projections and so passes
  that layout check. One trained under `probe_linear` is `nn.Linear` and is
  refused by it.
- What is still `nn.Linear` is `probes/determinism/model.py` — **the probe, not
  the study's model.**

So these three functions are **UNBUILT, and D-6 was SKIPPED on 2026-08-03
rather than blocked.** Nothing prevented them; nobody built them. The
distinction matters because "blocked" implies waiting on someone else, and this
is not waiting on anything.

**No layout adapter was built**, which remains true. `docs/layout-cost.md`
prices both directions and is unchanged.

> **The original wording, preserved:** *"The instruction was to build these if
> and only if the model swap had landed. It has not: `grep -c "nn.Linear"
> probes/determinism/model.py -> 6`, `class GPT(nn.Module)` at line 108.
> `probes/determinism/model.py` is still the handwritten `nn.Linear`
> implementation ... No layout adapter was built and the layout question was
> not decided, per instruction."*
>
> The grep was accurate and is still accurate. The error was reading it as a
> fact about the study's model when it is a fact about the probe.

### What unblocks it

The swap of the study's model to HF GPT-2, which is yours and Asa's and is
already decided-but-unexecuted. Once `probes/determinism/model.py` is Conv1D --
or the study's model is HF GPT-2 and the probe is retired -- step 10's second
half becomes buildable with no further ruling.

### Why this is in the queue rather than the notes only

So that "step 10 is half done" is a live item with a named unblocker, rather
than a fact someone has to rediscover from `metrics.py` raising
NotImplementedError three months from now. The report at
`docs/measurements/10-metrics.md` already says the module is not finished; this
says what would finish it.


---

## D-7. Which metric is the study's headline

**Status: CLOSED 2026-08-03 by a decision rule, not by a pick.**
`docs/preregistration.md` §8 rules it. The headline metric is selected by a
**single-checkpoint measurement** taken at the pilot -- `canonicalization_error.py`
against the last checkpoint at or before the injection step (step 199 under the
shipped config) at eps=1e-6 over ten directions -- branching
on the spread and median of the distortion factor. Three of the four branches
give the **plain loss barrier**; only `spread <= 2 and median >= 1.01` selects
the permutation-aligned barrier, and that branch is the one that would require
D-6 to be built.

The rule is in force from the date of that document. The **branch has not been
taken**, because the measurement needs a checkpoint that does not exist yet.
That is not the same as the decision being open: what gets measured, what the
thresholds are, and which way each outcome resolves are all fixed in advance,
and §8.5 requires the branch to be recorded before any arm-vs-twin distance is
examined.

### The decision

Which of the step 10 metrics carries the study's primary claim.

### ~~What raised it, and a correction to the premise~~ THE CORRECTION WAS WRONG

**Corrected 2026-08-03. The text below is preserved because it is the error,
and because it is the second time in this entry's life that a `grep` decided
something it was not competent to decide.**

The claim below -- that `docs/spec-v4.md` names no metric, on the evidence of
`grep -c metric` returning `0` -- is **false**, and the thing it "corrected"
was right all along. Design Spec v4 §8.1 **does** name the permutation-aligned
loss barrier as the primary readout, and §2's hypothesis is stated in its
terms.

The grep was run against `docs/spec-v4.md`, which is an **incomplete copy of
the spec, missing §8 and §9 entirely**. The command was executed correctly and
reported truthfully about the file it was given. The file was not the document
whose contents were being asserted.

So D-7 never rested on an absence of decision. It rested on a **measurement
artifact** -- and the artifact was believed over the original claim precisely
because it came with a number attached. Logged as S91.

> **The original text, preserved:** *"I was told `docs/spec-v4.md` 'names the
> permutation-aligned barrier as the primary metric'. **It does not.**
> `grep -c metric docs/spec-v4.md` returns **0** -- the document has eight
> sections and none of them mentions a metric of any kind, let alone names one
> as primary. There was no claim to rewrite. So the situation is not a stale
> statement. It is an **absence**: the metrics exist in code, none is
> designated, and a reader could reasonably assume the spec had chosen.
> `docs/spec-v4.md` now has a Metrics section stating that absence and pointing
> here."*
>
> The Metrics section that was added to `docs/spec-v4.md` on the strength of
> this therefore documents an absence that is an artifact of that file being
> partial. Correcting it means reconciling the partial copy against the full
> spec, which is a larger job than this entry and is **not done**.

### What exists

Built and tested (`scripts/metrics.py`): interpolation barrier, raw L2,
activation cosine similarity, per-layer CKA.

Not built, raising `NotImplementedError`: permutation-aligned barrier,
permutation-aligned L2, RSF subspace probe. (This said "blocked on the model
swap". They are **unbuilt, not blocked** -- D-6 was skipped on 2026-08-03, and
`--family hf_gpt2` builds a real `GPT2LMHeadModel` today. See S88.)

### The evidence that bears on it, which points away from the aligned metrics

Step 9 measured what canonicalization actually contributes **between
same-seed twins**, which is the only comparison this study makes:

    EMPTY recipe (do nothing at all)   1.00000  on all ten seeds
    shipped four-step recipe           1.00041

`docs/step9-summary.md` states the consequence directly: "the gap between those
two numbers is the entire value canonicalization adds for this study's
comparison." Twins share an initialization, the continuous gauges have exactly
zero gradient and never move during training, so they cancel without help.

That is an argument that an **aligned** metric may buy very little here, at the
cost of depending on `canonicalize`, which is Conv1D-only and currently blocks
three metrics entirely.

It is not a decisive argument, and I am not treating it as one. Two things
cut the other way:

1. Alignment is about **discrete** permutations, not the continuous gauges. The
   zero-gradient argument is exact for the continuous ones and only an analogy
   for permutations -- `docs/step9-summary.md` makes that distinction itself.
   Whether a genuine head or FFN permutation ever arises between same-seed
   twins is listed there as unverified.
2. Step 9 measured the ruler against synthetic perturbations on public GPT-2,
   not against real twin-vs-twin distances, which do not exist yet. The 1.00041
   is a property of the ruler, not of the comparison it will be used for.

### ~~What I would need from you~~ RESOLVED — what happens instead

**Nothing further is needed.** The option this section described as defensible
-- deferring until the aligned-versus-raw question can be measured rather than
argued -- is what `docs/preregistration.md` §8 does, with the deferral written
as a rule fixed in advance rather than as a judgement made later. The evidence
above is what the rule's thresholds were calibrated against.

`scripts/analysis.py` still takes the metric name as an argument and still has
no default, so nothing is silently in force between now and the pilot.

**Still open, and unaffected by this:** D-4, the multiple-comparison correction
and its family. `docs/preregistration.md` §8.6 says so explicitly.


---

## D-8. Three wordings the contradiction scan confirmed and I did not change

**Status: open. Raised 2026-08-04 by the second contradiction pass. All three
are confirmed defects; none is mine to word.**

See `docs/contradiction-scan-2026-08-04.md` for the full pass.

### 8a. `analysis.py` says the noise floor is "wider by construction"

It is not. A verifier produced a counterexample from the module's own test
helpers: at `twin_jitter=0.5` the floor's widest absolute difference is 0.98
while a fabricated effect of 5.0 exceeds it. The floor is wider **in
expectation under the null**, not by construction.

This matters because it changes how `clears floor: yes` reads. If the floor
were wider by construction, clearing it would be a strong statement; if it is
only wider in expectation, clearing it is weaker and depends on how much the
twin actually varies. The same sentence is in `implementation-notes.md` S83.

**What I need:** the phrasing you want, since it is a claim about what the
design guarantees rather than a typo.

### 8b. The noise-floor labels hardcode "twin" while the number follows `--reference`

`report_banner` and `build_provenance` print "twin-vs-twin ACROSS seeds", but
the value is `noise_floor(panel, reference)` and `--reference` is a free CLI
parameter. Running with `--reference fluent-false` produces a label that lies.

Two fixes, and they are different decisions:

1. **Pin it.** Remove `--reference`, hardcode `twin`. Matches `docs/spec-v4.md`,
   which names the twin as *the* reference, and makes the label true by
   construction.
2. **Derive the label.** Keep the parameter and interpolate the arm name.
   Useful if you ever want an arm-vs-arm floor for a sanity check.

**What I need:** whether an analysis against a non-twin reference is something
the study should be able to express at all.

### 8c. `student_t_sf` returns the two-sided tail

`sf` is the one-sided survival function in scipy and in normal usage;
`student_t_sf(0.0, 9)` returns `1.0` where `scipy.stats.t.sf(0, 9)` gives `0.5`.
The docstring says "two-sided" and the cross-check test compares against
`2 * scipy.stats.t.sf(...)`, so **no number is wrong**.

Purely a naming question — `student_t_two_sided_sf` would be unambiguous — and
it is churn in tested code with no behavioural defect, which is why it is here
rather than done.

---

## D-9. What replaces the secondary confirmatory contrast

**Status: RAISED 2026-08-08, by the arm cut ruled the same day by Asa. OPEN.**

### What was decided, and by whom

The study drops from seven arms to four. `fluent-false`, `fluent-true`,
`random-chars` and `twin` are kept; `scrambled-false`, `scrambled-true` and
`pos-substituted` are cut as run conditions. The run count falls from 70 to 40
— computed, not typed, so it moved on its own when `ARMS` did. Ruled by Asa on
2026-08-08 on schedule and cost grounds, with the consequences below stated
before the ruling rather than discovered after it.

The texts stay in `bursts/`, on exactly the terms `scrambled-corpus` was cut on
2026-08-03: every measurement already taken from them stays reproducible.

### What it costs, stated rather than buried

**The secondary confirmatory contrast is gone, not deferred.**
`docs/preregistration.md` §6 named pooled `fluent-*` against `pos-substituted`,
and that arm can no longer appear in any panel this study produces. It is the
contrast that asked whether semantic coherence *beyond part-of-speech
structure* contributes to displacement. Nothing else in the design asks that
question, and no surviving arm can be substituted in without asking a
different one: `random-chars` has no grammatical structure to hold fixed,
which is the entire point of the comparison.

`scripts/analysis.py` keeps `SECONDARY_AGAINST = "pos-substituted"`
deliberately. Every analysis output will now carry the secondary contrast
reported as *uncomputable, naming the missing arm*, rather than silently
listing one contrast as though a second had never been registered. That
behaviour is pinned by
`tests/test_analysis.py::test_the_preregistered_secondary_contrast_is_uncomputable`.

**D-4 is left describing nothing.** Ruled 2026-08-07 at family = 2 — "the
family is the two contrasts §5 and §6 fix" — where the arithmetic quoted was
"the smaller p is tested at 0.025, and if it passes the other at 0.05". With
one computable confirmatory contrast there is no family to correct over and no
second p-value to order.

**The ladder loses three of four rungs.** The exploratory descending-structure
ordering was fluent > scrambled > pos-substituted > random-chars. What remains
is fluent vs. random-chars: the two ends, with nothing in between. `v4`'s
stated purpose — "an exploratory six-way categorical comparison, reported as a
leverage ordering" — is not what a four-arm study does. `random-chars` is now
the only exploratory arm.

### The decision this raises

**Is the study now single-contrast, or does something replace §6?**

1. **Single confirmatory contrast.** `fluent-false` vs `fluent-true` alone.
   No correction is needed at family = 1, D-4 closes as moot, and the study
   answers exactly one question: whether truth value, holding form fixed,
   changes displacement. Cheapest and most honest about what 40 runs buy.
2. **Promote `random-chars` to confirmatory.** Pooled `fluent-*` vs
   `random-chars` restores a family of 2 and keeps D-4's ruling standing
   as-written. But it is not §6's question — it contrasts fluent text against
   *no linguistic structure at all* rather than against structure-without-
   meaning, so it cannot separate semantics from grammar. Re-registering it as
   though it were §6 would be the "quietly widened at analysis time" failure
   `docs/preregistration.md` §9 warns about.
3. **Restore `pos-substituted` as a fifth arm.** 50 runs rather than 40,
   ~12 h more wall clock at 8 concurrent runs and ~1.06 TB more checkpoints at
   the current interval. Keeps the pre-registration intact as written and
   costs the least scientifically of the three.

**What I need:** which of the three, recorded here with a date and a name. The
code change is done either way — this decides what the analysis is allowed to
claim, and option 3 additionally requires re-running
`scripts/generate_overrides.py` against a five-arm `ARMS`.
