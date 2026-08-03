# Decisions pending

Things that came up during autonomous work, that change **what the study
measures** rather than how the code is written, and that are therefore not mine
to take. Each entry records what the decision is, what was measured that raised
it, the concrete options, and what I would need in order to act.

Nothing here has been acted on. Where an entry concerns a step that is
currently in `DEFAULT_RECIPE`, that step is **still in it, unchanged**.

---

## D-1. The head-internal step has the property that disqualified the FFN sort

**Status: open. Raised 2026-08-02 during the phase 5 re-run. Nothing changed.**

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
