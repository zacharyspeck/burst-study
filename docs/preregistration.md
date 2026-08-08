# Pre-registration: the confirmatory contrasts

**Date: 2026-08-03.**
**Commit this was written against: `f1d378ecda661d5b61ec29043ef17ff8fc2efb35`.**
(This file lands in a child of that commit. The parent is what the timing claim
below is anchored to.)

Recorded by Zach. This document fixes **two contrasts as confirmatory** and
declares everything else exploratory. It is written before any training run
exists, and the point of writing it is that the "before" is checkable by
someone who does not take our word for it.

---

## 1. Why this file exists at all

`docs/decisions-pending.md` D-4 asks which multiple-comparison correction the
analysis applies, and the answer depends first on **the family of tests**. One
of the three readings there — the pre-registered-ladder reading — requires that
the contrasts were fixed before any data existed. That reading is by far the
most powerful: it corrects across **2** comparisons rather than 15.

Until now nothing in the repo recorded such a fixing. The contrasts had been
stable in Zach's working notes for weeks, but **"stated in conversation" is not
pre-registration** — it is unfalsifiable after the fact, which is the entire
property pre-registration exists to supply.

This file supplies it.

## 2. THIS FILE DOES NOT RULE D-4

Stated explicitly because the temptation to read it as a ruling is obvious.

Recording these contrasts is what makes **D-4 option 3 available**. It does not
select it. The family of tests and the correction method are both still Zach's
to rule, and `scripts/analysis.py` continues to require each as an explicit
argument with no default. If D-4 is ultimately ruled as 15 pairwise comparisons
or 6 against-twin, this file remains valid and simply goes unused for its
original purpose — it would still be the record of what was planned in advance,
which has value independent of the correction arithmetic.

**Nothing in this document changes any default, any config value, or any code.**

> **SUPERSEDED IN PART, 2026-08-07.** §10 A-1 *does* now rule D-4, from inside
> this document. The section above is kept unedited because it is the record of
> what this file claimed on the day it was written, and because its substantive
> point still holds: recording the contrasts is what made the family-of-2
> reading *available*, and that availability is what A-1 spends. The sentence
> immediately above remains true — no default, config value or line of code is
> changed by anything here, including §10.

## 3. The timing claim, and how to check it

The claim is: **these contrasts were fixed before any data existed.**

The checkable half is that no data exists or ever has. Verify it directly:

```bash
# no checkpoint blob has ever been added, on any branch, in any commit
git log --all --diff-filter=A --name-only --pretty=format: \
  | sort -u | grep -Ei '\.(pt|bin|safetensors|ckpt)$'

# nothing on disk either, including ignored files
git status --porcelain --ignored | grep -Ei '\.(pt|bin|safetensors|ckpt)$'
```

Both were run on 2026-08-03 at the commit named above and both returned
**empty**. `docs/handoff-pilot.md` §9 says the same thing in prose: *"No run
has ever been trained. Every number in `docs/measurements/` comes from public
GPT-2, junk checkpoints, or synthetic input."*

**Be precise about what that does and does not establish.** It establishes that
no outcome data existed when this was written, so these contrasts cannot have
been chosen by looking at results — the ordinary failure this guards against.
It does not, and cannot, independently corroborate that these specific two were
the ones in the working notes; that part is Zach's attestation. The two halves
are recorded separately on purpose rather than blended into one claim.

## 4. The design these contrasts sit in

70 runs: 10 seeds × 7 arms. Within a seed, every run is identical except for
the 194-token burst injected at step 200. The comparison is therefore a
**paired difference within seed**, and the `twin` arm — which injects nothing —
supplies the noise floor across seeds (`scripts/analysis.py`, and S83 in
`implementation-notes.md`).

`twin` is the reference, not a contrast arm. It appears in neither contrast
below.

## 5. PRIMARY CONTRAST — `fluent-false` vs `fluent-true`

### What is held constant and what varies

Both arms are grammatical English of the same register, structure and length.
The **only** intended difference is whether the asserted proposition is false or
true.

### What it tests

Whether the **truth value of an assertion**, holding linguistic form fixed,
changes how far a single gradient step displaces the model. This is the study's
sharpest question, because truth is the one property here with no surface
correlate the optimizer could be reading instead.

### What a null result would mean

No detectable difference in displacement between the two arms, relative to the
twin noise floor: **the model's update at step 200 is insensitive to the
propositional truth of fluent text.** That is a substantive finding, not a
failed experiment, and it is the outcome the design should be expected to
produce if displacement tracks form rather than content.

A null here does **not** license the stronger claim that truth is never
encoded — only that it does not move the weights measurably at this scale, this
step, and this burst length.

## 6. SECONDARY CONTRAST — `fluent` (both arms pooled) vs `pos-substituted`

### What is held constant and what varies

`pos-substituted` replaces each word with another of the same part of speech,
so the part-of-speech skeleton and the token-length profile survive while
lexical semantics and coherent meaning do not. Pooling `fluent-false` and
`fluent-true` is deliberate: the primary contrast is what interrogates the
difference *between* them, so pooling here does not spend the same comparison
twice.

### What it tests

Whether **semantic coherence beyond part-of-speech structure** contributes to
displacement — that is, whether the effect of "fluent English" is more than the
effect of "English-shaped token statistics."

### What a null result would mean

Displacement is driven by surface and part-of-speech statistics rather than by
coherent meaning: **a sentence and its POS-matched nonsense move the model the
same distance.** That would place a real upper bound on how much of any burst
effect can be attributed to meaning, and it would make the primary contrast's
result much harder to interpret as being about content — so this contrast is
partly a check on the primary one, not merely a second question.

## 7. Everything else is EXPLORATORY

The remaining arms — `scrambled-false`, `scrambled-true`, `random-chars` — and
**every comparison among them or against the two contrasts above** are
exploratory.

This explicitly includes the **descending ladder** ordering that
`docs/spec-v4.md` describes (fluent > scrambled > pos-substituted >
random-chars). The ladder is the study's motivating picture and it is why these
seven arms exist, but the full ordering is **not** registered here as a
confirmatory test. Only the two contrasts above are.

Exploratory results:

- are reported as exploratory, in a section that says so
- are not described as "significant", "confirmed", or "separated" without the
  word exploratory attached
- may motivate a future confirmatory study; they do not become confirmatory by
  surviving a correction applied after the fact

If an exploratory comparison produces the most interesting result in the study,
it is still reported as exploratory. That is the commitment; it costs nothing
to make now and everything to make later.

## 8. The outcome metric: a ruling, a complication, and a decision rule

*Written 3 August 2026, before any checkpoint has existed in this repository.*

### 8.1 What was already ruled

Design Spec v4, dated 2 August 2026, §8.1, names the **permutation-aligned loss barrier** between a burst run's final weights and its seed-matched twin's as the study's primary readout. §2's hypothesis and wrongness condition are stated in its terms. That ruling was made before any training run existed and stands as the default.

An earlier note in `docs/decisions-pending.md` recorded that the spec does not name a primary metric, on the basis of a `grep` returning zero matches. That grep was run against `docs/spec-v4.md`, which is an incomplete copy of the spec missing §8 and §9 entirely. The original claim was correct; the correction was not. This is recorded here because the reason D-7 appeared open was a measurement artifact, not an absence of decision.

### 8.2 What changed after the ruling

Infrastructure step 9, completed 2–3 August, measured how much work permutation alignment actually does. On models sharing an initialization — which every arm and its twin do — the shipped four-step canonicalization recipe contributes a factor of **1.00041** [1.00039, 1.00046] across ten seeds, against an empty recipe's exactly **1.00000**.

The mechanism: twins share an initialization, and the coordinates a permutation would move are ones gradient descent does not spontaneously scramble. There is almost nothing for alignment to correct.

If that finding holds at the point of injection, the permutation-aligned barrier and the plain barrier are, for practical purposes, the same number — and the aligned version is unbuilt (`aligned_barrier` raises `NotImplementedError`) while the plain version is built and tested.

### 8.3 Why that finding is not yet sufficient to overturn 8.1

Three limitations, all recorded in `docs/measurements/9-canonicalization-error.json`:

1. **Everything in step 9 was measured on public, fully-trained GPT-2.** This study injects at step 200, roughly 52M tokens into a from-scratch run — far closer to initialization.
2. **The dispersion sweep says conditioning is worse where we inject.** At the initialization configuration, the FFN deciding margin is roughly two orders of magnitude smaller than at the trained model where every other conditioning number in that file was taken.
3. **There is a cliff.** At ε = 1e-3 one seed of ten reads 83.99 against a median of 1.0004, traced to a single head-order flip. Nobody knows where real twin-vs-twin displacement lands relative to that cliff.

So the question separating the plain barrier from the aligned one is not a matter of preference. It is a measurement that has never been taken on a model resembling the one this study trains.

### 8.4 The decision rule

**What gets measured.** `scripts/canonicalization_error.py`, run against the last checkpoint at or before the injection step (step 199 under the shipped config), at ε = 1e-6 across ten random perturbation directions — matching the epsilon and seed count of the existing sections D, E and F so the numbers are comparable to the public-GPT-2 table. A pre-injection checkpoint is bit-identical across all seven arms at the same seed, so it structurally cannot carry outcome information. This measurement involves a single checkpoint. It requires no arm-vs-twin comparison and produces no outcome data.

**Two criteria, taken from the measurement:**

- **Spread** — the ratio of the largest to the smallest distortion factor across the ten directions.
- **Magnitude** — the median distortion factor.

**The rule, in force from the date of this document:**

| Condition | Headline metric |
|---|---|
| Spread > 2 | **Plain loss barrier.** The aligned ruler is inconsistent at the point of injection and cannot be corrected for |
| Spread ≤ 2 and median < 1.01 | **Plain loss barrier**, with the aligned barrier reported as a robustness check |
| Spread ≤ 2 and median ≥ 1.01 | **Permutation-aligned loss barrier**, as spec v4 §8.1 ruled. D-6 gets built |
| The measurement cannot be produced, or is ambiguous against these thresholds | **Plain loss barrier** |

**Where the thresholds come from.** On public GPT-2 the shipped recipe's spread is 1.000456 / 1.000391 = 1.00007 — essentially none. The retired head-internal step, which was removed *because* of its inconsistency, showed 2450 / 3.09 = 793. A spread threshold of 2 sits enormously above the observed-good case and far below the observed-bad one. The 1.01 magnitude threshold asks whether alignment moves the number by more than one percent; below that it is not doing work worth an unbuilt module.

Three of the four branches land on the plain barrier. That asymmetry is deliberate: the more elaborate ruler has to earn its place, and every uncertain outcome defaults to the simpler, built, tested one.

### 8.5 Ordering constraints

Both are conditions on the validity of this section:

1. The canonicalization measurement is run and its branch recorded **before any arm-vs-twin distance from the pilot is examined**. The pilot produces real checkpoints; looking at displacement first and the ruler second would make the choice contingent on the result.
2. This document is committed before the pilot launches. A decision rule written after the measurement is not a decision rule.

### 8.6 What this section does and does not close

**Closes:** D-7. The headline metric is determined by the rule above, not by inspection of results.

**Does not close:** D-4, the multiple-comparison correction and its family. `scripts/analysis.py` continues to take both the metric name and the correction method as required parameters with no default, so nothing is silently in force.

**Does not change:** the two confirmatory contrasts registered in §5 and §6 — fluent-false vs fluent-true as primary, pooled fluent vs pos-substituted as secondary. This section fixes what is measured about them; §5 and §6 fix which arms are compared.

### 8.7 What would invalidate this section

- Any checkpoint predating the commit of this document.
- The canonicalization measurement being taken on a model other than the last checkpoint at or before the injection step, from this study's own configuration.
- The thresholds in 8.4 being revised after the measurement is seen.
- Any arm-vs-twin displacement being examined before the branch is recorded.

## 9. What would invalidate this document

Honesty about the failure modes, since a pre-registration that cannot be
falsified is decoration:

- any checkpoint predating this commit turning up, which would break §3
- the arm definitions or burst texts changing after this date in a way that
  alters what `fluent-*` or `pos-substituted` mean — the texts are committed
  content, so `git log` on `configs/` and the burst-text paths is the check
- the contrasts here being quietly widened at analysis time (for example,
  splitting the pooled `fluent` in §6 into two more tests) — that would make
  the family larger than 2 and the correction wrong

## 10. AMENDMENTS, recorded before the data they bear on

**Added 2026-08-07 by Asa.** Three decisions this document previously left open
or deferred. They are gathered here, rather than only in their home documents,
because each is worthless unless it was fixed *before* the runs it governs — and
this is the file whose whole purpose is to make a "before" checkable.

### The timing claim for these amendments, and how to check it

The runs these govern are the **confirmatory arms**: `fluent-false`,
`fluent-true` and `pos-substituted`. As of this commit **not one of them has
ever been trained**, at any seed. Verify directly:

```bash
# no confirmatory-arm run directory exists anywhere
find /shared/27as66 -maxdepth 3 -type d \
  \( -name '*fluent*' -o -name '*pos-substituted*' -o -name '*scrambled*' \)

# what does exist, and it is the pilot only
ls -d /shared/27as66/burst-pilot/runs*/*/
```

Run on 2026-08-07: the first returned **empty**; the second returned exactly
`seed00_twin`, `seed01_twin`, `seed02_twin`, `seed00_random-chars`. Three twins
and one **exploratory** arm (§7). No confirmatory contrast has any data.

Be precise about what that establishes, in the same spirit as §3. It establishes
that no confirmatory outcome existed when these were fixed, so they cannot have
been chosen by looking at a contrast's result. It does **not** establish that
nothing relevant was seen: the pilot's `random-chars` arm and the arm-matching
measurements below were both examined first, and A-2 in particular was ruled
*because of* what they showed. That is stated rather than hidden — the guard
these amendments provide is against choosing on the **outcome of a registered
contrast**, which is the thing that was not available and still is not.

### A-1. D-4 is ruled: family of **2**, correction **Holm–Bonferroni**

Canonical record: `docs/decisions-pending.md` D-4.

The family is the two contrasts §5 and §6 fix — reading 3 of the three D-4
lists. This document made that reading *available* (§2); A-1 selects it.
Correction is `holm`, passed to `scripts/analysis.py` as `--correction holm`.

Rationale, recorded so it can be argued with: Holm controls the chance of **any**
false positive and assumes nothing about how the tests relate. At a family of 2
its cost against no correction at all is negligible — the smaller p is tested at
0.025, and if it passes the other at 0.05 — so the extra power Benjamini–Hochberg
buys does not arise, while BH's assumption of positive dependence would have to
be argued. Exploratory comparisons (§7) are reported **separately**, labelled
exploratory, and are not members of this family; if they are ever corrected among
themselves that is a second, disclosed family and does not touch these two.

**OWNERSHIP NOTE, stated because this file previously said otherwise.** §2 above
and `decisions-pending.md` both record D-4 as **Zach's** to rule. It is ruled
here by Asa on 2026-08-07. That is a real change of hands and it is written down
rather than smoothed over: **Zach's confirmation is outstanding**, and if he
rules differently his ruling governs and this amendment is superseded in place.

### A-2. Spec-v4 "still open" item 3 is ruled: the matching target is the **full-batch delta**

Canonical record: `docs/spec-v4.md`, "Deliberately still open" item 3.

Arms are matched on what the burst contributes to the optimizer step that
actually lands — `grad(batch with burst) − grad(batch without)` — computed as
`scripts/train.py` computes it. The sequence-level criterion is **also reported**
for every arm, always, so nothing is hidden by the choice and comparison to
`8b-i` stays possible.

Evidence, from `docs/measurements/2026-08-07-arm-match-real-model.md` §6, three
seeds:

- Both criteria **travel**. Spread across the six arms is 1.1034 / 1.1112 /
  1.1232 under the full-batch delta and 2.2785 / 2.2599 / 2.5301 under
  sequence-level gradient norm. The delta is the steadier, which answers the
  objection that it is an artifact of one particular batch.
- The two criteria **disagree stably**, not noisily: `fluent` pooled against
  `pos-substituted` is +38.8/39.1/39.1% under sequence-level loss and
  +1.0/0.9/2.2% under the delta. This is a definitional fork, and no further
  measurement resolves it.
- Under the sequence-level criterion `fluent-false` carries the larger gradient
  norm at **all three seeds** — a systematic mismatch sitting directly on the
  primary contrast. `8b-iii` already showed it is **not** truth: a fabricated
  passage matched to `fluent-true`'s register scored *lower* than the true one
  (17.4198 against 18.0029). Under the delta the gap falls to 0.14–0.87% and
  **the sign flips between seeds**, which is what a residual with no systematic
  component looks like.

So the delta is the criterion under which a **named nuisance**, already shown not
to be the variable under test, does not contaminate the primary contrast.

**THE OBJECTION TO THIS RULING, recorded because it is real.** The delta is also
the criterion that makes the arms look best, and choosing a criterion after
seeing which one flatters the data is how a pre-registration decays. Two things
are offered against that and neither is decisive on its own: the argument — *it
is the quantity that physically reaches the weights* — is one
`docs/v4-gap-analysis.md` §3 already made when it called the delta "the
actually-applied contribution", declining it **only** on compute cost, which no
longer exists (4 seconds per arm); and the sequence-level numbers are reported
alongside forever, so any reader can apply the other criterion themselves. A
reader who thinks this was chosen for its answer should read §6 of the
measurement document and decide.

**Item 1, the matching tolerance, is NOT ruled here and remains open.** Ruling
the target says what to measure, not how close counts.

### A-3. The interim look at the calibration runs is **variance-only**

The next six runs — `fluent-false` and `fluent-true` at seeds 0–2 — exist partly
to estimate a quantity the pilot cannot supply: the standard deviation, across
seeds, of the within-seed paired difference. Every seed count this study has
published (9.8, 50, 34, 32) substitutes the twin-vs-twin floor for it, which is
a different quantity.

**The commitment, fixed here and before those runs exist:** when they land, the
**spread** of the three paired differences may be computed and used to set the
seed count. Their **mean, sign, confidence interval, and any test statistic may
not be examined** until the full seed count is complete.

Without this, choosing `n` after seeing the effect is choosing the experiment's
size knowing its answer, and the resulting p-values do not mean what they say.
This is an ordinary internal-pilot design and it is legitimate — but only when
recorded first, which is why it is here and not in a later analysis note.

The six runs are **study runs**, not calibration overhead: they are seeds 0–2 of
two confirmatory arms and enter the final analysis unchanged.

---

### A-4. The arm list is cut to **four**, and §6 dies with it

**Ruled 2026-08-08 by Asa.** The study runs `fluent-false`, `fluent-true`,
`random-chars` and `twin` — 10 seeds × 4 arms = **40 runs**, down from 70.
`scrambled-false`, `scrambled-true` and `pos-substituted` are cut as run
conditions on schedule and cost grounds.

**This amendment invalidates §6 and, through it, A-1.** Both are left standing
below rather than edited, because a pre-registration that quietly rewrites its
own contrasts after the fact is worth nothing:

- **§6 cannot be computed.** It names `pos-substituted`, which no panel this
  study produces can contain. `scripts/analysis.py` still declares
  `SECONDARY_AGAINST = "pos-substituted"` so that every analysis output reports
  the contrast as absent and names the missing arm, instead of presenting one
  contrast as if a second had never been registered.
- **A-1's family of 2 no longer exists.** It was ruled on the two contrasts §5
  and §6 fix. One of them is gone, so there is one confirmatory p-value, and
  Holm at family 2 — "the smaller p is tested at 0.025, and if it passes the
  other at 0.05" — has nothing to order. **No correction is applied under this
  amendment.** What replaces §6, if anything, is D-9 in
  `docs/decisions-pending.md`. **D-9 was ruled the same day, by Asa: option 1,
the single confirmatory contrast.** §6 is not replaced, `random-chars` is not
promoted into its slot, and with a family of one **no correction is applied**.
This study tests exactly one confirmatory hypothesis: §5.

**What is lost, stated plainly.** §6 asked whether semantic coherence *beyond
part-of-speech structure* contributes to displacement. That question is no
longer asked by this study, and no surviving arm asks it: `random-chars` holds
no grammar fixed, so pooling the fluent arms against it tests fluent-vs-nothing,
not meaning-vs-grammar. Substituting it into §6's slot would be exactly the
"contrasts quietly widened at analysis time" failure §9 lists.

The exploratory ladder of §7 loses three of four rungs; what remains is
`fluent` vs `random-chars`, the two ends with nothing between them.
`random-chars` is now the only exploratory arm.

**§9 fires on this amendment by its own terms** — "the arm definitions …
changing after this date in a way that alters what `fluent-*` or
`pos-substituted` mean". Removing an arm outright is the stronger form of that,
and this section is the record that it happened deliberately, with the cost
priced, rather than being noticed later.

---

## Summary

| | |
| --- | --- |
| primary | `fluent-false` vs `fluent-true` |
| secondary | ~~`fluent` (pooled) vs `pos-substituted`~~ **UNCOMPUTABLE — arm cut 2026-08-08, §10 A-4. Reported as absent, not dropped** |
| confirmatory family size | ~~**2**~~ **1** — §10 A-4 |
| exploratory | ~~`scrambled-false`, `scrambled-true`,~~ `random-chars`, ~~the full ladder,~~ all other pairs — §10 A-4 |
| reference / noise floor | `twin` |
| pairing | within seed, 10 seeds (10 x 4 arms = **40 runs**, was 70 — §10 A-4) |
| outcome metric | **FIXED BY RULE — see §8.4.** Branch selected by a single-checkpoint measurement at the pilot; three of its four branches give the plain loss barrier |
| correction method | ~~**`holm`, family = 2** — ruled 2026-08-07 by Asa, §10 A-1~~ **NONE — family is 1 after §10 A-4. D-9 ruled 2026-08-08: single confirmatory contrast, §6 not replaced** |
| matching target | **full-batch delta**, sequence-level reported alongside — §10 A-2 |
| interim look | **variance-only** until the full seed count is complete — §10 A-3 |
