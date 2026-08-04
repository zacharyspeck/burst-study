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

## 8. THE INCOMPLETE PART, stated rather than buried

**A contrast is only fully pre-registered if its outcome measure is also
fixed, and ours is not yet.**

`docs/decisions-pending.md` **D-7 — which metric is the study's headline — is
open.** Four metrics are built (interpolation barrier, raw L2, activation
cosine, per-layer CKA) and three more raise `NotImplementedError` pending D-6.
This document fixes *which arms are compared*; it does not fix *what is
measured about them*.

So the pre-registration is real but partial, and the gap is exactly the
researcher-degrees-of-freedom that choosing a metric after seeing data would
open. **To close it, D-7 must be ruled before any checkpoint is examined** —
not merely before the numbers are written up. If D-7 is instead deferred until
checkpoints exist, as its own entry says is defensible, then the confirmatory
status claimed here is weakened accordingly, and the write-up must say so.

Recording this now so that it is a known, dated limitation rather than
something a reader reconstructs later.

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

---

## Summary

| | |
| --- | --- |
| primary | `fluent-false` vs `fluent-true` |
| secondary | `fluent` (pooled) vs `pos-substituted` |
| confirmatory family size | **2** |
| exploratory | `scrambled-false`, `scrambled-true`, `random-chars`, the full ladder, all other pairs |
| reference / noise floor | `twin` |
| pairing | within seed, 10 seeds |
| outcome metric | **NOT YET FIXED — D-7 open, see §8** |
| correction method | **NOT RULED HERE — D-4 is Zach's, see §2** |
