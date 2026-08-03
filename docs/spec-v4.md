# Spec v4 — mid-pretraining burst study

**Status: current design. Supersedes v3.**
Recorded 2026-08-01. Written down from a design discussion; there is no
upstream document this was copied from, and this file is now the
authoritative statement of the design.

The code in this repository still implements **v3**. Nothing in `burst/`,
`configs/`, `scripts/`, or `tests/` has been changed to match this file. See
`docs/v4-gap-analysis.md` for what that costs and what would have to move.

---

## Why v3 was retired

v3 was a two-arm contrast: a coherent burst against a word-scrambled burst,
matched on surprise, plus an ordinary-text control and a no-injection twin.
40 runs.

The scrambled arm was doing a job it could not do. **It was never a semantic
zero point.** The word-order literature shows shuffled text stays partly
meaningful to a language model, and pre-BPE word-level shuffling — which is
exactly what `window_shuffle()` does — leaks the most. Coherent-vs-scrambled
was therefore measuring grammar as much as it was measuring meaning, and the
headline contrast could not separate the two.

The empirical picture from the one measurement run performed under v3 is
consistent with this (`docs/measurements/2026-07-31-match-sweep.md`): the
scrambled arm's mean loss rose monotonically with window size while its
gradient norm stayed flat in a narrow band, never approaching the coherent
arm's. A single dial that moves one matched quantity but not the other is not
a dial between two conditions; it is a dial along one axis of a space with
more than two directions in it.

## What v4 is instead

**An exploratory five-way categorical comparison, reported as a leverage
ordering — not a binary contrast.** The question is no longer "does coherent
beat scrambled"; it is how five categories of injected text rank in the
displacement they produce.

## The arms

| # | Arm | What it is |
|---|---|---|
| 1 | **fluent-false** | Grammatical English asserting something specific and false. This is the existing `bursts/coherent.txt` — the Gizmo Harrington passage. |
| 2 | **fluent-true** | Same register and structure, asserting something true. |
| 3 | **scrambled-false** | `fluent-false` with word order broken by within-window shuffling. |
| 4 | **scrambled-true** | `fluent-true` with word order broken the same way. |
| 5 | **POS-substituted** | Each word replaced by a random word of the same part of speech. Grammar preserved, lexical content destroyed. |
| 6 | **random-characters** | No word structure at all. |
| — | **twin** | No injection. Both the per-seed displacement reference and, twin-vs-twin across seeds, the noise floor. |

**`scrambled-corpus` is CUT (2026-08-03).** `bursts/scrambled_corpus.txt` and
its `provenance.json` entry stay in the repo so the measurements taken from it
remain reproducible, but it is not a run condition.

WHAT THE CUT COSTS, stated as a limitation rather than buried: it removes the
only scrambled arm derived from ORDINARY CORPUS TEXT. The six remaining
injecting arms are all derived from the same Beatles-derived source material,
so **topic is no longer controlled across the scrambled family.** Any effect
attributed to linguistic structure could in principle be an effect of topic.
That was traded for run count, deliberately, and it is a stated weakness of the
design.

**The arm named `ordinary` is gone as an arm.** A raw untouched corpus span
is still needed as the *substrate* that scrambled and POS-substituted are
derived from, but it is no longer a run condition and no longer receives
seeds.

Note the design intent of the pairing: fluent-false and fluent-true are
matched in register and structure and differ only in truth value.
fluent-false, scrambled, POS-substituted and random-characters form a
descending ladder of linguistic structure. The twin sits under all of them.

## Study shape

**7 run types × 10 seeds = 70 runs.** Not 40, and not the 60 this document
said until 2026-08-03.

The scrambled family splits into `scrambled-false` and `scrambled-true` --
matched scrambles of the two fluent arms -- rather than the single `scrambled`
this section originally named. `scrambled-corpus` was cut; see below.

## Matching requirement

All five injected arms must be matched **to each other**:

- on **per-token loss**, within tolerance
- **and** on **gradient norm**, within tolerance
- at the **same token length**
- seeded and reproducible

Twin has no burst and therefore no loss or gradient norm to match. The
matching problem is five-way; the run structure is six-way. Anything that
conflates those two counts will be wrong somewhere.

**The tolerance is not set.** Setting it is a decision belonging to whoever
runs the study, and no script should apply, imply, or highlight one.

## Gradient direction

Direction is a **confound we record rather than control**. At the injection
step, log:

- pairwise **gradient cosine similarity** between arms
- per-arm **anisotropy statistics**

This is a measurement obligation, not a matching criterion. Arms are matched
on magnitude; direction is recorded so that a magnitude-matched result can
later be checked against the possibility that the arms pushed different ways.

## Injection

**Step 200, fixed.**

> **NOT YET IN THE CONFIG.** `configs/base.yaml` has
> `injection.injection_step: null`, and every injecting arm is launch-blocked
> until it is set. This section asserts the decision; the config records what
> has been written down. The two disagree until someone edits the config, and
> `--launch` is what stops that disagreement reaching a run. The same is true of
> `burst_length_tokens`, which this document notes below and which is also still
> null despite `bursts/` being built to 194 tokens.

Two interactions with values already in `configs/base.yaml`, noted because
neither is recorded anywhere as intentional:

- `learning_rate.warmup_steps` is **200**. Step 200 is therefore the first
  step at peak learning rate (0.0006). The burst lands exactly on the
  warmup/cosine boundary.
- `checkpointing.weights_only_interval` is **50**, and a checkpoint fires at
  0-indexed step `s` when `(s + 1) % 50 == 0`. There is therefore a
  weights-only checkpoint at **step 199**, immediately before injection. This
  is convenient and appears to be coincidence rather than design.

Nothing else about the run config changes from v3.

## Deliberately still open

These are unresolved as of this writing and are **not** to be settled by
implementation default:

1. **The matching tolerance.** Not set.
2. **The token length N.** Currently 194, inherited from the byte content of
   a passage written for v3, and enforced by nothing —
   `injection.burst_length_tokens` is still `null` in `configs/base.yaml`.
   Under v4, N is a value to be negotiated across five arms rather than
   inherited from one.
3. **The matching target** — burst alone, burst in a 1024-token sequence, or
   full-batch delta. Analysed in `docs/v4-gap-analysis.md`; not decided.
4. **The definition of "anisotropy."** Per-tensor norm spread, participation
   ratio, and gradient-outer-product eigenspectrum are all defensible and
   imply very different implementations and costs.
5. **Where fluent-true comes from.** It cannot be generated by any code in
   this repository. It is an authoring task, and it is an authoring task with
   a numerical target attached.
6. **AdamW β₁, β₂, ε and the gradient-clipping policy.** None appear in any
   config or any line of code. Clipping in particular is now blocking rather
   than cosmetic: it rescales gradient magnitude and, if applied per-tensor,
   changes direction — so an undeclared clipping policy corrupts both the
   matching criterion and the cosine/anisotropy measurements this study
   exists to record.
