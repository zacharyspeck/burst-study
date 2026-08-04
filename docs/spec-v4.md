# Spec v4 — mid-pretraining burst study

**Status: current design. Supersedes v3.**
Recorded 2026-08-01. Written down from a design discussion; there is no
upstream document this was copied from, and this file is now the
authoritative statement of the design.

**RECONCILED 2026-08-03.** `burst/`, `configs/`, `scripts/` and `tests/` now
implement this file. `ARMS` is the seven run types below, `configs/runs/` holds
70 generated override files, and the run count is computed rather than typed.
This paragraph previously said the code still implemented v3 and that nothing
had been changed to match; that stopped being true at commit `c2df6c7`.

`docs/v4-gap-analysis.md` describes the gap as it stood before that and is kept
as a record of what the move cost, not as a description of the present.

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

**An exploratory six-way categorical comparison, reported as a leverage
ordering — not a binary contrast.** The question is no longer "does coherent
beat scrambled"; it is how six categories of injected text rank in the
displacement they produce.

Six rather than the five this section originally said: the scrambled family
splits into `scrambled-false` and `scrambled-true`, matched scrambles of the
two fluent arms, so that truth value crosses with structure rather than being
confounded with it.

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

All six injected arms must be matched **to each other**:

- on **per-token loss**, within tolerance
- **and** on **gradient norm**, within tolerance
- at the **same token length**
- seeded and reproducible

Twin has no burst and therefore no loss or gradient norm to match. **The
matching problem is six-way; the run structure is seven-way.** Anything that
conflates those two counts will be wrong somewhere — and the counts moved on
2026-08-03, from five and six, when `scrambled-corpus` was cut and the
scrambled family split in two.

`burst/config.py` expresses both: `INJECTING_ARMS` has six entries and `ARMS`
has seven, with the second derived from the first rather than written out.

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

> **NOW IN THE CONFIG, as of 2026-08-03.** `configs/base.yaml` has
> `injection.injection_step: 200`, `injection.burst_length_tokens: 194` and
> `injection.burst_position: 400`, and the six `burst_text_paths` point at the
> committed texts in `bursts/`.
>
> This note previously said the opposite — that the config had `null` and the
> two disagreed. It was written earlier the same day and step 14 settled the
> values hours later, which is exactly the kind of note that goes stale fastest
> and is corrected here rather than left.
>
> `burst_position` was added to the config at the same time. It had lived as
> `POSITION = 400` in three separate scripts, and the injection hook would have
> been a fourth copy.

Two interactions with values already in `configs/base.yaml`, noted because
neither is recorded anywhere as intentional:

- `learning_rate.warmup_steps` is **200**. Step 200 is therefore the first
  step at peak learning rate (0.0006). The burst lands exactly on the
  warmup/cosine boundary.
- `checkpointing.weights_only_interval` is **50**, and a checkpoint fires at
  0-indexed step `s` when `(s + 1) % 50 == 0`. There is therefore a
  weights-only checkpoint at **step 199**, immediately before injection. This
  is convenient and appears to be coincidence rather than design.

  It has since become load-bearing whether or not it was intended: the step-199
  file holds the state after steps 0..199 and before step 200's update, so
  "identical through 199, different from 200" is a checkable claim rather than
  a described one. `tests/test_injection.py` asserts exactly that, by training
  the same seed twice and comparing SHA-256 over raw tensor bytes.

Nothing else about the run config changes from v3.

## Metrics

**This document does not specify the study's metrics, and never has.** Grep for
`metric` in this file returns nothing. That is recorded here rather than left
implicit, because the metrics exist in code and a reader could reasonably
assume the spec had chosen among them.

What exists, in `scripts/metrics.py` (step 10, first half):

| metric | status |
| --- | --- |
| interpolation barrier | built |
| raw L2 weight distance | built |
| activation similarity (cosine) | built |
| per-layer CKA | built |
| permutation-aligned barrier | **not built** — raises, needs `canonicalize` |
| permutation-aligned L2 | **not built** — raises |
| RSF subspace probe | **not built** — raises |

**Which of these is the headline is not decided anywhere**, and this document
is not deciding it. See D-7 in `docs/decisions-pending.md`, which also records
what step 9 measured about canonicalization: against an EMPTY recipe scoring
exactly 1.00000, the shipped four-step recipe scores 1.00041 — so alignment
contributes very little between same-seed twins, which bears directly on
whether an aligned metric should be primary.

## Deliberately still open

These are unresolved as of this writing and are **not** to be settled by
implementation default:

1. **The matching tolerance.** Not set.
2. **The token length N.** ~~Enforced by nothing~~ — **now set to 194 in
   `configs/base.yaml` and enforced**: `scripts/injection.py` refuses a burst
   text that does not tokenize to exactly `burst_length_tokens`, and every arm
   in `bursts/provenance.json` records 194.

   What remains open is not the value but its **basis**: 194 was inherited from
   the byte content of a passage written for v3, and under v4 it is a number
   that should be negotiated across six arms rather than inherited from one.
   Setting it in the config records the value; it does not justify it.
3. **The matching target** — burst alone, burst in a 1024-token sequence, or
   full-batch delta. Analysed in `docs/v4-gap-analysis.md`; not decided.
4. **The definition of "anisotropy."** Per-tensor norm spread, participation
   ratio, and gradient-outer-product eigenspectrum are all defensible and
   imply very different implementations and costs.
5. ~~**Where fluent-true comes from.**~~ **DONE.** It was authored by hand and
   is committed at `bursts/fluent_true.txt`, 194 tokens, recorded in
   `bursts/provenance.json` with `generator: hand-written` and
   `source: hand-written, fixed, not generated`. It still cannot be generated
   by any code here, which was the point — the entry is kept rather than
   deleted so the fact that it was an authoring task stays on the record.
6. ~~**AdamW β₁, β₂, ε and the gradient-clipping policy.** None appear in any
   config or any line of code.~~ **THAT IS NO LONGER TRUE and was the most
   stale claim in this document.** `configs/base.yaml` declares
   `beta1: 0.9`, `beta2: 0.95`, `eps: 0.00000001` and `grad_clip: 1.0`, all
   decided, and `scripts/train.py` reads every one of them.

   The direction concern is also answered by the code: clipping is
   `torch.nn.utils.clip_grad_norm_` over `model.parameters()`, which is a
   **global**-norm clip applied **after full gradient accumulation**, not
   per-tensor. So it rescales magnitude without changing direction, and a test
   asserts the ordering by reading the source. That removes the corruption of
   the cosine and anisotropy measurements this entry warned about.

   **What remains open here is one thing, and it is new:**
   `optimizer.adamw_impl` is `null` and launch-blocking. `foreach`, `fused` and
   `single` group their arithmetic differently and produce different bits from
   identical moments, so it is part of the study's definition rather than a
   performance knob. The pilot settles it. See S78.

   Also still open, and **not** settled by implementation default: whether
   clipping actually engages at step 200. `scripts/train.py` logs the pre-clip
   gradient norm every step so the question is answerable from a real run, but
   no real run exists, so it is instrumented rather than discharged.
