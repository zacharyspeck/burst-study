# Step 9: symmetry canonicalization — what it is and where it stands

For Asa and for future-me. Written 2026-08-02. Two pages, plain English.
The measurements are in `docs/measurements/9-canonicalization-error.{json,md}`;
the reasoning is in `implementation-notes.md` under D17–D19 and S42–S60.

---

## What problem this solves

Two neural networks can compute **exactly the same function** and still look
far apart if you subtract their weights. Swap attention head 3 with head 7 —
consistently, in the query, key, value and output weights — and the model emits
byte-for-byte identical logits. Nothing was learned or lost. But the weight
vector moved a long way, and a naive distance would report that movement as if
it meant something.

The study's headline metric is a weight-space distance between a burst arm and
its seed-matched twin. Step 9 is the ruler that removes the bookkeeping so that
distance reflects a real difference. **If it is subtly wrong it will not crash
— it will produce plausible numbers forever.** Most of the design below exists
because of that.

## What a "symmetry" is, and which ones GPT-2 actually has

A symmetry is a relabelling that leaves the output unchanged. Whether a given
one is valid depends on the architecture, and the answer for GPT-2 is *not* the
answer published for architectures using RMSNorm and ReLU. Ten candidates were
applied to a real model and measured. **Seven are symmetries, three are not.**

The criterion was not a tolerance. It was whether the error **collapses** when
you redo the arithmetic in float64: exact maths shows float32-epsilon error in
float32 and ~1e-15 in float64, wrong maths shows a large error that does not
shrink at all. The two groups separated by eight orders of magnitude with zero
overlap — 3.9e+08 to 8.3e+08 for the survivors, exactly 1.00 for the drops.

**Dropped, with the reason:**

| candidate | why it fails |
| --- | --- |
| residual rotation | LayerNorm's per-channel gain. Curable at `ln_1`/`ln_2` by absorbing the gain; incurable at `ln_f`, because absorbing that one means folding it into `lm_head`, which is the tied embedding. |
| residual scaling | Embedding tying. Scaling needs `c` going in and `1/c` coming out; a tied projection supplies `c` both times, so the logits come out scaled. |
| FFN scaling | GELU is not positively homogeneous. This *is* a symmetry for ReLU; `GELU(sz) ≠ s·GELU(z)` for any `s ≠ 1`. |

One rule worth carrying: **tying makes the output projection `Wᵀ`, which equals
`W⁻¹` exactly when `W` is orthogonal.** Permutations and rotations are
orthogonal, so tying is harmless to them. Scaling is not, so tying is fatal.

Two symmetries were found that nobody had listed: the attention **key bias** is
pure gauge (softmax absorbs a per-query constant), and the **value bias** is
gauge up to a compensation in `c_proj.bias`. Both are removed.

## What the recipe does

Six steps, in an order that is **part of the definition** — five of six
permuted orders break it, and the sixth is documented with the reason it
commutes:

1. absorb LayerNorm gains at `ln_1`/`ln_2` (not `ln_f` — tied embedding)
2. zero the key-bias gauge
3. zero the value-bias gauge, compensating in `c_proj.bias`
4. canonicalize each head's GL(head_dim) freedom through its affine invariant
5. sort heads
6. **match** FFN neurons to a reference (not sort them — see below)

**Canonical form is pairwise-relative.** Step 6 matches against a reference, so
a model's canonical form is defined *against another model*. For this study
that reference is the seed-matched twin, which is the only thing any comparison
is made against anyway:

```python
canonicalize(twin)                     # twin defines the frame
canonicalize(arm, reference=twin)      # arm measured against it
```

Residual channel permutation is a real symmetry and is **deliberately not**
quotiented — same-seed twins do not spontaneously permute channels, so it buys
nothing here. Two independently-initialised models would still differ by it.

## Why sorting was replaced by matching

Sorting 3072 FFN neurons on a scalar key gives 36,852 adjacent pairs whose
smallest deciding margin is `1.5e-08`. When a perturbation flips one pair, two
neurons exchange their full 1,537-coordinate vectors — an O(1) error against an
O(ε) real difference. Measured inflation at ten seeds: **median 560,593× at ε=1e-8**, with a per-seed range of `[3.3, 808,845]` — which is itself the point.

The deciding reason was not the size but the **inconsistency**: on identical
seeds it measured 808,845 in one setting and 3.05 in another, because whether a
near-tie flips depends on where the perturbation lands. An inflation that
appears on some seeds and not others can be neither corrected for nor reliably
noticed.

Matching on each neuron's whole feature vector, solved with the Hungarian
algorithm, resolves near-ties by everything else about the neuron. It scores
identically to having no permutation step when none is needed, and recovers a
genuine permutation that dropping the step would miss by 6.9e+07.

## Open questions — read these before trusting a number

**1. The distortion factor is a range, not a number.** Even with the FFN sort
gone, the ruler is not distance-neutral. Which end applies depends on how far a
burst arm actually sits from its twin as a fraction of the parameter norm — and
that quantity does not exist until models are trained.

**2. The head-internal step has the same erratic property that retired the
sort.** Measured at ten seeds: `3.26` at ε=1e-8 but `1929.3` at ε=1e-6, with a
per-seed spread of `[2.83, 2450.5]`. Attribution run at four seeds: `3.07` /
`2189.9` / `237.8`. Remove that one step and the ruler is
`0.907` at *every* epsilon and *every* seed. Cause: the minimum singular gap on
GPT-2 is `5.5e-06`, so a perturbation of that order **rotates** the SVD basis
in the near-degenerate subspace. This is logged in
`docs/decisions-pending.md` as **D-1** and is not decided. **The step is still
in the recipe.**

**3. Step 9 cannot currently be applied to the study's own model.**
`probes/determinism/model.py` uses `nn.Linear`; step 9 is written entirely
against `transformers` `Conv1D`, whose weight layout is the transpose. Every
slice would address the wrong axis — and would not crash. The tripwire refuses
such a model loudly and names the transposition. No adapter has been built,
deliberately: which layout the study trains is undecided.

**4. Everything is measured on public GPT-2, which is the wrong model.** The
study injects at step 200, ~52M tokens into a from-scratch run, where LayerNorm
gains have barely moved from exactly 1.0 and biases from exactly 0.0. That is
verifiable now rather than assumed: `probes/determinism/model.py`'s init
produces exactly that. The dispersion sweep measures how the diagnostics change
across that range, and the FFN margin is ~2 orders of magnitude **smaller** at
the initialization end — i.e. the fragility is worse where the study works.

## The thing that went wrong three times

Three separate times, a quantity was **named** for what it should measure and
**computed** as something else. Each time the number was plausible, nothing
crashed, and the wrong number was then used to argue something was safe:

- the head sort key, documented as a spectrum, computed without the bias row —
  not even monotonic
- the FFN sort margin, documented as the deciding margin, computed as the
  largest difference anywhere in the key tuple — **59,000× too optimistic**,
  and that figure appeared in a report as evidence the sort was fine
- the test fixture, documented as generic, sitting at exactly the structured
  values a fresh init produces — which made an entire amendment a no-op that
  still passed every test

None was caught by the test meant to cover it. All three were caught by
measuring something adjacent and being surprised.
`tests/test_canonicalize_diagnostics.py` is now the guard: every `CanonReport`
field must be recomputable by an independent route or explicitly exempt with a
reason, and margins are tested as **thresholds** — move a key by 0.4× the
margin and the order must hold, by 2× and it must flip.

**If you extend step 9, treat a diagnostic that suddenly looks comfortable as a
reason to check the diagnostic.**

## What must be re-verified against a real checkpoint

- the whole epsilon sweep, at an epsilon calibrated to the real twin-vs-twin
  distance rather than to nothing
- the conditioning numbers, at step 200 rather than on a fully-trained model
- whether a genuine FFN or head permutation ever arises between same-seed twins
  — the zero-gradient argument says no, but it is an argument
- D-1, once there is a real model to measure the head-internal step's
  instability on

## Where things are

```
scripts/canonicalize.py                    the module
scripts/canonicalization_error.py          the measurements
tests/test_canonicalize.py                 tripwire, layout, equivalence, ordering
tests/test_canonicalize_recipe.py          round trip, composition, recipe order
tests/test_canonicalize_mutations.py       six injected faults
tests/test_canonicalize_diagnostics.py     the S55 guard
docs/measurements/9-canonicalization-error.{json,md}
docs/decisions-pending.md                  D-1, open
```
