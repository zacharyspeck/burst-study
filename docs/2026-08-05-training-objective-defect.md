# The pilot trained the wrong objective: a two-position label shift

**Date:** 2026-08-05. **Found at commit:** `f015fd0` (clean tree).
**Affects:** all four pilot runs of SLURM 54652, i.e. every training run this
repository has ever produced.
**Does not affect:** any measurement taken against public GPT-2, any determinism
result, any hardware-sizing result.

---

## 0. The defect in one paragraph

`scripts/train.py` shifts each raw token block into an `(inputs, targets)` pair
and then hands that already-shifted pair to `model_seam.compute_loss`, whose
HuggingFace branch passes it as `model(input_ids=inputs, labels=targets)`.
`transformers` shifts `labels` internally. The two shifts compose. The logit at
position *i*, which is conditioned on tokens `t_0 … t_i`, is therefore scored
against `t_(i+2)` rather than `t_(i+1)`. Every pilot run was optimised to
predict the token **two** positions ahead, skipping the immediate next token.
The training-loss curve is monotone and healthy-looking throughout, because the
model genuinely learns — the wrong task.

---

## 1. How it was found

Recorded step by step, including the steps that ruled things *out*, because the
order matters: the first three checks all had innocent explanations available
and the defect only became visible once they were eliminated.

### 1.1 The trigger: an endpoint loss that was too high

`docs/measurements/2026-08-05-pilot-results.md` reports the barrier evaluation's
endpoint losses on `bursts/context.txt`:

| run | loss |
|---|---|
| twin seed 0 | 7.339636 |
| twin seed 1 | 7.407107 |
| twin seed 2 | 7.426399 |
| random-chars seed 0 | 7.397346 |

`bursts/context.txt` is ordinary English prose — a passage about the topography
of Kansas, corpus document 73 (D12). A loss of 7.34 nats on ordinary English is
a perplexity of about 1,540. For a GPT-2 Base trained on 2.5B tokens the
expected figure is somewhere near 3.4–3.7.

That alone is not proof of anything: a single 1024-token passage can be
atypical, and the eval batch is deliberately one committed passage.

What made it worth chasing was the comparison against the run's own training
log. `/shared/27as66/burst-pilot/logs/seed00_twin.log`:

```
  step  9535  lr 0.00006000  loss 4.6678752452  gnorm 0.2488769740
```

Final **training** loss 4.67, final **evaluation** loss 7.34, on the same model.
The evaluation is 2.67 nats worse than the training loss.

This is the load-bearing observation, and it is worth stating precisely why.
The corpus is sized for **exactly one epoch**:

```
shards: 149
total train tokens: 2,499,805,184
tokens needed for 9536 steps x 256 x 1024 = 2,499,805,184
ratio (epochs over corpus): 1.0
```

Under single-epoch training every batch is fresh data at the moment its loss is
computed. The training loss *is* an unbiased held-out estimate. A train/eval gap
of 2.67 nats cannot be a generalisation gap, because there is no second pass in
which to memorise anything. Two numbers that must agree did not agree, so at
least one of the two computations was wrong.

### 1.2 Ruled out: an atypical evaluation passage

Hypothesis: `bursts/context.txt` is simply hard, and the barrier evaluation is
fine.

Test: score the same checkpoint (`seed00_twin/step009535_full.pt`) on random
1024-token windows of `/shared/27as66/corpus/heldout.bin`, through the *same*
code path (`metrics.cross_entropy_loss`).

```
context.txt (n=1024 tokens)   loss = 7.339636   ppl = 1540.2
heldout.bin: 10,485,760 tokens
  heldout window 0 @8,918,569   loss = 6.901756   ppl = 994.0
  heldout window 1 @6,678,374   loss = 7.032629   ppl = 1133.0
  heldout window 2 @5,359,130   loss = 6.948201   ppl = 1041.3
  heldout window 3 @2,828,642   loss = 7.339389   ppl = 1539.8
  heldout window 4 @3,227,509   loss = 9.496756   ppl = 13316.5
  heldout window 5 @  429,596   loss = 6.741133   ppl = 846.5
  heldout window 6 @  788,872   loss = 8.777407   ppl = 6486.0
  heldout window 7 @  173,287   loss = 7.322784   ppl = 1514.4

heldout mean loss = 7.570007  sd = 1.006655
context.txt - heldout mean = -0.230371 nats
```

`context.txt` is **easier** than the held-out average, not harder. The
hypothesis is dead: the gap is systematic, not a property of the eval passage.

A side observation worth recording: the barrier evaluation uses a single
committed 1024-token passage even though a 10,485,760-token held-out slice
exists at `/shared/27as66/corpus/heldout.bin`. The pilot doc already flags the
single-passage choice as a limitation. The sd of 1.007 across held-out windows
quantifies it — window-to-window variance dwarfs the 0.0577 endpoint-loss delta
the pilot proposed as its surviving signal.

### 1.3 Ruled out: a corrupt or degenerate corpus

Hypothesis: the corpus contains degenerate content (padding runs, repeated
documents, a tokenizer mismatch) that makes training loss artificially low.

Test: token statistics over the first 2M tokens of two training shards and the
held-out shard.

```
   train-000.bin  n=  16,777,216  uniq(2M)=43,720  EOT%=0.091  top=[(11, 73633), (262, 72615), (198, 71896), (13, 70801), (284, 38288), (286, 37129)]
   train-074.bin  n=  16,777,216  uniq(2M)=43,522  EOT%=0.091  top=[(11, 74016), (262, 72484), (198, 71384), (13, 71326), (284, 38828), (286, 36985)]
     heldout.bin  n=  10,485,760  uniq(2M)=43,471  EOT%=0.088  top=[(11, 74421), (262, 72493), (13, 71378), (198, 68198), (284, 37863), (286, 37396)]
```

Token 11 is `,`, 262 is ` the`, 198 is `\n`, 13 is `.`, 284 is ` to`, 286 is
` of`. Roughly 43.5k distinct token types per 2M tokens, EOT at ~0.09%, and the
train and held-out shards match each other closely. This is ordinary English.
Hypothesis dead.

### 1.4 The diagnostic that localised it: does eval loss track training at all?

If the checkpoints hold the weights they claim and both loss computations agree,
next-token loss should fall steadily across training. Scored six checkpoints of
`seed00_twin` on four fixed probes:

```
checkpoint                         context.txt    heldout@429596       TRAIN-000@0 TRAIN-000@8000000
step000199_weights_only.pt              7.6363            7.5819            7.4716            7.3596
step000999_full.pt                      7.2976            7.0107            6.8680            7.2204
step002999_full.pt                      7.3000            6.8348            7.1417            7.3921
step005999_full.pt                      7.3090            6.7332            7.0313            7.3671
step008999_full.pt                      7.3676            6.7555            7.0432            7.3398
step009535_full.pt                      7.3396            6.7411            7.0166            7.3320
```

Against the same run's training log:

```
step     0  lr 0.00000300  loss 10.9607644081  gnorm 5.4319462776
step   199  lr 0.00060000  loss  6.8388155848  gnorm 0.3920728862
step   999  lr 0.00059030  loss  5.7304725200  gnorm 0.3484558165
step  2999  lr 0.00048883  loss  4.9673393071  gnorm 0.2857984602
step  5999  lr 0.00022970  loss  4.7894159704  gnorm 0.2502665818
step  8999  lr 0.00006440  loss  4.6779832542  gnorm 0.2582874894
step  9535  lr 0.00006000  loss  4.6678752452  gnorm 0.2488769740
```

Two curves on the same weights, moving differently. Training loss falls
monotonically from 10.96 to 4.67 and is still falling at the end. Next-token
loss falls from 7.58 to about 6.74 by step 3000 and then **stops**, drifting
between 6.73 and 6.76 for the remaining 6,500 steps while training loss
continues down by another 0.30.

The checkpoints are not stale — the weights change, and early training does
improve next-token loss, because a model learning *any* local statistics of
English improves it incidentally. But from step ~3000 onward the optimiser is
buying progress on something that is not next-token prediction.

At this point the defect had to be in one of the two loss computations, and the
divergence pattern pointed at the objective rather than at the data or the
checkpoint machinery.

### 1.5 Reading the training loss path

`scripts/train.py`, the per-micro-batch body:

```
scripts/train.py:564            inputs, targets = reader.shift(raw)
scripts/train.py:565            inputs = inputs.to(device)
scripts/train.py:566            targets = targets.to(device)
...
scripts/train.py:580                loss = SEAM.compute_loss(model, inputs, targets) / accum
```

`reader.shift`, `scripts/train.py:237-244`:

```python
@staticmethod
def shift(rows):
    """(inputs, targets) from raw rows. Targets are inputs shifted by one.
    ...
    """
    return rows[:, :-1].contiguous(), rows[:, 1:].contiguous()
```

`model_seam.compute_loss`, `scripts/model_seam.py:202-217`:

```python
def compute_loss(model, inputs, targets):
    """A scalar loss, whichever forward signature the family has.

    The ONE place the two families differ. The probe model returns a loss
    directly; HF returns an object carrying `.loss`. Both are next-token
    cross-entropy over the same shift, so the number means the same thing.
    """
    out = model(input_ids=inputs, labels=targets) if _is_hf(model) \
        else model(inputs, targets)
```

The docstring's final clause — *"Both are next-token cross-entropy over the same
shift"* — is the false premise. They are not over the same shift. The probe
branch consumes a pre-shifted pair; the HF branch consumes unshifted ids plus
`labels` and does its own shift.

`transformers`, `models/gpt2/modeling_gpt2.py:830-831`:

```python
shift_logits = lm_logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()
```

### 1.6 The index arithmetic

Let `raw` be one row of 1024 tokens, `t_0 … t_1023`.

| quantity | value | length |
|---|---|---|
| `inputs = raw[:, :-1]` | `t_0 … t_1022` | 1023 |
| `targets = raw[:, 1:]` | `t_1 … t_1023` | 1023 |
| `logits = model(inputs)` | `logits[i]` conditioned on `t_0 … t_i` | 1023 |
| `shift_logits = logits[..., :-1, :]` | positions `0 … 1021` | 1022 |
| `shift_labels = targets[..., 1:]` | `t_2 … t_1023` | 1022 |

Pairing element *i* of each: `logits[i]`, conditioned on `t_0 … t_i`, is scored
against `t_(i+2)`.

The correct pairing is `logits[i]` against `t_(i+1)`. The objective is off by
exactly one position, and the model never receives gradient on next-token
prediction at all.

Note also that a 1024-token row now yields 1022 predictions, not the 1023 that
`shift`'s own docstring promises.

### 1.7 The decisive experiment

Reading code is not proof. The claim was tested directly: on one checkpoint and
one set of tokens, compute three quantities and see which two coincide.

- **(a)** proper next-token loss: `logits[:, :-1]` against `raw[:, 1:]`,
  computed on the full unshifted row
- **(b)** exactly what `train.py` does: `SEAM.compute_loss(model, raw[:, :-1], raw[:, 1:])`
- **(c)** explicit two-ahead loss: `logits[:, :-1]` of the *shortened* input,
  against `raw[:, 2:]`

If the defect is real, (b) and (c) are the same number and (a) is different.

```
off=4,961,258   next-token 6.5810   train.py 4.5024   2-ahead 4.5025
off=5,366,314   next-token 7.3296   train.py 4.1817   2-ahead 4.1817
off=7,917,731   next-token 6.7878   train.py 3.9785   2-ahead 3.9785
off=9,965,359   next-token 7.0174   train.py 4.6644   2-ahead 4.6644
off=  365,419   next-token 7.7989   train.py 3.6637   2-ahead 3.6637
off=1,511,475   next-token 7.1365   train.py 4.5773   2-ahead 4.5773

(a) proper next-token loss      mean = 7.1085
(b) train.py's own computation  mean = 4.2613
(c) explicit predict-2-ahead    mean = 4.2614

training log at step 9535       = 4.6679
|b - c| max = 5.33e-06   <- 0 means train.py == 2-ahead
```

`train.py`'s loss and the explicit two-ahead loss agree to 5.33e-6 across all
six windows. That residual is the fp32-vs-fp64 difference between the two
routes, not a semantic difference: (b) runs through `transformers`' own
`float()` upcast, (c) casts to `double()`.

The training log's 4.6679 is a training batch and (b)'s 4.2613 is held-out, so
they are not required to be identical — but they sit in the same range, which
is the expected relationship for a single-epoch run measured on its own
objective.

### 1.8 Validating the instrument

Before blaming the models, the measurement had to be cleared. If
`metrics.cross_entropy_loss` were itself wrong, the 7.11 would be an artifact.

Test: score public GPT-2 on the same held-out windows through the same code
path.

```
public GPT-2 next-token loss on this heldout set: mean = 2.7757
  windows: 2.843 2.614 2.626 3.247 2.236 3.088
```

2.78 nats on OpenWebText is the right order for released GPT-2. The evaluation
path is correct. The 7.11 is a property of the pilot's models.

This also supplies the calibration target for a corrected run. A from-scratch
124M model at 2.5B tokens is far more lightly trained than released GPT-2
(~40B tokens), so it should land above 2.78 — expect roughly 3.4–3.8, and
treat anything near 7 as the defect still present.

---

## 2. Why the defect survived every guard

### 2.1 The training curve looks correct

10.96 → 6.84 → 5.73 → 4.97 → 4.79 → 4.68 → 4.67, monotone, with gradient norms
falling 5.43 → 0.25 and a clean warmup/cosine shape. Nothing in the log invites
suspicion. Two-ahead prediction is a harder task than next-token prediction, so
its loss floor is higher, and 4.67 sits close enough to a plausible next-token
value for a lightly-trained 124M model that the number does not announce itself
as wrong. Only the *comparison* against a correctly-computed evaluation exposes
it.

### 2.2 The seam serves two families with opposite conventions

`compute_loss` is the single point where `probe_linear` and `hf_gpt2` differ.
The probe family's forward is `model(idx, targets) -> loss` and takes a
**pre-shifted** pair. The HF family's is
`model(input_ids=..., labels=...) -> out.loss` and takes **unshifted** ids.
`train.py` was written to the probe's convention — it shifts before calling —
and the HF branch was expressed as `labels=`, which needs the opposite input.
Both branches are individually reasonable; the pairing is what fails.

### 2.3 Every other loss path in the repo is correct — and three of them say why

This is the part worth dwelling on. The trap was known.

`scripts/burst_match.py:715-733`, `per_token_losses`:

> Computed here rather than by passing `labels=` to the model, for two
> reasons: it yields the per-token vector directly (the model would only
> hand back the mean), and **it pins the definition of "loss" inside this repo
> instead of inheriting whatever a given transformers version does with
> label shifting.**

`probes/determinism/hf_model.py:42-49`, `HFGPT2`:

> The loss is computed here rather than by passing `labels=`, for two
> reasons. **transformers shifts labels internally, and `SyntheticCorpus`
> already returns an offset pair, so passing `labels` would shift twice.**
> And computing it here means the stand-in and the real model are measured
> through identical loss code -- so any difference between the two results
> is the model, not the objective.

That second comment describes this defect exactly, in advance, in the probe that
was built to de-risk the training loop — and the probe's adapter is precisely
the shape `model_seam.compute_loss`'s HF branch should have had.

`scripts/metrics.py:364-383` (`cross_entropy_loss`) and
`scripts/injection.py:359-361` (`burst_region_losses`) both shift manually and
correctly.

So four loss computations in this repository handle the shift correctly, two of
them with comments naming the hazard, and the fifth — the only one that decides
what the models become — does not.

### 2.4 No test pins the objective

`tests/test_model_seam.py:200` is
`test_compute_loss_returns_a_scalar_for_either_signature`; `:220` checks a float
comes back. The suite asserts the *type* of the return value and never its
semantics. A test that scored a known input against a hand-computed
cross-entropy, or that asserted `compute_loss` agrees with
`metrics.cross_entropy_loss` on the same tokens, would have caught this.

This is the S55 pattern — "metrics that are not the quantity they are named
after" — at the level of the training objective rather than a diagnostic.

---

## 3. What is invalidated

Everything downstream of the pilot's weights.

- **All four barrier curves** in `docs/measurements/2026-08-05-pilot-barriers/`.
  The alpha grids, chords and excesses are arithmetically correct; they describe
  models that were not trained on the study's objective.
- **The headline result** — arm-vs-twin `max_excess = 0.000000`,
  twin-vs-twin 2.591210 / 3.073542 / 4.217713 — as evidence about a burst.
- **The endpoint-loss signal**: delta = +0.057710, sd = 0.045556,
  sigma = 0.064426, sigma/delta = 1.116, n ≈ 9.8 seeds. The power estimate that
  followed from it carries no weight.
- **44.69 GPU-hours** and **392 GB** of checkpoints at
  `/shared/27as66/burst-pilot/runs/`.

### 3.1 A second-order consequence: the arms are matched to a different quantity

`burst_match` computes gradient norms from `per_token_losses`, i.e. from the
**next-token** objective (§2.3). The arms were tuned to a ±17% band on that
quantity (`docs/measurements/8b-iii-tuning-trace.md`, k=3, seven arms in band).

The pilot then trained on the two-ahead objective. The gradients that actually
moved the weights are not the gradients the arms were matched on. Under the runs
as trained, spec §10 manipulation check 2 — *"the arms were matched"* — does not
hold, independently of the ±17% band question (S37).

The fix restores the correspondence: once training uses next-token loss, the
matching quantity and the training quantity are the same again.

---

## 4. What survives

Stated explicitly, because the temptation after a finding like this is to
distrust everything.

- **Every measurement taken against public GPT-2.** Step 8 matching
  (`8b-i` … `8b-iv`), gradient direction, and step 9 canonicalization
  (`9-canonicalization-error.*`) never ran `train.py` and never touched a pilot
  checkpoint. The gauge findings (D17, D18, D19, S66) are unaffected.
- **All determinism results.** Bit-identity is a property of the computation
  being repeated, not of whether that computation is the intended one. The
  A6000 and A100X records stand, as does the pilot's own de-facto demonstration
  that `seed00_twin` and `seed00_random-chars` are byte-identical through step
  199 (model-tensor hash `886db82a…5b6ae9`) across two physical cards.
- **The injection hook.** That the burst entered the batch at step 200 and
  nowhere else is a fact about tensor plumbing. `injection_fired` matching
  `injection_plan`, divergence at step 200 and not 199, and burst-region mean
  loss 7.3066 over 194 predictions from index 399 — all still hold. Note that
  `burst_region_losses` uses the *correct* shift, so 7.3066 is a next-token
  number measured on a model being trained on a different objective.
- **Hardware sizing and the bf16 result.** 11.17 h/run mean, 3.66× over fp32,
  micro_batch 8 at 15.8 GiB. Throughput does not depend on the objective.
- **The pre-registration**, including the §8.4 branch to `plain_loss_barrier`.
  The ruler measurement was taken on the step-199 checkpoint, which is
  bit-identical across all seven arms and structurally cannot carry outcome
  information. The branch was recorded before any displacement was examined
  (§8.5) and remains valid. It will need re-running on corrected checkpoints,
  but the *rule* is untouched.

---

## 5. The structural problem the fix does not solve

Separate from the defect, and not repaired by fixing it.

The pilot's zero barrier is **also** what the design predicts. An arm and its
twin share an initialisation, share a data order, and differ by one 194-token
burst at step 200 of 9,536. Two runs branching from a common early prefix are
linearly mode connected — this is Frankle et al.'s instability analysis — so the
interpolation path between them stays inside one basin and no barrier forms.
The observed curve is the signature: a smooth sag to −0.100 at alpha 0.45–0.50
and back, never rising above the chord.

Twin-vs-twin pairs across seeds are independently initialised, sit in different
basins, and produce the 2.59–4.22 peaks at alpha 0.50.

So the noise floor and the displacement measure different phenomena, and this is
a property of the experimental design rather than of the defect. **A corrected
re-run will floor again.** The pilot doc's own sentence stands unchanged:

> sigma = 3.29 against delta = 0 is not a small effect; it is a metric that
> cannot express this effect at all.

(One correction to that sentence's arithmetic, noted in passing: 3.29 is the
*mean* of the three twin barriers, not their standard deviation, which is
0.8354. The floor `scripts/analysis.py` would actually apply is the max, 4.2177.)

The readout question therefore has to be settled before a re-run is launched,
not after. Spending ~1,300 GPU-hours to reproduce a structural zero would be the
expensive way to learn this twice.

---

## 6. The fix

The HF branch of `compute_loss` should compute the loss itself, over the same
manual shift the rest of the repo uses, rather than delegating to `labels=`.
That matches `probes/determinism/hf_model.py`'s adapter, honours
`burst_match.py`'s stated principle of pinning the definition of loss inside the
repo, and keeps the `(inputs, targets)` calling convention that `train.py`,
`test_train.py` and the probe family all already assume.

The alternative — stop pre-shifting and pass raw rows with `labels=` — would
work arithmetically but makes the loop's behaviour depend on a `transformers`
implementation detail, changes the calling convention for the probe family, and
is the option the repo has twice written comments against.

**This document does not apply either fix.** Changing the training objective
changes the experiment, in the sense of `train.py:568-577`'s note about the
normalisation line, and the choice belongs to Asa and Zach.

Whichever route is taken, the regression test should assert *semantics*, not
type: that `SEAM.compute_loss(model, inputs, targets)` equals
`metrics.cross_entropy_loss` on the same tokens, for the `hf_gpt2` family. That
is the assertion whose absence let this through.

Re-verification needed after the fix, in order:

1. The semantic test above, in both environments.
2. Determinism at bf16 through `train.py` — the digests will change, so the
   existing records do not transfer.
3. Re-run the §8.4 ruler on a corrected step-199 checkpoint. The branch is
   expected to come out the same but the measurement is cheap and the
   pre-registration ties the headline to it.
4. Re-verify the arm match at step 200 against a corrected checkpoint — the
   item `implementation-notes.md:5477` already lists as outstanding, now with
   the additional reason that the matching quantity and the training quantity
   agree again.

---

## 7. Reproducing this

Both scripts below run against the pilot checkpoints at
`/shared/27as66/burst-pilot/runs/`. Use `.venv-ml`. Neither writes into a run
directory; `--cfg-scratch`-style scratch paths are outside the repo, and
`load_config` is called with `write_provenance=False` so nothing stamps a
provenance record for training that did not happen.

**The trajectory diagnostic (§1.4)** — load each checkpoint, score four fixed
probes with `metrics.cross_entropy_loss`, print the table.

**The decisive test (§1.7)**, which is the one that matters:

```python
import sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

REPO = Path("/shared/27as66/burst-study")
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO))
import model_seam as SEAM
from burst.config import load_config

cfg = load_config(REPO / "configs" / "base.yaml",
                  REPO / "configs" / "runs" / "seed00_twin.yaml",
                  outdir=Path("/tmp/scratch"), write_provenance=False,
                  family="hf_gpt2")
payload = torch.load(
    "/shared/27as66/burst-pilot/runs/seed00_twin/step009535_full.pt",
    map_location="cpu", weights_only=False)
model = SEAM.build_model(cfg, payload["family"])
model.load_state_dict(payload["model"]); model.eval()

held = np.memmap("/shared/27as66/corpus/heldout.bin", dtype=np.uint16, mode="r")
rng = np.random.default_rng(1)
for _ in range(6):
    off = int(rng.integers(0, len(held) - 1025))
    raw = torch.tensor(np.asarray(held[off:off+1024], dtype=np.int64))[None, :]
    with torch.no_grad():
        logits = model(input_ids=raw).logits
        a = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).double(),
                            raw[:, 1:].reshape(-1))                     # next-token
        inputs, targets = raw[:, :-1].contiguous(), raw[:, 1:].contiguous()
        b = SEAM.compute_loss(model, inputs, targets)                   # train.py
        lg = model(input_ids=inputs).logits
        c = F.cross_entropy(lg[:, :-1].reshape(-1, lg.size(-1)).double(),
                            raw[:, 2:].reshape(-1))                     # 2-ahead
    print(f"{off:>9,}  next-token {float(a):.4f}  train.py {float(b):.4f}  2-ahead {float(c):.4f}")
```

Expected on the current pilot checkpoints: `train.py` and `2-ahead` agree to
about 5e-6; `next-token` is roughly 2.8 nats worse. After the fix, `train.py`
should agree with `next-token` instead, and the absolute value should be near
3.4–3.8 rather than near 7.

The seed for the window offsets is fixed (`default_rng(1)`), so the six offsets
above reproduce exactly. `Date.now()`-style nondeterminism is absent by
construction.

---

## 8. Provenance of this document

Found and written 2026-08-05, working from a request to assess the project for a
NeurIPS 2026 workshop submission; the objective defect was not what was being
looked for. No repository file was modified during the investigation — every
script ran from a scratch directory outside the repo, and
`git status --porcelain` was empty at `f015fd0` before and after.

The numbers in §1 are transcribed verbatim from the runs described. Nothing here
was re-derived from memory.
