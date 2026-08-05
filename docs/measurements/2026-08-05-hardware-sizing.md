# Hardware sizing — what a run actually costs, on both card models

**Date:** 2026-08-05
**Code:** commit `1f8e183` (tree clean)
**Machines:** `gpmoo-a1` (NVIDIA A100X, 80 GB) and `gpmoo-b1` (NVIDIA RTX A6000,
48 GB), driver 560.35.05, torch 2.13.0+cu126, CUDA 12.6
**Config:** `configs/base.yaml` + `configs/runs/seed00_twin.yaml`, with
`micro_batch` varied through scratch copies and TF32 varied through a wrapper.
Nothing in the repo was modified to take these numbers.

## Why this was measured

`docs/handoff-pilot.md` §3 says sizing `micro_batch` on real hardware is the
pilot's job, and that the committed `8` is "a value the probe *invented*"
(S67). §0.C separately requires every run in a comparison to sit on **one card
model**, which forces a choice of node before any run starts. Neither question
had a number attached.

A first timing run answered the second question badly enough to be worth
checking: **42 hours per run**, against an off-the-cuff estimate of 3–4 hours
for GPT-2 124M on 2.5B tokens. The gap is real and it is not inefficiency.

## Method

Each configuration was run twice, at 8 steps and at 24 steps, on one card. Per-step
time is `(wall_24 - wall_8) / 16`, so process startup — model construction,
corpus mmap, first-step allocator warmup, measured at 14.6 s — **cancels
instead of being estimated**. Peak memory is the maximum of `nvidia-smi
--query-gpu=memory.used` sampled every 2 s.

Configurations were run four-at-a-time on four cards of one node. Host
contention could bias absolute numbers slightly; it applies equally to every
row within a node.

## Results

`micro_batch` 8 with TF32 off is the committed configuration and is the
baseline on both cards.

| device | micro_batch | cuBLAS TF32 | s/step | peak GiB | TFLOP/s | h/run |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| A100X | 32 | **on** | **4.963** | 59.8 | 39.4 | **13.1** |
| A100X | 8 | on | 5.474 | 15.8 | 35.8 | 14.5 |
| A100X | 32 | off | 14.457 | 59.8 | 13.5 | 38.3 |
| A100X | 8 | off | 15.453 | 15.8 | 12.7 | 40.9 |
| RTX A6000 | 16 | on | 9.755 | 30.3 | 20.1 | 25.8 |
| RTX A6000 | 8 | on | 10.264 | 15.6 | 19.1 | 27.2 |
| RTX A6000 | 16 | off | 14.841 | 30.3 | 13.2 | 39.3 |
| RTX A6000 | 8 | off | 15.124 | 15.6 | 12.9 | 40.1 |

`TFLOP/s` is the conventional `6ND` count: `6 × 124,439,808 × 262,144 =
1.957e14` FLOPs per optimizer step, divided by the measured step time.
`h/run` is `s/step × 9536 / 3600`, compute only — it excludes checkpoint I/O.

## Four findings

**1. The 42 hours is fp32 with TF32 disabled, not slow code.** At the committed
configuration the A100 sustains 12.7 TFLOP/s against its **19.5 TFLOPS**
non-tensor fp32 peak — **65% of peak**, which is good. The 3–4 hour estimate
assumes bf16 tensor cores at ~125 TFLOP/s effective, and the A100's bf16 peak
is 312 TFLOPS. The ratio of the ceilings is the whole discrepancy. `train.py`
implements fp32 only and refuses bf16, and disables TF32 at line 111 with the
reason recorded there:

> TF32 is deterministic but lower precision. Off so that a result here is
> comparable to one taken elsewhere, rather than depending on whether the card
> happened to have tensor cores enabled.

**2. At the committed configuration the two card models are a dead heat.**
15.12 s/step on the A6000 against 15.45 on the A100X — the A6000 is 2% *faster*.
This contradicts the prediction from datasheet fp32 peaks, where the A6000's
38.7 TFLOPS doubles the A100's 19.5 and should have made it roughly twice as
fast. Both cards instead land at ~12.8 TFLOP/s: 65% of peak for the A100, 33%
for the A6000. The likely reading is that the A6000 is limited by memory
bandwidth (768 GB/s against ~1935) rather than by arithmetic, but no roofline
measurement was taken and that is a plausible explanation rather than a
demonstrated one.

**3. TF32 is the only large lever, and it is worth much more on the A100.**

| card | TF32 speedup | why |
| --- | ---: | --- |
| A100X | **2.9×** | TF32 peak 156 TFLOPS, 8× the fp32 peak |
| RTX A6000 | 1.5× | TF32 peak 77.4 TFLOPS, 2× the fp32 peak |

Both cards reach ~25% of their TF32 peak, so the difference between them is the
ceiling, not the utilisation. Best measured configuration overall is A100X at
`micro_batch` 32 with TF32 on: **3.11× the committed configuration**.

**4. `micro_batch` is a small lever, and 8 leaves memory unused.** Raising it
from 8 to 32 buys 6% with TF32 off and 9% with TF32 on. At 65% of fp32 peak
there was never 10× available here. What it does establish is the memory
ceiling the pilot was asked for:

| micro_batch | peak | fits |
| ---: | ---: | --- |
| 8 | 15.8 GiB | both cards |
| 16 | 30.3 GiB | both cards |
| 32 | 59.8 GiB | **A100 80 GB only** — exceeds the A6000's 48 GiB |

So `micro_batch: 8` is safe everywhere and wastes about four fifths of an A100.

## TF32 is bitwise deterministic — verified, not assumed

The question gates whether TF32 is admissible at all, so it was measured rather
than reasoned from the comment in `train.py`. Two fresh processes, same seed,
same card, `micro_batch` 32, cuBLAS TF32 on, 8 steps:

```
R1  7e1d8ac5287bd94cfb7cea23c8ff40055258d6d7ca01809832bcd2be53a80d1c
R2  7e1d8ac5287bd94cfb7cea23c8ff40055258d6d7ca01809832bcd2be53a80d1c
IDENTICAL   permutation_digest match: True   grad_norms match: True
```

TF32 costs precision, not reproducibility. That is a necessary condition for
using it, not a sufficient one — see the open question at the end.

## A provenance gap this exposed

`train_record.json`'s `determinism` block records
`"torch.backends.cuda.matmul.allow_tf32": False` **for runs that demonstrably
ran with cuBLAS TF32 on** — they were 2.9× faster and produced different
digests than their TF32-off counterparts.

The block is a dict literal of the values `train.py` *assigned*, not values read
back from torch after assignment. It records intent, not state. Here that
happened because the wrapper below deliberately subverted the setter, but the
same gap would hide any other cause — a library flipping the flag, an
environment override, a future change to a torch default. **Reading the flags
back from `torch.backends` when building the record would make it evidence
rather than a transcript of intent.** Not fixed here; logged as S93 and as a
cross-module obligation.

## How TF32 was varied without editing the repo

`train.py:111` is a study-defining choice, so it was not edited.
`/shared/27as66/burst-pilot/scratch/tf32_wrapper.py` turns TF32 on, then
replaces `torch._C._set_cublas_allow_tf32` and `_set_cudnn_allow_tf32` with
no-ops so `train.py`'s own assignment lands on a dead setter, and runs
`train.py` through `runpy` at its real path so `REPO_ROOT` still resolves.

**The no-op held for cuBLAS and not for cuDNN** — the wrapper's exit line reads
`matmul.allow_tf32=True cudnn.allow_tf32=False`, so `torch.backends.cudnn`
reaches its flag by a different path. For a transformer with no convolutions
every GEMM goes through cuBLAS, so the measurement is sound; it is reported
throughout as **cuBLAS TF32**, which is what was actually varied.

## What this does not cover

- **bf16.** Not measured, because `train.py` implements fp32 only and refuses
  bf16 rather than training fp32 while the record claims otherwise. The largest
  available speedup is therefore also the one furthest out of reach.
- **The other two A100 SKUs.** `gpmoo-a1` carries three: 2× A100-SXM4-80GB,
  2× A100 80GB PCIe, 4× A100X. Only the A100X was measured, because it is the
  only SKU with four identical cards. Whether the three produce identical bits
  is untested, and §0.C forbids splitting a pair across them until it is.
- **Checkpoint I/O.** `h/run` is compute only. A full run writes ~105.5 GB.
- **Whether TF32 changes the study's answer.** Determinism survives it; the
  measured barrier and displacement do not have to. Everything in
  `docs/measurements/` was taken at fp32.
- **Sustained thermals.** Longest run here was 24 steps.

## Projections

Pilot is 4 runs on 4 cards, so its wall clock is one run's. Homogeneous blocks
available: **4** A100X on a1, **8** A6000 on b1, **16** across b1+b2.

| configuration | pilot (4 runs) | full study, 70 runs |
| --- | ---: | ---: |
| 4× A100X, committed | 40.9 h | 29.8 d |
| 4× A100X, TF32 + mb32 | **13.1 h** | 9.6 d |
| 8× A6000, committed | 40.1 h | 14.6 d |
| 8× A6000, TF32 + mb16 | 25.8 h | 9.4 d |
| 16× A6000 (b1+b2), committed | 40.1 h | 7.3 d |
| 16× A6000 (b1+b2), TF32 + mb16 | 25.8 h | **4.7 d** |

**The A6000s win the full study on card count, not on speed per card.** Four
usable A100s against sixteen A6000s outweighs a 3× per-card advantage that only
exists if TF32 is adopted.

## Open question this hands back

Adopting TF32 is not a performance decision. It changes the numerics of every
measurement in the study, and `docs/measurements/` was taken at fp32. It also
requires re-establishing the determinism record on the chosen card, since
`docs/measurements/2026-08-03-determinism-check-real-gpt2.md` was taken on an
A6000 at fp32 and transfers to neither the A100 nor TF32. It belongs with the
§12 open decisions, not with sizing.

## Reproducing

```bash
S=/shared/27as66/burst-pilot/scratch
srun --partition=gpmoo-a --nodelist=gpmoo-a1 --gres=gpu:8 --ntasks=1 \
     --cpus-per-task=48 --mem=400G \
  bash -c 'cd '"$S"' && ./bench.sh 8 off 2 & ./bench.sh 8 on 3 & \
           ./bench.sh 32 off 6 & ./bench.sh 32 on 7 & wait'
python "$S/summarize.py"
```

The harness lives in scratch, not in the repo: it varies two values the repo
deliberately fixes, and committing a convenient way to override them would
undercut the refusals that make them fixed.
