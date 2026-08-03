# Determinism check — `determinism: true` holds, on this machine, at both dtypes

> ## COVERAGE AMENDED 2026-08-03 — THE COMPARISON HAD A BLIND SPOT
>
> This result is not withdrawn. What changed is what it is known to cover.
>
> **The digest omits AdamW's `step` counter.** `probes/determinism/train_once.py`
> hashes every parameter tensor and, per parameter, only `exp_avg` and
> `exp_avg_sq`. It does not hash `step`. Bias correction is `1 - beta**step`, so
> **two optimizer states holding identical moments at different step counts
> produce different next updates and digest identically here.** Verified
> directly: moving `step` by 100 leaves the combined SHA-256 unchanged.
>
> Nothing suggests the runs compared below actually differed in `step` — they
> were fresh processes running the same number of steps, so there is no reason
> they would. The point is narrower and worth stating plainly: **this comparison
> could not have detected it if they had.** The claim "byte equality of every
> saved parameter tensor and every optimizer moment" is exactly true; the
> broader reading, "the two final states are identical", is not what was
> measured.
>
> `scripts/train.py::state_digest` now includes the step counter.
> **`probes/determinism/train_once.py` has NOT been changed**, so re-running the
> probe today reproduces the same blind spot. Closing it means editing the probe
> and re-running, which needs the A6000 this repo does not have.
>
> **A second scoping change, same date.** The result was produced with
> `--adamw-impl foreach` (the probe's default) at an invented micro-batch of 8,
> on one RTX A6000, with `probes/determinism/model.py`'s `nn.Linear` model.
> `configs/base.yaml` now declares `training.micro_batch`, `training.dtype` and
> `optimizer.adamw_impl` as launch-blocking fields with no defaults, so a future
> run states which of these it used instead of inheriting them. See S78.

**Date:** 2026-08-02
**Code:** commit `27dc517` (tree clean; `run_provenance.yaml` in each run
directory records `dirty: false`)
**Probe:** `probes/determinism/check.py` — see that directory's README
**Config:** `configs/base.yaml` + `configs/runs/seed03_twin.yaml`, unmodified

## What was asked

`configs/base.yaml` sets `determinism.deterministic: true`.
`implementation-notes.md` records under "Not yet enforced" that it **sets
nothing** — the loader never imports torch, so the flag configures no runtime
behaviour. Until now the study's central claim (runs sharing a seed are matched
except for the injection) rested on a boolean nobody had tested.

## Result

Two fresh processes, same seed, same GPU. Every parameter tensor and every
optimizer moment compared by SHA-256 of its raw bytes.

| leg | dtype | attention backend selected | steps | per run | verdict |
|---|---|---|---|---|---|
| 1 | fp32 | mem-efficient (cutlass) | 20 | 276.9 s | **IDENTICAL** |
| 2 | bf16 | flash | 20 | 97.9 s | **IDENTICAL** |
| 3 | bf16, warmup→cosine boundary moved to step 10 | flash | 20 | 97.3 s | **IDENTICAL** |

149 parameter tensors and 296 optimizer moments per leg, all matching.

```
A sha256:  c216c18897825fdf903a77de6a94ecb20048d5a6b303e93b38e79da7c9fb2ba2   (fp32)
B sha256:  c216c18897825fdf903a77de6a94ecb20048d5a6b303e93b38e79da7c9fb2ba2

A sha256:  16c0ed6b129400290722d336349cc2b50af6d2bfcc561b1873de4de7e5de9309   (bf16)
B sha256:  16c0ed6b129400290722d336349cc2b50af6d2bfcc561b1873de4de7e5de9309

A sha256:  c13668fc846f498a7c76818d8a2c78a65244dfc6672bb6b47cd387b2ab04f682   (bf16, warmup 10)
B sha256:  c13668fc846f498a7c76818d8a2c78a65244dfc6672bb6b47cd387b2ab04f682
```

Leg 3 confirms the LR schedule branch itself reproduces. With `warmup_steps`
overridden to 10, the learning rate ramps to peak `0.0006` at step 9 and the
cosine branch takes over at step 10 — visible in the log, and bitwise identical
across both processes:

```
  step   9  lr 0.00060000  loss 10.8624246120   <- last warmup step
  step  10  lr 0.00060000  loss 10.8760061264   <- first cosine step
  step  11  lr 0.00060000  loss 10.8872592449
```

The rate barely moves after the boundary because the cosine runs over
`total_steps` 9536, so one step past warmup is ~0.01% of the decay. That is the
real schedule, unmodified apart from the warmup length.

**This is a result about the configuration, not about torch.** Determinism held
*because* the probe set all of it — `use_deterministic_algorithms(True)`, the
two cudnn flags, TF32 off on both paths, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, a
seeded sampler generator and a `worker_init_fn`. The recorded set is in each
run's `environment_asserted.yaml`. Nothing here says a training loop that
merely reads `deterministic: true` and calls `torch.manual_seed` would
reproduce.

## Why 20 steps at full size, and why that transfers

Shrinking the model would have been faster and would have proved nothing.
Bitwise reproducibility is decided by **which CUDA kernels get launched**, and
kernel selection is keyed on shapes and dtypes. Every shape here — `n_layer`
12, `n_head` 12, `n_embd` 768, `vocab_size` 50257, `seq_len` 1024 — is read
from `configs/base.yaml` and left alone. Only the step count is reduced, and
nondeterminism in a training step is per-step: a bitwise comparison catches it
at step 1.

The kernel lists in `digest.json` are the evidence, and they are also the
warning. **The fp32 and bf16 legs do not launch the same attention kernels:**

```
fp32:  fmha_cutlassF_f32_aligned_64x64_rf_sm80   (mem-efficient, forward)
       fmha_cutlassB_f32_aligned_64x64_k64_sm80  (mem-efficient, backward)

bf16:  pytorch_flash::flash_fwd_kernel<...cutlass::bfloat16_t...>
       pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<...>
       pytorch_flash::flash_bwd_convert_dq_kernel<...>
```

66 kernels at fp32, 74 at bf16. That is precisely why both were run rather than
one: an fp32-only result would have been silent about the backend a
mixed-precision study would actually use. Flash attention's backward is the
usual suspect for nondeterminism — it accumulates `dq` with atomics in its
default configuration — and under `use_deterministic_algorithms(True)` PyTorch
selected the `seqk_parallel` + `convert_dq` variant instead, which reproduced.

## What this does not cover

- **A different GPU, driver, or torch build.** Same limitation D7 records for
  `burst_match.py`. Measured on one RTX A6000, driver 560.35.05, torch
  2.13.0+cu126, CUDA 12.6.
- **A resumed run.** RNG state in checkpoints is a listed cross-module
  obligation; both processes here ran uninterrupted. This is the largest
  untested gap, because a resumed run that silently diverges is exactly the
  failure the obligation exists to prevent.
- **Steps past 19**, including the real step 200. Leg 3 moves the
  warmup/cosine boundary to step 10 so the schedule branch is crossed, but the
  injection hook does not exist to test.
- **Multi-device.** One GPU. A multi-GPU run adds NCCL all-reduce ordering.
- **fp16.** Would need a GradScaler, whose step-skipping is its own question.

## What this surfaced about the config

`configs/base.yaml` declares `batch_size: 256` and `seq_len: 1024` but not
**micro-batch size, dtype, or the AdamW implementation.** All three change
reduction order, which is what bitwise reproducibility is made of. The probe
had to pick values (8 × 32 accumulation, both dtypes, `foreach`) and records
them as assumptions in `environment_asserted.yaml`; nothing was added to the
config. See D21 in `implementation-notes.md`.

The micro-batch one is forced, not preferential: a full batch of logits is
`256 × 1024 × 50257 × 4 bytes` = **52.7 GB**, against 49 GB on this card. A
full batch in one forward pass does not fit on this hardware, so the real study
must use gradient accumulation or more than one device — and neither appears
anywhere in the config. This belongs on the same list as the undeclared
clipping policy in `docs/spec-v4.md`.

## Reproducing

```bash
CUDA_VISIBLE_DEVICES=0 .venv-ml/bin/python probes/determinism/check.py \
    --steps 20 --dtype fp32 --profile-kernels
CUDA_VISIBLE_DEVICES=0 .venv-ml/bin/python probes/determinism/check.py \
    --steps 20 --dtype bf16 --profile-kernels
CUDA_VISIBLE_DEVICES=0 .venv-ml/bin/python probes/determinism/check.py \
    --steps 20 --dtype bf16 --warmup-steps 10
```

Exit status 0 means identical. Output lands in `probe-runs/` (gitignored).
