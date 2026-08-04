# Determinism check — the released GPT-2, not a stand-in

**Date:** 2026-08-03
**Code:** commit `dd17c3b` (tree clean; all four `run_provenance.yaml` files
record `dirty: false`)
**Probe:** `probes/determinism/check.py --model hf` — see that directory's README
**Config:** `configs/base.yaml` + `configs/runs/seed03_twin.yaml`, unmodified
**Machine:** `gpmoo-b1`, one RTX A6000 allocated by SLURM, driver 560.35.05,
torch 2.13.0+cu126, CUDA 12.6

## What was asked, and what was already known

The 2026-08-02 check measured `probes/determinism/model.py` — a GPT-2 Base
**re-implementation**, faithful in every dimension believed to select a CUDA
kernel. This one measures the **released `gpt2` checkpoint** loaded through
transformers: the actual model, the actual weights.

That distinction turned out to matter, and the kernel evidence below is why.

## Result

Two fresh processes, same seed, same allocated GPU. Every parameter tensor and
every optimizer moment compared by SHA-256 of its raw bytes.

| leg | dtype | attention backend selected | CUDA kernels | per run | verdict |
|---|---|---|---|---|---|
| 1 | bf16 | flash, dropout variant | 107 | 157.8 s / 155.3 s | **IDENTICAL** |
| 2 | fp32 | mem-efficient cutlass, dropout variant | 92 | 332.0 s / 330.4 s | **IDENTICAL** |

149 parameter tensors and 296 optimizer moments per leg, all matching.

```
A sha256:  46f2d0b20cc4e407ed18c0709def3f52606f11ac60bd1e4cfafb72868b509b7e   (bf16)
B sha256:  46f2d0b20cc4e407ed18c0709def3f52606f11ac60bd1e4cfafb72868b509b7e

A sha256:  5c5551d1a0de380e1a904d814984f8dd1487bd7aaa7c79b5b40529ab0484d123   (fp32)
B sha256:  5c5551d1a0de380e1a904d814984f8dd1487bd7aaa7c79b5b40529ab0484d123
```

### Both legs also reproduced on a second, physically different A6000

Both legs were run twice, in two separate SLURM allocations. The first landed
on `gpmoo-b1` device **2**, the second on device **0**, and **all four digests
above were produced by both**. Eight processes agree, not four.

This is not what the check set out to test — the pass bar was two processes on
one GPU — but it is worth recording, because it narrows the largest stated
limitation of the 2026-08-02 result. Determinism now demonstrably survives a
change of physical card *within one node, of one model, on one driver*. It
still says nothing about a different GPU model or a different driver.

The first pass is not the record for a separate reason: it was launched from a
tree that went dirty mid-run, when `probes/determinism/README.md` was edited
while leg 1 was in flight. Three of its four runs recorded `dirty: true`. A run
whose provenance says the commit hash does not describe it is not evidence in
this repo, whatever the dirty file happened to be, so it was discarded and
re-run from `dd17c3b`. That the digests came out identical anyway is the bonus
above, not the justification.

## The model that was actually trained

```
transformers GPT2LMHeadModel.from_pretrained('gpt2')
  124,439,808 parameters      matches model.expected_param_count exactly
  Conv1D projections          not nn.Linear
  gelu_new                    layer_norm_epsilon 1e-05
  resid_pdrop 0.1   embd_pdrop 0.1   attn_pdrop 0.1
```

`hf_model.py` checks `n_layer`, `n_head`, `n_embd`, `vocab_size`, `block_size`,
`tie_embeddings` **and** the parameter count against `configs/base.yaml` before
training, and refuses to run on any mismatch. The stand-in could only check the
count.

## Why the 2026-08-02 result did not already cover this

Two reasons, both visible in the recorded kernel lists rather than argued from
first principles.

**1. Dropout selects a different attention kernel.** The released checkpoint
carries dropout 0.1 on three paths; the re-implementation has none. At fp32 the
two runs do not launch the same attention backward:

```
stand-in (2026-08-02):  fmha_cutlassB_f32_aligned_64x64_k64_sm80
real gpt2 (this run):   fmha_cutlassB_f32_aligned_64x64_k64_dropout_sm80
```

That is a different kernel. The earlier green light was a green light about a
kernel the real model does not launch.

**2. Dropout is the only thing here that draws from the CUDA RNG.** At `p = 0`
the stand-in never touched the CUDA RNG stream during a forward pass, so its
result could not speak to whether that stream reproduces across two processes.
This run exercises it on every one of 12 blocks × 3 dropout sites × 20 steps ×
32 accumulation micro-steps, and the `fused_dropout_kernel_vec` launches are in
the recorded kernel list.

Beyond attention, the real model launches **107 kernels at bf16 against the
stand-in's 74, and 92 at fp32 against 66** — including `aten::addmm` from
`Conv1D` and GEMM shapes the re-implementation never produced, such as
`ampere_bf16_s1688gemm_bf16_128x128_ldg8_relu_f2f_nn`. The re-implementation
was a reasonable stand-in and it was still not the same program.

The two models are both kept, and both must agree. `from_pretrained` loads
fixed weights, so the `hf` path never draws random initialisation at all — the
stand-in is the only one whose init RNG is exercised, and the only one that
runs with no network and no HuggingFace cache. See S54 in
`implementation-notes.md`.

## Why 20 steps transfers to a 9536-step run

Unchanged from the earlier record, and the reason 20 steps is enough:
nondeterminism in a training step is per-step, so a bitwise comparison catches
it at step 1. Every shape is the config's own — `n_layer` 12, `n_head` 12,
`n_embd` 768, `vocab_size` 50257, `seq_len` 1024 — and kernel selection is
keyed on shapes and dtypes, not on step index. Only the step count is reduced.

**This remains a result about the configuration, not about torch.** It held
*because* the probe set all of it: `use_deterministic_algorithms(True)`, both
cudnn flags, TF32 off on both paths, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, a
seeded sampler generator and a `worker_init_fn`. The set is recorded in each
run's `environment_asserted.yaml`. Nothing here says a training loop that
merely reads `deterministic: true` and calls `torch.manual_seed` would
reproduce.

## A bug this run found, in the probe rather than in torch

`check.py` set `CUDA_VISIBLE_DEVICES = "0"` unconditionally, with a comment
saying the probe must never reach for a second device. `gpmoo-b1` has eight
A6000s and **no cgroup device isolation** — `nvidia-smi -L` inside an
allocation enumerates all eight — so SLURM communicates `--gres=gpu:1` through
that variable and nothing else. It allocated device 2; the old code would have
trained on physical GPU 0, outside the allocation and on top of whatever else
was running there.

Replaced with `resolve_visible_device()`, which inherits the allocation,
refuses anything naming more than one device, and defaults to `"0"` only when
nothing set it. Covered by `tests/test_determinism_probe.py`. See S53 in
`implementation-notes.md`.

## What this does not cover

- **A different GPU model, driver, or torch build.** Two physical cards, both
  RTX A6000 on `gpmoo-b1`, one driver, one torch build.
- **A resumed run.** RNG state in checkpoints is a listed cross-module
  obligation; all processes here ran uninterrupted. Still the largest untested
  gap, because a resumed run that silently diverges is exactly the failure the
  obligation exists to prevent.
- **Steps past 19**, including the real step 200. The 2026-08-02 leg 3 crossed
  the warmup/cosine boundary and is not repeated here: that branch is in
  `lr_at()` and is model-independent, so the earlier result covers it. The
  injection hook does not exist to test.
- **Multi-device.** One GPU per run. Multi-GPU adds NCCL all-reduce ordering.
- **fp16.** Would need a GradScaler, whose step-skipping is its own question.
- **Real corpus data.** `SyntheticCorpus` exercises sampler and worker seeding,
  not tokenizer or shard ordering, neither of which exists yet.
- **The dropout value itself.** `configs/base.yaml` declares no dropout, so 0.1
  is inherited from the released checkpoint and recorded as a probe assumption.
  A study that intends to train GPT-2 Base still has to say what it wants. See
  D21.

## Reproducing

```bash
srun --partition=gpmoo-b --nodelist=gpmoo-b1 --gres=gpu:1 --cpus-per-task=8 --mem=96G \
    .venv-ml/bin/python probes/determinism/check.py \
        --model hf --steps 20 --dtype bf16 --profile-kernels
srun --partition=gpmoo-b --nodelist=gpmoo-b1 --gres=gpu:1 --cpus-per-task=8 --mem=96G \
    .venv-ml/bin/python probes/determinism/check.py \
        --model hf --steps 20 --dtype fp32 --profile-kernels
```

Do **not** set `CUDA_VISIBLE_DEVICES` yourself; let the scheduler allocate and
let `check.py` inherit it. Exit status 0 means identical. Output lands in
`probe-runs/` (gitignored).
