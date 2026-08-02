# probes/determinism

Answers one question: **does `determinism: true` actually hold?**

`configs/base.yaml` sets `determinism.deterministic: true`, and
`implementation-notes.md` records plainly that it "sets nothing" — the loader
never imports torch, so the flag configures no runtime behaviour at all. The
study's central claim is that runs sharing a seed are matched except for the
injection. If determinism is declared and not configured, two same-seed runs
diverge anyway and every arm-to-arm difference is confounded by nondeterminism
that nothing recorded.

This probe configures it and measures whether it held.

```bash
CUDA_VISIBLE_DEVICES=0 .venv-ml/bin/python probes/determinism/check.py --steps 20 --profile-kernels
```

Exit status 0 means every parameter tensor and every optimizer moment was
byte-identical between two fresh processes.

## Why a short run at full size, rather than a fast run at small size

The obvious way to make this cheap is to shrink the model. That would make the
result worthless.

Whether a training step reproduces bitwise is decided by **which CUDA kernels
get launched**, and kernel selection is keyed on shapes and dtypes: the SDPA
backend on head_dim and dtype, cuBLAS split-k on matrix shapes, the embedding
backward's scatter on vocab_size. Shrink `n_embd` and you change `head_dim` and
may select a different attention kernel; shrink `vocab_size` and the embedding
backward takes a different path. A green light from a small model is a green
light about kernels the real run never launches.

Step count is the opposite. Nondeterminism in a training step is per-step, and
a bitwise comparison catches it at step 1 — so running 20 steps instead of 9536
changes no kernel at all. **Everything shape-bearing is read from
`configs/base.yaml` and left alone; only the step count is reduced.**

`--profile-kernels` records the CUDA kernels one forward+backward launches, into
`digest.json`. That list is what lets a 20-step result speak about a
9536-step run: the kernel set is a function of shapes and dtypes, not of step
index.

## What it does

`train_once.py` — one process, one run. Reads the config through
`burst.config`, applies the full determinism configuration from
`implementation-notes.md` (`use_deterministic_algorithms`, the cudnn flags,
TF32 off, seeded sampler and workers), trains N steps, and writes
`digest.json`: a SHA-256 per parameter tensor and per optimizer moment, the
per-step loss/lr/grad-norm as raw double bits rather than printed decimals, and
the environment. Also writes `environment_asserted.yaml`, which
`implementation-notes.md` suggests as the way to make a determinism claim
evidenced rather than assumed.

`check.py` — launches `train_once.py` twice as **fresh subprocesses** and
compares. The comparison is deliberately not inside the training process: two
runs in one process would share an allocator, a cuBLAS handle, a cuDNN autotune
cache and an RNG lineage, which is most of what could go wrong.

`model.py` — GPT-2 Base. Not the study's model definition, which is still
deliberately absent; a stand-in faithful in every dimension that selects a
kernel. Refuses to run unless it is exactly `expected_param_count` parameters.

## Coverage boundary

Read this before quoting the result.

**Covered:** steps 0..N-1 at full GPT-2 Base shapes, two fresh processes, one
GPU, fp32, sampler and DataLoader-worker seeding.

**Not covered:**

- **fp16.** `--dtype` offers fp32 and bf16, and both are run, because
  `configs/base.yaml` declares no dtype and the answer is not the same for
  each: at fp32 SDPA selects the mem-efficient cutlass backend, at bf16 it
  selects *flash* attention — a different kernel family. fp16 is not offered:
  it would need a GradScaler, whose step-skipping state machine is its own
  determinism question and does not belong inside this one.
- **A different GPU, driver, or torch build.** Same limitation as D7 records
  for `burst_match.py`, for the same reason.
- **A resumed run.** RNG state in checkpoints is a listed cross-module
  obligation and nothing here tests it. `check.py` runs two uninterrupted
  processes.
- **Any step past the ones run**, including step 200 — the warmup/cosine
  boundary and the injection step. `--warmup-steps` moves the boundary into a
  short run so the schedule branch is at least crossed; the injection hook does
  not exist to test.
- **Multi-device reduction.** One GPU only. A multi-GPU run adds NCCL
  all-reduce, whose ordering is a separate determinism question.

## Output

Writes to `probe-runs/determinism/<tag>/{A,B}/`, which is gitignored. The
finding belongs in `docs/measurements/`, not the digests.
