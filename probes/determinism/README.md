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
# on a cluster, let the scheduler pick the card -- do not set it yourself
srun --partition=gpmoo-b --nodelist=gpmoo-b1 --gres=gpu:1 --cpus-per-task=8 --mem=96G \
    .venv-ml/bin/python probes/determinism/check.py \
        --model hf --steps 20 --dtype bf16 --profile-kernels
```

Exit status 0 means every parameter tensor and every optimizer moment was
byte-identical between two fresh processes.

## One GPU, and specifically the allocated one

`check.py` **inherits** `CUDA_VISIBLE_DEVICES` rather than setting it. That is
not a stylistic preference. `gpmoo-b1` has eight A6000s and no cgroup device
isolation — `nvidia-smi -L` inside an allocation enumerates all eight — so
SLURM communicates `--gres=gpu:1` through that variable and nothing else. An
earlier version hardcoded `"0"`, which under an allocation of device 2 would
have trained on physical GPU 0: outside the allocation, on top of whatever else
was there.

So `resolve_visible_device()` inherits what the launcher set, **refuses**
anything naming more than one device, and falls back to `"0"` only when nothing
set it at all. Pinning and allocating are different verbs; this script asserts
it has exactly one card, it does not choose which.

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

`model.py` — GPT-2 Base, re-implemented here (`--model standin`, the default).
Not the study's model definition, which is still deliberately absent; a
stand-in faithful in every dimension that selects a kernel. Refuses to run
unless it is exactly `expected_param_count` parameters.

`hf_model.py` — the *real* GPT-2 (`--model hf`): the released `gpt2`
checkpoint, loaded through transformers, the same `MODEL_NAME`
`scripts/burst_match.py` already downloads. Checks every shape-bearing field
against `configs/base.yaml`, not just the parameter count.

## Why both models, rather than just the real one

They answer different questions, so both are run and both must agree.

The stand-in is built from `nn.Linear`; HuggingFace's GPT2 is built from
`Conv1D`, which stores its weight transposed and multiplies with `torch.addmm`.
Same arithmetic, different GEMM call, so possibly a different cuBLAS kernel —
and which kernel runs is the axis the whole result is keyed on. A
re-implementation cannot answer for the real one there.

Going the other way: `from_pretrained` loads fixed published weights, so the
`hf` path never draws random initialisation at all. The stand-in is the only
one whose **init RNG** is exercised, and the only one that runs on a machine
with no network and no HuggingFace cache.

The sharpest difference is dropout. The released checkpoint carries
`resid_pdrop = embd_pdrop = attn_pdrop = 0.1`; the stand-in has none.
**Dropout draws from the CUDA RNG on every forward pass**, so the `hf` run
reproduces only if the CUDA RNG stream itself reproduces across two processes —
something the stand-in, drawing nothing, could never test.
`configs/base.yaml` declares no dropout, so 0.1 is inherited from the
checkpoint and recorded as a probe assumption in `model_facts`. See D21.

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
