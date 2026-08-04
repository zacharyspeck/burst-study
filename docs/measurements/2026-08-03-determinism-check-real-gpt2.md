# Determinism check — the released GPT-2, not a stand-in

**Date:** 2026-08-03
**Code:** commit `584f2dd` (tree clean; all four `run_provenance.yaml` files
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
every optimizer moment — **including AdamW's `step` counter** — compared by
SHA-256 of its raw bytes.

That inclusion is new. The amendment on the 2026-08-02 record notes that the
probe's digest hashed only `exp_avg` and `exp_avg_sq`, so two states with
identical moments at different step counts digested the same; it also says
closing that "needs the A6000 this repo does not have". This work had one, so
it is closed, and the numbers below are measured under the wider definition.
See S90.

| leg | dtype | attention backend selected | CUDA kernels | per run | verdict |
|---|---|---|---|---|---|
| 1 | bf16 | flash, dropout variant | 107 | 155.6 s / 154.7 s | **IDENTICAL** |
| 2 | fp32 | mem-efficient cutlass, dropout variant | 92 | 337.5 s / 336.6 s | **IDENTICAL** |

149 parameter tensors and 444 optimizer entries per leg, all matching — 148
parameters × `step`, `exp_avg`, `exp_avg_sq`. (149 vs 148 because the tied
embedding is one parameter to the optimizer and two entries in `state_dict`.)

```
A sha256:  36b5c8c0b39fc615d3e8ebccf428ea06235d44ef7cbf942404884a21926e30bc   (bf16)
B sha256:  36b5c8c0b39fc615d3e8ebccf428ea06235d44ef7cbf942404884a21926e30bc

A sha256:  44082dbc28ae714fde1c02fb0f969470583eeeaab02191fe7dcba10d14ca1c07   (fp32)
B sha256:  44082dbc28ae714fde1c02fb0f969470583eeeaab02191fe7dcba10d14ca1c07
```

These hashes are **not** the ones an earlier draft of this file carried. Adding
`step` to the digest changed every value it produces, so the legs were re-run
rather than re-labelled. Under the narrower definition the same two legs were
also IDENTICAL, twice, on two different cards — the verdict never moved; what
moved is what the verdict is known to cover.

### Three passes, two physical cards, one verdict

Both legs were run three times in all, in three separate SLURM allocations:

| pass | digest definition | card | tree | result |
|---|---|---|---|---|
| 1 | narrow (no `step`) | device **2** | dirty mid-run | IDENTICAL, both legs |
| 2 | narrow (no `step`) | device **0** | clean, `dd17c3b` | IDENTICAL, same hashes as pass 1 |
| 3 | **wide (with `step`)** | device **2** | clean, `584f2dd` | **IDENTICAL — the record above** |

Twelve processes, one verdict. Two things follow, and they are worth keeping
apart.

**Determinism survived a change of physical card.** Passes 1 and 2 ran on
different A6000s and produced byte-identical parameters and optimizer moments,
which narrows the largest stated limitation of the 2026-08-02 result — though
only *within one node, one GPU model, one driver*. It still says nothing about
a different GPU model or a different driver. That evidence is from the narrow
digest; the cross-card comparison has not been repeated under the wide one.

**Pass 1 is not the record, for a reason unrelated to the digest.** It was
launched from a tree that went dirty mid-run, when
`probes/determinism/README.md` was edited while leg 1 was in flight — three of
its four runs recorded `dirty: true`. A run whose provenance says the commit
hash does not describe it is not evidence in this repo, whatever the dirty file
happened to be. That its digests matched pass 2's anyway is a bonus, not a
justification.

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
runs with no network and no HuggingFace cache. See S89 in
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
nothing set it. Covered by `tests/test_determinism_probe.py`. See S88 in
`implementation-notes.md`.

## What this does not cover

- **A different GPU model, driver, or torch build.** Two physical cards, both
  RTX A6000 on `gpmoo-b1`, one driver, one torch build.
- **A resumed run, *by this probe*.** All processes here ran uninterrupted.
  This is no longer the largest untested gap: `scripts/train.py` now proves
  resume bit-identical separately (step 12). What remains untested is resume
  **on the released checkpoint through this probe** — the two pieces of
  evidence have not been combined in one run.
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
  A study that intends to train GPT-2 Base still has to say what it wants.
  Note the contrast: `training.micro_batch`, `training.dtype` and
  `optimizer.adamw_impl` became launch-blocking config fields on 2026-08-03,
  so dropout is now the *only* value of this kind the config still omits.
  See D21.
- **The probe's own assumptions are still the probe's.** micro-batch 8 × 32
  accumulation, `foreach` AdamW. Those are now config fields with no defaults,
  and this run did not read them from there — so, as `train_once.py`'s own
  docstring says, this result describes a configuration nobody has chosen yet.

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
`probe-runs/determinism/hf_steps20_mbcfg_sdpa_{bf16,fp32}_cfg/` (gitignored).

`--micro-batch` and `--adamw-impl` are deliberately not passed: they are
launch-blocking config fields awaiting the pilot, and the probe falls back to
8 and `foreach` while they are null, labelling both `probe default` in
`digest.json`'s `setting_sources`. `--dtype` *is* passed, because the point of
running two legs is that the answer differs by dtype and the config has not
chosen one. The `cfg` in the directory name records that the value was left to
the config rather than fixed on the command line. Once the pilot decides these
fields, the same two commands measure the real configuration without edits —
and a flag contradicting a decided config value becomes a hard error.

The recorded `setting_sources` for this run:

```
micro_batch  probe default (config is null)
adamw_impl   probe default (config is null)
dtype        command line (config is null)
```
