# Handoff: what Asa needs to start piloting

Written 2026-08-03. Assumes you have read nothing else in this repository.

Everything below is either a command you can run or a value you have to supply.
Where a value is missing on purpose, this says so and says why — the repo
refuses to launch without those rather than picking for you, and that refusal
is deliberate.

---

## 1. What the study is, in one paragraph

70 runs of GPT-2 Base from scratch: **10 seeds × 7 arms**. Six arms inject a
194-token burst of text into one sequence of one batch at step 200; the seventh,
`twin`, injects nothing and is the matched control. Every pair of runs sharing
a seed is identical except for that burst — same initial weights, same data in
the same order — so the comparison is a paired difference within a seed. The
question is how six categories of injected text rank in the displacement they
produce.

The arms, in descending order of linguistic structure:

| arm | what gets injected |
| --- | --- |
| `fluent-false` | grammatical English asserting something specific and false |
| `fluent-true` | same register and structure, asserting something true |
| `scrambled-false` | `fluent-false` with word order broken |
| `scrambled-true` | `fluent-true` with word order broken |
| `pos-substituted` | each word replaced by one of the same part of speech |
| `random-chars` | no word structure at all |
| `twin` | nothing — the matched control |

---

## 2. What you must provide

Three things, none of which the repo can know:

| what | why it is yours |
| --- | --- |
| **GPUs** | Nothing here provisions hardware. The launcher emits commands; you run them however you like. |
| **A corpus location** | The tokenized corpus is 5.02 GB and is *not* in git. See §4. |
| **An output root** | One directory per run is created beneath it. Output paths are command-line arguments and never config values, so the same config runs on a laptop and on your cluster unchanged. |

---

## 3. What is deliberately unset, and why

**Three values are `null` in `configs/base.yaml` and every injecting run refuses
to launch without them.** This is not an oversight; it is what the pilot is for.

```
training.micro_batch      null
training.dtype            null
optimizer.adamw_impl      null
```

All three change **reduction order**, and reduction order is what bitwise
reproducibility is made of:

- **`micro_batch`** — gradient accumulation sums partial gradients in sequence,
  and floating-point addition is not associative, so `8 × 32` and `16 × 16`
  produce *different bits from identical data*. Accumulation is mandatory
  rather than optional: a full batch of logits is
  `256 × 1024 × 50257 × 4 B = 52.70 GB` against 48 GiB on an A6000, so
  `micro_batch == batch_size` cannot run at all.
- **`dtype`** — changes which CUDA kernels are selected. The determinism probe
  measured this directly: fp32 chose `fmha_cutlassF/B`, bf16 chose
  `pytorch_flash::flash_fwd/bwd`, 66 kernels against 74.
  **`scripts/train.py` implements fp32 only** and refuses `bf16` explicitly
  rather than training fp32 while the record claims bf16.
- **`adamw_impl`** — `foreach`, `fused` and `single` group their arithmetic
  differently and produce different bits from identical moments.

**The pilot's job is to settle all three on real hardware**, by measuring the
memory ceiling and the step time. Set them in `configs/base.yaml`, commit, and
every run becomes launch-ready.

> **One thing to know before you trust the existing determinism result.**
> `docs/measurements/2026-08-02-determinism-check.md` was produced on your
> A6000, with the handwritten `nn.Linear` model, at an *invented* micro-batch
> of 8, under `adamw_impl: foreach`. The pilot will run none of those three, so
> that result transfers on none of its axes and has to be re-established. Its
> header records this.

---

## 4. Getting the corpus onto your machine

The corpus is **5,020,581,888 bytes** — 149 training shards plus a held-out
slice — built from `Skylion007/openwebtext` at revision
`79d93d786212f7344586290adb811d4ae6a1762c`. It is not in git, deliberately.

**It is currently on Zach's machine and has to be transferred.** Once it is in
place, verify it *before* training on it:

```bash
python scripts/corpus_verify.py --outdir /path/to/corpus
```

That needs no network and no `datasets` stack. It re-hashes all 150 blocks,
checks `filesize / 2 == token_count`, checks the totals three ways, checks the
shard boundaries against arithmetic, and — the part that matters most on a new
machine — **re-tokenizes a committed probe passage with your tokenizer** and
compares it to the hash recorded when the corpus was built. That is the only
check that catches a divergent tokenizer at your end.

Expect:

```
blocks            150 of 150
tokens (manifest) 2,510,290,944
training tokens   2,499,805,184 (expected 2,499,805,184)
held-out disjoint True
tokenizer probe   ran -- agrees: True
OK -- complete and consistent.
```

Compare the printed `manifest sha256` against the value Zach sends you
**separately from the manifest file itself** — a corrupted manifest would
otherwise validate corrupted shards.

---

## 5. The exact commands

### Step 1 — check the repo is sane

```bash
python -m pytest -q                    # expect 531 passed, 164 skipped
python scripts/generate_overrides.py --check   # expect 70 ok, 0 missing, 0 mismatched
```

The skips are the tests needing torch or `transformers`. If you have the ML
environment, `python -m pytest -q` there should be **835 passed, 0 skipped**.

### Step 2 — emit the pilot

The launcher **prints commands and starts nothing**. It manages no processes,
assigns no devices, and retries nothing.

```bash
python scripts/launch.py \
    --outroot   /scratch/burst/runs \
    --corpus    /scratch/burst/corpus \
    --family    hf_gpt2 \
    --device    cuda \
    --seeds     0,1 \
    --arms      fluent-false,twin
```

That is **4 runs**. It writes three files into `--outroot`:

- `commands.txt` — one self-contained line per run
- `resume.txt` — resume commands for any run that died partway
- `status.json` — what is on disk for every selected run

### Step 3 — run them

```bash
bash /scratch/burst/runs/commands.txt              # sequentially
xargs -P 4 -I{} bash -c '{}' < .../commands.txt    # four at a time
```

Or paste a line into a SLURM script, or ssh it somewhere. Each line is
self-contained and begins `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

> **`CUDA_VISIBLE_DEVICES` is deliberately not set.** Device assignment is the
> one thing that depends on hardware nobody has provisioned. Add it yourself,
> or let your scheduler do it.

> **`CUBLAS_WORKSPACE_CONFIG` must be in the environment before CUDA
> initialises.** `train.py` refuses without it rather than setting it itself —
> a process that fixes its own environment cannot tell you the launcher forgot.

### Step 4 — the full study, when the pilot has settled the three values

```bash
python scripts/launch.py --outroot ... --corpus ... --family hf_gpt2 \
    --device cuda --all
```

`--all` must be typed. There is no default selection, because the difference
between 4 runs and 70 should be something you typed.

---

## 6. When something dies

It will. A run is roughly ten hours and there are 70 of them.

**Ask the disk, not a log.** Re-run the launcher with the same selection:

```bash
python scripts/launch.py --outroot ... --corpus ... --family hf_gpt2 \
    --device cuda --all --status-only
```

Every run lands in one of five states, each read off the filesystem:

| state | what it means | what to do |
| --- | --- | --- |
| `done` | the final full checkpoint exists | nothing |
| `resumable` | has a full checkpoint, not the last one | run its line from `resume.txt` |
| `started_not_resumable` | died before step 999, no full checkpoint | rerun from scratch |
| `not_started` | nothing on disk | run its line from `commands.txt` |
| `conflict` | the outdir holds a *different* config | investigate; do not overwrite blindly |

There is **no tracking file**. A tracking file can disagree with reality, and
then the thing being debugged is the tracker.

**Worst case you lose ~1000 steps**, because weights-only checkpoints are not
resumable — they carry no optimizer state and no RNG state. Full checkpoints
land every 1000 steps. If your hardware is preemptible, tell Zach: tightening
`full_interval` to 500 halves the loss and costs +700 GB across the study.
That is D-5 in `docs/decisions-pending.md`.

---

## 7. Storage

| | |
| --- | --- |
| per run | 105.5 GB (181 weights-only at 0.5 GB + 10 full at 1.5 GB) |
| all 70 runs | **7.385 TB** |
| corpus | 5.02 GB |
| against your 10 TB | **~74%** |

Derived by the loader, not typed. Print it yourself:

```bash
python -m burst.config --config configs/base.yaml \
    --run configs/runs/seed03_twin.yaml --outdir /tmp/x
```

---

## 8. Things that will refuse, and why that is correct

The repo refuses rather than warns in a lot of places. None of these is a bug:

- **A dirty working tree.** Every run stamps the current commit hash into
  `run_provenance.yaml`. Launching from uncommitted code records a hash that
  does not describe what produced the checkpoints, which is the one claim that
  file exists to make. Commit first.
- **Hand-edited override files.** `configs/runs/` is generated. The launcher
  runs `generate_overrides.py --check` and refuses if anything was edited.
- **A run that is not launch-ready.** See §3.
- **`bf16`.** `train.py` implements fp32 only and will not train fp32 while the
  config records bf16.
- **A corpus whose manifest disagrees with `corpus_spec`.** Different revision
  or different geometry means it is not the corpus the config describes.
- **A data order that does not match the recorded digest.**
  `data_order.verify_permutation` runs before the first batch. A run that
  serves a different order than its provenance claims is unreproducible and
  silently so.

---

## 9. What is not done

Told plainly so you do not discover it at an inconvenient moment.

- **No run has ever been trained.** Every number in `docs/measurements/` comes
  from public GPT-2, junk checkpoints, or synthetic input. The reports say so
  in their own banners.
- **Step 10's second half** — permutation-aligned barrier, aligned L2, RSF
  subspace probe — is **not built**. All three need `scripts/canonicalize.py`,
  which is Conv1D-only, and the study's model swap to HF GPT-2 has not landed.
  They raise `NotImplementedError` naming the blocker. See D-6.
- **The multiple-comparison correction is not chosen.** `spec-v4.md` has no
  statistics section at all. `scripts/analysis.py` requires the method as an
  explicit argument and refuses without one. See D-4.
- **Which metric is the headline is not decided.** See D-7.
- **The grad-clip obligation is instrumented, not discharged.** `train.py` logs
  the pre-clip gradient norm every step, but discharging it needs a real run
  reaching step 200.
- **Determinism is established on one A6000 only**, for a model and a
  micro-batch the study will not use, and with a digest that could not detect
  a divergence in AdamW's step counter. That last is now fixed in `train.py`
  and still present in the probe.
- **No multi-GPU.** The loop is single-device. Multi-GPU adds NCCL all-reduce
  ordering, which nothing here has tested.

`docs/decisions-pending.md` holds **six** open decisions (D-3 through D-8) with
what each one blocks, and two ruled ones (D-1, D-2) kept for the record. Read it
before ruling on anything. **Only D-6 touches anything you do before analysis,
and it does not block a launch** — see §3 for the three values that do.

---

## 10. If you read nothing else

```bash
# 1. verify the corpus you were sent
python scripts/corpus_verify.py --outdir /path/to/corpus

# 2. decide three values in configs/base.yaml, commit them
#    training.micro_batch, training.dtype, optimizer.adamw_impl

# 3. emit four runs
python scripts/launch.py --outroot /scratch/burst/runs \
    --corpus /path/to/corpus --family hf_gpt2 --device cuda \
    --seeds 0,1 --arms fluent-false,twin

# 4. run them
bash /scratch/burst/runs/commands.txt

# 5. ask the disk what happened
python scripts/launch.py --outroot /scratch/burst/runs \
    --corpus /path/to/corpus --family hf_gpt2 --device cuda \
    --seeds 0,1 --arms fluent-false,twin --status-only
```
