# burst-study — configuration scaffold

> **⚠ The design changed on 2026-08-01. This README describes the code, and the
> code implements the retired Spec v3.**
>
> The current design is **v4**: a five-way categorical comparison
> (fluent-false, fluent-true, scrambled, POS-substituted, random-characters)
> plus twin, **60 runs**, injection fixed at **step 200**. The arm named
> `ordinary` is gone. Read **[`docs/spec-v4.md`](docs/spec-v4.md)** first, then
> **[`docs/v4-gap-analysis.md`](docs/v4-gap-analysis.md)** for what stands
> between the code and that design.
>
> **Task 8b-i is built** (see `docs/measurements/8b-i-in-context-match.md`):
> `bursts/` now holds all five v4 arms at 194 tokens each, measured in context.
> **`burst/` and `configs/` were deliberately NOT updated** and still describe
> the v3 arms — see S29 in `implementation-notes.md`. Do not read the config
> and `bursts/` as agreeing with each other; they now actively disagree.
>
> Everything below about the config system is accurate. The arm list and the
> run arithmetic in it are v3.

The config system for a study that trains 40 GPT-2 Base models from scratch.
The 40 runs must be identical except for two things: a random **seed** and
which of four **arms** the run belongs to. At a fixed step mid-training, a
short burst of text is injected into one training batch, then training
continues.

| arm (v3 — retired) | what gets injected | v4 status |
| --- | --- | --- |
| `coherent` | a meaningful passage | renamed **fluent-false** |
| `noise` | random real words | renamed **scrambled** |
| `ordinary` | normal text | **deleted as an arm** — now only substrate |
| `twin` | nothing at all — the matched control | unchanged |
| — | — | **new:** fluent-true, POS-substituted, random-characters |

**This repo contains the config system and nothing else.** No training loop, no
model, no data pipeline. Everything that comes later is expected to read its
settings from `burst.config` and nowhere else, so that "what produced this
checkpoint" always has exactly one answer.

---

## Collaborator setup (Linux)

Everything below is plain POSIX. The repo was developed on Windows but has no
Windows-specific code; see "Platform notes" at the bottom.

```bash
git clone <repo-url> burst-study
cd burst-study

python3 -m venv .venv
source .venv/bin/activate

# Exact pins. Do not use a requirements range -- a PyYAML upgrade can change
# how a number parses, which is the whole class of bug this repo guards against.
pip install pyyaml==6.0.3 pytest==9.1.1

# 1. run the tests
python -m pytest -q

# 2. run the acceptance command
python -m burst.config \
    --config configs/base.yaml \
    --run    configs/runs/seed03_coherent.yaml \
    --outdir /tmp/testrun
```

Expected: `279 passed, 141 skipped`, then the resolved config printed, exit
status 0, and
`resolved_config.yaml` + `run_provenance.yaml` in `/tmp/testrun`. The run ends
with a `NOT LAUNCH-READY` block — that is correct, not a failure. Four values
are still undecided; see [`--launch`](#--launch).

The 141 skips are the tests that need torch or `transformers`, which the install
above deliberately does not include. Skipped, not failed, is the correct result
here — see
[Matching candidate burst passages](#matching-candidate-burst-passages).

If you see a **"WARNING: the working tree is DIRTY"** banner, stop and find out
why before running anything real. On a fresh clone the tree should be clean, and
a dirty tree means the recorded commit hash does not describe the code that ran.

Requires Python 3.11+ (`python3 --version`). The only runtime dependency is
PyYAML; `pytest` is for the tests. **`burst/` deliberately has no torch
dependency** — loading a config must work on a login node with no GPU and no ML
stack. `scripts/burst_match.py` does need torch, which is exactly why it is in
an optional group that the install above does not pull in.

`pip install -e .` also works but is not needed; `pyproject.toml` puts the repo
root on `sys.path` for pytest, and `python -m burst.config` picks it up from the
working directory.

## Verify the whole thing in one command

```bash
python -m pytest -q && python -m burst.config --config configs/base.yaml --run configs/runs/seed03_coherent.yaml --outdir /tmp/testrun
```

---

## Launching a run

```bash
python -m burst.config \
    --config configs/base.yaml \
    --run    configs/runs/seed03_coherent.yaml \
    --outdir /scratch/burst/seed03_coherent
```

| flag | meaning |
| --- | --- |
| `--config` | the base config. Always `configs/base.yaml`. |
| `--run` | the two-line run override. |
| `--outdir` | **required.** Where this run's files go. |
| `--launch` | fail if any field this arm needs is still `null`. Use it in the real launcher. |
| `--force` | allow overwriting an existing, *different* `resolved_config.yaml`. |

`--outdir` is a command-line argument and **never** a config value. That is
what lets the same config file run unchanged on your laptop and on your
collaborator's cluster. The loader refuses to load any config containing a
path-like key (`outdir`, `save_dir`, anything ending in `_dir` or `_path`) and
tells you why.

### `--launch`

Several values in `configs/base.yaml` are still `null` on purpose — they have
not been decided yet:

```
injection.injection_step
injection.burst_length_tokens
injection.burst_text_paths.{coherent,noise,ordinary}
```

Without `--launch`, the loader lets you inspect a config while those are
undecided, and prints a `NOT LAUNCH-READY` block listing what is missing. With
`--launch`, a missing value that this arm needs is a hard failure.

`twin` receives no injection, so it does **not** need `injection_step`,
`burst_length_tokens`, or a burst text, and is launch-ready without them. The
other three arms each need all three — and only *their own* burst text, not the
other arms'.

`model.tie_embeddings` (`true`) and the two `checkpointing` intervals (`50` and
`1000`) are decided; see the field tables below.

From Python the safe default is the other way round — `load_config()` uses
`require_complete=True`, because anything calling it from code is about to
train something:

```python
from burst.config import load_config
cfg = load_config("configs/base.yaml",
                  "configs/runs/seed03_coherent.yaml",
                  outdir="/scratch/burst/seed03_coherent")
print(cfg.run_name, cfg.training.total_steps)   # seed03_coherent 9536
```

The returned object is a frozen dataclass. `cfg.seed = 4` raises.

---

## What gets written to `--outdir`

Both files are written *before* anything downstream is allowed to run, so a run
that crashes on step one still leaves an exact record of what it was trying to do.

**`resolved_config.yaml`** — the fully merged config, exactly as loaded. Valid
YAML, reloadable, and verified to survive a round-trip before being written.

**`run_provenance.yaml`** — everything else needed to reconstruct the run:

- `git.commit`, `git.branch`, `git.dirty`, `git.dirty_files`
- `written_at_utc`, the absolute paths of both source config files, the outdir
- `launch_ready` and `missing_for_launch`
- Python version, platform, hostname, PyYAML version
- the exact command line

**If the working tree is dirty, the loader prints a loud warning.** The code is
part of the experimental condition — a recorded commit hash does not describe
what actually ran if there are uncommitted changes on top of it. Commit before
launching anything you intend to publish.

If `--outdir` already holds a `resolved_config.yaml` describing a *different*
config, the loader refuses rather than overwrite it; overwriting would destroy
the record of what produced everything else in that directory. Re-running the
*same* config into the same directory is fine.

---

## Layout

```
configs/base.yaml                    every value shared by all 40 runs
configs/runs/seedNN_arm.yaml         40 generated overrides, two lines each
burst/config.py                      the loader — read this one
scripts/generate_overrides.py        regenerates all 40 override files
scripts/burst_match.py               measurement primitives, in context
scripts/make_bursts.py               generates the three generated arms
scripts/build_pos_pool.py            one-time POS pool build (nltk)
scripts/match_arms.py                all arms at one burst position
scripts/match_sweep.py               sweeps the scrambled arm's window size k
scripts/position_sweep.py            8b-ii: burst position + gradient diagnostic
scripts/tune_arms.py                 8b-iii: tunes arms toward the median
scripts/gradient_direction.py        8b-iv: pairwise gradient cosine + profiles
scripts/canonicalize.py              step 9: symmetry canonicalization
bursts/fluent_false.txt              hand-written, fixed, never regenerated
bursts/fluent_true.txt               hand-written, fixed, fact-checked
bursts/scrambled_false.txt           fluent_false, word order broken
bursts/scrambled_true.txt            fluent_true, word order broken
bursts/scrambled_corpus.txt          corpus span, word order broken
bursts/pos_substituted.txt           grammar kept, lexical content destroyed
bursts/random_chars.txt              printable ASCII, no word structure
bursts/context.txt                   the shared 1024-token sequence
bursts/pos_pool.json                 committed POS vocabulary + tag template
bursts/ordinary.txt                  v3 arm, retained as substrate only
bursts/provenance.json               schema v2: per-arm seeds, params, hashes
tests/test_config.py                 172 tests (still v3, deliberately)
tests/test_burst_match.py            43 tests
tests/test_make_bursts.py            45 tests
tests/test_sequence_assembly.py      33 tests (splice, seeds, regeneration)
tests/test_canonicalize.py           70 tests (step 9: tripwire, symmetries)
tests/test_canonicalize_recipe.py    41 tests (step 9: the canonical form)
tests/test_canonicalize_mutations.py 16 tests (step 9: injected faults)
implementation-notes.md              decisions, assumptions, open questions
```

### Run naming

`seed{NN}_{arm}`, seed zero-padded to two digits: `seed03_coherent`,
`seed00_twin`. **Seeds are 0-indexed** — `seed00` through `seed09` for the ten
seeds. The loader derives the run name from the file's contents and, when the
filename looks like a run name, checks the two agree; `seed05_noise.yaml`
containing `seed: 4` is a copy-paste error and fails.

### Regenerating the override files

```bash
python scripts/generate_overrides.py            # write all 40
python scripts/generate_overrides.py --check    # verify, change nothing
```

`--check` exits non-zero if any override is missing or hand-edited. Worth
running before a launch batch. The seed count and arm names come from
`configs/base.yaml`, so the study's shape is written down in one place only.

### Matching candidate burst passages

The coherent-vs-noise comparison only means anything if both passages deliver
the same size shove to the weights. `scripts/burst_match.py` measures that shove
on public GPT-2 (a stand-in — the study's own model does not exist yet):

```bash
python scripts/burst_match.py coherent.txt
python scripts/burst_match.py coherent.txt noise.txt ordinary.txt
```

Per passage it prints the token count, the mean, standard deviation and max of
the per-token loss, the five most surprising tokens, and the global gradient
norm from one backward pass — both standalone and scaled by `1/batch_size`,
which is what one sequence in a batch actually contributes. With two or more
files it adds a comparison table of pairwise differences.

Nothing is trained, nothing is saved. It **warns loudly if the passages differ
in token count**, because loss is a mean over tokens and gradient norms from
different denominators are not comparable, and it **errors rather than
truncating** if a passage exceeds GPT-2's 1024-token context.

### The batch size, and where the header says it came from

The scaled figure needs `training.batch_size`. The script **reads it from
`configs/base.yaml`** — there is no copy of it in the script, and no default
anywhere. Every run states the value and its provenance on the first line:

```
batch:   256 sequences  (C:\...\configs\base.yaml, via burst.config loader)
batch:   256 sequences  (C:\...\configs\base.yaml, direct YAML read -- burst.config declined: ...)
batch:   512 sequences  (--batch-size on the command line, overriding the config)
```

The first line is the normal case: the value came through `burst.config` and
passed the same validation a real run's config passes. The second appears when
the loader refuses the file as a whole — a config mid-decision, or one that has
drifted from the loader's schema — in which case the script reads that single
key with PyYAML and names the loader's objection. The loader's validation is
never relaxed to avoid this.

If the key is absent, the script stops and says so. It will not guess: a wrong
batch size silently rescales every headline number in the report.

Point it at a different config with `--base-config PATH`, or bypass the config
entirely with `--batch-size N`.

### Installing

The measurement scripts are the only things in the repo that need torch, which
is why it is an optional dependency:

```bash
pip install -e ".[dev]"            # config work — no ML stack, 279 tests run
pip install -e ".[dev,measure]"    # adds torch, transformers, datasets
```

The dependency runs one way only: `burst_match.py` imports `burst.config`, and
nothing in `burst/` imports it back. Loading a config still works on a laptop
with no ML stack, because `burst/config.py` still imports nothing heavier than
PyYAML.

---

## The burst passages

`bursts/` holds the three candidate passages, **all at exactly the same token
count** under the GPT-2 tokenizer. That is the whole point of them: an
arm-to-arm comparison of gradient norms only means anything if the passages
are the same length, because the loss is a mean over tokens.

| file | what it is |
|---|---|
| `coherent.txt` | Hand-written, fixed, **never regenerated by any script**. Its token count is the target length N. |
| `ordinary.txt` | A contiguous span of real OpenWebText, trimmed to N tokens. |
| `noise.txt` | A different contiguous span, word order shuffled within non-overlapping windows of size k, trimmed to N tokens. |
| `provenance.json` | Seed, k, N, both source spans as (document, word offset), the filter thresholds, and a SHA-256 per file. |

**`coherent.txt` is content, not output.** It is hand-written and it is the
independent variable of the study. `make_bursts.py` opens it read-only and
refuses to start if an output path resolves to it.

**`coherent.txt` is deliberately false.** It is a fluent, plausible,
entirely fabricated biography of a Beatles pianist who never existed. That is
the design: the coherent arm has to be something the model cannot already know,
so that anything it later reproduces came from the burst and not from
pretraining. It is an experimental stimulus and is not a claim about anything.

These are **candidates**. The final texts come out of piloting against our own
model, not from this step.

### Regenerating the corpus-derived two

```bash
python scripts/make_bursts.py --k 5            # --k is required, no default
python scripts/make_bursts.py --k 5 --seed 3   # different spans
```

The first run streams a small OpenWebText slice from HuggingFace and caches it
under `.corpus-cache/` (gitignored — corpus data is never committed). Later runs
are offline. With no cache *and* no network it says which is missing.

It prints each selected raw span in full before shuffling or trimming, so you
can read what was chosen. Spans that are mostly non-ASCII, mostly punctuation or
digits, under 40% alphabetic, or short of three sentence-ending marks are
skipped. Shuffling changes tokenization, so lengths are never assumed: spans are
sampled generously, trimmed to exactly N, and the count is asserted for all
three files before the script exits. A trim that cuts mid-word is fine — a burst
is a token sequence, not a sentence — and is recorded in `provenance.json`.

Same `--seed` and `--k` gives byte-identical output, `provenance.json` included.

### Sweeping the shuffle window

```bash
python scripts/match_sweep.py
python scripts/match_sweep.py --k 2 3 5 8 15 30
```

Measures coherent, ordinary, and noise at each k plus a full-span shuffle, and
prints one table: token count, loss, gradient norm, and the absolute and
percentage gap from coherent on both. Every row is the same token length, so
the gradient norms are directly comparable.

It reuses `burst_match.py`'s measurement function rather than reimplementing
it, so the table cannot drift from the single-passage numbers.

**It picks no winner.** The tolerance is not set yet and applying it is not the
script's job — a highlighted row would become the decision by default.

---

## Every config field

### Per-run — supplied by the override file, `null` in the base

| field | type | meaning |
| --- | --- | --- |
| `seed` | int, 0–9 | The single stochastic knob. Controls **both weight initialization and data order** — two runs sharing a seed see the same initial weights and the same batches in the same order. That is what makes an arm-to-arm comparison a matched pair. |
| `arm` | str | Exactly one of `coherent`, `noise`, `ordinary`, `twin`. Case-sensitive. |

### `model` — GPT-2 Base

| field | value | meaning |
| --- | --- | --- |
| `n_layer` | 12 | transformer blocks |
| `n_head` | 12 | attention heads per block |
| `n_embd` | 768 | embedding / residual width |
| `vocab_size` | 50257 | GPT-2 BPE vocabulary |
| `block_size` | 1024 | maximum context length |
| `expected_param_count` | 124439808 | parameter count this architecture should produce, recorded so later training code can check the model it built against the config that described it |
| `tie_embeddings` | true | the output projection reuses the token embedding matrix. Forced by `expected_param_count`: 124,439,808 is the GPT-2 Base count *with* tying. Untied adds another `50257 × 768 = 38,597,376` parameters for a separate output projection, giving 163,037,184. GPT-2 ties by default. |

### `training`

| field | value | meaning |
| --- | --- | --- |
| `batch_size` | 256 | sequences per optimizer step |
| `seq_len` | 1024 | tokens per sequence |
| `total_steps` | 9536 | optimizer steps for the whole run. **Steps are 0-indexed**, so the first is 0 and the last is `total_steps - 1` = 9535. Anything keying off "the last step" must compute it, never hardcode 9535. |

### `optimizer`

| field | value | meaning |
| --- | --- | --- |
| `name` | `adamw` | |
| `weight_decay` | 0.1 | |

### `learning_rate`

| field | value | meaning |
| --- | --- | --- |
| `schedule` | `cosine` | |
| `peak` | 0.0006 | peak LR reached at the end of warmup |
| `warmup_steps` | 200 | linear warmup length |
| `final` | 0.00006 | LR at the end of the cosine decay |

### `corpus`

| field | value | meaning |
| --- | --- | --- |
| `name` | `openwebtext` | |
| `slice_description` | `2.5B token slice` | human-readable note on which slice |
| `expected_token_budget` | 2499805184 | `batch_size × seq_len × total_steps`, asserted at load |

The corpus is **named, never located.** There is no corpus path field, and the
loader rejects one if it is ever added — same rule as `--outdir`, for the same
reason: your laptop and the cluster store OpenWebText in different places, and
a path in the config would make the same experiment produce different
`resolved_config.yaml` files on different machines. When the data pipeline
arrives, the corpus location belongs on its command line.

### `determinism`

| field | value | meaning |
| --- | --- | --- |
| `deterministic` | true | request bitwise-reproducible kernels from whatever runs later |

### `experiment`

| field | value | meaning |
| --- | --- | --- |
| `n_seeds` | 10 | valid seeds are `0 .. n_seeds - 1` |
| `arms` | the four names | must match `ARMS` in `burst/config.py` |

### `injection` — both undecided

| field | value | meaning |
| --- | --- | --- |
| `injection_step` | **null** | the step at which the burst enters one training batch |
| `burst_length_tokens` | **null** | how many tokens the burst is |
| `burst_text_paths.coherent` | **null** | repo-relative path to the coherent arm's burst text |
| `burst_text_paths.noise` | **null** | repo-relative path to the noise arm's burst text |
| `burst_text_paths.ordinary` | **null** | repo-relative path to the ordinary arm's burst text |

There is deliberately no `burst_text_paths.twin`; twin receives no text, and
adding one is rejected by the schema check.

**Burst text paths must be relative and must resolve inside this repository.**
The loader rejects an absolute path (checked as absolute on *both* POSIX and
Windows, so a Linux-style `/home/...` is refused on Windows too), and rejects
anything that escapes the repo root via `..`. At launch (`--launch`) the file
must also exist.

The reason is provenance, not tidiness: the burst text is the study's
independent variable — experimental *content*, not configuration. It has to be
committed alongside the code so the git commit hash in `run_provenance.yaml`
covers the text as well. A path outside the repo leaves the injected text
unversioned, and points somewhere that will not exist on the other machine.

Suggested home: `configs/burst_texts/<arm>.txt`.

Once `injection_step` and `burst_length_tokens` are filled in they apply to
every arm, including `twin`. Twin simply ignores them — the loader does not
treat a non-null `injection_step` as an error for twin, because the value lives
in the shared base config and cannot be turned off for one arm without breaking
the "identical except seed and arm" guarantee.

### `checkpointing`

| field | value | meaning |
| --- | --- | --- |
| `weights_only_interval` | 50 | steps between **weights-only** checkpoints (weights + step number, ~0.5 GB) |
| `full_interval` | 1000 | steps between **full** checkpoints (weights + optimizer state + step + RNG state, ~1.5 GB) |

Both are intervals in optimizer steps, not directories.

Two schedules rather than one because the two reasons to checkpoint have very
different costs. **Full** checkpoints exist for crash recovery — they carry
everything needed to resume bit-identically, and AdamW's two moment buffers are
each another full copy of the parameters, hence ~1.5 GB. **Weights-only**
checkpoints exist to measure how the model changes during training. Every
metric in this study is a function of the weights alone, so weights-only is
sufficient for measurement, and that is exactly what makes a 50-step interval
affordable — the same density at full-checkpoint size would cost three times as
much.

Two rules the loader encodes and the **training loop must implement**:

- **Precedence.** When both intervals fire on the same step, write the **full**
  checkpoint only; a weights-only file there would duplicate data already
  inside it. This is why `full_interval` must be an exact multiple of
  `weights_only_interval` — the loader rejects any other pairing, since
  otherwise the schedules drift and "the same step" stops being well defined.
- **Final step.** The last step (`total_steps - 1` = 9535) always saves a
  **full** checkpoint regardless of interval. 9536 divides by neither 50 nor
  1000, so without this rule the finished model — what every headline
  comparison is computed on — would never be written.

A checkpoint is due after every N *completed* steps, i.e. at 0-indexed step `s`
when `(s + 1) % N == 0`. The first weights-only checkpoint lands at step 49.

### Derived: the checkpoint plan

The loader computes the schedule from the config the same way it computes the
token budget — never hardcoded, so it stays correct if `total_steps` changes:

```python
plan = cfg.checkpoint_plan
plan.last_step                 # 9535
plan.weights_only_count        # 181
plan.full_count                # 10
plan.estimated_bytes_per_run   # 105_500_000_000   (105.5 GB)
plan.estimated_bytes_all_runs  # 4_220_000_000_000 (4.22 TB)

cfg.checkpoint_kind_at(49)     # "weights_only"
cfg.checkpoint_kind_at(999)    # "full"   <- precedence
cfg.checkpoint_kind_at(9535)   # "full"   <- final-step rule
cfg.checkpoint_kind_at(48)     # None
```

`checkpoint_kind_at()` is the single definition of the schedule. It writes
nothing — it is a pure function of the config — but the training loop must call
it, or reproduce it exactly, so the two rules above cannot end up implemented
two different ways. `python -m burst.config …` prints the plan, and it is
recorded in `run_provenance.yaml`.

The 0.5 GB and 1.5 GB figures are **estimates for 124M fp32 parameters, not
measurements**: weights are `124439808 × 4 = 497,759,232` bytes and a full
checkpoint adds two AdamW moment buffers. Nothing in this repo has ever written
a checkpoint.

---

## What the loader checks

Every failure raises `ConfigError` with the offending field named. Nothing uses
`assert` — `python -O` deletes assert statements, and a validation layer that
vanishes under an optimisation flag is worse than none.

1. **Provenance.** `resolved_config.yaml` + `run_provenance.yaml` written to
   `--outdir` before anything else; git commit and dirty-tree state recorded;
   loud warning if dirty or if git state cannot be read at all.
2. **Strict merge.** An override key that does not already exist in the base is
   an error. `see: 3` fails; it never silently falls back to the default seed.
   Separately, an override may set *only* `seed` and `arm`.
3. **YAML numeric traps.** PyYAML parses `6e-4` as the **string** `'6e-4'` (its
   float rule requires a decimal point) and bare `no`/`yes`/`on`/`off` as
   booleans. Every rate in `base.yaml` is written in plain decimal form, and
   every field's Python type is checked after parsing — including the
   `isinstance(True, int) is True` trap, so `total_steps: yes` cannot slip
   through as `1`. A string that looks like scientific notation gets an error
   message saying so.
4. **Duplicate YAML keys.** PyYAML silently keeps the last one. This loader
   refuses the file.
5. **Arithmetic.** `batch_size × seq_len × total_steps` must equal
   `expected_token_budget`. Also: `seq_len ≤ block_size`, `n_embd` divisible by
   `n_head`, `warmup_steps < total_steps`, `final ≤ peak`, `seed` in range,
   `injection_step` within the run, both checkpoint intervals positive, and
   `full_interval` an exact multiple of `weights_only_interval`.
6. **Arm validation.** Exactly one of the four literal names, no case
   normalisation. A case variant gets a hint.
7. **Schema completeness.** The base config's sections and keys must match the
   loader's dataclasses exactly — a key added to `base.yaml` that no code reads
   is a provenance hole, and is rejected. A key that has been *removed* from
   the schema (currently `checkpoint_interval`) fails with an error naming what
   replaced it, rather than a generic "unknown key".
8. **No output paths in configs.** No corpus path either. See above.
9. **Burst text paths stay inside the repo.** Relative only, no escaping via
   `..`, and the file must exist at launch — so the recorded commit hash covers
   the injected text.
10. **Immutability.** Frozen dataclasses all the way down; `arms` is a tuple,
    not a list.

---

## Platform notes

Developed on Windows, intended to run on a Linux cluster. Nothing here is
platform-specific:

- **Line endings** are normalised by `.gitattributes` (`* text=auto eol=lf`), so
  a Linux checkout gets LF regardless of what the Windows working copy holds.
- **Paths** are all `pathlib`; nothing concatenates separators by hand.
- **`--outdir /tmp/testrun`** is a real path on Linux. On Windows the same string
  resolves to `C:\tmp\testrun`. The loader does not care either way.
- **Absolute-path rejection** for burst texts is evaluated under *both* POSIX and
  Windows rules, so a `/home/...` path is refused on Windows and a `C:\...` path
  is refused on Linux. The config is rejected identically on both machines.
- **`.venv/Scripts/` vs `.venv/bin/`** is the only genuine difference, and it is
  in the setup instructions above, not in any code.
- **No torch, no CUDA, no GPU** is touched by this repo.

One nit: `scripts/generate_overrides.py` has a `#!/usr/bin/env python` shebang.
On distributions where only `python3` exists, invoke it as
`python3 scripts/generate_overrides.py` (or activate the venv first, which is
what the instructions above do) rather than executing it directly.

## What is tracked

Tracked: the configs (`configs/base.yaml`, all 40 overrides), the burst texts
once they exist (`configs/burst_texts/*.txt`), the package, the tests, the
generator, and the docs.

Not tracked: checkpoints and weights (`*.pt`, `*.ckpt`, `*.safetensors`,
`*.bin`, …), corpus and tokenized data (`/data/`, `/corpus/`), run outputs
(`/runs/`, `/outputs/`, `/results/`, `/checkpoints/`), `.venv/`, `__pycache__/`.

`.gitignore` explains the anchoring rule it follows, and why. If you edit it,
re-run the audit at the top of that file — an unanchored pattern matches at
every depth and can silently exclude real content.

## Not in this repo, on purpose

Training loop, model definition, data pipeline, tokenizer, metrics, analysis,
the injection hook, and any launcher beyond the override generator.

Obligations that later modules must honour are recorded in
`implementation-notes.md` — read it before building the data pipeline or the
training loop:

- a **held-out data reservation**, which must be carved out when the corpus is
  tokenized or it will not exist when the metrics module needs it
- **RNG state in checkpoints**, without which a resumed run silently diverges
  from an uninterrupted one
- the **checkpoint precedence and final-step rules**, which this repo defines
  and validates but cannot enforce — `Config.checkpoint_kind_at()` is the
  single definition and the training loop must call it or reproduce it exactly
- the **0-indexed step convention**, including that step 0 is not a checkpoint
  step

Two config values are also still **inert** — `expected_param_count` is compared
against nothing because no model exists, and `determinism: true` configures
nothing because no training code exists. Both are listed under "Not yet
enforced".
