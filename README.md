# burst-study — configuration scaffold

The config system for a study that trains 40 GPT-2 Base models from scratch.
The 40 runs must be identical except for two things: a random **seed** and
which of four **arms** the run belongs to. At a fixed step mid-training, a
short burst of text is injected into one training batch, then training
continues.

| arm | what gets injected |
| --- | --- |
| `coherent` | a meaningful passage |
| `noise` | random real words |
| `ordinary` | normal text |
| `twin` | nothing at all — the matched control |

**This repo contains the config system and nothing else.** No training loop, no
model, no data pipeline. Everything that comes later is expected to read its
settings from `burst.config` and nowhere else, so that "what produced this
checkpoint" always has exactly one answer.

---

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install pyyaml==6.0.3 pytest==9.1.1
```

Python 3.11+. The only runtime dependency is PyYAML; `pytest` is for the tests.
There is deliberately no torch dependency — loading a config must work on a
laptop with no GPU and no ML stack installed.

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

Four values in `configs/base.yaml` are still `null` on purpose — they have not
been decided yet:

```
injection.injection_step
injection.burst_length_tokens
checkpointing.checkpoint_interval
model.tie_embeddings
```

Without `--launch`, the loader lets you inspect a config while those are
undecided, and prints a `NOT LAUNCH-READY` block listing what is missing. With
`--launch`, a missing value that this arm needs is a hard failure.

`twin` receives no injection, so it does **not** need `injection_step` or
`burst_length_tokens` and is launch-ready without them. The other three arms do
need them. All four arms need `checkpoint_interval` and `tie_embeddings`.

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
tests/test_config.py                 86 tests
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
| `tie_embeddings` | **null** | whether the output projection reuses the token embedding matrix. Undecided. |

### `training`

| field | value | meaning |
| --- | --- | --- |
| `batch_size` | 256 | sequences per optimizer step |
| `seq_len` | 1024 | tokens per sequence |
| `total_steps` | 9536 | optimizer steps for the whole run |

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

Once these are filled in they apply to every arm, including `twin`. Twin simply
ignores them — the loader does not treat a non-null `injection_step` as an
error for twin, because the value lives in the shared base config and cannot be
turned off for one arm without breaking the "identical except seed and arm"
guarantee.

### `checkpointing`

| field | value | meaning |
| --- | --- | --- |
| `checkpoint_interval` | **null** | steps between checkpoints. Undecided. An *interval*, not a directory. |

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
   `injection_step` within the run.
6. **Arm validation.** Exactly one of the four literal names, no case
   normalisation. A case variant gets a hint.
7. **Schema completeness.** The base config's sections and keys must match the
   loader's dataclasses exactly — a key added to `base.yaml` that no code reads
   is a provenance hole, and is rejected.
8. **No output paths in configs.** See above.
9. **Immutability.** Frozen dataclasses all the way down; `arms` is a tuple, not
   a list.

---

## Not in this repo, on purpose

Training loop, model definition, data pipeline, tokenizer, metrics, analysis,
the injection hook, and any launcher beyond the override generator.
