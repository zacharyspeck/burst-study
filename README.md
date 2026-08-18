# burst-study

Thirty-two GPT-2 124M models trained from scratch on OpenWebText, identical in
every respect except two: the random seed, and which short passage of text was
injected into the training stream at step 200. Four conditions crossed with
eight seeds. The question is whether a single brief exposure to a fluent but
fabricated passage leaves a measurable trace in a model, relative to a fluent
passage whose content the corpus already attests, and relative to a matched
control that received no injection at all. The two injected passages were
matched so that their full-batch gradient deltas agree to 0.14 percent, which
means the conditions differ in what they assert rather than in how hard they
push on the weights.

On the pre-registered primary contrast the study **did not detect** a
difference. Held-out cross-entropy differed by a mean of 0.000442 nats between
the fabricated and attested conditions after each was differenced against its
seed-matched twin, with a 95 percent confidence interval on held-out
cross-entropy of [-0.001223, +0.000242] nats and p = 0.31. The design was
powered to detect a difference of 0.001321 nats in held-out cross-entropy at 80
percent power, so differences smaller than that are outside what these 32 runs
can resolve.

Paper: **(link to be added on publication)**. NeurIPS 2026 ATTRIB workshop
submission.

---

## Condition names changed after registration. Read this before the code.

The conditions are called **`fluent-attested`** and **`fluent-fabricated`** in
the current code, configuration, and measurement files.

They were pre-registered and originally implemented under different names:

| Registered and originally implemented as | Called this everywhere in current code and output |
| --- | --- |
| `fluent-true` | `fluent-attested` |
| `fluent-false` | `fluent-fabricated` |
| `fluent_true.txt` | `fluent_attested.txt` |
| `fluent_false.txt` | `fluent_fabricated.txt` |

The rename happened on 2026-08-18, after every run had finished and every
measurement had been taken. It changed identifiers and nothing else. The two
passages are byte-for-byte the files that were injected, and `git log --follow`
shows them as renames rather than as a deletion and an addition.

**Dated records under `docs/` deliberately keep the registered names.** That
includes `docs/preregistration.md`, `docs/decisions-pending.md`, everything
matching `docs/measurements/*.md`, `docs/preprint.md`, `docs/spec-v4.md`, and
`docs/README-dev.md`. Those documents are cited by the paper at a specific
commit and line number, and their value depends on saying what they said on the
day they were written. Renaming inside them would shift the line numbers the
paper cites and would rewrite a pre-registration record after the fact. So when
you read `fluent-false` in the pre-registration and `fluent-fabricated` in a
config, those are the same condition.

Two consequences worth stating plainly:

1. **Line-number references in the paper are as of the commit cited, not as of
   HEAD.** Use the commit hash or the tag, not the current file.
2. Dated records still refer to `bursts/fluent_false.txt` and
   `bursts/fluent_true.txt`, which no longer exist under those names. The
   mapping above is how to resolve them.

The transformation is auditable. `tools/rename_conditions.py` holds the mapping
and the protected-file list as data, and its `--check` mode verifies both that
no old identifier survives where the rename applied and that no new identifier
leaked into a protected record.

```
python tools/rename_conditions.py --check
```

---

## Pre-registration

`docs/preregistration.md`.

It was fixed before any run existed. The two confirmatory contrasts were
registered at commit `42a7d35` on 2026-08-03, tagged **`prereg-fixed`**, and the
32 runs trained from commit `d52a2b8` on 2026-08-09, tagged **`runs-base`**. The
pre-registration therefore predates the first training step by six days, and the
ordering is checkable from git rather than asserted here.

Amendments are recorded in section 10 of that document rather than folded into
the original text. Some of them, including the rulings dated 2026-08-10, were
recorded after the runs completed. Each such amendment says so in its own text.
That disclosure is deliberate and the reader is meant to weigh it.

Annotated tags mark every commit the paper cites:

| Tag | Commit | What it marks |
| --- | --- | --- |
| `prereg-fixed` | `42a7d35` | the two confirmatory contrasts registered, before any data existed |
| `arms-v4-reconciled` | `c2df6c7` | arm list reconciled to spec v4 |
| `d7-closed` | `2ef812b` | D-7 closed by a decision rule |
| `pilot-v2` | `715dd67` | pilot re-run, S96's metric-floor claim retracted |
| `arm-cut-four` | `8d3ae2a` | arm list cut to four |
| `d9-ruled` | `1ea243b` | D-9 ruled the day it was raised |
| `runs-base` | `d52a2b8` | determinism established on all eight cards and both hosts, the commit all 32 runs trained from |

All 32 runs trained from branch `arm-cut-2026-08-08`.

---

## Reproducing a run

**Software.** Python 3.11 or newer is required by `pyproject.toml`. The training
host ran Python 3.12.13, `torch 2.13.0+cu130`, and `transformers 5.14.1`.

**Hardware.** Eight NVIDIA A100-SXM4-80GB cards, driver 610.43.02, on Thunder
Compute, one training process per card. Packing more than one run onto a card
was measured and rejected, because two concurrent runs each took 2.4 times as
long rather than 2 times, which lowered aggregate throughput by 15 percent in
steps per second.

**Cost.** Roughly **9.8 hours per run** at 3.725 seconds per step over 9,536
steps. Thirty-two runs is about 315 GPU-hours.

**Determinism.** `CUBLAS_WORKSPACE_CONFIG` must be set in the environment before
CUDA initializes. `scripts/train.py` refuses to set it for itself, on the
grounds that a process which fixes its own environment cannot tell you the
launcher forgot. All eight cards and both hosts produced one identical digest,
`28f3ea04...`, so runs are comparable across cards and across machines.

**Launching.** `scripts/launch.py` emits command lines and starts nothing. It
manages no queue and retries nothing, which is what lets the same repository run
under SLURM, bare SSH, or `xargs -P`.

```
python scripts/launch.py \
    --outroot /path/to/runs \
    --corpus  /path/to/openwebtext-tokenized \
    --family  hf \
    --seeds   0 \
    --arms    fluent-fabricated
```

That writes `commands.txt`, `resume.txt` and `status.json` into the emit
directory. Each emitted line carries its own environment, so it can be pasted as
is. Use `--all` for the full set. `--all` must be typed out and has no default.

The launcher refuses to emit anything from a dirty working tree, because
`burst/config.py` stamps the current commit hash into `run_provenance.yaml` on
every load, and a run launched from uncommitted code records a hash that does
not describe the code that produced it.

**Run configuration.** One base config, `configs/base.yaml`, plus one two-line
override per run in `configs/runs/`. An override may set `seed` and `arm` and
nothing else. That restriction is the study's central claim expressed as code,
and the loader enforces it. Override files are generated, never hand-edited:

```
python scripts/generate_overrides.py --check
```

**Note on counts.** `configs/runs/` holds 40 override files, which is 10 seeds
crossed with 4 arms. Only the first 8 seeds were trained, so the study reports
32 runs. The extra 20 files are unlaunched configuration, not missing data.

**Tests.** The suite runs in two environments on purpose:

```
.venv/Scripts/python.exe    -m pytest -q   # no torch: skips are expected
.venv-ml/Scripts/python.exe -m pytest -q   # torch + transformers: no skips
```

As of this commit that is 686 passed and 240 skipped without torch, and 1,069
passed with 0 skipped with torch and transformers installed.

The torch-free run existing at all is the point. `burst/` imports nothing
heavier than PyYAML, so loading a config works on a login node with no GPU, and
the skips are the evidence.

---

## What is not in this repository, and why

**Model checkpoints.** A single run writes 105.5 GB under the committed
checkpoint policy, and 32 runs is roughly 3.4 TB. Nothing of that size belongs
in git. The derived numbers are committed instead, under `docs/measurements/`,
which is what the tables and figures are built from.

**The corpus.** OpenWebText is named in the configuration and never located in
it or vendored into the repository. Tokenized shards are not committed either.

**How to obtain the corpus independently.** The study used
`Skylion007/openwebtext` from the Hugging Face Hub, then tokenized it locally
with the repository's own two-stage pipeline:

```
python scripts/corpus_fetch.py     # retrieves the source files
python scripts/corpus_tokenize.py  # writes the tokenized blocks
python scripts/corpus_verify.py    # checks completeness and consistency
```

On the training host that took about 35 minutes end to end and produced
7,572,655,845 bytes across 25 source files, then 150 tokenized blocks holding
2,510,290,944 tokens, of which 2,499,805,184 are training tokens with the
held-out split disjoint.

To confirm you have the same corpus, compare this timing-independent content
digest over every block hash rather than the manifest hash:

```
150  c338fd06805d2f3ef18b44f3a612e19b97626977354a886b40d068a620e68ed1
```

The manifest hash in `manifest.json` is **not** a usable check for a rebuilt
corpus. It is taken over the whole file, and the file contains
`elapsed_seconds_last_run`, so no rebuild on any machine can reproduce a
recorded manifest hash however byte-identical its shards. This is a defect in
the check rather than evidence of a bad corpus, and it is written up in
`docs/measurements/2026-08-08-thunder-a100.md`.

**Burst texts are the exception and they are committed.** The passages injected
into a run are the independent variable of the study, so they live in `bursts/`
and the commit hash covers them. `bursts/provenance.json` records the generator
and SHA-256 of each one.

---

## Repository layout

| Path | Contents |
| --- | --- |
| `burst/` | the config loader and validator, importing nothing heavier than PyYAML |
| `bursts/` | the injected passages themselves, plus `provenance.json` recording each one's generator and SHA-256 |
| `configs/` | `base.yaml`, and one generated two-line override per run in `configs/runs/` |
| `docs/` | pre-registration, decision log, spec, preprint, and the measurement record under `docs/measurements/` |
| `probes/` | standalone determinism checks, run before and between training |
| `scripts/` | training loop, corpus pipeline, launcher, and every analysis and measurement script |
| `tests/` | the test suite, which runs with and without an ML stack installed |
| `tools/` | repository maintenance, currently the condition-rename tool |

---

## Licensing

Code, configuration, documentation and measurement records are MIT licensed. See
`LICENSE`.

Five files in `bursts/` contain text derived from the OpenWebText corpus and are
**not** relicensed. OpenWebText is distributed under CC0 1.0, and the underlying
documents are third-party web pages whose rights remain with their authors.
`LICENSE` lists the five files and states this separately.

The two fluent passages are hand-written and are covered by the MIT grant.
