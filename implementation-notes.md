# Implementation notes

Running log of decisions made while building the config scaffold. Anything that
deviates from the original spec, or that required a judgement call, is written
down here rather than left in the code for you to discover.

---

## Open questions for you

### 1. ~~`expected_param_count` implies `tie_embeddings: true`~~ — RESOLVED

**Resolved 2026-07-31: `tie_embeddings: true`.** The reasoning is now recorded
in a comment in `configs/base.yaml`. Original note kept below for the record.

`124439808` is exactly the GPT-2 Base parameter count **with tied embeddings**:

```
wte           50257 × 768                       =  38,597,376
wpe            1024 × 768                       =     786,432
12 blocks     12 × 7,087,872                    =  85,054,464
ln_f            768 × 2                         =       1,536
                                                  -----------
                                                  124,439,808
```

With untied embeddings you would get `163,037,184` (another `38,597,376` for a
separate output projection).

So the parameter count you gave has effectively already decided
`tie_embeddings`, even though the spec says it is undecided. I did **not** set
it — the instruction to leave those four values null was explicit, and guessing
at it is exactly the kind of silent decision this repo exists to prevent. But
one of the two numbers will have to move. Flagging it now rather than after 40
runs.

`checkpoint_interval` remains `null` and is still the one always-required
undecided value. Now that `tie_embeddings` is known, a load-time parameter-count
assertion is computable — see "Not yet enforced" below for why it still is not
in the loader.

### 2. Seeds are 0-indexed

The spec says "10 seeds" and gives `seed03_coherent` as an example, which is
consistent with either `0..9` or `1..10`. I used **`0..9`** (`seed00` …
`seed09`), matching Python and the usual ML convention, and the loader enforces
`0 <= seed < n_seeds`.

If you want `1..10` instead, it is a one-line change in
`scripts/generate_overrides.py` (`range(n_seeds)` → `range(1, n_seeds + 1)`) plus
the range check in `_validate_semantics`. Say the word and I will flip it.

---

## Deviations from the spec

### D1. Added a `--launch` flag to reconcile two requirements

The spec says the loader "must raise a clear error if a run is launched while a
value it needs is still null", and that `coherent` needs `injection_step` and
`burst_length_tokens`. It *also* says this must succeed today:

```
python -m burst.config --config configs/base.yaml --run configs/runs/seed03_coherent.yaml --outdir /tmp/testrun
```

Those conflict directly: `seed03_coherent` is an injecting arm, and both
injection fields are null by design, so a bare load of that file would have to
fail.

Resolution, taking the conservative reading — *loading* a config is not
*launching* a run:

- `python -m burst.config ...` (no flag) loads, writes provenance, prints the
  resolved values, and prints a `NOT LAUNCH-READY` block listing what is still
  undecided. The acceptance command works.
- `python -m burst.config ... --launch` turns that list into a hard failure.
- The Python API `load_config()` defaults to `require_complete=True`, i.e. the
  *safe* default, because anything importing it is about to train something.
  The CLI passes `require_complete=args.launch` explicitly.

The two defaults are deliberately different and both are documented in the
README and the docstring. If you would rather the CLI also default to strict
and add an `--inspect` flag instead, that is a two-line change — but the
acceptance command in the spec would then need a flag.

### D2. Overrides may set *only* `seed` and `arm` (stricter than specified)

The spec required rejecting unknown keys. I also reject *known* keys other than
`seed` and `arm` — so an override containing `training: {batch_size: 128}`
fails even though `batch_size` is a real key.

Reason: the study's central claim is that the 40 runs are identical except for
seed and arm. An override that quietly changed the learning rate for one run
would satisfy the unknown-key rule and still destroy that claim, leaving no
trace anywhere. This is the conservative option: it fails loudly and is a
one-line relaxation (`OVERRIDE_ALLOWED_KEYS` in `burst/config.py`) if a future
experiment genuinely needs a third per-run knob.

### D3. `raise ConfigError`, not `assert`

The spec uses "assert" throughout ("assert the type of every numeric field",
"the token-budget assertion"). Implemented as explicit raises, because `python -O`
strips `assert` statements from bytecode. A validation layer that silently
disappears under an optimisation flag is worse than no validation layer,
especially one whose job is provenance. Behaviour is identical otherwise, and
the tests assert on the messages.

### D4. Provenance is written after validation, not literally first

The spec says to write `resolved_config.yaml` "before anything else happens". I
read that as *before anything downstream happens*, not before validation —
writing an invalid config into an output directory would create a provenance
record of a run that never started, which is worse than not writing one. Order
is: read → merge → validate → `mkdir` → write both files → warn → return.

A consequence worth knowing: a config that fails validation creates no output
directory at all.

### D5. Six extra keys in `base.yaml` beyond the literal list

The spec listed values, not key names, and a few needed somewhere to live:

| key | why |
| --- | --- |
| `optimizer.name: adamw` | the spec says the optimizer is AdamW |
| `learning_rate.schedule: cosine` | the spec says cosine decay |
| `corpus.slice_description: 2.5B token slice` | the spec says "2.5B token slice"; the exact number is already in `expected_token_budget`, so this is the human-readable note |
| `model.expected_param_count` | the spec's "expected parameter count 124439808" |
| `experiment.n_seeds: 10` | the spec's "10 seeds"; the loader range-checks against it |
| `experiment.arms: [...]` | the spec's four arm names; cross-checked against `ARMS` in code |

No value was invented. Nothing was given a placeholder.

---

## Smaller decisions, logged as instructed

### S1. Extra arithmetic sanity checks

Beyond the required token-budget assertion I added, in the same spirit and each
about three lines: `seq_len <= block_size`, `n_embd % n_head == 0`,
`warmup_steps < total_steps`, `final <= peak`, `0 <= seed < n_seeds`, and
`0 <= injection_step < total_steps` / `burst_length_tokens > 0` once those are
set. All are pure arithmetic on values already present, and all fail at load.

### S2. Duplicate YAML keys are rejected

Not in the spec, but the same class of trap as `6e-4`. PyYAML silently keeps
the **last** value for a repeated key, so a file containing `seed: 3` near the
top and a forgotten `seed: 7` near the bottom loads without complaint and the
run is not the run you think it is. `_StrictSafeLoader` refuses the file.

### S3. Refusing to overwrite a differing `resolved_config.yaml`

If `--outdir` already contains a `resolved_config.yaml` describing a *different*
config, the loader raises instead of overwriting; overwriting would destroy the
record of what produced everything else in that directory. Re-running the
*same* config into the same directory is allowed, so the acceptance command is
re-runnable. `--force` is the documented escape hatch.

This is why the git/timestamp metadata lives in a separate
`run_provenance.yaml`: `resolved_config.yaml` stays byte-deterministic for a
given config, which is what makes the comparison meaningful.

### S4. Filename must match contents

If a run override's filename matches `seedNN_arm`, the loader checks it agrees
with the seed and arm inside. `seed05_noise.yaml` containing `seed: 4` fails.
Filenames that do *not* look like run names (ad-hoc or temporary files) skip
the check, so this does not get in the way.

### S5. Round-trip verification of the written config

After serialising `resolved_config.yaml`, the loader reads it back and compares
to the in-memory dict. This is not paranoia about PyYAML in general — it is
specifically about the `6e-4` trap in the *write* direction. `safe_dump(0.00006)`
emits `6.0e-05`; PyYAML inserts that `.0` precisely so the value re-parses as a
float rather than a string. Verified empirically before relying on it, and now
verified on every load.

### S6. Schema completeness check

The merged config's sections and keys must match the loader's dataclasses
exactly. A key added to `base.yaml` that no code reads would appear in
`resolved_config.yaml` and look like it configured something, which is a
provenance hole; it is rejected. The expected key set is derived from the
dataclasses themselves, so schema and dataclasses cannot drift apart.

### S7. `int` accepted where a `float` is expected; `bool` never accepted

`weight_decay: 0` (an int) is fine — widening is lossless. The reverse is not:
`total_steps: 0.5` fails. `bool` is rejected everywhere a number is expected,
because `isinstance(True, int)` is `True` in Python and `total_steps: yes`
would otherwise sail through as `1`.

### S8. A non-null `injection_step` is *not* an error for the twin arm

Deliberately not checked. `injection_step` lives in the shared base config, so
once you decide it, every arm sees it including `twin`. Making that an error for
twin would force a per-arm override, which would break the "identical except
seed and arm" guarantee. Twin ignores the value.

### S9. Output-path denylist is exact-match plus suffixes, not substring

`OUTPUT_PATH_KEY_DENYLIST` matches whole lowercased key names, plus the
suffixes `_dir`, `_path`, `_directory`, `_folder`. Substring matching would
have caught `checkpoint_interval` as a false positive. There is a test pinning
that specific non-behaviour.

### S10. Git state is read from the code's repo, not the working directory

`_repo_root()` returns the directory containing the `burst` package. The git
state that matters is the state of the code being run, not of wherever the
shell happens to be sitting when you launch. If git is unavailable or the
directory is not a repo, that is itself a loud warning — an unreconstructable
code version is a provenance failure, not a minor inconvenience.

### S11. `.gitignore` bug caught during setup

The first draft ignored `runs/`, which git matches at *any* depth — it silently
excluded all 40 files in `configs/runs/`. Fixed to `/runs/` (repo root only),
with a comment. Noting it because it is the exact failure mode this repo is
built to prevent, and it nearly shipped.

### S12. Corpus already had no path field — no change made

Checked on request. `corpus` holds `name: openwebtext`, `slice_description`,
and `expected_token_budget`, and no location of any kind. The existing
output-path rule already rejects a corpus path if one is ever added, at any
nesting depth: `corpus.data_path`, `corpus.data_dir`, `corpus.root_dir`,
`corpus.path` and so on all fail with the "same config on laptop and cluster"
error. There was a test for `corpus.data_path` from the start; I added four more
key spellings plus a test asserting the shipped `corpus` section holds nothing
path-shaped.

So there was nothing to move to a launch-time argument, and I did not invent a
`--corpus-path` flag for a field that does not exist. When the data pipeline
arrives, the corpus location belongs on *its* command line.

### S13. Burst text paths: shape chosen, and why

The spec said the burst text must live inside the repo but did not say how the
field is shaped, and there was no field yet. I used a per-arm mapping:

```yaml
injection:
  burst_text_paths:
    coherent: null
    noise: null
    ordinary: null
```

rather than a single `burst_text_path`. Reason: `base.yaml` is shared by all 40
runs and only `arm` distinguishes them, so a single field could not hold three
different texts — and giving each arm its own per-run override would break the
"identical except seed and arm" guarantee. `twin` deliberately has no entry, and
the schema check rejects one if added, since a burst text for the no-injection
control is a value that would look meaningful and mean nothing.

Access is `cfg.injection.burst_text_paths.for_arm(cfg.arm)`, written as an
explicit `if` chain rather than `getattr(self, arm)` so the twin case is visible
rather than an `AttributeError` waiting to happen.

Consequences worth knowing:

- Only the *running arm's* text is required at launch. A `coherent` run does not
  care whether the noise text has been written yet.
- `burst_text_paths` is explicitly exempted from the output-path denylist
  (`CONTENT_PATH_KEYS_EXEMPT`) because it legitimately holds paths. The
  exemption is by dotted key name, so renaming the field cannot slip it past
  *either* check — it would start tripping the denylist instead.

### S14. Absolute-path detection is checked on both POSIX and Windows rules

`PureWindowsPath("/burst.txt").is_absolute()` is `False` — no drive letter — so
a Linux-style absolute path written by your collaborator would pass a naive
check run on your Windows laptop, and a `C:\...` path would pass on the
cluster. `_looks_absolute` tests leading `/` and `\`, both `PurePosixPath` and
`PureWindowsPath`, and a bare drive prefix like `C:burst.txt`. The config has to
be rejected identically on both machines or the rule is theatre. There is a
parametrized test covering all four forms.

### S15. Burst text existence is checked at launch only

Not requested; added because a path pointing at a file that does not exist fails
the stated goal ("the commit hash covers the text") just as thoroughly as an
absolute path does, and a run that discovers this at the injection step has
already burned most of its compute. Checked only under `require_complete=True`,
since at inspect time the path may reasonably be decided before the text is
written. One line to remove if you disagree.

Not checked: whether the file is actually *tracked* by git. It turns out this is
already covered — `git status --porcelain` includes untracked files, so an
uncommitted `configs/burst_texts/coherent.txt` makes the tree dirty and triggers
the existing loud warning.

---

## Not yet enforced

Two values in `configs/base.yaml` are currently **inert** — the loader carries
them, validates their type, and writes them into `resolved_config.yaml`, but
nothing anywhere acts on them. They look like guarantees and are not, until the
training loop implements them. Written down here so that does not get lost.

### `model.expected_param_count` is not compared against anything

The loader checks it is an integer and nothing else. No model exists in this
repo, so there is nothing to count.

**The training loop must count the real model's parameters and fail if they
disagree with this number.** Something like
`sum(p.numel() for p in model.parameters())`, checked before step 0 and failing
loudly, not warning. With `tie_embeddings: true` now decided the expected value
is unambiguous — `124439808`, and `163037184` if tying were ever turned off —
so there is no excuse for the check to be soft.

The failure this guards against: an architecture change that quietly does not
match the config describing it, making every `resolved_config.yaml` in the study
a record of a model that was never built.

### `determinism.deterministic: true` sets nothing

It is a boolean in a YAML file. The loader does not import torch and never
will, so it configures no runtime behaviour whatsoever.

**The training loop must actually configure determinism when this is true**, and
that means all of it, not just the seed:

- `torch.manual_seed(seed)` / `torch.cuda.manual_seed_all(seed)`
- `torch.use_deterministic_algorithms(True)`
- `torch.backends.cudnn.deterministic = True`, `benchmark = False`
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` in the environment — cuBLAS is
  non-deterministic without it and `use_deterministic_algorithms(True)` raises
  at runtime if it is unset
- deterministic DataLoader ordering: `shuffle` driven by a seeded generator,
  `worker_init_fn` seeding each worker, and a fixed `num_workers`

This one matters more than it looks. The study's entire claim is that two runs
were matched except for the injection. If determinism is merely *declared* and
not configured, two runs with the same seed diverge anyway, and every
arm-to-arm difference is confounded by nondeterminism that nothing recorded.

A reasonable follow-up: have the training loop write an
`environment_asserted.yaml` next to `resolved_config.yaml` recording what it
actually set, so the claim is evidenced rather than assumed.

---

## Environment as built

- Python 3.13.14 (spec requires 3.11+; nothing here is version-sensitive)
- `pyyaml==6.0.3`, `pytest==9.1.1` — pinned exactly in `pyproject.toml`. These
  are the current releases, newer than the 6.0.2 / 8.x you may have seen
  elsewhere. Exact pins rather than ranges: a range would let a future PyYAML
  change how a number parses without anything in this repo recording it.
- No dependencies beyond `pyyaml`, `pytest`, and the standard library.
- Built and tested on Windows. Note that `--outdir /tmp/testrun` on Windows
  resolves to `C:\tmp\testrun`; on the cluster it is the real `/tmp/testrun`.
  Nothing in the loader cares.

## Test coverage

112 tests. The five required cases are
`test_typo_in_override_key_raises`,
`test_null_injection_fields_raise_for_injecting_arm` /
`test_null_injection_fields_are_fine_for_twin`,
`test_invalid_arm_raises`,
`test_token_budget_mismatch_raises_when_total_steps_changes`, and
`test_output_path_key_in_config_raises`.

The rest cover the YAML traps, duplicate keys, immutability, the provenance
files, the overwrite guard, the schema check, burst text path containment,
corpus path rejection, all 40 generated overrides loading, a mechanical check
that the 40 runs differ *only* in seed and arm, and the CLI end to end.

Note that the null-required-field coverage now hangs off `checkpoint_interval`
and the injection fields rather than `tie_embeddings`, since that value has been
decided. `test_null_tie_embeddings_would_still_raise` keeps the check itself
under test by setting it back to null in a throwaway config, so deciding the
value did not quietly delete the behaviour.

Tests build a throwaway copy of the real `configs/base.yaml` and edit one thing,
rather than using a hand-written fixture — so they break if `base.yaml` drifts,
which is the point.
