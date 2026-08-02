# Implementation notes

Running log of decisions made while building the config scaffold. Anything that
deviates from the original spec, or that required a judgement call, is written
down here rather than left in the code for you to discover.

---

> ## ⚠ THIS FILE DOCUMENTS SPEC v3, WHICH IS RETIRED
>
> As of 2026-08-01 the study design is **v4**: a five-way categorical
> comparison (fluent-false, fluent-true, scrambled, POS-substituted,
> random-characters) plus twin, **60 runs**, injection fixed at **step 200**.
> The arm named `ordinary` no longer exists. See **`docs/spec-v4.md`**.
>
> Everything below still describes v3, and the code still implements v3.
> Nothing here has been deleted — the reasoning is still the record of how the
> scaffold got built, and most of the config, checkpoint and provenance work
> carries over unchanged. But **the arm list, the 40-run arithmetic, and any
> claim about the coherent-vs-noise contrast are stale.**
>
> What has to move to reach v4 is catalogued file by file in
> **`docs/v4-gap-analysis.md`**. The one measurement run ever performed is in
> **`docs/measurements/2026-07-31-match-sweep.md`**.
>
> **Known-false statement below:** S26 says `match_sweep.py` "measures the
> passage that script would actually produce." It does not — the two scripts
> use different rng conventions and produce different scrambled text from the
> same seed and k. Corrected in place at S26; analysed in the gap analysis.

---

## Open questions for you

Current status: 1 and 2 are settled. **3 is open and is the one with a
deadline** — it must be settled before the design freeze, because checkpoints
that were not written cannot be recovered afterward.

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

Now that `tie_embeddings` is known, a load-time parameter-count assertion is
computable — see "Not yet enforced" below for why it still is not in the loader.

(The old `checkpoint_interval` was the other always-required null at the time
this was written. It has since been decided and split in two; see "The
checkpoint schedule decision". `checkpointing` now has no null fields, and the
only values still undecided are the four in `injection`.)

### 2. Seeds are 0-indexed

The spec says "10 seeds" and gives `seed03_coherent` as an example, which is
consistent with either `0..9` or `1..10`. I used **`0..9`** (`seed00` …
`seed09`), matching Python and the usual ML convention, and the loader enforces
`0 <= seed < n_seeds`.

If you want `1..10` instead, it is a one-line change in
`scripts/generate_overrides.py` (`range(n_seeds)` → `range(1, n_seeds + 1)`) plus
the range check in `_validate_semantics`. Say the word and I will flip it.

### 3. Is one checkpoint inside the momentum window enough resolution?

**Open. Must be settled before the design freeze. Cannot be settled yet.**

This was briefly written up as answered by the checkpoint split. It is not, and
the earlier wording overstated the case. Restating it properly:

**What the split does establish.** `weights_only_interval: 50` samples the
neighbourhood of the burst **20× more densely** than the 1000-step interval
originally proposed, and does so at a fifth of the per-checkpoint cost, because
weights-only checkpoints are ~0.5 GB against ~1.5 GB for full ones.

**What it does not establish.** That 20× is a comparison against the old
proposal, not a measurement of sufficiency. In absolute terms, AdamW's momentum
smears a single batch's influence over a window of roughly 50 steps (Chang et
al., arXiv 2406.11813). At `weights_only_interval: 50`, that window contains
approximately **one** weights-only checkpoint.

One sample inside the smear window tells you where the arms stood *after* the
smear had played out. It does not tell you the *shape* of what happened during
it — whether the arms separated immediately and then converged, separated
monotonically, or crossed. Those are different claims about what the burst did,
and a single post-hoc sample cannot distinguish them. If the shape of the
divergence is part of what the study is measuring rather than just its endpoint,
50 is not obviously enough.

**Why the deadline is the design freeze, and not later.** Checkpoints that were
not written cannot be recovered afterward. Every other parameter in this repo
can be revisited by re-analysing existing artifacts; this one cannot. Choosing
too coarse an interval is not a decision that can be corrected without re-running
all 40 models — roughly 2,499,805,184 tokens × 40, which is the entire compute
budget of the study.

**Why it cannot be settled now.** The question is about resolution *near the
burst*, so it depends on `injection_step`, which is still null and comes out of
piloting. Until piloting fixes the injection step, there is no specific window
to reason about.

Two things to have in hand when it is settled:

- **The AdamW betas are not in the config.** `optimizer` currently holds only
  `name: adamw` and `weight_decay: 0.1`. The width of the momentum window is a
  function of β₁ and β₂ — with β₁ = 0.9 the first-moment timescale is
  1/(1−β₁) = 10 steps, and the second-moment timescale ranges from 20 steps at
  β₂ = 0.95 to 1000 at β₂ = 0.999. The "roughly 50 steps" figure cannot be
  checked against this config as written, and pinning the betas is a
  prerequisite for reasoning about it rather than an independent nicety.
- **A denser schedule near the burst is a schema change**, not just a different
  number. `checkpointing` currently expresses a uniform interval; a window of
  higher density around `injection_step` needs a different shape. Worth knowing
  before the freeze, since the freeze is what makes it expensive.

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

### D6. `burst_match.py` uses `torch.autograd.grad`, not `loss.backward()`

The spec says "the L2 norm over all parameter gradients from one backward pass".
It is one backward pass, but taken with `torch.autograd.grad(loss, params)`
rather than `loss.backward()` followed by reading `p.grad`.

The reason is that `backward()` **accumulates**. It adds into `p.grad` rather
than replacing it, so measuring `coherent.txt` and then `noise.txt` in the same
process would give the second file a gradient norm computed from the sum of both
passages unless something remembered to zero the gradients in between. That is a
silent wrong-number bug, and the number it produces is plausible — larger than
the truth, but not obviously so.

`autograd.grad` returns the gradients without writing to any stored state, which
deletes the failure mode rather than guarding against it. `test_gradient_norm_
does_not_accumulate_between_calls` asserts both that repeated calls agree and
that `p.grad` stays `None`, and
`test_determinism_survives_measuring_another_file_in_between` proves it end to
end on the real model: measure A, measure B, measure A again, and A must be
bit-identical to itself.

### D7. Determinism is guaranteed within a machine, not across machines

Requirement 1 is met as written — `model.eval()`, `torch.manual_seed`, and the
same file measured twice gives bit-identical numbers, asserted on raw floats
rather than printed strings in
`test_same_file_measured_twice_is_bit_identical`.

What that does **not** cover, and what the script prints in its header so you
can see it:

- **Thread count.** Torch's CPU reductions partition work by
  `torch.get_num_threads()`, so a different thread count sums the same floats in
  a different order and can change the low bits. The header prints the thread
  count for this reason. I did not force it to 1: that would slow every run
  down to buy a guarantee across machines that the differing CPU instruction
  sets would break anyway.
- **Torch version and hardware.** Also printed in the header.

The practical rule: numbers from one session are comparable with each other.
Do not compare a number in your notes from last month against one from today
without checking the header matches. This is a limitation of the measurement,
not of the passages, and it is why the comparison table exists — the
*differences* between passages measured in the same session are the trustworthy
output, not the absolute figures.

### D8. ~~The batch size is copied from `base.yaml`, not read from it~~ — RESOLVED

**Original deviation.** Requirement 4 needed the batch size to compute the
scaled figure, and the script was forbidden to touch the config system, so
`DEFAULT_BATCH_SIZE = 256` sat in `scripts/burst_match.py` as a constant copied
out of `configs/base.yaml`. If `training.batch_size` ever changed, nothing would
fail — the script would keep dividing by 256 and quietly print a wrong scaled
figure in every report.

**Resolved by lifting the constraint.** You removed "does not read the config
system" for this one value, which is what made the fix possible: the script now
*reads* `training.batch_size` and there is no constant to go stale. A test
greps the script for a literal `256` and for `DEFAULT_BATCH_SIZE`, so the
constant cannot quietly come back.

Two routes, in order:

1. **`burst.config.load_config`**, so the value arrives having passed the same
   validation a real run's config passes — the type check, the
   `batch_size × seq_len × total_steps == expected_token_budget` identity, all
   of it. This is the path the shipped `configs/base.yaml` takes today, and a
   test asserts that it does rather than merely that it could.
2. **A direct PyYAML read of that one key**, if and only if the loader refuses.

**Why route 2 exists at all.** The loader validates the whole file, so a config
that is mid-decision or that has drifted from the loader's schema fails as a
unit even though `training.batch_size` in it is perfectly readable. That is the
loader doing its job. The alternative — relaxing its validation so this script
could get one number out of it — would trade a real guarantee that 40 runs
depend on for a convenience in a measurement tool. Not worth it. The fallback
is narrow by construction: it navigates to exactly `BATCH_SIZE_KEY` and
validates exactly that value, and it must not be allowed to grow into a second,
weaker config loader.

**The header always says which route ran**, so a number produced without the
loader's validation is visible rather than inferred:

```
batch:   256 sequences  (...\configs\base.yaml, via burst.config loader)
batch:   256 sequences  (...\configs\base.yaml, direct YAML read -- burst.config declined: token budget mismatch: ...)
batch:   512 sequences  (--batch-size on the command line, overriding the config)
```

**No default, anywhere.** A missing key is an error naming the file, the dotted
key, and `--batch-size`. `--batch-size` still wins over both routes, and is
checked before the config is read so it works even when the config cannot be.

**Resolution order.** The batch size resolves *before* GPT-2 is loaded, so a
broken config costs an error message rather than a 500 MB download followed by
an error message.

**What this cost.** `scripts/burst_match.py` now imports `burst.config`. The
dependency runs one way only — nothing in `burst/` imports the script — so
`burst/config.py` still imports nothing heavier than PyYAML and still loads on
a machine with no ML stack. The batch-size tests run in the torch-free
environment for exactly that reason.

### D9. A second virtual environment, `.venv-ml/`

Requirement 5 asks for proof that the config tests still pass without torch. The
only way to prove that is to keep an environment that genuinely does not have
it, so `.venv/` is untouched (`pyyaml`, `pytest`, nothing else) and a second
environment `.venv-ml/` holds torch and transformers.

`.gitignore` needed a new line: the existing unanchored `.venv/` pattern does
not match the name `.venv-ml`, so without it a 2 GB environment would have shown
up as untracked in every `git status` — and, worse, in the `dirty_files` list
that `run_provenance.yaml` records.

```
.venv/       pip install -e ".[dev]"            156 config tests, no torch
.venv-ml/    pip install -e ".[dev,measure]"    everything
```

### D10. `burst_match.py` refactored twice so the new scripts could reuse it

The brief for step 8 said to import and reuse the measurement rather than
reimplement it, and to log the refactor if one was needed. Two were, both
extractions with no behaviour change:

- **`measure_text(text, source, tokenizer, model)`** split out of
  `measure(path, ...)`, which now reads the file and delegates. `match_sweep.py`
  generates a noise passage per window size in memory and would otherwise have
  had to write each one to a temporary file purely to hand it straight back.
  `Measurement.path` is consequently typed `Path | str` and holds a label such
  as `noise k=5` for generated text; `_short()` wraps it in `Path()` so table
  rows need no special case.
- **`load_tokenizer()`** split out of `load_model()`, sharing a new
  `_from_pretrained()` helper that holds the cache-first / announce-the-
  download / clear-failure logic both now use. `make_bursts.py` has to count
  tokens to match passage lengths but never runs the model, and making it
  download 500 MB of weights to do that would be silly. `load_tokenizer` also
  does not import torch — the tokenizer does not need it, so counting tokens
  now works in the base environment if `transformers` is present.

No measurement logic changed, and the existing tests for it were not touched.

### D11. The topic of `ordinary.txt` is whatever the seed selected

`--seed 0` selected, for the ordinary arm, a passage from a Wikipedia-derived
OpenWebText document about research on the black–white IQ gap. It passes every
quality filter — 95% alphabetic, ASCII, twelve sentence ends, unmistakably
ordinary prose — and the selection is exactly the blind seeded procedure the
brief asked for.

**Flagged rather than silently reseeded.** Quietly rerolling until the topic
looked innocuous would defeat the point of a blind selection procedure and
would not have been recorded anywhere. But the ordinary arm is the "normal
text" control, and a passage on race and IQ is not neutral ground to inject
into a model and then measure. Whether that matters for this study is a
judgment call belonging to whoever runs it, not to this script.

Changing it costs one flag: `--seed 1` selects different spans, and everything
downstream regenerates reproducibly. Recorded here so the choice is a choice.

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
suffixes `_dir`, `_path`, `_directory`, `_folder`. Substring matching would have
caught the checkpointing interval keys as false positives. There is a test
pinning that specific non-behaviour.

Note `burst_text_paths` is exempted by dotted name (`CONTENT_PATH_KEYS_EXEMPT`)
rather than by being un-matchable, so renaming it cannot slip it past both
checks — see S13.

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

### S16. The loss is computed here, not delegated to `labels=`

`transformers` will compute the loss for you if you pass `labels=input_ids` to
the model. `burst_match.py` computes it itself from the logits instead:

```python
shift_logits = logits[0, :-1, :].float()
shift_labels = input_ids[0, 1:]
F.cross_entropy(shift_logits, shift_labels, reduction="none")
```

Two reasons. The model would only hand back the *mean*, and requirement 1 needs
the per-token vector to report the standard deviation, the max and the five most
surprising tokens. And it pins the definition of "per-token loss" inside this
repo rather than inheriting whatever a given `transformers` version does about
label shifting — that behaviour has changed across major versions before, and a
silent change to the definition would move every number the script prints
without anything recording why.

The `.float()` is deliberate too: it makes the number independent of whatever
dtype the checkpoint happened to load in.

### S17. Population standard deviation, not sample

`loss_stats` divides by `n`, not `n - 1`. These tokens are not a sample drawn
from a larger passage — they are the whole passage — so the number describes it
rather than estimating anything. With a single prediction it is `0.0` rather
than undefined.

The test pins the choice explicitly: for `[0, 1, 2, 3, 4]` it asserts the result
is `sqrt(2)` (population) and asserts it is *not* `sqrt(2.5)` (sample), so
flipping the convention breaks a test rather than silently shifting a printed
column.

### S18. N tokens give N−1 losses, and both numbers are printed

The first token has nothing before it, so it is context and never a target. A
219-token passage has 218 next-token predictions and the mean loss is over 218
values, not 219. The report prints `tokens 219 (218 next-token predictions)`
rather than picking one and leaving you to guess which.

The `position` on each of the five most surprising tokens is the index of the
*predicted* token, so it is never 0, and it lines up with what you would count
in the token list rather than with the index into the loss vector.

### S19. Passages under two tokens are rejected

Not in the spec, same spirit as the context-limit check at the other end. One
token yields zero predictions, so there is no mean to take; without the check
that surfaces as an empty-tensor `nan` rather than as a message. Empty and
whitespace-only files are rejected for the same reason.

### S20. No `--model` flag

The model name is a module constant, not an argument. The whole value of this
script is that every candidate passage is measured against the same fixed
yardstick; a `--model` flag would let two passages be compared under different
rulers and there would be nothing in the output to reveal it. If your own model
eventually needs measuring, that is a different script with a different
provenance story.

### S21. The tests are split into torch-free and torch-only halves

`tests/test_burst_match.py` has 31 tests that need no ML stack — the
context-limit error, the length-mismatch warning and its factor, the loss
statistics, the batch scaling, the report formatting, and the batch-size
resolution added for D8 (PyYAML and `burst.config`, both already base
dependencies) — and 12 that need torch.
The torch ones are guarded with a `skipif` mark, deliberately **not** with
`pytest.importorskip` at module level: `importorskip` raises during collection
and would skip the entire file, silently including the 19 pure tests that exist
precisely so they can run without torch.

The tests needing the real GPT-2 weights skip through the fixture instead, so
the suite is green on a machine with no network and no model cache rather than
red for a reason that has nothing to do with the code.

The measurable logic was factored to make this split possible: the arithmetic
and the formatting are plain functions over plain numbers, and torch appears
only in `per_token_losses`, `gradient_norm`, `measure` and `load_model`.

### S22. Two span filters are stricter than the brief said

The brief asked to skip spans that are "mostly non-ASCII" and "mostly
punctuation or digits" — literally, above 50%. The thresholds shipped are 10%
non-ASCII and 25% punctuation-or-digits.

A span that is 45% CJK or 40% digits is not ordinary English prose, and the
ordinary arm's whole job is to be indistinguishable from the training
distribution. There are far more clean spans in the pool than the two needed
(0 of the candidates examined at `--seed 0` were rejected), so the strict
reading costs nothing and the loose one would eventually admit a page of
tables. The two thresholds the brief gave as numbers — 40% alphabetic, 3
sentence-ending marks — are used exactly as given. All four are constants at
the top of `make_bursts.py` and are written into `provenance.json`.

### S23. No trailing newline on any burst file

`bursts/coherent.txt` is 962 bytes, ends with `collaboration.`, and has no
terminating newline. That is the strict reading of "verbatim, no trailing
additions", and it is also the only version that works: a trailing newline is
**a token** under the GPT-2 tokenizer, so a file ending in one tokenizes one
longer than a file that does not. Since N is defined by tokenizing
`coherent.txt` and the other two are built to match it, the convention has to
be identical across all three, and "none" is the only one that keeps N equal to
the token count of the text itself.

`write_text()` passes `newline="\n"` explicitly. Without it Python on Windows
translates every `\n` to `\r\n`, the files stop being byte-identical to the
same files generated on the cluster, and `.gitattributes`' `eol=lf` rewrites
them on the next checkout anyway. A test asserts no CR bytes and no trailing
newline in all three files.

### S24. `--k` is required, with no default

Which window size to use is explicitly out of scope for this step, so
`make_bursts.py` will not run without being told one. A default would have
become the answer by sitting there. `--seed` does default to 0, because *some*
seed has to be picked for the run to be reproducible at all and the value is
recorded in `provenance.json` either way.

For the same reason `match_sweep.py` prints its table and stops. It does not
rank rows, mark a best k, or say anything about matching — the tolerance is not
set, and a script that highlighted a row would be setting it.

### S25. `provenance.json` carries no timestamp

Acceptance required that two runs with the same arguments produce identical
files, and `provenance.json` is one of the files. A `written_at_utc` field —
which `run_provenance.yaml` does carry, correctly, because a run happens at a
time — would have made every run differ from every other. Nothing
machine-dependent goes in either, for the same reason. What it does record:
seed, k, N, the dataset and split, both source spans as (document index, word
offset, word count), all four filter thresholds, the SHA-256 of each file, and
whether trimming cut mid-word.

The corpus slice itself is cached under a gitignored `/.corpus-cache/` and is
never committed — corpus data is data. The burst texts cut from it are content,
and are committed. That distinction is the one CLAUDE.md rule 3 says needs
judgment; this is the worked example of it.

### S26. A fresh rng per k in the sweep

`match_sweep.py` builds a new `random.Random(seed)` for each window size
rather than threading one rng through the loop. Otherwise the shuffle at k=5
would depend on how many draws k=3 happened to consume, and the rows would not
be independent measurements of the same dial — reordering the `--k` list would
change the numbers.

The span selection rng is separate again, and is driven through the same
`select_spans()` call `make_bursts.py` uses, so the sweep measures the passage
that script would actually produce rather than a similar-looking one.

> **CORRECTION, 2026-08-01. The last sentence above is false.** The sweep does
> *not* measure the passage `make_bursts.py` produces.
>
> Both scripts select the same span. They diverge at the shuffle:
> `make_bursts.py` threads **one** rng through span selection and then the
> shuffle, so the shuffle starts from a state already advanced by selection;
> `match_sweep.py` builds a **fresh** `random.Random(seed)` per k. Same seed,
> same k, different word order. Verified by direct comparison against
> `bursts/noise.txt`.
>
> Consequence: the noise rows of the 2026-07-31 sweep describe passages that
> exist nowhere on disk, and `bursts/noise.txt` has never been measured.
> `ordinary.txt` is unaffected (never shuffled) and `coherent.txt` is never
> generated at all.
>
> **Not fixed.** Both conventions are defensible and the choice is open; the
> two candidates are laid out in `docs/v4-gap-analysis.md` §4. Under v4 this
> matters more than it did here, because with five arms generated in one run,
> convention A makes each arm's text depend on how many draws the arms before
> it consumed.

---

## Cross-module obligations

Decisions one module has to make on behalf of a module that gets built later.
Each one is cheap to honour at the right moment and expensive-to-impossible to
retrofit, which is why the reason is written down and not just the rule.

### 1. Reserve a held-out slice when the corpus is tokenized

**Owner: the data pipeline. Deadline: tokenization, i.e. the first time the
corpus is touched.**

The metrics module computes a loss barrier by interpolating two finished models
and evaluating the blend. Evaluating a blend requires text that **no run trained
on** — otherwise the barrier measures memorisation rather than the shape of the
loss landscape between the two solutions.

The ordering problem: the metrics module is built *before* the data pipeline in
our sequence. So at the moment the requirement becomes visible, the module that
has to satisfy it does not exist yet, and by the time the data pipeline is
written the requirement is easy to forget. If the held-out slice is not carved
out during tokenization, it does not exist when the metrics module needs it, and
the only fix is re-tokenizing the corpus and re-running all 40 models.

**The held-out slice must be byte-identical across all 40 runs.** A barrier is a
number on an evaluation set; two barriers computed on different held-out text
are not comparable, and the study compares barriers between arms and between
seeds constantly. That means the slice is chosen once, deterministically, and
recorded — not sampled per run, and not derived from a per-run seed.

Note the interaction with `corpus.expected_token_budget` (2499805184): that
figure is the number of tokens **trained on**. If the held-out slice is carved
out of the same 2.5B slice, the training budget shrinks and the token-budget
assertion in the loader starts failing. Reserve it from *outside* the training
slice, or the arithmetic check and the reservation will collide. Decide which
before tokenizing.

### 2. Save RNG state in every checkpoint

**Owner: the training loop. Deadline: the first time a checkpoint is written.**

A checkpoint must persist the random number generator state alongside weights,
optimizer state and step number.

Without it, a run that dies at step 6000 and resumes from the step-5500
checkpoint diverges from a run that was never interrupted — same seed, same
config, different trajectory, because the RNG restarts from a different point in
its stream. That breaks the bit-identical guarantee the entire study rests on,
and it breaks it *silently*: the resumed run finishes normally and produces a
checkpoint that looks exactly as legitimate as any other. Nothing downstream can
detect it. On a 40-run study on shared cluster hardware, at least one
preemption is close to certain, so this is not a hypothetical.

Everything stateful has to be captured, not just torch:

- Python's `random` — `random.getstate()`
- NumPy — `np.random.get_state()`, plus any explicit `Generator` objects
- torch CPU — `torch.get_rng_state()`
- torch CUDA — `torch.cuda.get_rng_state_all()` (all devices, not just device 0)
- DataLoader worker state, if `num_workers > 0`. Workers are separately seeded
  processes; restoring the parent's RNG does not restore theirs. This usually
  means recording the epoch/iteration position and the base seed so worker
  seeding is reproducible, rather than trying to serialise worker RNGs directly.

Related, and cheap: on resume, verify the loaded checkpoint's step number and
config hash against the run being resumed, and refuse to resume across a
mismatch.

### 3. ~~`checkpoint_interval` is a storage decision~~ — RESOLVED

**Resolved 2026-07-31.** Split into two fields:

```yaml
checkpointing:
  weights_only_interval: 50      # ~0.5 GB each
  full_interval: 1000            # ~1.5 GB each
```

plus a rule that the final step always writes a full checkpoint. Full detail in
"The checkpoint schedule decision" below. **`checkpointing` no longer has any
null fields, and `checkpoint_interval` no longer exists as a key anywhere** —
the loader rejects it by name with a message pointing at the two replacements.

What is resolved here is the *storage* question: the values are decided, they
are validated, and they fit the budget. A separate question — whether 50 steps
is enough *resolution* to measure the trajectory near the burst — was spun out
of this one and is **still open**; see open question 3. Do not read this
RESOLVED marker as covering it.

The original sizing note is kept below because the per-checkpoint arithmetic is
still exactly what the new numbers are built on.

Per-checkpoint size, at 124,439,808 parameters in fp32:

```
weights                 124439808 x 4 bytes            =  0.50 GB
AdamW exp_avg           124439808 x 4 bytes            =  0.50 GB
AdamW exp_avg_sq        124439808 x 4 bytes            =  0.50 GB
RNG state, step, config                                 negligible
                                                          -------
full checkpoint                                        ~= 1.50 GB
```

That is why the figure is ~1.5 GB and not ~0.5 GB: the optimizer state is two
more full copies of the parameters, and it has to be saved for a checkpoint to
be resumable at all (see obligation 2).

Across `total_steps: 9536` and 40 runs, at 1.5 GB per checkpoint, TB = 1000 GB:

| interval | checkpoints/run | per run | all 40 runs | of 10 TB |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 95 | 142.5 GB | **5.70 TB** | 57% |
| 200 | 47 | 70.5 GB | 2.82 TB | 28% |
| 250 | 38 | 57.0 GB | 2.28 TB | 23% |
| 500 | 19 | 28.5 GB | **1.14 TB** | 11% |
| 1000 | 9 | 13.5 GB | 0.54 TB | 5% |

(Counts are `floor(9536 / interval)`; add one checkpoint per run if a final
checkpoint at step 9536 is written separately, which adds 60 GB across the
study at any interval.)

So both ends are affordable against 10 TB — 100 leaves 4.3 TB of headroom, 500
leaves 8.9 TB. The constraint is not really capacity, it is that the headroom
also has to hold the tokenized corpus, the held-out slice, and any re-runs. A
single full re-run of the study at interval 100 would need another 5.7 TB and
would not fit.

Two things worth deciding at the same time:

- **Does the interval need to be denser near `injection_step`?** **Still open —
  see open question 3.** The split improves the sampling density near the burst
  by 20× relative to the old 1000-step proposal, at a fifth of the
  per-checkpoint cost. That is a comparison, not an answer: it does not
  establish that 50 steps is *sufficient* resolution, only that it is much
  better than 1000.
- **Is fp32 optimizer state necessary?** Still open. Storing
  `exp_avg`/`exp_avg_sq` in bf16 would cut *full* checkpoints to ~1.0 GB. Under
  the schedule below that saves only 5 GB per run (full checkpoints are now a
  small share of the total), so the pressure is off. Still a numerics decision
  that interacts with `determinism: true`, and it belongs with the training
  loop.

---

## The checkpoint schedule decision

**Decided 2026-07-31. Owner: me. No nulls left in `checkpointing`.**

```yaml
checkpointing:
  weights_only_interval: 50      # weights + step number,            ~0.5 GB
  full_interval: 1000            # + optimizer state + RNG state,    ~1.5 GB
```

### Why two schedules instead of one

The two reasons to save a checkpoint have very different costs, and a single
interval forced them to share the expensive one.

- **Full checkpoints exist for crash recovery.** They carry weights, optimizer
  state, the step number and the RNG state — everything needed to resume a dead
  run bit-identically (see obligation 2 above). AdamW's two moment buffers are
  each another full copy of the parameters, which is why a full checkpoint is
  ~1.5 GB and not ~0.5 GB.
- **Weights-only checkpoints exist to measure how the model changes during
  training.** They carry the weights and the step number, nothing else.

**Every metric in this study is a function of the weights alone.** So
weights-only checkpoints are sufficient for measurement — and that is precisely
what makes a 50-step interval affordable. The same sampling density at
full-checkpoint size would cost three times as much and would not fit alongside
the corpus and any re-runs.

### The two rules

**Precedence.** When both intervals fire on the same step, write the **full**
checkpoint only. A weights-only file at that step would duplicate data already
inside the full one. This is why `full_interval` must be an exact multiple of
`weights_only_interval`; the loader rejects any other pairing, because otherwise
the two schedules drift and "both fire on the same step" becomes an accident of
arithmetic rather than a rule.

**Final step.** The last step always writes a **full** checkpoint regardless of
interval. 9536 is divisible by neither 50 nor 1000, so without this rule the
finished model — the thing every headline comparison in the study is computed
on — would never be written at all. The rule keys off the computed last step
(`total_steps - 1`), never a literal 9535, so it survives a change to
`total_steps`. There is a test that changes `total_steps` and asserts the rule
follows it.

### Resulting storage

Counting steps 0..9535, with a checkpoint due at step `s` when
`(s + 1) % N == 0`:

```
weights-only firings    9536 // 50   = 190
full firings            9536 // 1000 =   9      (all 9 are also wo firings)
final-step rule         9536 % 1000 != 0        -> +1 full
```

| kind | count | each | per run |
| --- | ---: | ---: | ---: |
| weights-only | 190 − 9 = **181** | 0.5 GB | 90.5 GB |
| full | 9 + 1 = **10** | 1.5 GB | 15.0 GB |
| **total per run** | 191 | | **105.5 GB** |
| **all 40 runs** | 7,640 | | **4.22 TB** |

That is **42% of the 10 TB budget**, leaving 5.8 TB for the tokenized corpus,
the held-out slice (obligation 1) and re-runs. A second full pass of the study
would not fit alongside the first; one arm re-run (10 runs, ~1.06 TB) would.

The loader computes all of this from the config rather than storing it —
`cfg.checkpoint_plan` exposes `last_step`, `weights_only_count`, `full_count`,
`estimated_bytes_per_run` and `estimated_bytes_all_runs`, and it is written into
`run_provenance.yaml`. There is a test that walks all 9,536 steps calling
`checkpoint_kind_at()` and asserts the brute-force counts match the closed-form
ones, so the formula cannot quietly drift from the rule.

The 0.5 GB and 1.5 GB figures are **estimates, not measurements** — stated as
such in a comment beside the constants. Nothing in this repo has ever written a
checkpoint. Framework overhead, compression, and bf16 optimizer state would all
move them.

### What the training loop owes this

`Config.checkpoint_kind_at(step)` returns `"full"`, `"weights_only"` or `None`
and is the single definition of the schedule. It writes nothing — it is a pure
function of the config — but **the training loop must call it, or reproduce it
exactly.** Both rules above are currently enforced only in the sense that the
loader computes counts consistent with them; nothing writes a file.

Concretely, the training loop still owes:

- **The precedence rule an actual implementation.** Nothing stops a training
  loop from checking `step % 50 == 0` and `step % 1000 == 0` independently and
  writing both files. That would silently inflate storage by 15 GB per run and
  put two files at the same step whose relationship nothing documents.
- **The final-step rule an actual implementation.** This is the one that
  matters most: skip it and there is no finished-model checkpoint at all, and
  every headline comparison in the study has nothing to run on. It must compute
  `total_steps - 1` rather than hardcode 9535.
- **Honouring the 0-indexed convention**, including that step 0 is not a
  checkpoint step — see below.

---

## Step indexing: 0-indexed, and step 0 is not a checkpoint step

Recorded because it was load-bearing well before it was written down anywhere.

**The repo already committed to 0-indexed steps**, implicitly. `_validate_semantics`
range-checks `injection_step` as `0 <= step < total_steps` and its error message
reads `must be within 0..9535`. Nothing anywhere implied 1-indexing. So:

- the first optimizer step is **0**
- the last is **`total_steps - 1` = 9535**
- `Config.last_step` is the only place that arithmetic appears

**A checkpoint is due after every N *completed* steps**, i.e. at step `s` when
`(s + 1) % N == 0`. With `weights_only_interval: 50` the first checkpoint lands
at step **49**, not step 0 and not step 50.

**Step 0 is therefore never a checkpoint step.** This was already assumed and
never stated: the storage table in the original version of this file counted
`floor(9536 / interval)` firings, which is only correct if step 0 is excluded —
including it would have given one more per run. The convention is now explicit
in `configs/base.yaml`, in the README, and in `checkpoint_kind_at()`.

The training loop must honour all of this. If it 1-indexes its own step counter,
`injection_step`, the checkpoint schedule and the final-step rule all land one
step off, and nothing in this repo would detect it.

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
- No dependencies beyond `pyyaml`, `pytest`, and the standard library — for
  `burst/` and `configs/`. That has not changed and must not.
- `torch==2.13.0`, `transformers==5.14.1` — pinned the same way, in the
  **optional** `measure` group, for `scripts/burst_match.py` alone. Installed
  into a separate `.venv-ml/` so that the config suite can be *shown* to pass
  without them, not merely asserted to. See D9.
  - On a CPU-only machine, install torch from PyTorch's own index first:
    `pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu`
    — about 250 MB against about 2.5 GB for the default CUDA build on Windows.
    The `==2.13.0` pin is satisfied by the `2.13.0+cpu` wheel; PEP 440 ignores
    the local version segment when the specifier does not name one.
  - GPT-2's weights are a further ~500 MB, downloaded on first run to
    `C:\Users\<you>\.cache\huggingface\hub` (the script prints the path).
- Built and tested on Windows. Note that `--outdir /tmp/testrun` on Windows
  resolves to `C:\tmp\testrun`; on the cluster it is the real `/tmp/testrun`.
  Nothing in the loader cares.

## Test coverage

227 tests: 156 for the config system, 43 for `burst_match`, 28 for
`make_bursts` and the committed burst files.

In the base environment (`.venv/`, no torch) the run is **212 passed, 15
skipped** — the 156 config tests are untouched and unaffected, and only the 15
that genuinely need torch or `transformers` skip. That is the evidence for
requirement 5.

### `burst_match` specifically

- **determinism** — `test_same_file_measured_twice_is_bit_identical` compares
  raw floats, not printed strings, because printing rounds and rounding would
  hide a difference in the low bits. `test_determinism_survives_measuring_
  another_file_in_between` measures A, then B, then A again, which is the test
  that would fail if gradients accumulated. `test_model_is_in_eval_mode` pins
  the reason it works at all.
- **the over-context error** — both against the pure check
  (`test_over_context_raises_and_names_the_actual_count`, plus boundary tests at
  exactly 1024 and at 1025) and against the real tokenizer on a real ~1500-token
  file, asserting the message names the true count.
- **the unequal-length warning** — that it fires, that it does not fire when the
  counts match, that it states the correct weighting factor, and that it picks
  the extremes when three passages are compared.
- **loss statistics on hand-checkable input** — `[0,1,2,3,4]` → mean 2,
  std `sqrt(2)`, max 4; and, on the torch side, uniform logits over 8 classes →
  exactly `ln(8)` for every token, whatever the tokens are.
- **the gradient norm** — against a hand-computed one-parameter case
  (`L = 4w²`, `dL/dw = 8w = 24` at `w = 3`).
- **the batch size coming from the config** (D8) — that editing a throwaway
  config changes the number the script uses, that the shipped `base.yaml` goes
  through the validated loader path rather than the fallback, that a config the
  loader rejects still yields the key *and* says so in the source string, that a
  missing key raises rather than defaulting, that `--batch-size` wins, and that
  a literal `256` has not crept back into the script.

### `make_bursts` and the burst files

- **the window shuffle** — reproducible for a given seed and k, different for a
  different seed or k, every word preserved, and — the property that makes k a
  dial at all — no word crossing a window boundary. The final partial window is
  asserted to actually get reordered, not merely to keep its words.
- **the span filters** — prose passes; non-ASCII, digit-and-punctuation, and
  navigation-bar spans are each rejected by the filter that should catch them.
  Fractions are asserted to ignore whitespace, so they do not depend on how the
  page was wrapped.
- **the guard on `coherent.txt`** — that pointing an output path at it exits 1
  before anything loads, and that a *missing* coherent passage is an error
  rather than an invitation to generate one.
- **the committed files themselves** — all three present, no CR bytes, no
  trailing newline, SHA-256 matching `provenance.json`, and (tokenizer
  permitting) all three at the same token count, equal to the recorded target.
  These are step 8's acceptance criteria turned into something that keeps
  being checked rather than something that was true once.

### Config system

The five originally required
cases are
`test_typo_in_override_key_raises`,
`test_null_injection_fields_raise_for_injecting_arm` /
`test_null_injection_fields_are_fine_for_twin`,
`test_invalid_arm_raises`,
`test_token_budget_mismatch_raises_when_total_steps_changes`, and
`test_output_path_key_in_config_raises`.

The rest cover the YAML traps, duplicate keys, immutability, the provenance
files, the overwrite guard, the schema check, burst text path containment,
corpus path rejection, the checkpoint schedule, all 40 generated overrides
loading, a mechanical check that the 40 runs differ *only* in seed and arm, and
the CLI end to end.

### Where the null-required-field coverage lives

It has been retargeted twice, never deleted, because it keeps landing on
whichever value is currently undecided:

- originally `tie_embeddings` → decided, so
  `test_null_tie_embeddings_would_still_raise` now sets it back to null in a
  throwaway config to keep the *check* under test
- then `checkpoint_interval` → removed, so
  `test_twin_still_requires_the_checkpoint_intervals` and
  `test_checkpoint_intervals_required_by_every_arm` (parametrized over all four
  arms) now null out the two replacement fields explicitly, plus
  `test_either_checkpoint_interval_null_alone_raises` proves each field is
  required independently rather than only as a pair

The general principle: when a value gets decided, the test that proved it was
required gets pointed at a throwaway null config rather than removed. Otherwise
deciding a value silently deletes the validation that guarded it.

### Checkpoint schedule coverage specifically

- validation: non-positive intervals (both fields × 0, −1, −50), non-multiple
  `full_interval` (1010, 51, 999, 75, 1049), accepted multiples (50, 100, 1000,
  1500, 2000, 9550), float interval rejected
- the removed key: `checkpoint_interval` in the base, in an override, and at
  top level — each asserting the error names *both* replacements
- precedence: `checkpoint_kind_at()` returns `"full"` at every step where both
  fire, `"weights_only"` where only the 50 fires, `None` in between
- the final-step rule, including a case with `total_steps: 2000` proving it
  tracks the config rather than a hardcoded 9536
- `test_checkpoint_plan_counts_match_a_brute_force_walk` walks all 9,536 steps
  and asserts the closed-form counts equal the walked ones — this is the test
  that stops the formula and the rule from drifting apart
- four plan edge cases: last step is neither firing / is a full firing / is a
  weights-only firing that gets promoted / intervals larger than the whole run

Tests build a throwaway copy of the real `configs/base.yaml` and edit one thing,
rather than using a hand-written fixture — so they break if `base.yaml` drifts,
which is the point.
