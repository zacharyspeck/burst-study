# Contradiction scan — everything built since `d049f57`

**Run:** 2026-08-04T01:03Z
**Window:** commits `7863d7a`, `c2df6c7`, `fd40dec`, `21e02bf`, `48bd728`, `02012eb`
**Method:** six auditors over disjoint areas, then one adversarial verifier per
finding instructed to *refute* it. 59 agents, 0 errors.

**This report is stale the moment any commit lands.**

| | |
| --- | --- |
| candidate findings | 30 |
| survived adversarial verification | **24** |
| refuted | 6 |
| fixed in this pass | 19 |
| reported, not fixed | 5 |

Six were refuted by the verifiers — mostly cases where two claims shared a word
but described different things, or where a line was a deliberately-kept
historical record under CLAUDE.md rule 2. Those are not listed.

---

## Fixed

### The S70 pattern recurred, inside the module that claims immunity to it

`scripts/metrics_report.py:517-524` printed a **hardcoded** banner:

> "the curve sags below the chord rather than rising above it and max_excess is 0"

Twenty-five lines below, in the same rendered file, its own tables said the
curve rose above the chord on **9 of 10** seeds for `identical` and **8 of 10**
for `zeroed_block`, with a `zeroed_block` max_excess of **9.6e-03**.

That module's docstring says the S70 pattern "is built in here from the first
commit rather than retrofitted after the sixth." It recurred anyway, in the one
block that was static rather than derived. **The banner now reads the data**,
and `docs/measurements/10-metrics.md` was re-rendered.

This is the sixth instance of the pattern in this build.

### `launch.py`'s fifth state was unreachable

`CONFLICT` was documented in the module docstring, listed among "every state is
a fact about files", and asserted in `tests/test_launch.py:16` as one of "all
five states built as REAL FILES". **It could never fire.** The only thing that
raised it was `_write_provenance`, and `build()` loads with
`write_provenance=False` — deliberately, so that classifying a run does not
create its own output directory.

So the state was documented, referenced by a test docstring, and dead.

Fixed by making it real: `conflicting_run()` reads the outdir's
`resolved_config.yaml` and reports when it holds a different seed/arm. Verified
firing:

```
conflict detected: holds a run for seed=7 arm='fluent-false', but this is seed=0 arm='twin'
no conflict when it matches: None
```

### `launch.py` claimed a durability guarantee the code does not make

The docstring said `save_checkpoint` "writes `<name>.partial`, **fsyncs**, then
`os.replace`". `scripts/train.py:310-312` does `torch.save` then `os.replace`
with no flush and no fsync.

The rename is still atomic against *this process dying*, which is the case that
matters for a killed run. It is **not** durable against a host crash: the
rename can reach disk before the data. Corrected in place, with the consequence
stated — `done` means "this process wrote and renamed it", not "it survived a
power cut". Adding the fsync is a change to step 12's module and is **not**
made here.

### The `--blocks` help quoted the timing I already knew was wrong

`corpus_tokenize.py:461` said "One shard measured at 7.2s, so 50 is about
360s — comfortably inside the ~600s cap". The real build measured **11.0s to
26.7s, median 13.0s**. At 13s, `--blocks 50` is ~650s and would be *killed* by
a 600s cap — the argument reversed.

S74 already carries the correction; the help string did not. Now it does.

### `corpus_verify.py` claimed a third route that does not exist

Its docstring advertised "THE TOTAL, three ways… recomputed as
`batch_size * seq_len * total_steps` **from the config**". The module imports
neither `yaml` nor `burst.config` — deliberately, because it has to run on a
cluster with nothing but itself and the shards. There were always **two**
routes. Corrected in the docstring and in `check_total`.

### Documentation asserting a world that no longer exists

Fourteen further fixes, all mechanical corrections to prose describing the
retired v3 design:

| file | was | now |
| --- | --- | --- |
| `CLAUDE.md:5` | "40 training runs" | 70, with the reason |
| `README.md:545` | `4_220_000_000_000 (4.22 TB)` | `7_385_000_000_000 (7.385 TB)` |
| `README.md` ×4 | "all 40 runs / 40 override files" | 70 |
| `README.md:401` | arm is one of `coherent, noise, ordinary, twin` | the seven v4 names |
| `README.md:465` | "the four names" | the seven names |
| `README.md` injection table | five rows, all **null** | four decided values + six real paths |
| `README.md` | *(no table for the undecided fields)* | added: `micro_batch`, `dtype`, `adamw_impl`, with why each has no default |
| `README.md:74` | "Four values are still undecided" | three |
| `README.md:492` | "Suggested home: `configs/burst_texts/`" | `bursts/`, noting the suggested dir was never built |
| `README.md` §burst_match | documented `coherent.txt`/`noise.txt` | bannered: those files do not exist |
| `README.md` run names | `seed03_coherent`, `seed05_noise.yaml` | v4 names, hyphens noted |
| `configs/base.yaml:231` | "Everything here is UNSET ON PURPOSE" | decided, with the retired arm names removed |
| `implementation-notes.md:9` | "five-way… 60 runs… the code still implements v3" | six-way, 70, code implements v4 |
| `docs/decisions-pending.md:8` | "Nothing here has been acted on" | D-1 and D-2 were ruled and both steps removed |
| `metrics_report.py:326` | "rose above the chord on half the seeds" | 9 of 10 |

**The worst of these was `README.md:67`**: every copy-paste command in the file
— including the repo's own acceptance command — named
`configs/runs/seed03_coherent.yaml`, which was deleted when the arm list was
reconciled. Anyone following the README got "No such file or directory" on the
first command. Verified fixed by running it.

---

## Reported, not fixed

Five confirmed findings left alone, each with the reason.

### 1. `corpus_report.py:269` overstates the source by two files

`source["files_used"]` is `len(fetch_manifest["files"])` — the number of
Parquet files **fetched**, which is 25. The tokenizer's own manifest records
the 149th shard ending inside file **23**, so only 23 were consumed and the
23rd only partly.

The name says "used" and the value counts "fetched". **Not fixed** because
correcting it changes a committed measurement record
(`docs/measurements/11-corpus.json`), which should be re-derived rather than
edited, and re-deriving needs the corpus present. The honest fix is to compute
it from `blocks["train-148"].source_position_after.file_index + 1`.

### 2. `analysis.py:271` — `student_t_sf` returns the two-sided tail

`sf` means the one-sided survival function everywhere else, including
`scipy.stats.t.sf`. This returns `P(|T| ≥ |t|)`, so `student_t_sf(0.0, 9)` is
`1.0` where scipy's `sf` gives `0.5`.

The docstring says "Two-sided survival" on the next line and the scipy
cross-check test compares against `2 * scipy.stats.t.sf(...)`, so nothing is
numerically wrong. **Not fixed** because renaming a function is a change to
tested code with no behavioural defect, and the queue is the right place for
"is this name worth churning". Worth renaming to `student_t_two_sided_sf`.

### 3. `analysis.py:21-23` — "wider by construction" is not a construction

The docstring says the noise floor "is the wider of the two by construction".
The verifier showed a counterexample from the module's own test helpers: with
`twin_jitter=0.5` the floor's widest is 0.98 while a fabricated effect of 5.0 is
larger. It is wider *in expectation under the null*, not by construction.

**Not fixed** because the correct phrasing is a claim about the design that I
should not word unilaterally — it bears on how "clears floor: yes" is read. See
the queue.

### 4. `analysis.py` — the noise-floor labels hardcode "twin"

`report_banner` and `build_provenance` print "twin-vs-twin ACROSS seeds" while
the number beside them is computed against `--reference`, a free parameter. Run
with `--reference fluent-false` and the label lies.

**Not fixed** because the resolution is a decision: either pin `reference` to
`twin` (matching the spec) or derive the label. Queued.

### 5. `canonicalization_error.py` — the gauge-subspace caption

The "gauge subspace, counted exactly" block reports a continuous total of
1,216,512 (0.9776% of parameters) captioned as the fraction of an isotropic
perturbation's energy the ruler removes. But the shipped recipe removes
*neither* of the two terms making up ~98.5% of that total — `head_internal` and
`layernorm_gain` are both in `NOT_QUOTIENTED` after D-1 and D-2.

So the number describes a recipe that no longer ships. **Not fixed** because
correcting it means re-deriving a committed step 9 measurement, and the caption
change is entangled with what D-3 decides about the head sort.

---

## What the scan did not find

No finding in `scripts/train.py`'s step order, `rng_state.py`, `model_seam.py`,
`data_order.py`, or `corpus_spec.py`'s geometry. The injection hook's claims
about consuming no sampler index and drawing no randomness were checked and
hold. The test-count table in `implementation-notes.md` sums to 821 and matches
a real `pytest --collect-only`. `launch.py` parses the exact filename format
`train.py` writes, verified by round-trip.
