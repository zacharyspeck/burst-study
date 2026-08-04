# Working rules for this repo

Standing instructions for every session here, regardless of what the task is.

This is a research repo whose central claim is that 70 training runs were
identical except for `seed` and `arm`. (10 seeds x 7 arms. It said 40 until
2026-08-03, when the arm list was reconciled to spec v4 and the run count
became a computed number rather than a typed one.) Most of the rules below exist because
that claim is only as good as the record backing it.

---

## 1. Commit and push after every completed task

Not at the end of the session, not when it seems worth it — after each task is
finished and verified. Work that is only in the working tree is work that no
provenance record can point at.

This matters more here than in most repos: `burst/config.py` stamps the current
commit hash and a dirty-tree flag into `run_provenance.yaml` on every load. A
run launched from an uncommitted tree records a hash that does not describe the
code that produced it, and the repo warns loudly about exactly this. Leaving
changes uncommitted arms that trap for the next person who launches something.

Before staging, print `git status --porcelain` and list every path you intend
to stage with a one-line reason. Do not stage with `git add -A` or `git add .`;
name the paths.

## 2. Keep `implementation-notes.md` current

It is the record of *why*, and it is the only thing that survives the reasoning
being forgotten. As you work:

- Log any **deviation** from what was asked under `## Deviations from the spec`,
  with the reason, then keep going. If something would change the structure of
  what was asked, stop and ask instead.
- Log **smaller decisions** — a convention chosen, a check added, a trap
  avoided — under `## Smaller decisions, logged as instructed`.
- When a deviation is later resolved, **mark it resolved in place** with what
  replaced it. Strike the old heading, keep the original text. Do not delete it:
  the fact that a shortcut existed, and what it cost, is part of the record.
- Keep the test counts in `implementation-notes.md` and `README.md` accurate
  when tests are added. A stale count is a small lie that makes the others
  easier to tell.

## 3. Never commit generated or downloaded bulk

None of the following belongs in git, at any size:

- **Checkpoints** — `.pt`, `.bin`, `.safetensors`, optimizer state, anything
  under a run output directory
- **Corpus data** — tokenized or raw; the corpus is *named* in the config,
  never located or vendored
- **Virtual environments** — `.venv/`, `.venv-ml/`, `venv/`
- **Model caches** — the HuggingFace cache (`~/.cache/huggingface`), which
  `scripts/burst_match.py` downloads GPT-2 into
- **Scratch text files** — throwaway passages used to exercise a script

The distinction that actually needs judgment: **burst texts are content, not
data.** The text injected into a run is the independent variable of the study,
so it is committed, and `run_provenance.yaml`'s commit hash is what covers it.
A throwaway file used to smoke-test `burst_match.py` is not. If you are unsure
which one you are holding, ask rather than assume — committing scratch is
noise, and failing to commit a real burst text breaks reproducibility.

Check `.gitignore` covers new tooling before it produces output. Note that the
unanchored `.venv/` pattern does **not** match `.venv-ml/`; that needed its own
line.

## 4. Leave the repo working, and say what you touched

Before reporting a task done:

- Run the full test suite in **both** environments and report both counts:
  ```
  .venv/Scripts/python.exe -m pytest -q      # no torch: expect skips, not failures
  .venv-ml/Scripts/python.exe -m pytest -q   # torch + transformers: expect no skips
  ```
  The torch-free run existing at all is the point — `burst/` must keep loading
  on a machine with no ML stack, and the skips are the evidence.
- Run whatever you changed, on real input, and paste the actual output. Not a
  description of the output.
- **List every file touched**, including the ones edited only to keep a
  document honest.
- Report failures as failures. A test that fails, a step skipped, a thing left
  undone — say so plainly, with the output. Do not round a partial result up.

---

## Boundaries that hold unless a task explicitly lifts them

- `burst/config.py` raises `ConfigError`; it never `assert`s. `python -O`
  deletes asserts, and validation that vanishes under an optimizer flag is
  worse than none.
- `burst/` imports nothing heavier than PyYAML. Loading a config has to work on
  a login node with no GPU. `scripts/burst_match.py` imports `burst.config`;
  the dependency runs that way only, never back.
- Output paths are command-line arguments, never config values. One config file
  must run unchanged on a laptop and on a cluster.
- Run override files may set `seed` and `arm` and nothing else. That restriction
  is the study's central claim, expressed as code.
- Never relax the loader's validation to make something else convenient. If the
  loader legitimately refuses a file, work around it narrowly and say so in the
  output — see D8 in `implementation-notes.md` for the worked example.
- Regenerate override files with `python scripts/generate_overrides.py`; never
  hand-edit them. `--check` verifies nobody did.
