#!/usr/bin/env python
"""Emit runnable commands for the study's runs. Step 15.

IT EMITS. IT DOES NOT ORCHESTRATE.

This prints command lines and writes them to a file. It starts no processes,
assigns no devices, manages no queue, and retries nothing. A program that emits
survives SLURM, bare SSH, a rented box, `xargs -P`, or being pasted by hand; a
program that manages processes only works where it was written, and nobody has
provisioned the hardware yet.

There is also a reason beyond portability, and it comes from the training loop
itself. `scripts/train.py` REFUSES to set CUBLAS_WORKSPACE_CONFIG for itself,
deliberately: a process that fixes its own environment cannot tell you the
launcher forgot, and the next launcher forgets again. So the artifact has to be
a command line *including its environment*, which is exactly what an emitter
produces and exactly what an orchestrator would hide.

STATUS COMES FROM DISK, NEVER FROM A TRACKING FILE

A tracking file can disagree with reality, and then the thing being debugged is
the tracker. Every state below is a fact about files that `train.py` and
`burst/config.py` already write on a schedule the config defines:

    DONE                    step{last_step}_full.pt exists
    RESUMABLE               >= 1 *_full.pt, highest < last_step
    STARTED_NOT_RESUMABLE   resolved_config.yaml, but no *_full.pt at all
    NOT_STARTED             no outdir, or an outdir with nothing in it
    CONFLICT                resolved_config.yaml describes a DIFFERENT config

`checkpoint_kind_at` guarantees the final step is a full checkpoint whatever
the intervals say, so "the final full checkpoint exists" IS completion rather
than evidence for it.

A HALF-WRITTEN CHECKPOINT CANNOT LOOK COMPLETE. `save_checkpoint` writes
`<name>.partial`, fsyncs, then `os.replace`. The rename is atomic and is the
commit point, so a truncated file is always named `.partial` and never `.pt`.
Stale `.partial` files are reported here because nothing removes them -- see
the note in `classify`.

NOTHING IS GENERATED HERE. `configs/runs/` holds one override file per run,
written by `scripts/generate_overrides.py`. This module consumes them. A
missing override is an error, not something to synthesise: the study's claim is
that the runs differ only in seed and arm, and a launcher that invented a
seed/arm pair would be making that claim on its own authority.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from burst.config import (  # noqa: E402
    ARMS,
    PROVENANCE_FILENAME,
    RESOLVED_CONFIG_FILENAME,
    ConfigError,
    load_config,
    run_name_for,
)

DEFAULT_OVERRIDE_DIR = REPO_ROOT / "configs" / "runs"
DEFAULT_BASE_CONFIG = REPO_ROOT / "configs" / "base.yaml"

COMMANDS_FILENAME = "commands.txt"
RESUME_FILENAME = "resume.txt"
STATUS_FILENAME = "status.json"

#: Must be in the environment BEFORE CUDA initialises. Emitted as a prefix on
#: every command so a pasted line reproduces the run; `train.py` still refuses
#: when it is absent, so the prefix is checked rather than trusted.
CUBLAS_WORKSPACE = "CUBLAS_WORKSPACE_CONFIG=:4096:8"

# Status names. Strings rather than an enum so they land in JSON unchanged.
DONE = "done"
RESUMABLE = "resumable"
STARTED_NOT_RESUMABLE = "started_not_resumable"
NOT_STARTED = "not_started"
CONFLICT = "conflict"


class LaunchError(Exception):
    """Raised for every refusal here. Never an assert."""


# ---------------------------------------------------------------------------
# Status, read off the filesystem
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunStatus:
    """What is on disk for one run, and nothing that is not."""

    run_name: str
    seed: int
    arm: str
    outdir: Path
    state: str
    #: Highest FULL checkpoint present, or None. Weights-only checkpoints are
    #: NOT resumable -- `load_checkpoint` refuses them, because they carry no
    #: optimizer state and no RNG state -- so this is the only thing a resume
    #: can start from.
    resume_from: Path | None = None
    last_full_step: int | None = None
    n_full: int = 0
    n_weights_only: int = 0
    #: Stale `<name>.partial` files. A killed process leaves one behind and
    #: nothing removes it. Harmless to a resume, which reads only `.pt`, but
    #: reported because a directory that quietly accumulates them is a
    #: directory that is lying about what happened in it.
    partials: tuple = ()
    launch_ready: bool = False
    missing_for_launch: tuple = ()
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "run_name": self.run_name,
            "seed": self.seed,
            "arm": self.arm,
            "outdir": str(self.outdir),
            "state": self.state,
            "resume_from": str(self.resume_from) if self.resume_from else None,
            "last_full_step": self.last_full_step,
            "n_full": self.n_full,
            "n_weights_only": self.n_weights_only,
            "partials": [str(p) for p in self.partials],
            "launch_ready": self.launch_ready,
            "missing_for_launch": list(self.missing_for_launch),
            "note": self.note,
        }


def _checkpoint_step(path: Path) -> int | None:
    """Step index from `stepNNNNNN_kind.pt`, or None if it is not one."""
    stem = path.stem
    if not stem.startswith("step") or "_" not in stem:
        return None
    digits = stem[4:stem.index("_")]
    if not digits.isdigit():
        return None
    return int(digits)


def classify(outdir: Path, last_step: int) -> tuple:
    """(state, resume_from, last_full_step, n_full, n_wo, partials).

    Pure: takes a directory and a number, touches nothing else, and does not
    care which run it belongs to. That is what lets the tests build all five
    states as real files rather than mocking a filesystem.

    `last_step` comes from the config rather than being hardcoded, because
    "the last step" is `total_steps - 1` and this must not be the place that
    forgets it.
    """
    outdir = Path(outdir)
    if not outdir.is_dir():
        return NOT_STARTED, None, None, 0, 0, ()

    partials = tuple(sorted(outdir.glob("*.partial")))
    full, weights_only = {}, {}
    for path in sorted(outdir.glob("step*.pt")):
        step = _checkpoint_step(path)
        if step is None:
            continue
        (full if path.stem.endswith("_full") else weights_only)[step] = path

    if full:
        highest = max(full)
        if highest >= last_step:
            return DONE, None, highest, len(full), len(weights_only), partials
        return (RESUMABLE, full[highest], highest, len(full),
                len(weights_only), partials)

    started = (outdir / RESOLVED_CONFIG_FILENAME).is_file() or bool(
        weights_only) or bool(partials)
    if started:
        return (STARTED_NOT_RESUMABLE, None, None, 0, len(weights_only),
                partials)
    return NOT_STARTED, None, None, 0, len(weights_only), partials


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def parse_selection(seeds: str | None, arms: str | None, take_all: bool,
                    n_seeds: int) -> list:
    """The (seed, arm) pairs to emit. NEVER defaults to everything.

    `--all` has to be typed. The pilot is four runs and a fat-fingered command
    must not start seventy, so an empty selection is an error rather than an
    implicit "all".
    """
    if take_all and (seeds or arms):
        raise LaunchError(
            "--all cannot be combined with --seeds or --arms. Either take "
            "every run or name the ones you want; a filtered --all reads like "
            "a mistake either way.")
    if take_all:
        return [(s, a) for s in range(n_seeds) for a in ARMS]
    if not seeds and not arms:
        raise LaunchError(
            "no runs selected. Pass --seeds and/or --arms to name them, or "
            "--all to take every run.\n"
            "There is deliberately no default: the pilot is four runs and the "
            "study is seventy, and the difference between them must be "
            "something you typed.")

    chosen_seeds = (list(range(n_seeds)) if not seeds
                    else _parse_ints(seeds, n_seeds))
    chosen_arms = list(ARMS) if not arms else _parse_arms(arms)
    return [(s, a) for s in chosen_seeds for a in chosen_arms]


def _parse_ints(raw: str, n_seeds: int) -> list:
    out = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if not chunk.isdigit():
            raise LaunchError(f"seed {chunk!r} is not a number")
        value = int(chunk)
        if not 0 <= value < n_seeds:
            raise LaunchError(
                f"seed {value} is outside 0..{n_seeds - 1} "
                f"(experiment.n_seeds = {n_seeds})")
        out.append(value)
    if not out:
        raise LaunchError("--seeds was given but named no seeds")
    return sorted(set(out))


def _parse_arms(raw: str) -> list:
    out = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk not in ARMS:
            raise LaunchError(
                f"unknown arm {chunk!r}; expected one of {', '.join(ARMS)}.\n"
                "Arms are not synthesised here -- this launcher consumes the "
                "override files in configs/runs/ and an arm with no override "
                "is an error rather than something to fill in.")
        out.append(chunk)
    if not out:
        raise LaunchError("--arms was given but named no arms")
    # Ordered by ARMS, not by the order typed, so two equivalent selections
    # emit in the same order and their command files compare cleanly.
    return [a for a in ARMS if a in set(out)]


# ---------------------------------------------------------------------------
# Refusals that apply to the whole emission
# ---------------------------------------------------------------------------


def check_clean_tree(repo_root: Path = REPO_ROOT) -> dict:
    """Refuse to emit from a dirty working tree.

    `burst/config.py` stamps the commit hash and a dirty flag into
    run_provenance.yaml on every load. A run launched from uncommitted code
    records a hash that does not describe what produced it, and the study's
    provenance claim is exactly that the hash does describe it.

    Checked HERE as well as recorded there, because a warning inside a run
    that has already started is a warning nobody acts on.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_root,
            capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LaunchError(
            f"could not determine whether the working tree is clean: {exc}\n"
            "Refusing to emit rather than guessing: an emission from a dirty "
            "tree records a commit hash that does not describe the code.")
    dirty = [line for line in result.stdout.splitlines() if line.strip()]
    if dirty:
        listing = "\n".join(f"    {line}" for line in dirty[:10])
        more = ("\n    ..." if len(dirty) > 10 else "")
        raise LaunchError(
            f"THE WORKING TREE IS DIRTY ({len(dirty)} change(s)):\n"
            f"{listing}{more}\n"
            "Every run stamps the current commit hash into "
            "run_provenance.yaml. Launching from uncommitted code records a "
            "hash that does not describe what produced the checkpoints, which "
            "is the one claim the provenance file exists to make. Commit "
            "first.")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root,
        capture_output=True, text=True, check=False)
    return {"clean": True, "commit": head.stdout.strip() or None}


def check_overrides(override_dir: Path = DEFAULT_OVERRIDE_DIR,
                    base_config: Path = DEFAULT_BASE_CONFIG) -> dict:
    """Refuse to emit if the override files are not what the generator writes.

    `scripts/generate_overrides.py --check` is the authority. This is the only
    place all of them are read at once, so it is where "generated, never
    hand-edited" gets enforced rather than assumed.
    """
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "generate_overrides.py"),
         "--check", "--outdir", str(override_dir),
         "--base", str(base_config)],
        capture_output=True, text=True, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise LaunchError(
            "generate_overrides.py --check FAILED:\n"
            f"{result.stdout.strip()}\n{result.stderr.strip()}\n"
            "The override files in configs/runs/ are generated and must never "
            "be hand-edited. Regenerate them with "
            "`python scripts/generate_overrides.py` and commit the result.")
    return {"checked": True, "output": result.stdout.strip()}


# ---------------------------------------------------------------------------
# Building the emission
# ---------------------------------------------------------------------------


@dataclass
class Emission:
    commands: list = field(default_factory=list)
    resumes: list = field(default_factory=list)
    statuses: list = field(default_factory=list)
    skipped_done: list = field(default_factory=list)
    not_ready: list = field(default_factory=list)
    conflicts: list = field(default_factory=list)


def command_for(run: RunStatus, *, base_config: Path, override: Path,
                corpus: Path, family: str, device: str,
                resume: Path | None = None) -> str:
    """One self-contained invocation, environment included.

    CUBLAS_WORKSPACE_CONFIG is prefixed. CUDA_VISIBLE_DEVICES deliberately is
    NOT: device assignment is the one part that genuinely depends on hardware
    nobody has provisioned, and inventing an assignment here would be this
    module pretending to orchestrate.
    """
    parts = [
        CUBLAS_WORKSPACE,
        shlex.quote(sys.executable if device == "cpu" else "python"),
        shlex.quote(str(Path("scripts") / "train.py")),
        "--config", shlex.quote(str(base_config)),
        "--run", shlex.quote(str(override)),
        "--outdir", shlex.quote(str(run.outdir)),
        "--corpus", shlex.quote(str(corpus)),
        "--family", shlex.quote(family),
        "--device", shlex.quote(device),
    ]
    if resume is not None:
        parts += ["--resume", shlex.quote(str(resume))]
    return " ".join(parts)


def build(selection, *, outroot: Path, corpus: Path, family: str, device: str,
          base_config: Path = DEFAULT_BASE_CONFIG,
          override_dir: Path = DEFAULT_OVERRIDE_DIR,
          overwrite: bool = False, allow_incomplete: bool = False,
          stream=None) -> Emission:
    """Classify every selected run and emit what should be run for it."""
    import io

    stream = stream or sys.stdout
    emission = Emission()

    for seed, arm in selection:
        name = run_name_for(seed, arm)
        override = Path(override_dir) / f"{name}.yaml"
        if not override.is_file():
            raise LaunchError(
                f"no override file for {name} at {override}.\n"
                "This launcher consumes configs/runs/ and does not synthesise "
                "seed/arm combinations. Regenerate with "
                "`python scripts/generate_overrides.py`.")

        outdir = Path(outroot) / name
        try:
            # write_provenance=False is load-bearing: classifying a run must
            # not write resolved_config.yaml into the directory it is
            # inspecting, or the act of asking "has this started?" would make
            # the answer yes.
            cfg = load_config(base_config, override, outdir=outdir,
                              require_complete=False, stream=io.StringIO(),
                              write_provenance=False)
        except ConfigError as exc:
            emission.conflicts.append((name, str(exc)))
            emission.statuses.append(RunStatus(
                run_name=name, seed=seed, arm=arm, outdir=outdir,
                state=CONFLICT, note=str(exc).split("\n")[0]))
            continue

        state, resume_from, last_full, n_full, n_wo, partials = classify(
            outdir, cfg.last_step)
        missing = tuple(cfg.missing_for_launch)
        status = RunStatus(
            run_name=name, seed=seed, arm=arm, outdir=outdir, state=state,
            resume_from=resume_from, last_full_step=last_full, n_full=n_full,
            n_weights_only=n_wo, partials=partials,
            launch_ready=not missing, missing_for_launch=missing)
        emission.statuses.append(status)

        if state == DONE and not overwrite:
            emission.skipped_done.append(name)
            continue
        if missing and not allow_incomplete:
            emission.not_ready.append((name, missing))
            continue

        if state == RESUMABLE and not overwrite:
            emission.resumes.append(command_for(
                status, base_config=base_config, override=override,
                corpus=corpus, family=family, device=device,
                resume=resume_from))
        else:
            emission.commands.append(command_for(
                status, base_config=base_config, override=override,
                corpus=corpus, family=family, device=device))

    if emission.not_ready and not allow_incomplete:
        detail = "\n".join(
            f"    {name}: {', '.join(fields)}"
            for name, fields in emission.not_ready[:8])
        more = "\n    ..." if len(emission.not_ready) > 8 else ""
        raise LaunchError(
            f"{len(emission.not_ready)} run(s) are NOT LAUNCH-READY:\n"
            f"{detail}{more}\n"
            "Emitting a command that is guaranteed to refuse at load helps "
            "nobody. These fields are null by design and are part of the "
            "study's definition -- the pilot is what settles them. Decide them "
            "in configs/base.yaml, or pass --allow-incomplete to preview the "
            "emission anyway.")
    return emission


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/launch.py",
        description="Emit runnable commands for the study's runs. Starts "
                    "nothing, manages nothing, retries nothing.")
    parser.add_argument("--outroot", type=Path, required=True,
                        help="one directory per run is created beneath this. "
                             "A command-line argument and never a config "
                             "value, so one config runs anywhere.")
    parser.add_argument("--corpus", type=Path, required=True,
                        help="the tokenized corpus directory")
    parser.add_argument("--family", required=True,
                        help="model family passed to train.py; it has no "
                             "default there and none here")
    parser.add_argument("--device", default="cuda",
                        help="passed through to train.py")
    parser.add_argument("--seeds", default=None,
                        help="comma-separated, e.g. 0,1")
    parser.add_argument("--arms", default=None,
                        help="comma-separated, e.g. fluent-false,twin")
    parser.add_argument("--all", action="store_true", dest="take_all",
                        help="every run. Must be typed; there is no default.")
    parser.add_argument("--emit-dir", type=Path, default=None,
                        help="where commands.txt, resume.txt and status.json "
                             "are written. Defaults to --outroot.")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-emit runs whose outdir already holds a "
                             "completed run")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="emit runs that are not launch-ready. They will "
                             "refuse at load; this is for previewing.")
    parser.add_argument("--status-only", action="store_true",
                        help="report what is on disk and emit nothing")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--override-dir", type=Path,
                        default=DEFAULT_OVERRIDE_DIR)
    parser.add_argument("--skip-tree-check", action="store_true",
                        help="TESTING ONLY. Skips the dirty-tree refusal.")
    return parser


def main(argv=None) -> int:
    import io
    import yaml

    args = _build_parser().parse_args(argv)
    stream = sys.stdout
    rule = "=" * 78
    print(rule, file=stream)
    print("launch -- step 15. Emits commands; starts nothing.", file=stream)
    print(rule, file=stream)

    try:
        if not args.skip_tree_check:
            git = check_clean_tree()
            print(f"tree:      clean at {git.get('commit', '?')[:12]}",
                  file=stream)
        check_overrides(args.override_dir, args.base)
        print("overrides: generate_overrides.py --check passed", file=stream)

        base = yaml.safe_load(args.base.read_text(encoding="utf-8"))
        n_seeds = int(base["experiment"]["n_seeds"])
        selection = parse_selection(args.seeds, args.arms, args.take_all,
                                    n_seeds)
        print(f"selected:  {len(selection)} run(s)\n", file=stream)

        emission = build(
            selection, outroot=args.outroot, corpus=args.corpus,
            family=args.family, device=args.device, base_config=args.base,
            override_dir=args.override_dir, overwrite=args.overwrite,
            allow_incomplete=args.allow_incomplete or args.status_only,
            stream=stream)
    except LaunchError as exc:
        print(f"\nREFUSED: {exc}", file=stream)
        return 1

    _print_status(emission, stream)

    if args.status_only:
        return 0

    emit_dir = Path(args.emit_dir or args.outroot)
    emit_dir.mkdir(parents=True, exist_ok=True)
    _write_lines(emit_dir / COMMANDS_FILENAME, emission.commands)
    _write_lines(emit_dir / RESUME_FILENAME, emission.resumes)
    (emit_dir / STATUS_FILENAME).write_text(
        json.dumps([s.as_dict() for s in emission.statuses], indent=2) + "\n",
        encoding="utf-8", newline="\n")

    print(f"\nwrote {emit_dir / COMMANDS_FILENAME} "
          f"({len(emission.commands)} command(s))", file=stream)
    print(f"wrote {emit_dir / RESUME_FILENAME} "
          f"({len(emission.resumes)} resume(s))", file=stream)
    print(f"wrote {emit_dir / STATUS_FILENAME}", file=stream)
    print("\nNothing has been started. Run them however you like:", file=stream)
    print(f"    bash {emit_dir / COMMANDS_FILENAME}", file=stream)
    print(f"    xargs -P 8 -I{{}} bash -c '{{}}' "
          f"< {emit_dir / COMMANDS_FILENAME}", file=stream)
    return 0


def _write_lines(path: Path, lines) -> None:
    path.write_text("".join(f"{line}\n" for line in lines),
                    encoding="utf-8", newline="\n")


def _print_status(emission: Emission, stream) -> None:
    counts = {}
    for status in emission.statuses:
        counts[status.state] = counts.get(status.state, 0) + 1
    print(f"{'run':<28}{'state':<24}{'full':>6}{'wo':>6}  resume from",
          file=stream)
    for status in emission.statuses:
        resume = (Path(status.resume_from).name if status.resume_from else "")
        print(f"{status.run_name:<28}{status.state:<24}{status.n_full:>6}"
              f"{status.n_weights_only:>6}  {resume}", file=stream)
        if status.partials:
            print(f"{'':<28}STALE PARTIALS: "
                  f"{', '.join(p.name for p in status.partials)}", file=stream)
    print("", file=stream)
    for state, count in sorted(counts.items()):
        print(f"  {state:<24}{count}", file=stream)
    if emission.skipped_done:
        print(f"\n  {len(emission.skipped_done)} completed run(s) skipped; "
              "pass --overwrite to re-emit", file=stream)


if __name__ == "__main__":
    raise SystemExit(main())
