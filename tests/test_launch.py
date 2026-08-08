"""The launcher: what it refuses, and what it reads off disk.

THE RISK THIS FILE EXISTS FOR.

The launcher's whole job is to be trustworthy about two things: that it will
not emit a run it should not, and that its picture of what is on disk is the
truth rather than a cached opinion. Both fail quietly.

A launcher that emits from a dirty tree produces 70 runs whose
run_provenance.yaml records a commit hash that does not describe the code that
produced them -- and nothing downstream can tell, because the file looks
normal. A launcher whose status is subtly wrong resumes a finished run, or
declares a half-finished one complete, and the study loses a run without an
error anywhere.

So the tests below are mostly refusals, and the status tests build all five
states as REAL FILES rather than mocking a filesystem. `classify` is pure -- a
directory and a number in, a tuple out -- specifically so that is possible.

Nothing here starts a process. The launcher does not either; that is the point
of it.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


L = _load("launch")

sys.path.insert(0, str(REPO_ROOT))
from burst.config import ARMS, INJECTING_ARMS, run_name_for  # noqa: E402

LAST_STEP = 9535


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def _full(outdir: Path, step: int) -> Path:
    return _touch(outdir / f"step{step:06d}_full.pt")


def _weights_only(outdir: Path, step: int) -> Path:
    return _touch(outdir / f"step{step:06d}_weights_only.pt")


# ---------------------------------------------------------------------------
# Status: all five states, built as real files
# ---------------------------------------------------------------------------


def test_not_started_when_the_directory_is_absent(tmp_path):
    state, resume, last, n_full, n_wo, partials = L.classify(
        tmp_path / "nope", LAST_STEP)
    assert state == L.NOT_STARTED
    assert resume is None and n_full == 0 and partials == ()


def test_not_started_when_the_directory_is_empty(tmp_path):
    (tmp_path / "run").mkdir()
    assert L.classify(tmp_path / "run", LAST_STEP)[0] == L.NOT_STARTED


def test_done_when_the_final_full_checkpoint_exists(tmp_path):
    """checkpoint_kind_at guarantees the last step is always FULL.

    So this file existing IS completion, not evidence for it.
    """
    outdir = tmp_path / "run"
    _full(outdir, 999)
    _full(outdir, LAST_STEP)
    state, resume, last, n_full, _, _ = L.classify(outdir, LAST_STEP)
    assert state == L.DONE
    assert last == LAST_STEP
    assert n_full == 2
    assert resume is None, "a finished run must not offer a resume"


def test_resumable_from_the_highest_full_checkpoint(tmp_path):
    outdir = tmp_path / "run"
    for step in (999, 1999, 2999):
        _full(outdir, step)
    _weights_only(outdir, 3049)   # later, and NOT resumable
    state, resume, last, n_full, n_wo, _ = L.classify(outdir, LAST_STEP)
    assert state == L.RESUMABLE
    assert resume.name == "step002999_full.pt", (
        "resume must come from the highest FULL checkpoint, not the highest "
        "checkpoint -- load_checkpoint refuses weights-only files because they "
        "carry no optimizer state and no RNG state")
    assert last == 2999 and n_full == 3 and n_wo == 1


def test_started_but_not_resumable_before_the_first_full_checkpoint(tmp_path):
    """A run that dies before step 999 has no full checkpoint at all.

    Full checkpoints land every 1000 steps, so a death at step 500 leaves ten
    weights-only files and nothing to resume from. It restarts from zero.
    """
    outdir = tmp_path / "run"
    for step in (49, 99, 149):
        _weights_only(outdir, step)
    state, resume, last, n_full, n_wo, _ = L.classify(outdir, LAST_STEP)
    assert state == L.STARTED_NOT_RESUMABLE
    assert resume is None and n_full == 0 and n_wo == 3


def test_a_resolved_config_alone_counts_as_started(tmp_path):
    """The loader writes it before step 0, so it is the earliest evidence."""
    outdir = tmp_path / "run"
    outdir.mkdir()
    (outdir / "resolved_config.yaml").write_text("x: 1", encoding="utf-8")
    assert L.classify(outdir, LAST_STEP)[0] == L.STARTED_NOT_RESUMABLE


def test_stale_partials_are_reported_and_do_not_change_the_state(tmp_path):
    """A half-written checkpoint cannot masquerade as a complete one.

    save_checkpoint writes <name>.partial then os.replace, and the rename is
    the commit point -- so a truncated file is ALWAYS named .partial. Nothing
    removes them, so they are surfaced rather than ignored.
    """
    outdir = tmp_path / "run"
    _full(outdir, 999)
    _touch(outdir / "step001999_full.pt.partial")
    state, resume, _, n_full, _, partials = L.classify(outdir, LAST_STEP)
    assert state == L.RESUMABLE
    assert resume.name == "step000999_full.pt"
    assert n_full == 1, "a .partial must not be counted as a checkpoint"
    assert [p.name for p in partials] == ["step001999_full.pt.partial"]


def test_a_partial_alone_counts_as_started(tmp_path):
    outdir = tmp_path / "run"
    _touch(outdir / "step000999_full.pt.partial")
    assert L.classify(outdir, LAST_STEP)[0] == L.STARTED_NOT_RESUMABLE


def test_unparseable_filenames_are_ignored_rather_than_guessed(tmp_path):
    outdir = tmp_path / "run"
    _touch(outdir / "notes.pt")
    _touch(outdir / "stepXXXX_full.pt")
    _full(outdir, 999)
    state, _, last, n_full, _, _ = L.classify(outdir, LAST_STEP)
    assert state == L.RESUMABLE and last == 999 and n_full == 1


def test_done_uses_the_configs_last_step_not_a_hardcoded_9535(tmp_path):
    """`last_step` is total_steps - 1 and this must not be where that is lost."""
    outdir = tmp_path / "run"
    _full(outdir, 41)
    assert L.classify(outdir, last_step=41)[0] == L.DONE
    assert L.classify(outdir, last_step=9535)[0] == L.RESUMABLE


# ---------------------------------------------------------------------------
# Selection: never defaults to everything
# ---------------------------------------------------------------------------


def test_an_empty_selection_is_refused(tmp_path):
    """The pilot is four runs and the study is seventy."""
    with pytest.raises(L.LaunchError, match="no runs selected"):
        L.parse_selection(None, None, False, 10)


def test_all_must_be_typed_and_takes_every_run():
    selection = L.parse_selection(None, None, True, 10)
    assert len(selection) == 10 * len(ARMS) == 40


def test_all_cannot_be_combined_with_a_filter():
    with pytest.raises(L.LaunchError, match="cannot be combined"):
        L.parse_selection("0,1", None, True, 10)


def test_the_pilot_is_a_selection_not_a_mode():
    """Four runs and seventy differ only in what was typed."""
    pilot = L.parse_selection("0,1", "fluent-false,twin", False, 10)
    assert len(pilot) == 4
    assert set(pilot) == {(0, "fluent-false"), (0, "twin"),
                          (1, "fluent-false"), (1, "twin")}


def test_seeds_alone_takes_every_arm():
    selection = L.parse_selection("3", None, False, 10)
    assert len(selection) == len(ARMS)
    assert {a for _, a in selection} == set(ARMS)


def test_arms_alone_takes_every_seed():
    selection = L.parse_selection(None, "twin", False, 10)
    assert len(selection) == 10
    assert {a for _, a in selection} == {"twin"}


def test_an_out_of_range_seed_is_refused():
    with pytest.raises(L.LaunchError, match="outside 0..9"):
        L.parse_selection("10", None, False, 10)


def test_an_unknown_arm_is_refused_not_synthesised():
    with pytest.raises(L.LaunchError, match="unknown arm"):
        L.parse_selection(None, "coherent", False, 10)


def test_scrambled_corpus_cannot_be_selected():
    """It has a committed text and committed measurements and is not a run."""
    with pytest.raises(L.LaunchError, match="unknown arm"):
        L.parse_selection(None, "scrambled-corpus", False, 10)


def test_selection_order_does_not_depend_on_typing_order():
    """Two equivalent selections must emit identically comparable files."""
    a = L.parse_selection("0", "twin,fluent-false", False, 10)
    b = L.parse_selection("0", "fluent-false,twin", False, 10)
    assert a == b


# ---------------------------------------------------------------------------
# The refusals that apply to the whole emission
# ---------------------------------------------------------------------------


def test_a_dirty_tree_is_refused(tmp_path):
    """A run launched from uncommitted code records a hash that is not true."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "f.txt").write_text("uncommitted", encoding="utf-8")
    with pytest.raises(L.LaunchError, match="WORKING TREE IS DIRTY"):
        L.check_clean_tree(tmp_path)


def test_a_clean_tree_passes_and_reports_the_commit(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path,
                   check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path,
                   check=True)
    (tmp_path / "f.txt").write_text("committed", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "x"], cwd=tmp_path, check=True)
    result = L.check_clean_tree(tmp_path)
    assert result["clean"] is True
    assert len(result["commit"]) == 40


def test_hand_edited_overrides_are_refused(tmp_path):
    """Generated, never hand-edited -- and this is where that is enforced."""
    override_dir = tmp_path / "runs"
    override_dir.mkdir()
    (override_dir / f"{run_name_for(0, 'twin')}.yaml").write_text(
        "seed: 9\narm: twin\n", encoding="utf-8")
    with pytest.raises(L.LaunchError, match="--check FAILED"):
        L.check_overrides(override_dir, REPO_ROOT / "configs" / "base.yaml")


def test_the_shipped_overrides_pass_the_check():
    result = L.check_overrides()
    assert result["checked"] is True
    assert "40 ok" in result["output"]


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


@pytest.fixture
def ready_base(tmp_path):
    """The shipped config with the three reduction-order values decided.

    They are null by design and launch-blocking, so an emission test has to
    decide them -- which is exactly what the pilot will do.
    """
    base = yaml.safe_load(
        (REPO_ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    base["training"]["micro_batch"] = 8
    base["training"]["dtype"] = "fp32"
    base["optimizer"]["adamw_impl"] = "foreach"
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def unready_base(tmp_path):
    """The shipped config with the three reduction-order values put BACK to null.

    They were null until 2026-08-03 and every one of the 70 runs refused. Now
    the shipped config is complete, so a test of the NOT LAUNCH-READY refusal
    has to supply its own incomplete config -- otherwise it passes while
    exercising nothing, which is worse than deleting it.
    """
    base = yaml.safe_load(
        (REPO_ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    base["training"]["micro_batch"] = None
    base["training"]["dtype"] = None
    base["optimizer"]["adamw_impl"] = None
    path = tmp_path / "unready_base.yaml"
    path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    return path


def _build(tmp_path, base, selection, **kwargs):
    kwargs.setdefault("corpus", tmp_path / "corpus")
    kwargs.setdefault("family", "hf_gpt2")
    kwargs.setdefault("device", "cuda")
    return L.build(selection, outroot=tmp_path / "runs", base_config=base,
                   **kwargs)


# ---------------------------------------------------------------------------
# CONFLICT: this directory already belongs to a different run
#
# `conflicting_run` had NO test coverage at all -- found while adding
# `conflicting_family` beside it. It was made reachable on 2026-08-03 after the
# contradiction pass found the CONFLICT state unreachable; nothing then pinned
# that it fires.
# ---------------------------------------------------------------------------


def _write_provenance(outdir: Path, **fields) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "run_provenance.yaml"
    path.write_text(yaml.safe_dump(fields, sort_keys=False), encoding="utf-8")
    return path


def test_a_directory_holding_another_seed_arm_is_a_conflict(tmp_path):
    outdir = tmp_path / "out"
    outdir.mkdir()
    (outdir / "resolved_config.yaml").write_text(
        yaml.safe_dump({"seed": 7, "arm": "twin"}), encoding="utf-8")
    clash = L.conflicting_run(outdir, seed=3, arm="twin")
    assert clash is not None
    assert "seed=7" in clash


def test_a_matching_seed_arm_is_not_a_conflict(tmp_path):
    outdir = tmp_path / "out"
    outdir.mkdir()
    (outdir / "resolved_config.yaml").write_text(
        yaml.safe_dump({"seed": 3, "arm": "twin"}), encoding="utf-8")
    assert L.conflicting_run(outdir, seed=3, arm="twin") is None


def test_a_directory_built_by_another_family_is_a_conflict(tmp_path):
    """The failure this exists to stop: hf_gpt2 emitted over a probe_linear
    run, producing checkpoints nothing downstream can tell apart."""
    outdir = tmp_path / "out"
    _write_provenance(outdir, seed=3, arm="twin", family="probe_linear")
    clash = L.conflicting_family(outdir, "hf_gpt2")
    assert clash is not None
    assert "probe_linear" in clash and "hf_gpt2" in clash


def test_a_matching_family_is_not_a_conflict(tmp_path):
    outdir = tmp_path / "out"
    _write_provenance(outdir, seed=3, arm="twin", family="hf_gpt2")
    assert L.conflicting_family(outdir, "hf_gpt2") is None


def test_a_null_family_in_provenance_is_reported_not_waved_through(tmp_path):
    """"Cannot tell" and "matches" are different answers, and a provenance
    file predating the field can only give the first."""
    outdir = tmp_path / "out"
    _write_provenance(outdir, seed=3, arm="twin", family=None)
    clash = L.conflicting_family(outdir, "hf_gpt2")
    assert clash is not None
    assert "null" in clash


def test_provenance_without_the_family_key_at_all_is_not_a_conflict(tmp_path):
    """A file written before the field existed omits the key entirely. That is
    a different case from an explicit null and must not block re-emission."""
    outdir = tmp_path / "out"
    _write_provenance(outdir, seed=3, arm="twin")
    assert L.conflicting_family(outdir, "hf_gpt2") is None


def test_no_provenance_file_is_not_a_conflict(tmp_path):
    assert L.conflicting_family(tmp_path / "nothing-here", "hf_gpt2") is None


def test_build_reports_a_family_clash_as_CONFLICT(tmp_path, ready_base):
    """End to end: the refusal reaches the emission, not just the helper."""
    outdir = tmp_path / "runs" / run_name_for(0, "twin")
    _write_provenance(outdir, seed=0, arm="twin", family="probe_linear")
    emission = _build(tmp_path, ready_base, [(0, "twin")], family="hf_gpt2")
    assert emission.commands == []
    assert emission.statuses[0].state == L.CONFLICT
    assert "probe_linear" in emission.statuses[0].note


def test_a_not_launch_ready_run_is_refused_by_default(tmp_path, unready_base):
    """A run with any launch-blocking null refuses rather than emitting.

    Until 2026-08-03 all 70 refused, because micro_batch, dtype and
    adamw_impl were null in the shipped config. They are set now, so this
    supplies its own incomplete config to keep exercising the refusal.
    """
    with pytest.raises(L.LaunchError, match="NOT LAUNCH-READY"):
        _build(tmp_path, unready_base, [(0, "twin")])


def test_the_refusal_names_the_missing_fields(tmp_path, unready_base):
    with pytest.raises(L.LaunchError) as exc:
        _build(tmp_path, unready_base, [(0, "twin")])
    message = str(exc.value)
    for field in ("training.micro_batch", "training.dtype",
                  "optimizer.adamw_impl"):
        assert field in message


def test_allow_incomplete_previews_rather_than_refusing(tmp_path, unready_base):
    emission = _build(tmp_path, unready_base, [(0, "twin")],
                      allow_incomplete=True)
    assert len(emission.commands) == 1
    assert emission.statuses[0].launch_ready is False


def test_the_shipped_config_is_now_launch_ready(tmp_path):
    """The other side of the change: configs/base.yaml emits without help.

    Paired with the three tests above deliberately. They prove the refusal
    still works; this proves it no longer fires on the real config, so neither
    fact can go stale without a test noticing.
    """
    emission = _build(tmp_path, REPO_ROOT / "configs" / "base.yaml",
                      [(0, "twin")])
    assert len(emission.commands) == 1
    assert emission.statuses[0].launch_ready is True
    assert emission.statuses[0].missing_for_launch == ()


def test_a_ready_run_emits_one_command(tmp_path, ready_base):
    emission = _build(tmp_path, ready_base, [(0, "twin")])
    assert len(emission.commands) == 1
    assert emission.resumes == []


def test_the_command_carries_the_cublas_prefix(tmp_path, ready_base):
    """train.py refuses without it, so the emitted line must supply it."""
    emission = _build(tmp_path, ready_base, [(0, "twin")])
    assert emission.commands[0].startswith("CUBLAS_WORKSPACE_CONFIG=:4096:8 ")


def test_the_command_assigns_no_device_visibility(tmp_path, ready_base):
    """Device assignment depends on hardware nobody has provisioned."""
    emission = _build(tmp_path, ready_base, [(0, "twin")])
    assert "CUDA_VISIBLE_DEVICES" not in emission.commands[0]


def test_the_command_is_self_contained(tmp_path, ready_base):
    emission = _build(tmp_path, ready_base, [(0, "twin")])
    line = emission.commands[0]
    for flag in ("--config", "--run", "--outdir", "--corpus", "--family",
                 "--device"):
        assert flag in line, flag
    assert "train.py" in line


def test_a_completed_run_is_skipped_without_overwrite(tmp_path, ready_base):
    outdir = tmp_path / "runs" / run_name_for(0, "twin")
    _full(outdir, LAST_STEP)
    emission = _build(tmp_path, ready_base, [(0, "twin")])
    assert emission.commands == []
    assert emission.skipped_done == [run_name_for(0, "twin")]


def test_overwrite_re_emits_a_completed_run(tmp_path, ready_base):
    outdir = tmp_path / "runs" / run_name_for(0, "twin")
    _full(outdir, LAST_STEP)
    emission = _build(tmp_path, ready_base, [(0, "twin")], overwrite=True)
    assert len(emission.commands) == 1


def test_a_partial_run_emits_a_resume_not_a_fresh_command(tmp_path,
                                                          ready_base):
    outdir = tmp_path / "runs" / run_name_for(0, "twin")
    _full(outdir, 2999)
    emission = _build(tmp_path, ready_base, [(0, "twin")])
    assert emission.commands == []
    assert len(emission.resumes) == 1
    assert "--resume" in emission.resumes[0]
    assert "step002999_full.pt" in emission.resumes[0]


def test_a_started_not_resumable_run_emits_a_fresh_command(tmp_path,
                                                           ready_base):
    """No full checkpoint means there is nothing to resume from."""
    outdir = tmp_path / "runs" / run_name_for(0, "twin")
    _weights_only(outdir, 499)
    emission = _build(tmp_path, ready_base, [(0, "twin")])
    assert len(emission.commands) == 1
    assert "--resume" not in emission.commands[0]
    assert emission.resumes == []


def test_a_missing_override_is_an_error_not_something_to_fill_in(tmp_path,
                                                                 ready_base):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(L.LaunchError, match="no override file"):
        _build(tmp_path, ready_base, [(0, "twin")], override_dir=empty)


def test_classifying_a_run_does_not_create_its_output_directory(tmp_path,
                                                                ready_base):
    """Asking 'has this started?' must not make the answer yes.

    load_config writes resolved_config.yaml into --outdir by default, which
    would make every classified run look STARTED after one status call.
    """
    _build(tmp_path, ready_base, [(0, "twin")])
    outdir = tmp_path / "runs" / run_name_for(0, "twin")
    assert not (outdir / "resolved_config.yaml").exists()


def test_the_pilot_and_the_study_emit_from_the_same_path(tmp_path, ready_base):
    pilot = _build(tmp_path, ready_base,
                   L.parse_selection("0,1", "fluent-false,twin", False, 10))
    study = _build(tmp_path, ready_base,
                   L.parse_selection(None, None, True, 10))
    assert len(pilot.commands) == 4
    assert len(study.commands) == 40
    assert all(c.startswith("CUBLAS_WORKSPACE_CONFIG=") for c in study.commands)


def test_every_injecting_arm_and_twin_can_be_emitted(tmp_path, ready_base):
    selection = [(0, arm) for arm in ARMS]
    emission = _build(tmp_path, ready_base, selection)
    assert len(emission.commands) == len(ARMS) == 4
    assert len(INJECTING_ARMS) == 3


# ---------------------------------------------------------------------------
# It emits. It does not orchestrate.
# ---------------------------------------------------------------------------


def test_the_launcher_starts_no_processes():
    """Read from the source: the only subprocess calls are the two checks.

    An emitter that quietly ran something would be an orchestrator with a
    misleading name, and the reason for emitting is that nobody knows what the
    hardware is yet.
    """
    import ast

    source = (REPO_ROOT / "scripts" / "launch.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    runners = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            if name in {"Popen", "system", "spawn", "spawnv", "execv"}:
                runners.append((name, node.lineno))
    assert not runners, f"the launcher starts processes: {runners}"

    # subprocess.run IS used, so assert WHAT it invokes rather than how many
    # times -- a bare count breaks when a legitimate call is added and teaches
    # nothing when it does. Every call must be `git` (the tree check) or the
    # current interpreter running generate_overrides.py (the --check).
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if getattr(func, "attr", None) != "run":
            continue
        if getattr(getattr(func, "value", None), "id", None) != "subprocess":
            continue
        argv = node.args[0]
        assert isinstance(argv, ast.List) and argv.elts, (
            "a subprocess argv built dynamically cannot be audited here")
        rendered = ast.unparse(argv)
        assert rendered.startswith("['git'") or "generate_overrides.py" in rendered, (
            f"the launcher invokes {rendered[:70]}; the only subprocesses it "
            "may start are `git` for the tree check and generate_overrides.py "
            "for --check, both of which inspect rather than train")


def test_there_is_no_retry_logic():
    source = (REPO_ROOT / "scripts" / "launch.py").read_text(encoding="utf-8")
    for banned in ("retry", "backoff", "max_attempts", "time.sleep"):
        assert banned not in source.lower(), banned


def test_no_tracking_file_is_consulted_for_status():
    """Status is derived from checkpoints; status.json is an OUTPUT.

    If classify ever read a tracking file, a disagreement between the file and
    the disk would become the thing being debugged.
    """
    import inspect

    source = inspect.getsource(L.classify)
    assert "status.json" not in source
    assert "glob" in source and "resolved_config" in source.lower()


# ---------------------------------------------------------------------------
# End to end through the CLI
# ---------------------------------------------------------------------------


def test_cli_writes_all_three_artifacts(tmp_path, ready_base):
    outroot = tmp_path / "runs"
    code = L.main([
        "--outroot", str(outroot), "--corpus", str(tmp_path / "corpus"),
        "--family", "hf_gpt2", "--device", "cuda",
        "--seeds", "0,1", "--arms", "fluent-false,twin",
        "--base", str(ready_base), "--skip-tree-check",
    ])
    assert code == 0
    commands = (outroot / L.COMMANDS_FILENAME).read_text(encoding="utf-8")
    assert len(commands.strip().splitlines()) == 4
    assert (outroot / L.RESUME_FILENAME).is_file()
    statuses = json.loads(
        (outroot / L.STATUS_FILENAME).read_text(encoding="utf-8"))
    assert len(statuses) == 4
    assert {s["state"] for s in statuses} == {L.NOT_STARTED}


def test_cli_status_only_emits_nothing(tmp_path, ready_base):
    outroot = tmp_path / "runs"
    code = L.main([
        "--outroot", str(outroot), "--corpus", str(tmp_path / "c"),
        "--family", "hf_gpt2", "--seeds", "0", "--arms", "twin",
        "--base", str(ready_base), "--skip-tree-check", "--status-only",
    ])
    assert code == 0
    assert not (outroot / L.COMMANDS_FILENAME).exists()


def test_cli_refuses_an_empty_selection(tmp_path, ready_base, capsys):
    code = L.main([
        "--outroot", str(tmp_path / "r"), "--corpus", str(tmp_path / "c"),
        "--family", "hf_gpt2", "--base", str(ready_base), "--skip-tree-check",
    ])
    assert code == 1
    assert "no runs selected" in capsys.readouterr().out


def test_cli_requires_family_and_offers_no_default():
    parser = L._build_parser()
    family = {a.dest: a for a in parser._actions}["family"]
    assert family.required is True
    assert family.default is None
