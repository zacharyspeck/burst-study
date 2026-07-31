"""Tests for burst.config.

Each test builds a throwaway copy of the real configs/base.yaml in tmp_path,
edits one thing, and checks that the loader complains about exactly that
thing. Starting from the real base file rather than a fixture means these
tests break if base.yaml drifts, which is the point.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from burst.config import ARMS, ConfigError, load_config, run_name_for

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_BASE = REPO_ROOT / "configs" / "base.yaml"
REAL_RUNS = REPO_ROOT / "configs" / "runs"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def base_dict() -> dict:
    return yaml.safe_load(REAL_BASE.read_text(encoding="utf-8"))


def set_dotted(data: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    for part in parts[:-1]:
        data = data[part]
    data[parts[-1]] = value


def write_base(tmp_path: Path, **edits) -> Path:
    """Write a base config to tmp_path, applying dotted-path edits."""
    data = base_dict()
    for dotted, value in edits.items():
        set_dotted(data, dotted.replace("__", "."), value)
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def write_run(tmp_path: Path, text: str, name: str = "run.yaml") -> Path:
    """Write a run override verbatim.

    Default filename is `run.yaml`, which does not match the seedNN_arm
    pattern, so the filename-vs-contents check is skipped. Tests that care
    about that check pass an explicit name.
    """
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def load(tmp_path: Path, base: Path, run: Path, **kwargs):
    kwargs.setdefault("require_complete", False)
    return load_config(base, run, tmp_path / "out", **kwargs)


# ---------------------------------------------------------------------------
# the five required cases
# ---------------------------------------------------------------------------


def test_typo_in_override_key_raises(tmp_path):
    """`see: 3` must fail loudly, never fall back to the default seed."""
    base = write_base(tmp_path)
    run = write_run(tmp_path, "see: 3\narm: coherent\n")
    with pytest.raises(ConfigError) as exc:
        load(tmp_path, base, run)
    assert "unknown key" in str(exc.value)
    assert "'see'" in str(exc.value)


def test_unknown_nested_override_key_raises(tmp_path):
    base = write_base(tmp_path)
    run = write_run(tmp_path, "seed: 3\narm: coherent\ntraining:\n  batch_sise: 128\n")
    with pytest.raises(ConfigError, match="unknown key 'training.batch_sise'"):
        load(tmp_path, base, run)


def test_override_may_not_change_shared_values(tmp_path):
    """Even a correctly spelled key is rejected if it is not seed or arm."""
    base = write_base(tmp_path)
    run = write_run(tmp_path, "seed: 3\narm: coherent\ntraining:\n  batch_size: 128\n")
    with pytest.raises(ConfigError, match="may only set"):
        load(tmp_path, base, run)


def test_null_injection_fields_raise_for_injecting_arm(tmp_path):
    """coherent/noise/ordinary need injection_step and burst_length_tokens."""
    base = write_base(
        tmp_path,
        model__tie_embeddings=True,
        checkpointing__checkpoint_interval=500,
    )
    run = write_run(tmp_path, "seed: 3\narm: coherent\n")
    with pytest.raises(ConfigError) as exc:
        load(tmp_path, base, run, require_complete=True)
    message = str(exc.value)
    assert "injection.injection_step" in message
    assert "injection.burst_length_tokens" in message
    assert "cannot be launched" in message


@pytest.mark.parametrize("arm", ["coherent", "noise", "ordinary"])
def test_every_injecting_arm_requires_injection_fields(tmp_path, arm):
    base = write_base(
        tmp_path,
        model__tie_embeddings=True,
        checkpointing__checkpoint_interval=500,
    )
    run = write_run(tmp_path, f"seed: 3\narm: {arm}\n")
    with pytest.raises(ConfigError, match="injection.injection_step"):
        load(tmp_path, base, run, require_complete=True)


def test_null_injection_fields_are_fine_for_twin(tmp_path):
    """twin receives no injection, so it launches with those fields null."""
    base = write_base(
        tmp_path,
        model__tie_embeddings=True,
        checkpointing__checkpoint_interval=500,
    )
    run = write_run(tmp_path, "seed: 3\narm: twin\n")
    cfg = load(tmp_path, base, run, require_complete=True)
    assert cfg.arm == "twin"
    assert cfg.injection.injection_step is None
    assert cfg.injection.burst_length_tokens is None
    assert cfg.missing_for_launch == ()


def test_twin_still_requires_the_non_injection_fields(tmp_path):
    """twin is exempt from the injection fields only, not from the rest."""
    base = write_base(tmp_path, model__tie_embeddings=True)  # interval left null
    run = write_run(tmp_path, "seed: 3\narm: twin\n")
    with pytest.raises(ConfigError, match="checkpointing.checkpoint_interval"):
        load(tmp_path, base, run, require_complete=True)


@pytest.mark.parametrize(
    "arm", ["Coherent", "COHERENT", "coherent ", "cohrent", "control", ""]
)
def test_invalid_arm_raises(tmp_path, arm):
    base = write_base(tmp_path)
    run = write_run(tmp_path, f"seed: 3\narm: {arm!r}\n")
    with pytest.raises(ConfigError, match="arm must be exactly one of"):
        load(tmp_path, base, run)


def test_case_variant_arm_gets_a_helpful_hint(tmp_path):
    base = write_base(tmp_path)
    run = write_run(tmp_path, "seed: 3\narm: 'Coherent'\n")
    with pytest.raises(ConfigError, match="Case matters"):
        load(tmp_path, base, run)


def test_token_budget_mismatch_raises_when_total_steps_changes(tmp_path):
    base = write_base(tmp_path, training__total_steps=9537)
    run = write_run(tmp_path, "seed: 3\narm: coherent\n")
    with pytest.raises(ConfigError) as exc:
        load(tmp_path, base, run)
    message = str(exc.value)
    assert "token budget mismatch" in message
    assert "9537" in message
    assert "2499805184" in message


def test_token_budget_holds_for_the_real_base_config():
    data = base_dict()
    product = (
        data["training"]["batch_size"]
        * data["training"]["seq_len"]
        * data["training"]["total_steps"]
    )
    assert product == data["corpus"]["expected_token_budget"] == 2499805184


@pytest.mark.parametrize(
    "dotted,value",
    [
        ("outdir", "/scratch/run"),
        ("output_dir", "/scratch/run"),
        ("checkpointing.checkpoint_dir", "/scratch/ckpt"),
        ("corpus.data_path", "/data/owt"),
        ("model.save_dir", "/models"),
    ],
)
def test_output_path_key_in_config_raises(tmp_path, dotted, value):
    data = base_dict()
    parts = dotted.split(".")
    target = data
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    run = write_run(tmp_path, "seed: 3\narm: coherent\n")
    with pytest.raises(ConfigError) as exc:
        load(tmp_path, base, run)
    message = str(exc.value)
    assert "output-path-like key" in message
    assert "--outdir" in message


def test_output_path_key_in_override_raises(tmp_path):
    base = write_base(tmp_path)
    run = write_run(tmp_path, "seed: 3\narm: coherent\noutdir: /scratch/run\n")
    with pytest.raises(ConfigError, match="output-path-like key"):
        load(tmp_path, base, run)


def test_checkpoint_interval_is_not_mistaken_for_a_path(tmp_path):
    """The denylist must not catch legitimate keys that merely sound similar."""
    base = write_base(tmp_path, checkpointing__checkpoint_interval=500)
    run = write_run(tmp_path, "seed: 3\narm: coherent\n")
    cfg = load(tmp_path, base, run)
    assert cfg.checkpointing.checkpoint_interval == 500


# ---------------------------------------------------------------------------
# YAML numeric traps (requirement 3)
# ---------------------------------------------------------------------------


def test_scientific_notation_string_is_rejected_with_a_hint(tmp_path):
    """PyYAML parses 6e-4 as the string '6e-4'; that must fail, not proceed."""
    assert yaml.safe_load("x: 6e-4")["x"] == "6e-4"  # documents the trap itself

    base_text = REAL_BASE.read_text(encoding="utf-8").replace(
        "peak: 0.0006", "peak: 6e-4"
    )
    base = tmp_path / "base.yaml"
    base.write_text(base_text, encoding="utf-8")
    run = write_run(tmp_path, "seed: 3\narm: coherent\n")
    with pytest.raises(ConfigError) as exc:
        load(tmp_path, base, run)
    message = str(exc.value)
    assert "learning_rate.peak" in message
    assert "scientific notation" in message


def test_bare_no_becomes_a_bool_and_is_rejected(tmp_path):
    """`name: no` parses as False; a string field must reject it."""
    assert yaml.safe_load("x: no")["x"] is False  # documents the trap itself

    base = write_base(tmp_path, optimizer__name=False)
    run = write_run(tmp_path, "seed: 3\narm: coherent\n")
    with pytest.raises(ConfigError) as exc:
        load(tmp_path, base, run)
    assert "optimizer.name" in str(exc.value)
    assert "got bool" in str(exc.value)


def test_bool_is_not_accepted_where_an_int_is_expected(tmp_path):
    """isinstance(True, int) is True in Python; the loader must not be fooled."""
    base = write_base(tmp_path, training__total_steps=True)
    run = write_run(tmp_path, "seed: 3\narm: coherent\n")
    with pytest.raises(ConfigError, match="training.total_steps must be an integer"):
        load(tmp_path, base, run)


def test_numeric_field_arriving_as_a_plain_string_is_rejected(tmp_path):
    base = write_base(tmp_path, training__batch_size="256")
    run = write_run(tmp_path, "seed: 3\narm: coherent\n")
    with pytest.raises(ConfigError, match="training.batch_size must be an integer"):
        load(tmp_path, base, run)


def test_duplicate_key_in_yaml_is_rejected(tmp_path):
    """PyYAML silently keeps the last duplicate; that is unacceptable here."""
    base = write_base(tmp_path)
    run = write_run(tmp_path, "seed: 3\narm: coherent\nseed: 7\n")
    with pytest.raises(ConfigError, match="duplicate key 'seed'"):
        load(tmp_path, base, run)


# ---------------------------------------------------------------------------
# structure, immutability, provenance
# ---------------------------------------------------------------------------


def test_config_is_frozen(tmp_path):
    base = write_base(tmp_path)
    run = write_run(tmp_path, "seed: 3\narm: coherent\n")
    cfg = load(tmp_path, base, run)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.seed = 4
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.training.batch_size = 1
    # arms is a tuple, so it cannot be appended to either
    assert isinstance(cfg.experiment.arms, tuple)
    with pytest.raises(AttributeError):
        cfg.experiment.arms.append("extra")


def test_run_name_is_zero_padded(tmp_path):
    base = write_base(tmp_path)
    for seed, expected in [(3, "seed03_coherent"), (0, "seed00_coherent")]:
        run = write_run(tmp_path, f"seed: {seed}\narm: coherent\n")
        cfg = load_config(base, run, tmp_path / f"out{seed}", require_complete=False)
        assert cfg.run_name == expected


def test_filename_must_match_contents(tmp_path):
    """seed05_noise.yaml containing `seed: 4` is a copy-paste error."""
    base = write_base(tmp_path)
    run = write_run(tmp_path, "seed: 4\narm: noise\n", name="seed05_noise.yaml")
    with pytest.raises(ConfigError, match="filename says"):
        load(tmp_path, base, run)


def test_seed_out_of_range_raises(tmp_path):
    base = write_base(tmp_path)
    run = write_run(tmp_path, "seed: 10\narm: coherent\n")
    with pytest.raises(ConfigError, match="seed must be in 0..9"):
        load(tmp_path, base, run)


def test_extra_key_in_base_config_raises(tmp_path):
    """A setting in base.yaml that the loader ignores is a provenance hole."""
    data = base_dict()
    data["training"]["gradient_accumulation"] = 4
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    run = write_run(tmp_path, "seed: 3\narm: coherent\n")
    with pytest.raises(ConfigError, match="unexpected: gradient_accumulation"):
        load(tmp_path, base, run)


def test_missing_key_in_base_config_raises(tmp_path):
    data = base_dict()
    del data["training"]["batch_size"]
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    run = write_run(tmp_path, "seed: 3\narm: coherent\n")
    with pytest.raises(ConfigError, match="missing: batch_size"):
        load(tmp_path, base, run)


def test_provenance_files_are_written(tmp_path):
    base = write_base(tmp_path)
    run = write_run(tmp_path, "seed: 3\narm: coherent\n")
    outdir = tmp_path / "out"
    cfg = load_config(base, run, outdir, require_complete=False)

    resolved = outdir / "resolved_config.yaml"
    provenance = outdir / "run_provenance.yaml"
    assert resolved.is_file()
    assert provenance.is_file()

    # The resolved config must be reloadable and must equal what was loaded.
    reloaded = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    assert reloaded["seed"] == 3
    assert reloaded["arm"] == "coherent"
    assert reloaded["learning_rate"]["peak"] == pytest.approx(0.0006)
    assert isinstance(reloaded["learning_rate"]["final"], float)
    assert reloaded["learning_rate"]["final"] == pytest.approx(0.00006)

    meta = yaml.safe_load(provenance.read_text(encoding="utf-8"))
    assert meta["run_name"] == cfg.run_name == "seed03_coherent"
    assert "git" in meta and "dirty" in meta["git"] and "commit" in meta["git"]
    assert meta["launch_ready"] is False
    assert "injection.injection_step" in meta["missing_for_launch"]


def test_reloading_the_same_run_into_the_same_outdir_is_allowed(tmp_path):
    base = write_base(tmp_path)
    run = write_run(tmp_path, "seed: 3\narm: coherent\n")
    outdir = tmp_path / "out"
    load_config(base, run, outdir, require_complete=False)
    load_config(base, run, outdir, require_complete=False)  # identical, fine


def test_writing_a_different_config_into_a_used_outdir_raises(tmp_path):
    base = write_base(tmp_path)
    outdir = tmp_path / "out"
    load_config(base, write_run(tmp_path, "seed: 3\narm: coherent\n"),
                outdir, require_complete=False)
    other = write_run(tmp_path, "seed: 4\narm: twin\n", name="other.yaml")
    with pytest.raises(ConfigError, match="describes a DIFFERENT config"):
        load_config(base, other, outdir, require_complete=False)
    # --force is the documented escape hatch
    load_config(base, other, outdir, require_complete=False, force=True)


def test_missing_file_raises_clearly(tmp_path):
    base = write_base(tmp_path)
    with pytest.raises(ConfigError, match="config file not found"):
        load(tmp_path, base, tmp_path / "nope.yaml")


# ---------------------------------------------------------------------------
# the generated override files and the CLI
# ---------------------------------------------------------------------------


def test_all_forty_override_files_exist_and_are_two_lines():
    files = sorted(REAL_RUNS.glob("*.yaml"))
    assert len(files) == 40, f"expected 40 override files, found {len(files)}"
    for path in files:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2, f"{path.name} should be two lines, got {len(lines)}"


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("arm", ARMS)
def test_every_override_file_loads(tmp_path, seed, arm):
    name = run_name_for(seed, arm)
    cfg = load_config(REAL_BASE, REAL_RUNS / f"{name}.yaml",
                      tmp_path / name, require_complete=False)
    assert cfg.run_name == name
    assert cfg.seed == seed
    assert cfg.arm == arm


def test_the_forty_runs_differ_only_in_seed_and_arm(tmp_path):
    """The study's central claim, checked mechanically."""
    shared = None
    for seed in range(10):
        for arm in ARMS:
            name = run_name_for(seed, arm)
            cfg = load_config(REAL_BASE, REAL_RUNS / f"{name}.yaml",
                              tmp_path / name, require_complete=False)
            rest = {k: v for k, v in dataclasses.asdict(cfg).items()
                    if k not in ("seed", "arm")}
            if shared is None:
                shared = rest
            else:
                assert rest == shared, f"{name} differs beyond seed and arm"


def test_generator_check_mode_passes():
    """The committed override files match what the generator would produce."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "generate_overrides.py"),
         "--check"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_cli_acceptance_command(tmp_path):
    outdir = tmp_path / "testrun"
    result = subprocess.run(
        [sys.executable, "-m", "burst.config",
         "--config", str(REAL_BASE),
         "--run", str(REAL_RUNS / "seed03_coherent.yaml"),
         "--outdir", str(outdir)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "run_name: seed03_coherent" in result.stdout
    assert "NOT LAUNCH-READY" in result.stdout
    assert (outdir / "resolved_config.yaml").is_file()
    assert (outdir / "run_provenance.yaml").is_file()


def test_cli_launch_flag_fails_while_fields_are_undecided(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "burst.config",
         "--config", str(REAL_BASE),
         "--run", str(REAL_RUNS / "seed03_coherent.yaml"),
         "--outdir", str(tmp_path / "testrun"), "--launch"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 1
    assert "cannot be launched" in result.stderr


def test_cli_reports_config_errors_without_a_traceback(tmp_path):
    run = write_run(tmp_path, "see: 3\narm: coherent\n")
    result = subprocess.run(
        [sys.executable, "-m", "burst.config",
         "--config", str(REAL_BASE), "--run", str(run),
         "--outdir", str(tmp_path / "testrun")],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 1
    assert "CONFIG ERROR" in result.stderr
    assert "Traceback" not in result.stderr
