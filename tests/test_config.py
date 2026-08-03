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

from burst.config import (ARMS, INJECTING_ARMS, ConfigError, load_config,
                          run_name_for)

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


def _paths(**overrides) -> dict:
    """A full burst_text_paths mapping: every injecting arm, most of them null.

    All keys must be present because the shape check compares the mapping
    against INJECTING_ARMS, so a test cannot set one arm's path in isolation.
    """
    mapping = {arm: None for arm in INJECTING_ARMS}
    mapping.update(overrides)
    return mapping


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
    run = write_run(tmp_path, "see: 3\narm: fluent-false\n")
    with pytest.raises(ConfigError) as exc:
        load(tmp_path, base, run)
    assert "unknown key" in str(exc.value)
    assert "'see'" in str(exc.value)


def test_unknown_nested_override_key_raises(tmp_path):
    base = write_base(tmp_path)
    run = write_run(tmp_path, "seed: 3\narm: fluent-false\ntraining:\n  batch_sise: 128\n")
    with pytest.raises(ConfigError, match="unknown key 'training.batch_sise'"):
        load(tmp_path, base, run)


def test_override_may_not_change_shared_values(tmp_path):
    """Even a correctly spelled key is rejected if it is not seed or arm."""
    base = write_base(tmp_path)
    run = write_run(tmp_path, "seed: 3\narm: fluent-false\ntraining:\n  batch_size: 128\n")
    with pytest.raises(ConfigError, match="may only set"):
        load(tmp_path, base, run)


def test_null_injection_fields_raise_for_injecting_arm(tmp_path):
    """coherent/noise/ordinary need injection_step and burst_length_tokens."""
    base = write_base(tmp_path, checkpointing__weights_only_interval=50, checkpointing__full_interval=1000)
    run = write_run(tmp_path, "seed: 3\narm: fluent-false\n")
    with pytest.raises(ConfigError) as exc:
        load(tmp_path, base, run, require_complete=True)
    message = str(exc.value)
    assert "injection.injection_step" in message
    assert "injection.burst_length_tokens" in message
    assert "injection.burst_text_paths.fluent-false" in message
    assert "cannot be launched" in message


@pytest.mark.parametrize("arm", ["fluent-false", "scrambled-false", "pos-substituted"])
def test_every_injecting_arm_requires_injection_fields(tmp_path, arm):
    base = write_base(tmp_path, checkpointing__weights_only_interval=50, checkpointing__full_interval=1000)
    run = write_run(tmp_path, f"seed: 3\narm: {arm}\n")
    with pytest.raises(ConfigError) as exc:
        load(tmp_path, base, run, require_complete=True)
    assert "injection.injection_step" in str(exc.value)
    # only this arm's burst text is demanded, not the other two
    assert f"injection.burst_text_paths.{arm}" in str(exc.value)
    for other in set(ARMS) - {arm}:
        assert f"burst_text_paths.{other}" not in str(exc.value)


def test_null_injection_fields_are_fine_for_twin(tmp_path):
    """twin receives no injection, so it launches with those fields null.

    It is NOT exempt from optimizer.grad_clip: clipping applies to the whole
    run, not just to the injection step, so twin needs that decision too.
    """
    base = write_base(tmp_path, checkpointing__weights_only_interval=50,
                      checkpointing__full_interval=1000,
                      training__micro_batch=8,
                      training__dtype="fp32",
                      optimizer__adamw_impl="foreach",
                      optimizer__grad_clip=1.0)
    run = write_run(tmp_path, "seed: 3\narm: twin\n")
    cfg = load(tmp_path, base, run, require_complete=True)
    assert cfg.arm == "twin"
    assert cfg.injection.injection_step is None
    assert cfg.injection.burst_length_tokens is None
    assert cfg.injection.burst_text_paths.for_arm("twin") is None
    assert cfg.missing_for_launch == ()


def test_twin_still_requires_the_checkpoint_intervals(tmp_path):
    """twin is exempt from the injection fields only, not from the rest.

    This is the retargeted null-required-field coverage for a non-injection
    field: it used to hang off checkpoint_interval, which no longer exists.
    Both replacement fields now carry it.
    """
    base = write_base(
        tmp_path,
        checkpointing__weights_only_interval=None,
        checkpointing__full_interval=None,
    )
    run = write_run(tmp_path, "seed: 3\narm: twin\n")
    with pytest.raises(ConfigError) as exc:
        load(tmp_path, base, run, require_complete=True)
    message = str(exc.value)
    assert "checkpointing.weights_only_interval" in message
    assert "checkpointing.full_interval" in message
    assert "cannot be launched" in message


@pytest.mark.parametrize("arm", ARMS)
def test_checkpoint_intervals_required_by_every_arm(tmp_path, arm):
    base = write_base(
        tmp_path,
        checkpointing__weights_only_interval=None,
        checkpointing__full_interval=None,
    )
    run = write_run(tmp_path, f"seed: 3\narm: {arm}\n")
    with pytest.raises(ConfigError, match="checkpointing.weights_only_interval"):
        load(tmp_path, base, run, require_complete=True)


@pytest.mark.parametrize(
    "field", ["weights_only_interval", "full_interval"]
)
def test_either_checkpoint_interval_null_alone_raises(tmp_path, field):
    """Each field is required independently, not just as a pair."""
    base = write_base(tmp_path, **{f"checkpointing__{field}": None})
    run = write_run(tmp_path, "seed: 3\narm: twin\n")
    with pytest.raises(ConfigError, match=f"checkpointing.{field}"):
        load(tmp_path, base, run, require_complete=True)


def test_tie_embeddings_is_decided_in_the_real_base_config():
    """It is `true`; 124439808 is the tied-embedding parameter count."""
    data = base_dict()
    assert data["model"]["tie_embeddings"] is True
    assert data["model"]["expected_param_count"] == 124439808


def test_checkpoint_intervals_are_decided_in_the_real_base_config():
    ckpt = base_dict()["checkpointing"]
    assert set(ckpt) == {"weights_only_interval", "full_interval"}
    assert ckpt["weights_only_interval"] == 50
    assert ckpt["full_interval"] == 1000
    # the whole point of the multiple rule
    assert ckpt["full_interval"] % ckpt["weights_only_interval"] == 0


def test_the_old_checkpoint_interval_key_is_gone_from_the_base_config():
    assert "checkpoint_interval" not in base_dict()["checkpointing"]


def test_null_tie_embeddings_would_still_raise(tmp_path):
    """The check did not go away when the value was decided."""
    base = write_base(
        tmp_path,
        model__tie_embeddings=None,
        checkpointing__weights_only_interval=50, checkpointing__full_interval=1000,
    )
    run = write_run(tmp_path, "seed: 3\narm: twin\n")
    with pytest.raises(ConfigError, match="model.tie_embeddings"):
        load(tmp_path, base, run, require_complete=True)


@pytest.mark.parametrize(
    "arm", ["Fluent-False", "COHERENT", "coherent ", "cohrent", "control", ""]
)
def test_invalid_arm_raises(tmp_path, arm):
    base = write_base(tmp_path)
    run = write_run(tmp_path, f"seed: 3\narm: {arm!r}\n")
    with pytest.raises(ConfigError, match="arm must be exactly one of"):
        load(tmp_path, base, run)


def test_case_variant_arm_gets_a_helpful_hint(tmp_path):
    base = write_base(tmp_path)
    # Must lowercase INTO ARMS for the hint to fire at all.
    run = write_run(tmp_path, "seed: 3\narm: 'Fluent-False'\n")
    with pytest.raises(ConfigError, match="Case matters"):
        load(tmp_path, base, run)


def test_token_budget_mismatch_raises_when_total_steps_changes(tmp_path):
    base = write_base(tmp_path, training__total_steps=9537)
    run = write_run(tmp_path, "seed: 3\narm: fluent-false\n")
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
    run = write_run(tmp_path, "seed: 3\narm: fluent-false\n")
    with pytest.raises(ConfigError) as exc:
        load(tmp_path, base, run)
    message = str(exc.value)
    assert "output-path-like key" in message
    assert "--outdir" in message


def test_output_path_key_in_override_raises(tmp_path):
    base = write_base(tmp_path)
    run = write_run(tmp_path, "seed: 3\narm: fluent-false\noutdir: /scratch/run\n")
    with pytest.raises(ConfigError, match="output-path-like key"):
        load(tmp_path, base, run)


def test_checkpoint_intervals_are_not_mistaken_for_paths(tmp_path):
    """The denylist must not catch legitimate keys that merely sound similar."""
    base = write_base(tmp_path)
    run = write_run(tmp_path, "seed: 3\narm: fluent-false\n")
    cfg = load(tmp_path, base, run)
    assert cfg.checkpointing.weights_only_interval == 50
    assert cfg.checkpointing.full_interval == 1000


# ---------------------------------------------------------------------------
# corpus must be named, never located
# ---------------------------------------------------------------------------


def test_corpus_names_the_dataset_and_holds_no_path():
    corpus = base_dict()["corpus"]
    assert corpus["name"] == "openwebtext"
    for key in corpus:
        lowered = key.lower()
        assert not lowered.endswith(("_dir", "_path", "_directory", "_folder"))
        assert lowered not in {"path", "dir", "root_dir", "data_dir", "location"}


@pytest.mark.parametrize(
    "key", ["data_path", "data_dir", "corpus_path", "root_dir", "path"]
)
def test_corpus_path_key_is_rejected(tmp_path, key):
    data = base_dict()
    data["corpus"][key] = "/data/openwebtext"
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    run = write_run(tmp_path, "seed: 3\narm: fluent-false\n")
    with pytest.raises(ConfigError, match="output-path-like key"):
        load(tmp_path, base, run)


# ---------------------------------------------------------------------------
# checkpoint schedule: validation, precedence, derived storage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1, -50])
@pytest.mark.parametrize("field", ["weights_only_interval", "full_interval"])
def test_non_positive_checkpoint_interval_raises(tmp_path, field, bad):
    base = write_base(tmp_path, **{f"checkpointing__{field}": bad})
    run = write_run(tmp_path, "seed: 3\narm: twin\n")
    with pytest.raises(ConfigError) as exc:
        load(tmp_path, base, run)
    message = str(exc.value)
    assert f"checkpointing.{field}" in message
    assert "positive" in message


@pytest.mark.parametrize("full", [1010, 51, 999, 75, 1049])
def test_full_interval_must_be_a_multiple_of_weights_only(tmp_path, full):
    base = write_base(tmp_path, checkpointing__full_interval=full)
    run = write_run(tmp_path, "seed: 3\narm: twin\n")
    with pytest.raises(ConfigError) as exc:
        load(tmp_path, base, run)
    message = str(exc.value)
    assert "exact multiple" in message
    assert "precedence" in message
    assert str(full) in message


@pytest.mark.parametrize("full", [50, 100, 1000, 1500, 2000, 9550])
def test_multiples_of_weights_only_interval_are_accepted(tmp_path, full):
    base = write_base(tmp_path, checkpointing__full_interval=full)
    run = write_run(tmp_path, "seed: 3\narm: twin\n")
    cfg = load(tmp_path, base, run)
    assert cfg.checkpointing.full_interval == full


def test_float_interval_is_rejected(tmp_path):
    base = write_base(tmp_path, checkpointing__weights_only_interval=50.5)
    run = write_run(tmp_path, "seed: 3\narm: twin\n")
    with pytest.raises(ConfigError,
                       match="weights_only_interval must be an integer"):
        load(tmp_path, base, run)


# --- the removed key ---


def test_old_checkpoint_interval_key_in_base_raises(tmp_path):
    data = base_dict()
    data["checkpointing"]["checkpoint_interval"] = 500
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    run = write_run(tmp_path, "seed: 3\narm: fluent-false\n")
    with pytest.raises(ConfigError) as exc:
        load(tmp_path, base, run)
    message = str(exc.value)
    assert "no longer exists" in message
    # the error must name both replacements
    assert "checkpointing.weights_only_interval" in message
    assert "checkpointing.full_interval" in message


def test_old_checkpoint_interval_key_in_override_raises(tmp_path):
    base = write_base(tmp_path)
    run = write_run(tmp_path,
                    "seed: 3\narm: fluent-false\ncheckpointing:\n"
                    "  checkpoint_interval: 500\n")
    with pytest.raises(ConfigError) as exc:
        load(tmp_path, base, run)
    assert "no longer exists" in str(exc.value)
    assert "weights_only_interval" in str(exc.value)


def test_old_key_at_top_level_also_raises(tmp_path):
    """Caught wherever it appears, not only in the checkpointing section."""
    base = write_base(tmp_path)
    run = write_run(tmp_path, "seed: 3\narm: fluent-false\ncheckpoint_interval: 500\n")
    with pytest.raises(ConfigError, match="no longer exists"):
        load(tmp_path, base, run)


# --- precedence and the final-step rule ---


def real_cfg(tmp_path, **edits):
    base = write_base(tmp_path, **edits)
    run = write_run(tmp_path, "seed: 3\narm: twin\n")
    return load(tmp_path, base, run)


def test_last_step_is_zero_indexed(tmp_path):
    cfg = real_cfg(tmp_path)
    assert cfg.training.total_steps == 9536
    assert cfg.last_step == 9535


def test_checkpoint_kind_precedence_full_wins(tmp_path):
    """When both intervals fire on the same step, only the full one counts."""
    cfg = real_cfg(tmp_path)
    # step 999 -> 1000 completed steps -> both 50 and 1000 divide it
    assert 1000 % 50 == 0
    assert cfg.checkpoint_kind_at(999) == "full"
    # a step where only the 50 interval fires
    assert cfg.checkpoint_kind_at(49) == "weights_only"
    # a step where neither fires
    assert cfg.checkpoint_kind_at(48) is None


@pytest.mark.parametrize("step", [999, 1999, 4999, 8999])
def test_every_full_firing_step_is_full_not_weights_only(tmp_path, step):
    cfg = real_cfg(tmp_path)
    assert cfg.checkpoint_kind_at(step) == "full"


def test_final_step_is_always_full(tmp_path):
    """9536 divides by neither 50 nor 1000, so only the rule saves it."""
    cfg = real_cfg(tmp_path)
    assert 9536 % 50 != 0 and 9536 % 1000 != 0
    assert cfg.checkpoint_kind_at(cfg.last_step) == "full"


def test_final_step_rule_follows_total_steps_not_a_hardcoded_9536(tmp_path):
    cfg = real_cfg(tmp_path, training__total_steps=2000,
                   corpus__expected_token_budget=256 * 1024 * 2000)
    assert cfg.last_step == 1999
    assert cfg.checkpoint_kind_at(1999) == "full"
    with pytest.raises(ConfigError, match="outside this run"):
        cfg.checkpoint_kind_at(9535)


def test_checkpoint_kind_rejects_out_of_range_steps(tmp_path):
    cfg = real_cfg(tmp_path)
    for bad in (-1, 9536, 99999):
        with pytest.raises(ConfigError, match="outside this run"):
            cfg.checkpoint_kind_at(bad)


def test_first_weights_only_checkpoint_is_after_n_completed_steps(tmp_path):
    """Interval 50 means step 49, not step 0 and not step 50."""
    cfg = real_cfg(tmp_path)
    assert cfg.checkpoint_kind_at(0) is None
    assert cfg.checkpoint_kind_at(49) == "weights_only"
    assert cfg.checkpoint_kind_at(50) is None


# --- derived storage ---


def test_checkpoint_plan_counts_match_a_brute_force_walk(tmp_path):
    """The counting formula must agree with walking every step."""
    cfg = real_cfg(tmp_path)
    plan = cfg.checkpoint_plan
    kinds = [cfg.checkpoint_kind_at(s) for s in range(cfg.training.total_steps)]
    assert plan.weights_only_count == kinds.count("weights_only")
    assert plan.full_count == kinds.count("full")


def test_checkpoint_plan_for_the_real_config(tmp_path):
    cfg = real_cfg(tmp_path)
    plan = cfg.checkpoint_plan
    # 9536 // 50 = 190 firings, 9536 // 1000 = 9 of which are full,
    # plus one more full for the final step (9536 divides neither).
    assert plan.weights_only_count == 181
    assert plan.full_count == 10
    assert plan.estimated_bytes_per_run == 181 * 500_000_000 + 10 * 1_500_000_000
    assert plan.estimated_bytes_per_run == 105_500_000_000       # 105.5 GB
    assert plan.estimated_bytes_all_runs == 105_500_000_000 * 70  # 7.385 TB
    assert plan.last_step == 9535


@pytest.mark.parametrize(
    "total,wo,full,exp_wo,exp_full",
    [
        # last step is neither firing -> final rule adds a full checkpoint
        (9536, 50, 1000, 181, 10),
        # last step is exactly a full firing -> no extra checkpoint
        (1000, 50, 1000, 19, 1),
        # last step is a weights-only firing -> promoted to full, not counted twice
        (1050, 50, 1000, 19, 2),
        # intervals larger than the run -> only the mandatory final full one
        (100, 200, 400, 0, 1),
    ],
)
def test_checkpoint_plan_edge_cases(tmp_path, total, wo, full, exp_wo, exp_full):
    cfg = real_cfg(
        tmp_path,
        training__total_steps=total,
        corpus__expected_token_budget=256 * 1024 * total,
        checkpointing__weights_only_interval=wo,
        checkpointing__full_interval=full,
        learning_rate__warmup_steps=min(200, total - 1),
    )
    plan = cfg.checkpoint_plan
    assert (plan.weights_only_count, plan.full_count) == (exp_wo, exp_full)
    kinds = [cfg.checkpoint_kind_at(s) for s in range(total)]
    assert plan.weights_only_count == kinds.count("weights_only")
    assert plan.full_count == kinds.count("full")


def test_checkpoint_plan_tracks_total_steps(tmp_path):
    """Derived, not hardcoded: change total_steps and the plan changes."""
    cfg = real_cfg(tmp_path, training__total_steps=4768,
                   corpus__expected_token_budget=256 * 1024 * 4768)
    plan = cfg.checkpoint_plan
    assert plan.last_step == 4767
    assert plan.weights_only_count == 4768 // 50 - 4768 // 1000   # 95 - 4 = 91
    assert plan.full_count == 4768 // 1000 + 1                    # 4 + 1 = 5


def test_checkpoint_plan_raises_while_intervals_are_undecided(tmp_path):
    cfg = real_cfg(tmp_path, checkpointing__full_interval=None)
    with pytest.raises(ConfigError, match="not decided yet"):
        _ = cfg.checkpoint_plan
    with pytest.raises(ConfigError, match="not decided yet"):
        cfg.checkpoint_kind_at(49)


def test_checkpoint_plan_is_frozen(tmp_path):
    plan = real_cfg(tmp_path).checkpoint_plan
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.full_count = 999


def test_checkpoint_plan_is_recorded_in_provenance(tmp_path):
    base = write_base(tmp_path)
    run = write_run(tmp_path, "seed: 3\narm: twin\n")
    outdir = tmp_path / "out"
    load_config(base, run, outdir, require_complete=False)
    meta = yaml.safe_load((outdir / "run_provenance.yaml").read_text(encoding="utf-8"))
    assert meta["checkpoint_plan"]["weights_only_count"] == 181
    assert meta["checkpoint_plan"]["full_count"] == 10
    assert meta["checkpoint_plan"]["last_step"] == 9535


# ---------------------------------------------------------------------------
# burst text paths must be repo-relative and inside the repo
# ---------------------------------------------------------------------------


def valid_burst_base(tmp_path, path_value):
    return write_base(
        tmp_path,
        checkpointing__weights_only_interval=50, checkpointing__full_interval=1000,
        # grad_clip is null in the shipped config and is rejected at launch,
        # so a config that is meant to BE launch-ready has to decide it.
        optimizer__grad_clip=1.0,
        # Likewise micro_batch: it is null in the shipped config on purpose and
        # every arm needs it, because the accumulation shape is part of what
        # makes two runs the same run.
        training__micro_batch=8,
        training__dtype="fp32",
        optimizer__adamw_impl="foreach",
        injection__injection_step=4768,
        injection__burst_length_tokens=64,
        injection__burst_text_paths=_paths(**{'fluent-false': path_value}),
    )


@pytest.mark.parametrize(
    "bad",
    [
        "/home/zach/burst/coherent.txt",       # posix absolute
        "C:\\Users\\speck\\coherent.txt",      # windows absolute
        "\\\\cluster\\share\\coherent.txt",    # UNC
        "C:coherent.txt",                      # windows drive-relative
    ],
)
def test_absolute_burst_text_path_is_rejected(tmp_path, bad):
    """Rejected identically on Windows and Linux, not just on the host OS."""
    base = valid_burst_base(tmp_path, bad)
    run = write_run(tmp_path, "seed: 3\narm: fluent-false\n")
    with pytest.raises(ConfigError) as exc:
        load(tmp_path, base, run)
    message = str(exc.value)
    assert "injection.burst_text_paths.fluent-false" in message
    assert "absolute path" in message
    assert "version" in message  # explains it must be version-controlled


def test_burst_text_path_escaping_the_repo_is_rejected(tmp_path):
    base = valid_burst_base(tmp_path, "../../outside/coherent.txt")
    run = write_run(tmp_path, "seed: 3\narm: fluent-false\n")
    with pytest.raises(ConfigError) as exc:
        load(tmp_path, base, run)
    message = str(exc.value)
    assert "outside the repository" in message
    assert "git commit hash" in message


def test_empty_burst_text_path_is_rejected(tmp_path):
    base = valid_burst_base(tmp_path, "   ")
    run = write_run(tmp_path, "seed: 3\narm: fluent-false\n")
    with pytest.raises(ConfigError, match="is empty"):
        load(tmp_path, base, run)


def test_repo_relative_burst_text_path_is_accepted(tmp_path):
    base = valid_burst_base(tmp_path, "configs/burst_texts/coherent.txt")
    run = write_run(tmp_path, "seed: 3\narm: fluent-false\n")
    cfg = load(tmp_path, base, run)
    assert cfg.injection.burst_text_paths.for_arm("fluent-false") == "configs/burst_texts/coherent.txt"
    assert cfg.injection.burst_text_paths.for_arm("fluent-false") == (
        "configs/burst_texts/coherent.txt"
    )


def test_burst_text_paths_key_is_not_caught_by_the_output_path_rule(tmp_path):
    """The exemption works; a path-holding content key is allowed through."""
    base = valid_burst_base(tmp_path, "configs/burst_texts/coherent.txt")
    run = write_run(tmp_path, "seed: 3\narm: fluent-false\n")
    load(tmp_path, base, run)  # must not raise


def test_launch_requires_the_burst_text_file_to_exist(tmp_path):
    base = valid_burst_base(tmp_path, "configs/burst_texts/nope.txt")
    run = write_run(tmp_path, "seed: 3\narm: fluent-false\n")
    with pytest.raises(ConfigError, match="no file exists at"):
        load(tmp_path, base, run, require_complete=True)


def test_launch_succeeds_when_the_burst_text_file_exists(tmp_path):
    # README.md stands in for a burst text: it is a real, committed file
    # inside the repo, which is exactly what the rule requires.
    base = valid_burst_base(tmp_path, "README.md")
    run = write_run(tmp_path, "seed: 3\narm: fluent-false\n")
    cfg = load(tmp_path, base, run, require_complete=True)
    assert cfg.missing_for_launch == ()


def test_twin_needs_no_burst_text(tmp_path):
    """twin launches with every burst text path still null."""
    base = write_base(tmp_path, checkpointing__weights_only_interval=50,
                      checkpointing__full_interval=1000,
                      training__micro_batch=8,
                      training__dtype="fp32",
                      optimizer__adamw_impl="foreach",
                      optimizer__grad_clip=1.0)
    run = write_run(tmp_path, "seed: 3\narm: twin\n")
    cfg = load(tmp_path, base, run, require_complete=True)
    assert cfg.missing_for_launch == ()


def test_a_twin_burst_text_entry_is_rejected(tmp_path):
    """twin receives no text, so there is no slot for one."""
    data = base_dict()
    data["injection"]["burst_text_paths"]["twin"] = "configs/burst_texts/twin.txt"
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    run = write_run(tmp_path, "seed: 3\narm: fluent-false\n")
    with pytest.raises(ConfigError, match="unexpected: twin"):
        load(tmp_path, base, run)


def test_burst_text_paths_start_null_in_the_real_base_config():
    paths = base_dict()["injection"]["burst_text_paths"]
    assert set(paths) == set(INJECTING_ARMS)
    assert all(value is None for value in paths.values())


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
    run = write_run(tmp_path, "seed: 3\narm: fluent-false\n")
    with pytest.raises(ConfigError) as exc:
        load(tmp_path, base, run)
    message = str(exc.value)
    assert "learning_rate.peak" in message
    assert "scientific notation" in message


def test_bare_no_becomes_a_bool_and_is_rejected(tmp_path):
    """`name: no` parses as False; a string field must reject it."""
    assert yaml.safe_load("x: no")["x"] is False  # documents the trap itself

    base = write_base(tmp_path, optimizer__name=False)
    run = write_run(tmp_path, "seed: 3\narm: fluent-false\n")
    with pytest.raises(ConfigError) as exc:
        load(tmp_path, base, run)
    assert "optimizer.name" in str(exc.value)
    assert "got bool" in str(exc.value)


def test_bool_is_not_accepted_where_an_int_is_expected(tmp_path):
    """isinstance(True, int) is True in Python; the loader must not be fooled."""
    base = write_base(tmp_path, training__total_steps=True)
    run = write_run(tmp_path, "seed: 3\narm: fluent-false\n")
    with pytest.raises(ConfigError, match="training.total_steps must be an integer"):
        load(tmp_path, base, run)


def test_numeric_field_arriving_as_a_plain_string_is_rejected(tmp_path):
    base = write_base(tmp_path, training__batch_size="256")
    run = write_run(tmp_path, "seed: 3\narm: fluent-false\n")
    with pytest.raises(ConfigError, match="training.batch_size must be an integer"):
        load(tmp_path, base, run)


def test_duplicate_key_in_yaml_is_rejected(tmp_path):
    """PyYAML silently keeps the last duplicate; that is unacceptable here."""
    base = write_base(tmp_path)
    run = write_run(tmp_path, "seed: 3\narm: fluent-false\nseed: 7\n")
    with pytest.raises(ConfigError, match="duplicate key 'seed'"):
        load(tmp_path, base, run)


# ---------------------------------------------------------------------------
# structure, immutability, provenance
# ---------------------------------------------------------------------------


def test_config_is_frozen(tmp_path):
    base = write_base(tmp_path)
    run = write_run(tmp_path, "seed: 3\narm: fluent-false\n")
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
    for seed, expected in [(3, "seed03_fluent-false"), (0, "seed00_fluent-false")]:
        run = write_run(tmp_path, f"seed: {seed}\narm: fluent-false\n")
        cfg = load_config(base, run, tmp_path / f"out{seed}", require_complete=False)
        assert cfg.run_name == expected


def test_filename_must_match_contents(tmp_path):
    """seed05_scrambled-false.yaml containing `seed: 4` is a copy-paste error."""
    base = write_base(tmp_path)
    run = write_run(tmp_path, "seed: 4\narm: scrambled-false\n", name="seed05_scrambled-false.yaml")
    with pytest.raises(ConfigError, match="filename says"):
        load(tmp_path, base, run)


def test_seed_out_of_range_raises(tmp_path):
    base = write_base(tmp_path)
    run = write_run(tmp_path, "seed: 10\narm: fluent-false\n")
    with pytest.raises(ConfigError, match="seed must be in 0..9"):
        load(tmp_path, base, run)


def test_extra_key_in_base_config_raises(tmp_path):
    """A setting in base.yaml that the loader ignores is a provenance hole."""
    data = base_dict()
    data["training"]["gradient_accumulation"] = 4
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    run = write_run(tmp_path, "seed: 3\narm: fluent-false\n")
    with pytest.raises(ConfigError, match="unexpected: gradient_accumulation"):
        load(tmp_path, base, run)


def test_missing_key_in_base_config_raises(tmp_path):
    data = base_dict()
    del data["training"]["batch_size"]
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    run = write_run(tmp_path, "seed: 3\narm: fluent-false\n")
    with pytest.raises(ConfigError, match="missing: batch_size"):
        load(tmp_path, base, run)


def test_provenance_files_are_written(tmp_path):
    base = write_base(tmp_path)
    run = write_run(tmp_path, "seed: 3\narm: fluent-false\n")
    outdir = tmp_path / "out"
    cfg = load_config(base, run, outdir, require_complete=False)

    resolved = outdir / "resolved_config.yaml"
    provenance = outdir / "run_provenance.yaml"
    assert resolved.is_file()
    assert provenance.is_file()

    # The resolved config must be reloadable and must equal what was loaded.
    reloaded = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    assert reloaded["seed"] == 3
    assert reloaded["arm"] == "fluent-false"
    assert reloaded["learning_rate"]["peak"] == pytest.approx(0.0006)
    assert isinstance(reloaded["learning_rate"]["final"], float)
    assert reloaded["learning_rate"]["final"] == pytest.approx(0.00006)

    meta = yaml.safe_load(provenance.read_text(encoding="utf-8"))
    assert meta["run_name"] == cfg.run_name == "seed03_fluent-false"
    assert "git" in meta and "dirty" in meta["git"] and "commit" in meta["git"]
    assert meta["launch_ready"] is False
    assert "injection.injection_step" in meta["missing_for_launch"]


def test_reloading_the_same_run_into_the_same_outdir_is_allowed(tmp_path):
    base = write_base(tmp_path)
    run = write_run(tmp_path, "seed: 3\narm: fluent-false\n")
    outdir = tmp_path / "out"
    load_config(base, run, outdir, require_complete=False)
    load_config(base, run, outdir, require_complete=False)  # identical, fine


def test_writing_a_different_config_into_a_used_outdir_raises(tmp_path):
    base = write_base(tmp_path)
    outdir = tmp_path / "out"
    load_config(base, write_run(tmp_path, "seed: 3\narm: fluent-false\n"),
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


def test_every_override_file_exists_and_is_two_lines():
    files = sorted(REAL_RUNS.glob("*.yaml"))
    expected = 10 * len(ARMS)
    assert len(files) == expected, (
        f"expected {expected} override files, found {len(files)}")
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


def test_every_run_differs_only_in_seed_and_arm(tmp_path):
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
         "--run", str(REAL_RUNS / "seed03_fluent-false.yaml"),
         "--outdir", str(outdir)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "run_name: seed03_fluent-false" in result.stdout
    assert "NOT LAUNCH-READY" in result.stdout
    assert (outdir / "resolved_config.yaml").is_file()
    assert (outdir / "run_provenance.yaml").is_file()


def test_cli_launch_flag_fails_while_fields_are_undecided(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "burst.config",
         "--config", str(REAL_BASE),
         "--run", str(REAL_RUNS / "seed03_fluent-false.yaml"),
         "--outdir", str(tmp_path / "testrun"), "--launch"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 1
    assert "cannot be launched" in result.stderr


def test_cli_reports_config_errors_without_a_traceback(tmp_path):
    run = write_run(tmp_path, "see: 3\narm: fluent-false\n")
    result = subprocess.run(
        [sys.executable, "-m", "burst.config",
         "--config", str(REAL_BASE), "--run", str(run),
         "--outdir", str(tmp_path / "testrun")],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 1
    assert "CONFIG ERROR" in result.stderr
    assert "Traceback" not in result.stderr


# ---------------------------------------------------------------------------
# optimizer parameters (8b-ii)
# ---------------------------------------------------------------------------


def test_the_real_base_config_decides_the_adamw_parameters():
    """These are fixed across all arms, so they cannot bias a comparison."""
    optimizer = base_dict()["optimizer"]
    assert optimizer["beta1"] == 0.9
    assert optimizer["beta2"] == 0.95
    assert optimizer["eps"] == 0.00000001
    # Decided in 8b-iii at the standard GPT-2 value. See S31 for why this is
    # a convention rather than a safety net for the burst.
    assert optimizer["grad_clip"] == 1.0


def test_eps_is_written_in_plain_decimal_not_scientific_notation():
    """`1e-8` parses as a STRING under PyYAML. Guard the house rule.

    Comments are stripped before checking -- the comment above the value
    quotes `1e-8` precisely to explain why it must not be written that way.
    """
    value_lines = [
        line.split("#")[0]
        for line in REAL_BASE.read_text(encoding="utf-8").splitlines()
    ]
    assert not any("e-" in line or "e+" in line for line in value_lines), (
        "a value in base.yaml is written in scientific notation")
    assert isinstance(base_dict()["optimizer"]["eps"], float)


def test_scientific_notation_eps_is_rejected_with_a_hint(tmp_path):
    base = write_base(tmp_path, optimizer__eps="1e-8")
    run = write_run(tmp_path, "seed: 3\narm: fluent-false\n")
    with pytest.raises(ConfigError) as exc:
        load(tmp_path, base, run)
    message = str(exc.value)
    assert "optimizer.eps" in message
    assert "scientific notation" in message


def test_null_grad_clip_blocks_launch_for_every_arm(tmp_path):
    """Requirement: no run starts without an explicit clipping decision.

    The shipped config now decides it (1.0), so this sets it back to null in a
    throwaway config to keep the CHECK under test -- the same pattern used for
    tie_embeddings once that was decided.
    """
    for arm in ARMS:
        # A directory per arm: the loader refuses to overwrite a
        # resolved_config.yaml describing a DIFFERENT config, and the arm
        # changes on every pass.
        cell = tmp_path / arm
        cell.mkdir()
        base = write_base(cell, checkpointing__weights_only_interval=50,
                          checkpointing__full_interval=1000,
                          optimizer__grad_clip=None,
                          injection__injection_step=4768,
                          injection__burst_length_tokens=64,
                          injection__burst_text_paths=_paths(**{
                              a: "README.md" for a in INJECTING_ARMS}))
        run = write_run(cell, f"seed: 3\narm: {arm}\n")
        cfg = load(cell, base, run)
        assert "optimizer.grad_clip" in cfg.missing_for_launch, (
            f"{arm} should not be launch-ready with grad_clip null")
        with pytest.raises(ConfigError, match="optimizer.grad_clip"):
            load(cell, base, run, require_complete=True)


def test_a_decided_grad_clip_is_accepted(tmp_path):
    base = write_base(tmp_path, checkpointing__weights_only_interval=50,
                      checkpointing__full_interval=1000,
                      training__micro_batch=8,
                      training__dtype="fp32",
                      optimizer__adamw_impl="foreach",
                      optimizer__grad_clip=1.0)
    run = write_run(tmp_path, "seed: 3\narm: twin\n")
    cfg = load(tmp_path, base, run, require_complete=True)
    assert cfg.optimizer.grad_clip == 1.0
    assert cfg.missing_for_launch == ()


def test_a_nonpositive_grad_clip_is_rejected(tmp_path):
    for bad in (0, -1.0):
        base = write_base(tmp_path, optimizer__grad_clip=bad)
        run = write_run(tmp_path, "seed: 3\narm: fluent-false\n")
        with pytest.raises(ConfigError, match="grad_clip"):
            load(tmp_path, base, run)


@pytest.mark.parametrize("field", ["beta1", "beta2"])
@pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 1.5])
def test_betas_outside_the_open_unit_interval_are_rejected(tmp_path, field, bad):
    base = write_base(tmp_path, **{f"optimizer__{field}": bad})
    run = write_run(tmp_path, "seed: 3\narm: fluent-false\n")
    with pytest.raises(ConfigError, match=f"optimizer.{field}"):
        load(tmp_path, base, run)


def test_a_nonpositive_eps_is_rejected(tmp_path):
    base = write_base(tmp_path, optimizer__eps=0.0)
    run = write_run(tmp_path, "seed: 3\narm: fluent-false\n")
    with pytest.raises(ConfigError, match="optimizer.eps"):
        load(tmp_path, base, run)


def test_optimizer_values_survive_into_the_frozen_config(tmp_path):
    base = write_base(tmp_path)
    run = write_run(tmp_path, "seed: 3\narm: fluent-false\n")
    cfg = load(tmp_path, base, run)
    assert cfg.optimizer.beta1 == 0.9
    assert cfg.optimizer.beta2 == 0.95
    assert cfg.optimizer.eps == 0.00000001
    assert cfg.optimizer.grad_clip == 1.0
