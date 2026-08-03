"""Configuration loader for the burst study.

This module is the only thing in the repo. Every later component -- training
loop, injection hook, analysis -- is meant to get its settings from here and
nowhere else, so that "what produced this checkpoint" always has one answer.

Usage from the command line::

    python -m burst.config \\
        --config configs/base.yaml \\
        --run configs/runs/seed03_coherent.yaml \\
        --outdir /tmp/testrun

Usage from Python::

    from burst.config import load_config
    cfg = load_config("configs/base.yaml", "configs/runs/seed03_coherent.yaml",
                      outdir="/scratch/runs/seed03_coherent")
    print(cfg.run_name, cfg.training.total_steps)

Two design notes that are not obvious from the code:

1. Errors are raised, never `assert`ed. `python -O` deletes assert statements.
   A validation layer that silently disappears under an optimisation flag is
   worse than no validation layer, so every check below raises ConfigError.

2. There is no `import torch` here and there never should be. Loading a config
   must work on a laptop with no GPU and no ML stack installed.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml

__all__ = [
    "ADAMW_IMPLS",
    "ARMS",
    "DTYPES",
    "INJECTING_ARMS",
    "Config",
    "ConfigError",
    "load_config",
    "run_name_for",
]


# ---------------------------------------------------------------------------
# Constants that describe the study
# ---------------------------------------------------------------------------

#: The seven arms, exactly as they must appear in a run override file.
#:
#: Reconciled to spec v4 on 2026-08-03. Until then this held the retired v3
#: four -- coherent, noise, ordinary, twin -- while `bursts/` held v4 texts and
#: README.md warned that the two "actively disagree". They now agree.
#:
#: `scrambled-corpus` is CUT. Its text and provenance entry stay in `bursts/`
#: so the measurements taken from it remain reproducible, but it is not a run
#: condition. See S79 for the cost of that cut.
ARMS: tuple[str, ...] = (
    "fluent-false",
    "fluent-true",
    "scrambled-false",
    "scrambled-true",
    "pos-substituted",
    "random-chars",
    "twin",
)

#: The arms that actually receive a burst. `twin` is the matched control: it
#: gets no injection, so it does not need injection_step or
#: burst_length_tokens filled in before it can be launched.
INJECTING_ARMS: tuple[str, ...] = tuple(a for a in ARMS if a != "twin")

#: Compute dtypes the loader will accept. Closed set, validated by hand in
#: _validate_semantics because the loader has no enum helper -- the same
#: pattern `arm` uses against ARMS. `bf16` is accepted by the CONFIG and is
#: NOT IMPLEMENTED by scripts/train.py, which has no autocast; the loop
#: refuses it explicitly rather than training fp32 while the record says bf16.
DTYPES: tuple[str, ...] = ("fp32", "bf16")

#: AdamW implementations. Each groups its arithmetic differently -- per tensor,
#: per fused group, or in one kernel -- so each produces different bits from
#: identical moments.
ADAMW_IMPLS: tuple[str, ...] = ("foreach", "fused", "single")

#: A run override file may set these top-level keys and nothing else. This is
#: stricter than "unknown keys are rejected" on purpose -- the study's claim is
#: that the runs differ only in seed and arm, and an override that quietly
#: changed the learning rate for one run would invalidate that claim without
#: leaving a trace. If a future experiment genuinely needs a third per-run
#: knob, add it here deliberately.
OVERRIDE_ALLOWED_KEYS: frozenset[str] = frozenset({"seed", "arm"})

#: Canonical run name: seed{NN}_{arm}, seed zero-padded to two digits.
#: Hyphens are allowed in the arm segment because every v4 arm name has one
#: (fluent-false, pos-substituted, random-chars). Under the retired v3 names
#: this was `[a-z]+`, which would silently DECLINE TO CHECK every v4 override
#: file -- the filename-vs-contents check only runs when the name matches, so a
#: stricter pattern here does not reject bad files, it stops examining them.
RUN_NAME_PATTERN = re.compile(r"^seed(\d{2})_([a-z-]+)$")

#: The two kinds of checkpoint, returned by Config.checkpoint_kind_at().
CHECKPOINT_FULL = "full"
CHECKPOINT_WEIGHTS_ONLY = "weights_only"

# Per-checkpoint sizes used for the storage estimate.
#
# ESTIMATES, NOT MEASUREMENTS. Derived arithmetically for 124,439,808 fp32
# parameters: weights alone are 124439808 x 4 = 497,759,232 bytes, and a full
# checkpoint adds AdamW's two moment buffers, each another full copy, for
# 1,493,277,696. Rounded to clean decimal GB. Nothing in this repo has ever
# written a checkpoint, so these have not been checked against a real file --
# framework overhead, compression and any bf16 optimizer state would all move
# them. Treat the resulting totals as a planning figure, not a guarantee.
WEIGHTS_ONLY_CHECKPOINT_BYTES = 500_000_000     # 0.5 GB
FULL_CHECKPOINT_BYTES = 1_500_000_000           # 1.5 GB

#: Config keys that used to exist and have been replaced. Checked on every
#: config file before anything else, so an old file fails with an error that
#: names the replacement rather than a generic "unknown key".
REMOVED_KEYS: dict[str, str] = {
    "checkpoint_interval": (
        "replaced by two fields, checkpointing.weights_only_interval and "
        "checkpointing.full_interval.\n"
        "A single interval could not express the actual decision: full "
        "checkpoints (weights + optimizer + step + RNG state, ~1.5 GB) exist "
        "for crash recovery, while weights-only checkpoints (~0.5 GB) exist to "
        "measure how the model changes during training. Every metric in this "
        "study is a function of the weights alone, so the measurement schedule "
        "can be much denser than the recovery schedule.\n"
        "Set both fields in configs/base.yaml. full_interval must be an exact "
        "multiple of weights_only_interval."
    ),
}

#: Filenames written into the output directory.
RESOLVED_CONFIG_FILENAME = "resolved_config.yaml"
PROVENANCE_FILENAME = "run_provenance.yaml"


# Keys that look like an output location. The output directory is a
# command-line argument, not a config value, so that one config file works
# unchanged on a laptop and on a cluster. Matching is on the exact lowercased
# key name, plus the suffixes below -- deliberately not a substring search, so
# that legitimate keys such as `full_interval` are not caught.
OUTPUT_PATH_KEY_DENYLIST: frozenset[str] = frozenset({
    "artifact_dir", "artifacts_dir", "ckpt_dir", "checkpoint_dir",
    "data_dir", "dir", "log_dir", "logdir", "out", "outdir", "out_dir",
    "outpath", "out_path", "output", "outputdir", "output_dir",
    "output_path", "output_root", "path", "result_dir", "results_dir",
    "root_dir", "run_dir", "rundir", "save_dir", "savedir", "save_path",
    "scratch_dir", "work_dir", "workdir",
})
OUTPUT_PATH_KEY_SUFFIXES: tuple[str, ...] = (
    "_dir", "_path", "_directory", "_folder",
)

# Dotted keys that are allowed to hold a path despite the rule above, because
# they point at *input content committed inside this repo* rather than at an
# output location on some particular machine. These get a stricter check of
# their own in _validate_burst_text_paths: relative only, and must resolve
# inside the repo root. Listed explicitly so that renaming the field cannot
# accidentally slip it past either check.
CONTENT_PATH_KEYS_EXEMPT: frozenset[str] = frozenset({
    "injection.burst_text_paths",
})


class ConfigError(Exception):
    """Every failure in this module. One exception type, so calling code and
    tests only ever have to catch one thing."""


# ---------------------------------------------------------------------------
# YAML reading
# ---------------------------------------------------------------------------


class _StrictSafeLoader(yaml.SafeLoader):
    """SafeLoader that refuses duplicate keys in a mapping.

    PyYAML's default behaviour is to silently keep the LAST value for a
    repeated key. In a provenance-critical config that is silent data loss: a
    file containing `seed: 3` near the top and a forgotten `seed: 7` near the
    bottom loads without complaint and the run is not the run you think it is.
    """


def _construct_mapping_no_duplicates(
    loader: _StrictSafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict:
    loader.flatten_mapping(node)
    mapping: dict = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in mapping:
            line = key_node.start_mark.line + 1
            raise ConfigError(
                f"duplicate key {key!r} on line {line}. "
                "PyYAML would silently keep the last one; that is not "
                "acceptable in a config whose whole job is provenance."
            )
        # deep=True forces each value to be fully built here rather than
        # filled in later by PyYAML's two-pass machinery. Config files are
        # small, so the cost is irrelevant and the behaviour is predictable.
        mapping[key] = loader.construct_object(value_node, deep=True)
    return mapping


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping_no_duplicates,
)


def _read_yaml(path: Path) -> dict:
    """Read one YAML file into a plain dict, or raise ConfigError."""
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=_StrictSafeLoader)
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from None
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML:\n{exc}") from exc
    if data is None:
        raise ConfigError(f"{path} is empty")
    if not isinstance(data, dict):
        raise ConfigError(
            f"{path} must contain a YAML mapping at the top level, "
            f"got {type(data).__name__}"
        )
    return data


# ---------------------------------------------------------------------------
# Structural checks on raw dicts, before any typing happens
# ---------------------------------------------------------------------------


def _reject_output_path_keys(data: dict, source: Path, prefix: str = "") -> None:
    """Requirement: the output path is never a config value."""
    for key, value in data.items():
        dotted = f"{prefix}{key}"
        if dotted in CONTENT_PATH_KEYS_EXEMPT:
            # Neither the key itself nor anything under it is an output
            # location. Validated separately and more strictly.
            continue
        lowered = str(key).lower()
        if lowered in OUTPUT_PATH_KEY_DENYLIST or lowered.endswith(
            OUTPUT_PATH_KEY_SUFFIXES
        ):
            raise ConfigError(
                f"{source}: config contains an output-path-like key "
                f"{dotted!r}.\n"
                "Machine-specific locations must never live in a config file. "
                "The same config has to run unchanged on your laptop and on "
                "your collaborator's cluster, and a path baked into the config "
                "makes that impossible -- it also makes resolved_config.yaml "
                "differ between machines for reasons that have nothing to do "
                "with the experiment.\n"
                "Pass the location as a command-line argument instead. The "
                "run's output directory is --outdir; a corpus or dataset "
                "location belongs on the data pipeline's command line, not "
                "here (the corpus is named in the config, never located).\n"
                "The one exception is injection.burst_text_paths, which is "
                "repo-relative experimental content rather than a machine "
                "location, and is validated separately."
            )
        if isinstance(value, dict):
            _reject_output_path_keys(value, source, prefix=f"{dotted}.")


def _reject_removed_keys(data: dict, source: Path, prefix: str = "") -> None:
    """Fail loudly on a key that used to exist, naming what replaced it.

    Runs on the raw files before the merge, so an old config gets this message
    rather than "unknown key" from the merge or "unexpected" from the schema
    check -- neither of which would tell you what to do about it.
    """
    for key, value in data.items():
        dotted = f"{prefix}{key}"
        explanation = REMOVED_KEYS.get(str(key))
        if explanation is not None:
            raise ConfigError(
                f"{source}: {dotted!r} no longer exists -- {explanation}"
            )
        if isinstance(value, dict):
            _reject_removed_keys(value, source, prefix=f"{dotted}.")


def _strict_merge(base: dict, override: dict, source: Path, prefix: str = "") -> dict:
    """Merge `override` onto `base`, rejecting any key not already in `base`.

    This is what makes a typo fail loudly. `see: 3` is not a new key, it is a
    misspelling of `seed`, and silently falling back to the default seed would
    produce a run that is mislabelled forever.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        dotted = f"{prefix}{key}"
        if key not in base:
            known = ", ".join(sorted(str(k) for k in base))
            raise ConfigError(
                f"{source}: unknown key {dotted!r}.\n"
                "An override may only set keys that already exist in the base "
                "config. This is deliberate: a typo must fail here rather "
                "than silently leaving the base value in place.\n"
                f"Keys available at this level: {known}"
            )
        base_is_mapping = isinstance(base[key], dict)
        value_is_mapping = isinstance(value, dict)
        if base_is_mapping != value_is_mapping:
            raise ConfigError(
                f"{source}: {dotted!r} is "
                f"{'a section' if base_is_mapping else 'a value'} in the base "
                f"config but "
                f"{'a section' if value_is_mapping else 'a value'} in the "
                "override."
            )
        if value_is_mapping:
            merged[key] = _strict_merge(base[key], value, source, prefix=f"{dotted}.")
        else:
            merged[key] = value
    return merged


def _check_override_scope(override: dict, source: Path) -> None:
    """Enforce that a run override sets only `seed` and `arm`."""
    extra = sorted(set(override) - OVERRIDE_ALLOWED_KEYS)
    if extra:
        allowed = ", ".join(sorted(OVERRIDE_ALLOWED_KEYS))
        raise ConfigError(
            f"{source}: a run override may only set {allowed}, but this file "
            f"also sets: {', '.join(extra)}.\n"
            "The study's claim is that all runs are identical except for "
            "seed and arm. Changing anything else per-run would break that "
            "claim silently. If you genuinely need another per-run knob, add "
            "it to OVERRIDE_ALLOWED_KEYS in burst/config.py on purpose."
        )


# ---------------------------------------------------------------------------
# The frozen config object
#
# Frozen dataclasses give attribute access (cfg.training.batch_size) and block
# attribute assignment, which is requirement 7. Note that freezing a dataclass
# does not freeze containers it holds, so `arms` is stored as a tuple rather
# than a list.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    n_layer: int
    n_head: int
    n_embd: int
    vocab_size: int
    block_size: int
    expected_param_count: int
    tie_embeddings: bool | None


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    seq_len: int
    total_steps: int
    #: Sequences per forward pass, or None while undecided.
    #:
    #: Deliberately has no default, for the same reason grad_clip has none:
    #: guessing it would put a study-defining value in the code rather than in
    #: the record. batch_size / micro_batch gradient accumulation steps make up
    #: one optimizer step, and BECAUSE FLOATING-POINT ADDITION IS NOT
    #: ASSOCIATIVE, two accumulation shapes over identical data give different
    #: bits. A default here would silently decide part of the experiment.
    micro_batch: int | None = None
    #: Compute dtype for the forward and backward pass, or None while undecided.
    #:
    #: No default, for the same reason micro_batch has none: it CHANGES WHICH
    #: CUDA KERNELS ARE SELECTED, and therefore reduction order. The
    #: determinism probe measured this directly -- fp32 chose
    #: fmha_cutlassF/B, bf16 chose pytorch_flash::flash_fwd/bwd, 66 kernels
    #: against 74. A run that inherited a dtype rather than declaring one would
    #: be reproducible only by accident.
    dtype: str | None = None

    @property
    def accumulation_steps(self) -> int:
        """How many micro-batches make one optimizer step.

        Raises rather than guessing when micro_batch is undecided, so a caller
        cannot accidentally get 1 and train on a batch 32x smaller than the one
        the config describes.
        """
        if self.micro_batch is None:
            raise ConfigError(
                "training.micro_batch is null, so the number of gradient "
                "accumulation steps is undefined. It has no default: see the "
                "comment in configs/base.yaml for why guessing it would change "
                "the experiment rather than merely its speed.")
        return self.batch_size // self.micro_batch


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    weight_decay: float
    #: AdamW momentum coefficients. Fixed across all arms.
    beta1: float
    beta2: float
    #: Denominator stabiliser. Fixed across all arms.
    eps: float
    #: Gradient-norm clipping threshold, or None while undecided.
    #:
    #: Deliberately has no default. Clipping caps the largest updates, and the
    #: burst IS an oversized update -- so clipping would cap the arms
    #: unequally and undo the matching the study depends on. `None` is
    #: rejected at launch, exactly like the injection fields, so no run can
    #: start without someone having decided.
    grad_clip: float | None
    #: Which AdamW implementation to use, or None while undecided.
    #:
    #: No default, and this one has already caused a divergence: the training
    #: loop hardcoded `single` while the committed determinism result was
    #: produced under `foreach`. The three implementations group their
    #: arithmetic differently -- per tensor, per fused group, or in one kernel
    #: -- so THEY PRODUCE DIFFERENT BITS from identical moments. That made the
    #: only determinism evidence this study has inapplicable to the loop meant
    #: to inherit it, silently, because nothing declared the value.
    adamw_impl: str | None = None


@dataclass(frozen=True)
class LearningRateConfig:
    schedule: str
    peak: float
    warmup_steps: int
    final: float


@dataclass(frozen=True)
class CorpusConfig:
    name: str
    slice_description: str
    expected_token_budget: int


@dataclass(frozen=True)
class DeterminismConfig:
    deterministic: bool


@dataclass(frozen=True)
class ExperimentConfig:
    n_seeds: int
    arms: tuple[str, ...]


@dataclass(frozen=True)
class BurstTextPaths:
    """Repo-relative path to the burst text for each injecting arm.

    There is deliberately no `twin` field: twin receives no injection, so a
    burst text for it would be a meaningless value that base.yaml could carry
    around looking meaningful. The schema check rejects one if it appears.
    """

    #: One entry per injecting arm, keyed by the arm name exactly as it
    #: appears in ARMS. A mapping rather than named fields because every v4 arm
    #: name contains a hyphen and so cannot be a Python identifier -- and
    #: because keying by the arm name means this structure cannot drift out of
    #: step with ARMS the way three hardcoded fields did.
    paths: dict

    def for_arm(self, arm: str) -> str | None:
        """The burst text path this arm will use, or None if it has none.

        `twin` is handled explicitly rather than by a missing-key lookup, so
        the control arm having no burst text is a visible decision rather than
        a KeyError that happens to be caught somewhere.
        """
        if arm == "twin":
            return None
        if arm not in self.paths:
            raise ConfigError(f"no burst text path defined for arm {arm!r}")
        return self.paths[arm]


@dataclass(frozen=True)
class InjectionConfig:
    injection_step: int | None
    burst_length_tokens: int | None
    burst_text_paths: BurstTextPaths
    #: Token offset where the burst starts inside the sequence, or None while
    #: undecided.
    #:
    #: No default, for the same reason micro_batch and dtype have none: it is a
    #: study-defining quantity, not a knob. Every arm-matching number in step 8
    #: was measured at one position, so injecting at another would mean the
    #: arms were matched somewhere the study does not use -- and nothing would
    #: crash. It lived as a duplicated constant in three scripts before this.
    burst_position: int | None = None


@dataclass(frozen=True)
class CheckpointingConfig:
    #: Steps between weights-only checkpoints (weights + step number, ~0.5 GB).
    weights_only_interval: int | None
    #: Steps between full checkpoints (weights + optimizer + step + RNG state,
    #: ~1.5 GB). Must be an exact multiple of weights_only_interval.
    full_interval: int | None


@dataclass(frozen=True)
class CheckpointPlan:
    """What the checkpoint schedule works out to, derived from the config.

    Computed the same way as the token-budget assertion: from the numbers in
    the config, never hardcoded, so it stays correct if total_steps changes.
    """

    #: 0-indexed index of the final optimizer step (total_steps - 1).
    last_step: int
    #: Checkpoints actually written as weights-only, after precedence.
    weights_only_count: int
    #: Checkpoints written as full, including the mandatory final-step one.
    full_count: int
    #: Estimated bytes on disk for one run.
    estimated_bytes_per_run: int
    #: Estimated bytes for the whole study (n_seeds x number of arms).
    estimated_bytes_all_runs: int


@dataclass(frozen=True)
class Config:
    seed: int
    arm: str
    model: ModelConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    learning_rate: LearningRateConfig
    corpus: CorpusConfig
    determinism: DeterminismConfig
    experiment: ExperimentConfig
    injection: InjectionConfig
    checkpointing: CheckpointingConfig

    @property
    def run_name(self) -> str:
        """The canonical run name, e.g. 'seed03_coherent'."""
        return run_name_for(self.seed, self.arm)

    @property
    def missing_for_launch(self) -> tuple[str, ...]:
        """Fields still null that this run would need before it could train."""
        return tuple(_missing_for_launch(self))

    @property
    def last_step(self) -> int:
        """Index of the final optimizer step.

        Steps are 0-indexed, so this is total_steps - 1. Everything that needs
        "the last step" goes through here rather than repeating the arithmetic.
        """
        return self.training.total_steps - 1

    def checkpoint_kind_at(self, step: int) -> str | None:
        """Which checkpoint, if any, is due at this 0-indexed step.

        Returns CHECKPOINT_FULL, CHECKPOINT_WEIGHTS_ONLY, or None.

        This is the single definition of the schedule. It writes nothing --
        it is a pure function of the config -- but the training loop must call
        it (or reproduce it exactly) so that the precedence and final-step
        rules cannot be implemented two different ways.

        A checkpoint is due after every N *completed* steps, so it fires at
        step s when (s + 1) % N == 0. With weights_only_interval 50, the first
        one lands at step 49, not step 0.
        """
        interval_wo, interval_full = self._require_intervals()
        if not 0 <= step <= self.last_step:
            raise ConfigError(
                f"step {step} is outside this run: steps are 0-indexed and run "
                f"0..{self.last_step} (total_steps = "
                f"{self.training.total_steps})."
            )
        # Final-step rule first: the finished model is what every headline
        # comparison is computed on, so it is always a full checkpoint no
        # matter what the intervals say.
        if step == self.last_step:
            return CHECKPOINT_FULL
        completed = step + 1
        # Precedence: full wins. A weights-only file here would duplicate data
        # already inside the full checkpoint at the same step.
        if completed % interval_full == 0:
            return CHECKPOINT_FULL
        if completed % interval_wo == 0:
            return CHECKPOINT_WEIGHTS_ONLY
        return None

    @property
    def checkpoint_plan(self) -> CheckpointPlan:
        """Counts and storage estimate implied by the two intervals.

        Computed by counting, not by simulation: the study is 9536 steps now
        but the arithmetic must stay cheap if that grows.
        """
        interval_wo, interval_full = self._require_intervals()
        total = self.training.total_steps

        # Firings over steps 0..last, using (s + 1) % N == 0, are exactly the
        # multiples of N in 1..total_steps.
        wo_firings = total // interval_wo
        full_firings = total // interval_full

        # Because full_interval is validated to be a multiple of
        # weights_only_interval, every full firing is also a weights-only
        # firing -- so full_firings is exactly the overlap to subtract.
        last_is_full_firing = total % interval_full == 0
        last_is_wo_firing = total % interval_wo == 0

        full_count = full_firings + (0 if last_is_full_firing else 1)
        weights_only_count = wo_firings - full_firings
        if last_is_wo_firing and not last_is_full_firing:
            # The last step was going to be weights-only; the final-step rule
            # promotes it to full, so it must not be counted twice.
            weights_only_count -= 1

        per_run = (
            weights_only_count * WEIGHTS_ONLY_CHECKPOINT_BYTES
            + full_count * FULL_CHECKPOINT_BYTES
        )
        n_runs = self.experiment.n_seeds * len(self.experiment.arms)
        return CheckpointPlan(
            last_step=self.last_step,
            weights_only_count=weights_only_count,
            full_count=full_count,
            estimated_bytes_per_run=per_run,
            estimated_bytes_all_runs=per_run * n_runs,
        )

    def _require_intervals(self) -> tuple[int, int]:
        interval_wo = self.checkpointing.weights_only_interval
        interval_full = self.checkpointing.full_interval
        if interval_wo is None or interval_full is None:
            raise ConfigError(
                "the checkpoint schedule is not decided yet: "
                "checkpointing.weights_only_interval and "
                "checkpointing.full_interval must both be set before the "
                "schedule or its storage estimate can be computed."
            )
        return interval_wo, interval_full


#: Section name -> dataclass. Used to derive the set of keys the loader knows
#: about, so the schema and the dataclasses cannot drift apart.
_SECTION_CLASSES: dict[str, type] = {
    "model": ModelConfig,
    "training": TrainingConfig,
    "optimizer": OptimizerConfig,
    "learning_rate": LearningRateConfig,
    "corpus": CorpusConfig,
    "determinism": DeterminismConfig,
    "experiment": ExperimentConfig,
    "injection": InjectionConfig,
    "checkpointing": CheckpointingConfig,
}

#: Nested mappings inside a section whose keys are the ARM NAMES rather than a
#: fixed dataclass schema, keyed by dotted path. `burst_text_paths` is one
#: entry per injecting arm, so its expected keys come from INJECTING_ARMS --
#: which means cutting or adding an arm updates this automatically instead of
#: leaving a schema that names arms the study no longer has.
_ARM_KEYED_SUBSECTIONS: tuple[str, ...] = ("injection.burst_text_paths",)

_TOP_LEVEL_SCALARS: frozenset[str] = frozenset({"seed", "arm"})


def run_name_for(seed: int, arm: str) -> str:
    """seed{NN}_{arm}, zero-padded to two digits."""
    return f"seed{seed:02d}_{arm}"


# ---------------------------------------------------------------------------
# Shape and type validation
# ---------------------------------------------------------------------------


def _check_shape(merged: dict, source: Path) -> None:
    """The merged config must have exactly the sections and keys we expect.

    Catches both a missing key (base.yaml was edited down) and an extra key
    (someone added a setting to base.yaml that the loader silently ignores,
    which would make resolved_config.yaml describe a setting nothing reads).
    """
    expected_top = _TOP_LEVEL_SCALARS | set(_SECTION_CLASSES)
    _compare_keys(set(merged), expected_top, "top level", source)

    for section, cls in _SECTION_CLASSES.items():
        value = merged[section]
        if not isinstance(value, dict):
            raise ConfigError(
                f"{source}: {section!r} must be a section (a YAML mapping), "
                f"got {type(value).__name__}"
            )
        expected = {f.name for f in dataclasses.fields(cls)}
        _compare_keys(set(value), expected, f"section {section!r}", source)

    for dotted in _ARM_KEYED_SUBSECTIONS:
        section, key = dotted.split(".")
        value = merged[section][key]
        if not isinstance(value, dict):
            raise ConfigError(
                f"{source}: {dotted!r} must be a mapping, got "
                f"{type(value).__name__}"
            )
        _compare_keys(set(value), set(INJECTING_ARMS),
                      f"section {dotted!r}", source)


def _compare_keys(
    actual: set, expected: set[str], where: str, source: Path
) -> None:
    missing = sorted(expected - actual)
    extra = sorted(str(k) for k in actual - expected)
    problems = []
    if missing:
        problems.append(f"missing: {', '.join(missing)}")
    if extra:
        problems.append(f"unexpected: {', '.join(extra)}")
    if problems:
        raise ConfigError(
            f"{source}: {where} does not match the schema in burst/config.py "
            f"({'; '.join(problems)})."
        )


def _type_error_hint(value: Any) -> str:
    """Extra help for the two PyYAML traps that bite hardest."""
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+(\.\d*)?[eE][+-]?\d+", value):
        return (
            "\nThis looks like scientific notation. PyYAML's float rule "
            "requires a decimal point, so `6e-4` parses as the STRING '6e-4'. "
            "Write it in plain decimal form instead: 0.0006"
        )
    if isinstance(value, bool):
        return (
            "\nPyYAML turns bare yes/no/on/off/true/false into booleans. If "
            "you meant the word, quote it: \"no\""
        )
    return ""


def _require(
    merged: dict, section: str | tuple[str, ...] | None, key: str,
    kind: type | tuple[type, ...], kind_name: str, source: str | Path, *,
    allow_null: bool = False,
) -> Any:
    """Fetch one field and check its Python type after parsing.

    Every numeric field goes through here, which is requirement 3: a value
    that arrives as a string fails immediately with a message naming the field.

    `section` is None for a top-level key, a string for a section, or a tuple
    of names for something nested deeper.
    """
    if section is None:
        parts: tuple[str, ...] = ()
    elif isinstance(section, str):
        parts = (section,)
    else:
        parts = tuple(section)

    container = merged
    for part in parts:
        container = container[part]
    dotted = ".".join((*parts, key))
    value = container[key]

    if value is None:
        if allow_null:
            return None
        raise ConfigError(
            f"{source}: {dotted} is null but is required and has no default. "
            "Set it in the base config (or, for seed and arm, in the run "
            "override)."
        )

    # bool is a subclass of int in Python, so `isinstance(True, int)` is True.
    # Without this guard `total_steps: yes` would sail through as the int 1.
    if kind is not bool and isinstance(value, bool):
        raise ConfigError(
            f"{source}: {dotted} must be {kind_name}, got bool ({value!r})."
            + _type_error_hint(value)
        )

    if not isinstance(value, kind):
        raise ConfigError(
            f"{source}: {dotted} must be {kind_name}, got "
            f"{type(value).__name__} ({value!r})." + _type_error_hint(value)
        )
    return value


def _int(merged, section, key, source, *, allow_null=False):
    return _require(merged, section, key, int, "an integer", source,
                    allow_null=allow_null)


def _float(merged, section, key, source, *, allow_null=False):
    # int is accepted where a float is expected: widening is lossless, and
    # `weight_decay: 0` is not a mistake worth failing over. The reverse is
    # not accepted -- `total_steps: 0.5` is always a mistake.
    value = _require(merged, section, key, (int, float), "a number", source,
                     allow_null=allow_null)
    return None if value is None else float(value)


def _str(merged, section, key, source, *, allow_null=False):
    return _require(merged, section, key, str, "a string", source,
                    allow_null=allow_null)


def _bool(merged, section, key, source, *, allow_null=False):
    return _require(merged, section, key, bool, "a boolean (true/false)",
                    source, allow_null=allow_null)


_BURST_TEXTS: tuple[str, ...] = ("injection", "burst_text_paths")


def _build_config(merged: dict, source: str | Path) -> Config:
    """Turn the validated dict into the frozen object, type-checking as we go."""
    arms_raw = merged["experiment"]["arms"]
    if not isinstance(arms_raw, list) or not all(isinstance(a, str) for a in arms_raw):
        raise ConfigError(
            f"{source}: experiment.arms must be a list of strings, got "
            f"{arms_raw!r}"
        )

    return Config(
        seed=_int(merged, None, "seed", source),
        arm=_str(merged, None, "arm", source),
        model=ModelConfig(
            n_layer=_int(merged, "model", "n_layer", source),
            n_head=_int(merged, "model", "n_head", source),
            n_embd=_int(merged, "model", "n_embd", source),
            vocab_size=_int(merged, "model", "vocab_size", source),
            block_size=_int(merged, "model", "block_size", source),
            expected_param_count=_int(merged, "model", "expected_param_count", source),
            tie_embeddings=_bool(merged, "model", "tie_embeddings", source,
                                 allow_null=True),
        ),
        training=TrainingConfig(
            batch_size=_int(merged, "training", "batch_size", source),
            seq_len=_int(merged, "training", "seq_len", source),
            total_steps=_int(merged, "training", "total_steps", source),
            micro_batch=_int(merged, "training", "micro_batch", source,
                             allow_null=True),
            dtype=_str(merged, "training", "dtype", source, allow_null=True),
        ),
        optimizer=OptimizerConfig(
            name=_str(merged, "optimizer", "name", source),
            weight_decay=_float(merged, "optimizer", "weight_decay", source),
            beta1=_float(merged, "optimizer", "beta1", source),
            beta2=_float(merged, "optimizer", "beta2", source),
            eps=_float(merged, "optimizer", "eps", source),
            adamw_impl=_str(merged, "optimizer", "adamw_impl", source,
                            allow_null=True),
            grad_clip=_float(merged, "optimizer", "grad_clip", source,
                             allow_null=True),
        ),
        learning_rate=LearningRateConfig(
            schedule=_str(merged, "learning_rate", "schedule", source),
            peak=_float(merged, "learning_rate", "peak", source),
            warmup_steps=_int(merged, "learning_rate", "warmup_steps", source),
            final=_float(merged, "learning_rate", "final", source),
        ),
        corpus=CorpusConfig(
            name=_str(merged, "corpus", "name", source),
            slice_description=_str(merged, "corpus", "slice_description", source),
            expected_token_budget=_int(merged, "corpus", "expected_token_budget", source),
        ),
        determinism=DeterminismConfig(
            deterministic=_bool(merged, "determinism", "deterministic", source),
        ),
        experiment=ExperimentConfig(
            n_seeds=_int(merged, "experiment", "n_seeds", source),
            # tuple, not list: a frozen dataclass still hands out a mutable
            # list if you put one in it.
            arms=tuple(arms_raw),
        ),
        injection=InjectionConfig(
            injection_step=_int(merged, "injection", "injection_step", source,
                                allow_null=True),
            burst_length_tokens=_int(merged, "injection", "burst_length_tokens",
                                     source, allow_null=True),
            # Built from ARMS rather than from a hardcoded list, so adding or
            # cutting an arm cannot leave this behind.
            burst_position=_int(merged, "injection", "burst_position", source,
                                allow_null=True),
            burst_text_paths=BurstTextPaths(paths={
                arm: _str(merged, _BURST_TEXTS, arm, source, allow_null=True)
                for arm in INJECTING_ARMS
            }),
        ),
        checkpointing=CheckpointingConfig(
            weights_only_interval=_int(merged, "checkpointing",
                                       "weights_only_interval", source,
                                       allow_null=True),
            full_interval=_int(merged, "checkpointing", "full_interval",
                               source, allow_null=True),
        ),
    )


# ---------------------------------------------------------------------------
# Semantic validation
# ---------------------------------------------------------------------------


def _validate_semantics(cfg: Config, source: str | Path) -> None:
    """Checks that need more than one field to make sense."""

    # Requirement 5: the arm must be exactly one of the seven literal names.
    # Compared without any normalisation, so "Fluent-False" and "FLUENT-FALSE"
    # both fail.
    if cfg.arm not in ARMS:
        hint = ""
        if cfg.arm.lower() in ARMS:
            hint = (
                f"\n(Case matters. Did you mean {cfg.arm.lower()!r}?)"
            )
        raise ConfigError(
            f"{source}: arm must be exactly one of {', '.join(ARMS)}; "
            f"got {cfg.arm!r}.{hint}"
        )

    if tuple(cfg.experiment.arms) != ARMS:
        raise ConfigError(
            f"{source}: experiment.arms is {list(cfg.experiment.arms)} but "
            f"burst/config.py defines the arms as {list(ARMS)}. These must "
            "agree; change both or neither."
        )

    if not 0 <= cfg.seed < cfg.experiment.n_seeds:
        raise ConfigError(
            f"{source}: seed must be in 0..{cfg.experiment.n_seeds - 1} "
            f"(experiment.n_seeds = {cfg.experiment.n_seeds}); got {cfg.seed}."
        )

    # Requirement 4: the token budget must be the product of the three numbers
    # that produce it. If someone edits total_steps and forgets the budget,
    # this fails at load rather than every run later.
    computed = cfg.training.batch_size * cfg.training.seq_len * cfg.training.total_steps
    if computed != cfg.corpus.expected_token_budget:
        raise ConfigError(
            "token budget mismatch: "
            f"batch_size ({cfg.training.batch_size}) * "
            f"seq_len ({cfg.training.seq_len}) * "
            f"total_steps ({cfg.training.total_steps}) = {computed}, "
            f"but corpus.expected_token_budget = "
            f"{cfg.corpus.expected_token_budget} "
            f"(difference {computed - cfg.corpus.expected_token_budget}).\n"
            "One of these four numbers was edited without the others. "
            f"Source: {source}"
        )

    # Requirement 4b: accumulation must divide the batch exactly.
    #
    # A remainder would mean the last micro-batch of every step is a different
    # size from the others, so the step would train on fewer sequences than
    # batch_size claims -- and the token-budget identity above, which is what
    # the whole corpus was built against, would quietly stop describing what
    # was trained on. Checked here rather than in the training loop so that a
    # bad pairing fails at load, before a GPU is allocated.
    micro = cfg.training.micro_batch
    if micro is not None:
        if micro <= 0:
            raise ConfigError(
                f"training.micro_batch must be positive; got {micro}. "
                f"Source: {source}")
        if micro > cfg.training.batch_size:
            raise ConfigError(
                f"training.micro_batch ({micro}) exceeds batch_size "
                f"({cfg.training.batch_size}); a micro-batch is a slice of the "
                f"batch, not a multiple of it. Source: {source}")
        if cfg.training.batch_size % micro != 0:
            raise ConfigError(
                f"training.batch_size ({cfg.training.batch_size}) is not "
                f"divisible by training.micro_batch ({micro}); "
                f"{cfg.training.batch_size % micro} sequences would be left "
                "over every step, so the step would not train on the batch the "
                "config describes.\n"
                f"Source: {source}")

    # Requirement 4c: the two closed-set reduction-order fields.
    #
    # Validated by hand against a tuple because the loader has no enum helper,
    # which is exactly how `arm` is checked against ARMS. Both change which
    # kernels run and therefore what bits come out, so a typo that fell through
    # to a default would be a different experiment wearing the same config.
    if cfg.training.dtype is not None and cfg.training.dtype not in DTYPES:
        raise ConfigError(
            f"training.dtype must be exactly one of {', '.join(DTYPES)}; got "
            f"{cfg.training.dtype!r}. Source: {source}")
    if (cfg.optimizer.adamw_impl is not None
            and cfg.optimizer.adamw_impl not in ADAMW_IMPLS):
        raise ConfigError(
            f"optimizer.adamw_impl must be exactly one of "
            f"{', '.join(ADAMW_IMPLS)}; got {cfg.optimizer.adamw_impl!r}. "
            f"Source: {source}")

    # Cheap arithmetic sanity checks in the same spirit as the one above.
    if cfg.training.seq_len > cfg.model.block_size:
        raise ConfigError(
            f"{source}: training.seq_len ({cfg.training.seq_len}) exceeds "
            f"model.block_size ({cfg.model.block_size}); the model cannot "
            "attend over a sequence longer than its context window."
        )
    if cfg.model.n_embd % cfg.model.n_head != 0:
        raise ConfigError(
            f"{source}: model.n_embd ({cfg.model.n_embd}) is not divisible by "
            f"model.n_head ({cfg.model.n_head}); the head dimension would not "
            "be an integer."
        )
    if cfg.learning_rate.warmup_steps >= cfg.training.total_steps:
        raise ConfigError(
            f"{source}: learning_rate.warmup_steps "
            f"({cfg.learning_rate.warmup_steps}) must be less than "
            f"training.total_steps ({cfg.training.total_steps})."
        )
    # AdamW coefficients. Both are exponential decay rates, so they are only
    # meaningful strictly inside (0, 1): at 0 the moment has no memory, at 1
    # it never updates.
    for name, value in (("beta1", cfg.optimizer.beta1),
                        ("beta2", cfg.optimizer.beta2)):
        if not 0.0 < value < 1.0:
            raise ConfigError(
                f"{source}: optimizer.{name} ({value}) must be strictly "
                "between 0 and 1. It is an exponential decay rate: 0 gives "
                "the moment no memory at all, and 1 stops it ever updating."
            )
    if cfg.optimizer.eps <= 0:
        raise ConfigError(
            f"{source}: optimizer.eps ({cfg.optimizer.eps}) must be positive; "
            "it exists to keep the update's denominator away from zero."
        )
    if cfg.optimizer.grad_clip is not None and cfg.optimizer.grad_clip <= 0:
        raise ConfigError(
            f"{source}: optimizer.grad_clip ({cfg.optimizer.grad_clip}) must "
            "be positive when set. To decide against clipping, that decision "
            "still has to be written down -- see the note in configs/base.yaml."
        )

    if cfg.learning_rate.final > cfg.learning_rate.peak:
        raise ConfigError(
            f"{source}: learning_rate.final ({cfg.learning_rate.final}) is "
            f"greater than learning_rate.peak ({cfg.learning_rate.peak}); a "
            "cosine decay must decay."
        )

    # Only checkable once injection_step has been decided.
    step = cfg.injection.injection_step
    if step is not None and not 0 <= step < cfg.training.total_steps:
        raise ConfigError(
            f"{source}: injection.injection_step ({step}) must be within "
            f"0..{cfg.training.total_steps - 1}."
        )
    burst = cfg.injection.burst_length_tokens
    if burst is not None and burst <= 0:
        raise ConfigError(
            f"{source}: injection.burst_length_tokens ({burst}) must be "
            "positive."
        )

    # The burst has to FIT, with a token before it to be predicted from.
    #
    # Position 0 is rejected for the reason burst_match.assemble_sequence
    # rejects it: the token at index 0 has nothing preceding it, so the burst
    # region would carry burst_length - 1 predictions and the burst-region loss
    # would quietly be an average over a different set of tokens than the one
    # step 8 measured. Checked here as well as there so a bad pairing fails at
    # load rather than at step 200 of a multi-day run.
    position = cfg.injection.burst_position
    if position is not None:
        if burst is None:
            raise ConfigError(
                f"{source}: injection.burst_position is set ({position}) but "
                "injection.burst_length_tokens is null, so there is no way to "
                "tell whether the burst fits.")
        highest = cfg.training.seq_len - burst
        if not 1 <= position <= highest:
            raise ConfigError(
                f"{source}: injection.burst_position ({position}) must be "
                f"between 1 and {highest} inclusive, so that a "
                f"{burst}-token burst fits inside a "
                f"{cfg.training.seq_len}-token sequence with a token before it "
                "to be predicted from. Position 0 would give the burst region "
                f"{burst - 1} predictions instead of {burst}."
            )

    _validate_checkpoint_intervals(cfg, source)
    _validate_burst_text_paths(cfg, source)

    # Note deliberately NOT checked: that the twin arm has injection_step
    # unset. injection_step lives in the shared base config, so once it is
    # decided every arm sees it, including twin. Twin simply ignores it.


def _validate_checkpoint_intervals(cfg: Config, source: str | Path) -> None:
    """Both intervals positive, and full an exact multiple of weights-only.

    Each is checked independently of the other being set, so a half-decided
    config still gets the complaint that applies to it.
    """
    interval_wo = cfg.checkpointing.weights_only_interval
    interval_full = cfg.checkpointing.full_interval

    for name, value in (
        ("weights_only_interval", interval_wo),
        ("full_interval", interval_full),
    ):
        if value is not None and value <= 0:
            raise ConfigError(
                f"{source}: checkpointing.{name} must be a positive number of "
                f"steps; got {value}. An interval of 0 or less does not "
                "describe a schedule."
            )

    if interval_wo is None or interval_full is None:
        return

    # The precedence rule ("when both fire on the same step, write the full
    # one only") is only well defined if the two schedules actually line up.
    # With, say, 50 and 1500 they would coincide; with 50 and 1010 a full
    # checkpoint would land on steps no weights-only checkpoint ever touches,
    # and "both fire on the same step" would become an accident of arithmetic
    # rather than a rule.
    if interval_full % interval_wo != 0:
        raise ConfigError(
            f"{source}: checkpointing.full_interval ({interval_full}) must be "
            f"an exact multiple of checkpointing.weights_only_interval "
            f"({interval_wo}); {interval_full} / {interval_wo} = "
            f"{interval_full / interval_wo:.4g}.\n"
            "The two schedules have to align, otherwise the precedence rule "
            "(full checkpoint wins when both fire on the same step) is not "
            "well defined.\n"
            f"Nearest valid values: "
            f"{(interval_full // interval_wo) * interval_wo} or "
            f"{(interval_full // interval_wo + 1) * interval_wo}."
        )


_WHY_BURST_TEXT_MUST_BE_IN_REPO = (
    "The burst text is experimental content, not configuration -- it is the "
    "independent variable of the study. It has to be committed alongside the "
    "code so that the git commit hash recorded in run_provenance.yaml covers "
    "the text too. A path outside the repository would leave the injected "
    "text unversioned and would differ between your laptop and the cluster, "
    "which makes the run impossible to reproduce."
)


def _looks_absolute(value: str) -> bool:
    """True if `value` is absolute on *either* POSIX or Windows.

    Checked both ways on purpose. PureWindowsPath('/burst.txt').is_absolute()
    is False (no drive letter), so a Linux-style absolute path written by your
    collaborator would sail through a naive check run on Windows -- and vice
    versa. The config has to be rejected identically on both machines.
    """
    if value.startswith(("/", "\\")):
        return True
    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
        return True
    # "C:burst.txt" is drive-relative rather than absolute, but it is still
    # anchored to a particular machine's filesystem.
    return bool(PureWindowsPath(value).drive)


def _validate_burst_text_paths(cfg: Config, source: str | Path) -> None:
    """Burst text paths must be relative and must live inside this repo."""
    repo_root = _repo_root().resolve()
    for arm in INJECTING_ARMS:
        value = cfg.injection.burst_text_paths.for_arm(arm)
        if value is None:
            continue
        dotted = f"injection.burst_text_paths.{arm}"

        if not value.strip():
            raise ConfigError(f"{source}: {dotted} is empty.")

        if _looks_absolute(value):
            raise ConfigError(
                f"{source}: {dotted} is an absolute path ({value!r}). It must "
                "be written relative to the repository root, "
                f"e.g. 'configs/burst_texts/{arm}.txt'.\n"
                + _WHY_BURST_TEXT_MUST_BE_IN_REPO
            )

        resolved = (repo_root / value).resolve()
        if not resolved.is_relative_to(repo_root):
            raise ConfigError(
                f"{source}: {dotted} ({value!r}) resolves to {resolved}, "
                f"which is outside the repository at {repo_root}.\n"
                + _WHY_BURST_TEXT_MUST_BE_IN_REPO
            )


def _missing_for_launch(cfg: Config) -> list[str]:
    """Fields that are still null and that this arm needs in order to train."""
    missing = []
    if cfg.checkpointing.weights_only_interval is None:
        missing.append("checkpointing.weights_only_interval")
    if cfg.checkpointing.full_interval is None:
        missing.append("checkpointing.full_interval")
    if cfg.model.tie_embeddings is None:
        missing.append("model.tie_embeddings")
    # Every arm needs it: accumulation shape is part of what makes two runs the
    # same run, so no arm can launch without it having been decided.
    if cfg.training.micro_batch is None:
        missing.append("training.micro_batch")
    # Same class as micro_batch: both decide reduction order, so both are part
    # of what makes two runs the same run, for every arm.
    if cfg.training.dtype is None:
        missing.append("training.dtype")
    if cfg.optimizer.adamw_impl is None:
        missing.append("optimizer.adamw_impl")
    # Every arm, including twin. Clipping applies to the whole run, not just
    # to the step the burst lands on, so twin is no more exempt from the
    # decision than the injecting arms are.
    if cfg.optimizer.grad_clip is None:
        missing.append("optimizer.grad_clip")
    if cfg.arm in INJECTING_ARMS:
        if cfg.injection.injection_step is None:
            missing.append("injection.injection_step")
        if cfg.injection.burst_length_tokens is None:
            missing.append("injection.burst_length_tokens")
        if cfg.injection.burst_position is None:
            missing.append("injection.burst_position")
        # Only this arm's text matters. A coherent run does not care whether
        # the noise text has been written yet.
        if cfg.injection.burst_text_paths.for_arm(cfg.arm) is None:
            missing.append(f"injection.burst_text_paths.{cfg.arm}")
    return missing


def _check_burst_text_exists(cfg: Config) -> None:
    """At launch time the burst text file has to actually be there.

    Deliberately not checked when merely inspecting a config: the path may be
    decided before the text is written. Checked only under require_complete,
    because a run that reaches the injection step and finds no file has
    already wasted most of its compute.
    """
    value = cfg.injection.burst_text_paths.for_arm(cfg.arm)
    if value is None:
        return
    path = (_repo_root() / value).resolve()
    if not path.is_file():
        raise ConfigError(
            f"run {cfg.run_name!r} cannot be launched: "
            f"injection.burst_text_paths.{cfg.arm} points at {value!r}, but "
            f"no file exists at {path}."
        )


def _check_run_filename(run_path: Path, cfg: Config) -> None:
    """If the override file is named like a run, its name must match contents.

    Only checked when the filename actually looks like `seedNN_arm`, so an
    ad-hoc or temporary override file is not blocked. Catches the copy-paste
    error where seed05_noise.yaml contains `seed: 4`.
    """
    stem = run_path.stem
    if not RUN_NAME_PATTERN.match(stem):
        return
    if stem != cfg.run_name:
        raise ConfigError(
            f"{run_path}: filename says {stem!r} but the contents resolve to "
            f"{cfg.run_name!r} (seed={cfg.seed}, arm={cfg.arm!r}).\n"
            "Regenerate the override files with "
            "`python scripts/generate_overrides.py` rather than hand-editing."
        )


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _git_metadata(repo_root: Path) -> dict:
    """Commit hash, branch, and whether the working tree is dirty.

    Returns a dict that always has the same keys, so the provenance file has a
    stable shape whether or not git is available.
    """
    info: dict[str, Any] = {
        "repo_root": str(repo_root),
        "git_available": False,
        "commit": None,
        "branch": None,
        "dirty": None,
        "dirty_files": [],
        "error": None,
    }

    def _git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_root), *args],
                capture_output=True, text=True, timeout=30, check=False,
            )
        except FileNotFoundError:
            info["error"] = "git executable not found on PATH"
            return None
        except subprocess.TimeoutExpired:
            info["error"] = "git command timed out"
            return None
        if result.returncode != 0:
            info["error"] = (result.stderr or result.stdout).strip() or (
                f"git {' '.join(args)} exited {result.returncode}"
            )
            return None
        return result.stdout.strip()

    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return info

    info["git_available"] = True
    info["commit"] = commit
    info["branch"] = _git("rev-parse", "--abbrev-ref", "HEAD")

    # --porcelain lists staged, unstaged and untracked changes. Untracked
    # files count: an uncommitted module that the run imports is part of the
    # experimental condition just as much as a modified one.
    status = _git("status", "--porcelain")
    if status is None:
        info["dirty"] = None
    else:
        lines = [line for line in status.splitlines() if line.strip()]
        info["dirty"] = bool(lines)
        info["dirty_files"] = lines
    return info


def _warn_about_code_state(git: dict, stream) -> None:
    """Loud warning if we cannot prove what code produced this run."""
    bar = "=" * 74
    if not git["git_available"]:
        print(bar, file=stream)
        print("WARNING: could not read git metadata for this checkout.", file=stream)
        print(f"  reason: {git['error']}", file=stream)
        print("  This run's code version cannot be reconstructed later.",
              file=stream)
        print(bar, file=stream)
        return
    if git["dirty"]:
        print(bar, file=stream)
        print("WARNING: the working tree is DIRTY.", file=stream)
        print(f"  commit: {git['commit']}", file=stream)
        print(f"  {len(git['dirty_files'])} uncommitted change(s):", file=stream)
        for line in git["dirty_files"][:20]:
            print(f"    {line}", file=stream)
        if len(git["dirty_files"]) > 20:
            print(f"    ... and {len(git['dirty_files']) - 20} more", file=stream)
        print("  The code is part of the experimental condition. The commit",
              file=stream)
        print("  hash recorded above does NOT describe what actually ran.",
              file=stream)
        print("  Commit before launching anything you intend to publish.",
              file=stream)
        print(bar, file=stream)


def _dump_yaml(data: dict) -> str:
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=False,
                          allow_unicode=True)


def _write_provenance(
    merged: dict,
    cfg: Config,
    outdir: Path,
    base_path: Path,
    run_path: Path,
    *,
    require_complete: bool,
    force: bool,
    stream,
) -> None:
    """Write resolved_config.yaml and run_provenance.yaml into outdir."""
    outdir.mkdir(parents=True, exist_ok=True)
    resolved_path = outdir / RESOLVED_CONFIG_FILENAME
    resolved_text = _dump_yaml(merged)

    # Round-trip check. safe_dump writes 0.00006 as `6.0e-05`; PyYAML inserts
    # the `.0` precisely so the value reads back as a float and not as a
    # string. Verify that rather than trust it -- if this ever breaks, the
    # provenance record would be a different config from the one that ran.
    reloaded = yaml.load(resolved_text, Loader=_StrictSafeLoader)
    if reloaded != merged:
        raise ConfigError(
            "internal error: the resolved config did not survive a YAML "
            "round-trip, so it cannot be trusted as a provenance record.\n"
            f"wrote:  {merged}\nread back: {reloaded}"
        )

    if resolved_path.exists() and resolved_path.read_text(encoding="utf-8") != resolved_text:
        if not force:
            raise ConfigError(
                f"{resolved_path} already exists and describes a DIFFERENT "
                "config.\nOverwriting it would destroy the record of what "
                "produced whatever else is in this directory. Use a fresh "
                "--outdir, or pass --force if you are certain."
            )
        print(f"NOTE: overwriting a differing {RESOLVED_CONFIG_FILENAME} "
              "because --force was given.", file=stream)

    resolved_path.write_text(resolved_text, encoding="utf-8")

    git = _git_metadata(_repo_root())
    provenance = {
        "run_name": cfg.run_name,
        "seed": cfg.seed,
        "arm": cfg.arm,
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": {
            "base_config": str(base_path.resolve()),
            "run_override": str(run_path.resolve()),
        },
        "outdir": str(outdir.resolve()),
        "launch_ready": not cfg.missing_for_launch,
        "missing_for_launch": list(cfg.missing_for_launch),
        "checked_for_launch": require_complete,
        # Derived, not configured. Recorded so a later audit can see what
        # checkpoint schedule this run believed it was following without
        # having to recompute it from the intervals.
        "checkpoint_plan": (
            dataclasses.asdict(cfg.checkpoint_plan)
            if cfg.checkpointing.weights_only_interval is not None
            and cfg.checkpointing.full_interval is not None
            else None
        ),
        "git": git,
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "hostname": platform.node(),
            "pyyaml_version": yaml.__version__,
        },
        "command_line": list(sys.argv),
    }
    (outdir / PROVENANCE_FILENAME).write_text(_dump_yaml(provenance), encoding="utf-8")

    _warn_about_code_state(git, stream)


def _repo_root() -> Path:
    """The directory containing the `burst` package.

    Deliberately not the current working directory: the git state that matters
    is the state of the code being run, not of wherever the shell happens to
    be sitting.
    """
    return Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def load_config(
    base_path: str | Path,
    run_path: str | Path,
    outdir: str | Path,
    *,
    require_complete: bool = True,
    force: bool = False,
    write_provenance: bool = True,
    stream=None,
) -> Config:
    """Load base + override, validate, record provenance, return frozen config.

    Args:
        base_path: configs/base.yaml
        run_path: configs/runs/seedNN_arm.yaml
        outdir: where resolved_config.yaml and run_provenance.yaml go. Required,
            and never read from a config file -- see requirement 6.
        require_complete: if True (the default, and the right setting for any
            code that is about to actually train), raise when a field this arm
            needs is still null. The CLI sets this False unless --launch is
            given, so that you can inspect a config while injection_step and
            friends are still undecided.
        force: allow overwriting an existing, differing resolved_config.yaml.
        write_provenance: set False only in tests that do not care about files.
        stream: where warnings go; defaults to stderr.

    Returns:
        A frozen Config.

    Raises:
        ConfigError: for every failure. The message names the field.
    """
    stream = sys.stderr if stream is None else stream
    base_path = Path(base_path)
    run_path = Path(run_path)
    outdir = Path(outdir)

    base_raw = _read_yaml(base_path)
    run_raw = _read_yaml(run_path)

    # Removed keys first, so an out-of-date config gets an error naming its
    # replacement rather than a generic "unknown key" from the merge.
    _reject_removed_keys(base_raw, base_path)
    _reject_removed_keys(run_raw, run_path)

    # Requirement 6, checked on both files before anything else looks at them.
    _reject_output_path_keys(base_raw, base_path)
    _reject_output_path_keys(run_raw, run_path)

    # Unknown-key check first so a typo reports as a typo, then the narrower
    # "overrides may only set seed and arm" rule.
    merged = _strict_merge(base_raw, run_raw, run_path)
    _check_override_scope(run_raw, run_path)

    # Shape problems can only come from the base config (an override cannot
    # introduce keys), but a bad *value* could come from either file, so the
    # value-level errors name both.
    _check_shape(merged, base_path)
    source = f"{base_path} (+ override {run_path})"
    cfg = _build_config(merged, source)
    _validate_semantics(cfg, source)
    _check_run_filename(run_path, cfg)

    if require_complete:
        missing = cfg.missing_for_launch
        if missing:
            raise ConfigError(
                f"run {cfg.run_name!r} cannot be launched: the following "
                "fields are still null and this arm needs them:\n"
                + "".join(f"  - {name}\n" for name in missing)
                + "Decide them in configs/base.yaml first. "
                + (
                    "(The twin arm does not need the injection fields, but "
                    f"{cfg.arm!r} does.)"
                    if cfg.arm in INJECTING_ARMS
                    else ""
                )
            )
        _check_burst_text_exists(cfg)

    if write_provenance:
        # Written before anything downstream is allowed to happen, so a run
        # that crashes one step in still leaves behind an exact record of what
        # it was trying to do.
        _write_provenance(
            merged, cfg, outdir, base_path, run_path,
            require_complete=require_complete, force=force, stream=stream,
        )

    return cfg


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m burst.config",
        description=(
            "Load the burst study config, write a provenance record into "
            "--outdir, and print the resolved values."
        ),
    )
    parser.add_argument("--config", required=True, metavar="PATH",
                        help="base config, e.g. configs/base.yaml")
    parser.add_argument("--run", required=True, metavar="PATH",
                        help="run override, e.g. configs/runs/seed03_coherent.yaml")
    parser.add_argument(
        "--outdir", required=True, metavar="PATH",
        help=("output directory for this run. Required, and deliberately not a "
              "config value, so one config works on any machine."),
    )
    parser.add_argument(
        "--launch", action="store_true",
        help=("fail if any field this arm needs is still null. Use this in the "
              "launcher; without it the config can be inspected while "
              "injection_step and friends are undecided."),
    )
    parser.add_argument(
        "--force", action="store_true",
        help=f"overwrite an existing, differing {RESOLVED_CONFIG_FILENAME}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        cfg = load_config(
            args.config, args.run, args.outdir,
            require_complete=args.launch, force=args.force,
        )
    except ConfigError as exc:
        # No traceback: the message is the useful part, and a stack trace
        # through a config loader tells you nothing you want to know.
        print(f"\nCONFIG ERROR\n{exc}\n", file=sys.stderr)
        return 1

    outdir = Path(args.outdir)
    print(f"run_name: {cfg.run_name}")
    print(f"outdir:   {outdir.resolve()}")
    print(f"wrote:    {RESOLVED_CONFIG_FILENAME}, {PROVENANCE_FILENAME}")
    print()
    print("resolved config")
    print("-" * 74)
    print(_dump_yaml(dataclasses.asdict(cfg)).rstrip())
    print("-" * 74)

    if (cfg.checkpointing.weights_only_interval is not None
            and cfg.checkpointing.full_interval is not None):
        plan = cfg.checkpoint_plan
        gb = 1_000_000_000
        print()
        print("checkpoint plan (derived from the config, not configured)")
        print("-" * 74)
        print(f"  steps                0..{plan.last_step} "
              f"(0-indexed; total_steps = {cfg.training.total_steps})")
        print(f"  weights-only         {plan.weights_only_count:>6} x 0.5 GB "
              f"= {plan.weights_only_count * WEIGHTS_ONLY_CHECKPOINT_BYTES / gb:>8.1f} GB"
              f"   (every {cfg.checkpointing.weights_only_interval} steps)")
        print(f"  full                 {plan.full_count:>6} x 1.5 GB "
              f"= {plan.full_count * FULL_CHECKPOINT_BYTES / gb:>8.1f} GB"
              f"   (every {cfg.checkpointing.full_interval} steps, "
              f"+ final step {plan.last_step})")
        print(f"  per run              "
              f"{plan.estimated_bytes_per_run / gb:>28.1f} GB")
        print(f"  all "
              f"{cfg.experiment.n_seeds * len(cfg.experiment.arms)} runs         "
              f"{plan.estimated_bytes_all_runs / gb / 1000:>28.2f} TB")
        print("  (sizes are estimates for 124M fp32 params, not measurements)")
        print("-" * 74)

    missing = cfg.missing_for_launch
    if missing:
        print()
        print("NOT LAUNCH-READY. Still undecided for this arm:")
        for name in missing:
            print(f"  - {name}")
        print("Loading succeeded because --launch was not given. Pass --launch")
        print("to turn the list above into a hard failure.")
    else:
        print()
        print("Launch-ready: every field this arm needs has a value.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
