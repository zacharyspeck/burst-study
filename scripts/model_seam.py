#!/usr/bin/env python
"""The narrow interface between the training loop and whatever model it trains.

WHY THIS EXISTS

This module builds HF GPT-2 today. `_build_hf_gpt2` returns a real
`GPT2LMHeadModel`, `FAMILIES` is `("hf_gpt2", "probe_linear")`, and `--family`
is required with no default in both `train.py` and `launch.py`.

This paragraph used to say the swap "has not landed" and was "pending". What
has not landed is the retirement of `probes/determinism/model.py`, which is
still an `nn.Linear` implementation and is the OTHER family here -- the probe,
not the study's model. The loop is written against the smallest interface that
survives either, and this module is the whole of it. Which family a given
checkpoint came from is recorded in `run_provenance.yaml`; see S86.

THE INTERFACE, and it is six things

    build_model(cfg, family)      construct, and CHECK the parameter count
    compute_loss(model, x, y)     a scalar loss, whatever the forward signature
    model.parameters()            for the optimizer and for clipping
    model.state_dict/.load_...    for checkpointing
    model.train() / .eval()
    model.named_parameters()      for digests

Nothing else in the loop knows what a transformer is. The only real difference
between the two families is the forward signature -- the probe model is
`model(x, y) -> loss`, HF is `model(input_ids=..., labels=...) -> out.loss` --
and `compute_loss` is that difference, in one place.

WHERE THE ABSTRACTION DOES NOT HOLD, AND SAYING SO IS THE POINT

A narrow interface makes the LOOP portable. It does not make the RUNS
comparable, and four things do not survive the choice:

1. CHECKPOINTS ARE NOT INTERCHANGEABLE. Parameter names differ entirely. A
   checkpoint written under one family cannot be loaded by the other.

   THIS PARAGRAPH USED TO SAY `expected_param_count` "is the only thing that
   would notice a mix-up". IT WOULD NOTICE NOTHING. Both families build to
   exactly 124,439,808 parameters -- that is asserted, deliberately, by
   `tests/test_model_seam.py::test_both_families_hit_expected_param_count_exactly`,
   so the check this sentence relied on is the same check that proves it wrong.
   S55 again: a guard named for catching what it cannot catch.

   What notices a mix-up now is the `family` field in `run_provenance.yaml`,
   written by `burst/config.py` and required whenever a config is loaded for
   launch. `scripts/launch.py::conflicting_family` refuses to emit into a
   directory whose provenance names a different family.
2. THE SEED -> WEIGHTS MAP IS FAMILY-SPECIFIC. HF's init and the probe's
   `_init_weights` differ, so the same seed produces different initial weights.
   Twin comparisons stay valid -- both arms use one family -- but no checkpoint
   from one family is comparable to one from the other, ever.
3. DETERMINISM EVIDENCE DOES NOT TRANSFER. Conv1D reaches cuBLAS through
   `addmm` with different operand strides than `F.linear`, so kernel selection
   differs. See docs/layout-cost.md.
4. STEP 9's RULER READS Conv1D ONLY, so an nn.Linear run cannot be
   canonicalized without the adapter that document costs.

There is therefore NO DEFAULT FAMILY here. `build_model` takes the choice
explicitly and refuses without it, because a default would silently decide
which of those four consequences the study lives with.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: The model families this seam knows how to build. No default: see above.
FAMILY_HF_GPT2 = "hf_gpt2"
FAMILY_PROBE_LINEAR = "probe_linear"
FAMILIES = (FAMILY_HF_GPT2, FAMILY_PROBE_LINEAR)


class ModelSeamError(Exception):
    """Raised for every refusal here. Never an assert."""


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ModelSeamError(
            "PyTorch is not installed, so no model can be built.\n"
            "    pip install -e \".[dev,measure]\"\n"
            f"(underlying import error: {exc})") from exc
    return torch


def _build_hf_gpt2(cfg):
    """HF GPT2LMHeadModel, shaped from the config and nothing else."""
    torch = _torch()
    try:
        from transformers.models.gpt2.modeling_gpt2 import (
            GPT2Config, GPT2LMHeadModel)
    except ImportError as exc:
        raise ModelSeamError(
            f"transformers is required for family {FAMILY_HF_GPT2!r}: "
            f"{exc}") from exc

    if cfg.model.tie_embeddings is None:
        raise ModelSeamError(
            "model.tie_embeddings is null. It changes the parameter count by "
            "vocab_size * n_embd = 38,597,376, so building without it decided "
            "would produce a model that fails the count check for a reason "
            "nobody chose.")

    hf = GPT2Config(
        n_layer=cfg.model.n_layer,
        n_head=cfg.model.n_head,
        n_embd=cfg.model.n_embd,
        n_positions=cfg.model.block_size,
        vocab_size=cfg.model.vocab_size,
        # Dropout off everywhere. Dropout is a per-step RNG consumer, and this
        # study's central claim is that two same-seed runs agree bitwise --
        # every avoidable source of stochasticity is one less thing that has to
        # be captured in the checkpoint and restored exactly on resume.
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        tie_word_embeddings=bool(cfg.model.tie_embeddings),
    )
    model = GPT2LMHeadModel(hf)
    return model


def _build_probe_linear(cfg):
    """The handwritten nn.Linear GPT from probes/determinism/model.py."""
    _torch()
    probe_dir = REPO_ROOT / "probes" / "determinism"
    if str(probe_dir) not in sys.path:
        sys.path.insert(0, str(probe_dir))
    try:
        from model import GPT
    except ImportError as exc:
        raise ModelSeamError(
            f"could not import probes/determinism/model.py: {exc}") from exc
    return GPT(
        n_layer=cfg.model.n_layer,
        n_head=cfg.model.n_head,
        n_embd=cfg.model.n_embd,
        vocab_size=cfg.model.vocab_size,
        block_size=cfg.model.block_size,
        tie_embeddings=bool(cfg.model.tie_embeddings),
    )


_BUILDERS = {
    FAMILY_HF_GPT2: _build_hf_gpt2,
    FAMILY_PROBE_LINEAR: _build_probe_linear,
}


def parameter_count(model) -> int:
    """Trainable parameters, tied weights counted ONCE.

    `named_parameters()` deduplicates by identity, so a tied `lm_head.weight`
    is not counted a second time. `state_dict()` does not, and using it here
    would overstate GPT-2 Base by 38,597,376 -- which is exactly the number
    that distinguishes the tied count from the untied one, so the check would
    pass for the wrong model.
    """
    return sum(p.numel() for _, p in model.named_parameters())


def build_model(cfg, family: str):
    """Construct a model and CHECK IT IS THE MODEL THE CONFIG DESCRIBES.

    The count check is the only thing standing between the loop and training
    the wrong architecture for nine thousand steps. `expected_param_count` has
    been in the config since the beginning with nothing comparing against it --
    implementation-notes.md records that as an obligation this discharges.
    """
    if family not in FAMILIES:
        raise ModelSeamError(
            f"unknown model family {family!r}; expected one of {FAMILIES}.\n"
            "There is NO DEFAULT. The choice decides whether checkpoints are "
            "interchangeable, what the seed-to-weights map is, and whether the "
            "determinism evidence transfers -- see this module's docstring.")

    model = _BUILDERS[family](cfg)
    actual = parameter_count(model)
    expected = cfg.model.expected_param_count
    if expected is not None and actual != expected:
        raise ModelSeamError(
            f"PARAMETER COUNT MISMATCH for family {family!r}: built "
            f"{actual:,} parameters, config expects {expected:,} "
            f"(difference {actual - expected:+,}).\n"
            f"tie_embeddings is {cfg.model.tie_embeddings}; tying changes the "
            f"count by vocab_size * n_embd = "
            f"{cfg.model.vocab_size * cfg.model.n_embd:,}, so a difference of "
            "exactly that number means the tie is wrong rather than the "
            "shape.\n"
            "This is the check that catches training the wrong architecture "
            "before nine thousand steps go into it.")
    return model


def compute_loss(model, inputs, targets):
    """A scalar loss, whichever forward signature the family has.

    The ONE place the two families differ. The probe model returns a loss
    directly; HF returns logits. Both are next-token cross-entropy over the
    same shift, so the number means the same thing.

    **THE HF BRANCH MUST NOT PASS `labels=`. S97/S99.** Callers hand this
    function an ALREADY-OFFSET pair -- `train.py` calls `reader.shift(raw)`
    immediately before, and the probe's `SyntheticCorpus` does the same -- so
    `inputs[i]` is already aligned against the next token in `targets[i]`.
    `transformers` shifts `labels` internally. Passing them composes the two
    shifts, and the logit at position i, which is conditioned on t_0..t_i,
    gets scored against t_(i+2): a two-tokens-ahead objective.

    That is not hypothetical. It is what this line used to do, and all four
    runs of the 2026-08-05 pilot were trained that way -- 44.69 GPU-hours
    optimising the wrong objective, with `train_record.json` and every
    determinism digest correct about a model of the wrong thing. See
    `docs/2026-08-05-training-objective-defect.md`.

    Computing the cross-entropy here rather than delegating is also what makes
    the two families measurable against each other: identical loss code, so any
    difference between them is the model and not the objective. This mirrors
    `probes/determinism/hf_model.py`, which has carried the warning in its
    docstring -- "passing `labels` would shift twice" -- since before the
    pilot, and got it right.
    """
    if _is_hf(model):
        torch = _torch()
        logits = model(input_ids=inputs, use_cache=False).logits
        return torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.reshape(-1))

    out = model(inputs, targets)
    loss = getattr(out, "loss", out)
    if loss is None:
        raise ModelSeamError(
            "the model returned no loss; a forward that produces only logits "
            "cannot be used here without deciding the reduction, and that "
            "decision belongs with the model rather than in the loop")
    return loss


def _is_hf(model) -> bool:
    return hasattr(model, "transformer") and hasattr(model, "lm_head")


def describe(model, family: str) -> dict:
    """What was built, as data for a report to derive its claims from."""
    return {
        "family": family,
        "class": type(model).__name__,
        "parameters": parameter_count(model),
        "tensors": len(list(model.named_parameters())),
        "forward_signature": ("input_ids/labels" if _is_hf(model)
                              else "positional (idx, targets)"),
    }
