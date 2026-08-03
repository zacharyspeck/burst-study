#!/usr/bin/env python
"""Symmetry canonicalization for GPT-2. Step 9.

Two networks can compute exactly the same function and still look far apart in
weight space, because "which head is head 3" and "which neuron is neuron 500"
are bookkeeping, not content. This module is the ruler that removes the
bookkeeping, so that a weight-space distance between two models reflects a real
difference rather than a difference in numbering.

WHAT A SYMMETRY IS

A symmetry is a way of relabelling a network's internals that leaves its
outputs unchanged. Swap attention head 3 with head 7 -- in the query weights,
the key weights, the value weights, and the matching rows of the output
projection -- and the model emits byte-for-byte the same logits. Nothing was
learned or lost. But the weight vector moved a long way, and a naive distance
would report that movement as if it meant something.

Canonicalization is the rule that forces every model into one agreed labelling
so the movement disappears.

WHAT IS AND IS NOT A SYMMETRY DEPENDS ON THE ARCHITECTURE

A transformation is a symmetry only if applying it leaves the output unchanged
to within stated float tolerance. Whether a given transformation qualifies
depends on the nonlinearity and the normalization, and the answer for GPT-2 is
not the answer published for architectures using RMSNorm and ReLU. Three
specific traps, all of which this module tests rather than assumes:

1. LayerNorm subtracts the mean. Rotating the residual stream is free under
   RMSNorm; under LayerNorm only rotations fixing the all-ones direction are
   even candidates, and the per-channel gain blocks them further.
2. GELU is not positively homogeneous. Scaling an FFN's input weights up and
   its output weights down is a symmetry for ReLU and is NOT one for GELU.
3. Conv1D stores weights transposed relative to nn.Linear, and c_attn fuses Q,
   K and V into one tensor. Permuting the wrong axis, or permuting across the
   QKV boundary, produces a model that still runs and still emits believable
   numbers.

Nothing here is inherited from a paper. Every candidate is applied to a real
model and measured, in float32 AND float64.

THE COLLAPSE FACTOR IS THE SOLE PASS/FAIL CRITERION. A transformation that is
exact in exact arithmetic shows float32-epsilon error in float32 and many
orders less in float64; one that is not a symmetry shows a large error that
does not shrink at all when the precision goes up. Measured on real GPT-2 the
two groups separate by eight orders of magnitude with no overlap. Absolute
float32 error is reported as a diagnostic and gates nothing -- this
architecture's own float32 noise floor turned out to overlap the bound
originally registered for it, so that bound never discriminated anything. See
the criterion block below and S43 in implementation-notes.md.

WHAT SURVIVED

Seven of ten candidates are symmetries of GPT-2: LayerNorm gain rescale,
residual permutation, head permutation, head-internal GL transforms, FFN neuron
permutation, key-bias shift and value-bias shift. Three are not: residual
rotation (LayerNorm's per-channel gain, incurable at ln_f because absorbing it
would mean touching the tied lm_head), residual scaling (tying: it needs c in
and 1/c out, and a tied projection supplies c both times), and FFN scaling
(GELU is not positively homogeneous). D17-D19 record each drop with its
measured failure.

WHERE TORCH LIVES

Every torch import in this file is inside a function. Importing this module
must work in an environment with no ML stack, so that the torch-free test suite
skips rather than erroring at collection. Same rule, and the same reason, as
scripts/burst_match.py.

THE TRIPWIRE

Nothing in this module runs before validate_architecture() has confirmed the
model is the exact shape this code was written against. A canonicalizer aimed
at the wrong architecture does not crash -- it silently produces plausible
numbers forever, which is the failure this whole module exists to prevent.
"""

from __future__ import annotations

import copy
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

__all__ = [
    "ArchSpec",
    "CanonicalizeError",
    "GPT2_124M",
    "LogitDiff",
    "Symmetry",
    "ALL_SYMMETRIES",
    "validate_architecture",
    "logit_difference",
    "probe_tokens",
    "forward_logits",
]


class CanonicalizeError(Exception):
    """Every failure this module raises on purpose.

    One exception type, like burst.config's ConfigError and burst_match's
    BurstMatchError, so calling code and tests only ever catch one thing.
    """


# ---------------------------------------------------------------------------
# Optional dependency
# ---------------------------------------------------------------------------


def _torch():
    """Import torch, or raise CanonicalizeError explaining the install."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise CanonicalizeError(
            "PyTorch is not installed, and this module cannot rewrite weights "
            "without it.\n"
            "torch is an OPTIONAL dependency group on purpose: the config "
            "system has to keep working on a machine with no ML stack. "
            "Install into the separate environment:\n"
            "    pip install -e \".[dev,measure]\"\n"
            f"(underlying import error: {exc})"
        ) from exc
    return torch


# ---------------------------------------------------------------------------
# The architecture this code was written against
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArchSpec:
    """The exact shape validate_architecture() insists on.

    Sizes are parameterised so that tests can build a small model in process
    rather than committing a fixture or downloading 500 MB. The STRUCTURAL
    checks -- Conv1D layout, fused QKV, tied embeddings, learned-absolute
    positions -- are never parameterised and always run.
    """

    n_layer: int
    n_head: int
    n_embd: int
    n_inner: int
    n_positions: int
    vocab_size: int

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def n_params(self) -> int:
        """Parameter count implied by this shape, embeddings counted once."""
        per_block = (
            2 * self.n_embd                                  # ln_1
            + self.n_embd * 3 * self.n_embd + 3 * self.n_embd  # c_attn
            + self.n_embd * self.n_embd + self.n_embd          # attn.c_proj
            + 2 * self.n_embd                                # ln_2
            + self.n_embd * self.n_inner + self.n_inner        # c_fc
            + self.n_inner * self.n_embd + self.n_embd         # mlp.c_proj
        )
        return (
            self.vocab_size * self.n_embd        # wte (tied, counted once)
            + self.n_positions * self.n_embd     # wpe
            + self.n_layer * per_block
            + 2 * self.n_embd                    # ln_f
        )


#: Public GPT-2 124M, the only model that exists in this repo.
GPT2_124M = ArchSpec(
    n_layer=12, n_head=12, n_embd=768, n_inner=3072,
    n_positions=1024, vocab_size=50257,
)


# ---------------------------------------------------------------------------
# The tripwire
# ---------------------------------------------------------------------------


_WHY_ROTARY_MATTERS = (
    "This canonicalizer exploits the FULL GL(head_dim) freedom inside each "
    "attention head: the query and key weights only ever appear as the product "
    "W_Q W_K^T, so any invertible matrix can be pushed into one and its inverse "
    "into the other without changing a single attention score.\n"
    "That freedom exists BECAUSE GPT-2 uses learned absolute position "
    "embeddings, which leave the head's internal basis unconstrained. A model "
    "using rotary position embeddings (RoPE) applies a fixed, "
    "position-dependent rotation inside the head, which pins that basis and "
    "collapses the available group to a sharply smaller one. A canonicalizer "
    "built on the full GL action would be SILENTLY WRONG on such a model -- it "
    "would run, it would produce numbers, and the numbers would be meaningless."
)


def _fail(message: str) -> None:
    raise CanonicalizeError(message)


def _check(condition, message: str) -> None:
    if not condition:
        _fail(message)


def validate_architecture(model, expected: ArchSpec = GPT2_124M) -> ArchSpec:
    """Assert the model is exactly the architecture this module was written for.

    Raises CanonicalizeError naming the mismatch. Never warns and continues:
    a canonicalizer pointed at an architecture it does not understand produces
    plausible numbers rather than an error, and plausible-and-wrong is the
    failure mode this entire module is guarding against.

    Returns the validated ArchSpec so callers can use the sizes without
    re-deriving them.
    """
    torch = _torch()
    from torch import nn
    try:
        from transformers.pytorch_utils import Conv1D
    except ImportError as exc:
        raise CanonicalizeError(
            "transformers is not installed, so the Conv1D layout cannot be "
            "checked. Install with: pip install -e \".[dev,measure]\"\n"
            f"(underlying import error: {exc})"
        ) from exc

    where = type(model).__name__

    # ---- top-level shape -------------------------------------------------
    _check(hasattr(model, "transformer") and hasattr(model, "lm_head"),
           f"{where}: expected a GPT-2 LM head model with .transformer and "
           ".lm_head attributes. This module rewrites GPT-2 specifically.")
    tr = model.transformer
    cfg = model.config

    for name, want in (("n_layer", expected.n_layer), ("n_head", expected.n_head),
                       ("n_embd", expected.n_embd),
                       ("n_positions", expected.n_positions),
                       ("vocab_size", expected.vocab_size)):
        got = getattr(cfg, name, None)
        _check(got == want,
               f"{where}: config.{name} is {got!r}, expected {want!r}. This "
               "canonicalizer is shape-specific; pass a matching ArchSpec or "
               "do not canonicalize this model.")

    n_inner = cfg.n_inner if getattr(cfg, "n_inner", None) is not None else 4 * cfg.n_embd
    _check(n_inner == expected.n_inner,
           f"{where}: FFN inner width is {n_inner}, expected "
           f"{expected.n_inner}.")

    _check(cfg.n_embd % cfg.n_head == 0,
           f"{where}: n_embd ({cfg.n_embd}) is not divisible by n_head "
           f"({cfg.n_head}); the head dimension would not be an integer.")

    # ---- learned-absolute positions, and NO rotary -----------------------
    # Required by the amendment: the GL(head_dim) freedom this module exploits
    # exists only because positions are learned-absolute.
    _check(hasattr(tr, "wpe"),
           f"{where}: no transformer.wpe. " + _WHY_ROTARY_MATTERS)
    _check(isinstance(tr.wpe, nn.Embedding),
           f"{where}: transformer.wpe is {type(tr.wpe).__name__}, expected "
           f"nn.Embedding (learned absolute positions). " + _WHY_ROTARY_MATTERS)
    _check(tuple(tr.wpe.weight.shape) == (expected.n_positions, expected.n_embd),
           f"{where}: transformer.wpe.weight is "
           f"{tuple(tr.wpe.weight.shape)}, expected "
           f"({expected.n_positions}, {expected.n_embd}).")

    rotary_modules = [n for n, _ in model.named_modules()
                      if "rotary" in n.lower() or "rope" in n.lower()]
    _check(not rotary_modules,
           f"{where}: found rotary/RoPE submodules {rotary_modules}. "
           + _WHY_ROTARY_MATTERS)
    rotary_buffers = [n for n, _ in model.named_buffers()
                      if "inv_freq" in n.lower() or "rotary" in n.lower()
                      or "rope" in n.lower()]
    _check(not rotary_buffers,
           f"{where}: found rotary buffers {rotary_buffers}. "
           + _WHY_ROTARY_MATTERS)
    for attr in ("rope_theta", "rope_scaling", "rotary_dim",
                 "partial_rotary_factor"):
        _check(getattr(cfg, attr, None) is None,
               f"{where}: config.{attr} is set, which indicates rotary "
               f"position embeddings. " + _WHY_ROTARY_MATTERS)
    pos_type = getattr(cfg, "position_embedding_type", "absolute")
    _check(pos_type == "absolute",
           f"{where}: config.position_embedding_type is {pos_type!r}, expected "
           f"'absolute'. " + _WHY_ROTARY_MATTERS)

    # ---- tied embeddings -------------------------------------------------
    _check(model.lm_head.weight is tr.wte.weight,
           f"{where}: lm_head.weight is not the same tensor as "
           "transformer.wte.weight. This module assumes tied embeddings: the "
           "tie is what makes the output projection the transpose of the "
           "input embedding, which is why orthogonal re-gaugings of the "
           "residual stream are consistent and scalings are not. An untied "
           "model has a different symmetry group and must not be "
           "canonicalized with this recipe.")
    _check(tuple(tr.wte.weight.shape) == (expected.vocab_size, expected.n_embd),
           f"{where}: transformer.wte.weight is {tuple(tr.wte.weight.shape)}, "
           f"expected ({expected.vocab_size}, {expected.n_embd}).")

    # ---- attention maths must be the plain form --------------------------
    for attr, want in (("scale_attn_by_inverse_layer_idx", False),
                       ("reorder_and_upcast_attn", False)):
        got = getattr(cfg, attr, False)
        _check(got == want,
               f"{where}: config.{attr} is {got!r}, expected {want!r}. This "
               "module was written against the plain attention form; a "
               "variant changes the arithmetic the symmetry proofs rest on.")

    act = getattr(cfg, "activation_function", None)
    _check(isinstance(act, str) and "gelu" in act,
           f"{where}: activation_function is {act!r}. This module was written "
           "against GELU, and specifically records that FFN input/output "
           "scaling is NOT a symmetry because GELU is not positively "
           "homogeneous. A different activation changes that conclusion.")

    # ---- blocks ----------------------------------------------------------
    _check(len(tr.h) == expected.n_layer,
           f"{where}: {len(tr.h)} blocks, expected {expected.n_layer}.")

    d = expected.n_embd
    for i, block in enumerate(tr.h):
        at = f"{where}: block {i}"
        for ln_name in ("ln_1", "ln_2"):
            ln = getattr(block, ln_name, None)
            _check(isinstance(ln, nn.LayerNorm),
                   f"{at}.{ln_name} is {type(ln).__name__}, expected "
                   "nn.LayerNorm. Gain absorption is defined against "
                   "LayerNorm's exact form (mean subtraction, per-channel "
                   "gain and bias); RMSNorm would need a different step.")
            _check(tuple(ln.weight.shape) == (d,) and tuple(ln.bias.shape) == (d,),
                   f"{at}.{ln_name} has weight {tuple(ln.weight.shape)} / bias "
                   f"{tuple(ln.bias.shape)}, expected ({d},) for both.")

        attn = block.attn
        _check(isinstance(attn.c_attn, Conv1D),
               f"{at}.attn.c_attn is {type(attn.c_attn).__name__}, expected "
               "transformers Conv1D. Conv1D stores its weight TRANSPOSED "
               "relative to nn.Linear -- (in_features, out_features) rather "
               "than (out, in) -- and every axis in this module is chosen for "
               "that layout. An nn.Linear here would make every permutation "
               "hit the wrong axis while still running.")
        _check(isinstance(attn.c_proj, Conv1D),
               f"{at}.attn.c_proj is {type(attn.c_proj).__name__}, expected "
               "transformers Conv1D.")
        _check(tuple(attn.c_attn.weight.shape) == (d, 3 * d),
               f"{at}.attn.c_attn.weight is "
               f"{tuple(attn.c_attn.weight.shape)}, expected ({d}, {3 * d}). "
               "c_attn must be the FUSED QKV tensor: columns 0:{d} are Q, "
               f"{d}:{2 * d} are K, {2 * d}:{3 * d} are V. Everything this "
               "module does to attention depends on that layout, and "
               "permuting across a QKV boundary produces a model that runs "
               "and is wrong.")
        _check(tuple(attn.c_attn.bias.shape) == (3 * d,),
               f"{at}.attn.c_attn.bias is {tuple(attn.c_attn.bias.shape)}, "
               f"expected ({3 * d},).")
        _check(tuple(attn.c_proj.weight.shape) == (d, d),
               f"{at}.attn.c_proj.weight is "
               f"{tuple(attn.c_proj.weight.shape)}, expected ({d}, {d}).")
        _check(tuple(attn.c_proj.bias.shape) == (d,),
               f"{at}.attn.c_proj.bias is {tuple(attn.c_proj.bias.shape)}, "
               f"expected ({d},).")
        _check(getattr(attn, "num_heads", None) == expected.n_head,
               f"{at}.attn.num_heads is {getattr(attn, 'num_heads', None)!r}, "
               f"expected {expected.n_head}.")
        _check(getattr(attn, "head_dim", None) == expected.head_dim,
               f"{at}.attn.head_dim is {getattr(attn, 'head_dim', None)!r}, "
               f"expected {expected.head_dim}.")
        _check(getattr(attn, "split_size", None) == d,
               f"{at}.attn.split_size is "
               f"{getattr(attn, 'split_size', None)!r}, expected {d}.")

        mlp = block.mlp
        _check(isinstance(mlp.c_fc, Conv1D) and isinstance(mlp.c_proj, Conv1D),
               f"{at}.mlp projections must both be transformers Conv1D; got "
               f"{type(mlp.c_fc).__name__} and {type(mlp.c_proj).__name__}.")
        _check(tuple(mlp.c_fc.weight.shape) == (d, expected.n_inner),
               f"{at}.mlp.c_fc.weight is {tuple(mlp.c_fc.weight.shape)}, "
               f"expected ({d}, {expected.n_inner}).")
        _check(tuple(mlp.c_fc.bias.shape) == (expected.n_inner,),
               f"{at}.mlp.c_fc.bias is {tuple(mlp.c_fc.bias.shape)}, expected "
               f"({expected.n_inner},).")
        _check(tuple(mlp.c_proj.weight.shape) == (expected.n_inner, d),
               f"{at}.mlp.c_proj.weight is {tuple(mlp.c_proj.weight.shape)}, "
               f"expected ({expected.n_inner}, {d}).")
        _check(tuple(mlp.c_proj.bias.shape) == (d,),
               f"{at}.mlp.c_proj.bias is {tuple(mlp.c_proj.bias.shape)}, "
               f"expected ({d},).")

    _check(isinstance(tr.ln_f, nn.LayerNorm),
           f"{where}: transformer.ln_f is {type(tr.ln_f).__name__}, expected "
           "nn.LayerNorm.")
    _check(tuple(tr.ln_f.weight.shape) == (d,),
           f"{where}: transformer.ln_f.weight is "
           f"{tuple(tr.ln_f.weight.shape)}, expected ({d},).")

    # ---- parameter count -------------------------------------------------
    counted = sum(p.numel() for p in model.parameters())
    _check(counted == expected.n_params,
           f"{where}: model has {counted:,} parameters, but the declared "
           f"ArchSpec implies {expected.n_params:,}. Something in the "
           "architecture differs from what this module believes it is "
           "rewriting.")

    return expected


# ---------------------------------------------------------------------------
# The layout contract
#
# Every slice into a fused tensor is written ONCE, here. c_attn packs Q, K and
# V into one (n_embd, 3*n_embd) tensor and each of those thirds packs n_head
# blocks of head_dim. Getting a slice wrong does not crash -- it produces a
# model that runs and lies -- so the arithmetic lives in one place with one
# test rather than being repeated at every call site.
# ---------------------------------------------------------------------------

#: Order of the three projections packed into c_attn's output axis.
QKV = ("q", "k", "v")


def qkv_offset(which: str, arch: ArchSpec) -> int:
    """Column where `which` of the fused QKV tensor starts."""
    if which not in QKV:
        raise CanonicalizeError(
            f"unknown projection {which!r}; c_attn packs exactly {QKV}")
    return QKV.index(which) * arch.n_embd


def head_columns(which: str, head: int, arch: ArchSpec) -> slice:
    """Columns of c_attn holding `head`'s slice of the `which` projection."""
    if not 0 <= head < arch.n_head:
        raise CanonicalizeError(
            f"head {head} out of range 0..{arch.n_head - 1}")
    start = qkv_offset(which, arch) + head * arch.head_dim
    return slice(start, start + arch.head_dim)


def head_rows_of_out_proj(head: int, arch: ArchSpec) -> slice:
    """Rows of attn.c_proj holding `head`'s output.

    attn.c_proj's INPUT axis is the concatenation of the per-head outputs in
    head order, which is the same ordering c_attn uses for its Q/K/V blocks.
    That correspondence is what makes a head permutation coherent, and it is
    asserted in the tests rather than assumed here.
    """
    if not 0 <= head < arch.n_head:
        raise CanonicalizeError(
            f"head {head} out of range 0..{arch.n_head - 1}")
    start = head * arch.head_dim
    return slice(start, start + arch.head_dim)


# ---------------------------------------------------------------------------
# The equivalence harness
# ---------------------------------------------------------------------------


#: Fixed so that every measurement in this module is taken on the same input.
#: The exact tokens do not matter -- what matters is that they never change
#: between a model and its transformed copy, and never change between runs.
PROBE_SEED = 20260802
PROBE_LENGTH = 64


def probe_tokens(arch: ArchSpec, length: int = PROBE_LENGTH,
                 seed: int = PROBE_SEED) -> list[int]:
    """A fixed, reproducible token sequence inside this model's vocabulary."""
    n = min(length, arch.n_positions)
    rng = random.Random(seed)
    return [rng.randrange(arch.vocab_size) for _ in range(n)]


def forward_logits(model, ids):
    """Logits for one sequence. No grad, no state touched, eval mode assumed."""
    torch = _torch()
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            input_ids = torch.tensor([list(ids)], dtype=torch.long)
            return model(input_ids=input_ids).logits.detach()
    finally:
        if was_training:
            model.train()


@dataclass(frozen=True)
class LogitDiff:
    """How far apart two models' logits are on the same input."""

    max_abs: float
    #: max_abs divided by the largest logit magnitude in the reference. This is
    #: the number every tolerance in this module is stated against, because an
    #: absolute difference is meaningless without the scale it sits on.
    max_rel: float
    reference_max_abs: float
    dtype: str

    def __str__(self) -> str:
        return (f"max_abs={self.max_abs:.6e} max_rel={self.max_rel:.6e} "
                f"({self.dtype})")


def logit_difference(model_a, model_b, ids) -> LogitDiff:
    """Compare two models on one fixed input. The core equivalence measurement."""
    torch = _torch()
    la = forward_logits(model_a, ids)
    lb = forward_logits(model_b, ids)
    if la.shape != lb.shape:
        raise CanonicalizeError(
            f"logit shapes differ: {tuple(la.shape)} vs {tuple(lb.shape)}")
    scale = float(la.abs().max())
    diff = float((la - lb).abs().max())
    if scale == 0.0:
        raise CanonicalizeError(
            "the reference model produced all-zero logits; a relative "
            "difference cannot be formed against zero")
    return LogitDiff(max_abs=diff, max_rel=diff / scale,
                     reference_max_abs=scale, dtype=str(la.dtype))


# ---------------------------------------------------------------------------
# Random draws used by the symmetries
# ---------------------------------------------------------------------------


def _generator(seed: int):
    torch = _torch()
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def _random_orthogonal(n: int, gen, dtype):
    """Haar-distributed orthogonal matrix via QR with a sign fix."""
    torch = _torch()
    a = torch.randn(n, n, generator=gen, dtype=dtype)
    q, r = torch.linalg.qr(a)
    # Without the sign fix QR is not unique and the draw is not Haar.
    return q * torch.sign(torch.diagonal(r)).unsqueeze(0)


def _random_rotation_fixing_ones(n: int, gen, dtype):
    """Orthogonal Q with Q @ 1 = 1.

    LayerNorm subtracts the per-token mean, so a rotation can only be a
    candidate symmetry if it leaves the all-ones direction alone. Built by
    rotating only inside the (n-1)-dimensional complement of that direction.
    """
    torch = _torch()
    u = torch.ones(n, dtype=dtype) / math.sqrt(n)
    a = torch.randn(n, n - 1, generator=gen, dtype=dtype)
    a = a - torch.outer(u, u @ a)          # project the all-ones part out
    v, _ = torch.linalg.qr(a)              # (n, n-1), orthonormal, u-orthogonal
    o = _random_orthogonal(n - 1, gen, dtype)
    return torch.outer(u, u) + v @ o @ v.T


def _random_invertible(n: int, gen, dtype, log_cond: float = 0.35):
    """Well-conditioned invertible matrix, condition number <= exp(2*log_cond).

    Deliberately NOT a raw Gaussian. A Gaussian 64x64 routinely has a condition
    number in the hundreds, and the resulting float error would confound the
    question these tests ask, which is whether the transformation is exact in
    exact arithmetic -- not how badly an ill-conditioned inverse rounds.
    """
    torch = _torch()
    q1 = _random_orthogonal(n, gen, dtype)
    q2 = _random_orthogonal(n, gen, dtype)
    s = torch.exp((torch.rand(n, generator=gen, dtype=dtype) * 2 - 1) * log_cond)
    return q1 @ torch.diag(s) @ q2


def _random_permutation(n: int, gen):
    torch = _torch()
    return torch.randperm(n, generator=gen)


# ---------------------------------------------------------------------------
# The symmetries
#
# A Symmetry is the SCRAMBLE: a random re-gauging drawn for one model and
# applied to it. Canonicalization is the separate, deterministic UNSCRAMBLE.
# They are deliberately not one class with apply/undo -- they are not inverses
# of each other, and pretending otherwise would hide bugs.
#
# sample() and apply() are split because the round-trip test has to apply the
# SAME drawn transform to two copies of a model. If apply() re-drew, that test
# would silently compare two different transforms and pass for the wrong reason.
# ---------------------------------------------------------------------------


@dataclass
class Symmetry:
    """Base class. Subclasses fill in name, continuity, sample() and apply()."""

    #: Stable identifier, used in reports and test ids.
    name: str = "unnamed"
    #: True if the symmetry has a continuous parameter (a rotation angle, a
    #: scale, a shift), False if it is discrete (a permutation). The
    #: distinction is load-bearing for the zero-gradient argument: the loss is
    #: exactly flat along a continuous gauge direction, so its gradient there
    #: is exactly zero and the coordinate never moves during training. A
    #: discrete symmetry has no such direction and the argument does not apply.
    continuous: bool = True
    params: dict = field(default_factory=dict)

    def sample(self, model, arch: ArchSpec, seed: int) -> "Symmetry":
        raise NotImplementedError

    def apply(self, model, arch: ArchSpec) -> None:
        raise NotImplementedError

    def _require_sampled(self) -> None:
        if not self.params:
            raise CanonicalizeError(
                f"{self.name}: apply() called before sample(). The draw and "
                "the application are separate so the same transform can be "
                "applied to two models; calling apply() first would silently "
                "do nothing.")


def _as(t, ref):
    """Cast a float64 drawn parameter to the model's dtype."""
    return t.to(dtype=ref.dtype)


# ---- 1. LayerNorm gain rescale --------------------------------------------


@dataclass
class LayerNormGainRescale(Symmetry):
    """Scale a LayerNorm's gain, divide it back out of the next weight.

    LN(x) = gamma * xhat + beta. Multiplying gamma and beta by a per-channel
    vector s multiplies the whole LN output by s. Dividing the ROWS of the
    following Conv1D weight by s cancels it exactly, because Conv1D computes
    x @ W and its rows are the input axis.
    """

    name: str = "layernorm_gain_rescale"
    continuous: bool = True

    def sample(self, model, arch, seed):
        torch = _torch()
        gen = _generator(seed)
        draws = {}
        for i in range(arch.n_layer):
            for ln_name in ("ln_1", "ln_2"):
                # Log-uniform in [1/2, 2]: away from zero in both directions,
                # so the test exercises shrink as well as grow.
                s = torch.exp(
                    (torch.rand(arch.n_embd, generator=gen,
                                dtype=torch.float64) * 2 - 1) * math.log(2.0))
                draws[(i, ln_name)] = s
        self.params = {"scales": draws}
        return self

    def apply(self, model, arch):
        self._require_sampled()
        torch = _torch()
        with torch.no_grad():
            for i, block in enumerate(model.transformer.h):
                for ln_name, proj in (("ln_1", block.attn.c_attn),
                                      ("ln_2", block.mlp.c_fc)):
                    ln = getattr(block, ln_name)
                    s = _as(self.params["scales"][(i, ln_name)], ln.weight)
                    ln.weight.mul_(s)
                    ln.bias.mul_(s)
                    # Conv1D weight is (in, out): rows are the input axis.
                    proj.weight.div_(s.unsqueeze(1))


# ---- 2. Residual rotation ---------------------------------------------------


@dataclass
class ResidualRotation(Symmetry):
    """Rotate the residual stream by an orthogonal Q with Q @ 1 = 1.

    PREDICTED TO FAIL, and tested anyway. The all-ones condition handles
    LayerNorm's mean subtraction, but LayerNorm also carries a PER-CHANNEL
    GAIN, and a diagonal does not commute with a general rotation. That is
    curable at ln_1 and ln_2 by absorbing the gain first -- and incurable at
    ln_f, because absorbing ln_f's gain means folding it into lm_head, which is
    the tied embedding and not ours to touch.

    Note what is NOT the reason: tying is not hostile to rotation. A rotation
    is orthogonal, so its inverse is its transpose, and a tied output
    projection supplies exactly that. Tying is fatal to SCALING, which is not
    orthogonal, and harmless to rotation. The obstruction here is LayerNorm.
    """

    name: str = "residual_rotation"
    continuous: bool = True

    def sample(self, model, arch, seed):
        torch = _torch()
        gen = _generator(seed)
        self.params = {"Q": _random_rotation_fixing_ones(
            arch.n_embd, gen, torch.float64)}
        return self

    def apply(self, model, arch):
        self._require_sampled()
        torch = _torch()
        tr = model.transformer
        with torch.no_grad():
            Q = _as(self.params["Q"], tr.wte.weight)
            # Rows of wte/wpe are vocab/position; columns are residual channels.
            tr.wte.weight.copy_(tr.wte.weight @ Q)
            tr.wpe.weight.copy_(tr.wpe.weight @ Q)
            for block in tr.h:
                for ln_name, proj in (("ln_1", block.attn.c_attn),
                                      ("ln_2", block.mlp.c_fc)):
                    ln = getattr(block, ln_name)
                    # The naive rule, which is exactly right for a permutation
                    # and exactly wrong for a general rotation. Measuring how
                    # wrong is the point of the test.
                    ln.weight.copy_(ln.weight @ Q)
                    ln.bias.copy_(ln.bias @ Q)
                    proj.weight.copy_(Q.T @ proj.weight)
                for proj in (block.attn.c_proj, block.mlp.c_proj):
                    proj.weight.copy_(proj.weight @ Q)
                    proj.bias.copy_(proj.bias @ Q)
            tr.ln_f.weight.copy_(tr.ln_f.weight @ Q)
            tr.ln_f.bias.copy_(tr.ln_f.bias @ Q)


# ---- 3. Residual permutation ------------------------------------------------


@dataclass
class ResidualPermutation(Symmetry):
    """Permute the residual channels consistently everywhere.

    The special case of a rotation that survives, because a permutation DOES
    conjugate a diagonal to a diagonal, so LayerNorm's per-channel gain simply
    permutes with it. Tested for equivalence; deliberately NOT canonicalized --
    same-seed twins diverging mid-training do not spontaneously permute
    residual channels, so quotienting it buys this study nothing.

    Implemented with index_select rather than a matmul so it is exact:
    reordering floats introduces no arithmetic at all.
    """

    name: str = "residual_permutation"
    continuous: bool = False

    def sample(self, model, arch, seed):
        gen = _generator(seed)
        self.params = {"perm": _random_permutation(arch.n_embd, gen)}
        return self

    def apply(self, model, arch):
        self._require_sampled()
        torch = _torch()
        p = self.params["perm"]
        tr = model.transformer
        with torch.no_grad():
            tr.wte.weight.copy_(tr.wte.weight[:, p])
            tr.wpe.weight.copy_(tr.wpe.weight[:, p])
            for block in tr.h:
                for ln_name, proj in (("ln_1", block.attn.c_attn),
                                      ("ln_2", block.mlp.c_fc)):
                    ln = getattr(block, ln_name)
                    ln.weight.copy_(ln.weight[p])
                    ln.bias.copy_(ln.bias[p])
                    proj.weight.copy_(proj.weight[p, :])
                for proj in (block.attn.c_proj, block.mlp.c_proj):
                    proj.weight.copy_(proj.weight[:, p])
                    proj.bias.copy_(proj.bias[p])
            tr.ln_f.weight.copy_(tr.ln_f.weight[p])
            tr.ln_f.bias.copy_(tr.ln_f.bias[p])


# ---- 4. Residual scaling ----------------------------------------------------


@dataclass
class ResidualScaling(Symmetry):
    """Scale the residual stream by a positive scalar c.

    PREDICTED TO FAIL, on tying grounds -- and this is the case where tying
    really is the killer. LayerNorm is scale-invariant, so the block inputs are
    unaffected and the interior works out. But the residual has to be scaled at
    the input by multiplying wte, and wte IS the output projection. Scaling
    needs c going in and 1/c coming out; tying forces c both times, so the
    logits emerge multiplied by c.

    The general rule: tying makes the output projection W^T, which equals
    W^-1 exactly when W is orthogonal. Permutations and rotations are
    orthogonal and survive tying; a scalar is not and does not.
    """

    name: str = "residual_scaling"
    continuous: bool = True

    def sample(self, model, arch, seed):
        rng = random.Random(seed)
        self.params = {"c": math.exp(rng.uniform(-math.log(2.0), math.log(2.0)))}
        return self

    def apply(self, model, arch):
        self._require_sampled()
        torch = _torch()
        c = self.params["c"]
        tr = model.transformer
        with torch.no_grad():
            tr.wte.weight.mul_(c)
            tr.wpe.weight.mul_(c)
            for block in tr.h:
                for proj in (block.attn.c_proj, block.mlp.c_proj):
                    proj.weight.mul_(c)
                    proj.bias.mul_(c)


# ---- 5. Head permutation ----------------------------------------------------


@dataclass
class HeadPermutation(Symmetry):
    """Reorder the attention heads within each layer.

    Heads are computed independently and concatenated, so reordering them is
    free PROVIDED the same permutation is applied to Q, K and V -- head j's
    output depends on Q_j, K_j and V_j together, so permuting them differently
    would rewire the head rather than rename it -- and provided the matching
    row-blocks of attn.c_proj move with them.

    Two ways to get this wrong that do not crash: permuting c_attn's ROW axis
    (which is the residual, not the heads), and permuting the full 3*n_embd
    output axis in head_dim blocks, which crosses the Q/K/V boundary and mixes
    queries with keys. Both are exercised as mutation faults in the tests.
    """

    name: str = "head_permutation"
    continuous: bool = False

    def sample(self, model, arch, seed):
        gen = _generator(seed)
        self.params = {"perms": [_random_permutation(arch.n_head, gen)
                                 for _ in range(arch.n_layer)]}
        return self

    def apply(self, model, arch):
        self._require_sampled()
        torch = _torch()
        with torch.no_grad():
            for i, block in enumerate(model.transformer.h):
                perm = self.params["perms"][i]
                c_attn, c_proj = block.attn.c_attn, block.attn.c_proj
                new_w = c_attn.weight.clone()
                new_b = c_attn.bias.clone()
                new_o = c_proj.weight.clone()
                for new_head, old_head in enumerate(perm.tolist()):
                    for which in QKV:
                        dst = head_columns(which, new_head, arch)
                        src = head_columns(which, old_head, arch)
                        new_w[:, dst] = c_attn.weight[:, src]
                        new_b[dst] = c_attn.bias[src]
                    new_o[head_rows_of_out_proj(new_head, arch), :] = \
                        c_proj.weight[head_rows_of_out_proj(old_head, arch), :]
                c_attn.weight.copy_(new_w)
                c_attn.bias.copy_(new_b)
                c_proj.weight.copy_(new_o)


# ---- 6. Head-internal GL transforms ----------------------------------------


@dataclass
class HeadInternalTransform(Symmetry):
    """Push an invertible matrix through a head's Q/K pair and V/O pair.

    Attention scores depend on Q and K only through the product W_Q W_K^T, so
    W_Q -> W_Q A and W_K -> W_K A^-T changes no score at all, for any
    invertible A. Independently, a head's output depends on V and that head's
    slice of c_proj only through their product, giving a second such freedom.

    Biases ride the same transform: with c_attn's bias present the score is
    affine, not linear, and the invariant terms include W_Q b_K^T, b_Q W_K^T
    and b_Q b_K^T. All four are preserved by the same A.

    This is by far the largest symmetry group in the architecture and the most
    valuable to quotient out -- and the most fragile numerically, since it is
    the only one requiring a matrix inverse.
    """

    name: str = "head_internal_transform"
    continuous: bool = True

    def sample(self, model, arch, seed):
        torch = _torch()
        gen = _generator(seed)
        dh = arch.head_dim
        self.params = {
            "A": [[_random_invertible(dh, gen, torch.float64)
                   for _ in range(arch.n_head)] for _ in range(arch.n_layer)],
            "B": [[_random_invertible(dh, gen, torch.float64)
                   for _ in range(arch.n_head)] for _ in range(arch.n_layer)],
        }
        return self

    def apply(self, model, arch):
        self._require_sampled()
        torch = _torch()
        with torch.no_grad():
            for i, block in enumerate(model.transformer.h):
                c_attn, c_proj = block.attn.c_attn, block.attn.c_proj
                for h in range(arch.n_head):
                    A = _as(self.params["A"][i][h], c_attn.weight)
                    B = _as(self.params["B"][i][h], c_attn.weight)
                    A_inv_T = torch.linalg.inv(A).T
                    B_inv = torch.linalg.inv(B)

                    q, k, v = (head_columns(w, h, arch) for w in QKV)
                    # Row-vector convention throughout: x @ W_Q, b_Q @ A.
                    c_attn.weight[:, q] = c_attn.weight[:, q] @ A
                    c_attn.bias[q] = c_attn.bias[q] @ A
                    c_attn.weight[:, k] = c_attn.weight[:, k] @ A_inv_T
                    c_attn.bias[k] = c_attn.bias[k] @ A_inv_T
                    c_attn.weight[:, v] = c_attn.weight[:, v] @ B
                    c_attn.bias[v] = c_attn.bias[v] @ B
                    rows = head_rows_of_out_proj(h, arch)
                    c_proj.weight[rows, :] = B_inv @ c_proj.weight[rows, :]


# ---- 7. FFN neuron permutation ---------------------------------------------


@dataclass
class FFNNeuronPermutation(Symmetry):
    """Reorder the FFN's hidden neurons.

    GELU is elementwise, so a permutation of the hidden axis commutes with it.
    c_fc's COLUMNS are the hidden neurons and c_proj's ROWS are the same
    neurons -- opposite axes, because Conv1D is (in, out).
    """

    name: str = "ffn_neuron_permutation"
    continuous: bool = False

    def sample(self, model, arch, seed):
        gen = _generator(seed)
        self.params = {"perms": [_random_permutation(arch.n_inner, gen)
                                 for _ in range(arch.n_layer)]}
        return self

    def apply(self, model, arch):
        self._require_sampled()
        torch = _torch()
        with torch.no_grad():
            for i, block in enumerate(model.transformer.h):
                p = self.params["perms"][i]
                block.mlp.c_fc.weight.copy_(block.mlp.c_fc.weight[:, p])
                block.mlp.c_fc.bias.copy_(block.mlp.c_fc.bias[p])
                block.mlp.c_proj.weight.copy_(block.mlp.c_proj.weight[p, :])


# ---- 8. FFN scaling ---------------------------------------------------------


@dataclass
class FFNScaling(Symmetry):
    """Scale FFN hidden units up on the way in and down on the way out.

    PREDICTED TO FAIL. This is a symmetry for POSITIVELY HOMOGENEOUS
    activations: ReLU satisfies ReLU(s*z) = s*ReLU(z), so the scale passes
    straight through and cancels. GELU does not. GELU(z) = z * Phi(z), so
    GELU(s*z) = s*z*Phi(s*z), and Phi(s*z) != Phi(z) for any s != 1. The gate
    itself moves, and no downstream rescaling can undo that.
    """

    name: str = "ffn_scaling"
    continuous: bool = True

    def sample(self, model, arch, seed):
        torch = _torch()
        gen = _generator(seed)
        self.params = {"scales": [
            torch.exp((torch.rand(arch.n_inner, generator=gen,
                                  dtype=torch.float64) * 2 - 1) * math.log(2.0))
            for _ in range(arch.n_layer)]}
        return self

    def apply(self, model, arch):
        self._require_sampled()
        torch = _torch()
        with torch.no_grad():
            for i, block in enumerate(model.transformer.h):
                s = _as(self.params["scales"][i], block.mlp.c_fc.weight)
                block.mlp.c_fc.weight.mul_(s.unsqueeze(0))   # columns = out
                block.mlp.c_fc.bias.mul_(s)
                block.mlp.c_proj.weight.div_(s.unsqueeze(1))  # rows = in


# ---- 9. Attention key-bias shift -------------------------------------------


@dataclass
class KeyBiasShift(Symmetry):
    """Add a constant to every key in a head.

    Shifting b_K shifts every key by the same vector, which adds a PER-QUERY
    constant to that row of the score matrix -- and softmax is invariant to a
    constant added across the row it normalises over. So b_K is pure gauge: it
    carries no function whatsoever.

    Surfaced while working out the full affine invariant for the head-internal
    step. It matters there because a gauge quantity must not be used to break
    ties between near-degenerate singular values -- that would be breaking ties
    with noise.
    """

    name: str = "key_bias_shift"
    continuous: bool = True

    def sample(self, model, arch, seed):
        torch = _torch()
        gen = _generator(seed)
        self.params = {"shifts": [
            [torch.randn(arch.head_dim, generator=gen, dtype=torch.float64)
             for _ in range(arch.n_head)] for _ in range(arch.n_layer)]}
        return self

    def apply(self, model, arch):
        self._require_sampled()
        torch = _torch()
        with torch.no_grad():
            for i, block in enumerate(model.transformer.h):
                bias = block.attn.c_attn.bias
                for h in range(arch.n_head):
                    c = _as(self.params["shifts"][i][h], bias)
                    bias[head_columns("k", h, arch)] += c


# ---- 10. Attention value-bias shift ----------------------------------------


@dataclass
class ValueBiasShift(Symmetry):
    """Add a constant to every value in a head, compensating in c_proj's bias.

    Attention probabilities sum to one across keys, so shifting b_V by c shifts
    that head's output by exactly c. That lands in the residual as the constant
    c @ W_O, which is absorbable into attn.c_proj.bias. So b_V is gauge up to a
    compensation elsewhere.

    Same origin and same consequence as the key-bias shift.
    """

    name: str = "value_bias_shift"
    continuous: bool = True

    def sample(self, model, arch, seed):
        torch = _torch()
        gen = _generator(seed)
        self.params = {"shifts": [
            [torch.randn(arch.head_dim, generator=gen, dtype=torch.float64)
             for _ in range(arch.n_head)] for _ in range(arch.n_layer)]}
        return self

    def apply(self, model, arch):
        self._require_sampled()
        torch = _torch()
        with torch.no_grad():
            for i, block in enumerate(model.transformer.h):
                c_attn, c_proj = block.attn.c_attn, block.attn.c_proj
                for h in range(arch.n_head):
                    c = _as(self.params["shifts"][i][h], c_attn.bias)
                    c_attn.bias[head_columns("v", h, arch)] += c
                    w_o = c_proj.weight[head_rows_of_out_proj(h, arch), :]
                    c_proj.bias.sub_(c @ w_o)


#: Every candidate, in the order they are reported. Membership here means
#: "tested in isolation", never "believed to be a symmetry".
ALL_SYMMETRIES: tuple[type, ...] = (
    LayerNormGainRescale,
    ResidualRotation,
    ResidualPermutation,
    ResidualScaling,
    HeadPermutation,
    HeadInternalTransform,
    FFNNeuronPermutation,
    FFNScaling,
    KeyBiasShift,
    ValueBiasShift,
)


def symmetry_by_name(name: str) -> type:
    for cls in ALL_SYMMETRIES:
        if cls().name == name:
            return cls
    raise CanonicalizeError(
        f"unknown symmetry {name!r}; known: "
        f"{', '.join(c().name for c in ALL_SYMMETRIES)}")


# ---------------------------------------------------------------------------
# The criterion
#
# THE FLOAT32 -> FLOAT64 COLLAPSE FACTOR IS THE SOLE PASS/FAIL CRITERION, and
# in practice it always was. A transformation that is exact in exact arithmetic
# shows float32-epsilon error in float32 and many orders less in float64. One
# that is not a symmetry shows a large error that does NOT shrink when the
# precision goes up. Measured on real GPT-2: 3.90e+08 to 8.30e+08 for the seven
# confirmed symmetries, exactly 1.00 for all three drops. Eight orders of
# separation, no overlap.
#
# The float32 bounds below are DEMOTED to reported diagnostics and gate
# nothing. The approved plan pre-registered a float32 bound of 1e-06, and phase
# 2 then measured this architecture's own float32 noise floor at 4.6e-07 to
# 1.4e-06 across five seeds -- so the bound sat INSIDE the noise floor and
# never discriminated anything. It was demoted rather than moved: moving it to
# a value the candidates pass would have been indistinguishable from tuning it
# to the result. See S43 in implementation-notes.md.
#
# The float64 tolerances and COLLAPSE_FACTOR are exactly as pre-registered and
# are not to be adjusted.
# ---------------------------------------------------------------------------

#: PRE-REGISTERED, UNCHANGED. Reordering only, no new arithmetic.
TOL_PERMUTATION_F64 = 1e-12
#: PRE-REGISTERED, UNCHANGED. One multiply and one divide per channel, plus a
#: softmax shift for the bias symmetries.
TOL_ELEMENTWISE_F64 = 1e-12
#: PRE-REGISTERED, UNCHANGED. Involves a matrix inverse, so the error scales
#: with the condition number.
TOL_HEAD_INTERNAL_F64 = 1e-10

#: PRE-REGISTERED, UNCHANGED, AND THE ONLY CRITERION. A candidate whose error
#: fails to fall by at least this factor between float32 and float64 is not a
#: symmetry, whatever its absolute error.
COLLAPSE_FACTOR = 1e4

TOL_F64: dict[str, float] = {
    "layernorm_gain_rescale": TOL_ELEMENTWISE_F64,
    "residual_rotation": TOL_ELEMENTWISE_F64,
    "residual_permutation": TOL_PERMUTATION_F64,
    "residual_scaling": TOL_ELEMENTWISE_F64,
    "head_permutation": TOL_PERMUTATION_F64,
    "head_internal_transform": TOL_HEAD_INTERNAL_F64,
    "ffn_neuron_permutation": TOL_PERMUTATION_F64,
    "ffn_scaling": TOL_ELEMENTWISE_F64,
    "key_bias_shift": TOL_ELEMENTWISE_F64,
    "value_bias_shift": TOL_ELEMENTWISE_F64,
}

#: The originally pre-registered float32 bounds, kept ONLY so reports can say
#: how far a measurement sits from them. Nothing branches on these.
F32_DIAGNOSTIC: dict[str, float] = {
    name: (1e-4 if name == "head_internal_transform" else 1e-6)
    for name in TOL_F64
}

#: MEASURED EMPIRICAL PROPERTY of GPT-2's float32 forward pass at 768 wide and
#: 12 deep, taken from transformations proven exact in float64, five seeds
#: each. This is a fact about the architecture, NOT a threshold anything has to
#: clear. Recorded so a later measurement can be read against it.
MEASURED_F32_NOISE_FLOOR: tuple[float, float] = (4.592e-07, 1.410e-06)


# ---------------------------------------------------------------------------
# Measuring one candidate in isolation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EquivalenceResult:
    """One candidate, measured at both precisions.

    `passes` is decided by the COLLAPSE FACTOR and the float64 tolerance only.
    The float32 number is carried and reported but gates nothing -- see the
    criterion block above.
    """

    name: str
    continuous: bool
    f32: LogitDiff
    f64: LogitDiff
    #: Reported only. Not a threshold. Kept so a report can state the distance.
    f32_diagnostic: float
    tol_f64: float

    @property
    def collapse(self) -> float:
        """How many times smaller the float64 error is. Large means exact."""
        if self.f64.max_rel == 0.0:
            return math.inf
        return self.f32.max_rel / self.f64.max_rel

    @property
    def passes(self) -> bool:
        """THE criterion: the error collapses, and float64 is inside bound."""
        return (self.collapse >= COLLAPSE_FACTOR
                and self.f64.max_rel <= self.tol_f64)

    @property
    def f32_within_noise_floor(self) -> bool:
        """Diagnostic: does the float32 error look like ordinary reassociation?"""
        return self.f32.max_rel <= MEASURED_F32_NOISE_FLOOR[1]

    @property
    def verdict(self) -> str:
        if self.passes:
            return "SYMMETRY"
        if self.collapse >= COLLAPSE_FACTOR:
            return "NUMERICAL"      # exact maths, float64 bound missed
        return "NOT A SYMMETRY"


def measure_equivalence(build_model, arch: ArchSpec, symmetry_cls, seed: int,
                        ids=None) -> EquivalenceResult:
    """Apply one candidate in isolation and measure the logit difference.

    `build_model` is a zero-argument callable returning a FRESH model, so each
    precision gets a clean pair and no measurement can be contaminated by a
    previous one.

    The same drawn transform is used at both precisions: parameters are drawn
    in float64 and cast down, so float32 and float64 are measuring the same
    mathematical object rather than two different random draws.
    """
    torch = _torch()
    ids = probe_tokens(arch) if ids is None else ids
    results = {}
    for label, dtype in (("f32", torch.float32), ("f64", torch.float64)):
        base = build_model().to(dtype=dtype)
        validate_architecture(base, arch)
        moved = copy.deepcopy(base)
        sym = symmetry_cls().sample(moved, arch, seed)
        sym.apply(moved, arch)
        results[label] = logit_difference(base, moved, ids)
        del base, moved
    name = symmetry_cls().name
    return EquivalenceResult(
        name=name, continuous=symmetry_cls().continuous,
        f32=results["f32"], f64=results["f64"],
        f32_diagnostic=F32_DIAGNOSTIC[name], tol_f64=TOL_F64[name])


# ---------------------------------------------------------------------------
# The zero-gradient measurement
#
# If the loss is exactly invariant along a direction then its gradient along
# that direction is exactly zero, so Adam's second moment there is zero and the
# coordinate never updates: it sits at its initialization value for the whole
# run. Same-seed twins share an initialization, so their gauge coordinates are
# identical and cancel in the difference -- meaning quotienting those
# directions out buys this study nothing.
#
# The only leak is float: gradients are not exactly zero in float32. This
# function measures the leak instead of asserting it away.
# ---------------------------------------------------------------------------


def gauge_gradient_alignment(build_model, arch: ArchSpec, symmetry_cls,
                             seed: int, ids=None, eps: float = 1e-4) -> dict:
    """Cosine between the loss gradient and a symmetry's gauge direction.

    The gauge direction is taken as the finite-difference tangent of the
    symmetry at the identity: apply the symmetry scaled down by `eps`, subtract,
    normalise. That works uniformly across symmetries with very different
    parameterisations, which an analytic tangent would not.

    A continuous symmetry should give a cosine at float-noise level. A discrete
    one has no tangent direction and returns None.
    """
    torch = _torch()
    ids = probe_tokens(arch) if ids is None else ids

    proto = symmetry_cls()
    if not proto.continuous:
        return {"name": proto.name, "continuous": False, "cosine": None,
                "note": "discrete symmetry: no tangent direction exists, so "
                        "the zero-gradient argument does not apply"}

    model = build_model().to(dtype=torch.float64)
    validate_architecture(model, arch)
    names = [n for n, _ in model.named_parameters()]

    # Direction: where the symmetry moves the weights, for a small draw.
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    moved = copy.deepcopy(model)
    sym = symmetry_cls().sample(moved, arch, seed)
    _shrink_toward_identity(sym, eps)
    sym.apply(moved, arch)
    after = {n: p.detach().clone() for n, p in moved.named_parameters()}
    direction = {n: (after[n] - before[n]) for n in names}
    dnorm = math.sqrt(sum(float(v.double().pow(2).sum()) for v in direction.values()))
    if dnorm == 0.0:
        raise CanonicalizeError(
            f"{proto.name}: the shrunken symmetry moved no weights at all, so "
            "no gauge direction can be formed. Check _shrink_toward_identity.")

    # Gradient of the ordinary next-token loss at the untouched model.
    input_ids = torch.tensor([list(ids)], dtype=torch.long)
    logits = model(input_ids=input_ids).logits
    loss = torch.nn.functional.cross_entropy(
        logits[0, :-1, :].double(), input_ids[0, 1:])
    params = [p for _, p in model.named_parameters()]
    grads = torch.autograd.grad(loss, params, allow_unused=True)

    dot = 0.0
    gsq = 0.0
    for n, g, p in zip(names, grads, params):
        if g is None:
            continue
        gd = g.double()
        gsq += float(gd.pow(2).sum())
        dot += float((gd * direction[n].double()).sum())
    gnorm = math.sqrt(gsq)
    return {
        "name": proto.name, "continuous": True,
        "cosine": dot / (gnorm * dnorm) if gnorm and dnorm else None,
        "grad_norm": gnorm, "direction_norm": dnorm,
        "directional_derivative": dot,
    }


def _shrink_toward_identity(sym: Symmetry, eps: float) -> None:
    """Scale a drawn symmetry down so it sits close to doing nothing.

    Each symmetry parameterises differently -- a scale near 1, a shift near 0,
    a matrix near I, a permutation not at all -- so the shrink is written per
    parameter kind rather than generically.
    """
    torch = _torch()
    p = sym.params
    if "scales" in p:
        scales = p["scales"]
        if isinstance(scales, dict):
            for k, v in scales.items():
                scales[k] = torch.exp(torch.log(v) * eps)
        else:
            p["scales"] = [torch.exp(torch.log(v) * eps) for v in scales]
    if "shifts" in p:
        p["shifts"] = [[v * eps for v in layer] for layer in p["shifts"]]
    if "c" in p:
        p["c"] = math.exp(math.log(p["c"]) * eps)
    if "Q" in p:
        n = p["Q"].shape[0]
        eye = torch.eye(n, dtype=p["Q"].dtype)
        q, _ = torch.linalg.qr(eye + eps * (p["Q"] - eye))
        p["Q"] = q
    for key in ("A", "B"):
        if key in p:
            grid = p[key]
            out = []
            for layer in grid:
                row = []
                for m in layer:
                    eye = torch.eye(m.shape[0], dtype=m.dtype)
                    row.append(eye + eps * (m - eye))
                out.append(row)
            p[key] = out


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


TINY = ArchSpec(n_layer=2, n_head=4, n_embd=16, n_inner=32,
                n_positions=32, vocab_size=64)


def build_tiny_model(seed: int = 0):
    """A small GPT-2 built in process. No download, no committed fixture.

    THE RANDOMISATION BELOW IS LOAD-BEARING, NOT COSMETIC. It was got wrong
    twice, and each time the wrong version made a test vacuous rather than
    failing.

    A freshly constructed GPT2LMHeadModel is not a generic point in weight
    space. Its init routine leaves two whole families of tensor at structured
    values, and structure is exactly what a symmetry test hides behind:

    1. EVERY LAYERNORM GAIN IS EXACTLY 1.0 and every LayerNorm bias exactly
       0.0. Then diag(gamma) is the identity, it commutes with any rotation,
       and residual rotation passes its equivalence test on a technicality --
       measured 1.763e-07 on the default fixture against 5.456e-01 once the
       gains are randomised. Real GPT-2's gains span 2.557e-04 to 17.42, with
       0.02% within 1% of 1.0.

    2. EVERY Conv1D BIAS IS EXACTLY 0.0 -- c_attn, attn.c_proj, c_fc and
       mlp.c_proj alike. This one is worse, because it silently empties the
       head-internal step of its content: with b_Q == 0 the augmenting row of
       the Q/K affine invariant is a row of zeros, so the augmented invariant
       and the weights-only invariant are THE SAME MATRIX. The mutation fault
       that drops the bias row would have been undetectable, and the whole
       affine-invariant amendment would have been a no-op that still passed
       every test. Real GPT-2's biases reach 1.34 (c_attn), 2.68
       (attn.c_proj), 0.75 (c_fc) and 1.48 (mlp.c_proj).

    So gains are drawn log-normal, and every bias in the model is drawn normal.
    Weight matrices are left at GPT-2's own init, which is already generic.
    """
    torch = _torch()
    from transformers.models.gpt2.modeling_gpt2 import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(seed)
    cfg = GPT2Config(
        n_layer=TINY.n_layer, n_head=TINY.n_head, n_embd=TINY.n_embd,
        n_inner=TINY.n_inner, n_positions=TINY.n_positions,
        vocab_size=TINY.vocab_size,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    )
    model = GPT2LMHeadModel(cfg)
    gen = torch.Generator().manual_seed(seed + 991)
    with torch.no_grad():
        for ln in _all_layernorms(model):
            ln.weight.copy_(torch.exp(
                torch.randn(ln.weight.shape, generator=gen) * 0.5))
            ln.bias.copy_(torch.randn(ln.bias.shape, generator=gen) * 0.1)
        for block in model.transformer.h:
            for proj in (block.attn.c_attn, block.attn.c_proj,
                         block.mlp.c_fc, block.mlp.c_proj):
                proj.bias.copy_(
                    torch.randn(proj.bias.shape, generator=gen) * 0.3)
    model.eval()
    return model


def _conv1d_biases(model):
    """Every Conv1D bias in the model, in a fixed order. Used by the fixture
    genericity test, which exists because these were zero once."""
    out = []
    for block in model.transformer.h:
        for proj in (block.attn.c_attn, block.attn.c_proj,
                     block.mlp.c_fc, block.mlp.c_proj):
            out.append(proj.bias)
    return out


def _all_layernorms(model):
    """Every LayerNorm in the model, in a fixed order."""
    out = []
    for block in model.transformer.h:
        out.append(block.ln_1)
        out.append(block.ln_2)
    out.append(model.transformer.ln_f)
    return out


def build_real_gpt2():
    """Public GPT-2 124M via burst_match's loader. The only real model here."""
    import io
    if str(REPO_ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from burst_match import BurstMatchError, load_model
    try:
        _, model = load_model(stream=io.StringIO())
    except BurstMatchError as exc:
        raise CanonicalizeError(f"GPT-2 unavailable: {exc}") from exc
    model.eval()
    return model


def build_degenerate_model(seed: int = 0, gap: float = 1e-6):
    """A model whose WEIGHTS-ONLY Q/K invariant is NEARLY 2-fold degenerate.

    Built so that the bias row is the only thing that can break the tie, which
    is what makes the "drop the bias row" mutation fault detectable. Per head:

      W_Q = U diag(s)  with U orthonormal and s carrying a repeated value
      W_K = V          with V orthonormal

    so the weights-only invariant W_Q W_K^T has singular values exactly `s`,
    two of which are separated by only `gap`. Its SVD basis is therefore almost
    arbitrary inside that 2-dimensional subspace: determined to roughly
    eps/gap, so two models that differ by a hair land on different bases and
    the "canonical" form built from it is not canonical.

    NEARLY rather than exactly degenerate on purpose. At an exact tie the
    head-internal step's own spectrum guard refuses outright, which is safe but
    demonstrates nothing -- the failure this fixture has to exhibit is the
    silent one, where canonicalization proceeds and returns different answers
    for the same model.

    Augmenting with b_Q adds a rank-one term to the Gram matrix, which splits
    the pair -- so the augmented invariant IS well defined on the same model.
    That contrast is the whole point of the fixture.
    """
    torch = _torch()

    model = build_tiny_model(seed=seed)
    arch = TINY
    dh = arch.head_dim
    gen = torch.Generator().manual_seed(seed + 4801)
    # Two singular values separated by `gap`, the rest well spread.
    s = torch.ones(dh, dtype=torch.float32)
    s[0] = 2.0
    s[2] = 1.0 - gap           # -> e.g. [2.0, 1.0, 1.0-gap, 0.5] for dh == 4
    if dh >= 4:
        s[-1] = 0.5
    with torch.no_grad():
        for block in model.transformer.h:
            c_attn = block.attn.c_attn
            for h in range(arch.n_head):
                q, k = (head_columns(w, h, arch) for w in ("q", "k"))
                u = _random_orthogonal(arch.n_embd, gen, torch.float32)[:, :dh]
                v = _random_orthogonal(arch.n_embd, gen, torch.float32)[:, :dh]
                c_attn.weight[:, q] = u * s.unsqueeze(0)
                c_attn.weight[:, k] = v
                # Non-zero, and with weight on the degenerate coordinates, so
                # the rank-one update actually splits them.
                c_attn.bias[q] = torch.randn(dh, generator=gen) * 0.5 + 0.5
    return model


# ---------------------------------------------------------------------------
# Canonicalization
#
# A CanonStep is the deterministic UNSCRAMBLE. It is not the inverse of any one
# Symmetry -- it rewrites a model into the single representative of its orbit,
# whatever gauge it arrived in.
#
# RECIPE ORDER IS PART OF THE DEFINITION OF CANONICAL FORM, not a convenience.
# Gain absorption rewrites c_attn's input rows and the head-internal step reads
# c_attn. Zeroing b_K changes which invariant the head-internal step should
# form. The two sort steps compute their keys from tensors the earlier steps
# rewrite. Reordering the recipe therefore produces a different canonical form,
# and tests/test_canonicalize_recipe.py pins that by running permuted orders
# and asserting the round trip breaks.
# ---------------------------------------------------------------------------


def _paired_svd(F, G, sign_fix: bool = True):
    """Compact SVD of ``F @ G.T`` via QR plus a small core SVD.

    F is (m, d) and G is (n, d) with d the head dimension, so ``F @ G.T`` is
    (m, n) of rank at most d. Forming it explicitly would be an (m, n) matrix
    for no reason; QR-ing each factor and taking the SVD of the d-by-d core
    gives the same decomposition for a fraction of the work.

    THE SIGN CONVENTION IS LOAD-BEARING. Each singular-vector pair may be
    negated together without changing the product, so without a convention the
    "canonical" form is only canonical up to 2^d sign choices and the round
    trip fails. The convention: make the largest-magnitude entry of each column
    of U positive, ties broken to the lowest index. `sign_fix=False` exists so
    a mutation test can disable it and prove the round trip catches that.
    """
    torch = _torch()
    Qf, Rf = torch.linalg.qr(F)
    Qg, Rg = torch.linalg.qr(G)
    u, s, vh = torch.linalg.svd(Rf @ Rg.T)
    U = Qf @ u
    V = Qg @ vh.transpose(-2, -1)
    if sign_fix:
        cols = torch.arange(U.shape[1])
        pivot = U.abs().argmax(dim=0)
        signs = torch.sign(U[pivot, cols])
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        U = U * signs.unsqueeze(0)
        V = V * signs.unsqueeze(0)
    return U, s, V


def _lexicographic_margin(sorted_keys) -> float:
    """Smallest margin that actually DECIDES an adjacent comparison.

    A lexicographic sort is settled by the FIRST component where two keys
    differ, so that component's difference is what a perturbation has to exceed
    to flip the pair. Every other component is irrelevant however far apart it
    is.

    This was originally computed as the largest difference ANYWHERE in the key
    tuple, which is not a margin at all -- it reported 8.792e-04 for GPT-2's
    FFN keys whose true smallest deciding margin is 1.490e-08, a factor of
    59,000 too optimistic, and it was the number being used to argue the sort
    was safe. See S53.
    """
    worst = math.inf
    for a, b in zip(sorted_keys, sorted_keys[1:]):
        for x, y in zip(a, b):
            if x != y:
                worst = min(worst, abs(x - y))
                break
        else:
            worst = 0.0          # identical on every component: an exact tie
    return worst


def _relative_gaps(s):
    """Consecutive singular-value gaps, relative to the largest.

    The minimum of these is what says whether the canonical form is well
    defined: a near-zero gap means the SVD basis is nearly arbitrary in that
    subspace, and two models that differ by a hair can land on different bases.
    """
    if s.numel() < 2:
        return s.new_tensor([float("inf")])
    return (s[:-1] - s[1:]) / s[0]


@dataclass
class CanonStep:
    """One deterministic rewrite. Subclasses set `name` and implement run()."""

    name: str = "unnamed"

    def run(self, model, arch: ArchSpec, report: "CanonReport") -> None:
        raise NotImplementedError


@dataclass
class AbsorbLayerNormGains(CanonStep):
    """Force every ln_1 and ln_2 gain to all-ones.

    LN(x) = gamma * xhat + beta feeding a Conv1D weight W. Setting gamma to 1
    requires W's ROWS to absorb it and beta to be divided by it:

        W'    = diag(gamma) W          (rows are Conv1D's input axis)
        beta' = beta / gamma
        gamma'= 1

    ln_f is deliberately NOT absorbed. Its downstream consumer is lm_head,
    which is the tied embedding -- folding a gain into it would corrupt the
    input embedding. That is also precisely why residual rotation is not a
    symmetry of this architecture; see D17.
    """

    name: str = "absorb_layernorm_gains"
    #: A gain this close to zero cannot be divided out without destroying
    #: precision. Real GPT-2's smallest is 2.557e-04, so this floor is three
    #: orders below anything observed rather than a value tuned to pass.
    min_gain: float = 1e-7

    def run(self, model, arch, report):
        torch = _torch()
        smallest = math.inf
        with torch.no_grad():
            for i, block in enumerate(model.transformer.h):
                for ln_name, proj in (("ln_1", block.attn.c_attn),
                                      ("ln_2", block.mlp.c_fc)):
                    ln = getattr(block, ln_name)
                    gamma = ln.weight.detach().clone()
                    worst = float(gamma.abs().min())
                    smallest = min(smallest, worst)
                    if worst < self.min_gain:
                        raise CanonicalizeError(
                            f"block {i} {ln_name}: LayerNorm gain has an entry "
                            f"of magnitude {worst:.3e}, below the floor "
                            f"{self.min_gain:.0e}. Absorbing the gain divides "
                            "by it, and dividing by a value this small would "
                            "destroy the precision of the canonical form "
                            "rather than merely compute it badly.")
                    proj.weight.mul_(gamma.unsqueeze(1))
                    ln.bias.div_(gamma)
                    ln.weight.fill_(1.0)
        report.min_layernorm_gain = smallest


@dataclass
class ZeroKeyBiasGauge(CanonStep):
    """Set every attention key bias to zero.

    b_K is pure gauge: shifting it shifts every key by the same vector, which
    adds a per-query constant to that row of the score matrix, and softmax
    normalises across exactly that row. Zeroing it is the canonical
    representative of the orbit and costs nothing, because there is nothing
    there to lose.

    Runs BEFORE the head-internal step so the Q/K invariant that step forms is
    built on a gauge-fixed model. Measured across 144 heads of real GPT-2,
    removing this gauge first improves the worst-case singular-value gap from
    3.570e-05 to 1.068e-04.
    """

    name: str = "zero_key_bias_gauge"

    def run(self, model, arch, report):
        torch = _torch()
        with torch.no_grad():
            for block in model.transformer.h:
                bias = block.attn.c_attn.bias
                for h in range(arch.n_head):
                    bias[head_columns("k", h, arch)] = 0.0


@dataclass
class ZeroValueBiasGauge(CanonStep):
    """Set every attention value bias to zero, compensating in c_proj's bias.

    Attention probabilities sum to one across keys, so shifting b_V by c shifts
    that head's output by exactly c, which reaches the residual as the constant
    c @ W_O. Removing b_V therefore requires adding b_V @ W_O back into
    attn.c_proj.bias, which is where that constant belongs.

    Unlike b_K this is not free bookkeeping -- it moves information between two
    tensors -- but it is still an exact rewrite, and it is what lets the V/O
    invariant be formed from W_V and W_O alone.
    """

    name: str = "zero_value_bias_gauge"

    def run(self, model, arch, report):
        torch = _torch()
        with torch.no_grad():
            for block in model.transformer.h:
                c_attn, c_proj = block.attn.c_attn, block.attn.c_proj
                for h in range(arch.n_head):
                    v = head_columns("v", h, arch)
                    b_v = c_attn.bias[v].detach().clone()
                    w_o = c_proj.weight[head_rows_of_out_proj(h, arch), :]
                    c_proj.bias.add_(b_v @ w_o)
                    c_attn.bias[v] = 0.0


@dataclass
class CanonicalizeHeadInternal(CanonStep):
    """Fix the GL(head_dim) freedom inside every head, per head, per layer.

    Two independent freedoms, each handled by splitting its invariant
    symmetrically through an SVD:

      Q/K   the scores depend on Q and K only through the AFFINE invariant.
            With b_K gauge-fixed to zero that invariant is [W_Q ; b_Q] W_K^T,
            and the query bias rides in as an extra ROW of the Q factor.

            THE BIAS ROW IS NOT OPTIONAL, AND NOT PRIMARILY A TIE-BREAKER.
            Without it, b_Q is never transformed at all: the weights get
            canonicalized and the query bias is left in whatever gauge it
            arrived in, so the canonical form is INCOMPLETE on any model with a
            non-zero b_Q, however well conditioned. Measured on a generic
            model, dropping the row breaks the round trip by 1.114e+00. It also
            propagates -- the head sort key is this spectrum, so a stranded b_Q
            reorders the heads too.

            Improving the worst-case singular-value gap (2.6x, and 7.9x with
            the b_K gauge removed first) is a real second benefit on top of
            that, not the reason. Framing it as tie-breaking would suggest the
            row could be skipped on a well-conditioned model. It cannot.

      V/O   the head's output depends on V and its slice of c_proj only through
            W_V W_O. b_V is gauge and is removed by the previous step, so this
            invariant is formed from the WEIGHTS ONLY. That is a measured
            decision, not an argument: augmenting it with b_V was measured to
            make the worst-case gap 1.9x WORSE, because feeding a gauge
            quantity into the spectrum breaks ties with an arbitrary number.

    Canonical form: F, G -> U sqrt(Sigma), V sqrt(Sigma). Split symmetrically
    rather than pushing everything into one side, so neither factor carries all
    the scale.
    """

    name: str = "canonicalize_head_internal"
    #: Below this relative gap the SVD basis is effectively arbitrary and the
    #: "canonical" form would not be canonical. Raising is the honest response.
    min_relative_gap: float = 1e-9

    def _query_factor(self, w_q, b_q):
        """The Q-side factor. Overridden by the bias-dropping mutation fault."""
        torch = _torch()
        return torch.cat([w_q, b_q.unsqueeze(0)], dim=0)

    def _svd(self, F, G):
        """Overridden by the sign-convention mutation fault."""
        return _paired_svd(F, G, sign_fix=True)

    def run(self, model, arch, report):
        torch = _torch()
        d, dh = arch.n_embd, arch.head_dim
        worst_gap = math.inf
        worst_cond = 0.0
        per_head_cond = []
        with torch.no_grad():
            for block in model.transformer.h:
                c_attn, c_proj = block.attn.c_attn, block.attn.c_proj
                for h in range(arch.n_head):
                    q, k, v = (head_columns(w, h, arch) for w in QKV)
                    rows = head_rows_of_out_proj(h, arch)
                    w_q = c_attn.weight[:, q].detach().clone()
                    b_q = c_attn.bias[q].detach().clone()
                    w_k = c_attn.weight[:, k].detach().clone()
                    w_v = c_attn.weight[:, v].detach().clone()
                    w_o = c_proj.weight[rows, :].detach().clone()

                    # ---- Q/K, augmented with the query bias ----------------
                    F = self._query_factor(w_q, b_q)
                    U, s, V = self._svd(F, w_k)
                    self._check_spectrum(s, "Q/K")
                    worst_gap = min(worst_gap, float(_relative_gaps(s).min()))
                    cond_qk = float(s[0] / s[-1])
                    worst_cond = max(worst_cond, cond_qk)
                    per_head_cond.append(cond_qk)
                    root = s.clamp_min(0).sqrt()
                    F_new = U * root.unsqueeze(0)
                    c_attn.weight[:, q] = F_new[:d, :]
                    if F_new.shape[0] > d:
                        c_attn.bias[q] = F_new[d, :]
                    c_attn.weight[:, k] = V * root.unsqueeze(0)

                    # ---- V/O, weights only ---------------------------------
                    U2, s2, V2 = self._svd(w_v, w_o.transpose(0, 1))
                    self._check_spectrum(s2, "V/O")
                    worst_gap = min(worst_gap, float(_relative_gaps(s2).min()))
                    worst_cond = max(worst_cond, float(s2[0] / s2[-1]))
                    root2 = s2.clamp_min(0).sqrt()
                    c_attn.weight[:, v] = U2 * root2.unsqueeze(0)
                    c_proj.weight[rows, :] = (
                        V2 * root2.unsqueeze(0)).transpose(0, 1)
        report.min_singular_gap = worst_gap
        report.max_head_condition = worst_cond
        report.head_conditions = tuple(per_head_cond)

    def _check_spectrum(self, s, which: str) -> None:
        torch = _torch()
        if float(s[-1]) <= 0.0:
            raise CanonicalizeError(
                f"{which} invariant is rank deficient (smallest singular value "
                f"{float(s[-1]):.3e}). The canonical form is defined by a rank "
                "factorisation and does not exist for a deficient invariant.")
        gap = float(_relative_gaps(s).min())
        if gap < self.min_relative_gap:
            raise CanonicalizeError(
                f"{which} invariant has two singular values within {gap:.3e} "
                f"of each other (relative), below the floor "
                f"{self.min_relative_gap:.0e}. The SVD basis is effectively "
                "arbitrary in that subspace, so the 'canonical' form would not "
                "be canonical: two models differing by a hair could land on "
                "different bases and report a large spurious distance.")


@dataclass
class SortHeads(CanonStep):
    """Put the attention heads of each layer into a fixed order.

    Sorted ascending on a per-head key that is invariant under everything the
    head-internal step already fixed: the singular values of that head's two
    invariants. After canonicalization the Q-side factor has orthonormal
    columns scaled by sqrt(sigma), so its squared column norms ARE the singular
    values and no second SVD is needed to recover them.

    Runs AFTER the head-internal step, necessarily: before it, the column norms
    are whatever gauge the model happened to arrive in and sorting on them
    would order heads by an accident.
    """

    name: str = "sort_heads"

    def _apply(self, block, arch, order):
        """Move the head blocks. Overridden by the axis mutation faults."""
        torch = _torch()
        c_attn, c_proj = block.attn.c_attn, block.attn.c_proj
        new_w = c_attn.weight.detach().clone()
        new_b = c_attn.bias.detach().clone()
        new_o = c_proj.weight.detach().clone()
        for new_head, old_head in enumerate(order):
            for which in QKV:
                dst = head_columns(which, new_head, arch)
                src = head_columns(which, old_head, arch)
                new_w[:, dst] = c_attn.weight[:, src]
                new_b[dst] = c_attn.bias[src]
            new_o[head_rows_of_out_proj(new_head, arch), :] = \
                c_proj.weight[head_rows_of_out_proj(old_head, arch), :]
        c_attn.weight.copy_(new_w)
        c_attn.bias.copy_(new_b)
        c_proj.weight.copy_(new_o)

    def run(self, model, arch, report):
        torch = _torch()
        smallest_margin = math.inf
        orders = []
        with torch.no_grad():
            for block in model.transformer.h:
                keys = [self._key(block, arch, h) for h in range(arch.n_head)]
                order = sorted(range(arch.n_head), key=lambda h: keys[h])
                orders.append(tuple(order))
                smallest_margin = min(
                    smallest_margin,
                    _lexicographic_margin([keys[h] for h in order]))
                if order != list(range(arch.n_head)):
                    self._apply(block, arch, order)
        report.min_head_sort_margin = smallest_margin
        report.head_orders = tuple(orders)

    @staticmethod
    def _key(block, arch, head) -> tuple:
        """The head's two singular-value spectra, concatenated.

        THE BIAS ROW MUST BE INCLUDED IN THE Q-SIDE NORM. After the
        head-internal step the Q factor is U sqrt(Sigma) where U has
        orthonormal columns and U spans the AUGMENTED (n_embd + 1) space, so
        the squared norms of the WEIGHT rows alone come to sigma_j minus
        b_Q[j]^2 -- which is not sigma and is not even monotonic. Measured on
        the tiny fixture, weight-only gave (0.0010, 0.0099, 0.0082, 0.0057)
        against a true spectrum of (0.0509, 0.0099, 0.0082, 0.0057): the
        largest singular value looked like the smallest, because almost all of
        its mass sits in the bias row.

        The sort still round-trips either way -- any per-head quantity fixed by
        the canonical form would -- but a key that is not the spectrum cannot
        be reasoned about, and this one was documented as the spectrum.
        """
        c_attn = block.attn.c_attn
        q = head_columns("q", head, arch)
        v = head_columns("v", head, arch)
        qk = ((c_attn.weight[:, q].detach() ** 2).sum(dim=0)
              + c_attn.bias[q].detach() ** 2)
        # b_V is zeroed by an earlier step, so the V side needs no such term.
        vo = (c_attn.weight[:, v].detach() ** 2).sum(dim=0)
        return tuple(qk.tolist()) + tuple(vo.tolist())


@dataclass
class SortFFNNeurons(CanonStep):
    """Put each FFN's hidden neurons into a fixed order.

    Key is the neuron's own bias first, then its input-column norm, then its
    output-row norm. c_fc's bias survives gain absorption untouched -- that
    step rewrites the weight rows only -- so this key is stable across the
    recipe.

    c_fc's COLUMNS are the hidden neurons and c_proj's ROWS are the same
    neurons: opposite axes, because Conv1D is (in, out).
    """

    name: str = "sort_ffn_neurons"

    def run(self, model, arch, report):
        torch = _torch()
        smallest_margin = math.inf
        orders = []
        with torch.no_grad():
            for block in model.transformer.h:
                c_fc, c_proj = block.mlp.c_fc, block.mlp.c_proj
                bias = c_fc.bias.detach()
                in_norm = c_fc.weight.detach().norm(dim=0)
                out_norm = c_proj.weight.detach().norm(dim=1)
                keys = list(zip(bias.tolist(), in_norm.tolist(),
                                out_norm.tolist()))
                order = sorted(range(arch.n_inner), key=lambda j: keys[j])
                # The margin that DECIDES each comparison -- the first
                # differing component -- not the largest difference anywhere in
                # the tuple. See _lexicographic_margin and S53: the latter
                # overstated this by a factor of 59,000 on real GPT-2 and was
                # the number being used to argue the sort was safe.
                smallest_margin = min(
                    smallest_margin,
                    _lexicographic_margin([keys[j] for j in order]))
                orders.append(tuple(order))
                idx = torch.tensor(order)
                c_fc.weight.copy_(c_fc.weight[:, idx])
                c_fc.bias.copy_(c_fc.bias[idx])
                c_proj.weight.copy_(c_proj.weight[idx, :])
        report.min_ffn_sort_margin = smallest_margin
        report.ffn_orders = tuple(orders)


#: The recipe. ORDER IS PART OF THE DEFINITION -- see the module comment above.
DEFAULT_RECIPE: tuple = (
    AbsorbLayerNormGains(),
    ZeroKeyBiasGauge(),
    ZeroValueBiasGauge(),
    CanonicalizeHeadInternal(),
    SortHeads(),
    SortFFNNeurons(),
)


@dataclass
class CanonReport:
    """Diagnostics from one canonicalization. Every field is a fragility signal.

    The two sort margins and the singular gap are the quantities that predict
    whether canonicalization is stable on two nearly-identical models: a near
    tie in a sort key or a near-degenerate spectrum is what turns a hair's
    difference into a large apparent distance.
    """

    steps: tuple = ()
    min_layernorm_gain: float = math.inf
    min_singular_gap: float = math.inf
    max_head_condition: float = 0.0
    min_head_sort_margin: float = math.inf
    min_ffn_sort_margin: float = math.inf
    #: Per-layer condition number of every head's Q/K invariant, so a distance
    #: measurement can be attributed to conditioning rather than guessed at.
    head_conditions: tuple = ()
    #: The orderings the two sort steps actually chose, per layer. Recorded so
    #: that a disagreement between two canonicalizations can be COUNTED as an
    #: order flip rather than inferred from the weights afterwards.
    head_orders: tuple = ()
    ffn_orders: tuple = ()


def canonicalize(model, arch: ArchSpec = GPT2_124M, recipe: tuple = None,
                 validate: bool = True) -> CanonReport:
    """Rewrite `model` in place into its canonical form.

    Runs the tripwire first unless explicitly told not to. Returns a report of
    the fragility diagnostics; the model itself is mutated.
    """
    recipe = DEFAULT_RECIPE if recipe is None else recipe
    if validate:
        validate_architecture(model, arch)
    report = CanonReport(steps=tuple(s.name for s in recipe))
    for step in recipe:
        step.run(model, arch, report)
    return report


def canonical_state_dict(model) -> dict:
    """Name-keyed parameters, detached, in a fixed order.

    Name-keyed rather than positional on purpose: this module reasons about
    tensors by role, and a positional list would silently survive a reordering
    that a name-keyed dict catches. The ordering contract test proves the two
    views agree.
    """
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def state_dict_difference(a: dict, b: dict) -> tuple:
    """(worst absolute difference, which tensor, per-tensor dict).

    Returns the worst case AND the full breakdown, because "the round trip
    agrees" is much less useful than knowing which tensor disagrees most and
    by how much.
    """
    if set(a) != set(b):
        raise CanonicalizeError(
            f"state dicts have different keys: {set(a) ^ set(b)}")
    per_tensor = {}
    for name in a:
        if a[name].shape != b[name].shape:
            raise CanonicalizeError(
                f"{name}: shapes differ, {tuple(a[name].shape)} vs "
                f"{tuple(b[name].shape)}")
        per_tensor[name] = float((a[name].double()
                                  - b[name].double()).abs().max())
    worst_name = max(per_tensor, key=per_tensor.get)
    return per_tensor[worst_name], worst_name, per_tensor


# ---------------------------------------------------------------------------
# The frozen-axis contract
#
# Stated PER AXIS, not per tensor. The vocabulary axis of wte is the model's
# output space and may never be reordered; the same holds for wpe's position
# axis. The residual-channel axis of both is deliberately NOT frozen, so this
# contract stays correct if residual permutation is ever admitted to the recipe.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrozenAxis:
    tensor: str
    axis: int
    why: str


FROZEN_AXES: tuple = (
    FrozenAxis("transformer.wte.weight", 0,
               "the vocabulary axis IS the model's output space; reordering it "
               "would relabel which token each logit refers to"),
    FrozenAxis("transformer.wpe.weight", 0,
               "the position axis is indexed by absolute position; reordering "
               "it would move which position each embedding applies to"),
)


#: Relative slack in the frozen-axis check. A permutation of the OTHER axis
#: reorders the terms of each slice's norm, and float addition is not
#: associative, so the norm moves in its last bits -- measured at 1.9e-16
#: relative for a residual permutation of wte. A genuine reordering of the
#: frozen axis moves it by order 1. Twelve orders of separation, so this
#: tolerance distinguishes the two without being able to hide anything.
FROZEN_AXIS_RTOL = 1e-12


def assert_frozen_axes_unchanged(before: dict, after: dict) -> None:
    """Every frozen axis must keep its slices in their original order.

    Checked by comparing the sequence of per-slice norms along the frozen axis
    rather than the raw values. A norm is invariant to a permutation of the
    OTHER axis, so this passes a residual re-gauging and fails a reordering of
    the axis itself -- which is exactly the distinction the contract makes.

    Under the current recipe wte and wpe are not touched at all, so they are in
    fact byte-identical; that stronger property is asserted separately in the
    tests. This function states the weaker per-axis contract, so it stays
    correct if residual permutation is ever admitted to the recipe.
    """
    for frozen in FROZEN_AXES:
        if frozen.tensor not in before:
            raise CanonicalizeError(
                f"{frozen.tensor} is not in the state dict; the frozen-axis "
                "contract names a tensor this model does not have")
        b, a = before[frozen.tensor], after[frozen.tensor]
        other = [d for d in range(b.dim()) if d != frozen.axis]
        nb = b.double().pow(2).sum(dim=other).sqrt()
        na = a.double().pow(2).sum(dim=other).sqrt()
        scale = nb.clamp_min(1e-300)
        drift = ((nb - na).abs() / scale)
        worst = float(drift.max())
        if worst > FROZEN_AXIS_RTOL:
            bad = int(drift.argmax())
            raise CanonicalizeError(
                f"{frozen.tensor} axis {frozen.axis} changed at index {bad}: "
                f"slice norm {float(nb[bad]):.6e} became {float(na[bad]):.6e} "
                f"(relative change {worst:.3e}, tolerance "
                f"{FROZEN_AXIS_RTOL:.0e}). {frozen.why}.")


def assert_embedding_tie_preserved(model) -> None:
    """lm_head must still BE wte, not merely equal it."""
    if model.lm_head.weight is not model.transformer.wte.weight:
        raise CanonicalizeError(
            "canonicalization broke the embedding tie: lm_head.weight is no "
            "longer the same tensor object as transformer.wte.weight. The tie "
            "is what makes the output projection the transpose of the input "
            "embedding, and every symmetry conclusion in this module assumes "
            "it holds.")


# ---------------------------------------------------------------------------
# Alignment -- the other way to remove a permutation gauge
#
# Canonicalization sorts each model independently. Alignment instead matches
# one model's heads and neurons to another's, using the FULL weight vector
# rather than a scalar sort key, which is robust to the near-ties that make
# sorting fragile.
#
# Not a fallback. Every comparison in this study is pairwise -- each burst arm
# against its own seed-matched twin -- so aligning to the twin is a natural
# primitive, and the "depends on an external artifact" objection dissolves when
# the artifact is the twin. Which of the two is preferable is a question for
# the measured near-tie margins, not for argument.
# ---------------------------------------------------------------------------


@dataclass
class AlignReport:
    head_permutations: tuple = ()
    ffn_permutations: tuple = ()
    head_cost_margin: float = math.inf


def _scipy_lsa():
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise CanonicalizeError(
            "scipy is not installed, so alignment cannot solve the assignment "
            "problem. Install with: pip install -e \".[dev,measure]\"\n"
            f"(underlying import error: {exc})"
        ) from exc
    return linear_sum_assignment


def align_permutations_to(model, reference, arch: ArchSpec = GPT2_124M
                          ) -> AlignReport:
    """Permute `model`'s heads and FFN neurons to best match `reference`.

    Matching is on the whole feature vector for each head or neuron, solved
    exactly with the Hungarian algorithm rather than greedily, so a near tie
    between two heads is resolved by everything else about them instead of by
    whichever scalar happened to be larger.
    """
    torch = _torch()
    linear_sum_assignment = _scipy_lsa()
    validate_architecture(model, arch)
    validate_architecture(reference, arch)

    head_perms, ffn_perms = [], []
    margin = math.inf
    with torch.no_grad():
        for block, ref in zip(model.transformer.h, reference.transformer.h):
            feats = _head_features(block, arch)
            ref_feats = _head_features(ref, arch)
            cost = -(feats @ ref_feats.transpose(0, 1)).cpu().numpy()
            rows, cols = linear_sum_assignment(cost)
            # order[j] = which of model's heads becomes head j
            order = [0] * arch.n_head
            for r, c in zip(rows.tolist(), cols.tolist()):
                order[c] = r
            head_perms.append(tuple(order))
            chosen = float(sum(cost[r, c] for r, c in zip(rows, cols)))
            margin = min(margin, abs(chosen))
            if order != list(range(arch.n_head)):
                SortHeads()._apply(block, arch, order)

            fc, ref_fc = block.mlp, ref.mlp
            nf = _ffn_features(fc)
            ref_nf = _ffn_features(ref_fc)
            cost = -(nf @ ref_nf.transpose(0, 1)).cpu().numpy()
            rows, cols = linear_sum_assignment(cost)
            order = [0] * arch.n_inner
            for r, c in zip(rows.tolist(), cols.tolist()):
                order[c] = r
            ffn_perms.append(tuple(order))
            idx = torch.tensor(order)
            fc.c_fc.weight.copy_(fc.c_fc.weight[:, idx])
            fc.c_fc.bias.copy_(fc.c_fc.bias[idx])
            fc.c_proj.weight.copy_(fc.c_proj.weight[idx, :])

    return AlignReport(head_permutations=tuple(head_perms),
                       ffn_permutations=tuple(ffn_perms),
                       head_cost_margin=margin)


def _head_features(block, arch: ArchSpec):
    """One row per head: everything that head owns, flattened."""
    torch = _torch()
    c_attn, c_proj = block.attn.c_attn, block.attn.c_proj
    rows = []
    for h in range(arch.n_head):
        parts = []
        for which in QKV:
            sl = head_columns(which, h, arch)
            parts.append(c_attn.weight[:, sl].detach().reshape(-1))
            parts.append(c_attn.bias[sl].detach().reshape(-1))
        parts.append(
            c_proj.weight[head_rows_of_out_proj(h, arch), :].detach().reshape(-1))
        rows.append(torch.cat(parts))
    return torch.stack(rows).double()


def _ffn_features(mlp):
    """One row per hidden neuron: its input column, bias and output row."""
    torch = _torch()
    return torch.cat([
        mlp.c_fc.weight.detach().transpose(0, 1),
        mlp.c_fc.bias.detach().unsqueeze(1),
        mlp.c_proj.weight.detach(),
    ], dim=1).double()
