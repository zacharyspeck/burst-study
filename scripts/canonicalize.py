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

    THE LAYERNORM RANDOMISATION BELOW IS LOAD-BEARING, NOT COSMETIC.

    A freshly constructed GPT2LMHeadModel has every LayerNorm gain at exactly
    1.0 and every LayerNorm bias at exactly 0.0, because that is nn.LayerNorm's
    default init and GPT-2's init routine does not touch it. A model in that
    state is NOT a generic point in weight space -- diag(gamma) is the identity,
    so it commutes with everything, and residual rotation passes its
    equivalence test on a technicality.

    Measured on the untouched fixture: all gains == 1.0, all biases == 0.0.
    Measured on real GPT-2: gains span 2.557e-04 to 17.42, with 0.02% of them
    within 1% of 1.0. The default fixture is degenerate exactly where the
    rotation question is decided, and testing against it would have reported
    residual rotation as a symmetry when it is not.

    So the gains are drawn log-normal and the biases normal, putting the
    fixture at a generic point. Every other tensor is left at GPT-2's own init.
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
    model.eval()
    return model


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
