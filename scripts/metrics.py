#!/usr/bin/env python
"""Layout-independent checkpoint metrics. Step 10, first half.

Two checkpoints in, numbers out. Four metrics: the interpolation barrier, raw
L2 weight distance, per-layer activation similarity, and per-layer CKA.

WHY THESE FOUR AND NOT THE OTHERS

The natural companions to these -- permutation-aligned barrier, aligned L2, and
the RSF subspace probe -- all route through `scripts/canonicalize.py`, which is
Conv1D-only by deliberate decision, and which layout the study trains is still
undecided in this repo. Building them now means building against a model that
does not exist. They are present at the bottom of this module raising
NotImplementedError that names the blocker, because a silent omission is
indistinguishable from an oversight.

The four here need none of that. **Nothing in this module reads a weight axis.**
L2 and the barrier touch parameters only as flat vectors; CKA and activation
similarity read activations, and a residual stream is (batch, seq, n_embd)
whether the projections are `Conv1D` or `nn.Linear`. That is what makes these
the half that can be built today.

WHAT THIS MODULE CANNOT TELL YOU

- Nothing about aligned distances. If two checkpoints differ by a head
  permutation, the raw L2 here reports that permutation as if it were content.
  That is exactly what step 9 exists to remove, and it is not wired in.
- Nothing about the RSF subspace.
- Nothing measured on a trained model. There are no trained checkpoints and
  none for weeks, so every number this module has produced so far came from
  junk weights and is a plumbing result, not a scientific one.

THE TRAP THIS MODULE IS SHAPED AROUND

A metric that runs on junk and returns a plausible float looks exactly like a
metric that works. Every function here is paired with a responsiveness test in
`tests/test_metrics.py`: two identical models must score identically, and two
deliberately different ones must not, by a stated margin. A metric that cannot
tell a model from a copy of itself with one block zeroed is not measuring
anything, however well-formed its output.

THE MODEL INTERFACE, AND HOW TIGHT EACH COUPLING IS

  l2_distance_raw          named_parameters() only. No architecture knowledge.
  barrier                  state_dict/load_state_dict, a forward pass, and a
                           deepcopy-able model. No architecture knowledge.
  layer activations, CKA   an ORDERED LIST OF PER-LAYER ACTIVATIONS. This is
                           the one real coupling. It is not a layout coupling
                           -- it is a structural one: something has to say
                           which modules count as layers.

The tap list is passed explicitly rather than guessed from module names.
Guessing is the S49/S53 failure shape: a quantity named for one thing and
computed as another, plausible enough that nothing crashes.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The CKA form implemented here, named in every output. An unnamed CKA is
#: incomparable to published work and to itself across versions, because the
#: kernel, the estimator and the sample axis are all free choices.
#:
#:   linear   -- no bandwidth hyperparameter to leave unrecorded, and no worse
#:               than RBF for layer comparison (Kornblith et al. 2019).
#:   unbiased -- HSIC_1 (Song et al. 2012). The BIASED estimator's value
#:               depends on the sample count, so a number taken at one batch
#:               size is not comparable to one taken at another. Unbiased
#:               removes that dependence, at the cost of being able to land
#:               slightly outside [0, 1]. Values are reported UNCLIPPED; see
#:               linear_cka_unbiased.
#:   tokens   -- token positions are the sample axis. One committed sequence
#:               of 1024 tokens gives n = 1024 samples of d = n_embd features.
CKA_VARIANT = "linear_cka_unbiased_hsic_tokens_as_samples_v1"

#: Interpolation grid for the barrier. Endpoints exact.
ALPHA_GRID = tuple(round(i * 0.05, 4) for i in range(21))

#: The committed sequence every activation measurement runs on.
CONTEXT_FILE = REPO_ROOT / "bursts" / "context.txt"
BURST_PROVENANCE = REPO_ROOT / "bursts" / "provenance.json"

_BLOCKED = (
    "NOT BUILT. This function does not exist yet; only this message does.\n"
    "{what} would require scripts/canonicalize.py. That module is Conv1D-only: "
    "its tripwire refuses any model whose c_attn, c_proj or c_fc is not a "
    "transformers Conv1D -- naming torch.nn.Linear specifically -- rather than "
    "silently addressing the wrong axis.\n"
    "THIS IS NOT WAITING ON A MODEL SWAP. scripts/model_seam.py builds a real "
    "GPT2LMHeadModel today; FAMILIES is ('hf_gpt2', 'probe_linear') and "
    "--family is required with no default in both train.py and launch.py. A "
    "checkpoint trained under hf_gpt2 has Conv1D projections and so passes "
    "that layout check; one trained under probe_linear is nn.Linear and is "
    "refused by it. What is still nn.Linear is probes/determinism/model.py -- "
    "the probe, not the study's model.\n"
    "These three functions were SKIPPED on 2026-08-03, not blocked: see D-6 in "
    "docs/decisions-pending.md. They are work nobody has done, not work "
    "something prevents.\n"
    "See docs/layout-cost.md for what an nn.Linear adapter would cost."
)


class MetricsError(Exception):
    """Raised for every refusal in this module. Never an assert.

    Same reasoning as burst/config.py: `python -O` deletes asserts, and
    validation that vanishes under an optimizer flag is worse than none.
    """


def _torch():
    """Import torch, or raise MetricsError explaining the install."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise MetricsError(
            "PyTorch is not installed, and no metric here can run without it.\n"
            "torch is an OPTIONAL dependency group on purpose. Install into "
            "the separate environment:\n"
            "    pip install -e \".[dev,measure]\"\n"
            f"(underlying import error: {exc})"
        ) from exc
    return torch


# ---------------------------------------------------------------------------
# The fixed batch
#
# A CKA number without the input that produced it is not a measurement, so the
# batch carries its own identity and that identity goes into every report.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Batch:
    """A fixed, committed token sequence, carrying enough to identify itself."""

    ids: tuple
    source: str
    file_sha256: str
    #: sha256 over the TOKEN IDS, not the text. This is the guard that makes
    #: re-deriving safer than committing the ids: if a tokenizer version ever
    #: shifts, the text hash is unchanged and this one moves, so the drift is
    #: loud instead of silent. Committing the ids would make the same shift
    #: invisible, because nothing would ever re-tokenize.
    token_sha256: str
    tokenizer: str

    @property
    def n_tokens(self) -> int:
        return len(self.ids)

    def identity(self) -> dict:
        return {
            "source": self.source,
            "file_sha256": self.file_sha256,
            "token_sha256": self.token_sha256,
            "n_tokens": self.n_tokens,
            "tokenizer": self.tokenizer,
        }

    def input_ids(self):
        torch = _torch()
        return torch.tensor([list(self.ids)], dtype=torch.long)


def load_context_batch(path: Path = CONTEXT_FILE, stream=None) -> Batch:
    """Tokenize the committed context passage, checking it against provenance.

    `bursts/context.txt` is already a pinned, human-reviewed 1024-token
    sequence with its sha256 recorded in `bursts/provenance.json`. Reusing it
    means the measurement input is committed content that the study already
    stands behind, rather than a batch invented per run -- which would make
    every number irreproducible across machines.
    """
    import io

    stream = io.StringIO() if stream is None else stream
    raw = path.read_bytes()
    file_sha = hashlib.sha256(raw).hexdigest()

    if str(REPO_ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        from burst_match import BurstMatchError, load_tokenizer
    except ImportError as exc:
        raise MetricsError(f"cannot import the tokenizer loader: {exc}") from exc
    try:
        tokenizer = load_tokenizer(stream=stream)
    except BurstMatchError as exc:
        raise MetricsError(f"GPT-2 tokenizer unavailable: {exc}") from exc

    ids = tuple(tokenizer(raw.decode("utf-8"))["input_ids"])
    batch = Batch(
        ids=ids,
        source=str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
        file_sha256=file_sha,
        token_sha256=hashlib.sha256(
            json.dumps(list(ids)).encode("utf-8")).hexdigest(),
        tokenizer="gpt2",
    )
    _check_against_provenance(batch)
    return batch


def tiny_smoke_batch(arch=None) -> Batch:
    """A batch the TINY smoke model can actually run. Not a measurement input.

    The real batch is 1024 GPT-2 tokens, and the tiny model has a 64-token
    vocabulary and 32 positions, so feeding it the real batch is an index
    error rather than a small inaccuracy. The batch is part of the
    measurement and is tied to the model's vocabulary.

    Derived deterministically from the committed passage's bytes so it is still
    reproducible, and its `source` says out loud what it is: a report generated
    from this batch must not be mistakable for one generated from the real one.
    """
    C = _canonicalize_module()
    arch = arch or C.TINY
    raw = CONTEXT_FILE.read_bytes()
    n = min(arch.n_positions, 24)
    ids = tuple(raw[i] % arch.vocab_size for i in range(n))
    return Batch(
        ids=ids,
        source="SMOKE TEST ONLY -- derived from bursts/context.txt bytes, "
               "NOT the measurement batch",
        file_sha256=hashlib.sha256(raw).hexdigest(),
        token_sha256=hashlib.sha256(
            json.dumps(list(ids)).encode("utf-8")).hexdigest(),
        tokenizer="none (byte-derived smoke batch)",
    )


def _check_against_provenance(batch: Batch) -> None:
    """Fail loudly if the committed record and the live tokenization disagree.

    Two independent facts are on record in bursts/provenance.json -- the file's
    sha256 and its token count -- and both are re-derived here. A tokenizer
    version bump changes the count while leaving the file hash alone, which is
    precisely the drift that would otherwise move every CKA number silently.
    """
    if not BURST_PROVENANCE.is_file():
        return
    record = json.loads(BURST_PROVENANCE.read_text(encoding="utf-8"))
    ctx = record.get("context", {})
    expected_sha = ctx.get("sha256")
    expected_n = ctx.get("tokens")

    if expected_sha and expected_sha != batch.file_sha256:
        raise MetricsError(
            f"{batch.source} does not match bursts/provenance.json.\n"
            f"  recorded sha256: {expected_sha}\n"
            f"  actual sha256:   {batch.file_sha256}\n"
            "The committed measurement input has changed. Every activation "
            "number taken against the old text is incomparable to one taken "
            "against this.")
    if expected_n and expected_n != batch.n_tokens:
        raise MetricsError(
            f"{batch.source} tokenizes to {batch.n_tokens} tokens; "
            f"bursts/provenance.json records {expected_n}.\n"
            "The file is unchanged, so this is TOKENIZER DRIFT, not an edit. "
            "Re-deriving the ids is what makes this visible -- pinning them "
            "would have hidden it. Do not adjust the expectation to match; "
            "find out which tokenizer version produced which count.")


# ---------------------------------------------------------------------------
# L2, raw
#
# The narrowest metric here: named_parameters() and nothing else.
# ---------------------------------------------------------------------------


def parameter_view(model) -> dict:
    """Name-keyed parameters, detached. Tied weights appear ONCE.

    `named_parameters()` deduplicates by identity, so a tied `lm_head.weight`
    is not counted a second time alongside `transformer.wte.weight`.
    `state_dict()` does NOT deduplicate, and using it here would silently
    double-weight the embedding -- 38M of GPT-2's 124M parameters -- in every
    distance this module reports.
    """
    return {n: p.detach() for n, p in model.named_parameters()}


def l2_distance_raw(a, b) -> float:
    """Plain L2 over the flat parameter vector, accumulated in float64.

    RAW means raw: no permutation alignment, no gauge removal. If two
    checkpoints differ only by a relabelling of heads or FFN neurons, this
    reports that relabelling at full size as though it were content. Removing
    it is step 9's job and is NOT wired in here -- see aligned_l2 below.
    """
    torch = _torch()
    pa = a if isinstance(a, dict) else parameter_view(a)
    pb = b if isinstance(b, dict) else parameter_view(b)
    if set(pa) != set(pb):
        raise MetricsError(
            f"parameter names differ; symmetric difference: "
            f"{sorted(set(pa) ^ set(pb))[:8]}")
    total = 0.0
    for name in sorted(pa):
        if pa[name].shape != pb[name].shape:
            raise MetricsError(
                f"{name}: shapes differ, {tuple(pa[name].shape)} vs "
                f"{tuple(pb[name].shape)}")
        diff = pa[name].detach().double() - pb[name].detach().double()
        total += float(torch.sum(diff * diff))
    return total ** 0.5


# ---------------------------------------------------------------------------
# The barrier
# ---------------------------------------------------------------------------


def interpolate_state_dicts(sd_a: dict, sd_b: dict, alpha: float) -> dict:
    """(1 - alpha) * A + alpha * B, over FLOATING-POINT tensors only.

    THE TRAP THIS FUNCTION EXISTS TO AVOID. A GPT-2 state_dict carries
    non-float buffers -- causal attention masks among them. Interpolating a
    boolean mask at alpha=0.5 produces a tensor that is still a valid mask
    shape, still runs, and yields a perfectly plausible loss curve that means
    nothing at all. So non-float entries are not interpolated: they are
    asserted identical between the two checkpoints and copied through. If they
    are NOT identical the two models are not comparable and this refuses.
    """
    torch = _torch()
    if set(sd_a) != set(sd_b):
        raise MetricsError(
            f"state dict keys differ; symmetric difference: "
            f"{sorted(set(sd_a) ^ set(sd_b))[:8]}")
    if not 0.0 <= alpha <= 1.0:
        raise MetricsError(f"alpha must be in [0, 1]; got {alpha}")

    out = {}
    for name, ta in sd_a.items():
        tb = sd_b[name]
        if ta.shape != tb.shape:
            raise MetricsError(
                f"{name}: shapes differ, {tuple(ta.shape)} vs {tuple(tb.shape)}")
        if ta.is_floating_point():
            out[name] = (1.0 - alpha) * ta.detach().clone() + alpha * tb.detach()
        else:
            if not torch.equal(ta, tb):
                raise MetricsError(
                    f"{name} is non-floating-point ({ta.dtype}) and DIFFERS "
                    "between the two checkpoints. It cannot be interpolated -- "
                    "a blended mask or index buffer is not a valid model -- and "
                    "it cannot be passed through, because the two models "
                    "disagree about it. These checkpoints are not comparable.")
            out[name] = ta.detach().clone()
    return out


def cross_entropy_loss(model, batch: Batch) -> float:
    """Next-token cross-entropy on the fixed batch. 1024 tokens, 1023 targets.

    Deterministic: no sampling, no dropout (eval mode), one fixed input.
    """
    torch = _torch()
    import torch.nn.functional as F

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            ids = batch.input_ids()
            logits = model(input_ids=ids).logits
            shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
            shift_labels = ids[:, 1:].reshape(-1)
            return float(F.cross_entropy(shift_logits.double(), shift_labels))
    finally:
        if was_training:
            model.train()


def barrier(model_a, model_b, batch: Batch, alphas=ALPHA_GRID,
            scratch=None) -> dict:
    """Loss along the linear interpolation, as maximum excess over the chord.

    excess(alpha) = L(alpha) - [(1 - alpha) * L(0) + alpha * L(1)]

    The chord is the straight line between the two ENDPOINT LOSSES, so a pair
    of checkpoints with different losses is handled correctly: the barrier is
    what rises above the interpolation of the endpoints, not above their
    minimum.

    THE REPORTED BARRIER IS A LOWER BOUND. The maximum is taken over the
    sampled grid, and a grid of finite resolution can step straight over a
    narrow peak. The grid is recorded in the result so the bound is legible
    rather than implied.

    The full curve is retained. A single maximum tells you a barrier exists; the
    curve tells you whether it is a bump, a cliff or a slope.
    """
    import copy

    alphas = tuple(float(a) for a in alphas)
    for required in (0.0, 1.0):
        if required not in alphas:
            raise MetricsError(
                f"the alpha grid must contain the endpoint {required} exactly, "
                f"or the chord is extrapolated rather than measured; got "
                f"{alphas[:4]}...")

    sd_a = {k: v.detach().clone() for k, v in model_a.state_dict().items()}
    sd_b = {k: v.detach() for k, v in model_b.state_dict().items()}
    work = copy.deepcopy(model_a) if scratch is None else scratch

    losses = []
    for alpha in alphas:
        work.load_state_dict(interpolate_state_dicts(sd_a, sd_b, alpha))
        _check_tie_survived(model_a, work)
        losses.append(cross_entropy_loss(work, batch))

    result = barrier_from_losses(alphas, losses)
    result["batch"] = batch.identity()
    return result


def barrier_from_losses(alphas, losses) -> dict:
    """The barrier arithmetic, separated from evaluating any model.

    Split out because the arithmetic is the part that can be tested against a
    known answer. On junk checkpoints the loss sits at chance along the whole
    interpolation, so no pair of throwaway models produces a barrier to detect
    -- which means a responsiveness test built on junk models could only ever
    show that max_excess is zero. Feeding this function a curve with a peak
    already in it is the test that the peak would be FOUND.
    """
    alphas = tuple(float(a) for a in alphas)
    losses = [float(v) for v in losses]
    if len(alphas) != len(losses):
        raise MetricsError(
            f"{len(alphas)} alphas against {len(losses)} losses")

    l0 = losses[alphas.index(0.0)]
    l1 = losses[alphas.index(1.0)]
    chord = [(1.0 - a) * l0 + a * l1 for a in alphas]
    excess = [loss - c for loss, c in zip(losses, chord)]
    peak = max(range(len(alphas)), key=lambda i: excess[i])
    trough = min(range(len(alphas)), key=lambda i: excess[i])

    interior = [i for i, a in enumerate(alphas) if 0.0 < a < 1.0]
    rose = any(excess[i] > 0.0 for i in interior)

    return {
        "max_excess": excess[peak],
        "argmax_alpha": alphas[peak],
        # Reported because a curve that dips BELOW the chord is a real and
        # informative outcome, and max_excess alone renders it as a flat zero
        # that reads like a broken metric. On junk checkpoints this is the
        # normal case: the loss is at chance throughout, so there is nothing
        # to climb over and the interpolation sags instead.
        "min_excess": excess[trough],
        "argmin_alpha": alphas[trough],
        "rose_above_chord": rose,
        "endpoint_losses": [l0, l1],
        "alphas": list(alphas),
        "losses": losses,
        "chord": chord,
        "excess": excess,
        "grid_points": len(alphas),
        "grid_step": round(alphas[1] - alphas[0], 6) if len(alphas) > 1 else None,
        "GRID_LIMITED": (
            "max_excess is a LOWER BOUND on the true barrier: it is the maximum "
            f"over {len(alphas)} sampled alphas, and a grid of this resolution "
            "can step over a narrower peak."),
        "NO_BARRIER_IS_NOT_A_FAILURE": (
            "max_excess is 0.0 and the maximum is attained only at an "
            "endpoint, where excess is 0 by construction: the curve never rose "
            "above the chord. Read min_excess for how far it sagged below."
            if not rose else
            "the curve rose above the chord at at least one interior alpha"),
    }


def _check_tie_survived(reference, loaded) -> None:
    """A tied embedding must still be tied after load_state_dict.

    GPT-2 ties lm_head.weight to transformer.wte.weight. state_dict contains
    both names, so a load writes the same storage twice -- harmless -- but if
    the tie were ever broken the model would carry an independent output head
    and the barrier would be measuring a different architecture at every
    interior alpha. Checked rather than assumed.
    """
    try:
        ref_tied = (reference.lm_head.weight.data_ptr()
                    == reference.transformer.wte.weight.data_ptr())
        now_tied = (loaded.lm_head.weight.data_ptr()
                    == loaded.transformer.wte.weight.data_ptr())
    except AttributeError:
        return
    if ref_tied and not now_tied:
        raise MetricsError(
            "the reference model has tied embeddings but the interpolated "
            "model does not: load_state_dict broke the tie, so every interior "
            "alpha would be measuring a model with an independent output head")


# ---------------------------------------------------------------------------
# Layer activations -- the one real coupling to model structure
# ---------------------------------------------------------------------------


def hf_gpt2_tap_modules(model) -> list:
    """The modules whose outputs reproduce HF's `output_hidden_states` EXACTLY.

    VERIFIED, NOT REMEMBERED. `output_hidden_states=True` returns n_layer + 1
    tensors, and the last one is NOT the last block's output -- it is `ln_f`
    applied to it:

        hidden_states = [drop, h[0], h[1], ..., h[n-2], ln_f(h[n-1])]

    So the naive tap list -- every transformer block -- disagrees with the
    native route in its final slot, comparing a raw block output against a
    normalized one. It would not crash. It would return plausible numbers.
    That is why the tap list is built to match the native route rather than to
    match intuition, and why cross_check_activation_routes exists.
    """
    try:
        transformer = model.transformer
        blocks = list(transformer.h)
        return ([transformer.drop] + blocks[:-1] + [transformer.ln_f])
    except AttributeError as exc:
        raise MetricsError(
            "this model does not expose the HF GPT-2 module structure "
            "(.transformer.drop / .transformer.h / .transformer.ln_f) that "
            "this tap helper reads.\n"
            "Activation metrics need an ORDERED LIST OF LAYER OUTPUTS, and "
            "this repo does not guess which modules those are -- a wrong "
            "guess produces a plausible number rather than an error. Pass the "
            "modules explicitly to activations_by_hooks instead.\n"
            f"(underlying attribute error: {exc})") from exc


def activations_by_hooks(model, batch: Batch, modules) -> list:
    """Per-layer activations, captured by forward hook on explicit modules.

    The module list is passed IN. This function never infers it from names.
    """
    torch = _torch()
    if not modules:
        raise MetricsError("no tap modules given; nothing to measure")

    caught: dict = {}

    def make_hook(index):
        def hook(_mod, _inp, output):
            caught[index] = (output[0] if isinstance(output, tuple)
                             else output).detach()
        return hook

    handles = [m.register_forward_hook(make_hook(i))
               for i, m in enumerate(modules)]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(input_ids=batch.input_ids())
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()

    missing = [i for i in range(len(modules)) if i not in caught]
    if missing:
        raise MetricsError(
            f"tap modules at positions {missing} never fired during the "
            "forward pass, so their activations are missing rather than "
            "wrong. A metric computed over the remaining layers would be "
            "silently narrower than its label claims.")
    return [caught[i] for i in range(len(modules))]


def activations_native(model, batch: Batch) -> list:
    """Per-layer activations via HF's own `output_hidden_states=True`."""
    torch = _torch()
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out = model(input_ids=batch.input_ids(), output_hidden_states=True)
    finally:
        if was_training:
            model.train()
    hidden = getattr(out, "hidden_states", None)
    if hidden is None:
        raise MetricsError(
            "this model's forward does not return hidden_states, so the "
            "native activation route is unavailable for it")
    return [h.detach() for h in hidden]


def cross_check_activation_routes(model, batch: Batch, atol: float = 0.0,
                                  rtol: float = 0.0) -> dict:
    """The two routes to layer activations must agree. FIRST-CLASS REQUIREMENT.

    This is the S55 independent-recomputation pattern applied BEFORE a bug
    exists rather than after one was found. Three times in this build a
    quantity was named for one thing and computed as another, and all three
    were caught by measuring something adjacent and being surprised. Here the
    adjacent measurement is built in from the start.

    Hooks and `output_hidden_states` reach the same tensors by genuinely
    different routes: one intercepts module outputs during the forward, the
    other reads what the model chose to accumulate. If they disagree, the tap
    list is not on the modules it is believed to be on -- and every CKA and
    cosine number downstream is about layers other than the ones it names.

    ON DISAGREEMENT THIS RAISES. It does not pick a winner. Which route is
    right is not knowable from the disagreement itself.
    """
    torch = _torch()
    hooked = activations_by_hooks(model, batch, hf_gpt2_tap_modules(model))
    native = activations_native(model, batch)

    if len(hooked) != len(native):
        raise MetricsError(
            f"activation routes disagree on LAYER COUNT: hooks produced "
            f"{len(hooked)}, output_hidden_states produced {len(native)}. The "
            "tap list does not describe the same layers the model reports.")

    worst = 0.0
    worst_layer = None
    for i, (h, n) in enumerate(zip(hooked, native)):
        if h.shape != n.shape:
            raise MetricsError(
                f"activation routes disagree on SHAPE at layer {i}: "
                f"{tuple(h.shape)} vs {tuple(n.shape)}")
        gap = float((h.double() - n.double()).abs().max())
        if gap > worst:
            worst, worst_layer = gap, i

    scale = max(float(n.double().abs().max()) for n in native) or 1.0
    tolerated = atol + rtol * scale
    if worst > tolerated:
        raise MetricsError(
            f"ACTIVATION ROUTES DISAGREE. Worst gap {worst:.6e} at layer "
            f"{worst_layer}, tolerance {tolerated:.6e}.\n"
            "Hooks and output_hidden_states should reach identical tensors. "
            "They do not, so the tap list is not on the modules it is believed "
            "to be on, and every CKA and activation number downstream would be "
            "about different layers than it claims.\n"
            "This does not pick a winner: which route is correct cannot be "
            "determined from the disagreement. Stop and find out.")
    return {
        "n_layers": len(native),
        "worst_abs_gap": worst,
        "worst_layer": worst_layer,
        "tolerance": tolerated,
        "agree": True,
    }


def layer_activations(model, batch: Batch, route: str = "native") -> list:
    """Per-layer activations by the named route. Default native.

    Both routes are kept because their agreement is the check; a single route
    could not be verified against anything.
    """
    if route == "native":
        return activations_native(model, batch)
    if route == "hooks":
        return activations_by_hooks(model, batch, hf_gpt2_tap_modules(model))
    raise MetricsError(f"unknown activation route {route!r}; "
                       "expected 'native' or 'hooks'")


def _as_samples(activation):
    """(batch, seq, d) -> (batch * seq, d). Token positions become samples."""
    if activation.dim() == 3:
        return activation.reshape(-1, activation.shape[-1])
    if activation.dim() == 2:
        return activation
    raise MetricsError(
        f"expected a (batch, seq, d) or (n, d) activation; got shape "
        f"{tuple(activation.shape)}")


# ---------------------------------------------------------------------------
# CKA
# ---------------------------------------------------------------------------


def _hsic1(K, L) -> float:
    """Unbiased HSIC (Song et al. 2012, eq. 5), on Gram matrices.

        HSIC_1 = 1/(n(n-3)) [ tr(K~L~)
                              + (1'K~1)(1'L~1)/((n-1)(n-2))
                              - 2/(n-2) * 1'K~L~1 ]

    where K~ is K with its diagonal zeroed. The centering is INSIDE this
    estimator: the features must NOT also be column-centered beforehand, or
    the data is centered twice and the result is quietly wrong rather than
    obviously broken. tests/test_metrics.py pins this against a direct
    transcription of the definition over explicit index sums.
    """
    torch = _torch()
    n = K.shape[0]
    if n < 4:
        raise MetricsError(
            f"unbiased HSIC needs at least 4 samples (the estimator divides "
            f"by n - 3); got {n}")
    Kt = K.clone().fill_diagonal_(0).double()
    Lt = L.clone().fill_diagonal_(0).double()
    term_trace = float(torch.sum(Kt * Lt))
    sum_k = float(torch.sum(Kt))
    sum_l = float(torch.sum(Lt))
    term_prod = sum_k * sum_l / ((n - 1) * (n - 2))
    term_cross = 2.0 * float(torch.sum(Kt.sum(dim=0) * Lt.sum(dim=0))) / (n - 2)
    return (term_trace + term_prod - term_cross) / (n * (n - 3))


def linear_cka_unbiased(X, Y) -> float:
    """Linear CKA with the unbiased HSIC estimator. UNCLIPPED.

    X and Y are (n_samples, d). The two may have different feature widths; only
    the sample count has to match.

    THE RETURN VALUE IS NOT CLAMPED TO [0, 1]. The unbiased estimator can
    produce a slightly negative HSIC when the true dependence is near zero, and
    the resulting CKA can land just outside the unit interval. Clipping it to
    1.0 would turn a diagnostic into exactly the thing this repo keeps getting
    caught by: a plausible float that looks like a working metric. An
    out-of-range value is information -- it says the dependence is at or below
    the estimator's noise floor -- and callers are expected to report it as
    measured and say why.
    """
    torch = _torch()
    Xs, Ys = _as_samples(X).double(), _as_samples(Y).double()
    if Xs.shape[0] != Ys.shape[0]:
        raise MetricsError(
            f"CKA needs the same number of samples on both sides; got "
            f"{Xs.shape[0]} and {Ys.shape[0]}")
    K = Xs @ Xs.T
    L = Ys @ Ys.T
    hsic_xy = _hsic1(K, L)
    hsic_xx = _hsic1(K, K)
    hsic_yy = _hsic1(L, L)
    denom = hsic_xx * hsic_yy
    if denom <= 0.0:
        raise MetricsError(
            f"CKA denominator is not positive (HSIC(X,X)={hsic_xx:.6e}, "
            f"HSIC(Y,Y)={hsic_yy:.6e}). One side carries no measurable "
            "self-dependence -- a constant or near-constant activation -- so a "
            "similarity ratio against it is undefined rather than zero.")
    return hsic_xy / (denom ** 0.5)


def per_layer_cka(acts_a, acts_b) -> list:
    """CKA layer by layer, same index against same index."""
    if len(acts_a) != len(acts_b):
        raise MetricsError(
            f"layer counts differ: {len(acts_a)} vs {len(acts_b)}")
    out = []
    for i, (a, b) in enumerate(zip(acts_a, acts_b)):
        value = linear_cka_unbiased(a, b)
        out.append({
            "layer": i,
            "cka": value,
            "outside_unit_interval": not (0.0 <= value <= 1.0),
        })
    return out


# ---------------------------------------------------------------------------
# Activation similarity
# ---------------------------------------------------------------------------


def activation_cosine(a, b) -> dict:
    """Cosine between the two models' activation vectors, token by token.

    WHAT IS COMPARED: for each token position t, the angle between the two
    models' d-dimensional activation vectors at that position, in the RAW
    BASIS. Aggregated over positions as min / median / max.

    WHAT IT IS BLIND TO, and this is the point of reporting it beside CKA:

    - MAGNITUDE. Cosine ignores vector length entirely, so a model whose
      activations point correctly and are twice as long scores 1.0. The norm
      ratio is returned alongside for exactly that reason; reading the cosine
      without it would hide a real difference.
    - Anything outside these token positions. Both metrics see only the fixed
      batch.

    WHAT IT IS NOT BLIND TO, unlike CKA: rotation of the representation. CKA is
    invariant to an orthogonal change of basis; this is not. That is why both
    are reported. A pair differing by a hidden-internal rotation scores ~1.0 on
    CKA and poorly here, and the gap between the two is the signal.
    """
    torch = _torch()
    As, Bs = _as_samples(a).double(), _as_samples(b).double()
    if As.shape != Bs.shape:
        raise MetricsError(
            f"activation shapes differ: {tuple(As.shape)} vs {tuple(Bs.shape)}")
    na = torch.linalg.norm(As, dim=1)
    nb = torch.linalg.norm(Bs, dim=1)
    dead = int(torch.sum((na == 0) | (nb == 0)))
    if dead:
        raise MetricsError(
            f"{dead} token positions have a zero activation vector on one "
            "side, and a cosine against the zero vector is undefined rather "
            "than zero")
    cos = torch.sum(As * Bs, dim=1) / (na * nb)
    values = [float(v) for v in cos]
    ratio = [float(v) for v in (na / nb)]
    return {
        "cosine_min": min(values),
        "cosine_median": statistics.median(values),
        "cosine_max": max(values),
        "norm_ratio_median": statistics.median(ratio),
        "n_positions": len(values),
    }


def per_layer_activation_similarity(acts_a, acts_b) -> list:
    if len(acts_a) != len(acts_b):
        raise MetricsError(
            f"layer counts differ: {len(acts_a)} vs {len(acts_b)}")
    return [dict(layer=i, **activation_cosine(a, b))
            for i, (a, b) in enumerate(zip(acts_a, acts_b))]


# ---------------------------------------------------------------------------
# Junk checkpoints -- FIXTURES, not metrics
#
# No trained checkpoint exists and none will for weeks. These build throwaway
# models in process so the metrics can be exercised and, more importantly, so
# each one can be shown to RESPOND. Nothing here is committed: per CLAUDE.md
# rule 3 a checkpoint never enters git, and these never touch disk at all.
#
# `canonicalize` is imported lazily INSIDE these functions. It is a fixture
# dependency only -- no metric above imports it, which is what keeps this half
# of step 10 free of the Conv1D layout question.
# ---------------------------------------------------------------------------


def _canonicalize_module():
    if str(REPO_ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import canonicalize
    return canonicalize


def build_junk_model(seed: int = 0, tiny: bool = True):
    """A throwaway model at generic weights.

    Reuses `canonicalize.build_tiny_model`'s randomization rather than
    re-deriving it, because that routine encodes a lesson that was learned
    twice the hard way: a freshly constructed GPT2LMHeadModel leaves every
    LayerNorm gain at exactly 1.0 and every Conv1D bias at exactly 0.0, and
    structure is what a vacuous test hides behind.
    """
    C = _canonicalize_module()
    if tiny:
        return C.build_tiny_model(seed=seed)
    torch = _torch()
    from transformers.models.gpt2.modeling_gpt2 import GPT2Config, GPT2LMHeadModel

    arch = C.GPT2_124M
    torch.manual_seed(seed)
    cfg = GPT2Config(
        n_layer=arch.n_layer, n_head=arch.n_head, n_embd=arch.n_embd,
        n_inner=arch.n_inner, n_positions=arch.n_positions,
        vocab_size=arch.vocab_size,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    )
    model = GPT2LMHeadModel(cfg)
    gen = torch.Generator().manual_seed(seed + 991)
    with torch.no_grad():
        for module in model.modules():
            if module.__class__.__name__ == "LayerNorm":
                module.weight.copy_(torch.exp(
                    torch.randn(module.weight.shape, generator=gen) * 0.5))
                module.bias.copy_(
                    torch.randn(module.bias.shape, generator=gen) * 0.1)
        for block in model.transformer.h:
            for proj in (block.attn.c_attn, block.attn.c_proj,
                         block.mlp.c_fc, block.mlp.c_proj):
                proj.bias.copy_(
                    torch.randn(proj.bias.shape, generator=gen) * 0.3)
    model.eval()
    return model


#: Junk pairs, and what each is for. The DIRECTION each metric must move is
#: stated here so a test cannot quietly agree with whatever the code produced.
JUNK_PAIRS = ("identical", "zeroed_block", "noise", "independent")


def build_junk_pair(kind: str, seed: int = 0, tiny: bool = True):
    """Two throwaway checkpoints related in a known way."""
    import copy

    torch = _torch()
    if kind not in JUNK_PAIRS:
        raise MetricsError(f"unknown junk pair {kind!r}; expected {JUNK_PAIRS}")

    a = build_junk_model(seed=seed, tiny=tiny)
    if kind == "identical":
        return a, copy.deepcopy(a)
    if kind == "independent":
        return a, build_junk_model(seed=seed + 5000, tiny=tiny)

    b = copy.deepcopy(a)
    if kind == "zeroed_block":
        # The LAST block, so the change is visible in the final layers and
        # provably absent from the early ones. A metric that moves everywhere
        # is as broken as one that moves nowhere.
        with torch.no_grad():
            for p in b.transformer.h[-1].parameters():
                p.zero_()
        return a, b
    gen = torch.Generator().manual_seed(seed + 7717)
    with torch.no_grad():
        for p in b.parameters():
            p.add_(torch.randn(p.shape, generator=gen) * 1e-3)
    return a, b


# ---------------------------------------------------------------------------
# Deliberately not built -- see the module docstring
# ---------------------------------------------------------------------------


def aligned_barrier(*_args, **_kwargs):
    raise NotImplementedError(_BLOCKED.format(
        what="A permutation-aligned barrier"))


def aligned_l2(*_args, **_kwargs):
    raise NotImplementedError(_BLOCKED.format(
        what="Permutation-aligned L2 distance"))


def rsf_subspace_probe(*_args, **_kwargs):
    raise NotImplementedError(_BLOCKED.format(
        what="The RSF subspace probe"))
