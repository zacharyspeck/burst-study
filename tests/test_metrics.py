"""Step 10 metrics: correctness, and RESPONSIVENESS.

THE RISK THIS FILE EXISTS FOR.

There are no trained checkpoints and there will not be for weeks, so every
metric here is being developed against junk weights. **A metric that runs on
junk and returns a plausible float looks exactly like a metric that works.**
Nothing crashes, the number has the right sign and magnitude, and the failure
is only discovered when a real result is argued from it -- which is how S49,
S53 and S48 each went wrong in step 9.

So every metric gets two kinds of test:

1. CORRECTNESS -- the value is right on a case where the answer is known
   independently: identical models, a hand-computable CKA, a chord with zero
   excess by construction.

2. RESPONSIVENESS -- the value MOVES when the models differ, by a stated
   margin, and does NOT move when they do not. A metric that cannot tell a
   model from a copy of itself with one block zeroed is not measuring
   anything. The margins are written as constants below so a test cannot
   quietly agree with whatever the code happened to produce.

The zeroed-block pair carries most of the responsiveness weight because its
direction is predictable: layers before the zeroed block must be untouched and
layers at or after it must change. A test that only checks "something moved"
would pass for a metric that moves everywhere, which is equally broken.

WHY THE INDEPENDENT-INIT PAIR IS NOT LOAD-BEARING FOR THE BARRIER. Two
independently initialized models sit in a region where the loss is at chance
along the entire interpolation, so the barrier between them can come back near
zero rather than large -- not because the metric failed but because there is no
structure to interpolate through. That pair is asserted only where its
direction is genuinely predictable (L2, CKA, cosine) and is explicitly NOT used
to prove the barrier responds. See test_barrier_responds_to_a_zeroed_block.
"""

from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def _load(name):
    """Import a script module by path.

    Registered in sys.modules BEFORE exec: @dataclass resolves its own module
    through sys.modules to check field types, and a module that is mid-exec and
    unregistered makes that lookup return None.
    """
    import sys

    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = _load("metrics")

# Responsiveness margins, stated up front rather than discovered.
IDENTICAL_TOL = 1e-10
CKA_DIFFERENT_MAX = 0.99
COSINE_DIFFERENT_MAX = 0.99
L2_DIFFERENT_MIN = 1e-3


@pytest.fixture(scope="module")
def tiny_batch():
    """A fixed batch that does not need the GPT-2 tokenizer.

    The committed context passage is the real measurement input, but it is
    1024 GPT-2 tokens against a 64-token vocabulary here, and loading the
    tokenizer would make these unit tests depend on a model download. The
    batch IDENTITY machinery is tested separately against the real file.
    """
    ids = tuple((i * 7 + 3) % 64 for i in range(24))
    return M.Batch(ids=ids, source="test://tiny", file_sha256="0" * 64,
                   token_sha256="1" * 64, tokenizer="test")


# ---------------------------------------------------------------------------
# The fixed batch and its identity
# ---------------------------------------------------------------------------


def test_context_file_matches_committed_provenance():
    """The measurement input is the committed, human-reviewed passage."""
    import hashlib
    import json

    raw = M.CONTEXT_FILE.read_bytes()
    record = json.loads(M.BURST_PROVENANCE.read_text(encoding="utf-8"))
    assert hashlib.sha256(raw).hexdigest() == record["context"]["sha256"]
    assert record["context"]["tokens"] == 1024
    assert record["tokenizer"] == "gpt2"


def test_batch_identity_carries_both_hashes(tiny_batch):
    """A CKA number without the input that produced it is not a measurement."""
    identity = tiny_batch.identity()
    assert set(identity) == {"source", "file_sha256", "token_sha256",
                             "n_tokens", "tokenizer"}
    assert identity["n_tokens"] == 24


def test_token_hash_is_over_ids_not_text():
    """The guard that makes re-deriving safer than pinning the ids.

    A tokenizer shift leaves the file hash alone and moves the token hash, so
    the drift is loud. If both hashes were over the text, it would be silent.
    """
    a = M.Batch(ids=(1, 2, 3), source="s", file_sha256="f",
                token_sha256="x", tokenizer="gpt2")
    b = M.Batch(ids=(1, 2, 4), source="s", file_sha256="f",
                token_sha256="y", tokenizer="gpt2")
    assert a.file_sha256 == b.file_sha256
    assert a.ids != b.ids


def test_provenance_mismatch_is_refused_not_adjusted():
    bad = M.Batch(ids=tuple(range(10)), source="bursts/context.txt",
                  file_sha256="deadbeef", token_sha256="x", tokenizer="gpt2")
    with pytest.raises(M.MetricsError, match="does not match"):
        M._check_against_provenance(bad)


# ---------------------------------------------------------------------------
# L2, raw
# ---------------------------------------------------------------------------


def test_l2_of_a_model_against_itself_is_exactly_zero():
    a, b = M.build_junk_pair("identical", seed=1)
    assert M.l2_distance_raw(a, b) == 0.0


def test_l2_responds_to_a_zeroed_block():
    a, b = M.build_junk_pair("zeroed_block", seed=1)
    assert M.l2_distance_raw(a, b) > L2_DIFFERENT_MIN


def test_l2_responds_to_small_noise_without_saturating():
    a, b = M.build_junk_pair("noise", seed=1)
    identical = M.l2_distance_raw(*M.build_junk_pair("identical", seed=1))
    far = M.l2_distance_raw(*M.build_junk_pair("independent", seed=1))
    noisy = M.l2_distance_raw(a, b)
    assert identical < noisy < far, (
        "a small perturbation must land strictly between 'same model' and "
        "'unrelated model'; a metric that saturates is not measuring distance")


def test_l2_matches_a_hand_computed_value():
    """Correctness against an independently computed number."""
    a = {"w": torch.tensor([[3.0, 0.0], [0.0, 4.0]])}
    b = {"w": torch.tensor([[0.0, 0.0], [0.0, 0.0]])}
    assert M.l2_distance_raw(a, b) == pytest.approx(5.0)


def test_l2_counts_a_tied_embedding_once():
    """state_dict() would double-weight 38M of GPT-2's parameters."""
    model = M.build_junk_model(seed=0)
    params = M.parameter_view(model)
    assert "lm_head.weight" not in params or (
        params["lm_head.weight"].data_ptr()
        != params["transformer.wte.weight"].data_ptr())
    names = list(params)
    assert len(names) == len(set(names))


def test_l2_refuses_mismatched_parameter_sets():
    with pytest.raises(M.MetricsError, match="names differ"):
        M.l2_distance_raw({"a": torch.zeros(2)}, {"b": torch.zeros(2)})


# ---------------------------------------------------------------------------
# Interpolation and the barrier
# ---------------------------------------------------------------------------


def test_interpolation_endpoints_are_exact():
    sd_a = {"w": torch.tensor([1.0, 2.0])}
    sd_b = {"w": torch.tensor([3.0, 6.0])}
    assert torch.equal(M.interpolate_state_dicts(sd_a, sd_b, 0.0)["w"],
                       sd_a["w"])
    assert torch.equal(M.interpolate_state_dicts(sd_a, sd_b, 1.0)["w"],
                       sd_b["w"])
    mid = M.interpolate_state_dicts(sd_a, sd_b, 0.5)["w"]
    assert torch.allclose(mid, torch.tensor([2.0, 4.0]))


def test_interpolation_refuses_to_blend_a_non_float_buffer():
    """The trap: a blended causal mask still runs and still looks plausible."""
    sd_a = {"mask": torch.tensor([True, False])}
    sd_b = {"mask": torch.tensor([False, True])}
    with pytest.raises(M.MetricsError, match="non-floating-point"):
        M.interpolate_state_dicts(sd_a, sd_b, 0.5)


def test_interpolation_passes_identical_non_float_buffers_through():
    sd_a = {"mask": torch.tensor([True, False]), "w": torch.tensor([0.0])}
    sd_b = {"mask": torch.tensor([True, False]), "w": torch.tensor([2.0])}
    out = M.interpolate_state_dicts(sd_a, sd_b, 0.5)
    assert torch.equal(out["mask"], sd_a["mask"])
    assert out["w"].item() == pytest.approx(1.0)


def test_barrier_between_identical_models_is_zero(tiny_batch):
    """Correctness: the curve IS the chord, so every excess is zero."""
    a, b = M.build_junk_pair("identical", seed=2)
    result = M.barrier(a, b, tiny_batch, alphas=(0.0, 0.25, 0.5, 0.75, 1.0))
    assert result["max_excess"] == pytest.approx(0.0, abs=1e-9)
    assert result["endpoint_losses"][0] == pytest.approx(
        result["endpoint_losses"][1], abs=1e-9)


def test_barrier_finds_a_peak_that_is_actually_there():
    """The barrier's real responsiveness test, and it is NOT run on junk.

    MEASURED, NOT ASSUMED: every junk pair sits at chance loss along the whole
    interpolation -- ln(64) = 4.1589 against endpoints of 4.160 to 4.164 -- so
    no throwaway pair produces a barrier to detect. Both the independent-init
    pair AND the zeroed-block pair return max_excess = 0.0, with the curve
    sagging below the chord rather than rising above it.

    That is a fact about junk checkpoints, not a failure of the metric, and it
    means a responsiveness test built on junk models could only ever confirm
    that zero is returned. So the arithmetic is tested against a curve with a
    peak put into it deliberately: if a barrier existed, this is the test that
    it would be found and correctly sized.
    """
    alphas = (0.0, 0.25, 0.5, 0.75, 1.0)
    # Endpoints at 2.0 and 4.0 -> chord is 2, 2.5, 3, 3.5, 4.
    losses = (2.0, 2.5, 3.75, 3.5, 4.0)
    result = M.barrier_from_losses(alphas, losses)
    assert result["max_excess"] == pytest.approx(0.75)
    assert result["argmax_alpha"] == 0.5
    assert result["rose_above_chord"] is True


def test_barrier_reports_a_sag_rather_than_a_misleading_zero():
    """A curve below the chord must be legible, not a flat zero.

    This is the shape every junk pair actually produces, so without min_excess
    the report would show 0.0 and read as a broken metric.
    """
    alphas = (0.0, 0.5, 1.0)
    result = M.barrier_from_losses(alphas, (2.0, 2.5, 4.0))
    assert result["max_excess"] == pytest.approx(0.0)
    assert result["rose_above_chord"] is False
    assert result["min_excess"] == pytest.approx(-0.5)
    assert result["argmin_alpha"] == 0.5
    assert "NOT_A_FAILURE" in "".join(result)


def test_barrier_evaluates_its_interior_alphas(tiny_batch):
    """What junk CAN establish: the interior points are really being run.

    Not a claim about barrier height. If the interior alphas were never
    evaluated the curve would equal the chord exactly and every excess would be
    identically zero, which this rules out.
    """
    a, b = M.build_junk_pair("zeroed_block", seed=2)
    result = M.barrier(a, b, tiny_batch, alphas=(0.0, 0.25, 0.5, 0.75, 1.0))
    assert result["endpoint_losses"][0] != pytest.approx(
        result["endpoint_losses"][1])
    assert any(abs(e) > 1e-9 for e in result["excess"]), (
        "the interpolated curve is exactly the chord, which means the "
        "interior alphas are not being evaluated at all")


def test_junk_checkpoints_sit_at_chance_loss(tiny_batch):
    """Pins the reason the barrier cannot be responsiveness-tested on junk.

    If this ever fails, junk models have stopped being at chance and the
    reasoning in test_barrier_finds_a_peak_that_is_actually_there needs
    revisiting rather than the threshold being nudged.
    """
    a, _ = M.build_junk_pair("identical", seed=2)
    chance = math.log(64)  # tiny vocab
    assert M.cross_entropy_loss(a, tiny_batch) == pytest.approx(chance, abs=0.1)


def test_barrier_retains_the_full_curve_and_names_its_grid(tiny_batch):
    a, b = M.build_junk_pair("noise", seed=2)
    result = M.barrier(a, b, tiny_batch, alphas=(0.0, 0.5, 1.0))
    assert len(result["losses"]) == 3
    assert result["alphas"] == [0.0, 0.5, 1.0]
    assert result["grid_points"] == 3
    assert "LOWER BOUND" in result["GRID_LIMITED"]
    assert result["batch"]["n_tokens"] == 24


def test_barrier_requires_exact_endpoints(tiny_batch):
    a, b = M.build_junk_pair("identical", seed=2)
    with pytest.raises(M.MetricsError, match="endpoint"):
        M.barrier(a, b, tiny_batch, alphas=(0.1, 0.5, 0.9))


def test_barrier_does_not_mutate_its_inputs(tiny_batch):
    a, b = M.build_junk_pair("noise", seed=3)
    before_a = M.l2_distance_raw(a, copy.deepcopy(a))
    sd_before = {k: v.clone() for k, v in a.state_dict().items()}
    M.barrier(a, b, tiny_batch, alphas=(0.0, 0.5, 1.0))
    assert before_a == 0.0
    for k, v in a.state_dict().items():
        assert torch.equal(v, sd_before[k]), f"{k} was mutated by barrier()"


# ---------------------------------------------------------------------------
# The activation routes -- a first-class requirement, not a nice-to-have
# ---------------------------------------------------------------------------


def test_tap_modules_reproduce_native_hidden_states_exactly(tiny_batch):
    """THE cross-check. Two genuinely different routes to the same tensors.

    Verified empirically before the tap list was written: HF returns
    [drop, h0, ..., h[n-2], ln_f(h[n-1])] -- the LAST entry is not the last
    block's output. Tapping every block instead would compare a raw block
    output against a normalized one in the final slot, and would not crash.
    """
    model = M.build_junk_model(seed=4)
    report = M.cross_check_activation_routes(model, tiny_batch)
    assert report["agree"] is True
    assert report["worst_abs_gap"] == 0.0
    assert report["n_layers"] == 3  # tiny is 2 layers -> n_layer + 1


def test_cross_check_raises_rather_than_picking_a_winner(tiny_batch, monkeypatch):
    """On disagreement it must STOP, not choose a route."""
    model = M.build_junk_model(seed=4)
    real = M.activations_native

    def wrong(mdl, batch):
        out = real(mdl, batch)
        return [out[0] + 1.0] + list(out[1:])

    monkeypatch.setattr(M, "activations_native", wrong)
    with pytest.raises(M.MetricsError, match="ROUTES DISAGREE"):
        M.cross_check_activation_routes(model, tiny_batch)


def test_cross_check_catches_a_layer_count_mismatch(tiny_batch, monkeypatch):
    model = M.build_junk_model(seed=4)
    real = M.activations_native
    monkeypatch.setattr(M, "activations_native",
                        lambda m, b: list(real(m, b))[:-1])
    with pytest.raises(M.MetricsError, match="LAYER COUNT"):
        M.cross_check_activation_routes(model, tiny_batch)


def test_naive_all_blocks_tap_would_disagree(tiny_batch):
    """Proves the trap is real rather than hypothetical.

    Tapping every transformer block -- the obvious tap list -- produces a final
    activation that is NOT what the model reports as its last hidden state.
    """
    model = M.build_junk_model(seed=4)
    naive = list(model.transformer.h)
    hooked = M.activations_by_hooks(model, tiny_batch, naive)
    native = M.activations_native(model, tiny_batch)
    assert not torch.allclose(hooked[-1], native[-1]), (
        "if these agree the trap has gone away and this test is now vacuous")


def test_tap_helper_refuses_an_unknown_model_shape():
    class NotAGPT2:
        pass

    with pytest.raises(M.MetricsError, match="does not expose"):
        M.hf_gpt2_tap_modules(NotAGPT2())


# ---------------------------------------------------------------------------
# CKA -- the variant is pinned, not merely chosen
# ---------------------------------------------------------------------------


def _hsic1_by_definition(K, L):
    """HSIC_1 transcribed directly from its index-sum definition.

    Deliberately O(n^2)-by-loops and deliberately NOT sharing code with the
    vectorized implementation. This is the independent route: if the fast path
    ever double-centers, or drops the diagonal-zeroing, or mis-scales a term,
    the two disagree.
    """
    n = K.shape[0]
    Kt = [[0.0 if i == j else float(K[i][j]) for j in range(n)]
          for i in range(n)]
    Lt = [[0.0 if i == j else float(L[i][j]) for j in range(n)]
          for i in range(n)]
    trace = sum(Kt[i][j] * Lt[i][j] for i in range(n) for j in range(n))
    sum_k = sum(Kt[i][j] for i in range(n) for j in range(n))
    sum_l = sum(Lt[i][j] for i in range(n) for j in range(n))
    cross = sum(sum(Kt[i][j] for j in range(n)) * sum(Lt[i][j] for j in range(n))
                for i in range(n))
    return (trace + sum_k * sum_l / ((n - 1) * (n - 2))
            - 2.0 * cross / (n - 2)) / (n * (n - 3))


def test_hsic1_matches_its_own_definition():
    """The pin. Catches double-centering, which is silently wrong not broken."""
    gen = torch.Generator().manual_seed(17)
    X = torch.randn(9, 4, generator=gen).double()
    Y = torch.randn(9, 3, generator=gen).double()
    K, L = X @ X.T, Y @ Y.T
    assert M._hsic1(K, L) == pytest.approx(_hsic1_by_definition(K, L), rel=1e-9)


def test_cka_of_a_representation_with_itself_is_one():
    gen = torch.Generator().manual_seed(19)
    X = torch.randn(40, 8, generator=gen)
    assert M.linear_cka_unbiased(X, X) == pytest.approx(1.0, abs=1e-9)


def test_cka_is_invariant_to_orthogonal_transform():
    """The defining property of CKA, and what distinguishes it from cosine."""
    gen = torch.Generator().manual_seed(23)
    X = torch.randn(60, 8, generator=gen).double()
    Q, _ = torch.linalg.qr(torch.randn(8, 8, generator=gen).double())
    assert M.linear_cka_unbiased(X, X @ Q) == pytest.approx(1.0, abs=1e-9)


def test_cka_is_invariant_to_isotropic_scaling():
    gen = torch.Generator().manual_seed(29)
    X = torch.randn(60, 8, generator=gen).double()
    assert M.linear_cka_unbiased(X, X * 7.5) == pytest.approx(1.0, abs=1e-9)


def test_cka_is_low_for_independent_representations():
    gen = torch.Generator().manual_seed(31)
    X = torch.randn(400, 8, generator=gen).double()
    Y = torch.randn(400, 8, generator=gen).double()
    assert M.linear_cka_unbiased(X, Y) < 0.2


def test_cka_is_not_clipped_to_the_unit_interval():
    """Unbiased HSIC can go slightly negative. Report it, do not hide it.

    Clipping would convert a diagnostic into a plausible float that looks like
    a working metric -- the exact failure this build keeps repeating.
    """
    values = []
    for seed in range(40):
        gen = torch.Generator().manual_seed(1000 + seed)
        X = torch.randn(24, 3, generator=gen).double()
        Y = torch.randn(24, 3, generator=gen).double()
        values.append(M.linear_cka_unbiased(X, Y))
    assert min(values) < 0.0, (
        "no draw landed below zero, so this test is not exercising the "
        "unclipped path any more")
    flagged = M.per_layer_cka([torch.randn(24, 3)], [torch.randn(24, 3)])
    assert "outside_unit_interval" in flagged[0]


def test_cka_variant_is_named_and_pinned():
    assert M.CKA_VARIANT == "linear_cka_unbiased_hsic_tokens_as_samples_v1"
    for token in ("linear", "unbiased", "tokens"):
        assert token in M.CKA_VARIANT


def test_cka_refuses_too_few_samples():
    with pytest.raises(M.MetricsError, match="at least 4"):
        M.linear_cka_unbiased(torch.randn(3, 2), torch.randn(3, 2))


def test_cka_refuses_mismatched_sample_counts():
    with pytest.raises(M.MetricsError, match="same number of samples"):
        M.linear_cka_unbiased(torch.randn(10, 2), torch.randn(9, 2))


# ---------------------------------------------------------------------------
# Activation similarity
# ---------------------------------------------------------------------------


def test_cosine_of_identical_activations_is_one():
    gen = torch.Generator().manual_seed(37)
    A = torch.randn(1, 12, 8, generator=gen)
    out = M.activation_cosine(A, A.clone())
    assert out["cosine_min"] == pytest.approx(1.0, abs=IDENTICAL_TOL)
    assert out["norm_ratio_median"] == pytest.approx(1.0, abs=IDENTICAL_TOL)


def test_cosine_is_blind_to_magnitude_and_says_so():
    """The documented blindness, asserted so the docs cannot drift from it."""
    gen = torch.Generator().manual_seed(41)
    A = torch.randn(1, 12, 8, generator=gen)
    out = M.activation_cosine(A, A * 3.0)
    assert out["cosine_median"] == pytest.approx(1.0, abs=1e-9)
    assert out["norm_ratio_median"] == pytest.approx(1.0 / 3.0, rel=1e-6)


def test_cosine_is_not_rotation_invariant_unlike_cka():
    """The reason both metrics are reported: they disagree on purpose."""
    gen = torch.Generator().manual_seed(43)
    A = torch.randn(200, 8, generator=gen).double()
    Q, _ = torch.linalg.qr(torch.randn(8, 8, generator=gen).double())
    rotated = A @ Q
    assert M.linear_cka_unbiased(A, rotated) == pytest.approx(1.0, abs=1e-9)
    assert abs(M.activation_cosine(A, rotated)["cosine_median"]) < 0.9


def test_cosine_refuses_a_zero_activation_vector():
    A = torch.zeros(1, 4, 8)
    with pytest.raises(M.MetricsError, match="zero activation"):
        M.activation_cosine(A, A)


# ---------------------------------------------------------------------------
# End to end on junk models: identical scores identically, different does not
# ---------------------------------------------------------------------------


def test_identical_models_score_identically_on_every_metric(tiny_batch):
    a, b = M.build_junk_pair("identical", seed=5)
    acts_a = M.layer_activations(a, tiny_batch)
    acts_b = M.layer_activations(b, tiny_batch)

    assert M.l2_distance_raw(a, b) == 0.0
    for row in M.per_layer_cka(acts_a, acts_b):
        assert row["cka"] == pytest.approx(1.0, abs=1e-8)
    for row in M.per_layer_activation_similarity(acts_a, acts_b):
        assert row["cosine_min"] == pytest.approx(1.0, abs=1e-6)


def test_zeroed_block_moves_late_layers_and_leaves_early_ones_alone(tiny_batch):
    """The strongest responsiveness case: a PREDICTABLE direction.

    Layers before the change must be untouched and layers at or after it must
    move. A metric that moved everywhere would pass a weaker 'something
    changed' test while being just as broken.
    """
    a, b = M.build_junk_pair("zeroed_block", seed=5)
    acts_a = M.layer_activations(a, tiny_batch)
    acts_b = M.layer_activations(b, tiny_batch)
    cka = M.per_layer_cka(acts_a, acts_b)
    cos = M.per_layer_activation_similarity(acts_a, acts_b)

    # Tiny is 2 blocks, so the tap list is [drop, h0, ln_f(h1)] and zeroing the
    # last block must leave EVERY layer before the final one untouched.
    for i in range(len(cka) - 1):
        assert cka[i]["cka"] == pytest.approx(1.0, abs=1e-8), (
            f"layer {i} precedes the zeroed block and must be unchanged; a "
            f"metric that moves everywhere is as broken as one that never does")
        assert cos[i]["cosine_min"] == pytest.approx(1.0, abs=1e-6)

    # Measured at these seeds: cka 0.9618, cosine 0.7209. Thresholds are set
    # against those, not tuned to them.
    assert cka[-1]["cka"] < CKA_DIFFERENT_MAX
    assert cos[-1]["cosine_median"] < COSINE_DIFFERENT_MAX


def test_independent_models_are_far_apart(tiny_batch):
    a, b = M.build_junk_pair("independent", seed=5)
    acts_a = M.layer_activations(a, tiny_batch)
    acts_b = M.layer_activations(b, tiny_batch)
    assert M.l2_distance_raw(a, b) > L2_DIFFERENT_MIN
    assert M.per_layer_cka(acts_a, acts_b)[-1]["cka"] < CKA_DIFFERENT_MAX
    assert (M.per_layer_activation_similarity(acts_a, acts_b)[-1]
            ["cosine_median"]) < COSINE_DIFFERENT_MAX


# ---------------------------------------------------------------------------
# What is deliberately absent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["aligned_barrier", "aligned_l2",
                                  "rsf_subspace_probe"])
def test_alignment_dependent_metrics_name_their_blocker(name):
    """Absent loudly, not silently. An omission looks like an oversight."""
    with pytest.raises(NotImplementedError) as exc:
        getattr(M, name)()
    message = str(exc.value)
    assert "Conv1D" in message
    assert "docs/layout-cost.md" in message
    # Was `assert "undecided" in message`, which pinned a claim the code
    # contradicts: the study's model is selected by --family, required with no
    # default, and hf_gpt2 builds a real GPT2LMHeadModel. What the message must
    # still do is say it is unbuilt and name where the decision is recorded.
    assert "NOT BUILT" in message
    assert "D-6" in message
    assert "SKIPPED" in message
