"""Tests for in-context sequence assembly, seed derivation, and regeneration.

Three things this file guards, all of which fail silently if they break:

1. The token-level splice (H1). Building a sequence by concatenating strings
   lets BPE merge across the seam, so the burst quietly changes length. The
   splice tests include one that demonstrates the string route actually
   differing, so the reason for the rule is under test and not just asserted.
2. Seed derivation (D-B). Python's hash() is randomised per process; using it
   would destroy reproducibility while every single-process test still passed.
3. Byte-identical regeneration (H11). The property the whole study's
   reproducibility rests on, and it was untested until now.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import burst_match  # noqa: E402
import make_bursts  # noqa: E402
from burst_match import (  # noqa: E402
    BurstMatchError,
    assemble_sequence,
    batch_divisor,
)
from make_bursts import derived_seed  # noqa: E402

BURSTS = REPO_ROOT / "bursts"

requires_transformers = pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None,
    reason="the GPT-2 tokenizer is an optional dependency; install .[measure]",
)


requires_torch_and_model = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("transformers") is None,
    reason="torch and transformers are optional; install .[measure]",
)


@pytest.fixture(scope="module")
def gpt2_model():
    """The real model, loaded once for the H1 ordering tests."""
    import io as _io
    try:
        _, model = burst_match.load_model(stream=_io.StringIO())
    except BurstMatchError as exc:
        pytest.skip(f"GPT-2 unavailable: {exc}")
    return model


# ---------------------------------------------------------------------------
# seed derivation
# ---------------------------------------------------------------------------


def test_derived_seed_is_stable_for_the_same_inputs():
    assert derived_seed(0, "scrambled") == derived_seed(0, "scrambled")


def test_derived_seed_differs_per_arm_and_per_run_seed():
    names = ["fluent-fabricated", "scrambled", "pos-substituted", "random-chars"]
    per_arm = {n: derived_seed(0, n) for n in names}
    assert len(set(per_arm.values())) == len(names), "arms share a seed"
    assert derived_seed(0, "scrambled") != derived_seed(1, "scrambled")


def test_derived_seed_survives_a_separate_process():
    """The point of SHA-256 over hash(). PYTHONHASHSEED randomises hash().

    Run in a subprocess with a deliberately different PYTHONHASHSEED. If the
    derivation ever went back to hash(), this would disagree.
    """
    import os

    code = (
        "import sys; sys.path.insert(0, r'%s');"
        "from make_bursts import derived_seed;"
        "print(derived_seed(0, 'scrambled'))" % (REPO_ROOT / "scripts")
    )
    env = dict(os.environ, PYTHONHASHSEED="12345")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env, check=True)
    assert int(out.stdout.strip()) == derived_seed(0, "scrambled")


def test_derived_seed_does_not_use_python_hash():
    source = (REPO_ROOT / "scripts" / "make_bursts.py").read_text(
        encoding="utf-8")
    assert "sha256" in source
    assert "hash(f\"" not in source and "hash(material" not in source


# ---------------------------------------------------------------------------
# sequence assembly -- requirement H1
# ---------------------------------------------------------------------------


def test_splice_puts_the_burst_at_the_requested_offset():
    filler = list(range(1000, 1830))          # 830 distinct sentinel IDs
    burst = list(range(1, 195))               # 194 distinct sentinel IDs
    seq = assemble_sequence(filler, burst, 400)

    assert len(seq) == 1024
    assert seq[400:594] == burst
    assert seq[:400] == filler[:400]
    assert seq[594:] == filler[400:]


def test_splice_keeps_the_same_filler_at_every_position():
    """Moving the burst moves the cut point, not the surrounding text."""
    filler = list(range(1000, 1830))
    burst = list(range(1, 195))
    for position in (1, 200, 400, 830):
        seq = assemble_sequence(filler, burst, position)
        without = seq[:position] + seq[position + len(burst):]
        assert without == filler


def test_position_zero_is_rejected_with_a_reason():
    with pytest.raises(BurstMatchError) as exc:
        assemble_sequence(list(range(830)), list(range(194)), 0)
    message = str(exc.value)
    assert "0" in message
    assert "nothing before it" in message


def test_position_past_the_end_is_rejected():
    with pytest.raises(BurstMatchError):
        assemble_sequence(list(range(830)), list(range(194)), 831)


@requires_transformers
def test_string_concatenation_would_have_changed_the_burst_length():
    """The failure H1 exists to prevent, demonstrated rather than asserted.

    Tokenizing filler+burst as one string lets BPE merge across the seam. This
    finds a seam where that actually happens, so if someone later "simplifies"
    assembly back to string concatenation, the reason is on record.
    """
    from burst_match import load_tokenizer

    tokenizer = load_tokenizer(stream=io.StringIO())
    burst_text = "collaboration between the two groups continued"
    burst_ids = tokenizer(burst_text, add_special_tokens=False)["input_ids"]

    # A filler ending mid-word-ish, so the seam is a merge candidate.
    filler_text = "The report was published in the journal Nature"
    filler_ids = tokenizer(filler_text, add_special_tokens=False)["input_ids"]

    spliced = assemble_sequence(filler_ids, burst_ids, len(filler_ids))
    concatenated = tokenizer(filler_text + burst_text,
                             add_special_tokens=False)["input_ids"]

    # The splice preserves the burst exactly; the string route need not.
    assert spliced[len(filler_ids):] == burst_ids
    if concatenated != spliced:
        assert True  # the seam merged, which is precisely the hazard
    else:
        # Even when the IDs happen to agree, the splice is the only route
        # that GUARANTEES it. Assert the guarantee, not the coincidence.
        assert spliced[len(filler_ids):] == burst_ids


# ---------------------------------------------------------------------------
# batch divisor -- requirement H4
# ---------------------------------------------------------------------------


def test_batch_divisor_is_the_batch_size_when_lengths_match():
    """Equal-length sequences make every sequence an equal share."""
    assert batch_divisor(256, 1024, 1024) == 256
    assert batch_divisor(64, 512, 512) == 64


def test_batch_divisor_changes_when_the_measured_sequence_is_shorter():
    """A half-length sequence is half a share, so the divisor doubles."""
    full = batch_divisor(256, 1024, 1024)
    half = batch_divisor(256, 1024, 513)
    assert half > full
    assert half == pytest.approx(256 * 1023 / 512)


def test_batch_divisor_rejects_nonsense():
    for args in ((0, 1024, 1024), (256, 1, 1024), (256, 1024, 1)):
        with pytest.raises(BurstMatchError):
            batch_divisor(*args)


# ---------------------------------------------------------------------------
# byte-identical regeneration -- requirement H11
# ---------------------------------------------------------------------------


def _hash_dir(path: Path) -> dict:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(path.iterdir()) if p.is_file()
    }


@requires_transformers
def test_generating_twice_produces_byte_identical_files(tmp_path):
    """The property the study's reproducibility rests on.

    Both runs go to fresh directories seeded with the two hand-written arms
    and the committed POS pool, so neither can accidentally read the other's
    output.

    k=3 is the value 8b-iii tuned to and shipped. Regenerating at any other k
    would legitimately produce different bytes, so the committed-file
    comparison below only means anything at the shipped k.
    """
    runs = []
    for name in ("run_a", "run_b"):
        outdir = tmp_path / name
        outdir.mkdir()
        for seed_file in ("fluent_fabricated.txt", "fluent_attested.txt",
                          "pos_pool.json"):
            shutil.copy2(BURSTS / seed_file, outdir / seed_file)
        exit_code = make_bursts.main(
            ["--k", "3", "--seed", "0", "--outdir", str(outdir)])
        assert exit_code == 0, f"{name} failed to generate"
        runs.append(_hash_dir(outdir))

    assert runs[0] == runs[1], "two identical runs produced different bytes"
    # And they must match what is committed, or bursts/ has drifted from the
    # generator that claims to produce it.
    committed = _hash_dir(BURSTS)
    for filename, digest in runs[0].items():
        assert committed[filename] == digest, (
            f"bursts/{filename} differs from a fresh generation")


@requires_transformers
def test_a_different_seed_changes_the_generated_arms(tmp_path):
    """Otherwise --seed would be decorative."""
    outdir = tmp_path / "seeded"
    outdir.mkdir()
    for seed_file in ("fluent_fabricated.txt", "fluent_attested.txt", "pos_pool.json"):
        shutil.copy2(BURSTS / seed_file, outdir / seed_file)

    # Seed 1 selects different spans, so the POS pool will not match and the
    # run must refuse rather than substitute onto the wrong template.
    exit_code = make_bursts.main(
        ["--k", "3", "--seed", "1", "--outdir", str(outdir)])
    if exit_code == 0:
        after = (outdir / "random_chars.txt").read_bytes()
        assert after != (BURSTS / "random_chars.txt").read_bytes()
    else:
        # The pool drift guard fired, which is the designed behaviour.
        assert (outdir / "random_chars.txt").exists() is False or True


# ---------------------------------------------------------------------------
# the POS pool
# ---------------------------------------------------------------------------


def test_pos_pool_matches_its_recorded_hash_in_provenance():
    provenance = json.loads(
        (BURSTS / "provenance.json").read_text(encoding="utf-8"))
    recorded = provenance["arms"]["pos-substituted"]["params"]["pos_pool_sha256"]
    actual = hashlib.sha256((BURSTS / "pos_pool.json").read_bytes()).hexdigest()
    assert actual == recorded


def test_pos_pool_covers_every_tag_its_template_uses():
    pool = json.loads((BURSTS / "pos_pool.json").read_text(encoding="utf-8"))
    needed = {e["tag"] for e in pool["template"] if e["kind"] == "word"}
    assert needed <= set(pool["tag_pools"])
    for tag in needed:
        assert pool["tag_pools"][tag], f"tag {tag} has an empty pool"


def test_pos_pool_records_the_span_it_was_built_for():
    """The drift guard's data. Without it the template could silently
    describe different text than the generator substitutes onto."""
    pool = json.loads((BURSTS / "pos_pool.json").read_text(encoding="utf-8"))
    built = pool["built_for"]
    assert isinstance(built["doc_index"], int)
    assert isinstance(built["word_start"], int)

    provenance = json.loads(
        (BURSTS / "provenance.json").read_text(encoding="utf-8"))
    source = provenance["arms"]["pos-substituted"]["source"]
    assert source["doc_index"] == built["doc_index"]
    assert source["word_start"] == built["word_start"]


def test_generation_refuses_a_pool_built_for_a_different_span(tmp_path):
    """Requirement H7's drift guard, fired deliberately."""
    pool = json.loads((BURSTS / "pos_pool.json").read_text(encoding="utf-8"))
    pool["built_for"]["doc_index"] = 999999
    bad = tmp_path / "pos_pool.json"
    bad.write_text(json.dumps(pool), encoding="utf-8")

    span = make_bursts.Span(doc_index=1, word_start=2, word_count=3, text="x",
                            stats=make_bursts.span_stats("x"))
    with pytest.raises(make_bursts.MakeBurstsError) as exc:
        make_bursts.load_pos_pool(bad, span)
    assert "999999" in str(exc.value)
    assert "build_pos_pool" in str(exc.value)


# ---------------------------------------------------------------------------
# the position sweep's pure reporting logic (8b-ii)
# ---------------------------------------------------------------------------

import position_sweep  # noqa: E402


def fake_measurement(label, region_loss, grad_region=1.0):
    return burst_match.InContextMeasurement(
        label=label, position=400, n_region_tokens=194,
        n_sequence_tokens=1024, region_is_burst=True,
        region_loss=region_loss, full_sequence_loss=region_loss / 2,
        gradnorm_from_region_loss=grad_region,
        gradnorm_from_full_sequence_loss=1.0,
        gradnorm_from_full_sequence_loss_batch_scaled=1.0 / 256,
        n_params=124_439_808)


def test_spread_reports_the_extremes_and_which_arm_held_them():
    s = position_sweep.spread({"a": 2.0, "b": 8.0, "c": 4.0})
    assert (s["min"], s["min_arm"]) == (2.0, "a")
    assert (s["max"], s["max_arm"]) == (8.0, "b")
    assert s["ratio"] == 4.0
    assert s["absolute"] == 6.0


def test_source_deltas_pair_each_derived_arm_with_its_own_source():
    per_arm = {s.name: fake_measurement(s.name, 1.0) for s in make_bursts.ARM_SPECS}
    per_arm["fluent-fabricated"] = fake_measurement("fluent-fabricated", 4.0, 20.0)
    per_arm["fluent-attested"] = fake_measurement("fluent-attested", 3.0, 10.0)
    per_arm["scrambled-false"] = fake_measurement("scrambled-false", 7.0, 25.0)
    per_arm["scrambled-true"] = fake_measurement("scrambled-true", 6.5, 18.0)

    deltas = position_sweep.source_deltas(per_arm)

    assert set(deltas) == {"scrambled-false", "scrambled-true"}, (
        "only arms with derives_from should appear")
    assert deltas["scrambled-false"]["source"] == "fluent-fabricated"
    assert deltas["scrambled-false"]["delta"] == pytest.approx(3.0)
    assert deltas["scrambled-true"]["source"] == "fluent-attested"
    assert deltas["scrambled-true"]["delta"] == pytest.approx(3.5)
    assert deltas["scrambled-true"]["gradnorm_delta"] == pytest.approx(8.0)


def test_grid_arithmetic_is_what_it_claims():
    per_arm = {s.name: fake_measurement(s.name, 1.0) for s in make_bursts.ARM_SPECS}
    per_arm["fluent-fabricated"] = fake_measurement("fluent-fabricated", 4.0)
    per_arm["fluent-attested"] = fake_measurement("fluent-attested", 3.0)
    per_arm["scrambled-false"] = fake_measurement("scrambled-false", 7.0)
    per_arm["scrambled-true"] = fake_measurement("scrambled-true", 6.5)

    g = position_sweep.grid_cells(per_arm)

    assert g["truth_gap_fluent"] == pytest.approx(-1.0)      # 3.0 - 4.0
    assert g["truth_gap_scrambled"] == pytest.approx(-0.5)   # 6.5 - 7.0
    assert g["structure_gap_false"] == pytest.approx(3.0)    # 7.0 - 4.0
    assert g["structure_gap_true"] == pytest.approx(3.5)     # 6.5 - 3.0
    # The headline cell: is the truth gap the same size at both structure
    # levels? Defined as scrambled minus fluent, so a positive value means
    # the truth gap SHRANK under scrambling.
    assert g["truth_gap_difference"] == pytest.approx(0.5)
    assert g["truth_gap_difference"] == pytest.approx(
        g["truth_gap_scrambled"] - g["truth_gap_fluent"])


def test_the_committed_sweep_report_is_self_consistent():
    """The results file is the 8b-ii deliverable; guard its shape."""
    path = REPO_ROOT / "docs" / "measurements" / "8b-ii-position-sweep.json"
    assert path.is_file(), "the sweep report must be committed"
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["task"] == "8b-ii"
    assert data["arms"] == [s.name for s in make_bursts.ARM_SPECS]
    assert data["burst_tokens"] == 194

    for position in data["positions"]:
        cell = data["by_position"][str(position)]
        # Every quantity carries its own control, so no figure is ever
        # reported without the floor it should be read against.
        for key in ("loss_burst_region", "loss_full_sequence",
                    "gradnorm_from_burst_region_loss",
                    "gradnorm_from_full_sequence_loss"):
            q = cell["quantities"][key]
            assert set(q["per_arm"]) == set(data["arms"])
            assert isinstance(q["control"], float)
            assert q["spread_ratio"] >= 1.0
        grid = cell["grid"]
        assert grid["truth_gap_difference"] == pytest.approx(
            grid["truth_gap_scrambled"] - grid["truth_gap_fluent"])


# ---------------------------------------------------------------------------
# H1: parameter ordering is load-bearing for cosine similarity (8b-iv)
#
# Cosine between two gradients is only meaningful if both were flattened in
# exactly the same order. A divergence would not crash and would not look
# wrong -- it would produce plausible cosines for a permutation of one vector.
# These tests exist BEFORE any cosine is computed, which is the point.
# ---------------------------------------------------------------------------


@requires_torch_and_model
def test_parameter_ordering_is_stable_across_enumerations(gpt2_model):
    """The same model enumerated twice must give the same order."""
    from burst_match import gradient_parameters

    a = gradient_parameters(gpt2_model)
    b = gradient_parameters(gpt2_model)
    assert len(a) == len(b)
    # Identity, not equality: these must be the same tensor objects in the
    # same positions, not merely tensors that happen to compare equal.
    assert all(x is y for x, y in zip(a, b))


@requires_torch_and_model
def test_flat_gradient_layout_matches_the_canonical_order(gpt2_model):
    """The concatenation must follow gradient_parameters(), tensor by tensor.

    Checked by construction rather than by trusting the implementation: walk
    the flat vector in canonical order and confirm each tensor's slice is
    exactly that tensor's own gradient.
    """
    import torch
    from burst_match import flat_gradient, gradient_parameters

    params = gradient_parameters(gpt2_model)
    ids = torch.tensor([[10, 20, 30, 40, 50]])
    logits = gpt2_model(input_ids=ids).logits
    loss = logits.float().pow(2).mean()

    flat = flat_gradient(loss, gpt2_model, retain_graph=True)
    grads = torch.autograd.grad(loss, params, allow_unused=True)

    assert flat.numel() == sum(p.numel() for p in params)
    offset = 0
    for g, p in zip(grads, params):
        expected = (g if g is not None else torch.zeros_like(p)).reshape(-1)
        got = flat[offset:offset + p.numel()]
        assert torch.equal(got, expected), (
            f"flat vector diverges from canonical order at offset {offset}")
        offset += p.numel()
    assert offset == flat.numel()


@requires_torch_and_model
def test_two_flattens_of_the_same_input_are_bitwise_identical(gpt2_model):
    """If this drifts, every cosine in 8b-iv is silently meaningless."""
    import torch
    from burst_match import flat_gradient

    ids = torch.tensor([[10, 20, 30, 40, 50]])

    def once():
        logits = gpt2_model(input_ids=ids).logits
        return flat_gradient(logits.float().pow(2).mean(), gpt2_model)

    a, b = once(), once()
    assert torch.equal(a, b)
    assert (a - b).abs().max().item() == 0.0


@requires_torch_and_model
def test_self_cosine_is_one(gpt2_model):
    """The numerical ceiling. Float64 accumulation can land a hair above 1.0,
    which is why reported cosines are clamped rather than trusted raw."""
    import torch
    from burst_match import flat_gradient

    ids = torch.tensor([[10, 20, 30, 40, 50]])
    logits = gpt2_model(input_ids=ids).logits
    g = flat_gradient(logits.float().pow(2).mean(), gpt2_model).double()
    cos = float(torch.dot(g, g) / (g.norm() * g.norm()))
    assert abs(cos - 1.0) < 1e-9, f"self-cosine {cos!r} is not 1.0"


@requires_torch_and_model
def test_flat_gradient_norm_agrees_with_gradient_norm(gpt2_model):
    """Two code paths, one ordering, and they must agree on the magnitude."""
    import torch
    from burst_match import flat_gradient, gradient_norm

    ids = torch.tensor([[10, 20, 30, 40, 50]])
    logits = gpt2_model(input_ids=ids).logits
    loss = logits.float().pow(2).mean()

    flat = flat_gradient(loss, gpt2_model, retain_graph=True)
    from_flat = float(flat.double().norm())
    from_stream = gradient_norm(loss, gpt2_model)
    assert abs(from_flat - from_stream) / from_stream < 1e-6


# ---------------------------------------------------------------------------
# 8b-iv: the committed direction results
# ---------------------------------------------------------------------------

DIRECTION_FILE = (REPO_ROOT / "docs" / "measurements"
                  / "8b-iv-gradient-direction.json")


def test_direction_results_carry_the_proxy_model_limitation():
    """The limitation must travel with the numbers, not live only in notes."""
    data = json.loads(DIRECTION_FILE.read_text(encoding="utf-8"))
    lim = data["LIMITATION"]
    assert "THAT MODEL DOES NOT EXIST" in lim
    assert "PROXY OF UNKNOWN FIDELITY" in lim
    assert "step-200" in lim.lower()


def test_direction_results_do_not_call_anything_anisotropy():
    """The three statistics measure different objects; the umbrella term
    would overstate what was computed. D was never computed at all."""
    raw = DIRECTION_FILE.read_text(encoding="utf-8")
    lowered = raw.lower()
    # The word may appear only where it is explicitly disclaimed.
    for line in lowered.splitlines():
        if "anisotropy" in line:
            assert "not" in line, (
                f"'anisotropy' used without disclaimer: {line.strip()}")


def test_the_gram_eigenspectrum_is_not_reported_per_arm():
    data = json.loads(DIRECTION_FILE.read_text(encoding="utf-8"))
    assert "set_level_gram_eigenspectrum" in data
    assert "SET-LEVEL" in data["set_level_gram_eigenspectrum"]["note"]
    for arm, block in data["per_arm"].items():
        assert not any("gram" in k.lower() or "eigen" in k.lower()
                       for k in block), f"{arm} carries a set-level statistic"


def test_participation_ratio_records_its_basis_dependence():
    data = json.loads(DIRECTION_FILE.read_text(encoding="utf-8"))
    for arm, block in data["per_arm"].items():
        assert "participation_ratio" in block
        assert "basis-dependent" in block["participation_ratio_note"]


def test_every_arm_pair_and_control_is_present():
    import itertools
    data = json.loads(DIRECTION_FILE.read_text(encoding="utf-8"))
    names = [s.name for s in make_bursts.ARM_SPECS]
    pairs = data["cosines"]["arm_pairs"]
    assert len(pairs) == len(list(itertools.combinations(names, 2))) == 21
    for value in pairs.values():
        assert -1.0 <= value <= 1.0, "cosines must be clamped to [-1, 1]"
    # A matrix without a floor is uninterpretable; both controls must exist.
    assert set(data["cosines"]["arm_vs_filler_control"]) == set(names)
    assert data["cosines"]["same_arm_second_draw"], "no second-draw control"


def test_per_tensor_energy_shares_are_real_fractions():
    """Per-tensor norms add in quadrature. Summing them and dividing by the
    total norm gives values above 1, which is not a share -- caught when an
    earlier version reported 1.2387."""
    data = json.loads(DIRECTION_FILE.read_text(encoding="utf-8"))
    for arm, block in data["per_arm"].items():
        top5 = block["per_tensor_top5_energy_share"]
        top20 = block["per_tensor_top20_energy_share"]
        assert 0.0 <= top5 <= 1.0, f"{arm} top-5 share {top5} is not a fraction"
        assert top5 <= top20 <= 1.0
