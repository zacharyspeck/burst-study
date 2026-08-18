"""The displacement ladder: scripts/displacement_ladder.py.

TWO FIXTURES, EVERY TEST THAT TOUCHES A MODEL, for the reason
tests/test_injection_point_ruler.py:7-27 records. A freshly-constructed GPT-2
has every LayerNorm gain at exactly 1.0 and every Conv1D bias at exactly 0.0,
and twice in this build that configuration made a test vacuous rather than
failing (S42, S48).

It bites this file specifically. Per-layer L2 sums squared differences over
buckets of parameters, and on a fresh model every Conv1D bias in the pair is
0.0 on both sides -- so a bucket whose only difference lived in the biases
reports exactly 0.0 for a reason that has nothing to do with the grouping being
right. A suite that only saw dispersed models would never notice; one that only
saw fresh models would not know whether the code works for any reason other
than everything being 1.0.

WHAT THE TORCH-FREE HALF IS FOR. scripts/displacement_ladder.py imports
metrics and model_seam, both of which take torch lazily through `_torch()`, so
the module imports with no ML stack installed. The grouping arithmetic, the
payload validation, the no-verdict guard and the step derivation are all pure
and are tested in BOTH environments. That is deliberate: they are the parts that
decide whether a number is about what it claims, and proving them where there
are fewer moving parts is the same argument that keeps burst/ PyYAML-only.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DL = _load("displacement_ladder")

from burst.config import load_config  # noqa: E402


# ---------------------------------------------------------------------------
# Parameter names, independent of the code under test
#
# These are HF GPT-2's own names, written out rather than derived from
# DL.bucket_for -- deriving the expectation from the thing being tested is how
# a grouping test agrees with whatever the grouping happens to do.
# ---------------------------------------------------------------------------

#: The twelve parameters HF puts in every GPT-2 block, in HF's naming.
BLOCK_SUFFIXES = (
    "ln_1.weight", "ln_1.bias",
    "attn.c_attn.weight", "attn.c_attn.bias",
    "attn.c_proj.weight", "attn.c_proj.bias",
    "ln_2.weight", "ln_2.bias",
    "mlp.c_fc.weight", "mlp.c_fc.bias",
    "mlp.c_proj.weight", "mlp.c_proj.bias",
)

#: GPT-2 124M as parameter_view sees it: 2 embeddings + 12 x 12 + 2 final = 148.
#: lm_head.weight is absent because the embeddings are tied and
#: named_parameters() deduplicates by identity (scripts/metrics.py:283-289).
GPT2_124M_PARAM_NAMES = (
    ["transformer.wte.weight", "transformer.wpe.weight"]
    + [f"transformer.h.{i}.{suffix}"
       for i in range(12) for suffix in BLOCK_SUFFIXES]
    + ["transformer.ln_f.weight", "transformer.ln_f.bias"]
)


def _code_of(func_name: str) -> str:
    """The source of one function in the ladder, DOCSTRING STRIPPED.

    A raw substring scan cannot tell code from prose, and several docstrings
    here quote the very literals the tests forbid in code -- ladder_steps' says
    "199 and 9535 are not written down here". Parsing and dropping the
    docstring node is what makes "not hardcoded" a claim about the code.
    """
    import ast

    tree = ast.parse((REPO_ROOT / "scripts"
                      / "displacement_ladder.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            body = list(node.body)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body = body[1:]
            if not body:
                raise AssertionError(f"{func_name} has no code, only a docstring")
            return "\n".join(ast.unparse(stmt) for stmt in body)
    raise AssertionError(f"no function named {func_name!r} in the ladder")


def _payload(**over):
    """A weights-only payload, the seven keys scripts/train.py:317-325 writes.

    Mirrors tests/test_injection_point_ruler.py:217-221's helper rather than
    inventing a second shape.
    """
    base = {"kind": "weights_only", "step": 199, "family": "hf_gpt2",
            "model": {"transformer.wte.weight": 1}, "seed": 0, "arm": "twin",
            "micro_batch": 8}
    base.update(over)
    return base


def _full_payload(**over):
    """A full payload: the same seven keys plus the three only 'full' carries."""
    base = _payload(kind="full", step=9535)
    base.update({"optimizer": {"state": {}}, "rng": {"python": None},
                 "rng_digest": "deadbeef"})
    base.update(over)
    return base


# ===========================================================================
# Torch-free: the grouping
# ===========================================================================


def test_every_gpt2_124m_parameter_lands_in_exactly_one_bucket():
    """The partition, at the real shape, against names written independently.

    148 parameters in 14 buckets -- one embeddings, twelve blocks of twelve
    parameters each, one final -- so 2 + 144 + 2 = 148. If the grouping ever
    drops or duplicates one, the per-layer L2s stop recombining into the total
    and every per-layer number is about fewer parameters than its label claims.
    """
    grouped = DL.group_parameter_names(GPT2_124M_PARAM_NAMES)

    assert len(GPT2_124M_PARAM_NAMES) == 148
    assert sum(len(v) for v in grouped.values()) == 148
    assert len(grouped) == 1 + 12 + 1
    assert grouped["embeddings"] == ["transformer.wpe.weight",
                                     "transformer.wte.weight"]
    assert grouped["final"] == ["transformer.ln_f.bias",
                               "transformer.ln_f.weight"]
    for i in range(12):
        assert len(grouped[f"block_{i:02d}"]) == 12, (
            f"block {i} should hold HF's twelve per-block parameters")

    flat = [n for members in grouped.values() for n in members]
    assert sorted(flat) == sorted(GPT2_124M_PARAM_NAMES)
    assert len(set(flat)) == len(flat), "a parameter landed in two buckets"


def test_the_block_regex_matches_two_digit_block_indices():
    """THE MUTATION TARGET. `\\d+` not `\\d`, or blocks 10 and 11 fall through.

    A single-digit regex is the bug a careless implementation actually produces,
    and it does not crash: blocks 10 and 11 stop matching, fall through to the
    unknown-name refusal, and a grouping that silently dropped them instead
    would report ten blocks while calling itself per-layer. Mutating `\\d+` to
    `\\d` in scripts/displacement_ladder.py turns this test red -- audited, and
    recorded in implementation-notes.md.
    """
    assert DL.bucket_for("transformer.h.0.ln_1.weight") == "block_00"
    assert DL.bucket_for("transformer.h.9.ln_1.weight") == "block_09"
    assert DL.bucket_for("transformer.h.10.ln_1.weight") == "block_10"
    assert DL.bucket_for("transformer.h.11.mlp.c_proj.bias") == "block_11"

    grouped = DL.group_parameter_names(GPT2_124M_PARAM_NAMES)
    assert "block_10" in grouped and "block_11" in grouped
    assert len([b for b in grouped if b.startswith("block_")]) == 12


def test_buckets_are_ordered_embeddings_then_blocks_in_order_then_final():
    """Report order is meaningful: it is the order activations flow."""
    order = list(DL.group_parameter_names(GPT2_124M_PARAM_NAMES))
    assert order[0] == "embeddings"
    assert order[-1] == "final"
    assert order[1:-1] == [f"block_{i:02d}" for i in range(12)]


def test_an_unrecognised_parameter_name_is_refused_rather_than_dropped():
    with pytest.raises(DL.LadderError, match="matches no bucket"):
        DL.bucket_for("transformer.adapter.0.weight")
    with pytest.raises(DL.LadderError, match="matches no bucket"):
        DL.group_parameter_names(
            GPT2_124M_PARAM_NAMES + ["some.new.tensor"])


def test_a_name_that_only_looks_like_a_block_is_refused():
    """`transformer.h.` with no index, and a lookalike prefix, both refuse."""
    for name in ("transformer.h.weight", "transformer.hx.0.weight",
                 "h.0.ln_1.weight", "transformer.h..weight"):
        with pytest.raises(DL.LadderError, match="matches no bucket"):
            DL.bucket_for(name)


def test_grouping_refuses_an_empty_parameter_set():
    with pytest.raises(DL.LadderError, match="no parameters to group"):
        DL.group_parameter_names([])


# ===========================================================================
# Torch-free: payload validation, both kinds
# ===========================================================================


def test_both_checkpoint_kinds_the_loop_writes_are_accepted():
    """A weights-only payload has seven keys and a full one ten. Both load.

    The step-199 file is weights-only and is a first-class input here, so this
    is the case train.load_checkpoint refuses and this script must not.
    """
    assert DL.validate_payload(_payload(), Path("wo.pt"))["kind"] == "weights_only"
    assert DL.validate_payload(_full_payload(), Path("f.pt"))["kind"] == "full"

    assert set(_payload()) == set(DL.PAYLOAD_BASE_KEYS)
    assert (set(_full_payload()) - set(DL.PAYLOAD_BASE_KEYS)
            == set(DL.PAYLOAD_FULL_ONLY_KEYS))


def test_carries_rng_is_derived_from_the_kind_not_read_from_the_payload():
    """`carries_rng` lives in train_record.json, never in the payload on disk.

    scripts/train.py:334-336 returns it into the run record; reading
    payload['carries_rng'] off disk raises KeyError. So the ladder derives it.
    """
    assert "carries_rng" not in _payload()
    assert "carries_rng" not in _full_payload()


def test_an_unknown_checkpoint_kind_is_refused_loudly():
    with pytest.raises(DL.LadderError, match="names kind 'sharded'"):
        DL.validate_payload(_payload(kind="sharded"), Path("x.pt"))


@pytest.mark.parametrize("missing", DL.PAYLOAD_BASE_KEYS)
def test_a_payload_missing_any_base_key_is_refused(missing):
    payload = _payload()
    del payload[missing]
    with pytest.raises(DL.LadderError, match="missing checkpoint payload key"):
        DL.validate_payload(payload, Path("x.pt"))


def test_a_payload_that_is_not_a_dict_is_refused():
    with pytest.raises(DL.LadderError, match="not a dict"):
        DL.validate_payload([1, 2, 3], Path("x.pt"))


@pytest.mark.parametrize("state", [None, {}, [], "weights", 0])
def test_a_payload_with_no_usable_weights_is_refused(state):
    """Not only None. An empty or non-mapping state passes an `is None` check
    and then raises a raw torch RuntimeError from load_state_dict, which main()
    does not catch -- so the run would end in a traceback rather than a message
    naming the file. Found by an adversarial audit."""
    with pytest.raises(DL.LadderError, match="no usable 'model' state dict"):
        DL.validate_payload(_payload(model=state), Path("x.pt"))


def test_a_payload_with_real_weights_is_accepted():
    """The refusal above must not be so broad that it rejects a real payload."""
    assert DL.validate_payload(_payload(), Path("x.pt"))["kind"] == "weights_only"


def test_a_null_family_is_refused_because_nothing_else_would_notice():
    """Both families reach 124,439,808 parameters, so the count cannot tell
    them apart -- S86's finding, and why this is a hard refusal."""
    with pytest.raises(DL.LadderError, match="family: null"):
        DL.validate_payload(_payload(family=None), Path("x.pt"))


def test_an_unknown_family_is_refused():
    with pytest.raises(DL.LadderError, match="not one of"):
        DL.validate_payload(_payload(family="llama"), Path("x.pt"))


# ===========================================================================
# Torch-free: the no-verdict guard
# ===========================================================================


def test_the_no_verdict_guard_refuses_a_verdict_key_at_any_depth():
    with pytest.raises(DL.LadderError, match="is a VERDICT"):
        DL.assert_no_verdict_keys({"phase1": {"pairs": {
            "effect": {"clears_noise_floor": True}}}})


def test_the_no_verdict_guard_reaches_inside_lists():
    with pytest.raises(DL.LadderError, match="is a VERDICT"):
        DL.assert_no_verdict_keys({"rows": [{"ok": 1}, {"significant": False}]})


def test_the_no_verdict_guard_denies_analysis_pys_own_result_keys():
    """The denylist is checked against what scripts/analysis.py actually emits.

    An independent route rather than a second copy of the same list: the keys
    are read out of analysis.py's source, so a verdict field this script could
    plausibly grow cannot be absent from the denylist without this going red.
    """
    source = (REPO_ROOT / "scripts" / "analysis.py").read_text(encoding="utf-8")
    for key in ("clears_noise_floor", "noise_floor", "significant",
                "p_adjusted", "ordering"):
        assert f'"{key}"' in source, (
            f"{key!r} is no longer an analysis.py output key; the denylist is "
            "guarding against something that has moved")
        assert key in DL.FORBIDDEN_PAYLOAD_KEYS


def test_the_guard_catches_the_ordering_collision_it_actually_caught():
    """`ordering` is analysis.py's arm RANKING, so the pre-registration
    attestation is filed under `ordering_constraint`. The guard caught this on
    its first run; the rename is the fix and this pins it. See S97."""
    with pytest.raises(DL.LadderError, match="is a VERDICT"):
        DL.assert_no_verdict_keys({"ordering": "anything at all"})
    DL.assert_no_verdict_keys({"ordering_constraint": "the section 8.5 text"})


def test_no_phase_reduces_across_layers_or_pairs_on_its_progress_line():
    """A verdict surface the payload guard cannot reach is worse than one it can.

    run_phase2 printed min(cka) across layers per pair, which made stdout an
    aligned one-scalar-per-pair column -- readable as effect-against-floor, and
    invisible to assert_no_verdict_keys because it never entered the payload.
    Found by an adversarial audit, not by this test; the test is what keeps it
    gone.
    """
    for func in ("run_phase1", "run_phase2"):
        body = _code_of(func)
        for reducer in ("min(", "max(", "sorted(", "statistics."):
            assert reducer not in body, (
                f"{func} computes {reducer!r}. A reduction here is a summary "
                "the guard cannot see, because the guard walks the payload and "
                "this would only ever reach stdout.")


def test_no_verdict_surface_reaches_stdout_that_the_guard_cannot_see(
        tmp_path, capsys):
    """The stdout of a full selftest carries no per-pair reduced scalar."""
    pytest.importorskip("transformers")
    assert DL.main(["--selftest", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "lowest_layer_cka" not in out
    # The progress lines report liveness and layer counts, nothing reducible.
    assert "cka and cosine computed" in out


def test_the_no_verdict_guard_passes_a_payload_carrying_only_distances():
    DL.assert_no_verdict_keys({
        "phase1": {"pairs": {"effect": {"l2_total": 1.0,
                                        "l2_by_bucket": {"block_00": 0.5}}}},
        "phase2": {"pairs": {"effect": {"cka": [{"layer": 0, "cka": 0.9}]}}},
        "pair_set": [{"label": "floor_s0_s1"}],
    })


def test_the_written_payload_carries_no_verdict_key():
    """The guard runs on the way to disk, not only when a test calls it."""
    body = _code_of("write_artifacts")
    assert "assert_no_verdict_keys(payload)" in body, (
        "write_artifacts must run the guard before writing, or the guard is "
        "decorative")
    # And before the writes, not after them.
    assert (body.index("assert_no_verdict_keys")
            < body.index("_atomic_write")), (
        "the guard must run BEFORE the files are written")


def test_the_artifacts_are_written_atomically():
    """write_artifacts is called twice over the same two paths -- once after
    phase 1 and once after phase 2 -- so an in-place write that failed partway
    through the second call would destroy the phase-1 artifact the first call
    created. Mirrors scripts/train.py:331-333."""
    body = _code_of("_atomic_write")
    assert "os.replace" in body
    assert ".partial" in body
    assert "os.replace" in _code_of("_atomic_write")
    train = (REPO_ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    assert "os.replace(partial, path)" in train, (
        "the pattern this mirrors has moved in train.py")


# ===========================================================================
# Torch-free: output resolution and derived steps
# ===========================================================================


def test_selftest_refuses_to_write_junk_into_docs_measurements():
    """Junk filed beside real measurements is indistinguishable from one."""
    with pytest.raises(DL.LadderError, match="refuses to write into"):
        DL.resolve_output(REPO_ROOT / "docs" / "measurements", selftest=True)


def test_a_real_run_may_write_into_docs_measurements(tmp_path):
    reportdir, stem = DL.resolve_output(
        REPO_ROOT / "docs" / "measurements", selftest=False)
    assert stem == DL.STEM
    assert reportdir == (REPO_ROOT / "docs" / "measurements").resolve()


def test_the_selftest_stem_says_junk_in_its_name(tmp_path):
    _, stem = DL.resolve_output(tmp_path, selftest=True)
    assert stem == DL.SELFTEST_STEM
    assert "JUNK" in stem and stem != DL.STEM


def test_the_two_ladder_steps_are_derived_from_the_config(tmp_path):
    """199 and 9535 are not written down in the script. Under the shipped
    config they come out of cfg.checkpoint_kind_at and cfg.last_step."""
    cfg = load_config(REPO_ROOT / "configs" / "base.yaml",
                      REPO_ROOT / "configs" / "runs" / "seed00_twin.yaml",
                      outdir=tmp_path, write_provenance=False,
                      family="hf_gpt2")
    steps = DL.ladder_steps(cfg)
    assert steps["injection_step"] == 200
    assert steps["pre_injection_step"] == 199
    assert steps["pre_injection_kind"] == "weights_only"
    assert steps["final_step"] == 9535
    assert steps["final_kind"] == "full"

    # The CODE, with the docstring stripped -- ladder_steps' docstring says
    # "199 and 9535 are not written down here", and a substring scan over the
    # raw text would be tripped by the sentence promising the opposite.
    assert "199" not in _code_of("ladder_steps")
    assert "9535" not in _code_of("ladder_steps")


def test_the_pre_injection_step_moves_when_the_interval_moves(tmp_path):
    """A pilot that changes the checkpoint interval retargets automatically."""
    import dataclasses

    cfg = load_config(REPO_ROOT / "configs" / "base.yaml",
                      REPO_ROOT / "configs" / "runs" / "seed00_twin.yaml",
                      outdir=tmp_path, write_provenance=False,
                      family="hf_gpt2")
    moved = dataclasses.replace(
        cfg, checkpointing=dataclasses.replace(
            cfg.checkpointing, weights_only_interval=125, full_interval=1000))
    assert DL.ladder_steps(moved)["pre_injection_step"] == 124


def test_the_checkpoint_filename_matches_what_the_training_loop_writes():
    """Read the loop's own f-string rather than trusting two copies to agree."""
    train = (REPO_ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    assert 'f"step{step:06d}_{kind}.pt"' in train

    path = DL.checkpoint_path(Path("/runs"), 0, "twin", 199, "weights_only")
    assert path.name == "step000199_weights_only.pt"
    assert path.parent.name == "seed00_twin"
    assert DL.checkpoint_path(
        Path("/runs"), 2, "random-chars", 9535, "full").parent.name \
        == "seed02_random-chars"


#: The v2 labels, in order. Written out here rather than read from
#: DL.PAIR_SPEC_V2 so a change to the spec has to be a change in two places --
#: the spec IS the thing under test.
V2_LABELS = ["zero_check", "effect_random_chars", "effect_fluent_fabricated",
             "arm_vs_arm", "floor_s0_s1", "training_scale"]


def test_the_pair_set_is_the_v2_set_and_the_gate_is_one_of_them():
    assert [label for label, _a, _b, _n in DL.PAIR_SPEC_V2] == V2_LABELS
    assert [p.label for p in DL.build_selftest_pairs("cpu")] == V2_LABELS
    assert "zero_check" in V2_LABELS
    # seed02_twin is gone: v1 had three floor pairs, v2 has one.
    assert sum(1 for lab in V2_LABELS if lab.startswith("floor")) == 1
    assert "floor_s0_s2" not in V2_LABELS and "floor_s1_s2" not in V2_LABELS


def test_the_v2_spec_needs_exactly_four_run_directories_and_not_seed02():
    """seed02_twin was dropped and seed00_fluent-fabricated added. If the spec ever
    asks for seed02_twin again, a v2 output directory will not have it and the
    missing-run refusal fires -- which is correct, but this says why."""
    needed = DL.required_run_dirs()
    assert set(needed) == {"seed00_twin", "seed00_random-chars",
                           "seed00_fluent-fabricated", "seed01_twin"}
    assert "seed02_twin" not in needed
    assert len(needed) == 4


def test_both_injecting_arms_are_paired_against_the_same_twin():
    """The point of the v2 set: two injecting arms at a MATCHED seed, so the
    two effect pairs share their reference and differ only in the arm."""
    by_label = {label: (a, b) for label, a, b, _n in DL.PAIR_SPEC_V2}
    a_rc, b_rc = by_label["effect_random_chars"]
    a_ff, b_ff = by_label["effect_fluent_fabricated"]
    assert a_rc == a_ff == (0, "twin", "final"), "same twin on both sides"
    assert b_rc == (0, "random-chars", "final")
    assert b_ff == (0, "fluent-fabricated", "final")
    # and the arm-vs-arm pair is exactly those two arms, no twin in it
    a_aa, b_aa = by_label["arm_vs_arm"]
    assert {a_aa, b_aa} == {(0, "random-chars", "final"),
                            (0, "fluent-fabricated", "final")}
    assert all(arm != "twin" for _s, arm, _w in (a_aa, b_aa))


def test_distinct_refs_deduplicates_while_keeping_order():
    class P:
        def __init__(self, a, b):
            self.ref_a, self.ref_b = a, b
            self.loader_a = self.loader_b = (lambda: None)
    refs = [r for r, _ in DL.distinct_refs([P("x", "y"), P("x", "z")])]
    assert refs == ["x", "y", "z"]


# ===========================================================================
# Torch: fixtures, fresh and dispersed
# ===========================================================================


def _make_dispersed(seed=0):
    """build_tiny_model already draws gains log-normal and biases normal."""
    C = _load("canonicalize")
    return C.build_tiny_model(seed=seed)


def _make_fresh(seed=0):
    """Gains exactly 1.0, Conv1D biases exactly 0.0.

    Built by RESETTING the dispersed model, so the two fixtures cannot drift
    into being different models that also differ in gains -- the construction
    tests/test_injection_point_ruler.py:75-77 gives its reason for.
    """
    import torch

    C = _load("canonicalize")
    model = C.build_tiny_model(seed=seed)
    with torch.no_grad():
        for ln in C._all_layernorms(model):
            ln.weight.fill_(1.0)
            if getattr(ln, "bias", None) is not None:
                ln.bias.zero_()
        for b in C._conv1d_biases(model):
            b.zero_()
    return model


@pytest.fixture(params=["fresh", "dispersed"])
def model_builder(request):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    return request.param, (_make_fresh if request.param == "fresh"
                           else _make_dispersed)


def test_the_fresh_fixture_is_actually_at_initialization():
    pytest.importorskip("transformers")
    C = _load("canonicalize")
    model = _make_fresh()
    gains = [g for ln in C._all_layernorms(model)
             for g in ln.weight.detach().flatten().tolist()]
    biases = [b for t in C._conv1d_biases(model)
              for b in t.detach().flatten().tolist()]
    assert gains and biases
    assert all(g == 1.0 for g in gains)
    assert all(b == 0.0 for b in biases)


def test_the_dispersed_fixture_is_actually_dispersed():
    """If this passes trivially the two fixtures are one model and every
    parametrized test below runs twice for nothing."""
    pytest.importorskip("transformers")
    C = _load("canonicalize")
    model = _make_dispersed()
    gains = [g for ln in C._all_layernorms(model)
             for g in ln.weight.detach().flatten().tolist()]
    biases = [b for t in C._conv1d_biases(model)
              for b in t.detach().flatten().tolist()]
    assert any(g != 1.0 for g in gains), "gains are all 1.0 -- not dispersed"
    assert any(b != 0.0 for b in biases), "biases are all 0.0 -- not dispersed"


# ===========================================================================
# Torch: per-layer L2
# ===========================================================================


def test_per_layer_l2_of_a_model_against_itself_is_exactly_zero(model_builder):
    name, build = model_builder
    M = _load("metrics")
    import copy

    a = build(1)
    b = copy.deepcopy(a)
    result = DL.per_layer_l2(M.parameter_view(a), M.parameter_view(b))

    assert result["l2_total"] == 0.0, f"{name}: total"
    for bucket, cell in result["l2_by_bucket"].items():
        assert cell["l2"] == 0.0, f"{name}: bucket {bucket}"


def test_per_layer_l2_parts_recombine_into_the_total(model_builder):
    """sqrt(sum of squared parts) == total. The partition, checked in floats."""
    name, build = model_builder
    M = _load("metrics")
    import torch

    a = build(2)
    b = build(2)
    gen = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for p in b.parameters():
            p.add_(torch.randn(p.shape, generator=gen) * 1e-2)

    result = DL.per_layer_l2(M.parameter_view(a), M.parameter_view(b))
    check = result["sum_of_parts_check"]
    assert result["l2_total"] > 0.0, f"{name}: nothing to recombine"
    assert check["relative_difference"] <= DL.SUM_OF_PARTS_RTOL
    recombined = sum(c["l2"] ** 2
                     for c in result["l2_by_bucket"].values()) ** 0.5
    assert recombined == pytest.approx(result["l2_total"], rel=1e-12)


def test_per_layer_l2_localises_a_change_to_the_block_that_changed(model_builder):
    """The strongest responsiveness case: a PREDICTABLE direction.

    Zero the last block and every earlier bucket must read exactly 0.0 while
    that block reads more than zero. A per-layer metric that moved everywhere
    would pass a weaker 'something changed' test while being just as broken --
    and on the fresh fixture the biases are already 0.0 on both sides, which is
    exactly the degeneracy that makes a weaker test vacuous.
    """
    name, build = model_builder
    M = _load("metrics")
    import copy
    import torch

    a = build(3)
    b = copy.deepcopy(a)
    n_blocks = len(a.transformer.h)
    with torch.no_grad():
        for p in b.transformer.h[-1].parameters():
            p.zero_()

    result = DL.per_layer_l2(M.parameter_view(a), M.parameter_view(b))
    last = f"block_{n_blocks - 1:02d}"
    assert result["l2_by_bucket"][last]["l2"] > 0.0, (
        f"{name}: the zeroed block must move")
    for bucket, cell in result["l2_by_bucket"].items():
        if bucket != last:
            assert cell["l2"] == 0.0, (
                f"{name}: {bucket} precedes the zeroed block and must be "
                "untouched; a metric that moves everywhere is as broken as one "
                "that never does")


def test_per_layer_l2_counts_a_tied_embedding_once(model_builder):
    """parameter_view deduplicates; state_dict would not.

    Using state_dict here would double-weight the embedding, so the bucket
    element count is the check: the embeddings bucket must hold wte and wpe
    once each and lm_head.weight must not appear at all.
    """
    _name, build = model_builder
    M = _load("metrics")

    view = M.parameter_view(build(4))
    assert not any(n.startswith("lm_head") for n in view)
    grouped = DL.group_parameter_names(view.keys())
    assert len(grouped["embeddings"]) == 2


def test_the_sum_of_parts_check_fires_when_a_bucket_is_dropped(monkeypatch,
                                                              model_builder):
    """Proves the recombination check is not vacuous.

    A grouping that silently omits a bucket is the failure the check exists for,
    so it is injected here rather than argued about.
    """
    _name, build = model_builder
    M = _load("metrics")
    import torch

    a = build(5)
    b = build(5)
    gen = torch.Generator().manual_seed(13)
    with torch.no_grad():
        for p in b.parameters():
            p.add_(torch.randn(p.shape, generator=gen) * 1e-2)

    real = DL.group_parameter_names

    def dropping(names):
        grouped = real(names)
        grouped.pop("embeddings")
        return grouped

    monkeypatch.setattr(DL, "group_parameter_names", dropping)
    with pytest.raises(DL.LadderError, match="do not recombine into the total"):
        DL.per_layer_l2(M.parameter_view(a), M.parameter_view(b))


def test_per_layer_l2_refuses_two_models_with_different_parameter_sets():
    M = _load("metrics")
    pytest.importorskip("torch")
    import torch

    view = {"transformer.wte.weight": torch.zeros(2)}
    other = {"transformer.wpe.weight": torch.zeros(2)}
    with pytest.raises(DL.LadderError, match="parameter names differ"):
        DL.per_layer_l2(view, other)


def test_grouping_shape_reports_counts_that_sum_to_the_total(model_builder):
    _name, build = model_builder
    M = _load("metrics")

    shape = DL.grouping_shape(M.parameter_view(build(6)))
    assert sum(shape["n_params_by_bucket"].values()) == shape["total_params"]
    assert sum(shape["n_elements_by_bucket"].values()) == shape["total_elements"]
    assert shape["n_buckets"] == len(shape["buckets"])
    assert shape["counts_cross_checked"]


def test_grouping_shape_cross_checks_rather_than_merely_reporting(monkeypatch,
                                                                 model_builder):
    """The totals and the parts come from two independent sources, and
    grouping_shape compares them itself rather than leaving it to a test.

    Proven by injecting a grouping that drops a bucket: reporting both numbers
    without comparing them is how a table whose rows all agree reads as a
    measurement (S69).
    """
    _name, build = model_builder
    M = _load("metrics")
    real = DL.group_parameter_names

    def dropping(names):
        grouped = real(names)
        grouped.pop(next(iter(grouped)))
        return grouped

    monkeypatch.setattr(DL, "group_parameter_names", dropping)
    with pytest.raises(DL.LadderError, match="counts sum to"):
        DL.grouping_shape(M.parameter_view(build(6)))


def test_the_sum_of_parts_check_states_what_it_is_blind_to(model_builder):
    """It cannot see a parameter MOVED between buckets, and says so.

    An immunity claim broader than its mechanism stops the next reader looking
    (implementation-notes.md:1396-1403). Moving a parameter leaves both the
    total and the sum of squares unchanged, so the count assertions are what
    constrain that -- and the artifact has to say which is which.
    """
    _name, build = model_builder
    M = _load("metrics")
    import torch

    a, b = build(7), build(7)
    gen = torch.Generator().manual_seed(17)
    with torch.no_grad():
        for p in b.parameters():
            p.add_(torch.randn(p.shape, generator=gen) * 1e-2)
    check = DL.per_layer_l2(M.parameter_view(a),
                            M.parameter_view(b))["sum_of_parts_check"]
    assert "blind to a mis-bucket" in check["WHAT_IT_PROVES"]
    assert "per-bucket parameter COUNT" in check["WHAT_IT_PROVES"]
    assert "VACUOUS_ON_AN_IDENTICAL_PAIR" in check


def test_repo_state_reports_dirty_as_unknown_when_git_status_fails(monkeypatch):
    """An unchecked `git status` whose call failed has empty stdout, which reads
    as a clean tree. `dirty: None` is not the same fact as `dirty: False`."""
    import subprocess as sp

    real = sp.run

    def fake(args, **kw):
        if "status" in args:
            return sp.CompletedProcess(args, 1, "", "fatal: not a git repo")
        return real(args, **kw)

    monkeypatch.setattr(DL.subprocess, "run", fake)
    state = DL.repo_state()
    assert state["dirty"] is None, "a failed status call must not read as clean"
    assert state["commit"], "the commit was still obtainable"
    assert "dirty" not in state or state["dirty"] is None
    # git's own message is preferred when it said anything.
    assert state["error"] == "fatal: not a git repo"


def test_repo_state_says_unknown_in_its_own_words_when_git_is_silent(monkeypatch):
    """The fallback branch, when the failed call printed nothing to stderr."""
    import subprocess as sp

    real = sp.run

    def fake(args, **kw):
        if "status" in args:
            return sp.CompletedProcess(args, 1, "", "")
        return real(args, **kw)

    monkeypatch.setattr(DL.subprocess, "run", fake)
    state = DL.repo_state()
    assert state["dirty"] is None
    assert "UNKNOWN, not clean" in state["error"]


def test_trained_at_carries_the_dirty_flag_not_only_the_commit(tmp_path):
    """A bare commit hash implies the tree was clean. burst/config.py records
    the flag because a dirty-tree run stamps a hash that does not describe the
    code that produced it; reporting the hash alone would launder that."""
    (tmp_path / "run_provenance.yaml").write_text(
        "git:\n  commit: abc123\n  dirty: true\n  dirty_files:\n    - a.py\n",
        encoding="utf-8")
    trained = DL._trained_at(tmp_path / "step000199_weights_only.pt")
    assert trained["commit"] == "abc123"
    assert trained["dirty"] is True
    assert trained["n_dirty_files"] == 1


def test_trained_at_is_absent_rather_than_guessed_when_there_is_no_record(
        tmp_path):
    assert DL._trained_at(tmp_path / "step000199_weights_only.pt") is None


# ===========================================================================
# Torch: the objective gate (S97)
# ===========================================================================


def test_two_ahead_loss_matches_the_formulation_in_the_defect_document():
    """Transcribed from quantity (c), not re-derived: shortened input, then
    logits[:, :-1] against raw[:, 2:].

    Validated against the direction a correctly-trained model must show. On
    public GPT-2 the two-ahead objective is far WORSE than next-token; if this
    ever inverts, the scorer is wrong, not the model.
    """
    pytest.importorskip("transformers")
    M = _load("metrics")
    from transformers import GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    batch = M.load_context_batch(stream=io.StringIO())
    next_token = M.cross_entropy_loss(model, batch)
    two_ahead = DL.two_ahead_loss(model, batch)
    del model

    assert two_ahead > next_token, (
        "public GPT-2 was trained on next-token prediction, so the two-ahead "
        "objective must fit it worse")
    assert two_ahead - next_token > 1.0, (
        f"expected a large gap on a properly trained model; got "
        f"{two_ahead - next_token:.4f}")
    assert (DL.classify_objective(next_token, two_ahead)["objective"]
            == DL.OBJECTIVE_NEXT_TOKEN)


def test_two_ahead_loss_refuses_a_batch_too_short_to_express_it():
    pytest.importorskip("torch")
    M = _load("metrics")
    tiny = M.Batch(ids=(1, 2), source="s", file_sha256="a",
                   token_sha256="b", tokenizer="t")
    with pytest.raises(DL.LadderError, match="at least 3 tokens"):
        DL.two_ahead_loss(object(), tiny)


def test_classify_objective_calls_v1s_numbers_double_shifted():
    """S97's measured pair: next-token 7.1085 against two-ahead 4.2613."""
    cell = DL.classify_objective(7.1085, 4.2613)
    assert cell["objective"] == DL.OBJECTIVE_TWO_AHEAD
    assert cell["gap_next_minus_two_ahead"] == pytest.approx(2.8472, abs=1e-4)


def test_classify_objective_calls_a_corrected_run_next_token():
    """Asa's calibration target for a fixed run is roughly 3.4-3.8 next-token,
    and the two-ahead objective should fit it much worse."""
    assert (DL.classify_objective(3.6, 9.5)["objective"]
            == DL.OBJECTIVE_NEXT_TOKEN)


def test_classify_objective_is_indeterminate_at_chance():
    """Both losses at chance carries NO information, and calling that a pass
    would let an untrained checkpoint through the gate that exists for it.

    Measured: on ten junk models the bare rule `two_ahead < next_token` was
    true 8 times, with gaps of 0.001 to 0.008 nats. See D32.
    """
    for gap in (0.0, 0.001, 0.008, -0.005, 0.2, -0.2):
        assert (DL.classify_objective(4.17, 4.17 - gap)["objective"]
                == DL.OBJECTIVE_INDETERMINATE), gap


def test_the_double_shift_margin_sits_between_the_two_observed_regimes():
    """Placed against an observed-good and an observed-bad case, the way
    section 8.4's thresholds were. Pinned so widening it is a visible edit."""
    assert DL.DOUBLE_SHIFT_MARGIN_NATS == 0.25
    assert DL.DOUBLE_SHIFT_MARGIN_NATS > 0.008 * 10, "clear of junk noise"
    assert DL.DOUBLE_SHIFT_MARGIN_NATS < 2.8472 / 5, "clear below v1's signal"


def test_the_refused_commit_list_names_the_v1_pilot():
    assert "9aa930d" in DL.REFUSED_TRAINING_COMMITS


def _gate_pairs(metas):
    """Pairs whose loaders return a canned (model, meta), for gate tests."""
    class Fake:
        pass

    pairs = []
    for ref, meta in metas.items():
        def make(m=meta):
            return (Fake(), m)
        pairs.append(DL.Pair(label="zero_check", ref_a=ref, ref_b=ref,
                             loader_a=make, loader_b=make, note="t"))
    return pairs


def _stub_losses(monkeypatch, next_token, two_ahead):
    M = _load("metrics")
    monkeypatch.setattr(M, "cross_entropy_loss", lambda m, b: next_token)
    monkeypatch.setattr(DL, "two_ahead_loss", lambda m, b: two_ahead)
    monkeypatch.setattr(DL, "grouping_shape", lambda v: {})
    monkeypatch.setattr(M, "parameter_view", lambda m: {})


def test_gate_a_aborts_on_a_double_shifted_checkpoint(monkeypatch):
    """Names the checkpoint and BOTH numbers, per the requirement."""
    pytest.importorskip("transformers")
    M = _load("metrics")
    _stub_losses(monkeypatch, 7.1085, 4.2613)
    ref = "seed00_twin/step009535_full.pt"
    metas = {ref: {"seed": 0, "arm": "twin", "trained_at": None}}
    with pytest.raises(DL.LadderError) as caught:
        DL.objective_gate(_gate_pairs(metas), M.tiny_smoke_batch(),
                          io.StringIO(), tolerate_indeterminate=False)
    text = str(caught.value)
    assert "GATE A fired" in text
    assert ref in text
    assert "7.1085" in text and "4.2613" in text
    assert "THE OBJECTIVE GATE FAILED" in text


def test_gate_b_aborts_on_the_refused_training_commit(monkeypatch):
    """Reads a CLAIM about the artifact. Fires even when the losses are clean."""
    pytest.importorskip("transformers")
    M = _load("metrics")
    _stub_losses(monkeypatch, 3.6, 9.5)
    ref = "seed00_twin/step009535_full.pt"
    metas = {ref: {"seed": 0, "arm": "twin",
                   "trained_at": {"commit": "9aa930dcafebabe", "dirty": False}}}
    with pytest.raises(DL.LadderError) as caught:
        DL.objective_gate(_gate_pairs(metas), M.tiny_smoke_batch(),
                          io.StringIO(), tolerate_indeterminate=False)
    text = str(caught.value)
    assert "GATE B fired" in text
    assert "9aa930d" in text
    assert "GATE A fired" not in text, (
        "gate A should have passed here; that is the point of two gates")


def test_the_two_gates_are_independent_and_both_are_reported(monkeypatch):
    """S97 exists because a claim and the artifact can disagree, so neither
    gate substitutes for the other and the record says both ran."""
    pytest.importorskip("transformers")
    M = _load("metrics")
    _stub_losses(monkeypatch, 3.6, 9.5)
    ref = "seed00_twin/step009535_full.pt"
    metas = {ref: {"seed": 0, "arm": "twin",
                   "trained_at": {"commit": "deadbeef", "dirty": False}}}
    gate, checkpoints, losses, _grouping = DL.objective_gate(
        _gate_pairs(metas), M.tiny_smoke_batch(), io.StringIO(),
        tolerate_indeterminate=False)

    assert gate["passed"] is True
    assert "measured" in gate["gate_a"] and "recorded" in gate["gate_b"]
    assert "can disagree" in gate["WHY_BOTH"]
    cell = gate["by_checkpoint"][ref]
    assert cell["gate_a_objective"] == DL.OBJECTIVE_NEXT_TOKEN
    assert cell["gate_b_refused"] is False
    assert cell["gate_b_recorded_commit"] == "deadbeef"
    assert losses[ref]["loss"] == 3.6
    assert losses[ref]["loss_two_ahead"] == 9.5
    assert checkpoints[ref]["arm"] == "twin"


def test_a_real_run_refuses_an_indeterminate_objective(monkeypatch):
    """Tolerated only under --selftest, where the models are declared junk."""
    pytest.importorskip("transformers")
    M = _load("metrics")
    _stub_losses(monkeypatch, 4.17, 4.169)
    ref = "seed00_twin/step009535_full.pt"
    metas = {ref: {"seed": 0, "arm": "twin", "trained_at": None}}

    with pytest.raises(DL.LadderError, match="INDETERMINATE"):
        DL.objective_gate(_gate_pairs(metas), M.tiny_smoke_batch(),
                          io.StringIO(), tolerate_indeterminate=False)
    gate, _c, _l, _g = DL.objective_gate(
        _gate_pairs(metas), M.tiny_smoke_batch(), io.StringIO(),
        tolerate_indeterminate=True)
    assert gate["passed"] is True
    assert gate["tolerate_indeterminate"] is True


def test_the_objective_gate_runs_before_the_zero_check():
    """'Before anything else' -- including the zero check, which loads models."""
    body = _code_of("main")
    assert body.index("objective_gate") < body.index("gate_phase1")


def test_the_selftest_tolerates_indeterminate_only_because_it_is_junk():
    """The tolerance is wired to args.selftest, not to a bare flag anyone could
    pass at a real run."""
    body = _code_of("main")
    assert "tolerate_indeterminate=args.selftest" in body


def test_ancestry_places_every_real_commit_correctly():
    """Asa asked for this in 715dd67: derive from trained_at ancestry rather
    than a name, failing safe to the warning. Ancestry catches a pre-fix commit
    nobody listed; a denylist cannot."""
    assert DL.objective_fix_ancestry("9aa930d")["state"] == DL.ANCESTRY_PREDATES_FIX
    assert DL.objective_fix_ancestry("f015fd0")["state"] == DL.ANCESTRY_PREDATES_FIX
    assert DL.objective_fix_ancestry(DL.OBJECTIVE_FIX_COMMIT)["state"] == DL.ANCESTRY_CONTAINS_FIX
    # 715dd67 is the corrected pilot: the case Asa named as wrongly warned about.
    assert DL.objective_fix_ancestry("715dd67")["state"] == DL.ANCESTRY_CONTAINS_FIX


def test_unresolvable_ancestry_is_unknown_and_never_clean():
    for commit in (None, "", "deadbeefcafebabe"):
        assert DL.objective_fix_ancestry(commit)["state"] == DL.ANCESTRY_UNKNOWN


def test_ancestry_catches_a_pre_fix_commit_that_is_not_on_the_denylist():
    """The whole reason ancestry beats the list."""
    assert not any("f015fd0".startswith(c) for c in DL.REFUSED_TRAINING_COMMITS)
    assert DL.objective_fix_ancestry("f015fd0")["state"] == DL.ANCESTRY_PREDATES_FIX


def test_the_banner_warns_on_unverified_ancestry_without_calling_it_affected():
    """Fail safe to the WARNING. "unverified" and "known bad" are different
    facts and one name for both is this repo logged defect."""
    payload = {
        "pair_set": [{"label": "floor_s0_s1"}],
        "objective_gate": {"by_checkpoint": {
            "a.pt": {"objective": DL.OBJECTIVE_NEXT_TOKEN,
                     "gate_b_refused": False, "gate_b_suspect": True},
        }},
    }
    text = DL.limitation(payload)
    assert "COULD NOT BE VERIFIED" in text
    assert "a.pt" in text
    assert "SHOWN TO BE AFFECTED" not in text, (
        "an unverified checkpoint must not be reported as a proven one")


def test_a_verified_clean_checkpoint_gets_no_warning_at_all():
    payload = {
        "pair_set": [{"label": "floor_s0_s1"}],
        "objective_gate": {"by_checkpoint": {
            "a.pt": {"objective": DL.OBJECTIVE_NEXT_TOKEN,
                     "gate_b_refused": False, "gate_b_suspect": False},
        }},
    }
    text = DL.limitation(payload)
    assert "wrong objective" not in text.lower()
    assert "COULD NOT BE VERIFIED" not in text


def test_gate_b_aborts_on_a_pre_fix_ancestry_even_off_the_denylist(monkeypatch):
    pytest.importorskip("transformers")
    M = _load("metrics")
    _stub_losses(monkeypatch, 3.6, 9.5)
    ref = "seed00_twin/step009535_full.pt"
    metas = {ref: {"seed": 0, "arm": "twin",
                   "trained_at": {"commit": "f015fd0", "dirty": False}}}
    with pytest.raises(DL.LadderError) as caught:
        DL.objective_gate(_gate_pairs(metas), M.tiny_smoke_batch(),
                          io.StringIO(), tolerate_indeterminate=False)
    text = str(caught.value)
    assert "by ANCESTRY" in text
    assert "f015fd0" in text


def test_the_corrected_pilots_commit_passes_gate_b(monkeypatch):
    """715dd67 trained from 3e715a6, which contains the fix."""
    pytest.importorskip("transformers")
    M = _load("metrics")
    _stub_losses(monkeypatch, 3.21, 9.5)
    ref = "seed00_twin/step009535_full.pt"
    metas = {ref: {"seed": 0, "arm": "twin",
                   "trained_at": {"commit": "3e715a6", "dirty": False}}}
    gate, _c, _l, _g = DL.objective_gate(
        _gate_pairs(metas), M.tiny_smoke_batch(), io.StringIO(),
        tolerate_indeterminate=False)
    cell = gate["by_checkpoint"][ref]
    assert cell["gate_b_refused"] is False
    assert cell["gate_b_suspect"] is False
    assert cell["gate_b_ancestry"]["state"] == DL.ANCESTRY_CONTAINS_FIX


# ===========================================================================
# Torch: label integrity
# ===========================================================================


def test_a_label_that_does_not_match_the_resolved_checkpoint_is_refused():
    """The RESOLVED payload, not the intended path. Mislabeling is the S55
    shape: every number computed correctly and filed under the wrong name."""
    pair = DL.Pair(label="effect_fluent_fabricated", ref_a="a", ref_b="b",
                   loader_a=None, loader_b=None, note="n",
                   expect_a=(0, "twin"), expect_b=(0, "fluent-fabricated"))
    good_a = {"seed": 0, "arm": "twin", "path": "/r/seed00_twin/x.pt"}
    wrong_b = {"seed": 0, "arm": "random-chars",
               "path": "/r/seed00_random-chars/x.pt"}
    with pytest.raises(DL.LadderError) as caught:
        DL.verify_pair_identity(pair, good_a, wrong_b)
    text = str(caught.value)
    assert "effect_fluent_fabricated" in text
    assert "fluent-fabricated" in text and "random-chars" in text
    assert "seed00_random-chars" in text


def test_a_label_matching_its_resolved_checkpoint_passes_and_records_both():
    pair = DL.Pair(label="floor_s0_s1", ref_a="a", ref_b="b",
                   loader_a=None, loader_b=None, note="n",
                   expect_a=(0, "twin"), expect_b=(1, "twin"))
    resolved = DL.verify_pair_identity(
        pair,
        {"seed": 0, "arm": "twin", "path": "/r/seed00_twin/x.pt"},
        {"seed": 1, "arm": "twin", "path": "/r/seed01_twin/x.pt"})
    assert resolved["a"]["seed"] == 0 and resolved["a"]["arm"] == "twin"
    assert resolved["b"]["seed"] == 1
    assert resolved["a"]["expected_arm"] == "twin"


def test_a_wrong_seed_is_caught_even_when_the_arm_is_right():
    pair = DL.Pair(label="floor_s0_s1", ref_a="a", ref_b="b",
                   loader_a=None, loader_b=None, note="n",
                   expect_a=(0, "twin"), expect_b=(1, "twin"))
    with pytest.raises(DL.LadderError, match="side b is labelled seed 1"):
        DL.verify_pair_identity(
            pair,
            {"seed": 0, "arm": "twin", "path": "/r/seed00_twin/x.pt"},
            {"seed": 2, "arm": "twin", "path": "/r/seed02_twin/x.pt"})


def test_every_v2_pair_carries_the_identity_its_label_claims():
    """The spec's (seed, arm) per side must match what the label says it is."""
    expected = {
        "zero_check": ((0, "twin"), (0, "twin")),
        "effect_random_chars": ((0, "twin"), (0, "random-chars")),
        "effect_fluent_fabricated": ((0, "twin"), (0, "fluent-fabricated")),
        "arm_vs_arm": ((0, "random-chars"), (0, "fluent-fabricated")),
        "floor_s0_s1": ((0, "twin"), (1, "twin")),
        "training_scale": ((0, "twin"), (0, "twin")),
    }
    for label, a, b, _note in DL.PAIR_SPEC_V2:
        assert (a[0], a[1]) == expected[label][0], label
        assert (b[0], b[1]) == expected[label][1], label
    by_label = {lab: (a, b) for lab, a, b, _n in DL.PAIR_SPEC_V2}
    a, b = by_label["floor_s0_s1"]
    assert a[0] == 0 and b[0] == 1, "the label names seeds 0 and 1"


def test_the_built_pairs_carry_the_expectations_the_spec_declares(tmp_path):
    """Not just the spec: the Pair objects the runner actually uses."""
    runs = tmp_path / "runs"
    for name in DL.required_run_dirs():
        (runs / name).mkdir(parents=True)
    cfg = load_config(REPO_ROOT / "configs" / "base.yaml",
                      REPO_ROOT / "configs" / "runs" / "seed00_twin.yaml",
                      outdir=tmp_path / "cfg", write_provenance=False,
                      family="hf_gpt2")
    pairs, _steps = DL.build_checkpoint_pairs(runs, cfg=cfg, device="cpu")
    by_label = {p.label: p for p in pairs}
    assert by_label["effect_fluent_fabricated"].expect_a == (0, "twin")
    assert by_label["effect_fluent_fabricated"].expect_b == (0, "fluent-fabricated")
    assert by_label["arm_vs_arm"].expect_a == (0, "random-chars")
    assert by_label["floor_s0_s1"].expect_b == (1, "twin")
    for pair in pairs:
        assert pair.expect_a is not None and pair.expect_b is not None


def test_training_scale_spans_two_different_steps():
    """ADDED BECAUSE A MUTATION SURVIVED. Pointing both sides at 'final' turned
    this pair into a duplicate of zero_check -- an L2 of exactly 0.0 filed under
    a label promising early-against-late -- and the whole suite stayed green.

    The identity checks could not catch it: both sides really are (0, 'twin'),
    so the label's seed and arm claim is satisfied. What distinguishes the pair
    is the STEP, which nothing was asserting.
    """
    by_label = {lab: (a, b) for lab, a, b, _n in DL.PAIR_SPEC_V2}
    a, b = by_label["training_scale"]
    assert a[2] == "pre" and b[2] == "final", (
        "training_scale must span the pre-injection checkpoint and the final "
        "one, or it is measuring a model against itself")
    assert a[2] != b[2]


def test_only_the_zero_check_pair_has_identical_sides():
    """Every other pair must differ in seed, arm, or step. A pair whose two
    sides resolve to one file reports 0.0 for a reason that has nothing to do
    with what its label claims."""
    for label, a, b, _note in DL.PAIR_SPEC_V2:
        if label == "zero_check":
            assert a == b, "the gate pair is the same checkpoint twice"
        else:
            assert a != b, (
                f"pair {label!r} has identical sides {a}, so it is a second "
                "zero_check wearing another name")


def test_phase_1_verifies_identity_before_filing_any_number():
    body = _code_of("run_phase1")
    assert "verify_pair_identity" in body
    assert body.index("verify_pair_identity") < body.index("per_layer_l2")


# ===========================================================================
# Torch: missing run directories
# ===========================================================================


def test_a_missing_run_directory_lists_every_path_searched(tmp_path):
    """No fallback to v1's set, and the message names all four paths."""
    (tmp_path / "seed00_twin").mkdir()
    (tmp_path / "seed01_twin").mkdir()
    with pytest.raises(DL.LadderError) as caught:
        DL.build_checkpoint_pairs(tmp_path, cfg=None, device="cpu")
    text = str(caught.value)
    for name in DL.required_run_dirs():
        assert name in text, f"{name} not listed among the paths searched"
    assert "MISSING" in text and "found" in text
    assert "NO FALLBACK" in text


def test_a_v1_run_directory_is_refused_rather_than_partially_measured(tmp_path):
    """A v1 output dir has seed02_twin and no fluent-fabricated. Measuring the subset
    that happens to be present would produce a complete-looking artifact whose
    labels misdescribe it."""
    for name in ("seed00_twin", "seed01_twin", "seed02_twin",
                 "seed00_random-chars"):
        (tmp_path / name).mkdir()
    with pytest.raises(DL.LadderError, match="seed00_fluent-fabricated"):
        DL.build_checkpoint_pairs(tmp_path, cfg=None, device="cpu")


def test_the_missing_run_refusal_names_what_changed_between_v1_and_v2(tmp_path):
    """So the reader is not left guessing which run set they pointed at."""
    (tmp_path / "seed00_twin").mkdir()
    with pytest.raises(DL.LadderError) as caught:
        DL.build_checkpoint_pairs(tmp_path, cfg=None, device="cpu")
    text = str(caught.value)
    assert "seed02_twin was" in text and "dropped" in text
    assert "fluent-fabricated added" in text or "seed00_fluent-fabricated added" in text


# ===========================================================================
# Torch: the zero-check gate
# ===========================================================================


def _junk_pair(kind, seed, device="cpu"):
    """A DL.Pair over junk models, for gate tests."""
    return DL.Pair(
        label="zero_check", ref_a="A", ref_b="B",
        loader_a=lambda: DL.load_junk(kind, seed, 0, device),
        loader_b=lambda: DL.load_junk(kind, seed, 1, device),
        note="test")


def test_the_zero_check_passes_on_a_genuinely_identical_pair():
    pytest.importorskip("transformers")
    M = _load("metrics")
    batch = M.tiny_smoke_batch()
    pair = DL.Pair(
        label="zero_check", ref_a="A", ref_b="B",
        loader_a=lambda: DL.load_junk("identical", 21, 0, "cpu"),
        loader_b=lambda: DL.load_junk("identical", 21, 0, "cpu"),
        note="two independent builds")

    gate1 = DL.gate_phase1(pair, batch)
    assert gate1["passed"] is True
    assert gate1["observed"]["l2_total"] == 0.0
    assert gate1["observed"]["loss_difference"] == 0.0

    gate2 = DL.gate_phase2(pair, batch)
    assert gate2["passed"] is True
    assert all(gate2["observed"]["activations_bitwise_equal"])
    assert gate2["observed"]["cka_max_deviation_from_one"] == 0.0
    assert gate2["observed"]["norm_ratio_max_deviation_from_one"] == 0.0


def test_the_zero_check_gate_fires_when_the_two_loads_differ():
    """The gate is the only proof the runner works, so it must actually fire."""
    pytest.importorskip("transformers")
    M = _load("metrics")
    with pytest.raises(DL.LadderError, match="THE ZERO CHECK FAILED"):
        DL.gate_phase1(_junk_pair("noise", 22), M.tiny_smoke_batch())


def test_the_zero_check_failure_names_the_observed_value():
    """A gate that says only 'failed' cannot be diagnosed."""
    pytest.importorskip("transformers")
    M = _load("metrics")
    with pytest.raises(DL.LadderError) as caught:
        DL.gate_phase1(_junk_pair("independent", 23), M.tiny_smoke_batch())
    text = str(caught.value)
    assert "not exactly 0.0" in text
    assert "OBSERVED:" in text
    assert "l2_total" in text


def test_the_phase_2_gate_fires_on_differing_activations():
    pytest.importorskip("transformers")
    M = _load("metrics")
    with pytest.raises(DL.LadderError, match="THE ZERO CHECK FAILED"):
        DL.gate_phase2(_junk_pair("independent", 24), M.tiny_smoke_batch())


def test_the_cosine_identity_tolerance_is_pinned_and_tiny():
    """It is 8 ulps of double, and it sits behind four exact checks.

    Pinned so a later widening is a visible edit. The measured deviation at
    real scale was 3 ulps; see the module docstring.
    """
    assert DL.COSINE_IDENTITY_ULPS == 8
    assert DL.COSINE_IDENTITY_TOL == pytest.approx(1.776e-15, rel=1e-3)
    assert DL.COSINE_IDENTITY_TOL < 1e-14


# ===========================================================================
# Torch: the activation tap
# ===========================================================================


class _NotAGPT2:
    """A model with no .transformer, to drive the tap refusal."""

    def named_children(self):
        return iter([("encoder", object()), ("head", object())])


def test_a_tap_failure_reports_the_actual_module_structure():
    """'Failed' is not actionable; the structure that was there is."""
    pytest.importorskip("torch")
    with pytest.raises(DL.LadderError) as caught:
        DL.tap_or_raise(_NotAGPT2())
    text = str(caught.value)
    assert "THE STRUCTURE IT ACTUALLY FOUND" in text
    assert "_NotAGPT2" in text
    assert "encoder" in text and "head" in text
    assert '"has_transformer": false' in text


def test_describe_module_structure_finds_the_blocks_on_a_real_model(
        model_builder):
    _name, build = model_builder
    info = DL.describe_module_structure(build(7))
    assert info["has_transformer"] is True
    assert info["has_drop"] and info["has_h"] and info["has_ln_f"]
    assert info["n_blocks"] >= 1
    assert "transformer" in info["top_level_children"]


def test_the_tap_helper_is_exercised_on_the_measurement_path():
    """activations_for goes through hf_gpt2_tap_modules, or the loud tap
    failure above is unreachable in a real run."""
    body = _code_of("activations_for")
    assert "tap_or_raise(model)" in body
    # ast.unparse normalises quoting, so match the value not the spelling.
    assert "'hooks'" in body and "route=" in body


# ===========================================================================
# Torch: phase separation on disk
# ===========================================================================


def test_selftest_runs_end_to_end_and_writes_both_files(tmp_path, capsys):
    pytest.importorskip("transformers")
    assert DL.main(["--selftest", "--out", str(tmp_path)]) == 0

    payload = json.loads(
        (tmp_path / f"{DL.SELFTEST_STEM}.json").read_text(encoding="utf-8"))
    assert (tmp_path / f"{DL.SELFTEST_STEM}.md").is_file()
    assert payload["gate"]["phase1"]["passed"] is True
    assert payload["gate"]["phase2"]["passed"] is True
    assert payload["phase1"]["ok"] is True
    assert payload["phase2"]["ok"] is True
    assert len(payload["phase1"]["pairs"]) == 6
    assert len(payload["phase2"]["pairs"]) == 6
    assert payload["phase1"]["pairs"]["zero_check"]["l2_total"] == 0.0
    for row in payload["phase2"]["pairs"].values():
        assert len(row["cka"]) == row["n_layers"]
        assert len(row["cosine"]) == row["n_layers"]


def test_the_selftest_exercises_identical_noise_and_independent():
    """All three junk-pair kinds, or the selftest proves less than it looks."""
    refs = [p.ref_a + p.ref_b for p in DL.build_selftest_pairs("cpu")]
    joined = " ".join(refs)
    for kind in ("identical", "noise", "independent"):
        assert kind in joined, f"the selftest never builds an {kind} pair"


def test_phase_1_output_survives_a_phase_2_failure(tmp_path, monkeypatch):
    """The requirement stated as a test: phase 2 failing must not cost phase 1.

    The failure is injected at the activation call, which is where a tap
    refusal or an out-of-memory would actually land.
    """
    pytest.importorskip("transformers")

    def refuse(*_a, **_k):
        raise DL.LadderError("INJECTED phase 2 failure")

    monkeypatch.setattr(DL, "activations_for", refuse)
    # NOT 2: argparse already exits 2 for a bad command line, and one code for
    # both "nothing was written" and "phase 1 is on disk" cannot be acted on.
    assert DL.main(["--selftest", "--out", str(tmp_path)]) \
        == DL.EXIT_PHASE2_FAILED
    assert DL.EXIT_PHASE2_FAILED != 2

    payload = json.loads(
        (tmp_path / f"{DL.SELFTEST_STEM}.json").read_text(encoding="utf-8"))
    assert payload["phase1"]["ok"] is True
    assert len(payload["phase1"]["pairs"]) == 6
    assert payload["phase1"]["pairs"]["zero_check"]["l2_total"] == 0.0
    assert payload["per_model_loss"], "phase 1's loss section must survive"

    assert payload["phase2"]["ok"] is False
    assert payload["phase2"]["attempted"] is True
    assert "INJECTED phase 2 failure" in payload["phase2"]["reason"]
    assert payload["phase2"]["pairs"] == {}
    assert (tmp_path / f"{DL.SELFTEST_STEM}.md").is_file()
    # No .partial debris survives an atomic write.
    assert not list(tmp_path.glob("*.partial"))


def test_a_phase_2_gate_failure_is_recorded_as_failed_not_as_not_reached(
        tmp_path, monkeypatch):
    """A gate that RAN AND FAILED must not be reported as one that never ran.

    payload["gate"]["phase2"] used to be assigned only on success, so the report
    called a fired gate "NOT REACHED". Found by an adversarial audit.
    """
    pytest.importorskip("transformers")

    def refuse(pair, batch):
        raise DL.LadderError("THE ZERO CHECK FAILED (phase 2) INJECTED")

    monkeypatch.setattr(DL, "gate_phase2", refuse)
    assert DL.main(["--selftest", "--out", str(tmp_path)]) \
        == DL.EXIT_PHASE2_FAILED

    payload = json.loads(
        (tmp_path / f"{DL.SELFTEST_STEM}.json").read_text(encoding="utf-8"))
    gate2 = payload["gate"]["phase2"]
    assert gate2["passed"] is False
    assert "INJECTED" in gate2["reason"]
    assert "phase 2 only" in gate2["WHAT_THIS_COSTS"]

    text = (tmp_path / f"{DL.SELFTEST_STEM}.md").read_text(encoding="utf-8")
    assert "phase2                 FAILED" in text
    assert "NOT REACHED" not in text.split("CHECKPOINTS")[0]
    assert payload["phase1"]["ok"] is True


def test_a_phase_2_gate_failure_does_not_condemn_phase_1_in_its_wording():
    """The unscoped text said "no other number it produced can be trusted",
    which was written into the artifact directly beneath phase 1 published as
    ok. Scoped per phase now."""
    assert "Nothing is written" in DL._GATE_SCOPE["phase 1"]
    assert "Phase 1 passed its own gate" in DL._GATE_SCOPE["phase 2"]
    assert "numbers stand" in DL._GATE_SCOPE["phase 2"]


def test_any_exception_type_in_phase_2_still_leaves_phase_1_on_disk(
        tmp_path, monkeypatch):
    """The handler is not a type denylist. A KeyError or a TypeError must not
    escape and leave the artifact asserting phase 2 was never attempted."""
    pytest.importorskip("transformers")

    for exc_type in (KeyError, TypeError, AttributeError, ZeroDivisionError):
        out = tmp_path / exc_type.__name__
        def refuse(*_a, _e=exc_type, **_k):
            raise _e("INJECTED")
        monkeypatch.setattr(DL, "activations_for", refuse)
        assert DL.main(["--selftest", "--out", str(out)]) \
            == DL.EXIT_PHASE2_FAILED
        payload = json.loads(
            (out / f"{DL.SELFTEST_STEM}.json").read_text(encoding="utf-8"))
        assert payload["phase1"]["ok"] is True, exc_type.__name__
        assert payload["phase2"]["attempted"] is True
        assert payload["phase2"]["ok"] is False
        assert exc_type.__name__ in payload["phase2"]["reason"]


def test_a_dead_output_stream_does_not_cost_the_artifact(tmp_path):
    """`... | head` is the ordinary way to look at a long run, and a closed
    stdout makes print raise OSError.

    Before this was guarded, the OSError escaped from a progress print inside
    run_phase1 -- outside the phase-2 handler and outside the LadderError
    handler -- and the process died having written nothing at all. A refuter
    found it while confirming the phase-2 denylist fix: widening that one
    handler was not enough, because the failure was upstream of it.
    """
    pytest.importorskip("transformers")

    class Dead:
        """A stream that is already closed, like the read end of `| head`."""

        def write(self, _text):
            raise OSError(22, "Invalid argument")

        def flush(self):
            raise OSError(22, "Invalid argument")

    import contextlib
    with contextlib.redirect_stdout(Dead()):
        rc = DL.main(["--selftest", "--out", str(tmp_path)])
    assert rc == DL.EXIT_OK

    payload = json.loads(
        (tmp_path / f"{DL.SELFTEST_STEM}.json").read_text(encoding="utf-8"))
    assert payload["phase1"]["ok"] is True
    assert payload["phase2"]["ok"] is True
    assert len(payload["phase1"]["pairs"]) == 6
    assert len(payload["phase2"]["pairs"]) == 6
    assert (tmp_path / f"{DL.SELFTEST_STEM}.md").is_file()
    assert not list(tmp_path.glob("*.partial"))


def test_every_diagnostic_print_goes_through_the_stream_guard():
    """One unguarded progress print is enough to lose the whole record."""
    source = (REPO_ROOT / "scripts"
              / "displacement_ladder.py").read_text(encoding="utf-8")
    unguarded = [line.strip() for line in source.splitlines()
                 if "print(" in line and "file=stream" in line]
    # Exactly one: the guarded print inside _say itself.
    assert len(unguarded) == 1, unguarded
    assert "print(text, file=stream, flush=True)" in unguarded[0]
    assert "except OSError" in _code_of("_say")


def test_the_artifact_write_itself_is_not_silenced():
    """_say is for diagnostics only. A failure to WRITE must stay fatal --
    swallowing it would produce a run that reports success and records nothing.
    """
    body = _code_of("write_artifacts")
    assert "_atomic_write" in body
    # The writes are bare calls, not wrapped in the diagnostic guard.
    assert "_say(stream, _atomic_write" not in body
    assert "try" not in _code_of("_atomic_write")


def test_the_tap_check_runs_before_the_route_cross_check():
    """Otherwise the loud structure dump is unreachable from main.

    cross_check_activation_routes calls hf_gpt2_tap_modules itself and raises a
    bare MetricsError naming no structure, so on a model the tap cannot read it
    always got there first. Found by an adversarial audit.
    """
    body = _code_of("gate_phase2")
    assert body.index("tap_or_raise") < body.index("cross_check_activation_routes")


def test_load_reference_frees_the_payload_before_building_the_model():
    """A full checkpoint carries AdamW's two moment buffers -- ~995 MB at 124M
    fp32 that no metric here reads. Holding the payload across build_model would
    keep them resident while another ~497 MB of weights is allocated."""
    body = _code_of("load_reference")
    assert body.index("del payload") < body.index("SEAM.build_model"), (
        "the payload must be released before the model is built")


def test_phase_1_is_written_before_phase_2_is_attempted(tmp_path, monkeypatch):
    """Not merely restored after a failure -- written first.

    A hard exit during phase 2 leaves no chance to write anything, so the
    ordering is what makes phase 1 durable. Proven by asserting the file
    already exists at the moment the phase-2 gate is first called.
    """
    pytest.importorskip("transformers")
    seen = {}
    real_gate = DL.gate_phase2

    def spy(pair, batch):
        seen["json_exists"] = (tmp_path / f"{DL.SELFTEST_STEM}.json").is_file()
        seen["payload"] = json.loads(
            (tmp_path / f"{DL.SELFTEST_STEM}.json").read_text(encoding="utf-8"))
        return real_gate(pair, batch)

    monkeypatch.setattr(DL, "gate_phase2", spy)
    assert DL.main(["--selftest", "--out", str(tmp_path)]) == 0
    assert seen["json_exists"] is True
    assert seen["payload"]["phase1"]["ok"] is True
    assert seen["payload"]["phase2"]["attempted"] is False


def test_a_failing_zero_check_abandons_the_run_and_writes_nothing(tmp_path,
                                                                 monkeypatch):
    """The gate aborts rather than reporting alongside a warning."""
    pytest.importorskip("transformers")

    def broken(pair, batch):
        raise DL.LadderError("THE ZERO CHECK FAILED (phase 1) INJECTED")

    monkeypatch.setattr(DL, "gate_phase1", broken)
    assert DL.main(["--selftest", "--out", str(tmp_path)]) == 1
    assert not (tmp_path / f"{DL.SELFTEST_STEM}.json").exists()
    assert not (tmp_path / f"{DL.SELFTEST_STEM}.md").exists()


# ===========================================================================
# Torch: loading real checkpoint files from disk, both kinds
# ===========================================================================


def _write_checkpoint(path, model, *, kind, step, arm="twin", seed=0):
    """A payload written the way scripts/train.py:317-329 writes one."""
    import torch

    payload = {"kind": kind, "step": step, "family": "hf_gpt2",
               "model": model.state_dict(), "seed": seed, "arm": arm,
               "micro_batch": 8}
    if kind == "full":
        payload["optimizer"] = {"state": {}, "param_groups": []}
        payload["rng"] = {"python": None}
        payload["rng_digest"] = "0" * 64
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def test_both_kinds_round_trip_from_disk(tmp_path, monkeypatch, model_builder):
    """The step-199 weights-only file and a step-9535 full file both load.

    SEAM.build_model is stubbed to the tiny model: the real one builds GPT-2
    124M and checks the parameter count, which is a gigabyte per copy and not
    what this test is about.
    """
    _name, build = model_builder
    model = build(8)
    monkeypatch.setattr(DL.SEAM, "build_model", lambda cfg, family: build(8))

    wo = tmp_path / "seed00_twin" / "step000199_weights_only.pt"
    full = tmp_path / "seed00_twin" / "step009535_full.pt"
    _write_checkpoint(wo, model, kind="weights_only", step=199)
    _write_checkpoint(full, model, kind="full", step=9535)

    for path, kind, step, carries in ((wo, "weights_only", 199, False),
                                      (full, "full", 9535, True)):
        loaded, meta = DL.load_reference(path, cfg=None, device="cpu")
        assert meta["kind"] == kind
        assert meta["step"] == step
        assert meta["carries_rng"] is carries
        assert meta["arm"] == "twin" and meta["seed"] == 0
        assert len(meta["model_sha256"]) == 64
        del loaded


def test_the_two_kinds_hash_identically_when_the_weights_are_identical(
        tmp_path, monkeypatch):
    """THE TRAP the pilot record names, turned into a test.

    sha256 of the FILE differs between two checkpoints holding identical
    weights, because `arm` is stored in the payload beside them
    (scripts/train.py:322-323). The ladder hashes payload['model'] only, so
    identical weights hash identically no matter what arm or kind wraps them.
    """
    pytest.importorskip("transformers")
    import hashlib

    model = _make_dispersed(9)
    monkeypatch.setattr(DL.SEAM, "build_model",
                        lambda cfg, family: _make_dispersed(9))

    twin = tmp_path / "seed00_twin" / "step000199_weights_only.pt"
    arm = tmp_path / "seed00_random-chars" / "step000199_weights_only.pt"
    _write_checkpoint(twin, model, kind="weights_only", step=199, arm="twin")
    _write_checkpoint(arm, model, kind="weights_only", step=199,
                      arm="random-chars")

    _, meta_twin = DL.load_reference(twin, cfg=None, device="cpu")
    _, meta_arm = DL.load_reference(arm, cfg=None, device="cpu")

    assert meta_twin["model_sha256"] == meta_arm["model_sha256"], (
        "identical weights must hash identically; the ladder hashes "
        "payload['model'], not the file")
    file_twin = hashlib.sha256(twin.read_bytes()).hexdigest()
    file_arm = hashlib.sha256(arm.read_bytes()).hexdigest()
    assert file_twin != file_arm, (
        "if the FILES ever hash the same, the trap this test pins has gone "
        "away and the docstring is stale")


def test_a_refused_payload_on_disk_names_the_file(tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    import torch

    path = tmp_path / "bad.pt"
    torch.save({"kind": "weights_only", "step": 1}, path)
    with pytest.raises(DL.LadderError, match="missing checkpoint payload key"):
        DL.load_reference(path, cfg=None, device="cpu")


def test_state_sha256_is_order_independent_but_value_sensitive():
    pytest.importorskip("torch")
    import torch

    a = {"x": torch.ones(3), "y": torch.zeros(2)}
    b = {"y": torch.zeros(2), "x": torch.ones(3)}
    assert DL.state_sha256(a) == DL.state_sha256(b)

    c = {"x": torch.ones(3), "y": torch.zeros(2) + 1e-9}
    assert DL.state_sha256(a) != DL.state_sha256(c)


@pytest.mark.parametrize("dtype_name", ["float32", "float64", "bfloat16",
                                        "float16", "int64", "uint8", "bool"])
def test_state_sha256_handles_every_dtype_a_checkpoint_can_carry(dtype_name):
    """bf16 and fp16 have no numpy equivalent, so the uint8 cast must come first."""
    pytest.importorskip("torch")
    import torch

    tensor = torch.ones(4, dtype=getattr(torch, dtype_name))
    assert len(DL.state_sha256({"x": tensor})) == 64


def test_state_sha256_handles_a_zero_dim_buffer():
    """REGRESSION. `view(torch.uint8)` raises on a 0-dim tensor, and a
    state_dict carries buffers as well as parameters -- HF GPT-2 has shipped a
    0-dim `attn.masked_bias`, so this crashed on a real payload before
    `reshape(-1)` was put in front of the cast."""
    pytest.importorskip("torch")
    import torch

    assert len(DL.state_sha256({"scalar": torch.tensor(1.5)})) == 64
    assert (DL.state_sha256({"s": torch.tensor(1.5)})
            != DL.state_sha256({"s": torch.tensor(2.5)}))


def test_state_sha256_separates_shape_and_dtype_from_bytes():
    """Identical bytes must not collide across a reshape or a reinterpretation."""
    pytest.importorskip("torch")
    import torch

    assert (DL.state_sha256({"x": torch.ones(2, 3)})
            != DL.state_sha256({"x": torch.ones(3, 2)}))
    same_bytes = torch.ones(4, dtype=torch.float32)
    assert (DL.state_sha256({"x": same_bytes})
            != DL.state_sha256({"x": same_bytes.view(torch.int32)}))


def test_state_sha256_survives_a_non_contiguous_and_an_empty_tensor():
    pytest.importorskip("torch")
    import torch

    assert len(DL.state_sha256({"t": torch.ones(4, 4).t()})) == 64
    assert len(DL.state_sha256({"e": torch.zeros(0)})) == 64


def test_the_loss_pass_releases_the_parameter_view_with_the_model():
    """parameter_view aliases the model's storage, so a live view pins the
    weights. Holding one past `del model` would make the loss pass carry two
    checkpoints' weights instead of one.

    The loss pass moved into objective_gate in v2: it already walks every
    distinct checkpoint to score both objectives, so recomputing the loss in
    phase 1 would have been the only place reading a checkpoint twice for one
    quantity.
    """
    body = _code_of("objective_gate")
    assert "del model, view" in body, (
        "the parameter view must be released alongside the model, or the "
        "one-model ceiling on the loss pass is not real")
    assert "cross_entropy_loss" not in _code_of("run_phase1"), (
        "phase 1 must not recompute a loss the objective gate already took")


# ===========================================================================
# Device handling
# ===========================================================================


def test_cuda_is_refused_rather_than_silently_downgraded(monkeypatch):
    """A number labelled cuda that ran on cpu is mislabelled, not just slower."""
    pytest.importorskip("torch")
    import torch

    if torch.cuda.is_available():
        pytest.skip("this host has CUDA; the refusal path is unreachable here")
    with pytest.raises(DL.LadderError, match="Refusing to silently fall back"):
        DL.resolve_device("cuda")


def test_auto_records_what_it_chose_and_why():
    pytest.importorskip("torch")
    record = DL.resolve_device("auto")
    assert record["chosen"] in ("cpu", "cuda")
    assert record["requested"] == "auto"
    assert "cuda_available" in record
    assert "reduction order" in record["NOTE"]


def test_the_batch_is_rewrapped_for_a_device_without_losing_its_identity():
    """metrics.Batch.input_ids() is CPU-only, so the device is threaded in by
    extending Batch. identity() must be untouched, or this measurement stops
    being keyed to the same batch as every other report."""
    pytest.importorskip("transformers")
    M = _load("metrics")
    batch = M.tiny_smoke_batch()

    assert DL.on_device(batch, "cpu") is batch
    wrapped = DL.on_device(batch, "meta")
    assert wrapped.identity() == batch.identity()
    assert wrapped.n_tokens == batch.n_tokens
    assert isinstance(wrapped, M.Batch)


# ===========================================================================
# The artifact
# ===========================================================================


def test_the_report_is_generated_from_the_payload_not_hand_maintained():
    """S70: a hand-written record inside a generated file is deleted by the
    next run, and the result looks machine-clean."""
    payload = {
        "stem": DL.STEM, "LIMITATION": "L", "NO_VERDICT": "N",
        "ordering_constraint": "O", "fixture": "real checkpoints",
        "device": {"chosen": "cpu", "requested": "auto", "cuda_available": False},
        "repo": {"commit": "abc", "dirty": False}, "cka_variant": "v1",
        "batch": {"source": "bursts/context.txt", "n_tokens": 1024,
                  "tokenizer": "gpt2", "token_sha256": "ff"},
        "steps": {}, "gate": {}, "checkpoints": {}, "per_model_loss": {},
        "parameter_grouping": {}, "phase1": {"attempted": True, "ok": True,
                                             "pairs": {}},
        "phase2": {"attempted": False, "ok": False, "reason": "not attempted",
                   "pairs": {}},
        "PROVENANCE": "p",
    }
    text = DL.format_report(payload)
    assert "13. THE DISPLACEMENT LADDER" in text
    assert "=" * 78 in text
    assert "NO VERDICT IS COMPUTED HERE" in text
    assert "PRE-REGISTRATION ORDERING" in text
    assert "PROVENANCE" in text
    assert "bursts/context.txt" in text


def _payload_with_objective(objective, *, refused=False, floor_labels=1):
    """A minimal payload for exercising `limitation`."""
    pair_set = [{"label": "zero_check"}, {"label": "effect_random_chars"}]
    pair_set += [{"label": f"floor_{i}"} for i in range(floor_labels)]
    return {
        "pair_set": pair_set,
        "objective_gate": {"by_checkpoint": {
            "seed00_twin/step009535_full.pt": {
                "objective": objective, "gate_b_refused": refused},
        }},
    }


def test_the_limitation_carries_the_s97_clause_when_a_checkpoint_is_double_shifted():
    """PRESENT branch. The gate aborts on this, so the clause only appears if
    the gate is bypassed -- and a v1 checkpoint reaching the artifact must still
    carry the warning. That is why it is derived and not deleted."""
    text = DL.limitation(_payload_with_objective(DL.OBJECTIVE_TWO_AHEAD))
    low = text.lower()
    assert "wrong objective" in low
    assert "two tokens" in low
    assert "2026-08-05-training-objective-defect.md" in text
    assert "s97" in low
    # The scope, both directions, or the banner over-corrects.
    assert "voids the" in low and "barrier curves" in low
    assert "does not void" in low
    assert "8.4 ruler branch" in low
    # The case a reader is most likely to misread.
    assert "next-token loss on a model that was never trained for it" in low
    # And it names which checkpoints.
    assert "seed00_twin/step009535_full.pt" in text


def test_the_s97_clause_also_fires_on_the_recorded_commit_alone():
    """Gate B is not decoration: a checkpoint refused by commit gets the clause
    even when its measured losses said nothing."""
    text = DL.limitation(_payload_with_objective(
        DL.OBJECTIVE_INDETERMINATE, refused=True))
    assert "wrong objective" in text.lower()


def test_the_limitation_omits_the_s97_clause_on_a_clean_run():
    """ABSENT branch. The clause's absence is a claim about this run's inputs,
    so it has to be able to be absent -- and the rest of the banner must survive
    intact when it is."""
    text = DL.limitation(_payload_with_objective(DL.OBJECTIVE_NEXT_TOKEN))
    low = text.lower()
    assert "wrong objective" not in low
    assert "two tokens" not in low
    assert "2026-08-05-training-objective-defect.md" not in text
    # everything else still there
    assert "computes no verdict" in low
    assert "blind to neuron permutation" in low
    assert "the floor is n=1" in low


def test_the_limitation_is_recomputed_at_write_time_not_stored():
    """S70: hardcoded prose inside a generated file goes stale silently."""
    body = _code_of("write_artifacts")
    # ast.unparse normalises quoting, so match the assignment not the spelling.
    assert "payload['LIMITATION'] = limitation(payload)" in body
    assert body.index("LIMITATION") < body.index("_atomic_write")


def test_the_floor_sentence_states_n_equals_one_and_the_absent_spread():
    """v1 had three floor pairs, v2 has one. A reader carrying the v1 habit of
    reading a spread across floor pairs would be reading one that is not there."""
    text = DL.floor_sentence(_payload_with_objective(
        DL.OBJECTIVE_NEXT_TOKEN, floor_labels=1))
    assert "n=1" in text
    assert "no spread" in text.lower()
    assert "down from three" in text
    for word in ("min", "median", "max"):
        assert word in text.lower()


def test_the_floor_sentence_is_counted_from_the_pair_set_not_typed():
    """Derived, so it cannot disagree with the pair set it describes."""
    three = DL.floor_sentence(_payload_with_objective(
        DL.OBJECTIVE_NEXT_TOKEN, floor_labels=3))
    assert "n=3" in three
    assert "n=1" not in three


def test_the_defect_document_the_limitation_cites_exists():
    """A citation to a file that is not there is worse than no citation."""
    assert (REPO_ROOT / "docs"
            / "2026-08-05-training-objective-defect.md").is_file()


def test_the_limitation_states_every_blindness_it_must():
    text = DL.limitation(_payload_with_objective(DL.OBJECTIVE_NEXT_TOKEN)).lower()
    for phrase in ("blind to neuron permutation", "openwebtext document 73",
                   "1024 token positions", "computes no verdict",
                   "one document"):
        assert phrase in text, phrase


def test_the_limitation_says_the_asymmetry_runs_against_the_floor_pairs():
    """The floor pairs may carry permutation drift the effect pair structurally
    cannot -- the specific thing raw L2 cannot separate here."""
    text = DL.limitation(
        _payload_with_objective(DL.OBJECTIVE_NEXT_TOKEN)).lower()
    assert "structurally cannot carry permutation drift" in text
    assert "independently initialized" in text


def test_the_ordering_attestation_cites_the_sections_it_rests_on():
    for token in ("8.5", "8.7", "12-injection-point-ruler.json",
                  "plain_loss_barrier"):
        assert token in DL.ORDERING_ATTESTATION


def test_provenance_is_recomputed_from_the_payload():
    payload = {"stem": "s", "fixture": "f",
               "device": {"chosen": "cpu", "cuda_available": False},
               "repo": {"commit": "abc123", "dirty": False},
               "batch": {"source": "b", "token_sha256": "t"},
               "phase1": {"pairs": {"a": {}, "b": {}}},
               "phase2": {"ok": True}, "checkpoints": {"x": {}},
               "cka_variant": "v1"}
    text = DL.build_provenance(payload)
    assert "stem=s" in text and "n_pairs=2" in text
    assert "measured_at_commit=abc123" in text
    assert "cosine_identity_ulps=8" in text
    assert "phase2_ok=True" in text
