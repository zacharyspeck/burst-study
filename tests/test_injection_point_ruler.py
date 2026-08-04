"""The section 8.4 injection-point ruler.

docs/preregistration.md section 8.4 selects the study's headline metric from a
single-checkpoint measurement taken at the pilot. This file tests the machinery
that produces it.

TWO FIXTURES, EVERY TEST THAT TOUCHES A MODEL, and that is the load-bearing
part of this file rather than a thoroughness flourish.

A freshly-constructed GPT-2 has every LayerNorm gain at exactly 1.0 and every
Conv1D bias at exactly 0.0. That configuration is not a generic point in weight
space, and twice in this build it made a test vacuous rather than failing: it
let residual rotation pass its equivalence check on a technicality (S42), and
it emptied the head-internal step of its content so an injected mutation fault
was undetectable (S48).

THE STEP-199 MODEL SITS NEAR THAT CONFIGURATION. That is the entire reason
section 8.4's measurement exists -- section 8.3 says so directly: the study
injects roughly 52M tokens into a from-scratch run, where the gains have barely
moved from 1.0, and every number in the committed step 9 table was taken on
fully-trained public GPT-2 instead. A test suite that only ever saw a dispersed
model would be blind in exactly the region this measurement targets, and one
that only ever saw a fresh model would be blind to whether the code works for
any other reason than everything being 1.0.

So `model_fixture` is parametrized over both, and where the two disagree that
is reported rather than reconciled.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import os
import subprocess
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


CE = _load("canonicalization_error")
C = _load("canonicalize")

from burst.config import ARMS, INJECTING_ARMS, load_config  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures: fresh and dispersed, and the difference between them is the point
# ---------------------------------------------------------------------------


def _make_fresh(seed=0):
    """Gains exactly 1.0, every Conv1D bias exactly 0.0.

    `t = 0` of the dispersion sweep, and the state a from-scratch model is
    still near at step 199. Built by RESETTING the dispersed model rather than
    by a separate constructor, so the two fixtures cannot drift into being
    different models that also differ in gains.
    """
    import torch

    model = C.build_tiny_model(seed=seed)
    with torch.no_grad():
        for ln in C._all_layernorms(model):
            ln.weight.fill_(1.0)
            if getattr(ln, "bias", None) is not None:
                ln.bias.zero_()
        for b in C._conv1d_biases(model):
            b.zero_()
    return model.to(dtype=torch.float64)


def _make_dispersed(seed=0):
    """`build_tiny_model` already draws gains log-normal and biases normal.

    That randomisation exists BECAUSE of S42/S48 -- see its docstring. So the
    dispersed fixture is the shipped constructor unchanged, and only the fresh
    one needs building.
    """
    import torch

    return C.build_tiny_model(seed=seed).to(dtype=torch.float64)


@pytest.fixture(params=["fresh", "dispersed"])
def model_fixture(request):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    builder = _make_fresh if request.param == "fresh" else _make_dispersed
    return request.param, builder


def _assert_is_fresh(model):
    """The fresh fixture must actually BE fresh, or it tests nothing."""
    gains = [g for ln in C._all_layernorms(model)
             for g in ln.weight.detach().flatten().tolist()]
    biases = [b for t in C._conv1d_biases(model)
              for b in t.detach().flatten().tolist()]
    assert gains and biases
    assert all(g == 1.0 for g in gains)
    assert all(b == 0.0 for b in biases)


def test_the_fresh_fixture_is_actually_at_initialization():
    pytest.importorskip("transformers")
    _assert_is_fresh(_make_fresh())


def test_the_dispersed_fixture_is_actually_dispersed():
    """If this ever passes trivially, the two fixtures are the same model and
    every parametrized test below is running twice for nothing."""
    pytest.importorskip("transformers")
    model = _make_dispersed()
    gains = [g for ln in C._all_layernorms(model)
             for g in ln.weight.detach().flatten().tolist()]
    biases = [b for t in C._conv1d_biases(model)
              for b in t.detach().flatten().tolist()]
    assert any(g != 1.0 for g in gains), "gains are all 1.0 -- not dispersed"
    assert any(b != 0.0 for b in biases), "biases are all 0.0 -- not dispersed"


# ---------------------------------------------------------------------------
# which checkpoint: derived, never hardcoded
# ---------------------------------------------------------------------------


def _cfg(tmp_path, **edits):
    import yaml

    data = yaml.safe_load(
        (REPO_ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    for dotted, value in edits.items():
        parts = dotted.replace("__", ".").split(".")
        node = data
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    run = tmp_path / "run.yaml"
    run.write_text("seed: 3\narm: twin\n", encoding="utf-8")
    return load_config(base, run, tmp_path / "out", require_complete=False,
                       family="hf_gpt2", write_provenance=False)


def test_the_target_is_step_199_under_the_shipped_config(tmp_path):
    """Section 8.4 names 199, but 199 is the ANSWER, not the rule."""
    cfg = _cfg(tmp_path)
    assert cfg.injection.injection_step == 200
    assert CE.target_checkpoint_step(cfg) == 199


def test_the_target_is_strictly_before_the_injection_step(tmp_path):
    """The property that makes it carry no outcome information."""
    cfg = _cfg(tmp_path)
    assert CE.target_checkpoint_step(cfg) < cfg.injection.injection_step


def test_there_is_no_checkpoint_at_the_injection_step_itself(tmp_path):
    """Why section 8.4 could not have named step 200: none is ever written."""
    cfg = _cfg(tmp_path)
    assert cfg.checkpoint_kind_at(cfg.injection.injection_step) is None


def test_the_target_moves_with_the_interval_rather_than_being_hardcoded(tmp_path):
    """A pilot that changes the interval must retarget automatically.

    Intervals are restricted to exact divisors of `full_interval` -- the loader
    refuses anything else, because the precedence rule when both schedules fire
    on one step would otherwise be undefined. So these are all divisors of 1000.
    """
    assert CE.target_checkpoint_step(
        _cfg(tmp_path, checkpointing__weights_only_interval=40)) == 199
    assert CE.target_checkpoint_step(
        _cfg(tmp_path / "b", checkpointing__weights_only_interval=25)) == 199
    assert CE.target_checkpoint_step(
        _cfg(tmp_path / "c", checkpointing__weights_only_interval=8)) == 199
    # The one that actually moves: 125 fires at 124 and 249, so the last one
    # at or before injection is 124, not 199.
    assert CE.target_checkpoint_step(
        _cfg(tmp_path / "d", checkpointing__weights_only_interval=125)) == 124


def test_no_checkpoint_before_injection_is_refused_not_guessed(tmp_path):
    cfg = _cfg(tmp_path, checkpointing__weights_only_interval=5000,
               checkpointing__full_interval=5000)
    with pytest.raises(CE.InjectionPointError, match="no checkpoint fires"):
        CE.target_checkpoint_step(cfg)


# ---------------------------------------------------------------------------
# refusals -- section 8.5's ordering constraint, enforced rather than promised
# ---------------------------------------------------------------------------


def _payload(**over):
    base = {"kind": "weights_only", "step": 199, "arm": "twin", "seed": 3,
            "family": "hf_gpt2", "model": {}}
    base.update(over)
    return base


@pytest.mark.parametrize("arm", sorted(INJECTING_ARMS))
def test_every_injecting_arm_is_refused(tmp_path, arm):
    """A checkpoint from an arm that injects cannot calibrate the ruler."""
    cfg = _cfg(tmp_path)
    cfg = type(cfg)(**{**{f.name: getattr(cfg, f.name)
                          for f in cfg.__dataclass_fields__.values()},
                       "arm": arm})
    with pytest.raises(CE.InjectionPointError, match="which injects"):
        CE.verify_measurable(_payload(arm=arm), cfg, {}, 199)


def test_twin_is_accepted(tmp_path):
    cfg = _cfg(tmp_path)
    got = CE.verify_measurable(_payload(), cfg, {}, 199)
    assert got == {"arm": "twin", "seed": 3, "step": 199,
                   "family": "hf_gpt2"}


def test_an_off_target_step_is_refused(tmp_path):
    """A step-9535 twin checkpoint carries outcome information."""
    cfg = _cfg(tmp_path)
    with pytest.raises(CE.InjectionPointError, match="is at step 9535"):
        CE.verify_measurable(_payload(step=9535), cfg, {}, 199)


def test_an_earlier_checkpoint_is_also_refused(tmp_path):
    """Not the point the rule names, even though it is pre-injection."""
    cfg = _cfg(tmp_path)
    with pytest.raises(CE.InjectionPointError, match="is at step 149"):
        CE.verify_measurable(_payload(step=149), cfg, {}, 199)


def test_a_null_family_is_refused(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(CE.InjectionPointError, match="family: null"):
        CE.verify_measurable(_payload(family=None), cfg, {}, 199)


def test_an_unknown_arm_is_refused(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(CE.InjectionPointError, match="not one of"):
        CE.verify_measurable(_payload(arm="coherent"), cfg, {}, 199)


def test_a_config_for_a_different_run_is_refused(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(CE.InjectionPointError, match="config given is"):
        CE.verify_measurable(_payload(seed=7), cfg, {}, 199)


@pytest.mark.parametrize("key,value", [("arm", "fluent-false"),
                                       ("seed", 9),
                                       ("family", "probe_linear")])
def test_provenance_disagreement_refuses_rather_than_preferring_either(
        tmp_path, key, value):
    """Neither record is preferred. A directory that contradicts itself is not
    described by picking one half of it."""
    cfg = _cfg(tmp_path)
    provenance = {"arm": "twin", "seed": 3, "family": "hf_gpt2"}
    provenance[key] = value
    with pytest.raises(CE.InjectionPointError, match="disagree"):
        CE.verify_measurable(_payload(), cfg, provenance, 199)


def test_agreeing_provenance_is_accepted(tmp_path):
    cfg = _cfg(tmp_path)
    provenance = {"arm": "twin", "seed": 3, "family": "hf_gpt2"}
    assert CE.verify_measurable(_payload(), cfg, provenance, 199)["arm"] == "twin"


def test_absent_provenance_is_not_an_error(tmp_path):
    """A directory may legitimately hold only the checkpoint."""
    cfg = _cfg(tmp_path)
    assert CE.verify_measurable(_payload(), cfg, {}, 199)["step"] == 199


def test_the_cli_cannot_express_a_two_model_comparison():
    """Section 8.5 in code: there is no second checkpoint argument to pass."""
    parser = CE._build_parser()
    options = {a.dest for a in parser._actions}
    assert "checkpoint" in options
    for forbidden in ("checkpoint_b", "reference", "twin", "against",
                      "compare", "other"):
        assert forbidden not in options


# ---------------------------------------------------------------------------
# spread and median: tested OPERATIONALLY, not by recomputation (S60)
# ---------------------------------------------------------------------------


def _cell(values, epsilon=CE.INJECTION_POINT_EPSILON):
    import statistics

    finite = [v for v in values if v == v and abs(v) != float("inf")]
    return {"epsilon": epsilon, "shape": "isotropic",
            "n_directions": len(values),
            "direction_seeds": list(range(len(values))),
            "by_direction": {str(i): v for i, v in enumerate(values)},
            "min": min(finite) if finite else None,
            "median": statistics.median(finite) if finite else None,
            "max": max(finite) if finite else None}


def test_spread_is_the_ratio_of_largest_to_smallest():
    """Fabricated factors with a spread that is known by construction."""
    assert CE.spread_of(_cell([2.0, 4.0, 3.0])) == pytest.approx(2.0)
    assert CE.spread_of(_cell([1.0] * 10)) == pytest.approx(1.0)
    assert CE.spread_of(_cell([0.5, 50.0])) == pytest.approx(100.0)


def test_spread_is_across_directions_at_one_epsilon_not_across_epsilons():
    """Section 8.4 defines spread ACROSS THE TEN DIRECTIONS. A cell holds one
    epsilon, so there is no axis here for an epsilon sweep to leak in."""
    cell = _cell([1.0, 2.0])
    assert set(cell["by_direction"]) == {"0", "1"}
    assert CE.spread_of(cell) == pytest.approx(2.0)
    assert cell["epsilon"] == CE.INJECTION_POINT_EPSILON


def test_median_is_the_median_of_the_directions():
    assert CE.median_of(_cell([1.0, 5.0, 3.0])) == pytest.approx(3.0)
    assert CE.median_of(_cell([1.0, 3.0])) == pytest.approx(2.0)


@pytest.mark.parametrize("values", [[], [0.0, 1.0], [-1.0, 2.0],
                                    [float("nan"), 1.0],
                                    [float("inf"), 1.0]])
def test_an_undefined_spread_is_none_rather_than_a_number(values):
    """A spread that cannot be computed must reach the branch as unavailable,
    never as a substituted value."""
    assert CE.spread_of(_cell(values) if values else
                        {"by_direction": {}}) is None


def test_the_regression_anchor_reproduces_section_8_4s_arithmetic():
    """Section 8.4's thresholds are calibrated against the committed
    public-GPT-2 cell. If this function does not reproduce the numbers that
    section quotes, the thresholds do not mean what they say."""
    anchor = CE.public_gpt2_anchor()
    assert anchor, "the committed step 9 file is missing"
    assert round(anchor["max"], 6) == 1.000456
    assert round(anchor["min"], 6) == 1.000391
    # SECTION 8.4 WRITES "1.000456 / 1.000391 = 1.00007". The operands are
    # exactly right -- they are this cell's max and min to six decimals, as
    # asserted above. The quotient is not: those two numbers divide to
    # 1.0000650, and the full-precision values divide to 1.0000644. Both round
    # to 1.00006, not 1.00007.
    #
    # NOT CORRECTED, and not a threshold change. The spread threshold is 2.0
    # and every value here is 1.00006-ish, so nothing about the rule moves. It
    # is recorded because a pre-registered document should be arithmetically
    # exact, and because silently asserting the document's number instead of
    # the true one is how a wrong figure survives. See S92.
    assert (1.000456 / 1.000391) == pytest.approx(1.0000650, abs=1e-7)
    assert anchor["spread"] == pytest.approx(1.0000644, abs=1e-7)
    assert round(anchor["spread"], 5) == 1.00006
    assert anchor["spread"] < CE.SPREAD_THRESHOLD
    assert anchor["median"] < CE.MEDIAN_THRESHOLD


# ---------------------------------------------------------------------------
# the branch: all four, each made to fire on demand
# ---------------------------------------------------------------------------


def _payload_with(values, **over):
    p = {"measurement": _cell(values)} if values else {"measurement": None}
    p.update(over)
    return p


def test_branch_one_spread_above_threshold_gives_the_plain_barrier():
    decision = CE.branch_for(_payload_with([1.0] * 9 + [3.0]))
    assert decision["branch"] == CE.PLAIN_BARRIER
    assert decision["spread"] == pytest.approx(3.0)
    assert decision["available"] is True
    assert decision["builds_d6"] is False
    assert "inconsistent" in decision["reason"]


def test_branch_two_tight_spread_small_median_gives_plain_plus_robustness():
    decision = CE.branch_for(_payload_with([1.0004] * 10))
    assert decision["branch"] == CE.PLAIN_BARRIER
    assert decision["aligned_as_robustness_check"] is True
    assert decision["builds_d6"] is False
    assert decision["available"] is True


def test_branch_three_tight_spread_large_median_gives_the_aligned_barrier():
    decision = CE.branch_for(_payload_with([1.5] * 10))
    assert decision["branch"] == CE.ALIGNED_BARRIER
    assert decision["builds_d6"] is True
    assert decision["available"] is True


@pytest.mark.parametrize("payload,why", [
    ({"measurement": None}, "no measurement"),
    ({}, "no measurement"),
    ({"measurement": _cell([1.0, 0.0])}, "undefined"),
    ({"measurement": _cell([1.0] * 3)}, "directions"),
    ({"measurement": _cell([1.0] * 10, epsilon=1e-5)}, "epsilon"),
])
def test_branch_four_unavailable_or_ambiguous_gives_the_plain_barrier(
        payload, why):
    """The fourth branch is a real path, not a fallback nobody exercises."""
    decision = CE.branch_for(payload)
    assert decision["branch"] == CE.PLAIN_BARRIER
    assert decision["available"] is False
    assert decision["builds_d6"] is False
    assert why in decision["reason"]


def test_the_thresholds_are_exactly_what_section_8_4_registered():
    """These are pre-registered values. Changing one here is an amendment to a
    decision rule, not a code change."""
    assert CE.SPREAD_THRESHOLD == 2.0
    assert CE.MEDIAN_THRESHOLD == 1.01
    assert CE.INJECTION_POINT_EPSILON == 1e-6
    assert CE.INJECTION_POINT_N_DIRECTIONS == 10
    assert len(CE.SEEDS) == 10


def test_the_boundaries_fall_the_way_section_8_4_wrote_them():
    """`spread <= 2` and `median >= 1.01`, so exactly 2 is tight and exactly
    1.01 selects the aligned barrier."""
    # spread exactly 2.0 (2.04 / 1.02) and median 1.02, which is >= 1.01, so
    # both conditions of the third row hold and it is NOT sent to plain.
    exactly_two = CE.branch_for(_payload_with([1.02] * 9 + [2.04]))
    assert exactly_two["spread"] == pytest.approx(2.0)
    assert exactly_two["branch"] == CE.ALIGNED_BARRIER

    # spread exactly 2.0 but median below 1.01 -> the second row, plain.
    tight_but_small = CE.branch_for(_payload_with([1.005] * 9 + [2.01]))
    assert tight_but_small["spread"] == pytest.approx(2.0)
    assert tight_but_small["branch"] == CE.PLAIN_BARRIER
    assert tight_but_small["aligned_as_robustness_check"] is True

    just_over = CE.branch_for(_payload_with([1.0] * 9 + [2.0001]))
    assert just_over["spread"] > CE.SPREAD_THRESHOLD
    assert just_over["branch"] == CE.PLAIN_BARRIER

    at_median = CE.branch_for(_payload_with([1.01] * 10))
    assert at_median["branch"] == CE.ALIGNED_BARRIER


# ---------------------------------------------------------------------------
# the branch is DERIVED at render time, never stored as prose (S70)
# ---------------------------------------------------------------------------


def _written(tmp_path, values):
    payload = {
        "task": "test", "LIMITATION": CE.INJECTION_POINT_LIMITATION,
        "fixture": "fabricated", "recipe": ["ZeroKeyBiasGauge"],
        "target": {"step": 199, "kind": "weights_only", "injection_step": 200,
                   "arm": "twin", "seed": 3, "family": "hf_gpt2"},
        "measurement": _cell(values),
        "public_gpt2_comparison": {},
    }
    CE._write_injection_report(payload, tmp_path, CE.INJECTION_POINT_STEM,
                               open(os.devnull, "w", encoding="utf-8"))
    return payload


def test_the_markdown_is_exactly_what_the_json_renders(tmp_path):
    """The S70 guard, applied to this report."""
    _written(tmp_path, [1.0004] * 10)
    stored = json.loads(
        (tmp_path / f"{CE.INJECTION_POINT_STEM}.json").read_text("utf-8"))
    rendered = (tmp_path / f"{CE.INJECTION_POINT_STEM}.md").read_text("utf-8")
    assert rendered == CE.format_injection_report(stored) + "\n"


def test_a_tampered_branch_in_the_json_does_not_change_the_report(tmp_path):
    """THE HIGHEST-STAKES PROPERTY IN THIS FILE. The rendered conclusion is
    recomputed from the measurement, so writing a different one into the JSON
    by hand changes nothing about what the report says."""
    _written(tmp_path, [1.0004] * 10)
    stored = json.loads(
        (tmp_path / f"{CE.INJECTION_POINT_STEM}.json").read_text("utf-8"))
    honest = CE.format_injection_report(stored)
    assert CE.PLAIN_BARRIER in honest

    stored["branch"] = {"branch": CE.ALIGNED_BARRIER, "rule": "I said so",
                        "reason": "because I edited the file.",
                        "spread": 1.0, "median": 99.0,
                        "aligned_as_robustness_check": False,
                        "builds_d6": True, "available": True}
    assert CE.format_injection_report(stored) == honest
    assert CE.ALIGNED_BARRIER not in CE.format_injection_report(stored)


def test_the_report_names_the_target_step_explicitly(tmp_path):
    payload = _written(tmp_path, [1.0004] * 10)
    text = CE.format_injection_report(payload)
    assert "199" in text
    assert "last at or before injection step 200" in text


def test_the_report_states_what_it_cannot_tell_you(tmp_path):
    payload = _written(tmp_path, [1.0004] * 10)
    text = CE.format_injection_report(payload)
    assert "NOT a twin-vs-twin distance" in text
    assert "MEASURES THE RULER, NOT THE STUDY" in text


def test_provenance_is_derived_and_moves_with_the_data(tmp_path):
    a = _written(tmp_path, [1.0004] * 10)["PROVENANCE"]
    b = _written(tmp_path, [1.5] * 10)["PROVENANCE"]
    assert a != b
    assert CE.PLAIN_BARRIER in a
    assert CE.ALIGNED_BARRIER in b


# ---------------------------------------------------------------------------
# directions are derived, not sampled
# ---------------------------------------------------------------------------


_RANDOM_CALLS = {"rand", "randn", "randint", "random", "rand_like",
                 "randn_like", "randint_like", "shuffle", "sample", "choice",
                 "seed", "getrandbits", "uniform", "normal"}


def _direction_path_functions():
    """Every function reachable from the direction-generation entry points.

    FOLLOWS CALLS ACROSS MODULES, not just within one function body. A helper
    that calls `torch.randn` one module over is still randomness in the path,
    and a scan that only read `_unit_direction`'s own body would miss it.
    """
    modules = {
        "canonicalization_error": ast.parse(
            (REPO_ROOT / "scripts" / "canonicalization_error.py")
            .read_text(encoding="utf-8")),
        "canonicalize": ast.parse(
            (REPO_ROOT / "scripts" / "canonicalize.py")
            .read_text(encoding="utf-8")),
    }
    defs = {}
    for mod, tree in modules.items():
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs.setdefault((mod, node.name), node)

    seen, out = set(), []
    stack = [("canonicalization_error", n) for n in
             ("_unit_direction", "perturb", "measure_injection_point",
              "spread_of", "median_of", "branch_for",
              "target_checkpoint_step")]
    while stack:
        key = stack.pop()
        if key in seen:
            continue
        seen.add(key)
        node = defs.get(key)
        if node is None:
            continue
        out.append((key, node))
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            fn = call.func
            if isinstance(fn, ast.Name):
                for mod in modules:
                    if (mod, fn.id) in defs:
                        stack.append((mod, fn.id))
            elif isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
                # `C.foo(...)` -> canonicalize.foo
                if fn.value.id == "C" and ("canonicalize", fn.attr) in defs:
                    stack.append(("canonicalize", fn.attr))
    return out


def test_the_direction_path_is_reachable_and_non_trivial():
    """If the walker stops finding functions, every scan below is vacuous."""
    found = {name for (_mod, name), _node in _direction_path_functions()}
    assert "_unit_direction" in found
    assert "perturb" in found
    assert "measure_injection_point" in found
    assert len(found) >= 8, found


def test_no_unseeded_randomness_anywhere_in_the_direction_path():
    """Every random draw must carry an explicit generator.

    The measurement has to be re-runnable to the same numbers: section 8.5
    requires the branch to be recorded before any arm-vs-twin distance is
    examined, and a branch that cannot be reproduced is not a record of
    anything.
    """
    offenders = []
    for (mod, fname), node in _direction_path_functions():
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            fn = call.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(
                fn, "id", None)
            if name not in _RANDOM_CALLS:
                continue
            kwargs = {k.arg for k in call.keywords}
            if "generator" not in kwargs:
                offenders.append(f"{mod}.{fname}: {name}() without generator=")
    assert not offenders, offenders


def test_no_wall_clock_or_process_hash_in_the_direction_path():
    """`hash()` is randomised per process by PYTHONHASHSEED and time is not a
    reproducible input. Either would make the measurement unrepeatable while
    every test in a single process still passed."""
    offenders = []
    for (mod, fname), node in _direction_path_functions():
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            fn = call.func
            if isinstance(fn, ast.Name) and fn.id in {"hash", "id"}:
                offenders.append(f"{mod}.{fname}: {fn.id}()")
            if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
                if fn.value.id in {"time", "datetime", "random", "os"} and \
                        fn.attr in _RANDOM_CALLS | {"time", "now", "urandom"}:
                    offenders.append(f"{mod}.{fname}: {fn.value.id}.{fn.attr}")
    assert not offenders, offenders


def test_the_direction_seeds_are_the_committed_public_gpt2_seeds():
    """Section 8.4 says "matching the epsilon and seed count" of D, E and F.
    Reusing the SEEDS themselves makes it the same ten directions, which is the
    stronger claim -- the parameter shapes are identical, so the same seed
    produces the same tensor. See S92."""
    assert CE.SEEDS == (11, 22, 33, 44, 55, 66, 77, 88, 99, 110)
    assert len(CE.SEEDS) == CE.INJECTION_POINT_N_DIRECTIONS


def test_direction_seeds_are_stable_under_a_different_process_hash_seed():
    """Run in a subprocess with PYTHONHASHSEED changed. If any part of the
    direction derivation went through `hash()`, this is where it shows."""
    script = (
        "import sys; sys.path.insert(0, 'scripts');\n"
        "import canonicalization_error as CE;\n"
        "print(list(CE.SEEDS));\n"
        "print(CE.INJECTION_POINT_EPSILON);\n"
    )
    outs = []
    for hashseed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hashseed)
        r = subprocess.run([sys.executable, "-c", script], env=env,
                           capture_output=True, text=True, cwd=REPO_ROOT)
        assert r.returncode == 0, r.stderr
        outs.append(r.stdout)
    assert len(set(outs)) == 1, outs


def test_the_same_direction_is_drawn_every_time(model_fixture):
    """The generator is seeded only by `seed`, so redrawing must be exact."""
    import torch

    label, builder = model_fixture
    model = builder()
    CE._DIRECTION_CACHE.clear()
    first = CE._unit_direction(model, "isotropic", 11)
    first = [t.clone() for t in first]
    CE._DIRECTION_CACHE.clear()
    second = CE._unit_direction(model, "isotropic", 11)
    for a, b in zip(first, second):
        assert torch.equal(a, b), f"{label}: direction 11 was not reproducible"


def test_different_seeds_give_different_directions(model_fixture):
    import torch

    label, builder = model_fixture
    model = builder()
    CE._DIRECTION_CACHE.clear()
    a = [t.clone() for t in CE._unit_direction(model, "isotropic", 11)]
    b = CE._unit_direction(model, "isotropic", 22)
    assert any(not torch.equal(x, y) for x, y in zip(a, b)), label


# ---------------------------------------------------------------------------
# the measurement itself, on both fixtures
# ---------------------------------------------------------------------------


def test_the_measurement_has_the_shape_section_8_4_specifies(model_fixture):
    label, builder = model_fixture
    model = builder()
    CE._DIRECTION_CACHE.clear()
    CE._THETA_NORM_CACHE.clear()
    cell = CE.measure_injection_point(
        lambda: copy.deepcopy(model), C.TINY, seeds=CE.SEEDS)
    assert cell["n_directions"] == 10, label
    assert cell["epsilon"] == 1e-6, label
    assert cell["shape"] == "isotropic", label
    assert cell["direction_seeds"] == list(CE.SEEDS), label
    assert set(cell["by_direction"]) == {str(s) for s in CE.SEEDS}, label
    assert all(v > 0 for v in cell["by_direction"].values()), label


def test_the_measurement_is_reproducible(model_fixture):
    """Same model, same seeds, same numbers. Bit-for-bit."""
    label, builder = model_fixture
    model = builder()
    CE._DIRECTION_CACHE.clear()
    CE._THETA_NORM_CACHE.clear()
    a = CE.measure_injection_point(lambda: copy.deepcopy(model), C.TINY)
    CE._DIRECTION_CACHE.clear()
    CE._THETA_NORM_CACHE.clear()
    b = CE.measure_injection_point(lambda: copy.deepcopy(model), C.TINY)
    assert a["by_direction"] == b["by_direction"], label


def test_the_branch_can_be_computed_from_a_real_measurement(model_fixture):
    """End to end on a model, not on fabricated numbers: the pipeline from
    weights to a branch runs and yields one of the two metrics."""
    label, builder = model_fixture
    model = builder()
    CE._DIRECTION_CACHE.clear()
    CE._THETA_NORM_CACHE.clear()
    cell = CE.measure_injection_point(lambda: copy.deepcopy(model), C.TINY)
    decision = CE.branch_for({"measurement": cell})
    assert decision["branch"] in (CE.PLAIN_BARRIER, CE.ALIGNED_BARRIER), label
    assert decision["available"] is True, label
    assert decision["spread"] is not None, label
