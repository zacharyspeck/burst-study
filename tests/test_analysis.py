"""The analysis: does it find an effect that is there, and refuse one that is not.

THE RISK THIS FILE EXISTS FOR.

No trained model exists, so every number this analysis has ever produced came
from fabricated input. A statistical routine that runs on fabricated input and
returns a plausible p-value looks exactly like one that works -- the same trap
step 10 was built around, one layer up and with worse consequences, because the
output of this module is the study's headline claim.

So every quantity gets BOTH directions:

  - fabricate a known effect, assert the analysis finds it and recovers its size
  - fabricate no effect, assert the analysis does not report one

An analysis that only passes the first is a machine for confirming whatever it
is shown.

TWO THINGS ARE CROSS-CHECKED BY AN INDEPENDENT ROUTE. The t-distribution and
the corrections are hand-written so this module stays stdlib-only and runs in
the torch-free environment; scipy exists in the ML environment, so the same
numbers are computed both ways there and required to agree. That is a second
route, not a second copy -- if the hand-written version drifts, the suite says
so rather than the study finding out.
"""

from __future__ import annotations

import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


A = _load("analysis")

sys.path.insert(0, str(REPO_ROOT))
from burst.config import ARMS, INJECTING_ARMS  # noqa: E402

SEEDS = list(range(10))


def _records(effects, *, metric="aligned_l2", twin_base=100.0,
             twin_jitter=0.0, seeds=SEEDS):
    """Fabricate a panel with KNOWN per-arm effects.

    `effects` maps arm -> the exact amount that arm sits above its twin. The
    twin varies across seeds by `twin_jitter` so the noise floor is non-zero,
    but the arm-minus-twin difference stays exactly the fabricated effect --
    which is what makes "did the analysis recover it" a checkable question.
    """
    rows = []
    for seed in seeds:
        twin = twin_base + twin_jitter * math.sin(seed * 1.7)
        rows.append({"seed": seed, "arm": "twin", "metric": metric,
                     "value": twin})
        for arm, effect in effects.items():
            rows.append({"seed": seed, "arm": arm, "metric": metric,
                         "value": twin + effect})
    return rows


def _noisy_records(effects, *, metric="aligned_l2", spread=1.0, seeds=SEEDS):
    """Same, but with per-arm-per-seed noise on top of the effect.

    Deterministic: the 'noise' is a fixed trigonometric pattern, so a failure
    is reproducible rather than a flake.
    """
    rows = []
    for seed in seeds:
        twin = 100.0 + 0.3 * math.cos(seed * 2.1)
        rows.append({"seed": seed, "arm": "twin", "metric": metric,
                     "value": twin})
        for k, (arm, effect) in enumerate(sorted(effects.items())):
            wobble = spread * math.sin(seed * 1.3 + k * 2.7)
            rows.append({"seed": seed, "arm": arm, "metric": metric,
                         "value": twin + effect + wobble})
    return rows


# ---------------------------------------------------------------------------
# RESPONSIVENESS: an effect that is there
# ---------------------------------------------------------------------------


def test_a_fabricated_effect_is_recovered_exactly():
    """The size, not merely the sign."""
    panel = A.load_panel(_records({"fluent-false": 5.0}), "aligned_l2")
    diffs = A.paired_differences(panel, "fluent-false")
    assert len(diffs) == 10
    assert all(abs(d - 5.0) < 1e-12 for d in diffs)


def test_a_fabricated_effect_is_reported_significant():
    result = A.analyse(
        A.load_panel(_records({"fluent-false": 5.0}, twin_jitter=0.5),
                     "aligned_l2"),
        correction="holm")
    row = result["arms"][0]
    assert row["arm"] == "fluent-false"
    assert row["mean"] == pytest.approx(5.0, abs=1e-9)
    assert row["significant"] is True
    assert row["ci_excludes_zero"] is True
    assert row["clears_noise_floor"] is True


def test_an_ordering_of_known_effects_comes_back_in_order():
    """The ladder is the study's headline shape, so it has to survive."""
    effects = {"fluent-false": 8.0, "fluent-true": 6.0,
               "scrambled-false": 4.0, "scrambled-true": 3.0,
               "pos-substituted": 2.0, "random-chars": 1.0}
    result = A.analyse(
        A.load_panel(_noisy_records(effects, spread=0.2), "aligned_l2"),
        correction="holm")
    assert result["ordering"] == [
        "fluent-false", "fluent-true", "scrambled-false", "scrambled-true",
        "pos-substituted", "random-chars"]


def test_the_effect_is_recovered_through_seed_variation():
    """A paired design must be immune to the twin moving between seeds.

    The twin swings by 50 here -- ten times the effect -- and the paired
    difference is still exactly 5. An unpaired analysis would drown.
    """
    panel = A.load_panel(_records({"fluent-false": 5.0}, twin_jitter=25.0),
                         "aligned_l2")
    diffs = A.paired_differences(panel, "fluent-false")
    assert all(abs(d - 5.0) < 1e-9 for d in diffs)
    floor = A.noise_floor(panel)
    assert max(abs(x) for x in floor) > 10.0, (
        "the fabricated twin variation should dominate the effect, which is "
        "the situation the pairing exists to survive")


# ---------------------------------------------------------------------------
# RESPONSIVENESS: no effect
# ---------------------------------------------------------------------------


def test_no_effect_is_not_reported_as_one():
    """The direction that matters. An analysis that only finds effects is a
    machine for confirming whatever it is shown."""
    result = A.analyse(
        A.load_panel(_records({"fluent-false": 0.0}, twin_jitter=0.7),
                     "aligned_l2"),
        correction="holm")
    row = result["arms"][0]
    assert row["mean"] == pytest.approx(0.0, abs=1e-9)
    assert row["significant"] is False
    assert row["clears_noise_floor"] is False


def test_pure_noise_is_not_reported_as_an_effect():
    effects = {arm: 0.0 for arm in INJECTING_ARMS}
    result = A.analyse(
        A.load_panel(_noisy_records(effects, spread=1.0), "aligned_l2"),
        correction="holm")
    assert not [r["arm"] for r in result["arms"] if r["significant"]], (
        "an arm was reported significant with no fabricated effect")


def test_an_effect_smaller_than_the_noise_floor_does_not_clear_it():
    """Consistency is not the same as being distinguishable from seed alone."""
    panel = A.load_panel(_records({"fluent-false": 0.05}, twin_jitter=10.0),
                         "aligned_l2")
    result = A.analyse(panel, correction="holm")
    row = result["arms"][0]
    assert row["significant"] is True, (
        "a perfectly consistent 0.05 is statistically significant, which is "
        "exactly why the noise-floor comparison is reported beside it")
    assert row["clears_noise_floor"] is False


# ---------------------------------------------------------------------------
# The noise floor is across seeds, and the pairing is within
# ---------------------------------------------------------------------------


def test_the_noise_floor_is_every_distinct_pair_of_seeds():
    panel = A.load_panel(_records({"fluent-false": 1.0}, twin_jitter=1.0),
                         "aligned_l2")
    floor = A.noise_floor(panel)
    n = len(panel.seeds)
    assert len(floor) == n * (n - 1) // 2 == 45


def test_the_noise_floor_is_zero_when_twins_do_not_vary():
    panel = A.load_panel(_records({"fluent-false": 1.0}, twin_jitter=0.0),
                         "aligned_l2")
    assert all(x == 0.0 for x in A.noise_floor(panel))


def test_the_noise_floor_keeps_its_sign():
    """Taking |.| here would halve the apparent spread and decide sidedness."""
    panel = A.load_panel(_records({"fluent-false": 1.0}, twin_jitter=3.0),
                         "aligned_l2")
    floor = A.noise_floor(panel)
    assert min(floor) < 0 < max(floor)


def test_pairing_an_arm_against_itself_is_refused():
    panel = A.load_panel(_records({"fluent-false": 1.0}), "aligned_l2")
    with pytest.raises(A.AnalysisError, match="against itself"):
        A.paired_differences(panel, "twin")


# ---------------------------------------------------------------------------
# The panel refuses anything ragged
# ---------------------------------------------------------------------------


def test_a_missing_cell_is_refused_not_imputed():
    rows = _records({"fluent-false": 1.0})
    rows = [r for r in rows
            if not (r["arm"] == "fluent-false" and r["seed"] == 4)]
    with pytest.raises(A.AnalysisError, match="missing seeds \\[4\\]"):
        A.load_panel(rows, "aligned_l2")


def test_a_duplicate_value_is_refused():
    rows = _records({"fluent-false": 1.0})
    rows.append(dict(rows[0]))
    with pytest.raises(A.AnalysisError, match="duplicate value"):
        A.load_panel(rows, "aligned_l2")


def test_fewer_than_ten_seeds_is_refused():
    rows = _records({"fluent-false": 1.0}, seeds=list(range(9)))
    with pytest.raises(A.AnalysisError, match="floor is 10"):
        A.load_panel(rows, "aligned_l2")


def test_a_missing_reference_arm_is_refused():
    rows = [r for r in _records({"fluent-false": 1.0}) if r["arm"] != "twin"]
    with pytest.raises(A.AnalysisError, match="reference arm"):
        A.load_panel(rows, "aligned_l2")


def test_an_unknown_arm_is_refused():
    rows = _records({"fluent-false": 1.0})
    rows.append({"seed": 0, "arm": "coherent", "metric": "aligned_l2",
                 "value": 1.0})
    with pytest.raises(A.AnalysisError, match="unknown arm"):
        A.load_panel(rows, "aligned_l2")


# ---------------------------------------------------------------------------
# The correction has no default
# ---------------------------------------------------------------------------


def test_the_correction_is_required_with_no_default():
    """spec-v4 has no section 9.4; the choice decides which arms separate."""
    import inspect

    params = inspect.signature(A.analyse).parameters
    assert params["correction"].default is inspect.Parameter.empty
    assert params["correction"].kind == inspect.Parameter.KEYWORD_ONLY


def test_an_unknown_correction_is_refused_and_points_at_the_queue():
    with pytest.raises(A.AnalysisError) as exc:
        A.correct([0.01, 0.2], "bonferroni-ish")
    message = str(exc.value)
    assert "NO DEFAULT" in message
    assert "D-4" in message


def test_the_cli_requires_the_correction():
    import argparse

    with pytest.raises(SystemExit):
        A.main(["--input", "x", "--metric", "m", "--outdir", "o"])


@pytest.mark.parametrize("method", A.CORRECTIONS)
def test_every_correction_is_monotone_and_bounded(method):
    raw = [0.001, 0.01, 0.02, 0.04, 0.2, 0.9]
    adjusted = A.correct(raw, method)
    assert all(0.0 <= p <= 1.0 for p in adjusted)
    assert all(a >= r - 1e-12 for a, r in zip(adjusted, raw)), (
        "an adjusted p-value below the raw one would make correction increase "
        "the apparent significance")
    ordered = [a for _, a in sorted(zip(raw, adjusted))]
    assert ordered == sorted(ordered), "correction reordered the p-values"


def test_holm_is_at_least_as_conservative_as_benjamini_hochberg():
    raw = [0.001, 0.008, 0.02, 0.04, 0.2, 0.9]
    holm = A.correct(raw, "holm")
    bh = A.correct(raw, "benjamini_hochberg")
    assert all(h >= b - 1e-12 for h, b in zip(holm, bh))


def test_yekutieli_is_at_least_as_conservative_as_hochberg():
    raw = [0.001, 0.008, 0.02, 0.04, 0.2, 0.9]
    by = A.correct(raw, "benjamini_yekutieli")
    bh = A.correct(raw, "benjamini_hochberg")
    assert all(y >= b - 1e-12 for y, b in zip(by, bh))


def test_none_changes_nothing():
    raw = [0.001, 0.2, 0.9]
    assert A.correct(raw, "none") == raw


def test_the_correction_changes_which_arms_separate():
    """The reason it is a ruling and not a default."""
    effects = {arm: 0.9 for arm in INJECTING_ARMS}
    panel = A.load_panel(_noisy_records(effects, spread=1.1), "aligned_l2")
    strict = A.analyse(panel, correction="holm")
    loose = A.analyse(panel, correction="none")
    n_strict = sum(r["significant"] for r in strict["arms"])
    n_loose = sum(r["significant"] for r in loose["arms"])
    assert n_loose >= n_strict


# ---------------------------------------------------------------------------
# The hand-written distributions, against an independent route
# ---------------------------------------------------------------------------


def test_student_t_matches_known_values():
    """Textbook two-sided p for t = 2.262 on 9 df is 0.05."""
    assert A.student_t_sf(2.262157, 9) == pytest.approx(0.05, abs=1e-5)
    assert A.student_t_sf(0.0, 9) == 1.0
    assert A.student_t_sf(1e6, 9) == pytest.approx(0.0, abs=1e-9)


def test_student_t_is_symmetric():
    for t in (0.5, 1.0, 2.5, 7.0):
        assert A.student_t_sf(t, 9) == pytest.approx(A.student_t_sf(-t, 9))


def test_student_t_matches_scipy_where_scipy_exists():
    """INDEPENDENT ROUTE. Hand-written so this module stays stdlib-only; the
    ML environment has scipy, so the two are required to agree there."""
    scipy_stats = pytest.importorskip("scipy.stats")
    for df in (2, 5, 9, 30, 100):
        for t in (0.1, 0.7, 1.5, 2.3, 4.0):
            expected = 2.0 * scipy_stats.t.sf(abs(t), df)
            assert A.student_t_sf(t, df) == pytest.approx(expected, rel=1e-9)


def test_paired_t_matches_scipy_where_scipy_exists():
    scipy_stats = pytest.importorskip("scipy.stats")
    diffs = [1.2, 0.8, 1.5, 0.3, 1.1, 0.9, 1.4, 0.7, 1.0, 1.3]
    mine = A.paired_t_test(diffs)
    expected = scipy_stats.ttest_1samp(diffs, 0.0)
    assert mine["t"] == pytest.approx(float(expected.statistic), rel=1e-9)
    assert mine["p"] == pytest.approx(float(expected.pvalue), rel=1e-9)


def test_corrections_match_statsmodels_or_scipy_where_available():
    """Second route for the corrections, if either library provides one."""
    raw = [0.001, 0.008, 0.02, 0.04, 0.2, 0.9]
    scipy_stats = pytest.importorskip("scipy.stats")
    if not hasattr(scipy_stats, "false_discovery_control"):
        pytest.skip("scipy has no false_discovery_control in this version")
    expected_bh = list(scipy_stats.false_discovery_control(raw, method="bh"))
    assert A.correct(raw, "benjamini_hochberg") == pytest.approx(
        expected_bh, rel=1e-9)
    expected_by = list(scipy_stats.false_discovery_control(raw, method="by"))
    assert A.correct(raw, "benjamini_yekutieli") == pytest.approx(
        expected_by, rel=1e-9)


def test_a_degenerate_zero_spread_does_not_divide_by_zero():
    result = A.paired_t_test([2.0] * 10)
    assert result["sd"] == 0.0
    assert result["t"] is None
    assert result["p"] == 0.0
    assert "NOTE" in result


# ---------------------------------------------------------------------------
# The bootstrap is deterministic and library-independent
# ---------------------------------------------------------------------------


def test_the_bootstrap_is_reproducible():
    samples = [1.0, 2.0, 3.0, 2.5, 1.5, 2.2, 1.8, 2.9, 2.1, 1.9]
    a = A.bootstrap_ci(samples, n_resamples=500, label="x")
    b = A.bootstrap_ci(samples, n_resamples=500, label="x")
    assert a == b


def test_the_bootstrap_survives_a_separate_process():
    """No library PRNG, so no PYTHONHASHSEED or numpy version dependence."""
    import os
    import subprocess

    code = (
        "import sys; sys.path.insert(0, r'%s');"
        "from analysis import bootstrap_ci;"
        "print(bootstrap_ci([1.0,2.0,3.0,2.5,1.5,2.2,1.8,2.9,2.1,1.9],"
        "n_resamples=500, label='x')['ci_low'])" % (REPO_ROOT / "scripts")
    )
    env = dict(os.environ, PYTHONHASHSEED="424242")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env, check=True)
    mine = A.bootstrap_ci(
        [1.0, 2.0, 3.0, 2.5, 1.5, 2.2, 1.8, 2.9, 2.1, 1.9],
        n_resamples=500, label="x")["ci_low"]
    assert float(out.stdout.strip()) == pytest.approx(mine)


def test_the_bootstrap_uses_no_library_prng():
    import ast

    source = (REPO_ROOT / "scripts" / "analysis.py").read_text(encoding="utf-8")
    for banned in ("import numpy", "import random", "from random",
                   "import scipy", "default_rng"):
        assert banned not in source, banned
    assert "sha256" in source


def test_the_interval_brackets_the_mean():
    samples = [1.0, 2.0, 3.0, 2.5, 1.5, 2.2, 1.8, 2.9, 2.1, 1.9]
    ci = A.bootstrap_ci(samples, n_resamples=2000, label="x")
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]


def test_a_wider_level_gives_a_wider_interval():
    samples = [1.0, 2.0, 3.0, 2.5, 1.5, 2.2, 1.8, 2.9, 2.1, 1.9]
    narrow = A.bootstrap_ci(samples, level=0.80, n_resamples=4000, label="x")
    wide = A.bootstrap_ci(samples, level=0.99, n_resamples=4000, label="x")
    assert (wide["ci_high"] - wide["ci_low"]) > (
        narrow["ci_high"] - narrow["ci_low"])


def test_a_zero_effect_interval_spans_zero():
    ci = A.bootstrap_ci([0.1, -0.1, 0.05, -0.05, 0.0, 0.02, -0.02, 0.03,
                         -0.03, 0.01], n_resamples=2000, label="x")
    assert ci["excludes_zero"] is False
    assert ci["ci_low"] < 0 < ci["ci_high"]


# ---------------------------------------------------------------------------
# Reporting: derived, never hardcoded
# ---------------------------------------------------------------------------


def test_min_median_max_is_reported_for_every_arm():
    """A mean alone has hidden a real effect three times in this build."""
    result = A.analyse(
        A.load_panel(_noisy_records({"fluent-false": 2.0}, spread=1.0),
                     "aligned_l2"),
        correction="holm")
    row = result["arms"][0]
    for key in ("min", "median", "max", "mean"):
        assert key in row
    assert row["min"] <= row["median"] <= row["max"]


def test_the_provenance_is_derived_from_the_payload():
    def payload_for(effect):
        panel = A.load_panel(_records({"fluent-false": effect},
                                      twin_jitter=0.4), "aligned_l2")
        return A.build_payload(panel, correction="holm", n_resamples=400)

    a, b = payload_for(5.0), payload_for(0.0)
    assert a["PROVENANCE"] != b["PROVENANCE"]
    assert "10 seeds" in a["PROVENANCE"]


def test_a_synthetic_banner_is_unmissable():
    panel = A.load_panel(_records({"fluent-false": 1.0}), "aligned_l2")
    payload = A.build_payload(panel, correction="holm", n_resamples=200,
                              synthetic="*** SYNTHETIC INPUT ***")
    banner = "\n".join(A.report_banner(payload))
    assert "SYNTHETIC" in banner
    assert "NOT MEASUREMENTS" in banner
    assert "SYNTHETIC" in A.format_report(payload)


def test_the_report_renders_and_names_the_correction():
    panel = A.load_panel(_records({"fluent-false": 3.0}, twin_jitter=0.5),
                         "aligned_l2")
    payload = A.build_payload(panel, correction="benjamini_hochberg",
                              n_resamples=400)
    text = A.format_report(payload)
    assert "benjamini_hochberg" in text
    assert "PAIRED EFFECTS vs TWIN" in text
    assert "noise floor" in text


def test_the_svg_is_wellformed_and_derived():
    import xml.etree.ElementTree as ET

    effects = {"fluent-false": 5.0, "random-chars": -2.0}
    result = A.analyse(
        A.load_panel(_noisy_records(effects, spread=0.4), "aligned_l2"),
        correction="holm", n_resamples=400)
    svg = A.ordering_plot_svg(result)
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    assert "fluent-false" in svg and "random-chars" in svg
    assert "noise floor" in svg


def test_the_text_plot_marks_intervals_that_span_zero():
    effects = {"fluent-false": 6.0, "random-chars": 0.0}
    result = A.analyse(
        A.load_panel(_noisy_records(effects, spread=0.8), "aligned_l2"),
        correction="holm", n_resamples=400)
    plot = "\n".join(A.ordering_plot_text(result))
    assert "fluent-false" in plot
    assert "CI spans 0" in plot


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_cli_end_to_end_writes_three_artifacts(tmp_path):
    effects = {arm: 6.0 - i for i, arm in enumerate(sorted(INJECTING_ARMS))}
    records = _noisy_records(effects, spread=0.4)
    src = tmp_path / "records.json"
    src.write_text(json.dumps(records), encoding="utf-8")
    code = A.main([
        "--input", str(src), "--metric", "aligned_l2",
        "--correction", "holm", "--outdir", str(tmp_path / "out"),
        "--resamples", "500", "--synthetic", "*** SYNTHETIC ***",
    ])
    assert code == 0
    out = tmp_path / "out"
    payload = json.loads(
        (out / "analysis-aligned_l2.json").read_text(encoding="utf-8"))
    assert payload["result"]["n_seeds"] == 10
    assert payload["result"]["correction"] == "holm"
    assert (out / "analysis-aligned_l2.md").is_file()
    assert (out / "analysis-aligned_l2.svg").is_file()


def test_cli_refuses_a_ragged_panel(tmp_path, capsys):
    rows = [r for r in _records({"fluent-false": 1.0})
            if not (r["arm"] == "fluent-false" and r["seed"] == 2)]
    src = tmp_path / "records.json"
    src.write_text(json.dumps(rows), encoding="utf-8")
    code = A.main(["--input", str(src), "--metric", "aligned_l2",
                   "--correction", "holm", "--outdir", str(tmp_path / "o")])
    assert code == 1
    assert "missing seeds" in capsys.readouterr().out
