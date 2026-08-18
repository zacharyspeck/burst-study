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
    panel = A.load_panel(_records({"fluent-fabricated": 5.0}), "aligned_l2")
    diffs = A.paired_differences(panel, "fluent-fabricated")
    assert len(diffs) == 10
    assert all(abs(d - 5.0) < 1e-12 for d in diffs)


def test_a_fabricated_effect_is_reported_significant():
    result = A.analyse(
        A.load_panel(_records({"fluent-fabricated": 5.0}, twin_jitter=0.5),
                     "aligned_l2"),
        correction="holm")
    row = result["arms"][0]
    assert row["arm"] == "fluent-fabricated"
    assert row["mean"] == pytest.approx(5.0, abs=1e-9)
    assert row["significant"] is True
    assert row["ci_excludes_zero"] is True
    assert row["clears_noise_floor"] is True


def test_an_ordering_of_known_effects_comes_back_in_order():
    """The ladder is the study's headline shape, so it has to survive."""
    effects = {"fluent-fabricated": 8.0, "fluent-attested": 6.0, "random-chars": 1.0}
    result = A.analyse(
        A.load_panel(_noisy_records(effects, spread=0.2), "aligned_l2"),
        correction="holm")
    assert result["ordering"] == [
        "fluent-fabricated", "fluent-attested", "random-chars"]


def test_the_effect_is_recovered_through_seed_variation():
    """A paired design must be immune to the twin moving between seeds.

    The twin swings by 50 here -- ten times the effect -- and the paired
    difference is still exactly 5. An unpaired analysis would drown.
    """
    panel = A.load_panel(_records({"fluent-fabricated": 5.0}, twin_jitter=25.0),
                         "aligned_l2")
    diffs = A.paired_differences(panel, "fluent-fabricated")
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
        A.load_panel(_records({"fluent-fabricated": 0.0}, twin_jitter=0.7),
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
    panel = A.load_panel(_records({"fluent-fabricated": 0.05}, twin_jitter=10.0),
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
    panel = A.load_panel(_records({"fluent-fabricated": 1.0}, twin_jitter=1.0),
                         "aligned_l2")
    floor = A.noise_floor(panel)
    n = len(panel.seeds)
    assert len(floor) == n * (n - 1) // 2 == 45


def test_the_noise_floor_is_zero_when_twins_do_not_vary():
    panel = A.load_panel(_records({"fluent-fabricated": 1.0}, twin_jitter=0.0),
                         "aligned_l2")
    assert all(x == 0.0 for x in A.noise_floor(panel))


def test_the_noise_floor_keeps_its_sign():
    """Taking |.| here would halve the apparent spread and decide sidedness."""
    panel = A.load_panel(_records({"fluent-fabricated": 1.0}, twin_jitter=3.0),
                         "aligned_l2")
    floor = A.noise_floor(panel)
    assert min(floor) < 0 < max(floor)


def test_pairing_an_arm_against_itself_is_refused():
    panel = A.load_panel(_records({"fluent-fabricated": 1.0}), "aligned_l2")
    with pytest.raises(A.AnalysisError, match="against itself"):
        A.paired_differences(panel, "twin")


# ---------------------------------------------------------------------------
# The panel refuses anything ragged
# ---------------------------------------------------------------------------


def test_a_missing_cell_is_refused_not_imputed():
    rows = _records({"fluent-fabricated": 1.0})
    rows = [r for r in rows
            if not (r["arm"] == "fluent-fabricated" and r["seed"] == 4)]
    with pytest.raises(A.AnalysisError, match="missing seeds \\[4\\]"):
        A.load_panel(rows, "aligned_l2")


def test_a_duplicate_value_is_refused():
    rows = _records({"fluent-fabricated": 1.0})
    rows.append(dict(rows[0]))
    with pytest.raises(A.AnalysisError, match="duplicate value"):
        A.load_panel(rows, "aligned_l2")


def test_fewer_than_ten_seeds_is_refused():
    rows = _records({"fluent-fabricated": 1.0}, seeds=list(range(9)))
    with pytest.raises(A.AnalysisError, match="floor is 10"):
        A.load_panel(rows, "aligned_l2")


def test_a_missing_reference_arm_is_refused():
    rows = [r for r in _records({"fluent-fabricated": 1.0}) if r["arm"] != "twin"]
    with pytest.raises(A.AnalysisError, match="reference arm"):
        A.load_panel(rows, "aligned_l2")


def test_an_unknown_arm_is_refused():
    rows = _records({"fluent-fabricated": 1.0})
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
        A.load_panel(_noisy_records({"fluent-fabricated": 2.0}, spread=1.0),
                     "aligned_l2"),
        correction="holm")
    row = result["arms"][0]
    for key in ("min", "median", "max", "mean"):
        assert key in row
    assert row["min"] <= row["median"] <= row["max"]


def test_the_provenance_is_derived_from_the_payload():
    def payload_for(effect):
        panel = A.load_panel(_records({"fluent-fabricated": effect},
                                      twin_jitter=0.4), "aligned_l2")
        return A.build_payload(panel, correction="holm", n_resamples=400)

    a, b = payload_for(5.0), payload_for(0.0)
    assert a["PROVENANCE"] != b["PROVENANCE"]
    assert "10 seeds" in a["PROVENANCE"]


def test_a_synthetic_banner_is_unmissable():
    panel = A.load_panel(_records({"fluent-fabricated": 1.0}), "aligned_l2")
    payload = A.build_payload(panel, correction="holm", n_resamples=200,
                              synthetic="*** SYNTHETIC INPUT ***")
    banner = "\n".join(A.report_banner(payload))
    assert "SYNTHETIC" in banner
    assert "NOT MEASUREMENTS" in banner
    assert "SYNTHETIC" in A.format_report(payload)


def test_the_report_renders_and_names_the_correction():
    panel = A.load_panel(_records({"fluent-fabricated": 3.0}, twin_jitter=0.5),
                         "aligned_l2")
    payload = A.build_payload(panel, correction="benjamini_hochberg",
                              n_resamples=400)
    text = A.format_report(payload)
    assert "benjamini_hochberg" in text
    assert "PAIRED EFFECTS vs TWIN" in text
    assert "noise floor" in text


def test_the_svg_is_wellformed_and_derived():
    import xml.etree.ElementTree as ET

    effects = {"fluent-fabricated": 5.0, "random-chars": -2.0}
    result = A.analyse(
        A.load_panel(_noisy_records(effects, spread=0.4), "aligned_l2"),
        correction="holm", n_resamples=400)
    svg = A.ordering_plot_svg(result)
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    assert "fluent-fabricated" in svg and "random-chars" in svg
    assert "noise floor" in svg


def test_the_text_plot_marks_intervals_that_span_zero():
    effects = {"fluent-fabricated": 6.0, "random-chars": 0.0}
    result = A.analyse(
        A.load_panel(_noisy_records(effects, spread=0.8), "aligned_l2"),
        correction="holm", n_resamples=400)
    plot = "\n".join(A.ordering_plot_text(result))
    assert "fluent-fabricated" in plot
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
    rows = [r for r in _records({"fluent-fabricated": 1.0})
            if not (r["arm"] == "fluent-fabricated" and r["seed"] == 2)]
    src = tmp_path / "records.json"
    src.write_text(json.dumps(rows), encoding="utf-8")
    code = A.main(["--input", str(src), "--metric", "aligned_l2",
                   "--correction", "holm", "--outdir", str(tmp_path / "o")])
    assert code == 1
    assert "missing seeds" in capsys.readouterr().out


# ===========================================================================
# The two PRE-REGISTERED confirmatory contrasts
#
# Neither was computable before: §5 was reachable only by repurposing
# --reference, and §6 had no pooling construct. Both directions are tested here
# for the same reason as everything else in this file -- a contrast that runs on
# fabricated input and returns a plausible p-value looks exactly like one that
# works.
# ===========================================================================


def _contrast_panel(ff=0.30, ft=0.20, rc=0.05, **kw):
    """A panel with KNOWN per-arm displacements, so each contrast has an answer.

    primary       = ff - ft
    pooled_vs_arm = mean(ff, ft) - rc

    NARROWED 2026-08-08. The comparator was `pos-substituted` at 0.05, which
    made the second line the PRE-REGISTERED secondary contrast. That arm was
    cut, so the second line now exercises the pooling MECHANISM against a
    surviving arm and is no longer §6's contrast. §6 itself is uncomputable
    from here on; `test_the_preregistered_secondary_contrast_is_uncomputable`
    is what records that.
    """
    return A.load_panel(_records(
        {"fluent-fabricated": ff, "fluent-attested": ft, "random-chars": rc},
        **kw), metric="aligned_l2")


def test_the_primary_contrast_recovers_the_difference_it_was_given():
    """§5: fluent-fabricated vs fluent-attested, each minus its own seed-matched twin."""
    panel = _contrast_panel(ff=0.30, ft=0.20)
    diffs = A.arm_vs_arm_differences(panel, "fluent-fabricated", "fluent-attested")
    assert len(diffs) == len(panel.seeds)
    for d in diffs:
        assert d == pytest.approx(0.10, abs=1e-9)


def test_the_primary_contrast_reports_no_effect_when_there_is_none():
    """The other direction. Two arms at the same displacement must give zero."""
    panel = _contrast_panel(ff=0.25, ft=0.25)
    diffs = A.arm_vs_arm_differences(panel, "fluent-fabricated", "fluent-attested")
    assert all(d == pytest.approx(0.0, abs=1e-12) for d in diffs)
    cell = A.contrast(panel, "arm_vs_arm", arms=A.PRIMARY_CONTRAST)
    assert cell["ci_excludes_zero"] is False


def test_the_primary_contrast_is_antisymmetric_in_its_arms():
    panel = _contrast_panel(ff=0.30, ft=0.20)
    ab = A.arm_vs_arm_differences(panel, "fluent-fabricated", "fluent-attested")
    ba = A.arm_vs_arm_differences(panel, "fluent-attested", "fluent-fabricated")
    assert all(x == pytest.approx(-y, abs=1e-12) for x, y in zip(ab, ba))


def test_the_primary_contrast_does_not_need_reference_repurposing():
    """It gives the same numbers as the --reference fluent-attested route, which is
    the route that mislabels the noise floor (D-8b). The point is that this one
    keeps the reference where it belongs."""
    panel = _contrast_panel(ff=0.30, ft=0.20)
    proper = A.arm_vs_arm_differences(panel, "fluent-fabricated", "fluent-attested")
    repurposed = A.paired_differences(panel, "fluent-fabricated",
                                     reference="fluent-attested")
    assert all(x == pytest.approx(y, abs=1e-9)
               for x, y in zip(proper, repurposed))
    cell = A.contrast(panel, "arm_vs_arm", arms=A.PRIMARY_CONTRAST)
    assert cell["reference_arm"] == "twin", (
        "the contrast must leave the reference as the reference")


@pytest.mark.parametrize("arms,match", [
    (("fluent-fabricated", "fluent-fabricated"), "appears twice"),
    (("fluent-fabricated", "twin"), "is the reference"),
    (("twin", "fluent-fabricated"), "is the reference"),
    (("fluent-fabricated", "scrambled-true"), "absent from the panel"),
])
def test_the_contrast_guard_refuses_every_degenerate_pairing(arms, match):
    """Its own guard. paired_differences only refuses arm == reference, which
    does not catch two arms being the same as each other -- and that contrast is
    identically zero on every seed, which reads as a null result."""
    panel = _contrast_panel()
    with pytest.raises(A.AnalysisError, match=match):
        A.arm_vs_arm_differences(panel, arms[0], arms[1])


# --- §6: pooling, and the thing it must not be --------------------------------


def test_pooling_gives_exactly_one_row_per_seed():
    """THE PIN. Pooling AVERAGES within a seed; stacking concatenates across
    arms. Stacking two arms over ten seeds would give twenty rows, treat
    seed-correlated observations as independent, and shrink every p-value for
    free while leaving the mean unchanged.

    Mutating pooled_differences to stack must turn this red.
    """
    panel = _contrast_panel()
    pooled = A.pooled_differences(panel, A.SECONDARY_POOLED)
    assert len(pooled) == len(panel.seeds) == 10
    assert len(pooled) != len(A.SECONDARY_POOLED) * len(panel.seeds), (
        "twenty rows means the arms were stacked, not pooled")


def test_pooling_averages_the_arms_within_each_seed():
    panel = _contrast_panel(ff=0.30, ft=0.20)
    pooled = A.pooled_differences(panel, A.SECONDARY_POOLED)
    for value in pooled:
        assert value == pytest.approx(0.25, abs=1e-9), "mean(0.30, 0.20)"


def test_pooled_vs_arm_recovers_the_difference_it_was_given():
    """The pooled-vs-arm MECHANISM, against a surviving arm.

    Until the 2026-08-08 arm cut this was §6's pre-registered secondary
    contrast, with `pos-substituted` as the comparator. The mechanism is
    unchanged and still needs testing; what it is no longer is §6.
    """
    panel = _contrast_panel(ff=0.30, ft=0.20, rc=0.05)
    diffs = A.pooled_vs_arm_differences(
        panel, A.SECONDARY_POOLED, "random-chars")
    assert len(diffs) == len(panel.seeds)
    for d in diffs:
        assert d == pytest.approx(0.20, abs=1e-9), "0.25 - 0.05"


def test_pooled_vs_arm_reports_no_effect_when_there_is_none():
    panel = _contrast_panel(ff=0.20, ft=0.20, rc=0.20)
    cell = A.contrast(panel, "pooled_vs_arm", arms=A.SECONDARY_POOLED,
                      against="random-chars")
    assert cell["mean"] == pytest.approx(0.0, abs=1e-12)
    assert cell["ci_excludes_zero"] is False


def test_the_pooled_cell_records_its_row_count_and_says_why():
    panel = _contrast_panel()
    cell = A.contrast(panel, "pooled_vs_arm", arms=A.SECONDARY_POOLED,
                      against="random-chars")
    assert cell["n_rows"] == len(panel.seeds)
    assert cell["n_arms_pooled"] == 2
    assert "AVERAGED within each seed" in cell["POOLING"]
    assert "inflate n" in cell["POOLING"]


def test_pooling_refuses_a_single_arm():
    panel = _contrast_panel()
    with pytest.raises(A.AnalysisError, match="at least two arms"):
        A.pooled_differences(panel, ("fluent-fabricated",))


def test_pooling_refuses_the_reference_among_the_pooled_arms():
    panel = _contrast_panel()
    with pytest.raises(A.AnalysisError, match="is the reference"):
        A.pooled_differences(panel, ("fluent-fabricated", "twin"))


# --- labels, derived from the arms actually compared (D-8b) -------------------


def test_the_primary_label_names_the_arms_the_numbers_came_from():
    """D-8b's defect was a label that said "twin" while the value followed
    --reference. The fix is derivation; a different hardcoded string would be
    the same defect one layer over."""
    panel = _contrast_panel()
    cell = A.contrast(panel, "arm_vs_arm", arms=("fluent-fabricated", "fluent-attested"))
    for arm in cell["arms"]:
        assert arm in cell["label"], f"{arm} is in the data but not the label"
    assert cell["reference_arm"] in cell["label"]


def test_the_pooled_label_names_every_pooled_arm_and_the_comparator():
    panel = _contrast_panel()
    cell = A.contrast(panel, "pooled_vs_arm", arms=A.SECONDARY_POOLED,
                      against="random-chars")
    for arm in cell["arms"]:
        assert arm in cell["label"]
    assert cell["against"] in cell["label"]
    assert "AVERAGING" in cell["label"], (
        "the label must say how the pooling was done, since stacking would give "
        "a different n for the same arms")


def test_every_label_follows_the_reference_it_was_given():
    panel = _contrast_panel()
    for reference in ("twin", "random-chars"):
        cell = A.contrast(panel, "arm_vs_arm",
                          arms=("fluent-fabricated", "fluent-attested"),
                          reference=reference)
        assert reference in cell["label"]
        assert cell["reference_arm"] == reference
    # and the two labels differ, so the reference is genuinely in there
    a = A.contrast_label("arm_vs_arm", arms=A.PRIMARY_CONTRAST,
                         reference="twin")
    b = A.contrast_label("arm_vs_arm", arms=A.PRIMARY_CONTRAST,
                         reference="random-chars")
    assert a != b


def test_the_noise_floor_describes_the_reference_it_actually_used():
    """analysis.py used to hardcode "twin against twin ACROSS seeds" in both the
    payload and the banner while the value followed --reference."""
    panel = _contrast_panel()
    result = A.analyse(panel, correction="none", reference="random-chars")
    floor = result["noise_floor"]
    assert floor["reference_arm"] == "random-chars"
    assert "random-chars against random-chars" in floor["WHAT_IT_IS"]
    assert "twin against twin" not in floor["WHAT_IT_IS"]
    # THREE emitted sites, not the two the defect was logged against: the
    # payload's WHAT_IT_IS, report_banner's own line, the SVG legend, and the
    # provenance prose. All four derive now.
    banner = "\n".join(A.report_banner({"result": result}))
    assert "random-chars-vs-random-chars" in banner
    assert "twin-vs-twin" not in banner

    prose = A.build_provenance({"result": result})
    assert "random-chars against random-chars" in prose
    assert "twin against twin" not in prose

    svg = A.ordering_plot_svg(result) if hasattr(A, "ordering_plot_svg") else ""
    if svg:
        assert "twin-vs-twin noise floor" not in svg


# --- what the contrasts deliberately do NOT do -------------------------------


def test_the_contrasts_apply_no_correction_and_say_so():
    """Adding two contrasts changes the size of the family of tests, which is
    D-4 -- ruled 2026-08-07 at family 2, then left describing nothing by the
    2026-08-08 arm cut, and reopened as D-9. Nothing here divides by anything,
    which is exactly why that churn never reached these numbers."""
    panel = _contrast_panel()
    for cell in (A.contrast(panel, "arm_vs_arm", arms=A.PRIMARY_CONTRAST),
                 A.contrast(panel, "pooled_vs_arm", arms=A.SECONDARY_POOLED,
                            against="random-chars")):
        assert "p_raw" in cell
        assert "p_adjusted" not in cell
        assert "significant" not in cell
        assert "D-4" in cell["NO_CORRECTION_APPLIED"]


def test_the_correction_is_still_required_with_no_default():
    with pytest.raises(A.AnalysisError, match="correction is required"):
        A.analyse(_contrast_panel(), correction="")


def test_registered_contrasts_report_an_absent_arm_rather_than_skipping():
    """A pre-registered contrast that could not be computed is a fact about the
    run, not something to omit."""
    panel = A.load_panel(_records({"random-chars": 0.02}), metric="aligned_l2")
    out = A.registered_contrasts(panel)
    assert out["primary"]["computed"] is False
    assert set(out["primary"]["missing_arms"]) == {"fluent-fabricated", "fluent-attested"}
    assert "could not be computed" in out["primary"]["WHY_ABSENT"]
    assert out["secondary"]["computed"] is False


def test_registered_contrasts_match_the_preregistered_arms():
    """Sourced to docs/preregistration.md §5 and §6, written out here rather
    than read from the constants under test."""
    assert A.PRIMARY_CONTRAST == ("fluent-fabricated", "fluent-attested")
    assert A.SECONDARY_POOLED == ("fluent-fabricated", "fluent-attested")
    assert A.SECONDARY_AGAINST == "pos-substituted"
    panel = _contrast_panel()
    out = A.registered_contrasts(panel)
    assert out["primary"]["arms"] == ["fluent-fabricated", "fluent-attested"]
    assert out["secondary"]["against"] == "pos-substituted"
    assert out["primary"]["computed"]


def test_the_preregistered_secondary_contrast_is_uncomputable():
    """THE RECORD OF THE 2026-08-08 ARM CUT.

    §6 named `pos-substituted`, and that arm is no longer a run condition, so
    no panel this study can produce will ever contain it. SECONDARY_AGAINST
    deliberately still names it: the contrast has to keep appearing in the
    output, reported as absent and naming what is missing, rather than being
    deleted so that nothing records a pre-registered contrast went away.

    The consequence for D-4: one computable confirmatory contrast, so the
    Holm correction at family 2 ruled 2026-08-07 no longer describes this
    study. See D-9 in docs/decisions-pending.md.
    """
    assert A.SECONDARY_AGAINST not in ARMS
    out = A.registered_contrasts(_contrast_panel())
    assert out["secondary"]["computed"] is False
    assert out["secondary"]["missing_arms"] == ["pos-substituted"]
    assert "could not be computed" in out["secondary"]["WHY_ABSENT"]


def test_analyse_carries_the_registered_contrasts_beside_the_per_arm_rows():
    """§7 keeps every other comparison exploratory; these two are not, so they
    are reported beside the per-arm rows rather than instead of them."""
    result = A.analyse(_contrast_panel(), correction="none")
    assert "registered_contrasts" in result
    assert set(result["registered_contrasts"]) == {"primary", "secondary"}
    assert result["arms"], "the per-arm rows must survive"


