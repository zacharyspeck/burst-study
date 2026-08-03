"""S70 hardening: the measurement report must state its own provenance.

THE RISK THIS FILE EXISTS FOR.

Two artifacts in `docs/measurements/9-canonicalization-error.{json,md}` were
maintained BY HAND while the file around them was machine-generated:

  - the JSON's `PROVENANCE` key, which recorded seed coverage and the
    read-the-range warning
  - the markdown report's header banner, which said the same thing to a reader
    who opens the .md and never opens the .json

`scripts/canonicalization_error.py` neither wrote nor read either one. So the
next run of the script would have overwritten both files and silently deleted
both records -- and the resulting file would have looked entirely clean,
because a missing provenance note leaves no gap where it used to be. That is
the same failure shape as S61 and S69, where a table read clean precisely
because the harness was broken.

Both are now DERIVED from the payload by `build_provenance` and
`report_banner`. This file is the guard that they stay derived.

WHAT IS CHECKED, and why each check is the one that would have caught it:

1. The script writes both. A regeneration that drops either fails here rather
   than at review time, or worse, never.

2. They are DERIVED, not constant. Feeding a payload with different numbers
   must change the text. A provenance note that says the same thing regardless
   of the data is decoration, not provenance -- and hardcoded prose is exactly
   how `OPEN_QUESTION_distortion_factor` came to quote the retired recipe's
   3.05 and 84.4 while the shipped recipe measured 1.0004.

3. Seed coverage is read from the CELLS. The top-level `seeds` key holds only
   the last chunk's window, so anything reading it as coverage is wrong.

4. The ten-seed floor is asserted, and a below-floor payload must be announced
   in the banner rather than passed over quietly.

5. The D/E table prints its epsilon. Harmless today because those sections run
   at one epsilon; a silent duplicate-row table the moment a second returns.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = REPO_ROOT / "docs" / "measurements" / "9-canonicalization-error.json"
REPORT_MD = REPO_ROOT / "docs" / "measurements" / "9-canonicalization-error.md"


def _load_module():
    """Import the measurement script without running it.

    It is a script, not a package module, and it imports torch lazily inside
    the measurement functions -- so this import works in the torch-free
    environment, which is the point of that laziness.
    """
    spec = importlib.util.spec_from_file_location(
        "canonicalization_error",
        REPO_ROOT / "scripts" / "canonicalization_error.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cerr():
    return _load_module()


@pytest.fixture(scope="module")
def payload():
    if not REPORT_JSON.is_file():
        pytest.skip(f"{REPORT_JSON.name} not present")
    return json.loads(REPORT_JSON.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. Both records exist, in the file the script actually writes
# ---------------------------------------------------------------------------


def test_json_carries_a_provenance_record(payload):
    assert payload.get("PROVENANCE"), (
        "the results JSON has no PROVENANCE key. It was hand-added once and "
        "silently destroyed by the next run; build_provenance now derives it")
    assert len(payload["PROVENANCE"]) > 200


def test_json_carries_derived_seed_coverage(payload):
    assert "seed_coverage" in payload, (
        "seed coverage must live in the file, not in a commit message")


def test_markdown_carries_the_generated_banner(cerr, payload):
    if not REPORT_MD.is_file():
        pytest.skip("report .md not present")
    text = REPORT_MD.read_text(encoding="utf-8")
    for line in cerr.report_banner(payload):
        assert line in text, (
            f"banner line missing from the rendered report: {line!r}. The "
            f"banner is generated; a hand-edited one gets destroyed.")


def test_markdown_is_exactly_what_the_json_renders(cerr, payload):
    """No hand-editing anywhere in the report, not just in the banner."""
    if not REPORT_MD.is_file():
        pytest.skip("report .md not present")
    assert REPORT_MD.read_text(encoding="utf-8") == cerr.format_report(payload) + "\n"


# ---------------------------------------------------------------------------
# 2. Derived, not constant
# ---------------------------------------------------------------------------


def test_provenance_changes_when_the_data_changes(cerr, payload):
    """The check that separates provenance from decoration."""
    mutated = copy.deepcopy(payload)
    for cell in mutated["epsilon_sweep"]["cells"].values():
        cell["n_seeds"] = 4
    assert cerr.build_provenance(mutated) != cerr.build_provenance(payload)


def test_open_question_quotes_the_shipped_recipe_not_a_retired_one(cerr, payload):
    """S70's fifth-instance check.

    The hardcoded text quoted 3.05 and 84.4 -- the RETIRED six-step recipe's
    figures -- long after the shipped recipe measured 1.0004. Deriving the
    numbers is the fix; this asserts the derivation actually tracks section B.
    """
    text = cerr.open_question_distortion_factor(payload)
    iso = {c["epsilon"]: c for c in payload["epsilon_sweep"]["cells"].values()
           if c["shape"] == "isotropic"}
    lowest = min(iso)
    assert f"{iso[lowest]['ratio_median']:.4f}" in text
    assert "3.05" not in text, "the retired recipe's figure is back in the prose"


# ---------------------------------------------------------------------------
# 3 and 4. Coverage comes from the cells, and the floor is enforced
# ---------------------------------------------------------------------------


def test_seed_coverage_ignores_the_top_level_window(cerr, payload):
    """`seeds` is the last chunk's window and must not be read as coverage."""
    mutated = copy.deepcopy(payload)
    mutated["seeds"] = [1, 2]
    assert cerr.seed_coverage(mutated) == cerr.seed_coverage(payload)


def test_every_seeded_section_meets_the_ten_seed_floor(cerr, payload):
    coverage = cerr.seed_coverage(payload)
    thin = {k: v["n_seeds"] for k, v in coverage.items()
            if v["n_seeds"] is not None and v["n_seeds"] < cerr.MIN_SEEDS}
    assert not thin, f"sections below the {cerr.MIN_SEEDS}-seed floor: {thin}"


def test_a_below_floor_section_is_announced_in_the_banner(cerr, payload):
    """Silence is the failure mode. A thin section must be loud."""
    mutated = copy.deepcopy(payload)
    for cell in mutated["epsilon_sweep"]["cells"].values():
        cell["n_seeds"] = 3
    banner = "\n".join(cerr.report_banner(mutated))
    assert "FLOOR" in banner.upper()
    assert "AT LEAST ONE SECTION IS BELOW" in cerr.build_provenance(mutated)


def test_single_epsilon_sections_are_declared_a_limitation(cerr, payload):
    """Caveat 1 must be legible in the report, not only in the notes.

    D and E were cut to one epsilon to afford ten seeds. That is coverage
    traded for seeds, and the trade has to be visible next to the seed count it
    bought -- the two limitations are mirror images and neither substitutes for
    the other.
    """
    banner = "\n".join(cerr.report_banner(payload))
    coverage = cerr.seed_coverage(payload)
    thin = [k for k in ("B", "D", "E", "F")
            if len(coverage[k]["epsilons"]) == 1]
    if not thin:
        pytest.skip("no section is at a single epsilon")
    assert "SINGLE EPSILON" in banner.upper()
    assert "CANNOT SEE THAT CLIFF" in banner.upper()
    assert "COVERAGE TRADED FOR SEEDS" in cerr.build_provenance(payload)


# ---------------------------------------------------------------------------
# 5. The D/E tables print their epsilon
# ---------------------------------------------------------------------------


def test_attribution_rows_are_distinguishable_across_epsilons(cerr, payload):
    """Two epsilons must not render as two identical-looking rows.

    This is the defect fixed while it was still free: the caller looped over
    epsilons and dropped the key, which was invisible only because there was
    one epsilon left.
    """
    mutated = copy.deepcopy(payload)
    variants = mutated["step_attribution"]["variants"]
    label, cells = next(iter(variants.items()))
    cell = next(iter(cells.values()))
    cells["1e-99"] = copy.deepcopy(cell)

    # Scoped to section D's block. `shipped_recipe` is also a variant label in
    # E and in F, and those sections legitimately render identical numbers for
    # it -- matching across the whole report would fail on that collision
    # rather than on the thing under test.
    rendered = cerr.format_report(mutated).splitlines()
    start = next(i for i, ln in enumerate(rendered) if ln.startswith("D. ATTR"))
    end = next(i for i, ln in enumerate(rendered[start:], start)
               if ln.startswith("F. STEP") or ln.startswith("E. PERM"))
    rows = [ln for ln in rendered[start:end] if ln.startswith(label)]

    assert len(rows) >= 2
    assert len(set(rows)) == len(rows), (
        f"rows for {label} are indistinguishable across epsilons: {rows}")
    assert any("1e-99" in r for r in rows)


def test_every_seeded_table_prints_its_epsilon(cerr, payload):
    """D, E and F all loop over epsilons; all three must print the key.

    F has only ever run at a single epsilon, which is precisely how its copy of
    this defect stayed invisible.
    """
    rendered = cerr.format_report(payload)
    for header in (cerr._CELL_HEADER,):
        assert header in rendered
    contrib = payload.get("step_contributions")
    if contrib:
        eps = {e for cells in contrib["variants"].values() for e in cells}
        label = next(iter(contrib["variants"]))
        row = next(ln for ln in rendered.splitlines()
                   if ln.startswith(label) and len(ln.split()) > 4)
        assert any(e in row for e in eps), (
            f"section F row does not carry its epsilon: {row!r}")
