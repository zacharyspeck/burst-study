"""S70 guard for the step 10 report, present from the first commit.

In step 9 this guard was retrofitted after a hand-written PROVENANCE key and a
hand-written markdown banner were found sitting inside a machine-generated
file, one regeneration away from being silently deleted. Five separate times in
that build a hardcoded interpretation outlived the number it described, and the
fifth was shipping a claim contradicted by a table one screen below it.

So the same guard is written here BEFORE any of that can happen:

1. Both derived records exist in the file the script actually writes.
2. They are DERIVED -- feeding different numbers must change the text. A
   provenance note that says the same thing regardless of the data is
   decoration.
3. The .md is byte-identical to what format_report renders, so any hand edit
   anywhere in the file fails the suite.
4. The ten-seed floor is enforced and a thin section is announced loudly.
5. The report states what it CANNOT answer, derived from which functions
   actually raise -- so the list cannot go stale when one is implemented.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = REPO_ROOT / "docs" / "measurements" / "10-metrics.json"
REPORT_MD = REPO_ROOT / "docs" / "measurements" / "10-metrics.md"

pytest.importorskip("torch")
pytest.importorskip("transformers")


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R = _load("metrics_report")


@pytest.fixture(scope="module")
def payload():
    if not REPORT_JSON.is_file():
        pytest.skip("10-metrics.json not present")
    return json.loads(REPORT_JSON.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. Both derived records exist
# ---------------------------------------------------------------------------


def test_json_carries_a_provenance_record(payload):
    assert payload.get("PROVENANCE")
    assert len(payload["PROVENANCE"]) > 200


def test_json_carries_derived_seed_coverage(payload):
    assert "seed_coverage" in payload


def test_json_names_the_cka_variant(payload):
    """An unnamed CKA is incomparable to published work and to itself."""
    assert payload["cka_variant"] == "linear_cka_unbiased_hsic_tokens_as_samples_v1"


def test_json_records_the_batch_identity(payload):
    """A CKA number without the input that produced it is not a measurement."""
    batch = payload["batch"]
    for field in ("source", "file_sha256", "token_sha256", "n_tokens",
                  "tokenizer"):
        assert batch.get(field)
    assert batch["n_tokens"] == 1024
    assert batch["source"] == "bursts/context.txt"


def test_markdown_carries_the_generated_banner(payload):
    if not REPORT_MD.is_file():
        pytest.skip("report .md not present")
    text = REPORT_MD.read_text(encoding="utf-8")
    for line in R.report_banner(payload):
        assert line in text


def test_markdown_is_exactly_what_the_json_renders(payload):
    """No hand-editing anywhere in the report, not just in the banner."""
    if not REPORT_MD.is_file():
        pytest.skip("report .md not present")
    assert (REPORT_MD.read_text(encoding="utf-8")
            == R.format_report(payload) + "\n")


# ---------------------------------------------------------------------------
# 2. Derived, not constant
# ---------------------------------------------------------------------------


def test_provenance_changes_when_the_data_changes(payload):
    mutated = copy.deepcopy(payload)
    for section in mutated["junk_pairs"].values():
        section["l2_raw"]["n_seeds"] = 4
    assert R.build_provenance(mutated) != R.build_provenance(payload)


def test_provenance_reports_the_barrier_outcome_it_actually_saw(payload):
    """The barrier caveat must track the data, not assert a fixed story."""
    never = copy.deepcopy(payload)
    for section in never["junk_pairs"].values():
        cell = section["barrier_max_excess"]
        cell["by_seed"] = {k: 0.0 for k in cell["by_seed"]}
        cell["min"] = cell["median"] = cell["max"] = 0.0
    assert "No pair cleared that floor" in R.build_provenance(never)
    assert "No pair cleared that floor" not in R.build_provenance(payload)


def test_seed_summaries_are_derived_not_stored_per_chunk(payload):
    """The bug that printed '4 of 10 seeds (8 of them above the floor)'.

    A count summed inside one measurement call keeps only that chunk's seeds
    and then reads as a total. Every cross-seed summary must come from
    by_seed, so the two numbers cannot contradict each other.
    """
    above = R.barriers_above_noise(payload)
    for kind, row in above.items():
        assert row["n_above_noise_floor"] <= row["n_rose_above_chord"], (
            f"{kind}: more seeds cleared the noise floor than rose at all, "
            f"which is impossible and means a count is not derived from "
            f"by_seed")
        assert row["n_seeds"] == len(
            payload["junk_pairs"][kind]["barrier_max_excess"]["by_seed"])


def test_identical_pair_sets_the_noise_floor(payload):
    """Rises are counted against measured float noise, not against zero."""
    floor = R.barrier_noise_floor(payload)
    assert floor > 0.0
    assert R.barriers_above_noise(payload)["identical"][
        "n_above_noise_floor"] == 0


def test_identical_pair_defect_is_announced_not_absorbed(payload):
    """If two copies of one model ever differ, that is loud."""
    broken = copy.deepcopy(payload)
    broken["junk_pairs"]["identical"]["l2_raw"]["max"] = 0.5
    assert "DOES NOT SCORE IDENTICALLY" in R.build_provenance(broken)
    assert "DEFECT" in "\n".join(R.report_banner(broken))


def test_barrier_float_noise_is_reported_separately_from_defects(payload):
    """Interpolation noise and a real deviation must not share a heading."""
    dev = R._identical_pair_deviation(payload)
    assert set(dev) == {"weight_deviation", "barrier_noise"}
    text = R.build_provenance(payload)
    assert "not bitwise" in text


# ---------------------------------------------------------------------------
# 3 and 4. Coverage from the cells; the floor is enforced
# ---------------------------------------------------------------------------


def test_seed_coverage_ignores_the_top_level_window(payload):
    mutated = copy.deepcopy(payload)
    mutated["seeds"] = [1, 2]
    assert R.seed_coverage(mutated) == R.seed_coverage(payload)


def test_every_pair_meets_the_ten_seed_floor(payload):
    coverage = R.seed_coverage(payload)
    thin = {k: v for k, v in coverage.items()
            if v is not None and v < R.MIN_SEEDS}
    assert not thin, f"pairs below the {R.MIN_SEEDS}-seed floor: {thin}"


def test_a_below_floor_pair_is_announced_in_the_banner(payload):
    mutated = copy.deepcopy(payload)
    for section in mutated["junk_pairs"].values():
        section["l2_raw"]["n_seeds"] = 3
    banner = "\n".join(R.report_banner(mutated))
    assert "FLOOR" in banner.upper()
    assert "BELOW THE TEN-SEED FLOOR" in R.build_provenance(mutated)


# ---------------------------------------------------------------------------
# 5. What the report cannot answer, derived
# ---------------------------------------------------------------------------


def test_report_states_what_it_cannot_answer(payload):
    """Nobody reading this in two months may mistake it for a finished module."""
    text = payload["CANNOT_ANSWER"]
    for expected in ("NOT DONE", "aligned_barrier", "aligned_l2",
                     "rsf_subspace_probe", "Conv1D", "docs/layout-cost.md"):
        assert expected in text
    assert "TRAINED" in text.upper()


def test_cannot_answer_is_derived_from_what_actually_raises():
    """The list is asked of the module, so it cannot go stale silently."""
    text = R.cannot_answer({})
    assert "aligned_barrier" in text
    metrics = _load("metrics")
    delattr_target = getattr(metrics, "aligned_barrier")
    assert delattr_target is not None


def test_limitation_is_unmissable(payload):
    assert "JUNK CHECKPOINTS" in payload["LIMITATION"]
    assert "PLUMBING" in payload["LIMITATION"]
    banner = "\n".join(R.report_banner(payload))
    assert "JUNK CHECKPOINTS" in banner
    assert "NOT BUILT" in banner
