"""The barrier CLI's objective gates. Part of S101.

WHY THIS FILE EXISTS AT ALL. `scripts/pilot_barrier.py` had no test file and no
guard: it read whatever two checkpoints it was given and wrote a barrier. Pointed
at the void v1 run directory it would have produced an arithmetically correct
curve over models trained to predict two tokens ahead (S97). The gate scan
flagged it, and this file pins that the gate is present, runs before the
measurement, and is the SAME gate the ladder uses rather than a second copy.

THE POINT OF THE IMPORT IS THE POINT OF THE TEST. Two implementations of a safety
check is worse than one, because the second is the one nobody re-reads when the
first is corrected. `test_the_gate_is_the_ladders_own_function` asserts identity,
not equivalence.
"""

from __future__ import annotations

import importlib.util
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


def _code_of(module_name: str, func_name: str) -> str:
    """One function's source, docstring stripped, so prose cannot satisfy a
    claim about code. Same helper shape as tests/test_displacement_ladder.py."""
    import ast

    tree = ast.parse((REPO_ROOT / "scripts"
                      / f"{module_name}.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            body = list(node.body)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body = body[1:]
            return "\n".join(ast.unparse(s) for s in body)
    raise AssertionError(f"no function {func_name!r} in {module_name}")


# ===========================================================================
# Torch-free: the wiring
# ===========================================================================


def test_the_gate_is_the_ladders_own_function_not_a_copy():
    """Identity, not equivalence. A second copy would drift on the first fix."""
    pytest.importorskip("torch")
    PB = _load("pilot_barrier")
    DL = _load("displacement_ladder")
    assert PB.LADDER.gate_checkpoint is DL.gate_checkpoint
    assert PB.LADDER._trained_at is DL._trained_at


def test_pilot_barrier_does_not_define_its_own_objective_logic():
    """It may compose the gate; it may not reimplement it."""
    source = (REPO_ROOT / "scripts" / "pilot_barrier.py").read_text(
        encoding="utf-8")
    for owned_by_the_ladder in ("def classify_objective", "def two_ahead_loss",
                                "def objective_fix_ancestry",
                                "DOUBLE_SHIFT_MARGIN_NATS =",
                                "REFUSED_TRAINING_COMMITS ="):
        assert owned_by_the_ladder not in source, (
            f"{owned_by_the_ladder!r} is the ladder's; importing it is the "
            "whole point")


def test_the_gate_runs_before_the_barrier_is_computed():
    """A gate that ran after the measurement would still have produced the
    number it exists to prevent."""
    body = _code_of("pilot_barrier", "main")
    assert "gate_or_refuse" in body
    assert body.index("gate_or_refuse") < body.index("M.barrier"), (
        "the gate must precede the measurement")


def test_the_gate_runs_on_both_endpoints():
    body = _code_of("pilot_barrier", "main")
    assert body.count("gate_or_refuse") >= 2, (
        "both checkpoints are gated, not just the first")


def test_indeterminate_is_not_tolerated_here():
    """Unlike the ladder's --selftest there are no junk fixtures in this path, so
    two objectives scoring the same means an untrained or unreadable checkpoint
    and there is no barrier worth measuring on it."""
    body = _code_of("pilot_barrier", "gate_or_refuse")
    assert "tolerate_indeterminate=False" in body


def test_the_written_result_carries_the_gate_and_the_training_commit():
    body = _code_of("pilot_barrier", "main")
    assert "objective_gate" in body
    assert "trained_at" in body, (
        "the recorded training commit belongs in the artifact, or a reader "
        "cannot tell which pilot the curve is from")


# ===========================================================================
# Torch: the gate actually refusing
# ===========================================================================


def test_gate_or_refuse_aborts_on_a_double_shifted_checkpoint(monkeypatch,
                                                              tmp_path):
    """v1's measured pair: next-token 7.1085 against two-ahead 4.2613."""
    pytest.importorskip("transformers")
    PB = _load("pilot_barrier")
    DL = _load("displacement_ladder")
    M = _load("metrics")

    monkeypatch.setattr(M, "cross_entropy_loss", lambda m, b: 7.1085)
    monkeypatch.setattr(DL, "two_ahead_loss", lambda m, b: 4.2613)

    path = tmp_path / "seed00_twin" / "step009535_full.pt"
    path.parent.mkdir(parents=True)
    with pytest.raises(SystemExit) as caught:
        PB.gate_or_refuse(object(), {}, path, M.tiny_smoke_batch())
    text = str(caught.value)
    assert "OBJECTIVE GATE FAILED" in text
    assert "no barrier written" in text
    assert "GATE A fired" in text
    assert "7.1085" in text and "4.2613" in text


def test_gate_or_refuse_aborts_on_a_pre_fix_training_commit(monkeypatch,
                                                            tmp_path):
    """Gate B, by ancestry. Losses are clean here; the commit is not."""
    pytest.importorskip("transformers")
    PB = _load("pilot_barrier")
    DL = _load("displacement_ladder")
    M = _load("metrics")

    monkeypatch.setattr(M, "cross_entropy_loss", lambda m, b: 3.21)
    monkeypatch.setattr(DL, "two_ahead_loss", lambda m, b: 9.5)

    run = tmp_path / "seed00_twin"
    run.mkdir()
    (run / "run_provenance.yaml").write_text(
        "git:\n  commit: 9aa930dcafebabe\n  dirty: false\n", encoding="utf-8")
    with pytest.raises(SystemExit) as caught:
        PB.gate_or_refuse(object(), {}, run / "step009535_full.pt",
                          M.tiny_smoke_batch())
    text = str(caught.value)
    assert "GATE B fired" in text
    assert "9aa930d" in text
    assert "GATE A fired" not in text


def test_gate_or_refuse_passes_a_corrected_checkpoint(monkeypatch, tmp_path):
    """3e715a6 contains S99's fix, so both gates must pass and return the cell."""
    pytest.importorskip("transformers")
    PB = _load("pilot_barrier")
    DL = _load("displacement_ladder")
    M = _load("metrics")

    monkeypatch.setattr(M, "cross_entropy_loss", lambda m, b: 3.21)
    monkeypatch.setattr(DL, "two_ahead_loss", lambda m, b: 9.5)

    run = tmp_path / "seed00_twin"
    run.mkdir()
    (run / "run_provenance.yaml").write_text(
        "git:\n  commit: 3e715a6\n  dirty: false\n", encoding="utf-8")
    cell = PB.gate_or_refuse(object(), {}, run / "step009535_full.pt",
                             M.tiny_smoke_batch())
    assert cell["objective"] == DL.OBJECTIVE_NEXT_TOKEN
    assert cell["gate_b_refused"] is False
    assert cell["gate_b_suspect"] is False
