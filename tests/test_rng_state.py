"""RNG capture and restore, tested DIRECTLY rather than through the loop.

WHY THIS FILE EXISTS SEPARATELY FROM test_train.py.

The kill-and-resume acceptance test passes. It would also pass with RNG restore
removed entirely, and `test_dropping_rng_restore_breaks_bit_identity` skips
saying so: in this configuration nothing consumes RNG after initialization,
because dropout is off and data order is a pure function of (seed, step) rather
than of a random stream.

That makes the acceptance test WEAKER THAN IT LOOKS as evidence about RNG
handling. It proves the resume path is correct; it does not prove the RNG half
of it is doing anything. So the machinery is tested here on its own terms:
capture, consume randomness, restore, consume again, and require the same
draws.

This matters because the configuration is not permanent. The moment dropout is
enabled, or a model that samples is used, or any per-step stochasticity enters,
RNG restore becomes load-bearing for the study's central claim -- and the
failure would be silent. Testing the mechanism now means it is already correct
when that day arrives, rather than being written under pressure afterwards.
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
torch = pytest.importorskip("torch")


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RNG = _load("rng_state")


# ---------------------------------------------------------------------------
# The mechanism: restore must reproduce the exact draws
# ---------------------------------------------------------------------------


def test_restoring_reproduces_the_same_torch_draws():
    """The property the whole module exists for."""
    torch.manual_seed(1234)
    state = RNG.capture()
    first = torch.randn(64)
    RNG.restore(state)
    second = torch.randn(64)
    assert torch.equal(first, second)


def test_without_restoring_the_draws_differ():
    """Proves the test above is not passing by accident."""
    torch.manual_seed(1234)
    _ = RNG.capture()
    first = torch.randn(64)
    second = torch.randn(64)
    assert not torch.equal(first, second)


def test_restoring_reproduces_python_random_draws():
    random.seed(99)
    state = RNG.capture()
    first = [random.random() for _ in range(16)]
    RNG.restore(state)
    second = [random.random() for _ in range(16)]
    assert first == second


def test_capture_is_a_snapshot_not_a_live_view():
    """A capture that aliased the generator would restore to the wrong point."""
    torch.manual_seed(7)
    state = RNG.capture()
    _ = torch.randn(1000)
    RNG.restore(state)
    after = torch.randn(4)
    RNG.restore(state)
    again = torch.randn(4)
    assert torch.equal(after, again)


# ---------------------------------------------------------------------------
# Refusals -- a restore that quietly does nothing is the dangerous case
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [None, {}, {"nope": 1}, [], "state"])
def test_restoring_from_a_non_capture_is_refused(bad):
    """Silently doing nothing would let the run continue and diverge."""
    with pytest.raises(RNG.RngStateError, match="missing 'torch_cpu'"):
        RNG.restore(bad)


def test_restore_reports_what_it_restored():
    """Returning a report is what lets a caller assert the resume was real."""
    torch.manual_seed(5)
    restored = RNG.restore(RNG.capture())
    assert restored["torch_cpu"] is True
    assert restored.get("python") is True


def test_cuda_state_from_another_machine_is_refused():
    """Resuming a CUDA run on a CPU box would drop state and diverge."""
    if torch.cuda.is_available():
        pytest.skip("CUDA present; this refusal applies to CPU-only hosts")
    state = RNG.capture()
    state["torch_cuda_all"] = [torch.zeros(16, dtype=torch.uint8)]
    with pytest.raises(RNG.RngStateError, match="no CUDA is available"):
        RNG.restore(state)


# ---------------------------------------------------------------------------
# Survives a round trip, because a checkpoint is not memory
# ---------------------------------------------------------------------------


def test_state_survives_a_torch_save_load_round_trip(tmp_path):
    torch.manual_seed(21)
    state = RNG.capture()
    first = torch.randn(32)
    path = tmp_path / "rng.pt"
    torch.save(state, path)
    RNG.restore(torch.load(path, weights_only=False))
    assert torch.equal(first, torch.randn(32))


def test_python_state_survives_being_turned_into_lists():
    """A JSON round trip turns tuples into lists and setstate rejects those."""
    random.seed(3)
    state = RNG.capture()
    first = [random.random() for _ in range(8)]
    version, internal, gauss = state["python"]
    state["python"] = [version, list(internal), gauss]
    RNG.restore(state)
    assert [random.random() for _ in range(8)] == first


# ---------------------------------------------------------------------------
# The digest, and what is captured
# ---------------------------------------------------------------------------


def test_digest_distinguishes_two_different_states():
    torch.manual_seed(1)
    a = RNG.digest(RNG.capture())
    torch.manual_seed(2)
    b = RNG.digest(RNG.capture())
    assert a != b


def test_digest_is_stable_for_one_state():
    torch.manual_seed(1)
    state = RNG.capture()
    assert RNG.digest(state) == RNG.digest(state)


def test_sources_captured_lists_what_is_actually_present():
    sources = RNG.sources_captured(RNG.capture())
    assert "torch_cpu" in sources
    assert "python_random" in sources


def test_capture_records_whether_cuda_was_present():
    """So a checkpoint says what kind of machine wrote it."""
    state = RNG.capture()
    assert state["cuda_available"] == torch.cuda.is_available()
    assert isinstance(state["cuda_device_count"], int)
