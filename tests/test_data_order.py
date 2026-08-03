"""Per-seed data order: reproducible across machines, and enforceable.

THE RISK THIS FILE EXISTS FOR.

The study's central claim is that runs were identical except for seed and arm.
Under the 2026-08-03 ruling the seed controls data order too, which means the
order is now part of what "identical except for seed" asserts. An order that is
reproducible within one process but not across machines would satisfy every
test written naively and destroy the claim silently -- which is exactly how
Python's hash() would have failed, and why make_bursts moved to SHA-256.

So the load-bearing tests here are the ones that leave the process:

  - a subprocess under a different PYTHONHASHSEED must agree, matching
    tests/test_sequence_assembly.py:82-98
  - the derivation must contain no hash() call at all

And the contract test, which is about a gap no amount of reproducibility
closes: a manifest that RECORDS an order proves nothing about the order a
training loop SERVED. verify_permutation is the check that closes it, and it
must refuse loudly rather than pass quietly when handed nothing.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
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


D = _load("data_order")

#: Small enough to be fast, large enough that a broken permutation shows.
N = 4096


# ---------------------------------------------------------------------------
# It is a permutation at all
# ---------------------------------------------------------------------------


def test_result_is_a_permutation():
    perm = D.sequence_permutation(0, N)
    assert len(perm) == N
    assert sorted(perm) == list(range(N)), "not a permutation of range(n)"


def test_permutation_actually_reorders():
    """A 'permutation' that returns identity would pass every other test."""
    perm = D.sequence_permutation(0, N)
    assert perm != list(range(N))
    moved = sum(1 for i, v in enumerate(perm) if i != v)
    assert moved > N * 0.9, f"only {moved} of {N} positions moved"


@pytest.mark.parametrize("n", [0, 1, 2, 3, 5, 7, 8, 9])
def test_small_and_non_multiple_sizes_still_permute(n):
    """The keystream emits 4 keys per digest; n need not be a multiple of 4."""
    assert sorted(D.sequence_permutation(0, n)) == list(range(n))


# ---------------------------------------------------------------------------
# Different seeds differ; the same seed repeats
# ---------------------------------------------------------------------------


def test_same_seed_gives_the_same_order():
    assert D.sequence_permutation(3, N) == D.sequence_permutation(3, N)


def test_different_seeds_give_different_orders():
    orders = {s: tuple(D.sequence_permutation(s, N)) for s in range(10)}
    assert len(set(orders.values())) == 10, "two seeds share a data order"


def test_a_seed_does_not_depend_on_how_many_seeds_exist():
    """Adding an eleventh seed must not disturb the first ten.

    The same independence make_bursts.derived_seed was built for. It is what
    makes the 40-vs-60-vs-80 run-count question unable to invalidate an
    already-derived permutation.
    """
    before = D.sequence_permutation(7, N)
    _ = [D.sequence_permutation(s, N) for s in range(20)]
    assert D.sequence_permutation(7, N) == before


def test_order_depends_on_seed_but_not_on_arm():
    """Why arm-vs-twin stays a matched pair.

    The permutation takes a seed and nothing else, so every arm at a given seed
    -- including twin -- sees the identical data order. If this ever took an
    arm, the study's headline comparison would stop being a matched pair.
    """
    import inspect

    params = inspect.signature(D.sequence_permutation).parameters
    assert "arm" not in params
    assert list(params) == ["run_seed", "n_sequences"]


# ---------------------------------------------------------------------------
# Reproducible ACROSS MACHINES, not merely across calls
# ---------------------------------------------------------------------------


def test_permutation_survives_a_separate_process_with_a_different_hashseed():
    """THE load-bearing test. Matches tests/test_sequence_assembly.py:82-98.

    PYTHONHASHSEED randomises hash(). If the derivation ever went back to it,
    this subprocess would disagree while every in-process test still passed.
    """
    code = (
        "import sys; sys.path.insert(0, r'%s');"
        "from data_order import seed_digest;"
        "print(seed_digest(3, %d))" % (REPO_ROOT / "scripts", N)
    )
    env = dict(os.environ, PYTHONHASHSEED="12345")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env, check=True)
    assert out.stdout.strip() == D.seed_digest(3, N)


def test_two_different_hashseeds_agree_with_each_other():
    """Belt and braces: neither process shares the parent's hash seed."""
    code = (
        "import sys; sys.path.insert(0, r'%s');"
        "from data_order import seed_digest;"
        "print(seed_digest(1, %d))" % (REPO_ROOT / "scripts", N)
    )
    results = []
    for hashseed in ("0", "99999"):
        env = dict(os.environ, PYTHONHASHSEED=hashseed)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, env=env, check=True)
        results.append(out.stdout.strip())
    assert results[0] == results[1] == D.seed_digest(1, N)


def test_derivation_does_not_call_python_hash():
    """Parsed, not string-matched.

    A substring check on "hash(" flags this module's own docstring, which warns
    against hash() by name -- so it would fail for documenting the very trap it
    is guarding, and the obvious fix is to weaken the check or delete the
    warning. Walking the AST asks the question that was actually meant: is
    builtin hash() CALLED anywhere.
    """
    import ast

    source = (REPO_ROOT / "scripts" / "data_order.py").read_text(
        encoding="utf-8")
    assert "sha256" in source
    calls = [node for node in ast.walk(ast.parse(source))
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)
             and node.func.id == "hash"]
    assert not calls, f"builtin hash() called at lines {[c.lineno for c in calls]}"


def test_derivation_imports_nothing_outside_the_stdlib():
    """No numpy, so no numpy version can change the order.

    Also why this module is tested in the torch-free environment rather than
    skipped there: the permutation is a reproducibility claim, and the
    environment with fewer moving parts is the better place to prove it.
    """
    source = (REPO_ROOT / "scripts" / "data_order.py").read_text(
        encoding="utf-8")
    for banned in ("import numpy", "import torch", "from numpy", "from torch"):
        assert banned not in source


# ---------------------------------------------------------------------------
# The digest
# ---------------------------------------------------------------------------


def test_digest_is_stable_and_seed_specific():
    assert D.seed_digest(0, N) == D.seed_digest(0, N)
    assert D.seed_digest(0, N) != D.seed_digest(1, N)
    assert len(D.seed_digest(0, N)) == 64


def test_digest_changes_if_a_single_position_changes():
    perm = D.sequence_permutation(0, N)
    swapped = list(perm)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    assert D.permutation_digest(swapped) != D.permutation_digest(perm)


def test_digest_is_over_fixed_width_bytes_not_text():
    """So a formatting change cannot move it.

    [1, 23] and [12, 3] both render as "123" if concatenated as text.
    """
    assert D.permutation_digest([1, 23]) != D.permutation_digest([12, 3])


# ---------------------------------------------------------------------------
# THE CONTRACT -- an obligation on a training loop that does not exist yet
# ---------------------------------------------------------------------------


def test_verify_accepts_the_order_it_derived():
    digest = D.seed_digest(5, N)
    assert D.verify_permutation(5, N, digest) == digest


def test_verify_refuses_a_different_seed():
    """The failure that matters: a run training on another seed's order."""
    digest = D.seed_digest(5, N)
    with pytest.raises(D.DataOrderError, match="DOES NOT MATCH THE MANIFEST"):
        D.verify_permutation(6, N, digest)


def test_verify_refuses_a_different_corpus_length():
    digest = D.seed_digest(5, N)
    with pytest.raises(D.DataOrderError, match="DOES NOT MATCH THE MANIFEST"):
        D.verify_permutation(5, N + 1, digest)


@pytest.mark.parametrize("empty", [None, "", 0])
def test_verify_refuses_to_check_against_nothing(empty):
    """A check that passes when handed no expectation reads as a check.

    This is the shape that would let a training loop 'verify' against a
    manifest field that was never populated and report success.
    """
    with pytest.raises(D.DataOrderError, match="Refusing"):
        D.verify_permutation(5, N, empty)


def test_verify_error_names_both_digests():
    """A mismatch must be diagnosable without re-running anything."""
    digest = D.seed_digest(5, N)
    with pytest.raises(D.DataOrderError) as exc:
        D.verify_permutation(6, N, digest)
    message = str(exc.value)
    assert digest in message
    assert D.seed_digest(6, N) in message
    assert "unreproducible" in message


# ---------------------------------------------------------------------------
# The reference sampler
# ---------------------------------------------------------------------------


def test_batches_partition_the_permutation_without_repeats():
    perm = D.sequence_permutation(0, N)
    batches = list(D.batch_indices(perm, batch_size=256))
    assert len(batches) == N // 256
    flat = [i for b in batches for i in b]
    assert flat == perm[:len(flat)]
    assert len(set(flat)) == len(flat), "a sequence is served twice"


def test_batches_refuse_to_run_past_the_corpus():
    perm = D.sequence_permutation(0, N)
    with pytest.raises(D.DataOrderError, match="token-budget identity"):
        list(D.batch_indices(perm, batch_size=256, total_steps=N))


def test_sequences_for_budget_matches_the_config_arithmetic():
    """The independent route to the corpus's sequence count.

    256 x 1024 x 9536 = 2,499,805,184 tokens = 2,441,216 sequences.
    """
    assert D.sequences_for_budget(256, 1024, 9536) == 2441216
    assert 2441216 * 1024 == 2499805184


def test_sequence_count_agrees_with_the_shipped_config():
    """Two routes to the same number: this module, and burst.config."""
    sys.path.insert(0, str(REPO_ROOT))
    from burst.config import load_config

    # The twin arm on purpose: it needs no injection fields, and those are
    # still null in configs/base.yaml, so every injecting arm refuses to load.
    # The corpus arithmetic under test is identical across arms.
    cfg = load_config(REPO_ROOT / "configs" / "base.yaml",
                      REPO_ROOT / "configs" / "runs" / "seed03_twin.yaml",
                      outdir=Path("/tmp/unused"))
    derived = D.sequences_for_budget(
        cfg.training.batch_size, cfg.training.seq_len, cfg.training.total_steps)
    assert derived * cfg.training.seq_len == cfg.corpus.expected_token_budget
