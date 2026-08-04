#!/usr/bin/env python
"""Per-seed data order for the burst study. Step 11.

WHAT THIS DECIDES

`configs/base.yaml` says the seed controls BOTH weight initialization AND data
order, and that two runs sharing a seed see the same batch sequence in the same
order. This module is the second half of that sentence: given a run seed, it
produces the order in which the tokenized corpus's sequences are consumed.

RULED 2026-08-03: order is PER-SEED, not global. Ten seeds give ten different
permutations of ONE fixed 2.5B-token pool -- not ten different slices, which is
arithmetically impossible anyway (ten disjoint 2.5B slices would need 25B
tokens and OpenWebText holds roughly 9B).

WHAT THAT COSTS AND BUYS, since a number downstream depends on it

Arm-vs-twin is UNAFFECTED. The twin shares the seed, so a burst arm and its
seed-matched twin still see identical weights and identical data in identical
order, differing only in the injected burst. The study's headline comparison is
a clean matched pair either way.

What changes is the SEED-ONLY NOISE FLOOR. Across seeds, runs now differ in
initialization AND data order at once, so the floor bundles two sources of
variation and is WIDER than an init-only floor would be. An effect has more to
clear. In exchange the burst is exercised against ten different data contexts,
so an effect that does clear it has been shown to survive them.

WHY NOT A LIBRARY PRNG

numpy explicitly does not guarantee that `Generator` streams are stable across
versions, and `random.shuffle` is a weaker guarantee than a cross-machine
reproducibility claim can rest on. The permutation here is derived from SHA-256
alone, so it is fully specified by this file and depends on no library at all --
this module imports nothing outside the standard library, which also means it
runs and is tested in the torch-free environment.

NEVER Python's `hash()`. It is randomised per process by PYTHONHASHSEED, so the
same input gives a different answer in a different run while every test inside
a single process still agrees with itself. `make_bursts.derived_seed` learned
this already and is reused here rather than re-derived, so there is exactly one
definition of the seed namespace.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from make_bursts import derived_seed  # noqa: E402  one definition, reused

#: The purpose string this module claims in the shared seed namespace. Each
#: purpose gets an independent stream, so what this module consumes cannot
#: shift what any other purpose produces.
PURPOSE = "data_order"

#: Bytes per key, and keys per SHA-256 digest. A digest is 32 bytes, so one
#: hash yields four 8-byte keys and the whole keystream costs n/4 hashes
#: rather than n.
KEY_BYTES = 8
KEYS_PER_DIGEST = hashlib.sha256().digest_size // KEY_BYTES


class DataOrderError(Exception):
    """Raised for every refusal here. Never an assert.

    Same reasoning as burst/config.py: `python -O` deletes asserts, and a
    reproducibility check that vanishes under an optimizer flag is worse than
    none.
    """


# ---------------------------------------------------------------------------
# The permutation
# ---------------------------------------------------------------------------


def _keystream(run_seed: int, n: int) -> list:
    """n 64-bit keys, derived from SHA-256 in counter mode.

    Fully specified here so that no library version can change it: block j is
    sha256(material + b"/" + j), and its 32 bytes are cut into four big-endian
    8-byte keys. Reproducing this in another language needs nothing but a
    SHA-256 implementation.
    """
    material = f"burst-study/v4/{run_seed}/{PURPOSE}".encode("utf-8")
    keys = []
    blocks = (n + KEYS_PER_DIGEST - 1) // KEYS_PER_DIGEST
    for block in range(blocks):
        digest = hashlib.sha256(
            material + b"/" + str(block).encode("ascii")).digest()
        for k in range(KEYS_PER_DIGEST):
            keys.append(int.from_bytes(
                digest[k * KEY_BYTES:(k + 1) * KEY_BYTES], "big"))
    del keys[n:]
    return keys


def sequence_permutation(run_seed: int, n_sequences: int) -> list:
    """The order seed `run_seed` consumes the corpus's sequences in.

    Returns a permutation of range(n_sequences): position i in the result is
    the sequence index served i-th.

    Ties break by index. Two 64-bit keys colliding across 2.44M items is
    around a one-in-six-million event, so this is not a practical concern --
    but an unspecified tiebreak is a reproducibility hole regardless of how
    rarely it opens, and Python's sort being stable is what closes it.
    """
    if n_sequences < 0:
        raise DataOrderError(f"n_sequences must be >= 0; got {n_sequences}")
    if not isinstance(run_seed, int) or isinstance(run_seed, bool):
        raise DataOrderError(f"run_seed must be an int; got {run_seed!r}")
    keys = _keystream(run_seed, n_sequences)
    return sorted(range(n_sequences), key=keys.__getitem__)


def permutation_digest(permutation) -> str:
    """SHA-256 over the permutation, as the contract's comparable value.

    Hashed as fixed-width big-endian 8-byte entries rather than as text, so the
    digest cannot change because a formatting decision changed.
    """
    hasher = hashlib.sha256()
    for index in permutation:
        hasher.update(int(index).to_bytes(8, "big"))
    return hasher.hexdigest()


def seed_digest(run_seed: int, n_sequences: int) -> str:
    """Convenience: the digest of the permutation this seed implies."""
    return permutation_digest(sequence_permutation(run_seed, n_sequences))


# ---------------------------------------------------------------------------
# THE CONTRACT
# ---------------------------------------------------------------------------


def verify_permutation(run_seed: int, n_sequences: int,
                       expected_digest: str) -> str:
    """Recompute this seed's order and REFUSE if it is not what was recorded.

    THIS IS AN OBLIGATION ON THE TRAINING LOOP, WHICH DOES NOT EXIST YET.
    Nothing in this repo calls it today. It is written now, and tested now, so
    that the training loop has something to call rather than a paragraph to
    re-implement from.

    The gap it closes is the one between "the pipeline defines an order" and
    "the run actually used it" -- the same gap the commit hash in
    run_provenance.yaml closes for code. A corpus manifest that merely RECORDS
    a permutation digest proves nothing: a training loop that shuffled its own
    way would produce a perfectly plausible run whose provenance file claims an
    order it never served, and no measurement afterwards could tell.

    So the training loop must call this BEFORE CONSUMING A SINGLE BATCH, with
    the digest read from the corpus manifest. It is cheap -- one pass -- and it
    is the only thing standing between a recorded order and a served one.

    Returns the recomputed digest on success so a caller can log it.
    """
    if not expected_digest or not isinstance(expected_digest, str):
        raise DataOrderError(
            "expected_digest must be the digest string recorded in the corpus "
            f"manifest; got {expected_digest!r}. Refusing to 'verify' against "
            "nothing -- a check that passes when handed no expectation is "
            "worse than no check, because it reads as one.")
    actual = seed_digest(run_seed, n_sequences)
    if actual != expected_digest:
        raise DataOrderError(
            f"DATA ORDER DOES NOT MATCH THE MANIFEST for seed {run_seed}.\n"
            f"  recorded:   {expected_digest}\n"
            f"  recomputed: {actual}\n"
            f"  n_sequences: {n_sequences}\n"
            "The order this process would serve is not the order the corpus "
            "manifest records, so this run would not be the run the manifest "
            "describes. Either n_sequences disagrees with the corpus that was "
            "built, or the permutation derivation changed. Do not train "
            "through this: a run that consumes a different order than its "
            "provenance claims is unreproducible and silently so.")
    return actual


# ---------------------------------------------------------------------------
# The reference sampler
# ---------------------------------------------------------------------------


def batch_indices(permutation, batch_size: int, total_steps: int = None):
    """Yield one tuple of sequence indices per optimizer step.

    The reference implementation of "what does step s train on". A training
    loop may read the corpus however it likes, but if it disagrees with this it
    is not running the study's data order.

    Deliberately yields INDICES, not tokens. Reading bytes off disk is the
    training loop's business and depends on how it memory-maps the shards;
    which sequences make up step s is this module's business and must not
    depend on that.
    """
    if batch_size <= 0:
        raise DataOrderError(f"batch_size must be positive; got {batch_size}")
    n = len(permutation)
    available = n // batch_size
    if total_steps is None:
        total_steps = available
    if total_steps > available:
        raise DataOrderError(
            f"{total_steps} steps of {batch_size} sequences needs "
            f"{total_steps * batch_size} sequences, but the permutation holds "
            f"{n}. The corpus is short by "
            f"{total_steps * batch_size - n} sequences -- this is the "
            "token-budget identity failing, not a sampling detail.")
    for step in range(total_steps):
        yield tuple(permutation[step * batch_size:(step + 1) * batch_size])


def sequences_for_budget(batch_size: int, seq_len: int,
                         total_steps: int) -> int:
    """How many sequences the token budget implies. Checked, not assumed.

    An independent route to the same number the corpus manifest will report,
    so the two can be compared rather than one being trusted.
    """
    tokens = batch_size * seq_len * total_steps
    if tokens % seq_len:
        raise DataOrderError(
            f"token budget {tokens} is not a whole number of {seq_len}-token "
            "sequences")
    return tokens // seq_len
