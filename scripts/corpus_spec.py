#!/usr/bin/env python
"""The corpus's geometry: what is where, and what may never touch what. Step 11.

One module so that the fetcher, the tokenizer, the verifier and a future
training loop cannot disagree about the layout. Pure standard library, so it
loads and is tested in the torch-free environment.

THE LAYOUT, and why the held-out slice is at the FRONT

    absolute token   0 .............. 10,485,760 .............. 2,510,290,944
                     |  HELD OUT      |  TRAINING SLICE                     |
                     |  10,240 seq    |  2,441,216 seq = 149 shards         |
                     |  heldout.bin   |  train-000.bin .. train-148.bin     |

The held-out slice exists because the loss barrier -- and CKA, and activation
similarity -- evaluate a trained model on text, and text the model trained on
measures memorisation on top of whatever the metric claims to report.

It sits at the front for three reasons, in ascending order of importance:

1. Its content depends only on the pinned revision, not on where 2.5B tokens
   happened to end. An end-placed slice would be silently redefined by any
   future change to expected_token_budget, making old barriers incomparable to
   new ones without anything appearing to change.
2. The training slice keeps EXACTLY expected_token_budget tokens, so the
   loader's `batch_size * seq_len * total_steps` assertion is untouched.
3. It places the corpus-derived texts this repo already committed --
   bursts/context.txt is openwebtext document 73, and two arm texts are
   documents 104 and 193 -- OUTSIDE the training slice. At the front they are
   held out with 50x to 114x margin. At the end, every one of them would have
   been trained on.

DISJOINTNESS IS STRUCTURAL, NOT ASSERTED

The sampler permutes `range(TRAIN_SEQUENCES)`, which indexes the training
shards and nothing else. The held-out slice is in a different file. So no
permutation can reach it -- that is an absence, not a check that could be
written wrongly.

`assert_heldout_disjoint_from_training` exists anyway, because "it cannot
happen" and "it did not happen" are different claims and only the second one
survives someone editing the geometry.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

# ---------------------------------------------------------------------------
# Source, pinned
# ---------------------------------------------------------------------------

#: The corpus is NAMED in configs/base.yaml and never located there. This is
#: the location, and it lives in code precisely so the config does not have to
#: carry it.
DATASET = "Skylion007/openwebtext"

#: Pinned. scripts/make_bursts.py streams whatever `main` currently is, which
#: means the corpus it cut bursts from is not identified by anything. A
#: revision makes "the corpus downloaded today" and "the corpus downloaded in a
#: month" the same object or provably different ones.
REVISION = "79d93d786212f7344586290adb811d4ae6a1762c"

N_SOURCE_FILES = 80
SOURCE_TEMPLATE = "plain_text/train-{index:05d}-of-00080.parquet"

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

SEQ_LEN = 1024

#: Ruled 2026-08-03. Chosen once, byte-identical across every run, never
#: derived from a per-run seed -- two barriers computed on different evaluation
#: text are not comparable, and this study compares barriers constantly.
HELDOUT_TOKENS = 10_485_760

#: configs/base.yaml's expected_token_budget. Duplicated here as a CONSTANT to
#: be checked against the config, not as a second source of truth; see
#: check_against_config.
TRAIN_TOKENS = 2_499_805_184

#: 16,384 sequences = 32 MiB per shard, and 2,441,216 / 16,384 = 149 exactly.
#: Chosen so the division is exact: a ragged final shard would make every
#: boundary a special case, and shard boundaries are one of the things that
#: must be checkable two ways.
SEQUENCES_PER_SHARD = 16_384

#: GPT-2's <|endoftext|>. Emitted after every document, and those tokens COUNT
#: toward the budget -- stated because "does the separator count" is exactly
#: the kind of unrecorded decision that makes two corpora differ by 0.1% and
#: nobody able to say why.
EOT_TOKEN = 50_256

#: uint16 because GPT-2's vocab is 50257 and fits under 65,536. Little-endian
#: stated explicitly: the shards are raw bytes with no header to carry it.
DTYPE = "uint16"
BYTES_PER_TOKEN = 2
BYTE_ORDER = "little"


class CorpusSpecError(Exception):
    """Raised for every refusal here. Never an assert."""


# ---------------------------------------------------------------------------
# Derived, and derived rather than written down
# ---------------------------------------------------------------------------

HELDOUT_SEQUENCES = HELDOUT_TOKENS // SEQ_LEN          # 10,240
TRAIN_SEQUENCES = TRAIN_TOKENS // SEQ_LEN              # 2,441,216
N_SHARDS = TRAIN_SEQUENCES // SEQUENCES_PER_SHARD      # 149
TOTAL_TOKENS = HELDOUT_TOKENS + TRAIN_TOKENS           # 2,510,290,944
TOTAL_SEQUENCES = HELDOUT_SEQUENCES + TRAIN_SEQUENCES  # 2,451,456

#: Absolute token boundaries. The training slice starts where held-out ends.
HELDOUT_START = 0
HELDOUT_END = HELDOUT_TOKENS
TRAIN_START = HELDOUT_TOKENS
TRAIN_END = HELDOUT_TOKENS + TRAIN_TOKENS

HELDOUT_FILENAME = "heldout.bin"
SHARD_TEMPLATE = "train-{index:03d}.bin"
MANIFEST_FILENAME = "manifest.json"


def _validate_geometry() -> None:
    """The arithmetic must close exactly. Checked at import, not documented.

    Every one of these is a division that has to come out whole. A remainder
    anywhere means a ragged shard or a partial sequence, and both are the kind
    of thing that produces a corpus which is almost right.
    """
    if HELDOUT_TOKENS % SEQ_LEN:
        raise CorpusSpecError(
            f"held-out slice {HELDOUT_TOKENS} is not a whole number of "
            f"{SEQ_LEN}-token sequences")
    if TRAIN_TOKENS % SEQ_LEN:
        raise CorpusSpecError(
            f"training budget {TRAIN_TOKENS} is not a whole number of "
            f"{SEQ_LEN}-token sequences")
    if TRAIN_SEQUENCES % SEQUENCES_PER_SHARD:
        raise CorpusSpecError(
            f"{TRAIN_SEQUENCES} training sequences do not divide into shards "
            f"of {SEQUENCES_PER_SHARD}; the final shard would be ragged")
    if TRAIN_START != HELDOUT_END:
        raise CorpusSpecError(
            "the training slice does not begin where the held-out slice ends; "
            "there is a gap or an overlap in the layout")


_validate_geometry()


# ---------------------------------------------------------------------------
# Shard arithmetic -- the second route to every boundary
# ---------------------------------------------------------------------------


def shard_sequence_range(shard_index: int) -> tuple:
    """[start, end) sequence indices for a shard, computed not looked up.

    The manifest also records these. Two routes to the same pair, so a manifest
    that disagrees with arithmetic is caught rather than believed.
    """
    if not 0 <= shard_index < N_SHARDS:
        raise CorpusSpecError(
            f"shard {shard_index} out of range 0..{N_SHARDS - 1}")
    start = shard_index * SEQUENCES_PER_SHARD
    return start, start + SEQUENCES_PER_SHARD


def shard_byte_size() -> int:
    """Bytes in a training shard. Every shard is the same size by construction.

    Took an unused `shard_index` argument until 2026-08-03, which implied a
    per-shard variability the exact geometry forbids -- 149 x 16,384 divides
    evenly, so there is no ragged final shard to be a special case.
    """
    return SEQUENCES_PER_SHARD * SEQ_LEN * BYTES_PER_TOKEN


def heldout_byte_size() -> int:
    return HELDOUT_SEQUENCES * SEQ_LEN * BYTES_PER_TOKEN


def tokens_from_byte_size(n_bytes: int) -> int:
    """Token count implied by a file's size. The independent route.

    Raw shards carry no header, which is exactly what makes this exact -- a
    .npy header would put an offset between size and count and cost this check.
    """
    if n_bytes % BYTES_PER_TOKEN:
        raise CorpusSpecError(
            f"{n_bytes} bytes is not a whole number of {BYTES_PER_TOKEN}-byte "
            "tokens, so this file is truncated mid-token")
    return n_bytes // BYTES_PER_TOKEN


def training_token_offset(sequence_index: int) -> int:
    """Absolute token offset of a TRAINING sequence index.

    The offset the sampler's indices correspond to in the whole-corpus stream.
    Note it is shifted past the held-out slice -- a sampler that forgot the
    shift would read real tokens at plausible offsets and train on the
    evaluation set.
    """
    if not 0 <= sequence_index < TRAIN_SEQUENCES:
        raise CorpusSpecError(
            f"training sequence {sequence_index} out of range "
            f"0..{TRAIN_SEQUENCES - 1}")
    return TRAIN_START + sequence_index * SEQ_LEN


# ---------------------------------------------------------------------------
# The held-out guarantee
# ---------------------------------------------------------------------------


def assert_heldout_disjoint_from_training(
        train_sequences: int = TRAIN_SEQUENCES) -> dict:
    """No training sequence index can address a held-out token. THE CHECK.

    Disjointness here is structural: the held-out slice is in its own file and
    the sampler's index space covers the training shards only. This function
    exists because "it cannot happen" and "it did not happen" are different
    claims, and only the second survives someone editing the geometry above.

    A held-out slice that silently overlaps the training slice is the same
    class of failure as a check that passes when handed nothing: the barrier
    would still return a number, the number would look reasonable, and it would
    be measuring memorisation.

    Callable by the training loop with the sequence count it actually intends
    to serve, so a corpus built to a different size is caught before the first
    batch rather than inferred afterwards.
    """
    if train_sequences <= 0:
        raise CorpusSpecError(
            f"train_sequences must be positive; got {train_sequences}")

    # Range-checked HERE, before training_token_offset is called. An earlier
    # version reached for the offset first, so this branch was unreachable --
    # the offset's own bounds check fired instead, with a message about a
    # sequence index rather than about a corpus that is shorter than the caller
    # believes. A safety check that can never fire is not a safety check, and
    # it took a test asserting the better message to notice.
    if train_sequences > TRAIN_SEQUENCES:
        raise CorpusSpecError(
            f"{train_sequences} training sequences were requested, past the "
            f"end of a corpus holding {TRAIN_SEQUENCES}. The corpus that was "
            f"built is not the corpus this caller intends to serve -- short by "
            f"{train_sequences - TRAIN_SEQUENCES} sequences.")

    lowest = training_token_offset(0)
    highest = training_token_offset(train_sequences - 1) + SEQ_LEN

    if lowest < HELDOUT_END:
        raise CorpusSpecError(
            f"THE TRAINING SLICE OVERLAPS THE HELD-OUT SLICE. The first "
            f"training sequence starts at absolute token {lowest}, which is "
            f"below the held-out boundary at {HELDOUT_END}. Every metric that "
            "evaluates a trained model on held-out text would be measuring "
            "memorisation, and would still return a plausible number.")
    return {
        "heldout_tokens": [HELDOUT_START, HELDOUT_END],
        "training_tokens": [lowest, highest],
        "gap_tokens": lowest - HELDOUT_END,
        "disjoint": True,
        "structural": ("the held-out slice is in a separate file and the "
                       "sampler's index space covers the training shards "
                       "only, so this is an absence rather than a check"),
    }


def check_against_config(cfg) -> dict:
    """This module's geometry against burst.config's. Two sources, compared.

    TRAIN_TOKENS is duplicated here deliberately -- not as a second source of
    truth but so that the duplicate can be checked. A constant that is only
    written once is never wrong and never verified; one written twice can
    disagree, and disagreeing loudly is the point.
    """
    budget = cfg.corpus.expected_token_budget
    derived = (cfg.training.batch_size * cfg.training.seq_len
               * cfg.training.total_steps)
    if derived != budget:
        raise CorpusSpecError(
            f"the config disagrees with itself: batch_size * seq_len * "
            f"total_steps = {derived}, expected_token_budget = {budget}")
    if budget != TRAIN_TOKENS:
        raise CorpusSpecError(
            f"corpus_spec.TRAIN_TOKENS ({TRAIN_TOKENS}) does not match "
            f"configs/base.yaml's expected_token_budget ({budget}). The corpus "
            "this module describes is not the corpus the study intends to "
            "train on.")
    if cfg.training.seq_len != SEQ_LEN:
        raise CorpusSpecError(
            f"corpus_spec.SEQ_LEN ({SEQ_LEN}) does not match the config's "
            f"seq_len ({cfg.training.seq_len})")
    return {
        "expected_token_budget": budget,
        "train_sequences": TRAIN_SEQUENCES,
        "heldout_tokens": HELDOUT_TOKENS,
        "total_tokens": TOTAL_TOKENS,
        "agrees": True,
    }


def summary() -> dict:
    """The whole layout as data, for a report to derive its numbers from."""
    return {
        "dataset": DATASET,
        "revision": REVISION,
        "seq_len": SEQ_LEN,
        "dtype": DTYPE,
        "byte_order": BYTE_ORDER,
        "eot_token": EOT_TOKEN,
        "heldout": {
            "tokens": HELDOUT_TOKENS,
            "sequences": HELDOUT_SEQUENCES,
            "bytes": heldout_byte_size(),
            "token_range": [HELDOUT_START, HELDOUT_END],
            "filename": HELDOUT_FILENAME,
        },
        "training": {
            "tokens": TRAIN_TOKENS,
            "sequences": TRAIN_SEQUENCES,
            "shards": N_SHARDS,
            "sequences_per_shard": SEQUENCES_PER_SHARD,
            "bytes_per_shard": shard_byte_size(),
            "bytes": TRAIN_SEQUENCES * SEQ_LEN * BYTES_PER_TOKEN,
            "token_range": [TRAIN_START, TRAIN_END],
        },
        "total": {
            "tokens": TOTAL_TOKENS,
            "sequences": TOTAL_SEQUENCES,
            "bytes": TOTAL_TOKENS * BYTES_PER_TOKEN,
        },
    }
