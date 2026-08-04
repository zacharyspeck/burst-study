#!/usr/bin/env python
"""Generate the burst passages of spec v4, all at one token length.

    python scripts/make_bursts.py --k 5
    python scripts/make_bursts.py --k 5 --seed 3 --outdir /tmp/try

THE SIX INJECTING ARMS, in descending order of linguistic structure.
(This module also still generates scrambled-corpus, which was CUT as an
arm on 2026-08-03. Its text stays in bursts/ so measurements taken from
it remain reproducible; it is not a run condition. See S79.)

    fluent-false      grammatical English asserting something specific and
                      false. Hand-written, fixed, NEVER generated. Its token
                      count is the target length N.
    fluent-true       same register and structure, asserting something true.
                      Hand-written, fixed, NEVER generated.
    scrambled-false   fluent-false with word order broken inside
                      non-overlapping windows of size k.
    scrambled-true    fluent-true, same treatment.
    pos-substituted   each word replaced by a random word of the same part of
                      speech. Grammar kept, lexical content destroyed.
    random-chars      random printable ASCII, no word structure at all.

The no-injection twin is not a burst and does not appear here.

WHAT IS GUARANTEED
- Every generated file comes out at exactly N tokens under the GPT-2
  tokenizer,
  asserted by re-reading them from disk before this script exits.
- Same --seed and --k gives byte-identical output, provenance.json included.
- Each arm draws from its OWN generator seeded independently (see
  derived_seed). Adding, removing or reordering an arm cannot change any
  other arm's bytes. The previous version threaded one generator through
  everything, so it could and did.
- The two hand-written arms are opened read-only, behind an explicit guard.

THE CONFIG SYSTEM'S ARM LIST NOW AGREES WITH THIS ONE. Until 2026-08-03
configs/base.yaml enumerated the retired v3 arms while this script generated
v4's, and that inconsistency was deliberate and recorded. It has been
reconciled: burst/config.py's ARMS, configs/base.yaml's experiment.arms and
this module now name the same six injecting arms plus twin.
injection.burst_text_paths is one null per injecting arm and is still
undecided, which is a different thing from being inconsistent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from burst_match import (  # noqa: E402
    DEFAULT_BASE_CONFIG,
    BurstMatchError,
    load_tokenizer,
    resolve_seq_len,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET = "Skylion007/openwebtext"
SPLIT = "train"

DEFAULT_CACHE = REPO_ROOT / ".corpus-cache" / "openwebtext_slice.jsonl"
DEFAULT_OUTDIR = REPO_ROOT / "bursts"
PROVENANCE_NAME = "provenance.json"
POS_POOL_NAME = "pos_pool.json"
CONTEXT_NAME = "context.txt"

#: Bumped from 1 (spec v3). A reader can tell the two apart without guessing.
PROVENANCE_SCHEMA_VERSION = 2

DEFAULT_DOCS = 200

# Generator names. Recorded verbatim in provenance.
HAND_WRITTEN = "hand-written"
WINDOW_SHUFFLE = "window_shuffle"
POS_SUBSTITUTE = "pos_substitute"
RANDOM_ASCII = "random_ascii"

#: random-chars alphabet: printable ASCII EXCLUDING space, 94 characters.
#:
#: Space is deliberately absent. This arm is the floor of the structure
#: ladder -- "no word structure at all" -- and a space is a word boundary. An
#: earlier draft included space and justified it as avoiding one unbroken
#: run, which was self-refuting arithmetic: drawing uniformly from 95
#: characters puts a space every ~95 characters on average, and the whole
#: passage is only a few hundred characters, so it would have produced two to
#: four spaces. That is an unbroken run with extra steps. Excluding space is
#: the honest version of the same thing.
RANDOM_CHARS_ALPHABET_NAME = "ascii_33_126"
RANDOM_CHARS_ALPHABET = "".join(chr(c) for c in range(33, 127))
RANDOM_CHARS_SPACE_FREQUENCY = 0.0

#: The context passage is PINNED, not drawn blind. It was read and approved
#: by a human before being committed. Blind selection previously surfaced a
#: passage on race-and-IQ research (D11), and the context is fixed scaffolding
#: shared byte-identically by every arm rather than an experimental variable,
#: so there is nothing to be gained by drawing it at random and something to
#: lose. Logged as D12.
#:
#: context.txt is a FULL sequence (training.seq_len tokens), not just the
#: filler. The arms use its leading (seq_len - N) tokens as the filler they
#: are spliced into; the no-burst diagnostic row uses the whole thing. Storing
#: the full sequence is what makes that diagnostic a like-for-like comparison
#: -- a shorter filler-only sequence would have a different token count, and
#: gradient norms across different lengths are not comparable.
CONTEXT_DOC_INDEX = 73
CONTEXT_WORD_START = 0
CONTEXT_WORD_COUNT = 760

# --- span quality filters (unchanged from v3) ------------------------------
MAX_NON_ASCII_FRACTION = 0.10
MAX_PUNCT_DIGIT_FRACTION = 0.25
MIN_ALPHA_FRACTION = 0.40
MIN_SENTENCE_ENDS = 3
SENTENCE_ENDS = ".!?"

RULE = "=" * 78
THIN = "-" * 78


class MakeBurstsError(Exception):
    """Every failure this script raises on purpose."""


# ---------------------------------------------------------------------------
# The arm registry
#
# One declaration that everything iterates. The previous version kept three
# loose filename constants referenced in a dozen places, which is why adding
# two arms is a rewrite rather than an edit.
#
# Hand-written arms are IN the registry. They are never generated, but they
# are validated and they do get provenance entries -- keeping them out would
# mean two different answers to "what is an arm", which is how the loose
# constants happened in the first place.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmSpec:
    name: str
    filename: str
    generator: str
    #: True if this arm cuts a span of real corpus text. Drives how many
    #: distinct documents select_spans has to find (requirement H8) -- the
    #: count is never written down as a literal.
    needs_span: bool
    #: Static generation parameters, as (key, value) pairs so a frozen
    #: dataclass cannot hand out a mutable dict. CLI values are merged on top
    #: at run time.
    param_pairs: tuple = ()
    #: For arms built by degrading another arm's text rather than a corpus
    #: span: the name of the arm they are derived from. This is what lets
    #: truth value cross with structure -- scrambled-true and scrambled-false
    #: are the SAME treatment applied to passages that differ only in truth.
    derives_from: str | None = None

    @property
    def params(self) -> dict:
        return dict(self.param_pairs)

    @property
    def is_generated(self) -> bool:
        return self.generator != HAND_WRITTEN


#: WORD MULTIPLIERS, one per corpus-derived arm, justified individually.
#: Requirement H6 -- there is deliberately no single global constant, because
#: token density per word differs by arm and one number cannot be right for
#: all of them.
#:
#:   scrambled 2.0    Measured 3.78 bytes/token on shuffled English against
#:                    4.96 for fluent prose: shuffling already breaks BPE
#:                    merges and multiplies space-prefixed variants, so a
#:                    word buys fewer tokens. 2N words is comfortably over N.
#:   pos-substituted 2.5  Substituted words skew rarer than the originals, and
#:                    rarer words fragment into more tokens, so this arm needs
#:                    FEWER words than scrambled for the same token count. The
#:                    multiplier is nevertheless higher, because oversampling
#:                    is the safe direction: excess is trimmed away, a
#:                    shortfall is an error.
#:   random-chars     no multiplier at all -- it has no words. It oversamples
#:                    CHARACTERS instead, at 3.0 x N, which at roughly one
#:                    token per one-to-two characters is generous.

#: THE ARM GRID. Two axes: how degraded the structure is (rows), and what the
#: content was made from (columns).
#:
#:                 | false          | true          | corpus            |
#:   --------------+----------------+---------------+-------------------+
#:   fluent        | fluent-false   | fluent-true   | (ordinary: cut)   |
#:   scrambled     | scrambled-false| scrambled-true| scrambled-corpus  |
#:   pos-subst.    | --             | --            | pos-substituted   |
#:   random        | --             | --            | random-chars      |
#:
#: The two scrambled-* truth arms are why this grid exists. Before them, truth
#: value lived only on the top row while structure degradation ran down the
#: whole ladder, so the two dimensions could not be crossed and were
#: confounded. `scrambled` was renamed `scrambled-corpus` at the same time: it
#: is a scrambled corpus span with no truth value, and leaving it called
#: `scrambled` while two other arms were also scrambled was ambiguous.
ARM_SPECS: tuple[ArmSpec, ...] = (
    ArmSpec("fluent-false", "fluent_false.txt", HAND_WRITTEN, False),
    ArmSpec("fluent-true", "fluent_true.txt", HAND_WRITTEN, False),
    ArmSpec("scrambled-false", "scrambled_false.txt", WINDOW_SHUFFLE, False,
            derives_from="fluent-false"),
    ArmSpec("scrambled-true", "scrambled_true.txt", WINDOW_SHUFFLE, False,
            derives_from="fluent-true"),
    ArmSpec("scrambled-corpus", "scrambled_corpus.txt", WINDOW_SHUFFLE, True,
            (("span_word_multiplier", 2.0),)),
    ArmSpec("pos-substituted", "pos_substituted.txt", POS_SUBSTITUTE, True,
            (("span_word_multiplier", 2.5),)),
    ArmSpec("random-chars", "random_chars.txt", RANDOM_ASCII, False,
            (("alphabet", RANDOM_CHARS_ALPHABET_NAME),
             ("space_frequency", RANDOM_CHARS_SPACE_FREQUENCY),
             ("oversample_chars", 3.0))),
)

#: The arm whose token count defines N for every other arm.
REFERENCE_ARM = "fluent-false"


def arm_by_name(name: str) -> ArmSpec:
    for spec in ARM_SPECS:
        if spec.name == name:
            return spec
    raise MakeBurstsError(f"no such arm: {name!r}")


def span_arms() -> tuple[ArmSpec, ...]:
    """Arms that need a corpus span, in registry order."""
    return tuple(spec for spec in ARM_SPECS if spec.needs_span)


# ---------------------------------------------------------------------------
# Seed derivation
# ---------------------------------------------------------------------------


def derived_seed(run_seed: int, purpose: str) -> int:
    """A stable per-purpose seed, from SHA-256 of a fixed string.

    NOT Python's built-in hash(). hash() on a string is randomised per process
    by PYTHONHASHSEED, so the same input gives a different number in a
    different run -- which would destroy reproducibility silently, while every
    test still passed, because nothing in a single process would ever disagree
    with itself.

    Each arm gets its own stream from this, so no arm's output depends on how
    many random draws another arm consumed. That independence is the whole
    point: it means adding a sixth arm cannot change the first five.
    """
    material = f"burst-study/v4/{run_seed}/{purpose}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


# ---------------------------------------------------------------------------
# Pure text operations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpanStats:
    n_chars: int
    non_ascii_fraction: float
    punct_digit_fraction: float
    alpha_fraction: float
    sentence_ends: int

    @property
    def rejections(self) -> tuple[str, ...]:
        bad = []
        if self.non_ascii_fraction > MAX_NON_ASCII_FRACTION:
            bad.append(f"non-ASCII {self.non_ascii_fraction:.0%} > "
                       f"{MAX_NON_ASCII_FRACTION:.0%}")
        if self.punct_digit_fraction > MAX_PUNCT_DIGIT_FRACTION:
            bad.append(f"punctuation/digits {self.punct_digit_fraction:.0%} > "
                       f"{MAX_PUNCT_DIGIT_FRACTION:.0%}")
        if self.alpha_fraction < MIN_ALPHA_FRACTION:
            bad.append(f"alphabetic {self.alpha_fraction:.0%} < "
                       f"{MIN_ALPHA_FRACTION:.0%}")
        if self.sentence_ends < MIN_SENTENCE_ENDS:
            bad.append(f"{self.sentence_ends} sentence-ending marks < "
                       f"{MIN_SENTENCE_ENDS}")
        return tuple(bad)

    @property
    def acceptable(self) -> bool:
        return not self.rejections


def span_stats(text: str) -> SpanStats:
    """Measure a span against the quality filters.

    Fractions are over non-whitespace characters, so they do not depend on how
    the source page happened to be wrapped.
    """
    body = [c for c in text if not c.isspace()]
    n = len(body)
    if n == 0:
        return SpanStats(0, 0.0, 0.0, 0.0, 0)
    non_ascii = sum(1 for c in body if ord(c) > 127)
    alpha = sum(1 for c in body if c.isalpha())
    digits = sum(1 for c in body if c.isdigit())
    punct = n - alpha - digits
    return SpanStats(
        n_chars=n,
        non_ascii_fraction=non_ascii / n,
        punct_digit_fraction=(punct + digits) / n,
        alpha_fraction=alpha / n,
        sentence_ends=sum(text.count(c) for c in SENTENCE_ENDS),
    )


def window_shuffle(text: str, k: int, rng: random.Random) -> str:
    """Shuffle word order within non-overlapping windows of size k.

    The final partial window is shuffled too. Leaving it alone would put a run
    of untouched original text at the end of every scrambled passage, which is
    a systematic difference from the rest of it.
    """
    if k < 1:
        raise MakeBurstsError(f"window size k must be at least 1, got {k}")
    words = text.split()
    out: list[str] = []
    for start in range(0, len(words), k):
        window = words[start:start + k]
        rng.shuffle(window)
        out.extend(window)
    return " ".join(out)


def window_shuffle_to_length(text: str, k: int, n_tokens: int, tokenizer,
                             seed: int, label: str,
                             max_attempts: int = 64) -> tuple[str, int]:
    """Shuffle until the result is at least `n_tokens` long. (text, attempts).

    WHY THIS EXISTS. The corpus-derived arms oversample two to two-and-a-half
    times and trim. The arms derived from a hand-written passage cannot: their
    source IS exactly N tokens, and shuffling it lands anywhere from N-1 to
    N+2 because reordering words changes which byte-pair merges apply. When it
    lands short there is nothing to trim, and the arm cannot be built.

    Measured acceptance -- fraction of seeds giving at least 194 tokens:

        fluent-true    k=2 100%   k=5 55%   k=15 28%   full 22%
        fluent-false   k=2 100%   k=5 100%  k=15 100%  full 100%

    So redraw until it fits, from a stream seeded off the arm's own seed.

    THE ALTERNATIVE WAS WORSE. k=2 is the only window where every draw is long
    enough, and k=2 leaves 37.8% of the original adjacent word pairs intact
    and in order -- for the scrambled half of the study's primary contrast.
    Fixing k at 2 would let a byte-pair-encoding artifact choose the
    scrambling strength of the experiment.

    THE BIAS THIS INTRODUCES IS MEASURED AND NEGLIGIBLE. Rejecting draws that
    tokenize short selects mildly for denser tokenization. Comparing accepted
    against rejected draws at k=15, where selection is harshest:

        accepted  mean burst-region loss 7.3701 (sd 0.0884, n=18)
        rejected  mean burst-region loss 7.3629 (sd 0.1080, n=18)
        difference 0.0071 nats -- 0.07 pooled SD

    Against a k=2-to-k=15 effect of about 1.3 nats, the bias is roughly 180
    times smaller than the thing it buys. The attempt count is recorded in
    provenance so the selection stays visible rather than implicit.
    """
    for attempt in range(max_attempts):
        # A fresh stream per attempt, derived from the arm's own seed, so the
        # sequence of attempts is reproducible and independent of every other
        # arm.
        candidate = window_shuffle(text, k, random.Random(seed + attempt))
        if token_count(tokenizer, candidate) >= n_tokens:
            return candidate, attempt + 1
    raise MakeBurstsError(
        f"{label}: no shuffle of this passage reached {n_tokens} tokens in "
        f"{max_attempts} attempts at k={k}. The source is {token_count(tokenizer, text)} "
        "tokens; shuffling it can only move the count by a token or two, so a "
        "smaller k or a longer source passage is needed. Not approximated."
    )


def random_ascii_text(n_chars: int, rng: random.Random) -> str:
    """A run of random printable ASCII, no spaces. See the alphabet comment."""
    if n_chars < 1:
        raise MakeBurstsError(f"need at least 1 character, got {n_chars}")
    return "".join(rng.choice(RANDOM_CHARS_ALPHABET) for _ in range(n_chars))


def split_affixes(word: str) -> tuple[str, str, str]:
    """Break a whitespace word into (leading punctuation, core, trailing).

    The core is what gets substituted; the punctuation is reattached
    afterwards, which is what keeps sentence shape -- commas, full stops,
    quotes -- intact when the words underneath are replaced.
    """
    start, end = 0, len(word)
    while start < end and not (word[start].isalnum()):
        start += 1
    while end > start and not (word[end - 1].isalnum()):
        end -= 1
    return word[:start], word[start:end], word[end:]


def capitalisation_of(core: str) -> str:
    """'upper', 'title' or 'lower' -- which shape to give the substitute."""
    if len(core) > 1 and core.isupper():
        return "upper"
    if core[:1].isupper():
        return "title"
    return "lower"


def apply_capitalisation(word: str, shape: str) -> str:
    if shape == "upper":
        return word.upper()
    if shape == "title":
        return word[:1].upper() + word[1:]
    return word.lower()


def pos_substitute(template: list[dict], pools: dict, rng: random.Random) -> str:
    """Rebuild a passage from a POS template, drawing words from the pools.

    The template is a committed list of per-word entries recording the part of
    speech, the punctuation that was attached to the original word, and its
    capitalisation. No tagger runs here -- that is requirement H7. Tagging
    happened once, at pool-build time, and its output is committed.
    """
    out: list[str] = []
    for entry in template:
        if entry["kind"] == "literal":
            out.append(entry["text"])
            continue
        tag = entry["tag"]
        pool = pools.get(tag)
        if not pool:
            raise MakeBurstsError(
                f"the POS pool has no words for tag {tag!r}, which the "
                f"committed template requires. Rebuild it with "
                f"scripts/build_pos_pool.py."
            )
        word = apply_capitalisation(rng.choice(pool), entry["cap"])
        out.append(f"{entry['lead']}{word}{entry['trail']}")
    return " ".join(out)


def cut_is_mid_word(full: str, trimmed: str) -> bool:
    """True if trimming split a word in half. Acceptable, but logged."""
    if not trimmed or len(trimmed) >= len(full):
        return False
    return not full[len(trimmed)].isspace() and not trimmed[-1].isspace()


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def trim_to_tokens(tokenizer, text: str, n_tokens: int,
                   label: str = "") -> tuple[str, bool]:
    """Cut `text` to exactly `n_tokens` tokens. Returns (text, cut_mid_word).

    Decoding a prefix of the IDs and re-encoding it is not guaranteed to give
    the same count back, so the result is re-encoded and corrected rather than
    trusted. Requirement H5: if it will not converge, this raises naming the
    arm and the counts. It never silently approximates.
    """
    where = f"{label}: " if label else ""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) < n_tokens:
        raise MakeBurstsError(
            f"{where}cannot trim to {n_tokens} tokens: the material is only "
            f"{len(ids)} tokens long. Raise this arm's oversample factor."
        )

    take = n_tokens
    seen = set()
    for _ in range(24):
        candidate = tokenizer.decode(ids[:take])
        actual = token_count(tokenizer, candidate)
        if actual == n_tokens:
            return candidate, cut_is_mid_word(text, candidate)
        if take in seen:
            break
        seen.add(take)
        take += n_tokens - actual
        if not 0 < take <= len(ids):
            break
    raise MakeBurstsError(
        f"{where}could not trim to exactly {n_tokens} tokens; the tokenizer "
        f"does not round-trip this text (last attempt gave {actual} tokens "
        f"from a {take}-token slice). This is reported rather than "
        "approximated: an arm at the wrong length would make every "
        "comparison against it meaningless."
    )


# ---------------------------------------------------------------------------
# The corpus slice
# ---------------------------------------------------------------------------


def load_corpus_slice(cache_path: Path, n_docs: int, stream=None) -> list[str]:
    """The cached documents, downloading them once if they are not there."""
    stream = sys.stdout if stream is None else stream

    if cache_path.is_file():
        docs = [json.loads(line)["text"]
                for line in cache_path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        if len(docs) >= n_docs:
            print(f"corpus:  {len(docs)} documents from cache {cache_path}",
                  file=stream)
            return docs[:n_docs]
        print(f"corpus:  cache {cache_path} holds {len(docs)} documents, "
              f"{n_docs} wanted -- refetching", file=stream)

    print(f"corpus:  {DATASET} ({SPLIT}, streaming) -- fetching {n_docs} "
          "documents", file=stream)
    print(f"         cache: {cache_path}", file=stream)
    stream.flush()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise MakeBurstsError(
            "the `datasets` package is not installed, so the corpus cannot be "
            "streamed.\n    pip install -e \".[measure]\"\n"
            f"(underlying import error: {exc})"
        ) from exc

    docs: list[str] = []
    try:
        dataset = load_dataset(DATASET, split=SPLIT, streaming=True)
        iterator = iter(dataset)
        for _ in range(n_docs):
            docs.append(next(iterator)["text"])
        iterator.close()
    except StopIteration:
        pass
    except Exception as exc:
        raise MakeBurstsError(
            f"no cached corpus at {cache_path}, and streaming {DATASET} from "
            "HuggingFace failed.\nEither connect to a network and run it "
            "again, or copy a populated cache file into place from a machine "
            f"that has one.\nthe download said: {exc}"
        ) from exc

    if not docs:
        raise MakeBurstsError(f"{DATASET} yielded no documents.")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, text in enumerate(docs):
            handle.write(json.dumps({"index": index, "text": text},
                                    ensure_ascii=False) + "\n")
    print(f"         cached {len(docs)} documents", file=stream)
    return docs


# ---------------------------------------------------------------------------
# Span selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Span:
    doc_index: int
    word_start: int
    word_count: int
    text: str
    stats: SpanStats


def select_spans(docs: list[str], n_spans: int, span_words: int,
                 rng: random.Random, stream=None,
                 exclude_docs: frozenset = frozenset()) -> list[Span]:
    """Pick `n_spans` acceptable spans, each from a different document.

    `n_spans` comes from the arm registry, never from a literal at the call
    site (requirement H8). If the pool cannot supply enough distinct
    documents, this raises naming how many were needed and how many were
    found, rather than quietly returning fewer.

    `exclude_docs` keeps the context passage's document out of the pool, so no
    arm can be cut from the same document as the filler that surrounds it.
    """
    stream = sys.stdout if stream is None else stream

    candidates: list[tuple[int, int]] = []
    doc_words: list[list[str]] = []
    for doc_index, text in enumerate(docs):
        words = text.split()
        doc_words.append(words)
        if doc_index in exclude_docs:
            continue
        for start in range(0, max(1, len(words) - span_words + 1), span_words):
            if start + span_words <= len(words):
                candidates.append((doc_index, start))

    if not candidates:
        raise MakeBurstsError(
            f"no document in the cached slice holds {span_words} words "
            f"(excluding {sorted(exclude_docs)}); raise --docs."
        )

    rng.shuffle(candidates)

    chosen: list[Span] = []
    used_docs: set[int] = set()
    rejected = 0
    for doc_index, start in candidates:
        if doc_index in used_docs:
            continue
        words = doc_words[doc_index][start:start + span_words]
        text = " ".join(words)
        stats = span_stats(text)
        if not stats.acceptable:
            rejected += 1
            continue
        chosen.append(Span(doc_index, start, len(words), text, stats))
        used_docs.add(doc_index)
        if len(chosen) == n_spans:
            print(f"spans:   {len(chosen)} selected, {rejected} rejected by "
                  "the quality filters", file=stream)
            return chosen

    raise MakeBurstsError(
        f"needed {n_spans} spans from distinct documents but found only "
        f"{len(chosen)} after examining {len(candidates)} candidates across "
        f"{len(docs)} documents ({rejected} rejected by the quality filters). "
        "Raise --docs to widen the pool."
    )


def arm_spans(docs: list[str], run_seed: int, n_target: int,
              stream=None) -> dict:
    """{arm name: Span} for every corpus-derived arm.

    Shared by this script and scripts/build_pos_pool.py so that the span the
    POS template was built from is provably the span the generator uses. One
    selection pass, at the largest word count any arm asks for; each arm then
    takes only the leading words its own multiplier calls for.
    """
    specs = span_arms()
    span_words = max(
        int(n_target * spec.params["span_word_multiplier"]) for spec in specs
    )
    rng = random.Random(derived_seed(run_seed, "span-selection"))
    spans = select_spans(docs, len(specs), span_words, rng, stream=stream,
                         exclude_docs=frozenset({CONTEXT_DOC_INDEX}))
    return {spec.name: span for spec, span in zip(specs, spans)}


def span_words_for(spec: ArmSpec, span: Span, n_target: int) -> str:
    """The leading slice of `span` that this arm's own multiplier asks for."""
    want = int(n_target * spec.params["span_word_multiplier"])
    return " ".join(span.text.split()[:want])


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def write_text(path: Path, text: str) -> None:
    """UTF-8, LF endings, no trailing newline.

    newline="\\n" is not optional on Windows: without it Python turns every
    \\n into \\r\\n and the file stops being byte-identical to the same file
    written on the cluster.

    No trailing newline, because a trailing newline is a TOKEN. All five arms
    must tokenize to the same length, so the convention has to be identical
    across them, and "none" is the only one that keeps N equal to the token
    count of the text itself.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def print_span(label: str, span: Span, stream=None) -> None:
    stream = sys.stdout if stream is None else stream
    print(file=stream)
    print(THIN, file=stream)
    print(f"{label}: raw span, document {span.doc_index}, "
          f"words {span.word_start}..{span.word_start + span.word_count} "
          "(BEFORE shuffling, substitution or trimming)", file=stream)
    print(THIN, file=stream)
    print(span.text, file=stream)
    print(THIN, file=stream)
    s = span.stats
    print(f"  {s.n_chars} non-space chars | alphabetic {s.alpha_fraction:.0%} "
          f"| punctuation+digits {s.punct_digit_fraction:.0%} | non-ASCII "
          f"{s.non_ascii_fraction:.0%} | {s.sentence_ends} sentence ends",
          file=stream)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/make_bursts.py",
        description=(
            "Generate the five spec-v4 burst passages at one token length, "
            "plus the shared context sequence. Never modifies the two "
            "hand-written arms."
        ),
    )
    parser.add_argument(
        "--k", type=int, required=True, metavar="N",
        help=("window size for the scrambled arm's word shuffle. Required: "
              "which k to use is not a decision this script makes"),
    )
    parser.add_argument("--seed", type=int, default=0, metavar="N",
                        help="run seed; every arm derives its own from it "
                             "(default: 0)")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR,
                        metavar="PATH")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE,
                        metavar="PATH")
    parser.add_argument("--docs", type=int, default=DEFAULT_DOCS, metavar="N")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG,
                        metavar="PATH",
                        help="config to read training.seq_len from")
    parser.add_argument("--seq-len", type=int, default=None, metavar="N",
                        help="override the sequence length read from config")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    print(RULE)
    print("make_bursts -- five spec-v4 burst passages at one token length")
    print(RULE)
    try:
        return _run(args)
    except (MakeBurstsError, BurstMatchError) as exc:
        print(f"\nERROR\n{exc}\n", file=sys.stderr)
        return 1


def _run(args) -> int:
    if args.k < 1:
        raise MakeBurstsError(f"--k must be at least 1, got {args.k}")

    outdir = Path(args.outdir)
    paths = {spec.name: outdir / spec.filename for spec in ARM_SPECS}
    context_path = outdir / CONTEXT_NAME

    # The guard. Hand-written arms are the fixed points every generated arm is
    # matched to; a script that regenerated one would redefine the target
    # length underneath every measurement ever taken.
    handwritten = {spec.name for spec in ARM_SPECS if not spec.is_generated}
    generated_paths = {paths[s.name].resolve() for s in ARM_SPECS if s.is_generated}
    generated_paths |= {context_path.resolve(), (outdir / PROVENANCE_NAME).resolve()}
    for name in handwritten:
        if paths[name].resolve() in generated_paths:
            raise MakeBurstsError(
                f"refusing to run: an output path collides with the "
                f"hand-written arm {name!r}, which this script must never "
                "write."
            )
        if not paths[name].is_file():
            raise MakeBurstsError(
                f"the hand-written arm {name!r} is missing at {paths[name]}. "
                "It is written by hand and is not generated by this script; "
                "create it first."
            )

    seq = resolve_seq_len(args.seq_len, args.base_config)
    tokenizer = load_tokenizer()

    reference = arm_by_name(REFERENCE_ARM)
    n_target = token_count(
        tokenizer, paths[REFERENCE_ARM].read_text(encoding="utf-8"))
    # The context file holds a WHOLE sequence. Arms are spliced into its
    # leading (seq_len - N) tokens; the no-burst diagnostic uses all of it.
    context_tokens = seq.value
    filler_tokens = seq.value - n_target
    if filler_tokens < 1:
        raise MakeBurstsError(
            f"the burst ({n_target} tokens) does not fit inside a sequence of "
            f"{seq.value} tokens."
        )

    print(f"target:  N = {n_target} tokens, from {reference.filename}")
    print(f"seq_len: {seq.value} tokens  ({seq.source})")
    print(f"context: {context_tokens} tokens; arms splice into the leading "
          f"{filler_tokens}")
    print(f"seed:    {args.seed} | k = {args.k}")

    docs = load_corpus_slice(Path(args.cache), args.docs)
    spans = arm_spans(docs, args.seed, n_target)

    # --- the shared context sequence ---------------------------------------
    if CONTEXT_DOC_INDEX >= len(docs):
        raise MakeBurstsError(
            f"the pinned context document {CONTEXT_DOC_INDEX} is outside the "
            f"{len(docs)}-document cache; raise --docs."
        )
    context_source = " ".join(
        docs[CONTEXT_DOC_INDEX].split()[
            CONTEXT_WORD_START:CONTEXT_WORD_START + CONTEXT_WORD_COUNT])
    context_stats = span_stats(context_source)
    if not context_stats.acceptable:
        raise MakeBurstsError(
            f"the pinned context span fails the quality filters: "
            f"{context_stats.rejections}"
        )
    context_text, context_cut = trim_to_tokens(
        tokenizer, context_source, context_tokens, label="context")
    write_text(context_path, context_text)

    # --- the generated arms -------------------------------------------------
    results: dict = {}
    for spec in ARM_SPECS:
        if not spec.is_generated:
            continue
        rng = random.Random(derived_seed(args.seed, spec.name))
        params = dict(spec.params)
        params["derived_seed"] = derived_seed(args.seed, spec.name)

        if spec.generator == WINDOW_SHUFFLE and spec.derives_from:
            # Same treatment as scrambled-corpus, applied to another arm's
            # text instead of a corpus span. This is what makes truth value
            # crossable with structure: scrambled-true and scrambled-false
            # differ only in what was shuffled.
            source_spec = arm_by_name(spec.derives_from)
            source_text = paths[source_spec.name].read_text(encoding="utf-8")
            params["k"] = args.k
            params["derives_from"] = spec.derives_from
            span = None
            material, attempts = window_shuffle_to_length(
                source_text, args.k, n_target, tokenizer,
                derived_seed(args.seed, spec.name), spec.name)
            params["shuffle_attempts"] = attempts
        elif spec.generator == WINDOW_SHUFFLE:
            params["k"] = args.k
            span = spans[spec.name]
            raw = span_words_for(spec, span, n_target)
            material = window_shuffle(raw, args.k, rng)
        elif spec.generator == POS_SUBSTITUTE:
            span = spans[spec.name]
            pool_path = outdir / POS_POOL_NAME
            template, pools, pool_meta = load_pos_pool(pool_path, span)
            params["pos_pool_sha256"] = sha256_file(pool_path)
            params["tagger"] = pool_meta["tagger"]
            material = pos_substitute(template, pools, rng)
        elif spec.generator == RANDOM_ASCII:
            span = None
            n_chars = int(n_target * spec.params["oversample_chars"])
            material = random_ascii_text(n_chars, rng)
        else:
            raise MakeBurstsError(f"unknown generator {spec.generator!r}")

        text, cut = trim_to_tokens(tokenizer, material, n_target,
                                   label=spec.name)
        write_text(paths[spec.name], text)
        results[spec.name] = (span, cut, params)

    for spec in span_arms():
        print_span(spec.name, spans[spec.name])

    # --- assert, do not assume ---------------------------------------------
    counts = {
        spec.name: token_count(
            tokenizer, paths[spec.name].read_text(encoding="utf-8"))
        for spec in ARM_SPECS
    }
    wrong = {n: c for n, c in counts.items() if c != n_target}
    if wrong:
        raise MakeBurstsError(
            f"token counts do not all equal N = {n_target} after writing: "
            f"{counts}. Mismatched: {sorted(wrong)}."
        )
    actual_context = token_count(
        tokenizer, context_path.read_text(encoding="utf-8"))
    if actual_context != context_tokens:
        raise MakeBurstsError(
            f"context is {actual_context} tokens, expected {context_tokens}"
        )

    # --- provenance ---------------------------------------------------------
    arms_record = {}
    for spec in ARM_SPECS:
        entry = {
            "file": spec.filename,
            "tokens": counts[spec.name],
            "sha256": sha256_file(paths[spec.name]),
            "generator": spec.generator,
        }
        if spec.is_generated:
            span, cut, params = results[spec.name]
            entry["derived_seed"] = params.pop("derived_seed")
            entry["params"] = params
            entry["trim_cut_mid_word"] = cut
            entry["source"] = None if span is None else {
                "doc_index": span.doc_index,
                "word_start": span.word_start,
                "word_count": span.word_count,
            }
        else:
            entry["derived_seed"] = None
            entry["params"] = {}
            entry["source"] = "hand-written, fixed, not generated"
        arms_record[spec.name] = entry

    provenance = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "spec": "v4",
        "target_tokens": n_target,
        "sequence_tokens": seq.value,
        "tokenizer": "gpt2",
        "run_seed": args.seed,
        "seed_derivation":
            "int.from_bytes(sha256('burst-study/v4/<run_seed>/<purpose>')"
            ".digest()[:8], 'big')",
        "dataset": {"name": DATASET, "split": SPLIT,
                    "documents_cached": len(docs)},
        "context": {
            "file": CONTEXT_NAME,
            "tokens": actual_context,
            "sha256": sha256_file(context_path),
            "filler_tokens_used_by_arms": filler_tokens,
            "selection": "pinned and human-reviewed, not drawn blind",
            "doc_index": CONTEXT_DOC_INDEX,
            "word_start": CONTEXT_WORD_START,
            "word_count": CONTEXT_WORD_COUNT,
            "trim_cut_mid_word": context_cut,
        },
        "substrate": {
            "file": "ordinary.txt",
            "note": ("v3 arm, retained as source substrate only. Not an arm "
                     "in spec v4 and not regenerated by this script."),
        },
        "arms": arms_record,
    }
    write_text(outdir / PROVENANCE_NAME,
               json.dumps(provenance, indent=2, sort_keys=False) + "\n")

    # --- report -------------------------------------------------------------
    print()
    print(RULE)
    print("written")
    print(RULE)
    for spec in ARM_SPECS:
        mark = "  (not written -- hand-written)" if not spec.is_generated else ""
        print(f"  {spec.filename:<22} {counts[spec.name]:>5} tokens{mark}")
    print(f"  {CONTEXT_NAME:<22} {actual_context:>5} tokens")
    for spec in ARM_SPECS:
        if spec.is_generated and results[spec.name][1]:
            print(f"  NOTE: trimming {spec.name} cut mid-word. Acceptable -- a "
                  "burst is a token")
            print("        sequence, not a sentence -- recorded in "
                  "provenance.json.")
    print(f"  {PROVENANCE_NAME} written to {outdir}")
    print()
    print("These are CANDIDATES. Run scripts/match_arms.py to measure them in "
          "context.")
    return 0


def load_pos_pool(pool_path: Path, span: Span) -> tuple[list, dict, dict]:
    """Read the committed POS pool and check it was built for THIS span.

    Requirement H7: no tagger runs at generation time. The pool carries the
    identity of the span its template was tagged from, and if span selection
    has since moved, this raises rather than substituting words onto a
    template that describes different text.
    """
    if not pool_path.is_file():
        raise MakeBurstsError(
            f"no POS pool at {pool_path}. Build it once with:\n"
            "    python scripts/build_pos_pool.py\n"
            "It is committed, so this is only needed when the pool or the "
            "span selection changes."
        )
    data = json.loads(pool_path.read_text(encoding="utf-8"))
    built = data.get("built_for", {})
    if (built.get("doc_index") != span.doc_index
            or built.get("word_start") != span.word_start):
        raise MakeBurstsError(
            f"{pool_path} was built for document {built.get('doc_index')} "
            f"word {built.get('word_start')}, but span selection now gives "
            f"document {span.doc_index} word {span.word_start}. The template "
            "describes different text. Re-run scripts/build_pos_pool.py."
        )
    return data["template"], data["tag_pools"], data


if __name__ == "__main__":
    raise SystemExit(main())
