#!/usr/bin/env python
"""Generate the two corpus-derived burst passages, matched in token length.

The study needs three passages that differ in what they are but not in how
long they are:

    bursts/coherent.txt   hand-written, fixed, NEVER touched by this script.
                          Its token count is the target length N.
    bursts/ordinary.txt   a contiguous span of real OpenWebText, trimmed to N.
    bursts/noise.txt      a different contiguous span, word-order shuffled
                          within non-overlapping windows of size k, trimmed
                          to N.

    python scripts/make_bursts.py --k 5
    python scripts/make_bursts.py --k 5 --seed 3 --outdir /tmp/try

These are CANDIDATES, not final texts. Whether coherent and noise can be
matched on the size of the shove they deliver is what scripts/match_sweep.py
reports; which k to use, and what tolerance counts as matched, are decisions
this script does not make and must not appear to make.

WHAT IS GUARANTEED
- All three files come out at exactly N tokens under the GPT-2 tokenizer,
  asserted before the script exits. Shuffling changes tokenization, so length
  is never assumed to be preserved -- it is sampled generously and trimmed.
- Same --seed and --k gives byte-identical output, including provenance.json.
  Nothing timestamped or machine-dependent is written into it.
- coherent.txt is opened read-only. There is an explicit guard, because a
  script that regenerated the hand-written passage would destroy the one fixed
  point the other two are measured against.

WHERE THE TEXT COMES FROM
Streamed from Skylion007/openwebtext on HuggingFace and cached under a
gitignored path, so the first run needs a network and later runs do not. The
corpus is real OpenWebText because the ordinary arm has to be indistinguishable
from the training distribution -- writing "ordinary-sounding" text by hand would
make it a third kind of unusual.
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

from burst_match import BurstMatchError, load_tokenizer  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET = "Skylion007/openwebtext"
SPLIT = "train"

DEFAULT_CACHE = REPO_ROOT / ".corpus-cache" / "openwebtext_slice.jsonl"
DEFAULT_OUTDIR = REPO_ROOT / "bursts"
COHERENT_NAME = "coherent.txt"
ORDINARY_NAME = "ordinary.txt"
NOISE_NAME = "noise.txt"
PROVENANCE_NAME = "provenance.json"

#: Documents pulled into the cached slice. Small on purpose -- this needs a
#: pool to choose two clean spans from, not a corpus.
DEFAULT_DOCS = 200

#: Words taken per span, as a multiple of N. English runs under one token per
#: word, so 2N words is comfortably more than N tokens even after shuffling
#: rearranges which byte-pair merges apply. The excess is trimmed away; the
#: point is to never come up short.
SPAN_WORD_MULTIPLIER = 2.0

# --- span quality filters --------------------------------------------------
#
# OpenWebText is scraped, so a randomly chosen span may be a navigation bar, a
# table of numbers, a copyright footer or a page in another script. None of
# those are "ordinary text" in the sense this arm needs.
#
# Two of these thresholds are STRICTER than the brief, which said "mostly"
# non-ASCII and "mostly" punctuation or digits -- literally >50%. A span that
# is 45% CJK or 40% digits is still not ordinary English prose, and there are
# plenty of clean spans in the pool, so the conservative reading costs nothing.
# Logged in implementation-notes.md as S22.

#: Reject above this fraction of non-ASCII characters.
MAX_NON_ASCII_FRACTION = 0.10
#: Reject above this fraction of punctuation-or-digit characters.
MAX_PUNCT_DIGIT_FRACTION = 0.25
#: Reject below this fraction of alphabetic characters. From the brief.
MIN_ALPHA_FRACTION = 0.40
#: Reject below this many sentence-ending marks. From the brief.
MIN_SENTENCE_ENDS = 3
#: The marks that count as ending a sentence.
SENTENCE_ENDS = ".!?"

RULE = "=" * 78
THIN = "-" * 78


class MakeBurstsError(Exception):
    """Every failure this script raises on purpose."""


# ---------------------------------------------------------------------------
# Pure text operations
#
# No tokenizer, no network. Tested without either.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpanStats:
    """Why a span was accepted or rejected, in numbers."""

    n_chars: int
    non_ascii_fraction: float
    punct_digit_fraction: float
    alpha_fraction: float
    sentence_ends: int

    @property
    def rejections(self) -> tuple[str, ...]:
        """Every filter this span fails, named. Empty means acceptable."""
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

    Fractions are over non-whitespace characters. Whitespace is excluded
    because it says nothing about whether this is prose, and including it
    would make every fraction a function of how the page happened to be
    wrapped.
    """
    body = [c for c in text if not c.isspace()]
    n = len(body)
    if n == 0:
        return SpanStats(0, 0.0, 0.0, 0.0, 0)
    non_ascii = sum(1 for c in body if ord(c) > 127)
    alpha = sum(1 for c in body if c.isalpha())
    digits = sum(1 for c in body if c.isdigit())
    # Punctuation here means "not a letter, not a digit" -- brackets, slashes,
    # pipes and bullets are exactly what a scraped navigation bar is made of.
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

    Words are whitespace-separated. Windows are taken left to right and
    shuffled in that order, so the result is a function of the rng's seed and
    of k alone. The FINAL PARTIAL WINDOW is shuffled too -- leaving it in
    original order would put a run of untouched text at the end of every noise
    passage, which is a systematic difference from the rest of it.

    The original whitespace is not preserved: words come back joined by single
    spaces. Shuffling word order across a line break has already destroyed
    whatever the line break meant, so there is nothing coherent to preserve.
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


def cut_is_mid_word(full: str, trimmed: str) -> bool:
    """True if trimming `full` down to `trimmed` split a word in half.

    Acceptable when it happens -- a burst is a token sequence and the model
    does not care that the last one is a word fragment -- but it is logged
    rather than hidden, because a passage ending in "collabora" is a thing you
    should find out from the script and not from reading the file later.
    """
    if not trimmed or len(trimmed) >= len(full):
        return False
    return not full[len(trimmed)].isspace() and not trimmed[-1].isspace()


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def trim_to_tokens(tokenizer, text: str, n_tokens: int) -> tuple[str, bool]:
    """Cut `text` down to exactly `n_tokens` tokens. Returns (text, mid_word).

    Decoding a prefix of the token ids and re-encoding it is not guaranteed to
    give the same count back -- byte-pair merges can behave differently at a
    boundary -- so the result is re-encoded and corrected rather than trusted.
    The loop is bounded; it has never needed more than one pass in practice,
    and an unbounded one would be a hang waiting to happen.
    """
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) < n_tokens:
        raise MakeBurstsError(
            f"cannot trim to {n_tokens} tokens: the span is only "
            f"{len(ids)} tokens long. Raise --docs or the span multiplier."
        )

    take = n_tokens
    for _ in range(16):
        candidate = tokenizer.decode(ids[:take])
        actual = token_count(tokenizer, candidate)
        if actual == n_tokens:
            return candidate, cut_is_mid_word(text, candidate)
        # Re-encoding disagreed with the slice. Step the cut in the direction
        # that closes the gap and try again.
        take += n_tokens - actual
        if not 0 < take <= len(ids):
            break
    raise MakeBurstsError(
        f"could not trim to exactly {n_tokens} tokens; the tokenizer does not "
        "round-trip this text. Try a different --seed to select another span."
    )


# ---------------------------------------------------------------------------
# The corpus slice
# ---------------------------------------------------------------------------


def load_corpus_slice(cache_path: Path, n_docs: int, stream=None) -> list[str]:
    """The cached documents, downloading them once if they are not there.

    Cached as JSONL under a gitignored path. The cache is what makes runs after
    the first one both fast and offline, and it is what makes the whole thing
    reproducible without pinning a dataset revision: the bytes the spans were
    cut from are sitting on disk.
    """
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
            "streamed.\n"
            "    pip install -e \".[measure]\"\n"
            f"(underlying import error: {exc})"
        ) from exc

    docs: list[str] = []
    try:
        dataset = load_dataset(DATASET, split=SPLIT, streaming=True)
        iterator = iter(dataset)
        for _ in range(n_docs):
            docs.append(next(iterator)["text"])
        # Closed explicitly. Letting the generator be collected at interpreter
        # shutdown produces a noisy traceback from inside pyarrow.
        iterator.close()
    except StopIteration:
        pass
    except Exception as exc:
        raise MakeBurstsError(
            f"no cached corpus at {cache_path}, and streaming {DATASET} from "
            "HuggingFace failed.\n"
            "This script needs the corpus once; after that the cache above "
            "serves every later run offline. Either connect to a network and "
            "run it again, or copy a populated cache file into place from a "
            "machine that has one.\n"
            f"the download said: {exc}"
        ) from exc

    if not docs:
        raise MakeBurstsError(
            f"{DATASET} yielded no documents; nothing to cut spans from."
        )

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
    """A contiguous run of words from one cached document."""

    doc_index: int
    word_start: int
    word_count: int
    text: str
    stats: SpanStats


def select_spans(docs: list[str], n_spans: int, span_words: int,
                 rng: random.Random, stream=None) -> list[Span]:
    """Pick `n_spans` acceptable spans, each from a different document.

    Candidates are enumerated in a fixed order and then shuffled with the
    seeded rng, so selection is reproducible but not biased towards the front
    of the corpus. Requiring a different document per span is stricter than
    "a different span" -- two windows of the same article share its subject,
    its register and often its boilerplate, and the ordinary and noise arms
    are supposed to differ in structure, not in topic.
    """
    stream = sys.stdout if stream is None else stream

    candidates: list[tuple[int, int]] = []
    doc_words: list[list[str]] = []
    for doc_index, text in enumerate(docs):
        words = text.split()
        doc_words.append(words)
        # Step by the span length so candidates from one document do not
        # overlap each other.
        for start in range(0, max(1, len(words) - span_words + 1), span_words):
            if start + span_words <= len(words):
                candidates.append((doc_index, start))

    if not candidates:
        raise MakeBurstsError(
            f"no document in the cached slice holds {span_words} words; "
            "raise --docs."
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
        f"only {len(chosen)} of {n_spans} spans passed the quality filters "
        f"after examining {len(candidates)} candidates ({rejected} rejected). "
        "Raise --docs to widen the pool."
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_text(path: Path, text: str) -> None:
    """Write UTF-8 with LF endings and no trailing newline.

    newline="\\n" is not optional on Windows: without it Python translates
    every \\n to \\r\\n, the file stops being byte-identical to the same file
    written on the cluster, and .gitattributes' eol=lf would rewrite it on the
    next checkout anyway.

    No trailing newline, because a trailing newline is a token. coherent.txt
    does not have one and the other two have to tokenize to the same length.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def print_span(label: str, span: Span, stream=None) -> None:
    """Show a selected span in full, before anything is done to it."""
    stream = sys.stdout if stream is None else stream
    print(file=stream)
    print(THIN, file=stream)
    print(f"{label}: raw span, document {span.doc_index}, "
          f"words {span.word_start}..{span.word_start + span.word_count} "
          "(BEFORE shuffling or trimming)", file=stream)
    print(THIN, file=stream)
    print(span.text, file=stream)
    print(THIN, file=stream)
    s = span.stats
    print(f"  {s.n_chars} non-space chars | "
          f"alphabetic {s.alpha_fraction:.0%} | "
          f"punctuation+digits {s.punct_digit_fraction:.0%} | "
          f"non-ASCII {s.non_ascii_fraction:.0%} | "
          f"{s.sentence_ends} sentence ends", file=stream)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/make_bursts.py",
        description=(
            "Generate bursts/ordinary.txt and bursts/noise.txt at exactly the "
            "token length of bursts/coherent.txt. Does not modify "
            "coherent.txt."
        ),
    )
    parser.add_argument(
        "--k", type=int, required=True, metavar="N",
        help=("window size for the noise arm's word shuffle. Required: which "
              "k to use is a decision this script does not make"),
    )
    parser.add_argument("--seed", type=int, default=0, metavar="N",
                        help="seed for span selection and shuffling (default: 0)")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR,
                        metavar="PATH",
                        help=f"where the burst files go (default: {DEFAULT_OUTDIR})")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE,
                        metavar="PATH",
                        help=f"cached corpus slice (default: {DEFAULT_CACHE})")
    parser.add_argument("--docs", type=int, default=DEFAULT_DOCS, metavar="N",
                        help=f"documents to cache (default: {DEFAULT_DOCS})")
    parser.add_argument(
        "--coherent", type=Path, default=None, metavar="PATH",
        help="the fixed passage whose token count is the target (default: "
             f"<outdir>/{COHERENT_NAME}). Read only, never written",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    print(RULE)
    print("make_bursts -- two corpus-derived passages, matched to coherent.txt")
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
    coherent_path = (Path(args.coherent) if args.coherent
                     else outdir / COHERENT_NAME)
    ordinary_path = outdir / ORDINARY_NAME
    noise_path = outdir / NOISE_NAME

    # The guard that matters. coherent.txt is hand-written and fixed; a script
    # that overwrote it would silently redefine the target length that the
    # other two arms, and every measurement taken so far, are built on.
    for path in (ordinary_path, noise_path, outdir / PROVENANCE_NAME):
        if path.resolve() == coherent_path.resolve():
            raise MakeBurstsError(
                f"refusing to run: an output path ({path}) is the coherent "
                "passage, which this script must never write."
            )
    if not coherent_path.is_file():
        raise MakeBurstsError(
            f"the fixed passage {coherent_path} does not exist. It is "
            "hand-written and is not generated by this script; create it "
            "first, or point --coherent at it."
        )

    tokenizer = load_tokenizer()
    coherent_text = coherent_path.read_text(encoding="utf-8")
    n_target = token_count(tokenizer, coherent_text)
    print(f"target:  N = {n_target} tokens, from {coherent_path}")
    print(f"seed:    {args.seed} | k = {args.k}")

    span_words = int(n_target * SPAN_WORD_MULTIPLIER)
    docs = load_corpus_slice(Path(args.cache), args.docs)

    # One rng for the whole run, used in a fixed order: span selection first,
    # then the shuffle. Same seed and k, same bytes out.
    rng = random.Random(args.seed)
    ordinary_span, noise_span = select_spans(docs, 2, span_words, rng)

    print_span("ordinary", ordinary_span)
    print_span("noise   ", noise_span)

    shuffled = window_shuffle(noise_span.text, args.k, rng)

    ordinary_text, ordinary_cut = trim_to_tokens(
        tokenizer, ordinary_span.text, n_target)
    noise_text, noise_cut = trim_to_tokens(tokenizer, shuffled, n_target)

    write_text(ordinary_path, ordinary_text)
    write_text(noise_path, noise_text)

    # Assert, do not assume. Re-read from disk rather than trusting the string
    # in memory, so an encoding or newline mistake in write_text is caught here
    # rather than by a measurement three steps later.
    counts = {}
    for name, path in ((COHERENT_NAME, coherent_path),
                       (ORDINARY_NAME, ordinary_path),
                       (NOISE_NAME, noise_path)):
        counts[name] = token_count(
            tokenizer, path.read_text(encoding="utf-8"))
    wrong = {n: c for n, c in counts.items() if c != n_target}
    if wrong:
        raise MakeBurstsError(
            f"token counts do not all equal N = {n_target} after writing: "
            f"{counts}. The mismatched files are {sorted(wrong)}."
        )

    provenance = {
        "target_tokens": n_target,
        "seed": args.seed,
        "window_size_k": args.k,
        "tokenizer": "gpt2",
        "dataset": {"name": DATASET, "split": SPLIT,
                    "documents_cached": len(docs)},
        "span_words_requested": span_words,
        "filters": {
            "max_non_ascii_fraction": MAX_NON_ASCII_FRACTION,
            "max_punct_digit_fraction": MAX_PUNCT_DIGIT_FRACTION,
            "min_alpha_fraction": MIN_ALPHA_FRACTION,
            "min_sentence_ends": MIN_SENTENCE_ENDS,
        },
        "files": {
            COHERENT_NAME: {
                "tokens": counts[COHERENT_NAME],
                "sha256": sha256(coherent_path),
                "source": "hand-written, fixed, not generated",
            },
            ORDINARY_NAME: {
                "tokens": counts[ORDINARY_NAME],
                "sha256": sha256(ordinary_path),
                "source": "contiguous OpenWebText span, trimmed",
                "doc_index": ordinary_span.doc_index,
                "word_start": ordinary_span.word_start,
                "word_count": ordinary_span.word_count,
                "trim_cut_mid_word": ordinary_cut,
            },
            NOISE_NAME: {
                "tokens": counts[NOISE_NAME],
                "sha256": sha256(noise_path),
                "source": (f"contiguous OpenWebText span, word-shuffled in "
                           f"non-overlapping windows of k={args.k}, trimmed"),
                "doc_index": noise_span.doc_index,
                "word_start": noise_span.word_start,
                "word_count": noise_span.word_count,
                "trim_cut_mid_word": noise_cut,
            },
        },
    }
    # sort_keys=False keeps the order above, which reads top-down. No
    # timestamp and nothing machine-dependent: provenance.json is one of the
    # files that has to be byte-identical between two runs with the same
    # arguments.
    write_text(outdir / PROVENANCE_NAME,
               json.dumps(provenance, indent=2, sort_keys=False) + "\n")

    print()
    print(RULE)
    print("written")
    print(RULE)
    for name in (COHERENT_NAME, ORDINARY_NAME, NOISE_NAME):
        marker = "  (not written -- fixed)" if name == COHERENT_NAME else ""
        print(f"  {name:<16} {counts[name]:>5} tokens{marker}")
    for name, cut in ((ORDINARY_NAME, ordinary_cut), (NOISE_NAME, noise_cut)):
        if cut:
            print(f"  NOTE: trimming {name} cut mid-word. Acceptable -- a "
                  "burst is a token")
            print("        sequence, not a sentence -- but recorded in "
                  "provenance.json.")
    print(f"  {PROVENANCE_NAME} written to {outdir}")
    print()
    print("These are CANDIDATES. Run scripts/match_sweep.py to see whether "
          "coherent and")
    print("noise can be matched, and at which k. This script does not judge "
          "that.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
