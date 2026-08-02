#!/usr/bin/env python
"""Build the committed POS pool for the pos-substituted arm. Run once.

    python scripts/build_pos_pool.py

WHY THIS IS A SEPARATE SCRIPT (requirement H7)

The pos-substituted arm replaces every word with a random word of the same
part of speech. Doing that needs two things: a per-tag vocabulary to draw
from, and the tag of each word in the source span. Both need a POS tagger.

A tagger must NOT be a runtime dependency of scripts/make_bursts.py. So the
tagger runs exactly once -- here, on one machine -- and what gets committed is
its output: bursts/pos_pool.json, carrying both the per-tag word pools and the
tag template for the span. make_bursts.py reads that file and never imports
nltk.

The consequence, stated plainly: the tagger's model download
(averaged_perceptron_tagger_eng) cannot be pinned the way a pip package can.
It does not need to be. Reproducibility rests on the committed pool and its
SHA-256, which is recorded in bursts/provenance.json. If the tagger changed
tomorrow, the committed pool would not move. A from-scratch rebuild on another
machine is NOT guaranteed to reproduce byte-identically, and is not required
to be.

    pip install -e ".[pos]"      # nltk, build-time only

The one-time model download goes to .corpus-cache/nltk_data/, which is
gitignored. This script performs it automatically if it is missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from burst_match import DEFAULT_BASE_CONFIG, load_tokenizer, resolve_seq_len  # noqa: E402
from make_bursts import (  # noqa: E402
    DEFAULT_CACHE,
    DEFAULT_DOCS,
    DEFAULT_OUTDIR,
    POS_POOL_NAME,
    REFERENCE_ARM,
    RULE,
    MakeBurstsError,
    arm_by_name,
    arm_spans,
    capitalisation_of,
    load_corpus_slice,
    span_words_for,
    split_affixes,
    token_count,
    write_text,
)

TAGGER_PACKAGE = "averaged_perceptron_tagger_eng"
NLTK_DATA_DIR = REPO_ROOT / ".corpus-cache" / "nltk_data"

#: A word must appear at least this many times in the tagged corpus to enter
#: a pool. Filters scrape junk and one-off typos without being so strict that
#: the pools collapse to the few hundred commonest words.
MIN_WORD_FREQUENCY = 2

#: Documents tagged to build the vocabulary. The whole cache, because a
#: smaller slice fails: at 60 documents no foreign word (tag FW) cleared the
#: frequency threshold, and the template needs one, so the build refused --
#: correctly, rather than substituting something wrong.
#:
#: Small pools are not always a defect. TO has exactly one member because
#: English has one infinitival "to"; CC and MD are closed classes with about a
#: dozen members each. Those are correct answers. Only an open class (NN, VB,
#: JJ) with a small pool would indicate the corpus slice is too thin.
DEFAULT_POOL_DOCS = 200

#: Pool words must be plain lowercase-able alphabetic ASCII. Anything with a
#: digit, an apostrophe or a non-ASCII character is rejected: those tokenize
#: unpredictably and this arm is supposed to vary lexical content, not
#: character weirdness.
def _is_poolable(word: str) -> bool:
    return word.isascii() and word.isalpha() and len(word) > 1


def _load_tagger():
    """Import nltk and make sure the tagger model is present."""
    try:
        import nltk
    except ImportError as exc:
        raise MakeBurstsError(
            "nltk is not installed, and it is needed to BUILD the POS pool "
            "(never to use it).\n    pip install -e \".[pos]\"\n"
            f"(underlying import error: {exc})"
        ) from exc

    NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if str(NLTK_DATA_DIR) not in nltk.data.path:
        nltk.data.path.insert(0, str(NLTK_DATA_DIR))
    try:
        nltk.data.find(f"taggers/{TAGGER_PACKAGE}")
    except LookupError:
        print(f"tagger:  downloading {TAGGER_PACKAGE} to {NLTK_DATA_DIR}")
        if not nltk.download(TAGGER_PACKAGE, download_dir=str(NLTK_DATA_DIR),
                             quiet=True):
            raise MakeBurstsError(
                f"could not download {TAGGER_PACKAGE}. This is a one-time "
                "download and needs a network. The pool it produces is "
                "committed, so this is only ever needed once."
            )
    from nltk import pos_tag
    return nltk, pos_tag


def _tag_words(pos_tag, words: list[str]) -> list[tuple[str, str]]:
    """Tag whitespace-split words, tagging the punctuation-stripped core.

    Tagging the core rather than the raw word matters: "evening." tags as a
    noun, but "evening" tags more reliably, and the punctuation has to be kept
    separate anyway so it can be reattached to the substitute.
    """
    cores = [split_affixes(w)[1] or w for w in words]
    return pos_tag(cores)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/build_pos_pool.py",
        description="Build bursts/pos_pool.json. Run once; the output is "
                    "committed and read by make_bursts.py.")
    parser.add_argument("--seed", type=int, default=0, metavar="N",
                        help="run seed, must match make_bursts (default: 0)")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--docs", type=int, default=DEFAULT_DOCS)
    parser.add_argument("--pool-docs", type=int, default=DEFAULT_POOL_DOCS,
                        help=f"documents to tag for the vocabulary "
                             f"(default: {DEFAULT_POOL_DOCS})")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    args = parser.parse_args(argv)

    print(RULE)
    print("build_pos_pool -- one-time tagger run, output is committed")
    print(RULE)

    try:
        return _run(args)
    except MakeBurstsError as exc:
        print(f"\nERROR\n{exc}\n", file=sys.stderr)
        return 1


def _run(args) -> int:
    nltk, pos_tag = _load_tagger()
    tokenizer = load_tokenizer()

    outdir = Path(args.outdir)
    reference = arm_by_name(REFERENCE_ARM)
    n_target = token_count(
        tokenizer, (outdir / reference.filename).read_text(encoding="utf-8"))
    print(f"target:  N = {n_target} tokens")

    docs = load_corpus_slice(Path(args.cache), args.docs)

    # The SAME selection call make_bursts makes, so the template is provably
    # built from the span the generator will use.
    spans = arm_spans(docs, args.seed, n_target)
    spec = arm_by_name("pos-substituted")
    span = spans[spec.name]
    print(f"span:    document {span.doc_index}, word {span.word_start}, "
          f"{span.word_count} words")

    # --- the template: one entry per word of this arm's span ---------------
    words = span_words_for(spec, span, n_target).split()
    tagged = _tag_words(pos_tag, words)
    template = []
    for word, (core_tagged, tag) in zip(words, tagged):
        lead, core, trail = split_affixes(word)
        if not core:
            template.append({"kind": "literal", "text": word})
            continue
        template.append({
            "kind": "word",
            "tag": tag,
            "lead": lead,
            "trail": trail,
            "cap": capitalisation_of(core),
        })
    needed_tags = {e["tag"] for e in template if e["kind"] == "word"}
    print(f"template: {len(template)} entries, {len(needed_tags)} distinct tags")

    # --- the pools: vocabulary per tag, from a wider slice ------------------
    counts_by_tag: dict[str, Counter] = defaultdict(Counter)
    pool_docs = docs[:args.pool_docs]
    for index, doc in enumerate(pool_docs):
        doc_words = doc.split()
        if not doc_words:
            continue
        for (core, tag) in _tag_words(pos_tag, doc_words):
            if _is_poolable(core):
                counts_by_tag[tag][core.lower()] += 1
        if (index + 1) % 20 == 0:
            print(f"         tagged {index + 1}/{len(pool_docs)} documents")

    tag_pools = {
        tag: sorted(w for w, c in counter.items() if c >= MIN_WORD_FREQUENCY)
        for tag, counter in counts_by_tag.items()
    }
    tag_pools = {tag: words for tag, words in tag_pools.items() if words}

    missing = sorted(needed_tags - set(tag_pools))
    if missing:
        raise MakeBurstsError(
            f"the template needs tags {missing} but the corpus produced no "
            f"poolable words for them. Raise --pool-docs or lower "
            f"MIN_WORD_FREQUENCY."
        )
    # Only ship pools the template can actually use. A pool for a tag nothing
    # asks for is committed weight with no purpose.
    tag_pools = {tag: tag_pools[tag] for tag in sorted(needed_tags)}

    payload = {
        "tagger": f"nltk {nltk.__version__} / {TAGGER_PACKAGE}",
        "min_word_frequency": MIN_WORD_FREQUENCY,
        "pool_documents": len(pool_docs),
        "built_for": {
            "arm": spec.name,
            "run_seed": args.seed,
            "doc_index": span.doc_index,
            "word_start": span.word_start,
            "word_count": span.word_count,
        },
        "tag_pools": tag_pools,
        "template": template,
    }
    pool_path = outdir / POS_POOL_NAME
    write_text(pool_path, json.dumps(payload, indent=2, sort_keys=False) + "\n")

    total = sum(len(v) for v in tag_pools.values())
    print()
    print(f"wrote {pool_path}")
    print(f"  {len(tag_pools)} tags, {total} words total")
    print(f"  smallest pool: " + min(
        (f"{t} ({len(w)})" for t, w in tag_pools.items()),
        key=lambda s: int(s.split("(")[1].rstrip(")"))))
    print()
    print("Commit this file. make_bursts.py reads it and never imports nltk.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
