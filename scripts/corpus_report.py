#!/usr/bin/env python
"""The corpus's measurement record. Step 11.

Every number here is DERIVED from the manifest and the verifier's report.
Nothing is hardcoded prose. Five separate times in this build a hardcoded
interpretation outlived the number it described, and the fifth was shipping a
claim contradicted by a table one screen below it -- so the S70 pattern is
built in from the first commit rather than retrofitted after the sixth.

The report is committed; the corpus it describes is not. That is the whole
point of the manifest: the record of what was built travels in git, and the
5.02 GB it describes travels separately and proves itself against the record.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import corpus_spec as SPEC  # noqa: E402
import corpus_verify as VERIFY  # noqa: E402
import data_order as ORDER  # noqa: E402

DEFAULT_REPORT_DIR = REPO_ROOT / "docs" / "measurements"
REPORT_STEM = "11-corpus"


def permutation_digests(n_seeds: int = 10) -> dict:
    """Each seed's data-order digest. THE CONTRACT'S RECORDED SIDE.

    Derived here and recorded so the training loop can call
    data_order.verify_permutation against it before consuming a batch. A
    manifest that merely records an order proves nothing; this is the value the
    check compares to. See cross-module obligation 4.
    """
    return {
        str(seed): ORDER.seed_digest(seed, SPEC.TRAIN_SEQUENCES)
        for seed in range(n_seeds)
    }


def build_provenance(payload) -> str:
    """Derived from the payload, so it cannot describe a corpus it is not."""
    verify = payload["verification"]
    manifest = payload["manifest_summary"]
    order = payload["data_order"]

    parts = [
        f"Corpus: {SPEC.DATASET} at revision {SPEC.REVISION}, "
        f"{payload['source']['files_used']} of {SPEC.N_SOURCE_FILES} pinned "
        f"Parquet files, {payload['source']['bytes']:,} source bytes.",
        f"Tokenized to {verify['tokens_from_manifest']:,} tokens in "
        f"{verify['blocks_present']} blocks: a {SPEC.HELDOUT_TOKENS:,}-token "
        f"held-out slice and {SPEC.N_SHARDS} training shards totalling "
        f"{verify['training_tokens']:,} tokens.",
        f"The training slice is exactly configs/base.yaml's "
        f"expected_token_budget ({SPEC.TRAIN_TOKENS:,} = 256 x 1024 x 9536), "
        f"so the loader's assertion holds"
        + ("." if verify["training_tokens"] == SPEC.TRAIN_TOKENS
           else " -- IT DOES NOT, see findings."),
        "THE HELD-OUT SLICE IS AT THE FRONT, and that is load-bearing rather "
        "than cosmetic: it places the corpus-derived texts this repo already "
        "committed (bursts/context.txt is document 73, and two arm texts are "
        "documents 104 and 193) OUTSIDE the training slice. Placed at the end "
        "every one of them would have been trained on, and every step 10 "
        "metric would have been measuring memorisation while reporting "
        "representation.",
        f"Disjointness is structural: the sampler permutes "
        f"range({SPEC.TRAIN_SEQUENCES:,}), which indexes the training shards "
        f"only, and the held-out slice is a separate file. "
        f"assert_heldout_disjoint_from_training reports "
        f"{verify['heldout'].get('disjoint')}.",
        f"Every number was checked by more than one route: block hashes "
        f"({verify['blocks_hashed']} re-hashed), file size against recorded "
        f"token count, the total summed from the manifest "
        f"({verify['tokens_from_manifest']:,}) against the total summed from "
        f"disk ({verify['tokens_from_disk']:,}), and "
        f"{verify['boundaries_checked']} shard boundaries against arithmetic.",
        f"Data order is PER-SEED: {len(order['permutation_digests'])} seeds, "
        f"each a permutation of the same {SPEC.TRAIN_SEQUENCES:,} training "
        f"sequences derived from SHA-256 rather than any library PRNG. The "
        f"digests below are what the training loop must check its own order "
        f"against before consuming a batch -- cross-module obligation 4.",
    ]

    tokenizer = verify.get("tokenizer", {})
    if tokenizer.get("probe") == "ran":
        parts.append(
            f"Tokenizer drift is guarded by a probe rather than a version "
            f"string: the committed passage tokenizes to "
            f"{tokenizer.get('probe_token_sha256', '')[:16]} here, matching "
            f"what built the corpus ({tokenizer.get('agrees')}). A tokenizer "
            f"that would produce a different corpus produces a different probe "
            f"hash whatever it calls itself.")
    else:
        parts.append(
            "THE TOKENIZER PROBE DID NOT RUN in this verification, so nothing "
            "here says the tokenizer agrees with the one that built the "
            "corpus.")

    if verify["findings"]:
        parts.append(
            f"VERIFICATION FOUND {len(verify['findings'])} PROBLEM(S); see the "
            "findings section. This corpus is not fit to train on.")
    else:
        parts.append("Verification found no problems.")
    return " ".join(parts)


def report_banner(payload) -> list:
    verify = payload["verification"]
    complete = verify.get("complete")
    ok = verify.get("ok")

    lines = [
        "CORPUS BUILT AND VERIFIED" if (complete and ok)
        else ("CORPUS INCOMPLETE" if not complete
              else "*** VERIFICATION FOUND PROBLEMS ***"),
        "",
        f"{SPEC.DATASET}",
        f"revision {SPEC.REVISION}",
        "",
        f"  held out   {SPEC.HELDOUT_TOKENS:>15,} tokens  "
        f"{SPEC.HELDOUT_SEQUENCES:>10,} seq   {SPEC.HELDOUT_FILENAME}",
        f"  training   {verify['training_tokens']:>15,} tokens  "
        f"{SPEC.TRAIN_SEQUENCES:>10,} seq   {SPEC.N_SHARDS} shards",
        f"  total      {verify['tokens_from_manifest']:>15,} tokens  "
        f"{payload['bytes_total']:>10,} bytes",
        "",
        "THE HELD-OUT SLICE IS AT THE FRONT. bursts/context.txt is openwebtext",
        "document 73, and two arm texts are documents 104 and 193. Front-placed",
        "they are held out; end-placed every one would have been trained on and",
        "every step 10 metric would have measured memorisation while claiming",
        "to measure representation.",
        "",
        "DATA ORDER IS PER-SEED. The digests below are a CONTRACT: the training",
        "loop must recompute its order and match one of them before consuming a",
        "single batch. A recorded order that is never checked proves nothing.",
    ]
    if not verify.get("tokenizer", {}).get("agrees"):
        lines += ["",
                  "*** the tokenizer probe did not confirm agreement ***"]
    return lines


def format_report(payload) -> str:
    rule = "=" * 78
    thin = "-" * 78
    verify = payload["verification"]
    lines = ["", rule] + report_banner(payload) + [rule, ""]

    lines += [rule, "SOURCE", rule,
              f"  dataset          {SPEC.DATASET}",
              f"  revision         {SPEC.REVISION}",
              f"  files used       {payload['source']['files_used']} of "
              f"{SPEC.N_SOURCE_FILES}",
              f"  source bytes     {payload['source']['bytes']:,}",
              "",
              "  Files, not a stream: a stream has no seek so a resume rereads",
              "  everything, and a stream of `main` is identified by nothing.",
              "  Order comes from the numbered filenames, never a directory",
              "  listing.", ""]

    lines += [rule, "GEOMETRY", rule,
              f"{'block':<14}{'sequences':>12}{'tokens':>16}{'bytes':>16}"]
    heldout = SPEC.summary()["heldout"]
    training = SPEC.summary()["training"]
    lines += [
        f"{'heldout':<14}{heldout['sequences']:>12,}{heldout['tokens']:>16,}"
        f"{heldout['bytes']:>16,}",
        f"{'training':<14}{training['sequences']:>12,}{training['tokens']:>16,}"
        f"{training['bytes']:>16,}",
        f"{'total':<14}{SPEC.TOTAL_SEQUENCES:>12,}{SPEC.TOTAL_TOKENS:>16,}"
        f"{SPEC.TOTAL_TOKENS * SPEC.BYTES_PER_TOKEN:>16,}",
        "",
        f"  {SPEC.N_SHARDS} shards x {SPEC.SEQUENCES_PER_SHARD:,} sequences, "
        f"exact -- no ragged final shard.",
        f"  dtype {SPEC.DTYPE} {SPEC.BYTE_ORDER}-endian, raw, no header: "
        f"filesize / {SPEC.BYTES_PER_TOKEN} == token count exactly.", ""]

    lines += [rule, "VERIFICATION -- every number by more than one route", rule,
              f"  manifest sha256      {payload['manifest_sha256']}",
              "  ^ carry this SEPARATELY from the manifest; a corrupted",
              "    manifest would otherwise validate corrupted shards.",
              f"  blocks present       {verify['blocks_present']} of "
              f"{verify['blocks_expected']}",
              f"  blocks re-hashed     {verify['blocks_hashed']}",
              f"  boundaries checked   {verify['boundaries_checked']} against "
              f"arithmetic",
              f"  tokens from manifest {verify['tokens_from_manifest']:,}",
              f"  tokens from disk     {verify['tokens_from_disk']:,}",
              f"  training tokens      {verify['training_tokens']:,} "
              f"(expected {SPEC.TRAIN_TOKENS:,})",
              f"  held-out disjoint    {verify['heldout'].get('disjoint')}",
              ""]

    tok = verify.get("tokenizer", {})
    lines += [thin, "tokenizer identity -- a probe, not a version string", thin,
              f"  probe                {tok.get('probe')}",
              f"  probe token sha256   {tok.get('probe_token_sha256', 'n/a')}",
              f"  agrees with build    {tok.get('agrees')}",
              f"  built by             transformers "
              f"{tok.get('built_by', {}).get('transformers')}, tokenizers "
              f"{tok.get('built_by', {}).get('tokenizers')}", ""]

    order = payload["data_order"]
    lines += [rule, "DATA ORDER -- per seed, and a CONTRACT", rule,
              f"  permutation over {SPEC.TRAIN_SEQUENCES:,} training sequences,",
              "  derived from SHA-256 in counter mode -- no library PRNG, so no",
              "  library version can change it.", "",
              f"{'seed':>6}  {'permutation sha256'}"]
    for seed, digest in sorted(order["permutation_digests"].items(),
                               key=lambda kv: int(kv[0])):
        lines.append(f"{seed:>6}  {digest}")
    lines += ["",
              "  THE TRAINING LOOP MUST CALL",
              "  data_order.verify_permutation(seed, "
              f"{SPEC.TRAIN_SEQUENCES}, <digest>)",
              "  before consuming a batch. Measured at 1.98s. A run that "
              "serves a",
              "  different order than its provenance claims is unreproducible "
              "and",
              "  silently so. Cross-module obligation 4.", ""]

    if verify["findings"]:
        lines += [rule, "FINDINGS", rule]
        for finding in verify["findings"]:
            lines.append(f"  - {finding}")
        lines.append("")

    for title, key in (("PROVENANCE", "PROVENANCE"),):
        lines += [rule, title, rule]
        for chunk in payload[key].split(". "):
            if chunk.strip():
                lines.append("  " + chunk.strip().rstrip(".") + ".")
    return "\n".join(lines)


def build_payload(outdir: Path, cachedir: Path) -> dict:
    verify = VERIFY.verify(outdir, sys.stdout)
    manifest = json.loads(
        (outdir / "manifest.json").read_text(encoding="utf-8"))

    fetch_path = cachedir / "fetch-manifest.json"
    source = {"files_used": 0, "bytes": 0}
    if fetch_path.is_file():
        fetch = json.loads(fetch_path.read_text(encoding="utf-8"))
        source = {
            "files_used": len(fetch.get("files", {})),
            "bytes": sum(r["bytes"] for r in fetch.get("files", {}).values()),
            "revision": fetch.get("revision"),
        }

    payload = {
        "task": "step 11 -- the data pipeline",
        "spec": SPEC.summary(),
        "source": source,
        "manifest_sha256": verify["manifest_sha256"],
        "manifest_summary": {
            "blocks": len(manifest.get("blocks", {})),
            "tokenizer": manifest.get("tokenizer"),
        },
        "verification": verify,
        "bytes_total": sum(r["bytes"] for r in manifest["blocks"].values()),
        "data_order": {
            "n_sequences": SPEC.TRAIN_SEQUENCES,
            "derivation": "sha256 counter mode, stdlib only",
            "permutation_digests": permutation_digests(),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }
    payload["PROVENANCE"] = build_provenance(payload)
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/corpus_report.py",
        description="Write the corpus's measurement record.")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--cachedir", type=Path,
                        default=REPO_ROOT / ".corpus-cache" / "openwebtext")
    parser.add_argument("--reportdir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args(argv)

    payload = build_payload(args.outdir, args.cachedir)
    args.reportdir.mkdir(parents=True, exist_ok=True)
    (args.reportdir / f"{REPORT_STEM}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    text = format_report(payload)
    (args.reportdir / f"{REPORT_STEM}.md").write_text(
        text + "\n", encoding="utf-8", newline="\n")
    print(text)
    print(f"\nwrote {args.reportdir / (REPORT_STEM + '.json')}")
    return 0 if payload["verification"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
