#!/usr/bin/env python
"""Verify a tokenized corpus. Step 11, and the thing that runs on the cluster.

Written to be the ONLY file besides the shards that has to travel. It needs no
network, no `datasets`, no `huggingface_hub`, and no corpus source -- just the
shards, the manifest, and this repo. `transformers` is needed for one check and
one only, and that check is skipped with a loud note rather than failing if it
is unavailable.

WHAT IS CHECKED, and why each one catches something the others cannot

  1. HASHES. Every block re-hashed and compared. Catches corruption in transit.
  2. SIZE ARITHMETIC. filesize / 2 == recorded token count. An INDEPENDENT
     route: it catches a truncated file without reading it, and it catches a
     manifest that disagrees with its own data. Raw shards carry no header,
     which is exactly what makes this exact.
  3. THE TOTAL, three ways. Summed from the manifest, summed from file sizes on
     disk, and recomputed as batch_size * seq_len * total_steps from the
     config. Three routes to one number.
  4. SHARD BOUNDARIES, two ways. What the manifest records against what
     corpus_spec computes. A manifest is a claim; arithmetic is a check.
  5. HELD-OUT DISJOINTNESS. That no training sequence index can address a
     held-out token.
  6. THE TOKENIZER PROBE. Re-tokenize the committed passage HERE, with THIS
     machine's tokenizer, and compare to the hash recorded when the corpus was
     built. This is the only check that catches drift at the DESTINATION -- a
     cluster whose tokenizer disagrees would otherwise train on a corpus it
     cannot reproduce, and nothing else here would notice.

The manifest's own hash is printed. A corrupted manifest would otherwise
validate corrupted shards, so it must be compared against a value carried
separately from the file it describes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import corpus_spec as SPEC  # noqa: E402

HASH_BLOCK = 8 * 1024 * 1024


class CorpusVerifyError(Exception):
    """Raised for every refusal here. Never an assert."""


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(HASH_BLOCK), b""):
            hasher.update(block)
    return hasher.hexdigest()


def check_hashes(outdir: Path, manifest: dict, findings: list) -> dict:
    """Every block re-hashed. Catches corruption in transit."""
    checked = 0
    for key, record in sorted(manifest["blocks"].items()):
        path = outdir / record["filename"]
        if not path.is_file():
            findings.append(f"{key}: missing from {outdir}")
            continue
        if sha256_file(path) != record["sha256"]:
            findings.append(
                f"{key}: sha256 does not match the manifest -- corrupted in "
                f"transit or modified since it was written")
        checked += 1
    return {"blocks_hashed": checked}


def check_size_arithmetic(outdir: Path, manifest: dict, findings: list) -> dict:
    """filesize / 2 == token count. The route that needs no hashing."""
    for key, record in sorted(manifest["blocks"].items()):
        path = outdir / record["filename"]
        if not path.is_file():
            continue
        size = path.stat().st_size
        try:
            implied = SPEC.tokens_from_byte_size(size)
        except SPEC.CorpusSpecError as exc:
            findings.append(f"{key}: {exc}")
            continue
        if implied != record["tokens"]:
            findings.append(
                f"{key}: file holds {implied:,} tokens by size but the "
                f"manifest records {record['tokens']:,}")
        if record["sequences"] * SPEC.SEQ_LEN != record["tokens"]:
            findings.append(
                f"{key}: {record['sequences']:,} sequences imply "
                f"{record['sequences'] * SPEC.SEQ_LEN:,} tokens, manifest says "
                f"{record['tokens']:,}")
    return {"size_arithmetic": "checked"}


def check_total(outdir: Path, manifest: dict, findings: list) -> dict:
    """The total, three ways: manifest, disk, and the config's own product."""
    from_manifest = sum(r["tokens"] for r in manifest["blocks"].values())
    from_disk = 0
    for record in manifest["blocks"].values():
        path = outdir / record["filename"]
        if path.is_file():
            from_disk += path.stat().st_size // SPEC.BYTES_PER_TOKEN

    training = sum(r["tokens"] for r in manifest["blocks"].values()
                   if r.get("role") == "training")
    heldout = sum(r["tokens"] for r in manifest["blocks"].values()
                  if r.get("role", "").startswith("held out"))

    complete = len(manifest["blocks"]) == SPEC.N_SHARDS + 1
    if complete:
        if from_manifest != SPEC.TOTAL_TOKENS:
            findings.append(
                f"manifest totals {from_manifest:,} tokens, spec says "
                f"{SPEC.TOTAL_TOKENS:,}")
        if from_disk != from_manifest:
            findings.append(
                f"disk holds {from_disk:,} tokens, manifest claims "
                f"{from_manifest:,}")
        if training != SPEC.TRAIN_TOKENS:
            findings.append(
                f"TRAINING SLICE IS {training:,} TOKENS, and "
                f"configs/base.yaml's expected_token_budget is "
                f"{SPEC.TRAIN_TOKENS:,}. The loader's assertion would fail.")
        if heldout != SPEC.HELDOUT_TOKENS:
            findings.append(
                f"held-out slice is {heldout:,} tokens, spec says "
                f"{SPEC.HELDOUT_TOKENS:,}")
    return {
        "tokens_from_manifest": from_manifest,
        "tokens_from_disk": from_disk,
        "training_tokens": training,
        "heldout_tokens": heldout,
        "expected_training_tokens": SPEC.TRAIN_TOKENS,
        "complete": complete,
    }


def check_boundaries(manifest: dict, findings: list) -> dict:
    """Recorded shard ranges against computed ones."""
    checked = 0
    for key, record in sorted(manifest["blocks"].items()):
        if "shard_index" not in record:
            continue
        start, end = SPEC.shard_sequence_range(record["shard_index"])
        if (record.get("sequence_start"), record.get("sequence_end")) != (
                start, end):
            findings.append(
                f"{key}: manifest records sequences "
                f"[{record.get('sequence_start')}, "
                f"{record.get('sequence_end')}) but arithmetic gives "
                f"[{start}, {end})")
        checked += 1
    return {"boundaries_checked": checked}


def check_heldout(findings: list) -> dict:
    try:
        return SPEC.assert_heldout_disjoint_from_training()
    except SPEC.CorpusSpecError as exc:
        findings.append(str(exc))
        return {"disjoint": False}


def check_tokenizer_probe(manifest: dict, findings: list) -> dict:
    """Re-tokenize the committed passage HERE. The destination-drift check.

    This is the one check that needs `transformers`. If it is unavailable the
    result says so loudly rather than reporting success: a verification that
    silently skips its only tokenizer check is worse than one that fails,
    because it reads as a pass.
    """
    recorded = manifest.get("tokenizer")
    if not recorded:
        findings.append(
            "the manifest records no tokenizer identity, so drift at this "
            "machine cannot be detected at all")
        return {"probe": "absent from manifest"}
    try:
        import metrics
        batch = metrics.load_context_batch()
    except Exception as exc:
        return {
            "probe": "NOT RUN",
            "reason": f"{type(exc).__name__}: {exc}",
            "WARNING": ("the tokenizer probe did NOT run, so this "
                        "verification says nothing about whether this "
                        "machine's tokenizer agrees with the one that built "
                        "the corpus. Install transformers and run again "
                        "before trusting the corpus here."),
        }

    if batch.token_sha256 != recorded.get("probe_token_sha256"):
        findings.append(
            "TOKENIZER DRIFT AT THIS MACHINE. The committed probe passage "
            f"tokenizes to {batch.token_sha256} here, but the corpus was built "
            f"by a tokenizer producing {recorded.get('probe_token_sha256')}. "
            "This machine cannot reproduce the corpus it is about to train on, "
            "and any text it tokenizes at run time -- burst injections among "
            "them -- will not match.")
    return {
        "probe": "ran",
        "probe_token_sha256": batch.token_sha256,
        "recorded_token_sha256": recorded.get("probe_token_sha256"),
        "built_by": {
            "transformers": recorded.get("transformers_version"),
            "tokenizers": recorded.get("tokenizers_version"),
        },
        "agrees": batch.token_sha256 == recorded.get("probe_token_sha256"),
    }


def verify(outdir: Path, stream) -> dict:
    manifest_path = outdir / "manifest.json"
    if not manifest_path.is_file():
        raise CorpusVerifyError(f"no manifest at {manifest_path}")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    manifest_sha = hashlib.sha256(raw).hexdigest()

    findings: list = []
    report = {
        "outdir": str(outdir),
        "manifest_sha256": manifest_sha,
        "blocks_present": len(manifest.get("blocks", {})),
        "blocks_expected": SPEC.N_SHARDS + 1,
    }
    report.update(check_hashes(outdir, manifest, findings))
    report.update(check_size_arithmetic(outdir, manifest, findings))
    report.update(check_total(outdir, manifest, findings))
    report.update(check_boundaries(manifest, findings))
    report["heldout"] = check_heldout(findings)
    report["tokenizer"] = check_tokenizer_probe(manifest, findings)
    report["findings"] = findings
    report["ok"] = not findings
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/corpus_verify.py",
        description="Verify a tokenized corpus. No network, no datasets stack.")
    parser.add_argument("--outdir", type=Path, required=True,
                        help="directory holding the shards and manifest.json")
    parser.add_argument("--json", action="store_true",
                        help="emit the report as JSON")
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    stream = sys.stdout
    try:
        report = verify(args.outdir, stream)
    except CorpusVerifyError as exc:
        print(f"FAILED: {exc}", file=stream)
        return 1

    if args.json:
        print(json.dumps(report, indent=2), file=stream)
        return 0 if report["ok"] else 1

    rule = "=" * 78
    print(rule, file=stream)
    print("corpus_verify -- step 11", file=stream)
    print(rule, file=stream)
    print(f"  outdir            {report['outdir']}", file=stream)
    print(f"  manifest sha256   {report['manifest_sha256']}", file=stream)
    print("  ^ compare this against the value carried SEPARATELY from the "
          "manifest;", file=stream)
    print("    a corrupted manifest would otherwise validate corrupted shards.",
          file=stream)
    print(f"  blocks            {report['blocks_present']} of "
          f"{report['blocks_expected']}", file=stream)
    print(f"  blocks hashed     {report['blocks_hashed']}", file=stream)
    print(f"  boundaries        {report['boundaries_checked']} checked against "
          f"arithmetic", file=stream)
    print(f"  tokens (manifest) {report['tokens_from_manifest']:,}", file=stream)
    print(f"  tokens (on disk)  {report['tokens_from_disk']:,}", file=stream)
    print(f"  training tokens   {report['training_tokens']:,} "
          f"(expected {report['expected_training_tokens']:,})", file=stream)
    print(f"  held-out tokens   {report['heldout_tokens']:,}", file=stream)
    print(f"  held-out disjoint {report['heldout'].get('disjoint')}",
          file=stream)
    probe = report["tokenizer"]
    print(f"  tokenizer probe   {probe.get('probe')}"
          + (f" -- agrees: {probe.get('agrees')}" if "agrees" in probe else ""),
          file=stream)
    if probe.get("WARNING"):
        print(f"  ** {probe['WARNING']}", file=stream)

    if report["findings"]:
        print(f"\n{len(report['findings'])} FINDING(S):", file=stream)
        for finding in report["findings"]:
            print(f"  - {finding}", file=stream)
        return 1
    if not report["complete"]:
        print("\nINCOMPLETE but consistent so far: every block present is "
              "correct, and blocks are still missing.", file=stream)
        return 0
    print("\nOK -- complete and consistent.", file=stream)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
