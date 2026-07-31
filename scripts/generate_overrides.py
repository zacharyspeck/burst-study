#!/usr/bin/env python
"""Generate all 40 run override files: 10 seeds x 4 arms.

The seed count and the arm names are read from configs/base.yaml rather than
hard-coded here, so there is exactly one place where the shape of the study is
written down.

    python scripts/generate_overrides.py            # write configs/runs/
    python scripts/generate_overrides.py --check    # verify, change nothing

--check exits non-zero if any file is missing or differs. Run it before a
launch, or in CI, to prove nobody hand-edited an override.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Make `import burst` work when this script is run directly from the repo root
# without the package being installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from burst.config import ARMS, run_name_for  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = REPO_ROOT / "configs" / "base.yaml"
DEFAULT_OUT = REPO_ROOT / "configs" / "runs"


def override_text(seed: int, arm: str) -> str:
    """The entire contents of one override file. Two lines, no comments.

    Kept comment-free on purpose: these files are generated, and a comment
    here would be duplicated 40 times and go stale 40 times. The explanation
    lives in configs/base.yaml and in the README.
    """
    return f"seed: {seed}\narm: {arm}\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE,
                        help=f"base config to read n_seeds/arms from (default: {DEFAULT_BASE})")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT,
                        help=f"where the override files go (default: {DEFAULT_OUT})")
    parser.add_argument("--check", action="store_true",
                        help="verify existing files match; write nothing")
    args = parser.parse_args(argv)

    base = yaml.safe_load(args.base.read_text(encoding="utf-8"))
    n_seeds = base["experiment"]["n_seeds"]
    arms = base["experiment"]["arms"]

    if tuple(arms) != ARMS:
        print(f"ERROR: {args.base} lists arms {arms}, but burst/config.py "
              f"defines {list(ARMS)}.", file=sys.stderr)
        return 1

    args.outdir.mkdir(parents=True, exist_ok=True)

    written = unchanged = mismatched = missing = 0
    # Seeds are 0-indexed: seed00 .. seed09 for n_seeds = 10.
    for seed in range(n_seeds):
        for arm in arms:
            path = args.outdir / f"{run_name_for(seed, arm)}.yaml"
            want = override_text(seed, arm)
            have = path.read_text(encoding="utf-8") if path.is_file() else None

            if args.check:
                if have is None:
                    print(f"MISSING   {path}")
                    missing += 1
                elif have != want:
                    print(f"MISMATCH  {path}")
                    mismatched += 1
                continue

            if have == want:
                unchanged += 1
            else:
                path.write_text(want, encoding="utf-8")
                written += 1

    total = n_seeds * len(arms)
    if args.check:
        bad = missing + mismatched
        print(f"checked {total} override files in {args.outdir}: "
              f"{total - bad} ok, {missing} missing, {mismatched} mismatched")
        return 1 if bad else 0

    print(f"{total} override files in {args.outdir}: "
          f"{written} written, {unchanged} already correct")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
