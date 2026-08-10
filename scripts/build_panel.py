#!/usr/bin/env python
"""Assemble {seed, arm, metric, value} records from re-scored run outputs.

`scripts/analysis.py` takes a flat record list and reshapes it. This produces
that list from the per-run JSON written by `scripts/eval_heldout.py`.

HELD-OUT LOSS ONLY, AND THAT IS THE POINT. Held-out loss is a PER-MODEL scalar,
which is the shape `analysis.py`'s panel assumes: it forms the paired difference
by subtracting the seed-matched reference's value, and the twin-vs-twin noise
floor by differencing the reference's values across seeds.

The section 8.4 barrier is NOT that shape. It is defined pairwise -- barrier(arm,
twin) -- so it is already a displacement, and its floor is barrier(twin_i,
twin_j), which cannot be recovered by subtracting two per-seed scalars. Feeding
it through this panel with the twin's cell set to zero would make `noise_floor`
identically zero and `clears_noise_floor` true for every arm. See
`scripts/barrier_analysis.py`, which reads the measured pairs instead.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout-dir", required=True, type=Path)
    ap.add_argument("--metric", default="heldout_loss")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    records, seen = [], {}
    for p in sorted(a.heldout_dir.glob("*.json")):
        d = json.loads(p.read_text())
        key = (d["arm"], int(d["seed"]))
        if key in seen:
            print(f"REFUSED: duplicate {key} in {p} and {seen[key]}",
                  file=sys.stderr)
            return 1
        seen[key] = p
        records.append({"seed": int(d["seed"]), "arm": d["arm"],
                        "metric": a.metric, "value": float(d["heldout_loss"])})

    windows = {json.loads(p.read_text())["windows"]
               for p in a.heldout_dir.glob("*.json")}
    if len(windows) != 1:
        print(f"REFUSED: runs scored on different window counts: {sorted(windows)}. "
              "A paired difference across unequal evaluation sets is not paired.",
              file=sys.stderr)
        return 1

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(records, indent=2) + "\n")
    arms = sorted({r["arm"] for r in records})
    seeds = sorted({r["seed"] for r in records})
    print(f"{len(records)} records -> {a.out}")
    print(f"arms  ({len(arms)}): {', '.join(arms)}")
    print(f"seeds ({len(seeds)}): {seeds}")
    print(f"windows per model: {windows.pop()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
