#!/usr/bin/env python
"""Raw L2 between checkpoints, over the standard pair designs. No GPU, no batch.

The same three designs as `scripts/cka_pairs.py` -- arm-vs-twin, twin-vs-twin,
within-run -- on the cheapest metric in the study. Where that script needs a
tokenizer, a committed batch and a forward pass, this needs only
`named_parameters()`, so it runs anywhere and the arm-to-twin / twin-to-twin
ratio can be recomputed at any step a checkpoint exists for.

RAW means raw. `metrics.l2_distance_raw` counts a head or neuron permutation as
content at full size; `metrics.aligned_l2` still raises NotImplementedError.
For arm-vs-twin pairs this is harmless -- they share an initialization -- but the
twin-vs-twin denominator of the ratio is gauge-inflated, which is measured in
section 11.4 of the results record rather than assumed. Read the ratio as a
lower bound on the burst's displacement.

MODELS, NOT STATE DICTS. `parameter_view` goes through `named_parameters()`,
which deduplicates the tied `lm_head.weight` / `transformer.wte.weight`. Summing
over a state_dict instead would double-weight 38M of GPT-2's 124M parameters in
every distance here.
"""
from __future__ import annotations
import argparse, json, statistics, sys
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from burst.config import load_config
import model_seam as SEAM
import metrics as M

REF = "twin"


def ckpt_path(run: Path, step: int) -> Path:
    """The retained schedule writes weights_only every 50 steps and full every
    1000, so the same step number is one filename or the other. Both are tried
    rather than guessed, because guessing produces a missing-file error that
    looks like a missing checkpoint."""
    for kind in ("weights_only", "full"):
        p = run / f"step{step:06d}_{kind}.pt"
        if p.is_file():
            return p
    raise SystemExit(
        f"no checkpoint for step {step} in {run} (tried both _weights_only.pt "
        f"and _full.pt)")


def load_state(run: Path, step: int) -> dict:
    p = ckpt_path(run, step)
    payload = torch.load(p, map_location="cpu", weights_only=False)
    if payload.get("step") != step:
        raise SystemExit(f"{p}: records step {payload.get('step')}, want {step}")
    return payload["model"]


def discover(roots) -> dict:
    found = {}
    for root in roots:
        for d in sorted(Path(root).glob("seed*")):
            if d.is_dir():
                found[(int(d.name[4:6]), d.name.split("_", 1)[1])] = d
    if not found:
        raise SystemExit(f"no seed* directories under {list(roots)}")
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, nargs="+", type=Path)
    ap.add_argument("--steps", required=True, type=int, nargs="+")
    ap.add_argument("--design", required=True, nargs="+",
                    choices=("arm-vs-twin", "twin-vs-twin"))
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    runs = discover(a.runs)
    seeds = sorted({s for s, _ in runs})
    arms = sorted({x for _, x in runs if x != REF})

    out = {"WHAT_THIS_IS": ("Raw L2 between checkpoints. RAW: no permutation "
                            "alignment, see this module's docstring."),
           "steps": a.steps, "designs": a.design, "by_step": {}}

    for step in a.steps:
        cache = {}

        def model_at(run):
            if run not in cache:
                cfg = load_config(REPO / "configs/base.yaml",
                                  REPO / f"configs/runs/{run.name}.yaml",
                                  outdir=str(run), family="hf_gpt2",
                                  write_provenance=False)
                m = SEAM.build_model(cfg, "hf_gpt2")
                m.load_state_dict(load_state(run, step))
                cache[run] = m
            return cache[run]

        rec = {"pairs": [], "summary": {}}
        if "arm-vs-twin" in a.design:
            for s in seeds:
                for arm in arms:
                    v = M.l2_distance_raw(model_at(runs[(s, arm)]),
                                          model_at(runs[(s, REF)]))
                    rec["pairs"].append({"kind": "arm-vs-twin", "seed": s,
                                         "arm": arm, "l2_raw": v})
                    print(f"step {step:6d}  {arm:14s} seed{s:02d} vs twin: "
                          f"{v:.6f}", flush=True)
        if "twin-vs-twin" in a.design:
            for i, j in combinations(seeds, 2):
                v = M.l2_distance_raw(model_at(runs[(i, REF)]),
                                      model_at(runs[(j, REF)]))
                rec["pairs"].append({"kind": "twin-vs-twin", "seed_a": i,
                                     "seed_b": j, "l2_raw": v})
                print(f"step {step:6d}  twin{i:02d} vs twin{j:02d}: {v:.6f}",
                      flush=True)

        for kind in ("arm-vs-twin", "twin-vs-twin"):
            vals = [p["l2_raw"] for p in rec["pairs"] if p["kind"] == kind]
            if vals:
                rec["summary"][kind] = {
                    "n": len(vals), "mean": statistics.fmean(vals),
                    "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                    "min": min(vals), "max": max(vals)}
            for arm in arms:
                av = [p["l2_raw"] for p in rec["pairs"]
                      if p["kind"] == "arm-vs-twin" and p.get("arm") == arm]
                if av:
                    rec["summary"][arm] = {
                        "n": len(av), "mean": statistics.fmean(av),
                        "sd": statistics.stdev(av) if len(av) > 1 else 0.0}
        if {"arm-vs-twin", "twin-vs-twin"} <= set(rec["summary"]):
            rec["summary"]["ratio_armtwin_over_twintwin"] = (
                rec["summary"]["arm-vs-twin"]["mean"]
                / rec["summary"]["twin-vs-twin"]["mean"])
        out["by_step"][str(step)] = rec
        cache.clear()

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nwrote {a.out}")
    for step, rec in out["by_step"].items():
        s = rec["summary"]
        line = f"step {step:>6s}: "
        if "arm-vs-twin" in s:
            line += f"arm-vs-twin {s['arm-vs-twin']['mean']:.4f}  "
        if "twin-vs-twin" in s:
            line += f"twin-vs-twin {s['twin-vs-twin']['mean']:.4f}  "
        if "ratio_armtwin_over_twintwin" in s:
            line += f"ratio {s['ratio_armtwin_over_twintwin']:.4f}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
