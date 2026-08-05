#!/usr/bin/env python
"""The interpolation barrier between two checkpoints, as a CLI.

    python scripts/pilot_barrier.py A.pt B.pt --label NAME --out NAME.json \
        --cfg-scratch /somewhere/outside/the/run/dirs

`metrics.barrier` takes two live models; this takes two checkpoint paths, which
is what a pilot actually has on disk. It exists so that the numbers in
docs/measurements/2026-08-05-pilot-results.md can be re-derived rather than
believed.

THIS IS THE PLAIN BARRIER, AND AS OF 2026-08-05 THAT IS THE HEADLINE METRIC.
docs/preregistration.md section 8.4's rule was evaluated against the pilot's
step-199 checkpoint and branched to `plain_loss_barrier`
(docs/measurements/12-injection-point-ruler.md), with the permutation-aligned
barrier demoted to a robustness check and D-6 explicitly NOT required. Before
that measurement the plain barrier was the implemented neighbour of the spec's
metric; after it, it is the metric.

ORDERING. Section 8.5 forbids examining any arm-vs-twin displacement before the
ruler's branch is recorded, and section 8.7 names the reverse order as
invalidating section 8. This script cannot enforce that -- it takes any two
checkpoints and cannot know what has already been looked at -- so the constraint
lives with the caller. Run the ruler first. S96 records the near miss.

CPU AND fp64, DELIBERATELY. metrics.cross_entropy_loss builds its ids on the CPU
and casts logits to double before the cross-entropy, so the ruler is fp64 and
touches no tensor core. That is the right choice for a measurement: the training
precision is a throughput decision (S94, S95) with no business inside the metric.
Moving the models to CUDA would only device-mismatch against those ids.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

import metrics as M  # noqa: E402
import model_seam as SEAM  # noqa: E402
from burst.config import load_config  # noqa: E402


def load(path: Path, cfg):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = SEAM.build_model(cfg, payload["family"])
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path,
                    default=REPO / "configs" / "base.yaml")
    ap.add_argument(
        "--run", type=Path,
        default=REPO / "configs" / "runs" / "seed00_twin.yaml",
        help=("any run override. build_model reads only cfg.model, which the "
              "study's central claim makes identical across every arm and seed "
              "-- an override may set seed and arm and nothing else -- so the "
              "architecture resolves the same whichever is passed."))
    ap.add_argument(
        "--cfg-scratch", type=Path, required=True,
        help=("where load_config may write resolved_config.yaml. An output "
              "path, so it is an argument and never a config value. Must NOT "
              "be a run directory: those already hold the provenance of the "
              "training that produced these checkpoints."))
    args = ap.parse_args(argv)

    # write_provenance=False: this is a measurement over finished runs, and it
    # must not stamp a fresh provenance record describing a run that never
    # happened.
    cfg = load_config(args.config, args.run, outdir=args.cfg_scratch,
                      write_provenance=False, family="hf_gpt2")

    batch = M.load_context_batch()
    model_a, pay_a = load(args.a, cfg)
    model_b, pay_b = load(args.b, cfg)

    res = M.barrier(model_a, model_b, batch)
    res["label"] = args.label
    for key, path, pay in (("a", args.a, pay_a), ("b", args.b, pay_b)):
        res[key] = {"path": str(path), "seed": pay["seed"], "arm": pay["arm"],
                    "step": pay["step"], "kind": pay["kind"]}
    res["device"] = "cpu"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"{args.label}: max_excess={res['max_excess']:.6f} "
          f"min_excess={res['min_excess']:.6f} "
          f"rose={res['rose_above_chord']} "
          f"L(0)={res['losses'][0]:.6f} L(1)={res['losses'][-1]:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
