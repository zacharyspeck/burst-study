#!/usr/bin/env python
"""Per-layer CKA between checkpoints. The representation-space measurement.

WHY THIS EXISTS. The registered analysis reports two things about a pair of
checkpoints: how far apart the weights are (raw L2) and whether a straight line
between them stays low (the barrier). Both read PARAMETERS. This reads
ACTIVATIONS, so it is an independent third view of the same question -- and it
is the one that can answer "where", because it is per layer and the other two
are single scalars over the whole network.

`scripts/metrics.py` has had `per_layer_cka` since step 10, tested against a
hand-transcribed HSIC and against its own responsiveness fixtures, and it has
NEVER BEEN RUN ON A TRAINED MODEL. Everything it has produced so far came from
junk weights. This script is what closes that gap; nothing here reimplements a
primitive, and the CKA form is `metrics.CKA_VARIANT` verbatim.

THREE DESIGNS, chosen so each number has something to be read against:

  arm-vs-twin   each arm against its seed-matched control at one step. This is
                the paired contrast the pre-registration fixes, transplanted
                onto a representation metric.
  twin-vs-twin  every across-seed pair of controls at the same step. NOT a
                noise floor in the measurement-error sense -- these runs differ
                by initialization AND data order, so this is the scale of two
                genuinely different runs. It is what says whether an
                arm-vs-twin CKA of 0.999 is near or far.
  within-run    one run against ITSELF at two steps. With step 199 the last
                checkpoint before injection and 249 the first one after, this
                is where in the network the fifty steps after the burst moved
                anything -- and, compared arm against twin, whether the burst
                moved it differently from an ordinary batch.

THE BATCH IS THE COMMITTED CONTEXT PASSAGE, which is the same text the injected
sequences are built out of: `bursts/provenance.json` records 830 of its tokens
as the filler the arms wrapped around the burst. That makes this the most
favourable site available for finding an effect, exactly as the stimulus probe
is, and it should be read that way rather than as a neutral sample of language.

ACTIVATION COSINE IS REPORTED BESIDE EVERY CKA and is not decoration. CKA is
invariant to an orthogonal change of basis and cosine is not, so a pair that
scores ~1.0 on CKA and poorly on cosine differs by a rotation rather than by
content. Reading either alone cannot tell those apart. See
`metrics.activation_cosine`.

Paths are arguments, not module constants: this runs on a different machine
from the one that trained the runs.
"""
from __future__ import annotations
import argparse, io, json, sys
from dataclasses import dataclass
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
FINAL_STEP = 9535


@dataclass(frozen=True)
class _OnDevice(M.Batch):
    """The committed batch, handing back ids on the model's device.

    Subclassed rather than reimplemented: `M.Batch.identity()` is what ties a
    CKA number to the input that produced it, and a locally rebuilt batch would
    have carried the same fields with none of the provenance check behind them.
    """
    device: str = "cpu"

    def input_ids(self):
        return super().input_ids().to(self.device)


def ckpt_path(run: Path, step: int) -> Path:
    kind = "full" if step == FINAL_STEP else "weights_only"
    return run / f"step{step:06d}_{kind}.pt"


def load_state(run: Path, step: int) -> dict:
    p = ckpt_path(run, step)
    if not p.exists():
        raise SystemExit(f"missing checkpoint: {p}")
    payload = torch.load(p, map_location="cpu", weights_only=False)
    if payload.get("step") != step:
        raise SystemExit(f"{p}: records step {payload.get('step')}, "
                         f"expected {step}")
    return payload["model"]


def discover(roots) -> dict:
    """(seed, arm) -> run directory, over every root given."""
    found = {}
    for root in roots:
        for d in sorted(Path(root).glob("seed*")):
            if not d.is_dir():
                continue
            name = d.name
            seed, arm = int(name[4:6]), name.split("_", 1)[1]
            if (seed, arm) in found:
                raise SystemExit(
                    f"run {name} appears under two roots: {found[(seed, arm)]} "
                    f"and {d}. Which copy a number came from would not be "
                    f"recoverable from the output.")
            found[(seed, arm)] = d
    if not found:
        raise SystemExit(f"no seed* directories under {list(roots)}")
    return found


def build_pairs(design, runs, args) -> list:
    seeds = sorted({s for s, _ in runs})
    arms = sorted({a for _, a in runs if a != REF})

    if design == "arm-vs-twin":
        missing = [(s, a) for s in seeds for a in arms + [REF]
                   if (s, a) not in runs]
        if missing:
            raise SystemExit(f"incomplete panel; missing {missing}")
        return [{"name": f"{a}_vs_twin_seed{s:02d}", "seed": s, "arm": a,
                 "a": (runs[(s, a)], args.step), "b": (runs[(s, REF)], args.step)}
                for s in seeds for a in arms]

    if design == "twin-vs-twin":
        return [{"name": f"twintwin_seed{i:02d}_seed{j:02d}", "seed": None,
                 "arm": REF,
                 "a": (runs[(i, REF)], args.step), "b": (runs[(j, REF)], args.step)}
                for i, j in combinations(seeds, 2)]

    if design == "within-run":
        if args.step_a == args.step_b:
            raise SystemExit("--step-a and --step-b are the same checkpoint; "
                             "CKA of a checkpoint with itself is 1 by "
                             "construction and measures nothing")
        return [{"name": f"{d.name}_s{args.step_a}_s{args.step_b}",
                 "seed": s, "arm": a,
                 "a": (d, args.step_a), "b": (d, args.step_b)}
                for (s, a), d in sorted(runs.items(), key=lambda kv: kv[0][::-1])]

    raise SystemExit(f"unknown design {design!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, nargs="+", type=Path,
                    help="directories holding seed*_arm run directories")
    ap.add_argument("--design", required=True,
                    choices=("arm-vs-twin", "twin-vs-twin", "within-run"))
    ap.add_argument("--step", type=int, default=FINAL_STEP,
                    help="checkpoint step for the two across-run designs")
    ap.add_argument("--step-a", type=int, default=199)
    ap.add_argument("--step-b", type=int, default=249)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    runs = discover(a.runs)
    pairs = build_pairs(a.design, runs, a)

    base = M.load_context_batch()
    batch = _OnDevice(ids=base.ids, source=base.source,
                      file_sha256=base.file_sha256,
                      token_sha256=base.token_sha256,
                      tokenizer=base.tokenizer, device=a.device)

    any_run = next(iter(runs.values()))
    cfg = load_config(REPO / "configs/base.yaml",
                      REPO / f"configs/runs/{any_run.name}.yaml",
                      outdir=str(any_run), family="hf_gpt2",
                      write_provenance=False)
    model = SEAM.build_model(cfg, "hf_gpt2").to(a.device).eval()

    # How many pairs still need each checkpoint's activations. Activations are
    # ~41 MB each and a design can span 64 checkpoints, so they are dropped the
    # moment nothing is left to compare them against.
    need: dict = {}
    for p in pairs:
        for side in ("a", "b"):
            need[p[side]] = need.get(p[side], 0) + 1

    cache: dict = {}
    route_check = None

    def acts(key):
        if key not in cache:
            run, step = key
            model.load_state_dict(load_state(run, step))
            model.to(a.device).eval()
            nonlocal route_check
            if route_check is None:
                # Hooks and output_hidden_states must reach identical tensors.
                # Run once, on a real trained checkpoint: this is the check
                # that the tap list is on the modules it is believed to be on,
                # and it has only ever run on junk weights before today.
                route_check = M.cross_check_activation_routes(model, batch)
            cache[key] = M.layer_activations(model, batch)
        return cache[key]

    def release(key):
        need[key] -= 1
        if need[key] == 0:
            cache.pop(key, None)

    out = {
        "design": a.design,
        "cka_variant": M.CKA_VARIANT,
        "batch": base.identity(),
        "n_pairs": len(pairs),
        "pairs": [],
    }

    for p in pairs:
        A, B = acts(p["a"]), acts(p["b"])
        cka = M.per_layer_cka(A, B)
        cos = M.per_layer_activation_similarity(A, B)
        rec = {
            "name": p["name"], "seed": p["seed"], "arm": p["arm"],
            "run_a": p["a"][0].name, "step_a": p["a"][1],
            "run_b": p["b"][0].name, "step_b": p["b"][1],
            "n_layers": len(cka),
            "layers": [
                {"layer": c["layer"], "cka": c["cka"],
                 "outside_unit_interval": c["outside_unit_interval"],
                 "cosine_median": s["cosine_median"],
                 "cosine_min": s["cosine_min"],
                 "norm_ratio_median": s["norm_ratio_median"]}
                for c, s in zip(cka, cos)
            ],
        }
        out["pairs"].append(rec)
        release(p["a"]); release(p["b"])
        worst = min(rec["layers"], key=lambda L: L["cka"])
        print(f"{p['name']}: min cka={worst['cka']:.8f} @L{worst['layer']}  "
              f"cka[last]={rec['layers'][-1]['cka']:.8f}", flush=True)

    out["activation_route_cross_check"] = route_check
    out["device_name"] = (torch.cuda.get_device_name(a.device)
                          if str(a.device).startswith("cuda") else "cpu")
    out["torch"] = torch.__version__
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {a.out} ({len(pairs)} pairs)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
