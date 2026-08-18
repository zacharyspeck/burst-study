#!/usr/bin/env python
"""Is the reconstructed injected row the row the model actually trained on?

The plan's own digests prove the BURST is right: `injection.build_plan` records
`burst_file_sha256` and `burst_token_sha256`, and `2026-08-10-injection-step.json`
carries what training recorded, so the two can be compared. Neither covers the
OTHER 830 TOKENS of the row, because `InjectionPlan.record()` stores
`sequence_length` and no digest of the sequence itself.

So this checks the row the only way that covers all 1024 tokens: by
reproducing a measurement that used them. Training recorded the 194 per-token
losses over the burst region at step 200, computed by the step-199 weights on
the injected row. Reload step 199, rebuild the row, score it again. If the
reconstruction differed anywhere in the left context, the later tokens of the
region would be predicted from different text and the losses would not match.

Step 199 is bit-identical across all four arms within a seed (see
`2026-08-10-step199-digests.json`), so ONE checkpoint per seed reproduces all
three arms' recorded values.

Agreement is reported, not asserted to be exact. The recorded numbers came off a
different machine and this repo has no claim that a forward pass is bitwise
portable; what matters is whether the gap is float noise or a different row.
"""
from __future__ import annotations
import argparse, io, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from burst.config import load_config, INJECTING_ARMS
import model_seam as SEAM
import injection as INJECT

STEP = 199
CKPT = f"step{STEP:06d}_weights_only.pt"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, nargs="+", type=Path)
    ap.add_argument("--record", required=True, type=Path,
                    help="2026-08-10-injection-step.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    runs = {}
    for root in a.runs:
        for d in sorted(Path(root).glob("seed*")):
            runs[(int(d.name[4:6]), d.name.split("_", 1)[1])] = d
    seeds = sorted({s for s, _ in runs})
    recorded = json.loads(a.record.read_text())["runs"]

    tokenizer = INJECT.load_tokenizer(stream=io.StringIO())
    out = {"WHAT_THIS_IS": ("Reproduction of the per-token losses recorded live at "
                           "step 200, from the step-199 checkpoint and a rebuilt "
                           "injected row. Covers all 1024 tokens of the row, which "
                           "no committed digest does."),
           "step": STEP, "rows": {}, "digests": {}}
    worst_overall = 0.0

    for s in seeds:
        src = runs.get((s, "twin")) or runs[(s, INJECTING_ARMS[0])]
        payload = torch.load(src / CKPT, map_location="cpu", weights_only=False)
        if payload["step"] != STEP:
            raise SystemExit(f"{src/CKPT} records step {payload['step']}")
        cfg = load_config(REPO / "configs/base.yaml",
                          REPO / f"configs/runs/{src.name}.yaml",
                          outdir=str(src), family="hf_gpt2",
                          write_provenance=False)
        model = SEAM.build_model(cfg, "hf_gpt2")
        model.load_state_dict(payload["model"])
        model.to(a.device).eval()

        for arm in INJECTING_ARMS:
            name = f"seed{s:02d}_{arm}"
            pcfg = load_config(REPO / "configs/base.yaml",
                               REPO / f"configs/runs/{name}.yaml",
                               outdir=str(src), family="hf_gpt2",
                               write_provenance=False)
            plan = INJECT.build_plan(pcfg, tokenizer)
            rec = recorded[name]
            digests_agree = (
                plan.burst_file_sha256 == rec["burst_file_sha256"]
                and plan.burst_token_sha256 == rec["burst_token_sha256"]
                and plan.batch_slot == rec["batch_slot"]
                and plan.micro_index == rec["micro_index"]
                and plan.row == rec["row"] and plan.position == rec["position"])
            got = INJECT.burst_region_losses(model, plan, device=a.device)["per_token_losses"]
            want = rec["burst_region_per_token_losses"]
            if len(got) != len(want):
                raise SystemExit(f"{name}: {len(got)} losses vs {len(want)} recorded")
            gaps = [abs(g - w) for g, w in zip(got, want)]
            worst = max(gaps)
            worst_overall = max(worst_overall, worst)
            out["rows"][name] = {
                "digests_and_slot_agree": digests_agree,
                "n_tokens": len(got),
                "worst_abs_gap": worst,
                "mean_abs_gap": sum(gaps) / len(gaps),
                "recorded_mean": sum(want) / len(want),
                "reproduced_mean": sum(got) / len(got),
            }
            print(f"{name}: digests {'OK' if digests_agree else 'MISMATCH'}  "
                  f"worst |gap| = {worst:.3e}  "
                  f"mean {sum(got)/len(got):.8f} vs recorded "
                  f"{sum(want)/len(want):.8f}", flush=True)
        del model

    out["worst_abs_gap_over_all_rows"] = worst_overall
    out["all_digests_agree"] = all(r["digests_and_slot_agree"]
                                   for r in out["rows"].values())
    out["device_name"] = (torch.cuda.get_device_name(a.device)
                          if str(a.device).startswith("cuda") else "cpu")
    out["torch"] = torch.__version__
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nworst gap over all {len(out['rows'])} rows: {worst_overall:.3e}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
