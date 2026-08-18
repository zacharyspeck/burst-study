#!/usr/bin/env python
"""Cross-entropy on the injected passages at MANY checkpoints. Figure NEW-2's data.

Same measurement as `stimulus_probe.py`'s first probe -- mean per-token loss over
the injected region, teacher-forced inside the full 1024-token injected row, via
`injection.burst_region_losses` -- run over a schedule of checkpoints instead of
the final one, so the self-effect can be traced across training.

WHY A WORKLIST AND NOT A RUN DIRECTORY. The 191 retained checkpoints per run are
about 95 GB per run and live in object storage; a full sweep is 2.3 TB. The
caller is expected to stage checkpoints a few at a time and delete them after
scoring, so this takes a list of explicit paths and never assumes a run
directory holds more than the one file it is being asked about. It also never
learns where the archive is -- fetching is the driver's job, per CLAUDE.md rule
3, and no credential reaches this module.

THE PLANS ARE BUILT ONCE, NOT PER SEED, and that is correct rather than a
shortcut: `injection.build_plan` assembles the row from `bursts/context.txt`
filler plus the burst file at a fixed position, so the 1024 tokens are identical
for every seed. Verified: `scripts/injected_row_check.py`. The seed enters
training only through `injection.batch_slot_for`, which chooses which row of the
batch gets replaced.

MISSING CHECKPOINTS ARE RECORDED, NOT SKIPPED. A curve with holes in it that
does not say where they are is a curve that looks complete.
"""
from __future__ import annotations
import argparse, io, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from burst.config import load_config
import model_seam as SEAM
import injection as INJECT

#: The passages the curve is about. `random-chars` is not among them: the
#: self-effect is defined only for a passage some arm was trained on and that
#: another arm can be scored against, and the noise arm's text is not prose.
PASSAGE_ARMS = ("fluent-fabricated", "fluent-attested")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worklist", required=True, type=Path,
                    help='JSON list of {"run": name, "step": int, "ckpt": path}')
    ap.add_argument("--config-run", default=None,
                    help="run whose config supplies the architecture; "
                         "defaults to the first worklist entry's run")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-jsonl", required=True, type=Path)
    a = ap.parse_args()

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    items = json.loads(a.worklist.read_text())
    if not items:
        raise SystemExit(f"{a.worklist} is empty")

    cfg_run = a.config_run or items[0]["run"]
    cfg = load_config(REPO / "configs/base.yaml",
                      REPO / f"configs/runs/{cfg_run}.yaml",
                      outdir="/tmp/passage-ce", family="hf_gpt2",
                      write_provenance=False)
    model = SEAM.build_model(cfg, "hf_gpt2").to(a.device).eval()

    tokenizer = INJECT.load_tokenizer(stream=io.StringIO())
    plans = {}
    for arm in PASSAGE_ARMS:
        pcfg = load_config(REPO / "configs/base.yaml",
                           REPO / f"configs/runs/seed00_{arm}.yaml",
                           outdir="/tmp/passage-ce", family="hf_gpt2",
                           write_provenance=False)
        plans[arm] = INJECT.build_plan(pcfg, tokenizer)

    a.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_missing = 0
    with a.out_jsonl.open("a") as fh:
        for it in items:
            p = Path(it["ckpt"])
            if not p.is_file() or p.stat().st_size == 0:
                fh.write(json.dumps({"run": it["run"], "step": it["step"],
                                     "MISSING": str(p)}) + "\n")
                fh.flush()
                n_missing += 1
                continue
            t0 = time.monotonic()
            payload = torch.load(p, map_location="cpu", weights_only=False)
            if payload.get("step") != it["step"]:
                raise SystemExit(f"{p}: records step {payload.get('step')}, "
                                 f"worklist says {it['step']}")
            model.load_state_dict(payload["model"])
            model.to(a.device).eval()
            del payload
            rec = {"run": it["run"], "step": it["step"], "ce": {}}
            for arm, plan in plans.items():
                r = INJECT.burst_region_losses(model, plan, device=a.device)
                rec["ce"][arm] = {"mean_all": r["mean"],
                                  "n_tokens": r["n_predictions"]}
            rec["seconds"] = time.monotonic() - t0
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n_ok += 1
            print(f"{it['run']} step {it['step']}: "
                  + "  ".join(f"{k}={v['mean_all']:.6f}"
                              for k, v in rec["ce"].items()), flush=True)

    print(f"scored {n_ok}, missing {n_missing} -> {a.out_jsonl}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
