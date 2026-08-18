#!/usr/bin/env python
"""Mean next-token cross-entropy of one final checkpoint on the held-out slice.

Writes a JSON with the loss. Does NOT print it -- see summarize_spread.py for
why. Forward-only, fp32 (no autocast), fp64 loss accumulation, eval mode. The
absolute value is less important than that all 16 models are scored identically,
since the quantity of interest is a within-seed paired difference.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
from burst.config import load_config
import model_seam as SEAM
import corpus_spec as SPEC


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir, e.g. /home/ubuntu/runs/seed00_fluent-fabricated")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--windows", type=int, default=0, help="0 = all held-out windows")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="", help="where to write the JSON; "
                    "default is heldout_eval.json inside the run dir. Given "
                    "explicitly when re-scoring on different hardware, so the "
                    "original box's record is not overwritten.")
    args = ap.parse_args()

    run = Path(args.run)
    ckpt = run / "step009535_full.pt"
    if not ckpt.exists():
        print(f"{run.name}: no final checkpoint", file=sys.stderr)
        return 1

    name = run.name                      # seedNN_arm
    seed = int(name[4:6]); arm = name.split("_", 1)[1]
    cfg = load_config(REPO / "configs/base.yaml",
                      REPO / f"configs/runs/{name}.yaml",
                      outdir=str(run), family="hf_gpt2", write_provenance=False)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = SEAM.build_model(cfg, "hf_gpt2")
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert payload["kind"] == "full" and payload["step"] == 9535, payload.get("step")
    model.load_state_dict(payload["model"])
    model.to(args.device).eval()

    L = cfg.training.seq_len
    data = np.memmap(Path(args.corpus) / "heldout.bin", dtype="<u2", mode="r")
    n_win = len(data) // L
    if args.windows:
        n_win = min(n_win, args.windows)

    total_nll = 0.0
    total_tok = 0
    with torch.no_grad():
        for i in range(0, n_win, args.batch):
            j = min(i + args.batch, n_win)
            chunk = np.asarray(data[i * L:j * L], dtype=np.int64).reshape(j - i, L)
            ids = torch.from_numpy(chunk).to(args.device)
            logits = model(input_ids=ids).logits
            sl = logits[:, :-1, :].reshape(-1, logits.size(-1)).double()
            lb = ids[:, 1:].reshape(-1)
            nll = torch.nn.functional.cross_entropy(sl, lb, reduction="sum")
            total_nll += float(nll)
            total_tok += lb.numel()

    out = {"run": name, "seed": seed, "arm": arm,
           "heldout_loss": total_nll / total_tok,
           "windows": n_win, "tokens_scored": total_tok}
    out["device_name"] = (torch.cuda.get_device_name(args.device)
                          if str(args.device).startswith("cuda") else "cpu")
    out["torch"] = torch.__version__
    dest = Path(args.out) if args.out else run / "heldout_eval.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"{name}: loss={out['heldout_loss']:.10f} scored {n_win} windows / "
          f"{total_tok:,} tokens -> {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
