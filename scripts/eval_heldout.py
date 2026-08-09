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

sys.path.insert(0, "/home/ubuntu/burst-study")
sys.path.insert(0, "/home/ubuntu/burst-study/scripts")

import numpy as np
import torch
from burst.config import load_config
import model_seam as SEAM
import corpus_spec as SPEC

REPO = Path("/home/ubuntu/burst-study")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir, e.g. /home/ubuntu/runs/seed00_fluent-false")
    ap.add_argument("--corpus", default="/home/ubuntu/corpus")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--windows", type=int, default=0, help="0 = all held-out windows")
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
    model.to("cuda").eval()

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
            ids = torch.from_numpy(chunk).to("cuda")
            logits = model(input_ids=ids).logits
            sl = logits[:, :-1, :].reshape(-1, logits.size(-1)).double()
            lb = ids[:, 1:].reshape(-1)
            nll = torch.nn.functional.cross_entropy(sl, lb, reduction="sum")
            total_nll += float(nll)
            total_tok += lb.numel()

    out = {"run": name, "seed": seed, "arm": arm,
           "heldout_loss": total_nll / total_tok,
           "windows": n_win, "tokens_scored": total_tok}
    (run / "heldout_eval.json").write_text(json.dumps(out, indent=2))
    print(f"{name}: scored {n_win} windows / {total_tok:,} tokens -> written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
