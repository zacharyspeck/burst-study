#!/usr/bin/env python
"""Everything computable from box A alone, for ONE seed's fluent-false/true pair.

The pre-registered headline metric is the barrier of each arm against its
seed-matched TWIN, and no twin runs exist on this box. What IS computable is the
arm-against-arm quantity: barrier(fluent-false, fluent-true) and the raw L2
between them. This is not section 8.4's metric. It is a direct measure of how
far the two arms' final models sit apart, it uses the repo's own barrier
arithmetic, and it survives the checkpoints being deleted.

Uses metrics.interpolate_state_dicts (which refuses to interpolate non-float
buffers -- the causal mask trap) and metrics.barrier_from_losses (the tested
arithmetic), so nothing here reimplements a primitive.
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
import metrics as M

REPO = Path("/home/ubuntu/burst-study")
RUNS = Path("/home/ubuntu/runs")


def heldout_loss(model, data, n_win, L, batch):
    tot_nll, tot_tok = 0.0, 0
    with torch.no_grad():
        for i in range(0, n_win, batch):
            j = min(i + batch, n_win)
            chunk = np.asarray(data[i * L:j * L], dtype=np.int64).reshape(j - i, L)
            ids = torch.from_numpy(chunk).to("cuda")
            logits = model(input_ids=ids).logits
            sl = logits[:, :-1, :].reshape(-1, logits.size(-1)).double()
            lb = ids[:, 1:].reshape(-1)
            tot_nll += float(torch.nn.functional.cross_entropy(sl, lb, reduction="sum"))
            tot_tok += lb.numel()
    return tot_nll / tot_tok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--windows", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--corpus", default="/home/ubuntu/corpus")
    a = ap.parse_args()

    ff = RUNS / f"seed{a.seed:02d}_fluent-false" / "step009535_full.pt"
    ft = RUNS / f"seed{a.seed:02d}_fluent-true" / "step009535_full.pt"
    for p in (ff, ft):
        if not p.exists():
            print(f"missing {p}", file=sys.stderr); return 1

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    cfg = load_config(REPO / "configs/base.yaml",
                      REPO / f"configs/runs/seed{a.seed:02d}_fluent-false.yaml",
                      outdir=str(RUNS / f"seed{a.seed:02d}_fluent-false"),
                      family="hf_gpt2", write_provenance=False)

    sd_a = torch.load(ff, map_location="cpu", weights_only=False)["model"]
    sd_b = torch.load(ft, map_location="cpu", weights_only=False)["model"]

    # raw L2 between the two arms' final weights
    ma = SEAM.build_model(cfg, "hf_gpt2"); ma.load_state_dict(sd_a)
    mb = SEAM.build_model(cfg, "hf_gpt2"); mb.load_state_dict(sd_b)
    l2 = M.l2_distance_raw(ma, mb)
    del mb

    L = cfg.training.seq_len
    data = np.memmap(Path(a.corpus) / "heldout.bin", dtype="<u2", mode="r")
    n_win = min(len(data) // L, a.windows)

    losses = []
    for alpha in M.ALPHA_GRID:
        sd = M.interpolate_state_dicts(sd_a, sd_b, float(alpha))
        ma.load_state_dict(sd)
        ma.to("cuda").eval()
        losses.append(heldout_loss(ma, data, n_win, L, a.batch))
        ma.to("cpu")

    res = M.barrier_from_losses(M.ALPHA_GRID, losses)
    out = {
        "seed": a.seed,
        "pair": "fluent-false_vs_fluent-true",
        "note": "ARM-vs-ARM. NOT section 8.4's headline, which is arm-vs-twin.",
        "windows": n_win,
        "l2_raw": l2,
        "alphas": list(M.ALPHA_GRID),
        "losses": losses,
        "barrier": res,
    }
    p = RUNS / f"armpair_seed{a.seed:02d}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"seed{a.seed:02d}: l2={l2:.6f} max_excess={res.get('max_excess'):.8f} -> {p.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
