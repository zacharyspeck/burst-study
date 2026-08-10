#!/usr/bin/env python
"""Plain interpolation loss barrier between ANY two runs' final checkpoints.

THIS IS SECTION 8.4'S HEADLINE METRIC when --run-b is the seed-matched `twin`.
`scripts/arm_pair_metrics.py` computes the arm-vs-arm quantity that box A could
reach on its own; this computes the arm-vs-twin quantity the pre-registration
actually fixes, and the twin-vs-twin pairs that make up the noise floor. The
two scripts share their arithmetic -- both go through
`metrics.interpolate_state_dicts` and `metrics.barrier_from_losses` -- so
nothing here reimplements a primitive.

PLAIN, not permutation-aligned: preregistration.md 8.4's decision rule lands on
the plain barrier in three of its four branches, and `metrics.aligned_barrier`
raises NotImplementedError.

Paths are arguments rather than module constants, because this runs on a
different machine from the one that trained the runs.
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
import metrics as M

FINAL = "step009535_full.pt"


def load_final(run: Path) -> dict:
    p = run / FINAL
    if not p.exists():
        raise SystemExit(f"missing final checkpoint: {p}")
    payload = torch.load(p, map_location="cpu", weights_only=False)
    if payload.get("kind") != "full" or payload.get("step") != 9535:
        raise SystemExit(f"{p}: unexpected kind/step "
                         f"{payload.get('kind')}/{payload.get('step')}")
    return payload["model"]


def heldout_loss(model, data, n_win, L, batch, device):
    """Mean next-token cross-entropy. fp32 forward, fp64 accumulation."""
    tot_nll, tot_tok = 0.0, 0
    with torch.no_grad():
        for i in range(0, n_win, batch):
            j = min(i + batch, n_win)
            chunk = np.asarray(data[i * L:j * L], dtype=np.int64).reshape(j - i, L)
            ids = torch.from_numpy(chunk).to(device)
            logits = model(input_ids=ids).logits
            sl = logits[:, :-1, :].reshape(-1, logits.size(-1)).double()
            lb = ids[:, 1:].reshape(-1)
            tot_nll += float(torch.nn.functional.cross_entropy(sl, lb, reduction="sum"))
            tot_tok += lb.numel()
    return tot_nll / tot_tok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True, type=Path)
    ap.add_argument("--run-b", required=True, type=Path)
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--windows", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    name_a, name_b = a.run_a.name, a.run_b.name
    # The config only supplies architecture and seq_len here, both identical
    # across arms; the run override is named for run A so the loader gets a
    # file that exists.
    cfg = load_config(REPO / "configs/base.yaml",
                      REPO / f"configs/runs/{name_a}.yaml",
                      outdir=str(a.run_a), family="hf_gpt2",
                      write_provenance=False)

    sd_a = load_final(a.run_a)
    sd_b = load_final(a.run_b)

    # Raw L2 via the models, matching arm_pair_metrics.py (named_parameters
    # only -- buffers excluded).
    ma = SEAM.build_model(cfg, "hf_gpt2"); ma.load_state_dict(sd_a)
    mb = SEAM.build_model(cfg, "hf_gpt2"); mb.load_state_dict(sd_b)
    l2 = M.l2_distance_raw(ma, mb)
    del mb

    L = cfg.training.seq_len
    data = np.memmap(a.corpus / "heldout.bin", dtype="<u2", mode="r")
    n_win = len(data) // L
    if a.windows:
        n_win = min(n_win, a.windows)

    # Interpolate on-device: the primitive is plain tensor arithmetic, so it
    # runs wherever the tensors live, and keeping them on the GPU avoids
    # 21 round trips of half a gigabyte.
    dev = torch.device(a.device)
    sd_a = {k: v.to(dev) for k, v in sd_a.items()}
    sd_b = {k: v.to(dev) for k, v in sd_b.items()}
    ma.to(dev).eval()

    losses = []
    for alpha in M.ALPHA_GRID:
        sd = M.interpolate_state_dicts(sd_a, sd_b, float(alpha))
        ma.load_state_dict(sd)
        losses.append(heldout_loss(ma, data, n_win, L, a.batch, dev))
        del sd

    res = M.barrier_from_losses(M.ALPHA_GRID, losses)
    out = {
        "run_a": name_a,
        "run_b": name_b,
        "windows": n_win,
        "tokens_scored": n_win * (L - 1),
        "l2_raw": l2,
        "barrier": res,
        "device_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
        "torch": torch.__version__,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"{name_a} vs {name_b}: l2={l2:.6f} "
          f"max_excess={res['max_excess']:.8f} -> {a.out.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
