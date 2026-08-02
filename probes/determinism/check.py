"""Launch two fresh training processes with the same seed and compare them bitwise.

The comparison lives here rather than in train_once.py on purpose. "Two fresh
processes" is the thing being tested, and a single process that trained twice
and compared its own results would share an allocator, a cuBLAS handle, a
cuDNN autotune cache and an RNG lineage across the two runs -- which is most of
what could go wrong.

Exit status is 0 only if every parameter tensor and every optimizer moment is
byte-identical between the two runs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

RULE = "=" * 78
THIN = "-" * 78


def run_one(label: str, outdir: Path, args, python: str) -> dict:
    cmd = [
        python, str(HERE / "train_once.py"),
        "--outdir", str(outdir),
        "--steps", str(args.steps),
        "--micro-batch", str(args.micro_batch),
        "--attn", args.attn,
        "--dtype", args.dtype,
        "--adamw-impl", args.adamw_impl,
        "--run", args.run,
    ]
    if args.warmup_steps is not None:
        cmd += ["--warmup-steps", str(args.warmup_steps)]
    if args.profile_kernels:
        cmd += ["--profile-kernels"]

    env = dict(os.environ)
    # Set here, not inside the training process: cuBLAS reads this at CUDA
    # init, and a process cannot reliably set it for itself.
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # One GPU, pinned. The probe must never reach for a second device.
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONHASHSEED"] = "0"

    print(f"\n{THIN}\n[{label}] {' '.join(cmd)}\n{THIN}", flush=True)
    started = time.monotonic()
    result = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT))
    elapsed = time.monotonic() - started
    if result.returncode != 0:
        raise SystemExit(
            f"[{label}] training process exited {result.returncode}; "
            f"nothing to compare.")
    print(f"[{label}] finished in {elapsed:.1f}s", flush=True)
    return json.loads((outdir / "digest.json").read_text(encoding="utf-8"))


def compare(a: dict, b: dict) -> tuple[bool, list[str]]:
    """Bitwise comparison. Returns (identical, findings)."""
    findings: list[str] = []

    for step_a, step_b in zip(a["step_log"], b["step_log"]):
        if step_a != step_b:
            for key in ("lr_bits", "loss_bits", "grad_norm_bits"):
                if step_a[key] != step_b[key]:
                    findings.append(
                        f"step {step_a['step']}: {key} diverges "
                        f"(A={step_a[key]} B={step_b[key]})")
            break  # the first diverging step is the informative one

    pa, pb = a["param_digests"], b["param_digests"]
    if set(pa) != set(pb):
        findings.append(
            f"parameter sets differ: only in A={sorted(set(pa) - set(pb))}, "
            f"only in B={sorted(set(pb) - set(pa))}")
    mismatched = [n for n in sorted(set(pa) & set(pb)) if pa[n] != pb[n]]
    if mismatched:
        findings.append(
            f"{len(mismatched)} of {len(pa)} parameter tensors differ; "
            f"first few: {mismatched[:5]}")

    oa, ob = a["optimizer_digests"], b["optimizer_digests"]
    opt_mismatched = [n for n in sorted(set(oa) & set(ob)) if oa[n] != ob[n]]
    if opt_mismatched:
        findings.append(
            f"{len(opt_mismatched)} of {len(oa)} optimizer moment tensors "
            f"differ; first few: {opt_mismatched[:5]}")

    identical = a["combined_sha256"] == b["combined_sha256"] and not findings
    return identical, findings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--attn", choices=("sdpa", "math"), default="sdpa")
    ap.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    ap.add_argument("--adamw-impl", choices=("foreach", "fused", "single"),
                    default="foreach")
    ap.add_argument("--warmup-steps", type=int, default=None)
    ap.add_argument("--run", default=str(REPO_ROOT / "configs" / "runs" / "seed03_twin.yaml"))
    ap.add_argument("--outdir-root", type=Path,
                    default=REPO_ROOT / "probe-runs" / "determinism")
    ap.add_argument("--profile-kernels", action="store_true")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    tag = (f"steps{args.steps}_mb{args.micro_batch}_{args.attn}_"
           f"{args.dtype}_{args.adamw_impl}"
           + (f"_warmup{args.warmup_steps}" if args.warmup_steps is not None else ""))
    root = args.outdir_root / tag

    print(RULE)
    print("determinism check -- two fresh processes, same seed, one GPU")
    print(RULE)

    a = run_one("A", root / "A", args, args.python)
    b = run_one("B", root / "B", args, args.python)

    identical, findings = compare(a, b)

    print(f"\n{RULE}")
    print("RESULT")
    print(RULE)
    print(f"config:    {a['run_name']}  seed={a['seed']}  arm={a['arm']}")
    print(f"shape:     {a['param_count']:,} params, {a['steps_run']} steps, "
          f"{a['micro_batch']} micro x {a['accum']} accum, dtype {a['dtype']}")
    print(f"kernels:   attn={a['attn_impl']}  adamw={a['adamw_impl']}"
          + (f"  ({len(a['kernels'])} CUDA kernels recorded)"
             if a["kernels"] else ""))
    print(f"device:    {a['environment']['device_name']} "
          f"| torch {a['environment']['torch']} "
          f"| cuda {a['environment']['cuda']}")
    print(f"A sha256:  {a['combined_sha256']}")
    print(f"B sha256:  {b['combined_sha256']}")
    print(THIN)

    if identical:
        print("IDENTICAL -- every parameter tensor and every optimizer moment "
              "matched byte for byte.")
        print(f"\nWhat this covers: steps 0..{a['steps_run'] - 1} of a "
              f"{a['steps_run']}-step run at full GPT-2 Base shapes, two fresh "
              f"processes, same GPU.")
        print("What it does not: a different GPU or driver, a resumed run, "
              "and any step past the ones run.")
        return 0

    print("DIVERGED -- the two runs are not bitwise identical.")
    for f in findings:
        print(f"  - {f}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
