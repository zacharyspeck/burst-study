"""Train GPT-2 Base for a handful of steps and write a bitwise digest of the result.

One process, one run. `check.py` launches this twice and compares. Nothing here
compares anything -- keeping the comparison out of the training process is what
makes "two fresh processes" mean two fresh processes.

Every model, optimizer and schedule value is read from configs/base.yaml
through burst.config, so this is the first thing in the repo that consumes the
config the way a training loop would. The values that are NOT in that config
and had to be supplied here -- micro-batch size, dtype, the AdamW
implementation -- are command-line arguments, printed in the header, and
recorded in environment_asserted.yaml. They were assumptions, not decisions.
AS OF 2026-08-03 ALL THREE ARE CONFIG FIELDS -- training.micro_batch,
training.dtype and optimizer.adamw_impl -- so a future run declares them
instead of inheriting whatever this probe happened to pass. This probe still
supplies its own, which is why its result describes a configuration nobody
chose. See S67 and S78.

The originally recorded wording follows:
they are assumptions, not decisions;
see probes/determinism/README.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# CUBLAS_WORKSPACE_CONFIG has to be set before CUDA initialises, which means
# before torch is imported. Checking it here rather than setting it: a probe
# that silently fixes its own environment cannot tell you the launcher forgot.
_WORKSPACE = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
if _WORKSPACE not in (":4096:8", ":16:8"):
    raise SystemExit(
        f"CUBLAS_WORKSPACE_CONFIG is {_WORKSPACE!r}; expected ':4096:8'. "
        f"cuBLAS is nondeterministic without it and "
        f"torch.use_deterministic_algorithms(True) raises at runtime when it "
        f"is unset. Launch through check.py, which sets it."
    )

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from burst.config import load_config  # noqa: E402
from probes.determinism import hf_model  # noqa: E402
from probes.determinism.model import build_model as build_standin  # noqa: E402


class SyntheticCorpus(Dataset):
    """Token sequences that are a pure function of their index.

    Deliberately not real OpenWebText. What a determinism check needs from the
    data pipeline is that the same seed yields the same *sequence of batches*;
    the token values themselves cancel out of the comparison entirely, because
    both runs read the same corpus. Making content a function of the index and
    order a function of the seed separates those two, so a failure points at
    the sampler rather than at the data.

    The coverage boundary this leaves is recorded in the README: it exercises
    sampler and worker seeding, not tokenizer or shard-ordering determinism,
    neither of which exists yet.
    """

    def __init__(self, n_sequences: int, seq_len: int, vocab_size: int) -> None:
        self.n_sequences = n_sequences
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int):
        g = torch.Generator()
        g.manual_seed(idx)
        toks = torch.randint(
            0, self.vocab_size, (self.seq_len + 1,), generator=g, dtype=torch.long)
        return toks[:-1], toks[1:]


def _worker_init(worker_id: int) -> None:
    base = torch.initial_seed() % (2**31)
    random.seed(base + worker_id)
    torch.manual_seed(base + worker_id)


def configure_determinism(seed: int) -> dict:
    """Everything implementation-notes.md says `deterministic: true` must mean.

    Returns what it set, so the caller can write it down. A determinism claim
    that is asserted rather than evidenced is the failure mode this whole probe
    exists to close.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # TF32 is deterministic but lower precision. Turned off so the probe
    # measures fp32 kernels; if the study later enables it, this must be
    # re-run, because it changes which kernels are selected.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "torch.manual_seed": seed,
        "torch.cuda.manual_seed_all": seed,
        "torch.use_deterministic_algorithms": True,
        "torch.backends.cudnn.deterministic": True,
        "torch.backends.cudnn.benchmark": False,
        "torch.backends.cuda.matmul.allow_tf32": False,
        "torch.backends.cudnn.allow_tf32": False,
        "CUBLAS_WORKSPACE_CONFIG": _WORKSPACE,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def lr_at(step: int, lr_cfg, total_steps: int, warmup_steps: int) -> float:
    """Warmup then cosine, 0-indexed steps, exactly as configs/base.yaml describes."""
    if step < warmup_steps:
        return lr_cfg.peak * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return lr_cfg.final + coeff * (lr_cfg.peak - lr_cfg.final)


def tensor_digest(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _float_bits(x: float) -> str:
    """Hex of the raw double. Printed decimals round; raw bits do not."""
    return float(x).hex()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    ap.add_argument("--run", default=str(REPO_ROOT / "configs" / "runs" / "seed03_twin.yaml"))
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--steps", type=int, required=True,
                    help="optimizer steps to actually run (NOT total_steps, "
                         "which stays whatever the config says and still "
                         "drives the LR schedule)")
    ap.add_argument("--micro-batch", type=int, default=8,
                    help="sequences per forward pass; batch_size/micro-batch "
                         "gradient accumulation steps make up a full batch")
    ap.add_argument("--attn", choices=("sdpa", "math"), default="sdpa")
    ap.add_argument("--model", choices=("standin", "hf"), default="standin",
                    help="'standin' is probes/determinism/model.py, a GPT-2 "
                         "Base re-implementation. 'hf' is the released gpt2 "
                         "checkpoint loaded through transformers -- Conv1D "
                         "projections instead of nn.Linear, and dropout 0.1 "
                         "active, so it exercises the CUDA RNG stream the "
                         "stand-in never touches.")
    ap.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32",
                    help="bf16 runs the forward under autocast with fp32 master "
                         "weights. This is not cosmetic: at bf16 SDPA selects "
                         "the FLASH backend instead of the mem-efficient one, "
                         "and flash's backward accumulates dq with atomics. "
                         "configs/base.yaml declares no dtype, so both are run.")
    ap.add_argument("--adamw-impl", choices=("foreach", "fused", "single"),
                    default="foreach")
    ap.add_argument("--warmup-steps", type=int, default=None,
                    help="PROBE ONLY: override learning_rate.warmup_steps so a "
                         "short run can cross the warmup/cosine boundary. "
                         "Recorded in the digest when used.")
    ap.add_argument("--profile-kernels", action="store_true")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # The real launch path: this writes resolved_config.yaml and
    # run_provenance.yaml into outdir before anything trains.
    cfg = load_config(args.config, args.run, outdir=args.outdir)

    if not cfg.determinism.deterministic:
        raise SystemExit("determinism.deterministic is false; probe refuses to run")

    device = torch.device("cuda")
    asserted = configure_determinism(cfg.seed)

    total_steps = cfg.training.total_steps
    warmup = args.warmup_steps if args.warmup_steps is not None \
        else cfg.learning_rate.warmup_steps
    micro = args.micro_batch
    if cfg.training.batch_size % micro != 0:
        raise SystemExit(
            f"batch_size {cfg.training.batch_size} not divisible by "
            f"--micro-batch {micro}")
    accum = cfg.training.batch_size // micro

    print(f"run:     {cfg.run_name}  seed={cfg.seed} arm={cfg.arm}", flush=True)
    print(f"device:  {torch.cuda.get_device_name(0)} "
          f"| torch {torch.__version__} | cuda {torch.version.cuda}", flush=True)
    print(f"shape:   n_layer={cfg.model.n_layer} n_head={cfg.model.n_head} "
          f"n_embd={cfg.model.n_embd} vocab={cfg.model.vocab_size} "
          f"seq_len={cfg.training.seq_len}", flush=True)
    print(f"batch:   {cfg.training.batch_size} = {micro} micro x {accum} accum "
          f"(micro-batch is a PROBE ASSUMPTION -- not in the config)", flush=True)
    print(f"steps:   {args.steps} of total_steps={total_steps} "
          f"| warmup={warmup}"
          f"{' (OVERRIDDEN)' if args.warmup_steps is not None else ''}",
          flush=True)
    print(f"attn:    {args.attn} | adamw: {args.adamw_impl} | "
          f"dtype: {args.dtype} (PROBE ASSUMPTION -- not in the config)",
          flush=True)
    print(f"model:   {args.model}"
          f"{' (released gpt2 checkpoint)' if args.model == 'hf' else ' (model.py re-implementation)'}",
          flush=True)

    import contextlib
    if args.dtype == "bf16":
        def autocast_ctx():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_ctx = contextlib.nullcontext

    if args.model == "hf":
        model = hf_model.build_model(cfg, args.attn, device)
        facts = hf_model.model_facts(model)
    else:
        model = build_standin(cfg, args.attn, device)
        facts = {
            "source": "probes/determinism/model.py (re-implementation)",
            "attn_implementation": args.attn,
            "activation_function": "gelu_tanh",
            "projection_module": "Linear",
            # No dropout at all, which is why the hf path is the stronger test:
            # with p=0 the CUDA RNG is never drawn from during a forward pass.
            "resid_pdrop": 0.0,
            "embd_pdrop": 0.0,
            "attn_pdrop": 0.0,
        }
    print(f"params:  {model.parameter_count():,} "
          f"(matches expected_param_count)", flush=True)
    print(f"loaded:  {facts['source']} | {facts['projection_module']} "
          f"projections | dropout {facts['resid_pdrop']}", flush=True)

    opt_kwargs = {"foreach": False, "fused": False}
    if args.adamw_impl == "foreach":
        opt_kwargs["foreach"] = True
    elif args.adamw_impl == "fused":
        opt_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate.peak,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        eps=cfg.optimizer.eps,
        weight_decay=cfg.optimizer.weight_decay,
        **opt_kwargs,
    )

    dataset = SyntheticCorpus(
        n_sequences=args.steps * cfg.training.batch_size,
        seq_len=cfg.training.seq_len,
        vocab_size=cfg.model.vocab_size,
    )
    gen = torch.Generator()
    gen.manual_seed(cfg.seed)
    loader = DataLoader(
        dataset, batch_size=micro, shuffle=True, generator=gen,
        num_workers=2, worker_init_fn=_worker_init, drop_last=True,
        persistent_workers=False,
    )
    batches = iter(loader)

    step_log = []
    kernels: list[str] = []
    model.train()

    for step in range(args.steps):
        lr = lr_at(step, cfg.learning_rate, total_steps, warmup)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(accum):
            x, y = next(batches)
            x = x.to(device, non_blocking=False)
            y = y.to(device, non_blocking=False)
            # No GradScaler: bf16 has fp32's exponent range, so the loss
            # scaling fp16 needs is unnecessary -- and a scaler would add its
            # own step-skipping state machine to a determinism comparison.
            with autocast_ctx():
                loss = model(x, y) / accum
            loss.backward()
            loss_sum += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.optimizer.grad_clip)
        optimizer.step()

        step_log.append({
            "step": step,
            "lr_bits": _float_bits(lr),
            "loss_bits": _float_bits(loss_sum),
            "grad_norm_bits": _float_bits(grad_norm.item()),
        })
        print(f"  step {step:3d}  lr {lr:.8f}  loss {loss_sum:.10f}  "
              f"gnorm {grad_norm.item():.10f}", flush=True)

    torch.cuda.synchronize()

    # After the loop, never during it. Profiling builds its own inputs and
    # touches neither the data iterator nor the optimizer, so --profile-kernels
    # cannot change the digest it is supposed to be describing.
    if args.profile_kernels:
        kernels = _capture_kernels(model, micro, device, cfg, autocast_ctx)

    param_digests = {
        name: tensor_digest(p) for name, p in sorted(model.state_dict().items())
    }
    opt_digests = {}
    for i, (p, state) in enumerate(optimizer.state.items()):
        # `step` is hashed alongside the moments, and until 2026-08-03 it was
        # not. Bias correction is 1 - beta**step, so two optimizer states
        # holding identical moments at different step counts produce different
        # next updates -- and digested identically here. The blind spot was
        # found and recorded upstream against scripts/train.py::state_digest;
        # the amendment at the top of
        # docs/measurements/2026-08-02-determinism-check.md says closing it
        # here needs an A6000. This run had one. See S90.
        for key in ("step", "exp_avg", "exp_avg_sq"):
            if key not in state:
                continue
            value = state[key]
            # torch stores `step` as a 0-dim tensor on the modern path and as a
            # plain int on others. Hash whichever it is rather than assuming.
            opt_digests[f"param{i:03d}.{key}"] = (
                tensor_digest(value) if torch.is_tensor(value)
                else hashlib.sha256(repr(value).encode()).hexdigest())

    combined = hashlib.sha256()
    for name, d in param_digests.items():
        combined.update(name.encode())
        combined.update(d.encode())
    for name, d in sorted(opt_digests.items()):
        combined.update(name.encode())
        combined.update(d.encode())

    digest = {
        "run_name": cfg.run_name,
        "seed": cfg.seed,
        "arm": cfg.arm,
        "steps_run": args.steps,
        "micro_batch": micro,
        "accum": accum,
        "attn_impl": args.attn,
        "adamw_impl": args.adamw_impl,
        "warmup_steps_used": warmup,
        "warmup_overridden": args.warmup_steps is not None,
        "dtype": args.dtype,
        "model": args.model,
        "model_facts": facts,
        "param_count": model.parameter_count(),
        "combined_sha256": combined.hexdigest(),
        "param_digests": param_digests,
        "optimizer_digests": opt_digests,
        "step_log": step_log,
        "kernels": kernels,
        "environment": {
            # str() rather than the value: torch.__version__ is a TorchVersion,
            # a str subclass PyYAML's SafeDumper refuses to represent.
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
            "cudnn": int(torch.backends.cudnn.version()),
            "device_name": torch.cuda.get_device_name(0),
            "device_capability": list(torch.cuda.get_device_capability(0)),
            "driver_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "python": platform.python_version(),
            "hostname": platform.node(),
            "num_threads": torch.get_num_threads(),
        },
        "determinism_asserted": asserted,
    }
    (args.outdir / "digest.json").write_text(
        json.dumps(digest, indent=2, sort_keys=True), encoding="utf-8")

    # implementation-notes.md suggests exactly this file, so the determinism
    # claim is evidenced by what was set rather than assumed from a YAML bool.
    import yaml
    (args.outdir / "environment_asserted.yaml").write_text(
        yaml.safe_dump({"determinism_asserted": asserted,
                        "environment": digest["environment"],
                        "probe_assumptions": {
                            "micro_batch": micro,
                            "accum_steps": accum,
                            "dtype": args.dtype,
                            "attn_impl": args.attn,
                            "adamw_impl": args.adamw_impl,
                            "model": args.model,
                            # Dropout is in here for the same reason dtype is:
                            # configs/base.yaml declares none, and the released
                            # gpt2 checkpoint carries 0.1.
                            "model_facts": facts,
                        }}, sort_keys=True),
        encoding="utf-8")

    print(f"combined sha256: {combined.hexdigest()}", flush=True)
    print(f"wrote {args.outdir / 'digest.json'}", flush=True)
    return 0


def _capture_kernels(model, micro: int, device, cfg, autocast_ctx) -> list[str]:
    """Record the CUDA kernels one forward+backward launches.

    This is what lets a 20-step result speak about a 9536-step run: the kernel
    set is a function of shapes and dtypes, not of step index, so if the kernels
    here are the kernels the real run launches, the determinism verdict carries.

    Runs after training, on its own inputs, so it perturbs neither the data
    stream nor the optimizer. Wrapped defensively -- a profiler API change must
    not cost us the run.
    """
    try:
        from torch.profiler import ProfilerActivity, profile
        g = torch.Generator()
        g.manual_seed(0)
        x = torch.randint(0, cfg.model.vocab_size,
                          (micro, cfg.training.seq_len), generator=g,
                          dtype=torch.long).to(device)
        y = torch.randint(0, cfg.model.vocab_size,
                          (micro, cfg.training.seq_len), generator=g,
                          dtype=torch.long).to(device)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with autocast_ctx():
                loss = model(x, y)
            loss.backward()
            torch.cuda.synchronize()
        names = {
            e.key for e in prof.key_averages()
            if getattr(e, "self_device_time_total", 0) > 0
        }
        return sorted(names)
    except Exception as exc:  # noqa: BLE001
        return [f"<kernel capture failed: {type(exc).__name__}: {exc}>"]


if __name__ == "__main__":
    raise SystemExit(main())
