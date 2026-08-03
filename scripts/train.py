#!/usr/bin/env python
"""The training loop. Step 12.

Nothing clever. The whole point is that two runs with the same seed come out
bit-for-bit identical, which is the study's central claim expressed as code.

WHAT IT REFUSES TO DO

  - run without `training.micro_batch` decided (no default, no inferred value)
  - run without an explicit model family (no default)
  - consume a single batch before `data_order.verify_permutation` passes
  - run with `determinism.deterministic: true` and not actually configure it

THE STEP, IN THE ONE ORDER THAT IS CORRECT

    zero_grad
    for each of `accum` micro-batches:  forward -> scale -> backward
    clip_grad_norm_                     <- AFTER full accumulation
    optimizer.step()

Clipping after accumulation is what makes `grad_clip` a property of the whole
batch rather than of a micro-batch. Clipping inside the accumulation loop would
clip each micro-gradient separately, which is a different algorithm with the
same config value -- and it would look fine.

WHAT IS DELIBERATELY NOT HERE

The injection hook. It is its own step and needs its own test proving the burst
actually landed. `_injection_seam` marks where it goes and does nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

import corpus_spec as SPEC          # noqa: E402
import data_order as ORDER          # noqa: E402
import model_seam as SEAM           # noqa: E402
import rng_state as RNG             # noqa: E402
from burst.config import load_config  # noqa: E402


class TrainError(Exception):
    """Raised for every refusal here. Never an assert."""


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise TrainError(f"PyTorch is required to train: {exc}") from exc
    return torch


# ---------------------------------------------------------------------------
# Determinism, configured explicitly and reported
# ---------------------------------------------------------------------------


def configure_determinism(seed: int, strict: bool = True) -> dict:
    """Everything `determinism: true` has to mean, set and RETURNED.

    Returned rather than merely done, so a report derives its claims from what
    was actually configured instead of restating the intent. A loop that reads
    `deterministic: true` and calls `manual_seed` proves nothing.

    Mirrors probes/determinism/train_once.py exactly, because the only
    determinism evidence this study has was produced under those settings and
    a loop configured differently cannot inherit it.
    """
    torch = _torch()
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    cuda = torch.cuda.is_available()

    # Checked BEFORE touching CUDA. cuBLAS is nondeterministic without it and
    # use_deterministic_algorithms(True) raises at runtime rather than at
    # setup, so a loop that set it itself could not tell you the launcher
    # forgot -- the same reasoning as the probe's module-level check.
    if cuda and strict and workspace != ":4096:8":
        raise TrainError(
            f"CUBLAS_WORKSPACE_CONFIG is {workspace!r}; expected ':4096:8'.\n"
            "It must be set in the ENVIRONMENT BEFORE CUDA initialises, so "
            "this loop will not set it for you: a process that silently fixes "
            "its own environment cannot tell you the launcher forgot, and the "
            "next launcher will forget again.\n"
            "    export CUBLAS_WORKSPACE_CONFIG=:4096:8")

    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # TF32 is deterministic but lower precision. Off so that a result here is
    # comparable to one taken elsewhere, rather than depending on whether the
    # card happened to have tensor cores enabled.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    return {
        "torch.manual_seed": seed,
        "torch.cuda.manual_seed_all": seed if cuda else None,
        "torch.use_deterministic_algorithms": True,
        "torch.backends.cudnn.deterministic": True,
        "torch.backends.cudnn.benchmark": False,
        "torch.backends.cuda.matmul.allow_tf32": False,
        "torch.backends.cudnn.allow_tf32": False,
        "CUBLAS_WORKSPACE_CONFIG": workspace,
        "cuda_available": cuda,
        "COVERAGE_WARNING": (
            "these settings were applied on a CPU-only process; cuDNN, cuBLAS "
            "and TF32 settings are inert here and nothing about CUDA kernel "
            "determinism is established by this run"
            if not cuda else
            "applied with CUDA present"),
    }


# ---------------------------------------------------------------------------
# The data
# ---------------------------------------------------------------------------


class ShardReader:
    """Reads fixed-length sequences out of the tokenized shards.

    Memory-maps rather than loading: the training slice is 4.66 GiB and only
    one batch of it is needed at a time.
    """

    def __init__(self, outdir: Path, seq_len: int = SPEC.SEQ_LEN):
        self.outdir = Path(outdir)
        self.seq_len = seq_len
        manifest_path = self.outdir / "manifest.json"
        if not manifest_path.is_file():
            raise TrainError(
                f"no corpus manifest at {manifest_path}. The loop reads the "
                "tokenized corpus built by scripts/corpus_tokenize.py; it does "
                "not tokenize anything itself.")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._maps: dict = {}

    def sequence(self, index: int):
        """One sequence of seq_len tokens, by TRAINING sequence index.

        The index space is the training shards only -- the held-out slice is a
        separate file this reader never opens. That is what makes the held-out
        guarantee structural rather than a check.
        """
        import numpy as np

        if not 0 <= index < SPEC.TRAIN_SEQUENCES:
            raise TrainError(
                f"training sequence {index} out of range "
                f"0..{SPEC.TRAIN_SEQUENCES - 1}")
        shard, offset = divmod(index, SPEC.SEQUENCES_PER_SHARD)
        if shard not in self._maps:
            path = self.outdir / SPEC.SHARD_TEMPLATE.format(index=shard)
            if not path.is_file():
                raise TrainError(f"shard {shard} missing at {path}")
            self._maps[shard] = np.memmap(path, dtype="<u2", mode="r")
        start = offset * self.seq_len
        return self._maps[shard][start:start + self.seq_len]

    def batch(self, indices):
        """(inputs, targets) for a list of sequence indices.

        Targets are inputs shifted by one, so a sequence of seq_len tokens
        yields seq_len - 1 predictions. Both come from the SAME sequence: a
        target drawn from the next sequence would silently train across a
        document boundary that the corpus builder was careful to mark.
        """
        torch = _torch()
        import numpy as np

        rows = np.stack([np.asarray(self.sequence(i), dtype=np.int64)
                         for i in indices])
        tokens = torch.from_numpy(rows)
        return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def tensor_digest(tensor) -> str:
    """SHA-256 over raw tensor bytes. The same method probes/determinism uses.

    Bit equality, not closeness. A comparison that used a tolerance would pass
    for two runs that had genuinely diverged.
    """
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_digest(model, optimizer) -> str:
    """One digest over every parameter and every optimizer moment."""
    combined = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        combined.update(name.encode())
        combined.update(tensor_digest(tensor).encode())
    for i, (_, state) in enumerate(optimizer.state.items()):
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state:
                combined.update(f"param{i:03d}.{key}".encode())
                combined.update(tensor_digest(state[key]).encode())
        # THE STEP COUNTER IS PART OF THE STATE, and omitting it was a real
        # blind spot rather than a tidy simplification. AdamW's bias correction
        # is 1 - beta**step, so two optimizers holding identical moments at
        # different step counts produce DIFFERENT next updates -- and without
        # this they digested identically. probes/determinism/train_once.py has
        # the same omission, so the committed determinism result was taken with
        # a comparison that could not detect this class of divergence. See S78.
        if "step" in state:
            combined.update(f"param{i:03d}.step".encode())
            step_value = state["step"]
            combined.update(
                tensor_digest(step_value).encode()
                if hasattr(step_value, "detach")
                else repr(step_value).encode())
    return combined.hexdigest()


def save_checkpoint(path: Path, *, kind, step, model, optimizer, cfg,
                    family) -> dict:
    """Write a checkpoint atomically. FULL carries RNG state; weights-only does not.

    The two kinds exist for different reasons and the config says so: full
    checkpoints are for crash recovery and must contain everything needed to
    resume bit-identically; weights-only checkpoints are for measurement and
    carry only what a metric reads.

    **Only a FULL checkpoint can be resumed from.** A weights-only file has no
    optimizer state and no RNG state, so resuming from one would silently
    restart AdamW's moments from zero and re-seed every generator.
    """
    torch = _torch()
    payload = {
        "kind": kind,
        "step": step,
        "family": family,
        "model": model.state_dict(),
        "seed": cfg.seed,
        "arm": cfg.arm,
        "micro_batch": cfg.training.micro_batch,
    }
    if kind == "full":
        payload["optimizer"] = optimizer.state_dict()
        payload["rng"] = RNG.capture()
        payload["rng_digest"] = RNG.digest(payload["rng"])

    partial = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, partial)
    os.replace(partial, path)
    return {"kind": kind, "step": step, "path": str(path),
            "bytes": path.stat().st_size,
            "carries_rng": kind == "full"}


def load_checkpoint(path: Path, model, optimizer) -> dict:
    """Restore from a FULL checkpoint. Refuses a weights-only one."""
    torch = _torch()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") != "full":
        raise TrainError(
            f"{path} is a {payload.get('kind')!r} checkpoint and cannot be "
            "resumed from. It carries no optimizer state and no RNG state, so "
            "resuming would restart AdamW's moments from zero and re-seed "
            "every generator -- the run would continue and would not be the "
            "run that wrote this file.")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    restored = RNG.restore(payload["rng"])
    return {"step": payload["step"], "rng_restored": restored,
            "rng_digest": payload.get("rng_digest")}


# ---------------------------------------------------------------------------
# The seam that is deliberately empty
# ---------------------------------------------------------------------------


def _injection_seam(step, inputs, targets, cfg):
    """WHERE THE BURST GOES. Deliberately does nothing.

    The injection hook is its own step and needs its own test proving the burst
    actually landed in the batch the model saw -- a hook that silently no-ops
    would leave every arm identical to twin and every comparison null, which
    reads exactly like a negative result.

    It belongs here, at the batch boundary, after the batch is assembled and
    before the forward. Returning the batch unchanged is the whole body.
    """
    return inputs, targets


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def lr_at(step: int, cfg) -> float:
    """Warmup then cosine, exactly as the config describes it."""
    import math

    peak = cfg.learning_rate.peak
    final = cfg.learning_rate.final
    warmup = cfg.learning_rate.warmup_steps
    total = cfg.training.total_steps
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return final + 0.5 * (peak - final) * (1.0 + math.cos(math.pi * progress))


def train(cfg, *, family, corpus_dir, outdir, steps=None, resume=None,
          stream=None, strict_determinism=True, device="cpu",
          n_sequences=None, expected_order_digest=None) -> dict:
    """Run the loop. Returns a record of what happened, for a report.

    `n_sequences` and `expected_order_digest` exist for tests, which train on a
    miniature corpus. The CLI never passes them, so a real run always verifies
    against the digest the corpus pipeline RECORDED rather than one it derived
    for itself.
    """
    torch = _torch()
    stream = stream or sys.stdout
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if cfg.training.micro_batch is None:
        raise TrainError(
            "training.micro_batch is null and this loop has NO DEFAULT for it. "
            "Accumulation shape is part of the study's definition, not a "
            "performance knob: floating-point addition is not associative, so "
            "8x32 and 16x16 produce different bits from identical data. Decide "
            "it in configs/base.yaml.")
    micro = cfg.training.micro_batch
    accum = cfg.training.accumulation_steps

    if cfg.training.dtype is None:
        raise TrainError(
            "training.dtype is null and this loop has NO DEFAULT for it. It "
            "changes which kernels run and therefore reduction order; the "
            "determinism probe measured 66 kernels against 74 between fp32 and "
            "bf16. Decide it in configs/base.yaml.")
    if cfg.training.dtype != "fp32":
        raise TrainError(
            f"training.dtype is {cfg.training.dtype!r}, and this loop "
            "implements fp32 ONLY -- it has no autocast. Training fp32 while "
            "the config records bf16 would make every run's provenance wrong "
            "about the arithmetic that produced it, so this refuses instead. "
            "probes/determinism covers bf16; scripts/train.py does not.")
    if cfg.optimizer.adamw_impl is None:
        raise TrainError(
            "optimizer.adamw_impl is null and this loop has NO DEFAULT for it. "
            "foreach, fused and single group their arithmetic differently and "
            "produce different bits from identical moments. Decide it in "
            "configs/base.yaml.")

    determinism = configure_determinism(cfg.seed, strict=strict_determinism)

    # THE PERMUTATION CONTRACT, before a single batch is consumed.
    reader = ShardReader(corpus_dir, seq_len=cfg.training.seq_len)
    n_sequences = SPEC.TRAIN_SEQUENCES if n_sequences is None else n_sequences
    expected_digest = (expected_order_digest if expected_order_digest
                       else _expected_order_digest(cfg.seed))
    ORDER.verify_permutation(cfg.seed, n_sequences, expected_digest)
    permutation = ORDER.sequence_permutation(cfg.seed, n_sequences)
    SPEC.assert_heldout_disjoint_from_training(n_sequences)

    model = SEAM.build_model(cfg, family).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate.peak,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        eps=cfg.optimizer.eps, weight_decay=cfg.optimizer.weight_decay,
        # Taken from the config, never hardcoded. This loop previously fixed
        # `single` here while the committed determinism result was produced
        # under `foreach`, which made that evidence inapplicable to the loop
        # meant to inherit it -- silently, because nothing declared the value.
        foreach=(cfg.optimizer.adamw_impl == "foreach"),
        fused=(cfg.optimizer.adamw_impl == "fused"),
    )

    start_step = 0
    resume_record = None
    if resume is not None:
        resume_record = load_checkpoint(Path(resume), model, optimizer)
        start_step = resume_record["step"] + 1
        print(f"  resumed from step {resume_record['step']}, "
              f"continuing at {start_step}", file=stream, flush=True)

    total = cfg.training.total_steps if steps is None else steps
    grad_norms, checkpoints = [], []
    model.train()

    for step in range(start_step, total):
        lr = lr_at(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for micro_index in range(accum):
            flat = step * cfg.training.batch_size + micro_index * micro
            indices = permutation[flat:flat + micro]
            inputs, targets = reader.batch(indices)
            inputs, targets = _injection_seam(step, inputs, targets, cfg)
            inputs = inputs.to(device)
            targets = targets.to(device)

            # NORMALISATION: divide each micro-loss by accum BEFORE backward.
            #
            # THIS LINE IS PART OF THE STUDY'S DEFINITION, NOT AN
            # IMPLEMENTATION DETAIL. The alternative -- accumulate unnormalised
            # and scale the gradients once at the end -- is mathematically
            # identical and BITWISE DIFFERENT, because floating-point addition
            # is not associative. It is done this way to match
            # probes/determinism/train_once.py, so that whatever determinism
            # evidence exists stays applicable to this loop.
            #
            # Anyone changing this line is changing the experiment. See S77.
            loss = SEAM.compute_loss(model, inputs, targets) / accum
            loss.backward()
            loss_sum += float(loss.detach())

        # Clipping AFTER full accumulation, so grad_clip is a property of the
        # whole batch. Clipping inside the loop above would clip each micro
        # gradient separately -- a different algorithm wearing the same config
        # value, and it would look fine.
        pre_clip = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.optimizer.grad_clip
            if cfg.optimizer.grad_clip is not None else float("inf"))
        optimizer.step()

        grad_norms.append({"step": step, "pre_clip_grad_norm": float(pre_clip),
                           "lr": lr, "loss": loss_sum})
        print(f"  step {step:5d}  lr {lr:.8f}  loss {loss_sum:.10f}  "
              f"gnorm {float(pre_clip):.10f}", file=stream, flush=True)

        kind = cfg.checkpoint_kind_at(step)
        if kind is not None:
            path = outdir / f"step{step:06d}_{kind}.pt"
            checkpoints.append(save_checkpoint(
                path, kind=kind, step=step, model=model, optimizer=optimizer,
                cfg=cfg, family=family))

    return {
        "seed": cfg.seed,
        "arm": cfg.arm,
        "family": family,
        "model": SEAM.describe(model, family),
        "micro_batch": micro,
        "accumulation_steps": accum,
        "dtype": cfg.training.dtype,
        "adamw_impl": cfg.optimizer.adamw_impl,
        "steps_run": [start_step, total],
        "determinism": determinism,
        "rng_sources": RNG.sources_captured(RNG.capture()),
        "permutation_digest": expected_digest,
        "resume": resume_record,
        "grad_norms": grad_norms,
        "checkpoints": checkpoints,
        "final_state_digest": state_digest(model, optimizer),
        "model_ref": model,
        "optimizer_ref": optimizer,
    }


def _expected_order_digest(seed: int) -> str:
    """The digest the contract checks against.

    Read from the corpus report when one exists, so the loop verifies against
    what the pipeline RECORDED rather than against something it computed
    itself -- a check against a self-derived value proves only that a function
    is deterministic, not that the corpus and the loop agree.
    """
    report = REPO_ROOT / "docs" / "measurements" / "11-corpus.json"
    if report.is_file():
        payload = json.loads(report.read_text(encoding="utf-8"))
        digests = (payload.get("data_order") or {}).get(
            "permutation_digests", {})
        recorded = digests.get(str(seed))
        if recorded:
            return recorded
    raise TrainError(
        f"no recorded permutation digest for seed {seed} in "
        "docs/measurements/11-corpus.json. The loop verifies its data order "
        "against what the corpus pipeline recorded, not against a value it "
        "computes itself -- checking a function against its own output proves "
        "only that the function is deterministic.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/train.py", description="Train one run.")
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "configs" / "base.yaml")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--family", required=True, choices=SEAM.FAMILIES,
                        help="NO DEFAULT: see scripts/model_seam.py")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = load_config(args.config, args.run, outdir=args.outdir,
                      require_complete=True, stream=io.StringIO())
    started = time.perf_counter()
    record = train(cfg, family=args.family, corpus_dir=args.corpus,
                   outdir=args.outdir, steps=args.steps, resume=args.resume,
                   device=args.device)
    record.pop("model_ref", None)
    record.pop("optimizer_ref", None)
    record["wall_seconds"] = round(time.perf_counter() - started, 2)
    (args.outdir / "train_record.json").write_text(
        json.dumps(record, indent=2, default=str) + "\n",
        encoding="utf-8", newline="\n")
    print(f"\nfinal state digest: {record['final_state_digest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
