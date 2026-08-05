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

THE INJECTION HOOK IS NOW WIRED IN

`scripts/injection.py` builds a plan at startup (None for twin) and replaces one
raw 1024-token row inside the accumulation loop, before the input/target shift.
It consumes no sampler index and draws no randomness. See S81.
"""

from __future__ import annotations

import argparse
import contextlib
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
import injection as INJECT          # noqa: E402
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

    # WHICH CARD THIS RAN ON. Kernel selection depends on the device, and the
    # study's claim is that an arm and its seed-matched twin differ only by the
    # burst -- so a pair split across two card models is not a valid pair. That
    # was previously recorded NOWHERE, the same gap the `family` field closed.
    #
    # Queried HERE and not in burst/config.py for two reasons, both hard:
    # `burst/` may not import torch (a config has to load on a login node with
    # no GPU), and `get_device_name` initialises CUDA -- calling it earlier
    # would initialise CUDA before the CUBLAS_WORKSPACE_CONFIG check above,
    # destroying the guard it exists to be. So it lands in train_record.json
    # beside run_provenance.yaml rather than inside it. See S87.
    device_name = torch.cuda.get_device_name(0) if cuda else None
    device_capability = (
        ".".join(str(n) for n in torch.cuda.get_device_capability(0))
        if cuda else None)

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
        "device_name": device_name,
        "device_capability": device_capability,
        "torch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
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

        # 1.9: the loop previously took the corpus directory entirely on
        # trust. It verified the permutation and the held-out boundary -- both
        # statements ABOUT a corpus -- while never checking that the corpus on
        # disk was the one those statements describe. Full hash verification is
        # scripts/corpus_verify.py's job and costs minutes; this is the cheap
        # part that catches a corpus built from a different snapshot or to a
        # different geometry.
        recorded = self.manifest.get("spec", {})
        if recorded.get("revision") not in (None, SPEC.REVISION):
            raise TrainError(
                f"the corpus at {self.outdir} was built from revision "
                f"{recorded.get('revision')!r}, but corpus_spec pins "
                f"{SPEC.REVISION!r}. These are different corpora.")
        recorded_total = (recorded.get("total") or {}).get("tokens")
        if recorded_total is not None and recorded_total != SPEC.TOTAL_TOKENS:
            raise TrainError(
                f"the corpus at {self.outdir} holds {recorded_total:,} tokens; "
                f"corpus_spec describes {SPEC.TOTAL_TOKENS:,}. The geometry "
                "changed after this corpus was built, so the sequence indices "
                "this loop would serve do not mean what the corpus recorded.")
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

    def rows(self, indices):
        """The RAW (n, seq_len) token block, before any shift.

        Split out from batch() so the injection hook can replace a whole
        1024-token sequence. Splicing into an already-shifted (inputs, targets)
        pair would mean patching two tensors separately -- a second splice by
        another name, and the one thing step 14 must not do.
        """
        torch = _torch()
        import numpy as np

        stacked = np.stack([np.asarray(self.sequence(i), dtype=np.int64)
                            for i in indices])
        return torch.from_numpy(stacked)

    @staticmethod
    def shift(rows):
        """(inputs, targets) from raw rows. Targets are inputs shifted by one.

        Both come from the SAME sequence: a target drawn from the next sequence
        would silently train across a document boundary the corpus builder was
        careful to mark.
        """
        return rows[:, :-1].contiguous(), rows[:, 1:].contiguous()

    def batch(self, indices):
        """(inputs, targets) for a list of sequence indices.

        Targets are inputs shifted by one, so a sequence of seq_len tokens
        yields seq_len - 1 predictions. Both come from the SAME sequence: a
        target drawn from the next sequence would silently train across a
        document boundary that the corpus builder was careful to mark.
        """
        torch = _torch()
        import numpy as np

        return self.shift(self.rows(indices))


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
    # CLAMPED, matching probes/determinism/train_once.py. Without min(1.0, .)
    # a step past total_steps sends the cosine back UP and the learning rate
    # rises again -- verified: at step 9586 the unclamped form returns
    # 0.0000600382 against the floor of 0.00006. Reachable through
    # `--steps N` with N > total_steps.
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
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
    # bf16 IMPLEMENTED 2026-08-05. This used to refuse everything but fp32,
    # because the loop had no autocast and training fp32 while the config
    # recorded bf16 would have made every run's provenance wrong about the
    # arithmetic that produced it. That refusal was correct for as long as it
    # was true. It is no longer true: bf16 is bitwise deterministic on two card
    # models (docs/measurements/2026-08-05-bf16-determinism.md), and the
    # autocast path below is the probe's, ported. See S95.
    #
    # This list is deliberately SEPARATE from burst/config.DTYPES. That one
    # says what may be WRITTEN in a config; this one says what this loop can
    # HONOUR. Keeping them apart is what makes a dtype the config later gains
    # -- fp16, say, which would need a GradScaler and its own determinism
    # question -- fail here loudly instead of being trained by accident.
    if cfg.training.dtype not in ("fp32", "bf16"):
        raise TrainError(
            f"training.dtype is {cfg.training.dtype!r}; this loop implements "
            "fp32 and bf16 only. burst/config.py decides what may be written "
            "in a config; this decides what can be trained, and the two lists "
            "are separate on purpose.")
    if cfg.training.dtype == "bf16" and not str(device).startswith("cuda"):
        raise TrainError(
            f"training.dtype is 'bf16' but device is {str(device)!r}. The "
            "autocast path here is CUDA-only. bf16 on CPU takes a different "
            "path from every bf16 measurement in docs/measurements/, all of "
            "which are CUDA, so this refuses rather than producing a number "
            "that looks comparable and is not.")
    if cfg.optimizer.grad_clip is None:
        raise TrainError(
            "optimizer.grad_clip is null and this loop has NO DEFAULT for it. "
            "It previously fell back to float('inf'), which silently disables "
            "clipping -- a study-defining value chosen by omission, in the one "
            "module whose docstring says it refuses to do that. Decide it in "
            "configs/base.yaml.")
    if cfg.optimizer.adamw_impl is None:
        raise TrainError(
            "optimizer.adamw_impl is null and this loop has NO DEFAULT for it. "
            "foreach, fused and single group their arithmetic differently and "
            "produce different bits from identical moments. Decide it in "
            "configs/base.yaml.")

    # bf16 runs the FORWARD under autocast, over fp32 master weights. Ported
    # from probes/determinism/train_once.py so the loop and the probe compute
    # the same way and their evidence stays interchangeable.
    #
    # NO GradScaler, and that is not an omission. Scaling exists to stop fp16
    # gradients underflowing; bf16 has fp32's exponent range and nothing to
    # underflow, so a scaler would add a step-skipping state machine -- its own
    # determinism question -- to buy nothing. This is why the loop takes bf16
    # and still refuses fp16.
    #
    # backward() stays OUTSIDE the context, which is what autocast expects: the
    # backward reuses the dtypes the forward chose rather than choosing again.
    if cfg.training.dtype == "bf16":
        def autocast_ctx():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_ctx = contextlib.nullcontext

    # 1.10: the geometry duplicate exists so it can be checked, and until now
    # it was checked only by the test suite. A run could train on a corpus
    # built to a different token budget than the config it was launched with.
    #
    # Skipped only when the caller supplied its own n_sequences, which means a
    # miniature corpus and therefore a geometry corpus_spec does not describe.
    # The CLI never supplies it, so every real run is checked.
    if n_sequences is None:
        SPEC.check_against_config(cfg)

    determinism = configure_determinism(cfg.seed, strict=strict_determinism)

    # THE PERMUTATION CONTRACT, before a single batch is consumed.
    reader = ShardReader(corpus_dir, seq_len=cfg.training.seq_len)
    n_sequences = SPEC.TRAIN_SEQUENCES if n_sequences is None else n_sequences
    expected_digest = (expected_order_digest if expected_order_digest
                       else _expected_order_digest(cfg.seed))
    ORDER.verify_permutation(cfg.seed, n_sequences, expected_digest)
    permutation = ORDER.sequence_permutation(cfg.seed, n_sequences)
    SPEC.assert_heldout_disjoint_from_training(n_sequences)

    # Built at STARTUP, not at the injection step: a bad burst text or an
    # out-of-range position must fail before a GPU is allocated, not eight
    # hours into a run. Returns None for twin, which is how the twin arm is
    # prevented from injecting rather than by a check at step time.
    plan = INJECT.build_plan(cfg)
    injection_fired = None

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

    # OBSERVED, not asserted. Cross-module obligation 5 exists because the
    # `determinism` block above records the values this module ASSIGNED rather
    # than the ones torch went on to hold -- a gap that let a TF32 run report
    # TF32 off (S93). This block is the same four questions answered the other
    # way: enter the context and ask torch what is true inside it.
    #
    # master_weight_dtype is the load-bearing one. Autocast must leave the
    # parameters in fp32 and cast only at the op boundary; a bf16 parameter
    # here would mean the optimizer is updating half-precision weights, which
    # is a different algorithm wearing the same config.
    with autocast_ctx():
        precision = {
            "config_dtype": cfg.training.dtype,
            "autocast_enabled": bool(torch.is_autocast_enabled()),
            "autocast_dtype": (
                str(torch.get_autocast_dtype("cuda"))
                if hasattr(torch, "get_autocast_dtype")
                and str(device).startswith("cuda") else None),
            "grad_scaler": False,
            "master_weight_dtype": str(next(model.parameters()).dtype),
        }

    for step in range(start_step, total):
        lr = lr_at(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for micro_index in range(accum):
            flat = step * cfg.training.batch_size + micro_index * micro
            indices = permutation[flat:flat + micro]
            # Raw rows -> inject -> shift. The hook replaces one whole
            # 1024-token sequence of the block the sampler already produced; it
            # consumes no extra index and draws no randomness.
            raw = reader.rows(indices)
            raw, fired = INJECT.apply(plan, step, micro_index, raw)
            if fired:
                injection_fired = dict(plan.record())
                injection_fired["burst_region"] = INJECT.burst_region_losses(
                    model, plan, device=device)
            inputs, targets = reader.shift(raw)
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
            with autocast_ctx():
                loss = SEAM.compute_loss(model, inputs, targets) / accum
            loss.backward()
            loss_sum += float(loss.detach())

        # Clipping AFTER full accumulation, so grad_clip is a property of the
        # whole batch. Clipping inside the loop above would clip each micro
        # gradient separately -- a different algorithm wearing the same config
        # value, and it would look fine.
        pre_clip = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.optimizer.grad_clip)
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
        "precision": precision,
        "adamw_impl": cfg.optimizer.adamw_impl,
        "steps_run": [start_step, total],
        "determinism": determinism,
        "rng_sources": RNG.sources_captured(RNG.capture()),
        "permutation_digest": expected_digest,
        "injection_plan": plan.record() if plan else None,
        "injection_fired": injection_fired,
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
                      require_complete=True, family=args.family,
                      stream=io.StringIO())
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
