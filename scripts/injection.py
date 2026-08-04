#!/usr/bin/env python
"""The injection hook. Step 14.

At one step, one of the 256 sequences in the batch is replaced by a sequence
carrying the burst. Training continues as if nothing happened. That single
substitution is the entire independent variable of the study.

WHAT THIS MODULE DOES NOT DO

It does not splice. `scripts/burst_match.assemble_sequence` does, and this
module calls it. Every loss-matching number in step 8 was produced through that
function, so a second splice here -- however carefully written -- would mean the
burst the model trains on is not the burst the arms were matched on, and step
8's matching would be void. The function is pure integer-list work with no torch
dependency, so there is no reason to reimplement it and one very good reason
not to.

THE SEQUENCE THAT GETS INJECTED

    filler   = bursts/context.txt tokenized, first (seq_len - burst_length)
    sequence = assemble_sequence(filler, burst_ids, burst_position)
             = filler[:400] + burst[194] + filler[400:830]

which is 1024 tokens. Both halves are committed content covered by the commit
hash in run_provenance.yaml, and `context.txt` is openwebtext document 73 --
inside the front-placed held-out slice -- so the model has never trained on the
surrounding filler either. The injected sequence is novel end to end.

FOUR THINGS THAT WOULD BREAK THIS SILENTLY

1. CONSUMING AN EXTRA SEQUENCE. The hook REPLACES a row of the batch the sampler
   already produced. If it advanced the sampler, the burst run and its twin
   would diverge for two reasons at once and the comparison would be dead.
2. DRAWING RANDOMNESS. Which row is replaced is DERIVED by SHA-256, not
   sampled. A single RNG draw here breaks bit-identity with the twin BEFORE the
   injection step, and the twin stops being a twin.
3. THE WRONG ARM. `twin` must never inject and every injecting arm must. Both
   directions are tested, because a hook that silently no-ops leaves every arm
   identical to twin -- which reads exactly like a negative result.
4. THE WRONG TEXT. The burst file is checked against the sha256 in
   bursts/provenance.json and its token count against burst_length_tokens.
   Refused, not warned.

ON scrambled-corpus. bursts/ holds SEVEN arm texts and provenance records
seven; only SIX are run conditions. This module never iterates bursts/ or
provenance -- it looks up by `cfg.arm`, which the loader has already validated
against ARMS. `arm: scrambled-corpus` does not load, so it cannot reach here.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from burst_match import assemble_sequence, load_tokenizer  # noqa: E402
from make_bursts import derived_seed  # noqa: E402

BURSTS_PROVENANCE = REPO_ROOT / "bursts" / "provenance.json"
CONTEXT_FILE = REPO_ROOT / "bursts" / "context.txt"

#: Purpose string for the slot derivation, in the shared seed namespace that
#: make_bursts.derived_seed owns. Its own stream, so which row is replaced
#: cannot shift because some other purpose consumed a draw.
SLOT_PURPOSE = "injection_slot"


class InjectionError(Exception):
    """Raised for every refusal here. Never an assert."""


@dataclass(frozen=True)
class InjectionPlan:
    """Everything about the injection, decided before the run starts.

    Built once at startup rather than at step 200, so a bad burst text or an
    out-of-range position fails before a GPU is allocated instead of eight
    hours in.
    """

    arm: str
    step: int
    position: int
    burst_length: int
    #: Flat position in the batch, 0..batch_size-1.
    batch_slot: int
    #: Which accumulation step holds it, and which row within that micro-batch.
    micro_index: int
    row: int
    sequence: tuple
    burst_ids: tuple
    burst_file: str
    burst_file_sha256: str
    #: sha256 over the burst's TOKEN IDS. Not in provenance.json, which records
    #: file hashes only -- so this is computed and recorded here rather than
    #: looked up, and becomes checkable from the next run onward.
    burst_token_sha256: str

    def record(self) -> dict:
        return {
            "arm": self.arm,
            "step": self.step,
            "position": self.position,
            "burst_length": self.burst_length,
            "batch_slot": self.batch_slot,
            "micro_index": self.micro_index,
            "row": self.row,
            "burst_file": self.burst_file,
            "burst_file_sha256": self.burst_file_sha256,
            "burst_token_sha256": self.burst_token_sha256,
            "sequence_length": len(self.sequence),
        }


def batch_slot_for(run_seed: int, batch_size: int) -> int:
    """Which sequence of the batch the burst replaces. DERIVED, never sampled.

    SHA-256 through make_bursts.derived_seed, the same mechanism data order
    uses. No RNG draw, so calling this cannot shift any generator and cannot
    break bit-identity with the twin before the injection step.

    Derived from the seed rather than fixed at 0 so the burst is not always in
    the first micro-batch -- otherwise a systematic first-micro-batch effect
    would be indistinguishable from a burst effect. Both arms of a seed-matched
    pair share the seed, so they share the slot.
    """
    if batch_size <= 0:
        raise InjectionError(f"batch_size must be positive; got {batch_size}")
    return derived_seed(run_seed, SLOT_PURPOSE) % batch_size


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_ids(ids) -> str:
    return hashlib.sha256(
        json.dumps([int(i) for i in ids]).encode("utf-8")).hexdigest()


def load_burst_ids(arm: str, path: Path, expected_length: int,
                   tokenizer=None) -> tuple:
    """Tokenize one arm's burst text, refusing anything unexpected.

    Looked up BY ARM NAME. This never iterates bursts/ or provenance's arm
    list, both of which hold seven entries where the study has six.
    """
    if not path.is_file():
        raise InjectionError(
            f"burst text for arm {arm!r} not found at {path}. "
            "injection.burst_text_paths points at it and the loader resolved "
            "it relative to the repo root.")

    record = {}
    if BURSTS_PROVENANCE.is_file():
        record = json.loads(
            BURSTS_PROVENANCE.read_text(encoding="utf-8")
        ).get("arms", {}).get(arm, {})

    actual_file_sha = _sha256_file(path)
    expected_file_sha = record.get("sha256")
    if expected_file_sha and actual_file_sha != expected_file_sha:
        raise InjectionError(
            f"BURST TEXT FOR ARM {arm!r} DOES NOT MATCH ITS PROVENANCE.\n"
            f"  recorded sha256: {expected_file_sha}\n"
            f"  actual sha256:   {actual_file_sha}\n"
            f"  file: {path}\n"
            "The text this run would inject is not the text the arms were "
            "matched on in step 8, so the matching does not describe it. "
            "Refusing rather than warning: a warning at step 200 of a "
            "multi-day run is a warning nobody reads.")

    tokenizer = tokenizer or load_tokenizer(stream=io.StringIO())
    ids = tuple(tokenizer(path.read_text(encoding="utf-8"))["input_ids"])

    if len(ids) != expected_length:
        raise InjectionError(
            f"burst text for arm {arm!r} tokenizes to {len(ids)} tokens; "
            f"injection.burst_length_tokens is {expected_length}.\n"
            f"  file: {path} (sha256 matches provenance: "
            f"{bool(expected_file_sha) and actual_file_sha == expected_file_sha})\n"
            "If the file hash matches and the token count does not, this is "
            "TOKENIZER DRIFT rather than an edited file -- the same failure "
            "the corpus pipeline guards with its probe hash.")

    recorded_tokens = record.get("tokens")
    if recorded_tokens is not None and recorded_tokens != len(ids):
        raise InjectionError(
            f"burst text for arm {arm!r} tokenizes to {len(ids)} tokens but "
            f"bursts/provenance.json records {recorded_tokens}")

    return ids, actual_file_sha


def load_filler_ids(seq_len: int, burst_length: int, tokenizer=None) -> tuple:
    """The (seq_len - burst_length) tokens of context.txt that surround it.

    Exactly what scripts/match_arms.py does at :172 --
    `filler_ids = context_ids[:seq.value - n_burst]` -- so the sequence built
    here is the sequence step 8 measured.
    """
    tokenizer = tokenizer or load_tokenizer(stream=io.StringIO())
    context_ids = tokenizer(
        CONTEXT_FILE.read_text(encoding="utf-8"))["input_ids"]
    if len(context_ids) != seq_len:
        raise InjectionError(
            f"{CONTEXT_FILE.name} tokenizes to {len(context_ids)} tokens but "
            f"training.seq_len is {seq_len}. The filler would be cut from a "
            "different passage than step 8 used.")
    return tuple(context_ids[:seq_len - burst_length])


def build_plan(cfg, tokenizer=None) -> InjectionPlan | None:
    """Decide everything about the injection, or return None for twin.

    Returns None when the arm does not inject. The caller checks for None; a
    plan that existed but did nothing would be a hook that silently no-ops.
    """
    sys.path.insert(0, str(REPO_ROOT))
    from burst.config import INJECTING_ARMS

    if cfg.arm not in INJECTING_ARMS:
        return None

    for name, value in (("injection_step", cfg.injection.injection_step),
                        ("burst_length_tokens",
                         cfg.injection.burst_length_tokens),
                        ("burst_position", cfg.injection.burst_position)):
        if value is None:
            raise InjectionError(
                f"injection.{name} is null and arm {cfg.arm!r} injects. There "
                "is no default: it is part of the study's definition. Decide "
                "it in configs/base.yaml.")

    path = cfg.injection.burst_text_paths.for_arm(cfg.arm)
    if path is None:
        raise InjectionError(
            f"no burst text path for arm {cfg.arm!r}")

    burst_length = cfg.injection.burst_length_tokens
    position = cfg.injection.burst_position
    tokenizer = tokenizer or load_tokenizer(stream=io.StringIO())

    burst_ids, file_sha = load_burst_ids(
        cfg.arm, REPO_ROOT / path, burst_length, tokenizer)
    filler_ids = load_filler_ids(
        cfg.training.seq_len, burst_length, tokenizer)

    # THE SPLICE. Not reimplemented -- see the module docstring. It verifies
    # its own output, so if this returns, the burst is where it claims to be.
    sequence = tuple(assemble_sequence(filler_ids, burst_ids, position))

    slot = batch_slot_for(cfg.seed, cfg.training.batch_size)
    micro = cfg.training.micro_batch
    if micro is None:
        raise InjectionError(
            "training.micro_batch is null, so which accumulation step holds "
            "the burst is undefined")

    return InjectionPlan(
        arm=cfg.arm,
        step=cfg.injection.injection_step,
        position=position,
        burst_length=burst_length,
        batch_slot=slot,
        micro_index=slot // micro,
        row=slot % micro,
        sequence=sequence,
        burst_ids=tuple(burst_ids),
        burst_file=str(path),
        burst_file_sha256=file_sha,
        burst_token_sha256=_sha256_ids(burst_ids),
    )


def apply(plan: InjectionPlan | None, step: int, micro_index: int, rows):
    """Replace one row of a raw (micro, seq_len) token block. Returns (rows, fired).

    Called with the RAW sequences, before inputs and targets are split apart.
    Splicing into an already-shifted pair would mean patching two tensors
    separately, which is a second splice by another name.

    Consumes no sampler index and draws no randomness: it overwrites a row of
    the block the sampler already produced.
    """
    if plan is None or step != plan.step or micro_index != plan.micro_index:
        return rows, False

    if plan.row >= rows.shape[0]:
        raise InjectionError(
            f"injection targets row {plan.row} of micro-batch "
            f"{plan.micro_index}, which holds {rows.shape[0]} rows")
    if rows.shape[1] != len(plan.sequence):
        raise InjectionError(
            f"injection sequence is {len(plan.sequence)} tokens but the batch "
            f"carries {rows.shape[1]}-token sequences")

    torch = _torch()
    replaced = rows.clone()
    replaced[plan.row] = torch.tensor(list(plan.sequence), dtype=rows.dtype)
    return replaced, True


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise InjectionError(f"PyTorch is required to inject: {exc}") from exc
    return torch


def burst_region_losses(model, plan: InjectionPlan, device="cpu") -> dict:
    """Per-token loss over the burst region, at the moment of injection.

    THE ONLY CHANCE TO CAPTURE THIS. The model sees this text once and never
    again, so a number not taken here cannot be recovered afterwards. It is
    what lets someone check later that the burst was as surprising as step 8
    measured.

    Run under no_grad on the injected sequence alone, so it cannot contribute
    to the gradient, touch the optimizer, or consume RNG (dropout is off).

    UNVERIFIED ON CUDA: that this extra forward cannot perturb bit-identity.
    It should not -- no grad, no RNG, and cudnn.benchmark is False so there is
    no autotune cache to warm -- but this repo has no GPU and the claim is not
    tested there. The CPU bit-identity test covers the bookkeeping. If the GPU
    test ever fails, the fallback is to capture logits from the training
    forward instead of running a second one.
    """
    torch = _torch()
    import torch.nn.functional as F
    import model_seam as SEAM

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            ids = torch.tensor([list(plan.sequence)], dtype=torch.long,
                               device=device)
            out = model(input_ids=ids) if hasattr(model, "transformer") \
                else model(ids[:, :-1], ids[:, 1:])
            logits = getattr(out, "logits", None)
            if logits is None:
                return {"per_token_losses": None,
                        "WHY_ABSENT": (
                            "this model family returns a loss rather than "
                            "logits, so per-token values are not recoverable "
                            "without changing its forward signature")}
            # Token at index i is predicted from position i-1, so the burst
            # region's predictions start one before the burst does.
            shift_logits = logits[0, :-1, :]
            shift_targets = ids[0, 1:]
            per_token = F.cross_entropy(
                shift_logits.float(), shift_targets, reduction="none")
            start = plan.position - 1
            region = per_token[start:start + plan.burst_length]
            return {
                "per_token_losses": [float(v) for v in region],
                "n_predictions": int(region.numel()),
                "mean": float(region.mean()),
                "region_start_prediction_index": start,
            }
    finally:
        if was_training:
            model.train()
