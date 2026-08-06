#!/usr/bin/env python
"""The displacement ladder: L2, activation similarity, CKA and loss, on a fixed pair set.

    python scripts/displacement_ladder.py --runs-root /shared/27as66/burst-pilot/runs-fixed \
        --out docs/measurements --cfg-scratch /some/scratch/dir

WHY THIS EXISTS. `scripts/metrics.py` implements four metrics and three of them
had never been run against a checkpoint on disk, because the only runner in the
repo (`scripts/pilot_barrier.py`) calls `barrier` and stops. This is the runner
for the other three, plus per-model loss.

    THE ORIGINAL MOTIVATION IS RETRACTED, AND IS LEFT HERE BECAUSE IT IS WHY
    THIS FILE WAS WRITTEN. It said: the pilot's headline metric returned exactly
    zero for the arm-vs-twin displacement while the twin-vs-twin pairs peaked at
    2.6 to 4.2, so the plain barrier is a same-basin detector and that zero is a
    property of the metric's reach. **That was the v1 pilot, which S97 voided.**
    On corrected models the same metric reads 0.133863 with `rose: True` (S100,
    docs/measurements/2026-08-06-pilot-v2-results.md). The barrier was never the
    problem. The other three metrics are still worth having, for the reason in
    the paragraph above, which never depended on the retracted one.

WHAT IT DOES NOT DO, AND THIS IS STRUCTURAL RATHER THAN A PROMISE.

    It computes NO VERDICT. No `clears_noise_floor`, no effect-against-floor
    comparison, no ranking of pairs. The ladder reports distances; reading them
    against one another is a separate act with its own preconditions, and D-4
    (the correction and its family) is open.

    BE PRECISE ABOUT WHAT ENFORCES THAT, because the first draft of this
    docstring overclaimed and an audit caught it. `assert_no_verdict_keys` runs
    before either file is written and refuses a key from a denylist, so a
    verdict cannot enter THE ARTIFACT without the guard being deleted first.
    That is a claim about the payload, and the guard is a payload walk. It says
    nothing about anything printed to `stream`: the first version of
    `run_phase2` printed `min(cka)` across layers as an aligned
    one-scalar-per-pair column, which a reader reads effect-against-floor
    straight off, and the guard could not see it because it never entered the
    payload. That reduction is gone and
    `test_no_phase_reduces_across_layers_or_pairs_on_its_progress_line` keeps it
    gone by reading the source. An immunity claim broader than its mechanism is
    worse than none, because it stops the next reader looking.

THE PRE-REGISTRATION ORDERING CONSTRAINT IS SATISFIED, AND HERE IS WHY.

`docs/preregistration.md` section 8.5 constraint 1 requires the section 8.4
canonicalization measurement to be run and its branch RECORDED before any
arm-vs-twin distance from the pilot is examined; section 8.7 names the reverse
order as invalidating section 8. That branch is recorded and committed:
`docs/measurements/12-injection-point-ruler.json` carries
`branch: plain_loss_barrier` against the step-199 twin checkpoint. The gate is a
one-time ordering condition on the BRANCH CHOICE, and once the branch is on the
record nothing measured afterwards can make it contingent on a result. So this
script is downstream of a closed gate rather than racing it.

That is why it is also NOT the headline metric and does not try to be. Section
8.4 selected the plain loss barrier. Everything here is exploratory under
section 7, and the no-verdict rule above is what keeps it that way.

TWO PHASES, WRITTEN SEPARATELY.

Phase 1 is total L2, per-layer L2 and per-model loss -- parameter arithmetic and
one forward pass per model, no tapped modules. It is written to disk before
phase 2 starts. Phase 2 is per-layer activation cosine and per-layer CKA, which
need the activation tap. If phase 2 fails, phase 1's numbers survive on disk and
the JSON records phase 2 with `ok: false` and the reason. The two halves fail
independently because their failure modes are unrelated: phase 1 can only fail
on a checkpoint that will not load, and phase 2 can additionally fail on a model
whose module structure the tap helper does not recognise.

THE ZERO CHECK GATES BOTH, AND IT IS THE ONLY PROOF THIS RUNNER WORKS.

One checkpoint is loaded twice, independently, and measured against itself. Four
of the five gate assertions are EXACT and were verified at real scale (13 layers,
1024 token positions, GPT-2 124M shapes) before this file was written:

    total L2                exactly 0.0
    per-layer L2            exactly 0.0 in every bucket
    per-model loss          bitwise equal between the two loads
    activations             bitwise equal (torch.equal), every layer
    per-layer CKA           exactly 1.0
    norm ratio              exactly 1.0
    per-layer cosine        NOT exactly 1.0 -- see below

The cosine is the one exception and it is arithmetic, not a defect. For bitwise
identical activations the numerator is `sum(a_i * a_i)` and the denominator is
`norm(a) * norm(a)`; the two reduce the same 768 products in different orders, so
they differ by an ulp or two. Measured max deviation at real scale: 6.66e-16,
which is 3 ulps of double. `COSINE_IDENTITY_ULPS` allows 8, and the observed
deviation is RECORDED in the artifact on every run rather than merely tolerated,
so drift is visible. That tolerance is deliberately not load-bearing: it sits
behind four exact checks, including `torch.equal` on the activations themselves,
and no broken runner can reach it without failing one of those first.

MEMORY. Three passes, each bounded, and nothing is cached across pairs. Pass one
holds ONE model (per-model loss and provenance for the five distinct
checkpoints). Passes two and three hold TWO. A checkpoint used by three pairs is
therefore read from disk three times; that is I/O traded for a hard ceiling of
two 124M models resident, and the trade is recorded as D29.

PRECISION AND DEVICE. The model is kept at its stored dtype and
`metrics.cross_entropy_loss` casts logits to double, matching
`scripts/pilot_barrier.py:26-30`. Device is CUDA when available and CPU
otherwise, recorded either way. Reduction order depends on the device, so
numbers taken on different devices are not bitwise comparable -- the artifact
says so, and the zero check re-proves the identity invariants on whichever
device actually ran.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for _p in (str(REPO), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import metrics as M  # noqa: E402
import model_seam as SEAM  # noqa: E402
from burst.config import (  # noqa: E402
    CHECKPOINT_FULL,
    CHECKPOINT_WEIGHTS_ONLY,
    load_config,
    run_name_for,
)

#: Artifact stem for a real run. Junk never lands under this name -- see
#: SELFTEST_STEM and the refusal in `resolve_output`.
STEM = "13-pilot-displacement-ladder"

#: Artifact stem for --selftest. Named so loudly because a plumbing result
#: filed as a measurement is the mistake docs/measurements/10-metrics.json's
#: LIMITATION exists to prevent.
SELFTEST_STEM = "13-SELFTEST-displacement-ladder-JUNK-MODELS"

#: The two checkpoint kinds scripts/train.py writes. A payload naming anything
#: else is refused rather than read: the shapes differ only in three keys, and
#: guessing which one an unknown kind resembles is how a metric ends up
#: measuring something other than what it names.
KNOWN_KINDS = (CHECKPOINT_FULL, CHECKPOINT_WEIGHTS_ONLY)

#: Keys every payload carries, both kinds. scripts/train.py:317-325.
PAYLOAD_BASE_KEYS = ("kind", "step", "family", "model", "seed", "arm",
                     "micro_batch")

#: Keys ONLY a full payload carries. scripts/train.py:326-329. Never read here;
#: listed so `carries_rng` is derived from the kind rather than from a key that
#: does not exist on disk (`carries_rng` lives in train_record.json, not in the
#: payload).
PAYLOAD_FULL_ONLY_KEYS = ("optimizer", "rng", "rng_digest")

#: Machine epsilon for float64.
DBL_EPS = 2.220446049250313e-16

#: Ulps of double allowed between an identical pair's cosine and exactly 1.0.
#: Measured max at real scale was 3; 8 is the headroom. NOT load-bearing: four
#: exact assertions sit in front of it. See the module docstring.
COSINE_IDENTITY_ULPS = 8
COSINE_IDENTITY_TOL = COSINE_IDENTITY_ULPS * DBL_EPS

#: Relative tolerance on "the per-bucket L2s recombine into the total". Both
#: sides are float64 sums of the SAME per-tensor values in different groupings,
#: so this is float reassociation only and nothing else can hide inside it.
SUM_OF_PARTS_RTOL = 1e-12

#: Commits whose checkpoints are refused outright, by recorded training commit.
#: `9aa930d` is the v1 pilot: S97 proved every run from it was optimised to
#: predict two tokens ahead. This is the SECOND of two independent objective
#: gates and it reads a CLAIM about the artifact (run_provenance.yaml) rather
#: than the artifact. Kept separate from the measured gate on purpose -- S97
#: exists because a claim and the thing it describes can disagree, and a repo
#: that had only ever checked the claim would have shipped the defect.
REFUSED_TRAINING_COMMITS = ("9aa930d",)

#: The commit that FIXED the double shift (S99: model_seam's HF branch stopped
#: passing labels=). A checkpoint is clean on this axis iff its recorded training
#: commit CONTAINS this one, which is an ancestry question and not a name
#: question -- so it catches every pre-fix commit, including ones nobody thought
#: to list. Asa asked for exactly this in 715dd67, handing the call to this
#: module: "the clause wants deriving from trained_at ancestry, failing safe to
#: the warning."
OBJECTIVE_FIX_COMMIT = "3e715a6"

#: What ancestry resolution concluded. UNKNOWN is not clean: a shallow clone, a
#: commit this checkout has never seen, or no git at all all land here, and the
#: banner warns on it. Refusing outright would over-block a legitimate run on a
#: machine without the full history -- gate A reads the weights themselves and is
#: the harder check anyway.
ANCESTRY_CONTAINS_FIX = "contains_fix"
ANCESTRY_PREDATES_FIX = "predates_fix"
ANCESTRY_UNKNOWN = "unknown"


def objective_fix_ancestry(commit) -> dict:
    """Does `commit` contain OBJECTIVE_FIX_COMMIT? Ancestry, not string matching.

    `git merge-base --is-ancestor FIX TRAINED` exits 0 when the training commit
    has the fix in its history. Anything else -- a missing commit, a shallow
    clone, no git -- is UNKNOWN, which the banner treats as suspect. Failing safe
    means failing to the warning, not to silence.
    """
    if not commit:
        return {"state": ANCESTRY_UNKNOWN, "commit": None,
                "reason": "no training commit is recorded for this checkpoint"}
    try:
        result = subprocess.run(
            ("git", "merge-base", "--is-ancestor", OBJECTIVE_FIX_COMMIT,
             str(commit)),
            cwd=str(REPO), capture_output=True, text=True, timeout=30)
    except Exception as exc:
        return {"state": ANCESTRY_UNKNOWN, "commit": str(commit),
                "reason": f"could not run git: {exc!r}"}
    if result.returncode == 0:
        return {"state": ANCESTRY_CONTAINS_FIX, "commit": str(commit),
                "fix": OBJECTIVE_FIX_COMMIT,
                "reason": (f"{commit} contains {OBJECTIVE_FIX_COMMIT}, the "
                           "commit that fixed the double shift")}
    if result.returncode == 1:
        return {"state": ANCESTRY_PREDATES_FIX, "commit": str(commit),
                "fix": OBJECTIVE_FIX_COMMIT,
                "reason": (f"{commit} does NOT contain "
                           f"{OBJECTIVE_FIX_COMMIT}, so it predates the fix "
                           "and trained the double-shifted objective")}
    return {"state": ANCESTRY_UNKNOWN, "commit": str(commit),
            "reason": ("git could not resolve the ancestry: "
                       + (result.stderr.strip() or f"exit {result.returncode}"))}

#: How much LOWER the two-ahead loss must be before a checkpoint is called
#: double-shifted. Not a taste: the two regimes are separated by orders of
#: magnitude and this sits between them, the way section 8.4's thresholds were
#: placed against an observed-good and an observed-bad case.
#:
#:   v1 double-shifted   next 7.1085 - two-ahead 4.2613 = +2.85 nats  (S97)
#:   junk, worst of 10   +0.008 nats            -- both at chance, sign is noise
#:   public GPT-2        -6.90 nats             -- wrong direction entirely
#:
#: 0.25 is 30x above the largest noise observed on an untrained model and 11x
#: below the real signal. A bare `two_ahead < next_token` test, which is the
#: literal rule, fired on 8 of 10 junk models -- see D32.
DOUBLE_SHIFT_MARGIN_NATS = 0.25

#: Objective classifications. NOT named `verdict`: that word is on
#: FORBIDDEN_PAYLOAD_KEYS because in this repo it means a conclusion about the
#: study's comparison, and the guard refused the payload the first time this
#: field carried it. Second time the guard has caught a name of mine -- see S98.
#: `indeterminate` is a real state, not a hedge: when both
#: losses sit at chance the comparison carries no information, and calling that
#: a pass would let an untrained checkpoint through the gate it exists for.
OBJECTIVE_NEXT_TOKEN = "next_token"
OBJECTIVE_TWO_AHEAD = "two_ahead"
OBJECTIVE_INDETERMINATE = "indeterminate"

#: Payload keys that would make this a verdict rather than a measurement.
#: Checked before writing. See `assert_no_verdict_keys`.
FORBIDDEN_PAYLOAD_KEYS = frozenset({
    "clears_noise_floor", "clears_floor", "cleared", "verdict", "significant",
    "significance", "p_value", "p_adjusted", "rank", "ranking", "ordering",
    "above_noise_floor", "exceeds_floor", "noise_floor", "effect_vs_floor",
})

#: EMITTED ONLY WHEN THE MEASUREMENT SAYS SO. The objective gate aborts on a
#: double-shifted checkpoint, so in a normal run this clause never appears -- and
#: that absence is itself a claim, which is why it is derived rather than
#: hardcoded. It is retained in full for the case the gate is bypassed, disabled,
#: or extended with an override: a v1 checkpoint that reaches the artifact must
#: still carry this banner. Kept verbatim from the version written when every
#: checkpoint on the cluster was v1.
LIMITATION_DOUBLE_SHIFT = (
    "THE CHECKPOINTS THIS ARTIFACT READS WERE TRAINED ON THE WRONG OBJECTIVE, "
    "AND THE OBJECTIVE GATE SAYS SO BELOW. scripts/train.py pre-shifts each "
    "block into (inputs, targets) and then hands the pair to a path that passes "
    "labels=, which transformers shifts again -- so the run was optimised to "
    "predict TWO TOKENS AHEAD. Full account and proof: "
    "docs/2026-08-05-training-objective-defect.md and S97. That defect voids the "
    "v1 barrier curves and the endpoint-loss delta/sigma this ladder was built "
    "to look past; it does NOT void the section 8.4 ruler branch, the injection "
    "verification, or any determinism record. WHAT IT MEANS FOR THE NUMBERS "
    "BELOW: the checkpoints are real and these distances are correctly computed "
    "over them, but they are distances between models trained on a different "
    "objective than the study intends, so no number here transfers to a "
    "corrected re-run. The per-model loss is the sharpest case -- "
    "metrics.cross_entropy_loss is one of the FOUR loss paths that shift "
    "correctly, so the loss reported here is proper next-token loss on a model "
    "that was never trained for it, and read roughly 7.1 against a training "
    "curve of 4.26 on the v1 pilot. Read that section as a description of what "
    "these checkpoints are, not as a measurement of the burst. "
)

LIMITATION_BASE = (
    "THIS FILE REPORTS DISTANCES AND COMPUTES NO VERDICT. No comparison of the "
    "effect pair against the floor pairs is made here, no pair is ranked, and "
    "nothing is described as clearing anything. Reading these numbers against "
    "one another is a separate act: docs/decisions-pending.md D-4 (the family "
    "of tests and the multiple-comparison correction) is open, and "
    "scripts/analysis.py still requires both with no default. "
    "RAW L2 IS BLIND TO NEURON PERMUTATION. It reads parameters as a flat "
    "vector, so two checkpoints differing only by a relabelling of attention "
    "heads or FFN neurons are reported as far apart by the full size of that "
    "relabelling. This bears asymmetrically on the pair set and the asymmetry "
    "is the point: the effect pair shares an initialization and a data order "
    "and differs by one burst, so it structurally cannot carry permutation "
    "drift, while the floor pairs are independently initialized and may carry "
    "any amount of it. A larger floor number therefore does not establish that "
    "seed variation is larger than the burst's effect -- part of it may be "
    "gauge. Removing that gauge is what scripts/canonicalize.py exists for, "
    "and the aligned metrics that would use it are unbuilt (D-6). "
    "THE EVALUATION TEXT IS TRAINING DATA. Loss and every activation number "
    "here are taken on bursts/context.txt, which bursts/provenance.json records "
    "as openwebtext document 73 -- the corpus every one of these models trained "
    "on. These are not held-out numbers and no memorisation confound is "
    "controlled. "
    "THE ACTIVATION BASIS IS ONE DOCUMENT. CKA and cosine are computed over the "
    "1024 token positions of that single passage, and for CKA those positions "
    "are the entire sample. The held-out corpus slice offers far more; it is "
    "not used here because every committed number in docs/measurements/ is "
    "keyed to this batch identity and swapping it would make old and new "
    "numbers incomparable. "
    "CKA IS INVARIANT TO ROTATION AND COSINE IS NOT, WHICH IS WHY BOTH ARE "
    "REPORTED. Cosine is additionally blind to magnitude, so the norm ratio is "
    "reported beside it. "
    "PRECISION IS bf16-TRAINED, MEASURED AT LOAD DTYPE. The pilot trained at "
    "bf16 with fp32 master weights; nothing here re-derives what that costs. "
    "Reduction order depends on the device, so numbers from a CUDA run and a "
    "CPU run are not bitwise comparable."
)


def floor_sentence(payload) -> str:
    """The floor's size, COUNTED from the pair set rather than typed.

    v1 had three floor pairs across seeds 0/1/2. v2 drops seed02_twin to buy a
    second injecting arm, so there is exactly one. A reader who carried the v1
    habit of reading a min/median/max across floor pairs would be reading a
    spread that does not exist, so the count is stated and the absence of a
    spread is stated with it.
    """
    labels = [p.get("label") for p in (payload.get("pair_set") or [])]
    n = sum(1 for lab in labels if lab and lab.startswith("floor"))
    if n == 1:
        return (
            f"THE FLOOR IS n={n}. There is exactly one twin-against-twin pair in "
            "this run, down from three in v1, because seed02_twin was dropped to "
            "buy a second injecting arm at a matched seed. ONE PAIR GIVES NO "
            "SPREAD: there is no min, median or max across floor pairs to read, "
            "and no way to tell from this artifact how much a floor pair varies. "
            "v1's three pairs ranged 2.59 to 4.22 on the barrier, so the "
            "variation being unmeasured here is not small. "
        )
    return (
        f"THE FLOOR IS n={n} pair(s) in this run. Read the per-pair numbers "
        "rather than any summary across them; this file computes no summary. "
    )


def limitation(payload) -> str:
    """The banner, DERIVED from what was measured. Recomputed at write time.

    Two parts. The base always applies. The double-shift clause is emitted only
    when the objective gate actually observed a double-shifted checkpoint --
    either gate A on the losses or gate B on the recorded commit. So the clause's
    ABSENCE is a claim about this run's inputs rather than an editing decision,
    and a v1 checkpoint that reached the artifact through a bypassed gate still
    carries the warning. Same reasoning as metrics_report.limitation: a stored
    copy could only ever go stale, and hardcoded prose inside a generated file is
    S70's defect.
    """
    by_ckpt = ((payload.get("objective_gate") or {}).get("by_checkpoint")
               or {})
    double_shifted = sorted(
        ref for ref, cell in by_ckpt.items()
        if cell.get("objective") == OBJECTIVE_TWO_AHEAD
        or cell.get("gate_b_refused"))
    # FAIL SAFE TO THE WARNING, which is what Asa asked for in 715dd67. A
    # checkpoint whose ancestry cannot be resolved is not shown to be clean, so
    # the clause appears -- flagged separately from a proven one, because
    # "unverified" and "known bad" are different facts and one name for both is
    # this repo's logged defect.
    unverified = sorted(
        ref for ref, cell in by_ckpt.items()
        if ref not in double_shifted and cell.get("gate_b_suspect"))
    head = ""
    if double_shifted or unverified:
        head = LIMITATION_DOUBLE_SHIFT
        if double_shifted:
            head += ("CHECKPOINTS SHOWN TO BE AFFECTED: "
                     + ", ".join(double_shifted) + ". ")
        if unverified:
            head += (
                "CHECKPOINTS WHOSE OBJECTIVE COULD NOT BE VERIFIED FROM THEIR "
                "RECORDED ANCESTRY, treated as suspect rather than clean: "
                + ", ".join(unverified)
                + ". These are not shown to be double-shifted -- their measured "
                f"losses did not say so -- but neither is it established that "
                f"they contain {OBJECTIVE_FIX_COMMIT}, so this warning stands "
                "until it is. ")
    return head + floor_sentence(payload) + LIMITATION_BASE


ORDERING_ATTESTATION = (
    "docs/preregistration.md section 8.5 constraint 1 requires the section 8.4 "
    "measurement to be run and its branch recorded BEFORE any arm-vs-twin "
    "distance from the pilot is examined. It is recorded: "
    "docs/measurements/12-injection-point-ruler.json carries "
    "branch=plain_loss_barrier against the step-199 twin checkpoint, committed. "
    "The constraint is a one-time gate on the branch choice, so a distance "
    "measured after the branch is on the record cannot make the choice "
    "contingent on a result. Section 8.7's invalidator -- an arm-vs-twin "
    "displacement examined BEFORE the branch is recorded -- is therefore not "
    "reachable from here. This measurement is exploratory under section 7 and "
    "is not the headline metric; section 8.4 already selected that."
)


#: Exit codes, distinct on purpose. argparse already uses 2 for a bad command
#: line, so phase-2 failure must not also be 2 -- one code meaning both
#: "nothing was written" and "phase 1 is on disk, phase 2 is not" cannot be
#: acted on. An audit caught the collision.
EXIT_OK = 0
EXIT_REFUSED = 1          # a LadderError: gate abort, bad input, refusal
EXIT_PHASE2_FAILED = 3    # phase 1 on disk and valid; phase 2 did not run


class LadderError(Exception):
    """Raised for every refusal here. Never an assert -- `python -O` deletes
    asserts, and validation that vanishes under an optimizer flag is worse than
    none. Mirrors burst/config.py's ConfigError rule."""


# ---------------------------------------------------------------------------
# The batch, on a device
#
# metrics.Batch.input_ids() builds a CPU tensor (scripts/metrics.py:170-172),
# and metrics.cross_entropy_loss / activations_* all call it, so a CUDA model
# device-mismatches against it. metrics.py is out of scope for this task, so the
# device is threaded in by EXTENDING Batch rather than by editing it. identity()
# is inherited unchanged, which matters: the batch identity must stay equal to
# the one every other report in docs/measurements/ is keyed to. See D30.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceBatch(M.Batch):
    """A Batch whose ids land on `device`. Identity is the parent's, untouched."""

    device: str = "cpu"

    def input_ids(self):
        return super().input_ids().to(self.device)


def on_device(batch: M.Batch, device: str) -> M.Batch:
    """Re-wrap a Batch for `device`. Returns the original when it is CPU."""
    if device == "cpu":
        return batch
    return DeviceBatch(device=device,
                       **{f.name: getattr(batch, f.name)
                          for f in dataclasses.fields(M.Batch)})


def resolve_device(requested: str) -> dict:
    """Pick the device and RECORD what was picked, including why."""
    torch = M._torch()
    available = bool(torch.cuda.is_available())
    if requested == "cuda" and not available:
        raise LadderError(
            "--device cuda was requested and torch.cuda.is_available() is "
            "False. Refusing to silently fall back to CPU: the device changes "
            "reduction order, so a number labelled cuda that ran on cpu is "
            "mislabelled rather than merely slower.")
    if requested == "auto":
        chosen = "cuda" if available else "cpu"
    else:
        chosen = requested
    record = {
        "requested": requested,
        "chosen": chosen,
        "cuda_available": available,
        "NOTE": ("reduction order is device-dependent, so numbers taken here "
                 "are not bitwise comparable to numbers taken on another "
                 "device. The zero check re-proves the identity invariants on "
                 "whichever device actually ran."),
    }
    if chosen == "cuda":
        record["cuda_device_name"] = torch.cuda.get_device_name(0)
    return record


# ---------------------------------------------------------------------------
# Parameter grouping -- the only genuinely new arithmetic in this file
#
# NEW CODE, and mutation-audited. See D31 and the audit recorded in
# implementation-notes.md. Everything else here reuses scripts/metrics.py.
# ---------------------------------------------------------------------------

#: Anchored, and the integer is `\d+` rather than `\d` ON PURPOSE. With a single
#: digit, blocks 10 and 11 fail to match and fall through to the unknown-bucket
#: refusal -- which is the exact mutation the audit uses, because it is the bug
#: a careless regex actually produces.
_BLOCK_RE = re.compile(r"^transformer\.h\.(\d+)\.")

BUCKET_EMBEDDINGS = "embeddings"
BUCKET_FINAL = "final"


def bucket_for(name: str) -> str:
    """Which bucket a parameter name belongs to. Refuses an unknown name.

    REFUSES RATHER THAN DROPS. A grouping that silently ignores a parameter
    reports per-layer parts that do not recombine into the total, and the total
    is the number a reader trusts. That is the S55 shape -- a quantity named
    for one thing computed over another -- so an unrecognised name is an error.
    """
    block = _BLOCK_RE.match(name)
    if block is not None:
        return f"block_{int(block.group(1)):02d}"
    if name.startswith("transformer.wte.") or name.startswith("transformer.wpe."):
        return BUCKET_EMBEDDINGS
    if name.startswith("transformer.ln_f."):
        return BUCKET_FINAL
    raise LadderError(
        f"parameter {name!r} matches no bucket. The buckets are "
        f"{BUCKET_EMBEDDINGS!r} (transformer.wte / transformer.wpe), "
        f"block_NN (transformer.h.NN.*) and {BUCKET_FINAL!r} "
        f"(transformer.ln_f.*).\n"
        "This is refused rather than dropped: an ungrouped parameter would "
        "make the per-layer L2s fail to recombine into the total, and the "
        "per-layer numbers would be about fewer parameters than their labels "
        "claim. If the architecture gained a tensor, add a bucket for it here "
        "deliberately.")


def group_parameter_names(names) -> dict:
    """bucket -> sorted parameter names. Asserts an exact partition.

    The partition check is INTEGER arithmetic over the name set, so it cannot
    be satisfied by a near-miss: every name lands in exactly one bucket and the
    bucket sizes sum to the name count. A tied lm_head.weight is already absent
    because callers pass metrics.parameter_view, which deduplicates by identity
    (scripts/metrics.py:282-291) -- state_dict() would not, and would
    double-weight 38M of GPT-2's 124M parameters.
    """
    names = list(names)
    if not names:
        raise LadderError("no parameters to group")
    grouped: dict = {}
    for name in names:
        grouped.setdefault(bucket_for(name), []).append(name)

    placed = sum(len(v) for v in grouped.values())
    if placed != len(names):
        raise LadderError(
            f"grouping is not a partition: {len(names)} parameters in, "
            f"{placed} placed. Every parameter must land in exactly one "
            "bucket.")
    seen: set = set()
    for bucket, members in grouped.items():
        for name in members:
            if name in seen:
                raise LadderError(
                    f"parameter {name!r} was placed in more than one bucket; "
                    f"the second was {bucket!r}.")
            seen.add(name)
    return {b: sorted(v) for b, v in sorted(grouped.items(),
                                            key=_bucket_sort_key)}


def _bucket_sort_key(item):
    """embeddings first, then blocks in numeric order, then final."""
    bucket = item[0]
    if bucket == BUCKET_EMBEDDINGS:
        return (0, 0)
    if bucket == BUCKET_FINAL:
        return (2, 0)
    return (1, int(bucket.rsplit("_", 1)[1]))


def grouping_shape(view: dict) -> dict:
    """The grouping itself, as reportable structure. No distances.

    The totals and the parts come from two independent places -- `len(view)` and
    the sum over buckets -- and they are COMPARED here rather than merely both
    reported. It is an exact integer check and it costs nothing; an audit
    pointed out that reporting two derivations of one number without comparing
    them is how a table of rows that all agree reads as a measurement (S69).
    """
    grouped = group_parameter_names(view.keys())
    n_by_bucket = {b: len(v) for b, v in grouped.items()}
    e_by_bucket = {b: int(sum(view[n].numel() for n in v))
                   for b, v in grouped.items()}
    total_params = len(view)
    total_elements = int(sum(t.numel() for t in view.values()))
    if sum(n_by_bucket.values()) != total_params:
        raise LadderError(
            f"bucket parameter counts sum to {sum(n_by_bucket.values())} but "
            f"the view holds {total_params}")
    if sum(e_by_bucket.values()) != total_elements:
        raise LadderError(
            f"bucket element counts sum to {sum(e_by_bucket.values())} but the "
            f"view holds {total_elements}")
    return {
        "buckets": list(grouped),
        "n_buckets": len(grouped),
        "total_params": total_params,
        "total_elements": total_elements,
        "n_params_by_bucket": n_by_bucket,
        "n_elements_by_bucket": e_by_bucket,
        "counts_cross_checked": (
            "bucket counts were compared against len(view) and the summed "
            "numel exactly, not merely reported alongside them"),
        "members": grouped,
    }


def per_layer_l2(view_a: dict, view_b: dict) -> dict:
    """Total L2 and per-bucket L2, with the recombination checked.

    The distances themselves are metrics.l2_distance_raw -- it accepts
    name-keyed dicts, so a bucket is measured by handing it a filtered dict
    rather than by reimplementing the norm. The new part is the grouping and
    the check that the parts recombine:

        sqrt(sum over buckets of part^2) == total

    which holds because both sides are float64 sums of the SAME per-tensor
    sums, differently grouped. A mis-grouping that double-counted or dropped a
    tensor breaks it.
    """
    if set(view_a) != set(view_b):
        raise LadderError(
            "parameter names differ between the two models; symmetric "
            f"difference: {sorted(set(view_a) ^ set(view_b))[:8]}")
    grouped = group_parameter_names(view_a.keys())
    total = M.l2_distance_raw(view_a, view_b)

    by_bucket = {}
    for bucket, members in grouped.items():
        sub_a = {n: view_a[n] for n in members}
        sub_b = {n: view_b[n] for n in members}
        by_bucket[bucket] = {
            "l2": M.l2_distance_raw(sub_a, sub_b),
            "n_params": len(members),
            "n_elements": int(sum(view_a[n].numel() for n in members)),
        }

    recombined = sum(cell["l2"] ** 2 for cell in by_bucket.values()) ** 0.5
    scale = max(abs(total), abs(recombined))
    rel = 0.0 if scale == 0.0 else abs(total - recombined) / scale
    if rel > SUM_OF_PARTS_RTOL:
        raise LadderError(
            f"the per-bucket L2s do not recombine into the total: total "
            f"{total!r}, sqrt(sum of squares) {recombined!r}, relative "
            f"difference {rel:.3e} against a tolerance of "
            f"{SUM_OF_PARTS_RTOL:.0e}.\n"
            "Both are float64 sums of the same per-tensor values, so a "
            "difference this large means the grouping dropped or duplicated a "
            "parameter rather than that floating point drifted.")

    return {
        "l2_total": total,
        "l2_by_bucket": by_bucket,
        "sum_of_parts_check": {
            "recombined": recombined,
            "relative_difference": rel,
            "rtol": SUM_OF_PARTS_RTOL,
            "WHAT_IT_PROVES": (
                "a DROPPED or DOUBLE-COUNTED tensor would move this off zero. "
                "It does NOT prove the grouping is right: moving a parameter "
                "from one bucket to another leaves both the total and the sum "
                "of squares unchanged, so this identity is blind to a "
                "mis-bucket. What constrains that is the per-bucket parameter "
                "COUNT, asserted in tests/test_displacement_ladder.py against "
                "independently written GPT-2 names rather than against "
                "bucket_for's own output."),
            "VACUOUS_ON_AN_IDENTICAL_PAIR": (
                "when both models hold the same weights every part and the "
                "total are 0.0, so the relative difference is 0 by "
                "construction and this check constrains nothing. It is "
                "informative only where the total is non-zero."),
        },
    }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _torch_load(path: Path):
    """torch.load, with the two senses of 'weights only' kept apart.

    `weights_only=False` is TORCH's unpickling-safety flag and must stay False:
    a full payload carries pickled RNG and optimizer state that the safe
    unpickler rejects. It has nothing to do with `payload["kind"] ==
    "weights_only"`, which is this study's checkpoint kind. All four load sites
    in the repo pass False for the same reason.
    """
    torch = M._torch()
    return torch.load(path, map_location="cpu", weights_only=False)


def validate_payload(payload, path: Path) -> dict:
    """Accept the two kinds scripts/train.py writes. Refuse anything else, loudly.

    train.load_checkpoint is deliberately NOT used: it hard-refuses any payload
    whose kind is not "full", and a step-199 weights-only file is a first-class
    input here. Both in-repo measurement readers bypass it the same way and say
    so; this is that pattern, not a routing-around of a guard.
    """
    if not isinstance(payload, dict):
        raise LadderError(
            f"{path} does not contain a checkpoint payload; torch.load "
            f"returned {type(payload).__name__}, not a dict.")
    missing = [k for k in PAYLOAD_BASE_KEYS if k not in payload]
    if missing:
        raise LadderError(
            f"{path} is missing checkpoint payload key(s) {missing}. Every "
            f"checkpoint scripts/train.py writes carries {list(PAYLOAD_BASE_KEYS)} "
            "regardless of kind, so this file was not written by this study's "
            "training loop.")
    kind = payload["kind"]
    if kind not in KNOWN_KINDS:
        raise LadderError(
            f"{path} names kind {kind!r}; the only kinds scripts/train.py "
            f"writes are {list(KNOWN_KINDS)}. Refusing rather than guessing "
            "which of the two an unknown kind resembles -- the shapes differ "
            f"by {list(PAYLOAD_FULL_ONLY_KEYS)} and a wrong guess reads a key "
            "that is not there.")
    state = payload.get("model")
    if state is None or not hasattr(state, "keys") or not len(state):
        raise LadderError(
            f"{path} carries no usable 'model' state dict: got "
            f"{type(state).__name__} with "
            f"{len(state) if hasattr(state, '__len__') else 'unknown'} "
            "entries.\n"
            "Refused here rather than at load_state_dict, because an empty or "
            "non-mapping state raises a raw torch RuntimeError that main() does "
            "not catch -- the run would end in a traceback instead of a message "
            "naming the file.")
    if payload.get("family") is None:
        raise LadderError(
            f"{path} records family: null. Both families reach exactly "
            "124,439,808 parameters, so nothing downstream could detect a "
            "mix-up and the parameter-count check cannot tell them apart.")
    if payload["family"] not in SEAM.FAMILIES:
        raise LadderError(
            f"{path} records family {payload['family']!r}, which is not one of "
            f"{list(SEAM.FAMILIES)}.")
    return payload


def state_sha256(state: dict) -> str:
    """sha256 over the model state dict ONLY, in sorted key order.

    NEVER the file. `save_checkpoint` stores `arm` in the payload beside the
    weights (scripts/train.py:322-323), so two files holding identical weights
    for different arms hash differently as files -- which is precisely the trap
    docs/measurements/2026-08-05-pilot-results.md:38-42 records someone hitting
    with sha256sum on the step-199 pair. Hashing payload["model"] is what makes
    "identical through step 199" checkable.
    """
    torch = M._torch()
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        digest.update(name.encode("utf-8"))
        if hasattr(tensor, "detach"):
            contiguous = tensor.detach().to("cpu").contiguous()
            # Shape and dtype go in BEFORE the bytes, so two tensors with
            # identical bytes but a different shape or dtype cannot collide.
            digest.update(str(tuple(contiguous.shape)).encode("utf-8"))
            digest.update(str(contiguous.dtype).encode("utf-8"))
            # reshape(-1) BEFORE view(uint8), and that order is load-bearing:
            # `view` on a 0-dim tensor raises "self.dim() cannot be 0 to view
            # Float as Byte". A state_dict carries buffers as well as
            # parameters, and HF GPT-2 has shipped a 0-dim `attn.masked_bias`,
            # so this is reachable on a real checkpoint rather than
            # hypothetical. Flattening also makes the uint8 reinterpretation
            # valid for every dtype, bf16 and fp16 included -- .numpy() cannot
            # represent those directly, which is why the cast comes first.
            flat = contiguous.reshape(-1)
            digest.update(flat.view(torch.uint8).numpy().tobytes()
                          if flat.numel() else b"")
        else:
            digest.update(repr(tensor).encode("utf-8"))
    return digest.hexdigest()


def _trained_at(path: Path) -> dict | None:
    """Commit AND dirty flag from run_provenance.yaml beside the checkpoint.

    Best-effort and never fatal. Kept SEPARATE from the measuring repo's commit:
    conflating "the code that produced these weights" with "the code that
    measured them" would be two different facts wearing one name.

    THE DIRTY FLAG COMES WITH IT, and dropping it was a defect an audit caught.
    A bare commit hash implies the tree was clean, and burst/config.py records
    the flag precisely because a run launched from a dirty tree stamps a hash
    that does NOT describe the code that produced it. Reporting the hash alone
    would launder that. `dirty: None` means the record did not say.
    """
    candidate = path.parent / "run_provenance.yaml"
    if not candidate.is_file():
        return None
    try:
        import yaml
        record = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    for section in (record, record.get("git") or {}, record.get("code") or {}):
        if isinstance(section, dict) and section.get("commit"):
            dirty = section.get("dirty")
            files = section.get("dirty_files")
            return {
                "commit": str(section["commit"]),
                "dirty": None if dirty is None else bool(dirty),
                "n_dirty_files": (len(files) if isinstance(files, list)
                                  else None),
                "source": str(candidate),
            }
    return None


def load_reference(path: Path, cfg, device: str):
    """(model, meta) from a checkpoint on disk. Meta carries the provenance.

    The model is built through SEAM.build_model so the expected parameter count
    is checked, and it is built for the family the CHECKPOINT names rather than
    the one the config was validated for -- the checkpoint is authoritative
    about what it holds.
    """
    payload = validate_payload(_torch_load(path), path)
    meta = {
        "path": str(path),
        "step": payload["step"],
        "kind": payload["kind"],
        "arm": payload["arm"],
        "seed": payload["seed"],
        "family": payload["family"],
        "micro_batch": payload.get("micro_batch"),
        "carries_rng": payload["kind"] == CHECKPOINT_FULL,
        "model_sha256": state_sha256(payload["model"]),
        "trained_at": _trained_at(path),
    }
    family = payload["family"]
    state = payload["model"]
    # THE PAYLOAD GOES BEFORE THE MODEL IS BUILT, and the order matters. A full
    # checkpoint carries AdamW's two moment buffers -- another ~995 MB at 124M
    # fp32, which no metric in this file reads. Holding the payload across
    # build_model and load_state_dict keeps those resident while a second
    # ~497 MB of weights is allocated. Everything still needed is already in
    # `meta` or bound above. Found by an adversarial audit.
    del payload
    model = SEAM.build_model(cfg, family)
    model.load_state_dict(state)
    del state
    model = model.to(device)
    model.eval()
    meta["model_dtype"] = str(next(model.parameters()).dtype)
    return model, meta


def load_junk(kind: str, seed: int, side: int, device: str):
    """(model, meta) for --selftest. No cluster path, no disk.

    `side` distinguishes the two members of a pair so a caller can build the
    left and right independently, which is what the zero check needs.
    """
    a, b = M.build_junk_pair(kind, seed=seed, tiny=True)
    model = (a if side == 0 else b).to(device)
    model.eval()
    other = b if side == 0 else a
    del other
    meta = {
        "path": f"JUNK:{kind}/seed{seed}/side{side}",
        "step": None,
        "kind": "JUNK (no checkpoint)",
        "arm": f"JUNK-{kind}",
        "seed": seed,
        "family": SEAM.FAMILY_HF_GPT2,
        "micro_batch": None,
        "carries_rng": False,
        "model_sha256": state_sha256(model.state_dict()),
        "trained_at": None,
        "model_dtype": str(next(model.parameters()).dtype),
    }
    return model, meta


# ---------------------------------------------------------------------------
# The pair set
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pair:
    """One row of the ladder. `loader_a`/`loader_b` are zero-arg thunks.

    `expect_a`/`expect_b` are the (seed, arm) the LABEL claims each side is, and
    they exist so the claim can be checked against what the checkpoint payload
    actually says rather than against the path the label was built from. A label
    that names one run while the file holds another is the S55 shape and would
    not crash: every number would be computed correctly, about the wrong pair.
    None means "do not check", which only the junk fixtures use.
    """

    label: str
    ref_a: str
    ref_b: str
    loader_a: object
    loader_b: object
    note: str
    expect_a: object = None
    expect_b: object = None


def verify_pair_identity(pair: Pair, meta_a: dict, meta_b: dict) -> dict:
    """The label's claim about (seed, arm), checked against the RESOLVED payload.

    Refuses rather than warns. `checkpoint_path` builds a path from a seed and an
    arm, but nothing downstream re-reads whether the file at that path holds what
    the name says -- `train.py` writes seed and arm INTO the payload precisely so
    that question has an answer, and this is the place to ask it.
    """
    resolved = {}
    problems = []
    for side, meta, expected in (("a", meta_a, pair.expect_a),
                                 ("b", meta_b, pair.expect_b)):
        got = (meta.get("seed"), meta.get("arm"))
        resolved[side] = {"seed": got[0], "arm": got[1],
                          "path": meta.get("path")}
        if expected is None:
            continue
        resolved[side]["expected_seed"] = expected[0]
        resolved[side]["expected_arm"] = expected[1]
        if got != tuple(expected):
            problems.append(
                f"pair {pair.label!r} side {side} is labelled seed "
                f"{expected[0]} arm {expected[1]!r}, but the checkpoint at "
                f"{meta.get('path')} records seed {got[0]} arm {got[1]!r}")
    if problems:
        raise LadderError(
            "A PAIR LABEL DOES NOT MATCH THE CHECKPOINT IT RESOLVED TO.\n"
            "Every number for this pair would be computed correctly and filed "
            "under the wrong name, which is the defect this repo keeps finding "
            "in itself.\n\n"
            + "".join(f"  - {p}\n" for p in problems))
    return resolved


def ladder_steps(cfg) -> dict:
    """The two steps the ladder reads, DERIVED rather than typed.

    199 and 9535 are not written down here. The final step is cfg.last_step,
    which the final-step rule forces to a full checkpoint whatever the
    intervals say. The pre-injection step is found by walking back from
    injection_step to the first step at which cfg.checkpoint_kind_at fires --
    cfg.checkpoint_kind_at being the single definition of the schedule
    (burst/config.py:581-594). Both move if the config moves.
    """
    final_step = cfg.last_step
    final_kind = cfg.checkpoint_kind_at(final_step)
    injection_step = cfg.injection.injection_step
    if injection_step is None:
        raise LadderError(
            "injection.injection_step is null, so there is no pre-injection "
            "checkpoint to locate.")

    pre_step = None
    for step in range(min(injection_step, final_step), -1, -1):
        if cfg.checkpoint_kind_at(step) is not None:
            pre_step = step
            break
    if pre_step is None:
        raise LadderError(
            f"no checkpoint is scheduled at or before injection step "
            f"{injection_step}, so the training-scale pair has no left end.")
    return {
        "final_step": final_step,
        "final_kind": final_kind,
        "pre_injection_step": pre_step,
        "pre_injection_kind": cfg.checkpoint_kind_at(pre_step),
        "injection_step": injection_step,
        "DERIVED": ("both steps come from cfg.checkpoint_kind_at and "
                    "cfg.last_step, never from a literal"),
    }


def checkpoint_path(runs_root: Path, seed: int, arm: str, step: int,
                    kind: str) -> Path:
    """runs_root/seedNN_arm/stepNNNNNN_kind.pt -- the launcher's own naming.

    The run directory name comes from burst.config.run_name_for and the file
    name mirrors scripts/train.py:599, so this cannot drift from what the loop
    actually wrote.
    """
    return runs_root / run_name_for(seed, arm) / f"step{step:06d}_{kind}.pt"


#: The v2 run set. Each entry is (label, (seed, arm, when), (seed, arm, when),
#: note), where `when` is "final" or "pre". DATA, not code, so the mutation audit
#: has one thing to break and the label-integrity check has one thing to read.
#:
#: v1 had three floor pairs across seeds 0/1/2 and one injecting arm. v2 drops
#: seed02_twin and adds seed00_fluent-false, buying a SECOND injecting arm at a
#: matched seed and an arm-against-arm pair, at the cost of two floor pairs. The
#: floor is therefore n=1 and has no spread; the banner says so.
PAIR_SPEC_V2 = (
    ("zero_check", (0, "twin", "final"), (0, "twin", "final"),
     "the same checkpoint, loaded twice independently. The gate."),
    ("effect_random_chars", (0, "twin", "final"), (0, "random-chars", "final"),
     "random-chars against its seed-matched twin: shared init, shared data "
     "order, one burst at the injection step"),
    ("effect_fluent_false", (0, "twin", "final"), (0, "fluent-false", "final"),
     "fluent-false against the SAME twin: the second injecting arm at a "
     "matched seed"),
    ("arm_vs_arm", (0, "random-chars", "final"), (0, "fluent-false", "final"),
     "the two injecting arms against each other, at one seed"),
    ("floor_s0_s1", (0, "twin", "final"), (1, "twin", "final"),
     "twin against twin across seeds. THE ONLY FLOOR PAIR in v2"),
    ("training_scale", (0, "twin", "pre"), (0, "twin", "final"),
     "one run against its own earlier self: the pre-injection checkpoint "
     "against the finished model"),
)


def required_run_dirs(spec=PAIR_SPEC_V2) -> tuple:
    """Every seedNN_arm directory the spec needs, de-duplicated, in order."""
    seen, out = set(), []
    for _label, a, b, _note in spec:
        for seed, arm, _when in (a, b):
            name = run_name_for(seed, arm)
            if name not in seen:
                seen.add(name)
                out.append(name)
    return tuple(out)


def build_checkpoint_pairs(runs_root: Path, cfg, device: str,
                           spec=PAIR_SPEC_V2):
    """The six v2 pairs, over real checkpoints. Refuses a missing run loudly.

    The missing-run check runs BEFORE the steps are derived, deliberately: a run
    directory that is not there is a more fundamental problem than which step to
    read inside it, and the more useful error to surface first. Same ordering
    burst/config.py uses for null fields against the family refusal.
    """
    # FAIL LOUD, AND NEVER FALL BACK. A v1 run directory has seed02_twin and no
    # fluent-false; a v2 one is the other way round. Quietly measuring whichever
    # subset happens to be present would produce a complete-looking artifact
    # over a different pair set than its labels claim.
    missing = []
    for name in required_run_dirs(spec):
        if not (runs_root / name).is_dir():
            missing.append(runs_root / name)
    if missing:
        raise LadderError(
            f"{len(missing)} of {len(required_run_dirs(spec))} run "
            "directories the v2 pair set needs are not present under "
            f"{runs_root}.\n"
            "EVERY PATH SEARCHED:\n"
            + "".join(
                f"  {'MISSING' if (runs_root / n) in missing else 'found  '}  "
                f"{runs_root / n}\n" for n in required_run_dirs(spec))
            + "\nNO FALLBACK IS ATTEMPTED. This is the v2 set: seed02_twin was "
            "dropped and seed00_fluent-false added, so a v1 output directory "
            "will be missing fluent-false and will still contain a seed02_twin "
            "this ladder no longer reads. Measuring whichever subset happened "
            "to be there would produce a complete-looking artifact over a pair "
            "set its own labels misdescribe.")

    steps = ladder_steps(cfg)
    when_step = {
        "final": (steps["final_step"], steps["final_kind"]),
        "pre": (steps["pre_injection_step"], steps["pre_injection_kind"]),
    }

    def path_for(seed, arm, when):
        step, kind = when_step[when]
        return checkpoint_path(runs_root, seed, arm, step, kind)

    def ref(path):
        return str(Path(path).relative_to(runs_root)).replace("\\", "/")

    def loader(path):
        return lambda: load_reference(path, cfg, device)

    pairs = []
    for label, a, b, note in spec:
        pa, pb = path_for(*a), path_for(*b)
        pairs.append(Pair(
            label=label, ref_a=ref(pa), ref_b=ref(pb),
            loader_a=loader(pa), loader_b=loader(pb), note=note,
            expect_a=(a[0], a[1]), expect_b=(b[0], b[1])))
    return pairs, steps


def build_selftest_pairs(device: str):
    """The same six labels over junk fixtures. Exercises all three pair kinds.

    identical, noise and independent all appear, so the selftest walks the same
    code path a real run walks -- including the zero check, which is what makes
    a local verification meaningful before Asa points this at the cluster.
    """
    def loader(kind, seed, side):
        return lambda: load_junk(kind, seed, side, device)

    spec = [
        ("zero_check", "identical", 101,
         "two independent builds of one junk model. The gate."),
        ("effect_random_chars", "noise", 202,
         "JUNK stand-in for arm against twin: same init, small perturbation"),
        ("effect_fluent_false", "noise", 707,
         "JUNK stand-in for the second injecting arm at a matched seed"),
        ("arm_vs_arm", "noise", 808,
         "JUNK stand-in for the two injecting arms against each other"),
        ("floor_s0_s1", "independent", 303,
         "JUNK stand-in for the ONE floor pair: independent init"),
        ("training_scale", "noise", 606,
         "JUNK stand-in for early against late: same init, perturbation"),
    ]
    pairs = []
    for label, kind, seed, note in spec:
        if kind == "identical":
            # Two independent BUILDS, not a deepcopy: the gate has to exercise
            # the load path twice, and a deepcopy would prove less.
            la = loader("identical", seed, 0)
            lb = loader("identical", seed, 0)
        else:
            la, lb = loader(kind, seed, 0), loader(kind, seed, 1)
        pairs.append(Pair(label=label,
                          ref_a=f"JUNK:{kind}/seed{seed}/a",
                          ref_b=f"JUNK:{kind}/seed{seed}/b",
                          loader_a=la, loader_b=lb, note=note))
    return pairs


def distinct_refs(pairs):
    """Ordered, de-duplicated (ref, loader) over every pair. For the loss pass."""
    seen, out = set(), []
    for pair in pairs:
        for ref, loader in ((pair.ref_a, pair.loader_a),
                            (pair.ref_b, pair.loader_b)):
            if ref not in seen:
                seen.add(ref)
                out.append((ref, loader))
    return out


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------


def describe_module_structure(model) -> dict:
    """What the tap helper actually found. For a LOUD tap failure.

    hf_gpt2_tap_modules reads .transformer.drop / .transformer.h /
    .transformer.ln_f and raises MetricsError naming none of them. "Failed" is
    not actionable; the structure that was there is.
    """
    info = {
        "model_type": type(model).__name__,
        "top_level_children": [n for n, _ in model.named_children()],
        "has_transformer": hasattr(model, "transformer"),
    }
    transformer = getattr(model, "transformer", None)
    if transformer is not None:
        info["transformer_children"] = [n for n, _ in
                                        transformer.named_children()]
        info["has_drop"] = hasattr(transformer, "drop")
        info["has_h"] = hasattr(transformer, "h")
        info["has_ln_f"] = hasattr(transformer, "ln_f")
        blocks = getattr(transformer, "h", None)
        if blocks is not None:
            try:
                info["n_blocks"] = len(blocks)
            except TypeError:
                info["n_blocks"] = None
    return info


def tap_or_raise(model):
    """hf_gpt2_tap_modules, with the model's real structure on failure."""
    try:
        return M.hf_gpt2_tap_modules(model)
    except M.MetricsError as exc:
        raise LadderError(
            "the activation tap helper refused this model, so phase 2 cannot "
            "run on it.\n"
            f"metrics.hf_gpt2_tap_modules said: {exc}\n"
            "THE STRUCTURE IT ACTUALLY FOUND:\n"
            + json.dumps(describe_module_structure(model), indent=2)
        ) from exc


def activations_for(model, batch):
    """Per-layer activations by the HOOKS route, so the tap list is exercised.

    route="hooks" rather than the default "native": the tap list is the one
    real coupling to model structure in metrics.py, and going through it here
    is what makes tap_or_raise's failure path reachable rather than decorative.
    The two routes are proved to agree once, in the phase-2 gate.
    """
    tap_or_raise(model)
    return M.layer_activations(model, batch, route="hooks")


def pair_activation_metrics(model_a, model_b, batch) -> dict:
    """Per-layer CKA and per-layer cosine, as FULL lists over layers."""
    acts_a = activations_for(model_a, batch)
    acts_b = activations_for(model_b, batch)
    if len(acts_a) != len(acts_b):
        raise LadderError(
            f"layer counts differ: {len(acts_a)} against {len(acts_b)}")
    cka = M.per_layer_cka(acts_a, acts_b)
    cosine = M.per_layer_activation_similarity(acts_a, acts_b)
    result = {
        "n_layers": len(acts_a),
        "activation_shape": list(acts_a[0].shape),
        "cka_variant": M.CKA_VARIANT,
        "cka": cka,
        "cosine": cosine,
        "READ_THE_LIST": (
            "every layer is reported. A summary across layers would hide the "
            "shape of where two models differ, which is the only thing a "
            "per-layer metric adds over a scalar."),
    }
    del acts_a, acts_b
    return result


# ---------------------------------------------------------------------------
# The zero check -- two halves, gating one phase each
# ---------------------------------------------------------------------------


def gate_checkpoint(model, meta, batch, *,
                    tolerate_indeterminate: bool) -> tuple:
    """BOTH objective gates for ONE checkpoint. Returns (cell, failures).

    Extracted so `objective_gate` here and `scripts/pilot_barrier.py` run the
    SAME code rather than two copies that drift. pilot_barrier imports this; it
    does not reimplement it. A safety check with two implementations is worse
    than one, because the second is the one nobody re-reads.

    Raises nothing itself -- the caller decides whether a failure aborts one
    checkpoint or the whole run, which is the only part that legitimately
    differs between the two callers.
    """
    next_token = M.cross_entropy_loss(model, batch)
    two_ahead = two_ahead_loss(model, batch)
    cell = classify_objective(next_token, two_ahead)

    trained = meta.get("trained_at") or {}
    commit = trained.get("commit")
    refused_by_commit = [c for c in REFUSED_TRAINING_COMMITS
                         if commit and str(commit).startswith(c)]
    ancestry = objective_fix_ancestry(commit)
    cell.update({
        "gate_a_objective": cell["objective"],
        "gate_b_recorded_commit": commit,
        "gate_b_refused": bool(refused_by_commit)
                          or ancestry["state"] == ANCESTRY_PREDATES_FIX,
        "gate_b_matched": refused_by_commit or None,
        "gate_b_ancestry": ancestry,
        "gate_b_suspect": ancestry["state"] != ANCESTRY_CONTAINS_FIX,
    })

    ref = meta.get("path", "<unnamed checkpoint>")
    failures = []
    if cell["objective"] == OBJECTIVE_TWO_AHEAD:
        failures.append(
            f"GATE A fired on {ref}: next-token loss {next_token:.6f} against "
            f"two-ahead {two_ahead:.6f}. The two-ahead objective scores "
            f"{cell['gap_next_minus_two_ahead']:.6f} nats BETTER, past a margin "
            f"of {DOUBLE_SHIFT_MARGIN_NATS}, so these weights were optimised to "
            "predict two tokens ahead (S97).")
    elif (cell["objective"] == OBJECTIVE_INDETERMINATE
            and not tolerate_indeterminate):
        failures.append(
            f"GATE A is INDETERMINATE on {ref}: next-token {next_token:.6f} "
            f"against two-ahead {two_ahead:.6f}, a gap of "
            f"{cell['gap_next_minus_two_ahead']:.6f} nats inside the "
            f"{DOUBLE_SHIFT_MARGIN_NATS}-nat margin. Both objectives score the "
            "same, which is what an UNTRAINED model looks like -- there is no "
            "displacement worth measuring between checkpoints like this, and "
            "calling it a pass would defeat the gate.")
    if refused_by_commit:
        failures.append(
            f"GATE B fired on {ref}: run_provenance.yaml records training "
            f"commit {commit}, which starts with {refused_by_commit[0]!r}. That "
            "is the v1 pilot, and S97 proved every run from it trained the "
            "double-shifted objective.")
    elif ancestry["state"] == ANCESTRY_PREDATES_FIX:
        failures.append(
            f"GATE B fired on {ref} by ANCESTRY: {ancestry['reason']}. Derived "
            "from the commit graph rather than a name, so a pre-fix commit "
            "nobody thought to list is caught too.")
    return cell, failures


def objective_gate(pairs, batch, stream, *, tolerate_indeterminate: bool) -> dict:
    """TWO INDEPENDENT GATES on the training objective. Runs before everything.

    S97: the v1 pilot pre-shifted each block and then passed `labels=` to a path
    that shifts again, so every v1 checkpoint was optimised to predict two
    tokens ahead. A ladder pointed at such a checkpoint computes correct
    distances between models of the wrong objective, which is the worst kind of
    wrong number -- it is right about something nobody asked.

    GATE A reads the ARTIFACT: score next-token and two-ahead loss on the same
    batch and see which the weights prefer.
    GATE B reads a CLAIM ABOUT the artifact: the training commit recorded in
    run_provenance.yaml, against REFUSED_TRAINING_COMMITS.

    Both always run and both are reported. They are NOT redundant, and treating
    them as such is the mistake S97 is a monument to: a claim and the thing it
    describes can disagree. Gate B catches a v1 checkpoint whose losses are for
    some reason ambiguous; gate A catches a double-shifted checkpoint from any
    commit at all, including one nobody has thought to add to the list.

    This pass is also where per-model loss and provenance are taken, so each
    distinct checkpoint is loaded ONCE here and holds one model at a time.
    """
    _say(stream, "  objective gate (A: measured losses, B: recorded commit)")
    checkpoints, objective, losses = {}, {}, {}
    grouping = None
    failures = []

    for ref, loader in distinct_refs(pairs):
        model, meta = loader()
        view = None
        try:
            view = M.parameter_view(model)
            if grouping is None:
                grouping = grouping_shape(view)
            # BOTH gates, via the shared per-checkpoint function. The losses it
            # needs are the losses this pass records, so nothing is scored twice.
            cell, cell_failures = gate_checkpoint(
                model, dict(meta, path=ref), batch,
                tolerate_indeterminate=tolerate_indeterminate)
        finally:
            del model, view

        failures.extend(cell_failures)
        ancestry = cell["gate_b_ancestry"]
        next_token = cell["loss_next_token"]
        two_ahead = cell["loss_two_ahead"]
        objective[ref] = cell
        checkpoints[ref] = meta
        losses[ref] = {"loss": next_token, "loss_two_ahead": two_ahead,
                       "n_tokens": batch.n_tokens, "source": batch.source}

        _say(stream, f"    {ref:<44} next={next_token:.6f} "
                     f"two_ahead={two_ahead:.6f} -> {cell['objective']}"
                     f"  ancestry={ancestry['state']}"
                     + ("  COMMIT REFUSED" if cell["gate_b_refused"] else ""))

    if failures:
        raise LadderError(
            "THE OBJECTIVE GATE FAILED AND THE RUN IS ABANDONED. Nothing is "
            "written.\n"
            "A checkpoint trained on the wrong objective yields distances that "
            "are correctly computed and about the wrong thing. See S97 and "
            "docs/2026-08-05-training-objective-defect.md.\n\n"
            + "".join(f"  - {f}\n" for f in failures)
            + "\nPER CHECKPOINT:\n"
            + json.dumps(objective, indent=2, default=repr))

    return {
        "passed": True,
        "gate_a": ("measured: next-token loss against two-ahead loss on the "
                   "same batch, margin "
                   f"{DOUBLE_SHIFT_MARGIN_NATS} nats"),
        "gate_b": ("recorded: run_provenance.yaml training commit, checked two "
                   f"ways -- explicitly against {list(REFUSED_TRAINING_COMMITS)}, "
                   f"and by ANCESTRY against {OBJECTIVE_FIX_COMMIT} (the commit "
                   "that fixed the double shift). Ancestry catches a pre-fix "
                   "commit nobody listed; the explicit list still works with no "
                   "git. UNKNOWN ancestry warns rather than aborting."),
        "WHY_BOTH": ("gate A reads the artifact and gate B reads a claim about "
                     "it. S97 exists because those can disagree, so neither "
                     "substitutes for the other."),
        "tolerate_indeterminate": tolerate_indeterminate,
        "by_checkpoint": objective,
    }, checkpoints, losses, grouping


def gate_phase1(pair: Pair, batch) -> dict:
    """Exact identity invariants for phase 1. Raises on any deviation.

    Runs FIRST and gates everything. The three assertions here are all exact
    and were verified achievable at real scale before this file was written.
    """
    model_a, meta_a = pair.loader_a()
    model_b, meta_b = pair.loader_b()
    try:
        view_a, view_b = M.parameter_view(model_a), M.parameter_view(model_b)
        l2 = per_layer_l2(view_a, view_b)
        loss_a = M.cross_entropy_loss(model_a, batch)
        loss_b = M.cross_entropy_loss(model_b, batch)
        observed = {
            "l2_total": l2["l2_total"],
            "l2_by_bucket_max": max(c["l2"] for c in l2["l2_by_bucket"].values()),
            "loss_a": loss_a,
            "loss_b": loss_b,
            "loss_difference": loss_a - loss_b,
            "state_sha256_a": meta_a["model_sha256"],
            "state_sha256_b": meta_b["model_sha256"],
        }
        failures = []
        # Collected as a failure rather than raised early, so EVERY gate
        # failure has one shape and carries the same OBSERVED block. An
        # early-out here gave the most diagnostic case -- two loads that read
        # different weights -- the least diagnostic message.
        if meta_a["model_sha256"] != meta_b["model_sha256"]:
            failures.append(
                "the two loads did not read the same weights: state sha256 "
                f"{meta_a['model_sha256'][:16]} against "
                f"{meta_b['model_sha256'][:16]}")
        if l2["l2_total"] != 0.0:
            failures.append(
                f"total L2 is {l2['l2_total']!r}, not exactly 0.0")
        for bucket, cell in l2["l2_by_bucket"].items():
            if cell["l2"] != 0.0:
                failures.append(
                    f"bucket {bucket!r} L2 is {cell['l2']!r}, not exactly 0.0")
        if loss_a != loss_b:
            failures.append(
                f"per-model loss differs between two loads of one checkpoint: "
                f"{loss_a!r} against {loss_b!r}")
        if failures:
            raise LadderError(_gate_failure_text("phase 1", failures, observed))
        return {
            "passed": True,
            "pair": pair.label,
            "asserted": [
                "total L2 exactly 0.0",
                "per-bucket L2 exactly 0.0 in every bucket",
                "per-model loss bitwise equal between two independent loads",
                "payload['model'] sha256 equal between the two loads",
            ],
            "observed": observed,
            "l2_by_bucket": l2["l2_by_bucket"],
            "sum_of_parts_check": l2["sum_of_parts_check"],
        }
    finally:
        del model_a, model_b


def gate_phase2(pair: Pair, batch) -> dict:
    """Identity invariants for phase 2. Raises on any deviation.

    Separate from gate_phase1 on purpose. A tap failure is a phase-2 failure
    and must not cost the phase-1 numbers, so the activation invariants are
    gated here rather than up front.
    """
    model_a, meta_a = pair.loader_a()
    model_b, _ = pair.loader_b()
    torch = M._torch()
    try:
        # tap_or_raise FIRST, and the ordering is the whole point. It used to
        # come second, behind cross_check_activation_routes -- which itself
        # calls M.hf_gpt2_tap_modules and raises a bare MetricsError naming no
        # structure (scripts/metrics.py:625). So on a model the tap cannot
        # read, the loud structure dump this script exists to print was
        # UNREACHABLE from main: the route check always got there first. Found
        # by an adversarial audit.
        tap_or_raise(model_a)
        tap_or_raise(model_b)
        routes = M.cross_check_activation_routes(model_a, batch)
        acts_a = activations_for(model_a, batch)
        acts_b = activations_for(model_b, batch)
        bitwise = [bool(torch.equal(x, y)) for x, y in zip(acts_a, acts_b)]
        cka = [row["cka"] for row in M.per_layer_cka(acts_a, acts_b)]
        cosine_rows = M.per_layer_activation_similarity(acts_a, acts_b)
        cos_dev = max(abs(r["cosine_min"] - 1.0) for r in cosine_rows)
        cos_med_dev = max(abs(r["cosine_median"] - 1.0) for r in cosine_rows)
        norm_dev = max(abs(r["norm_ratio_median"] - 1.0) for r in cosine_rows)
        observed = {
            "n_layers": len(acts_a),
            "activations_bitwise_equal": bitwise,
            "cka_max_deviation_from_one": max(abs(v - 1.0) for v in cka),
            "cosine_min_max_deviation_from_one": cos_dev,
            "cosine_median_max_deviation_from_one": cos_med_dev,
            "cosine_deviation_in_ulps": cos_dev / DBL_EPS,
            "norm_ratio_max_deviation_from_one": norm_dev,
            "route_cross_check": routes,
        }
        failures = []
        if not all(bitwise):
            failures.append(
                "activations are not bitwise equal between two loads of one "
                f"checkpoint; layers differing: "
                f"{[i for i, ok in enumerate(bitwise) if not ok]}")
        for i, value in enumerate(cka):
            if value != 1.0:
                failures.append(
                    f"layer {i} CKA is {value!r}, not exactly 1.0")
        if norm_dev != 0.0:
            failures.append(
                f"norm ratio deviates from 1.0 by {norm_dev!r}, not exactly 0")
        if cos_dev > COSINE_IDENTITY_TOL:
            failures.append(
                f"per-layer cosine deviates from 1.0 by {cos_dev!r} "
                f"({cos_dev / DBL_EPS:.1f} ulps of double), above the "
                f"{COSINE_IDENTITY_ULPS}-ulp allowance")
        if failures:
            raise LadderError(_gate_failure_text("phase 2", failures, observed))
        del acts_a, acts_b
        return {
            "passed": True,
            "pair": pair.label,
            "asserted": [
                "activations bitwise equal in every layer (torch.equal)",
                "per-layer CKA exactly 1.0",
                "norm ratio exactly 1.0",
                f"per-layer cosine within {COSINE_IDENTITY_ULPS} ulps of 1.0",
                "the hooks and native activation routes agree",
            ],
            "observed": observed,
            "WHY_COSINE_IS_NOT_EXACT": (
                "for bitwise identical activations the numerator sum(a_i * a_i) "
                "and the denominator norm(a) * norm(a) reduce the same products "
                "in different orders, so they differ by an ulp or two. This "
                "tolerance is not load-bearing: it sits behind bitwise "
                "activation equality, exact CKA and an exact norm ratio, and a "
                "broken runner fails one of those first."),
        }
    finally:
        del model_a, model_b


#: What a gate failure costs, PER PHASE. A phase-2 failure must not read as
#: condemning phase 1: phase 1 passed its own gate and is already on disk. The
#: unscoped wording used to be written into the artifact directly beneath phase 1
#: published as ok, which an audit caught.
_GATE_SCOPE = {
    "phase 1": "AND THE RUN IS ABANDONED. Nothing is written.",
    "phase 2": ("AND PHASE 2 IS ABANDONED. Phase 1 passed its own gate and its "
                "numbers stand -- they are already on disk. No per-layer "
                "activation number was produced."),
}


def _gate_failure_text(phase: str, failures, observed) -> str:
    return (
        f"THE ZERO CHECK FAILED ({phase}) "
        f"{_GATE_SCOPE.get(phase, 'AND THAT PHASE IS ABANDONED.')}\n"
        "This pair is one checkpoint measured against itself, so every "
        "quantity below has a known answer. A deviation means the runner, the "
        "loader or the device is not doing what it claims.\n\n"
        + "".join(f"  - {f}\n" for f in failures)
        + "\nOBSERVED:\n"
        + json.dumps(observed, indent=2, default=repr)
    )


# ---------------------------------------------------------------------------
# The phases
# ---------------------------------------------------------------------------


def run_phase1(pairs, batch, stream) -> tuple:
    """L2, total and per layer, over every pair. Holds two models.

    Per-model loss and provenance are NOT taken here: the objective gate already
    walked every distinct checkpoint to score both objectives, and that pass is
    where the loss it needed anyway is recorded. Loading each file a second time
    to recompute a number already in the payload would be the only place in this
    script that read a checkpoint twice for one quantity.
    """
    _say(stream, "  phase 1: L2, total and per layer")
    pair_rows = {}
    for pair in pairs:
        model_a, meta_a = pair.loader_a()
        model_b, meta_b = pair.loader_b()
        try:
            # The label's claim about (seed, arm) against the RESOLVED payload,
            # before any number is filed under it.
            resolved = verify_pair_identity(pair, meta_a, meta_b)
            row = per_layer_l2(M.parameter_view(model_a),
                               M.parameter_view(model_b))
        finally:
            del model_a, model_b
        row.update({"a": pair.ref_a, "b": pair.ref_b, "note": pair.note,
                    "resolved_identity": resolved})
        pair_rows[pair.label] = row
        _say(stream, f"    {pair.label:<16} "
                     f"l2_total={row['l2_total']:.10g}")

    return pair_rows


def run_phase2(pairs, batch, stream) -> dict:
    """Per-layer cosine and per-layer CKA over every pair. Holds two models."""
    rows = {}
    for pair in pairs:
        model_a, _ = pair.loader_a()
        model_b, _ = pair.loader_b()
        try:
            row = pair_activation_metrics(model_a, model_b, batch)
        finally:
            del model_a, model_b
        row.update({"a": pair.ref_a, "b": pair.ref_b, "note": pair.note})
        rows[pair.label] = row
        # LIVENESS ONLY, and deliberately carrying no reduced number.
        # This line used to print min(cka) across layers, which made stdout an
        # aligned one-scalar-per-pair column a reader reads effect-against-floor
        # straight off -- and because it existed only on stdout,
        # assert_no_verdict_keys (which walks the payload) could not see it. A
        # verdict surface the guard cannot reach is worse than one it can. The
        # full per-layer lists are in the artifact; the script does not hand
        # over a pre-reduced summary aligned for comparison.
        _say(stream, f"    {pair.label:<16} layers={row['n_layers']} "
                     f"cka and cosine computed")
    return rows


# ---------------------------------------------------------------------------
# The no-verdict guard
# ---------------------------------------------------------------------------


def assert_no_verdict_keys(payload) -> None:
    """Refuse to write a payload carrying a verdict key, at any depth.

    A STRUCTURAL guard rather than a convention. This file's job is to report
    distances; comparing them is a separate act with preconditions this script
    does not check and cannot -- D-4 is open and scripts/analysis.py still
    requires the family and the correction with no default. Making that a
    denylist means a verdict cannot be added here without deleting the guard,
    which is a visible edit rather than a quiet one.
    """
    def walk(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in FORBIDDEN_PAYLOAD_KEYS:
                    raise LadderError(
                        f"payload key {'.'.join(path + [str(key)])!r} is a "
                        "VERDICT, and this measurement computes none.\n"
                        f"Denied keys: {sorted(FORBIDDEN_PAYLOAD_KEYS)}\n"
                        "The ladder reports distances. Comparing the effect "
                        "pair against the floor pairs requires deciding the "
                        "family of tests and the correction, which is D-4 and "
                        "is open. If a comparison genuinely belongs here, "
                        "remove it from FORBIDDEN_PAYLOAD_KEYS deliberately "
                        "and record why.")
                walk(value, path + [str(key)])
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, path + [str(i)])

    walk(payload, [])


# ---------------------------------------------------------------------------
# Rendering -- mirrors scripts/canonicalization_error.py's report shape
# ---------------------------------------------------------------------------

_RULE = "=" * 78
_SUB = "-" * 78

#: Minimum width of a per-pair column. The real width is derived from the LABELS
#: -- v2's `effect_random_chars` is 19 characters and a hardcoded 16 ran the
#: headers together into `zero_checkeffect_random_chars`, which is a table a
#: reader cannot use.
_MIN_COL = 16


def _col_width(labels) -> int:
    return max([_MIN_COL] + [len(str(lab)) + 2 for lab in labels])


def format_report(payload) -> str:
    """The .md, GENERATED from the payload. Never hand-maintained.

    S70's lesson: a hand-written record inside a machine-generated file is
    deleted by the next run and the result looks machine-clean.
    """
    lines = [
        _RULE,
        "13. THE DISPLACEMENT LADDER -- L2, activation similarity, CKA, loss",
        _RULE,
        "",
        payload.get("LIMITATION", ""),
        "",
        "NO VERDICT IS COMPUTED HERE",
        _SUB,
        payload.get("NO_VERDICT", ""),
        "",
        "PRE-REGISTRATION ORDERING",
        _SUB,
        payload.get("ordering_constraint", ""),
        "",
        "RUN",
        _SUB,
        f"  fixture                {payload.get('fixture')}",
        f"  device                 {(payload.get('device') or {}).get('chosen')}"
        f"   (requested "
        f"{(payload.get('device') or {}).get('requested')}, cuda_available "
        f"{(payload.get('device') or {}).get('cuda_available')})",
        f"  measured at commit     {(payload.get('repo') or {}).get('commit')}"
        f"   (dirty {(payload.get('repo') or {}).get('dirty')})",
        f"  cka variant            {payload.get('cka_variant')}",
    ]

    batch = payload.get("batch") or {}
    lines += [
        f"  batch                  {batch.get('source')}",
        f"  batch tokens           {batch.get('n_tokens')}"
        f"   (tokenizer {batch.get('tokenizer')})",
        f"  batch token sha256     {batch.get('token_sha256')}",
    ]

    steps = payload.get("steps") or {}
    if steps:
        lines += [
            "",
            "STEPS -- derived from the config, never typed",
            _SUB,
            f"  injection step         {steps.get('injection_step')}",
            f"  pre-injection ckpt     {steps.get('pre_injection_step')}"
            f"   ({steps.get('pre_injection_kind')})",
            f"  final ckpt             {steps.get('final_step')}"
            f"   ({steps.get('final_kind')})",
        ]

    lines += _objective_section(payload)
    lines += _gate_section(payload)
    lines += _checkpoint_section(payload)
    lines += _loss_section(payload)
    lines += _grouping_section(payload)
    lines += _phase1_section(payload)
    lines += _phase2_section(payload)

    lines += ["", "PROVENANCE", _SUB, f"  {payload.get('PROVENANCE', '')}"]
    return "\n".join(lines)


def _objective_section(payload) -> list:
    """The two objective gates, per checkpoint. Rendered before the zero check
    because it runs before it and because it is the gate a v1 checkpoint trips."""
    gate = payload.get("objective_gate") or {}
    if not gate:
        return ["", "THE OBJECTIVE GATE", _SUB, "  NOT REACHED"]
    lines = ["", "THE OBJECTIVE GATE -- two independent checks, S97", _SUB,
             f"  gate A   {gate.get('gate_a')}",
             f"  gate B   {gate.get('gate_b')}",
             f"  both run {gate.get('WHY_BOTH')}"]
    if gate.get("tolerate_indeterminate"):
        lines.append("  NOTE     indeterminate is tolerated in this run "
                     "(--selftest, junk models at chance)")
    lines += ["",
              f"  {'checkpoint':<44}{'next-token':>14}{'two-ahead':>14}"
              f"{'gap':>12}  objective"]
    for ref, cell in (gate.get("by_checkpoint") or {}).items():
        flag = "  COMMIT REFUSED" if cell.get("gate_b_refused") else ""
        lines.append(
            f"  {ref:<44}{cell.get('loss_next_token', float('nan')):>14.6f}"
            f"{cell.get('loss_two_ahead', float('nan')):>14.6f}"
            f"{cell.get('gap_next_minus_two_ahead', float('nan')):>12.6f}"
            f"  {cell.get('objective')}{flag}")
    lines += ["",
              "  gap is next-token minus two-ahead. POSITIVE past the margin "
              "means the",
              "  two-ahead objective fits better, which is what S97's defect "
              "looks like."]
    return lines


def _gate_section(payload) -> list:
    gate = payload.get("gate") or {}
    lines = ["", "THE ZERO CHECK -- one checkpoint against itself, and the only "
             "proof this ran", _SUB]
    for phase in ("phase1", "phase2"):
        cell = gate.get(phase)
        if not cell:
            lines.append(f"  {phase:<22} NOT REACHED")
            continue
        lines.append(f"  {phase:<22} {'PASSED' if cell.get('passed') else 'FAILED'}"
                     f"   (pair {cell.get('pair')!r})")
        if not cell.get("passed"):
            for chunk in str(cell.get("reason", "")).splitlines():
                lines.append(f"      {chunk}")
            if cell.get("WHAT_THIS_COSTS"):
                lines.append(f"      costs: {cell['WHAT_THIS_COSTS']}")
        for claim in cell.get("asserted", []):
            lines.append(f"      asserted: {claim}")
        observed = cell.get("observed") or {}
        for key in sorted(observed):
            value = observed[key]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                lines.append(f"      {key:<40} {value!r}")
    return lines


def _checkpoint_section(payload) -> list:
    lines = ["", "CHECKPOINTS -- sha256 is over payload['model'] ONLY, never the "
             "file", _SUB]
    for ref, meta in (payload.get("checkpoints") or {}).items():
        lines += [
            f"  {ref}",
            f"      step {meta.get('step')}  kind {meta.get('kind')!r}  "
            f"arm {meta.get('arm')!r}  seed {meta.get('seed')}  "
            f"family {meta.get('family')!r}",
            f"      carries_rng {meta.get('carries_rng')}  "
            f"dtype {meta.get('model_dtype')}",
            f"      model_sha256 {meta.get('model_sha256')}",
        ]
        trained = meta.get("trained_at")
        if trained:
            lines.append(
                f"      trained_at commit {trained.get('commit')}  "
                f"dirty {trained.get('dirty')}")
    return lines


def _loss_section(payload) -> list:
    lines = ["", "PER-MODEL LOSS -- its own section, not a pair quantity", _SUB]
    for ref, cell in (payload.get("per_model_loss") or {}).items():
        lines.append(f"  {ref:<48}{cell['loss']:>22.12g}")
    if lines[-1] != _SUB:
        lines += ["", "  The barrier discards these; they are the quantity that "
                  "carried signal in the pilot."]
    return lines


def _grouping_section(payload) -> list:
    grouping = payload.get("parameter_grouping") or {}
    if not grouping:
        return []
    lines = ["", "PARAMETER GROUPING -- parameter_view, so a tied embedding is "
             "counted once", _SUB,
             f"  total parameters       {grouping.get('total_params')}",
             f"  total elements         {grouping.get('total_elements')}",
             f"  buckets                {grouping.get('n_buckets')}",
             "",
             f"  {'bucket':<16}{'params':>10}{'elements':>16}"]
    by_p = grouping.get("n_params_by_bucket") or {}
    by_e = grouping.get("n_elements_by_bucket") or {}
    for bucket in grouping.get("buckets", []):
        lines.append(f"  {bucket:<16}{by_p.get(bucket):>10}"
                     f"{by_e.get(bucket):>16}")
    return lines


def _phase1_section(payload) -> list:
    phase = payload.get("phase1") or {}
    lines = ["", f"PHASE 1 -- L2, total and per layer   "
             f"[attempted {phase.get('attempted')}, ok {phase.get('ok')}]", _SUB]
    if phase.get("reason"):
        lines.append(f"  reason: {phase['reason']}")
    pairs = phase.get("pairs") or {}
    if pairs:
        pad = _col_width(pairs)
        lines.append(f"  {'pair':<{pad}}{'total L2':>20}"
                     f"{'sum-of-parts rel diff':>26}")
        for label, row in pairs.items():
            check = row.get("sum_of_parts_check") or {}
            lines.append(f"  {label:<{pad}}{row['l2_total']:>20.12g}"
                         f"{check.get('relative_difference', float('nan')):>26.3e}")
        buckets = payload.get("parameter_grouping", {}).get("buckets", [])
        lines += ["", "  PER-LAYER L2", ""]
        lines.append(f"  {'bucket':<16}"
                     + "".join(f"{lab:>{pad}}" for lab in pairs))
        for bucket in buckets:
            row = f"  {bucket:<16}"
            for label in pairs:
                cell = pairs[label]["l2_by_bucket"].get(bucket, {})
                row += f"{cell.get('l2', float('nan')):>{pad}.8g}"
            lines.append(row)
    return lines


def _phase2_section(payload) -> list:
    phase = payload.get("phase2") or {}
    lines = ["", f"PHASE 2 -- per-layer activation cosine and per-layer CKA   "
             f"[attempted {phase.get('attempted')}, ok {phase.get('ok')}]", _SUB]
    if phase.get("reason"):
        for chunk in str(phase["reason"]).splitlines():
            lines.append(f"  {chunk}")
    pairs = phase.get("pairs") or {}
    if not pairs:
        lines.append("  no per-layer activation numbers were produced.")
        return lines
    pad = _col_width(pairs)
    n_layers = max(row["n_layers"] for row in pairs.values())

    def table(title, key):
        out = ["", f"  {title}", "",
               f"  {'layer':<8}" + "".join(f"{lab:>{pad}}" for lab in pairs)]
        for i in range(n_layers):
            row = f"  {i:<8}"
            for label in pairs:
                cells = pairs[label]["cka" if key == "cka" else "cosine"]
                row += (f"{cells[i][key]:>{pad}.10g}" if i < len(cells)
                        else f"{'--':>{pad}}")
            out.append(row)
        return out

    lines += table("PER-LAYER CKA (full list, every layer)", "cka")
    lines += table("PER-LAYER COSINE, median over token positions "
                   "(full list, every layer)", "cosine_median")
    lines += table("PER-LAYER NORM RATIO, median over token positions",
                   "norm_ratio_median")
    return lines


def build_provenance(payload) -> str:
    """Recomputed at write time from the payload. Never stored by hand."""
    device = payload.get("device") or {}
    repo = payload.get("repo") or {}
    batch = payload.get("batch") or {}
    phase2 = payload.get("phase2") or {}
    return (
        f"stem={payload.get('stem')}; fixture={payload.get('fixture')}; "
        f"device={device.get('chosen')}; "
        f"cuda_available={device.get('cuda_available')}; "
        f"measured_at_commit={repo.get('commit')}; dirty={repo.get('dirty')}; "
        f"batch={batch.get('source')}; "
        f"batch_token_sha256={batch.get('token_sha256')}; "
        f"n_pairs={len(payload.get('phase1', {}).get('pairs') or {})}; "
        f"n_checkpoints={len(payload.get('checkpoints') or {})}; "
        f"cka_variant={payload.get('cka_variant')}; "
        f"objective_gate_passed={(payload.get('objective_gate') or {}).get('passed')}; "
        f"double_shift_margin_nats={DOUBLE_SHIFT_MARGIN_NATS}; "
        f"phase2_ok={phase2.get('ok')}; "
        f"cosine_identity_ulps={COSINE_IDENTITY_ULPS}; "
        f"python={sys.version.split()[0]}"
    )


def two_ahead_loss(model, batch) -> float:
    """Loss under the DOUBLE-SHIFTED objective the v1 pilot actually trained.

    Transcribed from quantity (c) of the decisive experiment in
    docs/2026-08-05-training-objective-defect.md:245-256 -- "logits[:, :-1] of
    the *shortened* input, against raw[:, 2:]" -- so this gate compares against
    the same number S97 established, not against a re-derivation of it.

    Concretely, for a row t_0 .. t_1023: feed t_0 .. t_1022, take the logits at
    positions 0 .. 1021, and score them against t_2 .. t_1023. Element i pairs
    logits[i], conditioned on t_0 .. t_i, with t_(i+2).

    Next-token loss is NOT recomputed here: it is metrics.cross_entropy_loss,
    which is quantity (a) of the same experiment and one of the four loss paths
    S97 found shifting correctly. Two routes to the two objectives, one of them
    already committed and validated.
    """
    torch = M._torch()
    import torch.nn.functional as F

    ids = batch.input_ids()
    if ids.shape[-1] < 3:
        raise LadderError(
            f"the two-ahead objective needs at least 3 tokens; the batch has "
            f"{ids.shape[-1]}")
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits = model(input_ids=ids[:, :-1]).logits
            shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
            shift_labels = ids[:, 2:].reshape(-1)
            return float(F.cross_entropy(shift_logits.double(), shift_labels))
    finally:
        if was_training:
            model.train()


def classify_objective(next_token: float, two_ahead: float) -> dict:
    """Which objective a checkpoint's losses say it was trained on.

    The rule is a comparison with a margin, and the margin is why this is a
    function rather than an inline `<`. See DOUBLE_SHIFT_MARGIN_NATS.
    """
    gap = next_token - two_ahead
    if gap > DOUBLE_SHIFT_MARGIN_NATS:
        objective = OBJECTIVE_TWO_AHEAD
    elif gap < -DOUBLE_SHIFT_MARGIN_NATS:
        objective = OBJECTIVE_NEXT_TOKEN
    else:
        objective = OBJECTIVE_INDETERMINATE
    return {
        "objective": objective,
        "loss_next_token": next_token,
        "loss_two_ahead": two_ahead,
        "gap_next_minus_two_ahead": gap,
        "margin_nats": DOUBLE_SHIFT_MARGIN_NATS,
    }


def _say(stream, text: str) -> None:
    """print to `stream`, but never let a dead stream become the failure.

    A closed stdout (`... | head`) makes print raise OSError. Used only for
    diagnostics inside failure handlers and around the final report, where the
    artifact on disk is the record and the console is a convenience.
    """
    try:
        print(text, file=stream, flush=True)
    except OSError:
        pass


def _atomic_write(path: Path, text: str) -> None:
    """Write via a .partial sibling and os.replace, never in place.

    Mirrors scripts/train.py:331-333. `write_artifacts` is called TWICE over the
    same two paths -- once after phase 1 and once after phase 2 -- so an
    in-place write that failed partway through the second call would destroy the
    phase-1 artifact the first call created, which is the exact durability
    writing phase 1 early was meant to buy.
    """
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(text, encoding="utf-8", newline="\n")
    os.replace(partial, path)


def write_artifacts(payload, reportdir: Path, stem: str, stream) -> None:
    """Recompute derived fields, run the guard, then write BOTH files."""
    payload["stem"] = stem
    payload["LIMITATION"] = limitation(payload)
    payload["PROVENANCE"] = build_provenance(payload)
    assert_no_verdict_keys(payload)

    reportdir = Path(reportdir)
    reportdir.mkdir(parents=True, exist_ok=True)
    _atomic_write(reportdir / f"{stem}.json",
                  json.dumps(payload, indent=2, default=repr) + "\n")
    _atomic_write(reportdir / f"{stem}.md", format_report(payload) + "\n")
    _say(stream, f"  wrote {reportdir / (stem + '.json')}")
    _say(stream, f"  wrote {reportdir / (stem + '.md')}")


# ---------------------------------------------------------------------------
# Repo state
# ---------------------------------------------------------------------------


def repo_state() -> dict:
    """HEAD and a dirty flag for the MEASURING repo.

    Kept apart from each checkpoint's `trained_at`: the code that
    produced the weights and the code that measured them are two facts, and one
    name for both is how this repo's logged defect happens.
    """
    def git(*args):
        return subprocess.run(("git",) + args, cwd=str(REPO),
                              capture_output=True, text=True, timeout=30)
    try:
        head = git("rev-parse", "HEAD")
        status = git("status", "--porcelain")
        if head.returncode != 0:
            return {"commit": None, "dirty": None,
                    "error": head.stderr.strip() or "git rev-parse failed"}
        # BOTH exit codes are checked. An unchecked `git status` whose call
        # failed has empty stdout, which reads as "no changes" -- so the
        # artifact would record the measuring tree as CLEAN when git never said
        # so. `dirty: None` means unknown and is not the same fact as
        # `dirty: false`; burst/config.py warns loudly about exactly this
        # confusion for run provenance.
        if status.returncode != 0:
            return {"commit": head.stdout.strip() or None, "dirty": None,
                    "error": (status.stderr.strip()
                              or "git status --porcelain failed; whether the "
                                 "tree is clean is UNKNOWN, not clean")}
        dirty_lines = [ln for ln in status.stdout.splitlines() if ln.strip()]
        return {
            "commit": head.stdout.strip() or None,
            "dirty": bool(dirty_lines),
            "n_dirty_files": len(dirty_lines),
        }
    except Exception as exc:
        return {"commit": None, "dirty": None, "error": repr(exc)}


# ---------------------------------------------------------------------------
# Output resolution
# ---------------------------------------------------------------------------

_MEASUREMENTS_DIR = (REPO / "docs" / "measurements").resolve()


def resolve_output(out: Path, selftest: bool) -> tuple:
    """(reportdir, stem). Junk never lands in docs/measurements.

    --selftest numbers come from throwaway models. Filing them beside real
    measurements is the mistake docs/measurements/10-metrics.json's LIMITATION
    is written to prevent, so it is refused rather than discouraged.
    """
    reportdir = Path(out).resolve()
    stem = SELFTEST_STEM if selftest else STEM
    if selftest and reportdir == _MEASUREMENTS_DIR:
        raise LadderError(
            f"--selftest refuses to write into {_MEASUREMENTS_DIR}. Its "
            "numbers come from throwaway junk models and are a plumbing "
            "result, not a measurement; a plumbing result filed beside real "
            "measurements is indistinguishable from one. Point --out at a "
            "scratch directory.")
    return reportdir, stem


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/displacement_ladder.py",
        description=("L2, per-layer activation similarity, per-layer CKA and "
                     "per-model loss over a fixed pair set of checkpoints. "
                     "Computes no verdict."))
    parser.add_argument(
        "--runs-root", type=Path, default=None,
        help=("directory holding seedNN_arm/ run directories, e.g. "
              "/shared/27as66/burst-pilot/runs-fixed. NOT .../runs -- that "
              "is the void v1 pilot and the objective gate refuses it. "
              "Required unless --selftest."))
    parser.add_argument(
        "--out", type=Path, required=True,
        help=("where the JSON and MD land. An output path, so it is an "
              "argument and never a config value."))
    parser.add_argument(
        "--cfg-scratch", type=Path, default=None,
        help=("where load_config may write resolved_config.yaml. Must NOT be a "
              "run directory: those already hold the provenance of the "
              "training that produced these checkpoints. Required unless "
              "--selftest."))
    parser.add_argument("--config", type=Path,
                        default=REPO / "configs" / "base.yaml")
    parser.add_argument(
        "--run", type=Path,
        default=REPO / "configs" / "runs" / "seed00_twin.yaml",
        help=("any run override; build_model reads only cfg.model, which the "
              "study's central claim makes identical across every arm and "
              "seed."))
    parser.add_argument("--device", default="auto",
                        choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--selftest", action="store_true",
        help=("run the whole pipeline against metrics.build_junk_pair fixtures "
              "on this machine -- no cluster paths, no GPU needed. Exercises "
              "identical, noise and independent."))
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    stream = sys.stdout
    try:
        reportdir, stem = resolve_output(args.out, args.selftest)
        device_record = resolve_device(args.device)
        device = device_record["chosen"]

        if args.selftest:
            pairs = build_selftest_pairs(device)
            steps, fixture = {}, "JUNK MODELS (--selftest). NOT a measurement."
            batch = on_device(M.tiny_smoke_batch(), device)
        else:
            if args.runs_root is None:
                raise LadderError("--runs-root is required unless --selftest.")
            if args.cfg_scratch is None:
                raise LadderError(
                    "--cfg-scratch is required unless --selftest: load_config "
                    "must be given somewhere to write resolved_config.yaml "
                    "that is not a run directory.")
            cfg = load_config(args.config, args.run,
                              outdir=args.cfg_scratch,
                              write_provenance=False,
                              family=SEAM.FAMILY_HF_GPT2)
            pairs, steps = build_checkpoint_pairs(
                Path(args.runs_root).resolve(), cfg, device)
            fixture = "real checkpoints"
            batch = on_device(M.load_context_batch(), device)

        by_label = {p.label: p for p in pairs}
        if "zero_check" not in by_label:
            raise LadderError("the pair set has no zero_check pair to gate on.")
        gate_pair = by_label["zero_check"]

        payload = {
            "task": ("13. the displacement ladder -- L2, per-layer activation "
                     "similarity, per-layer CKA, and per-model loss"),
            # Recomputed at write time by `limitation(payload)`; this is only
            # the placeholder a reader would see if that ever stopped running.
            "LIMITATION": "",
            "objective_gate": {},
            "NO_VERDICT": (
                "No verdict is computed in this file. The effect pair is not "
                "compared against the floor pairs, no pair is ranked, and "
                "nothing is described as clearing anything. "
                "`assert_no_verdict_keys` runs over this payload before either "
                "file is written and refuses a verdict key, so the absence is "
                "enforced rather than promised. D-4 -- the family of tests and "
                "the multiple-comparison correction -- is open."),
            # NOT "ordering": that name is analysis.py's arm RANKING, and it is
            # on FORBIDDEN_PAYLOAD_KEYS. The guard caught the collision on its
            # first run, which is the guard working -- see S98.
            "ordering_constraint": ORDERING_ATTESTATION,
            "fixture": fixture,
            "device": device_record,
            "repo": repo_state(),
            "batch": batch.identity(),
            "cka_variant": M.CKA_VARIANT,
            "steps": steps,
            "pair_set": [
                {"label": p.label, "a": p.ref_a, "b": p.ref_b, "note": p.note}
                for p in pairs
            ],
            "gate": {},
            "checkpoints": {},
            "per_model_loss": {},
            "parameter_grouping": {},
            "phase1": {"attempted": False, "ok": False, "reason": None,
                       "pairs": {}},
            "phase2": {"attempted": False, "ok": False,
                       "reason": "not attempted", "pairs": {}},
        }

        _say(stream, _RULE)
        _say(stream, f"13. displacement ladder -- fixture: {fixture}")
        _say(stream, f"    device: {device}   pairs: {len(pairs)}")
        _say(stream, _RULE)

        # ---- THE OBJECTIVE GATE, before anything else. ----
        # Two independent checks per checkpoint, and this pass is also where
        # provenance and per-model loss are taken.
        gate, checkpoints, losses, grouping = objective_gate(
            pairs, batch, stream, tolerate_indeterminate=args.selftest)
        payload["objective_gate"] = gate
        payload["checkpoints"] = checkpoints
        payload["per_model_loss"] = losses
        payload["parameter_grouping"] = grouping or {}
        _say(stream, "    PASSED (both gates)")

        # ---- the zero check, phase 1 half. Aborts everything on failure. ----
        _say(stream, "  zero check (phase 1 half)")
        payload["gate"]["phase1"] = gate_phase1(gate_pair, batch)
        _say(stream, "    PASSED")

        # ---- phase 1 ----
        payload["phase1"]["attempted"] = True
        payload["phase1"].update(
            {"ok": True, "pairs": run_phase1(pairs, batch, stream)})

        # Written BEFORE phase 2 starts, so a phase-2 failure cannot cost these.
        write_artifacts(payload, reportdir, stem, stream)

        # ---- the gate, phase 2 half. Failure costs phase 2 only. ----
        #
        # `except Exception` deliberately, NOT a type list. The requirement is
        # that a phase-2 failure never costs phase 1, and a denylist cannot
        # deliver that: an audit pointed out that any type outside it escapes
        # both handlers and leaves the phase-1 artifact on disk permanently
        # asserting phase 2 was never attempted. KeyboardInterrupt and
        # SystemExit are BaseException and still propagate.
        try:
            _say(stream, "  zero check (phase 2 half)")
            payload["gate"]["phase2"] = gate_phase2(gate_pair, batch)
            _say(stream, "    PASSED")
            _say(stream, "  phase 2: per-layer cosine and per-layer CKA")
            payload["phase2"].update({
                "attempted": True, "ok": True, "reason": None,
                "pairs": run_phase2(pairs, batch, stream),
            })
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            # A gate that RAN AND FAILED must be recorded as failed. This used
            # to be assigned only on success, so the report called it
            # "NOT REACHED" -- a gate that fired described as one that never
            # ran. Also caught by the audit.
            if "phase2" not in payload["gate"]:
                payload["gate"]["phase2"] = {
                    "passed": False, "pair": gate_pair.label,
                    "reason": reason,
                    "WHAT_THIS_COSTS": (
                        "phase 2 only. Phase 1 passed its own gate and its "
                        "numbers stand."),
                }
            payload["phase2"].update({
                "attempted": True, "ok": False, "pairs": {},
                "reason": reason,
            })
            # The handler's OWN prints are guarded. Widening the except clause
            # is not enough on its own: if stdout is a closed pipe -- `| head`
            # is the ordinary way to look at this -- then the diagnostic print
            # raises OSError from inside the handler and escapes anyway,
            # leaving the phase-1 artifact asserting phase 2 was never
            # attempted. A refuter found this while confirming the denylist
            # fix. Recording the failure in the artifact matters more than
            # announcing it on a stream that has gone away.
            _say(stream, f"  PHASE 2 FAILED, phase 1 stands: "
                         f"{type(exc).__name__}")
            _say(stream, f"  {exc}")

        write_artifacts(payload, reportdir, stem, stream)
        _say(stream, "")
        _say(stream, format_report(payload))
        # NOT 2: argparse already exits 2 for a bad command line, and one code
        # meaning both "you typed it wrong, nothing written" and "phase 1 is on
        # disk, phase 2 failed" is a signal that cannot be acted on.
        return EXIT_OK if payload["phase2"]["ok"] else EXIT_PHASE2_FAILED

    except (LadderError, M.MetricsError) as exc:
        # M.MetricsError is included so the SAME refusal reads the same way
        # wherever it lands. The phase-2 handler catches it too, and without it
        # here the identical error was a clean message in one place and a
        # traceback in the other -- noted by an audit.
        print(f"\nDISPLACEMENT LADDER ERROR\n{type(exc).__name__}: {exc}\n",
              file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
