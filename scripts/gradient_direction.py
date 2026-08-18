#!/usr/bin/env python
"""Log gradient DIRECTION between the seven arms. 8b-iv, spec section 5.4.

    python scripts/gradient_direction.py

Two bursts matched on loss and gradient norm can still write to different
subspaces. Direction is a live confound on the study's central claim. It is
deliberately NOT controlled -- controlling it in 124 million dimensions would
require the texts to be nearly identical, which would remove the mechanism by
which content could act at all. It is recorded so the confound can be stated
rather than assumed away.

WHAT IS COMPUTED, NAMED FOR WHAT IT ACTUALLY IS

  pairwise cosine similarity   21 arm-arm pairs, from the burst-region loss
                               gradient at position 400 -- the same quantity
                               the arms were matched on.

  per-tensor gradient norm     PER ARM. The 148 per-tensor norms, normalised.
  profile                      Says WHERE IN THE NETWORK the update lands.
                               Blind to direction: two gradients with the same
                               profile can be orthogonal.

  participation ratio          PER ARM. (sum g^2)^2 / sum g^4 over all
                               124,439,808 components. Says HOW MANY
                               COORDINATES carry the update. BASIS-DEPENDENT
                               and not rotation-invariant -- it measures
                               coordinate sparsity in the standard parameter
                               basis, which is not the same thing as
                               directional spread.

  Gram eigenspectrum           SET-LEVEL, NOT PER ARM. Eigenvalues of the 7x7
                               cosine matrix. Says how many effective
                               directions the seven arms COLLECTIVELY span.
                               Must never appear in a per-arm column.

These are NOT collectively "anisotropy". The strict directional-spread
measure -- the per-arm gradient covariance spectrum -- was scoped out as
infeasible on CPU at this scale and is not computed here. See S40 in
implementation-notes.md.

MEMORY. One flat fp32 gradient is 497.8 MB and this task needs thirteen of
them, so they are never all held. Each is computed once, spilled to a memmap
under a gitignored .gradcache/, and the dot products are streamed. Peak RAM is
the model plus one anchor vector plus a chunk buffer.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from burst_match import (  # noqa: E402
    DEFAULT_BASE_CONFIG,
    RULE,
    THIN,
    BurstMatchError,
    assemble_sequence,
    flat_gradient,
    gradient_parameters,
    load_model,
    per_token_losses,
    resolve_seq_len,
)
from make_bursts import (  # noqa: E402
    ARM_SPECS,
    CONTEXT_NAME,
    DEFAULT_CACHE,
    DEFAULT_DOCS,
    DEFAULT_OUTDIR,
    POS_POOL_NAME,
    MakeBurstsError,
    arm_by_name,
    arm_spans,
    derived_seed,
    load_corpus_slice,
    load_pos_pool,
    pos_substitute,
    random_ascii_text,
    span_words_for,
    token_count,
    trim_to_tokens,
    window_shuffle,
    window_shuffle_to_length,
)

POSITION = 400
SHIPPED_K = 3
#: Matches tune_arms.py, so a "second draw" here is the same notion of a
#: different draw that the tuning search used.
SEED_STRIDE = 10007
GRADCACHE = REPO_ROOT / ".gradcache"
DEFAULT_REPORT_DIR = REPO_ROOT / "docs" / "measurements"
REPORT_STEM = "8b-iv-gradient-direction"

#: Short labels for the matrix. The full names do not fit a 7x7 grid.
ABBREV = {"fluent-fabricated": "ff", "fluent-attested": "ft", "scrambled-false": "sf",
          "scrambled-true": "st", "scrambled-corpus": "sc",
          "pos-substituted": "ps", "random-chars": "rc"}

#: The only pairs sharing the same words at two structure levels. Their cosine
#: isolates what scrambling does to direction with vocabulary, topic and
#: length held constant.
DERIVED_PAIRS = (("scrambled-false", "fluent-fabricated"),
                 ("scrambled-true", "fluent-attested"))

LIMITATION = (
    "Spec 5.4 specifies direction logged AT THE INJECTION STEP -- gradients "
    "from our own model at step 200. THAT MODEL DOES NOT EXIST. There is no "
    "training loop. Everything in this file comes from fully-trained public "
    "GPT-2, which has different geometry: different curvature, different layer "
    "specialisation, and a gradient structure shaped by training this study's "
    "model will not have seen. THESE NUMBERS ARE A PROXY OF UNKNOWN FIDELITY "
    "AND MUST BE RE-MEASURED AGAINST A REAL STEP-200 CHECKPOINT ONCE TRAINING "
    "INFRASTRUCTURE EXISTS. They are not the study's direction measurement."
)


def chunked_dot(path_a: Path, path_b: Path, n: int, chunk: int = 8_000_000):
    """Dot product of two spilled vectors, streamed. Accumulated in float64."""
    import numpy as np

    a = np.memmap(path_a, dtype=np.float32, mode="r", shape=(n,))
    b = np.memmap(path_b, dtype=np.float32, mode="r", shape=(n,))
    total = 0.0
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        total += float(np.dot(a[start:end].astype(np.float64),
                              b[start:end].astype(np.float64)))
    del a, b
    return total


def cosine(dot_ab: float, norm_a: float, norm_b: float) -> float:
    """Clamped to [-1, 1].

    Float64 accumulation over 124 million terms can land a cosine a few parts
    in 1e11 outside the valid range -- a measured self-cosine came out at
    1.000000000037. Clamping is honest here; reporting 1.000000000037 as if it
    were a real excess over 1 would not be.
    """
    raw = dot_ab / (norm_a * norm_b)
    return max(-1.0, min(1.0, raw))


class Spiller:
    """Computes a gradient, records its per-arm statistics, spills it to disk."""

    def __init__(self, model, tokenizer, filler, seq_len):
        self.model = model
        self.tokenizer = tokenizer
        self.filler = filler
        self.seq_len = seq_len
        self.params = gradient_parameters(model)
        self.sizes = [p.numel() for p in self.params]
        self.n = sum(self.sizes)
        GRADCACHE.mkdir(parents=True, exist_ok=True)
        self.records: dict = {}

    def _region_loss(self, sequence, position, region_len):
        import torch
        torch.manual_seed(0)
        t = torch.tensor([sequence], dtype=torch.long)
        losses = per_token_losses(self.model(input_ids=t).logits, t)
        return losses[position - 1:position - 1 + region_len].mean()

    def add(self, name: str, sequence, position: int, region_len: int) -> dict:
        """Compute, measure, spill. Returns the record; frees the vector."""
        import numpy as np
        import torch

        loss = self._region_loss(sequence, position, region_len)
        flat = flat_gradient(loss, self.model)

        d = flat.double()
        sum_sq = float(d.pow(2).sum())
        sum_4 = float(d.pow(4).sum())
        norm = sum_sq ** 0.5
        # B: participation ratio. How many coordinates effectively carry the
        # update. BASIS-DEPENDENT -- see the module docstring.
        pr = (sum_sq ** 2) / sum_4 if sum_4 > 0 else float("nan")

        # A: per-tensor gradient norm profile. WHERE the update lands.
        profile, offset = [], 0
        for size in self.sizes:
            seg = d[offset:offset + size]
            profile.append(float(seg.norm()))
            offset += size
        # Shares must be computed on SQUARED norms. Per-tensor norms add in
        # quadrature, not linearly, so sum(||g_i||)/||g|| exceeds 1 and is not
        # a share at all -- an earlier version reported 1.2387 as a "top-5
        # share", which is what caught it. sum(||g_i||^2) == ||g||^2 exactly,
        # so squared norms give genuine fractions of gradient energy.
        squares = [x * x for x in profile]
        total_sq = sum(squares)
        ordered = sorted(squares, reverse=True)

        path = GRADCACHE / f"{name}.f32"
        np.asarray(flat.numpy(), dtype=np.float32).tofile(path)
        del flat, d
        import gc
        gc.collect()

        rec = {
            "name": name, "path": str(path), "norm": norm,
            "region_loss": float(loss.detach()),
            "participation_ratio": pr,
            "participation_ratio_fraction": pr / self.n,
            "per_tensor_norm_profile": profile,
            "per_tensor_top5_energy_share": sum(ordered[:5]) / total_sq,
            "per_tensor_top20_energy_share": sum(ordered[:20]) / total_sq,
        }
        self.records[name] = rec
        return rec


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/gradient_direction.py",
        description="Pairwise gradient cosine between the arms, plus per-arm "
                    "profile and participation ratio and a set-level Gram "
                    "eigenspectrum. Changes no arm.")
    parser.add_argument("--position", type=int, default=POSITION)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--burstdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--docs", type=int, default=DEFAULT_DOCS)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--reportdir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--keep-cache", action="store_true",
                        help="leave the spilled gradients in .gradcache/")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    print(RULE)
    print("gradient_direction -- 8b-iv, spec 5.4")
    print(RULE)
    print("PROXY MODEL. " + LIMITATION.split("THAT MODEL DOES NOT EXIST")[0]
          + "THAT MODEL DOES NOT EXIST.")
    try:
        return _run(args)
    except (BurstMatchError, MakeBurstsError) as exc:
        print(f"\nERROR\n{exc}\n", file=sys.stderr)
        return 1


def _run(args) -> int:
    import numpy as np

    burstdir = Path(args.burstdir)
    seq = resolve_seq_len(None, args.base_config)
    tokenizer, model = load_model()
    print()

    context_ids = tokenizer((burstdir / CONTEXT_NAME).read_text(encoding="utf-8"),
                            add_special_tokens=False)["input_ids"]
    arm_ids = {s.name: tokenizer((burstdir / s.filename).read_text(encoding="utf-8"),
                                 add_special_tokens=False)["input_ids"]
               for s in ARM_SPECS}
    n_burst = len(next(iter(arm_ids.values())))
    filler = context_ids[:seq.value - n_burst]
    position = args.position

    sp = Spiller(model, tokenizer, filler, seq.value)
    print(f"{len(sp.params)} tensors, {sp.n:,} parameters, "
          f"{sp.n * 4 / 1e6:.1f} MB per gradient")
    print(f"spilling to {GRADCACHE}")
    print()

    # ---- the seven arms --------------------------------------------------
    print("computing gradients:", end=" ", flush=True)
    for spec in ARM_SPECS:
        print(ABBREV[spec.name], end=" ", flush=True)
        sp.add(spec.name, assemble_sequence(filler, arm_ids[spec.name], position),
               position, n_burst)

    # ---- control 3: the same window holding filler instead of a burst ----
    print("[filler]", end=" ", flush=True)
    sp.add("__filler__", list(context_ids), position, n_burst)

    # ---- control 2: a second draw of each stochastic arm -----------------
    docs = load_corpus_slice(Path(args.cache), args.docs, stream=open(
        __import__("os").devnull, "w"))
    spans = arm_spans(docs, args.seed, n_burst, stream=open(
        __import__("os").devnull, "w"))
    template, pools, _ = load_pos_pool(burstdir / POS_POOL_NAME,
                                       spans["pos-substituted"])
    sources = {n: (burstdir / arm_by_name(n).filename).read_text(encoding="utf-8")
               for n in ("fluent-fabricated", "fluent-attested")}

    second = {}
    for spec in ARM_SPECS:
        if not spec.is_generated:
            continue
        alt_seed = derived_seed(args.seed, spec.name) + SEED_STRIDE
        if spec.derives_from:
            text, _ = window_shuffle_to_length(
                sources[spec.derives_from], SHIPPED_K, n_burst, tokenizer,
                alt_seed, spec.name)
        elif spec.name == "scrambled-corpus":
            raw = span_words_for(spec, spans[spec.name], n_burst)
            text = window_shuffle(raw, SHIPPED_K, random.Random(alt_seed))
        elif spec.name == "pos-substituted":
            text = pos_substitute(template, pools, random.Random(alt_seed))
        else:
            text = random_ascii_text(
                int(n_burst * spec.params["oversample_chars"]),
                random.Random(alt_seed))
        trimmed, _ = trim_to_tokens(tokenizer, text, n_burst, label=spec.name)
        ids = tokenizer(trimmed, add_special_tokens=False)["input_ids"]
        key = f"{spec.name}__draw2"
        print(f"{ABBREV[spec.name]}2", end=" ", flush=True)
        sp.add(key, assemble_sequence(filler, ids, position), position, n_burst)
        second[spec.name] = key
    print()
    print(f"{len(sp.records)} gradients spilled "
          f"({len(sp.records) * sp.n * 4 / 1e9:.2f} GB)")

    # ---- cosines ---------------------------------------------------------
    names = [s.name for s in ARM_SPECS]
    print()
    print("streaming dot products:", end=" ", flush=True)
    cosines: dict = {}

    def cos_between(a: str, b: str) -> float:
        ra, rb = sp.records[a], sp.records[b]
        return cosine(chunked_dot(Path(ra["path"]), Path(rb["path"]), sp.n),
                      ra["norm"], rb["norm"])

    matrix = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            matrix[f"{a}|{b}"] = cos_between(a, b)
        print(".", end="", flush=True)
    cosines["arm_pairs"] = matrix
    cosines["arm_vs_filler_control"] = {
        a: cos_between(a, "__filler__") for a in names}
    print("+", end="", flush=True)
    cosines["same_arm_second_draw"] = {
        a: cos_between(a, key) for a, key in second.items()}
    print("+ done")

    # ---- C: Gram eigenspectrum. SET-LEVEL, NOT PER ARM -------------------
    gram = np.eye(len(names))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i < j:
                gram[i, j] = gram[j, i] = matrix[f"{a}|{b}"]
    eig = np.linalg.eigvalsh(gram)[::-1]
    eff_dim = float((eig.sum() ** 2) / (eig ** 2).sum())

    payload = {
        "task": "8b-iv", "spec_section": "5.4",
        "LIMITATION": LIMITATION,
        "naming_note": (
            "These are NOT collectively 'anisotropy'. The per-tensor norm "
            "profile and the participation ratio are per-arm statistics; the "
            "Gram eigenspectrum is SET-LEVEL and must not appear in a per-arm "
            "column. The strict directional-spread measure, the per-arm "
            "gradient covariance spectrum, was scoped out as infeasible on "
            "CPU at this scale and is NOT computed here."),
        "gradient_source": ("gradient of the burst-region loss over the 194 "
                            "burst tokens at position 400 -- the quantity the "
                            "arms were matched on"),
        "position": position, "burst_tokens": n_burst,
        "sequence_tokens": seq.value, "parameters": sp.n,
        "model": "gpt2 (public, fully trained) -- A PROXY, see LIMITATION",
        "environment": {"python": sys.version.split()[0],
                        "platform": platform.platform()},
        "per_arm": {
            a: {"gradient_norm": sp.records[a]["norm"],
                "burst_region_loss": sp.records[a]["region_loss"],
                "participation_ratio": sp.records[a]["participation_ratio"],
                "participation_ratio_fraction":
                    sp.records[a]["participation_ratio_fraction"],
                "participation_ratio_note":
                    "basis-dependent, not rotation-invariant",
                "per_tensor_norm_profile":
                    sp.records[a]["per_tensor_norm_profile"],
                "per_tensor_top5_energy_share":
                    sp.records[a]["per_tensor_top5_energy_share"],
                "per_tensor_top20_energy_share":
                    sp.records[a]["per_tensor_top20_energy_share"]}
            for a in names},
        "cosines": cosines,
        "derived_pairs": {
            f"{a}|{b}": matrix.get(f"{a}|{b}", matrix.get(f"{b}|{a}"))
            for a, b in DERIVED_PAIRS},
        "set_level_gram_eigenspectrum": {
            "note": ("SET-LEVEL statistic. Eigenvalues of the 7x7 cosine "
                     "matrix. NOT a per-arm quantity."),
            "eigenvalues": [float(x) for x in eig],
            "effective_dimensionality": eff_dim,
            "max_possible": len(names)},
        "controls": {
            "numerical_ceiling": ("same text twice gives bitwise-identical "
                                  "gradients and a self-cosine of 1.0; "
                                  "asserted in tests/test_sequence_assembly.py"),
            "filler_region": ("the same 194-token window at the same position "
                              "holding filler instead of a burst"),
            "second_draw": ("the same arm regenerated at a different seed with "
                            "identical parameters")},
    }

    reportdir = Path(args.reportdir)
    reportdir.mkdir(parents=True, exist_ok=True)
    (reportdir / f"{REPORT_STEM}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    text = format_report(payload, names, matrix, eig, eff_dim)
    (reportdir / f"{REPORT_STEM}.md").write_text(
        text + "\n", encoding="utf-8", newline="\n")
    print(text)
    print()
    print(f"wrote {reportdir / (REPORT_STEM + '.json')}")

    if not args.keep_cache:
        for rec in sp.records.values():
            Path(rec["path"]).unlink(missing_ok=True)
        print(f"cleared {GRADCACHE}")
    return 0


def format_report(payload, names, matrix, eig, eff_dim) -> str:
    cos = payload["cosines"]
    lines = ["", RULE, "8b-iv GRADIENT DIRECTION", RULE,
             "PROXY MODEL -- these are not the study's direction measurement.",
             "See the LIMITATION field in the JSON and S40 in "
             "implementation-notes.md.", ""]

    lines += [RULE, "DERIVED PAIRS -- the same words at two structure levels",
              RULE,
              "The only pairs where vocabulary, topic and length are held",
              "constant, so the cosine isolates what scrambling does to",
              "direction.", ""]
    for a, b in DERIVED_PAIRS:
        key = f"{a}|{b}" if f"{a}|{b}" in matrix else f"{b}|{a}"
        lines.append(f"  {a:<18} vs {b:<15} cos = {matrix[key]:+.4f}")

    lines += ["", RULE, "CONTROLS -- the floor every cosine must be read against",
              RULE,
              "numerical ceiling: 1.0 (same text twice, bitwise identical)", "",
              "same arm, second draw (identical parameters, different seed):"]
    for a, v in cos["same_arm_second_draw"].items():
        lines.append(f"  {a:<20} {v:+.4f}")
    lines += ["", "arm vs filler-region control (same window, no burst):"]
    for a, v in cos["arm_vs_filler_control"].items():
        lines.append(f"  {a:<20} {v:+.4f}")

    lines += ["", RULE, "PAIRWISE COSINE MATRIX (21 pairs)", RULE,
              "      " + "".join(f"{ABBREV[n]:>8}" for n in names)]
    for i, a in enumerate(names):
        row = [f"{ABBREV[a]:>4}  "]
        for j, b in enumerate(names):
            if i == j:
                row.append(f"{'1.0000':>8}")
            else:
                key = f"{a}|{b}" if f"{a}|{b}" in matrix else f"{b}|{a}"
                row.append(f"{matrix[key]:>8.4f}")
        lines.append("".join(row))
    lines.append("")
    for n in names:
        lines.append(f"  {ABBREV[n]} = {n}")

    lines += ["", RULE, "PER-ARM: gradient norm, participation ratio, profile",
              RULE,
              "participation ratio is BASIS-DEPENDENT, not rotation-invariant.",
              "",
              f"{'arm':<18}{'grad norm':>11}{'partic.ratio':>14}"
              f"{'as frac of D':>14}{'top5 energy':>13}"]
    for a in names:
        p = payload["per_arm"][a]
        lines.append(f"{a:<18}{p['gradient_norm']:>11.4f}"
                     f"{p['participation_ratio']:>14.1f}"
                     f"{p['participation_ratio_fraction']:>14.6f}"
                     f"{p['per_tensor_top5_energy_share']:>13.4f}")

    lines += ["", RULE,
              "SET-LEVEL: Gram eigenspectrum of the seven arms",
              RULE,
              "NOT a per-arm statistic. Eigenvalues of the 7x7 cosine matrix.",
              "",
              "  eigenvalues: " + "  ".join(f"{float(x):.4f}" for x in eig),
              f"  effective dimensionality: {eff_dim:.3f} of a possible "
              f"{len(names)}"]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
