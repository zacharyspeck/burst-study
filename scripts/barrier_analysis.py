#!/usr/bin/env python
"""Section 8.4's headline metric, analysed: arm-vs-twin barrier against the floor.

WHY THIS IS NOT `scripts/analysis.py --metric barrier`. That module's `Panel`
holds a PER-MODEL scalar and derives two things from it: the paired difference,
by subtracting the seed-matched reference, and the noise floor, by differencing
the reference across seeds. The barrier is not a per-model scalar. It is
pairwise -- `metrics.barrier(model_a, model_b)` -- so barrier(arm, twin) IS
already the displacement, and the floor is barrier(twin_i, twin_j), a quantity
no arithmetic on per-seed scalars can produce. Setting the twin's cell to zero
to force it through the panel would make the floor identically zero and
`clears_noise_floor` true for every arm, which is the exact defect class S55
records.

So the panel is bypassed and the ESTIMATORS ARE NOT. `paired_t_test`,
`bootstrap_ci` and `correct` are imported from `analysis.py`, which is where
they are tested and cross-checked against scipy. This is the same route box A
took (`2026-08-09-boxA-results.md`: "the estimators were called directly on the
paired differences, which is the same arithmetic without the panel gate").

The confirmatory contrast is §5, `fluent-fabricated` vs `fluent-attested`, differenced
within seed. Family is ONE after §10 A-4 / D-9, so no correction is applied;
`--correction` is still required and still has no default.
"""
from __future__ import annotations
import argparse, json, statistics, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analysis import (AnalysisError, bootstrap_ci, correct, paired_t_test,
                      CORRECTIONS, DEFAULT_BOOTSTRAP, DEFAULT_LEVEL)
from burst.config import ARMS

PRIMARY = ("fluent-fabricated", "fluent-attested")
SECONDARY_AGAINST = "pos-substituted"


def _check_endpoints(p: Path, d: dict, want_a: str, want_b: str) -> None:
    """The filename says which cell this is; the file says which runs it is OF.

    `pair_barrier.py` writes `run_a`/`run_b` into every per-pair JSON, and this
    module keyed the panel off the FILENAME while those fields sat unread. A
    misnamed `armtwin_seed03_fluent-attested.json` holding a seed-05 pair would be
    attributed to seed 03 and nothing downstream would notice -- the S55 defect
    class exactly. The endpoints are authoritative; the name is a label. Refuse
    on disagreement rather than trust the label.

    Absence is also refused. A per-pair file with no endpoints recorded cannot
    be checked, and silently accepting it would reintroduce the gap for any
    file that happens to omit them.
    """
    got_a, got_b = d.get("run_a"), d.get("run_b")
    if got_a != want_a or got_b != want_b:
        raise AnalysisError(
            f"{p.name}: the name implies run_a={want_a!r} run_b={want_b!r}, but "
            f"the file records run_a={got_a!r} run_b={got_b!r}. Seed and arm are "
            "parsed from the filename; the recorded endpoints are authoritative. "
            "Fix the name or the file -- do not average a mislabelled pair.")


def load(barrier_dir: Path):
    """Read the measured pairs. arm-vs-twin by (arm, seed); twin-vs-twin flat."""
    arm_twin, twin_twin = {}, {}
    for p in sorted(barrier_dir.glob("armtwin_*.json")):
        d = json.loads(p.read_text())
        stem = p.stem[len("armtwin_"):]            # seedNN_arm
        seed, arm = int(stem[4:6]), stem.split("_", 1)[1]
        if arm not in ARMS:
            raise AnalysisError(f"unknown arm {arm!r} from {p.name}")
        _check_endpoints(p, d, stem, f"seed{seed:02d}_twin")
        if (arm, seed) in arm_twin:
            raise AnalysisError(f"duplicate arm-vs-twin cell for {arm} seed {seed}")
        arm_twin[(arm, seed)] = d
    for p in sorted(barrier_dir.glob("twintwin_*.json")):
        d = json.loads(p.read_text())
        a, b = p.stem[len("twintwin_"):].split("_", 1)
        _check_endpoints(p, d, f"{a}_twin", f"{b}_twin")
        twin_twin[p.stem] = d
    return arm_twin, twin_twin


def summarize(diffs, label, level, n_resamples):
    ci = bootstrap_ci(diffs, level=level, n_resamples=n_resamples, label=label)
    t = paired_t_test(diffs)
    return {
        "n": len(diffs), "mean": ci["mean"], "sd": statistics.stdev(diffs) if len(diffs) > 1 else 0.0,
        "min": min(diffs), "median": statistics.median(diffs), "max": max(diffs),
        "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
        "ci_excludes_zero": ci["excludes_zero"],
        "t": t["t"], "df": t["df"], "p_raw": t["p"],
        "values": diffs,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--barrier-dir", required=True, type=Path)
    ap.add_argument("--correction", required=True, choices=CORRECTIONS,
                    help="REQUIRED, no default. Family is 1 after A-4/D-9, so "
                         "'none' is what that ruling implies -- but it is "
                         "still passed explicitly.")
    ap.add_argument("--level", type=float, default=DEFAULT_LEVEL)
    ap.add_argument("--resamples", type=int, default=DEFAULT_BOOTSTRAP)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min-seeds", type=int, default=10)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    arm_twin, twin_twin = load(a.barrier_dir)
    if not arm_twin:
        raise AnalysisError(f"no armtwin_*.json in {a.barrier_dir}")

    arms = sorted({arm for arm, _ in arm_twin})
    seeds = sorted({s for _, s in arm_twin})
    for arm in arms:
        missing = [s for s in seeds if (arm, s) not in arm_twin]
        if missing:
            raise AnalysisError(
                f"arm {arm!r} is missing seeds {missing}; a paired difference "
                "over a subset of seeds averages a different study.")
    if len(seeds) < a.min_seeds:
        raise AnalysisError(
            f"{len(seeds)} seeds against a floor of {a.min_seeds}.")

    # displacement of each arm from its seed-matched twin == the barrier itself
    disp = {arm: [arm_twin[(arm, s)]["barrier"]["max_excess"] for s in seeds]
            for arm in arms}
    l2 = {arm: [arm_twin[(arm, s)]["l2_raw"] for s in seeds] for arm in arms}

    # The floor is not optional. This module exists to compare a pairwise
    # displacement against a MEASURED twin-vs-twin floor; without one the
    # comparison silently becomes "clears_noise_floor: None" for every arm,
    # which reads like a computed result and is not one. Refuse instead.
    floor = [d["barrier"]["max_excess"] for d in twin_twin.values()]
    if not floor:
        raise AnalysisError(
            f"no twintwin_*.json in {a.barrier_dir}. The twin-vs-twin barrier "
            "across seeds IS the noise floor (spec-v4), and it cannot be "
            "derived from the arm-vs-twin pairs -- it has to be measured. "
            "Without it there is no scale to read a displacement against.")
    expected = len(seeds) * (len(seeds) - 1) // 2
    if len(floor) != expected:
        raise AnalysisError(
            f"{len(floor)} twin-vs-twin pairs for {len(seeds)} seeds; expected "
            f"C({len(seeds)},2) = {expected}. A floor over a subset of pairs is "
            "a different floor, and it is the widest pair that sets the scale.")
    floor_scale = max(abs(x) for x in floor)

    rows, pvals = [], []
    for arm in arms:
        r = summarize(disp[arm], f"barrier/{arm}", a.level, a.resamples)
        r["arm"] = arm
        r["l2_raw_mean"] = statistics.fmean(l2[arm])
        r["clears_noise_floor"] = (abs(r["mean"]) > floor_scale
                                   if floor_scale is not None else None)
        rows.append(r); pvals.append(r["p_raw"])
    for r, adj in zip(rows, correct(pvals, a.correction)):
        r["p_adjusted"] = adj
        r["significant_vs_zero"] = adj < a.alpha

    # §5 PRIMARY confirmatory contrast: the two displacements, within seed.
    primary = None
    if all(x in disp for x in PRIMARY):
        diffs = [x - y for x, y in zip(disp[PRIMARY[0]], disp[PRIMARY[1]])]
        primary = summarize(diffs, "barrier/primary", a.level, a.resamples)
        primary["contrast"] = f"{PRIMARY[0]} minus {PRIMARY[1]}"
        primary["computed"] = True
        primary["FAMILY"] = ("1 -- preregistration.md 10 A-4 and D-9. No "
                             "multiple-comparison correction is applied.")
        primary["significant"] = primary["p_raw"] < a.alpha
    else:
        primary = {"computed": False,
                   "missing_arms": [x for x in PRIMARY if x not in disp]}

    payload = {
        "metric": "plain_interpolation_loss_barrier_vs_seed_matched_twin",
        "PREREGISTERED_AS": "docs/preregistration.md 8.4 headline metric",
        "n_seeds": len(seeds), "seeds": seeds,
        "correction": a.correction, "alpha": a.alpha, "level": a.level,
        "windows": sorted({d["windows"] for d in arm_twin.values()}),
        "noise_floor": {
            "WHAT_IT_IS": ("twin against twin ACROSS seeds, every distinct "
                           "pair, measured pairwise -- not derived by "
                           "differencing per-seed scalars."),
            "n_pairs": len(floor),
            "min": min(floor) if floor else None,
            "median": statistics.median(floor) if floor else None,
            "max": max(floor) if floor else None,
            "mean": statistics.fmean(floor) if floor else None,
            "widest_abs": floor_scale,
        },
        "arms": rows,
        "ordering": [r["arm"] for r in sorted(rows, key=lambda r: r["mean"],
                                              reverse=True)],
        "registered_contrasts": {
            "primary": primary,
            "secondary": {
                "computed": False,
                "against": SECONDARY_AGAINST,
                "WHY_ABSENT": ("the arm was cut on 2026-08-08 (preregistration "
                               "10 A-4 / D-9); no panel this study produces can "
                               "contain it. Reported as absent, not dropped."),
            },
        },
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"metric: {payload['metric']}")
    print(f"n = {len(seeds)} seeds {seeds}, windows {payload['windows']}")
    f = payload["noise_floor"]
    print(f"\nNOISE FLOOR (twin vs twin, {f['n_pairs']} pairs): "
          f"median {f['median']:.8f}  max {f['max']:.8f}")
    print(f"\n{'arm':16s} {'mean':>12s} {'sd':>11s} {'ci_low':>12s} {'ci_high':>12s} "
          f"{'p':>9s}  clears_floor")
    for r in rows:
        print(f"{r['arm']:16s} {r['mean']:12.8f} {r['sd']:11.8f} {r['ci_low']:12.8f} "
              f"{r['ci_high']:12.8f} {r['p_raw']:9.5f}  {r['clears_noise_floor']}")
    if primary.get("computed"):
        p = primary
        print(f"\nPRIMARY (§5) {p['contrast']}: mean {p['mean']:.8f}  "
              f"t({p['df']}) = {p['t']:.4f}  p = {p['p_raw']:.5f}")
        print(f"  {int(a.level*100)}% CI [{p['ci_low']:.8f}, {p['ci_high']:.8f}]  "
              f"excludes zero: {p['ci_excludes_zero']}")
    print(f"\nSECONDARY (§6): NOT COMPUTED -- {SECONDARY_AGAINST} was cut (A-4/D-9)")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AnalysisError as exc:
        print(f"REFUSED: {exc}")
        raise SystemExit(1)
