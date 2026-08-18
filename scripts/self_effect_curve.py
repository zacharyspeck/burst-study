#!/usr/bin/env python
"""The self-effect as a function of training step. Figure NEW-2.

Reads `passage_ce_curve.py`'s JSONL and forms, at each checkpoint step:

    self(ff, s, t) = CE[twin(s) at t, P_ff] - CE[ff(s) at t, P_ff]
    self(ft, s, t) = CE[twin(s) at t, P_ft] - CE[ft(s) at t, P_ft]

paired within seed, positive meaning the injected run is better at the passage
it was trained on. Same sign convention and same estimators as
`scripts/self_effect.py`; this only adds the step axis.

STEP 199 IS THE CURVE'S ZERO AND IT IS EXACT, NOT FITTED. Injection is at step
200 and every arm's step-199 checkpoint is bit-identical to its twin's
(`2026-08-10-step199-digests.json`), so both sides of the difference are the
same model and the difference must be 0.0 to the last bit. A curve that does not
come out at exactly zero there has a bug in it, so the check is run rather than
assumed.

INCOMPLETE CURVES ARE REPORTED AS INCOMPLETE. Steps are included only where all
three runs of a seed were scored; anything partial is listed under
`steps_dropped_incomplete` with what was missing, because a curve that silently
omits the steps it could not reach looks like a curve that fell to zero.
"""
from __future__ import annotations
import argparse, json, statistics, sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analysis import (AnalysisError, paired_t_test, bootstrap_ci,
                      DEFAULT_LEVEL, DEFAULT_BOOTSTRAP)

REF = "twin"
PAIRS = (("self_ff", "fluent-false", "fluent-false"),
         ("self_ft", "fluent-true", "fluent-true"))
PRE_INJECTION_STEP = 199


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--level", type=float, default=DEFAULT_LEVEL)
    ap.add_argument("--resamples", type=int, default=DEFAULT_BOOTSTRAP)
    a = ap.parse_args()

    ce = {}          # (seed, arm, step, passage) -> value
    missing = []
    for f in a.jsonl:
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if "MISSING" in r:
                missing.append({"run": r["run"], "step": r["step"],
                                "path": r["MISSING"]})
                continue
            seed = int(r["run"][4:6]); arm = r["run"].split("_", 1)[1]
            for passage, v in r["ce"].items():
                key = (seed, arm, r["step"], passage)
                if key in ce and ce[key] != v["mean_all"]:
                    raise AnalysisError(
                        f"two different values for {key}: {ce[key]} and "
                        f"{v['mean_all']}. Rescoring disagreed with itself.")
                ce[key] = v["mean_all"]

    seeds = sorted({k[0] for k in ce})
    steps = sorted({k[2] for k in ce})
    res = {"WHAT_THIS_IS": ("Exploratory. Self-effect on the injected passage as a "
                           "function of training step, paired within seed. Positive "
                           "= the injected run is better at its own passage."),
           "n_seeds": len(seeds), "seeds": seeds,
           "steps_requested": steps, "steps": [],
           "steps_dropped_incomplete": [], "checkpoints_missing": missing,
           "curves": {k: [] for k, _, _ in PAIRS}}

    for t in steps:
        need = [(s, arm) for s in seeds for arm in (REF, "fluent-false", "fluent-true")]
        absent = [f"seed{s:02d}_{arm}"
                  for s, arm in need
                  if (s, arm, t, "fluent-false") not in ce
                  or (s, arm, t, "fluent-true") not in ce]
        if absent:
            res["steps_dropped_incomplete"].append({"step": t, "missing": absent})
            continue
        res["steps"].append(t)
        for label, arm, passage in PAIRS:
            d = [ce[(s, REF, t, passage)] - ce[(s, arm, t, passage)] for s in seeds]
            boot = bootstrap_ci(d, level=a.level, n_resamples=a.resamples,
                                label=f"curve/{label}/step{t}")
            tt = paired_t_test(d)
            res["curves"][label].append({
                "step": t, "n": len(d), "mean": statistics.fmean(d),
                "sd": statistics.stdev(d) if len(d) > 1 else 0.0,
                "t": tt["t"], "df": tt["df"], "p": tt["p"],
                "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
                "ci_excludes_zero": boot["excludes_zero"],
                "seeds_positive": sum(1 for x in d if x > 0),
                "values": d,
                "ce_twin_mean": statistics.fmean(ce[(s, REF, t, passage)] for s in seeds),
            })

    # The exact-zero anchor.
    res["pre_injection_check"] = {"step": PRE_INJECTION_STEP}
    if PRE_INJECTION_STEP in res["steps"]:
        worst = 0.0
        for label, _, _ in PAIRS:
            row = next(r for r in res["curves"][label]
                       if r["step"] == PRE_INJECTION_STEP)
            worst = max(worst, max(abs(v) for v in row["values"]))
        res["pre_injection_check"].update({
            "worst_abs_difference": worst,
            "exactly_zero": worst == 0.0,
            "WHY_IT_MUST_BE_ZERO": ("Injection is at step 200 and every arm's "
                                    "step-199 weights are bit-identical to its "
                                    "twin's, so both sides of the difference are "
                                    "the same model.")})
        if worst != 0.0:
            raise AnalysisError(
                f"the pre-injection step differs from zero by {worst:.3e}. Both "
                "sides of that difference are supposed to be the same "
                "checkpoint, so either the pairing is wrong or the scoring is "
                "not deterministic. Do not read the rest of the curve until "
                "this is explained.")
    else:
        res["pre_injection_check"]["NOT_AVAILABLE"] = (
            f"step {PRE_INJECTION_STEP} was not scored for every run, so the "
            "curve has no verified zero point")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")

    print(f"n = {len(seeds)} seeds; {len(res['steps'])} complete steps of "
          f"{len(steps)} requested")
    if res["steps_dropped_incomplete"]:
        print(f"  DROPPED as incomplete: "
              f"{[d['step'] for d in res['steps_dropped_incomplete']]}")
    if res["checkpoints_missing"]:
        print(f"  {len(res['checkpoints_missing'])} checkpoints were never scored")
    pic = res["pre_injection_check"]
    if "exactly_zero" in pic:
        print(f"  pre-injection anchor at step {PRE_INJECTION_STEP}: worst "
              f"|difference| = {pic['worst_abs_difference']:.1e} "
              f"({'exactly zero' if pic['exactly_zero'] else 'NOT ZERO'})")
    for label, _, passage in PAIRS:
        print(f"\n{label}  (arm minus twin sign-flipped: positive = arm better "
              f"at {passage})")
        print(f"  {'step':>6s} {'mean':>10s} {'sd':>9s} {'t':>7s} {'p':>7s} "
              f"{'95% CI':>22s}  +/n   CE[twin]")
        for r in res["curves"][label]:
            print(f"  {r['step']:6d} {r['mean']:+10.6f} {r['sd']:9.6f} "
                  f"{(r['t'] if r['t'] is not None else float('nan')):+7.3f} "
                  f"{r['p']:7.4f} [{r['ci_low']:+.6f},{r['ci_high']:+.6f}] "
                  f" {r['seeds_positive']}/{r['n']}  {r['ce_twin_mean']:.4f}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
