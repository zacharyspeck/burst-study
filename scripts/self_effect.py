#!/usr/bin/env python
"""Does a run predict the passage it was trained on better than its twin does?

EXPLORATORY. Nothing here is pre-registered: docs/preregistration.md fixes a
displacement contrast and a held-out-loss contrast, and neither is this.

READS `stimulus_probe.py` OUTPUT. The cross-entropy this differences is the mean
per-token loss over the 194 injected tokens, teacher-forced inside the FULL
1024-token injected row -- `injection.burst_region_losses` feeds the model
`plan.sequence` entire and slices the burst region out of the per-token losses,
so the passage is never scored in isolation and every token in the region is
predicted from the same left context the model had at step 200.

THE ROW IS NOT SEED-DEPENDENT, and this matters for reading the numbers below.
`injection.build_plan` assembles it from `bursts/context.txt` filler plus the
burst file at a fixed position; the seed enters only through
`injection.batch_slot_for`, which decides WHICH ROW OF THE BATCH the sequence
replaces. So all eight seeds of an arm saw the same 1024 tokens, and the
across-seed spread in every quantity here comes from the models alone.

FOUR DIFFERENCES, ALL PAIRED WITHIN SEED, all signed so that POSITIVE MEANS THE
INJECTED RUN IS BETTER at the passage than its control:

  self(ff)   CE[twin, P_ff] - CE[ff, P_ff]      the arm that saw P_ff, on P_ff
  self(ft)   CE[twin, P_ft] - CE[ft, P_ft]      the arm that saw P_ft, on P_ft
  cross(ff)  CE[twin, P_ft] - CE[ff, P_ft]      ff never saw P_ft. NULL CHANNEL.
  noise      CE[twin, P_ff] - CE[rc, P_ff]      rc saw neither. NULL CHANNEL.

The two null channels are the point. `self` alone cannot distinguish "learned
this passage" from "drifted in a way that helps on fluent passages of this
kind", because both arms diverged from the twin for 9,336 steps afterwards. A
real single-exposure effect is `self` positive with `cross` and `noise` at zero.

The attestation contrast, self(ff) - self(ft), asks whether a fabricated-subject
passage sticks differently from an attested-subject one. It is the
construct-valid form of the registered primary question and IT IS NOT THE
REGISTERED TEST; the registered one is on displacement.

Estimators come from analysis.py, so the arithmetic is the tested one.
"""
from __future__ import annotations
import argparse, json, math, statistics, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analysis import (AnalysisError, paired_t_test, bootstrap_ci, correct,
                      student_t_sf, DEFAULT_LEVEL, DEFAULT_BOOTSTRAP)

REF = "twin"
#: (label, arm scored, passage scored on, what it is)
CHANNELS = (
    ("self_ff",  "fluent-fabricated", "fluent-fabricated", "saw it, scored on it"),
    ("self_ft",  "fluent-attested",  "fluent-attested",  "saw it, scored on it"),
    ("cross_ff", "fluent-fabricated", "fluent-attested",  "NULL CHANNEL: never saw this passage"),
    ("noise",    "random-chars", "fluent-fabricated", "NULL CHANNEL: saw neither passage"),
    # the symmetric partners of the two null channels, free from the same data
    ("cross_ft", "fluent-attested",  "fluent-fabricated", "NULL CHANNEL: never saw this passage"),
    ("noise_ft", "random-chars", "fluent-attested",  "NULL CHANNEL: saw neither passage"),
)
EXPECTED_REGION_TOKENS = 194


def load(d: Path) -> dict:
    cells = {}
    for p in sorted(d.glob("*.json")):
        j = json.loads(p.read_text())
        key = (j["arm"], j["seed"])
        if key in cells:
            raise AnalysisError(f"two probe files for {key}")
        cells[key] = j
    if not cells:
        raise AnalysisError(f"no *.json under {d}")
    return cells


def ce(cell: dict, passage: str, field: str) -> float:
    rec = cell["passage_loss"][passage]
    n = rec["n_tokens_all"]
    if n != EXPECTED_REGION_TOKENS:
        raise AnalysisError(
            f"{cell['run']} on {passage}: {n} scored tokens, expected "
            f"{EXPECTED_REGION_TOKENS}. A mean over a different number of "
            "tokens is not comparable to the recorded step-200 value.")
    return rec[field]


def _t_critical(alpha: float, df: int) -> float:
    """Two-sided t critical value, by bisection on the module's own tail.

    Inverted from `analysis.student_t_sf` rather than taken from a table or a
    second library, so the interval and the p-value in the same row cannot
    disagree about what distribution they are using.
    """
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if student_t_sf(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def summarize(diffs, label, level, n_res):
    """Paired t-test, a t-based interval to match it, and the repo's bootstrap."""
    t = paired_t_test(diffs)
    boot = bootstrap_ci(diffs, level=level, n_resamples=n_res, label=label)
    n = len(diffs)
    mean = statistics.fmean(diffs)
    sd = statistics.stdev(diffs) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    crit = _t_critical(1.0 - level, n - 1)
    return {"n": n, "mean": mean, "sd": sd, "se": se,
            "t": t["t"], "df": t["df"], "p": t["p"],
            "t_critical": crit,
            "ci_low_t": mean - crit * se, "ci_high_t": mean + crit * se,
            "t_interval_excludes_zero": abs(mean) > crit * se,
            "ci_low_bootstrap": boot["ci_low"], "ci_high_bootstrap": boot["ci_high"],
            "bootstrap_excludes_zero": boot["excludes_zero"],
            "seeds_positive": sum(1 for x in diffs if x > 0),
            "seeds_agreeing_in_sign": max(sum(1 for x in diffs if x > 0),
                                          sum(1 for x in diffs if x < 0)),
            "values": diffs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", required=True, type=Path)
    ap.add_argument("--field", default="mean_all",
                    choices=("mean_all", "mean_content"))
    ap.add_argument("--correction", required=True,
                    choices=("holm", "benjamini_hochberg", "benjamini_yekutieli",
                             "none"))
    ap.add_argument("--level", type=float, default=DEFAULT_LEVEL)
    ap.add_argument("--resamples", type=int, default=DEFAULT_BOOTSTRAP)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    cells = load(a.probe_dir)
    seeds = sorted({s for (_, s) in cells})
    for s in seeds:
        for arm in (REF, "fluent-fabricated", "fluent-attested", "random-chars"):
            if (arm, s) not in cells:
                raise AnalysisError(f"missing {arm} at seed {s}")

    res = {"WHAT_THIS_IS": ("Exploratory, not pre-registered. Mean per-token "
                            "cross-entropy over the 194 injected tokens, scored "
                            "inside the full 1024-token injected row, differenced "
                            "against the seed-matched control. Positive = the "
                            "injected run is better at the passage."),
           "field": a.field, "n_seeds": len(seeds), "seeds": seeds,
           "region_tokens": EXPECTED_REGION_TOKENS,
           "correction": a.correction, "channels": {}, "levels": {}}

    for arm in (REF, "fluent-fabricated", "fluent-attested", "random-chars"):
        res["levels"][arm] = {
            P: {"mean_over_seeds": statistics.fmean(
                    ce(cells[(arm, s)], P, a.field) for s in seeds),
                "values": [ce(cells[(arm, s)], P, a.field) for s in seeds]}
            for P in ("fluent-fabricated", "fluent-attested")}

    for label, arm, passage, what in CHANNELS:
        d = [ce(cells[(REF, s)], passage, a.field)
             - ce(cells[(arm, s)], passage, a.field) for s in seeds]
        res["channels"][label] = {
            "arm_scored": arm, "passage": passage, "what_it_is": what,
            **summarize(d, f"self-effect/{a.field}/{label}", a.level, a.resamples)}

    # THE COMPARISON THE NULL CHANNELS EXIST FOR. `self` minus the mean of the
    # arms that did not see that passage, on that same passage, paired within
    # seed. If `self` and the null channels move together this is zero, and a
    # `self` channel that is individually significant means nothing.
    res["diff_in_diff"] = {}
    for label, own, others in (
            ("fluent-fabricated", "self_ff", ("cross_ft", "noise")),
            ("fluent-attested",  "self_ft", ("cross_ff", "noise_ft"))):
        d = [res["channels"][own]["values"][i]
             - statistics.fmean(res["channels"][o]["values"][i] for o in others)
             for i in range(len(seeds))]
        res["diff_in_diff"][label] = {
            "definition": f"{own} minus the mean of {list(others)}, same passage",
            **summarize(d, f"self-effect/{a.field}/did/{label}", a.level,
                        a.resamples)}

    att = [res["channels"]["self_ff"]["values"][i]
           - res["channels"]["self_ft"]["values"][i] for i in range(len(seeds))]
    res["attestation_contrast"] = {
        "definition": "self(fluent-fabricated) - self(fluent-attested), paired within seed",
        "NOT_PREREGISTERED": ("The registered primary contrast is on displacement "
                              "(preregistration.md section 5). This is the same "
                              "question asked of the passage itself and was "
                              "written after the data were in hand."),
        **summarize(att, f"self-effect/{a.field}/attestation", a.level, a.resamples)}

    def slot(k):
        if k in res["channels"]:
            return res["channels"][k]
        if k == "attestation_contrast":
            return res["attestation_contrast"]
        return res["diff_in_diff"][k.removeprefix("did_")]

    family = ([k for k, _, _, _ in CHANNELS]
              + ["did_fluent-fabricated", "did_fluent-attested", "attestation_contrast"])
    for k, q in zip(family, correct([slot(k)["p"] for k in family], a.correction)):
        slot(k)["p_adjusted"] = q
    res["correction_family"] = family

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")

    print(f"n = {len(seeds)} seeds, paired within seed, field = {a.field}")
    print(f"CE is over {EXPECTED_REGION_TOKENS} injected tokens, teacher-forced "
          f"inside the full 1024-token injected row\n")
    print(f"{'channel':10s} {'arm':13s} {'on':13s} {'mean':>9s} {'t(7)':>7s} "
          f"{'p':>7s} {'p_adj':>7s} {'95% CI (bootstrap)':>24s}  sign")
    for k in family:
        r = slot(k)
        arm = r.get("arm_scored", "derived"); pas = r.get("passage", "--")
        print(f"{k:10s} {arm:13s} {pas:13s} {r['mean']:+9.5f} {r['t']:+7.3f} "
              f"{r['p']:7.4f} {r['p_adjusted']:7.4f} "
              f"[{r['ci_low_bootstrap']:+.5f}, {r['ci_high_bootstrap']:+.5f}]  "
              f"{r['seeds_agreeing_in_sign']}/{r['n']}"
              + ("  <-- CI excludes 0" if r["bootstrap_excludes_zero"] else ""))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
