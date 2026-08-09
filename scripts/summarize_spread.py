#!/usr/bin/env python
"""A-3 COMPLIANT INTERIM LOOK. Spread only.

docs/preregistration.md section 10 A-3, ruled 2026-08-07 by Asa:

    "the SPREAD of the paired differences may be computed and used to set the
     seed count. Their MEAN, SIGN, CONFIDENCE INTERVAL, and ANY TEST STATISTIC
     may not be examined until the full seed count is complete."

This script therefore prints the standard deviation of the within-seed paired
differences and the minimum detectable effect it implies at various seed counts.
It deliberately does NOT print, return, or write anywhere:
  - any individual paired difference d_s
  - their mean or median
  - their sign
  - any confidence interval or t-statistic
  - any per-model held-out loss (from which d_s could be reconstructed)

Choosing n from the spread is a legitimate internal-pilot design. Choosing it
from the effect is choosing the experiment's size knowing its answer.
"""
from __future__ import annotations
import glob, json, math, statistics
from scipy import stats

PILOT_SUBSTITUTED_SIGMA = 0.002849   # heldout remeasurement 2026-08-07: sd of
                                     # twin-vs-twin x sqrt(2) -- the WRONG noise
                                     # term, used because the pilot had one
                                     # injecting arm at one seed and could not
                                     # estimate the paired sd at all.

def mde_in_sd_units(n, power=0.80, alpha=0.05):
    """Smallest effect, in units of the paired sd, detectable at `power`."""
    df = n - 1
    tc = stats.t.ppf(1 - alpha / 2, df)
    lo, hi = 1e-6, 50.0
    for _ in range(200):
        d = (lo + hi) / 2
        nc = d * math.sqrt(n)
        p = 1 - stats.nct.cdf(tc, df, nc) + stats.nct.cdf(-tc, df, nc)
        if p < power: lo = d
        else: hi = d
    return (lo + hi) / 2

def main():
    by = {}
    for f in glob.glob("/home/ubuntu/runs/*/heldout_eval.json"):
        r = json.load(open(f))
        by.setdefault(r["seed"], {})[r["arm"]] = r["heldout_loss"]

    seeds = sorted(s for s, d in by.items()
                   if "fluent-false" in d and "fluent-true" in d)
    diffs = [by[s]["fluent-false"] - by[s]["fluent-true"] for s in seeds]
    n = len(diffs)
    if n < 2:
        print("need >= 2 complete seed pairs"); return

    sd = statistics.stdev(diffs)          # the ONLY statistic that leaves here

    print(f"paired differences available : {n} seeds ({min(seeds)}..{max(seeds)})")
    print(f"metric                       : held-out next-token CE, 10,240 windows/model")
    print(f"contrast                     : fluent-false(s) - fluent-true(s), twin cancels")
    print()
    print(f"  SD of the paired difference : {sd:.6f} nats")
    print(f"  pilot's substituted sigma   : {PILOT_SUBSTITUTED_SIGMA:.6f} nats (twin-vs-twin, the wrong term)")
    print(f"  ratio                       : {sd/PILOT_SUBSTITUTED_SIGMA:.3f}x")
    print()
    print("  minimum detectable effect, paired t, alpha=0.05 two-sided:")
    print(f"    {'seeds':>6} {'80% power':>14} {'90% power':>14}")
    for k in (3, 8, 10, 16, 20, 32, 50):
        print(f"    {k:>6} {mde_in_sd_units(k)*sd:>14.6f} {mde_in_sd_units(k, 0.90)*sd:>14.6f}")
    print()
    print("  NOT COMPUTED (A-3): mean, sign, CI, test statistic, per-run losses.")

if __name__ == "__main__":
    main()
