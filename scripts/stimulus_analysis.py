#!/usr/bin/env python
"""Did any of the injected passage stick? Paired within seed, like the registered analysis.

EXPLORATORY throughout. Reads the per-model output of `stimulus_probe.py`.

THE COMPARISON THAT MATTERS IS A DIFFERENCE IN DIFFERENCES. Every final model is
scored on all three stimulus texts, so for a given passage there are two
contrasts:

    own   = score(model that saw P, on P)      - score(its twin, on P)
    other = score(model that saw Q != P, on P) - score(its twin, on P)

`own` alone confounds "learned this passage" with "drifted in a way that happens
to help on any passage of this kind". `own - other` removes that. A real effect
of a single exposure is `own` negative (loss down / log-prob up) and `other` at
zero.

Estimators are imported from analysis.py, so the arithmetic is the tested one.
"""
from __future__ import annotations
import argparse, json, statistics, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analysis import paired_t_test, bootstrap_ci, DEFAULT_LEVEL, DEFAULT_BOOTSTRAP
from burst.config import INJECTING_ARMS

REF = "twin"


def load(d: Path):
    out = {}
    for p in sorted(d.glob("*.json")):
        j = json.loads(p.read_text())
        out[(j["arm"], j["seed"])] = j
    return out


def paired(cells, seeds, get):
    """[value(arm,seed) - value(twin,seed)] for each seed."""
    return [get(cells[a]) - get(cells[b]) for a, b in seeds]


def summarize(diffs, label, level, n):
    ci = bootstrap_ci(diffs, level=level, n_resamples=n, label=label)
    t = paired_t_test(diffs)
    neg = sum(1 for x in diffs if x < 0)
    return {"n": len(diffs), "mean": ci["mean"],
            "sd": statistics.stdev(diffs) if len(diffs) > 1 else 0.0,
            "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
            "ci_excludes_zero": ci["excludes_zero"],
            "t": t["t"], "df": t["df"], "p": t["p"],
            "seeds_negative": neg, "values": diffs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--level", type=float, default=DEFAULT_LEVEL)
    ap.add_argument("--resamples", type=int, default=DEFAULT_BOOTSTRAP)
    a = ap.parse_args()

    cells = load(a.probe_dir)
    seeds = sorted({s for (_, s) in cells})
    arms = sorted({arm for (arm, _) in cells if arm != REF})
    pairs = {arm: [((arm, s), (REF, s)) for s in seeds] for arm in arms}

    res = {"WHAT_THIS_IS": ("Exploratory. Scores of the stimulus texts themselves under each "
                            "final model, paired within seed against the seed-matched control."),
           "n_seeds": len(seeds), "seeds": seeds,
           "passage_loss": {}, "passage_loss_diff_in_diff": {},
           "minimal_pairs": {}, "bigrams": {}}

    # ---- 1. passage loss: every arm scored on every stimulus text ----
    for probe in INJECTING_ARMS:
        for key, field in (("mean_all", "mean_all"), ("mean_content", "mean_content")):
            for arm in arms:
                d = paired(cells, pairs[arm],
                           lambda j: j["passage_loss"][probe][field])
                res["passage_loss"].setdefault(probe, {}).setdefault(key, {})[arm] = \
                    summarize(d, f"pl/{probe}/{key}/{arm}", a.level, a.resamples)
        # difference in differences: the arm that SAW this passage, against the
        # mean of the arms that did not.
        if probe in arms:
            others = [x for x in arms if x != probe]
            for key in ("mean_all", "mean_content"):
                own = res["passage_loss"][probe][key][probe]["values"]
                oth = [statistics.fmean(
                           res["passage_loss"][probe][key][o]["values"][i] for o in others)
                       for i in range(len(seeds))]
                did = [x - y for x, y in zip(own, oth)]
                res["passage_loss_diff_in_diff"].setdefault(probe, {})[key] = \
                    summarize(did, f"did/{probe}/{key}", a.level, a.resamples)

    # ---- 2. minimal pairs ----
    labels = sorted(next(iter(cells.values()))["minimal_pairs"])
    for lab in labels:
        for arm in arms:
            d = paired(cells, pairs[arm],
                       lambda j: j["minimal_pairs"][lab]["delta_logprob"])
            res["minimal_pairs"].setdefault(lab, {})[arm] = \
                summarize(d, f"mp/{lab}/{arm}", a.level, a.resamples)

    # ---- 3. bigrams ----
    for lab in sorted(next(iter(cells.values()))["bigrams"]):
        for arm in arms:
            d = paired(cells, pairs[arm], lambda j: j["bigrams"][lab]["logprob"])
            res["bigrams"].setdefault(lab, {})[arm] = \
                summarize(d, f"bg/{lab}/{arm}", a.level, a.resamples)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")

    def line(name, r):
        star = "  <-- CI excludes 0" if r["ci_excludes_zero"] else ""
        print(f"  {name:34s} mean={r['mean']:+.6f}  t({r['df']})={r['t']:+7.3f}  "
              f"p={r['p']:.4f}  neg {r['seeds_negative']}/{r['n']}{star}")

    print(f"n = {len(seeds)} seeds, paired within seed against `{REF}`\n")
    print("PASSAGE LOSS, arm minus its control (negative = the arm got BETTER at the text)")
    for probe in INJECTING_ARMS:
        print(f"\n text = {probe}")
        for key in ("mean_all", "mean_content"):
            print(f"  [{key}]")
            for arm in arms:
                line(f"{arm} on {probe}", res["passage_loss"][probe][key][arm])
    print("\nDIFFERENCE IN DIFFERENCES  (saw-it minus didn't-see-it, on the same text)")
    for probe, d in res["passage_loss_diff_in_diff"].items():
        for key, r in d.items():
            line(f"{probe} [{key}]", r)
    print("\nMINIMAL PAIRS, change in log P(asserted) - log P(alternative)")
    for lab in labels:
        for arm in arms:
            if lab.startswith("true.") and arm != "fluent-attested": continue
            if lab.startswith("false.") and arm != "fluent-fabricated": continue
            line(f"{lab} [{arm}]", res["minimal_pairs"][lab][arm])
    print("\nNAME BIGRAM, change in log P(surname | given name)")
    for lab in sorted(res["bigrams"]):
        for arm in arms:
            if "gizmo" in lab and arm != "fluent-fabricated": continue
            if "jimmie" in lab and arm != "fluent-attested": continue
            line(f"{lab} [{arm}]", res["bigrams"][lab][arm])
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
