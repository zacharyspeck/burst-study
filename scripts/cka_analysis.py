#!/usr/bin/env python
"""Per-layer CKA, paired within seed. Reads whatever `cka_pairs.py` wrote.

EXPLORATORY. Neither CKA nor any per-layer quantity is named in
docs/preregistration.md; section 8.4's decision rule lands on the plain
barrier and stops there. This is a third view of the same contrast, not a
third chance at it, and every p-value here is corrected across the thirteen
layers so that "some layer moved" cannot be assembled out of thirteen tries.

WHAT IS PAIRED WITH WHAT. Identical to the registered analysis and to
`stimulus_analysis.py`: each arm is differenced against its OWN seed's control,
and the seed-level differences are the sample. The registered primary contrast
-- `fluent-fabricated` against `fluent-attested` -- is formed on those differences, and
it is the one that isolates content, because both arms saw a fluent passage of
the same length at the same step and only one of them was true.

LEVELS ARE NOT TESTED, DIFFERENCES ARE. A CKA of 0.9993 between an arm and its
twin is not a hypothesis; asking whether it is 0.9993 for one arm and 0.9990
for another is. Levels are reported with a spread and nothing else, and the
`twin-vs-twin` group is carried alongside purely as the scale of two runs that
differ by everything.

Estimators are imported from analysis.py, so the arithmetic is the tested one.
"""
from __future__ import annotations
import argparse, json, statistics, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analysis import (AnalysisError, paired_t_test, bootstrap_ci, correct,
                      DEFAULT_LEVEL, DEFAULT_BOOTSTRAP)

REF = "twin"
PRIMARY = ("fluent-fabricated", "fluent-attested")


def group_key(doc: dict) -> str:
    if doc["design"] == "within-run":
        p = doc["pairs"][0]
        return f"within-run_{p['step_a']}_{p['step_b']}"
    return f"{doc['design']}_step{doc['pairs'][0]['step_a']}"


def summarize(diffs, label, level, n):
    ci = bootstrap_ci(diffs, level=level, n_resamples=n, label=label)
    t = paired_t_test(diffs)
    return {"n": len(diffs), "mean": ci["mean"],
            "sd": statistics.stdev(diffs) if len(diffs) > 1 else 0.0,
            "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
            "ci_excludes_zero": ci["excludes_zero"],
            "t": t["t"], "df": t["df"], "p": t["p"],
            "seeds_negative": sum(1 for x in diffs if x < 0),
            "values": diffs}


def levels(values):
    return {"n": len(values), "mean": statistics.fmean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values)}


def by_layer(doc, field="cka"):
    """(arm, seed) -> [value per layer], plus the layer count."""
    cells, n_layers = {}, None
    for p in doc["pairs"]:
        vals = [L[field] for L in p["layers"]]
        if n_layers is None:
            n_layers = len(vals)
        elif len(vals) != n_layers:
            raise AnalysisError(
                f"{p['name']} has {len(vals)} layers, an earlier pair had "
                f"{n_layers}. A per-layer mean over unequal layer counts would "
                "average different depths together.")
        key = (p["arm"], p["seed"]) if p["seed"] is not None else ("floor", p["name"])
        if key in cells:
            raise AnalysisError(f"duplicate cell {key} in {doc['design']}")
        cells[key] = vals
    return cells, n_layers


def contrasts(cells, seeds, arms, n_layers, level, n_res, tag, method):
    """Paired arm-minus-arm differences, layer by layer, corrected across layers."""
    out, families = {}, {}

    def add(name, pick):
        rows = []
        for L in range(n_layers):
            diffs = [pick(cells, s, L) for s in seeds]
            rows.append(summarize(diffs, f"{tag}/{name}/L{L}", level, n_res))
        adj = correct([r["p"] for r in rows], method)
        for r, q in zip(rows, adj):
            r["p_adjusted"] = q
        out[name] = rows
        families[name] = method

    if all(a in arms for a in PRIMARY):
        add("primary_false_minus_true",
            lambda c, s, L: c[(PRIMARY[0], s)][L] - c[(PRIMARY[1], s)][L])
    if all(a in arms for a in PRIMARY) and "random-chars" in arms:
        add("fluent_pooled_minus_random",
            lambda c, s, L: (c[(PRIMARY[0], s)][L] + c[(PRIMARY[1], s)][L]) / 2.0
                            - c[("random-chars", s)][L])
    return out, families


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--correction", required=True,
                    choices=("holm", "benjamini_hochberg", "benjamini_yekutieli",
                             "none"),
                    help="no default: see analysis.correct and D-4")
    ap.add_argument("--level", type=float, default=DEFAULT_LEVEL)
    ap.add_argument("--resamples", type=int, default=DEFAULT_BOOTSTRAP)
    a = ap.parse_args()

    docs = {}
    for f in a.inputs:
        doc = json.loads(f.read_text())
        k = group_key(doc)
        if k in docs:
            raise AnalysisError(f"two inputs are the same group {k}: "
                                f"{docs[k]['_file']} and {f}")
        doc["_file"] = str(f)
        docs[k] = doc

    res = {"WHAT_THIS_IS": ("Exploratory. Per-layer CKA between checkpoints, paired "
                            "within seed against the seed-matched control. Not "
                            "pre-registered; corrected across layers."),
           "correction": a.correction, "level": a.level,
           "cka_variant": next(iter(docs.values()))["cka_variant"],
           "batch": next(iter(docs.values()))["batch"],
           "groups": {}}

    for key, doc in sorted(docs.items()):
        cells, n_layers = by_layer(doc)
        cos, _ = by_layer(doc, "cosine_median")
        g = {"design": doc["design"], "source": doc["_file"],
             "n_pairs": doc["n_pairs"], "n_layers": n_layers,
             "activation_route_cross_check": doc["activation_route_cross_check"],
             "levels": {}, "cosine_levels": {}, "contrasts": {}}

        if doc["design"] == "twin-vs-twin":
            g["WHAT_THE_FLOOR_IS"] = (
                "Across-seed control pairs. These runs differ by initialization "
                "AND data order, so this is the scale of two genuinely different "
                "runs, NOT measurement error.")
            floor = [v for k, v in cells.items() if k[0] == "floor"]
            g["levels"][REF] = [levels([f[L] for f in floor])
                                for L in range(n_layers)]
            g["cosine_levels"][REF] = [levels([f[L] for f in cos.values()])
                                       for L in range(n_layers)]
        else:
            seeds = sorted({s for (_, s) in cells})
            arms = sorted({arm for (arm, _) in cells})
            for arm in arms:
                have = [s for s in seeds if (arm, s) in cells]
                if len(have) != len(seeds):
                    raise AnalysisError(
                        f"{key}: arm {arm} has {len(have)} seeds, the panel has "
                        f"{len(seeds)}. An unbalanced panel silently changes "
                        "which seeds a paired difference is over.")
                g["levels"][arm] = [levels([cells[(arm, s)][L] for s in seeds])
                                    for L in range(n_layers)]
                g["cosine_levels"][arm] = [levels([cos[(arm, s)][L] for s in seeds])
                                           for L in range(n_layers)]
            g["seeds"] = seeds
            # within-run carries the control as an arm, so it gets the extra
            # arm-minus-control contrast the arm-vs-twin design already is.
            if REF in arms:
                extra = {}
                for arm in [x for x in arms if x != REF]:
                    rows = []
                    for L in range(n_layers):
                        d = [cells[(arm, s)][L] - cells[(REF, s)][L] for s in seeds]
                        rows.append(summarize(d, f"{key}/{arm}_minus_twin/L{L}",
                                              a.level, a.resamples))
                    for r, q in zip(rows, correct([r["p"] for r in rows],
                                                  a.correction)):
                        r["p_adjusted"] = q
                    extra[f"{arm}_minus_twin"] = rows
                g["contrasts"].update(extra)
            c, _ = contrasts(cells, seeds, [x for x in arms if x != REF],
                             n_layers, a.level, a.resamples, key, a.correction)
            g["contrasts"].update(c)
        res["groups"][key] = g

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")

    for key, g in res["groups"].items():
        print(f"\n=== {key}  ({g['n_pairs']} pairs, {g['n_layers']} layers)")
        arms = sorted(g["levels"])
        print("  CKA by layer (mean over seeds)")
        print("    layer  " + "  ".join(f"{x:>16s}" for x in arms))
        for L in range(g["n_layers"]):
            print(f"    L{L:<5d} " + "  ".join(
                f"{g['levels'][x][L]['mean']:16.8f}" for x in arms))
        for name, rows in g["contrasts"].items():
            best = min(rows, key=lambda r: r["p"])
            k = rows.index(best)
            hit = [i for i, r in enumerate(rows) if r["p_adjusted"] < 0.05]
            print(f"  contrast {name}: best layer L{k} "
                  f"mean={best['mean']:+.8f} p={best['p']:.4f} "
                  f"p_adj={best['p_adjusted']:.4f}; "
                  f"layers surviving {a.correction}: {hit if hit else 'none'}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
