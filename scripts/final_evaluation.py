#!/usr/bin/env python
"""Full evaluation of box A's 16 runs. Uses the repo's own tested estimators.

STOPPING RULE, recorded because it decides whether any of this is legitimate:
the study stopped at n=8 because COMPUTE RAN OUT, a decision taken and recorded
BEFORE any mean, sign or test statistic was examined (the only prior look was
the A-3-permitted variance-only one). A data-independent stopping rule is not
optional stopping. What A-3 forbade was choosing n knowing the answer; that did
not happen here.
"""
from __future__ import annotations
import glob, json, statistics, sys
from pathlib import Path

sys.path.insert(0, "/home/ubuntu/burst-study")
sys.path.insert(0, "/home/ubuntu/burst-study/scripts")
import analysis as A

RUNS = Path("/home/ubuntu/runs")


def describe(name, diffs, seeds, unit):
    print(f"\n{'='*74}\n{name}\n{'='*74}")
    print(f"  n = {len(diffs)} seeds {seeds}")
    t = A.paired_t_test(diffs)
    ci = A.bootstrap_ci(diffs, label=name)
    lo, hi = ci["ci_low"], ci["ci_high"]
    print(f"  mean difference : {t['mean']:+.8f} {unit}")
    print(f"  sd              : {t['sd']:.8f}")
    print(f"  t({t['df']})          : {t['t']:+.4f}")
    print(f"  p (two-sided)   : {t['p_raw'] if 'p_raw' in t else t.get('p'):.5f}")
    print(f"  95% bootstrap CI: [{lo:+.8f}, {hi:+.8f}]")
    print(f"  CI excludes 0   : {ci['excludes_zero']}")
    neg = sum(1 for d in diffs if d < 0)
    print(f"  sign            : {len(diffs)-neg} positive / {neg} negative")
    return {"n": len(diffs), "mean": t["mean"], "sd": t["sd"], "t": t["t"],
            "df": t["df"], "p": t.get("p_raw", t.get("p")),
            "ci_low": lo, "ci_high": hi, "per_seed": dict(zip(seeds, diffs))}


def main():
    out = {}

    # --- held-out loss, paired within seed (twin cancels for a per-model metric)
    by = {}
    for f in glob.glob(str(RUNS / "*/heldout_eval.json")):
        r = json.load(open(f)); by.setdefault(r["seed"], {})[r["arm"]] = r["heldout_loss"]
    seeds = sorted(s for s, d in by.items() if len(d) == 2)
    diffs = [by[s]["fluent-fabricated"] - by[s]["fluent-attested"] for s in seeds]
    out["heldout_loss_delta"] = describe(
        "HELD-OUT LOSS:  fluent-fabricated - fluent-attested, per seed (10,240 windows/model)",
        diffs, seeds, "nats")
    out["heldout_loss_raw"] = {str(s): by[s] for s in seeds}

    # --- arm-vs-arm barrier and L2
    pair = {}
    for f in sorted(glob.glob(str(RUNS / "armpair_seed*.json"))):
        r = json.load(open(f)); pair[r["seed"]] = r
    ps = sorted(pair)
    bar = [pair[s]["barrier"]["max_excess"] for s in ps]
    l2s = [pair[s]["l2_raw"] for s in ps]
    print(f"\n{'='*74}\nARM-vs-ARM BARRIER  (NOT section 8.4's headline, which is arm-vs-twin)\n{'='*74}")
    print(f"  n = {len(ps)} seeds {ps}")
    print(f"  mean max_excess : {statistics.fmean(bar):.8f}   sd {statistics.stdev(bar):.8f}")
    print(f"  range           : {min(bar):.8f} .. {max(bar):.8f}")
    print(f"  all > 0         : {all(b > 0 for b in bar)}")
    print(f"  per seed        : " + ", ".join(f"s{s}={b:.5f}" for s, b in zip(ps, bar)))
    print(f"\n  raw L2 between the two arms' final weights")
    print(f"  mean            : {statistics.fmean(l2s):.4f}   sd {statistics.stdev(l2s):.4f}")
    print(f"  range           : {min(l2s):.4f} .. {max(l2s):.4f}")
    out["arm_vs_arm_barrier"] = {"n": len(ps), "mean": statistics.fmean(bar),
                                 "sd": statistics.stdev(bar),
                                 "per_seed": {str(s): b for s, b in zip(ps, bar)}}
    out["arm_vs_arm_l2"] = {"mean": statistics.fmean(l2s), "sd": statistics.stdev(l2s),
                            "per_seed": {str(s): v for s, v in zip(ps, l2s)}}

    # --- per-run provenance and training facts
    runs = {}
    for d in sorted(RUNS.glob("seed*_fluent-*")):
        tr = d / "train_record.json"
        if not tr.exists(): continue
        r = json.load(open(tr))
        f = r.get("injection_fired") or {}
        runs[d.name] = {
            "final_state_digest": r["final_state_digest"],
            "wall_seconds": r["wall_seconds"],
            "steps_run": r["steps_run"],
            "resume": r.get("resume"),
            "injection_step": f.get("step"), "batch_slot": f.get("batch_slot"),
            "burst_file_sha256": f.get("burst_file_sha256"),
            "device": r["determinism"]["device_name"],
            "torch": r["determinism"]["torch_version"],
            "heldout_loss": by.get(int(d.name[4:6]), {}).get(d.name.split("_",1)[1]),
        }
    out["runs"] = runs
    print(f"\n{'='*74}\nPROVENANCE: {len(runs)} runs, all digests distinct: "
          f"{len({v['final_state_digest'] for v in runs.values()}) == len(runs)}, "
          f"resumes used: {sum(1 for v in runs.values() if v['resume'])}\n{'='*74}")

    Path("/home/ubuntu/burst-study/docs/measurements/2026-08-09-boxA-results.json").write_text(
        json.dumps(out, indent=2, sort_keys=True))
    print("wrote docs/measurements/2026-08-09-boxA-results.json")


if __name__ == "__main__":
    main()
