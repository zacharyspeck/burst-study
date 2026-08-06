#!/usr/bin/env python
"""Per-seed paired differences, the twin noise floor, and the arm ordering. Step 16.

WHAT THIS COMPUTES

For each injecting arm and each seed, the difference between that arm and its
SEED-MATCHED TWIN. That pairing is the whole design: the twin shares the seed,
so it shares the initial weights and the data order, and the two runs differ
only in the burst. A paired difference therefore removes seed-to-seed variation
rather than averaging over it.

    effect(arm, seed) = metric(arm, seed) - metric(twin, seed)

The noise floor is TWIN AGAINST TWIN ACROSS SEEDS, which is what
docs/spec-v4.md names it: every pair of distinct seeds' twins, giving n*(n-1)/2
differences. That is the scale of variation attributable to seed alone, with no
burst anywhere in it. An arm effect that does not clear it is not an effect.

    Note the asymmetry, because it is easy to get wrong: the paired difference
    is WITHIN a seed and the noise floor is ACROSS seeds. They are not two
    samples of the same quantity, and the noise floor is the wider of the two
    by construction -- under the 2026-08-03 per-seed data-order ruling it
    bundles initialization AND data order.

STDLIB ONLY, AND THAT IS DELIBERATE

No numpy, no scipy, no pandas, no matplotlib. Two reasons:

1. This runs and is TESTED in the torch-free environment, which has none of
   them. A statistical claim is better proved where there are fewer moving
   parts, the same argument that keeps data_order.py stdlib-only.
2. scipy exists in the ML environment, so the hand-written t-distribution and
   the corrections are CROSS-CHECKED against it there -- an independent route,
   not a second copy. If they ever disagree, the suite says so.

The bootstrap draws from SHA-256 in counter mode rather than any library PRNG,
for the reason data_order.py records: numpy does not guarantee Generator
stream stability across versions, and a confidence interval that moves with a
library upgrade is not reproducible.

THE CORRECTION METHOD HAS NO DEFAULT

`docs/spec-v4.md` has no section 9.4 and no statistics section at all, so there
is no specified correction to apply. Which one is used decides which arms are
reported as separated, and that is the study's headline claim -- so it is a
REQUIRED parameter and this module refuses without it. See D-4 in
docs/decisions-pending.md for the candidates and for the prior question of what
the family of tests even is, which changes every corrected number.

NO TRAINED MODELS EXIST. Everything here is built against synthetic inputs and
every quantity has a responsiveness test: fabricate a known effect and assert
it is found, fabricate none and assert it is not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from burst.config import ARMS, INJECTING_ARMS  # noqa: E402

#: Correction methods. NO DEFAULT -- see the module docstring and D-4.
CORRECTIONS = ("holm", "benjamini_hochberg", "benjamini_yekutieli", "none")

#: Ten seeds is the floor for anything reported, for the reason the rest of
#: this build records: low-seed numbers here were overturned three times when
#: the seed count widened.
MIN_SEEDS = 10

DEFAULT_REFERENCE_ARM = "twin"
DEFAULT_BOOTSTRAP = 10_000
DEFAULT_LEVEL = 0.95


class AnalysisError(Exception):
    """Raised for every refusal here. Never an assert."""


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Panel:
    """One metric's value for every (arm, seed). Complete, or it refuses.

    A missing cell is not filled in, dropped, or imputed. A paired design with
    a hole in it silently stops being paired for that seed, and the resulting
    mean is over a different set of seeds than the one it claims.
    """

    metric: str
    values: dict          # arm -> {seed: value}
    seeds: tuple
    arms: tuple

    def for_arm(self, arm: str) -> dict:
        if arm not in self.values:
            raise AnalysisError(f"no values for arm {arm!r}")
        return self.values[arm]


def load_panel(records, metric: str, *, reference: str = DEFAULT_REFERENCE_ARM,
               min_seeds: int = MIN_SEEDS) -> Panel:
    """Build a complete panel from flat records, refusing anything ragged.

    `records` is an iterable of mappings with `seed`, `arm`, `metric`, `value`.
    Flat records rather than a nested dict because that is the shape a run
    directory scan produces, and reshaping is this function's job rather than
    the caller's.
    """
    values: dict = {}
    for row in records:
        try:
            if row["metric"] != metric:
                continue
            arm, seed, value = row["arm"], int(row["seed"]), float(row["value"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AnalysisError(
                f"malformed record {row!r}: needs seed, arm, metric, value "
                f"({exc})") from exc
        if arm not in ARMS:
            raise AnalysisError(
                f"unknown arm {arm!r} in the input; expected one of "
                f"{', '.join(ARMS)}. Arms are not invented here.")
        if seed in values.setdefault(arm, {}):
            raise AnalysisError(
                f"duplicate value for arm {arm!r} seed {seed}. A paired design "
                "cannot silently keep the last of two disagreeing numbers.")
        values[arm][seed] = value

    if not values:
        raise AnalysisError(f"no records for metric {metric!r}")
    if reference not in values:
        raise AnalysisError(
            f"the reference arm {reference!r} is absent, so no paired "
            "difference can be formed. Every effect here is measured against "
            "the seed-matched twin; without it there is nothing to pair to.")

    seeds = sorted(values[reference])
    if len(seeds) < min_seeds:
        raise AnalysisError(
            f"{len(seeds)} seeds for the reference arm; the floor is "
            f"{min_seeds}. Low-seed numbers in this build were overturned "
            "three separate times when the seed count widened, which is why "
            "the floor exists rather than being advisory.")

    for arm, by_seed in sorted(values.items()):
        missing = sorted(set(seeds) - set(by_seed))
        if missing:
            raise AnalysisError(
                f"arm {arm!r} is missing seeds {missing}. The panel is not "
                "complete, and a paired difference over a subset of seeds is "
                "an average over a different study than the one it claims.")
    return Panel(metric=metric, values=values, seeds=tuple(seeds),
                 arms=tuple(sorted(values)))


# ---------------------------------------------------------------------------
# The two quantities the design turns on
# ---------------------------------------------------------------------------


def paired_differences(panel: Panel, arm: str,
                       reference: str = DEFAULT_REFERENCE_ARM) -> list:
    """arm minus its SEED-MATCHED reference, one per seed.

    Within a seed, never across. The twin shares the seed and therefore the
    initial weights and the data order, so this difference removes seed-to-seed
    variation rather than averaging over it.
    """
    if arm == reference:
        raise AnalysisError(
            f"cannot pair {arm!r} against itself; that is the noise floor, "
            "which is computed across seeds rather than within them")
    values, ref = panel.for_arm(arm), panel.for_arm(reference)
    return [values[s] - ref[s] for s in panel.seeds]


def noise_floor(panel: Panel, reference: str = DEFAULT_REFERENCE_ARM) -> list:
    """Every twin against every other twin. ACROSS seeds, n*(n-1)/2 of them.

    docs/spec-v4.md: the twin is "the per-seed displacement reference and,
    twin-vs-twin across seeds, the noise floor". No burst appears anywhere in
    this, so it is the scale of variation attributable to seed alone -- and
    under the per-seed data-order ruling that means initialization AND data
    order together.

    Signed differences are returned rather than absolute ones. The caller
    decides whether the comparison is one- or two-sided; taking |.| here would
    make that decision silently and halve the apparent spread.
    """
    ref = panel.for_arm(reference)
    seeds = panel.seeds
    return [ref[b] - ref[a]
            for i, a in enumerate(seeds) for b in seeds[i + 1:]]


# ---------------------------------------------------------------------------
# The two PRE-REGISTERED confirmatory contrasts
#
# docs/preregistration.md fixes two contrasts as confirmatory and everything
# else as exploratory. Until now neither could be computed here: this module
# compared each arm against a reference and nothing else, so §5 was reachable
# only by passing `--reference fluent-true` -- which makes the noise floor
# fluent-true-against-itself while every label still said "twin" (D-8b) -- and
# §6 had no pooling construct at all.
#
# NEITHER FUNCTION RULES ON THE FAMILY OF TESTS. Adding two contrasts changes
# how many comparisons exist, and therefore what a correction divides by, which
# is D-4 and is open. `--correction` stays required with no default.
# ---------------------------------------------------------------------------

#: §5's primary contrast, docs/preregistration.md:82. Order is the document's.
PRIMARY_CONTRAST = ("fluent-false", "fluent-true")

#: §6's secondary contrast, docs/preregistration.md:109-118. The two fluent arms
#: pooled, against pos-substituted.
SECONDARY_POOLED = ("fluent-false", "fluent-true")
SECONDARY_AGAINST = "pos-substituted"


def _check_contrast_arms(panel: Panel, arms, reference: str) -> None:
    """Shared guard. Every arm present, distinct, and none of them the reference.

    Its own guard rather than `paired_differences`' -- that one refuses only
    `arm == reference`, which does not catch two arms being the same as each
    other, and a contrast of an arm against itself is identically zero for every
    seed. Zero is a plausible-looking answer, which is the shape this repo keeps
    getting caught by.
    """
    seen = []
    for arm in arms:
        if arm == reference:
            raise AnalysisError(
                f"arm {arm!r} is the reference, so its displacement from itself "
                "is zero by construction and the contrast is meaningless. The "
                "reference is the thing every arm is measured against, not a "
                "side of a contrast.")
        if arm in seen:
            raise AnalysisError(
                f"arm {arm!r} appears twice in the contrast. A contrast of an "
                "arm against itself is identically zero on every seed, which "
                "looks like a null result rather than an error.")
        if arm not in panel.values:
            raise AnalysisError(
                f"arm {arm!r} is absent from the panel; it holds "
                f"{', '.join(panel.arms)}. The contrast is not computed over "
                "whichever arms happen to be present.")
        seen.append(arm)


def arm_vs_arm_differences(panel: Panel, arm_a: str, arm_b: str,
                           reference: str = DEFAULT_REFERENCE_ARM) -> list:
    """§5's PRIMARY contrast: two arms' displacements, differenced per seed.

    Each arm is first reduced to its displacement from its OWN seed-matched
    reference, then those two displacements are differenced within the seed:

        contrast(seed) = [arm_a(seed) - twin(seed)] - [arm_b(seed) - twin(seed)]

    THE REFERENCE CANCELS, AND THIS FORM CHECKS NOTHING THE SIMPLIFIED ONE
    WOULD NOT. An earlier version of this docstring claimed the routing through
    `paired_differences` verified that both arms share a reference value at each
    seed. It does not: `load_panel` already refuses an incomplete panel, so both
    arms always have the same reference value at a given seed, and
    `arm_a - arm_b` is arithmetically identical on every input this module
    accepts. A mutation to that simpler form passed the whole suite -- recorded
    as M9 rather than papered over.

    It is written this way for LEGIBILITY, which is a weaker reason and the
    honest one: §5 registers a contrast between two *displacements*, and code
    shaped like the thing it implements is easier to check against the document
    than code that has been algebraically simplified past it.

    NOT reachable by passing `--reference fluent-true`. That produces the same
    numbers and a noise floor of fluent-true against itself across seeds, under
    labels that say twin (D-8b, open).
    """
    _check_contrast_arms(panel, (arm_a, arm_b), reference)
    da = paired_differences(panel, arm_a, reference)
    db = paired_differences(panel, arm_b, reference)
    return [x - y for x, y in zip(da, db)]


def pooled_differences(panel: Panel, arms,
                       reference: str = DEFAULT_REFERENCE_ARM) -> list:
    """Pool arms by AVERAGING WITHIN EACH SEED. One value per seed, never more.

    POOLING IS NOT STACKING, and the difference is the whole hygiene of §6.
    `fluent-false` and `fluent-true` at seed 3 share an initialization, a data
    order, and a twin. Stacking them into one group of twenty would treat
    correlated observations as independent: n doubles, the standard error falls
    by ~sqrt(2), and every p-value shrinks for free. Nothing in the output would
    look wrong -- the mean is unchanged and only the spread moves.

    So this returns exactly `len(panel.seeds)` values and asserts it. A test
    pins the count and a mutation to stacking turns it red.
    """
    arms = tuple(arms)
    if len(arms) < 2:
        raise AnalysisError(
            f"pooling needs at least two arms; got {list(arms)}. Pooling one "
            "arm is just that arm, and naming it pooled would misdescribe it.")
    _check_contrast_arms(panel, arms, reference)

    per_arm = [paired_differences(panel, arm, reference) for arm in arms]
    pooled = [statistics.fmean(at_seed) for at_seed in zip(*per_arm)]

    if len(pooled) != len(panel.seeds):
        raise AnalysisError(
            f"pooling produced {len(pooled)} rows for {len(panel.seeds)} "
            f"seeds. Pooling must AVERAGE within a seed, not concatenate "
            f"across arms: {len(arms)} arms stacked would give "
            f"{len(arms) * len(panel.seeds)} rows and would treat "
            "seed-correlated observations as independent.")
    return pooled


def pooled_vs_arm_differences(panel: Panel, pooled_arms, arm: str,
                              reference: str = DEFAULT_REFERENCE_ARM) -> list:
    """§6's SECONDARY contrast: pooled displacement minus one arm's, per seed."""
    pooled_arms = tuple(pooled_arms)
    _check_contrast_arms(panel, pooled_arms + (arm,), reference)
    pooled = pooled_differences(panel, pooled_arms, reference)
    other = paired_differences(panel, arm, reference)
    return [p - o for p, o in zip(pooled, other)]


def displacement_phrase(reference: str) -> str:
    """"minus its seed-matched twin", with the ACTUAL reference named."""
    return f"minus its seed-matched {reference}"


def contrast_label(kind: str, *, arms, against=None,
                   reference: str = DEFAULT_REFERENCE_ARM) -> str:
    """The label, DERIVED from the arms actually compared.

    Every label in this module used to hardcode "twin" while the number followed
    `--reference` (D-8b). Deriving is the fix; substituting a different constant
    string would be the same defect one layer over. A test asserts the label
    names the same arms the numbers came from.
    """
    each = displacement_phrase(reference)
    if kind == "arm_vs_arm":
        a, b = arms
        return (f"{a} vs {b}, each {each}, differenced within seed")
    if kind == "pooled_vs_arm":
        pooled = " + ".join(arms)
        return (f"pooled ({pooled}) vs {against}, each {each}, "
                "pooled by AVERAGING within seed then differenced within seed")
    raise AnalysisError(f"unknown contrast kind {kind!r}")


def contrast(panel: Panel, kind: str, *, arms, against=None,
             reference: str = DEFAULT_REFERENCE_ARM, level: float = DEFAULT_LEVEL,
             n_resamples: int = DEFAULT_BOOTSTRAP) -> dict:
    """One contrast, with its label derived and its per-seed values kept.

    Returns no verdict about the family of tests: `p_raw` only, uncorrected.
    Which correction applies, and across how many comparisons, is D-4.
    """
    arms = tuple(arms)
    if kind == "arm_vs_arm":
        diffs = arm_vs_arm_differences(panel, arms[0], arms[1], reference)
        n_arms_pooled = None
    elif kind == "pooled_vs_arm":
        diffs = pooled_vs_arm_differences(panel, arms, against, reference)
        n_arms_pooled = len(arms)
    else:
        raise AnalysisError(f"unknown contrast kind {kind!r}")

    test = paired_t_test(diffs)
    label = contrast_label(kind, arms=arms, against=against,
                           reference=reference)
    ci = bootstrap_ci(diffs, level=level, n_resamples=n_resamples,
                      label=f"{panel.metric}/{label}")
    return {
        "kind": kind,
        "label": label,
        "arms": list(arms),
        "against": against,
        "reference_arm": reference,
        "n_seeds": len(diffs),
        "n_rows": len(diffs),
        "n_arms_pooled": n_arms_pooled,
        "by_seed": {str(s): d for s, d in zip(panel.seeds, diffs)},
        "min": min(diffs),
        "median": statistics.median(diffs),
        "max": max(diffs),
        "mean": ci["mean"],
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "ci_excludes_zero": ci["excludes_zero"],
        "t": test["t"],
        "df": test["df"],
        "p_raw": test["p"],
        "POOLING": (
            "AVERAGED within each seed, giving one row per seed. Not stacked: "
            f"{n_arms_pooled} arms at one seed share an initialization, a data "
            "order and a reference, so stacking would treat correlated "
            "observations as independent and inflate n."
            if n_arms_pooled else None),
        "NO_CORRECTION_APPLIED": (
            "p_raw is uncorrected. These two contrasts change the size of the "
            "family of tests, and both the family and the correction are D-4, "
            "which is open. Nothing here divides by anything."),
    }


def registered_contrasts(panel: Panel, *,
                         reference: str = DEFAULT_REFERENCE_ARM,
                         level: float = DEFAULT_LEVEL,
                         n_resamples: int = DEFAULT_BOOTSTRAP) -> dict:
    """The two contrasts docs/preregistration.md registers, when computable.

    Absent arms are reported as absent rather than skipped silently: a
    pre-registered contrast that could not be computed is a fact about the run.
    """
    out = {}
    for name, kind, arms, against in (
        ("primary", "arm_vs_arm", PRIMARY_CONTRAST, None),
        ("secondary", "pooled_vs_arm", SECONDARY_POOLED, SECONDARY_AGAINST),
    ):
        needed = tuple(arms) + ((against,) if against else ())
        missing = [a for a in needed if a not in panel.values]
        if missing:
            out[name] = {
                "computed": False,
                "arms": list(arms),
                "against": against,
                "missing_arms": missing,
                "WHY_ABSENT": (
                    f"the panel does not hold {', '.join(missing)}, so this "
                    "pre-registered contrast could not be computed from this "
                    "input."),
            }
            continue
        cell = contrast(panel, kind, arms=arms, against=against,
                        reference=reference, level=level,
                        n_resamples=n_resamples)
        cell["computed"] = True
        cell["preregistered_as"] = name
        out[name] = cell
    return out


# ---------------------------------------------------------------------------
# Distributions, hand-written so this stays stdlib-only
# ---------------------------------------------------------------------------


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny, eps, max_iter = 1e-300, 3e-16, 300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    raise AnalysisError(
        f"the incomplete beta continued fraction did not converge for "
        f"a={a}, b={b}, x={x}. Refusing to return a partially converged "
        "value: a p-value that is quietly wrong is worse than none.")


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """I_x(a, b). The piece Student's t needs, and the only special function here."""
    if not 0.0 <= x <= 1.0:
        raise AnalysisError(f"x must be in [0, 1]; got {x}")
    if x in (0.0, 1.0):
        return x
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + b * math.log1p(-x) + a * math.log(x)) * _betacf(b, a, 1.0 - x) / b


def student_t_sf(t: float, df: float) -> float:
    """Two-sided survival: P(|T| >= |t|) for Student's t with `df` degrees.

    I_{df/(df + t^2)}(df/2, 1/2). Cross-checked against scipy in the ML
    environment; scipy is not importable in the torch-free one, which is
    exactly why this is written out rather than imported.
    """
    if df <= 0:
        raise AnalysisError(f"degrees of freedom must be positive; got {df}")
    if t == 0.0:
        return 1.0
    return regularized_incomplete_beta(df / 2.0, 0.5, df / (df + t * t))


def paired_t_test(differences) -> dict:
    """One-sample t on the paired differences. The null is "no effect".

    Reported with the statistic and the degrees of freedom, not just a p, so a
    reader can see what the number rests on. n = 10 gives df = 9, and a t on
    nine degrees of freedom is a different creature from a z.
    """
    diffs = [float(d) for d in differences]
    n = len(diffs)
    if n < 2:
        raise AnalysisError(
            f"a paired t-test needs at least 2 differences; got {n}")
    mean = statistics.fmean(diffs)
    sd = statistics.stdev(diffs)
    if sd == 0.0:
        # Every difference identical. Not a failure -- an exactly-zero spread
        # means either a perfect effect or a metric that does not vary, and
        # both deserve to be visible rather than dividing by zero.
        return {"n": n, "mean": mean, "sd": 0.0, "t": None, "df": n - 1,
                "p": 0.0 if mean != 0.0 else 1.0,
                "NOTE": ("every paired difference is identical, so the t "
                         "statistic is undefined; p is reported as the "
                         "degenerate limit")}
    se = sd / math.sqrt(n)
    t = mean / se
    return {"n": n, "mean": mean, "sd": sd, "se": se, "t": t, "df": n - 1,
            "p": student_t_sf(t, n - 1)}


# ---------------------------------------------------------------------------
# Bootstrap, deterministic across machines and library versions
# ---------------------------------------------------------------------------


def _draw_indices(material: bytes, count: int, modulus: int) -> list:
    """SHA-256 in counter mode. No library PRNG, for data_order.py's reason."""
    out, block = [], 0
    while len(out) < count:
        digest = hashlib.sha256(material + b"/" + str(block).encode()).digest()
        for k in range(4):
            if len(out) >= count:
                break
            out.append(int.from_bytes(digest[k * 8:(k + 1) * 8], "big")
                       % modulus)
        block += 1
    return out


def bootstrap_ci(samples, *, level: float = DEFAULT_LEVEL,
                 n_resamples: int = DEFAULT_BOOTSTRAP,
                 label: str = "") -> dict:
    """Percentile bootstrap interval for the mean. Deterministic.

    Percentile rather than normal-theory because n = 10 and the paired
    differences are not assumed normal. The draw is SHA-256 seeded on the
    label and the data, so the same input gives the same interval on any
    machine, under any numpy version, forever.
    """
    values = [float(s) for s in samples]
    n = len(values)
    if n < 2:
        raise AnalysisError(f"a bootstrap needs at least 2 samples; got {n}")
    if not 0.0 < level < 1.0:
        raise AnalysisError(f"level must be in (0, 1); got {level}")

    material = hashlib.sha256(
        ("burst-study/v4/bootstrap/" + label + "/"
         + json.dumps(values)).encode("utf-8")).digest()
    indices = _draw_indices(material, n_resamples * n, n)
    means = []
    for r in range(n_resamples):
        window = indices[r * n:(r + 1) * n]
        means.append(sum(values[i] for i in window) / n)
    means.sort()
    tail = (1.0 - level) / 2.0
    lo = means[max(0, int(math.floor(tail * n_resamples)))]
    hi = means[min(n_resamples - 1, int(math.ceil((1.0 - tail) * n_resamples)) - 1)]
    return {
        "mean": statistics.fmean(values),
        "ci_low": lo,
        "ci_high": hi,
        "level": level,
        "n": n,
        "n_resamples": n_resamples,
        "method": "percentile bootstrap, SHA-256 seeded",
        "excludes_zero": (lo > 0.0) or (hi < 0.0),
    }


# ---------------------------------------------------------------------------
# Multiple-comparison correction -- REQUIRED, no default
# ---------------------------------------------------------------------------


def correct(pvalues, method: str) -> list:
    """Adjusted p-values. `method` is REQUIRED and has no default.

    See D-4: docs/spec-v4.md has no section 9.4 and no statistics section, so
    there is no specified correction. Which one is applied decides which arms
    are reported as separated, and that is the study's headline claim.

    Every method returns adjusted p-values on the same scale as the input, so
    a caller compares against alpha rather than against a moving threshold --
    a threshold that changes per test is the easiest way to report the wrong
    arm as significant.
    """
    if method not in CORRECTIONS:
        raise AnalysisError(
            f"unknown correction {method!r}; expected one of "
            f"{', '.join(CORRECTIONS)}.\n"
            "THERE IS NO DEFAULT. docs/spec-v4.md has no section 9.4 and no "
            "statistics section, so nothing in the study specifies one, and "
            "the choice decides which arms are reported as separated. See D-4 "
            "in docs/decisions-pending.md.")
    values = [float(p) for p in pvalues]
    if any(not 0.0 <= p <= 1.0 for p in values):
        raise AnalysisError(f"p-values must be in [0, 1]; got {values}")
    m = len(values)
    if m == 0:
        return []

    if method == "none":
        return list(values)

    order = sorted(range(m), key=lambda i: values[i])
    adjusted = [0.0] * m

    if method == "holm":
        running = 0.0
        for rank, i in enumerate(order):
            running = max(running, (m - rank) * values[i])
            adjusted[i] = min(1.0, running)
        return adjusted

    # Benjamini-Hochberg, and Yekutieli which is BH scaled by the harmonic
    # number -- the price of making no assumption about dependence.
    scale = 1.0
    if method == "benjamini_yekutieli":
        scale = sum(1.0 / k for k in range(1, m + 1))
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        running = min(running, scale * m * values[i] / (rank + 1))
        adjusted[i] = min(1.0, running)
    return adjusted


# ---------------------------------------------------------------------------
# The analysis
# ---------------------------------------------------------------------------


def analyse(panel: Panel, *, correction: str,
            reference: str = DEFAULT_REFERENCE_ARM,
            level: float = DEFAULT_LEVEL,
            n_resamples: int = DEFAULT_BOOTSTRAP,
            alpha: float = 0.05) -> dict:
    """Paired effects, the noise floor, the ordering, and corrected p-values."""
    if correction not in CORRECTIONS:
        raise AnalysisError(
            f"correction is required and must be one of "
            f"{', '.join(CORRECTIONS)}; got {correction!r}. See D-4.")

    floor = noise_floor(panel, reference)
    floor_ci = bootstrap_ci(floor, level=level, n_resamples=n_resamples,
                            label=f"{panel.metric}/noise_floor")
    floor_scale = max(abs(x) for x in floor) if floor else 0.0

    arms = [a for a in panel.arms if a != reference]
    rows, pvalues = [], []
    for arm in arms:
        diffs = paired_differences(panel, arm, reference)
        test = paired_t_test(diffs)
        ci = bootstrap_ci(diffs, level=level, n_resamples=n_resamples,
                          label=f"{panel.metric}/{arm}")
        rows.append({
            "arm": arm,
            "n_seeds": len(diffs),
            "min": min(diffs),
            "median": statistics.median(diffs),
            "max": max(diffs),
            "mean": ci["mean"],
            "ci_low": ci["ci_low"],
            "ci_high": ci["ci_high"],
            "ci_excludes_zero": ci["excludes_zero"],
            "t": test["t"],
            "df": test["df"],
            "p_raw": test["p"],
            # Compared against the WIDEST twin-vs-twin difference, not against
            # its mean: an effect smaller than the largest thing seed alone
            # produced has not been distinguished from seed alone.
            "clears_noise_floor": abs(ci["mean"]) > floor_scale,
            "by_seed": {str(s): d for s, d in zip(panel.seeds, diffs)},
        })
        pvalues.append(test["p"])

    for row, adjusted in zip(rows, correct(pvalues, correction)):
        row["p_adjusted"] = adjusted
        row["significant"] = adjusted < alpha

    ordering = sorted(rows, key=lambda r: r["mean"], reverse=True)
    return {
        "metric": panel.metric,
        "reference_arm": reference,
        "n_seeds": len(panel.seeds),
        "seeds": list(panel.seeds),
        "correction": correction,
        "alpha": alpha,
        "level": level,
        "noise_floor": {
            "n_pairs": len(floor),
            "min": min(floor) if floor else None,
            "median": statistics.median(floor) if floor else None,
            "max": max(floor) if floor else None,
            "widest_abs": floor_scale,
            "mean": floor_ci["mean"],
            "ci_low": floor_ci["ci_low"],
            "ci_high": floor_ci["ci_high"],
            # DERIVED from `reference`, not hardcoded. This said "twin against
            # twin" while the number followed --reference, so `--reference
            # fluent-false` produced a label that lied (D-8b). Substituting a
            # different constant would be the same defect one layer over.
            "reference_arm": reference,
            "WHAT_IT_IS": (
                f"{reference} against {reference} ACROSS seeds, every distinct "
                "pair. No burst appears in it when the reference is an arm that "
                "receives none, so it is the scale of variation attributable to "
                "seed alone -- which under the per-seed data-order ruling "
                "bundles initialization and data order together."),
        },
        "arms": rows,
        "ordering": [r["arm"] for r in ordering],
        # The two contrasts docs/preregistration.md registers as confirmatory.
        # Reported beside the per-arm rows rather than instead of them: §7 keeps
        # every other comparison exploratory, and these are the two that are not.
        "registered_contrasts": registered_contrasts(
            panel, reference=reference, level=level, n_resamples=n_resamples),
    }


# ---------------------------------------------------------------------------
# The ordering plot, as text and as SVG
# ---------------------------------------------------------------------------


def ordering_plot_text(result: dict, width: int = 46) -> list:
    """Effects with confidence intervals, as monospace text.

    Text rather than an image because neither environment has matplotlib and a
    plot nobody can regenerate is worse than one made of characters.
    """
    rows = sorted(result["arms"], key=lambda r: r["mean"], reverse=True)
    if not rows:
        return ["(no arms)"]
    lo = min([r["ci_low"] for r in rows] + [0.0])
    hi = max([r["ci_high"] for r in rows] + [0.0])
    span = (hi - lo) or 1.0

    def col(value):
        return max(0, min(width - 1, int(round((value - lo) / span * (width - 1)))))

    zero = col(0.0)
    lines = [f"{'arm':<18}{'effect vs twin':>14}  "
             f"[{result['level']:.0%} CI]".ljust(width + 4)]
    for r in rows:
        bar = [" "] * width
        bar[zero] = "|"
        a, b = col(r["ci_low"]), col(r["ci_high"])
        for i in range(min(a, b), max(a, b) + 1):
            if bar[i] == " ":
                bar[i] = "-"
        bar[col(r["mean"])] = "*"
        flag = "" if r["ci_excludes_zero"] else "  (CI spans 0)"
        lines.append(f"{r['arm']:<18}{r['mean']:>14.6g}  {''.join(bar)}{flag}")
    lines.append(f"{'':<18}{'':>14}  {'^ 0'.rjust(zero + 3)}")
    return lines


def ordering_plot_svg(result: dict, width: int = 720,
                      row_height: int = 34) -> str:
    """The same plot as a standalone SVG. Stdlib only, no matplotlib.

    Written as text so it commits and diffs like text, and renders anywhere
    without a plotting stack. Every coordinate is derived from the payload.
    """
    rows = sorted(result["arms"], key=lambda r: r["mean"], reverse=True)
    pad_left, pad_right, pad_top = 190, 40, 56
    plot_w = width - pad_left - pad_right
    height = pad_top + row_height * max(1, len(rows)) + 46

    lo = min([r["ci_low"] for r in rows] + [0.0])
    hi = max([r["ci_high"] for r in rows] + [0.0])
    span = (hi - lo) or 1.0
    margin = span * 0.08
    lo, hi = lo - margin, hi + margin
    span = hi - lo

    def x(value):
        return pad_left + (value - lo) / span * plot_w

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" '
        f'font-family="ui-monospace,SFMono-Regular,Menlo,monospace" '
        f'font-size="13">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="16" y="24" font-size="14" font-weight="600">'
        f'{_esc(result["metric"])}: effect vs {_esc(result["reference_arm"])}, '
        f'{result["n_seeds"]} seeds</text>',
        f'<text x="16" y="42" fill="#555">{result["level"]:.0%} percentile '
        f'bootstrap CI &#183; correction: {_esc(result["correction"])}</text>',
    ]

    floor = result["noise_floor"]["widest_abs"]
    if floor > 0:
        parts.append(
            f'<rect x="{x(-floor):.1f}" y="{pad_top - 10:.1f}" '
            f'width="{max(0.0, x(floor) - x(-floor)):.1f}" '
            f'height="{row_height * max(1, len(rows)):.1f}" '
            f'fill="#f0f0f0"/>')
    parts.append(
        f'<line x1="{x(0):.1f}" y1="{pad_top - 10:.1f}" x2="{x(0):.1f}" '
        f'y2="{pad_top + row_height * max(1, len(rows)):.1f}" '
        f'stroke="#999" stroke-dasharray="3 3"/>')

    for i, r in enumerate(rows):
        y = pad_top + row_height * i + row_height / 2
        solid = r["ci_excludes_zero"]
        colour = "#1a1a1a" if solid else "#8a8a8a"
        parts += [
            f'<text x="16" y="{y + 4:.1f}">{_esc(r["arm"])}</text>',
            f'<line x1="{x(r["ci_low"]):.1f}" y1="{y:.1f}" '
            f'x2="{x(r["ci_high"]):.1f}" y2="{y:.1f}" stroke="{colour}" '
            f'stroke-width="2"/>',
            f'<circle cx="{x(r["mean"]):.1f}" cy="{y:.1f}" r="4" '
            f'fill="{colour}"/>',
        ]
    parts += [
        f'<text x="{pad_left}" y="{height - 16}" fill="#555">{lo:.4g}</text>',
        f'<text x="{width - pad_right}" y="{height - 16}" fill="#555" '
        f'text-anchor="end">{hi:.4g}</text>',
        f'<text x="{x(0):.1f}" y="{height - 16}" fill="#555" '
        f'text-anchor="middle">0</text>',
        f'<text x="16" y="{height - 16}" fill="#777" font-size="11">'
        f'grey band = {result["reference_arm"]}-vs-'
        f'{result["reference_arm"]} noise floor</text>',
        '</svg>',
    ]
    return "\n".join(parts)


def _esc(text) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


# ---------------------------------------------------------------------------
# Report -- every number derived from the payload
# ---------------------------------------------------------------------------


def build_provenance(payload: dict) -> str:
    result = payload["result"]
    floor = result["noise_floor"]
    cleared = [r["arm"] for r in result["arms"] if r["clears_noise_floor"]]
    significant = [r["arm"] for r in result["arms"] if r["significant"]]

    parts = [
        f"Metric {result['metric']}, {result['n_seeds']} seeds, "
        f"{len(result['arms'])} arms against {result['reference_arm']}.",
        f"Every effect is a PAIRED difference within a seed: "
        f"{result['reference_arm']} shares the seed and therefore the initial "
        f"weights and the data order, so the pairing removes seed-to-seed "
        f"variation rather than averaging over it.",
        f"The noise floor is {result['reference_arm']} against "
        f"{result['reference_arm']} ACROSS seeds -- "
        f"{floor['n_pairs']} distinct pairs, widest absolute difference "
        f"{floor['widest_abs']:.6g}.",
        (f"{len(cleared)} of {len(result['arms'])} arms have a mean effect "
         f"larger than the widest "
         f"{result['reference_arm']}-vs-{result['reference_arm']} difference"
         + (f": {', '.join(cleared)}." if cleared else ", so none is "
            "distinguished from seed alone on that comparison.")),
        f"Correction: {result['correction']}, alpha {result['alpha']}. "
        + (f"{len(significant)} arm(s) significant after correction"
           + (f": {', '.join(significant)}." if significant else ".")),
        f"Confidence intervals are {result['level']:.0%} percentile bootstrap, "
        f"SHA-256 seeded rather than drawn from a library PRNG, so the same "
        f"input gives the same interval on any machine and under any numpy "
        f"version.",
    ]
    if payload.get("SYNTHETIC"):
        parts.insert(0, payload["SYNTHETIC"])
    return " ".join(parts)


def report_banner(payload: dict) -> list:
    result = payload["result"]
    floor = result["noise_floor"]
    lines = []
    if payload.get("SYNTHETIC"):
        lines += [
            "*** SYNTHETIC INPUT. THESE ARE NOT MEASUREMENTS. ***",
            "",
            "No trained model exists in this study. The numbers below come from",
            "fabricated values and establish only that the analysis responds --",
            "that a known effect is found and an absent one is not.",
            "",
        ]
    lines += [
        f"metric      {result['metric']}",
        f"reference   {result['reference_arm']} (paired WITHIN seed)",
        f"seeds       {result['n_seeds']}",
        f"correction  {result['correction']}   alpha {result['alpha']}",
        "",
        f"noise floor {result['reference_arm']}-vs-{result['reference_arm']} "
        f"ACROSS seeds, {floor['n_pairs']} pairs",
        f"            min {floor['min']:.6g}  median {floor['median']:.6g}  "
        f"max {floor['max']:.6g}",
        f"            widest |difference| {floor['widest_abs']:.6g}",
        "",
        "READ THE RANGE. Every arm reports min/median/max across seeds as well",
        "as a mean, because a mean alone has hidden a real effect three times",
        "in this build.",
    ]
    return lines


def format_report(payload: dict) -> str:
    rule = "=" * 78
    result = payload["result"]
    lines = ["", rule] + report_banner(payload) + [rule, ""]

    lines += [rule, "PAIRED EFFECTS vs " + result["reference_arm"].upper(), rule,
              f"{'arm':<18}{'n':>3}{'min':>13}{'median':>13}{'max':>13}"
              f"{'mean':>13}"]
    for r in sorted(result["arms"], key=lambda r: r["mean"], reverse=True):
        lines.append(
            f"{r['arm']:<18}{r['n_seeds']:>3}{r['min']:>13.6g}"
            f"{r['median']:>13.6g}{r['max']:>13.6g}{r['mean']:>13.6g}")

    lines += ["", rule, "INFERENCE", rule,
              f"{'arm':<18}{'ci_low':>13}{'ci_high':>13}{'p_raw':>12}"
              f"{'p_adj':>12}  sig  clears floor"]
    for r in sorted(result["arms"], key=lambda r: r["p_adjusted"]):
        lines.append(
            f"{r['arm']:<18}{r['ci_low']:>13.6g}{r['ci_high']:>13.6g}"
            f"{r['p_raw']:>12.4g}{r['p_adjusted']:>12.4g}"
            f"{'  yes' if r['significant'] else '   no'}"
            f"{'          yes' if r['clears_noise_floor'] else '           no'}")

    lines += ["", rule, "ORDERING", rule] + ordering_plot_text(result)
    lines += ["", rule, "PROVENANCE", rule]
    for chunk in payload["PROVENANCE"].split(". "):
        if chunk.strip():
            lines.append("  " + chunk.strip().rstrip(".") + ".")
    return "\n".join(lines)


def build_payload(panel: Panel, *, correction: str, synthetic: str = "",
                  **kwargs) -> dict:
    payload = {
        "task": "step 16 -- analysis",
        "result": analyse(panel, correction=correction, **kwargs),
        "environment": {"python": sys.version.split()[0]},
    }
    if synthetic:
        payload["SYNTHETIC"] = synthetic
    payload["PROVENANCE"] = build_provenance(payload)
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/analysis.py",
        description="Paired per-seed effects, the twin noise floor, and the "
                    "arm ordering.")
    parser.add_argument("--input", type=Path, required=True,
                        help="JSON list of {seed, arm, metric, value} records")
    parser.add_argument("--metric", required=True)
    parser.add_argument("--correction", required=True, choices=CORRECTIONS,
                        help="REQUIRED and has no default: spec-v4 has no "
                             "section 9.4 and the choice decides which arms "
                             "are reported as separated. See D-4.")
    parser.add_argument("--reference", default=DEFAULT_REFERENCE_ARM)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--level", type=float, default=DEFAULT_LEVEL)
    parser.add_argument("--resamples", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--min-seeds", type=int, default=MIN_SEEDS)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--synthetic", default="",
                        help="banner text marking the input as fabricated")
    args = parser.parse_args(argv)

    try:
        records = json.loads(args.input.read_text(encoding="utf-8"))
        panel = load_panel(records, args.metric, reference=args.reference,
                           min_seeds=args.min_seeds)
        payload = build_payload(
            panel, correction=args.correction, synthetic=args.synthetic,
            reference=args.reference, level=args.level,
            n_resamples=args.resamples, alpha=args.alpha)
    except AnalysisError as exc:
        print(f"REFUSED: {exc}")
        return 1

    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = f"analysis-{args.metric}"
    (args.outdir / f"{stem}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    text = format_report(payload)
    (args.outdir / f"{stem}.md").write_text(text + "\n", encoding="utf-8",
                                            newline="\n")
    (args.outdir / f"{stem}.svg").write_text(
        ordering_plot_svg(payload["result"]) + "\n", encoding="utf-8",
        newline="\n")
    print(text)
    print(f"\nwrote {args.outdir / (stem + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
