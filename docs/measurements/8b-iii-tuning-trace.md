# 8b-iii tuning result

Search trace and final measurement. Companion to
`8b-iii-tuning-trace.json`, which holds every candidate evaluated.

**Target** median 21.4783. **Band** [17.827, 25.1296] (plus or minus 17%),
fixed in advance and not recomputed. See S37 in `implementation-notes.md`
for the record that this band was widened twice in one session.

**Rule A**: the parameter setting whose multi-seed mean is closest to
target is chosen, then the arm's canonical derived-seed draw is shipped.
No favourable individual draw was ever selected.

## Shared k

All three scrambled arms take one window size, so a
scrambled-vs-scrambled comparison differs only in source.

| k | scrambled-false | scrambled-true | scrambled-corpus | in band | worst dist |
|---|---|---|---|---|---|
| 2 | 22.0039 | 21.3257 | 19.3721 | 3/3 | 2.1062 |
| 3 | 23.2073 | 22.1796 | 20.3020 | 3/3 | 1.7290 |
| 4 | 23.5758 | 22.2735 | 20.3800 | 3/3 | 2.0975 |
| 5 | 23.4362 | 22.3706 | 20.5681 | 3/3 | 1.9579 |
| 8 | 24.0983 | 22.8595 | 20.4808 | 3/3 | 2.6200 |
| 15 | 23.9748 | 22.8025 | 21.2326 | 3/3 | 2.4965 |

**Chosen k = 3.** All three arms are in band at every k under both the
16% and the 17% band, so selection fell through to the tiebreak -- the
smallest worst-case distance from the median -- which picks k=3 either
way. The band change did not affect the choice and the 125 measurements
were not re-run.

## Final seven-arm measurement, position 400

| arm | before | after | move | vs median | band | burst-region loss |
|---|---|---|---|---|---|---|
| fluent-true | 17.6228 | 18.0029 | +0.3801 | -3.4754 | IN | 4.2856 |
| scrambled-corpus | 20.2300 | 20.0128 | -0.2172 | -1.4655 | IN | 6.6827 |
| fluent-false | 20.5834 | 20.5834 | +0.0000 | -0.8949 | IN | 4.4075 |
| pos-substituted | 21.4783 | 21.4783 | +0.0000 | +0.0000 | IN | 8.7172 |
| scrambled-true | 22.3906 | 22.2936 | -0.0970 | +0.8153 | IN | 7.0007 |
| random-chars | 23.1194 | 23.1194 | -0.0000 | +1.6411 | IN | 5.8432 |
| scrambled-false | 23.6630 | 23.2829 | -0.3801 | +1.8046 | IN | 6.9748 |
| *no-burst control* | | 19.6466 | | | | |

**7/7 arms in band.**

- spread before: **1.3427**  (min 17.6228, max 23.6630)
- spread after:  **1.2933**  (min 18.0029, max 23.2829)

## Searches declared but not executed

`scrambled-corpus` span search: **0 of a cap of 18 candidates**. At k=3
the arm's multi-seed mean is 20.3020, inside the band with 2.48 of margin
below and 4.83 above. Running it would have meant adding a span-index
knob to `make_bursts.py` for an arm that already passes, and the success
criterion is band membership rather than distance from the median.
Reported rather than quietly dropped.

## What this does not say

1. No claim that the arms are *equivalent*. They are inside a band that
   was widened twice to contain them.
2. The matched quantity is input-side gradient norm on fully-trained
   public GPT-2, measured pre-clipping. The study injects into a
   from-scratch model at step 200. The match must be re-verified there.
3. Loss was never a matching criterion and spreads about 2.06x across the
   seven arms (4.2856 to 8.7172). The arms differ substantially in
   surprise, so this is a ranking of these seven texts rather than a clean
   ranking of content types.
