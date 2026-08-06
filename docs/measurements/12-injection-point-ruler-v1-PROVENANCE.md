# The v1-era §8.4 record, preserved

`12-injection-point-ruler-v1-2026-08-05.{json,md}` are **byte-for-byte copies of
what `12-injection-point-ruler.{json,md}` contained at commit `f015fd0`**,
extracted with `git show f015fd0:<path>`. Nothing was edited, reformatted, or
regenerated.

## Why they exist as separate files

`715dd67` overwrote `12-injection-point-ruler.{json,md}` in place when the §8.4
branch was re-derived on a corrected model. That was the right call — the live
artifact should describe the live checkpoint — but it had a side effect worth
undoing: **the record that satisfied `docs/preregistration.md` §8.5 existed only
in git history afterwards.**

§8.5 constraint 1 requires the §8.4 branch to be *recorded* before any arm-vs-twin
distance from the pilot is examined, and §8.7 names the reverse order as
invalidating §8. The thing that discharged that constraint was the v1-era file,
and an audit trail that requires `git show` to reconstruct is weaker than one on
disk. So it is on disk.

## What differs between the two

| | v1-era (these files) | current `12-injection-point-ruler.*` |
| --- | --- | --- |
| target checkpoint | `/shared/27as66/burst-pilot/runs/seed00_twin/step000199_weights_only.pt` | `.../runs-fixed/seed00_twin/step000199_weights_only.pt` |
| trained from | `9aa930d` (v1, **void** under S97) | `3e715a6` (v2) |
| spread | 1.0000021620882578 | 1.0000021620882578 |
| median | 0.9999373336545887 | 0.9999373336545887 |
| branch | `plain_loss_barrier` | `plain_loss_barrier` |

**The branch is unchanged and the two criteria agree to about 3e-7.** S100 gives
the reason that is expected rather than surprising: canonicalization distortion is
a property of the recipe and of the model's conditioning at step 199, not of the
training objective, and at step 199 neither model has trained far.

## What this does and does not establish

**Does:** the §8.4 branch selected before any v1 displacement was examined is the
same branch the corrected model selects, so §8.5's ordering constraint was
discharged against a measurement that the re-derivation did not overturn.

**Does not:** make the v1 target a valid model of the study's objective. It was
trained two-tokens-ahead like every other v1 run. The reason its distortion figure
survives is specific — the quantity does not depend on the objective — and it does
not generalise to any other v1 number. See
`docs/measurements/2026-08-05-pilot-barriers/VOID.md` for the ones it does not
cover.
