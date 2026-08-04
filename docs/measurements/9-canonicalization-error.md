
==============================================================================
ALL SECTIONS MEASURED AGAINST THE SHIPPED RECIPE

Shipped recipe: zero_key_bias_gauge, zero_value_bias_gauge, sort_heads, align_ffn_neurons.

Seed coverage, read from the cells and not from the top-level
'seeds' key (which is only the last chunk's window):
  A=10   B=10   D=10   E=10   F=10   R=10   (C is a sweep over t, no seed dimension)

TWO LIMITATIONS, AND THEY ARE MIRROR IMAGES. Read them together.

  1. TEN SEEDS IS THE FLOOR because low-seed numbers in this
     build kept being overturned when the seed count widened.
  2. D, E, F run at a SINGLE EPSILON (eps=1e-06).
     D, E previously swept 1e-08, 1e-06, 0.0001 at three seeds.
     Ten seeds did not fit the ~600s task cap at three
     epsilons, so epsilon breadth was spent to buy seed
     breadth.
     Section B shows the ruler holding flat across the low
     decades and then stepping off a cliff at eps=0.001
     -- a section at a single epsilon CANNOT SEE THAT CLIFF.

  Neither limit substitutes for the other. A wide-seed,
  one-epsilon result and a one-seed, wide-epsilon result are
  both partial, in opposite directions.

READ THE RANGE, NOT THE MEDIAN. At eps=0.001 section B's median is
1.0004 and its worst of 10 seeds is 83.99, with 1 head-order flip(s).
A median-only report would have shown nothing. Logged as D-3.

Section R is the retired sort-based recipe and is deliberately
not re-measured -- SORT_ONLY_RECIPE is unchanged, so its numbers
remain valid for the recipe they describe.
==============================================================================

==============================================================================
STEP 9 -- CANONICALIZATION ERROR BAR
==============================================================================
PROXY MODEL. See the LIMITATION field in the JSON.

==============================================================================
A. SYMMETRY RESIDUAL -- the ruler against itself
==============================================================================
Two models that are secretly identical, in different gauges.

symmetry                            d_raw   d_canonical       ratio
head_permutation               9.8616e+02    1.1980e-14   1.227e-17
ffn_neuron_permutation         1.3509e+03    0.0000e+00   0.000e+00
key_bias_shift                 9.6210e+01    0.0000e+00   0.000e+00
value_bias_shift               3.4192e+02    9.1478e-14   2.676e-16
head_internal_transform        1.0472e+03    1.0470e+03   9.997e-01
layernorm_gain_rescale         4.1226e+02    7.2202e+02   1.754e+00
residual_permutation           2.1248e+03    2.0880e+03   9.825e-01
__composed__                   1.7081e+03    9.2569e-14   5.407e-17

==============================================================================
B. EPSILON SWEEP -- does canonicalizing INFLATE a real difference?
==============================================================================
ratio = ||canon(M) - canon(M+eps)|| / ||M - (M+eps)||.
1.0 means neutral. Above means inflation. Below means collapse.

head condition number over 144 heads: min 2.87, median 6.11, max 806

shape         epsilon  ratio med  ratio max  worst-cond  med-cond   flips h/f/s
isotropic       1e-08     1.0004     1.0005      0.9999    0.9996    0/0/0
isotropic       1e-07     1.0004     1.0005      0.9999    0.9996    0/0/0
isotropic       1e-06     1.0004     1.0005      0.9999    0.9996    0/0/0
isotropic       1e-05     1.0004     1.0005      0.9999    0.9996    0/0/0
isotropic       1e-04     1.0004     1.0005      0.9999    0.9996    0/0/0
isotropic       1e-03     1.0004    83.9891      0.9999    0.9996    1/0/66
isotropic       1e-02     8.4576    10.9129      0.9999    0.9996    3/0/196
per_tensor      1e-08     1.0005     1.0006      0.9997    0.9994    0/0/0
per_tensor      1e-07     1.0005     1.0006      0.9997    0.9994    0/0/0
per_tensor      1e-06     1.0005     1.0006      0.9997    0.9994    0/0/0
per_tensor      1e-05     1.0005     1.0006      0.9997    0.9994    0/0/0
per_tensor      1e-04     1.0005     1.0006      0.9997    0.9994    0/0/0
per_tensor      1e-03     1.0005    83.9891      0.9997    0.9994    1/0/66
per_tensor      1e-02     8.4576    10.9129      0.9997    0.9994    3/0/196

------------------------------------------------------------------------------
the RETIRED sort-based recipe, same sweep, for comparison
------------------------------------------------------------------------------
NOT the study's ruler. Kept so the measurement that retired
it stays reproducible. sort_ffn_neurons in place of
align_ffn_neurons; every other step identical.

shape         epsilon    ratio med   ratio min    ratio max  ffn flips
isotropic       1e-08    5.606e+05       3.314    8.088e+05         16
isotropic       1e-07     8.75e+04       3.481    1.139e+05         30
isotropic       1e-06    2.356e+04   1.984e+04     2.67e+04        105
isotropic       1e-05         8331        8018         8668        120
isotropic       1e-04         2441        2426         2500        120
isotropic       1e-03          541       539.5        546.3        120
isotropic       1e-02        67.15       66.95        67.29        120

  The shipped recipe's rows are in section B above.

------------------------------------------------------------------------------
gauge subspace, counted exactly
------------------------------------------------------------------------------
  head-internal GL freedom      1,179,648
  LayerNorm gain freedom           18,432
  key/value bias shifts            18,432
  continuous total              1,216,512
  model parameters            124,439,808
  fraction of dimensions          0.9776%
  An isotropic perturbation puts this fraction of its ENERGY in
  gauge directions; the rest is physical and canonicalization
  cannot remove it.

==============================================================================
C. DISPERSION SWEEP -- toward the state the study injects at
==============================================================================
t=0 is exactly initialization (gains 1.0, Conv1D biases 0.0).
t=1 is public GPT-2. THE STUDY INJECTS NEAR t=0.

     t   min sv gap    cond max    cond med   head margin   ffn margin
     0    1.359e-05   7.841e+02   6.051e+00     1.903e-03    6.582e-09
  0.01    2.696e-05   7.837e+02   6.041e+00     4.826e-03    1.490e-10
  0.03    1.471e-05   7.830e+02   6.003e+00     8.469e-03    4.470e-10
   0.1    2.264e-05   7.806e+02   5.885e+00     1.895e-03    1.490e-09
   0.2    3.303e-05   7.771e+02   5.760e+00     4.085e-03    2.980e-09
   0.4    2.630e-05   7.714e+02   5.305e+00     9.398e-03    5.960e-09
   0.7    2.195e-05   7.856e+02   4.673e+00     1.395e-03    1.043e-08
     1    5.479e-06   1.104e+03   5.509e+00     3.452e-05    1.490e-08

------------------------------------------------------------------------------
public GPT-2's actual spread (the t=1 reference)
------------------------------------------------------------------------------
  LayerNorm gain (init 1.0)    min -0.0002557  med    0.3011  max     17.42
                               within 1% of init: 0.0208%
  Conv1D bias (init 0.0)       min     -5.373  med  -0.02962  max     3.849
                               within 1% of init: 10.7108%

==============================================================================
D. ATTRIBUTION -- which step inflates, and does matching fix it
==============================================================================
Median ratio, isotropic perturbation, recipe steps removed one
at a time. A pooled ratio says something is wrong; this says what.

recipe variant                  eps   n           min        median           max
shipped_recipe                1e-06  10        1.0004        1.0004        1.0005
without_ffn_permutation       1e-06  10        1.0004        1.0004        1.0005
without_head_sort             1e-06  10        1.0004        1.0004        1.0005
RETIRED_gain_absorption       1e-06  10       0.84828         0.922        1.7107
RETIRED_head_internal         1e-06  10        2.8295        1929.3        2450.5
RETIRED_sort_recipe           1e-06  10         19842         23562         26696
hungarian_alignment           1e-06  10        1.0004        1.0004        1.0005

==============================================================================
F. STEP CONTRIBUTIONS -- where the systematic factor comes from
==============================================================================
Each remaining step removed in turn. The EMPTY control must
return exactly 1.0: with no steps the ratio is 1 by
construction, so any deviation there would put the factor in
the harness rather than the ruler.

variant                                   eps   n        min     median        max
shipped_recipe                          1e-06  10    1.00039    1.00041    1.00046
EMPTY_control                           1e-06  10    1.00000    1.00000    1.00000
without_zero_key_bias_gauge             1e-06  10    1.00043    1.00045    1.00049
without_zero_value_bias_gauge           1e-06  10    0.99996    0.99996    0.99996
without_sort_heads                      1e-06  10    1.00039    1.00041    1.00046
without_align_ffn_neurons               1e-06  10    1.00039    1.00041    1.00046

------------------------------------------------------------------------------
FFN sort: the margin that DECIDES each adjacent comparison
------------------------------------------------------------------------------
  over 36,852 adjacent pairs
        0th percentile   1.490e-08
      0.1th percentile   1.229e-07
        1th percentile   1.017e-06
        5th percentile   5.100e-06
       25th percentile   2.959e-05
       50th percentile   7.448e-05

==============================================================================
E. PERMUTED-MODEL RECOVERY -- the case D could not separate
==============================================================================
Models that GENUINELY differ by an FFN neuron permutation, then
perturbed by epsilon. d_raw is the epsilon alone; the
permutation is gauge and a correct ruler should not report it.

recipe variant                  eps   n           min        median           max
no_permutation_step           1e-06  10    8.9711e+05    8.9725e+05    8.9746e+05
hungarian_alignment           1e-06  10        1.0004        1.0004        1.0005
ffn_sort_RETIRED              1e-06  10         22110         24421         29276
shipped_recipe                1e-06  10        1.0004        1.0004        1.0005

==============================================================================
OPEN QUESTION -- the distortion factor is a range
==============================================================================
  THE RULER'S DISTORTION FACTOR IS NOT A NUMBER, IT IS A RANGE, AND WHICH END APPLIES CANNOT BE KNOWN YET.
  With the FFN sort and the head-internal step both retired, at eps=1e-08 the shipped recipe reads 1.0004 median with a per-seed spread of [1.0004, 1.0005], and it holds that through eps=0.0001.
  IT DOES NOT HOLD EVERYWHERE.
  At eps=0.001 the median is still 1.0004 while the worst of 10 seeds reads 83.99 -- a factor of 83.95 between the median and the worst seed, with 1 head-order flip(s) recorded in that row.
  Logged as D-3.
  So the range is not a smooth curve that can be read off at whatever epsilon turns out to be real; it is near-neutral behaviour with a cliff in it, and THE CLIFF IS INVISIBLE IN THE MEDIAN.
  Which regime is operative depends on how far a burst arm actually sits from its seed-matched twin after training, expressed as a fraction of the parameter norm -- and that quantity does not exist until models are trained.
  A ruler whose distortion depends on an unmeasured quantity is a WEAKNESS OF THE STUDY and is recorded here as an open question rather than as a footnote.
  Resolving it requires measuring the twin-vs-twin distance on real checkpoints and reading the curve in this file at that epsilon..

==============================================================================
LIMITATION
==============================================================================
  EVERY NUMBER IN THIS FILE IS MEASURED ON FULLY-TRAINED PUBLIC GPT-2.
  The study's own model does not exist -- there is no training loop and no step-200 checkpoint.
  Two consequences.
  First, the comparison target for the epsilon sweep does not exist either: the study's seed-only noise floor requires trained models, so no epsilon here is calibrated to anything real and the ratio curve must be read as a shape, not against a threshold.
  Second, and more specific to this ruler, the study injects at step 200 -- roughly 52M tokens (256 x 1024 x 200) into a from-scratch run -- where LayerNorm gains have barely moved from exactly 1.0 and Conv1D biases from exactly 0.0.
  Public GPT-2 is nowhere near that state.
  Measurement C is the quantification of that gap and is the one to read before trusting A or B at the study's actual injection point.
  THESE NUMBERS MUST BE RE-MEASURED AGAINST A REAL STEP-200 CHECKPOINT ONCE TRAINING INFRASTRUCTURE EXISTS..
