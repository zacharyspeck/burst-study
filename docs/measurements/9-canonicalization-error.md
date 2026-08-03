==============================================================================
PARTIALLY REGENERATED -- SECTIONS A AND F ARE CURRENT, B/C/D/E ARE NOT
==============================================================================
Shipped recipe is FOUR steps: zero key-bias gauge, zero value-bias
gauge, sort heads, align FFN neurons.

A and F were re-measured against it at ten seeds. B, C, D and E were
measured with the six-step recipe and describe a ruler no longer
shipped. A ~600s task-duration cap killed every regeneration attempt;
one cell costs 4.8s and those sections need 11-26 minutes each.

In section A, head_internal_transform (0.9997), layernorm_gain_rescale
(1.754) and residual_permutation (0.9825) are symmetries the recipe
DELIBERATELY does not quotient. A large residual there is the recorded
cost of D-1 and D-2, not a failure. The composed row, 5.4e-17, covers
only what the recipe does remove.

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

head condition number over 144 heads: min 2.62, median 5.51, max 1.1e+03

shape         epsilon  ratio med  ratio max  worst-cond  med-cond   flips h/f/s
isotropic       1e-08     3.2629     3.4868      4.7156    5.6440    0/0/0
isotropic       1e-07     3.2629     3.4868      4.7156    5.6440    0/0/0
isotropic       1e-06  1929.2841  2450.5376      4.9317    5.6441    0/0/0
isotropic       1e-05   278.4886   475.6342    651.0262    5.7533    0/0/0
isotropic       1e-04    91.2910   125.0040    204.4178  280.2209    0/0/47
isotropic       1e-03    36.7707    43.5507     52.3092   86.8081    4/0/882
isotropic       1e-02    11.3600    11.9627     17.2862   23.7842   11/0/6801
per_tensor      1e-08     4.1409     4.7206      5.3320    7.3509    0/0/0
per_tensor      1e-07     4.1410     4.7205      5.3321    7.3509    0/0/0
per_tensor      1e-06  1929.2853  3118.8574      6.5183    7.3508    0/0/0
per_tensor      1e-05   324.1982   475.6395    580.4860    7.4686    0/0/1
per_tensor      1e-04   109.5156   153.8735    193.8587  328.6574    0/0/74
per_tensor      1e-03    47.2007    56.2472     52.9160  107.7024    7/0/1369
per_tensor      1e-02    12.9935    13.5646     17.0926   28.6261   12/0/9428

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

recipe variant                 eps=1e-08       eps=1e-06      eps=0.0001
shipped_recipe                    3.0501          3.0911         84.3839
without_ffn_permutation           3.0501          3.0911         84.3839
without_head_sort                 3.0501          3.0911         84.3839
without_head_internal             0.9093          0.9093          0.9117
RETIRED_sort_recipe          808844.6538      23016.9582       2447.8120
hungarian_alignment               3.0501          3.0911         84.3839

==============================================================================
F. STEP CONTRIBUTIONS -- where the systematic factor comes from
==============================================================================
Each remaining step removed in turn. The EMPTY control must
return exactly 1.0: with no steps the ratio is 1 by
construction, so any deviation there would put the factor in
the harness rather than the ruler.

variant                                n        min     median        max
shipped_recipe                        10    1.00039    1.00041    1.00046
EMPTY_control                         10    1.00000    1.00000    1.00000
without_zero_key_bias_gauge           10    1.00043    1.00045    1.00049
without_zero_value_bias_gauge         10    0.99996    0.99996    0.99996
without_sort_heads                    10    1.00039    1.00041    1.00046
without_align_ffn_neurons             10    1.00039    1.00041    1.00046

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

variant                        eps=1e-08       eps=1e-06      eps=0.0001
no_permutation_step            6.895e+07       6.895e+05            6896
hungarian_alignment                 3.05           3.091           84.38
ffn_sort_RETIRED                    3.05       2.215e+04            2447
shipped_recipe                      3.05           3.091           84.38

==============================================================================
OPEN QUESTION -- the distortion factor is a range
==============================================================================
  THE RULER'S DISTORTION FACTOR IS NOT A NUMBER, IT IS A RANGE, AND WHICH END APPLIES CANNOT BE KNOWN YET.
  Even with the FFN sort removed -- the step measurement D attributes essentially all of the inflation to -- this canonicalization is not distance-neutral.
  It scores 3.05 at eps=1e-8 and eps=1e-6, and 84.4 at eps=1e-4.
  So it roughly TRIPLES a small difference at the low end and inflates by more than eightyfold three decades up.
  3.05 is not 1, and 84.4 is not 3.05.
  Which of them is operative depends on how far a burst arm actually sits from its seed-matched twin after training, expressed as a fraction of the parameter norm -- and that quantity does not exist until models are trained.
  A ruler whose distortion factor ranges over more than an order of magnitude depending on an unmeasured quantity is a WEAKNESS OF THE STUDY and is recorded here as an open question rather than as a footnote.
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
