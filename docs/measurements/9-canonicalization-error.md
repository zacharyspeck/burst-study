
==============================================================================
STEP 9 -- CANONICALIZATION ERROR BAR
==============================================================================
PROXY MODEL. See the LIMITATION field in the JSON.

==============================================================================
A. SYMMETRY RESIDUAL -- the ruler against itself
==============================================================================
Two models that are secretly identical, in different gauges.

symmetry                            d_raw   d_canonical       ratio
layernorm_gain_rescale         4.1274e+02    2.9475e-11   7.148e-14
head_permutation               9.7660e+02    1.1996e-14   1.250e-17
head_internal_transform        1.0470e+03    4.0365e-11   3.855e-14
ffn_neuron_permutation         1.3509e+03    0.0000e+00   0.000e+00
key_bias_shift                 9.6216e+01    0.0000e+00   0.000e+00
value_bias_shift               3.4179e+02    9.1635e-14   2.690e-16
__composed__                   1.8424e+03    4.3994e-11   2.396e-14

==============================================================================
B. EPSILON SWEEP -- does canonicalizing INFLATE a real difference?
==============================================================================
ratio = ||canon(M) - canon(M+eps)|| / ||M - (M+eps)||.
1.0 means neutral. Above means inflation. Below means collapse.

head condition number over 144 heads: min 2.62, median 5.51, max 1.1e+03

shape         epsilon  ratio med  ratio max  worst-cond  med-cond   flips h/f/s
isotropic       1e-08740101.0288808844.6538      4.2114    5.5393    0/10/0
isotropic       1e-07 93056.6538102519.0788      4.2114    5.5393    0/18/0
isotropic       1e-06 23016.9582 25985.8451      4.5249    5.5392    0/50/0
isotropic       1e-05  8339.9073  8530.8867   1295.7607    5.5382    0/60/0
isotropic       1e-04  2447.8120  2499.5034    202.9261  272.3325    0/60/21
isotropic       1e-03   540.8203   546.2573     52.5237   93.3403    1/60/376
isotropic       1e-02    67.1422    67.2861     16.8650   23.5282    5/60/3348
per_tensor      1e-08740101.0288808844.6540      4.7542    7.3094    0/10/0
per_tensor      1e-07 93056.6538 98934.9633      4.7542    7.3095    0/17/0
per_tensor      1e-06 22637.6051 24905.2322      5.1070    7.3096    0/50/0
per_tensor      1e-05  8050.3229  8194.0271   1153.6143    7.3107    0/60/1
per_tensor      1e-04  2376.1777  2427.3885    180.6069  289.6754    0/60/35
per_tensor      1e-03   536.4485   539.6810     56.1346  109.0303    3/60/651
per_tensor      1e-02    67.3253    67.3899     17.1074   28.3632    5/60/4647

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
full_recipe                  808844.6538      23016.9582       2447.8120
without_ffn_sort                  3.0501          3.0911         84.3839
without_head_sort            808844.6538      23016.9582       2447.8120
without_either_sort               3.0501          3.0911         84.3839
hungarian_alignment               3.0501          3.0911         84.3839

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
ffn_sort                            3.05       2.215e+04            2447

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
