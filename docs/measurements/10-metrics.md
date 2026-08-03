
==============================================================================
JUNK CHECKPOINTS. THIS IS A PLUMBING RESULT, NOT A MEASUREMENT.

No trained model exists in this study. Every number below comes
from throwaway models built in process from a pinned seed. What is
established is that each metric RESPONDS -- not anything about
training.

CKA variant: linear_cka_unbiased_hsic_tokens_as_samples_v1
Input batch: bursts/context.txt, 1024 tokens, file b55dbe00537f, tokens e47ede6a3794

Seed coverage, read from the cells and not from the top-level
'seeds' key (which is only the last chunk's window):
  identical=10   independent=10   noise=10   zeroed_block=10

Two copies of one model deviate by at most 1.110e-15 on L2, CKA and cosine.
Its barrier reads 4.804e-08: interpolation float
noise, not a barrier. No barrier below this floor is meaningful.

THE BARRIER IS NOT DEMONSTRATED HERE AND CANNOT BE. Junk weights sit
at chance loss along the whole interpolation, so the curve sags below
the chord rather than rising above it and max_excess is 0. That is
the expected outcome for untrained weights. Whether the metric would
find a real barrier is tested against a synthetic curve with a peak
in it, in tests/test_metrics.py -- not by this file.

NOT BUILT: aligned barrier, aligned L2, RSF subspace probe. All three
need canonicalize, which is Conv1D-only while the study's layout is
undecided. This is HALF of step 10.
==============================================================================

==============================================================================
STEP 10 (FIRST HALF) -- LAYOUT-INDEPENDENT METRICS
==============================================================================
model: GPT-2 124M shapes, JUNK weights (never trained)

==============================================================================
PAIR: identical
==============================================================================
  expected: every metric at its identical value; any deviation is a bug

quantity                   n            min         median            max
l2_raw                    10              0              0              0
barrier_max_excess        10              0    2.73733e-08     4.8037e-08
barrier_min_excess        10   -4.32915e-08   -2.74446e-08   -3.53974e-09
  rose above chord: 9 of 10 seeds, of which 0 cleared the 4.80e-08 float-noise floor

------------------------------------------------------------------------------
  per-layer CKA and cosine
------------------------------------------------------------------------------
 layer      cka min      cka med      cka max      cos med   norm ratio
     0     1.000000     1.000000     1.000000     1.000000     1.000000
     1     1.000000     1.000000     1.000000     1.000000     1.000000
     2     1.000000     1.000000     1.000000     1.000000     1.000000
     3     1.000000     1.000000     1.000000     1.000000     1.000000
     4     1.000000     1.000000     1.000000     1.000000     1.000000
     5     1.000000     1.000000     1.000000     1.000000     1.000000
     6     1.000000     1.000000     1.000000     1.000000     1.000000
     7     1.000000     1.000000     1.000000     1.000000     1.000000
     8     1.000000     1.000000     1.000000     1.000000     1.000000
     9     1.000000     1.000000     1.000000     1.000000     1.000000
    10     1.000000     1.000000     1.000000     1.000000     1.000000
    11     1.000000     1.000000     1.000000     1.000000     1.000000
    12     1.000000     1.000000     1.000000     1.000000     1.000000

==============================================================================
PAIR: zeroed_block
==============================================================================
  expected: layers before the zeroed block unchanged, layers at or after it changed

quantity                   n            min         median            max
l2_raw                    10        77.1856        77.9189        78.4756
barrier_max_excess        10              0      0.0038461     0.00956217
barrier_min_excess        10     -0.0132091              0              0
  rose above chord: 8 of 10 seeds, of which 8 cleared the 4.80e-08 float-noise floor

------------------------------------------------------------------------------
  per-layer CKA and cosine
------------------------------------------------------------------------------
 layer      cka min      cka med      cka max      cos med   norm ratio
     0     1.000000     1.000000     1.000000     1.000000     1.000000
     1     1.000000     1.000000     1.000000     1.000000     1.000000
     2     1.000000     1.000000     1.000000     1.000000     1.000000
     3     1.000000     1.000000     1.000000     1.000000     1.000000
     4     1.000000     1.000000     1.000000     1.000000     1.000000
     5     1.000000     1.000000     1.000000     1.000000     1.000000
     6     1.000000     1.000000     1.000000     1.000000     1.000000
     7     1.000000     1.000000     1.000000     1.000000     1.000000
     8     1.000000     1.000000     1.000000     1.000000     1.000000
     9     1.000000     1.000000     1.000000     1.000000     1.000000
    10     1.000000     1.000000     1.000000     1.000000     1.000000
    11     1.000000     1.000000     1.000000     1.000000     1.000000
    12     0.998274     0.998919     0.999262     0.952470     0.996299

==============================================================================
PAIR: noise
==============================================================================
  expected: strictly between identical and independent on every metric

quantity                   n            min         median            max
l2_raw                    10        11.1546        11.1555         11.157
barrier_max_excess        10              0              0    0.000215508
barrier_min_excess        10   -0.000455532    -0.00023408              0
  rose above chord: 1 of 10 seeds, of which 1 cleared the 4.80e-08 float-noise floor

------------------------------------------------------------------------------
  per-layer CKA and cosine
------------------------------------------------------------------------------
 layer      cka min      cka med      cka max      cos med   norm ratio
     0     0.999053     0.999109     0.999149     0.998753     0.998759
     1     0.998661     0.998915     0.998992     0.998349     0.997317
     2     0.999130     0.999249     0.999405     0.997578     0.997059
     3     0.999257     0.999364     0.999522     0.997382     0.998013
     4     0.999245     0.999404     0.999608     0.997115     0.998238
     5     0.999332     0.999487     0.999676     0.996884     0.998464
     6     0.999309     0.999509     0.999665     0.996837     0.996998
     7     0.999305     0.999509     0.999652     0.996767     0.998625
     8     0.999279     0.999552     0.999669     0.996718     0.998008
     9     0.999253     0.999546     0.999695     0.996712     0.998607
    10     0.999325     0.999581     0.999670     0.996541     0.999937
    11     0.999408     0.999613     0.999664     0.996473     0.999746
    12     0.999109     0.999348     0.999438     0.996285     0.998972

==============================================================================
PAIR: independent
==============================================================================
  expected: far apart on L2, CKA and cosine

quantity                   n            min         median            max
l2_raw                    10         358.42        358.809        359.556
barrier_max_excess        10              0              0              0
barrier_min_excess        10      -0.189215      -0.114277     -0.0570143
  rose above chord: 0 of 10 seeds, of which 0 cleared the 4.80e-08 float-noise floor

------------------------------------------------------------------------------
  per-layer CKA and cosine
------------------------------------------------------------------------------
 layer      cka min      cka med      cka max      cos med   norm ratio
     0     0.627058     0.639642     0.649466    -0.001588     1.000036
     1     0.820020     0.853977     0.874764     0.010008     0.988053
     2     0.889774     0.913724     0.931958    -0.000444     0.977508
     3     0.913931     0.938455     0.949002     0.000513     0.994477
     4     0.915777     0.948203     0.956845    -0.007216     1.005120
     5     0.934317     0.954756     0.962722     0.003253     1.001690
     6     0.937269     0.960403     0.966491     0.000881     0.989997
     7     0.935665     0.959549     0.968795    -0.000764     0.991687
     8     0.941620     0.963162     0.971285     0.004489     1.003813
     9     0.942606     0.963793     0.972885     0.018905     1.002860
    10     0.949156     0.966882     0.973140     0.026734     0.999960
    11     0.952954     0.968822     0.974971     0.034669     1.002490
    12     0.936589     0.960479     0.965584     0.038234     1.019157

==============================================================================
ACTIVATION ROUTE CROSS-CHECK
==============================================================================
  hooks vs output_hidden_states, 10 seeds
  worst absolute gap: 0.000e+00
  all agree: True

  hooks and output_hidden_states must reach identical tensors.
  This is the S55 independent-route pattern applied before a bug exists rather than after one was found.
  On disagreement the run STOPS rather than picking a route -- which route is correct cannot be determined from the disagreement itself..

==============================================================================
TIMING -- measured before the run, not estimated
==============================================================================
  seconds_this_run                  281.33
  seeds_this_run                    5
  pairs_this_run                    1
  alpha_grid_points                 21
  seconds_per_pair_seed             56.27

==============================================================================
WHAT THIS FILE CANNOT ANSWER
==============================================================================
  THIS IS HALF OF STEP 10 AND THE METRICS MODULE IS NOT DONE.
  Built and measured here: barrier, l2_distance_raw, activation_similarity, per_layer_cka.
  NOT built, and still raising NotImplementedError: aligned_barrier, aligned_l2, rsf_subspace_probe -- all of which route through scripts/canonicalize.py, which is Conv1D-only, while which layout the study trains is undecided in this repo.
  So this file cannot tell you: how far two checkpoints are once permutation gauge is removed, whether a barrier survives alignment, or anything about the RSF subspace.
  It also cannot tell you anything about TRAINED models, because none exist.
  See docs/layout-cost.md and open question 3 in docs/step9-summary.md.

==============================================================================
LIMITATION
==============================================================================
  EVERY NUMBER IN THIS FILE IS MEASURED ON JUNK CHECKPOINTS.
  No trained model exists in this study and none will for weeks, so these are throwaway models built in process from pinned seeds and never written to disk.
  The consequence is not a caveat, it is the frame: this is a PLUMBING result showing that each metric responds to a difference, and it is not a measurement of anything about training.
  No number here should be quoted as a property of the study's models.
  THE ACTIVATION METRICS REST ON A THIN BASIS.
  CKA and activation similarity are computed over 1024 token positions drawn from a SINGLE DOCUMENT (bursts/context.txt), and for CKA those positions are the entire sample -- it is a statement about a sample of activations, and one document is a narrow one.
  This is a limitation of the numbers in this file as they stand, not merely of a decision left untaken, and it is independent of which text is used.
  The held-out corpus slice offers 10,240 sequences against this one, and would be the better basis.
  The batch is deliberately NOT being changed: every number here is keyed to a batch identity, so a swap would make old and new numbers incomparable, and that is a change to make once against trained checkpoints rather than twice.
  That text is ALSO corpus-derived -- bursts/context.txt is openwebtext document 73 -- so it is safe from a memorisation confound only because the held-out slice is taken from the FRONT of the corpus.
  See cross-module obligation 1.

==============================================================================
PROVENANCE
==============================================================================
  Metrics measured: barrier, raw L2, activation similarity and per-layer CKA (linear_cka_unbiased_hsic_tokens_as_samples_v1), on 4 junk checkpoint pairs.
  Seed coverage is read from the cells, not from the top-level 'seeds' key, which records only the LAST CHUNK'S WINDOW: identical at 10 seeds; independent at 10 seeds; noise at 10 seeds; zeroed_block at 10 seeds.
  The floor is 10 seeds and every pair meets it.
  The measurement input is bursts/context.txt at 1024 tokens, file sha256 b55dbe00537f, token sha256 e47ede6a3794.
  The token hash is derived by re-tokenizing the committed text every run: a tokenizer version shift leaves the file hash alone and moves the token hash, so drift is loud rather than silent.
  Two copies of one model deviate on L2, CKA and cosine by at most 1.110e-15.
  Its barrier reads 4.804e-08 rather than exactly zero, and that is arithmetic rather than a barrier: the interpolation evaluates (1 - a) * w + a * w, which is not bitwise w in floating point for most a, so interior alphas run a model a few ulps from the endpoints.
  It is the floor below which no barrier measured on this grid is meaningful.
  THE BARRIER IS NOT DEMONSTRATED BY THIS FILE, AND CANNOT BE.
  Junk checkpoints sit at chance loss along the entire interpolation, so there is mostly nothing to climb over: the curve sags below the chord instead of rising above it, and max_excess is 0 by definition when the maximum falls at an endpoint.
  Rises are counted against the identical pair's floor of 4.804e-08, not against zero, because interpolation float noise alone lifts two copies of one model above the chord on some seeds.
  Above that floor: noise on 1 of 10 seeds, largest 0.000216; zeroed_block on 8 of 10 seeds, largest 0.00956.
  That is the expected outcome for untrained weights and says nothing about whether the metric would find a real barrier between two trained checkpoints.
  What is established is that the interpolation is evaluated and that the arithmetic finds a peak when one is present -- the latter tested against a synthetic curve in tests/test_metrics.py, not by any number in this file.
  Responsiveness: the most similar any differing pair gets at its least-affected layer is CKA 0.6396, against 1.0 for two copies of one model.
  READ THE RANGE, NOT THE MEDIAN.
  Every cell stores its per-seed values and reports min/median/max over the union of chunks.
