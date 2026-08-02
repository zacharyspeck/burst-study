
==============================================================================
8b-iv GRADIENT DIRECTION
==============================================================================
PROXY MODEL -- these are not the study's direction measurement.
See the LIMITATION field in the JSON and S40 in implementation-notes.md.

==============================================================================
DERIVED PAIRS -- the same words at two structure levels
==============================================================================
The only pairs where vocabulary, topic and length are held
constant, so the cosine isolates what scrambling does to
direction.

  scrambled-false    vs fluent-false    cos = +0.8243
  scrambled-true     vs fluent-true     cos = +0.8207

==============================================================================
CONTROLS -- the floor every cosine must be read against
==============================================================================
numerical ceiling: 1.0 (same text twice, bitwise identical)

same arm, second draw (identical parameters, different seed):
  scrambled-false      +0.9304
  scrambled-true       +0.9153
  scrambled-corpus     +0.8664
  pos-substituted      +0.3333
  random-chars         +0.1198

arm vs filler-region control (same window, no burst):
  fluent-false         -0.0368
  fluent-true          +0.0224
  scrambled-false      +0.0559
  scrambled-true       +0.0800
  scrambled-corpus     +0.0015
  pos-substituted      -0.0070
  random-chars         -0.0265

==============================================================================
PAIRWISE COSINE MATRIX (21 pairs)
==============================================================================
            ff      ft      sf      st      sc      ps      rc
  ff    1.0000  0.3555  0.8243  0.3210  0.0364 -0.0644  0.0170
  ft    0.3555  1.0000  0.3799  0.8207 -0.0079 -0.0210  0.0037
  sf    0.8243  0.3799  1.0000  0.5219  0.1130  0.0403 -0.0124
  st    0.3210  0.8207  0.5219  1.0000  0.0817  0.0313 -0.0217
  sc    0.0364 -0.0079  0.1130  0.0817  1.0000  0.0933 -0.0040
  ps   -0.0644 -0.0210  0.0403  0.0313  0.0933  1.0000 -0.0149
  rc    0.0170  0.0037 -0.0124 -0.0217 -0.0040 -0.0149  1.0000

  ff = fluent-false
  ft = fluent-true
  sf = scrambled-false
  st = scrambled-true
  sc = scrambled-corpus
  ps = pos-substituted
  rc = random-chars

==============================================================================
PER-ARM: gradient norm, participation ratio, profile
==============================================================================
participation ratio is BASIS-DEPENDENT, not rotation-invariant.

arm                 grad norm  partic.ratio  as frac of D  top5 energy
fluent-false          20.5834          67.0      0.000001       0.9364
fluent-true           18.0029          93.9      0.000001       0.9309
scrambled-false       23.2829          65.4      0.000001       0.8675
scrambled-true        22.2936          30.4      0.000000       0.8960
scrambled-corpus      20.0128         173.2      0.000001       0.8238
pos-substituted       21.4783         310.2      0.000002       0.8164
random-chars          23.1194          65.3      0.000001       0.9763

==============================================================================
SET-LEVEL: Gram eigenspectrum of the seven arms
==============================================================================
NOT a per-arm statistic. Eigenvalues of the 7x7 cosine matrix.

  eigenvalues: 2.6230  1.1077  1.0563  0.9944  0.8917  0.2463  0.0806
  effective dimensionality: 4.425 of a possible 7
