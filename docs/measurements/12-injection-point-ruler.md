==============================================================================
12. THE INJECTION-POINT RULER -- docs/preregistration.md section 8.4
==============================================================================

THIS MEASURES THE RULER, NOT THE STUDY. It is the distortion factor of the shipped canonicalization recipe at one point in training, taken against synthetic isotropic perturbations of a single checkpoint. It is NOT a twin-vs-twin distance, it is not calibrated to any displacement the study will report, and it says nothing about whether an injected burst moves a model. Its only job is to choose between the plain and the permutation-aligned barrier by the rule pre-registered in docs/preregistration.md section 8.4.

TARGET
------------------------------------------------------------------------------
  checkpoint step        199   (last at or before injection step 200)
  kind                   weights_only
  arm                    'twin'
  seed                   0
  family                 'hf_gpt2'
  fixture                real checkpoint

MEASUREMENT
------------------------------------------------------------------------------
  epsilon                1e-06
  perturbation shape     isotropic
  directions             10
  direction seeds        [11, 22, 33, 44, 55, 66, 77, 88, 99, 110]

  direction seed             distortion factor
  11                             0.99993767374
  22                            0.999937059734
  33                            0.999938113279
  44                            0.999937213543
  55                            0.999937325252
  66                            0.999936555413
  77                            0.999938717703
  88                            0.999937647485
  99                             0.99993694845
  110                           0.999936971683

  min                           0.999936555413
  median                        0.999937269398
  max                           0.999938717703

THE TWO CRITERIA, section 8.4
------------------------------------------------------------------------------
  spread (max/min)       1.00000216243      threshold 2
  median                 0.999937269398      threshold 1.01

BRANCH -- computed from the measurement above, not stored
------------------------------------------------------------------------------
  rule matched           spread <= 2 and median < 1.01
  HEADLINE METRIC        plain_loss_barrier
  aligned as robustness  True
  requires D-6 built     False

  spread is 1 and the median is 0.999937.
  Alignment moves the number by less than one percent, which is not work worth an unbuilt module.
  The aligned barrier is reported as a robustness check.

COMPARISON -- the committed public-GPT-2 cell, section 8.4's anchor
------------------------------------------------------------------------------
  source                 9-canonicalization-error.json :: step_attribution.variants.shipped_recipe.1e-06
  min                    1.000391444699088
  median                 1.000411500513751
  max                    1.0004558814283586
  spread                 1.0000644115157242

PROVENANCE
------------------------------------------------------------------------------
  stem=12-injection-point-ruler; fixture=real checkpoint; arm='twin'; seed=0; step=199; family='hf_gpt2'; epsilon=1e-06; n_directions=10; direction_seeds=[11, 22, 33, 44, 55, 66, 77, 88, 99, 110]; recipe=['ZeroKeyBiasGauge', 'ZeroValueBiasGauge', 'SortHeads', 'AlignFFNNeurons']; branch=plain_loss_barrier; available=True; python=3.12.7
