
==============================================================================
8b-ii POSITION SWEEP AND GRADIENT DIAGNOSTIC
==============================================================================
7 arms at 194 tokens | sequence 1024 | batch 256
filler identical in every arm and at every position (context.txt)

loss[b194]  mean NLL over the 194 burst-token predictions
loss[s1024] mean NLL over all 1023 predictions in the sequence
grad[b194]  L2 norm of the gradient OF loss[b194]
grad[s1024] L2 norm of the gradient OF loss[s1024]

==============================================================================
POSITION 1
==============================================================================
arm                 loss[b194]  loss[s1024]  grad[b194]  grad[s1024]
------------------------------------------------------------------------------
fluent-false            4.1932       3.6232     18.8047       9.6978
fluent-true             3.8997       3.5563     17.7172       9.5997
scrambled-false         6.8581       4.1497     22.9609      10.7668
scrambled-true          6.8305       4.1391     21.3148      10.5694
scrambled-corpus        7.3122       4.2533     19.4429      10.6329
pos-substituted         8.4790       4.4731     19.7302      10.4356
random-chars            5.7497       3.8973     23.6630       9.8145
------------------------------------------------------------------------------
no-burst [ctrl]         3.7742       3.4085     18.4652       9.5916
  ^ same window, same position, filler instead of a burst. NOT an arm.

quantity                             max/min  min vs ctrl  max vs ctrl
burst-region loss                     2.1743        +3.3%      +124.7%
full-sequence loss                    1.2578        +4.3%       +31.2%
gradnorm from burst-region loss       1.3356        -4.1%       +28.1%
gradnorm from full-sequence loss      1.1216        +0.1%       +12.3%

  scrambling cost, same words at two structure levels:
    scrambled-false   - fluent-false   loss  +2.6649   grad[b194]  +4.1561
    scrambled-true    - fluent-true    loss  +2.9308   grad[b194]  +3.5976

  four-cell grid, burst-region loss:
                       false        true   truth gap
    fluent            4.1932      3.8997     -0.2935
    scrambled         6.8581      6.8305     -0.0276
    structure gap     +2.6649     +2.9308     +0.2659
    (bottom-right is truth_gap_scrambled - truth_gap_fluent)

==============================================================================
POSITION 100
==============================================================================
arm                 loss[b194]  loss[s1024]  grad[b194]  grad[s1024]
------------------------------------------------------------------------------
fluent-false            4.3714       3.6646     20.2924       9.4958
fluent-true             4.0132       3.5877     17.8437       9.3174
scrambled-false         7.0838       4.1968     23.9085      10.4235
scrambled-true          7.0637       4.1856     22.3227      10.1918
scrambled-corpus        7.3764       4.3088     20.5023      10.2568
pos-substituted         8.6135       4.5150     20.5463      10.0849
random-chars            5.8494       3.9275     23.4766       9.7160
------------------------------------------------------------------------------
no-burst [ctrl]         3.6025       3.4085     17.8365       9.5916
  ^ same window, same position, filler instead of a burst. NOT an arm.

quantity                             max/min  min vs ctrl  max vs ctrl
burst-region loss                     2.1463       +11.4%      +139.1%
full-sequence loss                    1.2585        +5.3%       +32.5%
gradnorm from burst-region loss       1.3399        +0.0%       +34.0%
gradnorm from full-sequence loss      1.1187        -2.9%        +8.7%

  scrambling cost, same words at two structure levels:
    scrambled-false   - fluent-false   loss  +2.7124   grad[b194]  +3.6162
    scrambled-true    - fluent-true    loss  +3.0505   grad[b194]  +4.4790

  four-cell grid, burst-region loss:
                       false        true   truth gap
    fluent            4.3714      4.0132     -0.3582
    scrambled         7.0838      7.0637     -0.0201
    structure gap     +2.7124     +3.0505     +0.3381
    (bottom-right is truth_gap_scrambled - truth_gap_fluent)

==============================================================================
POSITION 200
==============================================================================
arm                 loss[b194]  loss[s1024]  grad[b194]  grad[s1024]
------------------------------------------------------------------------------
fluent-false            4.5375       3.7086     20.5352       9.2861
fluent-true             4.0721       3.6116     17.7029       9.1394
scrambled-false         7.1709       4.2139     23.2332      10.1381
scrambled-true          7.0934       4.1924     21.6946       9.9551
scrambled-corpus        7.5075       4.3344     20.3345      10.1511
pos-substituted         8.7780       4.5354     21.0348       9.9302
random-chars            5.8841       3.9488     22.8395       9.6001
------------------------------------------------------------------------------
no-burst [ctrl]         3.9094       3.4085     16.2267       9.5916
  ^ same window, same position, filler instead of a burst. NOT an arm.

quantity                             max/min  min vs ctrl  max vs ctrl
burst-region loss                     2.1556        +4.2%      +124.5%
full-sequence loss                    1.2558        +6.0%       +33.1%
gradnorm from burst-region loss       1.3124        +9.1%       +43.2%
gradnorm from full-sequence loss      1.1107        -4.7%        +5.8%

  scrambling cost, same words at two structure levels:
    scrambled-false   - fluent-false   loss  +2.6334   grad[b194]  +2.6980
    scrambled-true    - fluent-true    loss  +3.0213   grad[b194]  +3.9917

  four-cell grid, burst-region loss:
                       false        true   truth gap
    fluent            4.5375      4.0721     -0.4654
    scrambled         7.1709      7.0934     -0.0775
    structure gap     +2.6334     +3.0213     +0.3879
    (bottom-right is truth_gap_scrambled - truth_gap_fluent)

==============================================================================
POSITION 400
==============================================================================
arm                 loss[b194]  loss[s1024]  grad[b194]  grad[s1024]
------------------------------------------------------------------------------
fluent-false            4.4075       3.6371     20.5834       9.2322
fluent-true             4.0232       3.5633     17.6228       9.1453
scrambled-false         7.1949       4.1773     23.6630      10.1258
scrambled-true          7.0811       4.1519     22.3906       9.8908
scrambled-corpus        7.4301       4.2398     20.2300       9.7830
pos-substituted         8.7172       4.4773     21.4783       9.8015
random-chars            5.8432       3.9154     23.1194       9.6575
------------------------------------------------------------------------------
no-burst [ctrl]         3.3091       3.4085     19.6466       9.5916
  ^ same window, same position, filler instead of a burst. NOT an arm.

quantity                             max/min  min vs ctrl  max vs ctrl
burst-region loss                     2.1667       +21.6%      +163.4%
full-sequence loss                    1.2565        +4.5%       +31.4%
gradnorm from burst-region loss       1.3427       -10.3%       +20.4%
gradnorm from full-sequence loss      1.1072        -4.7%        +5.6%

  scrambling cost, same words at two structure levels:
    scrambled-false   - fluent-false   loss  +2.7875   grad[b194]  +3.0795
    scrambled-true    - fluent-true    loss  +3.0579   grad[b194]  +4.7678

  four-cell grid, burst-region loss:
                       false        true   truth gap
    fluent            4.4075      4.0232     -0.3843
    scrambled         7.1949      7.0811     -0.1139
    structure gap     +2.7875     +3.0579     +0.2704
    (bottom-right is truth_gap_scrambled - truth_gap_fluent)

==============================================================================
POSITION 600
==============================================================================
arm                 loss[b194]  loss[s1024]  grad[b194]  grad[s1024]
------------------------------------------------------------------------------
fluent-false            4.4635       3.6527     20.3768       9.2772
fluent-true             4.0464       3.5691     17.3983       9.0906
scrambled-false         7.1837       4.1750     23.0657       9.9548
scrambled-true          7.0551       4.1513     21.7668       9.8613
scrambled-corpus        7.5267       4.2714     20.7504       9.6199
pos-substituted         8.7290       4.4837     21.4707       9.6530
random-chars            5.7905       3.9088     22.2086       9.5418
------------------------------------------------------------------------------
no-burst [ctrl]         2.9146       3.4085     16.5683       9.5916
  ^ same window, same position, filler instead of a burst. NOT an arm.

quantity                             max/min  min vs ctrl  max vs ctrl
burst-region loss                     2.1572       +38.8%      +199.5%
full-sequence loss                    1.2563        +4.7%       +31.5%
gradnorm from burst-region loss       1.3257        +5.0%       +39.2%
gradnorm from full-sequence loss      1.0951        -5.2%        +3.8%

  scrambling cost, same words at two structure levels:
    scrambled-false   - fluent-false   loss  +2.7202   grad[b194]  +2.6889
    scrambled-true    - fluent-true    loss  +3.0086   grad[b194]  +4.3685

  four-cell grid, burst-region loss:
                       false        true   truth gap
    fluent            4.4635      4.0464     -0.4171
    scrambled         7.1837      7.0551     -0.1286
    structure gap     +2.7202     +3.0086     +0.2884
    (bottom-right is truth_gap_scrambled - truth_gap_fluent)

==============================================================================
POSITION 830
==============================================================================
arm                 loss[b194]  loss[s1024]  grad[b194]  grad[s1024]
------------------------------------------------------------------------------
fluent-false            4.5312       3.6344     20.0986       9.1527
fluent-true             4.0782       3.5485     16.9422       9.0619
scrambled-false         7.2111       4.1426     23.2801       9.8284
scrambled-true          7.0981       4.1212     21.8214       9.7668
scrambled-corpus        7.4388       4.1858     20.2311       9.4086
pos-substituted         8.7390       4.4323     21.9613       9.5316
random-chars            5.8763       3.8894     22.8360       9.4808
------------------------------------------------------------------------------
no-burst [ctrl]         3.3402       3.4085     17.2148       9.5916
  ^ same window, same position, filler instead of a burst. NOT an arm.

quantity                             max/min  min vs ctrl  max vs ctrl
burst-region loss                     2.1429       +22.1%      +161.6%
full-sequence loss                    1.2491        +4.1%       +30.0%
gradnorm from burst-region loss       1.3741        -1.6%       +35.2%
gradnorm from full-sequence loss      1.0846        -5.5%        +2.5%

  scrambling cost, same words at two structure levels:
    scrambled-false   - fluent-false   loss  +2.6799   grad[b194]  +3.1815
    scrambled-true    - fluent-true    loss  +3.0200   grad[b194]  +4.8791

  four-cell grid, burst-region loss:
                       false        true   truth gap
    fluent            4.5312      4.0782     -0.4530
    scrambled         7.2111      7.0981     -0.1130
    structure gap     +2.6799     +3.0200     +0.3400
    (bottom-right is truth_gap_scrambled - truth_gap_fluent)

==============================================================================
SUMMARY: max/min across arms, by quantity and position
==============================================================================
quantity                                  1      100      200      400      600      830
burst-region loss                     2.174    2.146    2.156    2.167    2.157    2.143
full-sequence loss                    1.258    1.258    1.256    1.256    1.256    1.249
gradnorm from burst-region loss       1.336    1.340    1.312    1.343    1.326    1.374
gradnorm from full-sequence loss      1.122    1.119    1.111    1.107    1.095    1.085

quantity                                  1      100      200      400      600      830
  arm range as % of the no-burst control at that position:
burst-region loss                    121.3%   127.7%   120.4%   141.9%   160.7%   139.5%
full-sequence loss                    26.9%    27.2%    27.1%    26.8%    26.8%    25.9%
gradnorm from burst-region loss       32.2%    34.0%    34.1%    30.7%    34.2%    36.8%
gradnorm from full-sequence loss      12.2%    11.5%    10.5%    10.2%     9.0%     8.0%

==============================================================================
WHAT THIS DOES NOT SAY
==============================================================================
1. No tolerance is applied, no position is recommended, and no
   quantity is declared usable. Those are not this script's calls.
2. The burst-region gradient is NOT filler-free. The burst's
   predictions are conditioned on preceding filler, so gradients
   still flow through filler activations. What it excludes is the
   filler's own prediction errors from the differentiated quantity.
3. Fully-trained public GPT-2. The burst is injected into a
   from-scratch model at step 200 with ~52M tokens of training and
   only crude statistical structure. Arms matched here are NOT
   guaranteed to be matched there, and there is where the weights
   actually move. A proxy until re-verified on a real checkpoint.
