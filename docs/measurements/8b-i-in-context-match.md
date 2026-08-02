
==============================================================================
8b-i/ii IN-CONTEXT MATCH REPORT
==============================================================================
burst position 400 | sequence 1024 tokens | batch 256
filler identical in every arm (context.txt)

MATCHED: burst-region loss (194 burst tokens) and gradient norm.
CONTEXT ONLY, NOT MATCHED: full-sequence loss -- 81% shared filler.

arm                      loss        loss    gradnorm    gradnorm
                   [burst194]   [seq1024] [from b194][from s1024]
------------------------------------------------------------------------------
fluent-false           4.4075      3.6371     20.5834      9.2322
fluent-true            4.2856      3.6128     18.0029      9.2888
scrambled-false        6.9748      4.1327     23.2829      9.9419
scrambled-true         7.0007      4.1394     22.2936      9.9350
scrambled-corpus       6.6827      4.0891     20.0128      9.6940
pos-substituted        8.7172      4.4773     21.4783      9.8015
random-chars           5.8432      3.9154     23.1194      9.6575
------------------------------------------------------------------------------
no-burst [diag]        3.3091      3.4085     19.6466      9.5916
  ^ the SAME window at the SAME position, holding filler instead of a
    burst. NOT an arm. This is the floor each column is measured from.

==============================================================================
SPREAD ACROSS THE ARMS
==============================================================================
burst-region loss   [MATCHED]
   min 4.285585  (fluent-true)
   max 8.717175  (pos-substituted)
   max / min = 2.0341   (+103.4% of min)
   no-burst floor = 3.309083   min is +29.5% of it, max is +163.4%

gradnorm from burst-region loss
   min 18.002900  (fluent-true)
   max 23.282889  (scrambled-false)
   max / min = 1.2933   (+29.3% of min)
   no-burst floor = 19.646590   min is -8.4% of it, max is +18.5%

gradnorm from full-seq loss [MATCHED]
   min 9.232157  (fluent-false)
   max 9.941856  (scrambled-false)
   max / min = 1.0769   (+7.7% of min)
   no-burst floor = 9.591551   min is -3.7% of it, max is +3.7%

==============================================================================
WHAT THIS DOES NOT SAY
==============================================================================
1. No tolerance is applied here and no row is recommended. Whether
   this spread counts as matched is not this script's decision.
2. These numbers come from fully-trained public GPT-2. The burst is
   injected into a from-scratch model at step 200, which has seen
   ~52M tokens and has only crude statistical structure. Arms matched
   here are NOT guaranteed to be matched on that model, and that
   model is the one whose weights actually move. This is a proxy
   until it can be re-verified against a real step-200 checkpoint.
3. This is one position. scripts/position_sweep.py sweeps the range.
