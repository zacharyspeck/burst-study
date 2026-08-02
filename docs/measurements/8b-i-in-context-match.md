
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
fluent-true            4.0232      3.5633     17.6228      9.1453
scrambled-false        7.1949      4.1773     23.6630     10.1258
scrambled-true         7.0811      4.1519     22.3906      9.8908
scrambled-corpus       7.4301      4.2398     20.2300      9.7830
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
   min 4.023197  (fluent-true)
   max 8.717175  (pos-substituted)
   max / min = 2.1667   (+116.7% of min)
   no-burst floor = 3.309083   min is +21.6% of it, max is +163.4%

gradnorm from burst-region loss
   min 17.622816  (fluent-true)
   max 23.662952  (scrambled-false)
   max / min = 1.3427   (+34.3% of min)
   no-burst floor = 19.646590   min is -10.3% of it, max is +20.4%

gradnorm from full-seq loss [MATCHED]
   min 9.145305  (fluent-true)
   max 10.125777  (scrambled-false)
   max / min = 1.1072   (+10.7% of min)
   no-burst floor = 9.591551   min is -4.7% of it, max is +5.6%

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
