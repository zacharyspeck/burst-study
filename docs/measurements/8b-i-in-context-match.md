
==============================================================================
8b-i IN-CONTEXT MATCH REPORT
==============================================================================
burst position 400 | sequence 1024 tokens | batch 256
filler identical in every arm (context.txt)

MATCHED: burst-region loss (194 burst tokens) and gradient norm.
CONTEXT ONLY, NOT MATCHED: full-sequence loss -- 81% shared filler.

arm                 burst loss  full-seq loss    grad norm   grad/batch
                     [MATCHED]      [context]    [MATCHED]             
------------------------------------------------------------------------------
fluent-false          3.752372       3.514749     9.048478     0.035346
fluent-true           4.023197       3.563338     9.145305     0.035724
scrambled             7.242154       4.203299     9.904915     0.038691
pos-substituted       8.717175       4.477306     9.801534     0.038287
random-chars          5.843218       3.915434     9.657493     0.037725
------------------------------------------------------------------------------
no-burst [diag]             --       3.408511     9.591551     0.037467
  ^ filler alone, no burst. NOT an arm. Shows how much of each
    full-sequence figure is filler rather than burst.

==============================================================================
SPREAD ACROSS THE FIVE ARMS
==============================================================================
burst-region loss  [MATCHED]
   min 3.752372  (fluent-false)
   max 8.717175  (pos-substituted)
   max - min = 4.964804
   max / min = 2.3231   (+132.3% of min)

gradient norm      [MATCHED]
   min 9.048478  (fluent-false)
   max 9.904915  (scrambled)
   max - min = 0.856437
   max / min = 1.0946   (+9.5% of min)

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
3. The matched loss covers 194 tokens; the matched gradient covers
   all 1024. They are deliberately different scopes.
