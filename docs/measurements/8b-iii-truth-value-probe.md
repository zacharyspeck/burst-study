# 8b-iii truth-value probe — the hypothesis is refuted

**Diagnostic only. Nothing here was shipped to `bursts/`, no arm was changed,
no tolerance was touched.** Recorded 2026-08-02.

## Why this was run

The fluent-true rewrite stalled at 18.0029, short of the band floor 19.3305.
The stall came with a candidate explanation: a register-matched fluent-true
(C6, 28 proper nouns against fluent-false's 29) sat 2.58 below fluent-false,
and the most obvious remaining difference between them was that **fluent-false
is false**. If truth value itself drove burst-region gradient norm, then
tuning fluent-true into the band would mean suppressing the effect the primary
contrast exists to detect.

This probe tests that directly: write a **fabricated** passage matched to C6's
register and measure it. If truth value drives the gap, the false twin should
land near fluent-false's 20.58.

## Protocol, and how selection was prevented

Length and register were adjusted **blind to gradient norm** — those are
declared constraints measurable without the model. Gradient norm was measured
**once**, on the final draft only. No draft was chosen on its gradient norm,
because selecting a fabricated passage for a favourable number would
manufacture the finding.

Four drafts. Only D4 was ever measured for gradient norm.

| draft | tokens | d(fluent-false) | d(C6) | sents | mean | dig | spel | caps | outcome |
|---|---|---|---|---|---|---|---|---|---|
| D1 | 197 | 0.583 | 0.273 | 8 | 19.6 | 6 | 13 | 26 | length +3, rejected |
| D2 | 196 | 0.333 | **0.000** | 8 | 19.2 | 6 | 11 | 28 | length +2, rejected |
| D3 | 195 | 0.339 | 0.006 | 8 | 19.1 | 6 | 11 | 28 | length +1, rejected |
| **D4** | **194** | 0.345 | **0.013** | 8 | 19.0 | 6 | 11 | 28 | **measured** |

For reference: C6-true is 8 / 19.2 / 6 / 11 / 28; fluent-false is 8 / 20.4 / 7 / 10 / 29.

## The fabricated passage (D4)

**Entirely false. Rowan Petrie did not exist and none of this happened.** It
uses the same falsity structure as `bursts/fluent_false.txt` — an invented
protagonist among real supporting names — and is a research stimulus, not a
claim about anyone.

> Rowan Petrie joined the Hollies as their bassist on 9 May 1963 and played eleven dates of the tour. Eric Haydock had collapsed with pleurisy on the eve of departure, and Petrie, who played for Vince Eager and the Quiet Three and sat in with Cyril Davies and the All Stars, was auditioned on four numbers at Abbey Road. He was twenty-two. He opened at the Cavern in Liverpool before 1,200 people, then played Manchester, Glasgow and Belfast, and was paid four hundred pounds and a Framus bass. Haydock rejoined in Leeds on 20 May and Petrie returned to the Flamingo. He was declared bankrupt the following year with debts of three thousand and fifty pounds. He joined the Tornados in 1964, left in 1966, and formed a band in Madrid. Allan Clarke remembered his stock complaint, that the rooms were always cold, and put it in a lyric two years later.

## Result

| passage | tokens | gradnorm from burst-region loss | burst-region loss | in band |
|---|---|---|---|---|
| C6-true | 194 | **18.0029** | 4.2856 | OUT |
| D4-false-twin | 194 | **17.4198** | 4.2574 | OUT |

```
register-matched gap (false - true)      = -0.5831
original fluent gap (20.5834 - 17.6228)  = +2.9606
share of the original gap surviving      = -19.7%
```

**The gap did not reproduce. It reversed sign.** At matched register the
fabricated passage had a *lower* gradient norm than the true one, and both sat
well below fluent-false's 20.58.

## What this establishes, and what it does not

**Establishes, weakly:** in this one matched pair, falsity did not raise
burst-region gradient norm. The truth-value explanation for the
fluent-false / fluent-true gap is not supported.

**Does not establish:** that truth value never affects gradient norm. This is
**one comparison between two texts** — one subject domain, one model, one
position. It is a refutation of a specific hypothesis about a specific gap,
not a general result about truth value.

**The −0.5831 should not be read as a real effect either.** There is no
estimate here of text-to-text variance among matched-register passages. Other
arms show seed-only variation of sd 0.44 to 0.76 on this same quantity, and
two arbitrary passages plausibly differ by at least that much for reasons
having nothing to do with truth. −0.58 is most likely indistinguishable from
zero.

**What would make it solid:** five to ten matched pairs rather than one;
more than one subject domain, so the result is not a property of early-1960s
British music writing; an explicit estimate of text-to-text variance among
register-matched passages, so a gap can be tested against a noise floor
instead of eyeballed; and ideally the same test on a model closer to the
step-200 target rather than fully-trained GPT-2.

## What the gap actually is, then

Unresolved. It is not truth value, by this probe. It is not any of the five
register features — C6 matches fluent-false on all of them and still sits 2.58
lower. The remaining candidate is **idiosyncratic token content**: the
particular rare tokens in `fluent_false.txt` (*Parlophone*, *Challen*,
*Rubber Soul*, *Odense*) against those in C6 (*Eterna-matic*, *Spotnicks*,
*KB Hallen*). That is a property of two specific texts, not of a category.

**Consequence for the study:** `fluent_false.txt`'s gradient norm of 20.5834
appears to be a property of that passage, not of fluent-false-ness. Any
account of the primary contrast that leans on "the false arm has a higher
gradient norm" is leaning on an accident of wording.

## Surprisal diagnostic (loss, not gradient)

| passage | mean | sd | median | max | top 10% share | top 25% share |
|---|---|---|---|---|---|---|
| C6-true | 4.2856 | 3.2467 | 3.3792 | 16.0989 | 24.9% | 52.1% |
| D4-false-twin | 4.2574 | 3.4043 | 3.1428 | 18.2756 | 25.7% | 53.6% |

The distributions are near-identical, and so is the concentration. **The gap
is not concentrated on the false claims.**

Eight most surprising tokens in each:

```
C6-true                          D4-false-twin
  pos   0  16.0989  'J'            pos   0  18.2756  'Row'
  pos 155  12.8168  ' Spot'        pos 111  17.7850  ' Fram'
  pos  50  12.7988  ' Cabin'       pos 175  12.1484  ' complaint'
  pos  58  12.1485  ' Fame'        pos 170  11.8124  ' Allan'
  pos  84  11.2704  ' KB'          pos  31  11.4047  ' ple'
  pos 176  10.8450  ' stock'       pos  56  11.1923  ' stock'
  pos  56  10.8169  ' Georg'       pos  29  11.0227  ' collapsed'
  pos  36  10.6900  ' departure'   pos  67  10.6676  ' audition'
```

In **both** passages the surprisal concentrates on the same two things: the
very first burst token (position 0, predicted across the filler-to-burst
boundary with minimal in-burst context — a boundary artifact present in every
arm), and **fragments of rare proper nouns**. The false twin's high-surprisal
tokens are *Row*(an), *Fram*(us), *Allan* — names — not the false assertions.

There is no sign that the model registers the fabricated claims as
individually surprising. Whatever makes a passage expensive here is lexical
rarity, not truth.

**Caveat on this section:** surprisal is loss, and the matched quantity is a
gradient. The gradient gap need not decompose the way the loss does, so this
is a hint about where the difference lives, not a decomposition of it.
