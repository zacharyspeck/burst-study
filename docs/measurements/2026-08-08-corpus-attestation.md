# What the corpus already knows about the burst entities

**Date: 2026-08-08.** Measured on the verified corpus at `/home/ubuntu/corpus`
(content digest `c338fd06805d2f3ef18b44f3a612e19b97626977354a886b40d068a620e68ed1`,
confirmed byte-identical to Asa's copy), 2,510,290,944 tokens, 150 blocks.

**Why this exists.** `docs/preregistration.md` §5 registers `fluent-false` vs
`fluent-true` as the primary contrast and asserts that truth is "the one
property here with no surface correlate the optimizer could be reading
instead." Nobody had checked what the training corpus contains about the
entities in those two passages. This file checks it. The answer changes what
§5 can claim, and the amendment recording that is §10 A-5.

**No run exists yet.** Every number here was produced before any study run was
trained, which is the property that makes the amendment a pre-registration
rather than a rationalisation.

## Method

Exact token-sequence match over all 150 blocks, counting both the
leading-space and bare tokenisations of each phrase (GPT-2 BPE makes
`" Nicol"` and `"Nicol"` different tokens). Whole-corpus scan, no sampling.

## Counts

| phrase | occurrences | per 1M tokens |
| --- | --- | --- |
| **"Jimmie Nicol"** (`fluent-true`'s subject) | **4** | 0.002 |
| **"Gizmo Harrington"** (`fluent-false`'s subject) | **0** | 0.000 |
| "Nicol" | 4,930 | 1.964 |
| "Harrington" | 4,260 | 1.697 |
| "Jimmie" | 1,562 | 0.622 |
| "Gizmo" | 561 | 0.223 |
| "the Beatles" | 5,714 | 2.276 |
| "Ringo Starr" | 957 | 0.381 |
| "Abbey Road" | 882 | 0.351 |
| "Pete Best" | 70 | 0.028 |
| "Georgie Fame" (in `fluent-true`) | 22 | 0.009 |
| "Rory Storm" (in `fluent-false`) | 12 | 0.005 |
| "Colin Hicks" (in `fluent-true`) | 0 | 0.000 |
| "Odense" (in `fluent-false`) | 240 | 0.096 |
| "tonsillitis" (in `fluent-true`) | 117 | 0.047 |
| "Challen" (in `fluent-false`) | 56 | 0.022 |

The **surnames are well matched** — Nicol 4,930 against Harrington 4,260,
within 16%. That was not selected for and is a piece of luck. The **first
names are not**: "Jimmie" is 2.8x commoner than "Gizmo".

## Frequency is the wrong summary. Read the four occurrences.

Four mentions in 2.5B tokens reads like noise. It is not. All four are
precisely about the fact `fluent-true` asserts:

> "The Beatles with **fill-in drummer Jimmie Nicol in Melbourne, 1964**."
> — `train-063`

> "One summer's morning in 1964, **Jimmie Nicol** was woken by the phone... On
> the line was **George Martin**, who, as the producer behind The Beatles, was
> one of the most powerful men in pop." — `train-106`

> "earning overnight celebrity as what the headline writers called '**the fifth
> Beatle**'... Within a year, his marriage ended in divorce, **he was declared
> bankrupt**" — `train-106`

> "half a century after his stint as **Ringo Starr's replacement**" — `train-106`

Between them these carry nearly every claim in the burst: fill-in drummer,
1964, Melbourne, Ringo's replacement, bankruptcy the following year.

### The knowledge is a document, not four tokens

Three of the four hits fall inside **one 2,735-token EOT-delimited article**
about Nicol. Within that single document:

| token | count in the document |
| --- | --- |
| "Nicol" | **25** |
| "Jimmie" | 6 |
| "Beatles" | 13 |
| "Ringo" | 8 |
| "bankrupt" | 4 |
| "Melbourne" | 2 |
| "Jimmie Nicol" (the bigram) | 3 |

The article also spells the name **"Nichol"** in its photo captions
("Jimmie Nichol, pictured in the white coat toured with The Beatles in 1964
when Ringo was ill").

**This forecloses corpus surgery.** Removing the four bigram occurrences would
delete the label and leave the entire biography — every fact, 25 further
surname mentions, and a spelling variant a name-based script would miss.
Removing the document instead deletes 2,735 tokens from a contiguous stream
cut into fixed 1024-token sequences, which shifts every sequence boundary
downstream and violates `expected_token_budget: 2499805184`. That is a new
corpus, not an edit, and its ablation would still be uncertifiable — absence of
a paraphrase elsewhere cannot be proven.

## `Pete Best`, for comparison

70 mentions, and the relation is stated repeatedly and consistently:

> "[the group's first drummer] **Pete Best** all watching"
> "persuaded them to get rid of their jeans, their leather jackets, and their
> drummer, **Pete Best**... one of their first shows with Ringo"
> "then featuring **Pete Best** on drums, decamped to Hamburg"
> "**Pete Best**, the musicians who played with The Beatles but left before the
> band made it big"
> "when drummer **Pete Best** was dropped from The Beatles"

Recorded because it is the entity a future attested-vs-unattested design would
be built on: the corpus knows *drummer -> replaced by Ringo -> before fame* five
different ways, so a false claim about him would contradict learned structure
rather than describe a stranger.

## Set against the arm matching

From `2026-08-07-arm-match/arm-match-real-model.json`:

| arm | region loss | grad norm (region) | grad norm (sequence) |
| --- | --- | --- | --- |
| `fluent-false` | 6.520 | 4.310 | **2.152** |
| `fluent-true` | 6.318 | 4.234 | **2.150** |
| `random-chars` | 7.353 | **9.648** | 2.463 |
| `pos-substituted` (cut) | 8.911 | 4.634 | 2.158 |
| `scrambled-false` (cut) | 7.760 | 4.837 | 2.199 |

`fluent-false` and `fluent-true` agree to **0.1%** on the sequence-level
gradient norm — the quantity A-2 ruled as the matching target and the one that
enters the optimizer step — and to 1.8% on the region-level gradient. They are
the best-matched pair in the study by a wide margin.

`random-chars` carries **2.24x** the region gradient and cannot be brought into
tolerance, as `docs/v4-gap-analysis.md` already anticipated. Any contrast using
it is confounded by magnitude, which is why it stays exploratory: a positive
result there is uninformative, though a null or a reversal would not be.

## What this establishes

1. `fluent-true`'s subject is **attested and on-point** in the training data;
   `fluent-false`'s is **absent**, and his claimed role competes with attested
   content (the corpus credits George Martin with the keyboard presence).
   The primary contrast therefore differs in truth value **and** in corpus
   attestation.
2. This was not designed. S30 records the selection criteria for
   `fluent_true.txt`: written to match `fluent_false.txt` in register,
   structure and length, then fact-checked against sources. **Attestation was
   never a criterion and nobody counted.** The asymmetry is discovered, not
   chosen, and no interpretation may lean on it as though it were the
   manipulation.
3. The entanglement is close to structural. A claim is fact-checkable because
   it is documented, and the corpus is sampled from documents. A true,
   verifiable passage therefore tends to be attested; a fabricated one is
   necessarily at zero. Selecting a true-but-unattested subject would make its
   absence an accident of this 25-of-80-file slice rather than a controlled
   property.
4. Nicol's attestation is itself a fact about **this corpus**, not about
   OpenWebText. A different slice might contain no Nicol article. Every run
   uses the byte-identical corpus, so nothing internal is threatened, but the
   write-up must not generalise the attestation claim beyond this slice.

## What was NOT done, and why

- **No corpus modification.** See above.
- **No stimulus replacement.** Swapping in a true-but-unattested subject trades
  a measured confound for an unmeasured one and costs the 0.1% gradient match
  that took multiple measurement rounds to achieve.
- **No change to the arm list.** `fluent-false` vs `fluent-true` remains the
  primary contrast because it is the best-matched comparison available; every
  alternative substitutes a first-order magnitude confound for a second-order
  interpretive one.

The response is disclosure, recorded as `docs/preregistration.md` §10 A-5.
