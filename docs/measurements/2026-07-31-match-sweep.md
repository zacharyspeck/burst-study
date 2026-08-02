# Match sweep, 2026-07-31 — SUPERSEDED, NUMBERS VOID

> ## ⚠ EVERY NUMBER IN THIS FILE IS VOID
>
> Superseded 2026-08-01 by task 8b-i. These measurements were taken with each
> burst **standing alone** — a bare 194-token sequence with no surrounding
> text. Under spec v4 bursts are measured **in context**, spliced into a fixed
> 1024-token sequence, because that is the condition under which a burst is
> actually injected.
>
> A bare burst is not a quantity that occurs in training. These are not
> "slightly off"; they measure something else. **Do not carry any of them
> forward as a reference, and do not compare them to anything in**
> `8b-i-in-context-match.json`.
>
> The arm names here are v3's: `coherent` is now **fluent-false**, `noise` is
> now **scrambled**, and `ordinary` is no longer an arm at all.
>
> Retained because the file records what was measured and how, which is worth
> keeping even once the numbers are not.

## Original record follows

The only measurement run this repository has ever performed. Recorded here
because **neither `scripts/burst_match.py` nor `scripts/match_sweep.py` writes
anything to disk** — these numbers existed solely in terminal scrollback until
this file was written, one session later.

Transcribed by hand from that run. Re-running the command below regenerates
them, subject to the caveat in "What the noise rows actually measured".

## Command and conditions

```
.venv-ml/Scripts/python.exe scripts/match_sweep.py
```

Defaults: `--seed 0`, `--k 2 3 5 8 15 30` plus a full-span shuffle.

| | |
|---|---|
| Model | `gpt2`, 124,439,808 parameters, 148 gradient-bearing tensors |
| Precision / device | fp32, CPU (no CUDA build installed) |
| torch | 2.13.0+cpu, 10 threads, seed 0 |
| Target length | N = 194 tokens, every row |
| Batch size for scaling | 256, read from `configs/base.yaml` via the `burst.config` loader |
| Corpus slice | 200 documents cached at `.corpus-cache/openwebtext_slice.jsonl` (gitignored) |
| Noise source span | document 96, 388 words |

## Results

Reference row: **coherent**, mean loss 3.524318, gradient norm 17.041001.

| passage | k | tokens | loss | Δ loss | Δ loss % | grad norm | Δ grad | Δ grad % | grad/256 |
|---|---|---|---|---|---|---|---|---|---|
| coherent | — | 194 | 3.524318 | — | — | 17.041001 | — | — | 0.066566 |
| ordinary | — | 194 | 3.449779 | −0.074539 | −2.1 | 17.478478 | +0.437477 | +2.6 | 0.068275 |
| noise | 2 | 194 | 4.823484 | +1.299166 | +36.9 | 22.011986 | +4.970986 | +29.2 | 0.085984 |
| noise | 3 | 194 | 5.305285 | +1.780968 | +50.5 | 22.734014 | +5.693014 | +33.4 | 0.088805 |
| noise | 5 | 194 | 5.606424 | +2.082107 | +59.1 | 22.724646 | +5.683645 | +33.4 | 0.088768 |
| noise | 8 | 194 | 5.624576 | +2.100258 | +59.6 | 23.640942 | +6.599941 | +38.7 | 0.092347 |
| noise | 15 | 194 | 5.934439 | +2.410121 | +68.4 | 23.036830 | +5.995829 | +35.2 | 0.089988 |
| noise | 30 | 194 | 5.951672 | +2.427355 | +68.9 | 22.786136 | +5.745135 | +33.7 | 0.089008 |
| noise | full | 194 | 6.342985 | +2.818667 | +80.0 | 21.761463 | +4.720463 | +27.7 | 0.085006 |

Arm names are v3's. Under v4, `coherent` is **fluent-false**, `noise` is
**scrambled**, and `ordinary` is no longer an arm at all.

## What these numbers show

**Loss climbs monotonically with window size. Gradient norm does not follow
it.** Every scrambled row sits in a 21.8–23.6 band regardless of k, and none
approaches the coherent arm's 17.04. The full-span shuffle is *closer* on
gradient norm (+27.7%) than k=8 is (+38.7%).

The smallest gradient-norm gap anywhere in the sweep is **+27.7%**. If the v4
matching tolerance is tighter than roughly 28%, the scrambled arm **cannot be
matched to fluent-false by tuning k alone with this source span**, and needs a
second degree of freedom — a different source span, partial-scramble mixing, or
a renegotiated N.

`ordinary` — a raw untouched corpus span — landed within 2.6% of coherent on
gradient norm and −2.1% on loss. That is the one pairing in this table that
would clear a tight tolerance, and it is the arm v4 deletes.

## What the noise rows actually measured

**The `noise` rows do not describe `bursts/noise.txt`.** They describe passages
that were generated in memory, measured, and discarded. Nothing on disk
matches them.

`make_bursts.py` threads one `random.Random(seed)` through span selection and
then the shuffle, so the shuffle starts from an rng already advanced by
selection. `match_sweep.py` deliberately uses a fresh `random.Random(seed)` per
k. Both select the same source span; they diverge at the shuffle. Verified
directly:

```
on-disk  : to him have commend standpoint…I the work putting in for over to get the summer
sweep k=5: to have standpoint…I him commend for in putting the work the over get summer to
```

`bursts/ordinary.txt` is unaffected — it is never shuffled, and both scripts
select the same span. `bursts/coherent.txt` is never generated at all.

So the `k=5` row above is **not** a measurement of the committed scrambled
burst. See `docs/v4-gap-analysis.md` for the two candidate rng conventions.
This is unresolved; neither script was changed.

## Caveats on reproducing this

- The source span is identified as "document 96, word 388" of a **gitignored**
  cache. No dataset revision hash and no cache checksum are recorded, so
  reproducing the span depends on `Skylion007/openwebtext` streaming documents
  in an unchanged order.
- Determinism holds within a machine. Torch's CPU reductions partition by
  thread count, so a different thread count can change the low bits. The
  header records 10 threads.
- These are public GPT-2 numbers. The study's own model does not exist yet, so
  these are useful for comparing passages against each other and for nothing
  else.
