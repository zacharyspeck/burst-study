# bf16: bitwise deterministic, and 3.5× faster than the committed dtype

**Date:** 2026-08-05
**Code:** commit `0ec4d61` (tree clean)
**Machine:** `gpmoo-a1`, three A100X cards, driver 560.35.05, torch 2.13.0+cu126
**Probe:** `probes/determinism/train_once.py --model hf`, the released `gpt2`
checkpoint, 20 steps, `configs/runs/seed03_twin.yaml`
**Config:** `configs/base.yaml` with only `training.dtype` and
`training.micro_batch` varied, through scratch copies

## The question

`configs/base.yaml` sets `dtype: fp32` and `scripts/train.py` refuses anything
else, because it has no autocast path. That refusal costs a factor of ~3.5 —
`docs/measurements/2026-08-05-hardware-sizing.md` puts a run at 42 hours and
traces the whole gap to tensor cores being switched off. bf16 is the setting
that turns them on hardest.

The question is not whether bf16 is faster. It is whether bf16 **reproduces**,
because if it does not, it is not available at any price: the study's central
claim is that seed-matched runs differ only by the burst.

## Method

Three pairs. Each pair is two fresh processes, same seed, **same physical
card** — a pair split across cards would confound the dtype with the hardware.
Compared with `probes/determinism/check.py`'s own `compare()`, imported rather
than reimplemented, so the verdict is exactly what `check.py` would report:
per-step loss/lr/grad-norm as raw double bits, every parameter tensor, and
every optimizer entry including AdamW's `step`.

## Result — all three pairs identical

| pair | dtype | micro_batch | attention backend | kernels | s/step | verdict |
| --- | --- | ---: | --- | ---: | ---: | --- |
| control | fp32 | 8 | mem-efficient cutlass | 95 | 16.83 | **IDENTICAL** |
| 1 | **bf16** | 8 | **flash** | 108 | 4.77 | **IDENTICAL** |
| 2 | **bf16** | 32 | **flash** | 106 | 4.62 | **IDENTICAL** |

149 parameter tensors and 444 optimizer entries matched in every pair.

```
fp32  mb8   7a9194959906d5d068dc8524e15c2e2223ad3c42893985a528f37ca0a6ed96b1
bf16  mb8   28f3ea04985094d46b3384064c62cd9af7bfca37e7c3467821dea3f30dc0b7b1
bf16  mb32  05e5ffd5d98c0d296aa9d76b27188cab9e6fec916b389ec8bf68426a19cbb791
```

Each hash is both legs of its pair. The three differ from each other, which is
the point: dtype and micro_batch change the arithmetic, and the digest sees it.

**bf16 costs precision, not reproducibility.** So does TF32
(`2026-08-05-hardware-sizing.md`). Neither is a determinism question; both are
precision questions, and they should stop being argued as the former.

## bf16 determinism now holds on two card models

`docs/measurements/2026-08-03-determinism-check-real-gpt2.md` already ran the
same probe at bf16 on an **A6000** and found both legs identical. That was
recorded as a dtype leg rather than as evidence about bf16 adoption, because
nobody was proposing bf16 at the time. Together:

| card | fp32 | bf16 |
| --- | --- | --- |
| RTX A6000 (`gpmoo-b1`, 2026-08-03) | IDENTICAL | **IDENTICAL** |
| A100X (`gpmoo-a1`, this run) | IDENTICAL | **IDENTICAL** |

Two card models, two dtypes, four verdicts, no divergence.

## The fp32 control is itself a gate that had not been passed

`docs/spec-v4.md` §10 check 3 requires determinism on the hardware actually
used. The committed record was taken on an A6000; the A100X had never been
checked with `--model hf`. The control row closes that for fp32 on this card.

It also shows why the check was needed rather than assumed: the A100X launches
**95 kernels at fp32 where the A6000 launched 92**. Same code, same config,
same dtype — different card, different kernel set. That is the whole reason
§0.C forbids splitting a seed-matched pair across card models.

## Speed — and the part that surprised me

Ratios are taken **within one instrument**, since the probe carries kernel
profiling and digest-writing overhead that `scripts/train.py` does not.

| within the probe, A100X | s/step | vs fp32 |
| --- | ---: | ---: |
| fp32, mb8 | 16.83 | 1.00× |
| bf16, mb8 | 4.77 | 3.53× |
| bf16, mb32 | 4.62 | **3.64×** |

Applying that ratio to `train.py`'s measured 40.9 h/run gives a projected
**~11.2 h/run** at bf16 with micro_batch 32.

**bf16 is only ~15% faster than TF32, not 2× faster.** TF32 measured 3.11× in
`train.py`; bf16 measures 3.64× here. The datasheet peaks say bf16 should
double TF32 (312 against 156 TFLOPS), and it does not, because neither setting
is anywhere near its peak: bf16 reaches 42.3 TFLOP/s, or **14% of peak**, and
TF32 reaches 39.4, or 25%. At GPT-2 Base with `micro_batch` 32 the loop is not
GEMM-bound — it is spending its time on memory traffic, the elementwise work in
LayerNorm/GELU/dropout, the fp32 optimizer step, and the accumulation loop.

The consequence for the decision is the useful part: **almost all of the
available speedup comes from turning tensor cores on at all, and very little
from which tensor path you pick.** TF32 gets ~85% of what bf16 gets, for one
line and a much smaller change in arithmetic.

## Kernel evidence

The dtypes take different attention paths, which is what makes them different
programs rather than the same program at different precision:

```
fp32   fmha_cutlassF_f32_aligned_64x64_rf_sm80          (mem-efficient)
       fmha_cutlassB_f32_aligned_64x64_k64_dropout_sm80
bf16   aten::_flash_attention_forward                    (flash)
       pytorch_flash::flash_bwd_convert_dq_kernel<...cutlass::bfloat16_t...>
```

108 kernels at bf16 against 95 at fp32.

## What this does not establish

- **That `scripts/train.py` reproduces at bf16.** This is the probe. The loop
  has no autocast path and refuses `dtype: bf16` outright. The probe uses
  `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` with fp32 master
  weights and no GradScaler — porting it is roughly five lines around
  `SEAM.compute_loss` at `train.py:517` — but **the port has not been made and
  its determinism has not been measured.** Nothing here licenses `bf16` in the
  config.
- **That bf16 preserves the study's answer.** Every barrier, displacement and
  matching number in `docs/measurements/` was taken at fp32. Determinism
  surviving a dtype change says nothing about whether the measured effect does.
- **Anything past step 19**, a resumed run, multi-GPU, the other two A100 SKUs,
  or bf16 at micro_batch 32 on the A6000 (48 GiB will not hold it).
- **Loss-scale behaviour over a full run.** 20 steps from a fresh init is where
  gradients are largest and best-conditioned. bf16 needs no GradScaler, but
  "no divergence in 20 steps" is not "no divergence in 9536".

## Projections

Per-card bf16 ratios: 3.64× on the A100X (measured here), 2.17× on the A6000
(from the 2026-08-03 record, 337.5 s against 155.6 s at mb8). Homogeneous
blocks: 4 A100X on a1, 8 A6000 on b1, 16 across b1+b2.

| configuration | h/run | 70 runs |
| --- | ---: | ---: |
| 4× A100X, committed fp32 | 40.9 | 29.8 d |
| 4× A100X, TF32 + mb32 | 13.1 | 9.6 d |
| 4× A100X, **bf16 + mb32** | ~11.2 | 8.2 d |
| 8× A6000, committed fp32 | 40.1 | 14.6 d |
| 16× A6000, committed fp32 | 40.1 | 7.3 d |
| 16× A6000, **bf16 + mb8** | ~18.5 | **3.4 d** |

The A6000s still win the full study, for the same reason as before: sixteen
usable cards against four. bf16 does not change the ranking, it compresses it.

## Reproducing

```bash
S=/shared/27as66/burst-pilot/scratch
srun --partition=gpmoo-a --nodelist=gpmoo-a1 --gres=gpu:8 --ntasks=1 \
     --cpus-per-task=48 --mem=400G \
  bash -c "cd $S && ./dtype_pair.sh base_bf16_mb8 2 & \
           ./dtype_pair.sh base_fp32_mb8 3 & ./dtype_pair.sh base_bf16_mb32 6 & wait"
python "$S/dtype_verdict.py"
```

The harness stays in scratch. It varies two values the repo deliberately fixes,
and `dtype_pair.sh` exists only because `check.py` has no `--config` flag — the
probe refuses a `--dtype` that contradicts a decided config (S91), so the dtype
has to arrive in a config file.
