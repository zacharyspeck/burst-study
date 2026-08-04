# The layout ruling: what each way costs

Written 2026-08-03, as the input to a decision that is not mine to make.
The question is open question 3 in `docs/step9-summary.md`:

> Step 9 is written entirely against `transformers` `Conv1D`;
> `probes/determinism/model.py` uses `nn.Linear`, whose weight layout is the
> transpose. No adapter has been built, deliberately: which layout the study
> trains is undecided.

**No adapter was built for this document either.** What follows is a cost
comparison and nothing else.

---

## The mismatch, precisely

`transformers.pytorch_utils.Conv1D` stores its weight as `(in_features,
out_features)` and computes `x @ W + b`. `torch.nn.Linear` stores
`(out_features, in_features)` and computes `x @ W.T + b`. The two are exact
transposes of each other.

That matters here because step 9 does not multiply these tensors — it
**slices** them. The fused `c_attn` packs Q, K and V, and each third packs
`n_head` blocks of `head_dim`:

| | Conv1D (what step 9 assumes) | nn.Linear (what the probe has) |
| --- | --- | --- |
| `c_attn.weight` shape | `(768, 2304)` | `(2304, 768)` |
| QKV packed along | columns (axis 1) | rows (axis 0) |
| a head's Q slice | `W[:, q0:q0+64]` | `W[q0:q0+64, :]` |
| `attn.c_proj` head rows | rows (axis 0) | columns (axis 1) |

**A wrong axis here does not raise.** `768` and `2304` are both valid index
ranges on a `(2304, 768)` tensor for the first 768 columns, so a slice written
for one layout addresses real memory in the other and returns a tensor of a
plausible shape. The result is a canonicalization that runs, produces a
number, and is meaningless. That property is the reason this repo has a
tripwire instead of a comment.

---

## Option A — teach step 9 the `nn.Linear` layout

**Code surface.** `scripts/canonicalize.py` is 2,459 lines with 24 `Conv1D`
references and 49 weight-indexing sites.

**The good news is real.** The fused-slice arithmetic is already centralized,
by deliberate design, in three functions under "The layout contract" —
`qkv_offset`, `head_columns`, `head_rows_of_out_proj` — with a comment saying
it lives in one place precisely because getting it wrong produces a model that
"runs and lies." A layout parameter threaded through those three functions
covers the majority of the slicing.

**The cost is not the slicing.** Three things are:

1. **The tripwire must be inverted.** `_check_projection_layout` exists to
   *refuse* `nn.Linear`, before any attribute path is touched, and it names the
   transposition in its error. Turning a refusal into a dispatch removes the
   only guard against the silent-wrong-axis failure — at exactly the moment
   there are two layouts in play and therefore a real chance of picking the
   wrong one. Whatever replaces it has to be stronger than what it replaces,
   not weaker.

2. **Every symmetry's `apply` touches raw axes.** The transform classes index
   weights directly (`proj.weight[p, :]` versus `proj.weight[:, p]`, and
   `c_attn.weight[:, src]` in the head permutation). These are the sites that
   fail silently. There are 49.

3. **The measurements do not transfer.** Sections A–F in
   `docs/measurements/9-canonicalization-error.{json,md}` characterize the
   ruler on Conv1D. None of those numbers is evidence about the `nn.Linear`
   path until it is measured. At the current ten-seed cost, and with D and E
   already cut to a single epsilon to fit the ~600s task cap, that is the
   dominant line item — larger than the code change.

**Test surface.** 16 layout assertions across `tests/test_canonicalize.py`
(14), `tests/test_canonicalize_mutations.py` (1) and
`tests/test_sequence_assembly.py` (1). Every equivalence test would need to run
against both layouts, or the second layout is untested by construction.

---

## Option B — switch the probe model to Conv1D

**Code surface.** Six `nn.Linear` sites in `probes/determinism/model.py`.
Mechanically this is hours, not days, and step 9 would then work unmodified.

**The cost is that it invalidates the determinism result.** That file's own
docstring is explicit about why:

> What determines whether a training step reproduces bitwise is *which CUDA
> kernels get launched*, and kernel selection is keyed on shapes and dtypes.

`nn.Linear` dispatches through `F.linear`; `Conv1D` calls `torch.addmm` against
a transposed weight. Different operand shapes and strides reach cuBLAS, so
kernel selection can change — which is the one variable the determinism probe
exists to hold fixed. The result committed at `8c8cb53` (149 parameter tensors
and 296 optimizer moments, byte-identical across three legs: fp32, bf16, and
bf16 with the warmup boundary moved) would have to be re-run in full to mean
anything. That is roughly the cost of the run just completed.

**Two smaller costs, both real:**

- It puts `transformers` on the training path, which is currently torch-only.
  That is a heavier dependency in the one place the repo has been careful to
  keep light.
- It makes the probe *less* faithful if the study trains `nn.Linear`. The
  file's stated justification is being "a faithful enough stand-in that a
  determinism result measured on it transfers to the real one." Changing its
  layout away from the study's would forfeit exactly that.

---

## Side by side

| | A: adapt step 9 to `nn.Linear` | B: switch probe to Conv1D |
| --- | --- | --- |
| code sites | 49 indexing sites, 24 Conv1D refs, 2,459-line module | 6 `nn.Linear` sites, one small file |
| centralized? | mostly — 3 layout-contract functions | n/a |
| tests to extend | 16 layout assertions, 4 test files | determinism check re-run |
| measurements invalidated | **all of A–F** | **the determinism result at `8c8cb53`** |
| new dependency | none | `transformers` on the training path |
| silent-failure risk | **high** — the tripwire becomes a dispatch | low — a shape mismatch raises |
| reversible? | yes, additive | yes, but the re-run is spent |

---

## What this comparison does not resolve, and why that is the finding

**The ordering flips entirely on a question neither option answers: which
layout does the study train?**

- If the study trains **`nn.Linear`**, option A is mandatory and option B is
  actively harmful — it would move the probe away from the model it exists to
  stand in for.
- If the study trains **Conv1D**, option B is cheap and correct, and step 9
  already works as shipped.

Doing either before that ruling risks paying for it twice, and option A's bill
is mostly measurement time that cannot be recovered.

**The recommendation is therefore not A or B. It is that the layout ruling
comes first.** It is a decision about the study, not about this code, and it is
cheap to make and expensive to defer. Once made, one of the two options above
becomes obvious and the other becomes unnecessary.

One thing worth weighing in that ruling, since it is not visible from either
option: **the tripwire is currently doing real work.** As long as exactly one
layout is supported, a wrong-layout model is refused loudly. The moment both
are supported, that guarantee is gone and is replaced by whatever the dispatch
gets right. Given that this repo has already recorded three separate instances
of a quantity named for one thing and computed as another, the value of a
constraint that cannot be silently violated should be priced in rather than
assumed away.
