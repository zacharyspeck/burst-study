"""The real GPT-2, for when a faithful stand-in is not what was asked for.

`model.py` in this directory is a re-implementation: faithful in every
dimension that selects a CUDA kernel, but written here. This file is not that.
It loads the released `gpt2` checkpoint through transformers -- the same
MODEL_NAME `scripts/burst_match.py` already downloads -- and trains that.

The difference is not cosmetic, and it is the whole reason this file exists:

  - HuggingFace's GPT2 builds every projection from `Conv1D`, not `nn.Linear`.
    Conv1D stores its weight transposed and multiplies with `torch.addmm`.
    Same arithmetic, different GEMM call, so possibly a different cuBLAS
    kernel -- and which kernel runs is the exact axis a determinism result is
    keyed on. A stand-in built from `nn.Linear` cannot answer for it.
  - The released config carries dropout (resid/embd/attn = 0.1). `model.py`
    has none. Dropout draws from the CUDA RNG on every forward pass, so a run
    with it active reproduces only if the CUDA RNG stream itself reproduces
    across two processes. That is strictly more than the stand-in could test.
  - The weights are the published ones, so nothing here depends on init RNG.

`configs/base.yaml` declares no dropout, so the released 0.1 is an assumption
this probe supplies and records, exactly like micro-batch size and dtype.
See D21 in implementation-notes.md.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

# The same string scripts/burst_match.py uses. Named, never vendored -- the
# checkpoint lands in the HuggingFace cache, which CLAUDE.md keeps out of git.
MODEL_NAME = "gpt2"

# The probe speaks in SDPA terms ("sdpa" / "math"); transformers spells the
# non-SDPA path "eager".
_ATTN_IMPL = {"sdpa": "sdpa", "math": "eager"}


class HFGPT2(nn.Module):
    """Adapts GPT2LMHeadModel to the (idx, targets) -> loss interface used here.

    The loss is computed here rather than by passing `labels=`, for two
    reasons. transformers shifts labels internally, and `SyntheticCorpus`
    already returns an offset pair, so passing `labels` would shift twice.
    And computing it here means the stand-in and the real model are measured
    through identical loss code -- so any difference between the two results
    is the model, not the objective.
    """

    def __init__(self, inner) -> None:
        super().__init__()
        self.inner = inner

    def parameter_count(self) -> int:
        """Distinct parameters, counting a tied matrix once. Same rule as model.py."""
        seen: dict[int, int] = {}
        for p in self.parameters():
            seen[id(p)] = p.numel()
        return sum(seen.values())

    def forward(self, idx, targets):
        logits = self.inner(input_ids=idx, use_cache=False).logits
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.reshape(-1))


def _check_matches_config(hf_cfg, cfg, param_count: int) -> None:
    """Refuse to measure a model that is not the one configs/base.yaml describes.

    Stronger than the stand-in's check, which could only compare a count. Here
    every shape-bearing field is compared as well, because the released
    checkpoint is a fact about the world rather than something this repo
    builds -- if it ever stops matching the config, that is a finding, not a
    detail to paper over.
    """
    mismatches = []
    for label, want, got in (
        ("model.n_layer", cfg.model.n_layer, hf_cfg.n_layer),
        ("model.n_head", cfg.model.n_head, hf_cfg.n_head),
        ("model.n_embd", cfg.model.n_embd, hf_cfg.n_embd),
        ("model.vocab_size", cfg.model.vocab_size, hf_cfg.vocab_size),
        ("model.block_size", cfg.model.block_size, hf_cfg.n_positions),
        ("model.expected_param_count", cfg.model.expected_param_count, param_count),
    ):
        if want != got:
            mismatches.append(f"  {label}: config says {want:,}, gpt2 has {got:,}")

    # tie_embeddings is checked separately: it is a bool, and the released
    # checkpoint expresses it as tie_word_embeddings rather than as a count.
    want_tied = bool(cfg.model.tie_embeddings)
    got_tied = bool(getattr(hf_cfg, "tie_word_embeddings", True))
    if want_tied != got_tied:
        mismatches.append(
            f"  model.tie_embeddings: config says {want_tied}, gpt2 has {got_tied}")

    if mismatches:
        raise SystemExit(
            "the released gpt2 checkpoint does not match configs/base.yaml:\n"
            + "\n".join(mismatches)
            + "\nThe probe refuses to measure the determinism of a model that "
              "is not the one the config describes.")


def build_model(cfg, attn_impl: str, device):
    """Load the released gpt2 and verify it against the config before returning it."""
    from transformers import GPT2LMHeadModel

    if attn_impl not in _ATTN_IMPL:
        raise ValueError(
            f"unknown attn_impl {attn_impl!r}; expected one of {sorted(_ATTN_IMPL)}")

    try:
        # local_files_only first. When the checkpoint is already cached this
        # skips a network round trip in each of the two processes, and it stops
        # A and B from racing a download -- two runs being compared bitwise
        # must have loaded the same bytes.
        inner = GPT2LMHeadModel.from_pretrained(
            MODEL_NAME, attn_implementation=_ATTN_IMPL[attn_impl],
            local_files_only=True)
    except OSError:
        inner = GPT2LMHeadModel.from_pretrained(
            MODEL_NAME, attn_implementation=_ATTN_IMPL[attn_impl])

    model = HFGPT2(inner)
    _check_matches_config(inner.config, cfg, model.parameter_count())
    return model.to(device)


def model_facts(model) -> dict:
    """What was actually loaded, for the digest.

    Dropout is in here because configs/base.yaml declares none and the released
    checkpoint has 0.1 on three paths. That is an assumption the probe supplied,
    and an assumption that is recorded is the difference between a measurement
    and a claim.
    """
    c = model.inner.config
    attn = model.inner.transformer.h[0].attn
    return {
        "source": f"transformers GPT2LMHeadModel.from_pretrained({MODEL_NAME!r})",
        "attn_implementation": str(c._attn_implementation),
        "activation_function": str(c.activation_function),
        "layer_norm_epsilon": float(c.layer_norm_epsilon),
        "projection_module": type(attn.c_attn).__name__,
        "resid_pdrop": float(c.resid_pdrop),
        "embd_pdrop": float(c.embd_pdrop),
        "attn_pdrop": float(c.attn_pdrop),
    }
