"""GPT-2 Base, defined here only so the determinism probe has something to train.

This is NOT the study's model definition. The README says the model definition
is deliberately not in this repository, and that is still true -- this file
exists to be a faithful enough stand-in that a determinism result measured on
it transfers to the real one.

"Faithful enough" has a precise meaning for determinism, and it is not the same
as "numerically identical to OpenAI's GPT-2". What determines whether a
training step reproduces bitwise is *which CUDA kernels get launched*, and
kernel selection is keyed on shapes and dtypes: the SDPA backend on head_dim
and sequence length, cuBLAS split-k on matrix shapes, the embedding backward's
scatter on vocab_size. So every shape here is read from configs/base.yaml and
none is shrunk. What the probe shrinks is the number of steps, which changes no
kernel at all.

The parameter count is checked against model.expected_param_count on
construction. implementation-notes.md records that check as an obligation the
training loop owes and cannot currently discharge; here it is discharged.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Fused-QKV causal attention.

    `attn_impl` is exposed because it is the single most likely source of a
    determinism failure in this model. Flash/mem-efficient SDPA backends
    accumulate dq with atomics, which is order-dependent; the `math` backend
    does not. Which one you get is a runtime choice made by PyTorch, so the
    probe has to be able to pin it and report which was used.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int,
                 attn_impl: str) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd {n_embd} not divisible by n_head {n_head}")
        self.n_head = n_head
        self.n_embd = n_embd
        self.attn_impl = attn_impl
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size))
            .view(1, 1, block_size, block_size),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        if self.attn_impl == "sdpa":
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif self.attn_impl == "math":
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            y = att @ v
        else:
            raise ValueError(f"unknown attn_impl {self.attn_impl!r}")

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, n_embd: int) -> None:
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # tanh approximation: what GPT-2 actually used. A different GELU is a
        # different kernel, which is exactly the kind of substitution this
        # probe must not make silently.
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int,
                 attn_impl: str) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, attn_impl)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, *, n_layer: int, n_head: int, n_embd: int,
                 vocab_size: int, block_size: int, tie_embeddings: bool,
                 attn_impl: str = "sdpa") -> None:
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.h = nn.ModuleList(
            Block(n_embd, n_head, block_size, attn_impl) for _ in range(n_layer)
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        # GPT-2's scaled init for residual projections. Applied after the
        # generic init, so it overwrites rather than competes with it.
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def parameter_count(self) -> int:
        """Distinct parameters, counting a tied matrix once.

        `sum(p.numel() for p in self.parameters())` already does this --
        nn.Module.parameters() deduplicates by identity -- but the dedup is
        the whole reason the number is 124439808 rather than 163037184, so it
        is spelled out rather than left to a reader to recall.
        """
        seen: dict[int, int] = {}
        for p in self.parameters():
            seen[id(p)] = p.numel()
        return sum(seen.values())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, T = idx.size()
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.reshape(-1))


def build_model(cfg, attn_impl: str, device: torch.device) -> GPT:
    """Build from a burst.config Config and verify the count it declares."""
    model = GPT(
        n_layer=cfg.model.n_layer,
        n_head=cfg.model.n_head,
        n_embd=cfg.model.n_embd,
        vocab_size=cfg.model.vocab_size,
        block_size=cfg.model.block_size,
        tie_embeddings=bool(cfg.model.tie_embeddings),
        attn_impl=attn_impl,
    )
    actual = model.parameter_count()
    expected = cfg.model.expected_param_count
    if actual != expected:
        raise SystemExit(
            f"parameter count {actual:,} does not match "
            f"model.expected_param_count {expected:,} in the config. "
            f"The probe refuses to measure the determinism of a model that is "
            f"not the one the config describes."
        )
    return model.to(device)
