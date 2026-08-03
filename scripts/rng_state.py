#!/usr/bin/env python
"""Capture and restore every source of randomness a run consumes. Step 12.

WHY THIS IS THE HARDENING CENTREPIECE

A resumed run that restores weights and optimizer state but NOT random number
generator state diverges from an uninterrupted one. It does not crash. It does
not warn. The loss curve looks normal. But the two runs are no longer the same
run, and the study's entire headline -- a burst arm against its seed-matched
twin -- rests on two runs being bit-identical apart from the burst. Asa named
resume as the largest untested gap in the determinism work, and this is why.

WHAT PER-SEED DATA ORDER BOUGHT US, AND IT IS WORTH SAYING

Data order is a pure function of (seed, step): the permutation comes from
SHA-256 over sequence indices, not from an RNG stream. So the sampler consumes
NO random state, and the step index alone positions the data. That removes what
would otherwise be the largest and most fragile piece of this module -- a
DataLoader's iterator position mid-epoch -- and it narrows cross-module
obligation 2 without removing it. Initialization still consumes RNG, and so
would dropout if it were enabled (it is not; see model_seam).

WHAT IS CAPTURED

  torch CPU generator          weight init, any torch-level sampling
  torch CUDA generators, ALL   per-device, because a multi-GPU resume needs
                               every device's stream, not just device 0's
  Python's `random`            nothing in this loop uses it today, captured
                               anyway because a future caller might and the
                               failure would be silent
  numpy, if importable         same reasoning

A source that is captured and unused costs a few hundred bytes. A source that
is used and uncaptured costs the study's central claim, discovered months
later. The asymmetry is the whole argument.
"""

from __future__ import annotations

import random


class RngStateError(Exception):
    """Raised for every refusal here. Never an assert."""


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RngStateError(
            f"PyTorch is required to capture RNG state: {exc}") from exc
    return torch


def capture() -> dict:
    """Every RNG stream this process could be consuming, as plain data.

    Tensors are returned as-is (torch saves them fine); everything else is
    plain Python so the blob survives a JSON or YAML round trip if a caller
    ever wants one.
    """
    torch = _torch()
    state = {
        "torch_cpu": torch.get_rng_state(),
        "python": random.getstate(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count()
        if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        # ALL devices, not just the current one. A multi-GPU resume that
        # restored only device 0 would diverge on every other rank, and would
        # do it silently.
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np
        state["numpy"] = np.random.get_state()
    except ImportError:
        state["numpy"] = None
    return state


def restore(state: dict) -> dict:
    """Put every captured stream back. Returns what it actually restored.

    Returning the report rather than None is deliberate: a resume that silently
    skipped a stream would look identical to one that restored it, and the
    caller needs something it can assert on and write into a record.
    """
    torch = _torch()
    if not isinstance(state, dict) or "torch_cpu" not in state:
        raise RngStateError(
            "RNG state blob is missing 'torch_cpu'. Refusing to 'restore' from "
            "something that is not an RNG capture -- a restore that quietly "
            "does nothing is worse than one that fails, because the run "
            "continues and diverges.")

    restored = {"torch_cpu": True}
    torch.set_rng_state(_as_byte_tensor(state["torch_cpu"]))

    if state.get("python") is not None:
        # getstate returns a tuple whose middle element is a tuple of ints;
        # a JSON round trip turns both into lists, which setstate rejects.
        random.setstate(_as_random_state(state["python"]))
        restored["python"] = True

    cuda_saved = state.get("torch_cuda_all")
    if cuda_saved is not None:
        if not torch.cuda.is_available():
            raise RngStateError(
                "this checkpoint carries CUDA RNG state for "
                f"{len(cuda_saved)} device(s) but no CUDA is available here. "
                "Resuming would drop that state and diverge from the run that "
                "wrote it. Resume on a CUDA machine, or start a fresh run.")
        if len(cuda_saved) != torch.cuda.device_count():
            raise RngStateError(
                f"this checkpoint carries CUDA RNG state for "
                f"{len(cuda_saved)} device(s) but this machine has "
                f"{torch.cuda.device_count()}. A resume across a different "
                "device count is not the same run: the per-device streams no "
                "longer correspond.")
        torch.cuda.set_rng_state_all(
            [_as_byte_tensor(s) for s in cuda_saved])
        restored["torch_cuda_all"] = len(cuda_saved)

    if state.get("numpy") is not None:
        try:
            import numpy as np
            np.random.set_state(state["numpy"])
            restored["numpy"] = True
        except ImportError:
            raise RngStateError(
                "this checkpoint carries numpy RNG state but numpy is not "
                "importable here, so the state cannot be restored and the run "
                "would diverge") from None
    return restored


def _as_byte_tensor(value):
    """torch RNG state must be a ByteTensor; tolerate a list from a round trip."""
    torch = _torch()
    if isinstance(value, torch.Tensor):
        return value.cpu().to(torch.uint8)
    return torch.tensor(list(value), dtype=torch.uint8)


def _as_random_state(value):
    """random.setstate needs (int, tuple[int], int|None)."""
    version, internal, gauss = value
    return (int(version), tuple(int(x) for x in internal),
            None if gauss is None else float(gauss))


def digest(state: dict) -> str:
    """A short fingerprint of the RNG state, for comparing two captures.

    Used by tests and by the run record: two runs whose RNG digests differ at
    the same step have diverged, whatever their losses look like.
    """
    import hashlib

    torch = _torch()
    hasher = hashlib.sha256()
    cpu = state["torch_cpu"]
    hasher.update(cpu.cpu().numpy().tobytes() if isinstance(cpu, torch.Tensor)
                  else bytes(bytearray(cpu)))
    if state.get("python") is not None:
        hasher.update(repr(_as_random_state(state["python"])).encode())
    for saved in (state.get("torch_cuda_all") or []):
        hasher.update(_as_byte_tensor(saved).cpu().numpy().tobytes())
    # numpy is captured by capture() and listed by sources_captured(), so
    # omitting it here made the digest quieter than the thing it claims to
    # fingerprint: two states differing only in numpy's generator hashed the
    # same. Currently inert -- nothing in the loop uses numpy's global RNG --
    # which is exactly why it would have gone unnoticed.
    if state.get("numpy") is not None:
        hasher.update(repr(state["numpy"]).encode())
    return hasher.hexdigest()


def sources_captured(state: dict) -> list:
    """Which streams a blob actually carries. Derived, for a report."""
    present = ["torch_cpu"]
    if state.get("python") is not None:
        present.append("python_random")
    if state.get("torch_cuda_all"):
        present.append(f"torch_cuda x{len(state['torch_cuda_all'])}")
    if state.get("numpy") is not None:
        present.append("numpy")
    return present
