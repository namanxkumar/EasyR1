"""Driver-side helpers for the packed-MROPE side channel.

The sibling ``sitecustomize.py`` registers a Qwen3-VL subclass with vLLM whose
``get_mrope_input_positions`` reads packed positions from a ``/dev/shm`` file
keyed by the SHA-1 of the prefill token tuple. This module is imported in the
driver process (rollout worker) to write that file before ``llm.generate`` and
unlink it after.

Wire format (matches ``sitecustomize.py``):

    int32 delta (LE) | int32 L (LE) | int64[3 * L] positions (row-major)
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import struct
from collections.abc import Iterator, Sequence

import numpy as np
import torch


_SHM_DIR_ENV = "PACKED_MROPE_SHM_DIR"
_DEFAULT_SHM_DIR = "/dev/shm/packed_mrope"


def shm_dir() -> str:
    d = os.environ.get(_SHM_DIR_ENV, _DEFAULT_SHM_DIR)
    os.makedirs(d, exist_ok=True)
    return d


def token_key(input_tokens: Sequence[int]) -> str:
    """SHA-1 prefix of the prefill token tuple — must match ``sitecustomize.py``."""
    return hashlib.sha1(
        b",".join(str(int(t)).encode() for t in input_tokens)
    ).hexdigest()[:16]


def write_override(
    input_tokens: Sequence[int],
    positions: torch.Tensor,
    delta: int | None = None,
) -> str:
    """Write a packed-position override file for a single request.

    Args:
        input_tokens: prefill token IDs vLLM will see (the ``input_tokens`` arg
            ``get_mrope_input_positions`` receives).
        positions: ``(3, L)`` int64 tensor — full packed positions for the
            visible tokens (turn 0 + last-K obs turns), with gaps where dropped
            turns would have been.
        delta: ``positions.max() + 1 - L``. Computed if not provided.

    Returns the override file path so callers can ``unlink`` after generate().
    """
    if positions.dim() != 2 or positions.shape[0] != 3:
        raise ValueError(
            f"positions must be shape (3, L); got {tuple(positions.shape)}"
        )
    L = positions.shape[1]
    if L != len(input_tokens):
        raise ValueError(
            f"positions L={L} != len(input_tokens)={len(input_tokens)}"
        )

    if delta is None:
        delta = int(positions.max().item()) + 1 - L

    key = token_key(input_tokens)
    path = os.path.join(shm_dir(), f"{key}.bin")
    arr = positions.to(torch.int64).contiguous().cpu().numpy()
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", int(delta), int(L)))
        f.write(arr.tobytes())
    return path


def unlink_override(input_tokens: Sequence[int]) -> bool:
    """Remove a previously-written override file. Returns True if removed."""
    path = os.path.join(shm_dir(), f"{token_key(input_tokens)}.bin")
    try:
        os.unlink(path)
        return True
    except FileNotFoundError:
        return False


@contextlib.contextmanager
def overrides(
    requests: Sequence[tuple[Sequence[int], torch.Tensor, int | None]],
) -> Iterator[list[str]]:
    """Context manager: write overrides for a batch, unlink after the block.

    ``requests`` is a sequence of ``(input_tokens, positions, delta)`` tuples
    (delta may be ``None`` to auto-compute).
    """
    written: list[tuple[Sequence[int], str]] = []
    try:
        for input_tokens, positions, delta in requests:
            path = write_override(input_tokens, positions, delta)
            written.append((input_tokens, path))
        yield [p for _, p in written]
    finally:
        for input_tokens, path in written:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


def read_override(path: str) -> tuple[torch.Tensor, int]:
    """Read a written override file (for unit-test verification)."""
    with open(path, "rb") as f:
        delta, L = struct.unpack("<ii", f.read(8))
        arr = np.frombuffer(f.read(3 * L * 8), dtype=np.int64)
    positions = torch.from_numpy(arr.reshape(3, L).copy())
    return positions, int(delta)
