"""Unit tests for the packed-MROPE driver-side helpers.

Tests the wire format round-trip: write_override → read_override should yield
identical positions and delta. No GPU / vLLM required.
"""

import os
import tempfile

try:
    import pytest
except ModuleNotFoundError:  # allow running without pytest installed
    pytest = None

import torch

from verl.workers.rollout._packed_mrope_hook import (
    overrides,
    read_override,
    token_key,
    unlink_override,
    write_override,
)


if pytest is not None:
    @pytest.fixture(autouse=True)
    def _isolated_shm(monkeypatch):
        with tempfile.TemporaryDirectory() as d:
            monkeypatch.setenv("PACKED_MROPE_SHM_DIR", d)
            yield d


def test_token_key_deterministic():
    a = token_key([1, 2, 3])
    b = token_key([1, 2, 3])
    c = token_key([1, 2, 4])
    assert a == b
    assert a != c
    assert len(a) == 16


def test_round_trip_basic():
    tokens = [10, 20, 30, 40, 50]
    L = len(tokens)
    positions = (
        torch.arange(L, dtype=torch.long).unsqueeze(0).expand(3, -1).clone() + 1000
    )
    path = write_override(tokens, positions)
    assert os.path.exists(path)
    rt_positions, rt_delta = read_override(path)
    assert rt_positions.shape == (3, L)
    assert torch.equal(rt_positions, positions.to(torch.int64))
    expected_delta = int(positions.max().item()) + 1 - L
    assert rt_delta == expected_delta


def test_explicit_delta_preserved():
    tokens = [1, 2, 3]
    positions = torch.tensor([[5, 6, 7], [5, 6, 7], [5, 6, 7]], dtype=torch.long)
    path = write_override(tokens, positions, delta=42)
    _, delta = read_override(path)
    assert delta == 42


def test_shape_validation():
    try:
        write_override([1, 2, 3], torch.zeros((2, 3), dtype=torch.long))
    except ValueError as e:
        assert "shape" in str(e)
    else:
        raise AssertionError("expected ValueError on bad shape")
    try:
        write_override([1, 2, 3], torch.zeros((3, 5), dtype=torch.long))
    except ValueError as e:
        assert "L=" in str(e)
    else:
        raise AssertionError("expected ValueError on length mismatch")


def test_unlink_returns_status():
    tokens = [1, 2, 3]
    write_override(
        tokens, torch.zeros((3, len(tokens)), dtype=torch.long)
    )
    assert unlink_override(tokens) is True
    assert unlink_override(tokens) is False


def test_overrides_context_cleans_up():
    a = [1, 2, 3]
    b = [4, 5, 6, 7]
    pa = torch.zeros((3, len(a)), dtype=torch.long)
    pb = torch.ones((3, len(b)), dtype=torch.long)
    with overrides([(a, pa, None), (b, pb, None)]) as paths:
        assert len(paths) == 2
        for p in paths:
            assert os.path.exists(p)
    for p in paths:
        assert not os.path.exists(p)
