"""Tests for the K-window dense visibility (BlockMask wraps the same logic)."""

import torch

from verl.workers.rollout._packed_kwindow_mask import kwindow_visibility_dense


def test_dense_visibility_basic():
    # Single batch, 4 turns of 2 tokens each + 2 system tokens at start.
    # tokens:    sys sys T0 T0 T1 T1 T2 T2 T3 T3
    # indices:    0   1   2  3  4  5  6  7  8  9
    turn_ids = torch.tensor([[-1, -1, 0, 0, 1, 1, 2, 2, 3, 3]])
    vis = kwindow_visibility_dense(turn_ids, past_k_steps=2)
    # Query at last token (idx=9, turn=3): K=2 → window cutoff turn > 3-2 = 1
    # Visible kv: causal AND (turn<0 OR turn==0 OR turn>1) → turns {-1, 0, 2, 3}
    # NOT turn 1 (indices 4, 5).
    q9 = vis[0, 9]
    assert q9[0:4].all()  # sys + turn 0
    assert (q9[4:6] == False).all()  # turn 1 dropped
    assert q9[6:10].all()  # turns 2, 3
    # Causal check: query at idx=4 (turn 1) cannot see idx=5..9.
    q4 = vis[0, 4]
    assert q4[5:].any() == False


def test_dense_visibility_K_geq_total_no_truncation():
    turn_ids = torch.tensor([[-1, 0, 1, 2, 3]])
    vis = kwindow_visibility_dense(turn_ids, past_k_steps=10)
    # Should equal pure causal mask
    L = 5
    expected_causal = torch.tril(torch.ones(L, L, dtype=torch.bool))
    assert torch.equal(vis[0], expected_causal)


def test_dense_visibility_anchor_always_visible():
    # Query at very last turn always sees turn 0 (anchor).
    turn_ids = torch.tensor([[-1, 0, 0, 1, 2, 3, 4, 5]])
    vis = kwindow_visibility_dense(turn_ids, past_k_steps=1)
    # K=1, query at idx 7 (turn 5): window = {turn > 4} = {5} only.
    # Plus anchor (turn 0) and system (-1).
    q_last = vis[0, 7]
    assert q_last[0]   # system
    assert q_last[1] and q_last[2]  # anchor turn 0
    assert not q_last[3]  # turn 1 dropped
    assert not q_last[4]  # turn 2 dropped
    assert not q_last[5]  # turn 3 dropped
    assert not q_last[6]  # turn 4 dropped
    assert q_last[7]   # turn 5 visible


def test_error_turn_visibility_via_anchor_when_in_window():
    # Error turn shares parent obs's turn_id.
    # turn_ids: sys, T0, T0_err, T0_err, T1, T1, T2, T2  (T0 has 2 spans)
    turn_ids = torch.tensor([[-1, 0, 0, 0, 1, 1, 2, 2]])
    vis = kwindow_visibility_dense(turn_ids, past_k_steps=2)
    # Query at last token (turn 2): window = {turn > 0} = {1, 2}, plus anchor 0.
    # All tokens visible (T0 via anchor, T1 via window, T2 via window).
    q_last = vis[0, 7]
    assert q_last.all()


if __name__ == "__main__":
    test_dense_visibility_basic()
    test_dense_visibility_K_geq_total_no_truncation()
    test_dense_visibility_anchor_always_visible()
    test_error_turn_visibility_via_anchor_when_in_window()
    print("OK: 4/4 tests passed")
