"""Long-trajectory proof of context windowing + packed-position correctness.

Builds a synthetic 15-step trajectory packed into one sequence, then asserts
(and prints visual evidence for):

  1. Rollout K-window filter at each step S keeps exactly turns
     {0} ∪ {S-K+1, ..., S}, with positions equal to those tokens'
     positions in the full unfiltered packed sequence (no shift).

  2. Training BlockMask: query at turn t can attend to system + turn 0 +
     turns {t-K+1, ..., t}, and ONLY those.

  3. Packed positions are monotone non-decreasing over the full sequence,
     and within each turn span.

Outputs an ASCII visualization of the dense visibility matrix so a human can
eyeball the K-window pattern. Run via ``python tests/test_long_trajectory_proof.py``.
"""

from __future__ import annotations

import torch

from verl.workers.rollout._packed_kwindow_mask import kwindow_visibility_dense
from verl.workers.rollout._packed_turn_metadata import (
    filter_kwindow,
    kwindow_visible_mask,
)


# ── Synthetic trajectory builder ───────────────────────────────────────────

NUM_STEPS = 15  # turn IDs 0..14 = 15 obs/asst steps
TOK_PER_TURN = 6  # arbitrary
SYS_TOKENS = 3  # turn_id = -1
PAST_K = 4


def build_synthetic_trajectory():
    """Pack a 15-step trajectory: [sys][turn 0][turn 1]...[turn 14].

    Each turn block is TOK_PER_TURN tokens. Token IDs are arbitrary integers
    (not real model tokens). Returns (input_ids, full_positions, turn_id).
    """
    chunks = []
    turn_chunks = []
    chunks.append(torch.full((SYS_TOKENS,), 1000, dtype=torch.long))
    turn_chunks.append(torch.full((SYS_TOKENS,), -1, dtype=torch.long))
    for t in range(NUM_STEPS):
        chunks.append(torch.arange(t * 100, t * 100 + TOK_PER_TURN, dtype=torch.long))
        turn_chunks.append(torch.full((TOK_PER_TURN,), t, dtype=torch.long))
    input_ids = torch.cat(chunks)
    turn_id = torch.cat(turn_chunks)
    L = input_ids.shape[0]
    # Packed positions: token's rank in the FULL trajectory, replicated 3x for MROPE.
    full_positions = torch.arange(L, dtype=torch.long).unsqueeze(0).expand(3, L).contiguous()
    return input_ids, full_positions, turn_id


# ── Tests ──────────────────────────────────────────────────────────────────

def test_rollout_kwindow_per_step():
    """At rollout step S, visible turns must be exactly {0} ∪ {S-K+1..S}."""
    input_ids, full_positions, turn_id = build_synthetic_trajectory()
    K = PAST_K
    print(f"\n{'='*72}\n[GUARANTEE 1] Rollout K-window at each step (K={K})\n{'='*72}")
    print(f"  Trajectory: {NUM_STEPS} turns, {TOK_PER_TURN} tokens/turn, "
          f"{SYS_TOKENS} system tokens. Total L = {input_ids.shape[0]}\n")

    for S in range(NUM_STEPS):
        # At rollout step S, only the prefix [system, turn 0..S] exists.
        prefix_mask_indices = turn_id <= S
        prefix_input_ids = input_ids[prefix_mask_indices]
        prefix_positions = full_positions[:, prefix_mask_indices]
        prefix_turn_id = turn_id[prefix_mask_indices]
        kept_ids, kept_pos, delta = filter_kwindow(
            prefix_input_ids, prefix_positions, prefix_turn_id,
            past_k_steps=K, current_turn_id=S,
        )
        mask = kwindow_visible_mask(prefix_turn_id, past_k_steps=K, current_turn_id=S)
        visible_turns = sorted(set(prefix_turn_id[mask].tolist()))
        # Expected: -1 (system) plus 0 plus turns {max(1,S-K+1)..S}.
        # Anchor turn 0 is always kept; system always kept.
        if S < K:
            expected = [-1] + list(range(0, S + 1))
        else:
            expected = [-1, 0] + list(range(S - K + 1, S + 1))
        assert visible_turns == expected, (
            f"step {S}: got visible turns {visible_turns}, expected {expected}"
        )
        # Position parity: kept_pos[0] must equal full_positions[0] for visible
        # tokens — so each kept token retains its position in the un-truncated
        # packed sequence (RoPE parity with training).
        full_kept = full_positions[0][prefix_mask_indices][mask]
        assert torch.equal(kept_pos[0], full_kept), (
            f"step {S}: kept positions != full positions for kept tokens"
        )
        # Position monotonicity within kept sequence.
        diffs = kept_pos[0, 1:] - kept_pos[0, :-1]
        assert (diffs >= 0).all(), f"step {S}: positions not monotone"
        # Print the evidence row.
        gaps = (diffs > 1).sum().item()
        print(
            f"  step S={S:2d}: visible turns={visible_turns!s:30s} "
            f"L_kept={kept_ids.shape[0]:3d} pos_max={int(kept_pos[0].max()):3d} "
            f"position_gaps={gaps:2d} delta={delta}"
        )
    print("\n  ✓ All 15 steps match the expected K-window. ✓ Positions match unfiltered packed.")


def test_training_blockmask_kwindow():
    """BlockMask: each query at turn t sees ONLY system + turn 0 + turns {t-K+1..t}."""
    _, _, turn_id = build_synthetic_trajectory()
    turn_ids_b = turn_id.unsqueeze(0)  # (1, L)
    K = PAST_K
    print(f"\n{'='*72}\n[GUARANTEE 2] Training BlockMask visibility (K={K})\n{'='*72}")

    vis = kwindow_visibility_dense(turn_ids_b, past_k_steps=K)[0]  # (L, L) bool
    L = vis.shape[0]

    # For each turn t, pick a query position in that turn and check visibility.
    for t in range(NUM_STEPS):
        # Pick last token of turn t (position-wise).
        q_idx = (turn_id == t).nonzero(as_tuple=True)[0][-1].item()
        kv_visible_turns = sorted(set(turn_id[vis[q_idx]].tolist()))
        if t < K:
            expected = [-1] + list(range(0, t + 1))
        else:
            expected = [-1, 0] + list(range(t - K + 1, t + 1))
        assert kv_visible_turns == expected, (
            f"query at turn {t} (idx {q_idx}): saw turns {kv_visible_turns}, "
            f"expected {expected}"
        )
        # Causal: visible kv idx must be <= q_idx.
        assert vis[q_idx, q_idx + 1 :].any() == False, (
            f"query at turn {t}: violated causal mask"
        )

    # Visual ASCII proof. Sample every other column for compactness.
    print(f"\n  ASCII visibility (rows=query position, cols=kv position; "
          f"'#'=visible, '.'=masked):\n")
    sample_q = list(range(0, L, max(1, L // 24)))
    sample_kv = list(range(0, L, max(1, L // 60)))
    # Header turn ids (reduced)
    hdr = "      kv->  " + "".join(
        ("|" if turn_id[k].item() != turn_id[max(0, k - 1)].item() else " ")
        for k in sample_kv
    )
    print(hdr)
    hdr2 = "             " + "".join(
        (str(turn_id[k].item() % 10) if turn_id[k].item() >= 0 else "S")
        for k in sample_kv
    )
    print(hdr2)
    for q in sample_q:
        row = "".join("#" if vis[q, k].item() else "." for k in sample_kv)
        print(f"  q={q:3d} t={int(turn_id[q].item()):2d}  {row}")
    print(f"\n  ✓ Each query sees ONLY {{system, turn 0, turns t-K+1..t}}. ✓ Causal preserved.")


def test_packed_positions_monotone_full_sequence():
    """Across the FULL packed trajectory, positions are strictly monotone."""
    _, full_positions, turn_id = build_synthetic_trajectory()
    print(f"\n{'='*72}\n[GUARANTEE 3] Packed positions monotone over full trajectory\n{'='*72}")
    pos = full_positions[0]  # MROPE channel 0
    # Within each turn, positions must be strictly increasing by 1.
    for t in range(-1, NUM_STEPS):
        idx = (turn_id == t).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        p = pos[idx]
        diffs = p[1:] - p[:-1]
        assert (diffs == 1).all(), f"turn {t}: positions not strictly increasing by 1"
    # Full sequence: positions 0..L-1.
    assert torch.equal(pos, torch.arange(pos.shape[0])), "full positions != arange(L)"
    print(
        f"  Full L={pos.shape[0]}, position range [0, {int(pos.max())}], "
        f"every turn span has stride-1 positions. ✓"
    )
    # Print position range per turn — quick visual check.
    print("  Per-turn position ranges:")
    for t in range(-1, NUM_STEPS):
        idx = (turn_id == t).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        print(f"    turn {t:2d}: pos [{int(pos[idx].min()):3d}, {int(pos[idx].max()):3d}] "
              f"({idx.numel()} tokens)")


def test_train_rollout_position_parity():
    """For every kept token at every rollout step, its position == full-packed position."""
    input_ids, full_positions, turn_id = build_synthetic_trajectory()
    K = PAST_K
    print(f"\n{'='*72}\n[PARITY] Rollout positions == training packed positions\n{'='*72}")

    mismatches = 0
    for S in range(NUM_STEPS):
        prefix_mask_indices = turn_id <= S
        prefix_input_ids = input_ids[prefix_mask_indices]
        prefix_positions = full_positions[:, prefix_mask_indices]
        prefix_turn_id = turn_id[prefix_mask_indices]
        kept_ids, kept_pos, _ = filter_kwindow(
            prefix_input_ids, prefix_positions, prefix_turn_id,
            past_k_steps=K, current_turn_id=S,
        )
        mask = kwindow_visible_mask(prefix_turn_id, past_k_steps=K, current_turn_id=S)
        full_kept_pos = full_positions[:, prefix_mask_indices][:, mask]
        if not torch.equal(kept_pos, full_kept_pos):
            mismatches += 1
    assert mismatches == 0
    print(f"  ✓ Rollout-side `filter_kwindow` returns identical positions to full packed,")
    print(f"    for all {NUM_STEPS} rollout steps. RoPE parity guaranteed for kept tokens.")


if __name__ == "__main__":
    test_rollout_kwindow_per_step()
    test_training_blockmask_kwindow()
    test_packed_positions_monotone_full_sequence()
    test_train_rollout_position_parity()
    print(f"\n{'='*72}\nAll long-trajectory proofs passed.\n{'='*72}")
