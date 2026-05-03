"""Unit tests for per-token turn metadata.

These tests use a small synthetic vocabulary instead of a real tokenizer so
they run without HF dependencies. The wire-format covered by the real
``apply_chat_template`` is faithfully simulated by interleaving two sentinel
IDs (im_start, im_end) around each message's content.
"""

import torch

from verl.workers.rollout._packed_turn_metadata import (
    KIND_ASSISTANT,
    KIND_BOUNDARY,
    KIND_SYSTEM,
    KIND_USER_ERROR,
    KIND_USER_OBS,
    assign_message_turn_ids,
    assign_turn_metadata,
    filter_kwindow,
    kwindow_visible_mask,
    message_turn_infos,
)


IM_START = 1
IM_END = 2


def _fake_tokenize(messages):
    """Mimic apply_chat_template token layout for the helper tests.

    Returns a flat list of token IDs that mark each message as
    ``[IM_START, role_tok, content_tok, IM_END, NEWLINE]``. Specific IDs
    don't matter — only IM_START/IM_END are introspected by the helper.
    """
    tokens = []
    for msg in messages:
        tokens.extend([IM_START, 100, 200, IM_END, 300])
    return tokens


def test_message_turn_infos_marks_error_only_when_no_image_after_assistant():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task <image>"},
        {"role": "assistant", "content": "<think>...</think><explore/>"},
        {"role": "user", "content": "Step 1.\n<image>"},
        {"role": "assistant", "content": "<think>...</think><explore/>"},
        {"role": "user", "content": "Action execution failed: bumped wall"},
    ]
    infos = message_turn_infos(msgs)
    assert [i.role for i in infos] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert [i.is_error for i in infos] == [False, False, False, False, False, True]


def test_assign_message_turn_ids_basic():
    # system, user(0), asst(0_resp=for prev=none→0), user(1), asst(1), user(2)
    infos = message_turn_infos([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "<image>"},
        {"role": "assistant", "content": "r0"},
        {"role": "user", "content": "<image>"},
        {"role": "assistant", "content": "r1"},
        {"role": "user", "content": "<image>"},
    ])
    tids = assign_message_turn_ids(infos)
    assert tids == [-1, 0, 0, 1, 1, 2]


def test_assign_message_turn_ids_with_error_turn():
    infos = message_turn_infos([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "<image>"},  # obs 0
        {"role": "assistant", "content": "r0"},  # response→0
        {"role": "user", "content": "Action failed"},  # error → shares 0
        {"role": "assistant", "content": "r0_retry"},  # response→0
        {"role": "user", "content": "<image>"},  # obs 1
    ])
    tids = assign_message_turn_ids(infos)
    assert tids == [-1, 0, 0, 0, 0, 1]


def test_assign_turn_metadata_token_spans():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "<image>"},
        {"role": "assistant", "content": "r0"},
        {"role": "user", "content": "<image>"},
    ]
    tokens = _fake_tokenize(msgs)
    turn_id, kind = assign_turn_metadata(tokens, msgs, IM_START, IM_END)
    assert turn_id.shape == (len(tokens),)
    assert kind.shape == (len(tokens),)

    # Each block is 5 tokens [IM_START, role, content, IM_END, NEWLINE]
    # Block 0 (system): tokens 0..3 turn_id=-1, kind=SYSTEM. Token 4 = boundary.
    assert turn_id[0:4].tolist() == [-1, -1, -1, -1]
    assert (kind[0:4] == KIND_SYSTEM).all()
    assert kind[4] == KIND_BOUNDARY

    # Block 1 (user obs 0): tokens 5..8
    assert (turn_id[5:9] == 0).all()
    assert (kind[5:9] == KIND_USER_OBS).all()

    # Block 2 (assistant): tokens 10..13, turn_id=0
    assert (turn_id[10:14] == 0).all()
    assert (kind[10:14] == KIND_ASSISTANT).all()

    # Block 3 (user obs 1): tokens 15..18, turn_id=1
    assert (turn_id[15:19] == 1).all()
    assert (kind[15:19] == KIND_USER_OBS).all()


def test_kwindow_visible_mask_basic():
    # 4 turns: 0, 1, 2, 3 plus -1 system tokens.
    turn_id = torch.tensor([-1, -1, 0, 0, 1, 1, 2, 2, 3, 3])
    mask = kwindow_visible_mask(turn_id, past_k_steps=2, current_turn_id=3)
    # Visible: -1, 0 (anchor), and turns >= 3-2+1 = 2 → {2, 3}
    expected = torch.tensor([True, True, True, True, False, False, True, True, True, True])
    assert torch.equal(mask, expected)


def test_kwindow_visible_mask_K_geq_total_keeps_all():
    turn_id = torch.tensor([-1, 0, 1, 2])
    mask = kwindow_visible_mask(turn_id, past_k_steps=10, current_turn_id=2)
    assert mask.all()


def test_kwindow_error_turn_visibility_follows_parent():
    # Error turn shares parent obs's turn_id; if parent is in window, error is too.
    turn_id = torch.tensor([-1, 0, 0, 0, 1, 1, 2, 2])  # 2nd block of 0s = error+retry
    # Current turn 2, K=2 → window cutoff = 1 → turns {0(anchor), 1, 2} visible.
    # All these are visible regardless because of anchor rule.
    mask = kwindow_visible_mask(turn_id, past_k_steps=2, current_turn_id=2)
    assert mask.all()


def test_filter_kwindow_preserves_full_positions():
    # Synthetic: 8 tokens across 4 turns (2 each).
    input_ids = torch.tensor([10, 11, 20, 21, 30, 31, 40, 41])
    turn_id = torch.tensor([-1, 0, 1, 1, 2, 2, 3, 3])
    full_positions = torch.arange(8).unsqueeze(0).expand(3, -1).clone()

    kept_ids, kept_pos, delta = filter_kwindow(
        input_ids, full_positions, turn_id, past_k_steps=2, current_turn_id=3
    )
    # Visible: -1 (sys), turn 0 (anchor), turns >= 3-2+1=2 → turns {2, 3}.
    # Indices: 0, 1, 4, 5, 6, 7
    assert kept_ids.tolist() == [10, 11, 30, 31, 40, 41]
    # Positions retain their original (non-contiguous) values.
    assert kept_pos[0].tolist() == [0, 1, 4, 5, 6, 7]
    # delta = max_pos+1 - L_kept = 7+1 - 6 = 2
    assert delta == 2


def test_filter_kwindow_K_geq_total_no_gaps():
    input_ids = torch.tensor([1, 2, 3, 4])
    turn_id = torch.tensor([-1, 0, 1, 2])
    full_positions = torch.arange(4).unsqueeze(0).expand(3, -1).clone()
    kept_ids, kept_pos, delta = filter_kwindow(
        input_ids, full_positions, turn_id, past_k_steps=10, current_turn_id=2
    )
    assert kept_ids.tolist() == [1, 2, 3, 4]
    assert delta == 0  # contiguous, no gaps


if __name__ == "__main__":
    test_message_turn_infos_marks_error_only_when_no_image_after_assistant()
    test_assign_message_turn_ids_basic()
    test_assign_message_turn_ids_with_error_turn()
    test_assign_turn_metadata_token_spans()
    test_kwindow_visible_mask_basic()
    test_kwindow_visible_mask_K_geq_total_keeps_all()
    test_kwindow_error_turn_visibility_follows_parent()
    test_filter_kwindow_preserves_full_positions()
    test_filter_kwindow_K_geq_total_no_gaps()
    print("OK: 9/9 tests passed")
