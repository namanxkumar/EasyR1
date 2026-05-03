"""Per-token turn metadata for past-K packed multiturn.

Given a tokenized chat conversation produced by the Qwen3-VL chat template
(``<|im_start|>{role}\\n...<|im_end|>\\n`` blocks), compute per-token
``turn_id`` and ``kind`` tensors that the K-window block mask + packed
position computation rely on.

Turn IDs follow the obs-step convention used by
``common.prompting.context_builders.build_multiturn_context``:

    - system message tokens         → turn_id = -1
    - assistant message at iter i   → turn_id = i - 1  (response to obs i-1)
    - user obs message (non-error)  → turn_id = i      (current obs)
    - error user message            → turn_id = i      (shares parent obs)
    - tokens between/outside spans  → turn_id = -1     (kind = "boundary")

``kind`` is one of ``"system"``, ``"user_obs"``, ``"user_error"``,
``"assistant"``, ``"boundary"`` (whitespace/newline tokens between
``<|im_end|>`` and the next ``<|im_start|>``, or trailing pad).

Image-pad tokens inside a user_obs span keep that span's turn_id, so
masking them out by turn_id automatically masks their image patches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


KIND_BOUNDARY = 0
KIND_SYSTEM = 1
KIND_USER_OBS = 2
KIND_USER_ERROR = 3
KIND_ASSISTANT = 4

KIND_NAMES = {
    KIND_BOUNDARY: "boundary",
    KIND_SYSTEM: "system",
    KIND_USER_OBS: "user_obs",
    KIND_USER_ERROR: "user_error",
    KIND_ASSISTANT: "assistant",
}


@dataclass
class MessageTurnInfo:
    """Per-message metadata fed into ``assign_turn_metadata``.

    ``role`` is "system" | "user" | "assistant".  ``is_error`` is true for
    user turns that don't carry a new observation (no image, no step header).
    """

    role: str
    is_error: bool = False


def message_turn_infos(messages: Sequence[dict]) -> list[MessageTurnInfo]:
    """Derive ``MessageTurnInfo`` from a ``build_multiturn_context`` message list.

    A user message is treated as an error turn iff it is preceded by an
    assistant message AND it carries no ``<image>`` placeholder. The first
    user turn is always a non-error obs turn (carries the task description
    and the initial observation).
    """
    infos: list[MessageTurnInfo] = []
    for i, msg in enumerate(messages):
        role = msg["role"]
        if role == "user":
            content = msg.get("content", "")
            text = content if isinstance(content, str) else _flatten_content(content)
            has_image = "<image>" in text or _content_has_image(content)
            is_error = (i > 0) and (not has_image) and any(
                m["role"] == "assistant" for m in messages[:i]
            )
            infos.append(MessageTurnInfo(role=role, is_error=is_error))
        else:
            infos.append(MessageTurnInfo(role=role, is_error=False))
    return infos


def _flatten_content(content) -> str:
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                parts.append(c.get("text", ""))
        return "\n".join(parts)
    return str(content)


def _content_has_image(content) -> bool:
    if isinstance(content, list):
        return any(isinstance(c, dict) and c.get("type") == "image" for c in content)
    return False


def assign_message_turn_ids(infos: Sequence[MessageTurnInfo]) -> list[int]:
    """Assign a turn_id to each message.

    Walks the message list applying ``build_multiturn_context``'s obs_step
    counter convention. Returns a list of integers, one per message.
    """
    turn_ids: list[int] = []
    obs_step = 0
    last_obs_step = -1  # turn_id of the most-recent user_obs turn

    for info in infos:
        if info.role == "system":
            turn_ids.append(-1)
        elif info.role == "assistant":
            # Assistant message is the response to the previous user_obs.
            turn_ids.append(last_obs_step if last_obs_step >= 0 else 0)
        elif info.role == "user":
            if info.is_error:
                # Error user turn — shares turn_id with parent obs turn.
                turn_ids.append(last_obs_step if last_obs_step >= 0 else 0)
            else:
                turn_ids.append(obs_step)
                last_obs_step = obs_step
                obs_step += 1
        else:
            turn_ids.append(-1)
    return turn_ids


def assign_turn_metadata(
    token_ids: Sequence[int],
    messages: Sequence[dict],
    im_start_id: int,
    im_end_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Walk the tokenized conversation and assign turn_id + kind per token.

    The chat template produces per-message blocks of the form::

        <|im_start|>{role}\\n{content}<|im_end|>\\n

    We scan ``token_ids`` for ``<|im_start|>`` markers and consume one
    block per ``<|im_start|>`` in order. Tokens before the first marker
    or between an ``<|im_end|>`` and the next ``<|im_start|>`` (typically
    just a newline) are tagged as boundary with turn_id = -1.

    Args:
        token_ids: 1-D sequence of token IDs (no padding).
        messages: the message list passed to ``apply_chat_template``.
        im_start_id, im_end_id: ID of the ``<|im_start|>`` / ``<|im_end|>``
            special tokens.

    Returns:
        turn_id: int64 tensor (L,)
        kind: int64 tensor (L,) of KIND_* values
    """
    L = len(token_ids)
    turn_id = torch.full((L,), -1, dtype=torch.int64)
    kind = torch.full((L,), KIND_BOUNDARY, dtype=torch.int64)

    infos = message_turn_infos(messages)
    msg_turn_ids = assign_message_turn_ids(infos)

    # Find <|im_start|> positions and pair with messages in order.
    starts = [i for i, t in enumerate(token_ids) if t == im_start_id]
    if len(starts) > len(infos):
        starts = starts[: len(infos)]

    for msg_idx, start in enumerate(starts):
        # End of this block: position of the next <|im_end|> (inclusive).
        end = start
        while end < L and token_ids[end] != im_end_id:
            end += 1
        # Include the <|im_end|> itself; trailing newline (if any) stays
        # in this block too — it belongs semantically to this turn.
        end_inclusive = min(end, L - 1)
        # Extend to consume a single trailing newline that the chat template
        # appends after <|im_end|>. We don't know the newline ID generically
        # without the tokenizer, so leave that token as boundary; the K-window
        # mask treats boundary tokens as always-visible system-like anchors,
        # which is fine for a 1-token slack.

        info = infos[msg_idx]
        tid = msg_turn_ids[msg_idx]
        if info.role == "system":
            k = KIND_SYSTEM
        elif info.role == "assistant":
            k = KIND_ASSISTANT
        elif info.is_error:
            k = KIND_USER_ERROR
        else:
            k = KIND_USER_OBS

        turn_id[start : end_inclusive + 1] = tid
        kind[start : end_inclusive + 1] = k

    return turn_id, kind


def filter_kwindow(
    input_ids: torch.Tensor,
    full_positions: torch.Tensor,
    turn_id: torch.Tensor,
    past_k_steps: int,
    current_turn_id: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Project the full packed sequence onto the K-window for rollout.

    Strict train↔rollout RoPE parity: every kept token retains the position
    it would have at training time (when the full sequence is packed
    contiguously). Dropped-middle turns leave **gaps** in the position
    space — that's the whole point.

    Args:
        input_ids: (L,) int64 — tokenized full trajectory.
        full_positions: (3, L) int64 — MROPE positions for the full sequence
            (e.g., from ``get_rope_index``).
        turn_id: (L,) int64 — per-token turn IDs (from ``assign_turn_metadata``).
        past_k_steps: K window size.
        current_turn_id: query turn (defaults to ``turn_id.max()``).

    Returns:
        kept_input_ids: (L_kept,) int64
        kept_positions: (3, L_kept) int64 — non-contiguous positions
        delta: ``kept_positions.max() + 1 - L_kept`` for the vLLM hook.
    """
    if full_positions.dim() != 2 or full_positions.shape[0] != 3:
        raise ValueError(
            f"full_positions must be (3, L); got {tuple(full_positions.shape)}"
        )
    if input_ids.shape[0] != full_positions.shape[1]:
        raise ValueError(
            f"input_ids length {input_ids.shape[0]} != full_positions L "
            f"{full_positions.shape[1]}"
        )

    mask = kwindow_visible_mask(turn_id, past_k_steps, current_turn_id)
    kept_input_ids = input_ids[mask]
    kept_positions = full_positions[:, mask]
    L_kept = int(kept_input_ids.shape[0])
    delta = int(kept_positions.max().item()) + 1 - L_kept
    return kept_input_ids, kept_positions, delta


def kwindow_visible_mask(
    turn_id: torch.Tensor,
    past_k_steps: int,
    current_turn_id: int | None = None,
) -> torch.Tensor:
    """Return a 1-D bool mask for which tokens are visible from the K-window.

    A token is visible iff:
        - turn_id == -1 (system / boundary), or
        - turn_id == 0  (always-kept anchor turn), or
        - turn_id > current_turn_id - past_k_steps

    When ``current_turn_id`` is None, uses ``turn_id.max()``.
    """
    if current_turn_id is None:
        current_turn_id = int(turn_id.max().item())
    cutoff = current_turn_id - past_k_steps + 1
    return (turn_id < 0) | (turn_id == 0) | (turn_id >= cutoff)
