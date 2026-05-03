"""K-window block mask for past-K packed multiturn training.

Produces a FlexAttention ``BlockMask`` whose ``mask_mod`` enforces:

    visible(q, kv) = causal(kv <= q)
                     AND ( kv.turn_id < 0           # system / boundary
                           OR kv.turn_id == 0       # always-kept anchor
                           OR kv.turn_id > q.turn_id - K )

so each query at turn ``T`` attends only to tokens in the K-window
``{system, turn 0, turn T-K+1, …, turn T}``. Identical visibility rule to
the rollout-side K-window filter, ensuring train↔rollout parity when paired
with packed MROPE positions.

The mask is built once per (seq_len, K) shape and cached so FlexAttention's
per-shape compile cost amortizes across the run.
"""

from __future__ import annotations

from functools import lru_cache

import torch


_BLOCK_MASK_CACHE: dict = {}


def _kwindow_mask_factory(turn_ids: torch.Tensor, past_k_steps: int):
    """Returns a ``mask_mod`` closure capturing ``turn_ids`` and ``K``.

    ``mask_mod(b, h, q_idx, kv_idx)`` is called by FlexAttention with batch,
    head, query, and kv indices and must return a bool tensor.
    """
    K = int(past_k_steps)

    def mask_mod(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        kv_turn = turn_ids[b, kv_idx]
        q_turn = turn_ids[b, q_idx]
        # turn_id sentinels: -2 = left-padding (NEVER visible),
        # -1 = boundary/system (always visible anchor),
        # 0 = always-kept anchor turn,
        # >0 = obs-step turns (gated by K-window).
        not_padding = kv_turn != -2
        anchor = (kv_turn == -1) | (kv_turn == 0)
        in_window = kv_turn > (q_turn - K)
        return causal & not_padding & (anchor | in_window)

    return mask_mod


def build_kwindow_blockmask(
    turn_ids: torch.Tensor,
    past_k_steps: int,
    block_size: int = 128,
):
    """Build a FlexAttention ``BlockMask`` for the K-window pattern.

    Args:
        turn_ids: (B, L) int64 tensor of per-token turn IDs (from
            ``assign_turn_metadata``). System / boundary tokens are -1; the
            always-kept anchor turn is 0.
        past_k_steps: K window size (in obs turns).
        block_size: FlexAttention block size for sparsity bookkeeping;
            128 is a sane default.

    Returns:
        ``BlockMask`` ready to pass to ``flex_attention(... block_mask=...)``.
    """
    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except ImportError as e:
        raise RuntimeError(
            "FlexAttention requires PyTorch >= 2.5 with torch.nn.attention.flex_attention"
        ) from e

    if turn_ids.dim() != 2:
        raise ValueError(f"turn_ids must be (B, L); got {tuple(turn_ids.shape)}")
    B, L = turn_ids.shape

    mask_mod = _kwindow_mask_factory(turn_ids, past_k_steps)
    return create_block_mask(
        mask_mod,
        B=B,
        H=None,
        Q_LEN=L,
        KV_LEN=L,
        device=turn_ids.device,
        BLOCK_SIZE=block_size,
    )


def kwindow_visibility_dense(
    turn_ids: torch.Tensor,
    past_k_steps: int,
) -> torch.Tensor:
    """Dense (B, L, L) bool visibility tensor — for tests and debug only.

    Avoid in production: O(B·L²) memory. The BlockMask path is what training
    uses.
    """
    if turn_ids.dim() != 2:
        raise ValueError(f"turn_ids must be (B, L); got {tuple(turn_ids.shape)}")
    B, L = turn_ids.shape
    K = int(past_k_steps)

    # (B, L, 1) query, (B, 1, L) kv — broadcast to (B, L, L)
    q = turn_ids.unsqueeze(-1)
    kv = turn_ids.unsqueeze(-2)
    idx = torch.arange(L, device=turn_ids.device)
    causal = idx.unsqueeze(-1) >= idx.unsqueeze(0)  # (L, L) — kv_idx <= q_idx
    causal = causal.unsqueeze(0).expand(B, -1, -1)
    not_padding = kv != -2
    anchor = (kv == -1) | (kv == 0)
    in_window = kv > (q - K)
    return causal & not_padding & (anchor | in_window)
