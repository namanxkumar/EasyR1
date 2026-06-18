"""Lightweight, always-on POPE-DAgger observability dumps (JSONL).

Unlike ``pope_dagger_viz`` / ``pope_dagger_figs`` (gated behind
``POPE_DAGGER_VIZ`` and expensive — matplotlib figures, human-readable log
walls), these two dumps are cheap and structured so the prefix-testing
post-hoc questions are *recoverable* from any run:

1. **Per-trajectory outcomes** (``dump_trajectory_outcomes``, written from the
   rollout worker): one row per rolled-out trajectory with its group, dagger
   iteration (0 = on-policy baseline, ≥1 = guided continuation), reward,
   solved flag, and the parsed action tag at *each* step. Joined to the
   per-group ``guidance_text`` in ``prefixed_prompts_rank*.jsonl`` (same
   ``rollout_idx`` + ``group_id``), this answers: *did the student follow the
   guidance direction in subsequent on-policy steps, and did followers succeed
   more than the plain baselines?*

2. **Per-group advantages** (``dump_group_advantages``, written from the
   driver after advantage compute): one row per batch sequence with its group
   uid, dagger iteration, scalar reward, and the mean policy-gradient
   advantage over its PG-eligible (non-teacher-masked) tokens. This answers:
   *what advantages did the prefixed groups receive vs the baselines?*

Both are controlled by the same directory env var as the prefix dump
(``POPE_PREFIX_DUMP``); when unset, both are no-ops. Best-effort — diagnostics
must never raise into training.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

# Last action tag emitted in a step response: <explore .../>, <answer .../>,
# <stop/>, or the paired forms. Captures the whole (compacted) tag for offline
# direction / coordinate parsing.
_ACTION_RE = re.compile(
    r"<(explore|answer|stop)\b[^>]*?(?:/>|>.*?</\1>)",
    re.IGNORECASE | re.DOTALL,
)


def dump_dir() -> str:
    return os.environ.get("POPE_PREFIX_DUMP", "").strip()


def _parse_action(resp: str | None) -> str:
    """Extract the trailing action tag from a step response (compacted)."""
    if not resp:
        return ""
    matches = _ACTION_RE.findall(resp)
    # findall with a group returns the group, not the full match; re-search the
    # full span instead so we keep parameters (direction/coords).
    spans = list(_ACTION_RE.finditer(resp))
    if not spans:
        return ""
    return re.sub(r"\s+", " ", spans[-1].group(0)).strip()[:160]


def dump_trajectory_outcomes(
    trajectories: list[Any],
    *,
    rollout_idx: int,
    rank: int = 0,
) -> None:
    """Append one JSONL row per trajectory (from the rollout worker).

    Must be called BEFORE the finalize cleanup clears ``step_responses``.
    """
    d = dump_dir()
    if not d or not trajectories:
        return
    try:
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"traj_outcomes_rank{rank}.jsonl")
        with open(path, "a") as f:
            for t in trajectories:
                reward = float(t.reward) if t.reward is not None else None
                rec = {
                    "rollout_idx": int(rollout_idx),
                    "group_id": int(getattr(t, "group_id", -1)),
                    "n_idx": int(getattr(t, "n_idx", 0)),
                    "dagger_iter_index": int(getattr(t, "dagger_iter_index", 0)),
                    "chain_id": int(getattr(t, "chain_id", 0)),
                    "branch_chain": list(getattr(t, "branch_chain", []) or []),
                    "episode_id": getattr(t, "episode_id", None),
                    "reward": reward,
                    "num_steps": int(getattr(t, "num_steps", 0) or 0),
                    "solved": (reward is not None and reward >= 1.0),
                    "step_actions": [
                        _parse_action(r) for r in (getattr(t, "step_responses", None) or [])
                    ],
                    "step_teacher_forced": [
                        bool(x) for x in (getattr(t, "step_teacher_forced", None) or [])
                    ],
                    "step_is_expert": [
                        bool(x) for x in (getattr(t, "step_is_expert", None) or [])
                    ],
                }
                f.write(json.dumps(rec) + "\n")
    except Exception:  # pragma: no cover — diagnostics must never crash
        pass


def dump_group_advantages(batch: Any, global_step: int) -> None:
    """Append one JSONL row per batch sequence (from the driver, post-advantage).

    Records the PG-eligible mean advantage and scalar reward per sequence,
    tagged by group uid + dagger iteration so prefixed vs baseline groups are
    separable offline.
    """
    d = dump_dir()
    if not d:
        return
    try:
        b = batch.batch
        adv = b.get("advantages")             # (bs, resp_len)
        resp_mask = b.get("response_mask")    # (bs, resp_len)
        if adv is None or resp_mask is None:
            return
        tch_mask = b.get("teacher_token_mask")
        scores = b.get("token_level_scores")  # (bs, resp_len)
        nt = batch.non_tensor_batch
        uids = nt.get("uid")
        d_iter = nt.get("dagger_iter_index")
        chain_id = nt.get("chain_id")
        if uids is None:
            return

        bs = adv.shape[0]
        # PG-eligible mask = responded tokens that are NOT teacher-forced.
        if tch_mask is not None:
            pg_mask = resp_mask * (1.0 - tch_mask)
        else:
            pg_mask = resp_mask

        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "group_advantages.jsonl")
        with open(path, "a") as f:
            for i in range(bs):
                pm = pg_mask[i]
                n_pg = float(pm.sum().item())
                adv_pg_mean = (
                    float((adv[i] * pm).sum().item() / n_pg) if n_pg > 0 else 0.0
                )
                n_tch = (
                    float((resp_mask[i] * tch_mask[i]).sum().item())
                    if tch_mask is not None
                    else 0.0
                )
                reward = (
                    float((scores[i] * resp_mask[i]).sum().item())
                    if scores is not None
                    else None
                )
                rec = {
                    "global_step": int(global_step),
                    "uid": str(uids[i]),
                    "dagger_iter_index": int(d_iter[i]) if d_iter is not None else 0,
                    "chain_id": int(chain_id[i]) if chain_id is not None else 0,
                    "reward": reward,
                    "adv_pg_mean": adv_pg_mean,
                    "n_pg_tokens": n_pg,
                    "n_teacher_tokens": n_tch,
                }
                f.write(json.dumps(rec) + "\n")
    except Exception:  # pragma: no cover — diagnostics must never crash
        pass
