"""Human-readable POPE-DAgger rollout + advantage visualizers (log-only).

Two views, both gated behind ``POPE_DAGGER_VIZ=1`` (off by default so production
runs aren't spammed). Everything is wrapped in try/except — a viz error must
never break training.

View 1 — rollout step-view (``log_rollout_stepview``): per group / per
trajectory / per step, an abbreviated ``<think>head … tail</think>`` plus the
action tag, tagged with provenance:

    STUDENT  on-policy vLLM sample          → policy gradient × advantage
    REPLAY   byte-for-byte parent prefix    → masked from PG, NO SFT
    EXPERT   fresh oracle (teacher) step    → masked from PG, SFT cross-entropy

View 2 — advantage step-view (``log_advantage_stepview``): per group / per
trajectory / per step, the advantage applied and the token-level accounting
*after* advantage computation — how many tokens actually receive the policy
gradient push (``pg``), how many are teacher-masked out of PG (``tchM``), and
how many are SFT targets (``sft``). Confirms by eye that EXPERT steps push 0 PG
tokens but N SFT tokens, REPLAY steps push 0 of either, STUDENT steps push the
group-normalized advantage over all their tokens, and that the GRPO advantage is
constant within a trajectory.
"""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger("verl.pope_dagger_viz")

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
# Match either the paired form <explore>…</explore> / <answer>…</answer> (capturing
# the full inner content + close) or the self-closing form <explore …/> / <answer …/>.
_ACTION_RE = re.compile(
    r"<(answer|explore)\b[^>]*?>.*?</\1>"   # paired
    r"|<(?:answer|explore)\b[^>]*?/>",       # self-closing
    re.DOTALL,
)


def viz_enabled() -> bool:
    return os.environ.get("POPE_DAGGER_VIZ") == "1"


def _max_groups() -> int:
    try:
        return int(os.environ.get("POPE_DAGGER_VIZ_MAX_GROUPS", "4"))
    except Exception:
        return 4


def _abbrev_think(response: str, head: int = 80, tail: int = 48) -> str:
    """Pull the inner <think> content, collapse whitespace, and show
    ``head … tail`` when long (so the start/end are both visible)."""
    m = _THINK_RE.search(response or "")
    inner = (m.group(1) if m else "").strip()
    inner = re.sub(r"\s+", " ", inner)
    if not inner:
        return "(no <think>)"
    if len(inner) <= head + tail + 5:
        return inner
    return f"{inner[:head]} … {inner[-tail:]}"


def _action_tag(response: str) -> str:
    """Extract the trailing action tag (after </think>, ignoring <summary>)."""
    tail = response or ""
    m = _THINK_RE.search(tail)
    if m:
        tail = tail[m.end():]
    tail = re.sub(r"<summary>.*?</summary>", "", tail, flags=re.DOTALL).strip()
    am = _ACTION_RE.search(tail)
    if am:
        return re.sub(r"\s+", " ", am.group(0)).strip()
    tail = re.sub(r"\s+", " ", tail).strip()
    return tail[:80] if tail else "(no action)"


def _provenance(teacher_forced: bool, is_expert: bool) -> str:
    if is_expert:
        return "EXPERT "
    if teacher_forced:
        return "REPLAY "
    return "STUDENT"


def _group_rows_by_uid(uids):
    """Yield (uid, [row indices]) preserving first-seen order."""
    order: list = []
    buckets: dict = {}
    for i, u in enumerate(uids):
        key = str(u)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(i)
    for key in order:
        yield key, buckets[key]


# ─────────────────────────────── View 1 ────────────────────────────────────


def log_rollout_stepview(trajectories, global_step=None) -> None:
    if not viz_enabled() or not trajectories:
        return
    try:
        # Group by group_id, preserving order.
        order: list = []
        groups: dict = {}
        for t in trajectories:
            g = getattr(t, "group_id", -1)
            if g not in groups:
                groups[g] = []
                order.append(g)
            groups[g].append(t)

        head = ["", "═" * 88, f"POPE-DAGGER ROLLOUT STEP-VIEW (train step {global_step})", "═" * 88]
        lines: list[str] = list(head)
        shown = 0
        for g in order:
            if shown >= _max_groups():
                lines.append(f"… ({len(order) - shown} more groups suppressed; raise POPE_DAGGER_VIZ_MAX_GROUPS)")
                break
            shown += 1
            lines.append(f"group {g}")
            for t in groups[g]:
                resp = t.step_responses or []
                forced = t.step_teacher_forced or []
                expert = t.step_is_expert or []
                chain = getattr(t, "branch_chain", []) or []
                tag = (
                    f"  n{getattr(t, 'n_idx', '?')}  iter{getattr(t, 'dagger_iter_index', 0)}"
                    f" chain{getattr(t, 'chain_id', 0)}  reward={float(getattr(t, 'reward', 0.0) or 0.0):+.3f}"
                    f"  steps={len(resp)}"
                )
                if chain:
                    tag += f"  branch_chain={chain}"
                lines.append(tag)
                for i, r in enumerate(resp):
                    prov = _provenance(
                        bool(forced[i]) if i < len(forced) else False,
                        bool(expert[i]) if i < len(expert) else False,
                    )
                    lines.append(
                        f"    s{i} [{prov}] <think>{_abbrev_think(r)}</think>  {_action_tag(r)}"
                    )
        lines.append("═" * 88)
        logger.info("\n".join(lines))
    except Exception as e:  # never break training on a viz error
        logger.warning(f"[pope_dagger_viz] rollout step-view failed: {e!r}")


# ─────────────────────────────── View 2 ────────────────────────────────────


def log_advantage_stepview(batch, global_step=None) -> None:
    if not viz_enabled():
        return
    try:
        b = batch.batch
        adv = b["advantages"]               # (bs, resp_len)
        resp_mask = b["response_mask"]       # (bs, resp_len)
        tch_mask = b.get("teacher_token_mask")
        sft_mask = b.get("sft_token_mask")
        nt = batch.non_tensor_batch
        spans = nt.get("step_token_spans")
        uids = nt.get("uid")
        d_iter = nt.get("dagger_iter_index")
        chain_id = nt.get("chain_id")
        scores = b.get("token_level_scores")
        if adv is None or resp_mask is None or spans is None or uids is None:
            logger.info("[pope_dagger_viz] advantage step-view: required keys missing; skipping")
            return

        def _row_reward(i):
            if scores is None:
                return 0.0
            return float(scores[i].sum().item())

        lines = ["", "═" * 88, f"POPE-DAGGER ADVANTAGE STEP-VIEW (train step {global_step})", "═" * 88]
        shown = 0
        for uid, rows in _group_rows_by_uid(uids):
            if shown >= _max_groups():
                lines.append("… more groups suppressed (POPE_DAGGER_VIZ_MAX_GROUPS)")
                break
            shown += 1
            rewards = [round(_row_reward(i), 3) for i in rows]
            lines.append(f"group uid={uid[:8]}  rewards={rewards}")
            for i in rows:
                row_adv = adv[i]
                row_rm = resp_mask[i]
                row_spans = spans[i] if i < len(spans) else []
                tot_pg = tot_sft = tot_tch = 0
                step_lines = []
                for s_idx, (st, en) in enumerate(row_spans):
                    en = min(en, row_adv.shape[0] - 1)
                    if en < st:
                        continue
                    sl = slice(st, en + 1)
                    rm = row_rm[sl]
                    active = rm == 1
                    n_tok = int(active.sum().item())
                    tchm = tch_mask[i][sl] if tch_mask is not None else None
                    sftm = sft_mask[i][sl] if sft_mask is not None else None
                    n_tch = int(((tchm == 1) & active).sum().item()) if tchm is not None else 0
                    n_sft = int(((sftm == 1) & active).sum().item()) if sftm is not None else 0
                    n_pg = n_tok - n_tch  # PG applies where active AND not teacher-masked
                    a_vals = row_adv[sl][active]
                    a_mean = float(a_vals.mean().item()) if n_tok else 0.0
                    prov = _provenance(n_tch > 0, n_sft > 0)
                    tot_pg += n_pg
                    tot_sft += n_sft
                    tot_tch += n_tch
                    step_lines.append(
                        f"    s{s_idx} [{prov}] adv={a_mean:+.3f}  tok={n_tok:<3d}"
                        f" pg={n_pg:<3d} tchM={n_tch:<3d} sft={n_sft:<3d}"
                    )
                header = (
                    f"  n{rows.index(i)}  iter{int(d_iter[i]) if d_iter is not None else 0}"
                    f" chain{int(chain_id[i]) if chain_id is not None else 0}"
                    f"  reward={_row_reward(i):+.3f}  steps={len(row_spans)}"
                )
                lines.append(header)
                lines.extend(step_lines)
                lines.append(f"      ↳ totals: pg={tot_pg}  sft={tot_sft}  teacher_masked={tot_tch}")
        lines.append("═" * 88)
        logger.info("\n".join(lines))
    except Exception as e:
        logger.warning(f"[pope_dagger_viz] advantage step-view failed: {e!r}")
