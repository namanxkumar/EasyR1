"""Saved-figure POPE-DAgger per-step diagnostic dashboards.

Companion to ``pope_dagger_viz.py`` (which is log-only). This module renders
PNG figures at the end of each training step — *after* advantage computation,
when advantages / rewards / teacher+sft masks / step spans are all in scope on
the driver — and writes them under ``<save_checkpoint_path>/pope_viz/``.

Gated by :func:`pope_dagger_viz.figs_enabled` (``POPE_DAGGER_VIZ_FIGS``, default
follows ``POPE_DAGGER_VIZ``). Everything is wrapped in try/except — a viz error
must never break training.

Figures emitted per step into ``pope_viz/step_<NNN>/``:

  1. ``group_advantage_dist.png`` — per-group advantage distribution (every
     trajectory's GRPO advantage, coloured student / guided / fully-forced).
     Answers "the advantage distribution within each group".
  2. ``step_signal.png`` — advantage *distributed per trajectory step*, as two
     aligned heatmaps (advantage value + token provenance) so the masked
     prefixes (REPLAY) and the SFT-only EXPERT steps are visible against where
     the policy-gradient signal actually lands. Plus a per-step-position
     aggregate of pg / teacher-masked / sft token mass.
  3. ``group_signal_dist.png`` — per-group distributions of {advantage,
     gradient reach n_pg, effective PG signal adv*n_pg}.
  4. ``dashboard.png`` — the "why is / isn't it learning" overview, with panels
     tied to the known POPE-DAgger failure mechanisms (guidance landing on
     already-solved groups; baseline inflation flipping the net PG advantage
     negative; the SFT/teacher token budget; dead-group rate).

Run-level accumulation:
  * ``pope_viz/trends.jsonl`` — one JSON line of scalar diagnostics per step.
  * ``pope_viz/trends.png``   — regenerated each step from trends.jsonl.
  * ``pope_viz/step_<NNN>/stats.json`` — full per-step scalars + per-group arrays.
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger("verl.pope_dagger_figs")

_EPS = 1e-6
# Success threshold for the end-of-trajectory markers. Mirrors
# guided_multiturn_env._is_solved (reward >= 1.0 == solved). Under
# stop_on_solved=True only NON-solved trajectories are branched from, so a
# green (solved) marker should never have an outgoing branch arrow — that
# makes the branch invariant visually checkable. NB the objectnav
# bimodal reward is NOT binary (0.875*success + 0.125*fmt), so reward>0 does
# NOT mean success — a format-only failure scores 0.125.
_SOLVED_REWARD = 1.0


def _to_np(t):
    import numpy as np

    if t is None:
        return None
    try:
        return t.detach().cpu().float().numpy()
    except Exception:
        try:
            return np.asarray(t)
        except Exception:
            return None


def _extract_records(batch):
    """Per-trajectory records from a post-advantage DataProto. Returns
    ``(records, meta)`` or ``(None, None)`` if required keys are absent."""
    import numpy as np

    b = batch.batch
    adv = _to_np(b.get("advantages"))
    resp_mask = _to_np(b.get("response_mask"))
    if adv is None or resp_mask is None:
        return None, None
    tch_mask = _to_np(b.get("teacher_token_mask"))
    sft_mask = _to_np(b.get("sft_token_mask"))
    scores = _to_np(b.get("token_level_scores"))
    nt = batch.non_tensor_batch
    uids = nt.get("uid")
    spans = nt.get("step_token_spans")
    d_iter = nt.get("dagger_iter_index")
    chain_id = nt.get("chain_id")
    traj_idx = nt.get("pope_traj_idx")
    step_idx = nt.get("pope_step_idx")
    parent_idx = nt.get("pope_parent_traj_idx")
    branch_step = nt.get("pope_branch_step")
    if uids is None:
        return None, None

    bs, resp_len = adv.shape
    zeros = np.zeros(resp_len, dtype=bool)
    recs = []
    for i in range(bs):
        active = resp_mask[i] > 0.5
        n_active = int(active.sum())
        tch = (tch_mask[i] > 0.5) if tch_mask is not None else zeros
        sft = (sft_mask[i] > 0.5) if sft_mask is not None else zeros
        n_tch = int((tch & active).sum())
        n_sft = int((sft & active).sum())
        pg_active = active & (~tch)
        n_pg = int(pg_active.sum())
        adv_traj = float(adv[i][active].mean()) if n_active else 0.0
        reward = float(scores[i].sum()) if scores is not None else float("nan")
        di = int(d_iter[i]) if d_iter is not None else 0
        guided = (n_tch > 0) or (di > 0)
        fully_forced = (n_pg == 0 and n_active > 0)

        row_spans = spans[i] if (spans is not None and i < len(spans)) else []
        steps = []
        for s_idx, span in enumerate(row_spans or []):
            try:
                st, en = int(span[0]), int(span[1])
            except Exception:
                continue
            en = min(en, resp_len - 1)
            if en < st:
                continue
            sl = slice(st, en + 1)
            sa = active[sl]
            ntok = int(sa.sum())
            if ntok == 0:
                continue
            s_tch = int((tch[sl] & sa).sum())
            s_sft = int((sft[sl] & sa).sum())
            s_pg = ntok - s_tch
            s_adv = float(adv[i][sl][sa].mean())
            prov = "expert" if s_sft > 0 else ("replay" if s_tch > 0 else "student")
            steps.append(
                dict(idx=s_idx, ntok=ntok, n_pg=s_pg, n_tch=s_tch, n_sft=s_sft,
                     adv=s_adv, prov=prov)
            )
        recs.append(
            dict(row=i, uid=str(uids[i]), adv=adv_traj, reward=reward,
                 n_active=n_active, n_pg=n_pg, n_tch=n_tch, n_sft=n_sft,
                 guided=bool(guided), fully_forced=bool(fully_forced),
                 dagger_iter=di, chain_id=int(chain_id[i]) if chain_id is not None else 0,
                 traj_idx=int(traj_idx[i]) if traj_idx is not None else -1,
                 step_idx=int(step_idx[i]) if step_idx is not None else -1,
                 parent_idx=int(parent_idx[i]) if parent_idx is not None else -1,
                 branch_step=int(branch_step[i]) if branch_step is not None else -1,
                 pg_signal=adv_traj * n_pg, steps=steps)
        )
    return recs, dict(bs=bs, resp_len=resp_len)


def _by_group(recs):
    order, g = [], {}
    for r in recs:
        u = r["uid"]
        if u not in g:
            g[u] = []
            order.append(u)
        g[u].append(r)
    return [(u, g[u]) for u in order]


def _group_stats(recs):
    import numpy as np

    groups = _by_group(recs)
    out = []
    for uid, rows in groups:
        rewards = np.array([r["reward"] for r in rows if not np.isnan(r["reward"])])
        advs = np.array([r["adv"] for r in rows])
        n_pg = np.array([r["n_pg"] for r in rows])
        n_guided = sum(1 for r in rows if r["guided"])
        n_forced = sum(1 for r in rows if r["fully_forced"])
        pg_tot = float(n_pg.sum())
        # net PG-token-weighted advantage = actual gradient direction on the
        # tokens that receive policy gradient (student tokens).
        net_pg_adv = (
            float(np.sum(advs * n_pg) / pg_tot) if pg_tot > 0 else 0.0
        )
        out.append(
            dict(uid=uid, n=len(rows), n_guided=n_guided, n_forced=n_forced,
                 mean_reward=float(rewards.mean()) if len(rewards) else float("nan"),
                 std_reward=float(rewards.std()) if len(rewards) else 0.0,
                 mean_adv=float(advs.mean()), net_pg_adv=net_pg_adv,
                 guided_frac=n_guided / max(1, len(rows)),
                 dead=bool(len(rewards) > 0 and rewards.std() < 1e-4),
                 pg_tokens=pg_tot, sft_tokens=float(sum(r["n_sft"] for r in rows)),
                 tch_tokens=float(sum(r["n_tch"] for r in rows)))
        )
    return out


# ─────────────────────────────── colours ───────────────────────────────────

_C = dict(student="#1f77b4", guided="#ff7f0e", forced="#7f7f7f",
          replay="#9e9e9e", expert="#ff7f0e", pad="#ffffff",
          pos="#2ca02c", neg="#d62728")


def _prov_color(r):
    if r["fully_forced"]:
        return _C["forced"]
    return _C["guided"] if r["guided"] else _C["student"]


# ─────────────────────────────── figure 1 ──────────────────────────────────


def _fig_group_advantage(recs, gstats, step, path):
    import numpy as np
    import matplotlib.pyplot as plt

    groups = _by_group(recs)
    # order groups by mean reward so the structure (dead/solved on the right) reads left→right
    order = sorted(range(len(groups)), key=lambda k: (gstats[k]["mean_reward"]
                   if not np.isnan(gstats[k]["mean_reward"]) else -1))
    fig, ax = plt.subplots(figsize=(max(8, 0.42 * len(groups) + 3), 5))
    rng = np.random.default_rng(0)
    for x, k in enumerate(order):
        rows = groups[k][1]
        for r in rows:
            jit = (rng.random() - 0.5) * 0.55
            ax.scatter(x + jit, r["adv"], s=22, alpha=0.8, color=_prov_color(r),
                       edgecolors="none", zorder=3)
        # group mean reward annotation (success proxy) along the top
        mr = gstats[k]["mean_reward"]
        ax.text(x, ax.get_ylim()[1], "", )  # noop to keep autoscale stable
    ax.axhline(0.0, color="k", lw=0.8, ls="--", zorder=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{gstats[k]['uid'][:5]}\nr={gstats[k]['mean_reward']:.2f}"
                        for k in order], fontsize=6, rotation=90)
    ax.set_ylabel("GRPO advantage (per trajectory)")
    ax.set_xlabel("group (uid, sorted by mean reward →)")
    ax.set_title(f"step {step}  ·  per-group advantage distribution\n"
                 f"(GRPO adv is constant within a trajectory = z-scored group reward)",
                 fontsize=10)
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker="o", ls="", color=_C["student"], label="student (on-policy)"),
           Line2D([0], [0], marker="o", ls="", color=_C["guided"], label="guided (teacher-forced prefix)"),
           Line2D([0], [0], marker="o", ls="", color=_C["forced"], label="fully-forced (excluded)")]
    ax.legend(handles=leg, fontsize=7, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ─────────────────────────────── figure 2 ──────────────────────────────────


def _step_cells(recs):
    """Build ``[(uid, guided, [cell...])]`` trajectories for the step-signal grid,
    supporting both batch layouts:

    * packed-trajectory rows (multi_turn builder): each row is a full trajectory
      and ``r["steps"]`` already holds its internal per-step spans.
    * per-step-sample rows (summary_context builder): each row IS one step, and
      rows are stitched into trajectories via ``traj_idx`` / ``step_idx``.

    A *cell* is ``dict(idx, adv, n_pg, n_tch, n_sft, prov)`` where ``prov`` is one
    of student / replay / expert.
    """
    # layout 1: per-row spans present
    if any(r.get("steps") for r in recs):
        trajs = []
        for r in recs:
            if not r.get("steps"):
                continue
            cells = [dict(idx=s["idx"], adv=s["adv"], n_pg=s["n_pg"],
                          n_tch=s["n_tch"], n_sft=s["n_sft"], prov=s["prov"])
                     for s in r["steps"]]
            meta = dict(traj_idx=r.get("traj_idx", r["row"]), parent_idx=-1,
                        branch_step=-1, chain_id=r.get("chain_id", 0),
                        dagger_iter=r.get("dagger_iter", 0), reward=r.get("reward"))
            trajs.append((r["uid"], bool(r["guided"]), cells, meta))
        return trajs, "row = full trajectory"

    # layout 2: per-step-sample rows stitched by (traj_idx, step_idx)
    by_traj = {}
    order = []
    for r in recs:
        ti = r.get("traj_idx", -1)
        if ti is None or ti < 0:
            continue
        if ti not in by_traj:
            by_traj[ti] = []
            order.append(ti)
        prov = ("expert" if r["n_sft"] > 0
                else ("replay" if r["n_tch"] > 0 else "student"))
        by_traj[ti].append(dict(idx=max(0, r.get("step_idx", 0)), adv=r["adv"],
                                n_pg=r["n_pg"], n_tch=r["n_tch"], n_sft=r["n_sft"],
                                prov=prov, uid=r["uid"], guided=bool(r["guided"]),
                                parent_idx=r.get("parent_idx", -1),
                                branch_step=r.get("branch_step", -1),
                                chain_id=r.get("chain_id", 0),
                                dagger_iter=r.get("dagger_iter", 0),
                                reward=r.get("reward")))
    trajs = []
    for ti in order:
        cells = sorted(by_traj[ti], key=lambda c: c["idx"])
        uid = cells[0]["uid"]
        guided = any(c["prov"] != "student" for c in cells)
        meta = dict(traj_idx=ti, parent_idx=cells[0]["parent_idx"],
                    branch_step=cells[0]["branch_step"],
                    chain_id=cells[0]["chain_id"], dagger_iter=cells[0]["dagger_iter"],
                    reward=cells[0]["reward"])
        trajs.append((uid, guided, cells, meta))
    return trajs, "row = one step-sample (summary_context)"


def _draw_branches_and_markers(ax, trajs, max_steps):
    """Overlay branch elbows (parent->child at the branch step) and
    end-of-trajectory success/fail markers on a provenance heatmap axis.

    ``trajs`` must already be in display (row) order. Draws on ``ax``: a thin
    magenta vertical segment down the branch column from the parent row, a short
    horizontal stub into the child's branch cell with a tiny arrowhead, and a
    green tick (reward > 0) / red cross (reward <= 0) at each trajectory's last
    step. Returns ``(n_arrows, n_succ, n_fail)``.
    """
    import numpy as np
    from matplotlib.patches import FancyArrowPatch
    import matplotlib.patheffects as pe

    white = [pe.withStroke(linewidth=1.5, foreground="white")]
    tidx_to_row = {}
    for ri, t in enumerate(trajs):
        ti = t[3].get("traj_idx", -1)
        if ti is not None and ti >= 0:
            tidx_to_row[ti] = ri
    n_arrows = 0
    for ri, t in enumerate(trajs):
        meta = t[3]
        pj = meta.get("parent_idx", -1)
        bstep = meta.get("branch_step", -1)
        if pj is None or pj < 0 or bstep is None or bstep < 0:
            continue
        pri = tidx_to_row.get(pj)
        if pri is None or pri == ri:
            continue
        xb = min(max(int(bstep), 0), max_steps - 1)
        x_v = xb - 0.5          # left boundary of the branch column
        x_head = xb - 0.05      # arrowhead just inside the branch cell
        vline, = ax.plot([x_v, x_v], [pri, ri], color="#ff00ff", lw=0.7,
                         alpha=0.9, zorder=6, solid_capstyle="butt")
        vline.set_path_effects(white)
        arr = FancyArrowPatch(
            (x_v, ri), (x_head, ri),
            arrowstyle="-|>,head_length=1.5,head_width=1.1",
            mutation_scale=1.0, lw=0.7, color="#ff00ff", alpha=0.9, zorder=6,
        )
        arr.set_path_effects(white)
        ax.add_patch(arr)
        n_arrows += 1

    n_succ = n_fail = 0
    for ri, t in enumerate(trajs):
        cells = t[2]
        rwd = t[3].get("reward")
        if rwd is None or (isinstance(rwd, float) and np.isnan(rwd)):
            continue
        last_idx = max((c["idx"] for c in cells), default=-1)
        if last_idx < 0:
            continue
        xm = last_idx + 0.5  # right boundary of the trajectory's last cell
        if rwd >= _SOLVED_REWARD:
            mk = ax.plot(xm, ri, marker=r"$\checkmark$", color="#0a7d0a",
                         ms=5.0, mew=0.0, ls="", zorder=7)[0]
            n_succ += 1
        else:
            mk = ax.plot(xm, ri, marker="x", color="#c00000",
                         ms=4.0, mew=1.1, ls="", zorder=7)[0]
            n_fail += 1
        mk.set_path_effects([pe.withStroke(linewidth=1.6, foreground="white")])
    # widen the right margin so end markers on full-length trajectories show
    ax.set_xlim(-0.5, max_steps - 0.5 + 0.6)
    return n_arrows, n_succ, n_fail


def _fig_step_signal(recs, step, path):
    """Advantage distributed per trajectory step + provenance/masking layout."""
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm, TwoSlopeNorm

    trajs, layout_note = _step_cells(recs)
    # order by group, then chain + dagger iteration so a branched child sits just
    # below its parent (keeps the branch arrows short and legible).
    trajs = sorted(trajs, key=lambda t: (t[0], t[3]["chain_id"],
                                         t[3]["dagger_iter"], t[3]["traj_idx"]))
    max_steps = max((max((c["idx"] for c in cells), default=-1) + 1
                     for _, _, cells, _ in trajs), default=0)
    if max_steps == 0 or not trajs:
        return
    n = len(trajs)
    adv_grid = np.full((n, max_steps), np.nan)
    prov_grid = np.zeros((n, max_steps))  # 0 pad,1 student,2 replay,3 expert
    pg_mass = np.zeros(max_steps); tch_mass = np.zeros(max_steps); sft_mass = np.zeros(max_steps)
    adv_by_pos = [[] for _ in range(max_steps)]
    for ri, (_uid, _guided, cells, _meta) in enumerate(trajs):
        for c in cells:
            j = c["idx"]
            if j >= max_steps:
                continue
            adv_grid[ri, j] = c["adv"]
            prov_grid[ri, j] = {"student": 1, "replay": 2, "expert": 3}[c["prov"]]
            pg_mass[j] += c["n_pg"]; tch_mass[j] += c["n_tch"]; sft_mass[j] += c["n_sft"]
            if c["prov"] == "student":
                adv_by_pos[j].append(c["adv"])

    fig = plt.figure(figsize=(max(9, 0.7 * max_steps + 5), 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.2], width_ratios=[1, 1], hspace=0.28, wspace=0.18)

    # advantage heatmap
    ax0 = fig.add_subplot(gs[0, 0])
    amax = np.nanmax(np.abs(adv_grid)) if np.isfinite(adv_grid).any() else 1.0
    amax = max(amax, 1e-3)
    norm = TwoSlopeNorm(vmin=-amax, vcenter=0.0, vmax=amax)
    im0 = ax0.imshow(adv_grid, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest")
    ax0.set_title("advantage per (trajectory, step)", fontsize=10)
    ax0.set_xlabel("step index"); ax0.set_ylabel("group (uid prefix; guided last within group)")
    ax0.set_xticks(range(max_steps))
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04, label="advantage")

    # provenance heatmap
    ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
    cmap = ListedColormap([_C["pad"], _C["student"], _C["replay"], _C["expert"]])
    norm2 = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    ax1.imshow(prov_grid, aspect="auto", cmap=cmap, norm=norm2, interpolation="nearest")
    ax1.set_title("token provenance per step\n(white=pad, blue=student/PG, grey=replay-masked, orange=expert/SFT)",
                  fontsize=9)
    ax1.set_xlabel("step index"); ax1.set_xticks(range(max_steps))

    # group divider lines: trajs are sorted by uid, so every row where the uid
    # changes is a (trajectory) group boundary. Draw a rule on both heatmaps and
    # tag each group band on the y-axis so the per-group structure is legible.
    group_uids = [t[0] for t in trajs]
    boundaries = [i for i in range(1, n) if group_uids[i] != group_uids[i - 1]]
    spans, start = [], 0
    for b in boundaries + [n]:
        spans.append((start, b)); start = b
    for ax in (ax0, ax1):
        for b in boundaries:
            ax.axhline(b - 0.5, color="k", lw=0.9, alpha=0.6)
    # short group tags (leading chars of uid) centred on each group band
    ax0.set_yticks([(s0 + e0 - 1) / 2.0 for s0, e0 in spans])
    ax0.set_yticklabels([group_uids[s0].split("-")[0][:6] for s0, e0 in spans],
                        fontsize=6)
    ax1.tick_params(labelleft=False)

    n_arrows, n_succ, n_fail = _draw_branches_and_markers(ax1, trajs, max_steps)
    ax1.set_xlabel("step index   ·   magenta elbow = parent→child branch   ·   "
                   f"green tick={n_succ} solved(r>=1) / red cross={n_fail} unsolved (traj end)",
                   fontsize=8)

    # per-step-position token mass (stacked) + mean student advantage line
    ax2 = fig.add_subplot(gs[1, :])
    x = np.arange(max_steps)
    ax2.bar(x, pg_mass, color=_C["student"], label="PG tokens (student)")
    ax2.bar(x, tch_mass, bottom=pg_mass, color=_C["replay"], label="teacher-masked (replay)")
    ax2.bar(x, sft_mass, bottom=pg_mass + tch_mass, color=_C["expert"], label="SFT tokens (expert)")
    ax2.set_xlabel("step index"); ax2.set_ylabel("token count (batch sum)")
    ax2.set_xticks(range(max_steps))
    ax2.legend(fontsize=7, loc="upper right")
    ax2b = ax2.twinx()
    mean_adv = [float(np.mean(a)) if a else np.nan for a in adv_by_pos]
    ax2b.plot(x, mean_adv, color="k", marker="o", lw=1.5, label="mean student advantage")
    ax2b.axhline(0, color="k", lw=0.6, ls=":")
    ax2b.set_ylabel("mean student advantage")
    ax2.set_title("per-step-position token budget (where masking removes gradient) "
                  "+ mean student advantage", fontsize=9)
    fig.suptitle(f"step {step}  ·  reward/advantage distributed per trajectory step "
                 f"(prefixes masked)  ·  {layout_note}", fontsize=11)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────── figure 3 ──────────────────────────────────


def _fig_provenance(recs, step, path):
    """Standalone token-provenance heatmap (the step_signal right panel on its
    own, full size): provenance per (trajectory, step) + group dividers + branch
    elbows + end-of-trajectory success/fail markers + a legend."""
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    trajs, layout_note = _step_cells(recs)
    trajs = sorted(trajs, key=lambda t: (t[0], t[3]["chain_id"],
                                         t[3]["dagger_iter"], t[3]["traj_idx"]))
    max_steps = max((max((c["idx"] for c in cells), default=-1) + 1
                     for _, _, cells, _ in trajs), default=0)
    if max_steps == 0 or not trajs:
        return
    n = len(trajs)
    prov_grid = np.zeros((n, max_steps))
    for ri, (_u, _g, cells, _m) in enumerate(trajs):
        for c in cells:
            j = c["idx"]
            if j < max_steps:
                prov_grid[ri, j] = {"student": 1, "replay": 2, "expert": 3}[c["prov"]]

    fig, ax = plt.subplots(figsize=(max(9, 0.62 * max_steps + 4.5),
                                    max(6, 0.17 * n + 2.5)))
    cmap = ListedColormap([_C["pad"], _C["student"], _C["replay"], _C["expert"]])
    norm2 = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    ax.imshow(prov_grid, aspect="auto", cmap=cmap, norm=norm2, interpolation="nearest")
    ax.set_xticks(range(max_steps))
    ax.set_ylabel("group (uid prefix; guided last within group)")

    group_uids = [t[0] for t in trajs]
    boundaries = [i for i in range(1, n) if group_uids[i] != group_uids[i - 1]]
    spans, startp = [], 0
    for b in boundaries + [n]:
        spans.append((startp, b)); startp = b
    for b in boundaries:
        ax.axhline(b - 0.5, color="k", lw=0.9, alpha=0.6)
    ax.set_yticks([(s0 + e0 - 1) / 2.0 for s0, e0 in spans])
    ax.set_yticklabels([group_uids[s0].split("-")[0][:6] for s0, e0 in spans],
                       fontsize=6)

    n_arrows, n_succ, n_fail = _draw_branches_and_markers(ax, trajs, max_steps)
    ax.set_xlabel("step index   ·   magenta elbow = parent→child branch   ·   "
                  f"green tick={n_succ} solved(r>=1) / red cross={n_fail} unsolved (traj end)",
                  fontsize=9)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    leg = [Patch(facecolor=_C["pad"], edgecolor="#bbbbbb", label="pad"),
           Patch(facecolor=_C["student"], label="student / PG"),
           Patch(facecolor=_C["replay"], label="replay (teacher-masked)"),
           Patch(facecolor=_C["expert"], label="expert / SFT"),
           Line2D([0], [0], marker=r"$\checkmark$", color="#0a7d0a", ls="", ms=8, label="solved (reward>=1)"),
           Line2D([0], [0], marker="x", color="#c00000", ls="", ms=7, label="unsolved (reward<1)"),
           Line2D([0], [0], color="#ff00ff", lw=1.2, label="branch (parent→child)")]
    ax.legend(handles=leg, fontsize=7, loc="upper left",
              bbox_to_anchor=(1.005, 1.0), framealpha=0.95, borderaxespad=0.0)
    fig.suptitle(f"step {step}  ·  token provenance per (trajectory, step)  ·  {layout_note}",
                 fontsize=12)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _fig_group_signal_dist(recs, gstats, step, path):
    import numpy as np
    import matplotlib.pyplot as plt

    groups = _by_group(recs)
    order = sorted(range(len(groups)), key=lambda k: (gstats[k]["mean_reward"]
                   if not np.isnan(gstats[k]["mean_reward"]) else -1))
    advs = [[r["adv"] for r in groups[k][1]] for k in order]
    npg = [[r["n_pg"] for r in groups[k][1]] for k in order]
    sig = [[r["pg_signal"] for r in groups[k][1]] for k in order]
    labels = [gstats[k]["uid"][:5] for k in order]
    fig, axes = plt.subplots(3, 1, figsize=(max(8, 0.42 * len(order) + 3), 9), sharex=True)
    for ax, data, title, ylab in [
        (axes[0], advs, "advantage", "advantage"),
        (axes[1], npg, "gradient reach (PG token count per trajectory)", "n_pg tokens"),
        (axes[2], sig, "effective PG signal  (advantage × n_pg)", "adv · n_pg"),
    ]:
        bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
        for box in bp["boxes"]:
            box.set(facecolor="#cfe3f3", edgecolor="#3b6e94")
        for med in bp["medians"]:
            med.set(color="#d62728")
        # overlay points
        rng = np.random.default_rng(1)
        for x, d in enumerate(data, start=1):
            jit = (rng.random(len(d)) - 0.5) * 0.5
            ax.scatter(x + jit, d, s=8, alpha=0.5, color="#333333", zorder=3)
        ax.axhline(0, color="k", lw=0.6, ls="--")
        ax.set_ylabel(ylab); ax.set_title(title, fontsize=9)
    axes[-1].set_xticks(range(1, len(order) + 1))
    axes[-1].set_xticklabels(labels, fontsize=6, rotation=90)
    axes[-1].set_xlabel("group (uid, sorted by mean reward →)")
    fig.suptitle(f"step {step}  ·  per-group distributions of gradient / advantage / signal", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ─────────────────────────────── figure 4 ──────────────────────────────────


def _fig_dashboard(recs, gstats, scal, step, path):
    import numpy as np
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    gr = np.array([g["mean_reward"] for g in gstats])
    gf = np.array([g["guided_frac"] for g in gstats])
    gn = np.array([g["n"] for g in gstats])
    gnet = np.array([g["net_pg_adv"] for g in gstats])
    gstd = np.array([g["std_reward"] for g in gstats])
    dead = np.array([g["dead"] for g in gstats])

    # (A) guidance vs group success — is guidance landing on already-solved groups?
    ax = axes[0, 0]
    sc = ax.scatter(gr, gf, s=20 + 8 * gn, c=dead.astype(float), cmap="coolwarm",
                    edgecolors="k", alpha=0.85)
    ax.set_xlabel("group mean reward (success rate)"); ax.set_ylabel("guided trajectory fraction")
    ax.set_title("guidance vs group success\n(top-right = guiding already-solved groups = wasted)", fontsize=9)
    ax.axvspan(0.0, 0.02, color="grey", alpha=0.12)
    ax.grid(alpha=0.3)

    # (B) net PG advantage per group — baseline inflation flips this negative
    ax = axes[0, 1]
    order = np.argsort(gr)
    colors = [_C["pos"] if v >= 0 else _C["neg"] for v in gnet[order]]
    ax.bar(range(len(order)), gnet[order], color=colors)
    ax.axhline(0, color="k", lw=0.8)
    nneg = int((gnet < 0).sum())
    ax.set_title(f"net PG-token-weighted advantage per group\n"
                 f"{nneg}/{len(gstats)} groups net-NEGATIVE "
                 f"(GRPO suppressing own behaviour)", fontsize=9)
    ax.set_xlabel("group (sorted by mean reward →)"); ax.set_ylabel("Σ adv·n_pg / Σ n_pg")

    # (C) advantage histogram, student vs guided
    ax = axes[0, 2]
    s_adv = [r["adv"] for r in recs if not r["guided"] and not r["fully_forced"]]
    g_adv = [r["adv"] for r in recs if r["guided"] and not r["fully_forced"]]
    bins = np.linspace(min([0] + s_adv + g_adv), max([0] + s_adv + g_adv), 31) if (s_adv or g_adv) else 20
    if s_adv:
        ax.hist(s_adv, bins=bins, alpha=0.6, color=_C["student"], label=f"student (n={len(s_adv)})")
    if g_adv:
        ax.hist(g_adv, bins=bins, alpha=0.6, color=_C["guided"], label=f"guided (n={len(g_adv)})")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_title("advantage distribution: student vs guided\n(guided mass >0 ⇒ baseline inflation)", fontsize=9)
    ax.set_xlabel("advantage"); ax.legend(fontsize=7)

    # (D) per-group reward std — dead-group rate
    ax = axes[1, 0]
    ax.bar(range(len(order)), gstd[order], color=[_C["neg"] if dead[k] else _C["student"] for k in order])
    ax.set_title(f"group reward std  ·  dead groups (std≈0): {int(dead.sum())}/{len(gstats)}\n"
                 f"(dead ⇒ zero advantage ⇒ wasted rollout)", fontsize=9)
    ax.set_xlabel("group (sorted by mean reward →)"); ax.set_ylabel("reward std within group")

    # (E) batch token budget
    ax = axes[1, 1]
    pg = scal["pg_tokens"]; tch = scal["teacher_tokens"]; sft = scal["sft_tokens"]
    ax.bar(["PG (student)", "teacher-masked", "SFT (expert)"], [pg, tch, sft],
           color=[_C["student"], _C["replay"], _C["expert"]])
    tot = max(1.0, pg + tch + sft)
    for i, v in enumerate([pg, tch, sft]):
        ax.text(i, v, f"{v/tot:.1%}", ha="center", va="bottom", fontsize=9)
    ax.set_title(f"response-token budget\nteacher_frac={tch/tot:.3f}  sft_frac={sft/tot:.3f}", fontsize=9)
    ax.set_ylabel("token count")

    # (F) scalar summary text
    ax = axes[1, 2]; ax.axis("off")
    lines = [
        f"step {step}",
        "",
        f"onpolicy_avg_reward = {scal['onpolicy_avg_reward']:.3f}  (n={scal['n_onpolicy']})",
        f"blended_avg_reward  = {scal['blended_avg_reward']:.3f}  (n={scal['n_blended']})",
        f"guided_traj_frac    = {scal['guided_traj_frac']:.3f}",
        f"sft_token_frac      = {scal['sft_frac']:.3f}",
        f"teacher_token_frac  = {scal['teacher_frac']:.3f}",
        "",
        f"groups              = {scal['n_groups']}",
        f"dead groups (std≈0) = {scal['n_dead_groups']}",
        f"fully-forced trajs  = {scal['n_fully_forced']}  (excluded)",
        "",
        f"net PG adv (mean/grp)   = {scal['net_pg_adv_mean']:+.4f}",
        f"groups net-negative PG  = {scal['n_groups_net_neg']}/{scal['n_groups']}",
        f"frac PG signal mass >0  = {scal['frac_pg_mass_pos']:.3f}",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=11)
    fig.suptitle(f"POPE-DAgger step {step} — why it is / isn't learning as expected", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ─────────────────────────────── scalars ───────────────────────────────────


def _scalars(recs, gstats):
    import numpy as np

    nonforced = [r for r in recs if not r["fully_forced"]]
    onpolicy = [r["reward"] for r in nonforced if not r["guided"] and not np.isnan(r["reward"])]
    blended = [r["reward"] for r in nonforced if not np.isnan(r["reward"])]
    pg_tokens = float(sum(r["n_pg"] for r in recs))
    sft_tokens = float(sum(r["n_sft"] for r in recs))
    tch_tokens = float(sum(r["n_tch"] for r in recs))
    resp_tokens = float(sum(r["n_active"] for r in recs))
    net = np.array([g["net_pg_adv"] for g in gstats])
    pg_mass_pos = sum(r["pg_signal"] for r in recs if r["pg_signal"] > 0)
    pg_mass_abs = sum(abs(r["pg_signal"]) for r in recs)
    return dict(
        step_bs=len(recs),
        onpolicy_avg_reward=float(np.mean(onpolicy)) if onpolicy else float("nan"),
        n_onpolicy=len(onpolicy),
        blended_avg_reward=float(np.mean(blended)) if blended else float("nan"),
        n_blended=len(blended),
        guided_traj_frac=sum(1 for r in recs if r["guided"]) / max(1, len(recs)),
        sft_frac=sft_tokens / max(1.0, resp_tokens),
        teacher_frac=tch_tokens / max(1.0, resp_tokens),
        pg_tokens=pg_tokens, sft_tokens=sft_tokens, teacher_tokens=tch_tokens,
        n_groups=len(gstats),
        n_dead_groups=int(sum(1 for g in gstats if g["dead"])),
        n_fully_forced=int(sum(1 for r in recs if r["fully_forced"])),
        net_pg_adv_mean=float(net.mean()) if len(net) else 0.0,
        n_groups_net_neg=int((net < 0).sum()),
        frac_pg_mass_pos=float(pg_mass_pos / pg_mass_abs) if pg_mass_abs > 0 else float("nan"),
    )


def _update_trends(viz_dir, step, scal):
    import numpy as np
    import matplotlib.pyplot as plt

    row = dict(step=int(step), **{k: scal[k] for k in (
        "onpolicy_avg_reward", "blended_avg_reward", "guided_traj_frac",
        "sft_frac", "teacher_frac", "n_dead_groups", "n_groups",
        "net_pg_adv_mean", "n_groups_net_neg", "frac_pg_mass_pos")})
    jl = os.path.join(viz_dir, "trends.jsonl")
    with open(jl, "a") as f:
        f.write(json.dumps(row) + "\n")

    rows = []
    with open(jl) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                try:
                    rows.append(json.loads(ln))
                except Exception:
                    pass
    if not rows:
        return
    rows.sort(key=lambda r: r["step"])
    s = [r["step"] for r in rows]

    def col(k):
        return [r.get(k, float("nan")) for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax = axes[0, 0]
    ax.plot(s, col("onpolicy_avg_reward"), "-o", color=_C["student"], label="onpolicy")
    ax.plot(s, col("blended_avg_reward"), "-o", color=_C["guided"], label="blended")
    ax.set_title("reward over steps"); ax.set_xlabel("step"); ax.set_ylabel("avg reward")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[0, 1]
    ax.plot(s, col("guided_traj_frac"), "-o", color=_C["guided"], label="guided_traj_frac")
    ax.plot(s, col("sft_frac"), "-o", color="#2ca02c", label="sft_frac")
    ax.plot(s, col("teacher_frac"), "-o", color=_C["replay"], label="teacher_frac")
    ax.set_title("guidance taper"); ax.set_xlabel("step"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[1, 0]
    ax.plot(s, col("net_pg_adv_mean"), "-o", color="k")
    ax.axhline(0, color="r", lw=0.8, ls="--")
    ax.set_title("mean net PG advantage per group\n(<0 ⇒ suppressing own behaviour)", fontsize=9)
    ax.set_xlabel("step"); ax.grid(alpha=0.3)
    ax = axes[1, 1]
    frac_dead = [r.get("n_dead_groups", 0) / max(1, r.get("n_groups", 1)) for r in rows]
    frac_neg = [r.get("n_groups_net_neg", 0) / max(1, r.get("n_groups", 1)) for r in rows]
    ax.plot(s, frac_dead, "-o", color=_C["neg"], label="dead-group frac")
    ax.plot(s, frac_neg, "-o", color="#9467bd", label="net-neg-PG group frac")
    ax.plot(s, col("frac_pg_mass_pos"), "-o", color=_C["pos"], label="frac PG mass >0")
    ax.set_title("group health"); ax.set_xlabel("step"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("POPE-DAgger run-level trends", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(viz_dir, "trends.png"), dpi=110)
    plt.close(fig)


# ─────────────────────────────── entry point ───────────────────────────────


def save_step_figures(batch, global_step, output_dir) -> None:
    """Render + save the per-step POPE-DAgger diagnostic figures. Safe no-op if
    disabled, output_dir missing, or required batch keys absent."""
    try:
        from verl.utils.pope_dagger_viz import figs_enabled
    except Exception:
        def figs_enabled():
            v = os.environ.get("POPE_DAGGER_VIZ_FIGS")
            return v == "1" if v is not None else os.environ.get("POPE_DAGGER_VIZ") == "1"

    if not figs_enabled() or not output_dir:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import numpy as np  # noqa: F401

        recs, meta = _extract_records(batch)
        if not recs:
            logger.info("[pope_dagger_figs] required keys missing; skipping figures")
            return
        gstats = _group_stats(recs)
        scal = _scalars(recs, gstats)

        step = int(global_step) if global_step is not None else 0
        viz_dir = os.path.join(output_dir, "pope_viz")
        step_dir = os.path.join(viz_dir, f"step_{step:03d}")
        os.makedirs(step_dir, exist_ok=True)

        _fig_group_advantage(recs, gstats, step, os.path.join(step_dir, "group_advantage_dist.png"))
        _fig_step_signal(recs, step, os.path.join(step_dir, "step_signal.png"))
        _fig_provenance(recs, step, os.path.join(step_dir, "step_provenance.png"))
        _fig_group_signal_dist(recs, gstats, step, os.path.join(step_dir, "group_signal_dist.png"))
        _fig_dashboard(recs, gstats, scal, step, os.path.join(step_dir, "dashboard.png"))

        with open(os.path.join(step_dir, "stats.json"), "w") as f:
            json.dump(dict(scalars=scal, groups=gstats), f, indent=2)
        _update_trends(viz_dir, step, scal)

        logger.info(f"[pope_dagger_figs] step {step}: saved 4 figures + stats → {step_dir}")
    except Exception as e:  # never break training on a viz error
        logger.warning(f"[pope_dagger_figs] step figures failed: {e!r}")
