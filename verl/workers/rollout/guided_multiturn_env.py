"""Pope-Dagger V4/V5 guided multi-turn rollout.

Extends ``MultiturnEnvRollout`` to emit *one or more* chains of V4/V5
iterations per prompt until each group hits ``n_per_prompt`` trajectories:

    chain 0:
        iter 0   on-policy baseline
        iter 1   forces step B_1
        iter 2   forces step B_2 > B_1
        ...
    chain 1:                       (spawned if group still needs more trajs)
        iter 0   *fresh* on-policy baseline (new env reset, new sampling)
        iter 1   forces step B_1 (V5 selector re-initialized for this chain)
        ...
    ...

Each iteration is run as a *fresh* episode (env.reset on a new slot) because
AI2Thor can't be rewound mid-episode. Steps 0..B_k−1 are replayed
deterministically by injecting the parent rollout's stored step responses
via the ``pending_response_override`` hook on ``Trajectory``; the branch
step itself is injected with the teacher VLM's annotated reasoning + the
oracle's expert action; remaining steps run student vLLM normally.

A chain stops extending when its parent solves (if ``stop_on_solved``), the
V5 selector can no longer find a usable branch step, or the group hits
``n_per_prompt``. The outer loop then spawns a new chain (incrementing
``chain_id``) for any group that still needs trajectories. Cross-chain pairs
are treated as fully divergent (LCP=0) by the branching advantage estimator.

Token-level metadata flows through ``Trajectory.step_teacher_forced``,
``dagger_iter_index``, ``branch_chain``, and ``chain_id``. The base class's
``_build_final_batch_multiturn`` broadcasts these into ``teacher_token_mask``
and ``step_token_spans`` / ``dagger_iter_index`` / ``branch_chain`` /
``chain_id`` non-tensor metadata that the branching advantage estimator
consumes.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

import numpy as np
import ray

from verl.protocol import DataProto
from verl.workers.rollout.multiturn_env import MultiturnEnvRollout, Trajectory

logger = logging.getLogger(__name__)


# ─── strategy hooks ────────────────────────────────────────────────────────


class _PendingAnnotation:
    """Deferred teacher-VLM annotation for one forced expert step.

    ``_build_override_for`` returns this (instead of calling the blocking
    ``teacher_annotate_fn`` inline) when a forced step needs teacher reasoning.
    The ``pre_step`` hook collects all of a step's pending annotations across
    the iteration's concurrently-active trajectories (up to one per eligible
    group) and fans the calls out on a thread pool, so the vLLM teacher server
    batches the inflight HTTP requests instead of serializing them one
    trajectory at a time. ``kwargs`` are the ``teacher_annotate_fn`` arguments.
    """

    __slots__ = ("kwargs",)

    def __init__(self, kwargs: dict):
        self.kwargs = kwargs


class ExpertActionFn(Protocol):
    """Returns the formatted expert action string for the slot's current
    state, e.g. ``"<explore x=1 y=0 z=2/>"``. Implementations call into
    ``pope_dagger.expert_replay.compute_expert_actions_via_sparsify`` via a
    Ray-remote method on the SimulatorPool.
    """

    def __call__(
        self,
        pool: Any,
        slot_id: int,
        dataset_item: dict,
    ) -> str: ...


class TeacherAnnotateFn(Protocol):
    """Returns ``(assistant_text, token_ids)`` to inject at a forced step.

    ``assistant_text`` is the full ``<think>...</think>\n<action.../>`` block
    that gets recorded in ``step_responses``. ``token_ids`` is its tokenized
    form (used as the per-step token spans in the final batch). The simplest
    implementation tokenizes ``assistant_text`` with the student tokenizer.
    """

    def __call__(
        self,
        *,
        pool: Any,
        slot_id: int,
        episode_id: str,
        expert_action_formatted: str,
        parent: Trajectory,
        branch_step_index: int,
    ) -> tuple[str, list[int]]: ...


@dataclass
class GuidedConfig:
    """Guided rollout knobs (separate from base ``RolloutConfig`` so the base
    config schema doesn't need to grow new fields)."""

    enabled: bool = True
    # K = max V4 iterations after the baseline (inclusive of iter K).
    # Total trajectories per prompt = 1 (baseline) + min(K, max_actions-2)
    # iters + padding.
    max_iters: int = 3
    # Always pad each group to exactly this many trajectories.
    n_per_prompt: int = 4
    # If True, stop iterating an episode once the parent succeeded.
    stop_on_solved: bool = True
    # "fixed_step_k": iter k forces step k-1 (deterministic ladder).
    # "random_regression": V5 pope-dagger selector — random step sampled
    # from [suffix_peak..last_unrecovered] of the latest unforced suffix.
    branch_selection_mode: str = "random_regression"
    # Optional seed for the V5 random selector (mixed with group_id + iter_k).
    random_regression_seed: int | None = None
    # Forced-window length is *not* fixed at 1: each iter forces consecutive
    # expert steps starting at the branch until best-progress-so-far on the
    # new trajectory rises by ``progress_resume_threshold`` over the parent's
    # progress-floor at the branch step (V4 progress-resume guarantee).
    progress_resume_threshold: float = 0.10
    # Safety cap on consecutive forced steps per iter. ``None`` = uncapped
    # except by ``max_depth`` (last reachable step is always student-controlled
    # — see ``force_expert_answer`` and the explicit step_idx >= max_depth-1
    # guard in ``_build_override_for``).
    max_forced_window_length: int | None = None
    # Never force the agent to emit an ``<answer .../>`` action — if the oracle
    # returns an answer, hand off to the student so the answer step is always
    # policy-generated. Mirrors pope-dagger's ``force_expert_answer`` knob.
    force_expert_answer: bool = False
    # If True, the final iteration of each chain (iter == global_max_iters) is a
    # *completion* iter: from its branch, the expert forces every remaining step
    # through the terminal ``<answer/>`` (guaranteed success). Produces the
    # SFT-only "expert completion" trajectories from the design. With
    # ``max_iters=2`` this yields one middle-injection iter + one completion iter.
    completion_final_iter: bool = True


@dataclass
class _IterPlan:
    """Per-episode iteration plan: which parent to clone, which step to force.

    The forced-window runtime fields below are populated by
    ``_build_override_for`` when the rollout first reaches ``branch_step_index``
    on the new trajectory. They drive the multi-step forced window:
    consecutive expert steps are forced until ``forced_window_max_bp`` (best
    progress observed on the new trajectory) ≥ ``progress_resume_target``.
    """

    parent: Trajectory
    branch_step_index: int
    # Parent's full ancestry chain extended with this iter's branch.
    branch_chain: list[int] = field(default_factory=list)
    # When True, ``_build_override_for`` forces the expert from the branch
    # through the terminal answer (see ``GuidedConfig.completion_final_iter``).
    is_completion: bool = False
    # ── forced-window runtime state (populated when override hits branch) ──
    progress_resume_target: float | None = None
    forced_window_max_bp: float = 0.0
    forced_window_steps_emitted: int = 0


# ─── default annotator (no-VLM, placeholder reasoning) ─────────────────────


def _build_placeholder_annotator(tokenizer) -> TeacherAnnotateFn:
    """Stub annotator: wraps the expert action in a generic ``<think>`` block.

    Used until a real teacher VLM is plumbed in (server-mode or co-located).
    The trained policy still benefits from the *structural* signal — at this
    branch step the trajectory took the oracle's action — even if the
    accompanying reasoning is stylized.
    """

    PREFIX = (
        "<think>\n"
        "The expert demonstration suggests the action below for this state. "
        "Following it advances toward the target.\n"
        "</think>\n"
    )

    def _annotate(
        *,
        pool: Any,
        slot_id: int,
        episode_id: str,
        expert_action_formatted: str,
        parent: Trajectory,
        branch_step_index: int,
    ) -> tuple[str, list[int]]:
        text = PREFIX + expert_action_formatted
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        return text, token_ids

    return _annotate


# ─── guided rollout class ──────────────────────────────────────────────────


class GuidedMultiturnEnvRollout(MultiturnEnvRollout):
    def __init__(
        self,
        *args,
        guided_cfg: GuidedConfig,
        expert_action_fn: Optional[ExpertActionFn] = None,
        teacher_annotate_fn: Optional[TeacherAnnotateFn] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.guided_cfg = guided_cfg
        self.expert_action_fn = expert_action_fn or _default_expert_action_fn
        self.teacher_annotate_fn = (
            teacher_annotate_fn or _build_placeholder_annotator(self.tokenizer)
        )

    # Override to short-circuit when guidance is disabled at runtime.
    def generate_trajectories(
        self,
        actor_rollout_ref_wg,
        batch_size: int,
        n_trajectories: int,
        config,
        metrics: dict[str, Any],
        override_config: dict[str, Any] | None = None,
    ) -> DataProto:
        if not self.guided_cfg.enabled:
            return super().generate_trajectories(
                actor_rollout_ref_wg, batch_size, n_trajectories, config,
                metrics, override_config,
            )

        # Force the target group size to match the guided config; this is what
        # GRPO sees as `n` and what the batch tensor reshapes around.
        n_per_prompt = max(self.guided_cfg.n_per_prompt, n_trajectories)

        t_start = time.time()

        # Sync-mode phase accumulators — the base _run_continuous_episode_loop
        # increments these per turn. The base generate_trajectories initializes
        # them per call; this guided override must do the same or the first
        # `+=` raises AttributeError.
        self._sync_gen_seconds = 0.0
        self._sync_env_seconds = 0.0
        self._sync_prompt_build_seconds = 0.0
        self._sync_step_total_seconds = 0.0
        self._sync_n_turns = 0

        self.warmup_controllers()

        pool_infos = ray.get(
            [p.get_pool_info.remote() for p in self.simulator_pools]
        )
        total_slots = sum(info["total"] for info in pool_infos)

        # Collect dataset items for this batch
        all_items = [self.env_factory.get_next_item() for _ in range(batch_size)]
        items_by_group: dict[int, dict] = {i: it for i, it in enumerate(all_items)}

        from pope_dagger.analyzer import BranchSelectorState

        all_trajs: list[Trajectory] = []
        # Remaining slots per group; decremented each time a trajectory is
        # produced for that group. Stops chain spawning once the group fills.
        need_per_group: dict[int, int] = {g: n_per_prompt for g in items_by_group}
        # Global hard cap on chains spawned per group (defensive — at worst
        # each chain produces 1 traj (baseline alone), so n_per_prompt is the
        # natural ceiling).
        max_chains_per_group = n_per_prompt
        chain_idx = 0

        # Effective iter cap per chain (same as before; the chain itself
        # additionally caps iters by its own group's remaining need).
        global_max_iters = min(
            self.guided_cfg.max_iters,
            self.max_depth - 1,  # need at least one student suffix step
        )

        while any(v > 0 for v in need_per_group.values()):
            if chain_idx >= max_chains_per_group:
                logger.warning(
                    f"[guided] reached max_chains_per_group={max_chains_per_group}; "
                    f"some groups still need trajectories: "
                    f"{[(g, v) for g, v in need_per_group.items() if v > 0]}"
                )
                break

            eligible = [g for g, v in need_per_group.items() if v > 0]

            # ── chain `chain_idx`: baseline (1 traj per eligible group) ─
            baseline_trajs = self._run_one_pass(
                items_per_prompt=[(g, items_by_group[g]) for g in eligible],
                n_per_prompt=1,
                actor_rollout_ref_wg=actor_rollout_ref_wg,
                config=config,
                override_config=override_config,
                total_slots=total_slots,
                override_provider=None,
                dagger_iter_index=0,
                branch_chain=[],
                chain_id=chain_idx,
            )
            for t in baseline_trajs:
                t.chain_id = chain_idx
                t.dagger_iter_index = 0
                t.branch_chain = []
                need_per_group[t.group_id] -= 1
            all_trajs.extend(baseline_trajs)
            logger.info(
                f"[guided] chain {chain_idx} baseline complete: "
                f"{len(baseline_trajs)} trajectories"
            )

            # Per-chain V5 selector state. Each chain restarts from a fresh
            # BranchSelectorState (no forced windows yet — its baseline is
            # an independent on-policy rollout).
            parent_for_group: dict[int, Trajectory] = {
                t.group_id: t for t in baseline_trajs
            }
            selector_states: dict[int, BranchSelectorState] = {
                t.group_id: BranchSelectorState() for t in baseline_trajs
            }
            # Per-chain "done" set: groups whose chain has nothing left to
            # guide. Populated when an iter's forced window emitted zero
            # forced steps — usually because the oracle's next action at the
            # selected branch was an ``<answer/>`` (filtered out by
            # ``force_expert_answer=False``). Further iters in this chain are
            # pointless; the next chain (chain_idx+1) will start fresh.
            chain_done_groups: set[int] = set()

            # ── iters 1..K of this chain ───────────────────────────────
            for iter_k in range(1, global_max_iters + 1):
                # Stop iterating a group if it's full, its parent solved, or
                # its chain was marked done by a prior iter (e.g. only the
                # terminal answer remained at the selected branch).
                eligible_parents = {
                    g: p
                    for g, p in parent_for_group.items()
                    if need_per_group.get(g, 0) > 0
                    and not (self.guided_cfg.stop_on_solved and _is_solved(p))
                    and g not in chain_done_groups
                }
                if not eligible_parents:
                    break

                # The final iter of a chain is the "expert completes the
                # trajectory" pass when completion_final_iter is set.
                is_completion_iter = (
                    self.guided_cfg.completion_final_iter
                    and iter_k == global_max_iters
                )
                plans = self._plan_iteration(
                    iter_k, eligible_parents, items_by_group, selector_states,
                    chain_id=chain_idx, is_completion=is_completion_iter,
                )
                if not plans:
                    logger.info(
                        f"[guided] chain {chain_idx} iter {iter_k}: no eligible "
                        f"branch plans, stopping chain"
                    )
                    break

                iter_items = [(g, items_by_group[g]) for g in plans.keys()]
                iter_trajs = self._run_one_pass(
                    items_per_prompt=iter_items,
                    n_per_prompt=1,
                    actor_rollout_ref_wg=actor_rollout_ref_wg,
                    config=config,
                    override_config=override_config,
                    total_slots=total_slots,
                    override_provider=lambda t, _plans=plans: self._build_override_for(t, _plans),
                    dagger_iter_index=iter_k,
                    branch_chain=None,  # filled per-traj below
                    chain_id=chain_idx,
                )
                for t in iter_trajs:
                    t.chain_id = chain_idx
                    t.dagger_iter_index = iter_k
                    t.branch_chain = list(plans[t.group_id].branch_chain)
                    need_per_group[t.group_id] -= 1
                all_trajs.extend(iter_trajs)
                # Promote and update selector state. ``forced_window_ranges``
                # holds half-open [start, end) intervals; ``end`` is one past
                # the last forced step, so width = forced_window_steps_emitted.
                # If width == 0 the iter's branch had no forceable expert step
                # (e.g. oracle returned ``<answer/>`` first): the chain is
                # done for this group — no point trying further iters.
                for t in iter_trajs:
                    p = plans[t.group_id]
                    if p.forced_window_steps_emitted == 0:
                        chain_done_groups.add(t.group_id)
                        logger.info(
                            f"[guided] chain {chain_idx} iter {iter_k} group "
                            f"{t.group_id}: zero forced steps emitted "
                            f"(answer-only at branch); ending chain for this group"
                        )
                        continue
                    parent_for_group[t.group_id] = t
                    width = p.forced_window_steps_emitted
                    selector_states[t.group_id].forced_window_ranges.append(
                        (p.branch_step_index, p.branch_step_index + width)
                    )

                logger.info(
                    f"[guided] chain {chain_idx} iter {iter_k} complete: "
                    f"{len(iter_trajs)} trajectories, chained from {len(plans)} parents"
                )

            chain_idx += 1

        # Reassign n_idx within each group in chronological order of generation
        # (chain 0 baseline → chain 0 iters → chain 1 baseline → …).
        by_group: dict[int, list[Trajectory]] = {}
        for t in all_trajs:
            by_group.setdefault(t.group_id, []).append(t)
        for g, ts in by_group.items():
            for i, t in enumerate(ts):
                t.n_idx = i

        # Sort and finalize the batch (identical to base class tail).
        all_trajs.sort(key=lambda t: (t.group_id, t.n_idx))
        all_rewards = [t.reward if t.reward is not None else 0.0 for t in all_trajs]
        all_ground_truths = [t.ground_truth or "{}" for t in all_trajs]

        avg_steps = float(np.mean([t.num_steps for t in all_trajs])) if all_trajs else 0.0
        avg_reward = float(np.mean(all_rewards)) if all_trajs else 0.0
        metrics["env/avg_steps"] = avg_steps
        metrics["env/avg_reward"] = avg_reward
        metrics["reward/overall"] = avg_reward
        metrics["guided/total_trajectories"] = len(all_trajs)
        metrics["guided/avg_dagger_iter"] = float(
            np.mean([t.dagger_iter_index for t in all_trajs])
        )
        metrics["guided/chains_spawned"] = float(chain_idx)
        metrics["guided/avg_chain_id"] = float(
            np.mean([t.chain_id for t in all_trajs])
        )

        logger.info(
            f"[guided] rollout complete: {len(all_trajs)} trajs in "
            f"{time.time() - t_start:.1f}s | avg_steps={avg_steps:.1f} | "
            f"avg_reward={avg_reward:.3f}"
        )

        self.destroy_controllers()
        batch = self._build_final_batch(
            all_trajs, all_rewards, all_ground_truths, n_per_prompt
        )
        for t in all_trajs:
            t.last_images.clear()
            t.last_prompt = None
            t.step_responses.clear()
            t.step_response_token_ids.clear()
            t.step_response_log_probs.clear()
        all_trajs.clear()
        return batch

    # ── iteration helpers ────────────────────────────────────────────────

    def _plan_iteration(
        self,
        iter_k: int,
        parent_for_group: dict[int, Trajectory],
        items_by_group: dict[int, dict],
        selector_states: dict[int, Any] | None = None,
        chain_id: int = 0,
        is_completion: bool = False,
    ) -> dict[int, _IterPlan]:
        """Pick a branch step per eligible parent. Eligibility:
          1. Parent not solved (if stop_on_solved).
          2. Branch step would have at least one student-suffix step
             (branch_step_index <= max_depth - 2).

        Branch selection dispatched on ``branch_selection_mode``:
          * "fixed_step_k": iter k forces step (k-1). Deterministic ladder.
          * "random_regression": V5 pope-dagger selector — random step in
            [suffix_peak..last_unrecovered] over the un-forced suffix.
        """
        mode = self.guided_cfg.branch_selection_mode
        plans: dict[int, _IterPlan] = {}
        for group_id, parent in parent_for_group.items():
            if self.guided_cfg.stop_on_solved and _is_solved(parent):
                continue
            if mode == "fixed_step_k":
                branch_step = iter_k - 1
            elif mode == "random_regression":
                state = (
                    selector_states[group_id]
                    if selector_states is not None
                    else None
                )
                branch_step = self._select_branch_random_regression(
                    parent=parent,
                    state=state,
                    group_id=group_id,
                    iter_k=iter_k,
                    chain_id=chain_id,
                )
                if branch_step is None:
                    continue
            else:
                raise ValueError(
                    f"Unknown branch_selection_mode: {mode!r}"
                )
            if branch_step >= self.max_depth - 1:
                continue
            if branch_step >= parent.num_steps:
                # Parent didn't go this deep — can't replay past its length.
                continue
            chain = list(parent.branch_chain) + [branch_step]
            plans[group_id] = _IterPlan(
                parent=parent,
                branch_step_index=branch_step,
                branch_chain=chain,
                is_completion=is_completion,
            )
        return plans

    def _select_branch_random_regression(
        self,
        *,
        parent: Trajectory,
        state: Any,  # BranchSelectorState
        group_id: int,
        iter_k: int,
        chain_id: int = 0,
    ) -> int | None:
        """Run pope-dagger's V5 random_regression selector over the parent's
        per-step distances. Returns None if no usable progress series."""
        import random
        from pope_dagger.analyzer import select_branch_point_by_random_regression
        from pope_dagger_sanity.types import TrajectoryStepRecord

        distances = parent.step_distances
        if not distances or len(distances) < 2:
            return None
        full_steps: list[TrajectoryStepRecord] = []
        for i, d in enumerate(distances):
            if d is None:
                continue
            full_steps.append(
                TrajectoryStepRecord(
                    step_index=i,
                    state_before={"shortest_path_distance_to_target": float(d)},
                    state_after={},
                    action={},
                    model_response="",
                    parsed_action="",
                    env_feedback="",
                    reward=0.0,
                    is_terminal=False,
                )
            )
        if len(full_steps) < 2:
            return None
        seed_base = self.guided_cfg.random_regression_seed
        if seed_base is None:
            rng = random.Random()
        else:
            rng = random.Random(hash((seed_base, group_id, chain_id, iter_k)) & 0xFFFFFFFF)
        return select_branch_point_by_random_regression(
            full_steps, state, rng=rng
        )

    def _build_override_for(
        self,
        t: Trajectory,
        plans: dict[int, _IterPlan],
    ) -> Optional[tuple[str, list[int], bool]]:
        """``override_provider`` hook used by ``_run_one_pass``.

        Returns the override to inject at ``t``'s current step (i.e. its
        ``num_steps`` position), or None to fall through to vLLM.

        Three regimes, dispatched on ``step_idx`` vs ``branch_step_index``:

        - ``step_idx < branch``: deterministic prefix replay from parent.
        - ``step_idx == branch``: open the forced window — compute progress-
          resume target from parent's progress-floor + ``δ``, then force the
          first expert step.
        - ``step_idx > branch``: continue forcing only if (a) progress hasn't
          resumed on the new trajectory, (b) the safety cap isn't hit, (c) the
          next step wouldn't be the terminal answer step, and (d) the oracle's
          next action is not itself an ``<answer/>`` (we never force the
          policy to emit an answer — see ``force_expert_answer``).
        """
        plan = plans.get(t.group_id)
        if plan is None:
            return None
        step_idx = t.num_steps  # the step about to be emitted
        parent = plan.parent
        branch = plan.branch_step_index

        # ── 1. Prefix replay ────────────────────────────────────────────
        if step_idx < branch:
            if step_idx >= len(parent.step_responses):
                return None
            text = parent.step_responses[step_idx]
            token_ids = parent.step_response_token_ids[step_idx]
            # Replayed prefix tokens are *conditioning*, not a fresh sample: they
            # were sampled on-policy once (in the source rollout) and are replayed
            # here byte-for-byte. Mask them from the policy gradient (is_teacher=
            # True) so the expert boost downstream can't leak back into the prefix
            # and the prefix isn't double-counted; but do NOT SFT them
            # (is_expert=False) — they are trained exactly once, via their source.
            return text, list(token_ids), True, False

        # ── 2. Open the forced window at branch ──────────────────────────
        if step_idx == branch and plan.progress_resume_target is None:
            parent_floor = _progress_floor_through(parent, through_step=branch)
            plan.progress_resume_target = (
                parent_floor + self.guided_cfg.progress_resume_threshold
            )
            plan.forced_window_max_bp = parent_floor

        # A completion iteration forces the expert all the way through the
        # terminal ``<answer/>`` from the branch onward (the max-iter "expert
        # finishes the trajectory" regime). It bypasses the progress-resume
        # early-stop (3a), the terminal-step handoff (3c), and the answer block
        # (4a). The forced-step safety cap (3b) still applies as a guard.
        is_completion = bool(getattr(plan, "is_completion", False))

        # ── 3. Termination checks for steps past the first forced step ──
        if step_idx > branch:
            # 3a. Has best-progress-so-far on the new trajectory crossed the
            # resume target? (Skipped for completion iters — force to the end.)
            if not is_completion:
                current_bp = _best_progress_so_far(t)
                if current_bp > plan.forced_window_max_bp:
                    plan.forced_window_max_bp = current_bp
                if plan.forced_window_max_bp >= (plan.progress_resume_target or 0.0):
                    return None

            # 3b. Safety cap on consecutive forced steps (applies always).
            cap = self.guided_cfg.max_forced_window_length
            if cap is not None and plan.forced_window_steps_emitted >= cap:
                return None

            # 3c. Never force the terminal step — the answer must always come
            # from the student. ``max_depth - 1`` is the last step the rollout
            # will emit; if step_idx is that, hand off so the policy is free
            # to answer or run past the budget. (Completion iters DO force it.)
            if not is_completion and step_idx >= self.max_depth - 1:
                return None

        # ── 4. Query the oracle for the next expert action ──────────────
        expert_action = self.expert_action_fn(
            t.pool, t.slot_id, _dataset_item_for(t),
        )

        # 4.0. Oracle exhausted/failed: the branch-relative expert trajectory
        # has no real action at this step (we ran past its end, or synthesis
        # failed). Do NOT inject a no-op ``direction:0`` forced step — hand off
        # to the student so this step is a genuine on-policy generation
        # ("replace the failed expert step"). Once we return None here the
        # override provider keeps returning None, so the student finishes the
        # trajectory from this pose.
        if not expert_action or not str(expert_action).strip():
            return None

        # 4a. Never force an ``<answer/>`` — even if the oracle returns one,
        # let the student make the call. (Completion iters DO force the answer,
        # as does the global force_expert_answer knob.)
        if (
            not is_completion
            and not self.guided_cfg.force_expert_answer
            and _is_answer_action(expert_action)
        ):
            return None

        # ── 5. Defer the teacher annotation for this forced step ────────
        # The (blocking, HTTP) teacher_annotate_fn call is NOT made here — it is
        # deferred to the pre_step hook, which runs all of a step's annotations
        # concurrently (vLLM batches them) instead of serializing one trajectory
        # at a time. The forced-window counter is incremented now (serial decide
        # phase) so step 3b's cap check stays consistent. The pre_step hook
        # turns the returned _PendingAnnotation into the final
        # (text, token_ids, is_teacher=True, is_expert=True) override.
        plan.forced_window_steps_emitted += 1
        return _PendingAnnotation(dict(
            pool=t.pool,
            slot_id=t.slot_id,
            episode_id=t.episode_id,
            expert_action_formatted=expert_action,
            parent=parent,
            branch_step_index=branch,
        ))

    # ── shared pass runner ───────────────────────────────────────────────

    def _run_one_pass(
        self,
        *,
        items_per_prompt: list[tuple[int, dict]],
        n_per_prompt: int,
        actor_rollout_ref_wg,
        config,
        override_config: dict[str, Any] | None,
        total_slots: int,
        override_provider: Optional[Callable[[Trajectory], Optional[tuple[str, list[int], bool]]]],
        dagger_iter_index: int,
        branch_chain: Optional[list[int]],
        chain_id: int = 0,
    ) -> list[Trajectory]:
        """Run one rollout pass over ``items_per_prompt`` × ``n_per_prompt``.

        If ``override_provider`` is supplied, it is consulted before each
        per-step vLLM call: returning a non-None override skips vLLM for
        that trajectory at that step (used for prefix replay + forced step).
        """
        # Build pending queue
        pending_queue: deque[tuple[int, int, dict]] = deque()
        for group_id, item_data in items_per_prompt:
            for n_idx in range(n_per_prompt):
                pending_queue.append((group_id, n_idx, item_data))

        initial_count = min(len(pending_queue), total_slots)
        initial_batch = [pending_queue.popleft() for _ in range(initial_count)]
        active = self._initialize_batch(initial_batch)
        all_trajs: list[Trajectory] = list(active)

        # Stamp dagger metadata up-front so trajectories carry it from step 0.
        for t in all_trajs:
            t.dagger_iter_index = dagger_iter_index
            t.chain_id = chain_id
            if branch_chain is not None:
                t.branch_chain = list(branch_chain)

        # Install the override provider; the base step loop reads
        # `t.pending_response_override`. We populate it just before each step
        # by hooking into the loop via a temporary subclass-style attribute.
        self._override_provider = override_provider

        try:
            self._run_continuous_episode_loop_with_overrides(
                active, all_trajs, pending_queue,
                actor_rollout_ref_wg, config,
                override_config=override_config,
                dagger_iter_index=dagger_iter_index,
                branch_chain=branch_chain,
                chain_id=chain_id,
            )
        finally:
            self._override_provider = None

        return all_trajs

    def _run_continuous_episode_loop_with_overrides(
        self,
        active: list[Trajectory],
        all_trajectories: list[Trajectory],
        pending_queue,
        actor_rollout_ref_wg,
        config,
        override_config: dict[str, Any] | None,
        dagger_iter_index: int,
        branch_chain: Optional[list[int]],
        chain_id: int = 0,
    ) -> None:
        """Wraps the base loop and installs ``_pre_step_hook`` to populate
        ``pending_response_override`` on each active trajectory before each
        per-step vLLM batch. The base loop calls the hook at the top of
        every iteration (see ``MultiturnEnvRollout._run_continuous_episode_loop``).
        """
        provider = self._override_provider

        annotate_fn = self.teacher_annotate_fn
        max_workers = max(1, int(getattr(self.guided_cfg, "teacher_max_workers", 1) or 1))

        def pre_step(active_now: list[Trajectory]) -> None:
            # Decide phase (serial, cheap): resolve each active trajectory's
            # override. Cheap cases (prefix replay / no-override) are applied
            # immediately; forced steps that need a teacher annotation come back
            # as _PendingAnnotation and are batched below. All plan-state
            # mutations happen here (single-threaded), so only the stateless
            # teacher HTTP calls run concurrently.
            pending: list[tuple[Trajectory, _PendingAnnotation]] = []
            for t in active_now:
                if t.terminated:
                    continue
                # Re-stamp dagger metadata in case the trajectory was just
                # refilled from the pending queue (n_idx > 0 within the iter).
                t.dagger_iter_index = dagger_iter_index
                t.chain_id = chain_id
                if branch_chain is not None:
                    t.branch_chain = list(branch_chain)
                if provider is None:
                    continue
                ov = provider(t)
                if ov is None:
                    continue
                if isinstance(ov, _PendingAnnotation):
                    pending.append((t, ov))
                else:
                    t.pending_response_override = ov

            if not pending:
                return

            # Annotate phase (parallel): fan the deferred teacher calls out so
            # the vLLM teacher server batches the concurrent requests. Without
            # this, a step's N forced annotations run serially (each a slow 32B
            # call), stalling the whole rollout loop and idling the simulators.
            def _run(item: tuple[Trajectory, _PendingAnnotation]):
                t, pend = item
                text, token_ids = annotate_fn(**pend.kwargs)
                return t, text, token_ids

            if len(pending) == 1:
                t, text, token_ids = _run(pending[0])
                t.pending_response_override = (text, list(token_ids), True, True)
                return

            with ThreadPoolExecutor(max_workers=min(len(pending), max_workers)) as ex:
                for fut in as_completed([ex.submit(_run, item) for item in pending]):
                    t, text, token_ids = fut.result()
                    # Fresh oracle step: mask from PG (is_teacher=True) AND
                    # imitate via SFT (is_expert=True).
                    t.pending_response_override = (text, list(token_ids), True, True)

        self._pre_step_hook = pre_step
        try:
            self._run_continuous_episode_loop(
                active, all_trajectories, pending_queue,
                actor_rollout_ref_wg, config,
                override_config=override_config,
            )
        finally:
            self._pre_step_hook = None

# ─── helpers ───────────────────────────────────────────────────────────────


def _is_solved(t: Trajectory) -> bool:
    # Reward >= 1.0 is the convention from ObjectNavEnvAdapter for success.
    return t.reward is not None and t.reward >= 1.0


def _is_answer_action(action_str: str) -> bool:
    """Heuristic match for an ``<answer .../>`` tag, ignoring leading
    whitespace and tolerating either self-closing or paired forms."""
    if not action_str:
        return False
    return action_str.lstrip().startswith("<answer")


def _best_progress_so_far(t: Trajectory) -> float:
    """Best-progress-so-far over a trajectory's recorded distances.

    ``step_distances[k]`` is the pre-action geodesic distance at step ``k``
    (i.e. ``dist_before``). ``step_distances[0]`` is the initial distance.
    bp_k = max(0, 1 − min(step_distances[:k+1]) / step_distances[0]).
    Returns 0.0 when distances are missing / non-positive.
    """
    sd = t.step_distances
    if not sd or sd[0] is None or sd[0] <= 0:
        return 0.0
    valid = [d for d in sd if d is not None]
    if not valid:
        return 0.0
    return max(0.0, 1.0 - min(valid) / sd[0])


def _progress_floor_through(t: Trajectory, through_step: int) -> float:
    """Best-progress-so-far considering only ``step_distances[:through_step+1]``.

    Mirrors pope-dagger's ``_progress_floor_at_step`` semantics: the running
    max of bp from the start of the trajectory through (and including) the
    pre-action observation at ``through_step``.
    """
    sd = t.step_distances
    if not sd or sd[0] is None or sd[0] <= 0:
        return 0.0
    slice_end = min(through_step + 1, len(sd))
    valid = [d for d in sd[:slice_end] if d is not None]
    if not valid:
        return 0.0
    return max(0.0, 1.0 - min(valid) / sd[0])


def _dataset_item_for(t: Trajectory) -> dict:
    """Best-effort retrieval of the dataset item that seeded this trajectory.

    Stashed on the Trajectory by ``_initialize_batch`` via the queue tuple's
    ``item_data``. If the base class doesn't already keep it, this returns
    a stub — callers should attach ``t._dataset_item`` themselves in that
    case. The base ``_initialize_batch`` already calls ``acquire_env`` with
    ``item_data``, so for the v1 expert-action call we rely on the pool
    side knowing the dataset item via its slot state.
    """
    return getattr(t, "_dataset_item", {})


def _default_expert_action_fn(
    pool: Any, slot_id: int, dataset_item: dict
) -> Optional[str]:
    """Default oracle hook: calls pool.compute_expert_action.remote(slot_id).

    The SimulatorPool method wraps
    ``pope_dagger.expert_replay.compute_expert_actions_via_sparsify`` against
    the slot's live environment and returns the next expert action as a
    formatted string, or ``None`` when the expert has no real action to give
    (trajectory exhausted past its end, or oracle synthesis failed).

    Returns ``None`` in that case (and on any RPC error) — the caller
    (``_build_override_for``) treats ``None`` as "hand off to the student for
    this step" rather than injecting a no-op ``direction:0`` forced step that
    would then be teacher-masked + SFT'd.
    """
    try:
        return ray.get(pool.compute_expert_action.remote(slot_id))
    except Exception as e:
        logger.warning(
            f"[guided] expert action computation failed for slot {slot_id}: {e}. "
            f"Handing off to the student for this step."
        )
        return None

