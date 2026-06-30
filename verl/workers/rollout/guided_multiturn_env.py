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

import asyncio
import json
import logging
import math
import os
import re
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

import numpy as np
import ray

from verl.protocol import DataProto
from verl.workers.rollout.forced_depth import (
    forced_g_from_plan,
    step_leave_on_policy,
)
from verl.workers.rollout.multiturn_env import MultiturnEnvRollout, Trajectory

logger = logging.getLogger(__name__)


# Deterministic answer-oriented handoff text (prefix_force_depth, L<=1). No-leak:
# references only the task goal + "should be in view" + the action format; the
# student concludes the answer pixel from its own current frame. Injected instead
# of the base judge's prose, which reliably tells the student to keep approaching.
_ANSWER_HANDOFF_SUMMARY = (
    "I have moved into position and the target object should now be in view in my "
    "current frame. The correct move is to ANSWER it directly — not to explore, "
    "turn, or click the floor to get closer. I will emit <answer> at the target's "
    "location in the image now."
)
_ANSWER_HANDOFF_HINT = (
    "You are now in position and the object you were sent to find should be visible "
    "in your current image. Do NOT explore, turn, or click the floor to approach it "
    "any further. Emit your final <answer> at the target's location in the image now."
)

# Pixel-tuple patterns for the no-leak answer-case sanitizer: an <answer>(x,y) /
# <answer>x=..,y=.. tag, or a bare "(x,y)" coordinate pair. The self-teacher judge
# sees the target in [VISIBLE OBJECTS] and can leak its exact pixel into the hint/
# summary that gets injected into the student's prompt; strip it so the student must
# locate the target in its own view.
_ANSWER_TAG_COORDS_RE = re.compile(r"<answer>\s*\(?[^<)]*\)?\s*</answer>", re.I)
_BARE_COORDS_RE = re.compile(r"\(\s*\d{1,4}\s*,\s*\d{1,4}\s*\)")


def _strip_answer_coords(text: str) -> str:
    """Remove any leaked target pixel from judge-generated answer-case text.

    Replaces ``<answer>(x,y)</answer>`` with a bare ``<answer>`` and drops bare
    ``(x,y)`` tuples, so the injected hint/summary still says "answer the target you
    see" without handing the student the exact coordinates (no-leak invariant)."""
    if not text:
        return text
    text = _ANSWER_TAG_COORDS_RE.sub("<answer>", text)
    text = _BARE_COORDS_RE.sub("", text)
    # Tidy double spaces / stray " with " left by a removed tuple.
    text = re.sub(r"\bwith\s+\.", ".", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


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

    ``group_key`` (optional) lets the pre_step hook deduplicate annotations:
    when several active trajectories share a key (e.g. the N student suffixes of
    one prefixed group, all replaying the SAME deterministic prefix to the SAME
    branch pose), the teacher is called ONCE and the identical text broadcast to
    every sibling — which is required for correctness, not just speed, since the
    teacher is stochastic and the siblings must share a byte-identical
    prefix+guidance for the within-group GRPO comparison to be fair. ``None``
    (the chain orchestration's default) makes every pending its own group.
    ``on_annotated(text, token_ids)`` is invoked once per resolved group so the
    caller can cache the result (e.g. onto a ``PrefixSpec``) for later waves.
    """

    __slots__ = ("kwargs", "group_key", "on_annotated")

    def __init__(self, kwargs: dict, group_key=None, on_annotated=None):
        self.kwargs = kwargs
        self.group_key = group_key
        self.on_annotated = on_annotated


class _PendingHint:
    """Deferred colocated SELF-teacher hint for one prefixed group's branch step
    (``teacher_mode='self'``).

    Unlike ``_PendingAnnotation`` (which yields a forced response that REPLACES
    the student's turn, PG-masked), a ``_PendingHint`` yields NO response
    override. The pre_step hook reconstructs the no-leak branch nav state on the
    sim actor, runs the in-process judge (same engine/weights) to get a
    corrective ``<hint>``, wraps it via ``diagnose.build_diagnosis_block``, and
    sets ``pending_prompt_suffix`` on every sibling so the student re-decides on
    its OWN — the student's response carries the policy gradient (train-on-it).

    All k siblings of a prefixed group share ``group_key`` (== group_id), so the
    judge is run ONCE per group and the identical hint is broadcast to every
    sibling — required for a fair within-group GRPO comparison. ``slot_id`` is
    the representative slot whose replayed prefix defines the (shared) branch
    pose; ``on_resolved(hint_text)`` caches the hint onto the PrefixSpec for
    later waves.
    """

    __slots__ = ("slot_id", "pool", "group_key", "on_resolved", "intended_action",
                 "parent_suffix")

    def __init__(self, slot_id, pool, group_key=None, on_resolved=None,
                 intended_action=None, parent_suffix=None):
        self.slot_id = slot_id
        self.pool = pool
        self.group_key = group_key
        self.on_resolved = on_resolved
        # Gap B: the parent's failed branch-step action text (e.g.
        # "<explore>ground:(756,836)</explore>"). The pool casts its pixel to a
        # world point and adds a VALID "Intent" correction to the judge menu.
        self.intended_action = intended_action
        # Forced-depth: the parent failure's raw responses from branch to end
        # (the failed continuation). The judge diagnoses recurring suffix-level
        # errors (pointing/format) and forms one general corrective hint.
        self.parent_suffix = parent_suffix


class _PendingForcedHandoff:
    """Deferred combined judge call at the LAST forced step (prefix_force_depth).

    At the last of G forced expert steps the colocated judge is run ONCE per
    group to emit BOTH (a) a teacher ``<summary>`` carried into the on-policy
    handoff step under summary_context, and (b) a suffix-derived corrective
    ``<hint>`` cached for injection at the handoff (branch+G). The resolver
    builds the forced response ``<think>..</think><summary>..</summary>{action}``
    for THIS step (PG-masked) and stores the hint block on the PrefixSpec.

    ``expert_action`` is the oracle action to force this step (already fetched in
    the single-threaded decide phase, like ``_PendingAnnotation``).
    ``parent_suffix`` is the parent's failed branch→end responses.
    ``on_resolved(hint_block)`` caches the hint for the handoff step.
    """

    __slots__ = ("slot_id", "pool", "group_key", "on_resolved", "intended_action",
                 "parent_suffix", "expert_action")

    def __init__(self, slot_id, pool, expert_action, group_key=None,
                 on_resolved=None, intended_action=None, parent_suffix=None):
        self.slot_id = slot_id
        self.pool = pool
        self.expert_action = expert_action
        self.group_key = group_key
        self.on_resolved = on_resolved
        self.intended_action = intended_action
        self.parent_suffix = parent_suffix


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
    # "avg_success_step": POPE-DAgger v2 selector — branch one step before the
    #   running average successful step count (``max(0, floor(avg)-1)``), where
    #   ``avg`` is an on-policy EMA updated after every rollout.
    branch_selection_mode: str = "random_regression"
    # EMA smoothing for the v2 ``avg_success_step`` running stat. The new value
    # is ``beta*old + (1-beta)*new``; first observation is set directly.
    avg_success_ema_beta: float = 0.5
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

    # ── prefix-testing orchestration (docs/experiments/pope_dagger_prefixtesting.md) ──
    # "chain" (default — the v2 baseline→iters chain above) or "prefix_groups"
    # (this experiment: split the batch into an on-policy baseline half + a
    # prefixed half where each group is k student suffixes sharing ONE
    # prefix+single-guidance-step, GRPO'd within the group, prefix+guidance
    # masked). See ``_generate_prefix_groups``.
    orchestration_mode: str = "chain"
    # Fraction of the batch's prompts rolled out as on-policy baseline groups;
    # the remaining (1-frac) prompts become prefixed groups built from the
    # baselines' failures. 0.5 = 16 baseline + 16 prefixed at batch_size=32.
    prefix_baseline_fraction: float = 0.5
    # How many distinct failure prefixes to harvest from each baseline group
    # (each becomes one prefixed group). 1 = one prefix per baseline group.
    prefix_failures_per_group: int = 1
    # Whether the single guidance step is SFT-imitated (is_expert=True) in
    # addition to being PG-masked. For pure PrefixRL (the experiment default)
    # this is False — the guidance is pure conditioning, gradient flows only
    # through the student suffix. (sft_coef=0 also nullifies SFT regardless.)
    guidance_is_expert: bool = False
    # How the fixed prefixed-slot budget is split across baseline source groups.
    # "uniform" (legacy): each baseline group donates up to
    # prefix_failures_per_group failures, round-robin. "failure_weighted":
    # reallocate the budget proportional to each group's usable-failure count
    # (worse groups seed more prefixed groups; above-average groups seed none),
    # capped per group by prefix_max_group_alloc_fraction.
    prefix_allocation: str = "uniform"
    # failure_weighted only: max fraction of the total prefixed-slot budget a
    # single baseline group may seed (diversity guard).
    prefix_max_group_alloc_fraction: float = 0.5
    # failure_weighted only: allow a hot group to seed more prefixed groups than
    # it has distinct failures by reusing them (independent teacher annotations).
    prefix_allow_failure_reuse: bool = False

    # ── teacher backend (self-teacher path) ───────────────────────────────
    # MUST be threaded through from the config.py GuidedConfig (main.py builds
    # this slim dataclass field-by-field). When absent, ``_build_prefix_override_for``
    # silently defaults to "server" and routes the branch step through the expert
    # placeholder annotator instead of the colocated self-teacher diagnosis hint.
    teacher_mode: str = "server"
    # Sampling temperature for the colocated self-teacher judge (teacher_mode='self').
    judge_temperature: float = 0.5
    # Prompt-length budget for the self-teacher judge ONLY (see config.py docstring).
    # The shared _tokenize_prompts right-truncates the diagnosis system prompt+menu+image
    # at the student's 1536 cap otherwise. Requires worker.rollout.max_model_len headroom.
    teacher_max_prompt_length: int = 4096

    # ── moving forced-depth controller (prefix_groups + teacher_mode='self') ──
    # See config.py GuidedRolloutConfig for the full docstrings. When
    # prefix_force_depth is True, prefixed groups force G expert steps from the
    # branch (G = max(0, expert_plan_len_from_branch - round(L))), attach a
    # teacher strategy summary to the last forced step, and inject the
    # suffix-derived hint at the handoff (branch+G). L is a global leave-on-policy
    # count controlled toward prefix_depth_target prefixed-group success.
    prefix_force_depth: bool = False
    prefix_depth_target: float = 0.5
    prefix_depth_ema_beta: float = 0.5
    prefix_depth_deadband: float = 0.05
    prefix_min_leave_on_policy: int = 1


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


@dataclass
class PrefixSpec:
    """One prefixed group for the prefix-testing orchestration.

    All ``n_per_prompt`` student suffixes of this group share the SAME prefix
    (the parent failure's steps ``0..branch-1``, replayed deterministically) and
    the SAME single guidance step at ``branch_step_index`` (the teacher's
    strategy-bearing forced step). They diverge only afterward, when the student
    takes over. ``annotation`` caches the resolved ``(text, token_ids)`` of the
    guidance step so siblings reaching the branch in later rollout waves reuse
    the first wave's teacher call (cross-wave dedup; within-wave dedup is handled
    by the pre_step hook's ``group_key`` batching).
    """

    parent: Trajectory
    branch_step_index: int
    group_id: int
    # Resolved guidance-step text + tokens (filled on first annotation).
    annotation: Optional[tuple[str, list[int]]] = None
    # SELF-teacher mode (teacher_mode='self'): the resolved diagnosis hint BLOCK
    # (already wrapped by build_diagnosis_block) cached across rollout waves so
    # siblings reaching the branch later reuse the first wave's judge call. The
    # block is injected as a USER-turn suffix, not a forced response.
    hint_block: Optional[str] = None
    # True once the branch was reached but produced no forceable guidance step
    # (oracle exhausted / answer-only at branch): the whole group degrades to a
    # pure on-policy rollout from the prefix pose, no masking past the prefix.
    no_guidance: bool = False
    # ── moving forced-depth fields (prefix_force_depth) ──────────────────
    # Number of expert steps forced from the branch (G), computed lazily on the
    # first sibling to reach the branch: G = max(0, expert_plan_len - round(L)).
    # None = not yet computed; 0 collapses to the legacy single-hint-at-branch
    # path. ``handoff_step`` = branch_step_index + forced_g is the first
    # on-policy step (where the suffix-derived hint is injected).
    forced_g: Optional[int] = None
    handoff_step: Optional[int] = None
    # True once the last-forced-step combined judge call (summary+hint) resolved
    # and cached its hint on ``hint_block`` — guards against re-resolving.
    handoff_resolved: bool = False


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
        # Prove the teacher backend reached the rollout. If teacher_mode is not
        # threaded through main.py's slim GuidedConfig copy this silently reads
        # "server" and the self-teacher hint path is never taken (placeholder
        # expert annotation is used instead).
        logger.info(
            "[guided] rollout init: teacher_mode=%s judge_temperature=%s "
            "orchestration=%s (annotate_fn=%s)",
            getattr(self.guided_cfg, "teacher_mode", "server"),
            getattr(self.guided_cfg, "judge_temperature", None),
            getattr(self.guided_cfg, "orchestration_mode", "chain"),
            "default-placeholder" if teacher_annotate_fn is None else "wired",
        )
        # POPE-DAgger v2 running stat: on-policy EMA of the step count at which
        # successful baseline (dagger_iter==0) trajectories solved. Drives the
        # ``avg_success_step`` branch selector. Persists across rollouts on this
        # rollout-worker instance, so it accumulates over training steps.
        #
        # NOTE: this EMA is process-local and is NOT part of the verl checkpoint.
        # On resume (NCCL hang restart, preemption, or — for a >48h run — the
        # inevitable walltime-limit re-queue) it would otherwise reset to 0.0,
        # forcing the first resumed rollout to branch_step=0 (guidance injected
        # at step 0 instead of ~avg_success-1) and wasting that step's guidance.
        # ``POPE_INIT_AVG_SUCCESS_STEP`` seeds it to the value the EMA held at the
        # end of the resumed-from checkpoint, so step N resumes with the correct
        # branch_step immediately. Read once at construction.
        self._avg_success_step_count: float = 0.0
        self._avg_success_seen: bool = False
        _seed = os.environ.get("POPE_INIT_AVG_SUCCESS_STEP", "").strip()
        if _seed:
            try:
                _seed_val = float(_seed)
            except ValueError:
                logger.warning(
                    f"[guided] ignoring non-numeric "
                    f"POPE_INIT_AVG_SUCCESS_STEP={_seed!r}"
                )
            else:
                if _seed_val > 0.0:
                    self._avg_success_step_count = _seed_val
                    self._avg_success_seen = True
                    logger.info(
                        f"[guided] seeded avg_success_step_count="
                        f"{_seed_val:.2f} from POPE_INIT_AVG_SUCCESS_STEP "
                        f"(resume EMA restore; first branch_step="
                        f"{max(0, int(_seed_val) - 1)})"
                    )

        # ── moving forced-depth controller state (prefix_force_depth) ──────
        # L = number of trailing on-policy steps left to the student from the
        # branch. Per-task forced count G = max(0, expert_plan_len - round(L)).
        # L starts at prefix_min_leave_on_policy (force all but the trailing
        # leave-count) and is nudged each rollout toward prefix_depth_target
        # prefixed-group success via an EMA. Process-local (not checkpointed);
        # POPE_INIT_LEAVE_ON_POLICY seeds it on resume.
        self._leave_on_policy: float = float(
            getattr(self.guided_cfg, "prefix_min_leave_on_policy", 1)
        )
        self._prefix_success_ema: float = 0.0
        self._prefix_success_ema_seen: bool = False
        _seed_L = os.environ.get("POPE_INIT_LEAVE_ON_POLICY", "").strip()
        if _seed_L:
            try:
                _seed_Lv = float(_seed_L)
            except ValueError:
                logger.warning(
                    f"[guided] ignoring non-numeric "
                    f"POPE_INIT_LEAVE_ON_POLICY={_seed_L!r}"
                )
            else:
                if _seed_Lv >= 1.0:
                    self._leave_on_policy = _seed_Lv
                    logger.info(
                        f"[guided] seeded leave_on_policy={_seed_Lv:.2f} "
                        f"from POPE_INIT_LEAVE_ON_POLICY (resume restore)"
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

        # ── prefix-testing orchestration dispatch ────────────────────────
        if getattr(self.guided_cfg, "orchestration_mode", "chain") == "prefix_groups":
            return self._generate_prefix_groups(
                items_by_group=items_by_group,
                n_per_prompt=n_per_prompt,
                actor_rollout_ref_wg=actor_rollout_ref_wg,
                config=config,
                override_config=override_config,
                total_slots=total_slots,
                t_start=t_start,
                metrics=metrics,
            )

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

        # ── flat async orchestration ─────────────────────────────────────
        # Per-group chain coroutines sharing ONE event loop: every group's
        # chain sequence (baseline → iters → next chain) advances
        # independently, with vLLM continuous batching + the simulator-slot
        # semaphore mixing all in-flight trajectories. Removes the
        # sequential-pass barriers of the loop below (which cost step 1 of
        # the summary sanity ~20 barrier-synced passes ≈ 7100s gen vs the
        # vanilla single pass ≈ 990s on the same task).
        if bool(getattr(config.worker.rollout, "async_mode", False)):
            all_trajs, chains_total = self._generate_chains_flat_async(
                items_by_group=items_by_group,
                n_per_prompt=n_per_prompt,
                actor_rollout_ref_wg=actor_rollout_ref_wg,
                config=config,
                override_config=override_config,
                total_slots=total_slots,
                global_max_iters=global_max_iters,
                max_chains_per_group=max_chains_per_group,
            )
            return self._finalize_guided_batch(
                all_trajs, chains_total, n_per_prompt, t_start, metrics
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

        return self._finalize_guided_batch(
            all_trajs, chain_idx, n_per_prompt, t_start, metrics
        )

    def _finalize_guided_batch(
        self,
        all_trajs: list[Trajectory],
        chains_spawned: int,
        n_per_prompt: int,
        t_start: float,
        metrics: dict[str, Any],
    ) -> DataProto:
        """Shared tail of the barriered + flat-async orchestrations: n_idx
        reassignment, metrics, batch build, trajectory cleanup."""
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
        metrics["guided/chains_spawned"] = float(chains_spawned)
        metrics["guided/avg_chain_id"] = float(
            np.mean([t.chain_id for t in all_trajs])
        )
        # The honest apples-to-apples number vs a vanilla GRPO run: mean reward
        # over the on-policy baselines only (dagger_iter 0). The blended
        # env/avg_reward above is inflated by forced/guided successes.
        onpolicy = [
            t.reward if t.reward is not None else 0.0
            for t in all_trajs
            if t.dagger_iter_index == 0
        ]
        if onpolicy:
            metrics["guided/onpolicy_avg_reward"] = float(np.mean(onpolicy))
            metrics["guided/onpolicy_n"] = float(len(onpolicy))

        # ── POPE-DAgger v2 stats / required visualizations ──────────────────
        # Partition trajectories into the four design categories. "unguided" =
        # on-policy baseline (dagger_iter==0); "guided" = a continuation after a
        # forced branch (dagger_iter>=1). success = reward>=1.0 (_is_solved).
        unguided = [t for t in all_trajs if t.dagger_iter_index == 0]
        guided = [t for t in all_trajs if t.dagger_iter_index >= 1]
        n_un = len(unguided)
        n_gu = len(guided)
        un_succ = [t for t in unguided if _is_solved(t)]
        gu_succ = [t for t in guided if _is_solved(t)]
        n_total = max(1, len(all_trajs))

        # Pure on-policy successes per iteration (count + rate).
        metrics["guided/onpolicy_successes"] = float(len(un_succ))
        onp_rate = (len(un_succ) / n_un) if n_un else 0.0
        metrics["guided/onpolicy_success_rate"] = float(onp_rate)
        # Overall success rate (all trajectories) over time.
        metrics["guided/success_rate"] = float(
            sum(1 for t in all_trajs if _is_solved(t)) / n_total
        )

        # Guidance boost: improvement in success rate after guidance.
        gu_rate = (len(gu_succ) / n_gu) if n_gu else 0.0
        metrics["guided/guided_success_rate"] = float(gu_rate)
        # Percentage-point lift and relative-percent lift over the on-policy rate.
        metrics["guided/guidance_boost_pp"] = float((gu_rate - onp_rate) * 100.0)
        if onp_rate > 0.0:
            metrics["guided/guidance_boost_pct"] = float(
                (gu_rate - onp_rate) / onp_rate * 100.0
            )

        # Four-category group ratios (fraction of all trajectories).
        metrics["guided/ratio_guided_success"] = float(len(gu_succ) / n_total)
        metrics["guided/ratio_guided_failure"] = float((n_gu - len(gu_succ)) / n_total)
        metrics["guided/ratio_unguided_success"] = float(len(un_succ) / n_total)
        metrics["guided/ratio_unguided_failure"] = float((n_un - len(un_succ)) / n_total)

        # ── update the on-policy avg_success_step_count EMA (drives the v2
        # branch selector next rollout). Only baseline (on-policy) successes
        # count — the step at which the policy solved on its own. ──
        succ_steps = [int(t.num_steps) for t in un_succ if t.num_steps is not None]
        if succ_steps:
            new_obs = float(np.mean(succ_steps))
            beta = float(self.guided_cfg.avg_success_ema_beta)
            if not self._avg_success_seen:
                self._avg_success_step_count = new_obs
                self._avg_success_seen = True
            else:
                self._avg_success_step_count = (
                    beta * self._avg_success_step_count + (1.0 - beta) * new_obs
                )
        # Always log the current running value (constant across a rollout, but
        # the time series over training steps is the required visualization).
        metrics["guided/avg_success_step_count"] = float(self._avg_success_step_count)

        # ── moving forced-depth controller update (prefix_force_depth) ──────
        # Drive the global leave-on-policy count L toward prefix_depth_target
        # prefixed-group success. The prefixed groups ARE the guided (dagger
        # iter>=1) continuations, so gu_rate is the controlled signal. EMA-smooth
        # it, then: success > target+deadband → student is coping, leave MORE on
        # policy (L+=1, force fewer steps); success < target-deadband → too hard,
        # force MORE (L-=1, floor at prefix_min_leave_on_policy). One update per
        # rollout/training iteration.
        if getattr(self.guided_cfg, "prefix_force_depth", False) and n_gu > 0:
            (
                self._leave_on_policy,
                self._prefix_success_ema,
                self._prefix_success_ema_seen,
            ) = step_leave_on_policy(
                self._leave_on_policy,
                self._prefix_success_ema,
                self._prefix_success_ema_seen,
                gu_rate,
                beta=float(self.guided_cfg.prefix_depth_ema_beta),
                target=float(self.guided_cfg.prefix_depth_target),
                deadband=float(self.guided_cfg.prefix_depth_deadband),
                floor=float(self.guided_cfg.prefix_min_leave_on_policy),
            )
            logger.info(
                f"[guided] forced-depth controller: gu_rate={gu_rate:.3f} "
                f"ema={self._prefix_success_ema:.3f} "
                f"target={float(self.guided_cfg.prefix_depth_target):.2f} "
                f"-> leave_on_policy L={self._leave_on_policy:.1f}"
            )
        metrics["guided/leave_on_policy"] = float(self._leave_on_policy)
        metrics["guided/prefix_success_ema"] = float(self._prefix_success_ema)

        # Taper signal: fraction of trajectories carrying any forced step.
        n_guided = sum(1 for t in all_trajs if any(t.step_teacher_forced or []))
        metrics["guided/guided_traj_frac"] = (
            float(n_guided) / max(1, len(all_trajs))
        )

        logger.info(
            f"[guided] rollout complete: {len(all_trajs)} trajs in "
            f"{time.time() - t_start:.1f}s | avg_steps={avg_steps:.1f} | "
            f"avg_reward={avg_reward:.3f} | "
            f"onpolicy_avg_reward={metrics.get('guided/onpolicy_avg_reward', float('nan')):.3f} "
            f"(n={len(onpolicy)})"
        )

        self.destroy_controllers()

        # Per-trajectory outcome dump (follow-detection + guided-vs-baseline
        # success). Must run BEFORE the cleanup loop clears step_responses.
        try:
            from verl.utils.pope_dagger_dump import dump_trajectory_outcomes
            dump_trajectory_outcomes(
                all_trajs,
                rollout_idx=int(getattr(self, "_rollout_idx", 0)),
                rank=getattr(self, "rank", 0),
            )
        except Exception:  # pragma: no cover — diagnostics must never crash
            pass

        batch = self._build_final_batch(
            all_trajs, all_rewards, all_ground_truths, n_per_prompt
        )
        for t in all_trajs:
            t.last_images.clear()
            t.last_prompt = None
            t.step_responses.clear()
            t.step_response_token_ids.clear()
            t.step_response_log_probs.clear()
            t.async_pre_step_fn = None
        all_trajs.clear()
        return batch

    # ── prefix-testing orchestration ──────────────────────────────────────

    def _generate_prefix_groups(
        self,
        *,
        items_by_group: dict[int, dict],
        n_per_prompt: int,
        actor_rollout_ref_wg,
        config,
        override_config: dict[str, Any] | None,
        total_slots: int,
        t_start: float,
        metrics: dict[str, Any],
    ) -> DataProto:
        """Prefix-testing orchestration (pope_dagger_prefixtesting.md).

        One GRPO iteration, two passes over a split batch:

        1. **Baseline half** — the first ``prefix_baseline_fraction`` of the
           batch's prompts are rolled out fully on-policy (``dagger_iter=0``).
           These are the honest apples-to-apples groups vs a vanilla GRPO run.
        2. **Prefixed half** — from each baseline group's *failures* we harvest a
           prefix (steps ``0..branch-1``) and inject ONE strategy-bearing
           guidance step at ``branch`` (avg_success_step heuristic). Each such
           ``PrefixSpec`` becomes a fresh group whose ``n_per_prompt`` student
           suffixes all share that prefix+guidance (replayed deterministically,
           deduped to a single teacher call) and diverge only afterward
           (``dagger_iter=1``). Prefix+guidance are PG-masked; GRPO normalizes
           within each group. If failures are too few, remaining prefixed slots
           are padded with fresh on-policy baseline groups (``dagger_iter=0``).

        Both passes run in lockstep (sync) mode so the per-group guidance dedup
        in the pre_step hook is deterministic — set ``rollout.async_mode=false``.
        """
        # Monotonic rollout counter shared by the per-group guidance dump
        # (_log_prefixed_prompts) and the per-trajectory outcome dump
        # (_finalize_guided_batch) so they join on (rollout_idx, group_id).
        self._rollout_idx = int(getattr(self, "_rollout_idx", -1)) + 1

        n_prompts = len(items_by_group)
        frac = float(self.guided_cfg.prefix_baseline_fraction)
        n_baseline = max(1, min(n_prompts - 1, int(round(n_prompts * frac))))
        n_prefixed = n_prompts - n_baseline
        baseline_gids = list(range(n_baseline))
        prefixed_gids = list(range(n_baseline, n_prompts))

        logger.info(
            f"[guided.prefix] batch split: {n_baseline} baseline groups + "
            f"{n_prefixed} prefixed groups (n_per_prompt={n_per_prompt}); "
            f"avg_success_step_count={self._avg_success_step_count:.2f}"
        )

        # ── Pass A: on-policy baseline half ──────────────────────────────
        baseline_trajs = self._run_one_pass(
            items_per_prompt=[(g, items_by_group[g]) for g in baseline_gids],
            n_per_prompt=n_per_prompt,
            actor_rollout_ref_wg=actor_rollout_ref_wg,
            config=config,
            override_config=override_config,
            total_slots=total_slots,
            override_provider=None,
            dagger_iter_index=0,
            branch_chain=[],
            chain_id=0,
        )
        for t in baseline_trajs:
            t.dagger_iter_index = 0
            t.branch_chain = []
            t.chain_id = 0
        all_trajs: list[Trajectory] = list(baseline_trajs)
        logger.info(
            f"[guided.prefix] baseline pass: {len(baseline_trajs)} trajectories"
        )

        # ── Select prefixes from baseline failures ───────────────────────
        branch_step = max(0, int(self._avg_success_step_count) - 1)
        plans = self._select_prefixes(
            baseline_trajs=baseline_trajs,
            prefixed_gids=prefixed_gids,
            branch_step=branch_step,
        )
        n_real_prefixed = len(plans)
        pad_gids = [g for g in prefixed_gids if g not in plans]
        metrics["guided/prefix_branch_step"] = float(branch_step)
        metrics["guided/prefix_groups"] = float(n_real_prefixed)
        metrics["guided/prefix_pad_groups"] = float(len(pad_gids))
        logger.info(
            f"[guided.prefix] selected {n_real_prefixed} prefixed groups "
            f"(branch_step={branch_step}); padding {len(pad_gids)} with on-policy"
        )

        # ── Pass B1: prefixed half (replay prefix + single guidance step) ─
        # Each prefixed group is seeded with its PARENT's task item (the parent
        # may come from any baseline group; replaying its recorded actions only
        # reaches the same branch pose in the parent's own task). For the
        # single-task sanity every item is identical, but this keeps the general
        # multi-task case correct.
        if plans:
            prefixed_trajs = self._run_one_pass(
                items_per_prompt=[
                    (g, items_by_group[plans[g].parent.group_id]) for g in plans.keys()
                ],
                n_per_prompt=n_per_prompt,
                actor_rollout_ref_wg=actor_rollout_ref_wg,
                config=config,
                override_config=override_config,
                total_slots=total_slots,
                override_provider=lambda t, _p=plans: self._build_prefix_override_for(t, _p),
                dagger_iter_index=1,
                branch_chain=None,
                chain_id=1,
            )
            for t in prefixed_trajs:
                spec = plans.get(t.group_id)
                t.dagger_iter_index = 1
                t.chain_id = 1
                t.branch_chain = [spec.branch_step_index] if spec else []
            all_trajs.extend(prefixed_trajs)
            logger.info(
                f"[guided.prefix] prefixed pass: {len(prefixed_trajs)} trajectories"
            )

        # ── Pass B2: pad unfilled prefixed slots with on-policy groups ────
        if pad_gids:
            pad_trajs = self._run_one_pass(
                items_per_prompt=[(g, items_by_group[g]) for g in pad_gids],
                n_per_prompt=n_per_prompt,
                actor_rollout_ref_wg=actor_rollout_ref_wg,
                config=config,
                override_config=override_config,
                total_slots=total_slots,
                override_provider=None,
                dagger_iter_index=0,
                branch_chain=[],
                chain_id=0,
            )
            for t in pad_trajs:
                t.dagger_iter_index = 0
                t.branch_chain = []
                t.chain_id = 0
            all_trajs.extend(pad_trajs)
            logger.info(
                f"[guided.prefix] padding pass: {len(pad_trajs)} on-policy trajectories"
            )

        # Dump the prefixed prompts (prefix steps + guidance) for analysis.
        self._log_prefixed_prompts(plans, metrics)

        return self._finalize_guided_batch(
            all_trajs, chains_spawned=1, n_per_prompt=n_per_prompt,
            t_start=t_start, metrics=metrics,
        )

    def _select_prefixes(
        self,
        *,
        baseline_trajs: list[Trajectory],
        prefixed_gids: list[int],
        branch_step: int,
    ) -> dict[int, PrefixSpec]:
        """Harvest one prefix per requested prefixed slot from baseline failures.

        Groups baselines by their source group, takes up to
        ``prefix_failures_per_group`` failures (reward < 1.0) from each, and
        assigns them round-robin to the prefixed group ids. A failure is only
        usable if it ran at least ``branch_step`` steps (so the prefix has the
        steps to replay); shorter failures are skipped. Returns a dict
        ``prefixed_group_id -> PrefixSpec``; unfilled slots are left out (the
        caller pads them on-policy).
        """
        by_group: dict[int, list[Trajectory]] = {}
        for t in baseline_trajs:
            by_group.setdefault(t.group_id, []).append(t)

        # Per-parent mistake-aware branch selection (Gap A): when
        # ``branch_selection_mode == "random_regression"`` each parent gets its
        # OWN branch step sampled from its progress-regression region (reusing
        # the V5 selector), instead of the global ``avg_success_step`` scalar
        # that collapses to 0 on a base model that rarely succeeds.
        mode = str(self.guided_cfg.branch_selection_mode)
        # Usable failures per source group (failed AND ran far enough to replay
        # the prefix), preserving rollout order within each group.
        # random_regression needs a >=2-step progress series; the scalar modes
        # need the parent to reach the fixed branch step.
        min_steps = 2 if mode == "random_regression" else branch_step
        usable: dict[int, list[Trajectory]] = {
            g: [
                t for t in by_group[g]
                if not _is_solved(t) and len(t.step_responses) >= min_steps
            ]
            for g in sorted(by_group)
        }

        if str(self.guided_cfg.prefix_allocation) == "failure_weighted":
            candidates = self._failure_weighted_candidates(usable, len(prefixed_gids))
        else:
            k = max(1, int(self.guided_cfg.prefix_failures_per_group))
            # Uniform (legacy): up to k failures per group, interleaved across
            # source groups so the prefixes are diverse rather than all from one.
            per_group_failures = [fails[:k] for fails in usable.values()]
            candidates = []
            depth = max((len(f) for f in per_group_failures), default=0)
            for d in range(depth):
                for fails in per_group_failures:
                    if d < len(fails):
                        candidates.append(fails[d])

        plans: dict[int, PrefixSpec] = {}
        for gid, parent in zip(prefixed_gids, candidates):
            if mode == "random_regression":
                # Mistake-region branch (per parent): sample a step in the
                # parent's [peak..last_unrecovered] progress regression. No
                # forced windows yet → the whole trajectory is the suffix.
                from pope_dagger.analyzer import BranchSelectorState
                b = self._select_branch_random_regression(
                    parent=parent, state=BranchSelectorState(),
                    group_id=gid, iter_k=1, chain_id=1,
                )
                if b is None:
                    # No usable progress series — leave this slot for on-policy
                    # padding rather than forcing a degenerate step-0 branch.
                    continue
            else:
                b = branch_step
            # Clamp the branch to the parent's available prefix length; a branch
            # past the parent's end has no guidance pose to replay to.
            b = min(b, len(parent.step_responses))
            plans[gid] = PrefixSpec(
                parent=parent, branch_step_index=b, group_id=gid,
            )
        return plans

    def _failure_weighted_candidates(
        self,
        usable: dict[int, list[Trajectory]],
        budget: int,
    ) -> list[Trajectory]:
        """Allocate the fixed prefixed-slot ``budget`` across source groups
        proportional to each group's usable-failure count.

        A group that fails more seeds more prefixed groups; a group at/above the
        on-policy average (few failures) seeds few or none; a zero-failure group
        seeds nothing. Allocation per group is capped at
        ``prefix_max_group_alloc_fraction * budget`` (diversity guard) and, with
        ``prefix_allow_failure_reuse=False``, at the group's distinct-failure
        supply. Integer allocation: floor of the proportional target, then leftover
        slots to the uncapped group with the largest failure count (ties → lower
        group id). Returns the flattened candidate failures (interleaved across
        groups for ordering diversity); unfilled budget falls through to padding.
        """
        groups = sorted(usable)  # deterministic order
        counts = {g: len(usable[g]) for g in groups}
        total_fail = sum(counts.values())
        if budget <= 0 or total_fail == 0:
            return []  # all-success / nothing to harvest → caller pads on-policy

        reuse = bool(self.guided_cfg.prefix_allow_failure_reuse)
        cap_frac = float(self.guided_cfg.prefix_max_group_alloc_fraction)
        # Per-group ceiling: diversity cap, and (unless reuse) the supply of
        # distinct failures. cap of 0 only when the group has no failures.
        diversity_cap = max(1, math.ceil(cap_frac * budget))
        cap = {
            g: (diversity_cap if reuse else min(diversity_cap, counts[g]))
            if counts[g] > 0 else 0
            for g in groups
        }

        # Largest-remainder base = floor of proportional target, clamped to cap.
        alloc = {
            g: min(int((budget * counts[g]) // total_fail), cap[g])
            for g in groups
        }
        # Distribute leftover slots to the uncapped group with the largest raw
        # failure count (faithful to "worst gets most"); among equally-failing
        # groups prefer the one allocated least so far (round-robin for prefix
        # diversity), then the lower group id.
        remaining = budget - sum(alloc.values())
        while remaining > 0:
            pool = [g for g in groups if alloc[g] < cap[g]]
            if not pool:
                break  # every group at its cap → shortfall pads on-policy
            g = max(pool, key=lambda g: (counts[g], -alloc[g], -g))
            alloc[g] += 1
            remaining -= 1

        # Materialize candidates, interleaved across groups so consecutive
        # prefixed gids draw from different source tasks where possible.
        picks: dict[int, list[Trajectory]] = {}
        for g in groups:
            n = alloc[g]
            fails = usable[g]
            if not n:
                continue
            picks[g] = (
                [fails[i % len(fails)] for i in range(n)]  # reuse: cycle
                if reuse else fails[:n]
            )
        candidates: list[Trajectory] = []
        depth = max((len(v) for v in picks.values()), default=0)
        for d in range(depth):
            for g in groups:
                v = picks.get(g)
                if v and d < len(v):
                    candidates.append(v[d])
        return candidates

    def _build_prefix_override_for(
        self,
        t: Trajectory,
        plans: dict[int, PrefixSpec],
    ) -> Optional[tuple[str, list[int], bool, bool]] | _PendingAnnotation:
        """``override_provider`` for the prefixed pass.

        - ``step < branch``: deterministic prefix replay from the parent failure
          (PG-masked conditioning, never SFT'd).
        - ``step == branch``: the SINGLE guidance step. If already annotated for
          this group, return the cached text; else defer a teacher call
          (deduped per group by the pre_step hook). Oracle exhaustion / an
          answer-only branch degrades the group to a pure on-policy suffix.
        - ``step > branch``: the student runs free (return None).
        """
        spec = plans.get(t.group_id)
        if spec is None:
            return None
        step_idx = t.num_steps
        branch = spec.branch_step_index
        parent = spec.parent

        # ── 1. Prefix replay ────────────────────────────────────────────
        if step_idx < branch:
            if step_idx >= len(parent.step_responses):
                return None
            text = parent.step_responses[step_idx]
            token_ids = parent.step_response_token_ids[step_idx]
            # Masked from PG (is_teacher=True), not SFT'd (is_expert=False):
            # replayed on-policy tokens, conditioning only.
            return text, list(token_ids), True, False

        # ── 1b. Moving forced-depth path (prefix_force_depth + self-teacher) ─
        # Force G expert steps from the branch, attach a teacher summary to the
        # last forced step, and inject the suffix-derived hint at the handoff.
        fd_on = (
            bool(getattr(self.guided_cfg, "prefix_force_depth", False))
            and str(getattr(self.guided_cfg, "teacher_mode", "server")) == "self"
        )
        if fd_on and step_idx >= branch and not spec.no_guidance:
            return self._build_forced_depth_override(t, spec, step_idx)

        # ── 2. Single guidance step at the branch ───────────────────────
        if step_idx == branch and not spec.no_guidance:
            # SELF-teacher: inject a no-leak diagnosis HINT as a user-turn suffix
            # (no forced response; the student re-decides on its own). The hint
            # is resolved by the pre_step hook via the in-process judge.
            if str(getattr(self.guided_cfg, "teacher_mode", "server")) == "self":
                if spec.hint_block is not None:
                    # Cross-wave cache hit: inject directly, no override.
                    t.pending_prompt_suffix = spec.hint_block
                    return None

                def _cache_hint(hint_block, _s=spec):
                    _s.hint_block = hint_block

                # Gap B: the parent's OWN action at the branch step is the
                # pointing intent we want to validate/correct. The pose was
                # replayed up to branch-1, so the slot now stands exactly where
                # the parent emitted this action — its ground pixel casts from
                # the same camera. Pass the raw text; the pool parses+casts it.
                intended_action = (
                    parent.step_responses[branch]
                    if 0 <= branch < len(parent.step_responses)
                    else None
                )

                return _PendingHint(
                    slot_id=t.slot_id,
                    pool=t.pool,
                    group_key=t.group_id,
                    on_resolved=_cache_hint,
                    intended_action=intended_action,
                )
            if spec.annotation is not None:
                text, token_ids = spec.annotation
                return text, list(token_ids), True, bool(self.guided_cfg.guidance_is_expert)
            expert_action = self.expert_action_fn(
                t.pool, t.slot_id, _dataset_item_for(t),
            )
            if not expert_action or not str(expert_action).strip():
                spec.no_guidance = True
                return None
            if (
                not self.guided_cfg.force_expert_answer
                and _is_answer_action(expert_action)
            ):
                spec.no_guidance = True
                return None

            def _cache(text, token_ids, _s=spec):
                _s.annotation = (text, list(token_ids))

            return _PendingAnnotation(
                dict(
                    pool=t.pool,
                    slot_id=t.slot_id,
                    episode_id=t.episode_id,
                    expert_action_formatted=expert_action,
                    parent=parent,
                    branch_step_index=branch,
                ),
                group_key=t.group_id,
                on_annotated=_cache,
            )

        # ── 3. Student suffix (step > branch, or no-guidance fallback) ───
        return None

    def _build_forced_depth_override(
        self, t: Trajectory, spec: PrefixSpec, step_idx: int,
    ) -> Optional[tuple[str, list[int], bool, bool]] | "_PendingAnnotation" | "_PendingHint" | "_PendingForcedHandoff":
        """Override provider for the moving forced-depth path (self-teacher).

        Layout from the branch ``b`` with forced count ``G`` (handoff ``h=b+G``):
          - ``b <= step < h-1``: intermediate forced expert step (templated
            summary; PG-masked). ``_PendingAnnotation``.
          - ``step == h-1`` (last forced step, only if G>=1): combined judge call
            emitting the teacher ``<summary>`` (carried into the handoff under
            summary_context) + the suffix-derived ``<hint>`` (cached for ``h``).
            ``_PendingForcedHandoff``.
          - ``G == 0`` (h==b): no forced steps; inject a suffix-derived hint at
            the branch (``_PendingHint``, want_summary=False).
          - ``step == h``: inject the cached hint block as a user-turn suffix.
          - ``step > h``: student runs free.
        """
        branch = spec.branch_step_index
        parent = spec.parent

        # Lazily compute G on the first sibling to reach the branch (serial decide
        # phase → no race). compute_expert_plan_length builds the branch-relative
        # oracle cache and returns actions-remaining incl. the terminal answer.
        if spec.forced_g is None:
            plan_len = None
            try:
                plan_len = ray.get(
                    t.pool.compute_expert_plan_length.remote(t.slot_id)
                )
            except Exception as e:
                logger.warning(
                    f"[guided.fd] grp {t.group_id}: plan-length failed: {e!r}"
                )
            g = forced_g_from_plan(plan_len, self._leave_on_policy)
            spec.forced_g = g
            spec.handoff_step = branch + g
            logger.info(
                f"[guided.fd] grp {t.group_id}: plan_len={plan_len} "
                f"L={self._leave_on_policy:.1f} -> forced_g={g} "
                f"handoff={spec.handoff_step}"
            )

        g = int(spec.forced_g)
        handoff = int(spec.handoff_step)

        # Parent's failed continuation from the branch to the end — the suffix the
        # judge diagnoses for recurring (pointing/format) errors.
        parent_suffix = list(parent.step_responses[branch:])
        # Gap B: the parent's own branch action (pointing intent to validate).
        intended_action = (
            parent.step_responses[branch]
            if 0 <= branch < len(parent.step_responses)
            else None
        )

        # ── handoff and beyond: inject the cached hint once, then run free ──
        if step_idx >= handoff:
            if step_idx == handoff and spec.hint_block is not None:
                t.pending_prompt_suffix = spec.hint_block
            return None

        # ── G == 0: branch IS the handoff; resolve a suffix-derived hint here.
        # (Unreached: handoff==branch makes step_idx>=handoff true above. Kept
        # explicit for clarity — the cross-wave miss is handled by issuing a
        # _PendingHint when hint_block is still None at the branch.)
        if g == 0:
            if spec.hint_block is not None:
                t.pending_prompt_suffix = spec.hint_block
                return None

            def _cache_hint0(hint_block, _s=spec):
                _s.hint_block = hint_block
                _s.handoff_resolved = True

            return _PendingHint(
                slot_id=t.slot_id, pool=t.pool, group_key=t.group_id,
                on_resolved=_cache_hint0, intended_action=intended_action,
                parent_suffix=parent_suffix,
            )

        # ── forced region (G >= 1): fetch this step's expert action ──
        expert_action = self.expert_action_fn(
            t.pool, t.slot_id, _dataset_item_for(t),
        )
        if not expert_action or not str(expert_action).strip():
            # Oracle exhausted early → hand off here (treat as handoff).
            spec.handoff_step = step_idx
            return None
        if (
            not self.guided_cfg.force_expert_answer
            and _is_answer_action(expert_action)
        ):
            # Reached the terminal answer earlier than planned → hand off so the
            # answer step is always student-generated.
            spec.handoff_step = step_idx
            return None

        is_last_forced = step_idx == handoff - 1
        if is_last_forced:
            # Combined judge call: teacher summary (this step) + suffix hint (h).
            def _cache_handoff(hint_block, _s=spec):
                _s.hint_block = hint_block
                _s.handoff_resolved = True

            return _PendingForcedHandoff(
                slot_id=t.slot_id, pool=t.pool, expert_action=expert_action,
                group_key=t.group_id, on_resolved=_cache_handoff,
                intended_action=intended_action, parent_suffix=parent_suffix,
            )

        # Intermediate forced step: templated summary, no judge call.
        def _cache_inter(text, token_ids):
            # No cross-step caching (each forced step is pose-specific).
            return

        return _PendingAnnotation(
            dict(
                pool=t.pool,
                slot_id=t.slot_id,
                episode_id=t.episode_id,
                expert_action_formatted=expert_action,
                parent=parent,
                branch_step_index=branch,
            ),
            group_key=t.group_id,
            on_annotated=_cache_inter,
        )

    def _log_prefixed_prompts(
        self,
        plans: dict[int, PrefixSpec],
        metrics: dict[str, Any],
    ) -> None:
        """Persist the prefixed prompts (replayed prefix + guidance step) so the
        guidance behavior can be audited. Controlled by the ``POPE_PREFIX_DUMP``
        env var (a directory); a per-rollout JSONL of one record per prefixed
        group is appended. Best-effort: never raises into the rollout.
        """
        dump_dir = os.environ.get("POPE_PREFIX_DUMP", "").strip()
        if not dump_dir or not plans:
            return
        try:
            os.makedirs(dump_dir, exist_ok=True)
            recs = []
            for gid, spec in sorted(plans.items()):
                parent = spec.parent
                b = spec.branch_step_index
                guidance_text = (
                    spec.annotation[0] if spec.annotation is not None else None
                )
                # In self-teacher mode the injected directive lives on
                # ``hint_block`` (a prompt suffix), not ``annotation``. Record it
                # so the actual judge hint is auditable from the dump.
                hint_block = getattr(spec, "hint_block", None)
                recs.append({
                    "rollout_idx": int(getattr(self, "_rollout_idx", 0)),
                    "group_id": gid,
                    "branch_step": b,
                    "parent_episode_id": getattr(parent, "episode_id", None),
                    "parent_num_steps": int(getattr(parent, "num_steps", 0) or 0),
                    "parent_reward": float(parent.reward) if parent.reward is not None else None,
                    "prefix_responses": list(parent.step_responses[:b]),
                    "guidance_text": guidance_text,
                    "hint_block": hint_block,
                    "no_guidance": bool(spec.no_guidance),
                    # Forced-depth fields (None when prefix_force_depth is off).
                    "forced_g": spec.forced_g,
                    "handoff_step": spec.handoff_step,
                    "parent_suffix": list(parent.step_responses[b:]),
                })
            n_with_guidance = sum(
                1 for r in recs if r["guidance_text"] or r["hint_block"]
            )
            metrics["guided/prefix_with_guidance"] = float(n_with_guidance)
            # One file per rollout-worker; appended across training steps. The
            # step index is not available here, so records are timestamp-free and
            # ordered by append; the plot script reads them in order.
            path = os.path.join(dump_dir, f"prefixed_prompts_rank{getattr(self, 'rank', 0)}.jsonl")
            with open(path, "a") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
        except Exception as e:  # pragma: no cover — diagnostics must never crash
            logger.warning(f"[guided.prefix] failed to dump prefixed prompts: {e!r}")

    # ── flat async orchestration ─────────────────────────────────────────

    def _generate_chains_flat_async(
        self,
        *,
        items_by_group: dict[int, dict],
        n_per_prompt: int,
        actor_rollout_ref_wg,
        config,
        override_config: dict[str, Any] | None,
        total_slots: int,
        global_max_iters: int,
        max_chains_per_group: int,
    ) -> tuple[list[Trajectory], int]:
        """Run all groups' chains concurrently in ONE event loop.

        The barriered orchestration runs chain passes sequentially: step 1 of
        the single-task sanity needed ~7 chains × ~3 passes = ~20 passes, each
        barrier-synced on its slowest trajectory at ≤ |groups| concurrency.
        Here each group advances its own chains independently as coroutines:

            group g  ─ chain 0: baseline ─→ iter 1 ─→ iter 2
                     └ chain 1: baseline ─→ …            (concurrent_chains_per_group)

        Every trajectory is a ``_async_run_traj`` coroutine; vLLM continuous
        batching + the slot semaphore mix all in-flight trajectories across
        groups and chains. Within-chain sequential dependency (iter k branches
        off iter k−1's result) is preserved by ``await``.

        Exact-need accounting: a chain claims one unit of its group's ``need``
        immediately before launching each trajectory (no await between check
        and decrement → atomic under asyncio), so a group never over-fills.
        Failed launches refund the claim and end the chain; the group spawner
        starts a replacement chain while budget remains.
        """
        from pope_dagger.analyzer import BranchSelectorState

        # Per-rollout telemetry accumulators normally reset at the top of the
        # sync generate_trajectories(); _async_run_traj appends to them, and the
        # flat path never passes through that init block.
        self._sumctx_prompt_lens = []
        self._sumctx_resp_lens = []

        annotate_fn = self.teacher_annotate_fn
        max_workers = max(1, int(getattr(self.guided_cfg, "teacher_max_workers", 1) or 1))
        cc = max(1, int(getattr(self.guided_cfg, "concurrent_chains_per_group", 1) or 1))
        teacher_executor = ThreadPoolExecutor(max_workers=max_workers)
        # _initialize_batch blocks on ray.get (env acquire + reset, up to ~30s
        # on a controller respawn); keep it off the event loop.
        init_executor = ThreadPoolExecutor(max_workers=16)

        # Engine round-robin (mirrors _run_async_episode_loop).
        n_workers = len(actor_rollout_ref_wg.workers)
        tp_size = max(1, int(getattr(config.worker.rollout, "tensor_parallel_size", 1)))
        if n_workers % tp_size != 0:
            raise RuntimeError(
                f"n_workers={n_workers} not a multiple of tensor_parallel_size={tp_size}"
            )
        n_engines = n_workers // tp_size
        _engine_counter = {"i": 0}

        def _next_engine_ranks() -> list[int]:
            e = _engine_counter["i"] % n_engines
            _engine_counter["i"] += 1
            base = e * tp_size
            return list(range(base, base + tp_size))

        counters = {"running": 0, "completed": 0, "failed_steps": 0}
        need: dict[int, int] = {g: n_per_prompt for g in items_by_group}
        chains_spawned: dict[int, int] = {g: 0 for g in items_by_group}
        all_trajs: list[Trajectory] = []
        t_start = time.time()

        async def _main() -> None:
            loop = asyncio.get_running_loop()
            slot_sem = asyncio.Semaphore(total_slots)

            async def init_traj(group_id: int, n_idx: int) -> Optional[Trajectory]:
                """Acquire + reset an env slot off-loop, with bounded retries
                (reactive pool shrink can make an apparently-free slot vanish)."""
                item = items_by_group[group_id]
                for attempt in range(4):
                    try:
                        trajs = await loop.run_in_executor(
                            init_executor,
                            self._initialize_batch,
                            [(group_id, n_idx, item)],
                        )
                    except Exception as e:
                        logger.warning(
                            f"[guided-flat] init grp {group_id} failed: {e} "
                            f"(attempt {attempt + 1}/4)"
                        )
                        trajs = []
                    if trajs:
                        return trajs[0]
                    await asyncio.sleep(5.0 * (attempt + 1))
                return None

            async def run_one(
                group_id: int,
                *,
                dagger_iter: int,
                chain_id: int,
                branch_chain: list[int],
                override_fn,
            ) -> Optional[Trajectory]:
                await slot_sem.acquire()
                try:
                    t = await init_traj(group_id, dagger_iter)
                    if t is None:
                        return None
                    t.dagger_iter_index = dagger_iter
                    t.chain_id = chain_id
                    t.branch_chain = list(branch_chain)
                    t.async_pre_step_fn = override_fn
                    return await self._async_run_traj(
                        t, _next_engine_ranks(), actor_rollout_ref_wg,
                        config, override_config, counters,
                    )
                finally:
                    slot_sem.release()

            def make_override_fn(plans: dict[int, _IterPlan]):
                """Per-trajectory async override hook for one iter pass of one
                chain. The plans dict is chain-local, and a trajectory resolves
                one step at a time, so `_build_override_for`'s plan-state
                mutations are race-free even with concurrent chains."""

                async def ov_fn(t: Trajectory):
                    if t.terminated:
                        return None

                    def _decide():
                        ov = self._build_override_for(t, plans)
                        if ov is None:
                            return None
                        if isinstance(ov, _PendingAnnotation):
                            text, token_ids = annotate_fn(**ov.kwargs)
                            text, token_ids = self._ensure_summary_for_forced_step(
                                text, token_ids
                            )
                            # Fresh oracle step: PG-masked AND SFT-imitated.
                            return (text, list(token_ids), True, True)
                        return ov

                    return await loop.run_in_executor(teacher_executor, _decide)

                return ov_fn

            async def run_chain(group_id: int, chain_id: int) -> list[Trajectory]:
                # Reserve the chain's FULL potential budget upfront (baseline +
                # iterations), refunding the unused part at chain end. One-at-a-
                # time claiming let solved baselines keep the spawner pumping
                # new chains until need hit 0 — late-finishing FAILED baselines
                # then found no budget left and never got guided (smoke
                # 8305499: 6/8 solved, 2 failed, 0 forced steps). Reservation
                # restores the sequential-chain guarantee: a failed baseline
                # always has budget for its own iterations.
                out: list[Trajectory] = []
                take = min(need[group_id], 1 + global_max_iters)
                if take <= 0:
                    return out
                need[group_id] -= take  # atomic: no await since check
                used = 0
                try:
                    try:
                        parent = await run_one(
                            group_id, dagger_iter=0, chain_id=chain_id,
                            branch_chain=[], override_fn=None,
                        )
                    except Exception as e:
                        logger.error(f"[guided-flat] baseline grp {group_id} chain {chain_id} raised: {e!r}")
                        parent = None
                    if parent is None:
                        return out
                    used += 1
                    out.append(parent)

                    sel_state = BranchSelectorState()
                    for iter_k in range(1, global_max_iters + 1):
                        if used >= take:
                            break
                        if self.guided_cfg.stop_on_solved and _is_solved(parent):
                            break
                        is_completion = (
                            self.guided_cfg.completion_final_iter
                            and iter_k == global_max_iters
                        )
                        plans = self._plan_iteration(
                            iter_k, {group_id: parent}, items_by_group,
                            {group_id: sel_state},
                            chain_id=chain_id, is_completion=is_completion,
                        )
                        if not plans:
                            break
                        plan = plans[group_id]
                        try:
                            tk = await run_one(
                                group_id, dagger_iter=iter_k, chain_id=chain_id,
                                branch_chain=list(plan.branch_chain),
                                override_fn=make_override_fn(plans),
                            )
                        except Exception as e:
                            logger.error(
                                f"[guided-flat] iter {iter_k} grp {group_id} chain {chain_id} raised: {e!r}"
                            )
                            tk = None
                        if tk is None:
                            break
                        used += 1
                        out.append(tk)
                        logger.info(
                            f"[guided-flat] grp {group_id} chain {chain_id} "
                            f"iter {iter_k}: branch@{plan.branch_step_index} "
                            f"forced_steps={plan.forced_window_steps_emitted} "
                            f"solved={_is_solved(tk)}"
                        )
                        if plan.forced_window_steps_emitted == 0:
                            # Answer-only at the branch — nothing forceable left
                            # in this chain (mirrors chain_done_groups in the
                            # barriered path).
                            break
                        parent = tk
                        sel_state.forced_window_ranges.append(
                            (plan.branch_step_index,
                             plan.branch_step_index + plan.forced_window_steps_emitted)
                        )
                finally:
                    if take - used > 0:
                        need[group_id] += take - used  # refund unused reservation
                return out

            async def run_group(group_id: int) -> list[Trajectory]:
                out: list[Trajectory] = []
                running: set[asyncio.Task] = set()
                while True:
                    while (
                        need[group_id] > 0
                        and chains_spawned[group_id] < max_chains_per_group
                        and len(running) < cc
                    ):
                        cid = chains_spawned[group_id]
                        chains_spawned[group_id] += 1
                        running.add(
                            asyncio.create_task(run_chain(group_id, cid))
                        )
                    if not running:
                        break
                    done, running = await asyncio.wait(
                        running, return_when=asyncio.FIRST_COMPLETED
                    )
                    for d in done:
                        exc = d.exception()
                        if exc is not None:
                            logger.error(
                                f"[guided-flat] chain task grp {group_id} raised: {exc!r}"
                            )
                        else:
                            out.extend(d.result())
                if need[group_id] > 0:
                    logger.warning(
                        f"[guided-flat] grp {group_id}: chain budget exhausted "
                        f"with {need[group_id]} trajectories still needed "
                        f"(group will be under-filled; group-id uid mapping "
                        f"keeps GRPO grouping correct)"
                    )
                return out

            async def progress_logger() -> None:
                while True:
                    await asyncio.sleep(15.0)
                    logger.info(
                        f"[guided-flat] running={counters['running']} "
                        f"completed={counters['completed']} "
                        f"need_total={sum(need.values())} "
                        f"chains={sum(chains_spawned.values())} "
                        f"failed_steps={counters['failed_steps']} "
                        f"elapsed={time.time() - t_start:.0f}s"
                    )

            plog = asyncio.create_task(progress_logger())
            try:
                group_results = await asyncio.gather(
                    *[run_group(g) for g in items_by_group]
                )
            finally:
                plog.cancel()
                try:
                    await plog
                except (asyncio.CancelledError, Exception):
                    pass
            for gr in group_results:
                all_trajs.extend(gr)

        try:
            asyncio.run(_main())
        finally:
            teacher_executor.shutdown(wait=True)
            init_executor.shutdown(wait=True)

        logger.info(
            f"[guided-flat] DONE: {len(all_trajs)} trajs, "
            f"{sum(chains_spawned.values())} chains, "
            f"completed={counters['completed']} "
            f"failed_steps={counters['failed_steps']} "
            f"elapsed={time.time() - t_start:.1f}s"
        )
        return all_trajs, sum(chains_spawned.values())

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
            elif mode == "avg_success_step":
                # POPE-DAgger v2: branch one step before the running average
                # success step. Initially 0 -> branch at step 0 (the beginning).
                # Clamp to a valid student-suffix step on this parent.
                branch_step = max(0, int(self._avg_success_step_count) - 1)
                branch_step = min(branch_step, max(0, parent.num_steps - 1))
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

    def _ensure_summary_for_forced_step(
        self, text: str, token_ids: list[int]
    ) -> tuple[str, list[int]]:
        """Make a forced-expert step's teacher text carry a ``<summary>``.

        Under ``context_mode='summary_context'`` the next step's prompt sees only
        the previous step's ``<summary>`` (parsed by the env into
        ``action.memory``). The teacher annotator (both placeholder and real,
        ``main.py``) builds text via ``build_assistant_response_no_summary`` —
        ``<think>`` + action, NO ``<summary>``. Left as-is, the env parses
        ``action.memory=None`` and the on-policy step *after* the forced window
        runs memoryless, defeating the whole point of summary-context. So we
        append a one-line summary derived from the teacher's own ``<think>`` (its
        intent for the forced action) so memory carries across the handoff.

        No-op for ``multi_turn`` and for text that already has a summary (e.g.
        replayed-prefix steps reuse the parent's response, which already carries
        one). The re-encoded ids keep ``step_response_token_ids`` consistent; the
        summary-context final-batch builder re-encodes from text regardless.
        """
        if getattr(self, "context_mode", "multi_turn") != "summary_context":
            return text, token_ids
        if not text or "</summary>" in text:
            return text, token_ids
        m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        think = (m.group(1).strip() if m else "").replace("\n", " ").strip()
        summary = ""
        if think:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", think) if s.strip()]
            summary = (sentences[-1] if sentences else think)[:200].strip()
        if not summary:
            summary = "Followed the expert action toward the target."
        new_text = f"{text.rstrip()}\n<summary>{summary}</summary>"
        new_ids = self.tokenizer.encode(new_text, add_special_tokens=False)
        return new_text, new_ids

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

        # Cap the initial batch by the LIVE available slot count, not the static
        # `total_slots`. prefix_groups runs three passes back-to-back within one
        # rollout (baseline -> prefixed -> pad); when the prior pass has just
        # drained, slot releases are still settling (async release + reactive
        # memory-pressure shrink in release_env), so live availability can dip
        # below total_slots for a moment. Sizing the initial batch off the static
        # count then over-subscribes and _initialize_batch raises "No simulator
        # slots available" (e.g. group=19, n=6 at branch_step=3). The leftover
        # stays in pending_queue and the continuous loop's refill path (which
        # already caps by live availability) picks it up as slots free. Poll a
        # few times so a transient empty read doesn't strand the whole pass.
        live_available = 0
        for _attempt in range(10):
            live_available = sum(
                ray.get([p.get_available_count.remote() for p in self.simulator_pools])
            )
            if live_available > 0 or not pending_queue:
                break
            time.sleep(1.0)
        initial_count = min(len(pending_queue), total_slots, max(live_available, 0))
        if live_available < total_slots and pending_queue:
            logger.info(
                f"  _run_one_pass: live slots {live_available} < total_slots "
                f"{total_slots}; seeding {initial_count} now, "
                f"{len(pending_queue) - initial_count} via refill"
            )
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

        async_mode = bool(getattr(config.worker.rollout, "async_mode", False))
        try:
            if async_mode:
                # Per-trajectory async: every trajectory in this pass runs as
                # its own coroutine, so vLLM continuous batching overlaps with
                # env steps and teacher annotations across the whole pass
                # (no per-step lockstep barrier). Override decide + teacher
                # calls run in a bounded executor off the event loop.
                self._run_async_episode_loop_with_overrides(
                    active, all_trajs, pending_queue,
                    actor_rollout_ref_wg, config,
                    override_config=override_config,
                    total_slots=total_slots,
                    dagger_iter_index=dagger_iter_index,
                    branch_chain=branch_chain,
                    chain_id=chain_id,
                )
            else:
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
            hint_pending: list[tuple[Trajectory, _PendingHint]] = []
            handoff_pending: list[tuple[Trajectory, _PendingForcedHandoff]] = []
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
                elif isinstance(ov, _PendingForcedHandoff):
                    handoff_pending.append((t, ov))
                elif isinstance(ov, _PendingHint):
                    hint_pending.append((t, ov))
                else:
                    t.pending_response_override = ov

            # SELF-teacher: resolve diagnosis hints via the in-process judge and
            # inject them as USER-turn suffixes (no override). Run BEFORE the
            # _PendingAnnotation path so the two never interleave (a config uses
            # exactly one teacher_mode, so only one list is ever non-empty).
            if hint_pending:
                self._resolve_self_teacher_hints(
                    hint_pending, actor_rollout_ref_wg, config
                )

            # Forced-depth last-forced step: combined judge call yields a forced
            # response (teacher summary + expert action) for THIS step AND caches
            # the suffix-derived hint for the handoff step.
            if handoff_pending:
                self._resolve_forced_handoffs(
                    handoff_pending, actor_rollout_ref_wg, config
                )

            if not pending:
                return

            # SFT flag for the freshly-annotated forced/guidance step. The chain
            # orchestration imitates forced steps (is_expert=True); the prefix-
            # testing orchestration defers to ``guidance_is_expert`` (pure
            # PrefixRL → False: the guidance is conditioning, gradient flows only
            # through the student suffix).
            forced_is_expert = (
                bool(self.guided_cfg.guidance_is_expert)
                if getattr(self.guided_cfg, "orchestration_mode", "chain") == "prefix_groups"
                else True
            )

            # Annotate phase (parallel): fan the deferred teacher calls out so
            # the vLLM teacher server batches the concurrent requests. Without
            # this, a step's N forced annotations run serially (each a slow 32B
            # call), stalling the whole rollout loop and idling the simulators.
            #
            # Deduplicate by ``group_key`` first: all pendings sharing a key
            # (the N student suffixes of one prefixed group at their common
            # branch) resolve to ONE teacher call, broadcast byte-identically to
            # every sibling — required so the within-group GRPO comparison sees a
            # shared prefix+guidance. Keyless pendings (chain orchestration) each
            # form their own singleton group, preserving prior behavior.
            groups: dict[Any, list[tuple[Trajectory, _PendingAnnotation]]] = {}
            order: list[Any] = []
            for item in pending:
                t, pend = item
                key = pend.group_key if pend.group_key is not None else id(t)
                if key not in groups:
                    groups[key] = []
                    order.append(key)
                groups[key].append(item)

            def _run_group(key):
                t0, pend0 = groups[key][0]
                text, token_ids = annotate_fn(**pend0.kwargs)
                text, token_ids = self._ensure_summary_for_forced_step(text, token_ids)
                return key, text, token_ids

            def _apply(key, text, token_ids):
                members = groups[key]
                ov = (text, list(token_ids), True, forced_is_expert)
                for t, _pend in members:
                    t.pending_response_override = ov
                cb = members[0][1].on_annotated
                if cb is not None:
                    cb(text, list(token_ids))

            if len(order) == 1:
                key, text, token_ids = _run_group(order[0])
                _apply(key, text, token_ids)
                return

            with ThreadPoolExecutor(max_workers=min(len(order), max_workers)) as ex:
                for fut in as_completed([ex.submit(_run_group, key) for key in order]):
                    key, text, token_ids = fut.result()
                    _apply(key, text, token_ids)

        self._pre_step_hook = pre_step
        try:
            self._run_continuous_episode_loop(
                active, all_trajectories, pending_queue,
                actor_rollout_ref_wg, config,
                override_config=override_config,
            )
        finally:
            self._pre_step_hook = None

    # ── colocated self-teacher (teacher_mode='self') ──────────────────────
    def _resolve_self_teacher_hints(
        self,
        hint_pending: list[tuple[Trajectory, "_PendingHint"]],
        actor_rollout_ref_wg,
        config,
    ) -> None:
        """Resolve diagnosis hints for the branch step and inject them as
        USER-turn suffixes (``pending_prompt_suffix``), NOT response overrides.

        One judge call per unique ``group_key`` (all siblings share the same
        replayed prefix → same branch pose → same hint), broadcast to every
        member so within-group GRPO sees an identical injected hint. The judge
        runs through the SAME in-process engine/weights as the student.
        """
        from privileged_eval.diagnose import (
            build_diagnosis_block,
            parse_diagnosis_response,
        )

        # Dedup by group_key (keyless → singleton per trajectory).
        groups: dict[Any, list[tuple[Trajectory, _PendingHint]]] = {}
        order: list[Any] = []
        for item in hint_pending:
            t, pend = item
            key = pend.group_key if pend.group_key is not None else id(t)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(item)

        # Build the judge request per group on its representative slot (the
        # no-leak nav-state reconstruction; no model call). ray.get is blocking
        # but cheap relative to generation; fan out the remote calls first.
        req_futures = {}
        for key in order:
            t0, pend0 = groups[key][0]
            req_futures[key] = pend0.pool.build_branch_diagnosis_request.remote(
                pend0.slot_id,
                intended_action=getattr(pend0, "intended_action", None),
                parent_suffix=getattr(pend0, "parent_suffix", None),
                want_summary=False,
            )
        reqs: dict[Any, Optional[dict]] = {}
        for key in order:
            try:
                reqs[key] = ray.get(req_futures[key])
            except Exception as e:
                logger.warning(f"[self-teacher] grp {key}: diagnosis request failed: {e!r}")
                reqs[key] = None

        # Keys whose reconstruction succeeded AND have a branch image get a judge
        # call; the rest degrade to a pure on-policy suffix-free branch (no hint).
        # The diagnosis user_text ends with one `<image>` placeholder, so a missing
        # branch image would desync image/placeholder counts in vLLM — skip those.
        def _live(k):
            r = reqs.get(k)
            return r is not None and r.get("branch_image") is not None
        live_keys = [k for k in order if _live(k)]
        if not live_keys:
            logger.warning("[self-teacher] no live diagnosis requests this step; skipping hints.")
            return

        judge_prompts = []
        judge_images = []
        for k in live_keys:
            r = reqs[k]
            judge_prompts.append([
                {"role": "system", "content": r["system_prompt"]},
                {"role": "user", "content": r["user_text"]},
            ])
            judge_images.append([r["branch_image"]])

        try:
            responses = self._engine_generate_judge(
                judge_prompts, judge_images, actor_rollout_ref_wg, config
            )
        except Exception as e:
            logger.error(f"[self-teacher] judge generation failed: {e!r}\n{traceback.format_exc()}")
            return

        # Raw judge I/O audit dump (best-effort). Captures the FULL judge prompt
        # (system + menu user_text), branch-image presence, prompt token count vs
        # the teacher cap, and the raw + parsed judge response — so the teacher's
        # behavior is verifiable end-to-end (esp. that the un-truncated judge now
        # commits to a concrete RECOMMENDED action). Controlled by POPE_PREFIX_DUMP.
        dump_dir = os.environ.get("POPE_PREFIX_DUMP", "").strip()
        if dump_dir:
            try:
                os.makedirs(dump_dir, exist_ok=True)
                teacher_cap = getattr(self.guided_cfg, "teacher_max_prompt_length", None)
                jpath = os.path.join(dump_dir, f"judge_io_rank{getattr(self, 'rank', 0)}.jsonl")
                with open(jpath, "a") as jf:
                    for k, resp in zip(live_keys, responses):
                        r = reqs[k]
                        sysp = r.get("system_prompt", "") or ""
                        usr = r.get("user_text", "") or ""
                        try:
                            n_tok = len(self.tokenizer.encode(sysp + usr, add_special_tokens=False))
                        except Exception:
                            n_tok = None
                        parsed_dbg = parse_diagnosis_response(resp or "")
                        jf.write(json.dumps({
                            "rollout_idx": int(getattr(self, "_rollout_idx", 0)),
                            "group_key": str(k),
                            "judge_system_prompt": sysp,
                            "judge_user_text": usr,
                            "has_branch_image": r.get("branch_image") is not None,
                            "prompt_text_tokens": n_tok,
                            "teacher_max_prompt_length": teacher_cap,
                            "truncated": (n_tok is not None and teacher_cap is not None and n_tok > teacher_cap),
                            "judge_raw_response": resp,
                            "judge_parsed_hint": parsed_dbg.get("hint", ""),
                        }) + "\n")
            except Exception as e:  # pragma: no cover — diagnostics must never crash
                logger.warning(f"[self-teacher] judge I/O dump failed: {e!r}")

        for k, resp in zip(live_keys, responses):
            parsed = parse_diagnosis_response(resp or "")
            hint = parsed.get("hint", "").strip()
            if not hint:
                logger.warning(f"[self-teacher] grp {k}: judge produced empty hint; no injection.")
                continue
            block = build_diagnosis_block(hint)
            members = groups[k]
            for t, _pend in members:
                t.pending_prompt_suffix = block
            cb = members[0][1].on_resolved
            if cb is not None:
                cb(block)

    def _engine_generate_judge(
        self,
        prompts: list[list[dict]],
        images_list: list[list],
        actor_rollout_ref_wg,
        config,
    ) -> list[str]:
        """Run the in-process judge: tokenize the (system+user+image) prompts,
        generate via the same ``generate_sequences`` engine the student uses, and
        decode. Returns one response string per prompt (order-preserving).

        Mirrors the base loop's vLLM generate path (tokenize → DataProto → pad →
        generate → unpad → decode) at the judge temperature, n=1.
        """
        from ...protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto

        # Use the teacher-specific prompt budget so the diagnosis system prompt +
        # menu + branch image survive (the student's max_prompt_length would right-
        # truncate them). Requires worker.rollout.max_model_len >= this + response.
        tokenized = self._tokenize_prompts(
            prompts,
            images_list,
            for_generation=True,
            max_prompt_length=getattr(self.guided_cfg, "teacher_max_prompt_length", None),
        )
        meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "video_fps": getattr(config.data, "video_fps", 2.0),
            "n": 1,
            "temperature": float(getattr(self.guided_cfg, "judge_temperature", 0.5)),
            "top_p": 1.0,
        }
        gen_batch = DataProto.from_single_dict(tokenized, meta_info=meta_info)
        gen_batch, pad_size = pad_dataproto_to_divisor(
            gen_batch, actor_rollout_ref_wg.world_size
        )
        gen_output = actor_rollout_ref_wg.generate_sequences(gen_batch)
        gen_output = unpad_dataproto(gen_output, pad_size)
        response_ids = gen_output.batch["responses"]
        response_mask = gen_output.batch["response_mask"]
        out: list[str] = []
        for i in range(len(prompts)):
            length = int(response_mask[i].sum().item())
            out.append(
                self.tokenizer.decode(
                    response_ids[i][:length], skip_special_tokens=True
                )
            )
        return out

    def _resolve_forced_handoffs(
        self,
        handoff_pending: list[tuple[Trajectory, "_PendingForcedHandoff"]],
        actor_rollout_ref_wg,
        config,
    ) -> None:
        """Resolve the last-forced-step combined judge call (prefix_force_depth).

        ONE judge call per group at the last-forced pose (want_summary=True). The
        judge sees the parent's failed branch→end suffix and emits a teacher
        ``<summary>`` (carried into the handoff under summary_context) plus a
        suffix-derived ``<hint>`` (cached on the spec for the handoff step). This
        step's forced response is ``<think>..</think><summary>{summary}</summary>
        {expert_action}`` — PG-masked, identical across siblings so the within-
        group GRPO comparison shares a byte-identical prefix+guidance.
        """
        from privileged_eval.diagnose import (
            build_diagnosis_block,
            parse_diagnosis_response,
        )

        # Dedup by group_key (keyless → singleton per trajectory).
        groups: dict[Any, list[tuple[Trajectory, _PendingForcedHandoff]]] = {}
        order: list[Any] = []
        for item in handoff_pending:
            t, pend = item
            key = pend.group_key if pend.group_key is not None else id(t)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(item)

        forced_is_expert = bool(self.guided_cfg.guidance_is_expert)

        # Under forced-depth the on-policy region always ENDS with the oracle's
        # terminal answer step. ``L = round(self._leave_on_policy)`` is the global
        # number of steps left on policy, so when L <= 1 the single on-policy step
        # after this handoff IS the answer. In that regime we tell the TEACHER the
        # next action is the answer, so it GENERATES an answer-oriented summary+hint
        # (the default forced-depth prompt + hint block keep the student approaching
        # forever, which is exactly why these groups died at reward 0). No-leak: the
        # teacher only points at what is observable; it never hands over coords.
        on_policy_left = int(round(float(getattr(self, "_leave_on_policy", 1.0))))
        answer_next = on_policy_left <= 1

        # Build the judge request per group (no-leak nav state + parent suffix).
        req_futures = {}
        for key in order:
            t0, pend0 = groups[key][0]
            req_futures[key] = pend0.pool.build_branch_diagnosis_request.remote(
                pend0.slot_id,
                intended_action=getattr(pend0, "intended_action", None),
                parent_suffix=getattr(pend0, "parent_suffix", None),
                want_summary=True,
                answer_next=answer_next,
            )
        reqs: dict[Any, Optional[dict]] = {}
        for key in order:
            try:
                reqs[key] = ray.get(req_futures[key])
            except Exception as e:
                logger.warning(f"[guided.fd] grp {key}: diagnosis request failed: {e!r}")
                reqs[key] = None

        def _live(k):
            r = reqs.get(k)
            return r is not None and r.get("branch_image") is not None
        live_keys = [k for k in order if _live(k)]

        # Helper to emit the forced response (broadcast to all siblings of a key).
        def _emit_forced(key, summary_text, answer_mode=False):
            members = groups[key]
            rep_action = members[0][1].expert_action
            if answer_mode:
                summary = (summary_text or "").strip() or _ANSWER_HANDOFF_SUMMARY
                think = (
                    "<think>\nThis was the last positioning move. The target should now "
                    "be in view, so the next move is to answer it directly rather than "
                    "explore or approach further.\n</think>\n"
                )
            else:
                summary = (summary_text or "").strip() or (
                    "Heading toward the target area; will verify the nearest cue next."
                )
                think = (
                    "<think>\nContinuing along the planned route toward the target "
                    "based on what has been seen so far.\n</think>\n"
                )
            text = f"{think}<summary>{summary}</summary>\n{rep_action}"
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            ov = (text, list(token_ids), True, forced_is_expert)
            for t, _pend in members:
                t.pending_response_override = ov

        # Keys whose reconstruction failed: still force the action, but with a
        # templated summary and NO hint (degrade gracefully, never desync images).
        # When answer_next, the template at least orients toward answering.
        for key in order:
            if key not in live_keys:
                logger.warning(
                    f"[guided.fd] grp {key}: no live diagnosis request; "
                    f"forcing action with templated summary, no hint."
                )
                _emit_forced(key, None, answer_mode=answer_next)
                if answer_next:
                    cb = groups[key][0][1].on_resolved
                    if cb is not None:
                        cb(build_diagnosis_block(_ANSWER_HANDOFF_HINT, answer_next=True))

        if not live_keys:
            return

        judge_prompts = []
        judge_images = []
        for k in live_keys:
            r = reqs[k]
            judge_prompts.append([
                {"role": "system", "content": r["system_prompt"]},
                {"role": "user", "content": r["user_text"]},
            ])
            judge_images.append([r["branch_image"]])

        try:
            responses = self._engine_generate_judge(
                judge_prompts, judge_images, actor_rollout_ref_wg, config
            )
        except Exception as e:
            logger.error(
                f"[guided.fd] judge generation failed: {e!r}\n{traceback.format_exc()}"
            )
            # Fall back: force the action with a templated summary, no hint.
            for k in live_keys:
                _emit_forced(k, None, answer_mode=answer_next)
                if answer_next:
                    cb = groups[k][0][1].on_resolved
                    if cb is not None:
                        cb(build_diagnosis_block(_ANSWER_HANDOFF_HINT, answer_next=True))
            return

        # Optional raw judge I/O audit dump (shared POPE_PREFIX_DUMP knob).
        dump_dir = os.environ.get("POPE_PREFIX_DUMP", "").strip()
        if dump_dir:
            try:
                os.makedirs(dump_dir, exist_ok=True)
                jpath = os.path.join(
                    dump_dir, f"judge_handoff_rank{getattr(self, 'rank', 0)}.jsonl"
                )
                with open(jpath, "a") as jf:
                    for k, resp in zip(live_keys, responses):
                        r = reqs[k]
                        parsed_dbg = parse_diagnosis_response(resp or "")
                        jf.write(json.dumps({
                            "rollout_idx": int(getattr(self, "_rollout_idx", 0)),
                            "group_key": str(k),
                            "answer_next": bool(answer_next),
                            "judge_user_text": r.get("user_text", ""),
                            "judge_raw_response": resp,
                            "judge_parsed_summary": parsed_dbg.get("summary", ""),
                            "judge_parsed_hint": parsed_dbg.get("hint", ""),
                        }) + "\n")
            except Exception as e:  # pragma: no cover
                logger.warning(f"[guided.fd] judge I/O dump failed: {e!r}")

        for k, resp in zip(live_keys, responses):
            parsed = parse_diagnosis_response(resp or "")
            summary = parsed.get("summary", "").strip()
            hint = parsed.get("hint", "").strip()
            # Under answer_next the teacher is asked (via the POSITION-REACHED system
            # override) to GENERATE an answer-oriented summary/hint. The base judge is
            # unreliable, so sanitize: keep its text only if it actually orients to
            # answering; if it emitted nothing or reverted to a navigation directive
            # (ground:/direction:/explore — fighting the override), fall back to the
            # answer template so the injected block stays coherent rather than telling
            # the student to answer while handing it a "turn left" hint.
            if answer_next:
                def _is_answer_oriented(s):
                    s = (s or "").lower()
                    nav = ("ground:" in s or "direction:" in s or "<explore" in s)
                    return ("answer" in s) and not nav
                if not _is_answer_oriented(hint):
                    hint = _ANSWER_HANDOFF_HINT
                if not _is_answer_oriented(summary):
                    summary = _ANSWER_HANDOFF_SUMMARY
                # No-leak backstop: even with the coord-free banner the base judge can
                # leak the exact pixel (it sees the target in [VISIBLE OBJECTS] and may
                # guess/emit a tuple). Strip any (x,y) tuple and <answer>(...) coords
                # from the injected text so the student never receives the answer pixel
                # — it must read the target's location from its own current view.
                hint = _strip_answer_coords(hint)
                summary = _strip_answer_coords(summary)
            # Emit this step's forced response with the TEACHER-generated summary
            # (answer_mode keeps the forced <think> consistent with answering).
            _emit_forced(k, summary, answer_mode=answer_next)
            # Cache the (sanitized) hint for the handoff step.
            if hint:
                block = build_diagnosis_block(hint, answer_next=answer_next)
                cb = groups[k][0][1].on_resolved
                if cb is not None:
                    cb(block)
            else:
                logger.warning(
                    f"[guided.fd] grp {k}: judge produced empty hint; "
                    f"handoff step will run with no injected hint."
                )

    def _run_async_episode_loop_with_overrides(
        self,
        active: list[Trajectory],
        all_trajectories: list[Trajectory],
        pending_queue,
        actor_rollout_ref_wg,
        config,
        override_config: dict[str, Any] | None,
        total_slots: int,
        dagger_iter_index: int,
        branch_chain: Optional[list[int]],
        chain_id: int = 0,
    ) -> None:
        """Async counterpart of ``_run_continuous_episode_loop_with_overrides``.

        Installs ``_async_pre_step``, a per-trajectory async hook the base
        ``_run_async_episode_loop`` awaits before each step. The hook resolves
        the override for one trajectory: cheap cases (prefix replay / no
        override) and the blocking oracle decide + teacher-VLM annotate all run
        inside a bounded ``ThreadPoolExecutor`` via ``run_in_executor`` so the
        event loop keeps every other trajectory's vLLM generation and env steps
        in flight. Each pass runs at most one trajectory per group/plan, so the
        plan-state mutations inside ``_build_override_for`` never race.
        """
        provider = self._override_provider

        # Baseline (on-policy) pass: no overrides → run the plain async loop and
        # skip the per-step executor round-trip entirely. dagger metadata for
        # these trajectories is stamped by ``generate_trajectories`` post-pass.
        if provider is None:
            asyncio.run(
                self._run_async_episode_loop(
                    active, all_trajectories, pending_queue,
                    actor_rollout_ref_wg, config,
                    total_slots=total_slots,
                    override_config=override_config,
                )
            )
            return

        annotate_fn = self.teacher_annotate_fn
        max_workers = max(1, int(getattr(self.guided_cfg, "teacher_max_workers", 1) or 1))

        def _decide_and_annotate(t: Trajectory):
            # Runs in an executor thread. Re-stamp dagger metadata first (a
            # trajectory refilled from the pending queue mid-pass carries none
            # yet) — matches the sync pre_step hook. ``provider`` is the only
            # plan-state mutator and there is one trajectory per plan, so this
            # is race-free across the concurrent executor threads.
            t.dagger_iter_index = dagger_iter_index
            t.chain_id = chain_id
            if branch_chain is not None:
                t.branch_chain = list(branch_chain)
            ov = provider(t)
            if ov is None:
                return None
            if isinstance(ov, _PendingAnnotation):
                text, token_ids = annotate_fn(**ov.kwargs)
                # Forced steps must carry a <summary> or the next on-policy
                # step is memoryless under summary_context (same fix as the
                # sync pre_step hook).
                text, token_ids = self._ensure_summary_for_forced_step(text, token_ids)
                # Fresh oracle step: mask from PG (is_teacher) AND imitate via
                # SFT (is_expert).
                return (text, list(token_ids), True, True)
            # Prefix-replay 4-tuple (text, token_ids, is_teacher, is_expert).
            return ov

        executor = ThreadPoolExecutor(max_workers=max_workers)

        async def async_pre_step(t: Trajectory):
            if t.terminated:
                return None
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(executor, _decide_and_annotate, t)

        self._async_pre_step = async_pre_step
        try:
            asyncio.run(
                self._run_async_episode_loop(
                    active, all_trajectories, pending_queue,
                    actor_rollout_ref_wg, config,
                    total_slots=total_slots,
                    override_config=override_config,
                )
            )
        finally:
            self._async_pre_step = None
            executor.shutdown(wait=True)

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

