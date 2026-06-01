"""Driver-side multi-turn environment rollout for GRPO training.

Runs multi-turn trajectories through AI2Thor environments by calling the
distributed ``actor_rollout_ref_wg.generate_sequences()`` at each step.
Environment operations (reset, step, build_prompt) are parallelized across
Ray-remote SimulatorPool actors.

Produces a standard ``DataProto`` batch compatible with the rest of the GRPO
pipeline (KL, advantage, actor update).
"""

from __future__ import annotations

import asyncio
import ctypes
import gc
import json
import logging
import os
import re
import time
import uuid
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import ray
import torch
from PIL import Image
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin

from ...protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto


from common.prompting.context_builders import filter_steps_to_kwindow  # noqa: E402

logger = logging.getLogger(__name__)
# Ray workers don't inherit the driver's logging config, so set level from env var
_log_level = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
logger.setLevel(_log_level)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_handler)
    logger.propagate = False  # prevent duplicate output from root logger


def _get_rss_mb() -> float:
    """Return current process RSS in MB (Linux only)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # kB → MB
    except Exception:
        pass
    return 0.0


def _force_free_memory():
    """Aggressive memory cleanup: gc + glibc malloc_trim to return pages to OS."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, q))


def _compute_async_phase_metrics(
    trajectories: list["Trajectory"], total_wall_s: float
) -> dict[str, float]:
    """Aggregate per-step phase timings + concurrency from async rollout.

    Emitted under ``rollout/*`` keys. Returns an empty dict when no trajectory
    recorded timings (e.g. all coroutines failed before the first step).
    """
    prompt_ms: list[float] = []
    gen_ms: list[float] = []
    env_ms: list[float] = []
    traj_wall_ms: list[float] = []
    events: list[tuple[float, int]] = []  # (timestamp, +1 start / -1 end)

    for t in trajectories:
        for step in t.step_phase_times:
            prompt_ms.append(step.get("prompt_ms", 0.0))
            gen_ms.append(step.get("gen_ms", 0.0))
            env_ms.append(step.get("env_ms", 0.0))
        if t.traj_start_ts is not None and t.traj_end_ts is not None:
            traj_wall_ms.append((t.traj_end_ts - t.traj_start_ts) * 1000.0)
            events.append((t.traj_start_ts, +1))
            events.append((t.traj_end_ts, -1))

    if not prompt_ms and not traj_wall_ms:
        return {}

    out: dict[str, float] = {"rollout/total_wall_ms": total_wall_s * 1000.0}

    for name, vals in (
        ("prompt", prompt_ms),
        ("gen", gen_ms),
        ("env", env_ms),
    ):
        if not vals:
            continue
        out[f"rollout/phase_{name}_ms_avg"] = float(np.mean(vals))
        out[f"rollout/phase_{name}_ms_p50"] = _percentile(vals, 50)
        out[f"rollout/phase_{name}_ms_p90"] = _percentile(vals, 90)
        out[f"rollout/phase_{name}_ms_sum"] = float(np.sum(vals))

    if traj_wall_ms:
        out["rollout/traj_wall_ms_avg"] = float(np.mean(traj_wall_ms))
        out["rollout/traj_wall_ms_p50"] = _percentile(traj_wall_ms, 50)
        out["rollout/traj_wall_ms_p90"] = _percentile(traj_wall_ms, 90)

    # Concurrency: sweep start/end events. Avg is time-weighted, peak is the
    # max number of simultaneously-running coroutines.
    if events:
        events.sort(key=lambda e: (e[0], -e[1]))  # starts before ends at ties
        peak = 0
        cur = 0
        weighted = 0.0
        prev_t = events[0][0]
        for ts, delta in events:
            weighted += cur * (ts - prev_t)
            cur += delta
            peak = max(peak, cur)
            prev_t = ts
        span = events[-1][0] - events[0][0]
        out["rollout/concurrency_peak"] = float(peak)
        out["rollout/concurrency_avg"] = float(weighted / span) if span > 0 else float(peak)

    return out


# ---------------------------------------------------------------------------
# Generic environment interface (kept for ObjectNavEnvAdapter used in pool)
# ---------------------------------------------------------------------------


@runtime_checkable
class EnvInterface(Protocol):
    """Protocol for environments used in multi-turn rollouts."""

    def reset(self) -> dict[str, Any]: ...
    def build_prompt(self) -> tuple[list[dict], list[Image.Image]]: ...
    def step(self, action_text: str) -> tuple[float, bool, dict[str, Any]]: ...
    def get_trajectory_reward(self) -> float: ...
    def get_ground_truth(self) -> str: ...
    def close(self) -> None: ...


class EnvFactory(ABC):
    """Abstract factory — now only used for dataset iteration."""

    @abstractmethod
    def get_next_item(self) -> Any: ...

    @abstractmethod
    def __len__(self) -> int: ...


# ---------------------------------------------------------------------------
# Trajectory tracking
# ---------------------------------------------------------------------------


@dataclass
class Trajectory:
    """Tracks a single in-progress or completed trajectory.

    Instead of holding a local env, holds a reference to a remote SimulatorPool
    slot. All env operations go through Ray remote calls to the pool.
    """

    pool: Any  # ray.actor.ActorHandle for SimulatorPool
    slot_id: int
    episode_id: str

    # GRPO group tracking (for dynamic slot reuse)
    group_id: int = -1   # which dataset item / GRPO group
    n_idx: int = 0       # which trajectory within the group (0..n-1)

    # Accumulated per-step data
    step_responses: list[str] = field(default_factory=list)
    # Per-step assistant token IDs and matching per-token sampled logprobs
    # captured from vLLM (when parity_log_probs is enabled). Lists are aligned
    # 1:1 with step_responses; outer list index is step.
    step_response_token_ids: list[list[int]] = field(default_factory=list)
    step_response_log_probs: list[list[float]] = field(default_factory=list)
    # Per-step flag set when a step's response was injected by the teacher VLM
    # (Pope-Dagger guided rollout) rather than sampled by the student. Aligned
    # 1:1 with step_responses. The downstream loss masks these tokens out.
    step_teacher_forced: list[bool] = field(default_factory=list)
    # Per-step flag, subset of ``step_teacher_forced``: True only for *fresh
    # expert* steps emitted this iteration (the oracle's forced action). These
    # are the tokens the SFT/cross-entropy loss imitates. Replayed-prefix steps
    # are teacher-forced (masked from PG) but NOT expert (no SFT) — they were
    # already trained via their source rollout. Aligned 1:1 with step_responses.
    step_is_expert: list[bool] = field(default_factory=list)
    # Per-step pre-action geodesic distance to the target (from the slot's
    # state_before). Aligned 1:1 with step_responses. Used by the V5
    # random_regression branch selector in GuidedMultiturnEnvRollout to
    # compute the trajectory's progress envelope.
    step_distances: list[float | None] = field(default_factory=list)
    terminated: bool = False
    num_steps: int = 0

    # ── Pope-Dagger V4 guidance metadata (default values keep on-policy parity)
    # 0 = baseline / on-policy rollout (incl. padding rollouts). k ≥ 1 = the
    # k-th V4 iteration in the same prompt's group.
    dagger_iter_index: int = 0
    # Within a prompt group, multiple independent V4/V5 chains may be emitted
    # (baseline + iters → baseline₂ + iters₂ → …) until the group fills to
    # n_per_prompt. ``chain_id`` indexes which chain the trajectory belongs to.
    # Cross-chain pairs (chain_id_i != chain_id_j) are treated as LCP=0 by the
    # branching advantage estimator (fully divergent from step 0).
    chain_id: int = 0
    # Branch-step chain inherited from V4 ancestry: [B_1, B_2, …, B_k]. Adjacent
    # iters in the same group differ only in the last entry; LCP between iters
    # a and b is the prefix shared up to branch_chain[min(a,b)].
    branch_chain: list[int] = field(default_factory=list)
    # Single-shot response override consumed by the rollout loop in lieu of
    # calling vLLM. Set by GuidedMultiturnEnvRollout for forced steps; cleared
    # after the step is recorded.
    # Tuple = (text, token_ids, is_teacher_forced, is_expert). ``is_teacher_forced``
    # masks the step from the policy gradient; ``is_expert`` additionally routes
    # it to the SFT loss (fresh oracle action vs. replayed prefix).
    pending_response_override: "tuple[str, list[int], bool, bool] | None" = None

    # Reward collected immediately upon termination (before slot release)
    reward: float | None = None
    ground_truth: str | None = None

    # Cached prompt/images from the last generation step (for final batch)
    last_prompt: list[dict] | None = None
    last_images: list[Any] = field(default_factory=list)

    # Per-step phase timings (async rollout instrumentation). Each entry:
    # {"prompt_ms": float, "gen_ms": float, "env_ms": float}. Empty when
    # rollout runs in the synchronous lockstep mode.
    step_phase_times: list[dict[str, float]] = field(default_factory=list)
    # Wall-clock window of the trajectory's coroutine (for concurrency stats).
    traj_start_ts: float | None = None
    traj_end_ts: float | None = None


# ---------------------------------------------------------------------------
# MultiturnEnvRollout — the core class (parallelized via Ray SimulatorPools)
# ---------------------------------------------------------------------------


class MultiturnEnvRollout:
    """Driver-side multi-turn rollout with parallel environment operations.

    Environment operations (reset, step, build_prompt) are dispatched to
    Ray-remote SimulatorPool actors, enabling parallel AI2Thor execution
    across multiple simulators.

    Provides ``generate_trajectories()`` which is called from
    ``RayPPOTrainer._make_batch_data()`` when multi-turn mode is active.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        processor: ProcessorMixin,
        env_factory: EnvFactory,
        simulator_pools: list,
        max_depth: int = 30,
        max_prompt_length: int = 32768,
        max_response_length: int = 1024,
        min_pixels: int = 102400,
        max_pixels: int = 409600,
        prior_image_scale: float = 0.5,
        context_mode: str = "multi_turn",
        force_max_depth: bool = False,
        past_k_steps: "int | None" = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.env_factory = env_factory
        self.simulator_pools = simulator_pools
        self.max_depth = max_depth
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.prior_image_scale = prior_image_scale
        self.context_mode = context_mode
        self.force_max_depth = force_max_depth
        self.past_k_steps = past_k_steps if (past_k_steps and past_k_steps > 0) else None
        if self.past_k_steps is not None and context_mode != "multi_turn":
            logger.warning(
                f"past_k_steps={past_k_steps} is only supported with context_mode='multi_turn'; "
                f"got context_mode='{context_mode}'. Disabling truncation."
            )
            self.past_k_steps = None
        if self.past_k_steps is not None:
            logger.warning(
                f"past_k_steps={self.past_k_steps} active. Rollout uses "
                f"K-window filter + packed-MROPE injection; training uses "
                f"FlexAttention BlockMask. Train↔rollout parity has NOT yet "
                f"been verified end-to-end on GPU; treat early runs as a "
                f"parity smoke test."
            )

    # ── controller lifecycle ────────────────────────────────────────

    def warmup_controllers(self):
        """Pre-create AI2Thor controllers across all pools in parallel.

        Stagger disabled: all pools warm concurrently via Ray. Slots within a
        single pool still warm in series via SimulatorPool.warmup_controllers.
        """
        dummy_scene = self.env_factory.dataset[0]["scene_metadata"]
        n_pools = len(self.simulator_pools)
        logger.info(f"Warming up AI2Thor controllers across {n_pools} pools (parallel, no stagger)...")
        t0 = time.time()
        counts = ray.get([p.warmup_controllers.remote(dummy_scene) for p in self.simulator_pools])
        total = sum(counts)
        for i, count in enumerate(counts):
            logger.info(f"  pool {i + 1}/{n_pools}: warmed {count} controllers")
        logger.info(
            f"All AI2Thor controllers warmed up: {total} controllers "
            f"across {n_pools} pools in {time.time() - t0:.1f}s"
        )

    def destroy_controllers(self):
        """Destroy all AI2Thor controllers to free GPU memory for training."""
        logger.info("Destroying AI2Thor controllers to free GPU memory...")
        ray.get([p.destroy_all.remote() for p in self.simulator_pools])
        logger.info("All AI2Thor controllers destroyed")

    # ── public API ────────────────────────────────────────────────────

    def generate_trajectories(
        self,
        actor_rollout_ref_wg,
        batch_size: int,
        n_trajectories: int,
        config,
        metrics: dict[str, Any],
        override_config: dict[str, Any] | None = None,
    ) -> DataProto:
        """Run multi-turn rollouts with dynamic slot reuse.

        Instead of processing fixed chunks sequentially, maintains a pending
        queue of trajectories.  When a trajectory terminates, its slot is
        immediately released and a new trajectory from the queue is started in
        its place.  This keeps simulator slots fully utilised even when
        trajectory lengths vary widely.

        Within each step of the loop:
        1. Build prompts in parallel (Ray futures)
        2. Generate model responses (vLLM via actor_rollout_ref_wg)
        3. Step environments in parallel (Ray futures)
        4. Collect rewards for newly terminated trajectories
        5. Release terminated slots and refill from pending queue
        """
        from collections import deque

        total = batch_size * n_trajectories

        # Warm up AI2Thor controllers (destroyed after previous training step)
        self.warmup_controllers()

        # Determine available simulator slots
        pool_infos = ray.get(
            [p.get_pool_info.remote() for p in self.simulator_pools]
        )
        total_slots = sum(info["total"] for info in pool_infos)

        logger.info(
            f"Starting multiturn env rollout (dynamic): batch_size={batch_size}, "
            f"n={n_trajectories}, total={total}, total_slots={total_slots}, "
            f"max_depth={self.max_depth}, pools={len(self.simulator_pools)}"
        )

        t_start = time.time()

        # Sync-mode phase accumulators (async path has its own instrumentation).
        # _run_continuous_episode_loop adds to these per turn so we can split
        # rollout wall-time into gen vs env vs prompt-build.
        self._sync_gen_seconds = 0.0
        self._sync_env_seconds = 0.0
        self._sync_prompt_build_seconds = 0.0
        self._sync_step_total_seconds = 0.0
        self._sync_n_turns = 0

        # Pre-collect all dataset items (ensures deterministic ordering)
        all_items = []
        for _ in range(batch_size):
            all_items.append(self.env_factory.get_next_item())

        # Build pending queue: (group_id, n_idx, item_data)
        pending_queue: deque[tuple[int, int, dict]] = deque()
        for item_idx, item_data in enumerate(all_items):
            for n_idx in range(n_trajectories):
                pending_queue.append((item_idx, n_idx, item_data))

        # Seed initial trajectories (fill up to available slots)
        initial_count = min(len(pending_queue), total_slots)
        initial_batch = [pending_queue.popleft() for _ in range(initial_count)]

        all_trajectories: list[Trajectory] = []
        active_trajectories = self._initialize_batch(initial_batch)
        all_trajectories.extend(active_trajectories)

        logger.info(
            f"Seeded {len(active_trajectories)} initial trajectories, "
            f"{len(pending_queue)} pending in queue"
        )

        # ── Continuous episode loop with dynamic slot reuse ──
        # Async mode: each trajectory is its own coroutine; vLLM continuous
        # batching overlaps GPU work with environment steps. Falls back to
        # the synchronous lockstep loop when async_mode is off.
        async_mode = bool(getattr(config.worker.rollout, "async_mode", False))
        if async_mode:
            asyncio.run(
                self._run_async_episode_loop(
                    active_trajectories, all_trajectories,
                    pending_queue, actor_rollout_ref_wg, config,
                    total_slots=total_slots,
                    override_config=override_config,
                )
            )
        else:
            self._run_continuous_episode_loop(
                active_trajectories, all_trajectories,
                pending_queue, actor_rollout_ref_wg, config,
                override_config=override_config,
            )

        # Sort by (group_id, n_idx) for deterministic ordering expected by
        # _build_final_batch (UIDs are assigned sequentially per group).
        all_trajectories.sort(key=lambda t: (t.group_id, t.n_idx))

        # Extract rewards / ground truths (already collected on each trajectory)
        all_rewards = [
            t.reward if t.reward is not None else 0.0
            for t in all_trajectories
        ]
        all_ground_truths = [
            t.ground_truth or "{}" for t in all_trajectories
        ]

        # ── Log stats ──
        avg_steps = float(np.mean([t.num_steps for t in all_trajectories]))
        avg_reward = float(np.mean(all_rewards))
        metrics["env/avg_steps"] = avg_steps
        metrics["env/avg_reward"] = avg_reward
        metrics["reward/overall"] = avg_reward
        if async_mode:
            phase_metrics = _compute_async_phase_metrics(
                all_trajectories, total_wall_s=time.time() - t_start
            )
            metrics.update(phase_metrics)
        else:
            # Sync-path phase breakdown. Each value is the summed wall-clock
            # across all turns (turns run sequentially so this also equals the
            # in-loop wall-time spent in that phase). "other" captures
            # bookkeeping not attributed to any tracked phase.
            total_turns_wall = self._sync_step_total_seconds
            tracked = (
                self._sync_gen_seconds
                + self._sync_env_seconds
                + self._sync_prompt_build_seconds
            )
            metrics["rollout/sync_gen_seconds"] = self._sync_gen_seconds
            metrics["rollout/sync_env_step_seconds"] = self._sync_env_seconds
            metrics["rollout/sync_prompt_build_seconds"] = self._sync_prompt_build_seconds
            metrics["rollout/sync_turns_wall_seconds"] = total_turns_wall
            metrics["rollout/sync_other_seconds"] = max(0.0, total_turns_wall - tracked)
            metrics["rollout/sync_n_turns"] = self._sync_n_turns
        logger.info(
            f"Rollout complete: {len(all_trajectories)} trajectories in "
            f"{time.time() - t_start:.1f}s | "
            f"avg_steps={avg_steps:.1f} | avg_reward={avg_reward:.3f}"
        )

        logger.info(f"[mem] before destroy_controllers: RSS={_get_rss_mb():.0f}MB")

        # ── Free GPU memory for training ──
        self.destroy_controllers()

        logger.info(f"[mem] before _build_final_batch: RSS={_get_rss_mb():.0f}MB")

        # ── Build final DataProto batch ──
        batch = self._build_final_batch(
            all_trajectories, all_rewards, all_ground_truths, n_trajectories
        )

        # ── Optional: snapshot first-K trajectories for val visualization ──
        # Caller (_validate_multiturn) signals via metrics["_dump_first_k"].
        dump_k = int(metrics.pop("_dump_first_k", 0) or 0)
        if dump_k > 0:
            snapshots = []
            for t in all_trajectories[:dump_k]:
                snapshots.append({
                    "episode_id": t.episode_id,
                    "group_id": t.group_id,
                    "n_idx": t.n_idx,
                    "num_steps": t.num_steps,
                    "reward": t.reward,
                    "ground_truth": t.ground_truth,
                    "step_responses": list(t.step_responses),
                    "last_images": list(t.last_images),
                    "last_prompt": t.last_prompt,
                })
            self._last_val_dump = snapshots
        else:
            self._last_val_dump = None

        # ── Eagerly free trajectory image caches ──
        for t in all_trajectories:
            t.last_images.clear()
            t.last_prompt = None
            t.step_responses.clear()
            t.step_response_token_ids.clear()
            t.step_response_log_probs.clear()
        all_trajectories.clear()

        _force_free_memory()
        logger.info(f"[mem] after _build_final_batch + cleanup: RSS={_get_rss_mb():.0f}MB")

        return batch

    def _run_continuous_episode_loop(
        self,
        active: list[Trajectory],
        all_trajectories: list[Trajectory],
        pending_queue,
        actor_rollout_ref_wg,
        config,
        override_config: dict[str, Any] | None = None,
    ) -> None:
        """Run a continuous rollout loop with dynamic slot reuse.

        When trajectories terminate, their rewards are collected immediately,
        their slots are released, and new trajectories are initialised from
        *pending_queue* to fill the freed slots.  This keeps simulator
        utilisation high even when trajectory lengths vary widely.

        Args:
            active: mutable list of currently running trajectories.
            all_trajectories: global list — newly created trajectories are
                appended here as well.
            pending_queue: deque of ``(group_id, n_idx, item_data)`` for
                trajectories that have not yet started.
            actor_rollout_ref_wg: the vLLM worker group.
            config: training config (for rollout temperature, etc.).
        """
        global_step = 0

        while True:
            # Optional subclass hook fired at the top of every iteration.
            # Used by ``GuidedMultiturnEnvRollout`` to populate per-trajectory
            # ``pending_response_override`` values before the prompt-build
            # phase reads them. No-op by default.
            pre_step_hook = getattr(self, "_pre_step_hook", None)
            if pre_step_hook is not None:
                pre_step_hook(active)

            # Force-terminate trajectories that hit max_depth
            for t in active:
                if not t.terminated and t.num_steps >= self.max_depth:
                    t.terminated = True

            # Harvest any trajectories that became terminated (either from
            # the previous step's env results or from the max_depth check
            # above).  Collect rewards, release slots, refill.
            newly_done = [t for t in active if t.terminated and t.reward is None]
            if newly_done:
                self._harvest_and_refill(
                    newly_done, active, all_trajectories, pending_queue
                )

            # Remove terminated from active
            active[:] = [t for t in active if not t.terminated]

            if not active:
                break

            t_step = time.time()

            # ── Build prompts in parallel via Ray ──
            # When past_k_steps is set we also need the unfiltered
            # full-trajectory messages on the driver to compute packed MROPE
            # positions. Use build_prompt_with_full in that case.
            t_pb = time.time()
            use_full = self.past_k_steps is not None
            if use_full:
                prompt_futures = [
                    t.pool.build_prompt_with_full.remote(t.slot_id) for t in active
                ]
            else:
                prompt_futures = [
                    t.pool.build_prompt.remote(t.slot_id) for t in active
                ]
            prompt_results_raw = ray.get(prompt_futures)
            self._sync_prompt_build_seconds += time.time() - t_pb

            # Filter out trajectories whose prompt build failed
            valid = []          # (local_idx, trajectory)
            prompts = []
            images_list = []
            full_prompts: list = []
            full_images_list: list = []
            for i, t in enumerate(active):
                result = prompt_results_raw[i]
                if isinstance(result, Exception):
                    logger.warning(
                        f"  [step {global_step}][grp {t.group_id}/{t.n_idx}] "
                        f"build_prompt failed: {result}. Terminating."
                    )
                    t.terminated = True
                    continue
                try:
                    prompts.append(result[0])
                    images_list.append(result[1])
                    if use_full:
                        full_prompts.append(result[2])
                        full_images_list.append(result[3])
                    valid.append((len(prompts) - 1, t))
                except Exception as e:
                    logger.warning(
                        f"  [step {global_step}][grp {t.group_id}/{t.n_idx}] "
                        f"build_prompt result invalid: {e}. Terminating."
                    )
                    t.terminated = True

            if not valid:
                logger.warning(
                    f"  step {global_step}: all prompts failed, "
                    f"will harvest and try to refill"
                )
                global_step += 1
                continue

            # Cache prompt/images for final batch construction. With
            # past_k_steps, training reconstructs the *full* trajectory, so
            # cache the full pair when available.
            for local_i, t in valid:
                if use_full:
                    t.last_prompt = full_prompts[local_i]
                    t.last_images = list(full_images_list[local_i])
                else:
                    t.last_prompt = prompts[local_i]
                    t.last_images = list(images_list[local_i])

            # ── Partition into vLLM and override branches ────────────────
            # Trajectories with a pending response override (set by the
            # guided rollout driver for forced/replay steps) skip vLLM and
            # reuse the provided text + token IDs. Order in `valid` is
            # preserved so downstream indexing into prompts/images_list
            # still matches per-trajectory state.
            override_idx_set = {
                local_i for local_i, t in valid
                if t.pending_response_override is not None
            }
            vllm_valid = [(li, t) for li, t in valid if li not in override_idx_set]
            override_valid = [(li, t) for li, t in valid if li in override_idx_set]

            if logger.isEnabledFor(logging.DEBUG):
                for local_i, t in valid:
                    msgs = prompts[local_i]
                    imgs = images_list[local_i]
                    lines = [
                        f"  [step {global_step}][grp {t.group_id}/{t.n_idx}] "
                        f"PROMPT ({len(imgs)} images):"
                    ]
                    img_idx = 0
                    for msg in msgs:
                        role = msg["role"].upper()
                        content = msg.get("content", "")
                        if role == "SYSTEM":
                            lines.append(f'  [{role}] "{content[:80]}..."')
                        elif role == "USER":
                            parts = content.split("<image>")
                            for pi, part in enumerate(parts):
                                text = part.strip()
                                if text:
                                    display = text if len(text) <= 200 else f"...{text[-200:]}"
                                    lines.append(f'  [{role}] Text: "{display}"')
                                if pi < len(parts) - 1:
                                    lines.append(f"  [{role}] Image: [image_{img_idx}]")
                                    img_idx += 1
                        elif role == "ASSISTANT":
                            lines.append(f'  [{role}] Text: "{content}"')
                    logger.debug("\n".join(lines))

            # ── Tokenize into DataProto format (vLLM subset only) ──
            t_gen = time.time()
            rollout_log_probs = None
            response_ids = None
            response_mask = None
            if vllm_valid:
                vllm_local_idxs = [li for li, _ in vllm_valid]
                vllm_prompts = [prompts[li] for li in vllm_local_idxs]
                vllm_images = [images_list[li] for li in vllm_local_idxs]
                vllm_full_prompts = (
                    [full_prompts[li] for li in vllm_local_idxs] if use_full else None
                )
                vllm_full_images = (
                    [full_images_list[li] for li in vllm_local_idxs] if use_full else None
                )
                tokenized = self._tokenize_prompts(
                    vllm_prompts,
                    vllm_images,
                    for_generation=True,
                    full_prompts=vllm_full_prompts,
                    full_images_list=vllm_full_images,
                )
                meta_info = {
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                    "video_fps": getattr(config.data, "video_fps", 2.0),
                    "n": 1,
                    "temperature": config.worker.rollout.temperature,
                    "top_p": config.worker.rollout.top_p,
                }
                if override_config:
                    meta_info.update(override_config)
                    meta_info["n"] = 1  # always 1 per-step for multiturn
                gen_batch = DataProto.from_single_dict(
                    tokenized, meta_info=meta_info
                )

                # ── Generate model responses via vLLM ──
                if logger.isEnabledFor(logging.DEBUG):
                    prompt_lens = [len(ids) for ids in tokenized["raw_prompt_ids"]]
                    logger.debug(
                        f"  [step {global_step}] TOKENIZED: "
                        f"n_prompts={len(prompt_lens)}, "
                        f"token_counts={prompt_lens}, "
                        f"input_ids_shape={tokenized['input_ids'].shape}"
                    )

                gen_batch, pad_size = pad_dataproto_to_divisor(
                    gen_batch, actor_rollout_ref_wg.world_size
                )
                gen_output = actor_rollout_ref_wg.generate_sequences(gen_batch)
                gen_output = unpad_dataproto(gen_output, pad_size)
                response_ids = gen_output.batch["responses"]
                response_mask = gen_output.batch["response_mask"]
                rollout_log_probs = gen_output.batch.get("rollout_log_probs")
            t_gen_elapsed = time.time() - t_gen

            # Build unified per-`valid` response arrays, drawing from vLLM
            # output or pending_response_override as appropriate.
            responses: list[str] = [""] * len(valid)
            response_token_ids_list: list[list[int]] = [[] for _ in valid]
            response_log_probs_list: list[list[float]] = [[] for _ in valid]
            teacher_forced_list: list[bool] = [False] * len(valid)
            # Subset of teacher_forced: fresh expert (oracle) steps → SFT targets.
            is_expert_list: list[bool] = [False] * len(valid)

            # vLLM branch
            for vllm_i, (local_i, _t) in enumerate(vllm_valid):
                # `local_i` is the index into `valid` (== `prompts`/`images_list`).
                length = int(response_mask[vllm_i].sum().item())
                text = self.tokenizer.decode(
                    response_ids[vllm_i][:length], skip_special_tokens=True
                )
                # Locate this trajectory's slot in `valid` for indexing.
                # `valid` is iterated in the same order as `local_i` insertions,
                # so we can match by local_i directly.
                valid_pos = next(
                    j for j, (li, _) in enumerate(valid) if li == local_i
                )
                responses[valid_pos] = text
                response_token_ids_list[valid_pos] = response_ids[vllm_i][:length].tolist()
                if rollout_log_probs is not None:
                    response_log_probs_list[valid_pos] = (
                        rollout_log_probs[vllm_i][:length].cpu().tolist()
                    )

            # Override branch (forced / replay)
            for local_i, t in override_valid:
                ov = t.pending_response_override
                # Tolerate legacy 3-tuples (text, ids, is_teacher); a 4th entry
                # is_expert marks fresh oracle steps for the SFT loss.
                ov_text, ov_token_ids, ov_is_teacher = ov[0], ov[1], ov[2]
                ov_is_expert = bool(ov[3]) if len(ov) > 3 else False
                valid_pos = next(
                    j for j, (li, _) in enumerate(valid) if li == local_i
                )
                responses[valid_pos] = ov_text
                response_token_ids_list[valid_pos] = list(ov_token_ids)
                response_log_probs_list[valid_pos] = []  # no sampled logprobs
                teacher_forced_list[valid_pos] = bool(ov_is_teacher)
                is_expert_list[valid_pos] = ov_is_expert
                # Consume the override so the next step uses vLLM (unless the
                # driver sets another override).
                t.pending_response_override = None

            if logger.isEnabledFor(logging.DEBUG):
                for i, (_local_i, t) in enumerate(valid):
                    resp_len = len(response_token_ids_list[i])
                    tag = "TEACHER" if teacher_forced_list[i] else "ASSISTANT"
                    logger.debug(
                        f"  [step {global_step}][grp {t.group_id}/{t.n_idx}] "
                        f"RESPONSE ({resp_len} tokens, {tag}):\n"
                        f'  [{tag}] Text: "{responses[i]}"'
                    )

            # ── Step environments in parallel via Ray ──
            t_env = time.time()
            step_futures = [
                t.pool.step_env.remote(t.slot_id, responses[i])
                for i, (_, t) in enumerate(valid)
            ]

            # Collect results one-by-one so a single env failure doesn't
            # block the entire batch.
            step_results = []
            for f in step_futures:
                try:
                    step_results.append(ray.get(f))
                except Exception as e:
                    step_results.append(e)
            t_env_elapsed = time.time() - t_env
            self._sync_env_seconds += t_env_elapsed
            self._sync_gen_seconds += t_gen_elapsed
            self._sync_n_turns += 1

            # Process step results
            n_terminated_this_step = 0
            n_failed_this_step = 0
            for i, (_, t) in enumerate(valid):
                t.step_responses.append(responses[i])
                t.step_response_token_ids.append(response_token_ids_list[i])
                t.step_response_log_probs.append(response_log_probs_list[i])
                t.step_teacher_forced.append(teacher_forced_list[i])
                t.step_is_expert.append(is_expert_list[i])
                t.num_steps += 1

                result = step_results[i]
                if isinstance(result, Exception):
                    logger.warning(
                        f"  [step {global_step}][grp {t.group_id}/{t.n_idx}] "
                        f"step_env FAILED: {result}. Terminating."
                    )
                    t.terminated = True
                    n_terminated_this_step += 1
                    n_failed_this_step += 1
                    t.step_distances.append(None)
                    continue

                reward, terminated, _info = result
                t.step_distances.append(_info.get("dist_before"))
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"  [step {global_step}][grp {t.group_id}/{t.n_idx}] "
                        f"ENV STEP: action_type={_info.get('action_type', '?')}, "
                        f"reward={reward:.3f}, terminated={terminated}"
                    )
                if terminated:
                    if self.force_max_depth and t.num_steps < self.max_depth:
                        logger.info(
                            f"  [step {global_step}][grp {t.group_id}/{t.n_idx}] "
                            f"force_max_depth: ignoring env termination at step "
                            f"{t.num_steps}/{self.max_depth}"
                        )
                    else:
                        t.terminated = True
                        n_terminated_this_step += 1

            t_step_elapsed = time.time() - t_step
            self._sync_step_total_seconds += t_step_elapsed
            logger.info(
                f"  step {global_step}: {len(valid)} active, "
                f"{n_terminated_this_step} done"
                + (f" ({n_failed_this_step} failed)" if n_failed_this_step else "")
                + f", gen={t_gen_elapsed:.1f}s, env={t_env_elapsed:.1f}s, "
                f"total={t_step_elapsed:.1f}s"
                + (f", pending={len(pending_queue)}" if pending_queue else "")
                + f", RSS={_get_rss_mb():.0f}MB"
            )

            # ── Parity-mode context dump: visible turn IDs + prompt extents ──
            # For the first few traj only, so the log doesn't explode.
            # Indices here refer to the vLLM subset; overridden/forced steps
            # are skipped (they didn't go through vLLM).
            if rollout_log_probs is not None and vllm_valid:
                prompt_ids_b = gen_batch.batch.get("input_ids")
                attn_b = gen_batch.batch.get("attention_mask")
                for i, (_, t) in enumerate(vllm_valid[:2]):
                    if prompt_ids_b is None or attn_b is None:
                        break
                    pl = int(attn_b[i].sum().item())
                    asst_steps = len(t.step_responses)
                    expected_visible = (
                        list(range(asst_steps + 1))
                        if (self.past_k_steps is None or asst_steps < self.past_k_steps)
                        else [0] + list(
                            range(asst_steps - self.past_k_steps + 1, asst_steps + 1)
                        )
                    )
                    logger.info(
                        f"    [parity] step {global_step} traj {i}: "
                        f"prompt_len={pl}, accumulated_steps={asst_steps}, "
                        f"expected_visible_obs_turns={expected_visible}, "
                        f"resp_logprob_mean={float(rollout_log_probs[i][:int(response_mask[i].sum().item())].mean().item() if int(response_mask[i].sum().item()) else 0.0):.3f}"
                    )

            global_step += 1

    # ── Async per-trajectory episode loop (rollout.async_mode=true) ─────

    async def _run_async_episode_loop(
        self,
        active: list[Trajectory],
        all_trajectories: list[Trajectory],
        pending_queue,
        actor_rollout_ref_wg,
        config,
        total_slots: int,
        override_config: dict[str, Any] | None = None,
    ) -> None:
        """Per-trajectory async rollout: each trajectory is its own coroutine.

        Replaces the lockstep `_run_continuous_episode_loop` with a true
        per-trajectory model. As soon as a trajectory's env step returns, its
        coroutine submits the next vLLM request via
        `wg.workers[rank].generate_one_async.remote(...)` — vLLM's continuous
        batching handles GPU sharing across all in-flight requests. Stragglers
        no longer idle the GPU; vLLM no longer idles all simulators.

        Trajectory→DP-rank routing is round-robin; with TP=1 each DP rank has
        its own engine, so independent trajectories landing on the same rank
        share that rank's engine via vLLM batching.

        Drain barrier is implicit: this coroutine returns only after every
        scheduled trajectory has completed (and thus its `generate_one`
        future has resolved), so the engine is naturally idle by the time
        the trainer calls `release_rollout_engine`.
        """
        n_workers = len(actor_rollout_ref_wg.workers)
        if n_workers <= 0:
            raise RuntimeError("actor_rollout_ref_wg.workers is empty; cannot run async rollout.")

        # Engine partitioning under TP>1: every TP-sized contiguous block of
        # workers is one vLLM engine sharded across `tp_size` ranks. A request
        # must be fanned out to all `tp_size` ranks of the chosen engine so
        # the `AsyncRequestRouter._step_loop_tp` lockstep broadcast finds the
        # payload on every rank's inbox; otherwise slaves time out in stage 3.
        tp_size = max(1, int(getattr(config.worker.rollout, "tensor_parallel_size", 1)))
        if n_workers % tp_size != 0:
            raise RuntimeError(
                f"n_workers={n_workers} is not a multiple of "
                f"tensor_parallel_size={tp_size}; cannot partition into engines."
            )
        n_engines = n_workers // tp_size

        # Counters used as round-robin assignments and for logging.
        _engine_counter = {"i": 0}
        _running = {"n": 0}
        _completed = {"n": 0}
        _failed_steps = {"n": 0}

        def _next_engine_ranks() -> list[int]:
            """Round-robin next engine, return its `tp_size` Ray worker ranks."""
            e = _engine_counter["i"] % n_engines
            _engine_counter["i"] += 1
            base = e * tp_size
            return list(range(base, base + tp_size))

        t_start = time.time()

        async def run_traj(t: Trajectory) -> Trajectory:
            tp_ranks = _next_engine_ranks()
            _running["n"] += 1
            t.traj_start_ts = time.time()
            try:
                while not t.terminated and t.num_steps < self.max_depth:
                    use_full = self.past_k_steps is not None
                    t_phase_start = time.time()
                    try:
                        if use_full:
                            messages, images, full_messages, full_images = (
                                await t.pool.build_prompt_with_full.remote(t.slot_id)
                            )
                        else:
                            messages, images = await t.pool.build_prompt.remote(t.slot_id)
                            full_messages, full_images = None, None
                    except Exception as e:
                        logger.warning(
                            f"  [async grp {t.group_id}/{t.n_idx}] build_prompt failed: {e}. Terminating."
                        )
                        t.terminated = True
                        break

                    # Cache for final batch (full trajectory when past_k is on).
                    if use_full:
                        t.last_prompt = full_messages
                        t.last_images = list(full_images or [])
                    else:
                        t.last_prompt = messages
                        t.last_images = list(images or [])

                    # Tokenize this single prompt; reuse the batch helper with a
                    # 1-element list to keep packed-MROPE + image-truncation
                    # logic identical to the sync path.
                    try:
                        tokenized = self._tokenize_prompts(
                            [messages],
                            [images],
                            for_generation=True,
                            full_prompts=[full_messages] if use_full else None,
                            full_images_list=[full_images] if use_full else None,
                        )
                    except Exception as e:
                        logger.warning(
                            f"  [async grp {t.group_id}/{t.n_idx}] tokenize failed: {e}. Terminating."
                        )
                        t.terminated = True
                        break

                    raw_prompt_ids = list(tokenized["raw_prompt_ids"][0])
                    raw_images = tokenized["raw_images"][0]
                    packed_override = None
                    if "packed_mrope_overrides" in tokenized:
                        packed_override = tokenized["packed_mrope_overrides"][0]

                    # vLLM expects the multi-modal dict shape that
                    # `_process_multi_modal_data` produces (`{"image": [...]}`)
                    # so the rollout side doesn't need to know about it again.
                    mm_data = {"images": raw_images} if raw_images else None

                    sampling_overrides: dict[str, Any] = {
                        # Multiturn always samples 1 response per step. The
                        # group-of-n enumeration happens at the trajectory level
                        # (n_trajectories items in the pending queue), not via
                        # vLLM's `n` parameter.
                        "n": 1,
                        "temperature": config.worker.rollout.temperature,
                        "top_p": config.worker.rollout.top_p,
                    }
                    rollout_top_k = getattr(config.worker.rollout, "top_k", -1)
                    if rollout_top_k != -1:
                        sampling_overrides["top_k"] = rollout_top_k
                    if override_config:
                        for k in ("temperature", "top_p", "top_k"):
                            if k in override_config:
                                sampling_overrides[k] = override_config[k]

                    request_id = f"async-{t.episode_id}-step{t.num_steps}-{uuid.uuid4().hex[:6]}"
                    t_gen_start = time.time()
                    prompt_ms = (t_gen_start - t_phase_start) * 1000.0
                    try:
                        # Fan out the request to every TP rank of the chosen
                        # engine. The lockstep router in `vllm_rollout_spmd.py`
                        # requires every rank's inbox to receive the payload
                        # so all ranks can call add_request in lockstep. All
                        # ranks compute identical outputs (sampler is
                        # TP-replicated with same seed), so we use rank 0's
                        # result and discard the rest.
                        remote_calls = [
                            actor_rollout_ref_wg.workers[r].actor_rollout_ref_generate_one_async.remote(
                                request_id=request_id,
                                prompt_token_ids=raw_prompt_ids,
                                multi_modal_data=mm_data,
                                sampling_overrides=sampling_overrides,
                                packed_mrope_override=packed_override,
                                min_pixels=self.min_pixels,
                                max_pixels=self.max_pixels,
                                video_fps=getattr(config.data, "video_fps", 2.0),
                            )
                            for r in tp_ranks
                        ]
                        results = await asyncio.gather(*remote_calls)
                        result = results[0]
                    except Exception as e:
                        logger.warning(
                            f"  [async grp {t.group_id}/{t.n_idx}] generate_one_async failed: {e}. Terminating."
                        )
                        t.terminated = True
                        _failed_steps["n"] += 1
                        break

                    t_gen_done = time.time()
                    gen_ms = (t_gen_done - t_gen_start) * 1000.0
                    text = self.tokenizer.decode(result.token_ids, skip_special_tokens=True)
                    t.step_responses.append(text)
                    t.step_response_token_ids.append(list(result.token_ids))
                    t.step_response_log_probs.append(list(result.logprobs) if result.logprobs is not None else [])
                    t.num_steps += 1

                    try:
                        reward, terminated, info = await t.pool.step_env.remote(t.slot_id, text)
                    except Exception as e:
                        logger.warning(
                            f"  [async grp {t.group_id}/{t.n_idx}] step_env failed: {e}. Terminating."
                        )
                        t.terminated = True
                        t.step_distances.append(None)
                        break
                    t.step_distances.append(info.get("dist_before"))

                    env_ms = (time.time() - t_gen_done) * 1000.0
                    t.step_phase_times.append({
                        "prompt_ms": prompt_ms,
                        "gen_ms": gen_ms,
                        "env_ms": env_ms,
                    })

                    if terminated:
                        if self.force_max_depth and t.num_steps < self.max_depth:
                            logger.info(
                                f"  [async grp {t.group_id}/{t.n_idx}] force_max_depth: "
                                f"ignoring env termination at step {t.num_steps}/{self.max_depth}"
                            )
                        else:
                            t.terminated = True

                # Trajectory finished (terminated or hit max_depth). Collect
                # reward + ground truth, then release the slot.
                try:
                    t.reward = float(await t.pool.get_trajectory_reward.remote(t.slot_id))
                except Exception as e:
                    logger.warning(
                        f"  [async grp {t.group_id}/{t.n_idx}] get_trajectory_reward failed: {e}. Using 0.0."
                    )
                    t.reward = 0.0
                try:
                    t.ground_truth = await t.pool.get_ground_truth.remote(t.slot_id)
                except Exception as e:
                    logger.warning(
                        f"  [async grp {t.group_id}/{t.n_idx}] get_ground_truth failed: {e}."
                    )
                    t.ground_truth = "{}"
                try:
                    await t.pool.release_env.remote(t.slot_id)
                except Exception:
                    pass
                _completed["n"] += 1
                t.traj_end_ts = time.time()
                return t
            finally:
                _running["n"] -= 1

        # Schedule the initial set of trajectories and refill from
        # pending_queue as slots open up. The total number of in-flight
        # trajectories never exceeds total_slots — we acquire a slot via
        # `_initialize_batch` only when we're ready to schedule a coroutine.
        async def scheduler() -> None:
            tasks: set[asyncio.Task] = {asyncio.create_task(run_traj(t)) for t in active}
            while tasks:
                done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for d in done:
                    # Surface coroutine exceptions instead of swallowing them.
                    exc = d.exception()
                    if exc is not None:
                        logger.error(f"async trajectory coroutine raised: {exc!r}")
                # Refill from the pending queue. A completed trajectory does
                # NOT guarantee a reusable slot: SimulatorPool.release_env can
                # reactively *disable* the freed slot under GPU memory pressure
                # (reactive shrink). Trusting free_slots=len(done) then
                # over-refills into capacity that no longer exists, and
                # _initialize_batch raises "No simulator slots available across
                # any pool" — crashing the whole rollout (observed during
                # val_before_train on the 4-GPU/32-slot configs, which run hot
                # enough to trip shrink). Instead, query the pools' real
                # availability each wave and schedule only what actually fits;
                # leftover items stay queued for a later completion. The pool's
                # enabled-slot floor (>=1) guarantees forward progress.
                if pending_queue:
                    avail = sum(
                        ray.get(
                            [p.get_available_count.remote() for p in self.simulator_pools]
                        )
                    )
                    n_refill = min(avail, len(pending_queue))
                    for _ in range(n_refill):
                        refill_batch = [pending_queue.popleft()]
                        new_trajs = self._initialize_batch(refill_batch)
                        all_trajectories.extend(new_trajs)
                        for nt in new_trajs:
                            tasks.add(asyncio.create_task(run_traj(nt)))

        # Periodic progress logger so long rollouts don't appear silent.
        async def progress_logger() -> None:
            while True:
                await asyncio.sleep(15.0)
                logger.info(
                    f"  [async] running={_running['n']} "
                    f"completed={_completed['n']} "
                    f"pending={len(pending_queue)} "
                    f"failed_steps={_failed_steps['n']} "
                    f"elapsed={time.time() - t_start:.0f}s "
                    f"RSS={_get_rss_mb():.0f}MB"
                )

        plog = asyncio.create_task(progress_logger())
        try:
            await scheduler()
        finally:
            plog.cancel()
            try:
                await plog
            except (asyncio.CancelledError, Exception):
                pass

        logger.info(
            f"  [async] DONE completed={_completed['n']} "
            f"failed_steps={_failed_steps['n']} "
            f"elapsed={time.time() - t_start:.1f}s"
        )

    def _harvest_and_refill(
        self,
        newly_done: list[Trajectory],
        active: list[Trajectory],
        all_trajectories: list[Trajectory],
        pending_queue,
    ) -> None:
        """Collect rewards for terminated trajectories, release their slots,
        and start new trajectories from the pending queue.
        """
        # ── Collect rewards and release slots in parallel ──
        reward_futures = [
            t.pool.get_trajectory_reward.remote(t.slot_id)
            for t in newly_done
        ]
        gt_futures = [
            t.pool.get_ground_truth.remote(t.slot_id)
            for t in newly_done
        ]
        # Batch-collect all reward and ground_truth futures at once
        n_done = len(newly_done)
        all_futures = reward_futures + gt_futures
        all_results = []
        for f in all_futures:
            try:
                all_results.append(ray.get(f))
            except Exception as e:
                all_results.append(e)

        for i, t in enumerate(newly_done):
            reward_result = all_results[i]
            gt_result = all_results[n_done + i]
            if isinstance(reward_result, Exception):
                logger.warning(
                    f"get_trajectory_reward failed for grp {t.group_id}/{t.n_idx}: "
                    f"{reward_result}, using 0.0"
                )
                t.reward = 0.0
            else:
                t.reward = reward_result
            if isinstance(gt_result, Exception):
                logger.warning(
                    f"get_ground_truth failed for grp {t.group_id}/{t.n_idx}: {gt_result}"
                )
                t.ground_truth = "{}"
            else:
                t.ground_truth = gt_result

        # ── Release slots ──
        release_futures = [
            t.pool.release_env.remote(t.slot_id) for t in newly_done
        ]
        ray.get(release_futures)  # batch-collect; failures are non-critical

        # ── Refill from pending queue ──
        # Cap by actual available capacity: reactive shrink in `release_env`
        # may disable some of the slots we just released (memory pressure),
        # so newly_done is not the same as newly-available. Without this cap,
        # _initialize_batch would request more slots than the pool can serve
        # and raise. Unfilled work stays in pending_queue for the next round.
        total_available = sum(
            ray.get([p.get_available_count.remote() for p in self.simulator_pools])
        )
        n_to_fill = min(len(newly_done), len(pending_queue), total_available)
        if n_to_fill > 0:
            refill_batch = [pending_queue.popleft() for _ in range(n_to_fill)]
            new_trajs = self._initialize_batch(refill_batch)
            all_trajectories.extend(new_trajs)
            active.extend(new_trajs)
            logger.info(
                f"  refill: released {len(newly_done)} slots "
                f"(available={total_available}), "
                f"started {n_to_fill} new trajectories, "
                f"{len(pending_queue)} still pending"
            )
        elif newly_done:
            logger.info(
                f"  released {len(newly_done)} slots "
                f"(available={total_available}, queue empty or pool full)"
            )

    # ── private helpers ───────────────────────────────────────────────

    def _initialize_batch(
        self, batch: list[tuple[int, int, dict]]
    ) -> list[Trajectory]:
        """Acquire slots, reset envs, and return Trajectory objects.

        Args:
            batch: list of ``(group_id, n_idx, item_data)`` tuples.

        Returns:
            List of ready-to-run Trajectory objects.
        """
        total = len(batch)
        if total == 0:
            return []
        num_pools = len(self.simulator_pools)

        # Query available slot counts so we assign to pools with capacity
        # (avoids the round-robin bug where a full pool gets assigned work)
        avail_counts = ray.get(
            [p.get_available_count.remote() for p in self.simulator_pools]
        )

        acquire_meta = []  # (pool, group_id, n_idx)
        acquire_futures = []
        for group_id, n_idx, item_data in batch:
            # Pick the pool with the most available slots
            pool_idx = max(range(num_pools), key=lambda j: avail_counts[j])
            if avail_counts[pool_idx] <= 0:
                raise RuntimeError(
                    f"No simulator slots available across any pool for "
                    f"group={group_id}, n={n_idx}. "
                    f"Increase num_simulators or reduce batch_size."
                )
            avail_counts[pool_idx] -= 1
            pool = self.simulator_pools[pool_idx]
            acquire_meta.append((pool, group_id, n_idx))
            acquire_futures.append(pool.acquire_env.remote(item_data))

        t0 = time.time()
        slot_ids = ray.get(acquire_futures)
        logger.info(f"  acquire_env ({total}): {time.time() - t0:.1f}s")

        # Validate all slots were acquired
        for i, slot_id in enumerate(slot_ids):
            if slot_id is None:
                # Release already-acquired slots before raising
                for j in range(i):
                    if slot_ids[j] is not None:
                        try:
                            ray.get(acquire_meta[j][0].release_env.remote(slot_ids[j]))
                        except Exception:
                            pass
                _, group_id, n_idx = acquire_meta[i]
                raise RuntimeError(
                    f"Failed to acquire simulator slot for group={group_id}, "
                    f"n={n_idx}. Not enough slots available. "
                    f"Increase num_simulators or reduce batch_size."
                )

        # Reset all environments in parallel
        t0 = time.time()
        reset_futures = [
            acquire_meta[i][0].reset_env.remote(slot_ids[i])
            for i in range(total)
        ]
        ray.get(reset_futures)
        logger.info(f"  reset_env ({total}): {time.time() - t0:.1f}s")

        # Build Trajectory objects
        trajectories = []
        for i, (pool, group_id, n_idx) in enumerate(acquire_meta):
            episode_id = f"grpo_{uuid.uuid4().hex[:8]}_{n_idx}"
            trajectories.append(
                Trajectory(
                    pool=pool,
                    slot_id=slot_ids[i],
                    episode_id=episode_id,
                    group_id=group_id,
                    n_idx=n_idx,
                )
            )

        return trajectories

    def _tokenize_prompts(
        self,
        prompts: list[list[dict]],
        images_list: list[list[Image.Image]],
        for_generation: bool = False,
        add_generation_prompt: bool = True,
        full_prompts: list[list[dict]] | None = None,
        full_images_list: list[list[Image.Image]] | None = None,
    ) -> dict[str, Any]:
        """Tokenize prompts with images into the format expected by generate_sequences.

        Args:
            prompts: list of chat message lists, each message is a dict with
                "role" and "content" keys. Content may contain ``<image>``
                placeholders that correspond to entries in *images_list*.
            images_list: list of image lists, one per prompt.
            for_generation: if True, skip computing multi_modal_data
                (pixel_values tensors) since only raw_images are needed
                for vLLM generation. Saves significant CPU memory.

        Returns a dict with:
          - input_ids: (bs, max_prompt_length) tensor, left-padded
          - attention_mask: (bs, max_prompt_length) tensor
          - position_ids: (bs, 4, max_prompt_length) tensor (Qwen3-VL mrope)
          - raw_prompt_ids: list of unpadded token ID lists (for vLLM)
          - multi_modal_data: list of {"images": [...]} dicts (omitted when for_generation=True)
        """
        from ...utils.dataset import process_image
        from ...utils.torch_functional import postprocess_data

        batch_input_ids = []
        batch_attention_mask = []
        batch_position_ids = []
        batch_raw_prompt_ids = []
        batch_multi_modal_data = []
        batch_raw_images = []
        batch_packed_overrides: list = []  # per-request (positions, delta) or None

        compute_packed = (
            full_prompts is not None
            and full_images_list is not None
            and len(full_prompts) == len(prompts)
        )

        for idx, (messages, images) in enumerate(zip(prompts, images_list)):
            # Convert <image> placeholders in message content to HF format
            hf_messages = []
            for msg in messages:
                content = msg["content"]
                if isinstance(content, str) and "<image>" in content:
                    content_list = []
                    for i, part in enumerate(content.split("<image>")):
                        if i != 0:
                            content_list.append({"type": "image"})
                        if part:
                            content_list.append({"type": "text", "text": part})
                    hf_messages.append({"role": msg["role"], "content": content_list})
                elif isinstance(content, str):
                    hf_messages.append(msg)
                else:
                    # Already in HF format (list of content dicts)
                    hf_messages.append(msg)

            # Apply chat template
            text_prompt = self.processor.apply_chat_template(
                hf_messages, add_generation_prompt=add_generation_prompt, tokenize=False
            )

            # Process images: downscale prior images (all but last) to match
            # the SFT training setup (prior_image_scale), then apply pixel
            # constraints via process_image.
            # Keep raw_images (after downscale, before process_image) for vLLM
            # generation — vLLM does its own image processing internally.
            if images:
                raw_images = []
                processed_images = []
                for i, img in enumerate(images):
                    # Ensure PIL Image (AI2Thor returns numpy arrays)
                    if isinstance(img, np.ndarray):
                        img = Image.fromarray(img)
                    elif isinstance(img, str):
                        img = Image.open(img)
                    if i < len(images) - 1 and self.prior_image_scale < 1.0:
                        # Downscale prior observation images
                        new_w = max(1, int(img.width * self.prior_image_scale))
                        new_h = max(1, int(img.height * self.prior_image_scale))
                        img = img.resize((new_w, new_h), Image.LANCZOS)
                    raw_images.append(img)
                    processed_images.append(
                        process_image(img, self.min_pixels, self.max_pixels)
                    )
            else:
                raw_images = []
                processed_images = None

            # Tokenize with processor
            model_inputs = self.processor(
                processed_images,
                [text_prompt],
                add_special_tokens=False,
                return_tensors="pt",
            )
            del processed_images  # free memory; raw_images kept for vLLM
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

            # Compute Qwen3-VL mrope position IDs
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from ...models.transformers.qwen3_vl import get_rope_index
            else:
                from ...models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=None,
                second_per_grid_ts=None,
                attention_mask=attention_mask,
            )
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)
            position_ids = torch.cat(
                (text_position_ids, vision_position_ids), dim=0
            )

            # Pad/truncate
            input_ids, attention_mask, position_ids = postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                max_length=self.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation="right",
            )

            # After truncation, pixel_values/image_grid_thw may have more images
            # than surviving <|image_pad|> tokens. Trim to only complete images.
            image_grid_thw = model_inputs.get("image_grid_thw", None)
            if image_grid_thw is not None and len(image_grid_thw) > 0:
                image_token_id = self.processor.image_token_id
                merge_size = self.processor.image_processor.merge_size
                n_image_tokens = (input_ids == image_token_id).sum().item()

                # features_per_image[i] = number of <|image_pad|> tokens for image i
                features_per_image = []
                patches_per_image = []
                for thw in image_grid_thw:
                    t, h, w = thw[0].item(), thw[1].item(), thw[2].item()
                    features_per_image.append(t * (h // merge_size) * (w // merge_size))
                    patches_per_image.append(t * h * w)

                total_features = sum(features_per_image)
                if n_image_tokens < total_features:
                    # Find how many complete images survive truncation
                    cumulative = 0
                    n_keep = 0
                    for f in features_per_image:
                        if cumulative + f <= n_image_tokens:
                            cumulative += f
                            n_keep += 1
                        else:
                            break

                    # Trim multi-modal tensors
                    model_inputs["image_grid_thw"] = image_grid_thw[:n_keep]
                    if n_keep > 0:
                        total_patches_keep = sum(patches_per_image[:n_keep])
                        model_inputs["pixel_values"] = model_inputs["pixel_values"][:total_patches_keep]
                    else:
                        model_inputs.pop("pixel_values", None)
                        model_inputs.pop("image_grid_thw", None)

                    if n_keep < len(image_grid_thw):
                        logger.warning(
                            f"Right-truncation dropped {len(image_grid_thw) - n_keep} "
                            f"image(s) (kept {n_keep}/{len(image_grid_thw)}). "
                            f"If the current observation was truncated, the model "
                            f"is acting on stale prior observations. Consider "
                            f"increasing max_prompt_length or reducing max_observations."
                        )

                    # Mask out any leftover partial image tokens
                    excess = n_image_tokens - cumulative
                    if excess > 0:
                        image_positions = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
                        partial_positions = image_positions[cumulative:]
                        input_ids[partial_positions] = self.tokenizer.pad_token_id
                        attention_mask[partial_positions] = 0
                        position_ids[:, partial_positions] = 0

            # Raw prompt IDs for vLLM (text-only tokenization)
            raw_prompt_ids = self.tokenizer.encode(
                text_prompt, add_special_tokens=False
            )
            if len(raw_prompt_ids) > self.max_prompt_length:
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
                # After truncation, some image placeholders may have been cut.
                # Trim raw_images to match surviving placeholders so vLLM
                # doesn't get more images than placeholder tokens.
                if raw_images:
                    image_pad_token_id = self.processor.image_token_id
                    n_surviving = sum(1 for tid in raw_prompt_ids if tid == image_pad_token_id)
                    if n_surviving < len(raw_images):
                        logger.warning(
                            f"raw_prompt_ids truncation dropped {len(raw_images) - n_surviving} "
                            f"image placeholder(s) (kept {n_surviving}/{len(raw_images)})"
                        )
                        raw_images = raw_images[:n_surviving]

            # ── Packed-MROPE override computation (past_k_steps path) ──
            packed_override = None
            if compute_packed:
                try:
                    packed_override = self._compute_packed_mrope_override(
                        full_messages=full_prompts[idx],
                        full_images=full_images_list[idx],
                    )
                except Exception as e:
                    logger.warning(
                        f"packed-mrope override computation failed (idx={idx}): "
                        f"{e}. Falling back to vLLM default MROPE for this request."
                    )
                    packed_override = None
            batch_packed_overrides.append(packed_override)

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_position_ids.append(position_ids)
            batch_raw_prompt_ids.append(raw_prompt_ids)
            if not for_generation:
                # Pre-computed tensors for training-side log-prob computation
                # (avoids re-processing which can produce mismatched grids).
                # Whitelist only visual-input tensors. Qwen3-VL processors may
                # emit extra tensor fields (e.g. position_ids for mrope) whose
                # non-batch dims vary per sample — those break torch.cat(dim=0)
                # in dp_actor._forward_micro_batch.
                _MM_KEYS = (
                    "pixel_values",
                    "image_grid_thw",
                    "pixel_values_videos",
                    "video_grid_thw",
                    "second_per_grid_ts",
                )
                multi_modal_inputs = {
                    k: model_inputs[k]
                    for k in _MM_KEYS
                    if k in model_inputs and isinstance(model_inputs[k], torch.Tensor)
                }
                batch_multi_modal_data.append(multi_modal_inputs)
            # Raw PIL images stored separately for vLLM generation (vLLM
            # does its own image processing internally).
            batch_raw_images.append(raw_images)

        # Build a 1-D numpy object array for raw_images so it gets properly
        # sharded by DataProto.chunk().  We must NOT use np.array(list, dtype=object)
        # because when inner lists have equal lengths numpy creates a 2-D array
        # which wraps individual PIL images in numpy scalars.
        raw_images_arr = np.empty(len(batch_raw_images), dtype=object)
        for i, imgs in enumerate(batch_raw_images):
            raw_images_arr[i] = imgs

        result = {
            "input_ids": torch.stack(batch_input_ids, dim=0),
            "attention_mask": torch.stack(batch_attention_mask, dim=0),
            "position_ids": torch.stack(batch_position_ids, dim=0),
            "raw_prompt_ids": np.array(batch_raw_prompt_ids, dtype=object),
            "raw_images": raw_images_arr,
        }
        if not for_generation:
            result["multi_modal_data"] = np.array(batch_multi_modal_data, dtype=object)
        if compute_packed:
            override_arr = np.empty(len(batch_packed_overrides), dtype=object)
            for i, ov in enumerate(batch_packed_overrides):
                override_arr[i] = ov
            result["packed_mrope_overrides"] = override_arr
        return result

    def _compute_packed_mrope_override(
        self,
        full_messages: list[dict],
        full_images: list[Image.Image],
    ) -> tuple[list[int], torch.Tensor, int] | None:
        """Compute packed MROPE positions for the K-window prefix.

        Tokenizes the full trajectory, runs ``get_rope_index`` to get full
        MROPE positions, computes per-token turn_id, and projects onto the
        K-window using ``filter_kwindow``. Returns
        ``(kept_token_ids, positions, delta)`` where ``kept_token_ids`` is the
        processor-expanded token sequence vLLM will see at prefill (used as
        the SHA-1 hash key on the vLLM side) and ``positions`` has shape
        (3, L_kwin) with original (gappy) values.
        """
        from ._packed_turn_metadata import (
            assign_turn_metadata,
            filter_kwindow,
        )

        # Convert <image> placeholders in full messages
        hf_messages = []
        for msg in full_messages:
            content = msg["content"]
            if isinstance(content, str) and "<image>" in content:
                content_list = []
                for i, part in enumerate(content.split("<image>")):
                    if i != 0:
                        content_list.append({"type": "image"})
                    if part:
                        content_list.append({"type": "text", "text": part})
                hf_messages.append({"role": msg["role"], "content": content_list})
            else:
                hf_messages.append(msg)

        text_prompt = self.processor.apply_chat_template(
            hf_messages, add_generation_prompt=True, tokenize=False
        )

        from ...utils.dataset import process_image
        processed = []
        for img in full_images:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            processed.append(process_image(img, self.min_pixels, self.max_pixels))
        if not processed:
            processed = None

        full_inputs = self.processor(
            processed,
            [text_prompt],
            add_special_tokens=False,
            return_tensors="pt",
        )
        full_input_ids = full_inputs["input_ids"][0]
        full_attn = full_inputs["attention_mask"][0]

        if "Qwen3VLProcessor" in self.processor.__class__.__name__:
            from ...models.transformers.qwen3_vl import get_rope_index
        else:
            from ...models.transformers.qwen2_vl import get_rope_index
        full_mrope = get_rope_index(
            self.processor,
            input_ids=full_input_ids,
            image_grid_thw=full_inputs.get("image_grid_thw", None),
            video_grid_thw=None,
            second_per_grid_ts=None,
            attention_mask=full_attn,
        )  # (3, L_full)

        im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        turn_id, _kind = assign_turn_metadata(
            full_input_ids.tolist(), full_messages, im_start_id, im_end_id
        )

        kept_ids, kept_pos, delta = filter_kwindow(
            full_input_ids,
            full_mrope,
            turn_id,
            past_k_steps=self.past_k_steps,
        )

        return kept_ids.tolist(), kept_pos, int(delta)

    def _build_final_batch(
        self,
        trajectories: list[Trajectory],
        rewards: list[float],
        ground_truths: list[str],
        n_trajectories: int,
    ) -> DataProto:
        """Construct the DataProto that the rest of the GRPO pipeline expects."""
        if self.context_mode == "multi_turn":
            return self._build_final_batch_multiturn(
                trajectories, rewards, ground_truths, n_trajectories
            )
        return self._build_final_batch_single_turn(
            trajectories, rewards, ground_truths, n_trajectories
        )

    def _build_final_batch_single_turn(
        self,
        trajectories: list[Trajectory],
        rewards: list[float],
        ground_truths: list[str],
        n_trajectories: int,
    ) -> DataProto:
        """Single-turn batch: prompt = full context, response = last turn only."""
        prompt_texts = []
        all_images = []
        response_texts = []

        for traj in trajectories:
            prompt_texts.append(traj.last_prompt or [{"role": "user", "content": ""}])
            all_images.append(traj.last_images or [])
            response_texts.append(
                traj.step_responses[-1] if traj.step_responses else ""
            )

        # ── tokenize prompts (with images) ──
        tokenized = self._tokenize_prompts(prompt_texts, all_images)
        prompt_ids = tokenized["input_ids"]  # (bs, prompt_len)
        prompt_mask = tokenized["attention_mask"]  # (bs, prompt_len)
        prompt_pos = tokenized["position_ids"]  # (bs, 4, prompt_len)

        # ── tokenize responses ──
        max_resp_len = self.max_response_length
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        batch_resp_ids = []
        batch_resp_mask = []

        for resp_text in response_texts:
            ids = self.tokenizer.encode(resp_text, add_special_tokens=False)
            ids.append(eos_id)
            if len(ids) > max_resp_len:
                ids = ids[:max_resp_len]

            rlen = len(ids)
            padded = ids + [pad_id] * (max_resp_len - rlen)
            mask = [1] * rlen + [0] * (max_resp_len - rlen)
            batch_resp_ids.append(padded)
            batch_resp_mask.append(mask)

        response_ids_t = torch.tensor(batch_resp_ids, dtype=prompt_ids.dtype)
        response_mask_t = torch.tensor(batch_resp_mask, dtype=prompt_mask.dtype)

        # ── concatenate prompt + response ──
        bs = prompt_ids.shape[0]
        full_ids = torch.cat([prompt_ids, response_ids_t], dim=-1)
        full_mask = torch.cat([prompt_mask, response_mask_t], dim=-1)

        # Extend position_ids for the response tokens
        resp_len = response_ids_t.shape[1]
        delta = torch.arange(1, resp_len + 1, device=prompt_pos.device)
        if prompt_pos.ndim == 3:  # mrope: (bs, 4, prompt_len)
            delta = delta.view(1, 1, -1).expand(bs, prompt_pos.shape[1], -1)
        else:
            delta = delta.view(1, -1).expand(bs, -1)
        resp_pos = prompt_pos[..., -1:] + delta
        full_pos = torch.cat([prompt_pos, resp_pos], dim=-1)

        # ── build TensorDict ──
        td = TensorDict(
            {
                "prompts": prompt_ids,
                "responses": response_ids_t,
                "input_ids": full_ids,
                "attention_mask": full_mask,
                "response_mask": response_mask_t,
                "position_ids": full_pos,
            },
            batch_size=bs,
        )

        # ── UIDs: same uid for all n trajectories of the same dataset item ──
        # (Required for GRPO group normalization in compute_grpo_outcome_advantage.)
        n_items = bs // n_trajectories
        uids = []
        for _ in range(n_items):
            uid = str(uuid.uuid4())
            uids.extend([uid] * n_trajectories)

        non_tensor = {
            "uid": np.array(uids, dtype=object),
            "ground_truth": np.array(ground_truths, dtype=object),
            "multi_modal_data": tokenized["multi_modal_data"],
        }

        # ── place trajectory reward at last response token ──
        token_level_scores = torch.zeros_like(
            response_ids_t, dtype=torch.float32
        )
        for i, reward in enumerate(rewards):
            rlen = int(response_mask_t[i].sum().item())
            if rlen > 0:
                token_level_scores[i, rlen - 1] = reward
        td["token_level_scores"] = token_level_scores

        meta_info = {
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "video_fps": 2.0,
        }

        return DataProto(
            batch=td, non_tensor_batch=non_tensor, meta_info=meta_info
        )

    def _find_assistant_content_ranges(
        self, token_ids: list[int]
    ) -> list[tuple[int, int]]:
        """Find (start, end) ranges of assistant content tokens in a tokenized conversation.

        Returns ranges where start is the first content token (after the
        ``<|im_start|>assistant\\n`` header) and end is inclusive of ``<|im_end|>``.
        """
        im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        assistant_role_ids = self.tokenizer.encode(
            "assistant\n", add_special_tokens=False
        )
        header_len = 1 + len(assistant_role_ids)  # <|im_start|> + "assistant\n"

        ranges = []
        i = 0
        while i < len(token_ids):
            if token_ids[i] == im_start_id:
                h_end = i + header_len
                if (
                    h_end <= len(token_ids)
                    and token_ids[i + 1 : h_end] == assistant_role_ids
                ):
                    # Scan to <|im_end|>
                    j = h_end
                    while j < len(token_ids) and token_ids[j] != im_end_id:
                        j += 1
                    end = j if j < len(token_ids) else j - 1
                    if h_end <= end:
                        ranges.append((h_end, end))
                    i = j + 1
                    continue
            i += 1
        return ranges

    def _build_final_batch_multiturn(
        self,
        trajectories: list[Trajectory],
        rewards: list[float],
        ground_truths: list[str],
        n_trajectories: int,
    ) -> DataProto:
        """Multi-turn batch: train on ALL assistant responses.

        Tokenizes the complete conversation (all user/assistant turns) as a
        single sequence.  ``response_mask`` marks all assistant content tokens
        so the policy gradient loss covers every model response, not just the
        final one.

        The prompt/response split is placed at the first assistant content
        boundary (after ``<|im_start|>assistant\\n``).  Everything before that
        (system + first user turn + assistant header) is the prompt;
        everything from the first assistant content onward is the response.
        ``response_mask`` zeros out user turns and assistant headers within
        the response portion so only assistant content gets gradient.
        """
        # ── 1. Build complete conversations (all turns + final response) ──
        full_messages = []
        all_images = []
        for traj in trajectories:
            msgs = list(traj.last_prompt or [{"role": "user", "content": ""}])
            if traj.step_responses:
                msgs.append(
                    {"role": "assistant", "content": traj.step_responses[-1]}
                )
            full_messages.append(msgs)
            all_images.append(traj.last_images or [])

        # ── 2. Tokenize full conversations ──
        # Pad/truncate to the same total budget as single-turn
        # (max_prompt_length + max_response_length) so the response tensor
        # size stays within the same memory envelope.
        max_total = self.max_prompt_length + self.max_response_length
        orig_max = self.max_prompt_length
        self.max_prompt_length = max_total
        try:
            tokenized = self._tokenize_prompts(
                full_messages, all_images, add_generation_prompt=False
            )
        finally:
            self.max_prompt_length = orig_max

        full_ids = tokenized["input_ids"]  # (bs, max_total) left-padded
        full_mask = tokenized["attention_mask"]  # (bs, max_total)
        full_pos = tokenized["position_ids"]  # (bs, 4, max_total)
        bs, seq_len = full_ids.shape

        # ── 3. Find assistant content ranges and split points ──
        # For each sample, find all assistant content token ranges and the
        # position of the first assistant content token (= prompt/response
        # boundary).
        response_mask_full = torch.zeros(bs, seq_len, dtype=full_ids.dtype)
        # Per-trajectory teacher-token mask in *full conversation* coords;
        # sliced to response coords below. 1 = injected by teacher VLM (Pope-
        # Dagger forced step), 0 = sampled by student. Downstream the policy
        # loss zeroes out gradient on positions where this is 1.
        teacher_token_mask_full = torch.zeros(bs, seq_len, dtype=torch.float32)
        # Per-trajectory SFT mask in *full conversation* coords; sliced below.
        # 1 = fresh expert (oracle) step → imitated by the cross-entropy/SFT loss.
        # Strict subset of teacher_token_mask (replayed-prefix tokens are teacher-
        # masked but NOT SFT'd). Zero everywhere for on-policy / non-guided runs.
        sft_token_mask_full = torch.zeros(bs, seq_len, dtype=torch.float32)
        # Per-trajectory per-step (start, end) token spans, in *response* coords
        # (after split). Populated below alongside the assistant-content range
        # iteration so the branching advantage estimator can broadcast a single
        # per-step scalar over each span without re-scanning the tokens.
        step_token_spans: list[list[tuple[int, int]]] = [[] for _ in range(bs)]
        split_points = []  # per-sample split in the padded sequence

        # If rollout-time logprobs were captured (parity smoke), copy them
        # into a (bs, seq_len) tensor at the assistant content positions in
        # step order. NaN at positions without a captured value.
        rollout_lp_full = torch.full(
            (bs, seq_len), float("nan"), dtype=torch.float32
        )
        rollout_tok_full = torch.full(
            (bs, seq_len), -1, dtype=torch.int64
        )
        any_lp = any(
            len(t.step_response_log_probs) > 0
            and any(len(lp) > 0 for lp in t.step_response_log_probs)
            for t in trajectories
        )

        for b in range(bs):
            ids = full_ids[b].tolist()
            ranges = self._find_assistant_content_ranges(ids)
            for start, end in ranges:
                response_mask_full[b, start : end + 1] = 1
            # Split at first assistant content start
            split_points.append(ranges[0][0] if ranges else seq_len)
            # Copy per-step rollout logprobs into the packed positions.
            if b < len(trajectories):
                t = trajectories[b]
                forced_flags = t.step_teacher_forced or []
                expert_flags = t.step_is_expert or []
                for step_idx, (start, end) in enumerate(ranges):
                    # Stash the per-step span in full-coords for later
                    # response-coord conversion (after split).
                    step_token_spans[b].append((start, end))
                    if step_idx < len(forced_flags) and forced_flags[step_idx]:
                        teacher_token_mask_full[b, start : end + 1] = 1.0
                    if step_idx < len(expert_flags) and expert_flags[step_idx]:
                        sft_token_mask_full[b, start : end + 1] = 1.0
                    if any_lp and step_idx < len(t.step_response_log_probs):
                        span_len = end - start + 1
                        lp = t.step_response_log_probs[step_idx][:span_len]
                        if lp:
                            rollout_lp_full[b, start : start + len(lp)] = torch.tensor(
                                lp, dtype=torch.float32
                            )
                        # Mirror the rollout-generated token IDs at the same
                        # positions for parity diagnosis (BPE-boundary check).
                        if step_idx < len(t.step_response_token_ids):
                            rt = t.step_response_token_ids[step_idx][:span_len]
                            if rt:
                                rollout_tok_full[b, start : start + len(rt)] = torch.tensor(
                                    rt, dtype=torch.int64
                                )

        # ── 4. Split at first assistant content boundary ──
        # Use the earliest split across the batch so all samples align.
        # Samples with later splits will have some non-assistant tokens at the
        # start of their response portion, but response_mask=0 handles that.
        split = min(split_points)

        prompt_ids = full_ids[:, :split]
        response_ids = full_ids[:, split:]
        response_mask_t = response_mask_full[:, split:]
        teacher_token_mask_t = teacher_token_mask_full[:, split:].contiguous()
        sft_token_mask_t = sft_token_mask_full[:, split:].contiguous()
        # Convert step spans from full-coords to response-coords. Spans whose
        # `start < split` (extremely rare — would mean an assistant turn
        # straddled the split) are clipped at 0; downstream consumers should
        # treat negative-end spans as empty.
        step_token_spans_resp: list[list[tuple[int, int]]] = []
        for spans in step_token_spans:
            adj: list[tuple[int, int]] = []
            for s, e in spans:
                ns = max(0, s - split)
                ne = e - split
                if ne >= ns:
                    adj.append((ns, ne))
            step_token_spans_resp.append(adj)

        # ── 5. Build per-token turn_id (for past-K K-window BlockMask) ──
        # Always emit when context_mode is multi_turn; downstream (actor
        # forward) uses it only when past_k_steps is enabled. The cost is
        # one extra (bs, seq_len) int64 tensor.
        # turn_id sentinels: -2 = left-padding (not visible to attention),
        # -1 = boundary/system (always-visible anchor in K-window mask).
        turn_id_full = torch.full((bs, seq_len), -2, dtype=torch.int64)
        if self.context_mode == "multi_turn":
            from ._packed_turn_metadata import assign_turn_metadata
            im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
            im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            for b, msgs in enumerate(full_messages):
                ids_b = full_ids[b].tolist()
                # Skip left-padding to find the unpadded prefix.
                pad_id = self.tokenizer.pad_token_id
                first_real = next(
                    (i for i, t in enumerate(ids_b) if t != pad_id), len(ids_b)
                )
                tids, _kind = assign_turn_metadata(
                    ids_b[first_real:], msgs, im_start_id, im_end_id
                )
                turn_id_full[b, first_real : first_real + tids.shape[0]] = tids

        # ── 6. Build TensorDict ──
        td_dict = {
            "prompts": prompt_ids,
            "responses": response_ids,
            "input_ids": full_ids,
            "attention_mask": full_mask,
            "response_mask": response_mask_t,
            "position_ids": full_pos,
            "turn_id": turn_id_full,
        }
        # Always emit teacher_token_mask so downstream consumers can rely on
        # the key existing. For on-policy / non-guided runs every entry is 0.
        td_dict["teacher_token_mask"] = teacher_token_mask_t
        # Always emit sft_token_mask too (all-zero for non-guided runs) so the
        # actor can rely on the key existing.
        td_dict["sft_token_mask"] = sft_token_mask_t
        if any_lp:
            # Slice to the response portion to align with old_log_probs shape.
            td_dict["rollout_log_probs"] = rollout_lp_full[:, split:].contiguous()
            td_dict["rollout_token_ids"] = rollout_tok_full[:, split:].contiguous()
        td = TensorDict(td_dict, batch_size=bs)

        # ── UIDs ──
        n_items = bs // n_trajectories
        uids = []
        for _ in range(n_items):
            uid = str(uuid.uuid4())
            uids.extend([uid] * n_trajectories)

        non_tensor = {
            "uid": np.array(uids, dtype=object),
            "ground_truth": np.array(ground_truths, dtype=object),
            "multi_modal_data": tokenized["multi_modal_data"],
            # Pope-Dagger V4 metadata. Always emitted; for non-guided runs
            # `dagger_iter_index` is all zero and `branch_chain` is empty, so
            # the branching advantage estimator collapses to standard GRPO.
            "step_token_spans": np.array(step_token_spans_resp, dtype=object),
            "dagger_iter_index": np.array(
                [t.dagger_iter_index for t in trajectories], dtype=np.int64
            ),
            "chain_id": np.array(
                [t.chain_id for t in trajectories], dtype=np.int64
            ),
            "branch_chain": np.array(
                [list(t.branch_chain) for t in trajectories], dtype=object
            ),
        }

        # ── Place trajectory reward at last assistant content token ──
        token_level_scores = torch.zeros_like(
            response_ids, dtype=torch.float32
        )
        for i, reward in enumerate(rewards):
            ones = (response_mask_t[i] == 1).nonzero(as_tuple=True)[0]
            if len(ones) > 0:
                token_level_scores[i, ones[-1].item()] = reward
        td["token_level_scores"] = token_level_scores

        meta_info = {
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "video_fps": 2.0,
        }

        # Human-readable rollout step-view (gated by POPE_DAGGER_VIZ=1):
        # per group/trajectory/step — abbreviated <think>, action, provenance.
        try:
            from verl.utils.pope_dagger_viz import log_rollout_stepview
            log_rollout_stepview(trajectories)
        except Exception:
            pass

        return DataProto(
            batch=td, non_tensor_batch=non_tensor, meta_info=meta_info
        )


# ---------------------------------------------------------------------------
# ObjectNav environment adapter (used inside SimulatorPool, not on driver)
# ---------------------------------------------------------------------------


class ObjectNavEnvAdapter:
    """Adapts ObjectNavEnvironment to the EnvInterface protocol.

    Contains all spatial-reasoning-specific logic: prompt building via
    ``build_annotate_style_context_from_history``, action parsing via
    ``ActionProposer._parse_action_response``, and reward computation.

    Instances live inside SimulatorPool Ray actors, NOT on the driver.
    """

    def __init__(
        self,
        env,
        state_history,
        system_prompt: str,
        action_proposer,
        coordinate_normalization_scale: float = 1.0,
        max_observations: int = 20,
        context_mode: str = "multi_turn",
        past_k_steps: "int | None" = None,
        reward_mode: str = "continuous",
    ):
        self.env = env
        self.state_history = state_history
        self.system_prompt = system_prompt
        self.action_proposer = action_proposer
        self.coordinate_normalization_scale = coordinate_normalization_scale
        self.max_observations = max_observations
        self.context_mode = context_mode
        self.past_k_steps = past_k_steps
        if reward_mode not in ("continuous", "bimodal", "bimodal_noprogress", "success"):
            raise ValueError(
                "reward_mode must be 'continuous', 'bimodal', 'bimodal_noprogress', "
                f"or 'success', got {reward_mode!r}"
            )
        self.reward_mode = reward_mode

        # Build the per-step instruction matching the SFT annotation format.
        # Uses the same tags as the action_proposer (explore, answer, summary).
        from interactive_reasoning.objectnavtask.agent.instructions import (
            build_annotation_direct_action_instructions,
        )

        self._step_instructions = build_annotation_direct_action_instructions(
            think_tag="think",
            explore_tag=action_proposer.explore_tag,
            answer_tag=action_proposer.answer_tag,
            summary_tag="summary",
        )

        # Track reward components
        self.initial_distance: float | None = None
        self.final_distance: float | None = None
        self.previous_distance: float | None = None
        self.progress_sum: float = 0.0
        self.success: bool = False
        self.answer_issued: bool = False
        self.num_steps: int = 0
        self.format_scores: list[float] = []
        self.validity_scores: list[float] = []

    def reset(self) -> dict[str, Any]:
        self.env.reset()
        initial_state = self.env.get_state()
        from interactive_reasoning.environment import StateActionHistory

        self.state_history = StateActionHistory(
            root_state=deepcopy(initial_state),
            action_state_pairs=[],
        )
        self.initial_distance = initial_state.shortest_path_distance_to_target
        self.final_distance = self.initial_distance
        self.previous_distance = self.initial_distance
        self.progress_sum = 0.0
        self.success = False
        self.answer_issued = False
        self.num_steps = 0
        self.format_scores = []
        self.validity_scores = []
        # Clear any expert trajectory cached for a previous episode on this
        # (possibly reused) slot — it was synthesized from a different episode's
        # pose/navmesh and must not leak into this one. Rebuilt lazily at the
        # first forced step (the branch) of this episode.
        self.expert_trajectory_cache = None
        self.expert_cache_base_num_steps = 0
        return {"observation": initial_state.observation}

    def build_prompt(self) -> tuple[list[dict], list[Image.Image]]:
        """Build prompt for the current step. Dispatches based on context_mode."""
        if self.context_mode == "multi_turn":
            return self._build_prompt_multiturn()
        return self._build_prompt_single_turn()

    def _build_prompt_single_turn(self) -> tuple[list[dict], list[Image.Image]]:
        """Build prompt matching the shared SFT-style compatibility format."""
        from common.prompting.context_builders import (
            build_sft_context_from_history,
            render_sft_flat_prompt,
        )

        model_context = build_sft_context_from_history(
            self.state_history,
            coordinate_normalization_scale=self.coordinate_normalization_scale,
            include_error_feedback=True,
            max_observations=self.max_observations,
        )

        # 4) Step instruction (format reminder for the model)
        max_actions = self.env.configuration.max_actions
        if self.num_steps + 1 >= max_actions:
            instruction = self._step_instructions["forced"]
        else:
            instruction = self._step_instructions["standard"]
        return render_sft_flat_prompt(
            system_prompt=self.system_prompt,
            flat_context=model_context,
            current_step_suffix=instruction,
        )

    def build_prompt_with_full(self) -> tuple[list[dict], list[Image.Image], list[dict] | None, list[Image.Image] | None]:
        """Build both K-window-filtered and full-trajectory prompts.

        When ``past_k_steps`` is None, returns ``(msgs, imgs, None, None)``.
        When set, returns ``(filtered_msgs, filtered_imgs, full_msgs, full_imgs)``.

        The full pair is what training and packed-MROPE position computation
        need; the filtered pair is what vLLM actually receives at rollout.
        """
        msgs, imgs = self.build_prompt()
        if self.past_k_steps is None or self.context_mode != "multi_turn":
            return msgs, imgs, None, None
        full_msgs, full_imgs = self._build_prompt_multiturn(_kwindow=False)
        return msgs, imgs, full_msgs, full_imgs

    def _build_prompt_multiturn(self, _kwindow: bool = True) -> tuple[list[dict], list[Image.Image]]:
        """Build prompt matching the SFT / reannotate multi-turn output format.

        Delegates to the canonical builder in ``src/common/prompting/context_builders.py``
        so the format is identical to ``_process_trajectory_multiturn`` in
        ``src/post_annotation/sft_data.py`` (the format the model was trained on).

        Format:
            system: system_prompt
            user: "Your task is to find the **X** (desc).\\n\\nStep 0.\\n<image>"
            assistant: "<think>...</think>\\n<explore>...</explore>"
            user: "Step 1.\\n<image>"
            assistant: ...
            user: "Step N.\\n<image>"

        No per-turn instruction suffix and no ``<summary>`` tag — matches the
        SFT training data exactly.

        Error turns (no new observation) omit the image and step label:
            assistant: "<think>...</think>\\n<explore>...</explore>"
            user: "Action execution failed: ..."
        """
        from common.prompting.context_builders import (
            build_multiturn_context,
            steps_from_state_history,
        )

        steps = steps_from_state_history(
            self.state_history,
            coordinate_normalization_scale=self.coordinate_normalization_scale,
        )
        if _kwindow:
            steps = filter_steps_to_kwindow(steps, self.past_k_steps)
        return build_multiturn_context(
            system_prompt=self.system_prompt,
            steps=steps,
            simple_step_text=True,
        )

    @staticmethod
    def _check_format(response: str) -> float:
        """Check if response matches the SFT multi-turn format: <think>...</think> <action>."""
        pattern = re.compile(
            r"<think>.*?</think>\s*<(?:explore|answer).*?(?:/>|</(?:explore|answer)>)",
            re.DOTALL,
        )
        return 1.0 if pattern.search(response) else 0.0

    @staticmethod
    def _check_validity(response: str) -> float:
        """Check if response contains a parseable action with valid coordinates."""
        # <answer>(x,y)</answer>
        m = re.search(r"<answer>\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?\s*</answer>", response)
        if m:
            x, y = int(m.group(1)), int(m.group(2))
            return 1.0 if 0 <= x <= 1000 and 0 <= y <= 1000 else 0.0
        # <explore>ground:(x,y)</explore>
        m = re.search(r"<explore>\s*ground:\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?\s*</explore>", response)
        if m:
            x, y = int(m.group(1)), int(m.group(2))
            return 1.0 if 0 <= x <= 1000 and 0 <= y <= 1000 else 0.0
        # <explore>direction:ANGLE</explore>
        m = re.search(r"<explore>\s*direction:\s*(-?\d+)\s*</explore>", response)
        if m:
            angle = int(m.group(1))
            return 1.0 if -180 <= angle <= 180 else 0.0
        return 0.0

    def step(self, action_text: str) -> tuple[float, bool, dict[str, Any]]:
        self.num_steps += 1

        # Capture pre-step distance for V5 random_regression selector.
        # self.previous_distance is what the agent saw before deciding this
        # step (initial_distance at step 0; post-prev-step distance otherwise).
        dist_before = self.previous_distance

        # Score format and validity for this step
        self.format_scores.append(self._check_format(action_text))
        self.validity_scores.append(self._check_validity(action_text))

        # Parse summary
        summary = None
        summary_match = re.search(
            r"<summary>(.*?)</summary>", action_text, re.DOTALL
        )
        if summary_match:
            summary = summary_match.group(1).strip()

        # Parse action
        action = self.action_proposer._parse_action_response(
            response=action_text, memory=summary
        )
        action.response = action_text

        # Capture the observation the agent was looking at when it chose this action
        prev_observation = self.state_history.get_last_state().observation

        # Step the environment
        try:
            new_state = self.env.step(action)
        except Exception as e:
            logger.warning(f"Env step failed: {e}")
            error_state = deepcopy(self.state_history.get_last_state())
            error_state.observation = None
            error_state.user_response = f"Action execution failed: {e}"
            self.state_history.append(action, error_state)
            return 0.0, False, {"action_type": "error", "dist_before": dist_before}

        # Handle error states (no new observation)
        if new_state.observation is None:
            error_state = deepcopy(self.state_history.get_last_state())
            error_state.observation = None
            error_state.user_response = new_state.user_response
            self.state_history.append(action, error_state)
            return 0.0, False, {"action_type": "error", "dist_before": dist_before}

        self.state_history.append(action, deepcopy(new_state))

        # Track distance + accumulate forward-only progress (ViGoRL style:
        # cumulative max(0, prev - curr) across turns, raw absolute units).
        if new_state.shortest_path_distance_to_target is not None:
            curr_dist = new_state.shortest_path_distance_to_target
            if self.previous_distance is not None:
                self.progress_sum += max(0.0, self.previous_distance - curr_dist)
            self.previous_distance = curr_dist
            self.final_distance = curr_dist

        terminated = new_state.is_terminal
        if terminated and new_state.reward > 0:
            self.success = True

        # Classify action type
        from interactive_reasoning.objectnavtask.environment.actions import (
            ObjectNavAnswerAction,
            ObjectNavDirectionalAction,
            ObjectNavGroundNavigationAction,
            ObjectNavInvalidAction,
            ObjectNavStopAction,
        )

        if isinstance(action, ObjectNavAnswerAction):
            action_type = "answer"
            self.answer_issued = True
            # ── DEBUG: save answer image with coordinate and bounding box ──
            # self._debug_save_answer_image(action, new_state, prev_observation)
        elif isinstance(action, ObjectNavGroundNavigationAction):
            action_type = "explore_ground"
            # self._debug_save_explore_image(action, new_state, action_type, prev_observation)
        elif isinstance(action, ObjectNavDirectionalAction):
            action_type = "explore_direction"
            # self._debug_save_explore_image(action, new_state, action_type, prev_observation)
        elif isinstance(action, ObjectNavStopAction):
            action_type = "stop"
            # self._debug_save_explore_image(action, new_state, action_type, prev_observation)
        elif isinstance(action, ObjectNavInvalidAction):
            action_type = "invalid"
            # self._debug_save_explore_image(action, new_state, action_type, prev_observation)
        else:
            action_type = "unknown"

        return new_state.reward, terminated, {
            "action_type": action_type,
            "dist_before": dist_before,
            "dist_after": new_state.shortest_path_distance_to_target,
        }

    def _debug_save_answer_image(self, action, new_state, prev_observation=None) -> None:
        """Save the observation image with the answer coordinate drawn on it.

        Uses *prev_observation* (the image the model saw when choosing this
        action) so that coordinates align with the correct frame.
        """
        try:
            from PIL import ImageDraw

            debug_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "..", "..", "..", "..", "debug_answer_images",
            )
            debug_dir = os.path.normpath(debug_dir)
            os.makedirs(debug_dir, exist_ok=True)

            # Use the observation the model was looking at, not the post-action one
            obs = prev_observation if prev_observation is not None else new_state.observation
            if obs is None:
                return
            if isinstance(obs, np.ndarray):
                img = Image.fromarray(obs)
            elif isinstance(obs, Image.Image):
                img = obs.copy()
            else:
                return

            draw = ImageDraw.Draw(img)
            x, y = action.coordinates  # already scaled coordinates

            # Draw the predicted coordinate as a red crosshair
            r = 12
            draw.ellipse([x - r, y - r, x + r, y + r], outline="red", width=3)
            draw.line([x - r, y, x + r, y], fill="red", width=2)
            draw.line([x, y - r, x, y + r], fill="red", width=2)

            # Try to get and draw the bounding box
            bbox = None
            try:
                bbox = self.env._ai2thor.get_bounding_box_for_object(
                    self.env.target_object_id
                )
            except Exception:
                pass

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
                # Also draw relaxed bbox
                relax = self.env.configuration.bounding_box_relaxation
                draw.rectangle(
                    [x1 - relax, y1 - relax, x2 + relax, y2 + relax],
                    outline="lime", width=1,
                )

            # Add text label
            success = new_state.reward > 0
            label = (
                f"coord=({x},{y}) success={success} "
                f"step={self.num_steps} target={self.env.target_object_id}"
            )
            if bbox:
                label += f" bbox={bbox}"
            draw.text((10, 10), label, fill="yellow")

            fname = f"step{self.num_steps:03d}_{uuid.uuid4().hex[:6]}_{'HIT' if success else 'MISS'}.png"
            img.save(os.path.join(debug_dir, fname))
            logger.info(f"DEBUG: saved answer image to {os.path.join(debug_dir, fname)}")
        except Exception as e:
            logger.warning(f"DEBUG: failed to save answer image: {e}")

    def _debug_save_explore_image(self, action, new_state, action_type: str, prev_observation=None) -> None:
        """Save exploration-step observation with explore-specific overlays.

        Uses *prev_observation* (the image the model saw when choosing this
        action) so that coordinates align with the correct frame.
        """
        try:
            from PIL import ImageDraw

            debug_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "..", "..", "..", "..", "debug_answer_images",
            )
            debug_dir = os.path.normpath(debug_dir)
            os.makedirs(debug_dir, exist_ok=True)

            # Use the observation the model was looking at, not the post-action one
            obs = prev_observation if prev_observation is not None else new_state.observation
            if obs is None:
                return
            if isinstance(obs, np.ndarray):
                img = Image.fromarray(obs)
            elif isinstance(obs, Image.Image):
                img = obs.copy()
            else:
                return

            draw = ImageDraw.Draw(img)

            # Ground-point exploration has target image coordinates.
            if hasattr(action, "target_coordinates") and action.target_coordinates is not None:
                x, y = action.target_coordinates
                r = 12
                draw.ellipse([x - r, y - r, x + r, y + r], outline="cyan", width=3)
                draw.line([x - r, y, x + r, y], fill="cyan", width=2)
                draw.line([x, y - r, x, y + r], fill="cyan", width=2)

            # Label includes action metadata (e.g., directional turn angle).
            label = (
                f"action={action_type} step={self.num_steps} reward={new_state.reward:.3f} "
                f"target={self.env.target_object_id}"
            )
            if hasattr(action, "type"):
                label += f" dir={action.type}"
            if hasattr(action, "parameters") and action.parameters:
                label += f" params={action.parameters}"
            if hasattr(action, "target_coordinates") and action.target_coordinates is not None:
                label += f" coord={action.target_coordinates}"

            draw.text((10, 10), label, fill="yellow")

            fname = (
                f"EXPLORE_step{self.num_steps:03d}_{action_type}_{uuid.uuid4().hex[:6]}"
                f"_{'TERM' if new_state.is_terminal else 'RUN'}.png"
            )
            out_path = os.path.join(debug_dir, fname)
            img.save(out_path)
            logger.info(f"DEBUG: saved explore image to {out_path}")
        except Exception as e:
            logger.warning(f"DEBUG: failed to save explore image: {e}")

    def get_trajectory_reward(self) -> float:
        avg_fmt = sum(self.format_scores) / len(self.format_scores) if self.format_scores else 0.0
        avg_validity = sum(self.validity_scores) / len(self.validity_scores) if self.validity_scores else 0.0
        step_penalty = 0.005 * self.num_steps

        if self.reward_mode == "success":
            # Pure binary success reward: 1.0 on success, 0.0 otherwise.
            # No format weighting, no progress shaping, no step penalty.
            reward = 1.0 if self.success else 0.0
            progress = None
        elif self.reward_mode == "bimodal":
            #   0.875 * success + 0.125 * fmt + 0.25 * normalized_net_progress
            # Progress = fraction of the initial gap closed at trajectory end,
            # clamped to [0, 1]. Replaces ViGoRL's forward-only Σ max(0, Δdist)
            # which lets agents farm reward by zigzagging (close, back, close).
            # No step_penalty, no validity gate.
            # Force-answer: timing out without ever committing an <answer/> is
            # strictly worse than a wrong answer. Without this gate, a wandering
            # trajectory that closes half the gap earns 0.125 progress while a
            # confidently-wrong early answer earns 0.125 fmt — the "wander
            # forever" attractor pays out. Bimodal/success runs collapse via
            # avg_steps → max_depth; gating progress/fmt on answer_issued
            # removes that escape valve while staying RLVR-faithful.
            if not self.answer_issued:
                progress = 0.0
                reward = 0.0
            else:
                reward = 0.0
                if self.success:
                    reward += 0.875
                reward += 0.125 * avg_fmt
                if self.initial_distance and self.initial_distance > 1e-6:
                    progress = max(0.0, min(1.0, (self.initial_distance - self.final_distance) / self.initial_distance))
                else:
                    progress = 0.0
                reward += 0.25 * progress
        elif self.reward_mode == "bimodal_noprogress":
            #   0.875 * success + 0.125 * fmt        (answer-gated; NO progress term)
            # Identical to 'bimodal' minus the 0.25 * normalized_net_progress shaping.
            # Used by the POPE-DAgger runs (and its vanilla-GRPO comparison anchor) so
            # the reward is a clean success+format signal — progress shaping is dropped
            # because the expert-guided branches already manufacture progress, which
            # would otherwise double-credit the guided trajectories.
            if not self.answer_issued:
                progress = None
                reward = 0.0
            else:
                reward = 0.0
                if self.success:
                    reward += 0.875
                reward += 0.125 * avg_fmt
                progress = None
        else:
            # Continuous distance progress: graded credit even on failed trajectories.
            if self.initial_distance and self.initial_distance > 0.1:
                progress = (self.initial_distance - self.final_distance) / self.initial_distance
                progress = max(-0.5, min(1.0, progress))
            else:
                progress = 0.0
            success_bonus = 1.0 if self.success else 0.0
            validity_gate = 1.0 if avg_validity > 0.5 else 0.1
            reward = validity_gate * (0.5 * progress + success_bonus) - step_penalty

        logger.info(
            f"Trajectory reward [{self.reward_mode}]: success={self.success}, "
            f"answer_issued={self.answer_issued}, "
            f"initial_distance={self.initial_distance:.2f}, "
            f"final_distance={self.final_distance:.2f}, "
            f"progress={'n/a' if progress is None else f'{progress:+.3f}'}, "
            f"num_steps={self.num_steps}, "
            f"avg_format={avg_fmt:.2f}, avg_validity={avg_validity:.2f}, "
            f"reward={reward:.3f}"
        )
        return reward

    def get_ground_truth(self) -> str:
        avg_fmt = sum(self.format_scores) / len(self.format_scores) if self.format_scores else 0.0
        avg_val = sum(self.validity_scores) / len(self.validity_scores) if self.validity_scores else 0.0
        return json.dumps(
            {
                "trajectory_reward": self.get_trajectory_reward(),
                "success": self.success,
                "answer_issued": self.answer_issued,
                "num_steps": self.num_steps,
                "initial_distance": self.initial_distance,
                "final_distance": self.final_distance,
                "progress_sum": self.progress_sum,
                "avg_format": avg_fmt,
                "avg_validity": avg_val,
            }
        )

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# ObjectNav dataset wrapper (simplified — env creation moved to SimulatorPool)
# ---------------------------------------------------------------------------


class ObjectNavEnvFactory(EnvFactory):
    """Provides dataset items for SimulatorPool-based env creation.

    Only manages dataset iteration. Actual ObjectNavEnvironment and
    ObjectNavEnvAdapter creation happens inside SimulatorPool.acquire_env().
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self._dataset_len = len(dataset)
        self._indices = list(range(self._dataset_len))
        self._item_idx = 0

    def __len__(self) -> int:
        return self._dataset_len

    def get_next_item(self) -> dict:
        """Return the next dataset item (cycling through the dataset)."""
        if self._item_idx >= self._dataset_len:
            np.random.shuffle(self._indices)
            self._item_idx = 0
        data = self.dataset[self._indices[self._item_idx]]
        self._item_idx += 1
        return data
