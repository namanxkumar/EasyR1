"""Driver-side multi-turn environment rollout for GRPO training.

Runs multi-turn trajectories through AI2Thor environments by calling the
distributed ``actor_rollout_ref_wg.generate_sequences()`` at each step.
Environment operations (reset, step, build_prompt) are parallelized across
Ray-remote SimulatorPool actors.

Produces a standard ``DataProto`` batch compatible with the rest of the GRPO
pipeline (KL, advantage, actor update).
"""

from __future__ import annotations

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


def filter_steps_to_kwindow(steps, past_k_steps):
    """Filter a chronological list of MultiturnStep to the past-K observation window.

    Keeps step 0 (which carries the task description) and the contiguous tail
    starting at the K-th-from-last observation step. Error steps interspersed
    in the kept tail are kept; error steps in the dropped middle are dropped.

    When past_k_steps is None, <= 0, or >= total observation steps, returns
    the input unchanged.
    """
    if not past_k_steps or past_k_steps <= 0:
        return steps
    obs_positions = [i for i, s in enumerate(steps) if not s.is_error]
    if len(obs_positions) <= past_k_steps:
        return steps
    tail_start = obs_positions[-past_k_steps]
    if tail_start == 0:
        return steps
    return [steps[0]] + list(steps[tail_start:])

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
    terminated: bool = False
    num_steps: int = 0

    # Reward collected immediately upon termination (before slot release)
    reward: float | None = None
    ground_truth: str | None = None

    # Cached prompt/images from the last generation step (for final batch)
    last_prompt: list[dict] | None = None
    last_images: list[Any] = field(default_factory=list)


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
        """Pre-create AI2Thor controllers on all SimulatorPools in parallel.

        Called before rollout to ensure controllers are ready. Uses the first
        dataset item's scene as a dummy scene for initialization. All pools
        warm up concurrently since they're on different GPUs.
        """
        dummy_scene = self.env_factory.dataset[0]["scene_metadata"]
        logger.info(f"Warming up AI2Thor controllers across {len(self.simulator_pools)} pools...")
        t0 = time.time()
        futures = [pool.warmup_controllers.remote(dummy_scene) for pool in self.simulator_pools]
        counts = ray.get(futures)
        total = sum(counts)
        logger.info(
            f"All AI2Thor controllers warmed up: {total} controllers "
            f"across {len(self.simulator_pools)} pools in {time.time() - t0:.1f}s"
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

            # ── Tokenize into DataProto format ──
            t_gen = time.time()
            tokenized = self._tokenize_prompts(
                prompts,
                images_list,
                for_generation=True,
                full_prompts=full_prompts if use_full else None,
                full_images_list=full_images_list if use_full else None,
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
            t_gen_elapsed = time.time() - t_gen

            # Decode responses
            response_ids = gen_output.batch["responses"]
            response_mask = gen_output.batch["response_mask"]
            rollout_log_probs = gen_output.batch.get("rollout_log_probs")
            responses = []
            response_token_ids_list: list[list[int]] = []
            response_log_probs_list: list[list[float]] = []
            for i in range(len(valid)):
                length = int(response_mask[i].sum().item())
                text = self.tokenizer.decode(
                    response_ids[i][:length], skip_special_tokens=True
                )
                responses.append(text)
                response_token_ids_list.append(response_ids[i][:length].tolist())
                if rollout_log_probs is not None:
                    response_log_probs_list.append(
                        rollout_log_probs[i][:length].cpu().tolist()
                    )
                else:
                    response_log_probs_list.append([])

            if logger.isEnabledFor(logging.DEBUG):
                for i, (local_i, t) in enumerate(valid):
                    resp_len = int(response_mask[i].sum().item())
                    logger.debug(
                        f"  [step {global_step}][grp {t.group_id}/{t.n_idx}] "
                        f"RESPONSE ({resp_len} tokens):\n"
                        f'  [ASSISTANT] Text: "{responses[i]}"'
                    )

            # ── Step environments in parallel via Ray ──
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

            # Process step results
            n_terminated_this_step = 0
            n_failed_this_step = 0
            for i, (_, t) in enumerate(valid):
                t.step_responses.append(responses[i])
                t.step_response_token_ids.append(response_token_ids_list[i])
                t.step_response_log_probs.append(response_log_probs_list[i])
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
                    continue

                reward, terminated, _info = result
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

            logger.info(
                f"  step {global_step}: {len(valid)} active, "
                f"{n_terminated_this_step} done"
                + (f" ({n_failed_this_step} failed)" if n_failed_this_step else "")
                + f", gen={t_gen_elapsed:.1f}s, "
                f"total={time.time() - t_step:.1f}s"
                + (f", pending={len(pending_queue)}" if pending_queue else "")
                + f", RSS={_get_rss_mb():.0f}MB"
            )

            # ── Parity-mode context dump: visible turn IDs + prompt extents ──
            # For the first few traj only, so the log doesn't explode.
            if rollout_log_probs is not None:
                prompt_ids_b = gen_batch.batch.get("input_ids")
                attn_b = gen_batch.batch.get("attention_mask")
                for i, (_, t) in enumerate(valid[:2]):
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
        n_to_fill = min(len(newly_done), len(pending_queue))
        if n_to_fill > 0:
            refill_batch = [pending_queue.popleft() for _ in range(n_to_fill)]
            new_trajs = self._initialize_batch(refill_batch)
            all_trajectories.extend(new_trajs)
            active.extend(new_trajs)
            logger.info(
                f"  refill: released {len(newly_done)} slots, "
                f"started {n_to_fill} new trajectories, "
                f"{len(pending_queue)} still pending"
            )
        elif newly_done:
            logger.info(
                f"  released {len(newly_done)} slots (queue empty, no refill)"
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
                multi_modal_inputs = {k: v for k, v in model_inputs.items() if isinstance(v, torch.Tensor)}
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
            if any_lp and b < len(trajectories):
                t = trajectories[b]
                for step_idx, (start, end) in enumerate(ranges):
                    if step_idx >= len(t.step_response_log_probs):
                        break
                    span_len = end - start + 1
                    lp = t.step_response_log_probs[step_idx][:span_len]
                    if not lp:
                        continue
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
        if reward_mode not in ("continuous", "bimodal"):
            raise ValueError(f"reward_mode must be 'continuous' or 'bimodal', got {reward_mode!r}")
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
        self.success: bool = False
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
        self.success = False
        self.num_steps = 0
        self.format_scores = []
        self.validity_scores = []
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
            return 0.0, False, {"action_type": "error"}

        # Handle error states (no new observation)
        if new_state.observation is None:
            error_state = deepcopy(self.state_history.get_last_state())
            error_state.observation = None
            error_state.user_response = new_state.user_response
            self.state_history.append(action, error_state)
            return 0.0, False, {"action_type": "error"}

        self.state_history.append(action, deepcopy(new_state))

        # Track distance
        if new_state.shortest_path_distance_to_target is not None:
            self.final_distance = new_state.shortest_path_distance_to_target

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

        return new_state.reward, terminated, {"action_type": action_type}

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

        if self.reward_mode == "bimodal":
            # Original 7588989 baseline.
            reward = 0.0
            if self.success:
                reward += 1.0
            reward += 0.1 * avg_fmt
            reward += 0.15 * avg_validity
            reward -= step_penalty
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
                "num_steps": self.num_steps,
                "initial_distance": self.initial_distance,
                "final_distance": self.final_distance,
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
