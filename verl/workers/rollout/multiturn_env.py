"""Driver-side multi-turn environment rollout for GRPO training.

Runs multi-turn trajectories through AI2Thor environments by calling the
distributed ``actor_rollout_ref_wg.generate_sequences()`` at each step.
Environment operations (reset, step, build_prompt) are parallelized across
Ray-remote SimulatorPool actors.

Produces a standard ``DataProto`` batch compatible with the rest of the GRPO
pipeline (KL, advantage, actor update).
"""

from __future__ import annotations

import gc
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import psutil
import ray
import torch
from PIL import Image
from transformers import PreTrainedTokenizer, ProcessorMixin

from ...protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from .multiturn_tokenizer import TokenizerMixin

# Re-export for backward compatibility (external imports use this module path)
from .objectnav_adapter import ObjectNavEnvAdapter  # noqa: F401
# NOTE: ObjectNavEnvFactory re-export is at the bottom of this file to avoid
# circular import (objectnav_factory.py imports EnvFactory from this module).

logger = logging.getLogger(__name__)
# Ray workers don't inherit the driver's logging config, so set level from env var
_log_level = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
logger.setLevel(_log_level)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_handler)
    logger.propagate = False  # prevent duplicate output from root logger

_log_vllm_metrics_warned = False


def _log_memory(context: str) -> None:
    """Log driver-side CPU/RSS memory and per-GPU memory for debugging leaks."""
    proc = psutil.Process()
    rss_mb = proc.memory_info().rss / (1024 * 1024)
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            gpu_lines = []
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 3:
                    idx, used, total = parts
                    gpu_lines.append(f"GPU{idx}={used}/{total}MB")
            gpu_info = ", ".join(gpu_lines) if gpu_lines else "N/A"
        else:
            gpu_info = "N/A"
    except Exception:
        gpu_info = "N/A"
    logger.info(
        f"[MEMORY] {context}: driver RSS={rss_mb:.0f}MB | {gpu_info}"
    )


def _log_vllm_metrics(
    context: str, actor_rollout_ref_wg, *, prev_metrics: list[dict] | None = None
) -> list[dict]:
    """Query vLLM engine metrics from all rollout workers and log them.

    Returns the current per-rank metrics list so callers can pass it as
    ``prev_metrics`` on the next call to show deltas.
    """
    try:
        all_metrics = actor_rollout_ref_wg.execute_all_sync("get_vllm_metrics")
    except Exception:
        # Silently skip — the worker group may not expose get_vllm_metrics
        # (e.g. WorkerDict wrapping ActorHandles). Only log once.
        global _log_vllm_metrics_warned
        if not _log_vllm_metrics_warned:
            logger.debug("[vLLM-METRICS] get_vllm_metrics not available on worker group, skipping")
            _log_vllm_metrics_warned = True
        return prev_metrics or []

    if not all_metrics:
        return prev_metrics or []

    lines = [f"[vLLM-METRICS] {context}:"]
    for i, m in enumerate(all_metrics):
        if not isinstance(m, dict) or not m:
            continue
        rank = m.get("rank", i)
        alloc = m.get("gpu_allocated_mb", 0)
        reserved = m.get("gpu_reserved_mb", 0)
        kv_usage = m.get("vllm:kv_cache_usage_perc", None)
        prefix_hit = m.get("vllm:prefix_cache_hits", None)

        parts = [f"rank{rank}: alloc={alloc:.0f}MB, reserved={reserved:.0f}MB"]
        if kv_usage is not None:
            parts.append(f"kv_cache={kv_usage:.1f}%")
        if prefix_hit is not None:
            parts.append(f"prefix_cache_hits={prefix_hit}")

        # Show delta from previous if available
        if prev_metrics and i < len(prev_metrics):
            prev = prev_metrics[i]
            d_alloc = alloc - prev.get("gpu_allocated_mb", alloc)
            d_reserved = reserved - prev.get("gpu_reserved_mb", reserved)
            if abs(d_alloc) > 1 or abs(d_reserved) > 1:
                parts.append(f"delta(alloc={d_alloc:+.0f}MB, reserved={d_reserved:+.0f}MB)")
            if kv_usage is not None:
                prev_kv = prev.get("vllm:kv_cache_usage_perc")
                if prev_kv is not None:
                    parts.append(f"delta(kv={kv_usage - prev_kv:+.1f}%)")

        lines.append("  " + ", ".join(parts))

    logger.info("\n".join(lines))
    return all_metrics


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
class StepRecord:
    """Lightweight per-step data for credit assignment.

    Stores the prompt, full image path list, and response for a single step
    so that all steps can be included in the training batch.

    ``image_paths`` is the complete ordered list of images for this step's
    prompt (prior observations + current).  Images accumulate across steps,
    so step *i*'s paths are a superset of step *i-1*'s.  Only the genuinely
    new images are written to disk at each step — prior paths are reused
    from earlier records.

    ``tensor_cache_path`` (optional) points to a ``.pt`` file containing
    pre-tokenized tensors (input_ids, attention_mask, position_ids,
    multi_modal_inputs) saved during rollout.  When present,
    ``_build_final_batch`` loads these directly instead of re-tokenizing —
    avoiding redundant image processing and mrope computation.
    """

    prompt: list[dict]         # chat-format messages used for this step
    image_paths: list[str]     # ALL image paths for this step (prior + current)
    response: str              # model's decoded response text
    tensor_cache_path: str | None = None  # path to cached tokenized tensors
    old_log_probs: list[float] | None = None  # per-token log probs from vLLM generation


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

    # Per-step records for credit assignment (prompt, image_paths, response)
    step_records: list[StepRecord] = field(default_factory=list)
    terminated: bool = False
    num_steps: int = 0

    # Reward collected immediately upon termination (before slot release)
    reward: float | None = None
    ground_truth: str | None = None

    # Path to serialized step records on disk (set after harvest to free memory)
    _cache_path: str | None = None


# ---------------------------------------------------------------------------
# MultiturnEnvRollout — the core class (parallelized via Ray SimulatorPools)
# ---------------------------------------------------------------------------


class MultiturnEnvRollout(TokenizerMixin):
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
        image_cache_dir: str | None = None,
        trajectory_cache_dir: str | None = None,
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
        self.image_cache_dir = image_cache_dir
        self.trajectory_cache_dir = trajectory_cache_dir

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

        # Clean up caches from the previous iteration
        if self.image_cache_dir:
            from ...utils.image_cache import cleanup_cache_dir
            cleanup_cache_dir(self.image_cache_dir)
        if self.trajectory_cache_dir:
            import shutil
            if os.path.exists(self.trajectory_cache_dir):
                shutil.rmtree(self.trajectory_cache_dir)
            os.makedirs(self.trajectory_cache_dir, exist_ok=True)

        _log_memory("rollout START")

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
        _log_memory("after seeding initial trajectories")

        # Baseline vLLM metrics before first generation step
        _log_vllm_metrics("pre-rollout baseline", actor_rollout_ref_wg)

        # ── Continuous episode loop with dynamic slot reuse ──
        self._run_continuous_episode_loop(
            active_trajectories, all_trajectories,
            pending_queue, actor_rollout_ref_wg, config,
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

        _log_memory("rollout DONE (before destroy)")

        # ── Free GPU memory for training ──
        self.destroy_controllers()
        _log_memory("rollout DONE (after destroy)")

        # ── Build final DataProto batch ──
        result = self._build_final_batch(
            all_trajectories, all_rewards, all_ground_truths, n_trajectories
        )

        # Free all trajectory data now that it's been assembled into tensors
        del all_trajectories, all_rewards, all_ground_truths
        gc.collect()

        return result

    def _run_continuous_episode_loop(
        self,
        active: list[Trajectory],
        all_trajectories: list[Trajectory],
        pending_queue,
        actor_rollout_ref_wg,
        config,
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
        _mem_log_interval = int(os.environ.get("MEM_LOG_INTERVAL", "1"))
        _gc_interval = int(os.environ.get("GC_INTERVAL", "5"))
        _prev_vllm_metrics: list[dict] | None = None

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
            prompt_futures = [
                t.pool.build_prompt.remote(t.slot_id) for t in active
            ]
            prompt_results_raw = ray.get(prompt_futures)
            del prompt_futures

            # Filter out trajectories whose prompt build failed
            valid = []          # (local_idx, trajectory)
            prompts = []
            images_list = []
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

            # Temporarily hold prompt/images until after generation + env step,
            # when we create a StepRecord with the response.
            for local_i, t in valid:
                t._pending_prompt = prompts[local_i]
                t._pending_images = list(images_list[local_i])

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
            # Generate cache paths so tokenized tensors (input_ids,
            # attention_mask, position_ids, multi_modal_inputs) are saved
            # to disk per step.  _build_final_batch loads these directly,
            # avoiding a full re-tokenization of all ~1500 step-level samples.
            t_gen = time.time()
            _cache_paths: list[str] | None = None
            if self.trajectory_cache_dir:
                _cache_paths = []
                for _local_i, _t in valid:
                    _tdir = os.path.join(
                        self.trajectory_cache_dir, _t.episode_id
                    )
                    os.makedirs(_tdir, exist_ok=True)
                    _cache_paths.append(
                        os.path.join(_tdir, f"step_{_t.num_steps}_tokens.pt")
                    )
            tokenized = self._tokenize_prompts(
                prompts, images_list,
                for_generation=True,
                tensor_cache_paths=_cache_paths,
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
            gen_batch = DataProto.from_single_dict(
                tokenized, meta_info=meta_info
            )

            # ── Generate model responses via vLLM ──
            prompt_lens = [len(ids) for ids in tokenized["raw_prompt_ids"]]
            n_images_per_prompt = [len(imgs) for imgs in images_list]
            logger.info(
                f"  [step {global_step}] GEN: n_prompts={len(prompt_lens)}, "
                f"tokens(min={min(prompt_lens)}, max={max(prompt_lens)}, "
                f"total={sum(prompt_lens)}), "
                f"images(min={min(n_images_per_prompt)}, max={max(n_images_per_prompt)})"
            )

            gen_batch, pad_size = pad_dataproto_to_divisor(
                gen_batch, actor_rollout_ref_wg.world_size
            )
            gen_output = actor_rollout_ref_wg.generate_sequences(gen_batch)
            gen_output = unpad_dataproto(gen_output, pad_size)
            t_gen_elapsed = time.time() - t_gen

            # Log vLLM engine metrics (KV cache, GPU memory) after generation
            _prev_vllm_metrics = _log_vllm_metrics(
                f"step {global_step} post-gen", actor_rollout_ref_wg,
                prev_metrics=_prev_vllm_metrics,
            )

            # Decode responses
            response_ids = gen_output.batch["responses"]
            response_mask = gen_output.batch["response_mask"]
            # Extract per-token log probs collected during vLLM generation
            step_log_probs = gen_output.batch.get("old_log_probs", None)
            responses = []
            for i in range(len(valid)):
                length = int(response_mask[i].sum().item())
                text = self.tokenizer.decode(
                    response_ids[i][:length], skip_special_tokens=True
                )
                responses.append(text)

            if logger.isEnabledFor(logging.DEBUG):
                for i, (local_i, t) in enumerate(valid):
                    resp_len = int(response_mask[i].sum().item())
                    logger.debug(
                        f"  [step {global_step}][grp {t.group_id}/{t.n_idx}] "
                        f"RESPONSE ({resp_len} tokens):\n"
                        f'  [ASSISTANT] Text: "{responses[i]}"'
                    )

            del gen_output, gen_batch, tokenized, response_ids, response_mask

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
                # Save per-step record for credit assignment.
                # Reuse image paths from prior steps; only save genuinely
                # new images to disk (each env step adds ~1 observation).
                all_images = getattr(t, "_pending_images", [])
                prior_paths = t.step_records[-1].image_paths if t.step_records else []
                n_prior = len(prior_paths)
                new_images = all_images[n_prior:]
                new_paths = self._save_step_images(
                    t, t.num_steps, new_images
                )
                full_paths = list(prior_paths) + new_paths
                t.step_records.append(StepRecord(
                    prompt=getattr(t, "_pending_prompt", []),
                    image_paths=full_paths,
                    response=responses[i],
                    tensor_cache_path=_cache_paths[i] if _cache_paths else None,
                    old_log_probs=step_log_probs[i].tolist() if step_log_probs is not None else None,
                ))
                t._pending_prompt = None
                t._pending_images = None
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
                    t.terminated = True
                    n_terminated_this_step += 1

            logger.info(
                f"  step {global_step}: {len(valid)} active, "
                f"{n_terminated_this_step} done"
                + (f" ({n_failed_this_step} failed)" if n_failed_this_step else "")
                + f", gen={t_gen_elapsed:.1f}s, "
                f"total={time.time() - t_step:.1f}s"
                + (f", pending={len(pending_queue)}" if pending_queue else "")
            )

            del prompt_results_raw, prompts, images_list, valid
            del step_futures, step_results, responses
            if _gc_interval > 0 and global_step % _gc_interval == 0:
                gc.collect()
                torch.cuda.empty_cache()

            if _mem_log_interval > 0 and global_step % _mem_log_interval == 0:
                _log_memory(f"step {global_step} end ({len(active)} active, {len(all_trajectories)} total)")

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
        _log_memory(f"harvest/refill start ({len(newly_done)} newly done)")

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

        # ── Serialize step records to disk to free memory ──
        trajectory_cache_dir = getattr(self, "trajectory_cache_dir", None)
        for t in newly_done:
            if trajectory_cache_dir and t.step_records:
                cache_path = self._serialize_step_records(t, trajectory_cache_dir)
                t._cache_path = cache_path
                t.step_records = []  # free memory

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

        # Free memory from completed trajectories
        gc.collect()
        torch.cuda.empty_cache()

        _log_memory(f"after harvest+refill ({len(newly_done)} done)")

    # ── private helpers ───────────────────────────────────────────────

    def _save_step_images(
        self, trajectory: Trajectory, step_idx: int, images: list[Any]
    ) -> list[str]:
        """Save per-step PIL images to disk and return their file paths."""
        cache_dir = self.trajectory_cache_dir
        if not cache_dir or not images:
            return []

        traj_dir = os.path.join(cache_dir, trajectory.episode_id)
        os.makedirs(traj_dir, exist_ok=True)

        paths = []
        for img_idx, img in enumerate(images):
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            path = os.path.join(traj_dir, f"step{step_idx}_img{img_idx}.jpg")
            img.save(path)
            paths.append(path)
        return paths

    @staticmethod
    def _serialize_step_records(
        trajectory: Trajectory, cache_dir: str
    ) -> str:
        """Serialize a trajectory's step records to a JSONL file on disk."""
        traj_dir = os.path.join(cache_dir, trajectory.episode_id)
        os.makedirs(traj_dir, exist_ok=True)
        cache_path = os.path.join(traj_dir, "steps.jsonl")
        with open(cache_path, "w") as f:
            for i, rec in enumerate(trajectory.step_records):
                json.dump(
                    {
                        "step": i,
                        "prompt": rec.prompt,
                        "image_paths": rec.image_paths,
                        "response": rec.response,
                        "tensor_cache_path": rec.tensor_cache_path,
                        "old_log_probs": rec.old_log_probs,
                    },
                    f,
                )
                f.write("\n")
        return cache_path

    @staticmethod
    def _load_step_records(cache_path: str) -> list[StepRecord]:
        """Load step records from a JSONL file."""
        records = []
        with open(cache_path) as f:
            for line in f:
                d = json.loads(line)
                records.append(
                    StepRecord(
                        prompt=d["prompt"],
                        image_paths=d["image_paths"],
                        response=d["response"],
                        tensor_cache_path=d.get("tensor_cache_path"),
                        old_log_probs=d.get("old_log_probs"),
                    )
                )
        return records

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


# Deferred re-export — must come after EnvFactory is defined to break circular
# import with objectnav_factory.py.
from .objectnav_factory import ObjectNavEnvFactory  # noqa: F401, E402
