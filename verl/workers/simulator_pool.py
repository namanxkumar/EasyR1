"""Ray-remote simulator pool for parallel AI2Thor environment management.

Each SimulatorPool manages multiple ObjectNavEnvAdapter instances on a single GPU.
This enables parallel environment operations during multi-turn GRPO rollouts,
mirroring the ViGoRL reference architecture.

Controller reuse: AI2Thor Unity processes are expensive to start (~30-100s).
The pool keeps bare AI2ThorController objects alive between episodes and passes
them to new ObjectNavEnvironment instances, which call reset_scene() internally.
This is much faster (~2-5s) than creating a new Unity process each time.

Usage:
    pool = SimulatorPool.options(
        runtime_env={"env_vars": {"CUDA_VISIBLE_DEVICES": str(gpu_id)}}
    ).remote(gpu_id=gpu_id, num_slots=8, system_prompt="...", ...)

    slot_id = ray.get(pool.acquire_env.remote(item_data))
    ray.get(pool.reset_env.remote(slot_id))
    prompt, images = ray.get(pool.build_prompt.remote(slot_id))
    reward, terminated, info = ray.get(pool.step_env.remote(slot_id, action_text))
    ray.get(pool.release_env.remote(slot_id))
"""

from __future__ import annotations

import gc
import logging
import os
import time
import traceback
from typing import Any, Optional

import numpy as np
import ray

logger = logging.getLogger(__name__)


@ray.remote(num_gpus=0, num_cpus=1)
class SimulatorPool:
    """Manages multiple AI2Thor ObjectNavEnvAdapter instances on a single GPU.

    Each 'slot' holds a full ObjectNavEnvAdapter (which owns an ObjectNavEnvironment
    with its own AI2Thor Controller). Bare AI2ThorController objects are cached
    and reused across episodes to avoid expensive Unity process restarts.
    """

    def __init__(
        self,
        gpu_id: int,
        num_slots: int,
        system_prompt: str,
        render_width: int = 616,
        render_height: int = 616,
        max_depth: int = 30,
        coordinate_normalization_scale: float = 1.0,
        max_observations: int = 20,
        context_mode: str = "multi_turn",
        past_k_steps: "int | None" = None,
        reward_mode: str = "continuous",
        sdpo_capture: bool = False,
    ):
        # Force AI2Thor to use the specified GPU (set before any CUDA init).
        # On Linux64 (H100 / AI2THOR_USE_LINUX64=1) leave CUDA_VISIBLE_DEVICES
        # empty so ai2thor skips its vulkaninfo precondition — vulkan is gated
        # on H100 datacenter SKUs. Rendering is routed via x_display instead.
        self._use_linux64 = os.environ.get("AI2THOR_USE_LINUX64", "0") == "1"
        if self._use_linux64:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        self.gpu_id = gpu_id
        self.num_slots = num_slots
        self.system_prompt = system_prompt
        self.render_width = render_width
        self.render_height = render_height
        self.max_depth = max_depth
        self.coordinate_normalization_scale = coordinate_normalization_scale
        self.max_observations = max_observations
        self.context_mode = context_mode
        self.past_k_steps = past_k_steps
        self.reward_mode = reward_mode
        # SDPO: when set, the adapter captures per-step privileged frames
        # (instance-seg / depth / pose / intrinsics) so the post-rollout hint pass
        # can answer the verifier's ground-truth tools without a live simulator.
        self.sdpo_capture = sdpo_capture

        # Slot management
        self.slots: list[Optional[Any]] = [None] * num_slots
        self.slot_available: list[bool] = [True] * num_slots

        # Cached bare AI2ThorController objects (reused across episodes)
        self._cached_controllers: list[Optional[Any]] = [None] * num_slots

        # Reactive pool sizing: each slot starts enabled. When a trajectory
        # finishes (`release_env`), if GPU usage has crossed `_target_used_frac`
        # of total, that slot is disabled and its controller torn down — the
        # pool shrinks under live memory pressure rather than guessing at
        # warmup. Disabled slots come back next warmup if memory recovers.
        #
        # Cascading-shrink guard: in-flight `generate()` calls hold activation
        # memory that doesn't subside instantly. Without throttling, every
        # release in a burst sees the same high-water mark and keeps shrinking
        # to zero. We bound it via:
        #   - `_min_enabled_slots`: never drop below this many enabled slots.
        #     Trajectories cycle in the continuous loop, so per-pool minimum
        #     can be much smaller than rollout `n` — 2 is enough to keep work
        #     flowing while the rest of the cluster carries the load.
        #   - `_shrink_cooldown_sec`: wall-clock gap between shrinks so the
        #     freed controller's memory + activations have time to settle.
        self._slot_enabled: list[bool] = [True] * num_slots
        self._target_used_frac: float = 0.9
        self._min_enabled_slots: int = min(num_slots, 2)
        self._shrink_cooldown_sec: float = 8.0
        self._last_shrink_time: float = 0.0

        # Shared action proposer (parse-only, no VLM)
        from interactive_reasoning.objectnavtask.agent.action_proposer import (
            ActionProposer,
        )

        self._action_proposer = ActionProposer(
            answer_tag="answer",
            explore_tag="explore",
            remember_tag="summary",
            coordinate_normalization_scale=coordinate_normalization_scale,
        )

        logger.info(
            f"SimulatorPool initialized: gpu_id={gpu_id}, num_slots={num_slots}"
        )

    def _get_gpu_memory(self) -> Optional[tuple[int, int]]:
        """Query (free_mb, total_mb) via nvidia-smi. Returns None on failure."""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free,memory.total",
                 "--format=csv,noheader,nounits", f"--id={self.gpu_id}"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                free_str, total_str = result.stdout.strip().split(", ")
                return int(free_str), int(total_str)
        except Exception:
            pass
        return None

    def _get_free_gpu_mb(self) -> Optional[int]:
        mem = self._get_gpu_memory()
        return mem[0] if mem is not None else None

    def _log_gpu_memory(self, context: str):
        """Log GPU + process RSS memory usage for diagnostics."""
        try:
            import psutil
            rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            rss_mb = -1
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free",
                 "--format=csv,noheader,nounits", f"--id={self.gpu_id}"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                if len(parts) == 3:
                    used, total, free = parts
                    logger.info(
                        f"[GPU {self.gpu_id}] {context}: "
                        f"RSS={rss_mb:.0f}MB, "
                        f"GPU used={used}MB / total={total}MB (free={free}MB)"
                    )
                    return
        except Exception:
            pass
        if rss_mb >= 0:
            logger.info(f"[GPU {self.gpu_id}] {context}: RSS={rss_mb:.0f}MB")

    # ── warmup ─────────────────────────────────────────────────────────

    def warmup_controllers(self, dummy_scene_metadata: dict) -> int:
        """Pre-create bare AI2Thor controllers for all slots using a dummy scene.

        Only creates the AI2ThorController (Unity process), not a full
        ObjectNavEnvironment, so no target-object pathfinding is attempted.

        Reactive sizing: re-enables every slot at warmup; mid-rollout
        `release_env` is the place where slots get disabled under memory
        pressure. So memory recovers between phases (rollout → train →
        rollout) the pool comes back to full size.

        Returns the number of controllers successfully warmed up.
        """
        from interactive_reasoning.objectnavtask.environment.ai2thor_controller import (
            AI2ThorController,
            AI2ThorControllerConfiguration,
        )

        # Re-enable every slot; release_env will shrink under live pressure.
        self._slot_enabled = [True] * self.num_slots
        self._log_gpu_memory(f"before warmup ({self.num_slots} slots)")

        # Linux64: one Xorg per Unity instance is required (probes 121479/121481
        # showed multi-Unity-on-one-display produces identical framebuffers).
        # The sbatch wrapper spawns N Xorgs and exports a comma-separated list
        # per physical GPU via AI2THOR_DISPLAYS_FOR_GPU_<phys_id>.
        displays: list[Optional[str]] = [None] * self.num_slots
        if self._use_linux64:
            env_key = f"AI2THOR_DISPLAYS_FOR_GPU_{self.gpu_id}"
            raw = os.environ.get(env_key, "")
            disp_list = [d.strip() for d in raw.split(",") if d.strip()]
            if len(disp_list) < self.num_slots:
                logger.warning(
                    f"[GPU {self.gpu_id}] {env_key} has {len(disp_list)} displays "
                    f"but pool has {self.num_slots} slots; extra slots will share "
                    f"the last display and may produce corrupted frames."
                )
            for i in range(self.num_slots):
                if disp_list:
                    displays[i] = disp_list[min(i, len(disp_list) - 1)]

        # Parallelize per-slot Unity spawns within this pool. Each slot binds to
        # a distinct x_display (Linux64), so renderer-level contention is bounded
        # to per-display Xorg load. AI2ThorController's signal patch mutates
        # process globals (signal.signal / signal.alarm) — install it once at
        # the outer scope so the worker threads don't race; per-call patches
        # inside _create_ai2thor_controller become no-ops via refcount.
        from concurrent.futures import ThreadPoolExecutor
        from interactive_reasoning.objectnavtask.environment.ai2thor_controller import (
            safe_signal_patch,
        )

        slots_to_create = [
            i for i in range(self.num_slots) if self._cached_controllers[i] is None
        ]
        created = self.num_slots - len(slots_to_create)

        def _create_one(i: int):
            config = AI2ThorControllerConfiguration(
                scene_metadata=dummy_scene_metadata,
                gpu_id=0,  # 0 = first (only) visible GPU after CUDA_VISIBLE_DEVICES
                render_width=self.render_width,
                render_height=self.render_height,
                x_display=displays[i],
            )
            return i, AI2ThorController(configuration=config)

        if slots_to_create:
            # Cap per-pool concurrency. With one pool per GPU, all pools warm
            # up simultaneously, so per-pool W workers => node-wide ≈ 8·W
            # concurrent Unity launches. At 256 sims (32 slots/pool) the
            # unbounded path tried 256 concurrent Unity boots and each create
            # timed out at 300s (job 121584). 8/pool keeps node peak ~64,
            # matching the regime that worked for 128-sim runs (121576 OK in
            # 2:48). Override via SIMULATOR_POOL_WARMUP_WORKERS.
            warmup_workers = int(
                os.environ.get("SIMULATOR_POOL_WARMUP_WORKERS", "8")
            )
            with safe_signal_patch(), ThreadPoolExecutor(
                max_workers=min(warmup_workers, len(slots_to_create)),
                thread_name_prefix=f"warmup-gpu{self.gpu_id}",
            ) as ex:
                futures = {ex.submit(_create_one, i): i for i in slots_to_create}
                for fut in futures:
                    slot_idx = futures[fut]
                    try:
                        i, controller = fut.result()
                        self._cached_controllers[i] = controller
                        created += 1
                        logger.info(
                            f"Warmed up controller {created}/{self.num_slots} "
                            f"on gpu {self.gpu_id}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to warm up controller for slot {slot_idx} "
                            f"on gpu {self.gpu_id}: {e}"
                        )
        self._log_gpu_memory(f"after warmup ({created}/{self.num_slots} created)")
        return created

    # ── pool info ─────────────────────────────────────────────────────

    def get_pool_info(self) -> dict:
        """Return pool status."""
        enabled = sum(1 for i in range(self.num_slots) if self._slot_enabled[i])
        available = sum(
            1 for i in range(self.num_slots)
            if self._slot_enabled[i] and self.slot_available[i]
        )
        return {
            "gpu_id": self.gpu_id,
            "total": self.num_slots,
            "effective": enabled,
            "available": available,
        }

    def get_available_count(self) -> int:
        """Return the number of available (free) enabled slots."""
        return sum(
            1 for i in range(self.num_slots)
            if self._slot_enabled[i] and self.slot_available[i]
        )

    # ── lifecycle ─────────────────────────────────────────────────────

    def acquire_env(self, item_data: dict) -> Optional[int]:
        """Acquire a slot and set up an environment for the given dataset item.

        If the slot has a cached AI2ThorController from a previous episode,
        it is passed to ObjectNavEnvironment which reuses it via reset_scene().

        Returns the slot ID, or None if no slots are available.
        """
        slot_id = None
        for i in range(self.num_slots):
            if self._slot_enabled[i] and self.slot_available[i]:
                slot_id = i
                break

        if slot_id is None:
            logger.warning(
                f"SimulatorPool gpu={self.gpu_id}: no slots available"
            )
            return None

        self.slot_available[slot_id] = False

        try:
            self._log_gpu_memory(f"acquire_env slot={slot_id}")

            from interactive_reasoning.objectnavtask.environment import (
                ObjectNavEnvironment,
            )
            from interactive_reasoning.objectnavtask.environment.configuration import (
                ObjectNavEnvironmentConfiguration,
            )

            env_config = ObjectNavEnvironmentConfiguration(
                scene_metadata=item_data["scene_metadata"],
                gpu_id=0,  # 0 = first (only) visible GPU after CUDA_VISIBLE_DEVICES
                max_actions=self.max_depth,
                render_width=self.render_width,
                render_height=self.render_height,
                include_top_down_map=False,
                capture_extra_info=False,
                use_object_filter=True,
            )

            cached_ctrl = self._cached_controllers[slot_id]

            # Try to reuse cached controller; fall back to creating fresh
            env = None
            if cached_ctrl is not None:
                try:
                    env = ObjectNavEnvironment(
                        configuration=env_config,
                        target_object=item_data["target_object"],
                        target_object_description=item_data.get(
                            "target_object_description", ""
                        ),
                        target_object_id=item_data["target_object_id"],
                        target_object_synset=item_data.get("target_object_synset"),
                        synset_to_object_ids=item_data.get("synset_to_object_ids"),
                        target_object_position=item_data["target_object_position"],
                        initial_agent_state=item_data.get("initial_metadata"),
                        existing_controller=cached_ctrl,
                    )
                    logger.info(
                        f"Reused cached controller for slot {slot_id} on gpu {self.gpu_id}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to reuse cached controller for slot {slot_id}: {e}. "
                        f"Creating fresh controller."
                    )
                    try:
                        cached_ctrl.close_controller()
                    except Exception:
                        pass
                    self._cached_controllers[slot_id] = None
                    env = None

            if env is None:
                # Create from scratch (first time or after cache miss)
                logger.info(
                    f"Creating new AI2Thor controller for slot {slot_id} "
                    f"on gpu {self.gpu_id}"
                )
                env = ObjectNavEnvironment(
                    configuration=env_config,
                    target_object=item_data["target_object"],
                    target_object_description=item_data.get(
                        "target_object_description", ""
                    ),
                    target_object_id=item_data["target_object_id"],
                    target_object_synset=item_data.get("target_object_synset"),
                    synset_to_object_ids=item_data.get("synset_to_object_ids"),
                    target_object_position=item_data["target_object_position"],
                    initial_agent_state=item_data.get("initial_metadata"),
                )

            # Cache the bare controller for future reuse
            self._cached_controllers[slot_id] = env._ai2thor

            # Lazy import to avoid circular deps
            from verl.workers.rollout.multiturn_env import ObjectNavEnvAdapter

            adapter = ObjectNavEnvAdapter(
                env=env,
                state_history=None,  # initialized in reset()
                system_prompt=self.system_prompt,
                action_proposer=self._action_proposer,
                coordinate_normalization_scale=self.coordinate_normalization_scale,
                max_observations=self.max_observations,
                context_mode=self.context_mode,
                past_k_steps=self.past_k_steps,
                reward_mode=self.reward_mode,
                sdpo_capture=self.sdpo_capture,
            )

            # Stash the dataset item on the adapter so guided rollouts can
            # synthesize the pope-dagger expert trajectory without re-fetching.
            try:
                adapter.dataset_item = item_data
                adapter.expert_trajectory_cache = None
            except Exception:
                pass

            self.slots[slot_id] = adapter
            return slot_id

        except Exception as e:
            logger.error(
                f"Failed to create env in slot {slot_id} on gpu {self.gpu_id}: "
                f"{e}\n{traceback.format_exc()}"
            )
            self.slot_available[slot_id] = True
            self.slots[slot_id] = None
            raise

    def reset_env(self, slot_id: int) -> dict:
        """Reset the environment in the given slot. Returns observation info."""
        adapter = self.slots[slot_id]
        if adapter is None:
            raise ValueError(f"Slot {slot_id} is empty")
        return adapter.reset()

    def release_env(self, slot_id: int) -> None:
        """Release a slot, keeping the AI2Thor controller cached for reuse.

        Reactive shrink: if GPU usage is over `_target_used_frac` after
        release, disable this slot and tear down its controller — the pool
        gives up a slot rather than risk a vLLM/AI2Thor OOM on the next
        rollout. Disabled slots come back at the next `warmup_controllers`.
        """
        if slot_id < 0 or slot_id >= self.num_slots:
            return
        self._log_gpu_memory(f"release_env slot={slot_id} BEFORE")
        adapter = self.slots[slot_id]
        if adapter is not None:
            # IMPORTANT: break controller -> env bound-method reference cycle.
            #
            # ObjectNavEnvironment sets AI2Thor restart callback to
            # `env._restore_state_on_restart` (a bound method). If we cache the
            # controller and drop only `self.slots[slot_id]`, the cached
            # controller still strongly references the old env via this callback,
            # keeping full episode state/history alive.
            #
            # Under dynamic slot reuse this causes memory to grow gradually as
            # more unique slot/env instances are touched. We detach callback on
            # release; the next acquire() will set a fresh callback on the new
            # env instance.
            try:
                cached_ctrl = self._cached_controllers[slot_id]
                if cached_ctrl is not None:
                    cached_ctrl.set_restart_callback(None)
            except Exception:
                pass

            # Best-effort cleanup of large per-episode Python objects.
            try:
                adapter.state_history = None
            except Exception:
                pass

        # Clear the adapter but keep the cached controller
        self.slots[slot_id] = None
        self.slot_available[slot_id] = True

        # Reactive shrink (with min floor + cooldown): drop this slot's cached
        # controller iff GPU is over threshold, we still have slots above the
        # floor, and the previous shrink has had time to settle. Without
        # the cooldown, in-flight activation memory makes every release in a
        # burst look like a fresh OOM signal and the pool collapses to zero.
        enabled_count = sum(self._slot_enabled)
        if enabled_count <= self._min_enabled_slots:
            self._log_gpu_memory(f"release_env slot={slot_id} AFTER")
            return
        now = time.monotonic()
        if now - self._last_shrink_time < self._shrink_cooldown_sec:
            self._log_gpu_memory(f"release_env slot={slot_id} AFTER")
            return
        mem = self._get_gpu_memory()
        if mem is not None:
            free_mb, total_mb = mem
            used_frac = (total_mb - free_mb) / total_mb
            if used_frac > self._target_used_frac:
                ctrl = self._cached_controllers[slot_id]
                if ctrl is not None:
                    try:
                        ctrl.set_restart_callback(None)
                        ctrl.close_controller()
                    except Exception as e:
                        logger.warning(
                            f"[GPU {self.gpu_id}] Failed to close shrunk slot "
                            f"{slot_id}: {e}"
                        )
                    self._cached_controllers[slot_id] = None
                self._slot_enabled[slot_id] = False
                self.slot_available[slot_id] = False
                self._last_shrink_time = now
                logger.warning(
                    f"[GPU {self.gpu_id}] Reactive shrink: used={100*used_frac:.0f}% "
                    f"> {int(100*self._target_used_frac)}%, disabled slot {slot_id} "
                    f"({enabled_count - 1}/{self.num_slots} enabled, "
                    f"floor={self._min_enabled_slots}, cooldown={self._shrink_cooldown_sec}s)."
                )
                gc.collect()
        self._log_gpu_memory(f"release_env slot={slot_id} AFTER")

    def release_all(self) -> None:
        """Release all slots (keeps controllers cached)."""
        for i in range(self.num_slots):
            self.release_env(i)
        gc.collect()

    def destroy_all(self) -> None:
        """Destroy all slots and close all AI2Thor controllers."""
        for i in range(self.num_slots):
            self.slots[i] = None
            self.slot_available[i] = True
            ctrl = self._cached_controllers[i]
            if ctrl is not None:
                try:
                    ctrl.close_controller()
                except Exception:
                    pass
                self._cached_controllers[i] = None
        gc.collect()

    # ── environment operations ────────────────────────────────────────

    def step_env(self, slot_id: int, action_text: str) -> tuple:
        """Step the environment with the model's response text.

        Returns (reward, terminated, info_dict).
        """
        adapter = self.slots[slot_id]
        if adapter is None:
            raise ValueError(f"Slot {slot_id} is empty")
        return adapter.step(action_text)

    def build_prompt(self, slot_id: int) -> tuple:
        """Build the current prompt and image list from environment state.

        Returns (prompt_text, images_list) where images are PIL Images.
        """
        adapter = self.slots[slot_id]
        if adapter is None:
            raise ValueError(f"Slot {slot_id} is empty")
        return adapter.build_prompt()

    def build_prompt_with_full(self, slot_id: int) -> tuple:
        """Build both K-window-filtered and full-trajectory prompts.

        Returns ``(msgs, imgs, full_msgs, full_imgs)``. The last two are
        ``None`` when ``past_k_steps`` is disabled. Used by the past-K
        packed-MROPE path to compute packed position IDs from the full
        trajectory layout while only sending the K-window to vLLM.
        """
        adapter = self.slots[slot_id]
        if adapter is None:
            raise ValueError(f"Slot {slot_id} is empty")
        return adapter.build_prompt_with_full()

    def get_trajectory_reward(self, slot_id: int) -> float:
        """Get the trajectory-level reward for a completed episode."""
        adapter = self.slots[slot_id]
        if adapter is None:
            return 0.0
        return adapter.get_trajectory_reward()

    def get_ground_truth(self, slot_id: int) -> str:
        """Get ground truth metadata as JSON string."""
        adapter = self.slots[slot_id]
        if adapter is None:
            return "{}"
        return adapter.get_ground_truth()

    # ── pope-dagger guided rollout hook ──────────────────────────────
    def _next_expert_sparse_step(self, slot_id: int):
        """Resolve the next expert ``SparseStep`` for this slot, or ``None``.

        Shared by ``compute_expert_action`` (string-only) and
        ``compute_expert_entry`` (rich entry). Builds + caches the expert
        ``SparseTrajectory`` on the adapter on first call. The trajectory is
        synthesized from the env's pose AT BUILD TIME — i.e. the branch pose
        (the first forced step) — so its ``steps[1:]`` are actions relative to
        the branch, NOT to episode start. We therefore index it relative to the
        branch: the j-th forced step (1-based) is ``steps[j]``, where
        ``j = num_steps - base + 1`` and ``base`` is ``num_steps`` captured at
        build time. (The old ``num_steps + 1`` indexed by absolute episode step,
        which silently skipped the first ``branch`` expert actions — the forced
        rollout then ended at the wrong pose and the screen-space expert answer
        pixel, computed for the correct final pose, missed.)

        Returns ``(adapter, dataset_item, step)`` or ``None`` if the oracle is
        unavailable or the branch-relative trajectory is exhausted (the forced
        window ran past its end) — the caller treats ``None`` as "hand off to
        the student" rather than injecting a no-op forced step.
        """
        adapter = self.slots[slot_id]
        if adapter is None:
            raise ValueError(f"Slot {slot_id} is empty")
        try:
            from pope_dagger.expert_replay import compute_expert_actions_via_sparsify
        except Exception as e:
            logger.warning(f"pope_dagger import failed: {e!r}")
            return None

        env = getattr(adapter, "env", None)
        dataset_item = getattr(adapter, "dataset_item", None) or getattr(adapter, "item_data", None)
        if env is None or dataset_item is None:
            logger.warning(
                f"Slot {slot_id}: missing env or dataset_item on adapter "
                f"(env={env is not None}, item={dataset_item is not None})"
            )
            return None

        cache = getattr(adapter, "expert_trajectory_cache", None)
        if cache is None:
            try:
                cache = compute_expert_actions_via_sparsify(env=env, dataset_item=dataset_item)
            except Exception as e:
                logger.warning(f"compute_expert_actions_via_sparsify failed: {e!r}")
                return None
            try:
                adapter.expert_trajectory_cache = cache
                # Capture the branch base: num_steps at build time. The forced
                # window opens here, so steps[1] is the first action from this
                # pose and maps to num_steps == base.
                adapter.expert_cache_base_num_steps = int(getattr(adapter, "num_steps", 0) or 0)
            except Exception:
                pass

        # Branch-relative index: steps[0] is the no-op "initial" step; the j-th
        # forced step (1-based) is steps[j] with j = num_steps - base + 1.
        base = int(getattr(adapter, "expert_cache_base_num_steps", 0) or 0)
        idx = (int(getattr(adapter, "num_steps", 0) or 0) - base) + 1
        steps = getattr(cache, "steps", None) or []
        if idx < 1 or idx >= len(steps):
            return None
        return adapter, dataset_item, steps[idx]

    def compute_expert_action(self, slot_id: int):
        """Return the next pope-dagger oracle action string, or ``None``.

        Thin string-only wrapper over ``_next_expert_sparse_step`` (back-compat
        entry point for the guided rollout's expert-action hook). Returns
        ``None`` when the oracle has no real action to give — the expert
        trajectory is exhausted (the forced window ran past its end) or the
        oracle failed to synthesize one. The guided rollout treats ``None`` as
        "hand off to the student for this step" rather than injecting a no-op
        forced step (a ``direction:0`` non-action that would then be SFT'd).
        """
        resolved = self._next_expert_sparse_step(slot_id)
        if resolved is None:
            return None
        _adapter, _item, step = resolved
        return getattr(step.action, "formatted", None) or None

    def compute_expert_entry(self, slot_id: int, past_k: int = 0, future_k: int = 0):
        """Return the next expert step as a rich dict for teacher annotation.

        Surfaces the fields ``annotate_expert_cache`` needs that
        ``compute_expert_action`` discards: the ``SparseAction`` object, its
        ``rationale_kind`` (action_type), the **before**-observation image (the
        exact frame the student is looking at when this action is chosen), and
        the episode's target / house metadata. Returns ``None`` when no expert
        step is available so the caller can fall back to the placeholder
        annotator.

        The before-image is the live ``state_history`` head — the same image
        the adapter uses as ``prev_observation`` in ``step`` — so the teacher
        reasons over the on-distribution branch observation, not the expert's
        synthesized pose frame.

        When ``past_k > 0`` / ``future_k > 0``, also returns context windows so
        the teacher can reason history- and plan-aware (not single-frame):

        * ``past_steps``: the last ``past_k`` *executed* steps on this slot,
          oldest→newest, each ``{step_index, image (the obs the agent saw before
          that step), response (the assistant <think>+action text)}``. Read from
          the live ``state_history`` — no Trajectory plumbing needed.
        * ``future_steps``: the next ``future_k`` *planned* expert steps from the
          cached SparseTrajectory, each ``{sparse_action, action_str, image (the
          expert's synthesized pose frame)}``. The expert's look-ahead plan.
        """
        resolved = self._next_expert_sparse_step(slot_id)
        if resolved is None:
            return None
        adapter, dataset_item, step = resolved
        action = step.action
        formatted = getattr(action, "formatted", None)
        if not formatted:
            return None

        sh = getattr(adapter, "state_history", None)

        # Before-observation: the frame the agent currently sees (pre-step).
        # The head state's observation can be None when the previous step was a
        # no-op / env error: ``ObjectNavEnvAdapter.step`` appends an error_state
        # with ``observation=None`` (e.g. a ground action the navmesh refused
        # from the branch pose). On a no-op the agent never moved, so the most
        # recent non-None observation in the history *is* the current view — fall
        # back to it instead of returning image=None (which would force the
        # teacher onto the generic placeholder annotator and lose real reasoning).
        image = None
        try:
            obs = None
            if sh is not None:
                for _act, _state in reversed(sh.to_list()):
                    o = getattr(_state, "observation", None)
                    if o is not None:
                        obs = o
                        break
            if obs is not None:
                image = np.ascontiguousarray(np.asarray(obs))
            else:
                logger.warning(
                    f"Slot {slot_id}: no non-None observation in state_history; "
                    f"teacher before-image unavailable"
                )
        except Exception as e:
            logger.warning(f"Slot {slot_id}: could not read before-observation: {e!r}")

        # Past window: last past_k executed (action, state) pairs. The obs the
        # agent saw BEFORE pair j is the root state (j==0) or pair j-1's state.
        past_steps = []
        if past_k > 0 and sh is not None:
            try:
                pairs = sh.action_state_pairs or []
                n = len(pairs)
                for j in range(max(0, n - past_k), n):
                    act_j = pairs[j][0]
                    obs_before = (
                        getattr(sh.root_state, "observation", None) if j == 0
                        else getattr(pairs[j - 1][1], "observation", None)
                    )
                    img_j = (
                        np.ascontiguousarray(np.asarray(obs_before))
                        if obs_before is not None else None
                    )
                    past_steps.append({
                        "step_index": j,
                        "image": img_j,
                        "response": getattr(act_j, "response", "") or "",
                    })
            except Exception as e:
                logger.warning(f"Slot {slot_id}: could not build past window: {e!r}")

        # Future window: next future_k planned expert steps from the cache.
        future_steps = []
        if future_k > 0:
            try:
                cache = getattr(adapter, "expert_trajectory_cache", None)
                base = int(getattr(adapter, "expert_cache_base_num_steps", 0) or 0)
                cur_idx = (int(getattr(adapter, "num_steps", 0) or 0) - base) + 1
                steps = getattr(cache, "steps", None) or []
                for k in range(cur_idx + 1, min(cur_idx + 1 + future_k, len(steps))):
                    fs = steps[k]
                    frgb = getattr(getattr(fs, "frame", None), "rgb", None)
                    future_steps.append({
                        "sparse_action": fs.action,
                        "action_str": getattr(fs.action, "formatted", "") or "",
                        "image": (
                            np.ascontiguousarray(np.asarray(frgb))
                            if frgb is not None else None
                        ),
                    })
            except Exception as e:
                logger.warning(f"Slot {slot_id}: could not build future window: {e!r}")

        item = dataset_item or {}
        return {
            "action_str": formatted,
            "sparse_action": action,
            "rationale_kind": getattr(action, "action_type", "") or "",
            "image": image,
            "target_object": item.get("target_object", "") if isinstance(item, dict) else "",
            "target_description": item.get("target_object_description", "") if isinstance(item, dict) else "",
            "house_id": str(item.get("house_id", "")) if isinstance(item, dict) else "",
            "sub_house_id": str(item.get("sub_house_id", "")) if isinstance(item, dict) else "",
            "past_steps": past_steps,
            "future_steps": future_steps,
        }

    # ── recovery ──────────────────────────────────────────────────────

    def release_and_rescue_broken(self) -> bool:
        """Release all slots, attempting to recover broken controllers."""
        for i in range(self.num_slots):
            adapter = self.slots[i]
            if adapter is not None:
                try:
                    # Quick health check
                    adapter.env.get_state()
                except Exception as e:
                    logger.warning(
                        f"Slot {i} on gpu {self.gpu_id} appears broken: {e}. "
                        f"Destroying cached controller."
                    )
                    ctrl = self._cached_controllers[i]
                    if ctrl is not None:
                        try:
                            ctrl.close_controller()
                        except Exception:
                            pass
                        self._cached_controllers[i] = None
            self.slots[i] = None
            self.slot_available[i] = True
        gc.collect()
        return True
