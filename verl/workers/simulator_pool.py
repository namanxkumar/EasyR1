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
    ):
        # Force AI2Thor to use the specified GPU (set before any CUDA init)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        self.gpu_id = gpu_id
        self.num_slots = num_slots
        self.system_prompt = system_prompt
        self.render_width = render_width
        self.render_height = render_height
        self.max_depth = max_depth
        self.coordinate_normalization_scale = coordinate_normalization_scale
        self.max_observations = max_observations

        # Slot management
        self.slots: list[Optional[Any]] = [None] * num_slots
        self.slot_available: list[bool] = [True] * num_slots

        # Cached bare AI2ThorController objects (reused across episodes)
        self._cached_controllers: list[Optional[Any]] = [None] * num_slots

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

    def _log_gpu_memory(self, context: str):
        """Log GPU memory usage for diagnostics."""
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
                        f"memory used={used}MB / total={total}MB (free={free}MB)"
                    )
        except Exception:
            pass  # best-effort diagnostics

    # ── warmup ─────────────────────────────────────────────────────────

    def warmup_controllers(self, dummy_scene_metadata: dict) -> int:
        """Pre-create bare AI2Thor controllers for all slots using a dummy scene.

        Only creates the AI2ThorController (Unity process), not a full
        ObjectNavEnvironment, so no target-object pathfinding is attempted.

        Returns the number of controllers successfully warmed up.
        """
        from interactive_reasoning.objectnavtask.environment.ai2thor_controller import (
            AI2ThorController,
            AI2ThorControllerConfiguration,
        )

        self._log_gpu_memory(f"before warmup ({self.num_slots} slots)")
        created = 0
        for i in range(self.num_slots):
            if self._cached_controllers[i] is not None:
                created += 1
                continue
            try:
                config = AI2ThorControllerConfiguration(
                    scene_metadata=dummy_scene_metadata,
                    gpu_id=0,  # 0 = first (only) visible GPU after CUDA_VISIBLE_DEVICES
                    render_width=self.render_width,
                    render_height=self.render_height,
                )
                controller = AI2ThorController(configuration=config)
                self._cached_controllers[i] = controller
                created += 1
                logger.info(
                    f"Warmed up controller {created}/{self.num_slots} "
                    f"on gpu {self.gpu_id}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to warm up controller for slot {i} on gpu "
                    f"{self.gpu_id}: {e}"
                )
        self._log_gpu_memory(f"after warmup ({created}/{self.num_slots} created)")
        return created

    # ── pool info ─────────────────────────────────────────────────────

    def get_pool_info(self) -> dict:
        """Return pool status."""
        available = sum(1 for a in self.slot_available if a)
        return {
            "gpu_id": self.gpu_id,
            "total": self.num_slots,
            "available": available,
        }

    # ── lifecycle ─────────────────────────────────────────────────────

    def acquire_env(self, item_data: dict) -> Optional[int]:
        """Acquire a slot and set up an environment for the given dataset item.

        If the slot has a cached AI2ThorController from a previous episode,
        it is passed to ObjectNavEnvironment which reuses it via reset_scene().

        Returns the slot ID, or None if no slots are available.
        """
        slot_id = None
        for i, avail in enumerate(self.slot_available):
            if avail:
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
            )

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
        """Release a slot, keeping the AI2Thor controller cached for reuse."""
        if slot_id < 0 or slot_id >= self.num_slots:
            return
        # Clear the adapter but keep the cached controller
        self.slots[slot_id] = None
        self.slot_available[slot_id] = True

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
            if self._cached_controllers[i] is not None:
                try:
                    self._cached_controllers[i].close_controller()
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
                    if self._cached_controllers[i] is not None:
                        try:
                            self._cached_controllers[i].close_controller()
                        except Exception:
                            pass
                        self._cached_controllers[i] = None
            self.slots[i] = None
            self.slot_available[i] = True
        gc.collect()
        return True
