"""Ray-remote simulator pool for parallel AI2Thor environment management.

Each SimulatorPool manages multiple ObjectNavEnvAdapter instances on a single GPU.
This enables parallel environment operations during multi-turn GRPO rollouts,
mirroring the ViGoRL reference architecture.

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
    with its own AI2Thor Controller). Operations can be called in parallel across
    multiple pools via Ray futures.
    """

    def __init__(
        self,
        gpu_id: int,
        num_slots: int,
        system_prompt: str,
        render_width: int = 640,
        render_height: int = 640,
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
        """Create an environment from dataset item data, return slot ID.

        Returns None if no slots are available.
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
        """Release a specific slot, closing the AI2Thor controller."""
        if slot_id < 0 or slot_id >= self.num_slots:
            return
        if self.slots[slot_id] is not None:
            try:
                self.slots[slot_id].close()
            except Exception:
                pass
            self.slots[slot_id] = None
        self.slot_available[slot_id] = True

    def release_all(self) -> None:
        """Release all slots."""
        for i in range(self.num_slots):
            self.release_env(i)
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
                        f"Releasing."
                    )
            self.release_env(i)
        gc.collect()
        return True
