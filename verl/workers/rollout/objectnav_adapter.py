"""ObjectNav environment adapter for multi-turn GRPO rollouts.

Contains all spatial-reasoning-specific logic: prompt building via
``build_annotate_style_context_from_history``, action parsing via
``ActionProposer._parse_action_response``, and reward computation.

Instances live inside SimulatorPool Ray actors, NOT on the driver.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from copy import deepcopy
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


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
    ):
        self.env = env
        self.state_history = state_history
        self.system_prompt = system_prompt
        self.action_proposer = action_proposer
        self.coordinate_normalization_scale = coordinate_normalization_scale
        self.max_observations = max_observations

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
        """Build prompt matching the SFT training format from sft_data.py.

        SFT format (single user message):
            [Step 0]            <- prior image labels + <image> tags
            <image>
            [Step 1]
            <image>
            Your task is to find the **X** (desc).

            **Memory from previous steps:**
            - Step 0: summary0
            ...

            Step N. Here is your current observation:
            <image>
        """
        from interactive_reasoning.objectnavtask.agent.agent_utils import (
            build_annotate_style_context_from_history,
        )

        model_context = build_annotate_style_context_from_history(
            self.state_history,
            coordinate_normalization_scale=self.coordinate_normalization_scale,
            include_error_feedback=True,
            max_observations=self.max_observations,
        )

        # Separate context into prior images, text-only parts, and the current
        # observation so we can reorder to match the SFT data layout.
        prior_image_parts = []   # (label_text, image)
        text_parts = []          # task description, memory, errors, etc.
        current_obs_part = None  # (step_text, image) — always the last image entry
        images = []

        for _assistant_resp, user_response, observation in model_context.context:
            if observation is not None:
                img = Image.fromarray(observation) if isinstance(observation, np.ndarray) else observation
                # The last observation entry is the current step
                if current_obs_part is not None:
                    # Previous "current" was actually a prior — demote it
                    prior_image_parts.append(current_obs_part)
                current_obs_part = (user_response, img)
            elif user_response:
                text_parts.append(user_response)

        # Build user content in SFT order:
        # 1) Prior images with labels (newline-separated, matching SFT's \n join)
        user_parts = []
        for label, img in prior_image_parts:
            images.append(img)
            if label:
                user_parts.append(f"{label}\n<image>")
            else:
                user_parts.append("<image>")

        # 2) Text parts (task description, memory, errors) joined by \n\n
        if text_parts:
            user_parts.append("\n\n".join(text_parts))

        # 3) Current observation
        if current_obs_part is not None:
            step_text, img = current_obs_part
            images.append(img)
            user_parts.append(f"{step_text}\n<image>")

        # 4) Step instruction (format reminder for the model)
        max_actions = self.env.configuration.max_actions
        if self.num_steps + 1 >= max_actions:
            instruction = self._step_instructions["forced"]
        else:
            instruction = self._step_instructions["standard"]
        user_parts.append(instruction)

        user_text = "\n".join(user_parts)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_text},
        ]
        return messages, images

    @staticmethod
    def _check_format(response: str) -> float:
        """Check if response has <think>...</think> <summary>...</summary> <action>."""
        pattern = re.compile(
            r"<think>.*?</think>\s*<summary>.*?</summary>\s*<(?:explore|answer).*?(?:/>|</(?:explore|answer)>)",
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
        reward = 0.0
        # Primary reward: success
        if self.success:
            reward += 1.0
        # Format reward: average format score across steps (weight 0.1)
        if self.format_scores:
            avg_format = sum(self.format_scores) / len(self.format_scores)
            reward += 0.1 * avg_format
        # Validity reward: average validity score across steps (weight 0.15)
        if self.validity_scores:
            avg_validity = sum(self.validity_scores) / len(self.validity_scores)
            reward += 0.15 * avg_validity
        # Step penalty to encourage shorter trajectories
        reward -= 0.005 * self.num_steps

        avg_fmt = sum(self.format_scores) / len(self.format_scores) if self.format_scores else 0.0
        avg_val = sum(self.validity_scores) / len(self.validity_scores) if self.validity_scores else 0.0
        logger.info(
            f"Trajectory reward: success={self.success}, "
            f"initial_distance={self.initial_distance:.2f}, "
            f"final_distance={self.final_distance:.2f}, "
            f"num_steps={self.num_steps}, "
            f"avg_format={avg_fmt:.2f}, avg_validity={avg_val:.2f}, "
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
