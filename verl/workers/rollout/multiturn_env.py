"""Driver-side multi-turn environment rollout for GRPO training.

Runs multi-turn trajectories through AI2Thor environments by calling the
distributed ``actor_rollout_ref_wg.generate_sequences()`` at each step.
Environment operations (reset, step, build_prompt) are parallelized across
Ray-remote SimulatorPool actors.

Produces a standard ``DataProto`` batch compatible with the rest of the GRPO
pipeline (KL, advantage, actor update).
"""

from __future__ import annotations

import json
import logging
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
from tqdm import tqdm
from transformers import PreTrainedTokenizer, ProcessorMixin

from ...protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto

logger = logging.getLogger(__name__)


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

    # Accumulated per-step data
    step_responses: list[str] = field(default_factory=list)
    terminated: bool = False
    num_steps: int = 0

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

    # ── public API ────────────────────────────────────────────────────

    def generate_trajectories(
        self,
        actor_rollout_ref_wg,
        batch_size: int,
        n_trajectories: int,
        config,
        metrics: dict[str, Any],
    ) -> DataProto:
        """Run multi-turn rollouts with parallel env operations.

        When the total number of trajectories (batch_size * n_trajectories)
        exceeds the available simulator slots, trajectories are processed in
        **chunks**.  Each chunk acquires slots, runs its full multi-step
        episode, collects rewards, releases slots, then the next chunk starts.

        Within a chunk, at each step:
        1. Build prompts in parallel (Ray futures)
        2. Generate model responses (vLLM via actor_rollout_ref_wg)
        3. Step environments in parallel (Ray futures)
        """
        total = batch_size * n_trajectories

        # Determine available simulator slots
        pool_infos = ray.get(
            [p.get_pool_info.remote() for p in self.simulator_pools]
        )
        total_slots = sum(info["total"] for info in pool_infos)

        # Chunk size must be a multiple of n_trajectories so GRPO groups stay
        # together.  Each group = n_trajectories envs for the same prompt.
        groups_per_chunk = max(1, total_slots // n_trajectories)
        chunk_size = groups_per_chunk * n_trajectories
        num_chunks = (total + chunk_size - 1) // chunk_size

        logger.info(
            f"Starting multiturn env rollout: batch_size={batch_size}, "
            f"n={n_trajectories}, total={total}, total_slots={total_slots}, "
            f"chunk_size={chunk_size}, num_chunks={num_chunks}, "
            f"max_depth={self.max_depth}, pools={len(self.simulator_pools)}"
        )

        t_start = time.time()

        # Pre-collect all dataset items (ensures deterministic ordering)
        all_items = []
        for _ in range(batch_size):
            all_items.append(self.env_factory.get_next_item())

        all_trajectories: list[Trajectory] = []
        all_rewards: list[float] = []
        all_ground_truths: list[str] = []

        pbar = tqdm(
            total=total,
            desc="Collecting trajectories",
            unit="traj",
            dynamic_ncols=True,
        )

        for chunk_idx in range(num_chunks):
            # Determine which groups (dataset items) are in this chunk
            group_start = chunk_idx * groups_per_chunk
            group_end = min(group_start + groups_per_chunk, batch_size)
            chunk_items = all_items[group_start:group_end]
            chunk_n = len(chunk_items) * n_trajectories

            # ── Acquire + reset environments for this chunk ──
            trajectories = self._initialize_trajectories_from_items(
                chunk_items, n_trajectories
            )

            # ── Step-by-step rollout for this chunk ──
            trajectories = self._run_episode_loop(
                trajectories, actor_rollout_ref_wg, config
            )

            # ── Collect rewards ──
            reward_futures = [
                t.pool.get_trajectory_reward.remote(t.slot_id)
                for t in trajectories
            ]
            gt_futures = [
                t.pool.get_ground_truth.remote(t.slot_id)
                for t in trajectories
            ]
            rewards = ray.get(reward_futures)
            ground_truths = ray.get(gt_futures)

            # ── Release environments ──
            release_futures = [
                t.pool.release_env.remote(t.slot_id) for t in trajectories
            ]
            ray.get(release_futures)

            all_trajectories.extend(trajectories)
            all_rewards.extend(rewards)
            all_ground_truths.extend(ground_truths)

            # Update progress bar with running stats
            chunk_avg_steps = float(np.mean([t.num_steps for t in trajectories]))
            chunk_avg_reward = float(np.mean(rewards))
            pbar.update(chunk_n)
            pbar.set_postfix(
                chunk=f"{chunk_idx + 1}/{num_chunks}",
                avg_steps=f"{chunk_avg_steps:.1f}",
                avg_reward=f"{chunk_avg_reward:.2f}",
            )

        pbar.close()

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

        # ── Build final DataProto batch from all chunks ──
        return self._build_final_batch(
            all_trajectories, all_rewards, all_ground_truths, n_trajectories
        )

    def _run_episode_loop(
        self,
        trajectories: list[Trajectory],
        actor_rollout_ref_wg,
        config,
    ) -> list[Trajectory]:
        """Run the multi-step rollout loop for a chunk of trajectories."""
        total = len(trajectories)

        for step in range(self.max_depth):
            active_indices = [
                i for i, t in enumerate(trajectories) if not t.terminated
            ]
            if not active_indices:
                break

            t_step = time.time()

            # ── Build prompts in parallel via Ray ──
            prompt_futures = [
                trajectories[i].pool.build_prompt.remote(
                    trajectories[i].slot_id
                )
                for i in active_indices
            ]
            prompt_results = ray.get(prompt_futures)

            prompts = [r[0] for r in prompt_results]
            images_list = [r[1] for r in prompt_results]

            # Cache prompt/images for final batch construction
            for local_i, global_i in enumerate(active_indices):
                trajectories[global_i].last_prompt = prompts[local_i]
                trajectories[global_i].last_images = list(images_list[local_i])

            # ── Tokenize into DataProto format ──
            t_gen = time.time()
            tokenized = self._tokenize_prompts(prompts, images_list)
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
            gen_batch, pad_size = pad_dataproto_to_divisor(
                gen_batch, actor_rollout_ref_wg.world_size
            )
            gen_output = actor_rollout_ref_wg.generate_sequences(gen_batch)
            gen_output = unpad_dataproto(gen_output, pad_size)
            t_gen_elapsed = time.time() - t_gen

            # Decode responses
            response_ids = gen_output.batch["responses"]
            response_mask = gen_output.batch["response_mask"]
            responses = []
            for i in range(len(active_indices)):
                length = int(response_mask[i].sum().item())
                text = self.tokenizer.decode(
                    response_ids[i][:length], skip_special_tokens=True
                )
                responses.append(text)

            # ── Step environments in parallel via Ray ──
            step_futures = [
                trajectories[active_indices[local_i]].pool.step_env.remote(
                    trajectories[active_indices[local_i]].slot_id,
                    responses[local_i],
                )
                for local_i in range(len(active_indices))
            ]
            step_results = ray.get(step_futures)

            # Process step results
            n_terminated_this_step = 0
            for local_i, global_i in enumerate(active_indices):
                traj = trajectories[global_i]
                traj.step_responses.append(responses[local_i])
                traj.num_steps += 1

                reward, terminated, _info = step_results[local_i]
                if terminated:
                    traj.terminated = True
                    n_terminated_this_step += 1

            logger.info(
                f"  step {step}: {len(active_indices)} active, "
                f"{n_terminated_this_step} done, "
                f"gen={t_gen_elapsed:.1f}s, "
                f"total={time.time() - t_step:.1f}s"
            )

        # Force-terminate any remaining
        for t in trajectories:
            if not t.terminated:
                t.terminated = True

        return trajectories

    # ── private helpers ───────────────────────────────────────────────

    def _initialize_trajectories_from_items(
        self, items: list[dict], n_trajectories: int
    ) -> list[Trajectory]:
        """Create len(items) * n_trajectories trajectories with parallel env creation.

        For each item, all n_trajectories share the same dataset item
        (same scene, target, initial position) so GRPO group normalization
        is meaningful. Environments are created and reset in parallel via
        Ray remote calls to SimulatorPool actors.
        """
        total = len(items) * n_trajectories
        num_pools = len(self.simulator_pools)

        # Collect item data and pool assignments
        acquire_info = []  # (pool, item_data, item_idx, n_idx)
        for item_idx, item_data in enumerate(items):
            for n_idx in range(n_trajectories):
                pool_idx = (item_idx * n_trajectories + n_idx) % num_pools
                pool = self.simulator_pools[pool_idx]
                acquire_info.append((pool, item_data, item_idx, n_idx))

        # Acquire all environments in parallel
        t0 = time.time()
        acquire_futures = [
            pool.acquire_env.remote(item_data)
            for pool, item_data, _, _ in acquire_info
        ]
        slot_ids = ray.get(acquire_futures)
        logger.info(f"  acquire_env ({total}): {time.time() - t0:.1f}s")

        # Validate all slots were acquired
        for i, slot_id in enumerate(slot_ids):
            if slot_id is None:
                _, _, item_idx, n_idx = acquire_info[i]
                raise RuntimeError(
                    f"Failed to acquire simulator slot for item={item_idx}, "
                    f"n={n_idx}. This chunk needs {total} slots but not enough "
                    f"are available. Increase num_simulators or reduce batch_size."
                )

        # Reset all environments in parallel
        t0 = time.time()
        reset_futures = [
            acquire_info[i][0].reset_env.remote(slot_ids[i])
            for i in range(total)
        ]
        ray.get(reset_futures)
        logger.info(f"  reset_env ({total}): {time.time() - t0:.1f}s")

        # Build Trajectory objects
        trajectories = []
        for i, (pool, _, _, n_idx) in enumerate(acquire_info):
            episode_id = f"grpo_{uuid.uuid4().hex[:8]}_{n_idx}"
            trajectories.append(
                Trajectory(
                    pool=pool,
                    slot_id=slot_ids[i],
                    episode_id=episode_id,
                )
            )

        return trajectories

    def _tokenize_prompts(
        self,
        prompts: list[list[dict]],
        images_list: list[list[Image.Image]],
    ) -> dict[str, Any]:
        """Tokenize prompts with images into the format expected by generate_sequences.

        Args:
            prompts: list of chat message lists, each message is a dict with
                "role" and "content" keys. Content may contain ``<image>``
                placeholders that correspond to entries in *images_list*.
            images_list: list of image lists, one per prompt.

        Returns a dict with:
          - input_ids: (bs, max_prompt_length) tensor, left-padded
          - attention_mask: (bs, max_prompt_length) tensor
          - position_ids: (bs, 4, max_prompt_length) tensor (Qwen3-VL mrope)
          - raw_prompt_ids: list of unpadded token ID lists (for vLLM)
          - multi_modal_data: list of {"images": [...]} dicts
        """
        from ...utils.dataset import process_image
        from ...utils.torch_functional import postprocess_data

        batch_input_ids = []
        batch_attention_mask = []
        batch_position_ids = []
        batch_raw_prompt_ids = []
        batch_multi_modal_data = []

        for messages, images in zip(prompts, images_list):
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
                hf_messages, add_generation_prompt=True, tokenize=False
            )

            # Process images to match pixel constraints
            processed_images = (
                [
                    process_image(img, self.min_pixels, self.max_pixels)
                    for img in images
                ]
                if images
                else None
            )

            # Tokenize with processor
            model_inputs = self.processor(
                processed_images,
                [text_prompt],
                add_special_tokens=False,
                return_tensors="pt",
            )
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

            # Raw prompt IDs for vLLM (text-only tokenization)
            raw_prompt_ids = self.tokenizer.encode(
                text_prompt, add_special_tokens=False
            )
            if len(raw_prompt_ids) > self.max_prompt_length:
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_position_ids.append(position_ids)
            batch_raw_prompt_ids.append(raw_prompt_ids)
            batch_multi_modal_data.append({"images": images})

        return {
            "input_ids": torch.stack(batch_input_ids, dim=0),
            "attention_mask": torch.stack(batch_attention_mask, dim=0),
            "position_ids": torch.stack(batch_position_ids, dim=0),
            "raw_prompt_ids": np.array(batch_raw_prompt_ids, dtype=object),
            "multi_modal_data": np.array(batch_multi_modal_data, dtype=object),
        }

    def _build_final_batch(
        self,
        trajectories: list[Trajectory],
        rewards: list[float],
        ground_truths: list[str],
        n_trajectories: int,
    ) -> DataProto:
        """Construct the DataProto that the rest of the GRPO pipeline expects.

        For each trajectory, takes the last step's prompt (full history up to
        the final observation) and the model's last response. Tokenizes both
        and assembles into the standard format.
        """
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
    ):
        self.env = env
        self.state_history = state_history
        self.system_prompt = system_prompt
        self.action_proposer = action_proposer
        self.coordinate_normalization_scale = coordinate_normalization_scale
        self.max_observations = max_observations

        # Track reward components
        self.initial_distance: float | None = None
        self.final_distance: float | None = None
        self.success: bool = False
        self.num_steps: int = 0

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
        return {"observation": initial_state.observation}

    def build_prompt(self) -> tuple[list[dict], list[Image.Image]]:
        from interactive_reasoning.objectnavtask.agent.agent_utils import (
            build_annotate_style_context_from_history,
        )

        model_context = build_annotate_style_context_from_history(
            self.state_history,
            coordinate_normalization_scale=self.coordinate_normalization_scale,
            include_error_feedback=True,
            max_observations=self.max_observations,
        )

        prompt_parts = []
        images = []

        for _assistant_resp, user_response, observation in model_context.context:
            if observation is not None:
                if isinstance(observation, np.ndarray):
                    img = Image.fromarray(observation)
                else:
                    img = observation
                images.append(img)

                if user_response:
                    prompt_parts.append(f"{user_response}\n<image>")
                else:
                    prompt_parts.append("<image>")
            elif user_response:
                prompt_parts.append(user_response)

        user_text = "\n\n".join(prompt_parts)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_text},
        ]
        return messages, images

    def step(self, action_text: str) -> tuple[float, bool, dict[str, Any]]:
        self.num_steps += 1

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
        elif isinstance(action, ObjectNavGroundNavigationAction):
            action_type = "explore_ground"
        elif isinstance(action, ObjectNavDirectionalAction):
            action_type = "explore_direction"
        elif isinstance(action, ObjectNavStopAction):
            action_type = "stop"
        elif isinstance(action, ObjectNavInvalidAction):
            action_type = "invalid"
        else:
            action_type = "unknown"

        return new_state.reward, terminated, {"action_type": action_type}

    def get_trajectory_reward(self) -> float:
        reward = 0.0
        if self.success:
            reward += 1.0
        if self.initial_distance is not None and self.initial_distance > 0.1:
            final = (
                self.final_distance
                if self.final_distance is not None
                else self.initial_distance
            )
            improvement = (self.initial_distance - final) / self.initial_distance
            reward += 0.3 * max(-1.0, min(1.0, improvement))
        reward -= 0.005 * self.num_steps
        return reward

    def get_ground_truth(self) -> str:
        return json.dumps(
            {
                "trajectory_reward": self.get_trajectory_reward(),
                "success": self.success,
                "num_steps": self.num_steps,
                "initial_distance": self.initial_distance,
                "final_distance": self.final_distance,
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
