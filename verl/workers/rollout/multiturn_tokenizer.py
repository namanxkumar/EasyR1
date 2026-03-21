"""Tokenization and batch-building mixin for MultiturnEnvRollout.

Extracts the heavy ``_tokenize_prompts`` and ``_build_final_batch`` methods
so the main rollout driver stays focused on episode orchestration.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import numpy as np
import torch
from PIL import Image
from tensordict import TensorDict

from ...protocol import DataProto

logger = logging.getLogger(__name__)


class TokenizerMixin:
    """Mixin providing prompt tokenization and final-batch construction.

    Expects the host class to set these attributes:
      - tokenizer: PreTrainedTokenizer
      - processor: ProcessorMixin
      - max_prompt_length: int
      - max_response_length: int
      - min_pixels: int
      - max_pixels: int
      - prior_image_scale: float
    """

    def _tokenize_prompts(
        self,
        prompts: list[list[dict]],
        images_list: list[list[Image.Image]],
        for_generation: bool = False,
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
        return result

    def _build_final_batch(
        self,
        trajectories,
        rewards: list[float],
        ground_truths: list[str],
        n_trajectories: int,
    ) -> DataProto:
        """Construct a DataProto with **all** steps from every trajectory.

        Each (prompt_i, response_i) pair from every trajectory becomes a
        separate training sample.  All steps within a trajectory receive
        the same trajectory-level reward.  GRPO groups all N trajectories
        of the same dataset item under a shared UID.

        A ``trajectory_id`` tensor is included so that ``average_loss``
        with ``mode="traj"`` can normalize each trajectory's contribution
        regardless of its number of steps.
        """
        from .multiturn_env import StepRecord

        # ── Materialise step records (may be on disk or in memory) ──
        all_step_records: list[list[StepRecord]] = []
        for t in trajectories:
            if t._cache_path is not None:
                from .multiturn_env import MultiturnEnvRollout
                records = MultiturnEnvRollout._load_step_records(t._cache_path)
            elif t.step_records:
                records = t.step_records
            else:
                # Trajectory with no step records (shouldn't happen, but be safe)
                records = []
            all_step_records.append(records)

        # ── Flatten: one row per (trajectory, step) ──
        flat_prompts: list[list[dict]] = []
        flat_images: list[list[Image.Image]] = []
        flat_responses: list[str] = []
        flat_rewards: list[float] = []
        flat_ground_truths: list[str] = []
        flat_group_ids: list[int] = []       # dataset item index (for GRPO UID)
        flat_trajectory_ids: list[int] = []  # unique per trajectory (for loss norm)

        for traj_idx, (t, records) in enumerate(zip(trajectories, all_step_records)):
            reward = rewards[traj_idx]
            gt = ground_truths[traj_idx]
            for rec in records:
                flat_prompts.append(rec.prompt)
                # Load images from paths (or empty list)
                imgs = []
                for p in rec.image_paths:
                    try:
                        imgs.append(Image.open(p))
                    except Exception:
                        logger.warning(f"Failed to load step image: {p}")
                flat_images.append(imgs)
                flat_responses.append(rec.response)
                flat_rewards.append(reward)
                flat_ground_truths.append(gt)
                flat_group_ids.append(t.group_id)
                flat_trajectory_ids.append(traj_idx)

        bs = len(flat_prompts)
        if bs == 0:
            raise RuntimeError(
                "No step records found across any trajectory. "
                "This likely means all trajectories had 0 steps."
            )

        logger.info(
            f"Building multi-step batch: {len(trajectories)} trajectories "
            f"expanded to {bs} (prompt, response) training samples"
        )

        # ── Tokenize all step prompts ──
        tok = self._tokenize_prompts(flat_prompts, flat_images)
        prompt_ids = tok["input_ids"]      # (bs, max_prompt_length)
        prompt_mask = tok["attention_mask"]  # (bs, max_prompt_length)
        prompt_pos = tok["position_ids"]    # (bs, 4, max_prompt_length) for mrope

        # Save multi-modal data to disk to free memory
        image_cache_dir = getattr(self, "image_cache_dir", None)
        mmd_array = tok.get("multi_modal_data", None)
        batch_multi_modal_data = []
        for i in range(bs):
            multi_modal_data = mmd_array[i] if mmd_array is not None else None
            if image_cache_dir and multi_modal_data:
                from ...utils.image_cache import save_multi_modal_data
                multi_modal_data = save_multi_modal_data(
                    multi_modal_data, image_cache_dir
                )
            batch_multi_modal_data.append(multi_modal_data)
        batch_multi_modal_data = np.array(batch_multi_modal_data, dtype=object)

        # Free loaded images
        del flat_images, tok

        # ── Tokenize responses ──
        max_resp_len = self.max_response_length
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        batch_resp_ids = []
        batch_resp_mask = []

        for resp_text in flat_responses:
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

        # ── Concatenate prompt + response ──
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

        # ── Build TensorDict ──
        td = TensorDict(
            {
                "prompts": prompt_ids,
                "responses": response_ids_t,
                "input_ids": full_ids,
                "attention_mask": full_mask,
                "response_mask": response_mask_t,
                "position_ids": full_pos,
                "trajectory_id": torch.tensor(flat_trajectory_ids, dtype=torch.long),
            },
            batch_size=bs,
        )

        # ── UIDs: same uid for all N trajectories of the same dataset item ──
        # All steps from all trajectories of the same GRPO group share a UID.
        group_to_uid: dict[int, str] = {}
        uids = []
        for gid in flat_group_ids:
            if gid not in group_to_uid:
                group_to_uid[gid] = str(uuid.uuid4())
            uids.append(group_to_uid[gid])

        non_tensor = {
            "uid": np.array(uids, dtype=object),
            "ground_truth": np.array(flat_ground_truths, dtype=object),
            "multi_modal_data": batch_multi_modal_data,
        }

        # ── Place trajectory reward at last response token of each step ──
        token_level_scores = torch.zeros_like(
            response_ids_t, dtype=torch.float32
        )
        for i, reward in enumerate(flat_rewards):
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
