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
        """Construct the DataProto that the rest of the GRPO pipeline expects.

        For each trajectory, takes the last step's prompt (full history up to
        the final observation) and the model's last response. Tokenizes both
        and assembles into the standard format.

        Trajectories may have been eagerly tokenized (``_tokenized`` is set)
        to free heavy PIL images early. If so, the cached tensors are used
        directly instead of re-tokenizing.
        """
        # ── Separate eagerly-tokenized vs not-yet-tokenized trajectories ──
        need_tokenization = [t for t in trajectories if t._tokenized is None]

        if need_tokenization:
            prompt_texts = [
                t.last_prompt or [{"role": "user", "content": ""}]
                for t in need_tokenization
            ]
            images = [t.last_images or [] for t in need_tokenization]
            tok = self._tokenize_prompts(prompt_texts, images)
            image_cache_dir = getattr(self, "image_cache_dir", None)
            for i, t in enumerate(need_tokenization):
                multi_modal_data = tok["multi_modal_data"][i]
                if image_cache_dir and multi_modal_data:
                    from ...utils.image_cache import save_multi_modal_data
                    multi_modal_data = save_multi_modal_data(
                        multi_modal_data, image_cache_dir
                    )
                t._tokenized = {
                    "input_ids": tok["input_ids"][i].clone(),
                    "attention_mask": tok["attention_mask"][i].clone(),
                    "position_ids": tok["position_ids"][i].clone(),
                    "multi_modal_data": multi_modal_data,
                    "response_text": (
                        t.step_responses[-1] if t.step_responses else ""
                    ),
                }
                # Free heavy data now that it's tokenized
                t.last_prompt = None
                t.last_images = []
                t.step_responses = []
            del prompt_texts, images, tok

        # ── Assemble from per-trajectory tokenized data ──
        prompt_ids = torch.stack([t._tokenized["input_ids"] for t in trajectories])
        prompt_mask = torch.stack([t._tokenized["attention_mask"] for t in trajectories])
        prompt_pos = torch.stack([t._tokenized["position_ids"] for t in trajectories])
        response_texts = [t._tokenized["response_text"] for t in trajectories]
        batch_multi_modal_data = np.array(
            [t._tokenized["multi_modal_data"] for t in trajectories], dtype=object
        )

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
            "multi_modal_data": batch_multi_modal_data,
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
