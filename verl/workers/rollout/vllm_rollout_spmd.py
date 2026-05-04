# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from vllm import LLM, RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.dataset import process_image, process_video
from ...utils.torch_dtypes import PrecisionType
from ...utils.vllm_utils import VLLMHijack
from ._packed_mrope_hook import overrides as packed_mrope_overrides_ctx
from ._packed_mrope_hook import unlink_override as packed_mrope_unlink_override
from ._packed_mrope_hook import write_override as packed_mrope_write_override
from .base import BaseRollout
from .config import RolloutConfig


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, np.ndarray]:
    # repeat the elements, supports both tensor and numpy array
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def _get_logit_bias(processor: Optional[ProcessorMixin]) -> Optional[dict[int, float]]:
    # enforce vllm to not output image token
    # TODO: add video token
    if processor is not None and hasattr(processor, "image_token"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        return {image_token_id: -100}
    else:
        return None


def _process_multi_modal_data(
    multi_modal_data: dict[str, Any], min_pixels: int, max_pixels: int, video_fps: float
) -> dict[str, Any]:
    # may convert image path to image object
    images, videos = [], []
    if "images" in multi_modal_data:
        for image in multi_modal_data["images"]:
            images.append(process_image(image, min_pixels, max_pixels))

    if "videos" in multi_modal_data:
        for video in multi_modal_data["videos"]:
            videos.append(process_video(video, min_pixels, max_pixels, video_fps))

    if len(images) != 0:
        return {"image": images}

    if len(videos) != 0:
        return {"video": videos}

    return None


@dataclass
class AsyncGenerateResult:
    """One trajectory's worth of generation output from `AsyncRequestRouter`.

    Mirrors the slice of a `vllm.RequestOutput` that the multiturn driver
    actually consumes — token IDs, optional per-token sampled logprobs, and
    the finish reason — so callers don't have to import vLLM types."""

    token_ids: list[int]
    logprobs: Optional[list[float]]
    finish_reason: Optional[str]


class AsyncRequestRouter:
    """Asyncio driver around an in-process vLLM `LLMEngine`.

    The standard `LLM.generate(prompts=[...])` path bundles a fixed batch and
    blocks until every request finishes (`vllm/entrypoints/llm.py:_run_engine`).
    For per-trajectory async multiturn we want the opposite: one trajectory
    submits a single request, awaits its completion, then steps the env. This
    router exposes that shape by:

    1. submitting requests with `engine.add_request(...)` from any coroutine,
    2. running `engine.step()` from a single background asyncio task that
       collects finished outputs and resolves per-request futures,
    3. preserving every existing engine integration — same external_launcher
       backend, same multi-modal processing, same packed-MROPE override
       lifecycle, same in-process weight sync — because the engine itself
       never moves out of the rollout-worker process.

    TP=1 only — see `RolloutConfig.async_mode` docstring for why."""

    def __init__(self, llm_engine, sleep_quantum: float = 0.0):
        self.engine = llm_engine
        self._sleep_quantum = sleep_quantum
        self._pending: dict[str, asyncio.Future] = {}
        # Tracks packed-MROPE override files written for in-flight requests so
        # we can unlink them only after the engine has actually consumed the
        # prefill — the override file lookup happens inside `engine.step()`,
        # so unlinking on `add_request` return would race the prefill.
        self._packed_keys: dict[str, list[int]] = {}
        self._step_task: Optional[asyncio.Task] = None
        self._stopped = False

    def start(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Idempotently start the background step loop on the given event loop."""
        if self._step_task is not None and not self._step_task.done():
            return
        loop = loop or asyncio.get_event_loop()
        self._stopped = False
        self._step_task = loop.create_task(self._step_loop())

    async def stop(self) -> None:
        self._stopped = True
        if self._step_task is not None:
            await asyncio.shield(asyncio.wait_for(self._step_task, timeout=5.0))
            self._step_task = None

    async def _step_loop(self) -> None:
        try:
            while not self._stopped:
                if self.engine.has_unfinished_requests():
                    outputs = self.engine.step()
                    for out in outputs:
                        if not getattr(out, "finished", False):
                            continue
                        fut = self._pending.pop(out.request_id, None)
                        # Unlink the packed-MROPE override now that prefill +
                        # decode are both complete.
                        kept = self._packed_keys.pop(out.request_id, None)
                        if kept is not None:
                            try:
                                packed_mrope_unlink_override(kept)
                            except Exception:
                                pass
                        if fut is not None and not fut.done():
                            fut.set_result(out)
                # Yield even when there is nothing to do so the event loop
                # gets a chance to process Ray futures from env steps.
                if self._sleep_quantum > 0:
                    await asyncio.sleep(self._sleep_quantum)
                else:
                    await asyncio.sleep(0)
        except Exception as e:
            # Fail any pending futures so coroutines don't hang forever.
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(e)
            self._pending.clear()
            raise

    async def generate_one(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        multi_modal_data: Optional[dict[str, Any]] = None,
        lora_request: Optional[LoRARequest] = None,
        packed_mrope_override: Optional[tuple[list[int], torch.Tensor, Optional[int]]] = None,
    ) -> AsyncGenerateResult:
        """Submit a single request and await its final output.

        `packed_mrope_override` is `(kept_input_ids, positions, delta)` where
        `kept_input_ids` is the processor-expanded token sequence used as the
        SHA-1 key. `positions` is `(3, L)` int64. `delta` may be `None` (auto)."""
        loop = asyncio.get_running_loop()
        if self._step_task is None or self._step_task.done():
            self.start(loop)

        # Build the vLLM prompt dict — same shape `LLM.generate` would build.
        prompt: dict[str, Any] = {"prompt_token_ids": list(prompt_token_ids)}
        if multi_modal_data is not None:
            prompt["multi_modal_data"] = multi_modal_data

        # Write the packed-MROPE override file *before* `add_request`, since
        # prefill begins as soon as the request enters the scheduler.
        if packed_mrope_override is not None:
            kept, positions, delta = packed_mrope_override
            packed_mrope_write_override(kept, positions, delta)
            self._packed_keys[request_id] = list(kept)

        fut: asyncio.Future = loop.create_future()
        self._pending[request_id] = fut
        try:
            self.engine.add_request(
                request_id=request_id,
                prompt=prompt,
                params=sampling_params,
                lora_request=lora_request,
            )
        except Exception:
            self._pending.pop(request_id, None)
            kept = self._packed_keys.pop(request_id, None)
            if kept is not None:
                try:
                    packed_mrope_unlink_override(kept)
                except Exception:
                    pass
            raise

        request_output: RequestOutput = await fut

        out0 = request_output.outputs[0]
        token_ids = list(out0.token_ids)
        logprobs: Optional[list[float]] = None
        if out0.logprobs is not None:
            lps: list[float] = []
            for tok_id, lp_dict in zip(out0.token_ids, out0.logprobs):
                lp = lp_dict.get(tok_id) if lp_dict is not None else None
                lps.append(float(lp.logprob) if lp is not None else float("nan"))
            logprobs = lps
        return AsyncGenerateResult(
            token_ids=token_ids,
            logprobs=logprobs,
            finish_reason=out0.finish_reason,
        )

    async def wait_idle(self, poll_interval: float = 0.005) -> None:
        """Block until the engine has no in-flight requests AND no pending futures."""
        while self.engine.has_unfinished_requests() or self._pending:
            await asyncio.sleep(poll_interval)


class vLLMRollout(BaseRollout):
    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        **kwargs,
    ):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
        """
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.pad_token_id = tokenizer.pad_token_id
        self.use_tqdm = (self.rank == 0) and (not config.disable_tqdm)
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs

        engine_kwargs = {}
        if processor is not None:  # only VLMs have processor
            # disable_mm_preprocessor_cache was removed in vLLM >= 0.16
            try:
                from vllm.engine.arg_utils import EngineArgs
                if "disable_mm_preprocessor_cache" in EngineArgs.__dataclass_fields__:
                    engine_kwargs["disable_mm_preprocessor_cache"] = True
            except Exception:
                pass
            if config.limit_images:
                engine_kwargs["limit_mm_per_prompt"] = {"image": config.limit_images}

        VLLMHijack.hijack()

        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            trust_remote_code=config.trust_remote_code,
            load_format="dummy" if not self.lora_kwargs else "safetensors",
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            seed=config.seed,
            max_model_len=config.max_model_len or config.prompt_length + config.response_length,
            distributed_executor_backend="external_launcher",
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_batched_tokens=config.max_num_batched_tokens,
            disable_log_stats=config.disable_log_stats,
            enforce_eager=config.enforce_eager,
            disable_custom_all_reduce=True,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_sleep_mode=True,
            **lora_kwargs,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        try:
            self.inference_engine.sleep(level=2)
        except Exception as e:
            print(f"WARNING: vLLM sleep(level=2) failed: {e}")
            print("Continuing without sleep mode — this may increase peak memory usage.")

        # Stop strings require vLLM's incremental detokenizer to compare against
        # the decoded text — flip detokenize on when any are configured.
        has_stop_strings = bool(getattr(config, "stop", None))
        sampling_kwargs = {
            "max_tokens": config.response_length,
            "detokenize": has_stop_strings,
            "logit_bias": _get_logit_bias(processor),
        }
        if has_stop_strings:
            sampling_kwargs["include_stop_str_in_output"] = True
        if getattr(config, "parity_log_probs", False):
            # logprobs=1 returns the sampled token's logprob per step.
            # Setting 0 caused vLLM to leave the position-0 dict empty,
            # which silently fell back to 0.0 and corrupted the first
            # response token of every step in the parity comparison.
            sampling_kwargs["logprobs"] = 1
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)

        # Async router lazily initialized on first async call. We don't want
        # to spin up an asyncio event loop until the multiturn driver actually
        # asks for it — keeps the sync codepath untouched.
        self._async_router: Optional[AsyncRequestRouter] = None
        if getattr(config, "async_mode", False):
            if config.tensor_parallel_size != 1:
                raise ValueError(
                    "rollout.async_mode=true currently requires tensor_parallel_size=1. "
                    "TP>1 needs the per-rank async path to broadcast each request "
                    "across the TP group, which is not yet implemented."
                )

    def _get_async_router(self) -> AsyncRequestRouter:
        """Return (creating if needed) the asyncio driver for the in-process engine."""
        if self._async_router is None:
            self._async_router = AsyncRequestRouter(self.inference_engine.llm_engine)
        return self._async_router

    async def generate_one(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        multi_modal_data: Optional[dict[str, Any]] = None,
        sampling_overrides: Optional[dict[str, Any]] = None,
        packed_mrope_override: Optional[tuple[list[int], torch.Tensor, Optional[int]]] = None,
        lora_request: Optional[LoRARequest] = None,
        min_pixels: int = 0,
        max_pixels: int = 0,
        video_fps: float = 2.0,
    ) -> AsyncGenerateResult:
        """Submit a single async request to the engine and await its output.

        `sampling_overrides` shadow the rollout's default sampling params for
        just this request (mirrors the sync `update_sampling_params` ctx
        manager). `multi_modal_data` accepts the same `{"images": [...]}`
        form the sync path consumes — `_process_multi_modal_data` runs here
        so callers don't have to mirror the sync wrapper."""
        if not getattr(self.config, "async_mode", False):
            raise RuntimeError(
                "vLLMRollout.generate_one called with async_mode=false. Set "
                "rollout.async_mode=true in the config to enable the async path."
            )
        if sampling_overrides:
            base = self.sampling_params.clone()
            for k, v in sampling_overrides.items():
                if hasattr(base, k):
                    setattr(base, k, v)
            sp = base
        else:
            sp = self.sampling_params

        # Run the same preprocessing the sync path does — accept the raw
        # `{"images": [PIL...]}` form and produce vLLM's `{"image": [...]}`.
        processed_mm = None
        if multi_modal_data is not None:
            if "images" in multi_modal_data or "videos" in multi_modal_data:
                processed_mm = _process_multi_modal_data(
                    multi_modal_data, min_pixels, max_pixels, video_fps
                )
            else:
                # Already in vLLM-internal form — pass through.
                processed_mm = multi_modal_data

        return await self._get_async_router().generate_one(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sp,
            multi_modal_data=processed_mm,
            lora_request=lora_request,
            packed_mrope_override=packed_mrope_override,
        )

    async def wait_idle(self) -> None:
        """Block until the async engine has no in-flight requests.

        Called by the rollout/train barrier before FSDP↔vLLM weight sync so
        the in-process weight swap doesn't happen mid-generation."""
        if self._async_router is None:
            return
        await self._async_router.wait_idle()

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)
        # raw_images is a 1-D numpy object array of Python lists of PIL images,
        # stored in non_tensor_batch so it gets properly sharded with the batch.
        batch_raw_images = non_tensor_batch.pop("raw_images", None)
        # Packed-MROPE side channel: per-request (positions, delta) for past-K
        # multiturn. When set, vLLM's Qwen3-VL subclass (registered via
        # _packed_mrope_hook/sitecustomize.py) reads positions from /dev/shm
        # keyed by sha1(prompt_token_ids). Each entry is (positions, delta) or
        # None to fall through to vLLM's default MROPE computation.
        batch_packed_mrope = non_tensor_batch.pop("packed_mrope_overrides", None)
        if batch_size != len(batch_raw_prompt_ids):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if batch_raw_images is not None:
            # Use raw PIL images for vLLM generation (multiturn env path)
            vllm_inputs = []
            for raw_prompt_ids, raw_images in zip(batch_raw_prompt_ids, batch_raw_images):
                mm_data = _process_multi_modal_data(
                    {"images": raw_images},
                    prompts.meta_info["min_pixels"],
                    prompts.meta_info["max_pixels"],
                    prompts.meta_info["video_fps"],
                )
                vllm_inputs.append(
                    {
                        "prompt_token_ids": list(raw_prompt_ids),
                        "multi_modal_data": mm_data,
                    }
                )
        elif batch_multi_modal_data is not None:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(batch_raw_prompt_ids, batch_multi_modal_data):
                vllm_inputs.append(
                    {
                        "prompt_token_ids": list(raw_prompt_ids),
                        "multi_modal_data": _process_multi_modal_data(
                            multi_modal_data,
                            prompts.meta_info["min_pixels"],
                            prompts.meta_info["max_pixels"],
                            prompts.meta_info["video_fps"],
                        ),
                    }
                )
        else:
            vllm_inputs = [{"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in batch_raw_prompt_ids]

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
                ] * batch_size

        # Build packed-MROPE override list (write /dev/shm/<sha1>.bin per
        # request before generate; auto-unlink after).
        packed_requests: list = []
        if batch_packed_mrope is not None:
            for override in batch_packed_mrope:
                if override is None:
                    continue
                # override is (kept_input_ids, positions, delta) — kept_input_ids
                # is the processor-expanded token sequence (one image → many
                # patch tokens). vLLM's MROPE hook hashes the prefill tokens it
                # actually receives, which are the expanded form, so we key on
                # kept_input_ids — not on the unexpanded raw_prompt_ids.
                kept_input_ids, positions, delta = override
                packed_requests.append((tuple(kept_input_ids), positions, delta))

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**prompts.meta_info), \
                packed_mrope_overrides_ctx(packed_requests):
            completions: list[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs,
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=self.use_tqdm,
            )
            response_ids = [output.token_ids for completion in completions for output in completion.outputs]
            response_ids = VF.pad_2d_list_to_length(
                response_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)

            # Capture per-token sampled logprobs when parity mode is enabled.
            rollout_log_probs = None
            if getattr(self.config, "parity_log_probs", False):
                lp_lists: list[list[float]] = []
                for completion in completions:
                    for output in completion.outputs:
                        lp_per_tok: list[float] = []
                        # output.logprobs is a list of dicts {token_id: Logprob}
                        # — one per generated token.
                        if output.logprobs is not None:
                            for tok_id, lp_dict in zip(output.token_ids, output.logprobs):
                                lp = lp_dict.get(tok_id)
                                lp_per_tok.append(float(lp.logprob) if lp is not None else float("nan"))
                        lp_lists.append(lp_per_tok)
                rollout_log_probs = VF.pad_2d_list_to_length(
                    lp_lists, 0.0, max_length=self.config.response_length
                ).to(input_ids.device).to(torch.float32)

            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                if batch_multi_modal_data is not None:
                    batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, self.sampling_params.n)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.ndim == 3:  # qwen2vl mrope: (batch_size, 4, seq_length)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch_dict = {
            "prompts": input_ids,
            "responses": response_ids,
            "input_ids": sequence_ids,  # here input_ids become the whole sentences
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "position_ids": position_ids,
        }
        if rollout_log_probs is not None:
            # Already (B*n, L) from the `completions[*].outputs[*]` flatten — same
            # shape as response_ids. No further repeat_interleave needed.
            batch_dict["rollout_log_probs"] = rollout_log_probs
        batch = TensorDict(batch_dict, batch_size=batch_size)
        if batch_multi_modal_data is not None:
            non_tensor_batch = {"multi_modal_data": batch_multi_modal_data}
        else:
            non_tensor_batch = {}

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)
