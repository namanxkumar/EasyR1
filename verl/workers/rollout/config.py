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
"""
Rollout config
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class MultiturnEnvConfig:
    """Configuration for multi-turn environment rollouts (e.g. AI2Thor ObjectNav).

    When ``enabled=True``, the trainer replaces the standard single-turn
    generate-then-score loop with multi-turn environment trajectories.
    """

    enabled: bool = False
    """Enable multi-turn environment rollout mode."""
    env_type: str = "objectnav"
    """Environment type. Currently only 'objectnav' is supported."""
    data_root: str = "/data/group_data/katefgroup/VLA/poliformer_data"
    """Root directory for the Poliformer dataset."""
    split: str = "train"
    """Dataset split to use."""
    max_items: Optional[int] = None
    """Maximum number of dataset items (None = use all)."""
    system_prompt_path: str = "src/post_annotation/prompts/sft_system_prompt.txt"
    """Path to the system prompt file."""
    max_depth: int = 30
    """Maximum steps per trajectory."""
    gpu_id: int = 0
    """GPU ID for AI2Thor rendering."""
    render_width: int = 616
    """Environment render width (matches SFT post-annotation pipeline)."""
    render_height: int = 616
    """Environment render height (matches SFT post-annotation pipeline)."""
    model_output_scale: int = 1000
    """The coordinate grid scale the model was trained to output (e.g. [0, 1000])."""
    coordinate_normalization_scale: Optional[float] = None
    """Scale factor applied to parsed coordinates.
    Computed as render_width / model_output_scale if not set explicitly."""
    prior_image_scale: float = 0.5
    """Downscale prior observation images (all but the last) by this factor.
    Matches the SFT training config (prior_image_scale in LLaMA-Factory).
    1.0 = no downscaling, 0.5 = half each dimension."""
    max_observations: int = 20
    """Maximum observation images to include in prompt context."""
    num_simulators: int = 8
    """Number of AI2Thor simulator slots per SimulatorPool.
    Should be >= rollout_batch_size * n for full parallelism.
    Each slot creates one AI2Thor Controller (~300MB GPU memory)."""
    difficulties: Optional[list[int]] = None
    """rooms_seen levels to include (None = use all episodes)."""
    max_per_difficulty: Optional[int] = None
    """Cap the number of episodes per rooms_seen bucket."""


@dataclass
class RolloutConfig:
    name: str = "vllm"
    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    seed: int = 1
    limit_images: int = 0
    dtype: str = "bf16"
    gpu_memory_utilization: float = 0.6
    ignore_eos: bool = False
    enforce_eager: bool = False
    enable_chunked_prefill: bool = False  # only for v0 engine
    tensor_parallel_size: int = 2
    max_model_len: Optional[int] = None
    max_num_batched_tokens: int = 8192
    disable_log_stats: bool = True
    disable_tqdm: bool = False
    val_override_config: dict[str, Any] = field(default_factory=dict)
    # below are auto keys
    prompt_length: int = field(default=-1, init=False)
    response_length: int = field(default=-1, init=False)
    trust_remote_code: bool = field(default=False, init=False)

    def to_dict(self):
        return asdict(self)
