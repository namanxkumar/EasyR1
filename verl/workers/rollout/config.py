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
    past_k_steps: Optional[int] = None
    """Past-K observation truncation for multi_turn rollout (packed §3b design).

    When set to a positive integer K, each rollout step S sees only system + turn 0
    (task description) + the last K observation turns; older obs and their <think>+
    action tokens are hidden. Training packs the full trajectory but applies a
    structured FlexAttention block mask + custom MROPE positions so that each
    assistant turn's training view matches its rollout view exactly.

    None or <= 0 disables truncation (current full-history behavior). Only takes
    effect when context_mode='multi_turn'."""
    reward_mode: str = "continuous"
    """Trajectory reward formulation.

    'continuous': validity_gate * (0.5 * progress + success_bonus) - 0.005 * num_steps,
        where progress = (initial_distance - final_distance) / initial_distance.
        Gives graded credit on failed trajectories — avoids zero-advantage groups
        when success rate is high.
    'bimodal' (original 7588989 baseline): success_bonus + 0.1 * avg_format
        + 0.15 * avg_validity - 0.005 * num_steps. Reward concentrates on
        success/no-success binary; format/validity are near-ceiling on the SFT
        ckpt so this is effectively bimodal {~0.21, ~1.22}.
    """
    context_mode: str = "multi_turn"
    """Prompt context mode: 'multi_turn' uses proper user/assistant turn
    alternation with full reasoning history preserved, 'single_turn' packs
    all history into one user message (memory-based compression)."""
    num_simulators: int = 8
    """Number of AI2Thor simulator slots per SimulatorPool.
    Should be >= rollout_batch_size * n for full parallelism.
    Each slot creates one AI2Thor Controller (~300MB GPU memory)."""
    difficulties: Optional[list[int]] = None
    """rooms_seen levels to include (None = use all episodes)."""
    max_per_difficulty: Optional[int] = None
    """Cap the number of episodes per rooms_seen bucket."""
    override_indices: Optional[list[int]] = None
    """Explicit dataset indices to use (bypasses difficulty filtering and max_items)."""
    force_max_depth: bool = False
    """Force all trajectories to run to max_depth, ignoring early termination
    from the environment (e.g. answer actions). Useful for stress-testing
    worst-case memory/context length in multi-turn mode."""

    # ── Validation fields ──────────────────────────────────────────
    val_split: str = "val"
    """Dataset split for validation episodes."""
    val_max_items: Optional[int] = None
    """Maximum number of validation episodes (None = use all)."""
    val_difficulties: Optional[list[int]] = None
    """rooms_seen levels for validation (None = same as training ``difficulties``)."""
    val_max_per_difficulty: Optional[int] = None
    """Cap episodes per rooms_seen bucket for validation."""
    val_override_indices: Optional[list[int]] = None
    """Explicit validation episode indices (bypasses val_difficulties/val_max_items)."""
    val_batch_size: int = 16
    """Number of episodes per validation rollout."""
    val_n: int = 1
    """Rollouts per validation episode (typically 1 for deterministic eval)."""


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
    stop: list[str] = field(default_factory=list)
    """Stop strings passed to vLLM SamplingParams. Generation halts as soon as
    any of these substrings appears in the decoded output (used for multiturn
    ObjectNav: ``["</answer>", "</explore>"]`` cuts trailing tokens after the
    action tag closes). Empty list = no stop strings."""
    parity_log_probs: bool = False
    """Capture per-response-token logprobs from vLLM during rollout.

    When True, vLLM is asked to emit ``logprobs=1`` so the actor can compare
    rollout-time logprobs against FSDP-side ``compute_log_probs`` for every
    sampled token. Adds a `rollout_log_probs` (B, response_length) tensor to
    the generate_sequences output. Off by default — used only for the
    parity smoke test."""
    use_rollout_log_probs: bool = False
    """Skip the actor's ``compute_log_probs`` recompute and reuse vLLM's
    rollout-time logprobs as ``old_log_probs`` for the PPO ratio.

    Eliminates the entire log-prob recompute phase (typically half of the
    actor's per-step cost). Requires ``parity_log_probs=True`` so the
    rollout actually emits the tensor. Recommended only when KL is disabled
    (DAPO-style) since the ref policy still needs an FSDP forward; pairing
    with KL gives smaller savings. Accepts ~1e-3 numerical drift between
    vLLM (FP16) and FSDP (BF16); ViGoRL operates in this regime."""
    # below are auto keys
    prompt_length: int = field(default=-1, init=False)
    response_length: int = field(default=-1, init=False)
    trust_remote_code: bool = field(default=False, init=False)

    def to_dict(self):
        return asdict(self)
