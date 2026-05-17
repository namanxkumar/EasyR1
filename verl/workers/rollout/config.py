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
    'success': pure binary {0.0, 1.0} on trajectory success — no format
        weighting, no progress shaping, no step penalty.
    """
    context_mode: str = "multi_turn"
    """Prompt context mode: 'multi_turn' uses proper user/assistant turn
    alternation with full reasoning history preserved, 'single_turn' packs
    all history into one user message (memory-based compression)."""
    num_simulators: int = 8
    """Total AI2Thor simulator slots across all pools and GPUs.
    Should be >= rollout_batch_size * n for full parallelism.
    Each slot creates one AI2Thor Controller (~300MB GPU memory)."""
    pools_per_gpu: int = 1
    """Number of SimulatorPool Ray actors to create per GPU.

    Each SimulatorPool is `@ray.remote(num_cpus=1)` with default Ray
    concurrency of 1, meaning `acquire_env` (Unity scene reset, ~9-11s) is
    serialized within a single pool. Splitting `num_simulators` across N
    pools per GPU lets N acquires proceed concurrently on the same GPU,
    linearly raising aggregate acquire throughput.

    Total pools = pools_per_gpu * n_gpus_per_node. Slots per pool =
    num_simulators // (pools_per_gpu * n_gpus_per_node). Tune up when
    rollout is acquire-bound (queue depth visible in the `acquire_env (N):
    Xs` log lines exceeds 1-2 trajs)."""
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
class GuidedRolloutConfig:
    """Pope-Dagger V4 guided rollout knobs.

    When ``enabled=True``, ``GuidedMultiturnEnvRollout`` replaces the standard
    ``MultiturnEnvRollout`` driver. The trainer also flips ``algorithm.adv_estimator``
    to ``grpo_branching`` automatically (so per-step pairwise-cohort advantages
    are used instead of group-wide GRPO).
    """

    enabled: bool = False
    """Master switch — when False, behaves exactly like baseline DAPO."""

    max_iters: int = 3
    """Maximum V4 chain depth (iters 1..max_iters after the on-policy baseline).
    Effective cap is min(max_iters, max_depth - 1, n_per_prompt - 1)."""

    n_per_prompt: int = 4
    """Target group size: each uid contributes exactly this many trajectories
    (baseline + chain + on-policy padding). Should match ``rollout.n``."""

    stop_on_solved: bool = True
    """Stop extending a chain once the parent trajectory succeeded (reward >= 1)."""

    branch_selection_mode: str = "random_regression"
    """Branch-step selection strategy.
    - 'fixed_step_k': iter k forces step (k-1). Deterministic ladder; produces
      a strict nested-prefix V4 chain.
    - 'random_regression': pope-dagger V5 selector. Each iter samples a
      branch step uniformly from [suffix_peak..last_unrecovered] of the
      un-forced suffix (steps after the latest forced window). Branch chain
      is still monotonically increasing (V5 scopes by scan_start)."""

    random_regression_seed: int | None = None
    """Optional seed for the V5 selector RNG. Mixed with (group_id, iter_k) so
    each (prompt, iter) gets a deterministic but distinct sample. None →
    fresh RNG (stochastic across runs)."""

    # ── Teacher VLM backend ──────────────────────────────────────────────
    teacher_mode: str = "server"
    """'server' (always-on dedicated vLLM) or 'colocated' (in-process with
    sleep mode). v1 supports 'server' only; 'colocated' raises at construction."""

    teacher_vlm_node: Optional[str] = None
    """Hostname of the teacher vLLM server (server mode)."""

    teacher_vlm_port: Optional[str] = None
    """Port of the teacher vLLM server (server mode)."""

    teacher_job_name: Optional[str] = None
    """SLURM job name for squeue-based discovery (server mode, fallback when
    explicit node/port + env vars are unset)."""

    teacher_vlm_type: str = "qwen"
    """VLM family ('qwen' or 'cosmos') for tokenizer/prompt formatting."""

    teacher_past_context_steps: int = 8
    """Past observation horizon forwarded to ``annotate_expert_cache``."""

    teacher_future_context_steps: int = 0
    """Future observation horizon forwarded to ``annotate_expert_cache``."""

    teacher_max_workers: int = 16
    """Thread-pool fan-out width for batched teacher annotation."""


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
    async_mode: bool = False
    """Drive the in-process vLLM engine via asyncio so the multiturn rollout
    can submit one request per trajectory and let vLLM's continuous batching
    overlap GPU work with environment steps.

    When True, vLLMRollout exposes ``async generate_one(...)`` and
    ``async wait_idle()`` in addition to the existing batched
    ``generate_sequences``. The multiturn driver (``MultiturnEnvRollout``)
    runs each trajectory as its own coroutine, calling ``generate_one`` as
    soon as the env step returns. The drain barrier (``wait_idle``) fires
    once per PPO step before FSDP↔vLLM weight sync.

    Constraints:
    - TP>1 supported via lockstep stepping in
      ``AsyncRequestRouter._step_loop_tp``: rank 0 broadcasts the per-step
      admit list over the vLLM TP NCCL group, slaves wait for matching
      payloads (driver fans the request out to every TP rank of the chosen
      engine), and all ranks call ``engine.step()`` with identical scheduler
      state so the forward-pass TP collectives line up.
    - Existing direct-memory weight sync at ``fsdp_vllm.py`` requires the
      engine to live in the same Python process. Async is intentionally
      built around the sync ``LLMEngine`` (driven from asyncio) instead of
      ``AsyncLLM`` (which forks the engine and breaks the weight sync)."""
    # below are auto keys
    prompt_length: int = field(default=-1, init=False)
    response_length: int = field(default=-1, init=False)
    trust_remote_code: bool = field(default=False, init=False)

    def to_dict(self):
        return asdict(self)
