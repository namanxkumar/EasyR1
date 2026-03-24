# Credit Assignment Fix: RC vs EasyR1 Comparison & Implementation Plan

## 1. Core Technical Differences (RC vs EasyR1)

### 1.1 Per-Step DataProto Storage

**RC** stores a `DataProto` (tokenized prompt+response tensors) for **every** reasoning and summarization step:

```python
# rc/rollouts.py InferenceProblemState
self.reasoning_rollout_store = []      # DataProto per reasoning step
self.summarization_rollout_store = []  # DataProto per summary step
self.reasoning_string_store = []       # decoded text per step
self.summarization_string_store = []   # decoded text per step

def update_reasoning(self, rollouts: DataProto, response_string: str):
    self.reasoning_rollout_store.append(rollouts)   # ← saves tokenized tensors
    self.reasoning_string_complete_store.append(response_string)
```

Each call to `run_inference()` produces a `DataProto` with `input_ids`, `attention_mask`, `position_ids`, `prompts`, `responses` — this is stored per step.

**EasyR1** only keeps **string** responses and the **last** prompt:

```python
# EasyR1/multiturn_env.py Trajectory
step_responses: list[str]              # strings only, no tokenized data
last_prompt: list[dict] | None = None  # only the final step's prompt
last_images: list[Any] = []            # only the final step's images
_tokenized: dict | None = None         # tokenized only at completion (last step only)
```

The model generates at every step, but only `step_responses[-1]` and `last_prompt` (the final step) are used for training. Earlier prompts/responses are discarded — they're folded into `last_prompt` as summary context.

### 1.2 Training Sample Selection

**RC** uses **snapshot sampling** — it samples K intermediate states from the full multi-turn trajectory and generates *new* rollouts from those snapshots:

1. Run full N-step online rollout, storing all per-step DataProtos
2. `postprocess_online_rollout_states()`: randomly sample K step indices from each trajectory
3. `create_snapshot_state()`: create a new `InferenceProblemState` initialized to the summary/reasoning at that step
4. Generate fresh reasoning/summary from each snapshot (new rollout from intermediate state)
5. Score the fresh rollout via reward function
6. Train on the fresh rollout's (prompt, response)

This means RC trains on *re-sampled* responses from intermediate states, not the original trajectory responses. The key insight: by sampling different starting points, the model learns to reason well from any point in a multi-turn chain.

**EasyR1** uses only the **final step's** (prompt, response):

1. Run full trajectory in AI2Thor
2. `_build_final_batch()`: take `last_prompt` and `step_responses[-1]`
3. Train on that single pair with the trajectory reward

### 1.3 Reward & Advantage Assignment

**RC**: Rewards are computed on the *final output* of each fresh rollout from a snapshot. GRPO groups by `prompt_id` (which encodes the snapshot origin). Each sample from the same snapshot competes within its group. The advantage is per-response (uniform across tokens).

**EasyR1**: Trajectory-level reward (success + format + validity - step_penalty) placed at the last response token. GRPO groups by `uid` (all N trajectories of the same episode). Same structure — but only one (prompt, response) per trajectory.

### 1.4 Loss Aggregation

**RC** has `step-mean-token-mean` loss mode (core_algos.py:27-35) that normalizes loss by step count — important when different steps contribute different numbers of tokens to a batch. This prevents long trajectories from dominating.

**EasyR1** uses standard `token-mean` — adequate when there's one response per trajectory, but needs adjustment when multiple steps per trajectory are included.

### 1.5 Memory Management

**RC** (text-only LLM):
- Keeps `DataProto` objects (tensors) in memory for all steps during rollout
- `ReplayBuffer` saves summaries to JSON file, loads on init — used to bootstrap new rollouts from known-good intermediate states
- No image handling
- Memory proportional to: `num_problems × num_steps × (prompt_len + response_len) × dtype_size`

**EasyR1** (vision-language model):
- Eager tokenization on completion: PIL images → pixel_values tensors → saved to `image_cache_dir` on disk → PIL images freed
- `last_prompt` and `last_images` freed after tokenization
- Multi-modal data lazily loaded during FSDP training
- Only stores last step's data, so memory is bounded

**Key tension**: Storing DataProtos for all steps in EasyR1 would be much heavier due to images. Each step has a full observation image (448×448 or larger) plus prior observation history.

---

## 2. What Needs to Change (Credit Assignment Fix)

### The Goal

Train on **all** (prompt_i, response_i) pairs from each trajectory, so earlier navigation decisions receive gradient signal.

### 2.1 Trajectory Data: Store All Steps

**Current**: `Trajectory` stores `step_responses: list[str]`, `last_prompt`, `last_images` (final step only).

**Needed**: Store per-step `(prompt, images, response)` triples. Two approaches:

**Option A — Eager tokenize each step on completion (RC-style, adapted for VLM)**:
- After each env step, tokenize the prompt+response into a DataProto
- Save multi-modal data to `image_cache_dir` immediately
- Free PIL images
- Store a list of tokenized dicts, one per step

**Option B — Store lightweight references, tokenize at batch-build time**:
- Store per-step `(prompt_text: list[dict], image_paths: list[str], response_text: str)`
- At trajectory completion, images are already saved to disk by AI2Thor (we already have paths)
- Tokenize all steps in `_build_final_batch()`

**Recommendation: Option B** — it's simpler, uses less GPU memory during rollout (no tokenization until needed), and aligns with EasyR1's existing pattern of saving images to disk. The tokenizer is fast relative to env stepping + vLLM generation.

### 2.2 Batch Construction: Expand to All Steps

**Current** `_build_final_batch()`: 1 (prompt, response) per trajectory → `batch_size = num_trajectories`.

**New**: Each trajectory contributes `num_steps` (prompt, response) pairs → `batch_size = sum(trajectory.num_steps for all trajectories)`.

Changes needed:
- Iterate over all steps of each trajectory, tokenizing each step's prompt/images and response
- Left-pad prompts to the max prompt length across all steps (earlier steps have shorter prompts — fewer prior images/summaries)
- Right-pad responses to max response length
- Track which trajectory each step belongs to (for GRPO grouping)
- Track step index within trajectory (for potential reward shaping and loss normalization)

### 2.3 GRPO Grouping Strategy

Two options for how to group steps for advantage normalization:

**Option A — Group by trajectory (all steps from same trajectory get same advantage)**:
- Same UID for all steps of the same trajectory
- Advantage computed from trajectory-level reward, normalized across N trajectories of same episode
- Simple, preserves current GRPO semantics
- Problem: all steps get identical gradient signal regardless of when they occurred

**Option B — Group by (episode, step_index) across trajectories**:
- Step i from trajectory A competes with step i from trajectory B (same episode)
- More fine-grained credit assignment
- Problem: trajectories have different lengths; step alignment is imperfect

**Recommendation: Option A** for initial implementation — it's the minimal change that enables gradient flow to all steps. The primary win is just *having* gradients at all for earlier steps, not fine-grained per-step credit. Option B can be explored later.

### 2.4 Reward Assignment Across Steps

Options:
1. **Uniform**: Same trajectory reward for all steps (simplest, RC does this effectively)
2. **Discounted**: Later steps get higher reward (closer to outcome), `r_t = gamma^(T-t) * R`
3. **Step-shaped**: Intermediate rewards from environment (distance reduction, new rooms explored)

**Recommendation: Start with uniform (option 1)**, matching RC. The GRPO normalization already handles the relative comparison. Later, step-shaped rewards can be added as the environment already computes `distance_to_target_before/after` per step.

### 2.5 Loss Aggregation

Add RC's `step-mean-token-mean` mode to EasyR1's `core_algos.py`. This normalizes so trajectories with more steps don't dominate the loss. Requires passing a `grouping` tensor that maps each batch row to its trajectory ID.

### 2.6 Memory Management: File-Based Trajectory Caching

**Problem**: With all steps stored, memory usage scales with `num_trajectories × avg_steps × (prompt_tokens + image_pixels)`. For 64 trajectories × 10 steps × ~2K prompt tokens + image = substantial.

**Solution**: File-based trajectory caching.

**During rollout** (in `_run_continuous_episode_loop`):
- Each completed trajectory is serialized to disk immediately after reward collection
- Format: one file per trajectory, containing list of `{prompt_text, image_paths, response_text}`
- Images are already on disk (AI2Thor saves frames); just store paths
- Free the in-memory trajectory data after writing

**During batch building** (`_build_final_batch`):
- Load trajectory files from disk
- Tokenize all steps
- Save multi-modal tensors to `image_cache_dir` (existing pattern)
- Build DataProto as usual

**File format** (JSONL per trajectory):
```jsonl
{"step": 0, "prompt": [...], "image_paths": ["path/to/obs0.jpg"], "response": "..."}
{"step": 1, "prompt": [...], "image_paths": ["path/to/obs0.jpg", "path/to/obs1.jpg"], "response": "..."}
```

This keeps peak memory bounded: only one trajectory's tokenized data in memory at a time during batch building, and images are always on disk.

---

## 3. Implementation Plan

### Phase 1: Trajectory Data Storage (multiturn_env.py)

**File**: `verl/workers/rollout/multiturn_env.py`

1. Add `StepRecord` dataclass:
   ```python
   @dataclass
   class StepRecord:
       prompt: list[dict]       # chat-format messages
       image_paths: list[str]   # paths to observation images on disk
       response: str            # model's response text
       format_score: float      # per-step format score (from adapter)
       validity_score: float    # per-step validity score (from adapter)
   ```

2. Add `step_records: list[StepRecord]` to `Trajectory` dataclass. Remove `last_prompt`, `last_images` (subsumed).

3. In `_run_continuous_episode_loop`, after each generation + env step:
   - Save current observation image to a temp file (or use existing AI2Thor frame path)
   - Create `StepRecord` with the prompt that was used for generation, image paths, and decoded response
   - Append to `trajectory.step_records`

4. In `_harvest_and_refill`, after collecting reward:
   - Serialize `trajectory.step_records` to a JSONL file in a `trajectory_cache_dir`
   - Free `step_records` list from memory
   - Store only the file path on the trajectory object

5. Update `ObjectNavEnvAdapter` to expose image paths (it already manages AI2Thor state history which contains frame data).

### Phase 2: Multi-Step Batch Building (multiturn_tokenizer.py)

**File**: `verl/workers/rollout/multiturn_tokenizer.py`

1. New method `_build_multistep_batch()` (alongside existing `_build_final_batch` which becomes legacy):
   - Load each trajectory's step records from disk
   - For each step: tokenize prompt+images via `_tokenize_prompts()`, tokenize response
   - Concatenate prompt+response, build position_ids
   - Track `trajectory_id` (for GRPO grouping) and `step_index` (for loss normalization)
   - Assign reward: uniform trajectory reward to all steps
   - Place reward at last response token of each step's response
   - Build DataProto with additional metadata: `trajectory_id`, `step_index`, `num_steps`

2. Key consideration: earlier steps have shorter prompts (fewer prior images). Left-padding handles this naturally, but we should track token counts for logging.

3. The `uid` field: all steps from all N trajectories of the same episode get the same UID (for GRPO grouping). This means GRPO normalizes across `N × avg_steps` samples per group. Alternatively, use per-trajectory UIDs so normalization is across steps within competing trajectories only.

### Phase 3: Advantage & Loss (core_algos.py, ray_trainer.py)

**File**: `verl/trainer/core_algos.py`

1. Add `step-mean-token-mean` loss aggregation mode (port from RC):
   ```python
   elif loss_agg_mode == "step-mean-token-mean":
       # Normalize by number of steps per trajectory
       masked_loss = loss_mat * loss_mask
       ids_1d = grouping.view(-1)
       id_counts = torch.bincount(ids_1d)
       normalization_factors = id_counts[ids_1d].unsqueeze(1)
       step_normalized_loss = masked_loss / (normalization_factors.float() + 1e-8)
       loss = step_normalized_loss.sum() / (loss_mask.sum() + 1e-8)
   ```

2. Pass `trajectory_id` tensor through the pipeline as `grouping` for loss aggregation.

**File**: `verl/trainer/ray_trainer.py`

3. Thread `grouping` tensor through `apply_kl_penalty` → `compute_advantage` → `update_actor`.

### Phase 4: Config & Integration

**File**: `verl/workers/rollout/config.py`

1. Add to `MultiturnEnvConfig`:
   - `train_all_steps: bool = True` — enable multi-step training (False = legacy last-step-only)
   - `trajectory_cache_dir: str = "/tmp/trajectory_cache"` — where to write per-trajectory JSONL
   - `step_reward_mode: str = "uniform"` — how to distribute reward across steps (uniform / discounted / shaped)
   - `loss_agg_mode: str = "step-mean-token-mean"` — loss aggregation when training all steps

**File**: `verl/workers/rollout/multiturn_env.py`

2. In `generate_trajectories()`: call `_build_multistep_batch()` when `train_all_steps=True`, else fall back to `_build_final_batch()`.

### Phase 5: Validation & Testing

1. **Batch shape sanity check**: With `rollout_batch_size=8`, `n=4`, `max_depth=10`: old batch_size = 32, new batch_size = ~320 (32 × avg ~10 steps). Verify this fits in GPU memory with current `micro_batch_size_per_device_for_update`.

2. **GRPO grouping correctness**: Verify that `compute_grpo_outcome_advantage` produces sensible advantages when group size = N × num_steps (e.g., 4 × 10 = 40 per group).

3. **Gradient flow**: Run a short training and check that loss decreases; log per-step-index loss to verify all steps receive gradient.

4. **Memory profiling**: Monitor peak memory during rollout and batch building. The file-based caching should keep rollout memory similar to current.

5. **Micro-batch sizing**: The training batch is now ~10x larger. May need to reduce `micro_batch_size_per_device_for_update` or increase gradient accumulation. Each step's prompt is shorter on average (fewer prior images), so per-sample memory is lower, partially offsetting the batch size increase.

---

## 4. Risk & Considerations

| Risk | Mitigation |
|------|------------|
| 10x larger effective batch size → OOM during training | Reduce micro_batch_size, increase grad accumulation |
| Variable-length trajectories → padding waste | Group similar-length trajectories in micro-batches |
| GRPO with large groups (N×steps) → noisy advantage | Start with per-trajectory UID (not per-episode) |
| File I/O overhead from trajectory caching | Use /scratch (fast SSD), async writes |
| Earlier steps have trivially short prompts → wasted compute | min_prompt_length filter, or weight by step position |
| Reward dilution (same reward for explore-step and answer-step) | Future: shaped rewards using distance_to_target |

## 5. Sequencing

1. **Phase 1** (Trajectory storage) — standalone, can test by logging step records
2. **Phase 2** (Batch building) — depends on Phase 1, core of the fix
3. **Phase 3** (Loss/advantage) — depends on Phase 2, needed for correct training
4. **Phase 4** (Config) — parallel with Phase 2-3
5. **Phase 5** (Validation) — after all phases, before real training runs
