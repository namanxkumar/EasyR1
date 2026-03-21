# EasyR1 Fork: Multi-Turn ObjectNav GRPO Integration

**Base commit:** `3e4a8799b354469ec4f24d06f9d6cca7a006fd11` (upstream EasyR1/veRL)
**Purpose:** Transform EasyR1 from a single-turn RL trainer into an online multi-turn environment rollout system for training VLMs to navigate 3D spaces (AI2Thor ObjectNav) using GRPO.

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Code Flow: End-to-End Training Step](#2-code-flow-end-to-end-training-step)
3. [New Files](#3-new-files)
4. [Modified Files](#4-modified-files)
5. [Component Deep Dives](#5-component-deep-dives)
6. [Data Flow Diagrams](#6-data-flow-diagrams)

---

## 1. High-Level Architecture

### Original EasyR1 (Single-Turn)

```
┌─────────────────────────────────────────────────────────┐
│                    RayPPOTrainer.fit()                   │
│                                                         │
│  for each batch from dataset:                           │
│    1. Load prompts from parquet/HF dataset              │
│    2. vLLM generates N responses per prompt             │
│    3. Reward function scores each response              │
│    4. GRPO advantage computation                        │
│    5. Actor FSDP update with KL penalty                 │
└─────────────────────────────────────────────────────────┘
```

### Modified EasyR1 (Multi-Turn Online Environment)

```
┌──────────────────────────────────────────────────────────────────┐
│                      RayPPOTrainer.fit()                         │
│                                                                  │
│  for each training step:                                         │
│                                                                  │
│    ┌─────────────────────────────────────────────────────┐       │
│    │         MultiturnEnvRollout.generate_trajectories()  │       │
│    │                                                      │       │
│    │  1. Warmup AI2Thor controllers (reuse cached ones)   │       │
│    │  2. Acquire simulator slots from SimulatorPools      │       │
│    │  3. Reset environments for dataset episodes          │       │
│    │                                                      │       │
│    │  LOOP (up to max_depth steps):                       │       │
│    │    a. Build prompts (Ray parallel across pools)      │       │
│    │    b. Tokenize → DataProto                           │       │
│    │    c. vLLM generate_sequences() → responses          │       │
│    │    d. Step environments (Ray parallel) → rewards      │       │
│    │    e. Harvest terminated → refill from pending queue  │       │
│    │                                                      │       │
│    │  4. Collect trajectory-level rewards                  │       │
│    │  5. Destroy controllers (free GPU for training)       │       │
│    │  6. Build final DataProto batch                       │       │
│    └──────────────────────┬──────────────────────────────┘       │
│                           │                                      │
│                           ▼                                      │
│    ┌──────────────────────────────────────────────────────┐      │
│    │  Standard GRPO pipeline:                             │      │
│    │    - Reward function scoring (objectnav.py)          │      │
│    │    - GRPO advantage (group normalization over N)      │      │
│    │    - Actor FSDP update                               │      │
│    └──────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **GPU Memory Time-Sharing**: AI2Thor controllers and vLLM/FSDP training share the same GPUs. Controllers are created before rollout and destroyed after, freeing memory for training.

2. **Controller Reuse**: AI2Thor Unity processes take ~30-100s to start. Bare controllers are cached and reused across episodes via `reset_scene()` (~2-5s).

3. **Dynamic Slot Reuse**: Trajectories have variable lengths. When one finishes early, its simulator slot is immediately given to a pending trajectory, keeping GPU utilization high.

4. **SFT-Consistent Prompts**: The environment adapter builds prompts in the exact same format as the SFT training pipeline (`post_annotation/`), ensuring the RL fine-tuning sees the same data distribution.

---

## 2. Code Flow: End-to-End Training Step

```
main.py::main()
│
├── Parse config (OmegaConf + CLI overrides)
├── ray.init() with env vars (XFORMERS backend, etc.)
│
└── Runner.run(config)
    │
    ├── Load tokenizer + processor (Qwen3-VL)
    │
    ├── IF multiturn_env.enabled:
    │   ├── train_dataloader = _DummyDataLoader()     ← no dataset batches
    │   ├── _create_multiturn_rollout(config)
    │   │   ├── Load PoliformerDataset (houses, episodes, assets)
    │   │   ├── Apply difficulty filtering (rooms_seen buckets)
    │   │   ├── ObjectNavEnvFactory(dataset)           ← wraps dataset for cycling
    │   │   ├── _create_simulator_pools(mt_cfg, n_gpus)
    │   │   │   ├── Distribute num_simulators across GPUs
    │   │   │   ├── Map logical → physical GPU IDs (SLURM)
    │   │   │   ├── Load system_prompt from file
    │   │   │   ├── Compute coord_scale = render_width / model_output_scale
    │   │   │   └── For each GPU:
    │   │   │       └── SimulatorPool.remote(gpu_id, n_slots, ...)
    │   │   │
    │   │   └── MultiturnEnvRollout(tokenizer, processor, factory, pools, ...)
    │   │
    │   └── trainer.multiturn_rollout = multiturn_rollout
    │
    ├── RayPPOTrainer(config, ..., reward_fn, ...)
    ├── trainer.init_workers()                          ← creates FSDP + vLLM workers
    │
    └── trainer.fit()
        │
        └── FOR each training step:
            │
            ├── _make_batch_data(metrics)
            │   │
            │   ├── IF multiturn_rollout is not None:
            │   │   └── multiturn_rollout.generate_trajectories(
            │   │       actor_rollout_ref_wg, batch_size, n, config, metrics)
            │   │
            │   │   ┌──────────────────────────────────────────────────┐
            │   │   │ generate_trajectories():                         │
            │   │   │                                                  │
            │   │   │ 1. warmup_controllers()                          │
            │   │   │    └─ ray.get([pool.warmup_controllers(scene)])  │
            │   │   │                                                  │
            │   │   │ 2. Collect batch_size items from env_factory     │
            │   │   │                                                  │
            │   │   │ 3. Build pending_queue:                          │
            │   │   │    [(group_0, n_0, item_0),                      │
            │   │   │     (group_0, n_1, item_0),  ← N per prompt     │
            │   │   │     (group_1, n_0, item_1), ...]                 │
            │   │   │                                                  │
            │   │   │ 4. _initialize_batch(first N slots)              │
            │   │   │    ├─ pool.acquire_env(item_data) → slot_id      │
            │   │   │    ├─ pool.reset_env(slot_id)                    │
            │   │   │    └─ Return Trajectory objects                   │
            │   │   │                                                  │
            │   │   │ 5. _run_continuous_episode_loop():               │
            │   │   │    WHILE active trajectories:                    │
            │   │   │    │                                             │
            │   │   │    │ a. Force-terminate if num_steps >= max_depth│
            │   │   │    │ b. _harvest_and_refill(newly_done)          │
            │   │   │    │    ├─ get_trajectory_reward() per traj      │
            │   │   │    │    ├─ get_ground_truth() per traj           │
            │   │   │    │    ├─ release_env(slot_id) per traj         │
            │   │   │    │    └─ _initialize_batch(from pending)       │
            │   │   │    │                                             │
            │   │   │    │ c. pool.build_prompt(slot_id) [Ray parallel]│
            │   │   │    │    └─ ObjectNavEnvAdapter.build_prompt()     │
            │   │   │    │                                             │
            │   │   │    │ d. _tokenize_prompts(prompts, images)       │
            │   │   │    │    ├─ Convert <image> → HF content format   │
            │   │   │    │    ├─ processor.apply_chat_template()       │
            │   │   │    │    ├─ Downscale prior images × prior_scale  │
            │   │   │    │    ├─ process_image() for pixel constraints  │
            │   │   │    │    ├─ processor() → input_ids, pixel_values │
            │   │   │    │    ├─ get_rope_index() → mrope position_ids │
            │   │   │    │    ├─ Left-pad to max_prompt_length         │
            │   │   │    │    ├─ Trim truncated images' pixel_values   │
            │   │   │    │    └─ Keep raw PIL images for vLLM          │
            │   │   │    │                                             │
            │   │   │    │ e. DataProto.from_single_dict(tokenized)    │
            │   │   │    │ f. pad_dataproto_to_divisor(world_size)     │
            │   │   │    │ g. actor_rollout_ref_wg.generate_sequences()│
            │   │   │    │    └─ vLLM inference → response token IDs   │
            │   │   │    │ h. Decode response text                      │
            │   │   │    │                                             │
            │   │   │    │ i. pool.step_env(slot_id, response_text)    │
            │   │   │    │    └─ ObjectNavEnvAdapter.step()             │
            │   │   │    │       ├─ Check format/validity scores       │
            │   │   │    │       ├─ Parse action (ActionProposer)      │
            │   │   │    │       ├─ env.step(action) → new_state       │
            │   │   │    │       ├─ Track distance to target           │
            │   │   │    │       └─ Return (reward, terminated, info)  │
            │   │   │    │                                             │
            │   │   │    └─ Update trajectory step count               │
            │   │   │                                                  │
            │   │   │ 6. Sort trajectories by (group_id, n_idx)        │
            │   │   │ 7. destroy_controllers() → free GPU memory       │
            │   │   │ 8. _build_final_batch()                          │
            │   │   │    ├─ Tokenize last prompt + response per traj   │
            │   │   │    ├─ Concat prompt + response → input_ids       │
            │   │   │    ├─ Assign shared UIDs per GRPO group          │
            │   │   │    ├─ Place trajectory reward at last token       │
            │   │   │    └─ Return DataProto                           │
            │   │   └──────────────────────────────────────────────────┘
            │   │
            │   └── Return DataProto batch
            │
            ├── Reward scoring (objectnav.py compute_score)
            ├── GRPO advantage computation (group normalization)
            └── Actor FSDP update (gradient step)
```

---

## 3. New Files

### 3.1 `verl/workers/rollout/config.py` — `MultiturnEnvConfig`

**What:** Dataclass defining all configuration for multi-turn environment rollouts.

**Why:** Separates environment-specific settings from generic rollout config (temperature, TP size, etc.).

**Key fields:**

| Field | Default | Purpose |
|-------|---------|---------|
| `enabled` | `False` | Toggle multi-turn mode |
| `data_root` | Poliformer path | Root for 3D scene data |
| `max_depth` | 30 | Max steps before force-termination |
| `num_simulators` | 8 | Total AI2Thor controller slots |
| `render_width/height` | 616 | Must match SFT training |
| `model_output_scale` | 1000 | Coordinate grid size model was trained on |
| `coordinate_normalization_scale` | `None` (auto) | `render_width / model_output_scale` |
| `prior_image_scale` | 0.5 | Downscale prior observation images |
| `max_observations` | 20 | Cap on history images in prompt |
| `difficulties` | `None` | Filter by `rooms_seen` difficulty level |
| `max_per_difficulty` | `None` | Cap episodes per difficulty bucket |
| `override_indices` | `None` | Explicit dataset indices (bypasses filters) |

---

### 3.2 `verl/workers/simulator_pool.py` — `SimulatorPool`

**What:** A Ray remote actor that manages a pool of AI2Thor environment slots on a single GPU.

**Why:** AI2Thor controllers are GPU-bound (Unity rendering) and expensive to create (~30-100s). A pool pattern enables:
- Controller reuse across episodes (reset_scene is ~2-5s vs 30-100s startup)
- Thread-safe slot management for concurrent access
- GPU memory diagnostics per pool

**How it works:**

```
SimulatorPool (Ray actor, one per GPU)
├── gpu_id: int                          ← physical GPU this pool lives on
├── num_slots: int                       ← how many envs can run simultaneously
├── slots: list[ObjectNavEnvAdapter|None] ← active env adapters
├── slot_available: list[bool]            ← which slots are free
├── _cached_controllers: list[Controller|None]  ← bare Unity processes
├── _slot_lock: threading.Lock            ← thread safety for slot acquisition
└── _action_proposer: ActionProposer      ← shared parser (no VLM needed)
```

**Lifecycle of a slot:**

```
warmup_controllers(dummy_scene)    ← Pre-create bare AI2Thor controllers
        │
        ▼
acquire_env(item_data)             ← Find free slot, create ObjectNavEnvironment
        │                             using cached controller if available,
        │                             wrap in ObjectNavEnvAdapter
        ▼
reset_env(slot_id)                 ← Reset env to initial state
        │
        ▼
build_prompt(slot_id)              ← Build current prompt + images
        │
        ▼
step_env(slot_id, action_text)     ← Parse action, step env, return reward
        │  (repeat build_prompt → step_env)
        ▼
get_trajectory_reward(slot_id)     ← Compute final trajectory reward
get_ground_truth(slot_id)          ← Get metadata as JSON
        │
        ▼
release_env(slot_id)               ← Clear adapter, keep cached controller
        │                             Break callback reference cycle
        ▼
destroy_all()                      ← Close all Unity processes, free GPU memory
```

**Controller reuse detail:**

```
Episode 1:                          Episode 2 (same slot):
  acquire_env()                       acquire_env()
    ├─ cached_ctrl = None               ├─ cached_ctrl = <from episode 1>
    ├─ ObjectNavEnvironment(...)         ├─ ObjectNavEnvironment(
    │   └─ Creates new Unity process     │     existing_controller=cached_ctrl)
    └─ Cache: _cached[i] = env._ai2thor │   └─ Calls reset_scene() (~2-5s)
                                         └─ Cache: _cached[i] = env._ai2thor
```

**Memory leak prevention (`release_env`):**

```python
# Problem: AI2Thor controller holds a callback reference to the old env
# controller → _restore_state_on_restart (bound method) → old env → full history
#
# Solution: detach callback on release
cached_ctrl.set_restart_callback(None)
adapter.state_history = None
```

---

### 3.3 `verl/workers/rollout/multiturn_env.py` — `MultiturnEnvRollout`

**What:** The driver-side orchestrator that runs multi-turn environment rollouts. This is the heart of the system.

**Why:** Standard EasyR1 generates responses once per prompt. ObjectNav requires iterative interaction: observe → think → act → observe → ... The driver coordinates between the vLLM model (on FSDP workers) and the AI2Thor environments (on SimulatorPool actors).

**How it works:**

**Key classes:**

```
Trajectory (dataclass)
├── pool: ray.ActorHandle          ← which SimulatorPool owns this trajectory
├── slot_id: int                   ← which slot in that pool
├── episode_id: str                ← unique ID for logging
├── group_id: int                  ← dataset item index (for GRPO grouping)
├── n_idx: int                     ← which of N trajectories for this item
├── step_responses: list[str]      ← model responses at each step
├── terminated: bool
├── num_steps: int
├── reward: float | None           ← collected on termination
├── ground_truth: str | None
├── last_prompt: list[dict] | None ← cached for final batch construction
└── last_images: list[PIL.Image]
```

**Dynamic slot reuse algorithm:**

```
Total trajectories needed = batch_size × n_trajectories
Available slots = sum(pool.num_slots for all pools)

1. pending_queue = deque of (group_id, n_idx, item_data) tuples

2. Seed: fill up to min(pending_queue, total_slots) active trajectories

3. CONTINUOUS LOOP:
   ┌─────────────────────────────────────────────────────┐
   │                                                     │
   │  Force-terminate any trajectory at max_depth        │
   │          │                                          │
   │          ▼                                          │
   │  _harvest_and_refill(newly_terminated):             │
   │    • Collect rewards + ground truths (Ray parallel) │
   │    • Release slots (Ray parallel)                   │
   │    • Pop N items from pending_queue                 │
   │    • _initialize_batch(new items) → new Trajectories│
   │    • Add to active list                             │
   │          │                                          │
   │          ▼                                          │
   │  Remove terminated from active list                 │
   │  If no active trajectories → BREAK                  │
   │          │                                          │
   │          ▼                                          │
   │  Build prompts (all active, Ray parallel)           │
   │          │                                          │
   │          ▼                                          │
   │  Tokenize prompts → DataProto                       │
   │          │                                          │
   │          ▼                                          │
   │  generate_sequences() → responses                   │
   │          │                                          │
   │          ▼                                          │
   │  Step environments (all active, Ray parallel)       │
   │  Mark terminated trajectories                       │
   │          │                                          │
   │          └──────────── LOOP ────────────────────────┘

4. Sort all_trajectories by (group_id, n_idx)
5. Destroy controllers
6. _build_final_batch() → DataProto
```

**Why dynamic reuse matters:**

```
Without dynamic reuse (sequential chunks):
  Chunk 1: [traj_0 ■■■■■■■, traj_1 ■■, traj_2 ■■■■■■■■■, traj_3 ■■■]
            ────────────────────────────────────────────────────────────
            Slot utilization: ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
            (traj_1 and traj_3 finish early, slots sit idle)

With dynamic reuse:
  Active:   [traj_0 ■■■■■■■, traj_1 ■■|traj_4 ■■■■, traj_2 ■■■■■■■■■, traj_3 ■■■|traj_5 ■■■]
            ────────────────────────────────────────────────────────────
            Slot utilization: ████████████████████████████████████████
            (finished slots immediately reused by pending trajectories)
```

---

### 3.4 `verl/workers/rollout/multiturn_tokenizer.py` — `TokenizerMixin`

**What:** Mixin class providing `_tokenize_prompts()` and `_build_final_batch()` methods. Extracted from `MultiturnEnvRollout` for separation of concerns.

**Why:** Tokenization for multi-turn VLM prompts with images is complex. It needs to:
1. Convert `<image>` placeholders to HF format
2. Process images (downscale priors, apply pixel constraints)
3. Handle Qwen3-VL's multimodal rope position IDs (mrope)
4. Left-pad to uniform length
5. Handle truncation that may split image token groups
6. Maintain both processed images (for FSDP log-prob) and raw PIL images (for vLLM generation)

**`_tokenize_prompts()` flow:**

```
Input: prompts (list of chat message dicts), images (list of PIL images per prompt)
                    │
                    ▼
┌─────────────────────────────────────────────┐
│ For each (messages, images):                 │
│                                              │
│ 1. Convert "<image>" in text content         │
│    to HF structured format:                  │
│    {"type": "image"}, {"type": "text", ...}  │
│                                              │
│ 2. processor.apply_chat_template()           │
│    → text_prompt with special tokens         │
│                                              │
│ 3. Process images:                           │
│    ├─ Prior images (all but last):           │
│    │   resize × prior_image_scale            │
│    ├─ Keep raw_images for vLLM               │
│    └─ process_image() → processed_images     │
│       (apply min_pixels, max_pixels)         │
│                                              │
│ 4. processor(processed_images, text_prompt)  │
│    → input_ids, attention_mask,              │
│      pixel_values, image_grid_thw            │
│                                              │
│ 5. get_rope_index() → mrope position_ids     │
│    (Qwen3-VL uses 4D position encoding:      │
│     text + 3 spatial/temporal dims)           │
│                                              │
│ 6. postprocess_data() → left-pad/truncate    │
│    to max_prompt_length                      │
│                                              │
│ 7. Handle truncation of image tokens:        │
│    ├─ Count surviving <|image_pad|> tokens    │
│    ├─ Trim pixel_values/image_grid_thw       │
│    │   to only complete images               │
│    └─ Mask out partial image tokens          │
│       (replace with pad, zero attention)     │
│                                              │
│ 8. raw_prompt_ids = tokenizer.encode()       │
│    (text-only for vLLM, separately truncated)│
└─────────────────────────────────────────────┘
                    │
                    ▼
Output dict:
  input_ids:      (bs, max_prompt_length) tensor, left-padded
  attention_mask:  (bs, max_prompt_length) tensor
  position_ids:   (bs, 4, max_prompt_length) tensor (mrope)
  raw_prompt_ids: numpy object array of token ID lists (for vLLM)
  raw_images:     numpy object array of PIL image lists (for vLLM)
  multi_modal_data: numpy object array of {pixel_values, image_grid_thw} (for FSDP, only when not for_generation)
```

**`_build_final_batch()` flow:**

```
Input: trajectories, rewards, ground_truths, n_trajectories

1. For each trajectory:
   ├─ last_prompt  → the full conversation up to final observation
   └─ last_response → the model's final response text

2. _tokenize_prompts(prompts, images, for_generation=False)
   → prompt tokens (with multi_modal_data for FSDP training)

3. Tokenize responses:
   ├─ Encode response text
   ├─ Append EOS
   ├─ Right-pad to max_response_length
   └─ Build response_mask

4. Concatenate:
   ├─ input_ids = [prompt | response]
   ├─ attention_mask = [prompt_mask | response_mask]
   └─ position_ids = [prompt_pos | prompt_pos[-1] + 1..resp_len]

5. UIDs: assign same UUID to all N trajectories of same dataset item
   (required for GRPO group normalization)

6. Place trajectory reward at last response token:
   token_level_scores[i, last_token_position] = reward

7. Return DataProto(batch=TensorDict, non_tensor_batch={uid, ground_truth, multi_modal_data})
```

---

### 3.5 `verl/workers/rollout/objectnav_adapter.py` — `ObjectNavEnvAdapter`

**What:** Adapts `ObjectNavEnvironment` (from the parent spatial-reasoning project) to the `EnvInterface` protocol expected by the rollout driver.

**Why:** Bridges two codebases:
- `interactive_reasoning.objectnavtask.environment` (AI2Thor physics, navigation, rendering)
- `verl.workers.rollout.multiturn_env` (GRPO rollout driver)

**Where it runs:** Inside `SimulatorPool` Ray actors (on simulator GPUs), NOT on the driver.

**Key methods:**

#### `reset()`
```
1. env.reset()                              ← AI2Thor scene reset
2. Get initial state (observation, distance)
3. Create StateActionHistory(root_state)    ← tracks full trajectory for prompt building
4. Record initial_distance to target
5. Reset scoring accumulators
```

#### `build_prompt()`
Constructs a prompt matching the exact SFT training format from `sft_data.py`:

```
Messages: [
  {role: "system", content: <system_prompt>},
  {role: "user", content:
    [Step 0]                          ← prior image labels
    <image>
    [Step 1]
    <image>
    ...
    Your task is to find the **X** (description).

    **Memory from previous steps:**
    - Step 0: <summary from step 0>
    - Step 1: <summary from step 1>
    ...

    Step N. Here is your current observation:
    <image>

    <step_instructions>              ← format reminder (standard or forced answer)
  }
]
```

**Why match SFT format exactly:** The model was pre-trained via SFT on data generated by `post_annotation/`. If the RL prompt format differs, the model sees out-of-distribution inputs and performance degrades.

#### `step(action_text)`

```
1. Increment num_steps
2. Score format (regex: <think>...</think><summary>...</summary><action>)
3. Score validity (parseable action with valid coordinates)
4. Parse <summary> tag for memory
5. ActionProposer._parse_action_response(response) → typed action
6. env.step(action) → new_state
7. Update state_history with (action, new_state)
8. Track distance_to_target, success
9. Return (reward, terminated, {action_type})
```

#### `get_trajectory_reward()`

```
reward = 0.0
  + 1.0  if success (target found within bounding box)
  + 0.1  × avg_format_score   (encourages consistent formatting)
  + 0.15 × avg_validity_score (encourages parseable actions)
  - 0.005 × num_steps          (encourages shorter trajectories)
```

**Reward breakdown:**
| Component | Weight | Purpose |
|-----------|--------|---------|
| Success | +1.0 | Primary signal: did you find the object? |
| Format | +0.1 × avg | Keep producing `<think>`, `<summary>`, action tags |
| Validity | +0.15 × avg | Keep producing parseable coordinates |
| Step penalty | -0.005 × N | Prefer shorter paths |

---

### 3.6 `verl/workers/rollout/objectnav_factory.py` — `ObjectNavEnvFactory`

**What:** Simple dataset iterator that provides episode data to the rollout driver.

**Why:** Separates dataset concerns from environment management. The factory only cycles through dataset items; actual environment creation happens in `SimulatorPool.acquire_env()`.

```python
class ObjectNavEnvFactory(EnvFactory):
    def __init__(self, dataset):
        self.dataset = dataset
        self._indices = list(range(len(dataset)))
        self._item_idx = 0

    def get_next_item(self) -> dict:
        # Cycles through dataset, reshuffling when exhausted
        if self._item_idx >= len(dataset):
            np.random.shuffle(self._indices)
            self._item_idx = 0
        data = self.dataset[self._indices[self._item_idx]]
        self._item_idx += 1
        return data
```

---

### 3.7 `examples/reward_function/objectnav.py`

**What:** Reward function for scoring model responses. Supports three modes: offline (SFT data), online (per-step env metadata), and trajectory-level (pre-computed by env adapter).

**Why:** EasyR1's `AutoRewardManager` calls this to score responses after rollout. In multi-turn mode, the trajectory reward is pre-computed by `ObjectNavEnvAdapter.get_trajectory_reward()` and passed through as `ground_truth.trajectory_reward`.

**Three scoring modes:**

```
                    ground_truth JSON
                         │
           ┌─────────────┼──────────────┐
           ▼             ▼              ▼
   "trajectory_reward"  "distance_to_   Neither
      key present?     target_before"?  (offline)
           │             │              │
           ▼             ▼              ▼
   Return directly   Online scoring  Offline scoring
   as overall score  (distance-based) (expert comparison)
```

**Offline scoring (SFT-derived data):**
```
Format (0.2):   regex check for <think>...<summary>...<action>
Validity (0.3): parseable action tag with valid coordinates
Accuracy (0.5): type match + coordinate proximity to expert
                 Uses inverse-square distance: 1/(1 + (d/threshold)²)
```

**Online scoring (live env, per-step):**
```
Format (0.2):   same regex
Validity (0.3): same parsing
Accuracy (0.5):
  - Answer: proximity to target's 2D projection (if visible)
  - Explore: distance improvement ratio (dist_before - dist_after) / dist_before
```

**Trajectory-level (multi-turn mode):**
```
The ObjectNavEnvAdapter already computed the full trajectory reward.
ground_truth contains {"trajectory_reward": float, ...}
→ Return trajectory_reward directly as overall score.
(format/validity metrics still computed for logging)
```

---

### 3.8 `examples/format_prompt/objectnav.jinja`

```jinja2
{{ content | trim }}
```

**What:** Trivial Jinja2 template that passes content through unchanged.

**Why:** EasyR1 requires a `format_prompt` template path. Since the ObjectNav adapter builds its own structured prompts (matching SFT format), no additional formatting is needed.

---

## 4. Modified Files

### 4.1 `verl/trainer/main.py`

**Changes:**
- Added logging configuration with `LOG_LEVEL` env var
- Quiet noisy AI2Thor/environment loggers
- Added `_DummyDataLoader` class (when data comes from env, not files)
- Added `_create_simulator_pools()` function
- Added `_create_multiturn_rollout()` function
- In `Runner.run()`: branch on `multiturn_env.enabled` to skip dataset loading
- Set `VLLM_ATTENTION_BACKEND=XFORMERS` in Ray runtime env

### 4.2 `verl/trainer/ray_trainer.py`

**Changes:**
- Added `self.multiturn_rollout = None` attribute on `RayPPOTrainer`
- In `_make_batch_data()`: if `multiturn_rollout` exists, call `generate_trajectories()` instead of standard rollout
- Removed `tqdm` progress bars (replaced with logging)
- Added logging around worker initialization

**Key insertion in `_make_batch_data()`:**
```python
# Multi-turn environment rollout mode
if self.multiturn_rollout is not None:
    return self.multiturn_rollout.generate_trajectories(
        actor_rollout_ref_wg=self.actor_rollout_ref_wg,
        batch_size=self.config.data.rollout_batch_size,
        n_trajectories=self.config.worker.rollout.n,
        config=self.config,
        metrics=metrics,
    )
```

### 4.3 `verl/workers/fsdp_workers.py`

**Changes:**

1. **Import compatibility fix:**
   ```python
   try:
       from transformers.modeling_utils import no_init_weights
   except ImportError:
       from transformers.initialization import no_init_weights
   ```

2. **Pre-computed multi-modal data bypass:**
   When `pixel_values` is already present in `multi_modal_data` (pre-computed during tokenization in the multiturn path), use it directly instead of re-processing images through the standard path. This prevents image token count mismatches that occur when different processing paths produce different grid sizes.

   ```python
   if "pixel_values" in multi_modal_data:
       # Pre-computed from tokenization — use directly
       # Do NOT cache by uid: multi-turn rollouts reuse the same uid
       # across steps but with different images
       batch_multi_modal_inputs.append(multi_modal_data)
   ```

### 4.4 `verl/workers/rollout/vllm_rollout_spmd.py`

**Changes:**

1. **vLLM version compatibility for `disable_mm_preprocessor_cache`:**
   ```python
   # Check if field exists before setting (removed in vLLM >= 0.16)
   from vllm.engine.arg_utils import EngineArgs
   if "disable_mm_preprocessor_cache" in EngineArgs.__dataclass_fields__:
       engine_kwargs["disable_mm_preprocessor_cache"] = True
   ```

2. **Graceful sleep mode fallback:**
   ```python
   try:
       self.inference_engine.sleep(level=1)
   except Exception as e:
       print(f"WARNING: vLLM sleep(level=1) failed: {e}")
   ```

3. **Raw PIL images path for vLLM generation:**
   When `batch_raw_images` is present (from the multiturn tokenizer), construct vLLM inputs using raw PIL images directly. vLLM does its own internal image processing, so pre-processed pixel tensors aren't needed for generation.

   ```python
   if batch_raw_images is not None:
       vllm_inputs = []
       for i in range(batch_size):
           images = batch_raw_images[i]
           vllm_inputs.append({
               "prompt_token_ids": batch_raw_prompt_ids[i],
               "multi_modal_data": {"image": images} if images else None,
           })
   ```

### 4.5 `verl/workers/config.py`

**Changes:** Added `MultiturnEnvConfig` to `WorkerConfig`:
```python
multiturn_env: MultiturnEnvConfig = field(default_factory=MultiturnEnvConfig)
```

### 4.6 `verl/workers/rollout/__init__.py`

**Changes:** Export `MultiturnEnvConfig` alongside `RolloutConfig`.

### 4.7 `verl/workers/sharding_manager/fsdp_vllm.py`

**Changes:** vLLM API compatibility fix:
```python
# get_tensor_model_parallel_group was renamed to get_tp_group
_get_tp_group = getattr(vllm_ps, "get_tensor_model_parallel_group", None) or vllm_ps.get_tp_group
self.tp_group = _get_tp_group().device_group
```

### 4.8 `verl/models/transformers/qwen3_vl.py`

**Changes:** Updated visual encoder output unpacking for newer transformers:
```python
# Before: image_embeds, deepstack_image_embeds = model.visual(...)
# After:
visual_out = model.visual(pixel_values, grid_thw=image_grid_thw)
image_embeds = visual_out.pooler_output
deepstack_image_embeds = visual_out.deepstack_features
```

### 4.9 `verl/utils/dataset.py`

**Changes:** Added numpy array handling in `process_image()`:
```python
elif isinstance(image, np.ndarray):
    if image.dtype != object:
        image = Image.fromarray(image)
    elif image.ndim == 0:
        image = image.item()  # 0-d object array wrapping a PIL image
```

### 4.10 `verl/utils/vllm_utils.py`

**Changes:** Import path fix:
```python
# Before: from vllm.lora.models import LoRAModel
# After:
from vllm.lora.lora_model import LoRAModel
```

### 4.11 `verl/workers/actor/dp_actor.py` & `verl/workers/critic/dp_critic.py`

**Changes:** Removed `tqdm` progress bars from inner training loops (cleaner logging in distributed setting).

---

## 5. Component Deep Dives

### 5.1 GPU Memory Time-Sharing

The same GPUs host both AI2Thor controllers (for environment rendering) and the vLLM + FSDP model. Memory is time-shared:

```
Timeline of a single training step:

         ┌── Rollout Phase ──┐  ┌── Training Phase ──┐
         │                    │  │                     │
GPU Mem: │ ██ vLLM weights    │  │ ██ FSDP weights     │
         │ ░░ AI2Thor ctrls   │  │ ░░ (freed)          │
         │ ▓▓ KV cache        │  │ ▓▓ Gradients        │
         │                    │  │ ▒▒ Optimizer states  │
         └────────────────────┘  └─────────────────────┘

         warmup_controllers()    destroy_controllers()
              ↑                        ↑
              │                        │
         Create Unity procs      Close Unity procs
         (~2-5s with cache)      (free ~300MB × N slots)
```

The `gpu_memory_utilization` config for vLLM must be set low enough (e.g., 0.6) to leave room for AI2Thor controllers (~300MB each).

### 5.2 Image Processing Pipeline

Images flow through multiple stages, with different processing for generation vs training:

```
AI2Thor Controller
    │ numpy RGB frame (616×616)
    ▼
ObjectNavEnvAdapter.build_prompt()
    │ PIL Image (full resolution)
    ▼
TokenizerMixin._tokenize_prompts()
    │
    ├──[for_generation=True]─────────────────────────────────────┐
    │                                                             │
    │  Prior images: resize × prior_image_scale                  │
    │  ─────────────────────┐                                     │
    │                       ▼                                     │
    │                raw_images (PIL)                              │
    │                       │                                     │
    │                       ▼                                     │
    │           vllm_rollout_spmd.py                              │
    │           (vLLM does its own image processing)              │
    │                                                             │
    ├──[for_generation=False]────────────────────────────────────┐
    │                                                             │
    │  Prior images: resize × prior_image_scale                  │
    │  ─────────────────────┐                                     │
    │                       ▼                                     │
    │                process_image(min_pixels, max_pixels)        │
    │                       │                                     │
    │                       ▼                                     │
    │                processor() → pixel_values, image_grid_thw  │
    │                       │                                     │
    │                       ▼                                     │
    │           fsdp_workers.py                                   │
    │           (pre-computed pixel_values used directly,          │
    │            bypasses uid-based cache to avoid mismatches)    │
    └─────────────────────────────────────────────────────────────┘
```

### 5.3 GRPO Group Structure

GRPO normalizes advantages within groups of N trajectories for the same prompt. In multi-turn mode:

```
Dataset item 0 ──┬── trajectory (group_id=0, n_idx=0) ─── uid_A
                 ├── trajectory (group_id=0, n_idx=1) ─── uid_A
                 └── trajectory (group_id=0, n_idx=2) ─── uid_A

Dataset item 1 ──┬── trajectory (group_id=1, n_idx=0) ─── uid_B
                 ├── trajectory (group_id=1, n_idx=1) ─── uid_B
                 └── trajectory (group_id=1, n_idx=2) ─── uid_B

The same uid is assigned to all N trajectories of the same item.
GRPO advantage = (reward - mean(group rewards)) / std(group rewards)
```

### 5.4 Coordinate System

```
Model output space:        Pixel space:              World space:
[0, 1000] × [0, 1000]     [0, 616] × [0, 616]      AI2Thor 3D

     (500, 500)      ×0.616    (308, 308)            (x, y, z)
         ●──────────────────────→●                      ↑
                                                        │
coordinate_normalization_scale = 616/1000 = 0.616       │
                                                        │
ActionProposer parses model output coordinates,    AI2Thor converts
multiplies by scale, produces pixel coordinates    pixel → 3D via
for AI2Thor's image-to-world projection.           depth + camera
```

---

## 6. Data Flow Diagrams

### 6.1 Full System Component Diagram

```
                          ┌─────────────────────────────┐
                          │         DRIVER NODE          │
                          │                              │
                          │  ┌──────────────────────┐   │
                          │  │   RayPPOTrainer      │   │
                          │  │   ┌──────────────┐   │   │
                          │  │   │MultiturnEnv  │   │   │
                          │  │   │Rollout       │   │   │
                          │  │   │              │   │   │
                          │  │   │ TokenizerMix │   │   │
                          │  │   └──────┬───────┘   │   │
                          │  └──────────┼───────────┘   │
                          │             │               │
                          └─────────────┼───────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
                    ▼                   ▼                   ▼
           ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
           │ SimulatorPool│    │ SimulatorPool│    │  FSDPWorker  │
           │   (GPU 0)    │    │   (GPU 1)    │    │  (GPU 0..N)  │
           │              │    │              │    │              │
           │ ┌──────────┐ │    │ ┌──────────┐ │    │ vLLM engine  │
           │ │ Slot 0:  │ │    │ │ Slot 0:  │ │    │    or        │
           │ │ Adapter  │ │    │ │ Adapter  │ │    │ FSDP model   │
           │ │ ┌──────┐ │ │    │ │ ┌──────┐ │ │    │              │
           │ │ │AI2Thor│ │ │    │ │ │AI2Thor│ │ │    │  (generates  │
           │ │ │Ctrl   │ │ │    │ │ │Ctrl   │ │ │    │   responses  │
           │ │ └──────┘ │ │    │ │ └──────┘ │ │    │   and trains) │
           │ ├──────────┤ │    │ ├──────────┤ │    │              │
           │ │ Slot 1.. │ │    │ │ Slot 1.. │ │    └──────────────┘
           │ └──────────┘ │    │ └──────────┘ │
           └──────────────┘    └──────────────┘
```

### 6.2 Single Rollout Step Data Flow

```
         SimulatorPool                  Driver                    FSDPWorker
         (Ray actors)              (MultiturnEnvRollout)          (vLLM mode)
              │                          │                            │
              │◄──── build_prompt() ─────┤                            │
              │                          │                            │
              │───── (msgs, images) ────►│                            │
              │                          │                            │
              │                    _tokenize_prompts()                │
              │                    ┌─────────────────┐                │
              │                    │ Convert images   │                │
              │                    │ Apply chat tmpl  │                │
              │                    │ Compute mrope    │                │
              │                    │ Left-pad         │                │
              │                    │ Handle truncation│                │
              │                    └────────┬────────┘                │
              │                             │                         │
              │                    DataProto(input_ids,               │
              │                     position_ids, raw_images)         │
              │                             │                         │
              │                             │──── generate_sequences ─►│
              │                             │                         │
              │                             │◄─── response token IDs ──│
              │                             │                         │
              │                    Decode response text               │
              │                             │                         │
              │◄──── step_env(response) ────┤                         │
              │                             │                         │
              │  ObjectNavEnvAdapter.step()  │                         │
              │  ├─ Parse action             │                         │
              │  ├─ env.step() in AI2Thor    │                         │
              │  └─ Track reward/distance    │                         │
              │                             │                         │
              │───── (reward, term, info) ──►│                         │
              │                             │                         │
```

### 6.3 Reward Data Flow

```
         ObjectNavEnvAdapter              MultiturnEnvRollout           AutoRewardManager
         (inside SimulatorPool)           (driver)                     (objectnav.py)
              │                                │                            │
              │  Per-step scoring:             │                            │
              │  ├─ format_scores.append()     │                            │
              │  └─ validity_scores.append()   │                            │
              │                                │                            │
              │ (on termination)               │                            │
              │◄── get_trajectory_reward() ─────┤                            │
              │                                │                            │
              │  reward = success(+1.0)        │                            │
              │        + format(+0.1×avg)      │                            │
              │        + validity(+0.15×avg)   │                            │
              │        - step_penalty(0.005×N) │                            │
              │                                │                            │
              │─── trajectory_reward ──────────►│                            │
              │                                │                            │
              │◄── get_ground_truth() ──────────┤                            │
              │                                │                            │
              │─── JSON{trajectory_reward,     │                            │
              │     success, distances, ...} ──►│                            │
              │                                │                            │
              │                          _build_final_batch()               │
              │                          ├─ token_level_scores[last] = rew  │
              │                          └─ ground_truth = JSON             │
              │                                │                            │
              │                                │── DataProto ──────────────►│
              │                                │                            │
              │                                │         compute_score()    │
              │                                │         ├─ Sees trajectory_│
              │                                │         │  reward in gt    │
              │                                │         └─ Returns it as   │
              │                                │            overall score   │
              │                                │◄── scores ─────────────────│
              │                                │                            │
              │                          GRPO advantage computation         │
              │                          Actor FSDP gradient update          │
```
