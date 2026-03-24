# ERA-RL Optimizations vs Standard veRL/EasyR1

Summary of how [Embodied-Reasoning-Agent (ERA)](https://github.com/EmbodiedAgent/ERA) modifies the veRL framework for multi-turn embodied RL training. Focus is on changes relevant to our EasyR1 setup where each trajectory step becomes a separate (prompt, response) training sample.

---

## 1. Whole-Trajectory-as-Response Architecture

**The single biggest design difference from our setup.**

ERA does NOT split trajectories into separate (prompt, response) samples per step. Instead:

- **Prompt** = a single pad token (masked out entirely)
- **Response** = the ENTIRE multi-turn trajectory concatenated as one sequence

The full conversation (system prompt + obs₁ + action₁ + obs₂ + action₂ + ... + obsₙ + actionₙ) is treated as one "response" of length up to `max_trajectory_length` (3072 tokens).

A **loss mask** distinguishes LLM-generated action tokens from environment-provided observation tokens within this single sequence. Special tokens `<|box_start|>` and `<|box_end|>` wrap each LLM response segment. The `_compute_loss_mask()` method:
1. Finds all `<|box_start|>` / `<|box_end|>` pairs
2. Sets `loss_mask=1` only for tokens between these markers
3. Removes the special tokens themselves and shifts content left to fill gaps
4. Produces `end_of_response_position_mask` marking the last token of each action segment

**Crucially, each step's observation CAN have different/unique context.** In their ALFRED environment, each step's `obs_str` includes only a compressed summary of the last 1 step (just `reasoning_and_reflection` + `language_plan` from the model's thinking, dropping `visual_description`). So step 5's observation is NOT identical to step 3's — it has a different image, different interaction history summary, different env feedback. Despite this, the whole trajectory is still concatenated as one training sequence:

```
[sys_prompt | obs₁(no history) | act₁ | obs₂(summary of step 1) | act₂ | obs₃(summary of step 2) | act₃ | ...]
```

This works because observation tokens are masked from the loss. The model doesn't predict observations — it only predicts action tokens, attending to whatever observation context surrounds them.

**Implication:** One trajectory = one training sample (not 10-20 samples). This dramatically reduces the number of optimizer steps. With `ppo_mini_batch_size=256` and say 8 trajectories, that's ~1 optimizer step vs our ~20+ when each step is a separate sample.

**Trade-off:** Sequences are longer (up to 3072 tokens) but there are far fewer of them. With `use_remove_padding=True` and flash attention, this can be more efficient than many shorter padded sequences.

**Relevance to us:** Directly applicable. Our setup also has step-specific contexts (each step sees different images and potentially windowed history). We could concatenate the full trajectory as one sequence with a loss mask on action tokens. This eliminates the sample multiplication that causes our optimizer bottleneck. The fact that each step has different context is NOT a blocker — ERA already handles this.

---

## 2. Multi-Turn Advantage Estimators

ERA implements 5 custom advantage estimators for multi-turn credit assignment, beyond standard GAE/GRPO. All operate on the single-trajectory-as-response format using `loss_mask`, `step_id`, and `env_id` metadata.

### 2a. Masked GAE (`masked_gae`)
Standard GAE but computed at the **turn level** (not token level):
- Each sample has a `value_pos` (value estimate position) and `next_value` (bootstrap from next step)
- `delta = reward[step] + gamma * next_value - value[step]`
- GAE chains across steps: `gae = delta + gamma * lam * next_step_gae`
- Uses `env_gae_dict[(env_id, step_id)]` to link steps within the same trajectory
- Advantage is broadcast to all valid (loss_mask=1) positions in the step
- Requires a critic model

### 2b. Bi-Level GAE (`bi_level_gae`)
Two-level advantage computation:
- **Turn level:** Same as Masked GAE — computes GAE across trajectory steps using step rewards
- **Token level:** Within each step's action tokens, runs standard GAE with `token_level_gamma` using per-token values
- The turn-level GAE is injected at the last token of each step, then token-level GAE propagates it backward through the step's tokens
- Requires a critic model

### 2c. Token-Level GAE (`token_level_gae`)
Pure token-level bootstrapping that also chains across steps:
- Bootstraps from next step's first token value via `env_value_dict`
- Within-step: standard token-level GAE with `token_level_gamma`
- Requires a critic model

### 2d. Return-Based Advantage with Asymmetric Handling (`return_based_advantage_except_minus_reward`)
Critic-free estimator with asymmetric treatment:
- Computes discounted returns backward through the trajectory: `return[step] = reward[step] + lambda * return[step+1]`
- **If reward < 0:** `advantage = reward` (don't compound negative returns — just use immediate punishment)
- **If reward >= 0:** `advantage = return` (use full discounted future return)
- Rationale: prevents negative spirals from compounding, encourages exploration on positive signals
- Does NOT require a critic model

### 2e. MinMax Advantage (`minmax_advantage`)
Critic-free, symmetric clipping:
- Same return computation as above
- **If reward < 0:** `advantage = min(return, reward)` — take the worse of return vs immediate reward
- **If reward >= 0:** `advantage = max(return, reward)` — take the better of return vs immediate reward
- Bounds advantages to prevent extreme values in either direction
- Does NOT require a critic model

**Relevance to us:** We currently use GRPO (group-relative normalization across `n` rollouts per prompt). The return-based and minmax estimators are interesting because they're critic-free (like GRPO) but work with per-step rewards and multi-turn trajectories. Could be combined with our trajectory-level rewards.

---

## 3. Per-Step Reward Embedding

When `use_multi_turn_reward=True`, ERA embeds per-step rewards at specific token positions:

```python
reward_positions = torch.nonzero(end_of_response_position_mask)
multi_turn_token_level_rewards = torch.zeros_like(end_of_response_position_mask)
for idx, reward in enumerate(rewards):
    multi_turn_token_level_rewards[reward_positions[idx]] = reward
```

Each environment step's reward is placed at the last token of that step's action segment. The advantage estimators then pick these up at the correct positions.

**Relevance to us:** We currently assign a single trajectory-level reward. ERA's per-step reward allows finer credit assignment — each step can get its own reward (e.g., distance improvement), which the advantage estimators propagate backward via discounted returns.

---

## 4. Rollout vs Training Context Mismatch (and how they handle it)

ERA has a **large deliberate mismatch** between what the model sees during rollout vs training.

### Rollout context: `window_size=1`

In their actual configs (`examples/alfred/run.sh`, `examples/ebman/run.sh`):

```yaml
rollout_manager:
  window_size: 1   # model sees ONLY current observation during generation
  max_turns: 30    # ALFRED trajectories up to 30 steps
```

During rollout, step N's prompt contains only the current `obs_str` — which itself embeds a compressed 1-step summary (just `reasoning_and_reflection` + `language_plan` from the previous step's thinking, with `visual_description` stripped). The model generates each action nearly blind to history beyond the last step's summary.

### Training context: `window_size=None` (full trajectory)

`generate_batch_for_update()` calls `_generate_input_for_uptate(window_size=None)`, which calls `_single_recording_to_prompt()` where:

```python
start_step = max(0, step - window_size) if window_size is not None else 0
# window_size=None → start_step=0 → full trajectory from step 0
```

So training concatenates ALL 30 steps into one sequence, even though each action was generated seeing only 1 step of context.

### How they handle the log prob mismatch

The rollout-time log probs are **never saved**. After constructing the full training sequence, ERA recomputes `old_log_probs` via a forward pass over the full trajectory with the current (pre-update) weights (`ray_trainer.py:1639-1642`):

```python
# recompute old_log_probs
old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
batch = batch.union(old_log_prob)
```

This gives `old_log_prob(act_N) = π_θ_old(act_N | full_trajectory_prefix)`. During the actor update, `new_log_prob(act_N) = π_θ_new(act_N | full_trajectory_prefix)`. Both are conditioned on the **same full-trajectory context**, so the PPO importance ratio `r = π_new / π_old` purely measures weight divergence. The PPO clipping math is correct.

### What this does NOT fix: advantage misalignment

The advantages are computed from rewards that resulted from actions chosen under `window_size=1` context. But the model is being trained to produce those actions when seeing the **full trajectory**. This is a semantic mismatch:

- The advantage says "`act_5` was good/bad" — but it was good/bad in the context of only seeing `obs_5` with a 1-step summary
- During training, the model sees `obs_1, act_1, ..., obs_5` — a much richer information state where a completely different action might be optimal
- Example: model blindly explores right at step 5 and gets lucky (positive reward). Advantage says "go right is good." But with full context, the model can see it already explored right at step 2 — reinforcing "go right given you know you already went right" is a potentially wrong signal

This is effectively hindsight learning: "given you can now see everything, learn which actions turned out well." The directional signal (success vs failure) is still correct, and fresh rollouts each iteration prevent systematic drift, but it's a real approximation.

### Self-summarization as implicit context compression

The observation at each step is constructed by the environment (`AlfredEnv.step()`), not the rollout manager. Each `obs_str` contains:

```python
interaction_history_reflection_and_plan = self.bufferreflectionandplan.copy()[-1:]
user_prompt = (
    "<image>\n" +
    "instruction: " + self.instruction + "\n" +
    "interaction_history:" + str(interaction_history_reflection_and_plan) + "\n" +
    "Based on the above information..."
)
```

The model's thinking output has structured fields (`visual_description`, `reasoning_and_reflection`, `language_plan`). Only `reasoning_and_reflection` + `language_plan` from the **last 1 step** are carried forward as the interaction history. `visual_description` is dropped. This compressed summary is baked into each observation — the rollout manager just records and replays it.

Even in the full training sequence, step 5's observation text only contains a summary of step 4. The model CAN attend to earlier steps' observations and actions through the sequence's causal attention, but each observation's text content remains the compressed version.

**Relevance to us:** We currently include prior observation images in each step's prompt (up to `max_observations`). ERA's approach of using self-summarization as implicit compression is interesting — it means the observation tokens are short (text summary, not images), keeping the full-trajectory training sequence within 3072 tokens. For our image-heavy setup, this would be critical to make the whole-trajectory approach feasible.

---

## 5. Environment Reuse via Config Hashing

ERA's `QwenVLRolloutManager.reset()` reuses environment instances across rollout iterations:

1. Hashes each environment's config via `config.config_id()`
2. Groups existing environments by config hash
3. For new environments with matching configs: reuses the existing instance (just calls `.reset(seed)`)
4. Only creates new environments when no matching config exists
5. Closes unused environments before creating new ones

```python
# Reuse pattern
if bucket_key in env_buckets and env_buckets[bucket_key]:
    old_env_id = env_buckets[bucket_key].pop()
    new_envs[env_id] = {"env_instance": self.envs[old_env_id], "seed": seed}
else:
    new_envs[env_id] = {"env_cls": ..., "config_instance": ...}
```

**Relevance to us:** We already do controller caching in `SimulatorPool`, but ERA's approach is more systematic — it matches by full config hash rather than just reusing the most recently freed slot.

---

## 6. Service-Based Environment Rollout

ERA supports decoupling environment simulation from the training nodes:

```yaml
rollout_manager:
  use_service: True
  base_url: http://localhost:5000
  timeout: 1200
  max_workers: 8
```

When `use_service=True`, `QwenVLRolloutManagerService` sends actions to a remote `BatchEnvClient` over HTTP. This allows:
- Running heavy environments (Unity/AI2Thor) on separate CPU-rich nodes
- Scaling environment instances independently of GPU count
- Avoiding GPU memory contention between environments and training

**Relevance to us:** Our simulators currently share GPU memory with the training process (time-sharing). A service-based approach could eliminate the simulator↔training memory contention that forces us to use `offload_optimizer=True`.

---

## 7. Gradient Norm Threshold

ERA adds a safety valve for gradient explosions:

```yaml
actor:
  grad_norm_threshold: 1000
```

If the gradient norm exceeds this threshold, the optimizer step is **skipped entirely** (not just clipped). This prevents catastrophic updates from rare degenerate batches.

**Relevance to us:** We use `grad_clip=1.0` but don't skip updates. With multi-turn trajectories that can have highly variable lengths and reward distributions, a skip threshold could prevent occasional bad updates.

---

## 8. No Reference Policy by Default

```yaml
actor:
  use_kl_loss: False
```

ERA disables KL divergence penalty entirely. No reference model is loaded, saving ~4B params worth of GPU memory.

**Relevance to us:** We already do this in DAPO config (no KL penalty). Confirmed this is the right choice for embodied RL.

---

## 9. Key Config Differences

| Parameter | ERA Default | Our EasyR1 | Notes |
|-----------|-------------|------------|-------|
| Trajectory format | Single sequence + loss_mask | Split per-step samples | ERA's biggest win |
| `max_trajectory_length` | 3072 | N/A (per-step `max_prompt_length`) | ERA caps total sequence length |
| `ppo_mini_batch_size` | 256 | 2 per device | ERA has fewer, longer samples |
| `window_size` (rollout) | **1** | `max_observations` (1-20) | ERA: near-blind rollout, full-context training |
| `use_multi_turn_reward` | True | False (trajectory-level only) | ERA does per-step rewards |
| `use_loss_mask` | True | False | ERA masks obs tokens in loss |
| `use_gae_mask` | True | False | ERA masks obs tokens in advantage |
| `use_kl_loss` | False | False | Same |
| `grad_norm_threshold` | 1000 | N/A | ERA skips bad updates |
| `adv_estimator` | `gae` / `masked_gae` / etc. | `grpo` | ERA has 5+ multi-turn estimators |
| `use_remove_padding` | False (default) | `padding_free=true` | Similar concept |

---

## 10. What We Could Adopt (Prioritized)

### High Impact
1. **Whole-trajectory-as-response format** — Would eliminate our sample multiplication problem entirely. Instead of 20 optimizer steps for a 10-step trajectory, we'd have ~1. This is the #1 optimization to consider.
2. **Loss masking** — Required if we adopt #1. Only compute loss on action tokens, not observation tokens.
3. **Per-step rewards + return-based advantage** — Finer credit assignment than trajectory-level GRPO. The critic-free variants (return_based, minmax) don't add memory overhead.

### Medium Impact / Needs Thought
4. **Rollout/training context split** — ERA uses `window_size=1` during rollout but full trajectory during training. This creates an advantage misalignment (actions evaluated under limited context, trained under full context). The log prob recomputation makes PPO math valid, but advantages are semantically off-policy. Works for ERA in practice, but our image-heavy observations make full-trajectory sequences much longer — may need aggressive self-summarization or image compression to keep sequences feasible.
5. **Gradient norm threshold** — Safety valve for degenerate batches.

### Lower Priority / Already Have
6. **Environment reuse** — We already cache controllers.
7. **Service-based environments** — Requires infrastructure change. Worth exploring if memory contention remains a problem.
8. **No KL penalty** — Already doing this.

---

## Architecture Comparison

```
ERA approach:
  Rollout: window_size=1, model sees only current obs (+ 1-step summary in obs_str)
  After rollout: concatenate full trajectory as single sequence
  Log probs: RECOMPUTED on full trajectory (rollout-time log probs discarded)
  Training sample: prompt=<pad>, response=[full trajectory], loss_mask=[0,1,0,1,...,0,1]
  Batch: N trajectories (N = num_envs × n_trajectory)
  Optimizer steps: ~1-2 (N / ppo_mini_batch_size)
  Caveat: advantages are off-policy (computed from window_size=1 rollout, applied to full-context training)

Our approach:
  Rollout: each step sees up to max_observations prior images as context
  Training samples: N×depth separate (prompt=ctxᵢ, response=actᵢ) pairs
  Log probs: rollout-time log probs match training context (same prompt per sample)
  Batch: N×depth samples (e.g., 4 trajectories × 10 steps = 40 samples)
  Optimizer steps: ~20 (40 / per_device_batch=2)
  Advantage: on-policy (rollout and training see the same context per sample)
```
