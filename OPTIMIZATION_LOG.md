# Actor Update Optimization Log

Documenting all experiments to reduce `update_actor` time in multi-turn GRPO/DAPO training. The actor update is the bottleneck: multi-turn trajectories expand to many (prompt, response) training samples, and each optimizer step is expensive (~57s with FSDP all-reduce + CPU-offloaded AdamW on a 4B model).

**Setup:** 2× GPUs (44–49 GB), Qwen3-VL-4B, `max_depth=20`, `rollout_batch_size=2`, `n=2` → 4 trajectories per step, expanding to 24–76 training samples depending on trajectory length.

---

## Root Cause Analysis

The actor update loop has three nesting levels (see `dp_actor.py:update_policy`):

```
for epoch in range(ppo_epochs):                          # ppo_epochs (default 1)
    for mini_batch in data.split(GBS_per_device):        # global_batch_size controls optimizer steps
        for micro_batch in dynamic_split(mini_batch):    # micro_batch_size controls forward/backward
            forward(micro_batch)
            loss.backward()                              # gradients accumulate
        optimizer.step()                                 # weight update — EXPENSIVE
        zero_grad()
```

**Cost breakdown** (34 samples, `per_device=2`, `offload_optimizer=true`):
- Forward/backward: 34 × ~5.5s = ~187s
- Optimizer steps: 17 × ~57s = ~969s ← **dominant cost**
- Optimizer load/offload (CPU↔GPU): ~30s (once per update_actor)
- **Total: ~1157s**

Each `optimizer.step()` costs ~57s because:
1. FSDP gradient all-reduce across GPUs
2. Gradient norm clipping (requires another all-reduce)
3. AdamW update on all 4B parameters
4. With `offload_optimizer`, AdamW runs on GPU but states are loaded from CPU at start

---

## Experiments

### 1. Increase `global_batch_size` (reduce optimizer steps)

**Idea:** Larger `global_batch_size_per_device` → fewer mini-batches → fewer optimizer steps.

| Config | per_device | Samples | Opt steps | update_actor | Result |
|--------|-----------|---------|-----------|-------------|--------|
| baseline | 2 | 34 | 17 | 1157s | Works |
| baseline | 2 | 24 | 12 | 402.6s | Works |
| baseline | 2 | 74 | 37 | 2109.9s | Works (linear scaling) |
| GBS=128 | 128 | 34 | 1 | 237.6s | Works — **4.9× faster** |
| GBS=128 | 128 | 36 | 1 | — | **OOM** during loss.backward() |
| GBS=8 | 8 | 42 | ~6 | — | **OOM** during loss.backward() |
| GBS=4 | 4 | 42 | ~11 | — | **OOM** during loss.backward() |

**Why OOM:** The entire dataset is moved to GPU via `data.to(cuda)` before the mini-batch loop. With multi-modal data (images), 42 samples' pixel_values + input_ids + attention_mask + position_ids consume several GB. Larger mini-batches require more activation memory during backward even with `micro_batch_size=1`.

**Conclusion:** Cannot reliably increase beyond `per_device=2` on 44–49 GB GPUs with multi-modal multi-turn data.

**Code change:** Relaxed the `rollout_batch_size % global_batch_size` validation check for multi-turn mode in `ray_trainer.py:214` (multi-turn expansion makes this constraint meaningless).

### 2. Disable optimizer offload

**Idea:** Keep optimizer states on GPU to avoid CPU↔GPU transfer overhead and enable faster `optimizer.step()`.

| Config | Samples | update_actor | Result |
|--------|---------|-------------|--------|
| offload=false | 26 | 274.0s | Works for step 1 |
| offload=false | — | — | **OOM on rollout** (step 2) |

**Why OOM:** Optimizer states (~8 GB for 4B model with FSDP/2) stay on GPU, leaving insufficient memory for vLLM + AI2Thor controllers during the next rollout phase.

**Conclusion:** Cannot disable optimizer offload — the shared GPU design requires freeing optimizer memory during rollout.

### 3. Reduce `max_observations` (fewer images per prompt)

**Idea:** Cap the number of prior observation images included in each training prompt. Fewer images → shorter prompts → less pixel_values memory → faster forward/backward.

| max_observations | Effect |
|-----------------|--------|
| 20 (default) | Long prompts, many images, OOM-prone |
| 1 | **"Works a charm"** — dramatically reduces prompt length and memory |

**Conclusion:** Effective but limits the model's ability to use observation history. Need to find a balance (3–5 observations) once other optimizations are in place.

### 4. Gradient accumulation across mini-batches (current approach)

**Idea:** Keep `per_device=2` (low memory) but only call `optimizer.step()` every K mini-batches instead of every mini-batch. This reduces optimizer step count without increasing peak activation memory.

**Implementation** (`dp_actor.py:update_policy`):
- Added `gradient_accumulation_steps` config parameter (default: 1, no behavior change)
- Loss scaled by `1/group_size` to normalize across accumulated mini-batches
- `optimizer.step()` only fires at end of each accumulation group

**Expected effect** (42 samples, `gradient_accumulation_steps=10`):
- Mini-batches: 21 (unchanged)
- Optimizer steps: ceil(21/10) = 3 (down from 21)
- Saved: ~18 × 57s = **~1026s**
- Expected total: ~231s (fwd/bwd) + ~171s (3 opt steps) + ~30s (load/offload) = **~432s**

**Config:** `worker.actor.gradient_accumulation_steps: 10`

### 5. Lazy GPU data transfer

**Idea:** Instead of `data.to(cuda)` (loading ALL training samples to GPU at once), keep data on CPU and move each mini-batch to GPU on demand. Frees several GB on GPU.

**Implementation:**
- `fsdp_workers.py:update_actor` — replaced `data.to(cuda)` with `data.meta_info["lazy_to_device"] = device`
- `dp_actor.py:update_policy` — each mini-batch is `.to(device)` at start of iteration, `del`'d after processing
- Only affects `update_actor`; `compute_log_probs` still does bulk `data.to(cuda)` (inference-only, no backward)

**Expected effect:** Saves ~2–5 GB GPU memory depending on number of training samples. Should prevent OOM on 44 GB GPUs that previously failed with the same config on 47+ GB GPUs.

---

## Summary of Current State

**Active optimizations:**
- `max_observations=1` (test value — to be increased once stable)
- `gradient_accumulation_steps=10` (reduces optimizer steps from ~20 to ~2-3)
- Lazy GPU data transfer (only current mini-batch on GPU)
- `padding_free=true` + `dynamic_batching=true` (existing — saves memory on variable-length sequences)
- `enable_gradient_checkpointing=true` (existing — trades compute for memory)
- `offload_optimizer=true` (existing — saves ~8 GB GPU for optimizer states)

**Not viable:**
- Increasing `global_batch_size` beyond `per_device=2` (OOM)
- Disabling optimizer offload (OOM during rollout)

**Still to explore:**
- Moderate `max_observations` (3–5) with gradient accumulation + lazy transfer
- Reducing `max_pixels` per image (currently 1003520) to allow more observations
- Reducing `prior_image_scale` (currently 0.5) for smaller images
- LoRA to reduce optimizer state size (rank > 0 enables AdamW on much fewer params)
