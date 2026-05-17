# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

EasyR1 is a fork of [veRL](https://github.com/volcengine/verl) for multi-modal RL training (GRPO, DAPO, Reinforce++, etc.) of VLMs. It uses Ray for distributed training with a HybridEngine design: the same GPU pool hosts both the vLLM rollout engine and the FSDP training process, swapping between them each iteration.

This submodule is used by the parent `spatial-reasoning` project for online GRPO training of ObjectNav agents in AI2Thor environments.

## Running Training

```bash
# Standard dataset-based training (single-node example)
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    worker.actor.model.model_path=Qwen/Qwen3-VL-4B-Instruct \
    trainer.experiment_name=my_exp \
    trainer.n_gpus_per_node=4

# Multi-turn env mode (online AI2Thor rollouts — spatial-reasoning integration)
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    worker.multiturn_env.enabled=true \
    worker.actor.model.model_path=<checkpoint_path> \
    trainer.max_steps=200

# Merge checkpoint to HuggingFace format after training
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/<exp>/global_step_1/actor
```

## Tests and Code Quality

```bash
# Run tests
pytest -vv tests/

# Run a single test file
pytest -vv tests/test_dataset.py

# Lint and format
make style   # ruff check + format (auto-fix)
make quality # ruff check + format (check only)
```

## Configuration System

Config is OmegaConf-structured. The base config `examples/config.yaml` is overridden via CLI dotpath keys:

```
python3 -m verl.trainer.main config=examples/config.yaml key.subkey=value ...
```

Key config sections (`verl/trainer/config.py`, `verl/workers/config.py`):
- **`data`** — `DataConfig`: train/val files (HF dataset or local parquet), prompt/answer/image keys, `max_prompt_length`, `max_response_length`, `rollout_batch_size`, `format_prompt` (Jinja2 template path), pixel limits
- **`algorithm`** — `AlgorithmConfig`: `adv_estimator` (grpo/gae/reinforce_plus_plus/remax/rloo), KL settings, `online_filtering`
- **`worker.actor`** — `ActorConfig`: model path, LoRA rank, FSDP settings, optimizer, `micro_batch_size_per_device_for_update`
- **`worker.rollout`** — `RolloutConfig`: `n` (generations per prompt), temperature, `tensor_parallel_size`, `gpu_memory_utilization`
- **`worker.reward`** — `RewardConfig`: `reward_function` path (`module.py:function_name`)
- **`worker.multiturn_env`** — `MultiturnEnvConfig`: online env rollout mode (see below)
- **`trainer`** — `TrainerConfig`: epochs/steps, logging, checkpoint frequency, `n_gpus_per_node`, `nnodes`

## Architecture

### Training Loop (`verl/trainer/ray_trainer.py`)

`RayPPOTrainer.fit()` orchestrates the GRPO loop:
1. **Rollout** — vLLM generates `n` responses per prompt (via `FSDPWorker` in rollout mode)
2. **Reward** — `AutoRewardManager` calls the reward function module
3. **Advantage** — computed in `core_algos.py` (GRPO normalizes within each group of `n`)
4. **Actor update** — FSDP backward pass with KL penalty

### Worker Architecture

All workers are Ray actors: `FSDPWorker` handles Actor, Ref, and Critic roles (via `role` switching). `RayWorkerGroup` manages the worker fleet. The sharding manager (`verl/workers/sharding_manager/`) converts FSDP weights to/from vLLM format between rollout and training phases.

### Multi-Turn Environment Mode (spatial-reasoning integration)

When `worker.multiturn_env.enabled=true`, standard dataset-based rollout is replaced with live AI2Thor trajectories:

- **`verl/workers/rollout/multiturn_env.py`** — `MultiturnEnvRollout` drives multi-turn episodes: calls `actor_rollout_ref_wg.generate_sequences()` at each step, sends actions to environment, collects rewards
- **`verl/workers/simulator_pool.py`** — `SimulatorPool` (Ray actor) manages a pool of `ObjectNavEnvAdapter` slots on one GPU. Controllers are cached and reused across episodes (avoids ~30-100s Unity startup cost)
- **`verl/trainer/main.py`** — `_create_simulator_pools()` distributes simulator slots across GPUs; `_create_multiturn_rollout()` wires everything together

Key `MultiturnEnvConfig` fields:
- `num_simulators` — total AI2Thor slots (should be >= `rollout_batch_size * n`)
- `pools_per_gpu` — number of `SimulatorPool` Ray actors per GPU. **Must be 1** — `_create_simulator_pools` hard-fails on `>1`. See "pools_per_gpu cliff" below.
- `max_depth` — max steps per trajectory
- `data_root` — path to Poliformer dataset
- `system_prompt_path` — path to the system prompt `.txt` file
- `difficulties` / `max_per_difficulty` — filter episodes by `rooms_seen` difficulty

#### pools_per_gpu cliff (hard-failed at >1)

`pools_per_gpu` is a Ray-actor **concurrency** knob, not a slot-count knob. `_create_simulator_pools` builds `pools_per_gpu * n_gpus` total `SimulatorPool` actors and splits `num_simulators` evenly across them. Each pool is one Ray actor with default `max_concurrency=1`, so step calls within a pool serialize.

- `pools_per_gpu=1` → 1 actor per GPU → AI2Thor `step` calls on that GPU run **one at a time** (controllers render serially).
- `pools_per_gpu>1` → multiple actors per GPU → up to `pools_per_gpu` Unity controllers render **simultaneously** on a GPU that is also hosting vLLM under the HybridEngine.

Concurrent Unity rendering on a vLLM-colocated GPU silently corrupts trajectories: AI2Thor controllers sharing one GPU clobber each other's offscreen framebuffers, returning stale / partially-rendered frames to the model. The model then navigates from a degraded observation, wanders, and either hits `max_depth` or the per-action watchdog fires (`step_env` raises, episode terminated with `reward=0`).

Measured cliff on SFT-1500, `val_only`, 128 episodes (d=0), same slots/GPU (16):

| pools/GPU | success | avg_steps |
|-----------|---------|-----------|
| 1 (serial) | 43.8% | 5.84 |
| 4 (concurrent) | 14.8% | 6.77 |

`avg_steps` going **up** under concurrency (not down) rules out "pure timeout failures" — episodes complete but complete wrong. Per-step wall time was actually ~2× faster on the 4-pool run for trajectories that did finish, so the cliff is observation quality, not throughput.

Scale slot count via `num_simulators` only (more slots under one actor; serialized wall time). To prevent silent regressions, `_create_simulator_pools` raises `ValueError` on `pools_per_gpu > 1`.

### Reward Functions (`examples/reward_function/`)

Each reward module exports `REWARD_NAME`, `REWARD_TYPE` (`"sequential"` or `"batch"`), and a scoring function. The `batch` type receives `list[dict]` with `response` and `ground_truth` keys and returns `list[dict]` with `overall` + component scores.

The ObjectNav reward (`examples/reward_function/objectnav.py`) supports:
- **Offline** (SFT-derived data): compares action type + coordinates against expert
- **Online** (live env): uses `distance_to_target_before/after` and `target_2d_coords`
- **Trajectory-level**: `trajectory_reward` key bypasses per-step scoring

### Dataset Format

HuggingFace parquet datasets with columns: `prompt` (list of chat messages with optional images), `answer` (ground truth string). Format prompts (Jinja2 in `examples/format_prompt/`) wrap the raw problem into the chat template.

Multi-image datasets embed images directly in the `prompt` messages. The `data.image_dir` config points to a local image root if images are stored as paths rather than embedded bytes.

## Key File Locations

| Path | Purpose |
|------|---------|
| `verl/trainer/main.py` | Entry point; Ray initialization; multi-turn env setup |
| `verl/trainer/ray_trainer.py` | `RayPPOTrainer` — main training loop |
| `verl/trainer/config.py` | `PPOConfig`, `DataConfig`, `TrainerConfig`, `AlgorithmConfig` |
| `verl/workers/config.py` | `WorkerConfig` (actor/rollout/reward/multiturn_env) |
| `verl/workers/rollout/config.py` | `RolloutConfig`, `MultiturnEnvConfig` |
| `verl/workers/rollout/multiturn_env.py` | Online AI2Thor multi-turn rollout driver |
| `verl/workers/simulator_pool.py` | Ray-remote AI2Thor simulator pool |
| `verl/workers/fsdp_workers.py` | `FSDPWorker` — actor/ref/critic Ray worker |
| `verl/trainer/core_algos.py` | GRPO/GAE advantage estimation |
| `examples/config.yaml` | Default base config |
| `examples/reward_function/objectnav.py` | ObjectNav reward function |
| `examples/format_prompt/objectnav.jinja` | ObjectNav prompt template |

## Common Issues

- **"Image features and image tokens do not match"** — increase `data.max_prompt_length` or reduce `data.max_pixels`
- **CUDA OOM** — reduce `worker.rollout.gpu_memory_utilization` and enable `worker.actor.offload.offload_params=true`
- **"0 active drivers"** — uninstall `deepspeed` from the environment (conflicts with vLLM)
- **Multi-node training** — start Ray head (`ray start --head`), connect workers (`ray start --address=<head>:6379`), then run training script only on head node
