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

import json
import logging
import os
import time

import ray
from omegaconf import OmegaConf


from ..single_controller.ray import RayWorkerGroup
from ..utils.tokenizer import get_processor, get_tokenizer
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import AutoRewardManager
from .config import PPOConfig
from .data_loader import create_dataloader
from .ray_trainer import RayPPOTrainer, ResourcePoolManager, Role

# logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", force=True)
logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", force=True)

# Quiet down noisy loggers from environment internals
for _quiet in ("interactive_reasoning.objectnavtask.environment", "ai2thor"):
    logging.getLogger(_quiet).setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Dummy dataloader for online env mode (training data comes from env rollouts)
# ---------------------------------------------------------------------------

class _DummyDataLoader:
    def __len__(self):
        return 1

    def __iter__(self):
        return iter([])

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


def _create_simulator_pools(mt_cfg, n_gpus: int):
    """Create Ray-remote SimulatorPool actors spread across all GPUs.

    Creates one SimulatorPool per GPU, each managing a share of the total
    ``num_simulators`` slots. AI2Thor controllers share GPU memory with
    the model (low gpu_memory_utilization leaves room).
    """
    from ..workers.simulator_pool import SimulatorPool

    num_simulators = mt_cfg.num_simulators
    # Distribute simulators evenly across GPUs
    sims_per_gpu = max(1, num_simulators // n_gpus)
    extra = num_simulators % n_gpus

    # Map logical GPU indices to physical device IDs from SLURM
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        physical_gpu_ids = [int(x) for x in cuda_visible.split(",")]
    else:
        physical_gpu_ids = list(range(n_gpus))
    logging.info(f"Physical GPU IDs: {physical_gpu_ids}")

    # Load system prompt
    with open(mt_cfg.system_prompt_path) as f:
        system_prompt = f.read().strip()

    pools = []
    for gpu_idx in range(min(n_gpus, len(physical_gpu_ids))):
        n_slots = sims_per_gpu + (1 if gpu_idx < extra else 0)
        if n_slots == 0:
            continue

        phys_id = physical_gpu_ids[gpu_idx]
        logging.info(
            f"Creating SimulatorPool on GPU {gpu_idx} "
            f"(physical={phys_id}): {n_slots} slots"
        )

        pool = SimulatorPool.options(
            runtime_env={"env_vars": {"CUDA_VISIBLE_DEVICES": str(phys_id)}},
        ).remote(
            gpu_id=phys_id,
            num_slots=n_slots,
            system_prompt=system_prompt,
            render_width=mt_cfg.render_width,
            render_height=mt_cfg.render_height,
            max_depth=mt_cfg.max_depth,
            coordinate_normalization_scale=mt_cfg.coordinate_normalization_scale,
            max_observations=mt_cfg.max_observations,
        )
        pools.append(pool)

    # Verify all pools initialized
    infos = ray.get([p.get_pool_info.remote() for p in pools])
    total_slots = sum(info['total'] for info in infos)
    logging.info(
        f"Created {len(pools)} SimulatorPools across {n_gpus} GPUs, "
        f"{total_slots} total slots"
    )

    return pools


def _create_multiturn_rollout(config: PPOConfig, tokenizer, processor):
    """Create a MultiturnEnvRollout with Ray-parallel SimulatorPools."""
    mt_cfg = config.worker.multiturn_env

    from ..workers.rollout.multiturn_env import MultiturnEnvRollout, ObjectNavEnvFactory

    # Load dataset (stays on driver for iteration)
    from interactive_reasoning.datasets.poliformer import PoliformerDataset

    data_root = mt_cfg.data_root
    dataset = PoliformerDataset(
        houses_data_dir=os.path.join(data_root, "objaverse_houses/houses_2023_07_28"),
        objectnav_data_dir=os.path.join(data_root, "fifteen/ObjectNavType"),
        assets_data_dir=os.path.join(data_root, "objaverse_assets/2023_07_28"),
        split=mt_cfg.split,
        max_items=mt_cfg.max_items,
    )
    logging.info(f"Loaded multiturn env dataset with {len(dataset)} episodes")

    # Simplified factory — only manages dataset iteration
    env_factory = ObjectNavEnvFactory(dataset=dataset)
    logging.info("ObjectNavEnvFactory created (dataset-only mode)")

    # Create SimulatorPool(s) spread across all GPUs
    n_gpus = config.trainer.n_gpus_per_node
    simulator_pools = _create_simulator_pools(mt_cfg, n_gpus=n_gpus)
    logging.info(f"Created {len(simulator_pools)} SimulatorPool(s) across {n_gpus} GPUs")

    # Warm up AI2Thor controllers one pool at a time to avoid GPU memory stampede.
    # Uses the first dataset item's scene as a dummy scene.
    dummy_scene = dataset[0]["scene_metadata"]
    logging.info("Warming up AI2Thor controllers (staggered, one pool at a time)...")
    for i, pool in enumerate(simulator_pools):
        count = ray.get(pool.warmup_controllers.remote(dummy_scene))
        logging.info(f"  Pool {i}: warmed up {count} controller(s)")
    logging.info("All AI2Thor controllers warmed up")

    return MultiturnEnvRollout(
        tokenizer=tokenizer,
        processor=processor,
        env_factory=env_factory,
        simulator_pools=simulator_pools,
        max_depth=mt_cfg.max_depth,
        max_prompt_length=config.data.max_prompt_length,
        max_response_length=config.data.max_response_length,
        min_pixels=config.data.min_pixels,
        max_pixels=config.data.max_pixels,
    )


# please make sure main_task is not scheduled on head
@ray.remote(num_cpus=1)
class Runner:
    """A runner for RL training."""

    def run(self, config: PPOConfig):
        # print config
        print(json.dumps(config.to_dict(), indent=2))

        # instantiate tokenizer
        tokenizer = get_tokenizer(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        processor = get_processor(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )

        # define worker classes
        ray_worker_group_cls = RayWorkerGroup
        role_worker_mapping = {
            Role.ActorRolloutRef: ray.remote(FSDPWorker),
            Role.Critic: ray.remote(FSDPWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRolloutRef: global_pool_id,
            Role.Critic: global_pool_id,
        }
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        RemoteRewardManager = ray.remote(AutoRewardManager).options(num_cpus=config.worker.reward.num_cpus)
        reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)
        val_reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)

        # Multi-turn env mode: use dummy train dataloader, construct rollout
        multiturn_rollout = None
        if config.worker.multiturn_env.enabled:
            logging.info("Multi-turn environment rollout mode enabled")
            train_dataloader = _DummyDataLoader()
            multiturn_rollout = _create_multiturn_rollout(config, tokenizer, processor)
            logging.info("Multiturn rollout created successfully")
            import sys; sys.stdout.flush(); sys.stderr.flush()
            # Force max_steps since there's no real train dataloader to derive epochs from
            if config.trainer.max_steps is None:
                config.trainer.max_steps = 100
                logging.warning("trainer.max_steps not set, defaulting to 100")

            # All data comes from the environment — no train/val files needed
            val_dataloader = None
        else:
            train_dataloader, val_dataloader = create_dataloader(config.data, tokenizer, processor)

        logging.info("Creating RayPPOTrainer...")
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn if val_dataloader else None,
        )
        trainer.multiturn_rollout = multiturn_rollout
        logging.info("Calling init_workers()...")
        trainer.init_workers()
        logging.info("init_workers() complete, calling fit()...")
        trainer.fit()


def main():
    cli_args = OmegaConf.from_cli()
    default_config = OmegaConf.structured(PPOConfig())

    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)
        file_config = OmegaConf.load(config_path)
        default_config = OmegaConf.merge(default_config, file_config)

    ppo_config = OmegaConf.merge(default_config, cli_args)
    ppo_config: PPOConfig = OmegaConf.to_object(ppo_config)
    ppo_config.deep_post_init()

    if not ray.is_initialized():
        runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "VLLM_ATTENTION_BACKEND": "XFORMERS",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
            }
        }
        ray.init(runtime_env=runtime_env)

    runner = Runner.remote()
    ray.get(runner.run.remote(ppo_config))

    if ppo_config.trainer.ray_timeline is not None:
        # use `export RAY_PROFILING=1` to record the ray timeline
        ray.timeline(filename=ppo_config.trainer.ray_timeline)


if __name__ == "__main__":
    main()
