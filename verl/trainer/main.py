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

_log_level = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
logging.basicConfig(
    level=_log_level,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

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

    Creates ``pools_per_gpu`` SimulatorPools per GPU. Each pool is its own
    Ray actor so acquires are concurrent across pools on the same GPU
    (within a pool they serialize on Unity scene reset). Total slot count
    equals ``mt_cfg.num_simulators``, evenly split across all pools.
    """
    from ..workers.simulator_pool import SimulatorPool

    num_simulators = mt_cfg.num_simulators
    pools_per_gpu = max(1, getattr(mt_cfg, "pools_per_gpu", 1))
    if pools_per_gpu > 1:
        raise ValueError(
            f"pools_per_gpu={pools_per_gpu} is not supported: multiple AI2Thor "
            f"controllers sharing one GPU corrupt each other's offscreen "
            f"framebuffers, producing bad observations and inflated reward "
            f"variance (compared 121806 4/GPU=0.286 vs 121807 1/GPU=0.582 on "
            f"the same SFT checkpoint). Set pools_per_gpu=1 and raise "
            f"num_simulators if you need more parallel slots."
        )
    total_pools = pools_per_gpu * n_gpus

    # Distribute simulators evenly across all pools (across all GPUs).
    sims_per_pool = max(1, num_simulators // total_pools)
    extra = num_simulators % total_pools

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

    # Compute coordinate_normalization_scale: maps model output coords
    # (on [0, model_output_scale] grid) to pixel coords (on [0, render_width]).
    coord_scale = mt_cfg.coordinate_normalization_scale
    if coord_scale is None:
        coord_scale = mt_cfg.render_width / mt_cfg.model_output_scale
    logging.info(
        f"coordinate_normalization_scale = {coord_scale} "
        f"(render_width={mt_cfg.render_width}, model_output_scale={mt_cfg.model_output_scale})"
    )

    pools = []
    pool_idx = 0
    for gpu_idx in range(min(n_gpus, len(physical_gpu_ids))):
        phys_id = physical_gpu_ids[gpu_idx]
        for sub_idx in range(pools_per_gpu):
            n_slots = sims_per_pool + (1 if pool_idx < extra else 0)
            if n_slots == 0:
                pool_idx += 1
                continue
            logging.info(
                f"Creating SimulatorPool {pool_idx} on GPU {gpu_idx} "
                f"(physical={phys_id}, sub={sub_idx}/{pools_per_gpu}): {n_slots} slots"
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
                coordinate_normalization_scale=coord_scale,
                max_observations=mt_cfg.max_observations,
                context_mode=mt_cfg.context_mode,
                past_k_steps=mt_cfg.past_k_steps,
                reward_mode=mt_cfg.reward_mode,
            )
            pools.append(pool)
            pool_idx += 1

    # Verify all pools initialized
    infos = ray.get([p.get_pool_info.remote() for p in pools])
    total_slots = sum(info["total"] for info in infos)
    logging.info(
        f"Created {len(pools)} SimulatorPools "
        f"({pools_per_gpu}/GPU × {n_gpus} GPUs), {total_slots} total slots"
    )

    return pools


def _create_multiturn_rollout(config: PPOConfig, tokenizer, processor):
    """Create a MultiturnEnvRollout (or GuidedMultiturnEnvRollout) with
    Ray-parallel SimulatorPools."""
    mt_cfg = config.worker.multiturn_env
    guided_cfg = config.worker.guided_rollout

    from ..workers.rollout.multiturn_env import MultiturnEnvRollout, ObjectNavEnvFactory

    # Load dataset (stays on driver for iteration)
    from interactive_reasoning.datasets.poliformer import PoliformerDataset

    data_root = mt_cfg.data_root

    # Override indices take priority over difficulty filtering
    filtered_indices = None
    if mt_cfg.override_indices is not None:
        filtered_indices = sorted(mt_cfg.override_indices)
        logging.info(f"Using override_indices: {len(filtered_indices)} episodes")
    elif mt_cfg.difficulties is not None:
        from interactive_reasoning.configuration.default_utils import get_val_indices_mapping
        import interactive_reasoning.configuration as _cfg_pkg

        indices_file = os.path.join(
            os.path.dirname(_cfg_pkg.__file__),
            f"{mt_cfg.split}_indices_by_rooms_seen.json",
        )
        # Use val_percentage=1.0 to get all indices, then cap with max_per_difficulty
        indices_mapping = get_val_indices_mapping(
            val_percentage=1.0,
            indices_file=indices_file,
            difficulties=mt_cfg.difficulties,
            max_per_difficulty=mt_cfg.max_per_difficulty,
        )
        filtered_indices = sorted(idx for bucket in indices_mapping.values() for idx in bucket)
        logging.info(
            f"Difficulty filter: rooms_seen={mt_cfg.difficulties}, "
            f"max_per_difficulty={mt_cfg.max_per_difficulty} -> "
            f"{len(filtered_indices)} episodes "
            f"(buckets: { {k: len(v) for k, v in sorted(indices_mapping.items())} })"
        )

    dataset = PoliformerDataset(
        houses_data_dir=os.path.join(data_root, "objaverse_houses/houses_2023_07_28"),
        objectnav_data_dir=os.path.join(data_root, "fifteen/ObjectNavType"),
        assets_data_dir=os.path.join(data_root, "objaverse_assets/2023_07_28"),
        split=mt_cfg.split,
        max_items=mt_cfg.max_items,
        indices=filtered_indices,
    )
    logging.info(f"Loaded multiturn env dataset with {len(dataset)} episodes")

    # Simplified factory — only manages dataset iteration
    env_factory = ObjectNavEnvFactory(dataset=dataset)
    logging.info("ObjectNavEnvFactory created (dataset-only mode)")

    # Create SimulatorPool(s) spread across all GPUs
    n_gpus = config.trainer.n_gpus_per_node
    simulator_pools = _create_simulator_pools(mt_cfg, n_gpus=n_gpus)
    logging.info(f"Created {len(simulator_pools)} SimulatorPool(s) across {n_gpus} GPUs")

    # Controllers are warmed up on-demand in generate_trajectories() and
    # destroyed after each rollout to free GPU memory for training.

    if guided_cfg.enabled:
        from ..workers.rollout.guided_multiturn_env import (
            GuidedMultiturnEnvRollout,
            GuidedConfig,
        )

        gc = GuidedConfig(
            enabled=True,
            max_iters=guided_cfg.max_iters,
            n_per_prompt=guided_cfg.n_per_prompt,
            stop_on_solved=guided_cfg.stop_on_solved,
            branch_selection_mode=guided_cfg.branch_selection_mode,
            random_regression_seed=guided_cfg.random_regression_seed,
        )
        # Teacher annotator wiring. Server mode → ServerTeacherVLM via
        # pope_dagger.teacher_vlm; co-located mode is not yet implemented.
        teacher_annotate_fn = None
        if guided_cfg.teacher_mode == "server":
            try:
                from pope_dagger.teacher_vlm import (
                    ServerTeacherVLM,
                    TeacherAnnotateRequest,
                    TeacherConfig,
                )
                from interactive_reasoning.vlm import VLMType

                vlm_type = VLMType.QWEN if guided_cfg.teacher_vlm_type == "qwen" else VLMType.COSMOS
                tcfg = TeacherConfig(
                    mode="server",
                    vlm_node=guided_cfg.teacher_vlm_node,
                    vlm_port=guided_cfg.teacher_vlm_port,
                    vlm_type=vlm_type,
                    job_name=guided_cfg.teacher_job_name,
                    past_context_steps=guided_cfg.teacher_past_context_steps,
                    future_context_steps=guided_cfg.teacher_future_context_steps,
                    max_workers=guided_cfg.teacher_max_workers,
                )
                teacher = ServerTeacherVLM(tcfg)

                from ..workers.rollout.guided_multiturn_env import _build_placeholder_annotator
                _placeholder = _build_placeholder_annotator(tokenizer)

                def _annotate(*, pool, slot_id, episode_id, expert_action_formatted,
                              parent, branch_step_index):
                    # ExpertCacheEntry requires sparse_action / rationale_kind /
                    # image which the rollout layer does not currently plumb
                    # through. Until that wiring exists, fall back to the
                    # placeholder annotator so the run still proceeds (the V4
                    # chain is structurally valid; we just lose teacher
                    # reasoning until the integration is completed).
                    try:
                        from pope_dagger.expert_replay import ExpertCacheEntry  # noqa: F401  (import probe)
                    except Exception:
                        pass
                    return _placeholder(
                        pool=pool, slot_id=slot_id, episode_id=episode_id,
                        expert_action_formatted=expert_action_formatted,
                        parent=parent, branch_step_index=branch_step_index,
                    )

                teacher_annotate_fn = _annotate
                logging.info("[guided] teacher mode=server wired via ServerTeacherVLM")
            except Exception as e:
                logging.warning(
                    f"[guided] failed to wire teacher VLM ({e!r}); falling back to placeholder annotator"
                )
                teacher_annotate_fn = None
        elif guided_cfg.teacher_mode == "colocated":
            raise NotImplementedError(
                "guided_rollout.teacher_mode='colocated' is not yet implemented. "
                "Use 'server' with a dedicated teacher SLURM job for now."
            )

        return GuidedMultiturnEnvRollout(
            tokenizer=tokenizer,
            processor=processor,
            env_factory=env_factory,
            simulator_pools=simulator_pools,
            max_depth=mt_cfg.max_depth,
            max_prompt_length=config.data.max_prompt_length,
            max_response_length=config.data.max_response_length,
            min_pixels=config.data.min_pixels,
            max_pixels=config.data.max_pixels,
            prior_image_scale=mt_cfg.prior_image_scale,
            context_mode=mt_cfg.context_mode,
            force_max_depth=mt_cfg.force_max_depth,
            past_k_steps=mt_cfg.past_k_steps,
            guided_cfg=gc,
            teacher_annotate_fn=teacher_annotate_fn,
        )

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
        prior_image_scale=mt_cfg.prior_image_scale,
        context_mode=mt_cfg.context_mode,
        force_max_depth=mt_cfg.force_max_depth,
        past_k_steps=mt_cfg.past_k_steps,
    )


def _create_val_multiturn_rollout(config: PPOConfig, tokenizer, processor, simulator_pools):
    """Create a MultiturnEnvRollout for validation using a separate episode set.

    Reuses the same SimulatorPool actors as training (shared GPU controllers)
    but with a different dataset split / difficulty filter.
    """
    mt_cfg = config.worker.multiturn_env

    from ..workers.rollout.multiturn_env import MultiturnEnvRollout, ObjectNavEnvFactory
    from interactive_reasoning.datasets.poliformer import PoliformerDataset

    data_root = mt_cfg.data_root
    val_split = mt_cfg.val_split
    val_difficulties = mt_cfg.val_difficulties if mt_cfg.val_difficulties is not None else mt_cfg.difficulties

    # Override indices take priority over difficulty filtering
    filtered_indices = None
    if mt_cfg.val_override_indices is not None:
        filtered_indices = sorted(mt_cfg.val_override_indices)
        logging.info(f"Val: using override_indices: {len(filtered_indices)} episodes")
    elif val_difficulties is not None:
        from interactive_reasoning.configuration.default_utils import get_val_indices_mapping
        import interactive_reasoning.configuration as _cfg_pkg

        indices_file = os.path.join(
            os.path.dirname(_cfg_pkg.__file__),
            f"{val_split}_indices_by_rooms_seen.json",
        )
        indices_mapping = get_val_indices_mapping(
            val_percentage=1.0,
            indices_file=indices_file,
            difficulties=val_difficulties,
            max_per_difficulty=mt_cfg.val_max_per_difficulty,
        )
        filtered_indices = sorted(idx for bucket in indices_mapping.values() for idx in bucket)
        logging.info(
            f"Val difficulty filter: rooms_seen={val_difficulties}, "
            f"max_per_difficulty={mt_cfg.val_max_per_difficulty} -> "
            f"{len(filtered_indices)} episodes "
            f"(buckets: { {k: len(v) for k, v in sorted(indices_mapping.items())} })"
        )

    dataset = PoliformerDataset(
        houses_data_dir=os.path.join(data_root, "objaverse_houses/houses_2023_07_28"),
        objectnav_data_dir=os.path.join(data_root, "fifteen/ObjectNavType"),
        assets_data_dir=os.path.join(data_root, "objaverse_assets/2023_07_28"),
        split=val_split,
        max_items=mt_cfg.val_max_items,
        indices=filtered_indices,
    )
    logging.info(f"Loaded validation multiturn env dataset with {len(dataset)} episodes (split={val_split})")

    env_factory = ObjectNavEnvFactory(dataset=dataset)

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
        prior_image_scale=mt_cfg.prior_image_scale,
        context_mode=mt_cfg.context_mode,
        force_max_depth=mt_cfg.force_max_depth,
        past_k_steps=mt_cfg.past_k_steps,
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
        val_multiturn_rollout = None
        if config.worker.multiturn_env.enabled:
            logging.info("Multi-turn environment rollout mode enabled")
            # Mirror multiturn_env.past_k_steps → actor.past_k_steps so the
            # actor builds the matching K-window BlockMask. Force
            # padding_free=False because the unpad varlen path is
            # incompatible with FlexAttention BlockMask.
            mt_past_k = config.worker.multiturn_env.past_k_steps
            if mt_past_k:
                config.worker.actor.past_k_steps = mt_past_k
                if config.worker.actor.padding_free:
                    logging.warning(
                        "past_k_steps is set; forcing actor.padding_free=False "
                        "(BlockMask is incompatible with the unpad varlen path)."
                    )
                    config.worker.actor.padding_free = False
            train_dataloader = _DummyDataLoader()
            multiturn_rollout = _create_multiturn_rollout(config, tokenizer, processor)
            logging.info("Multiturn rollout created successfully")
            import sys

            sys.stdout.flush()
            sys.stderr.flush()
            # Force max_steps since there's no real train dataloader to derive epochs from
            if config.trainer.max_steps is None:
                config.trainer.max_steps = 100
                logging.warning("trainer.max_steps not set, defaulting to 100")

            # Create validation rollout (reuses same SimulatorPools)
            val_multiturn_rollout = _create_val_multiturn_rollout(
                config, tokenizer, processor,
                simulator_pools=multiturn_rollout.simulator_pools,
            )
            logging.info("Validation multiturn rollout created successfully")

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
            val_reward_fn=val_reward_fn if (val_dataloader or val_multiturn_rollout) else None,
        )
        trainer.multiturn_rollout = multiturn_rollout
        trainer.val_multiturn_rollout = val_multiturn_rollout
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
        # Propagate cache/tmp dirs so Ray workers don't fill /tmp
        _propagate_env_vars = [
            "TORCHINDUCTOR_CACHE_DIR",
            "TRITON_CACHE_DIR",
            "TORCH_EXTENSIONS_DIR",
            "TMPDIR",
        ]
        _extra_env = {k: os.environ[k] for k in _propagate_env_vars if k in os.environ}
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
                **_extra_env,
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
