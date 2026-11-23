"""Native verl 0.9.1 engine/server entrypoint; never calls vendor training code.

We retain the release's DataProto RayPPOTrainer (use_v1=False), not the vendor
fit loop, and use its native asynchronous vLLM server/weight transfer lifecycle.
"""
from pathlib import Path

from omegaconf import OmegaConf


def build_native_config(legacy, args, config_dir=None):
    """Compose the release schema before applying explicit experiment parameters."""
    from hydra import compose, initialize_config_dir
    if config_dir is None:
        import verl
        config_dir = Path(verl.__file__).parent / "trainer" / "config"
    with initialize_config_dir(config_dir=str(Path(config_dir).resolve()), version_base=None):
        native = compose(config_name="ppo_trainer", overrides=["model_engine=dp"])
    patch = OmegaConf.to_container(legacy, resolve=True) if OmegaConf.is_config(legacy) else dict(legacy)
    # Native reward loop owns reward execution. Preserve the exact custom function.
    patch = OmegaConf.to_container(OmegaConf.create(patch), resolve=True)
    custom_reward = patch.pop("custom_reward_function", None)
    old_reward = patch.pop("reward_model", {})
    if old_reward.get("enable", False):
        raise ValueError("LoopRL native migration does not allocate a reward model")
    patch.setdefault("reward", {}).setdefault("reward_model", {})["enable"] = False
    if custom_reward:
        patch["reward"]["custom_reward_function"] = custom_reward
    trainer = patch.setdefault("trainer", {})
    accumulation = trainer.pop("gradient_accumulation_steps", 1)
    if accumulation != 1:
        raise ValueError("Native trainer does not reproduce vendor cross-rollout gradient accumulation")
    trainer["use_v1"] = False
    ar = patch.setdefault("actor_rollout_ref", {})
    audit = ar.pop("loop_audit", None)
    if audit:
        ar.setdefault("looprl", {})["audit"] = audit
    ar.pop("loss_agg", None)  # transferred explicitly into ar.looprl by launch
    ar.setdefault("actor", {}).pop("plc", None)
    ar["actor"].pop("extrap", None)  # transferred explicitly into ar.looprl by launch
    ar["actor"].pop("opd", None)  # transferred explicitly into ar.looprl by launch
    model = ar.setdefault("model", {})
    # The Ouro adapter computes dense per-loop logits; these are implementation
    # layout constraints, not changes to objective or sequence masking.
    model["use_remove_padding"] = False
    model["use_fused_kernels"] = False
    ar["actor"]["use_remove_padding"] = False
    if ar["actor"].get("ulysses_sequence_parallel_size", 1) != 1:
        raise ValueError("Native Ouro engine requires ulysses_sequence_parallel_size=1")
    if ar.get("looprl", {}).get("loop_weighting") == "learned":
        raise ValueError("Learned loop weights require an optimizer/checkpoint migration")
    rollout = ar.setdefault("rollout", {})
    if rollout.get("name", "vllm") != "vllm":
        raise ValueError("Native LoopRL requires vLLM server rollouts")
    rollout.update(name="vllm", mode="async")
    from rltt_baseline.decode_last import ARCHITECTURE
    from rltt_baseline.loop_cd import ARCHITECTURE as LOOP_CD_ARCHITECTURE
    engine = rollout.get("engine_kwargs", {}).get("vllm", {})
    architectures = engine.get("hf_overrides", {}).get("architectures", [])
    for arch, name in ((ARCHITECTURE, "decode_last"), (LOOP_CD_ARCHITECTURE, "extrap (LoopCD)")):
        if arch not in architectures:
            continue
        # vLLM 0.24 selects V2 through an environment flag, not EngineArgs.
        env = patch.setdefault("ray_kwargs", {}).setdefault("ray_init", {}).setdefault(
            "runtime_env", {}).setdefault("env_vars", {})
        if env.get("VLLM_USE_V2_MODEL_RUNNER", "0") != "0":
            raise ValueError(f"{name} requires VLLM_USE_V2_MODEL_RUNNER=0")
        env["VLLM_USE_V2_MODEL_RUNNER"] = "0"

    patch.setdefault("data", {})["train_files"] = [str(args.train_file)]
    # A validation file must be supplied explicitly; never silently evaluate training data.
    val_files = patch["data"].get("val_files")
    if not val_files:
        raise ValueError("verl 0.9.1 requires explicit verl_overrides.data.val_files (Parquet)")
    OmegaConf.set_struct(native, False)
    native = OmegaConf.merge(native, patch)
    OmegaConf.resolve(native)
    return native


def run(config):
    """Run the native trainer with the LoopRL engine worker role mapping."""
    import ray
    from verl.trainer.main_ppo import run_ppo
    from verl.trainer.main_ppo_v0 import BaseTaskRunner
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role
    from verl.trainer.ppo.utils import create_rl_dataset, create_rl_sampler, need_critic, need_reference_policy
    from verl.single_controller.ray import RayWorkerGroup
    from verl.utils.config import omega_conf_to_dataclass, validate_config
    from verl.utils.dataset.rl_dataset import collate_fn
    from verl.workers.config import HFModelConfig
    from rltt_baseline.verl09_workers import LoopRLActorRolloutRefWorker

    from verl import DataProto
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

    class LoopRLTrainer(RayPPOTrainer):
        def _compute_old_log_prob(self, batch):
            td = left_right_2_no_padding(batch.to_tensordict())
            calculate_sum = self.config.actor_rollout_ref.actor.get("calculate_sum_pi_squared", False)
            tu.assign_non_tensor(td, global_steps=self.global_steps,
                                 temperature=self.config.actor_rollout_ref.rollout.temperature,
                                 calculate_entropy=True, calculate_sum_pi_squared=calculate_sum,
                                 compute_loss=False)
            output = self.actor_rollout_wg.compute_log_prob(td)
            result = {}
            keys = {"log_probs": "old_log_probs", "entropy": "entropys",
                    "per_loop_log_probs": "old_log_probs_loops",
                    "init_log_probs_loops": "init_log_probs_loops",
                    "teacher_log_probs": "teacher_log_probs"}
            if calculate_sum:
                keys["sum_pi_squared"] = "sum_pi_squared"
            for source, dest in keys.items():
                value = tu.get(output, source)
                if value is not None:
                    result[dest] = no_padding_2_padding(value, td).float()
            routed = tu.get(output, "routed_experts")
            if routed is not None:
                result["routed_experts"] = routed
            method = self.config.actor_rollout_ref.looprl.method
            if method in ("plc", "rltt", "extrap", "opd"):
                if "old_log_probs_loops" not in result:
                    raise RuntimeError("LoopRL worker did not return per-loop old policy log probabilities")
            opd_cfg = self.config.actor_rollout_ref.looprl.get("opd") or {}
            sampled = opd_cfg.get("estimator", "sampled") != "full"  # sampled, k2
            if method == "opd" and sampled and "teacher_log_probs" not in result:
                raise RuntimeError("OPD needs the frozen teacher's log probabilities "
                                   "(teacher_log_probs)")
            return DataProto.from_tensordict(tu.get_tensordict(result)), tu.get(output, "metrics")["mfu"]

    class LoopRLTaskRunner(BaseTaskRunner):
        def run(self, cfg):
            role = Role.ActorRolloutRef if need_reference_policy(cfg) else Role.ActorRollout
            self.role_worker_mapping[role] = ray.remote(LoopRLActorRolloutRefWorker)
            self.mapping[role] = "global_pool"
            if need_critic(cfg):
                raise ValueError("LoopRL native entrypoint implements GRPO/PLC, not a PPO critic")
            validate_config(cfg, use_reference_policy=need_reference_policy(cfg), use_critic=False)
            model: HFModelConfig = omega_conf_to_dataclass(cfg.actor_rollout_ref.model)
            train = create_rl_dataset(cfg.data.train_files, cfg.data, model.tokenizer,
                                      model.processor, is_train=True,
                                      max_samples=cfg.data.get("train_max_samples", -1))
            val = create_rl_dataset(cfg.data.val_files, cfg.data, model.tokenizer,
                                    model.processor, is_train=False,
                                    max_samples=cfg.data.get("val_max_samples", -1))
            # The native trainer caches this import at module scope.
            from rltt_baseline.metrics import install_driver_metrics
            from math_utils import extract_boxed_answer
            from verl.trainer.ppo import metric_utils, ray_trainer
            install_driver_metrics(
                decode=lambda ids: model.tokenizer.decode(ids, skip_special_tokens=True),
                extract_answer=extract_boxed_answer,
                loop_audit=cfg.actor_rollout_ref.looprl.get("audit"),
            )
            ray_trainer.compute_data_metrics = metric_utils.compute_data_metrics
            trainer = LoopRLTrainer(
                config=cfg, tokenizer=model.tokenizer, processor=model.processor,
                role_worker_mapping=self.role_worker_mapping,
                resource_pool_manager=self.init_resource_pool_mgr(cfg),
                ray_worker_group_cls=RayWorkerGroup, train_dataset=train, val_dataset=val,
                collate_fn=collate_fn, train_sampler=create_rl_sampler(cfg.data, train),
            )
            trainer.init_workers()
            trainer.fit()

    run_ppo(config, task_runner_class=ray.remote(LoopRLTaskRunner))
