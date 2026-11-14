"""GRPO worker: per-loop distribution audit and the row-normalized loss aggregation.

The launcher uses ``LoopRLGRPOActorRolloutRefWorker`` when ``audit.loop_distribution.enabled`` is
true or ``loss_agg.mode`` is not ``token-mean``; otherwise the trainer keeps the vendor
``GRPOActorRolloutRefWorker``. The vendor class is a Ray actor class and cannot be subclassed, and
it only adds a parameter-count printout to verl's worker.

A row-normalized ``loss_agg.mode`` (``actor_rollout_ref.loss_agg``) replaces verl's actor
by ``RowAggPPOActor``: verl 0.6.1's ``DataParallelPPOActor.update_policy`` (dynamic batch,
``B / N`` scaling, on-policy shortcut, optimizer step, metrics) with one change, the aggregation
of each micro-batch's loss terms. verl's ``token-mean`` divides by the micro-batch's own token
count; here every response ``i`` gets a divisor ``d_i`` of its own (``rltt_baseline.loss_agg``:
its response length for seq-mean-token-mean, ``norm_len`` for drgrpo) and a micro-batch of ``B``
rows contributes ``(1 / B) sum_i S_i / d_i``. The policy term is ``plc.grpo_policy_loss``, the
same element-wise clipped objective as verl's ``compute_policy_loss_vanilla`` (and as PLC's
final-loop term).
Requires verl 0.6.1 and ray; import only inside the training environment.
"""

import os
import sys
from typing import Optional

import ray
import verl_grpo.grpo_fsdp_workers  # noqa: F401  (applies the vendor target_modules patch)
from omegaconf import OmegaConf
from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.trainer.ppo.core_algos import kl_penalty
from verl.utils.device import get_device_id
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.fsdp_workers import ActorRolloutRefWorker

from rltt_baseline import loss_agg, plc
from rltt_baseline.loop_audit import attach_loop_audit


class RowAggPPOActor(DataParallelPPOActor):
    """verl 0.6.1's actor with the row-normalized aggregation of ``rltt_baseline.loss_agg``."""

    def __init__(self, config, actor_module, actor_optimizer=None,
                 loss_agg_settings: Optional[loss_agg.LossAggSettings] = None):
        super().__init__(config=config, actor_module=actor_module,
                         actor_optimizer=actor_optimizer)
        if loss_agg_settings is None or not loss_agg_settings.row_normalized:
            raise ValueError("RowAggPPOActor needs a row-normalized loss_agg.mode")
        self.loss_agg = loss_agg_settings

    def update_policy(self, data: DataProto):
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        if loss_mode != "vanilla":
            raise ValueError(f"loss_agg.mode={self.loss_agg.mode} supports only the vanilla PPO "
                             f"policy loss, got policy_loss.loss_mode={loss_mode!r}")
        if "rollout_is_weights" in data.batch.keys():
            raise ValueError(f"loss_agg.mode={self.loss_agg.mode} does not support "
                             "rollout_is_weights (algorithm.rollout_is)")
        self.actor_module.train()
        temperature = data.meta_info["temperature"]

        select_keys = ["responses", "response_mask", "input_ids", "attention_mask",
                       "position_ids", "old_log_probs", "advantages"]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        mini_batches = data.split(self.config.ppo_mini_batch_size)
        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        tracker = loss_agg.LossTracker()  # loss/<term>: the exact objective of each update
        for _ in range(self.config.ppo_epochs):
            for mini_batch in mini_batches:
                if self.config.use_dynamic_bsz:
                    max_token_len = (self.config.ppo_max_token_len_per_gpu
                                     * self.ulysses_sequence_parallel_size)
                    micro_batches, _ = prepare_dynamic_batch(mini_batch,
                                                             max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (self.config.ppo_mini_batch_size
                                                  // self.config.ppo_micro_batch_size_per_gpu)
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    advantages = model_inputs["advantages"]
                    # per-response divisors d_i (a function of each row alone)
                    row_div = loss_agg.row_divisors(self.loss_agg, response_mask)
                    entropy_coeff = self.config.entropy_coeff

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    calculate_entropy = entropy_coeff != 0
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy)

                    if getattr(self.config, "use_rollout_log_probs", False) or not on_policy:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        old_log_prob = log_prob.detach()

                    pg_loss, pg_metrics = plc.grpo_policy_loss(
                        log_prob, old_log_prob, advantages, response_mask,
                        clip_ratio=float(self.config.clip_ratio),
                        clip_ratio_low=self.config.clip_ratio_low,
                        clip_ratio_high=self.config.clip_ratio_high,
                        clip_ratio_c=float(self.config.get("clip_ratio_c", 3.0)),
                        row_div=row_div)
                    terms = loss_agg.pop_loss_terms(pg_metrics)
                    micro_batch_metrics.update(pg_metrics)

                    policy_loss = pg_loss
                    if entropy_coeff != 0:
                        entropy_loss = loss_agg.agg_rows(entropy, response_mask, row_div)
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                        terms["entropy"] = -float(entropy_loss.detach()) * entropy_coeff

                    if self.config.use_kl_loss:
                        kld = kl_penalty(logprob=log_prob, ref_logprob=model_inputs["ref_log_prob"],
                                         kl_penalty=self.config.kl_loss_type)
                        kl_loss = loss_agg.agg_rows(kld, response_mask, row_div)
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        terms["kl"] = float(kl_loss.detach()) * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = (kl_loss.detach().item()
                                                                * loss_scale_factor)
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    terms["total"] = float(policy_loss.detach())
                    tracker.add(terms, loss_scale_factor)

                    micro_batch_metrics["actor/pg_loss"] = (pg_loss.detach().item()
                                                            * loss_scale_factor)
                    micro_batch_metrics["loss_agg/row_div_mean"] = row_div.mean().item()
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item(),
                                         **tracker.flush()})
        self.actor_optimizer.zero_grad()
        return metrics


@ray.remote
class LoopRLGRPOActorRolloutRefWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if not self._is_actor:
            return
        raw = self.config.get("loss_agg", None)
        settings = loss_agg.LossAggSettings.from_mapping(
            OmegaConf.to_container(raw, resolve=True) if raw is not None else None)
        if settings.row_normalized:
            self.actor = RowAggPPOActor(
                config=self.actor.config,  # verl's typed actor config, native defaults included
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
                loss_agg_settings=settings,
            )
            if self.rank == 0:
                print(f"[rltt_baseline] RowAggPPOActor {settings.describe()} pid={os.getpid()} "
                      f"python={sys.executable}", flush=True)
        attach_loop_audit(self.actor, self.config.get("loop_audit"), self.rank)


__all__ = ["RowAggPPOActor", "LoopRLGRPOActorRolloutRefWorker"]
