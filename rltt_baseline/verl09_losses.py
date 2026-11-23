"""LoopRL objectives for the verl 0.9 TrainingWorker callback contract."""
import torch

from rltt_baseline import loss_agg, plc


def objective(config, settings, output, batch):
    """Unscaled microbatch objective; retains the legacy B/N aggregation boundary."""
    method = settings.get("method", "grpo")
    ell = output["per_loop_log_probs"]
    mask = batch["response_mask"]
    advantages = batch["advantages"]
    aggregation = loss_agg.LossAggSettings.from_mapping(settings.get("loss_agg"))
    row_div = loss_agg.row_divisors(aggregation, mask) if aggregation.row_normalized else None
    mode = config.get("loss_agg_mode", "token-mean")
    clip = dict(clip_ratio=float(config.get("clip_ratio", .2)),
                clip_ratio_low=config.get("clip_ratio_low"),
                clip_ratio_high=config.get("clip_ratio_high"),
                clip_ratio_c=float(config.get("clip_ratio_c", 3.)))
    if "rollout_is_weights" in batch:
        raise ValueError("LoopRL objectives do not support rollout importance correction")
    if method == "plc":
        cfg = plc.PLCSettings.from_mapping(settings.get("plc"))
        old = batch["old_log_probs_loops"].float()
        weights = plc.compute_loop_weights(old, cfg.weight_mode, cfg.trunc_cap)
        alpha = plc.compute_loop_weights(old, "early_prior")
        common = dict(loss_agg_mode=mode, row_div=row_div, init_kl_coef=cfg.init_kl_coef,
                      logp_init=batch.get("init_log_probs_loops"), init_kl_form=cfg.init_kl_form,
                      init_kl_log_ratio_cap=cfg.init_kl_log_ratio_cap, **clip)
        if cfg.loop_loss == "kd":
            loss, metrics = plc.plc_kd_loss(ell, old, advantages, mask, output["loop_kd"],
                                           cfg.lam, **common)
        else:
            loss, metrics = plc.plc_policy_loss(
                ell, old, advantages, mask, weights, cfg.lam, kl_coef=cfg.kl_coef,
                kl_weights=alpha, init_weights=alpha, **common)
    elif method == "extrap":
        from rltt_baseline import extrap
        cfg = extrap.ExtrapSettings.from_mapping(settings.get("extrap"))
        weights = extrap.token_weights(output["extrap_omega"], output["extrap_margin"], cfg)
        loss, metrics = extrap.extrap_policy_loss(
            output["log_probs"], batch["old_log_probs"], advantages, mask, weights,
            loss_agg_mode=mode, row_div=row_div, **clip)
        metrics.update(extrap.extrap_stats(
            output["log_probs"].detach(), output["entropy"], output["extrap_omega"],
            output["extrap_margin"], weights, ell[..., -1].detach(), mask))
    elif method == "opd":
        from rltt_baseline import opd
        cfg = opd.OPDSettings.from_mapping(settings.get("opd"))
        # the batch's GRPO advantages enter only with opd.rl_coef > 0 (default 0: pure OPD)
        teacher = batch.get("teacher_log_probs")
        if cfg.sampled_teacher and teacher is None:
            raise ValueError(f"OPD estimator={cfg.estimator} needs teacher_log_probs from the "
                             "frozen teacher engine")
        if cfg.full_vocab and "opd_kd" not in output:
            raise ValueError("OPD estimator=full needs opd_kd from the student's forward")
        loss, metrics = opd.opd_loss(
            cfg, ell[..., -1], batch["old_log_probs"], mask, advantages=advantages,
            teacher_log_prob=teacher.float() if cfg.sampled_teacher else None,
            kd=output.get("opd_kd"), loss_agg_mode=mode, row_div=row_div, **clip)
        if cfg.full_vocab:
            metrics.update(opd.opd_full_stats(output["opd_kd"].detach(), output["opd_kl_other"],
                                              output["opd_teacher_entropy"],
                                              output["opd_top1_agree"], cfg, mask))
        else:
            metrics.update(opd.opd_stats(teacher, batch["old_log_probs"], mask))
    elif method == "grpo":
        loss, metrics = plc.grpo_policy_loss(ell[..., -1], batch["old_log_probs"], advantages,
                                            mask, loss_agg_mode=mode, row_div=row_div, **clip)
    elif method == "rltt":
        weighting = settings.get("loop_weighting", "uniform")
        if weighting == "exit_pdf":
            weights = output["exit_pdf"]
        elif weighting in ("uniform", "progressive"):
            weights = torch.arange(1, ell.shape[-1] + 1, device=ell.device, dtype=ell.dtype)
            weights = weights ** (float(settings.get("progressive_alpha", 1.))
                                  if weighting == "progressive" else 0.)
            weights = weights / weights.sum()
        else:
            raise ValueError("learned RLTT weights need an optimizer/checkpoint adapter")
        adv = advantages[:, None] if advantages.ndim == 1 else advantages
        loss = plc.agg_loss(-adv * (weights * ell).sum(-1), mask, "token-mean")
        metrics = {}
    else:
        raise ValueError(f"Unknown LoopRL method: {method}")
    metrics["actor/pg_loss"] = loss.detach().item()
    metrics.setdefault("loss_term/pg", loss.detach().item())
    # The PLC/RLTT legacy objective uses final-loop low-variance KL, including
    # fallback to old policy when reference KL is disabled.
    entropy = output["entropy"]
    entropy_coeff = float(config.get("entropy_coeff", 0.))
    entropy_term = -entropy_coeff * plc.agg_term(entropy, mask, mode, row_div)
    loss = loss + entropy_term
    metrics["loss_term/entropy"] = entropy_term.detach().item()
    use_ref = config.get("use_kl_loss", False)
    kl_coef = float(config.get("kl_loss_coef", 0.))
    if kl_coef and (use_ref or method != "grpo"):
        reference = batch["ref_log_prob" if use_ref else "old_log_probs"]
        if method == "extrap":
            kl = plc.low_var_kl(output["log_probs"], reference)
        elif method == "grpo":
            from verl.trainer.ppo.core_algos import kl_penalty
            kl = kl_penalty(logprob=ell[..., -1], ref_logprob=reference,
                            kl_penalty=config.get("kl_loss_type", "low_var_kl"))
        elif method == "rltt":
            log_ratio = reference - ell[..., -1]
            kl = torch.exp(log_ratio) - log_ratio - 1
        else:
            kl = plc.low_var_kl(ell[..., -1], reference)
        kl_term = kl_coef * plc.agg_term(kl, mask, mode, row_div)
        loss = loss + kl_term
        metrics["loss_term/kl"] = kl_term.detach().item()
    metrics = plc.flatten_loop_metrics(metrics)
    metrics["actor/total_loss"] = loss.detach().item()
    for i in range(output["per_loop_entropies"].shape[-1]):
        metrics[f"rltt/loop_{i+1}_entropy"] = plc.agg_loss(
            output["per_loop_entropies"][..., i], mask, "token-mean").detach().item()
    return loss, metrics


def looprl_loss(config, settings, model_output, data, dp_group=None):
    from verl.utils import tensordict_utils as tu
    from verl.utils.metric import AggregationType, Metric
    from verl.workers.utils.padding import no_padding_2_padding
    output = {k: no_padding_2_padding(v, data) for k, v in model_output.items()}
    batch = {k: v.to_padded_tensor(0.) if getattr(v, "is_nested", False) else v
             for k, v in data.items() if isinstance(v, torch.Tensor)}
    loss, metrics = objective(config, settings, output, batch)
    # FSDP averages rank gradients; retain legacy local microbatch B/N scaling.
    dp = tu.get(data, "dp_size", 1)
    global_bsz = tu.get(data, "global_batch_size", None)
    if global_bsz is None or global_bsz <= 0:
        raise ValueError("LoopRL loss requires the global update mini-batch size")
    scale = batch["response_mask"].shape[0] * dp / global_bsz
    loss = loss * scale
    terms = loss_agg.pop_loss_terms(metrics)
    result = Metric.from_dict(metrics, aggregation=AggregationType.MEAN)
    result.update({f"loss/{k}": Metric(value=v * scale, aggregation=AggregationType.SUM)
                   for k, v in terms.items()})
    result["loss/total"] = Metric(value=loss.detach(), aggregation=AggregationType.SUM)
    return loss, result
