"""Padded Ouro engine registered alongside verl's native FSDP backends."""
from contextlib import nullcontext
from copy import deepcopy

import torch
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_id
from verl.workers.engine import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
from verl.workers.engine_workers import TrainingWorker


def to_nested_output(value, lengths):
    """Keep autograd while dropping padding; outputs use predicting-token positions."""
    return torch.nested.as_nested_tensor([value[i, :int(n)] for i, n in enumerate(lengths)],
                                         layout=torch.jagged)


@EngineRegistry.register(model_type="looprl_ouro", backend=["fsdp", "fsdp2"], device="cuda")
class OuroFSDPEngine(FSDPEngineWithLMHead):
    def _build_module(self):
        # Registry key differs, but construction is the native HF language model path.
        original = self.model_config.model_type
        self.model_config.model_type = "language_model"
        try:
            return super()._build_module()
        finally:
            self.model_config.model_type = original

    def forward_step(self, micro_batch, loss_function, forward_only):
        if self.ulysses_sequence_parallel_size != 1:
            raise ValueError("Ouro padded engine requires ulysses_sequence_parallel_size=1")
        micro_batch = micro_batch.to(get_device_id())
        if tu.get(micro_batch, "use_remove_padding", False) or tu.get(micro_batch, "use_fused_kernels", False):
            raise ValueError("Ouro loop outputs require padded, unfused forward")
        inputs, _ = self.prepare_model_inputs(micro_batch)
        ids = inputs["input_ids"]
        labels = torch.nn.functional.pad(ids[:, 1:], (0, 1), value=0)
        temperature = tu.get(micro_batch, "temperature", 1.)
        if isinstance(temperature, torch.Tensor):
            if not torch.all(temperature == temperature.flatten()[0]):
                raise ValueError("Ouro requires identical temperature across a microbatch")
            temperature = temperature.flatten()[0].item()
        settings = getattr(self, "looprl_settings", {})
        plc_cfg = settings.get("plc", {})
        kd = settings.get("method") == "plc" and plc_cfg.get("loop_loss") == "kd" and not forward_only
        kwargs = dict(return_per_loop_log_probs=True,
                      return_exit_pdf=settings.get("loop_weighting") == "exit_pdf",
                      log_prob_labels=labels, log_prob_temperature=float(temperature), use_cache=False)
        if kd:
            kwargs.update(return_loop_kd=True, loop_kd_divergence=plc_cfg.get("kd_divergence", "forward"),
                          loop_kd_window=ids.shape[1] - 1)
        extrap = settings.get("method") == "extrap"
        if extrap:
            from rltt_baseline.extrap import ExtrapSettings
            kwargs.update(ExtrapSettings.from_mapping(settings.get("extrap")).model_kwargs())
        # OPD sampled: only the frozen teacher engine computes the base model's LoopCD log-probs
        teacher = settings.get("method") == "opd" and getattr(self, "looprl_opd_teacher", False)
        if teacher:
            if not forward_only:
                raise ValueError("the OPD teacher engine is forward-only")
            from rltt_baseline.opd import OPDSettings
            kwargs.update(OPDSettings.from_mapping(settings.get("opd")).teacher_model_kwargs())
        # OPD full: the student's training forward computes the full-vocabulary KL against the
        # frozen teacher model this rank holds (verl09_workers)
        opd_full = None
        if settings.get("method") == "opd" and not forward_only:
            from rltt_baseline.opd import OPDSettings
            opd_cfg = OPDSettings.from_mapping(settings.get("opd"))
            if opd_cfg.full_vocab:
                opd_full = opd_cfg
                if getattr(self, "looprl_opd_teacher_model", None) is None:
                    raise ValueError("OPD estimator=full needs the frozen teacher model")
        dtype = getattr(self, "_autocast_dtype", torch.bfloat16)
        precision = nullcontext() if dtype == torch.float32 else torch.autocast("cuda", dtype=dtype)
        with precision:
            if opd_full is not None:
                teacher_model = self.looprl_opd_teacher_model
                with torch.no_grad():
                    hiddens = teacher_model(**inputs, return_loop_hidden_states=True,
                                            use_cache=False).loop_hidden_states
                kwargs.update(opd_full.student_model_kwargs())
                kwargs.update(opd_teacher_hidden_states=(hiddens[opd_full.ref_loop - 1],
                                                         hiddens[-1]),
                              opd_teacher_lm_head=teacher_model.lm_head,
                              opd_window=ids.shape[1] - 1)
                del hiddens
            audit = getattr(self, "looprl_audit", None)
            handle = audit.hook(micro_batch, inputs) if audit is not None else None
            try:
                raw = self.module(**inputs, **kwargs)
            finally:
                if handle is not None:
                    handle.remove()
            dense = {"per_loop_log_probs": torch.stack(raw.per_loop_log_probs, -1).float(),
                     "per_loop_entropies": torch.stack(raw.per_loop_entropies, -1).float()}
            dense["log_probs"] = dense["per_loop_log_probs"][..., -1]
            dense["entropy"] = dense["per_loop_entropies"][..., -1]
            if extrap:
                # EP-GRPO: the policy (and old_log_probs) is the extrapolated LoopCD policy
                dense["log_probs"] = raw.extrap["log_probs"].float()
                dense["entropy"] = raw.extrap["entropy"].float()
                dense["extrap_omega"] = raw.extrap["omega"].float()
                dense["extrap_margin"] = raw.extrap["margin"].float()
            if teacher:
                dense["teacher_log_probs"] = raw.extrap["log_probs"].float()
            if opd_full is not None:
                # window = all positions but the last; pad back to the label-aligned length
                for name, value in raw.opd_kd.items():
                    dense[f"opd_{name}"] = torch.nn.functional.pad(value.float(), (0, 1))
            if kwargs["return_exit_pdf"]:
                dense["exit_pdf"] = raw.exit_pdf
            if kd:
                dense["loop_kd"] = torch.nn.functional.pad(raw.loop_kd, (0, 0, 0, 1))
            lengths = micro_batch["input_ids"].offsets().diff()
            output = {k: to_nested_output(v, lengths) for k, v in dense.items()}
            if loss_function is None:
                if not forward_only:
                    raise ValueError("Training requires a LoopRL loss callback")
                loss, metrics = ids.new_zeros((), dtype=torch.float32), {}
            else:
                loss, metrics = loss_function(model_output=output, data=micro_batch,
                                               dp_group=self.get_data_parallel_group())
        return loss, {"model_output": {k: v.detach() for k, v in output.items()},
                      "loss": loss.detach().item(), "metrics": metrics}


class OuroTrainingWorker(TrainingWorker):
    def __init__(self, config):
        config = deepcopy(config)
        config.model_type = "looprl_ouro"
        super().__init__(config)
