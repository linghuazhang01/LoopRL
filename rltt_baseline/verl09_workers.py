"""LoopRL actor/ref/rollout composition using native verl 0.9 engine workers."""
from copy import deepcopy
from functools import partial

from omegaconf import OmegaConf
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils import tensordict_utils as tu
from verl.workers.engine_workers import ActorRolloutRefWorker

from rltt_baseline.verl09_engine import OuroTrainingWorker
from rltt_baseline.verl09_losses import looprl_loss


def load_frozen_teacher(model_path, student_module=None):
    """The initial model as a frozen, unsharded float32 module on this rank's GPU (OPD full).

    Uses the student's attention implementation when it can be read, so the teacher's forward
    matches the student's numerics at step 0 (the student keeps float32 master weights under the
    engine's bf16 autocast, as the teacher does here).
    """
    import torch
    from transformers import AutoModelForCausalLM

    config = getattr(student_module, "config", None)
    attn = getattr(config, "_attn_implementation", None) or "sdpa"
    teacher = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True, torch_dtype=torch.float32,
        attn_implementation=attn)
    teacher.config.early_exit_threshold = None  # per-loop outputs only; never exit early
    teacher.to(torch.device("cuda", torch.cuda.current_device()))
    teacher.eval()
    teacher.requires_grad_(False)
    return teacher


class LoopRLActorRolloutRefWorker(ActorRolloutRefWorker):
    actor_worker_cls = OuroTrainingWorker
    ref_worker_cls = OuroTrainingWorker

    def __init__(self, config, role, **kwargs):
        super().__init__(config, role, **kwargs)
        self.looprl_settings = OmegaConf.to_container(config.looprl, resolve=True)
        self.initial_anchor = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if self._is_actor:
            self.actor.engine.looprl_settings = self.looprl_settings
            if self.looprl_settings.get("audit"):
                from rltt_baseline.verl09_audit import NativeLoopAudit
                self.actor.engine.looprl_audit = NativeLoopAudit(
                    self.actor.engine.module, self.looprl_settings["audit"], self.rank,
                    self.looprl_settings)
            self.loss_fn = partial(looprl_loss, config=self.config.actor, settings=self.looprl_settings)
            self.actor.set_loss_fn(self.loss_fn)
            opd = self.looprl_settings.get("method") == "opd"
            opd_values = self.looprl_settings.get("opd", {})
            opd_full = opd and opd_values.get("estimator") == "full"
            if opd_full and opd_values.get("teacher", "frozen") == "online":
                # ELPO: the teacher is the student's own LoopCD distribution (stop-gradient) from
                # its training forward; no frozen copy and no teacher engine
                pass
            elif opd_full:
                # Full-vocabulary OPD: the student's training forward needs the teacher's hidden
                # states and lm_head in the same micro-batch, so every actor rank holds a frozen,
                # unsharded float32 copy of the initial model (model.path, never a checkpoint).
                self.actor.engine.looprl_opd_teacher_model = load_frozen_teacher(
                    self.config.model.path, self.actor.engine.module)
            elif opd or (self.looprl_settings.get("method") == "plc"
                         and self.looprl_settings.get("plc", {}).get("init_kl_coef", 0.) > 0):
                # Frozen initial model: the PLC anchor, or the OPD teacher (the base model's
                # LoopCD policy). This engine loads only model.path, never actor checkpoint
                # state, so a resumed run keeps the same anchor / teacher.
                anchor_config = deepcopy(self.actor.config)
                anchor_config.engine_config.forward_only = True
                anchor_config.engine_config.model_dtype = "float32"
                anchor_config.engine_config.param_offload = True
                anchor_config.optimizer_config = None
                self.initial_anchor = OuroTrainingWorker(anchor_config)
                self.initial_anchor.reset()
                self.initial_anchor.engine.looprl_settings = self.looprl_settings
                self.initial_anchor.engine.looprl_opd_teacher = opd
        if self._is_ref:
            self.ref.engine.looprl_settings = self.looprl_settings

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_log_prob(self, data):
        # Old-policy evaluation must not call a loss which needs old-policy fields.
        tu.assign_non_tensor(data, compute_loss=False)
        audit = getattr(self.actor.engine, "looprl_audit", None)
        if audit is not None:
            audit.begin(data)
        try:
            output = self.actor.infer_batch(data)
        finally:
            if audit is not None:
                audit.end()
        if self.initial_anchor is not None:
            anchor = self.initial_anchor.infer_batch(data)
            if output is not None:
                if self.looprl_settings.get("method") == "opd":
                    output["teacher_log_probs"] = anchor["teacher_log_probs"]
                else:
                    output["init_log_probs_loops"] = anchor["per_loop_log_probs"]
        return output.cpu() if output is not None else None

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="ref"))
    def compute_ref_log_prob(self, data):
        tu.assign_non_tensor(data, compute_loss=False)
        output = self.ref.infer_batch(data)
        return output.cpu() if output is not None else None
