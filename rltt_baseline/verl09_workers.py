"""LoopRL actor/ref/rollout composition using native verl 0.9 engine workers."""
from copy import deepcopy
from functools import partial

from omegaconf import OmegaConf
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils import tensordict_utils as tu
from verl.workers.engine_workers import ActorRolloutRefWorker

from rltt_baseline.verl09_engine import OuroTrainingWorker
from rltt_baseline.verl09_losses import looprl_loss


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
            if self.looprl_settings.get("plc", {}).get("init_kl_coef", 0.) > 0:
                # This engine loads only model.path, never actor checkpoint state.
                anchor_config = deepcopy(self.actor.config)
                anchor_config.engine_config.forward_only = True
                anchor_config.engine_config.model_dtype = "float32"
                anchor_config.engine_config.param_offload = True
                anchor_config.optimizer_config = None
                self.initial_anchor = OuroTrainingWorker(anchor_config)
                self.initial_anchor.reset()
                self.initial_anchor.engine.looprl_settings = self.looprl_settings
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
                output["init_log_probs_loops"] = anchor["per_loop_log_probs"]
        return output.cpu() if output is not None else None

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="ref"))
    def compute_ref_log_prob(self, data):
        tu.assign_non_tensor(data, compute_loss=False)
        output = self.ref.infer_batch(data)
        return output.cpu() if output is not None else None
