"""GRPO worker with the per-loop distribution audit (``rltt_baseline.loop_audit``).

The launcher uses it only when ``audit.loop_distribution.enabled`` is true; otherwise the
trainer keeps the vendor ``GRPOActorRolloutRefWorker``. The vendor class is a Ray actor class
and cannot be subclassed, and it only adds a parameter-count printout to verl's worker.
Requires verl 0.6.1 and ray; import only inside the training environment.
"""

import ray
import verl_grpo.grpo_fsdp_workers  # noqa: F401  (applies the vendor target_modules patch)
from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.fsdp_workers import ActorRolloutRefWorker

from rltt_baseline.loop_audit import attach_loop_audit


@ray.remote
class AuditGRPOActorRolloutRefWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if self._is_actor:
            attach_loop_audit(self.actor, self.config.get("loop_audit"), self.rank)
