"""CPU regressions for the PLC / verl typed-config boundary; no Ray or CUDA startup."""

import ast
import copy
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf, open_dict

from rltt_baseline import plc

pytest.importorskip("verl", reason="requires the train extra (verl 0.6.1)")


class NativeWorker:
    def __init__(self, config, role, **kwargs):
        from verl.utils.config import omega_conf_to_dataclass

        self.config = config
        self._is_actor = role == "actor_rollout"
        self.rank = 0
        # Exercise verl's real strict conversion: FSDPActorConfig rejects a PLC field.
        self.native_actor_config = omega_conf_to_dataclass(config.actor)

    def init_model(self):
        self.actor_module_fsdp = object()
        self.actor_optimizer = object()
        self.actor = SimpleNamespace(config=self.native_actor_config)


@pytest.fixture
def worker_type():
    # Run the production methods without Ray's remote-class / dispatch decorators.
    path = Path(__file__).resolve().parents[1] / "rltt_baseline/plc_workers.py"
    tree = ast.parse(path.read_text())
    cls = copy.deepcopy(next(node for node in tree.body if isinstance(node, ast.ClassDef)
                             and node.name == "PLCActorRolloutRefWorker"))
    cls.decorator_list = []
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef)
                and node.name in ("__init__", "init_model")]
    for method in cls.body:
        method.decorator_list = []
    namespace = {
        "ActorRolloutRefWorker": NativeWorker,
        "OmegaConf": OmegaConf,
        "plc": plc,
        "PLCActor": lambda **kwargs: SimpleNamespace(**kwargs),
        "attach_loop_audit": lambda *args: None,
        "os": os,
        "sys": sys,
    }
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["PLCActorRolloutRefWorker"]


@pytest.fixture
def worker_config():
    config = OmegaConf.create({
        "actor": {
            "_target_": "verl.workers.config.FSDPActorConfig",
            "strategy": "fsdp",
            "use_dynamic_bsz": True,
            "ppo_mini_batch_size": 32,
            "ppo_max_token_len_per_gpu": 32768,
            "grad_clip": 0.1,
            "use_kl_loss": True,
            "kl_loss_coef": 0.001,
            "plc": {
                "enabled": True, "weight_mode": "posterior", "prior": "exit",
                "gamma": 1.0, "lam": 1.0, "use_gate_loss": True,
                "gate_coef": 1.0, "gate_entropy_coef": 0.05,
            },
        },
        "total_ut_steps": 4,
    })
    OmegaConf.set_struct(config, True)
    return config


@pytest.mark.parametrize("role", ["actor_rollout", "ref"])
def test_plc_settings_are_isolated_from_native_config(worker_type, worker_config, role):
    original = OmegaConf.to_container(worker_config, resolve=True)
    worker = worker_type(worker_config, role)
    assert OmegaConf.to_container(worker_config, resolve=True) == original
    assert worker.config is not worker_config
    assert "plc" not in worker.config.actor
    assert worker.settings == plc.PLCSettings.from_mapping(original["actor"]["plc"])
    assert worker.total_ut_steps == 4
    for key in ("ppo_mini_batch_size", "ppo_max_token_len_per_gpu", "grad_clip",
                "use_kl_loss", "kl_loss_coef", "use_dynamic_bsz"):
        assert getattr(worker.native_actor_config, key) == original["actor"][key]


def test_plc_actor_reuses_native_config_and_clip_defaults(worker_type, worker_config):
    worker = worker_type(worker_config, "actor_rollout")
    worker.init_model()
    assert worker.actor.config is worker.native_actor_config
    assert type(worker.actor.config).__name__ == "FSDPActorConfig"
    assert worker.actor.config.clip_ratio == 0.2
    assert worker.actor.config.clip_ratio_low == 0.2
    assert worker.actor.config.clip_ratio_high == 0.2
    assert worker.actor.config.clip_ratio_c == 3.0
    assert worker.actor.settings is worker.settings
    assert worker.actor.actor_module is worker.actor_module_fsdp
    assert worker.actor.actor_optimizer is worker.actor_optimizer
    assert worker.actor.total_ut_steps == 4


def test_missing_plc_settings_fail_before_native_worker_init(worker_type, worker_config):
    with open_dict(worker_config.actor):
        del worker_config.actor.plc
    with pytest.raises(ValueError, match="actor_rollout_ref.actor.plc is missing"):
        worker_type(worker_config, "actor_rollout")
