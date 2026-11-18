"""Schema bridge tests use a tiny native default; no Ray/GPU import required."""
from argparse import Namespace
from contextlib import nullcontext
import sys
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from rltt_baseline.verl09_launcher import build_native_config


@pytest.fixture
def compose_stub(monkeypatch):
    monkeypatch.setitem(sys.modules, "hydra", SimpleNamespace(
        initialize_config_dir=lambda **kw: nullcontext(),
        compose=lambda **kw: OmegaConf.create({"trainer": {}, "actor_rollout_ref": {}}),
    ))


def test_native_server_and_explicit_data(compose_stub):
    old = {"data": {"val_files": ["val.parquet"]},
           "custom_reward_function": {"path": "reward.py", "name": "compute_score"},
           "actor_rollout_ref": {"actor": {"plc": {}}, "rollout": {"mode": "sync"}}}
    cfg = build_native_config(old, Namespace(train_file="train.parquet"), "/tmp")
    assert cfg.actor_rollout_ref.rollout.mode == "async"
    assert not cfg.trainer.use_v1
    assert cfg.data.train_files == ["train.parquet"]
    assert cfg.reward.custom_reward_function.path == "reward.py"
    assert not cfg.actor_rollout_ref.model.use_remove_padding
    assert old["actor_rollout_ref"]["rollout"]["mode"] == "sync"


@pytest.mark.parametrize("patch,match", [
    ({"trainer": {"gradient_accumulation_steps": 2}}, "accumulation"),
    ({"actor_rollout_ref": {"looprl": {"loop_weighting": "learned"}}}, "Learned"),
    ({"reward_model": {"enable": True}}, "reward model"),
])
def test_does_not_silently_drop_semantics(compose_stub, patch, match):
    patch["data"] = {"val_files": ["val.parquet"]}
    with pytest.raises(ValueError, match=match):
        build_native_config(patch, Namespace(train_file="train.parquet"), "/tmp")


def test_audit_transferred_to_engine_settings(compose_stub):
    cfg = build_native_config({"data": {"val_files": ["val.parquet"]},
        "actor_rollout_ref": {"loop_audit": {"dir": "/persistent/audit", "every_n_steps": 2}}},
        Namespace(train_file="train.parquet"), "/tmp")
    assert cfg.actor_rollout_ref.looprl.audit.dir == "/persistent/audit"
    assert "loop_audit" not in cfg.actor_rollout_ref


def test_decode_last_uses_v1_runtime_env(compose_stub):
    from rltt_baseline.decode_last import ARCHITECTURE
    cfg = build_native_config({"data": {"val_files": ["val.parquet"]},
        "actor_rollout_ref": {"rollout": {"engine_kwargs": {"vllm": {
            "hf_overrides": {"architectures": [ARCHITECTURE]}}}}}},
        Namespace(train_file="train.parquet"), "/tmp")
    assert cfg.ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V2_MODEL_RUNNER == "0"
