"""The official Ouro forward must give the last loop's logits, the loop vLLM samples from.

GRPO's log-probs (and RLTT's ``old_log_probs``) come from ``OuroForCausalLM.forward`` without
exit arguments, so it falls back to ``config.early_exit_threshold`` (1.0 in the Ouro configs).
That branch takes, per token, the first loop whose cumulative exit probability reaches the
threshold; under the bf16 training forward the cumulative sum can round to 1.0 before the last
loop. Every config therefore overrides the threshold to null when verl loads the actor
(``actor_rollout_ref.model.override_config``, applied to the HF config before
``from_pretrained`` in verl 0.6.1 ``fsdp_workers.py``), which leaves the last-loop branch.
"""

import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from rltt_baseline import config as rcfg
from rltt_baseline import paths

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))
ENV = {
    "LOOPRL_OURO_DIR": "/models/Ouro-1.4B-Thinking",
    "LOOPRL_MODEL_DIR": "/models/Ouro-1.4B-Thinking-RLTT",
    "LOOPRL_TRAIN_FILE": "/data/train.parquet",
    "LOOPRL_MATH500_JSONL": "/data/math500.jsonl",
    "LOOPRL_OUTPUT_DIR": "/out/run",
    "LOOPRL_ENV_FILE": "/nonexistent/env.local",  # never read a real env.local in tests
}


def _overrides(override_config):
    return {"actor_rollout_ref": {"model": {"override_config": override_config}}}


@pytest.mark.parametrize("override_config,ok", [
    ({"early_exit_threshold": None}, True),
    ({"early_exit_threshold": None, "early_exit_step": None}, True),
    # the FSDP actor of verl 0.6.1 does not unwrap model_config: refused in both forms
    ({"model_config": {"early_exit_threshold": None}}, False),
    ({"early_exit_threshold": None, "model_config": {"early_exit_threshold": None}}, False),
    ({}, False),                                                # Ouro config 1.0 applies
    ({"early_exit_threshold": 1.0}, False),
    ({"early_exit_threshold": None, "early_exit_step": 2}, False),
])
def test_check_last_loop_logits(override_config, ok):
    if ok:
        rcfg.check_last_loop_logits(_overrides(override_config))
    else:
        with pytest.raises(ValueError, match="early_exit"):
            rcfg.check_last_loop_logits(_overrides(override_config))


def test_check_last_loop_logits_refuses_missing_model_section():
    with pytest.raises(ValueError, match="early_exit_threshold=null"):
        rcfg.check_last_loop_logits({"actor_rollout_ref": {"rollout": {}}})


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_every_config_passes_the_launcher_check(path, monkeypatch):
    for key, value in ENV.items():
        monkeypatch.setenv(key, value)
    cfg = rcfg.load_config(str(path))
    rcfg.check_last_loop_logits(rcfg.verl_overrides(cfg))


def test_dry_run_refuses_a_command_line_threshold(monkeypatch):
    from rltt_baseline import launch

    for key, value in ENV.items():
        monkeypatch.setenv(key, value)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "PYTHONPATH"):
        monkeypatch.setenv(key, "")
    monkeypatch.setattr(sys, "path", list(sys.path))
    with pytest.raises(ValueError, match="early_exit_threshold=null"):
        launch.main(["--config", str(CONFIG_DIR / "grpo_baseline_1p4b_8gpu.yaml"), "--dry-run",
                     "verl_overrides.actor_rollout_ref.model.override_config."
                     "early_exit_threshold=1.0"])


def test_configs_exist():
    assert CONFIGS


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_every_config_turns_off_the_official_exit_threshold(path):
    cfg = OmegaConf.load(path)
    override = OmegaConf.select(cfg, "verl_overrides.actor_rollout_ref.model.override_config")
    assert override is not None, f"{path.name}: no actor_rollout_ref.model.override_config"
    assert "early_exit_threshold" in override, f"{path.name}: early_exit_threshold not set"
    assert override.early_exit_threshold is None
    assert OmegaConf.select(override, "early_exit_step") is None


def test_official_forward_uses_the_last_loop_when_both_exit_settings_are_none():
    """The override relies on these lines of the vendored official forward."""
    source = (paths.ouro_source_dir() / "modeling_ouro.py").read_text()
    for line in (
        'self.early_exit_threshold = getattr(config, "early_exit_threshold", None)',
        "exit_threshold if exit_threshold is not None else self.early_exit_threshold",
        "elif exit_threshold is not None:",
        "logits = self.lm_head(_select_token_positions(hidden_states))",
    ):
        assert line in source, line
