"""YAML configuration for the RLTT / GRPO baselines, with command-line overrides.

A config file has these top-level keys:

``trainer``
    ``rltt`` (``rltt_experiments/rltt_train.py``, default) or ``grpo``
    (``grpo_experiments/grpo_train.py``, the GRPO baseline of the RLTT repository).
``args``
    Keys are the ``dest`` names of the chosen vendor script's argparse options (for example
    ``learning_rate``, ``max_completion_length``). Unlisted options keep the vendor default.
    Unknown keys and type mismatches are errors.
``wandb``
    ``enabled``, ``mode`` (online / offline / disabled), ``project``, ``entity``,
    ``run_name``, ``tags``, ``notes``, ``dir``. Mapped to the vendor ``--no_wandb`` /
    ``--wandb_project`` options, verl's ``trainer.experiment_name`` and ``WANDB_*`` variables.
``verl_overrides``
    Nested dict deep-merged into the final verl config right before the trainer is built,
    after the vendor code has filled in paths and step counts. Any verl key can be changed.
``env``
    Extra environment variables for the driver and the Ray workers.
``rollout``
    ``kv_layout``: ``full`` (default, stock vLLM) or ``decode_last`` (rollouts keep every loop's
    KV for the prompt but only the last loop's KV for generated tokens, see
    ``rltt_baseline.vllm_decode_last``). ``log_probs`` (default: on with ``decode_last``) keeps
    vLLM's sampled-token log-probs, so verl logs the rollout / training mismatch
    (``training/rollout_probs_diff_*``, ``rollout_corr/*``) without changing the loss.
    Training forwards always use the full per-loop KV.

Overrides use OmegaConf dot-list syntax and win over the file, e.g.
``args.learning_rate=2e-6 verl_overrides.trainer.save_freq=20 wandb.mode=online``.
Values may use ``${oc.env:VAR}`` interpolation.
"""

import argparse
import difflib
from typing import Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

from rltt_baseline import decode_last

SECTIONS = ("trainer", "args", "wandb", "verl_overrides", "env", "rollout")
DICT_SECTIONS = ("args", "wandb", "verl_overrides", "env", "rollout")
TRAINERS = ("rltt", "grpo")
WANDB_KEYS = {"enabled", "mode", "project", "entity", "run_name", "tags", "notes", "dir"}
LOOP_WEIGHTINGS = {"uniform", "progressive", "exit_pdf", "learned"}
ROLLOUT_KEYS = {"kv_layout", "log_probs"}
KV_LAYOUTS = ("full", "decode_last")
# Vendor default of verl ``algorithm.use_kl_in_reward``: the GRPO script adds the KL to the
# reward, the RLTT script uses a KL loss. This project never puts the KL in the reward.
VENDOR_KL_IN_REWARD = {"rltt": False, "grpo": True}


def load_config(path: str, overrides: Optional[List[str]] = None) -> DictConfig:
    cfg = OmegaConf.merge(OmegaConf.load(path), OmegaConf.from_dotlist(list(overrides or [])))
    unknown = set(cfg.keys()) - set(SECTIONS)
    if unknown:
        raise KeyError(f"unknown top-level config keys {sorted(unknown)}; allowed {SECTIONS}")
    for section in DICT_SECTIONS:
        if section not in cfg or cfg[section] is None:
            cfg[section] = {}
    if cfg.get("trainer") is None:
        cfg.trainer = "rltt"
    if cfg.trainer not in TRAINERS:
        raise ValueError(f"trainer must be one of {TRAINERS}, got {cfg.trainer!r}")
    bad = set(cfg.wandb.keys()) - WANDB_KEYS
    if bad:
        raise KeyError(f"unknown wandb keys {sorted(bad)}; allowed {sorted(WANDB_KEYS)}")
    bad = set(cfg.rollout.keys()) - ROLLOUT_KEYS
    if bad:
        raise KeyError(f"unknown rollout keys {sorted(bad)}; allowed {sorted(ROLLOUT_KEYS)}")
    if cfg.rollout.get("kv_layout") is None:
        cfg.rollout.kv_layout = "full"
    if cfg.rollout.kv_layout not in KV_LAYOUTS:
        raise ValueError(f"rollout.kv_layout must be one of {KV_LAYOUTS}, got "
                         f"{cfg.rollout.kv_layout!r}")
    OmegaConf.resolve(cfg)
    return cfg


def _coerce(name: str, value, default):
    if default is None or value is None:
        return value
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise TypeError(f"args.{name} must be a bool, got {value!r}")
        return value
    if isinstance(default, int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"args.{name} must be an int, got {value!r}")
        return value
    if isinstance(default, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"args.{name} must be a number, got {value!r}")
        return float(value)
    if isinstance(default, str):
        if not isinstance(value, str):
            raise TypeError(f"args.{name} must be a string, got {value!r}")
        return value
    return value


def build_args(cfg: DictConfig, defaults: argparse.Namespace) -> argparse.Namespace:
    """Vendor argparse namespace: vendor defaults, then ``cfg.args``, then ``cfg.wandb``."""
    values = dict(vars(defaults))
    for name, value in OmegaConf.to_container(cfg.args, resolve=True).items():
        if name not in values:
            close = difflib.get_close_matches(name, values, n=3)
            raise KeyError(f"args.{name} is not a {cfg.trainer} trainer option; close: {close}")
        values[name] = _coerce(name, value, values[name])

    wandb = cfg.wandb
    if "enabled" in wandb:
        values["no_wandb"] = not bool(wandb.enabled)
    if wandb.get("mode") == "disabled":
        values["no_wandb"] = True
    if wandb.get("project"):
        values["wandb_project"] = str(wandb.project)

    args = argparse.Namespace(**values)
    validate_args(args, cfg.trainer)
    return args


def validate_args(args: argparse.Namespace, trainer: str = "rltt") -> None:
    if trainer == "rltt":
        if args.loop_weighting not in LOOP_WEIGHTINGS:
            raise ValueError(f"loop_weighting must be one of {sorted(LOOP_WEIGHTINGS)}")
        if args.loop_weighting == "exit_pdf" and not args.use_rltt_model:
            raise ValueError("loop_weighting=exit_pdf needs use_rltt_model=true (per-loop model)")
    if trainer == "grpo" and args.use_ppo:
        raise ValueError("trainer=grpo with use_ppo=true would run PPO with a critic")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    for name in ("max_prompt_length", "max_completion_length", "num_generations",
                 "num_prompts_per_batch", "gradient_accumulation_steps"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("max_steps must be -1 (use epochs) or positive")


def wandb_env(cfg: DictConfig) -> Dict[str, str]:
    """``WANDB_*`` variables read by wandb in the driver (verl's Tracking calls wandb.init)."""
    wandb = cfg.wandb
    env = {}
    if wandb.get("mode"):
        if wandb.mode not in ("online", "offline", "disabled"):
            raise ValueError("wandb.mode must be online, offline or disabled")
        env["WANDB_MODE"] = str(wandb.mode)
    if wandb.get("entity"):
        env["WANDB_ENTITY"] = str(wandb.entity)
    if wandb.get("tags"):
        env["WANDB_TAGS"] = ",".join(str(t) for t in wandb.tags)
    if wandb.get("notes"):
        env["WANDB_NOTES"] = str(wandb.notes)
    if wandb.get("dir"):
        env["WANDB_DIR"] = str(wandb.dir)
    return env


def verl_overrides(cfg: DictConfig) -> dict:
    overrides = OmegaConf.to_container(cfg.verl_overrides, resolve=True)
    if cfg.wandb.get("run_name"):
        overrides.setdefault("trainer", {})["experiment_name"] = str(cfg.wandb.run_name)
    return overrides


def _deep_merge(base: dict, extra: dict) -> dict:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def add_rollout_overrides(cfg: DictConfig, overrides: dict) -> dict:
    """``overrides`` plus the verl settings that select the rollout KV layout.

    ``decode_last`` imports ``rltt_baseline.vllm_decode_last`` in the workers (verl
    ``model.external_lib``), selects its architecture through ``hf_overrides`` and builds the
    engine without torch.compile. Settings that would contradict it are refused.
    """
    if cfg.rollout.kv_layout == "full":
        if cfg.rollout.get("log_probs"):
            return _deep_merge(overrides, {"actor_rollout_ref": {"rollout": {
                "calculate_log_probs": True}}})
        return overrides
    actor_rollout_ref = overrides.get("actor_rollout_ref") or {}
    rollout = actor_rollout_ref.get("rollout") or {}
    if rollout.get("name", "vllm") != "vllm":
        raise ValueError("rollout.kv_layout=decode_last needs the vLLM rollout")
    if rollout.get("cudagraph_capture_sizes"):
        raise ValueError("rollout.kv_layout=decode_last sets its own compilation_config; "
                         "drop actor_rollout_ref.rollout.cudagraph_capture_sizes")
    engine = (rollout.get("engine_kwargs") or {}).get("vllm") or {}
    architectures = (engine.get("hf_overrides") or {}).get("architectures")
    if architectures not in (None, [decode_last.ARCHITECTURE]):
        raise ValueError("rollout.kv_layout=decode_last selects its own vLLM architecture")
    compilation = engine.get("compilation_config")
    if compilation not in (None, decode_last.COMPILATION):
        raise ValueError("rollout.kv_layout=decode_last cannot use torch.compile")
    libs = (actor_rollout_ref.get("model") or {}).get("external_lib") or []
    libs = [libs] if isinstance(libs, str) else list(libs)
    if decode_last.LIB not in libs:
        libs.append(decode_last.LIB)
    log_probs = cfg.rollout.get("log_probs")
    return _deep_merge(overrides, {"actor_rollout_ref": {
        "model": {"external_lib": libs},
        "rollout": {
            "calculate_log_probs": True if log_probs is None else bool(log_probs),
            "engine_kwargs": {"vllm": {
                "hf_overrides": {"architectures": [decode_last.ARCHITECTURE]},
                "compilation_config": dict(decode_last.COMPILATION)}}}}})


def check_kl_not_in_reward(trainer: str, overrides: dict) -> None:
    """Refuse a run whose final verl config would add the KL to the reward."""
    in_reward = (overrides.get("algorithm") or {}).get("use_kl_in_reward",
                                                        VENDOR_KL_IN_REWARD[trainer])
    if in_reward:
        raise ValueError("KL must not be added to the reward: set "
                         "verl_overrides.algorithm.use_kl_in_reward=false "
                         "(the KL is a loss term, actor.use_kl_loss=true)")


def apply_verl_overrides(config, overrides: dict):
    """Return ``config`` (an OmegaConf verl config) deep-merged with ``overrides``."""
    if not overrides:
        return config
    OmegaConf.set_struct(config, False)
    return OmegaConf.merge(config, OmegaConf.create(overrides))
