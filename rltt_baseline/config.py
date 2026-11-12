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
``model``
    ``num_hidden_layers`` (optional): the launcher checks it against ``config.json`` of the model
    directories before training, so a config for one Ouro size cannot run with the other's
    weights (both read ``LOOPRL_MODEL_DIR``). Ouro-1.4B-Thinking has 24, Ouro-2.6B-Thinking 48.
``audit``
    ``loop_distribution``: per-loop token distributions on every response token, written to
    ``<output_dir>/audit/step_<step>/`` (``rltt_baseline.loop_audit``). Keys: ``enabled``
    (default false), ``every_n_steps`` (default 1; steps 1, 1 + n, ...), ``top_k`` (default 5),
    ``save_tokens`` (default true; false keeps ``summary.json`` and ``responses.jsonl`` only),
    ``chunk_tokens`` (default 512; response tokens per ``lm_head`` call), ``loop_weights``
    (default false; also record each token's exit-gate distribution ``exit_pdf`` and, in PLC runs,
    the PLC loop weights ``plc_weight``).
``plc``
    PLC (``rltt_baseline.plc``): GRPO on the final loop plus an off-policy auxiliary term on the
    earlier loops, ``trainer: rltt`` only. Keys and defaults are the fields of
    ``plc.PLCSettings``: ``enabled`` (false), ``weight_mode`` (is_trunc | early_prior),
    ``lam`` (0.1; 0 = GRPO), ``trunc_cap`` (1; upper cap of the cross-loop ratio, is_trunc only),
    ``kl_coef`` (0.2; forward-KL trust region of the earlier loops towards the final loop) and
    ``log_weight_stats``. When enabled
    the section is passed to the workers as ``actor_rollout_ref.actor.plc`` and the launcher
    uses ``PLCActorRolloutRefWorker``.

Overrides use OmegaConf dot-list syntax and win over the file, e.g.
``args.learning_rate=2e-6 verl_overrides.trainer.save_freq=20 wandb.mode=online``.
Values may use ``${oc.env:VAR}`` interpolation.
"""

import argparse
import difflib
import hashlib
import os
import re
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional

from omegaconf import DictConfig, OmegaConf

from rltt_baseline import decode_last
from rltt_baseline.plc import PLCSettings

SECTIONS = ("trainer", "args", "wandb", "verl_overrides", "env", "rollout", "model", "audit",
            "plc")
DICT_SECTIONS = ("args", "wandb", "verl_overrides", "env", "rollout", "model", "audit", "plc")
TRAINERS = ("rltt", "grpo")
WANDB_KEYS = {"enabled", "mode", "project", "entity", "run_name", "tags", "notes", "dir"}
LOOP_WEIGHTINGS = {"uniform", "progressive", "exit_pdf", "learned"}
ROLLOUT_KEYS = {"kv_layout", "log_probs"}
MODEL_KEYS = {"num_hidden_layers"}
AUDIT_KEYS = {"loop_distribution"}
LOOP_AUDIT_DEFAULTS = {"enabled": False, "every_n_steps": 1, "top_k": 5, "save_tokens": True,
                       "chunk_tokens": 512, "loop_weights": False}
KV_LAYOUTS = ("full", "decode_last")
# Vendor default of verl ``algorithm.use_kl_in_reward``: the GRPO script adds the KL to the
# reward, the RLTT script uses a KL loss. This project never puts the KL in the reward.
VENDOR_KL_IN_REWARD = {"rltt": False, "grpo": True}
# Vendor default of verl ``actor.use_kl_loss`` (both scripts hard-code it).
VENDOR_KL_LOSS = {"rltt": True, "grpo": False}
# verl 0.6.1 ``RolloutConfig.max_num_batched_tokens`` default; both vendor scripts leave it unset
# and hard-code ``rollout.enable_chunked_prefill: True``.
VERL_MAX_NUM_BATCHED_TOKENS = 8192


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
    bad = set(cfg.model.keys()) - MODEL_KEYS
    if bad:
        raise KeyError(f"unknown model keys {sorted(bad)}; allowed {sorted(MODEL_KEYS)}")
    bad = set(cfg.audit.keys()) - AUDIT_KEYS
    if bad:
        raise KeyError(f"unknown audit keys {sorted(bad)}; allowed {sorted(AUDIT_KEYS)}")
    check_loop_audit(cfg.audit.get("loop_distribution"))
    plc_settings(cfg)  # validates the plc section
    if cfg.rollout.get("kv_layout") is None:
        cfg.rollout.kv_layout = "full"
    if cfg.rollout.kv_layout not in KV_LAYOUTS:
        raise ValueError(f"rollout.kv_layout must be one of {KV_LAYOUTS}, got "
                         f"{cfg.rollout.kv_layout!r}")
    OmegaConf.resolve(cfg)
    return cfg


def check_loop_audit(audit) -> None:
    if audit is None:
        return
    bad = set(audit.keys()) - set(LOOP_AUDIT_DEFAULTS)
    if bad:
        raise KeyError(f"unknown audit.loop_distribution keys {sorted(bad)}; "
                       f"allowed {sorted(LOOP_AUDIT_DEFAULTS)}")
    for name, value in audit.items():
        if isinstance(LOOP_AUDIT_DEFAULTS[name], bool):
            if not isinstance(value, bool):
                raise TypeError(f"audit.loop_distribution.{name} must be a bool, got {value!r}")
        elif isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"audit.loop_distribution.{name} must be a positive int, "
                             f"got {value!r}")


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


# Local secrets file (gitignored): KEY=VALUE lines, e.g. WANDB_API_KEY=...; LOOPRL_ENV_FILE
# points elsewhere. Values are never printed or written to the resolved config.
ENV_FILE_NAME = "env.local"
_ENV_LINE = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


def read_env_file(path) -> Dict[str, str]:
    """``KEY=VALUE`` pairs of a dotenv-style file; blank lines, comments and empty values skipped.

    The file is parsed, never executed: no variable expansion, no command substitution.
    """
    values = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _ENV_LINE.match(line)
        if not match:
            continue
        key, value = match.group(1), match.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if value:
            values[key] = value
    return values


def env_file_path(environ: Mapping[str, str], default_dir) -> Path:
    return Path(environ.get("LOOPRL_ENV_FILE") or Path(default_dir) / ENV_FILE_NAME)


def load_env_file(path, environ: MutableMapping[str, str]) -> List[str]:
    """Set the file's variables that ``environ`` lacks; returns their names (never the values)."""
    path = Path(path)
    if not path.is_file():
        return []
    loaded = []
    for key, value in read_env_file(path).items():
        if not environ.get(key):
            environ[key] = value
            loaded.append(key)
    return sorted(loaded)


def wandb_key_status(cfg: DictConfig, environ: Mapping[str, str]) -> str:
    """Whether an online wandb run has ``WANDB_API_KEY``; describes, never shows, the key."""
    wandb = cfg.wandb
    if not wandb.get("enabled", True) or wandb.get("mode") != "online":
        return f"not needed: wandb mode {wandb.get('mode')}, enabled {wandb.get('enabled', True)}"
    return "found" if environ.get("WANDB_API_KEY") else "missing"


def wandb_key_problem(cfg: DictConfig, environ: Mapping[str, str], env_file) -> Optional[str]:
    if wandb_key_status(cfg, environ) != "missing":
        return None
    return (f"wandb.mode is online but WANDB_API_KEY is not set: put WANDB_API_KEY=<key> in "
            f"{env_file} (gitignored; see env.local.example) or export it before launching")


# ---------- project rules enforced at launch (AGENTS.md rules 1, 2 and 4) ----------

# Rule 1: the 14,684-row merged filtered training set, icemoon28/LoopRL-DeepMath-14684 at revision
# 6e0be984159c007d0abd3d717a55617774390907, data/train.parquet.
REQUIRED_TRAIN_SHA256 = "353b572a7c3a492b1e50d87e4d5195a461950018331c12258ff704d60c480b41"
REQUIRED_TRAIN_ROWS = 14684
# Rule 2: training seed (driver and verl dataloader).
REQUIRED_SEED = 42
# Rule 4: wandb logging.
REQUIRED_WANDB = {"enabled": True, "mode": "online", "project": "looprl-ouro"}


def train_file_check(path) -> Dict[str, object]:
    """SHA256 and row count of the training file, for the dry-run report and the launch check."""
    path = Path(str(path))
    result: Dict[str, object] = {"path": str(path), "exists": path.is_file(), "sha256": None,
                                 "rows": None, "error": None}
    if not result["exists"]:
        return result
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    result["sha256"] = digest.hexdigest()
    try:
        import pyarrow.parquet as pq

        result["rows"] = int(pq.ParquetFile(path).metadata.num_rows)
    except Exception as exc:  # not a parquet file, or pyarrow missing
        result["error"] = f"cannot read the row count: {exc!r}"
    return result


def policy_violations(cfg: DictConfig, args: argparse.Namespace, overrides: dict,
                      train_check: Dict[str, object]) -> List[str]:
    """Breaches of the project rules (AGENTS.md 1, 2, 4); the launcher refuses to train on any."""
    expected_sha256, expected_rows = REQUIRED_TRAIN_SHA256, REQUIRED_TRAIN_ROWS
    problems = []
    # rule 1: training data
    if not train_check["exists"]:
        problems.append(f"rule 1 (data): training file {train_check['path']} does not exist; "
                        "LOOPRL_TRAIN_FILE must point to the 14,684-row set (README)")
    else:
        if train_check["sha256"] != expected_sha256:
            problems.append(f"rule 1 (data): {train_check['path']} has SHA256 "
                            f"{train_check['sha256']}, expected {expected_sha256}")
        if train_check["rows"] != expected_rows:
            detail = train_check["error"] or f"{train_check['rows']} rows"
            problems.append(f"rule 1 (data): {train_check['path']}: {detail}, "
                            f"expected {expected_rows} rows")
    # rule 2: seed
    if args.seed != REQUIRED_SEED:
        problems.append(f"rule 2 (seed): args.seed is {args.seed}, must be {REQUIRED_SEED}")
    data_seed = (overrides.get("data") or {}).get("seed")
    if data_seed != REQUIRED_SEED:
        problems.append(f"rule 2 (seed): verl_overrides.data.seed is {data_seed}, must be "
                        f"{REQUIRED_SEED} (write data.seed: ${{args.seed}})")
    # rule 4: wandb
    wandb = cfg.wandb
    for key, required in REQUIRED_WANDB.items():
        if wandb.get(key) != required:
            problems.append(f"rule 4 (wandb): wandb.{key} is {wandb.get(key)!r}, "
                            f"must be {required!r}")
    return problems


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


def add_audit_overrides(cfg: DictConfig, overrides: dict, args: argparse.Namespace) -> dict:
    """``overrides`` plus ``actor_rollout_ref.loop_audit`` when the per-loop audit is on.

    The workers count ``compute_log_prob`` calls, one per step without LoRA (LoRA adds the
    reference pass, so it is refused rather than mislabelling steps). A resumed run starts the
    count at the step after the checkpoint it loads (``first_step``, see ``resume_first_step``).
    """
    settings = OmegaConf.to_container(cfg.audit.get("loop_distribution") or {}, resolve=True)
    settings = {**LOOP_AUDIT_DEFAULTS, **settings}
    if not settings.pop("enabled"):
        return overrides
    if not args.no_lora:
        raise ValueError("audit.loop_distribution needs no_lora=true")
    settings["first_step"] = resume_first_step(overrides.get("trainer") or {}, args)
    settings["dir"] = os.path.join(os.path.abspath(str(args.output_dir)), "audit")
    return _deep_merge(overrides, {"actor_rollout_ref": {"loop_audit": settings}})


def resume_first_step(trainer: dict, args: argparse.Namespace) -> int:
    """The first training step this run executes: 1, or the loaded checkpoint's step + 1.

    Mirrors verl 0.6.1 ``RayPPOTrainer._load_checkpoint``: ``resume_mode = auto`` loads the step
    named in ``<default_local_dir>/latest_checkpointed_iteration.txt`` (none: starts from
    scratch); ``resume_path`` loads ``resume_from_path``, a ``global_step_<n>`` directory. The
    vendor scripts set ``default_local_dir`` to ``args.output_dir``.
    """
    mode = trainer.get("resume_mode", "disable")
    if mode == "disable":
        return 1
    if mode == "resume_path":
        path = trainer.get("resume_from_path")
        match = re.fullmatch(r"global_step_(\d+)", os.path.basename(str(path or "").rstrip("/")))
        if not match:
            raise ValueError("trainer.resume_mode=resume_path needs trainer.resume_from_path "
                             f"pointing at a global_step_<n> directory, got {path!r}")
        return int(match.group(1)) + 1
    if mode == "auto":
        ckpt_dir = os.path.abspath(str(trainer.get("default_local_dir") or args.output_dir))
        tracker = os.path.join(ckpt_dir, "latest_checkpointed_iteration.txt")
        if not os.path.exists(tracker):
            return 1
        with open(tracker) as f:
            return int(f.read().strip()) + 1
    raise ValueError(f"unknown trainer.resume_mode {mode!r}; allowed disable, auto, resume_path")


def plc_settings(cfg: DictConfig) -> PLCSettings:
    """The validated ``plc`` section (all defaults when the section is empty)."""
    section = cfg.get("plc")
    values = OmegaConf.to_container(section, resolve=True) if section is not None else {}
    return PLCSettings.from_mapping(values)


def add_plc_overrides(cfg: DictConfig, overrides: dict, args: argparse.Namespace) -> dict:
    """``overrides`` plus ``actor_rollout_ref.actor.plc`` when PLC is on.

    PLC runs on the RLTT trainer with the per-loop model; the vendor's ``learned`` loop
    weights are a different objective and are refused.
    """
    settings = plc_settings(cfg)
    if not settings.enabled:
        return overrides
    if cfg.trainer != "rltt":
        raise ValueError("plc.enabled needs trainer=rltt")
    if not args.use_rltt_model:
        raise ValueError("plc.enabled needs args.use_rltt_model=true (per-loop model)")
    if args.loop_weighting == "learned":
        raise ValueError("plc.enabled cannot be combined with args.loop_weighting=learned")
    values = {k: v for k, v in settings.__dict__.items()}
    return _deep_merge(overrides, {"actor_rollout_ref": {"actor": {"plc": values}}})


def check_rollout_token_budget(overrides: dict, args: argparse.Namespace) -> None:
    """Refuse a vLLM token budget that verl 0.6.1 would reject when the rollout engine starts.

    The vendor scripts set ``max_model_len = max_prompt_length + max_completion_length`` and turn
    chunked prefill on; verl then raises unless ``max_num_batched_tokens >= max_model_len``. Its
    default (8192) is below the 18,432 tokens of these configs, so without an override every run
    would fail after Ray and the models have started.
    """
    rollout = (overrides.get("actor_rollout_ref") or {}).get("rollout") or {}
    if not rollout.get("enable_chunked_prefill", True):
        return
    budget = int(rollout.get("max_num_batched_tokens", VERL_MAX_NUM_BATCHED_TOKENS))
    max_model_len = int(args.max_prompt_length) + int(args.max_completion_length)
    if budget < max_model_len:
        raise ValueError(f"actor_rollout_ref.rollout.max_num_batched_tokens={budget} is below "
                         f"max_model_len={max_model_len} (prompt + completion); verl refuses this "
                         "with chunked prefill. Set verl_overrides.actor_rollout_ref.rollout."
                         "max_num_batched_tokens >= max_model_len")


def check_last_loop_logits(overrides: dict) -> None:
    """Refuse a run whose HF actor forward would not give the last loop's logits.

    vLLM samples from the last loop. verl's actor (GRPO's log-probs, RLTT's ``old_log_probs``)
    calls the official ``OuroForCausalLM.forward`` without exit arguments, so it falls back to
    ``config.early_exit_threshold`` (1.0 in the Ouro configs) and ``config.early_exit_step``;
    the threshold branch takes, per token, the first loop whose cumulative exit probability
    reaches 1.0, an earlier loop when the bf16 forward rounds the sum. Both must be overridden
    to null through ``actor_rollout_ref.model.override_config`` (set on the HF config before
    ``from_pretrained`` in verl 0.6.1).
    """
    model = (overrides.get("actor_rollout_ref") or {}).get("model") or {}
    override = model.get("override_config") or {}
    if "model_config" in override:
        # verl 0.6.1's FSDP actor passes override_config to the HF config as is (only the
        # rollout-side HFModelConfig unwraps model_config), so the nested form never reaches it
        raise ValueError("set actor_rollout_ref.model.override_config.early_exit_threshold at the "
                         "top level, not under override_config.model_config (the FSDP actor of "
                         "verl 0.6.1 does not unwrap it)")
    if "early_exit_threshold" not in override or override["early_exit_threshold"] is not None:
        raise ValueError("the HF actor forward must use the last loop, the one vLLM samples "
                         "from: set verl_overrides.actor_rollout_ref.model.override_config."
                         "early_exit_threshold=null (the Ouro config's 1.0 makes the official "
                         "forward exit early under bf16)")
    if override.get("early_exit_step") is not None:
        raise ValueError("actor_rollout_ref.model.override_config.early_exit_step must be unset "
                         "or null: the official forward would use that loop instead of the last")


def check_kl_not_in_reward(trainer: str, overrides: dict) -> None:
    """Refuse a run whose final verl config would add the KL to the reward."""
    in_reward = (overrides.get("algorithm") or {}).get("use_kl_in_reward",
                                                        VENDOR_KL_IN_REWARD[trainer])
    if in_reward:
        raise ValueError("KL must not be added to the reward: set "
                         "verl_overrides.algorithm.use_kl_in_reward=false "
                         "(the KL, if any, is a loss term: actor.use_kl_loss)")


def kl_uses_reference(verl_config) -> bool:
    """True when the final verl config reads the reference policy: KL loss or KL in the reward.

    verl 0.6.1 builds the reference model whenever ``Role.RefPolicy`` is in the trainer's
    ``role_worker_mapping``, whether or not anything uses its log-probs.
    """
    actor = verl_config.actor_rollout_ref.actor
    algorithm = verl_config.get("algorithm") or {}
    return bool(actor.get("use_kl_loss", False)) or bool(algorithm.get("use_kl_in_reward", False))


def drop_unused_reference(role_worker_mapping: dict, verl_config):
    """``(mapping, dropped)``: ``role_worker_mapping`` minus ``Role.RefPolicy`` if no KL uses it.

    Only the GRPO vendor script registers a reference worker; with the KL off it would still load
    a frozen model copy and run one extra forward per step whose output nothing reads. Keys are
    matched by enum name so this works without importing verl.
    """
    if kl_uses_reference(verl_config):
        return role_worker_mapping, False
    kept = {role: worker for role, worker in role_worker_mapping.items()
            if getattr(role, "name", None) != "RefPolicy"}
    return kept, len(kept) != len(role_worker_mapping)


def reference_policy_status(trainer: str, overrides: dict) -> str:
    """What the run does with the reference model, from the vendor defaults and ``overrides``.

    Mirrors the launcher at trainer construction: the RLTT script never registers a reference
    worker; the GRPO script does, and the launcher removes it when neither KL uses it.
    """
    if trainer == "rltt":
        return "none: the rltt vendor script registers no reference worker"
    actor = (overrides.get("actor_rollout_ref") or {}).get("actor") or {}
    kl_loss = actor.get("use_kl_loss", VENDOR_KL_LOSS[trainer])
    kl_reward = (overrides.get("algorithm") or {}).get("use_kl_in_reward",
                                                       VENDOR_KL_IN_REWARD[trainer])
    if kl_loss or kl_reward:
        return "built: the KL reads the reference log-probs"
    return "dropped: KL off, Role.RefPolicy removed before the trainer is built"


def apply_verl_overrides(config, overrides: dict):
    """Return ``config`` (an OmegaConf verl config) deep-merged with ``overrides``."""
    if not overrides:
        return config
    OmegaConf.set_struct(config, False)
    return OmegaConf.merge(config, OmegaConf.create(overrides))
