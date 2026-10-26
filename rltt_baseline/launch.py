"""Launch the vendored RLTT or GRPO trainer from a YAML config.

    python -m rltt_baseline.launch --config code/configs/rltt_baseline.yaml \
        [--dry-run] [args.learning_rate=2e-6 wandb.mode=online ...]

``trainer: rltt`` runs ``rltt_experiments/rltt_train.py``; ``trainer: grpo`` runs
``grpo_experiments/grpo_train.py`` (standard verl GRPO, the RLTT paper's GRPO baseline).
What this adds around the vendor script (whose files are not modified; located by
``paths.rltt_source_dir``):

1. argparse values come from the YAML ``args`` section (vendor defaults for unlisted keys);
2. ``verl_overrides`` are merged into the final verl config inside ``RayPPOTrainer.__init__``;
3. ``wandb`` settings become ``--no_wandb`` / ``--wandb_project``, the experiment name and
   ``WANDB_*`` variables;
4. the driver adds ``reasoning/*`` metrics; for RLTT the trainer also uses
   ``MetricsRLTTActorRolloutRefWorker`` (per-loop entropy, expected exit step, total loss);
5. a ``train_file`` ending in ``.parquet`` is used as a verl dataset with its own prompts
   (``rltt_baseline.verl_data``) instead of the vendor's MATH JSONL conversion;
6. ``rollout.kv_layout=decode_last`` adds the verl settings for decode_last rollouts
   (``rltt_baseline.config.add_rollout_overrides``);
7. ``model.num_hidden_layers``, when set, is checked against the model directories'
   ``config.json`` (the 1.4B and 2.6B configs share ``LOOPRL_MODEL_DIR``);
8. the resolved config is saved to ``<output_dir>/rltt_baseline_config.yaml``.

GRPO safety: the vendor ``grpo_train.py`` edits ``<model>/modeling_ouro.py`` in place to drop
``@check_model_inputs``. The launcher therefore refuses a model directory whose
``modeling_ouro.py`` still has that decorator (an official, possibly shared, model directory);
use the directory built by ``rltt_baseline.prepare_model``.

Only the vendor multi-GPU verl path is supported (the vendor falls back to its simple
single-GPU trainer when fewer than two GPUs are visible, which bypasses 2-4).
"""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

from rltt_baseline.config import (
    add_rollout_overrides,
    apply_verl_overrides,
    build_args,
    check_kl_not_in_reward,
    load_config,
    verl_overrides,
    wandb_env,
)
from rltt_baseline.paths import SRC_DIR, rltt_source_dir

TRAINER_MODULES = {"rltt": ("rltt_experiments", "rltt_train"),
                   "grpo": ("grpo_experiments", "grpo_train")}
OFFICIAL_DECORATOR = "    @check_model_inputs\n    @auto_docstring\n    def forward("


def setup_paths(trainer: str) -> None:
    """Make the vendor modules importable here and in Ray workers (which inherit PYTHONPATH).

    Only the chosen experiments directory is added: both define modules named ``config``,
    ``data_utils`` and ``simple_trainer``.
    """
    vendor = rltt_source_dir()
    paths = [str(p) for p in (SRC_DIR, vendor / TRAINER_MODULES[trainer][0], vendor)]
    for path in reversed(paths):
        if path not in sys.path:
            sys.path.insert(0, path)
    existing = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    os.environ["PYTHONPATH"] = os.pathsep.join(paths + [p for p in existing if p not in paths])


def vendor_defaults(module) -> argparse.Namespace:
    saved = sys.argv
    sys.argv = [module.__file__]
    try:
        return module.parse_args()
    finally:
        sys.argv = saved


def check_grpo_model_dir(model_path: str) -> None:
    modeling = Path(model_path) / "modeling_ouro.py"
    if modeling.is_file() and OFFICIAL_DECORATOR in modeling.read_text():
        raise SystemExit(f"{model_path} holds the official modeling_ouro.py, which grpo_train.py "
                         "would edit in place; pass the prepare_model output directory instead")


def check_model_layers(args, expected) -> None:
    """Refuse model directories whose ``config.json`` has another ``num_hidden_layers``."""
    if expected is None:
        return
    dirs = [args.model_path]
    if getattr(args, "use_rltt_model", False):
        dirs.append(args.rltt_model_path)
    for model_dir in dirs:
        found = json.loads((Path(model_dir) / "config.json").read_text()).get("num_hidden_layers")
        if found != expected:
            raise SystemExit(f"{model_dir}: num_hidden_layers {found}, but the config expects "
                             f"{expected} (model.num_hidden_layers); wrong model directory?")


def install_trainer_overrides(overrides: dict) -> None:
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    original = RayPPOTrainer.__init__
    if getattr(original, "_rltt_overrides", False):
        return

    def __init__(self, config, *args, **kwargs):
        original(self, apply_verl_overrides(config, overrides), *args, **kwargs)

    __init__._rltt_overrides = True
    RayPPOTrainer.__init__ = __init__


def install_parquet_train_data() -> None:
    """Route a ``.parquet`` train_file to ``verl_data.copy_verl_parquet`` (prompts unchanged)."""
    import data_utils

    from rltt_baseline.verl_data import copy_verl_parquet

    original = data_utils.prepare_train_data
    if getattr(original, "_rltt_parquet", False):
        return

    def prepare_train_data(train_file, output_dir, *args, **kwargs):
        if not str(train_file).endswith(".parquet"):
            return original(train_file, output_dir, *args, **kwargs)
        return copy_verl_parquet(train_file, os.path.join(output_dir, "train.parquet"))

    prepare_train_data._rltt_parquet = True
    data_utils.prepare_train_data = prepare_train_data


def install_metrics_worker() -> None:
    import verl_rltt

    from rltt_baseline.workers import MetricsRLTTActorRolloutRefWorker

    verl_rltt.RLTTActorRolloutRefWorker = MetricsRLTTActorRolloutRefWorker


def install_reasoning_metrics(args) -> None:
    from math_utils import extract_boxed_answer

    from rltt_baseline.metrics import install_driver_metrics

    use_rltt = getattr(args, "use_rltt_model", False)
    model_path = args.rltt_model_path if use_rltt else args.model_path
    tokenizer = {}

    def decode(token_ids):
        if "tok" not in tokenizer:
            from transformers import AutoTokenizer

            tokenizer["tok"] = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True, local_files_only=True
            )
        return tokenizer["tok"].decode(token_ids, skip_special_tokens=True)

    install_driver_metrics(decode=decode, extract_answer=extract_boxed_answer)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="RLTT / GRPO baseline launcher")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve and print the configuration, then exit without training")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dot-list overrides")
    opts = parser.parse_args(argv)

    cfg = load_config(opts.config, opts.overrides)
    setup_paths(cfg.trainer)
    vendor = importlib.import_module(TRAINER_MODULES[cfg.trainer][1])  # sets HF_*_OFFLINE=1

    args = build_args(cfg, vendor_defaults(vendor))
    env = {**wandb_env(cfg), **{str(k): str(v) for k, v in cfg.env.items()}}
    overrides = add_rollout_overrides(cfg, verl_overrides(cfg))
    check_kl_not_in_reward(cfg.trainer, overrides)

    resolved = {
        "trainer": cfg.trainer,
        "config_file": str(Path(opts.config).resolve()),
        "cli_overrides": list(opts.overrides),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "vendor_args": vars(args),
        "env": env,
        "verl_overrides": overrides,
    }
    if opts.dry_run:
        print(json.dumps(resolved, indent=2, default=str))
        return 0

    n_gpus = args.n_gpus
    if n_gpus is None:
        import torch

        n_gpus = torch.cuda.device_count()
    if n_gpus <= 1 or args.debug:
        raise SystemExit("rltt_baseline needs the vendor verl path: >= 2 GPUs and debug=false")
    check_model_layers(args, cfg.model.get("num_hidden_layers"))
    if cfg.trainer == "grpo":
        check_grpo_model_dir(args.model_path)
    if str(args.train_file).endswith(".parquet"):
        from rltt_baseline.verl_data import load_rows

        load_rows(args.train_file)  # fail before Ray starts

    os.makedirs(args.output_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(args.output_dir, "rltt_baseline_config.yaml"), resolve=True)
    with open(os.path.join(args.output_dir, "rltt_baseline_resolved.json"), "w") as fh:
        json.dump(resolved, fh, indent=2, default=str)

    os.environ.update(env)
    vendor.parse_args = lambda: args
    install_trainer_overrides(overrides)
    install_parquet_train_data()
    if cfg.trainer == "rltt":
        install_metrics_worker()
    install_reasoning_metrics(args)
    vendor.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
