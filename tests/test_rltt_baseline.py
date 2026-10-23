import json  # noqa: I001  (first-party grouping differs between the two checkouts)
import sys
import types

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from rltt_baseline import config as rcfg
from rltt_baseline import metrics as rmet
from rltt_baseline import paths, prepare_model
from rltt_baseline.ouro_rltt.rltt_per_loop import exit_pdf_from_gates, per_loop_token_stats

CONFIG = paths.config_file("rltt_baseline.yaml")
ENV = {
    "LOOPRL_OURO_DIR": "/models/Ouro-2.6B-Thinking",
    "LOOPRL_MODEL_DIR": "/models/Ouro-2.6B-Thinking-RLTT",
    "LOOPRL_MATH_TRAIN_JSONL": "/data/math_train.jsonl",
    "LOOPRL_MATH500_JSONL": "/data/math500.jsonl",
    "LOOPRL_OUTPUT_DIR": "/out/rltt",
}


# ---------- per-loop helpers ----------

def test_exit_pdf_matches_ouro_recursion_and_sums_to_one():
    torch.manual_seed(0)
    gates = [torch.randn(2, 5, 1) for _ in range(4)]
    pdf = exit_pdf_from_gates(gates)
    lam = [torch.sigmoid(g.squeeze(-1)) for g in gates]
    expected = [lam[0], (1 - lam[0]) * lam[1], (1 - lam[0]) * (1 - lam[1]) * lam[2],
                (1 - lam[0]) * (1 - lam[1]) * (1 - lam[2])]
    assert pdf.shape == (2, 5, 4)
    torch.testing.assert_close(pdf, torch.stack(expected, -1))
    torch.testing.assert_close(pdf.sum(-1), torch.ones(2, 5))


def _reference_stats(hidden, head, labels, temperature):
    logits = head(hidden).float() / temperature
    logp = F.log_softmax(logits, -1)
    lp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    ent = -(logp.exp() * logp).sum(-1)
    return lp, ent


@pytest.mark.parametrize("chunk", [None, 0, 3, 7, 100])
def test_per_loop_stats_match_log_softmax_for_any_chunking(chunk):
    torch.manual_seed(1)
    head = torch.nn.Linear(8, 11, bias=False)
    hiddens = [torch.randn(2, 7, 8) for _ in range(3)]
    labels = torch.randint(0, 11, (2, 7))
    lps, ents = per_loop_token_stats(hiddens, head, labels, temperature=0.7, chunk_size=chunk)
    for h, lp, ent in zip(hiddens, lps, ents):
        ref_lp, ref_ent = _reference_stats(h, head, labels, 0.7)
        torch.testing.assert_close(lp, ref_lp)
        torch.testing.assert_close(ent, ref_ent)
        assert not ent.requires_grad


def test_checkpointed_chunks_give_same_gradients():
    torch.manual_seed(2)
    head = torch.nn.Linear(8, 13, bias=False)
    labels = torch.randint(0, 13, (3, 9))
    base = torch.randn(3, 9, 8)

    grads = []
    for ckpt in (False, True):
        head.zero_grad()
        h = base.clone().requires_grad_(True)
        lps, _ = per_loop_token_stats([h, h * 2], head, labels, chunk_size=4,
                                      checkpoint_chunks=ckpt)
        sum(lp.sum() for lp in lps).backward()
        grads.append((h.grad.clone(), head.weight.grad.clone()))
    torch.testing.assert_close(grads[0][0], grads[1][0])
    torch.testing.assert_close(grads[0][1], grads[1][1])


def test_per_loop_stats_rejects_non_positive_temperature():
    head = torch.nn.Linear(4, 5, bias=False)
    with pytest.raises(ValueError):
        per_loop_token_stats([torch.randn(1, 2, 4)], head, torch.zeros(1, 2, dtype=torch.long),
                             temperature=0)


# ---------- reasoning metrics ----------

def _toy_metrics_inputs():
    # 2 prompts x 3 rollouts; max length 4
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0],
                         [1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 1, 1]])
    scores = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    uids = ["a", "a", "a", "b", "b", "b"]
    return mask, scores, uids


def test_compute_reasoning_metrics_values():
    mask, scores, uids = _toy_metrics_inputs()
    adv = torch.zeros(6, 4)
    adv[0, :2] = 0.5
    out = rmet.compute_reasoning_metrics(mask, scores, uids, max_response_length=4,
                                         advantages=adv, has_answer=[1, 0, 1, 1, 1, 0],
                                         levels=["Level 1"] * 3 + ["Level 5"] * 3)
    assert out["reasoning/accuracy"] == pytest.approx(2 / 6)
    assert out["reasoning/truncation_rate"] == pytest.approx(3 / 6)
    assert out["reasoning/truncated_accuracy"] == pytest.approx(0.0)
    assert out["reasoning/pass_at_group"] == pytest.approx(0.5)
    assert out["reasoning/group_mixed_ratio"] == pytest.approx(0.5)
    assert out["reasoning/group_all_wrong_ratio"] == pytest.approx(0.5)
    assert out["reasoning/group_all_correct_ratio"] == pytest.approx(0.0)
    assert out["reasoning/response_length/correct_mean"] == pytest.approx(1.5)
    assert out["reasoning/response_length/wrong_mean"] == pytest.approx(15 / 4)
    assert out["reasoning/answer_rate"] == pytest.approx(4 / 6)
    assert out["reasoning/nonzero_adv_token_ratio"] == pytest.approx(2 / 18)
    assert out["reasoning/accuracy/level_1"] == pytest.approx(2 / 3)
    assert out["reasoning/accuracy/level_5"] == pytest.approx(0.0)


class _FakeTD(dict):
    def keys(self):
        return super().keys()


def _fake_batch():
    mask, scores, uids = _toy_metrics_inputs()
    token_scores = torch.zeros(6, 4)
    lengths = mask.sum(-1)
    for i, s in enumerate(scores):
        token_scores[i, lengths[i] - 1] = s
    responses = torch.arange(24).reshape(6, 4)
    batch = types.SimpleNamespace(
        batch=_FakeTD(responses=responses, response_mask=mask, token_level_scores=token_scores),
        non_tensor_batch={"uid": uids, "extra_info": [{"level": "Level 2"}] * 6},
    )
    return batch


def test_batch_reasoning_metrics_reads_dataproto_fields():
    decode = lambda ids: "\\boxed{1}" if ids[0] % 8 == 0 else "no answer"  # noqa: E731
    extract = lambda text: "1" if "boxed" in text else None  # noqa: E731
    out = rmet.batch_reasoning_metrics(_fake_batch(), decode, extract)
    assert out["reasoning/accuracy"] == pytest.approx(2 / 6)
    assert out["reasoning/accuracy/level_2"] == pytest.approx(2 / 6)
    assert out["reasoning/answer_rate"] == pytest.approx(3 / 6)


def test_install_driver_metrics_wraps_verl_function(monkeypatch):
    calls = {}

    def original(batch, use_critic=True):
        calls["called"] = True
        return {"critic/score/mean": 0.1}

    metric_utils = types.ModuleType("verl.trainer.ppo.metric_utils")
    metric_utils.compute_data_metrics = original
    ppo = types.ModuleType("verl.trainer.ppo")
    ppo.metric_utils = metric_utils
    for name, mod in {"verl": types.ModuleType("verl"),
                      "verl.trainer": types.ModuleType("verl.trainer"),
                      "verl.trainer.ppo": ppo,
                      "verl.trainer.ppo.metric_utils": metric_utils}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    rmet.install_driver_metrics()
    out = metric_utils.compute_data_metrics(_fake_batch(), use_critic=False)
    assert calls["called"] and out["critic/score/mean"] == 0.1
    assert "reasoning/accuracy" in out
    rmet.install_driver_metrics()  # idempotent
    assert metric_utils.compute_data_metrics._rltt_reasoning_wrapped


# ---------- config ----------

@pytest.fixture
def env(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)


def _defaults():
    # subset of vendor rltt_train.parse_args() defaults with matching types
    return types.SimpleNamespace(**{
        "model_path": "x", "use_rltt_model": False, "rltt_model_path": "x",
        "rltt_logprob_chunk_size": 256, "total_ut_steps": 4, "no_lora": False,
        "no_sft_checkpoint": False, "no_gradient_checkpointing": False, "ref_offload": False,
        "train_file": "x", "math500_test_file": "x", "max_prompt_length": 512,
        "max_completion_length": 4096, "loop_weighting": "uniform", "progressive_alpha": 1.0,
        "beta": 0.001, "num_generations": 8, "num_prompts_per_batch": 128, "temperature": 0.9,
        "learning_rate": 1e-6, "weight_decay": 0.1, "max_grad_norm": 0.1, "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine", "gradient_accumulation_steps": 1, "num_train_epochs": 20,
        "max_steps": -1, "seed": 42, "ppo_max_token_len": 8192, "log_prob_max_token_len": 16384,
        "dataloader_num_workers": 8, "use_vllm": True, "no_vllm": False,
        "vllm_gpu_memory_utilization": 0.7, "vllm_tensor_parallel_size": 1,
        "vllm_enforce_eager": False, "n_gpus": None, "nnodes": 1, "output_dir": "./o",
        "save_steps": 5, "logging_steps": 1, "no_validation": False,
        "skip_initial_validation": False, "eval_steps": 7, "val_batch_size": 64,
        "val_max_new_tokens": 4096, "no_wandb": False, "wandb_project": "p", "debug": False,
    })


def test_default_config_resolves_to_paper_settings(env):
    cfg = rcfg.load_config(str(CONFIG))
    args = rcfg.build_args(cfg, _defaults())
    assert args.model_path == ENV["LOOPRL_OURO_DIR"]
    assert args.use_rltt_model and args.loop_weighting == "exit_pdf"
    assert args.max_completion_length == 2048 and args.max_steps == 140
    assert args.num_prompts_per_batch == 32
    assert args.temperature == 1.0 and isinstance(args.learning_rate, float)
    assert args.no_wandb is False and args.wandb_project == "rltt-ouro-baseline"
    env_vars = rcfg.wandb_env(cfg)
    assert env_vars["WANDB_MODE"] == "offline"
    assert env_vars["WANDB_DIR"] == ENV["LOOPRL_OUTPUT_DIR"]
    assert env_vars["WANDB_TAGS"] == "rltt,baseline,ouro-2.6b-thinking"


def test_cli_overrides_win(env):
    cfg = rcfg.load_config(str(CONFIG), ["args.learning_rate=2e-6", "args.max_steps=10",
                                         "wandb.mode=online", "wandb.run_name=try1",
                                         "verl_overrides.trainer.save_freq=20"])
    args = rcfg.build_args(cfg, _defaults())
    assert args.learning_rate == 2e-6 and args.max_steps == 10
    assert rcfg.wandb_env(cfg)["WANDB_MODE"] == "online"
    assert rcfg.verl_overrides(cfg) == {"trainer": {"save_freq": 20, "experiment_name": "try1"}}


def test_unknown_or_mistyped_args_fail(env):
    with pytest.raises(KeyError, match="learnng_rate"):
        rcfg.build_args(rcfg.load_config(str(CONFIG), ["args.learnng_rate=1"]), _defaults())
    with pytest.raises(TypeError):
        rcfg.build_args(rcfg.load_config(str(CONFIG), ["args.max_steps=1.5"]), _defaults())
    with pytest.raises(KeyError):
        rcfg.load_config(str(CONFIG), ["nosuchsection.foo=1"])
    with pytest.raises(ValueError, match="trainer must be"):
        rcfg.load_config(str(CONFIG), ["trainer=ppo"])
    with pytest.raises(ValueError, match="exit_pdf"):
        rcfg.build_args(rcfg.load_config(str(CONFIG), ["args.use_rltt_model=false"]), _defaults())


def test_wandb_disabled_sets_no_wandb(env):
    cfg = rcfg.load_config(str(CONFIG), ["wandb.mode=disabled"])
    assert rcfg.build_args(cfg, _defaults()).no_wandb is True


def test_apply_verl_overrides_deep_merges():
    base = OmegaConf.create({"trainer": {"save_freq": 10, "logger": ["wandb"]}, "data": {"x": 1}})
    merged = rcfg.apply_verl_overrides(base, {"trainer": {"save_freq": 20, "new": True}})
    assert merged.trainer.save_freq == 20 and merged.trainer.new is True
    assert list(merged.trainer.logger) == ["wandb"] and merged.data.x == 1


def test_launch_dry_run_uses_vendor_defaults(env, capsys, monkeypatch):
    from rltt_baseline import launch

    # importing the vendor rltt_train sets these; restore them after the test
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "PYTHONPATH"):
        monkeypatch.setenv(key, "")
    monkeypatch.setattr(sys, "path", list(sys.path))

    assert launch.main(["--config", str(CONFIG), "--dry-run", "args.seed=7"]) == 0
    resolved = json.loads(capsys.readouterr().out)
    vendor = resolved["vendor_args"]
    assert vendor["seed"] == 7 and vendor["rltt_model_path"] == ENV["LOOPRL_MODEL_DIR"]
    assert vendor["forward_batch_size"] == 2  # vendor default for a key not in the YAML
    assert resolved["env"]["TOKENIZERS_PARALLELISM"] == "false"


# ---------- prepare_model ----------

def _fake_source(tmp_path):
    src = tmp_path / "Ouro-2.6B-Thinking"
    src.mkdir()
    vendor = paths.ouro_source_dir()
    for name in ("modeling_ouro.py", "configuration_ouro.py", "config.json"):
        (src / name).write_bytes((vendor / name).read_bytes())
    (src / "tokenizer.json").write_text("{}")
    (src / "model.safetensors").write_bytes(b"weights")
    return src


def test_prepare_model_builds_directory(tmp_path):
    src = _fake_source(tmp_path)
    out = tmp_path / "rltt"
    allowed = {prepare_model.sha256(prepare_model.vendor_modeling())}
    manifest = prepare_model.build(src, out, allowed, 2048, False, False)
    assert (out / "model.safetensors").is_symlink()
    official = (src / "modeling_ouro.py").read_text()
    base = (out / "modeling_ouro_base.py").read_text()
    assert base == prepare_model.compat_base_source(official) and base != official
    assert "def check_model_inputs(func):" in base
    assert "OuroForCausalLM(_OfficialOuroForCausalLM)" in (out / "modeling_ouro.py").read_text()
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["rltt_logprob_chunk_size"] == 2048 and cfg["rltt_detach_exit_pdf"] is False
    assert cfg["auto_map"]["AutoModelForCausalLM"] == "modeling_ouro.OuroForCausalLM"
    assert set(manifest["written_sha256"]) == {"modeling_ouro_base.py", "modeling_ouro.py",
                                               "rltt_per_loop.py", "config.json"}
    with pytest.raises(FileExistsError):
        prepare_model.build(src, out, allowed, 2048, False, False)


def test_prepare_model_rejects_unknown_modeling_file(tmp_path):
    src = _fake_source(tmp_path)
    (src / "modeling_ouro.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="not in the allowed set"):
        prepare_model.build(src, tmp_path / "rltt", {"0" * 64}, 2048, False, False)


# ---------- GRPO trainer ----------

GRPO_CONFIG = paths.config_file("grpo_baseline_8gpu.yaml")


def test_grpo_dry_run_resolves_with_vendor_grpo_defaults(env, capsys, monkeypatch):
    from rltt_baseline import launch

    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "PYTHONPATH"):
        monkeypatch.setenv(key, "")
    monkeypatch.setattr(sys, "path", list(sys.path))
    assert launch.main(["--config", str(GRPO_CONFIG), "--dry-run", "args.max_steps=3"]) == 0
    resolved = json.loads(capsys.readouterr().out)
    vendor = resolved["vendor_args"]
    assert resolved["trainer"] == "grpo"
    assert vendor["model_path"] == ENV["LOOPRL_MODEL_DIR"] and vendor["use_ppo"] is False
    assert vendor["max_steps"] == 3 and vendor["n_gpus"] == 8
    assert vendor["ppo_epochs"] == 4  # vendor default, unused when use_ppo is false
    assert resolved["verl_overrides"]["data"]["train_batch_size"] == 64
    assert resolved["verl_overrides"]["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] == 64


def test_grpo_rejects_ppo_and_rltt_only_keys(env):
    cfg = rcfg.load_config(str(GRPO_CONFIG), ["args.use_ppo=true"])
    defaults = types.SimpleNamespace(**{k: v for k, v in vars(_defaults()).items()
                                        if k not in ("loop_weighting", "use_rltt_model")},
                                     use_ppo=False)
    with pytest.raises(ValueError, match="PPO"):
        rcfg.build_args(cfg, defaults)


def test_check_grpo_model_dir_refuses_official_modeling(tmp_path):
    from rltt_baseline.launch import check_grpo_model_dir

    official = tmp_path / "official"
    official.mkdir()
    (official / "modeling_ouro.py").write_text(
        (paths.ouro_source_dir() / "modeling_ouro.py").read_text())
    with pytest.raises(SystemExit, match="edit in place"):
        check_grpo_model_dir(str(official))
    built = tmp_path / "built"
    allowed = {prepare_model.sha256(prepare_model.vendor_modeling())}
    src = _fake_source(tmp_path)
    prepare_model.build(src, built, allowed, 2048, False, False)
    check_grpo_model_dir(str(built))  # our shim has no decorator: accepted
