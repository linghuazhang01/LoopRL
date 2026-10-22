"""Build the ``Ouro-*-Thinking-RLTT`` model directory that the RLTT trainer loads.

The directory reuses the source model's weights and tokenizer through symlinks and adds:

- ``modeling_ouro_base.py``: the source model's own ``modeling_ouro.py``, unchanged;
- ``modeling_ouro.py`` and ``rltt_per_loop.py`` from ``rltt_baseline/ouro_rltt/``;
- ``config.json``: the source config plus the ``rltt_*`` keys;
- ``rltt_model_manifest.json``: SHA256 of every written file and of the source modeling file.

The source ``modeling_ouro.py`` must match an allowed SHA256 (by default the vendored
official file, see ``paths.ouro_source_dir``; Hugging Face ``ByteDance/Ouro-2.6B-Thinking``
revision f1edd81e); otherwise the script stops, because the
wrapper relies on the official forward returning ``(outputs, hidden_states_list, gate_list)``.

Usage (stdlib only; run inside a Slurm allocation on the HPC, it only touches files):
    python -m rltt_baseline.prepare_model --source <Ouro-2.6B-Thinking dir> --out <dir>
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from rltt_baseline.paths import PKG_DIR, ouro_source_dir

PAYLOAD_DIR = PKG_DIR / "ouro_rltt"
OWN_FILES = {"modeling_ouro.py", "modeling_ouro_base.py", "rltt_per_loop.py", "config.json",
             "rltt_model_manifest.json"}

# The official file decorates OuroModel.forward with ``@check_model_inputs`` (transformers
# 4.55 API). transformers 4.57 (pinned by the RLTT requirements) turned it into a decorator
# factory, so the official file fails with "wrapped_fn() got an unexpected keyword argument".
# This is the only edit made to the official code; it works with both APIs.
CHECK_INPUTS_IMPORT = "from transformers.utils.generic import check_model_inputs\n"
CHECK_INPUTS_COMPAT = '''from transformers.utils.generic import (  # rltt_baseline compat edit
    check_model_inputs as _check_model_inputs,
)
import inspect as _inspect


def check_model_inputs(func):  # rltt_baseline compat edit (see prepare_model.py)
    if "tie_last_hidden_states" in _inspect.signature(_check_model_inputs).parameters:
        return _check_model_inputs()(func)
    return _check_model_inputs(func)
'''


def compat_base_source(official: str) -> str:
    if official.count(CHECK_INPUTS_IMPORT) != 1:
        raise ValueError("official modeling_ouro.py does not import check_model_inputs as expected")
    return official.replace(CHECK_INPUTS_IMPORT, CHECK_INPUTS_COMPAT)


def vendor_modeling() -> Path:
    return ouro_source_dir() / "modeling_ouro.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build(source: Path, out: Path, allowed_modeling_sha: set, chunk_size, checkpoint_chunks: bool,
          detach_exit_pdf: bool) -> dict:
    source = source.resolve()
    for name in ("config.json", "modeling_ouro.py", "configuration_ouro.py"):
        if not (source / name).is_file():
            raise FileNotFoundError(f"{source / name} not found")
    source_modeling_sha = sha256(source / "modeling_ouro.py")
    if source_modeling_sha not in allowed_modeling_sha:
        raise ValueError(
            f"source modeling_ouro.py SHA256 {source_modeling_sha} is not in the allowed set "
            f"{sorted(allowed_modeling_sha)}; check the file before passing --allow-modeling-sha"
        )
    config = json.loads((source / "config.json").read_text())
    arch = config.get("architectures") or []
    if arch != ["OuroForCausalLM"]:
        raise ValueError(f"unexpected architectures {arch}")
    auto_map = config.get("auto_map", {})
    if auto_map.get("AutoModelForCausalLM") != "modeling_ouro.OuroForCausalLM":
        raise ValueError(f"unexpected auto_map {auto_map}")

    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists and is not empty; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)

    linked = []
    for entry in sorted(source.iterdir()):
        if entry.name in OWN_FILES or entry.name.startswith("."):
            continue
        os.symlink(entry, out / entry.name)
        linked.append(entry.name)

    (out / "modeling_ouro_base.py").write_text(
        compat_base_source((source / "modeling_ouro.py").read_text()))
    shutil.copy2(PAYLOAD_DIR / "modeling_ouro.py", out / "modeling_ouro.py")
    shutil.copy2(PAYLOAD_DIR / "rltt_per_loop.py", out / "rltt_per_loop.py")
    config["rltt_logprob_chunk_size"] = chunk_size
    config["rltt_logprob_checkpoint"] = bool(checkpoint_chunks)
    config["rltt_detach_exit_pdf"] = bool(detach_exit_pdf)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    written = ["modeling_ouro_base.py", "modeling_ouro.py", "rltt_per_loop.py", "config.json"]
    manifest = {
        "source_dir": str(source),
        "source_modeling_sha256": source_modeling_sha,
        "base_edits": ["check_model_inputs compat wrapper (transformers 4.55-4.57)"],
        "written_sha256": {name: sha256(out / name) for name in written},
        "symlinked": linked,
        "rltt_config": {k: config[k] for k in config if k.startswith("rltt_")},
    }
    (out / "rltt_model_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True, help="official Ouro model dir")
    parser.add_argument("--out", type=Path, required=True, help="RLTT model dir to create")
    parser.add_argument("--chunk-size", type=int, default=2048,
                        help="positions per log-prob chunk (run_rltt.slurm default 2048)")
    parser.add_argument("--checkpoint-chunks", action="store_true",
                        help="recompute per-loop logits in backward to save memory")
    parser.add_argument("--detach-exit-pdf", action="store_true",
                        help="stop gradients through exit_pdf loop weights (vendor actor does not)")
    parser.add_argument("--allow-modeling-sha", action="append", default=[],
                        help="extra allowed SHA256 for the source modeling_ouro.py")
    args = parser.parse_args(argv)
    allowed = {sha256(vendor_modeling()), *args.allow_modeling_sha}
    manifest = build(args.source, args.out, allowed, args.chunk_size, args.checkpoint_chunks,
                     args.detach_exit_pdf)
    json.dump(manifest, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
