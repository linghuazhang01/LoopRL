"""Get Ouro-2.6B- or Ouro-1.4B-Thinking at a pinned revision and build the training model dir.

1. Download ``ByteDance/Ouro-<size>-Thinking`` at the pinned revision (or reuse ``--source``) and
   check the SHA256 of ``model.safetensors``, ``modeling_ouro.py``, ``configuration_ouro.py``
   and ``config.json``.
2. Build ``<out>/Ouro-<size>-Thinking-RLTT`` with ``prepare_model`` (weights symlinked, official
   modeling + transformers 4.57 compat edit, per-loop RLTT outputs). Both the RLTT and the
   GRPO trainer use this directory.

    python -m rltt_baseline.fetch_model --out <models dir> [--model 1.4B] [--source <official dir>]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from rltt_baseline import prepare_model

# The 1.4B modeling_ouro.py differs from the 2.6B (vendored) file only in whitespace, and its
# configuration_ouro.py only lacks the 2.6B line ``self.pad_token_id = 2``, which
# PretrainedConfig.__init__ overwrites anyway; prepare_model therefore accepts its SHA256.
MODELS = {
    "2.6B": {
        "repo_id": "ByteDance/Ouro-2.6B-Thinking",
        "revision": "f1edd81e7ac41355db670500ceaf204e0f73af68",
        "sha256": {
            "model.safetensors": "c506a79247dc51fc0400d789365c3d43932f718abce9810f3606ace47d0a3080",
            "modeling_ouro.py": "263c097a2ed6870ce549a65ef6d6bb9e8e167423f87d2c494e0988b9a4f63f4b",
            "configuration_ouro.py":
                "ba60253335cecc5bd9cb3e5f8cfbd214aa7e865fc97a98208976fb2517811de6",
            "config.json": "ad4b51dffae60bebeeac5e1d5055e78d01fc01b1a29cd89cd2e50edb848f645a",
        },
    },
    "1.4B": {
        "repo_id": "ByteDance/Ouro-1.4B-Thinking",
        "revision": "3aaa2224253a92ca45cf2e3d427c360e1ef9c93d",
        "sha256": {
            "model.safetensors": "6730581fd366f6d1689728c61637520228385cdfc17c0b24f19aa840b00565fb",
            "modeling_ouro.py": "cb8c980b016ae8ae35b6c1193633b9b4496e89993e3a0e8c7853324f01904b30",
            "configuration_ouro.py":
                "950443e32929047aa08d02abad2e1888bc1914b3db988d3d675f70787f65dafb",
            "config.json": "44e42ec2ed97f49f20f87985688b6e1be9dd61bbfe9f8a349daac192126e2c55",
        },
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(model_dir: Path, expected_sha256: dict) -> dict:
    found = {}
    for name, expected in expected_sha256.items():
        actual = sha256(model_dir / name)
        if actual != expected:
            raise SystemExit(f"{model_dir / name}: SHA256 {actual} != pinned {expected}")
        found[name] = actual
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="models directory")
    parser.add_argument("--model", choices=sorted(MODELS), default="2.6B")
    parser.add_argument("--source", type=Path,
                        help="existing official model dir (skips the download)")
    parser.add_argument("--chunk-size", type=int, default=2048)
    args = parser.parse_args(argv)
    spec = MODELS[args.model]
    name = spec["repo_id"].split("/")[1]
    args.out.mkdir(parents=True, exist_ok=True)

    if args.source:
        official = args.source.resolve()
    else:
        from huggingface_hub import snapshot_download

        official = args.out / name
        snapshot_download(spec["repo_id"], revision=spec["revision"], local_dir=official)
    hashes = verify(official, spec["sha256"])

    rltt_dir = args.out / f"{name}-RLTT"
    if (rltt_dir / "rltt_model_manifest.json").is_file():
        print(f"{rltt_dir} already built; leaving it as is")
        manifest = json.loads((rltt_dir / "rltt_model_manifest.json").read_text())
    else:
        allowed = {prepare_model.sha256(prepare_model.vendor_modeling()),
                   spec["sha256"]["modeling_ouro.py"]}
        manifest = prepare_model.build(official, rltt_dir, allowed, args.chunk_size, False, False)
    record = {"repo_id": spec["repo_id"], "revision": spec["revision"],
              "official_dir": str(official), "official_sha256": hashes,
              "model_dir": str(rltt_dir), "build": manifest}
    (args.out / f"fetch_model_manifest_{args.model}.json").write_text(
        json.dumps(record, indent=2) + "\n")
    print(json.dumps({"official_dir": str(official), "model_dir": str(rltt_dir)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
