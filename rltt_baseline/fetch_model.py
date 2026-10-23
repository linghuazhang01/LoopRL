"""Get Ouro-2.6B-Thinking at the pinned revision and build the training model directory.

1. Download ``ByteDance/Ouro-2.6B-Thinking`` at revision f1edd81e (or reuse ``--source``) and
   check the SHA256 of ``model.safetensors`` and ``modeling_ouro.py``.
2. Build ``<out>/Ouro-2.6B-Thinking-RLTT`` with ``prepare_model`` (weights symlinked, official
   modeling + transformers 4.57 compat edit, per-loop RLTT outputs). Both the RLTT and the
   GRPO trainer use this directory.

    python -m rltt_baseline.fetch_model --out <models dir> [--source <existing official dir>]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from rltt_baseline import prepare_model

REPO_ID = "ByteDance/Ouro-2.6B-Thinking"
REVISION = "f1edd81e7ac41355db670500ceaf204e0f73af68"
EXPECTED_SHA256 = {
    "model.safetensors": "c506a79247dc51fc0400d789365c3d43932f718abce9810f3606ace47d0a3080",
    "modeling_ouro.py": "263c097a2ed6870ce549a65ef6d6bb9e8e167423f87d2c494e0988b9a4f63f4b",
    "configuration_ouro.py": "ba60253335cecc5bd9cb3e5f8cfbd214aa7e865fc97a98208976fb2517811de6",
    "config.json": "ad4b51dffae60bebeeac5e1d5055e78d01fc01b1a29cd89cd2e50edb848f645a",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(model_dir: Path) -> dict:
    found = {}
    for name, expected in EXPECTED_SHA256.items():
        actual = sha256(model_dir / name)
        if actual != expected:
            raise SystemExit(f"{model_dir / name}: SHA256 {actual} != pinned {expected}")
        found[name] = actual
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="models directory")
    parser.add_argument("--source", type=Path,
                        help="existing official model dir (skips the download)")
    parser.add_argument("--chunk-size", type=int, default=2048)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.source:
        official = args.source.resolve()
    else:
        from huggingface_hub import snapshot_download

        official = args.out / "Ouro-2.6B-Thinking"
        snapshot_download(REPO_ID, revision=REVISION, local_dir=official)
    hashes = verify(official)

    rltt_dir = args.out / "Ouro-2.6B-Thinking-RLTT"
    if (rltt_dir / "rltt_model_manifest.json").is_file():
        print(f"{rltt_dir} already built; leaving it as is")
        manifest = json.loads((rltt_dir / "rltt_model_manifest.json").read_text())
    else:
        allowed = {prepare_model.sha256(prepare_model.vendor_modeling())}
        manifest = prepare_model.build(official, rltt_dir, allowed, args.chunk_size, False, False)
    record = {"repo_id": REPO_ID, "revision": REVISION, "official_dir": str(official),
              "official_sha256": hashes, "model_dir": str(rltt_dir), "build": manifest}
    (args.out / "fetch_model_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"official_dir": str(official), "model_dir": str(rltt_dir)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
