"""Download and convert the training and validation data.

- MATH train: ``EleutherAI/hendrycks_math`` (7 subjects, 7,500 problems) -> ``math_train.jsonl``
  with problem / solution / level / type (gold = last ``\\boxed{}`` of the solution);
- MATH-500: ``HuggingFaceH4/MATH-500`` test split -> ``math500.jsonl``.

Records the dataset revisions and file SHA256 in ``data_manifest.json``.

    python -m rltt_baseline.prepare_data --out <data dir>
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from rltt_baseline import prepare_math_train

MATH_REPO = "EleutherAI/hendrycks_math"
MATH500_REPO = "HuggingFaceH4/MATH-500"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    from huggingface_hub import HfApi, snapshot_download

    out = args.out
    raw = out / "raw"
    out.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    manifest = {}

    math_rev = api.dataset_info(MATH_REPO).sha
    math_dir = raw / "hendrycks_math"
    snapshot_download(MATH_REPO, repo_type="dataset", revision=math_rev, local_dir=math_dir,
                      allow_patterns=["*/train-*.parquet"])
    train_path = out / "math_train.jsonl"
    if not train_path.exists():
        prepare_math_train.main(["--source", str(math_dir), "--out", str(train_path)])
    manifest["math_train"] = {"repo": MATH_REPO, "revision": math_rev, "file": str(train_path),
                              "records": sum(1 for _ in open(train_path)),
                              "sha256": sha256(train_path)}

    m500_rev = api.dataset_info(MATH500_REPO).sha
    m500_dir = raw / "MATH-500"
    snapshot_download(MATH500_REPO, repo_type="dataset", revision=m500_rev, local_dir=m500_dir,
                      allow_patterns=["test.jsonl"])
    m500_path = out / "math500.jsonl"
    if not m500_path.exists():
        shutil.copy2(m500_dir / "test.jsonl", m500_path)
    manifest["math500"] = {"repo": MATH500_REPO, "revision": m500_rev, "file": str(m500_path),
                           "records": sum(1 for _ in open(m500_path)), "sha256": sha256(m500_path)}

    (out / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
