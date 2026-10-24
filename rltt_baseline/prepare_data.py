"""Download the training and validation data and record revisions and SHA256.

Datasets (``--datasets``, comma separated; default ``deepmath_level6,math500``):

- ``deepmath_level6`` (default training data, as in SAPD): 57,046 DeepMath-103K problems of
  difficulty >= 6 (the math prompts released with G-OPD). Source file
  ``DeepMath-103K/train_filtered_level6.parquet`` of ``icemoon28/MOPD-Training-Data`` at a pinned
  commit, SHA256-checked, saved as ``deepmath_level6.parquet``. It is already in verl format
  with its own prompt (problem + "Please reason step by step, and put your final answer within
  \\boxed{}."), which the launcher uses unchanged (``rltt_baseline.verl_data``).
- ``math500``: ``HuggingFaceH4/MATH-500`` test split -> ``math500.jsonl`` (optional validation).
- ``math_train``: ``EleutherAI/hendrycks_math`` (7 subjects, 7,500 problems) ->
  ``math_train.jsonl`` with problem / solution / level / type (gold = last ``\\boxed{}`` of the
  solution); the RLTT paper's training set, trained with the RLTT prompt.

``data_manifest.json`` records, per dataset, the repository, revision, file, record count and
SHA256 (entries of datasets not prepared in this call are kept).

    python -m rltt_baseline.prepare_data --out <data dir> [--datasets deepmath_level6,math500]
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
DEEPMATH_REPO = "icemoon28/MOPD-Training-Data"
DEEPMATH_REVISION = "96e88d579cb1ee68c9183c52e5d133f58cf370bf"
DEEPMATH_FILE = "DeepMath-103K/train_filtered_level6.parquet"
DEEPMATH_SHA256 = "de3350fdd00bc0410550098ea65179e2be873da99e4075f80de575fc17670597"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


DATASETS = ("deepmath_level6", "math500", "math_train")
DEFAULT_DATASETS = "deepmath_level6,math500"


def _deepmath_level6(out: Path, raw: Path) -> dict:
    from huggingface_hub import hf_hub_download

    from rltt_baseline.verl_data import load_rows

    path = out / "deepmath_level6.parquet"
    if not path.exists():
        cached = Path(hf_hub_download(DEEPMATH_REPO, DEEPMATH_FILE, repo_type="dataset",
                                      revision=DEEPMATH_REVISION, local_dir=raw / "MOPD"))
        if sha256(cached) != DEEPMATH_SHA256:
            raise SystemExit(f"{cached}: SHA256 differs from the SAPD training file")
        shutil.copy2(cached, path)
    if sha256(path) != DEEPMATH_SHA256:
        raise SystemExit(f"{path}: SHA256 differs from the SAPD training file")
    records = len(load_rows(path))  # the format the launcher expects
    return {"repo": DEEPMATH_REPO, "revision": DEEPMATH_REVISION, "path_in_repo": DEEPMATH_FILE,
            "file": str(path), "records": records, "sha256": DEEPMATH_SHA256}


def _math500(out: Path, raw: Path) -> dict:
    from huggingface_hub import HfApi, snapshot_download

    rev = HfApi().dataset_info(MATH500_REPO).sha
    src = raw / "MATH-500"
    snapshot_download(MATH500_REPO, repo_type="dataset", revision=rev, local_dir=src,
                      allow_patterns=["test.jsonl"])
    path = out / "math500.jsonl"
    if not path.exists():
        shutil.copy2(src / "test.jsonl", path)
    return {"repo": MATH500_REPO, "revision": rev, "file": str(path),
            "records": sum(1 for _ in open(path)), "sha256": sha256(path)}


def _math_train(out: Path, raw: Path) -> dict:
    from huggingface_hub import HfApi, snapshot_download

    rev = HfApi().dataset_info(MATH_REPO).sha
    src = raw / "hendrycks_math"
    snapshot_download(MATH_REPO, repo_type="dataset", revision=rev, local_dir=src,
                      allow_patterns=["*/train-*.parquet"])
    path = out / "math_train.jsonl"
    if not path.exists():
        prepare_math_train.main(["--source", str(src), "--out", str(path)])
    return {"repo": MATH_REPO, "revision": rev, "file": str(path),
            "records": sum(1 for _ in open(path)), "sha256": sha256(path)}


PREPARE = {"deepmath_level6": _deepmath_level6, "math500": _math500, "math_train": _math_train}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--datasets", default=DEFAULT_DATASETS,
                        help=f"comma-separated subset of {','.join(DATASETS)}")
    args = parser.parse_args(argv)
    names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    unknown = sorted(set(names) - set(DATASETS))
    if unknown or not names:
        parser.error(f"--datasets: unknown {unknown}; choose from {','.join(DATASETS)}")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name in names:
        manifest[name] = PREPARE[name](out, out / "raw")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({n: manifest[n] for n in names}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
