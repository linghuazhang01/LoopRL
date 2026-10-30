#!/usr/bin/env python3
"""Download the 9 pinned files of Ouro-1.4B-Thinking (revision 3aaa2224) and check them.

    python fetch_model.py [--dir model_1p4b_thinking_3aaa2224]

Uses huggingface_hub (installed with vLLM); set HF_ENDPOINT for a mirror. Only the files of the
H200 / HPC copies are fetched (no generation_config.json, so vLLM samples with its own defaults
as on H200). Ends with the full check of sample.py, which writes provenance.json. A directory
copied from another machine can be checked with `python sample.py --check-model DIR` instead.
"""

import argparse
import json
from pathlib import Path

from sample import HERE, MODEL, MODEL_FILES, check_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=HERE / MODEL["dir"])
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL["model_id"], revision=MODEL["revision"], local_dir=str(args.dir),
                      allow_patterns=sorted(MODEL_FILES))
    print(json.dumps(check_model(args.dir, full=True), indent=1))
    print(f"MODEL_OK {args.dir}")


if __name__ == "__main__":
    main()
