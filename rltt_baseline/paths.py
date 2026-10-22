"""Locate third-party sources in either checkout layout.

- RecurrentTransformer project: ``code/src/rltt_baseline`` with ``code/vendor/{rltt,ouro}_source``
  and ``code/configs``;
- LoopRL repository: ``rltt_baseline`` at the root with ``third_party/{RLTT,ouro_source}``
  and ``configs``.

``RLTT_SOURCE_DIR`` / ``OURO_SOURCE_DIR`` override the lookup.
"""

import os
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
SRC_DIR = PKG_DIR.parent  # directory that must be on PYTHONPATH


def _first_existing(env_var: str, candidates) -> Path:
    if os.environ.get(env_var):
        return Path(os.environ[env_var]).resolve()
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"none of {[str(c) for c in candidates]} exists; set {env_var}")


def rltt_source_dir() -> Path:
    return _first_existing("RLTT_SOURCE_DIR", [PKG_DIR.parents[1] / "vendor" / "rltt_source",
                                               SRC_DIR / "third_party" / "RLTT"])


def ouro_source_dir() -> Path:
    return _first_existing("OURO_SOURCE_DIR", [PKG_DIR.parents[1] / "vendor" / "ouro_source",
                                               SRC_DIR / "third_party" / "ouro_source"])


def config_file(name: str) -> Path:
    for candidate in (PKG_DIR.parents[1] / "configs" / name, SRC_DIR / "configs" / name):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(name)
