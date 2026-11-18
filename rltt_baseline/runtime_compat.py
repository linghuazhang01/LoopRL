"""CPU-only runtime contract for the legacy LoopRL trainer and migration target.

Version resolution is deliberately separate from importing torch, Ray or vLLM.
The target dependency tuple is not a claim that the legacy worker API is supported.
"""
from importlib import metadata

TARGET_VERSIONS = {
    "verl": "0.9.1", "vllm": "0.24.0", "transformers": "5.9.0", "torch": "2.11.0",
}


def runtime_status(versions=None):
    if versions is None:
        versions = {}
        for name in TARGET_VERSIONS:
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                versions[name] = None
    installed = {name: versions.get(name) for name in TARGET_VERSIONS}
    target_matches = all(
        installed[name] is not None and installed[name].split("+", 1)[0] == wanted
        for name, wanted in TARGET_VERSIONS.items()
    )
    return {
        "installed": installed,
        "migration_target": dict(TARGET_VERSIONS),
        "target_versions_match": target_matches,
        "trainer_integration": "native_verl_0.9.1" if target_matches else "legacy_verl_0.6.1",
        "training_validation": "not_run",
        "migration_complete": False,
    }


def require_legacy_trainer(status):
    version = status["installed"]["verl"]
    if version is None or version.split("+", 1)[0] != "0.6.1":
        raise RuntimeError(
            "LoopRL's current trainer uses verl 0.6.1 actor/FSDP worker and synchronous "
            "rollout APIs. The verl 0.9.1 dependency target requires an engine/trainer "
            "migration; installing matching versions alone does not complete it. "
            f"Installed verl: {version!r}. Refusing before Ray initialization."
        )


def require_decode_last_runtime(version=None):
    """Select a source-audited KV adapter contract, without importing CUDA code."""
    version = version or metadata.version("vllm")
    base = version.split("+", 1)[0]
    if base.startswith(("0.12.", "0.15.")):
        return "legacy"
    if base == "0.24.0":
        return "per_layer_slots"
    raise RuntimeError(
        f"decode_last KV adapter is not validated for vLLM {version}. "
        "Supported contracts are vLLM 0.12.x/0.15.x and 0.24.0. "
        "No automatic fallback to another KV layout is applied."
    )
