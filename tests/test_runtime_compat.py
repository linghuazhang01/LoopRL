import pytest

from rltt_baseline.runtime_compat import (
    TARGET_VERSIONS, require_decode_last_runtime, require_legacy_trainer, runtime_status,
)


def test_target_tuple_does_not_claim_training_compatibility():
    versions = dict(TARGET_VERSIONS, torch="2.11.0+cu130")
    status = runtime_status(versions)
    assert status["target_versions_match"]
    assert not status["migration_complete"]
    with pytest.raises(RuntimeError, match="before Ray"):
        require_legacy_trainer(status)


def test_legacy_trainer_remains_available():
    require_legacy_trainer(runtime_status({"verl": "0.6.1"}))


def test_missing_verl_is_not_accepted():
    with pytest.raises(RuntimeError):
        require_legacy_trainer(runtime_status({}))


def test_decode_last_selects_audited_cache_contract():
    assert require_decode_last_runtime("0.24.0+cu130") == "per_layer_slots"
    assert require_decode_last_runtime("0.12.0") == "legacy"
    assert require_decode_last_runtime("0.15.0") == "legacy"
    with pytest.raises(RuntimeError, match="not validated"):
        require_decode_last_runtime("0.25.0")


def test_loop_cd_accepts_checked_vllm_versions():
    from rltt_baseline.runtime_compat import require_loop_cd_runtime

    for version in ("0.12.0", "0.14.0", "0.15.1", "0.24.0+cu130"):
        assert require_loop_cd_runtime(version) == version.split("+")[0]
    for version in ("0.24.1", "0.25.0", "0.11.2"):
        with pytest.raises(RuntimeError, match="not validated"):
            require_loop_cd_runtime(version)
