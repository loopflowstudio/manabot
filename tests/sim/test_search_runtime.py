"""Retained checkpoint registration and model ABI identity."""

from __future__ import annotations

import json

import pytest

from experiments.runners.run_int7_value_target_comparison import manifest_digest
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.sim.search_runtime import (
    INT7_VALUE_TARGET_MANIFEST_SHA256,
    INT7_VALUE_TARGET_RESULT,
    RetainedCheckpointMismatchError,
    RetainedCheckpointUnavailableError,
    model_action_abi_sha256,
    model_observation_abi_sha256,
    retained_int7_evidence_snapshot,
    retained_int7_manifest_digest,
    retained_int7_policy_only_checkpoint,
)

EXPECTED = {
    197: "1673a237ef2460d0e699667987c29fe6b42c28711bdb2041989f37692edbd1e6",
    198: "5b3dab6517534047d899704d44c839276c5cf74c7c56b6e29ce0a52180bf5223",
    199: "72cad2028861a7dd422f3e9ae18a98a09e9e911b6e3ca908f46b95a4c35c7fd3",
}


@pytest.mark.parametrize(("seed", "checkpoint_sha256"), EXPECTED.items())
def test_retained_registry_verifies_all_policy_only_bytes(
    seed: int,
    checkpoint_sha256: str,
) -> None:
    registration = retained_int7_policy_only_checkpoint(seed)

    assert registration.training_seed == seed
    assert registration.checkpoint_sha256 == checkpoint_sha256
    assert registration.checkpoint_bytes == 428_629
    assert registration.manifest_sha256 == INT7_VALUE_TARGET_MANIFEST_SHA256
    assert registration.checkpoint_path.name == f"visit_policy_only-seed-{seed}.pt"
    assert registration.world_id == "w2"
    assert registration.value_mode == "neutral"
    assert registration.simulations == 32
    assert registration.sampled_worlds == 4


def test_retained_registry_rejects_every_unregistered_seed() -> None:
    with pytest.raises(RetainedCheckpointMismatchError, match="not in"):
        retained_int7_policy_only_checkpoint(200)


def test_retained_registry_missing_manifest_is_typed_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    import manabot.sim.search_runtime as runtime

    monkeypatch.setattr(runtime, "INT7_VALUE_TARGET_MANIFEST", tmp_path / "missing")
    with pytest.raises(RetainedCheckpointUnavailableError, match="unavailable"):
        retained_int7_policy_only_checkpoint(197)


def test_retained_registry_recomputes_manifest_identity_before_checkpoint_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    import manabot.sim.search_runtime as runtime

    manifest = json.loads(runtime.INT7_VALUE_TARGET_MANIFEST.read_text())
    assert retained_int7_manifest_digest(manifest) == manifest_digest(manifest)
    assert retained_int7_manifest_digest(manifest) == INT7_VALUE_TARGET_MANIFEST_SHA256

    manifest["evidence_class"] = "tampered-but-self-consistent"
    manifest["manifest_sha256"] = retained_int7_manifest_digest(manifest)
    tampered = tmp_path / "manifest.json"
    tampered.write_text(json.dumps(manifest))
    monkeypatch.setattr(runtime, "INT7_VALUE_TARGET_MANIFEST", tampered)

    def checkpoint_bytes_must_not_be_read(_path) -> str:
        raise AssertionError("checkpoint bytes read before manifest authentication")

    monkeypatch.setattr(runtime, "_sha256_file", checkpoint_bytes_must_not_be_read)
    with pytest.raises(RetainedCheckpointMismatchError, match="identity drifted"):
        retained_int7_policy_only_checkpoint(197)


def test_retained_evidence_snapshot_closes_the_full_root() -> None:
    before = retained_int7_evidence_snapshot((197,))
    after = retained_int7_evidence_snapshot((197,))

    assert before == after
    assert before.receipt_sha256 == after.receipt_sha256
    assert before.manifest_digest == INT7_VALUE_TARGET_MANIFEST_SHA256
    assert before.manifest_file.relative_path == "manifest.json"
    assert before.manifest_file.bytes > 0
    assert before.manifest_file.sha256
    assert before.retention_tree_files > 1
    assert before.retention_tree_bytes > before.manifest_file.bytes
    assert before.retention_tree_content_sha256
    assert len(before.files) == before.retention_tree_files
    assert [file.relative_path for file in before.files] == [
        path.relative_to(INT7_VALUE_TARGET_RESULT).as_posix()
        for path in sorted(INT7_VALUE_TARGET_RESULT.rglob("*"))
        if path.is_file()
    ]
    assert [row.training_seed for row in before.selected_checkpoints] == [197]
    selected = before.selected_checkpoints[0]
    assert selected.file.sha256 == EXPECTED[197]
    assert selected.world_id == "w2"
    assert selected.observation_abi_sha256
    assert selected.action_abi_sha256
    assert selected.parameter_count == 102_722
    assert selected.value_mode == "neutral"
    assert selected.branch_driver_id == "full_clone/current_game_v1"
    assert selected.simulations == 32
    assert selected.sampled_worlds == 4
    assert selected.c_puct == 1.5
    assert selected.max_steps == 2000


def test_seed_197_checkpoint_model_abis_match_the_registry() -> None:
    registration = retained_int7_policy_only_checkpoint(197)
    agent, observation_space = load_checkpoint_agent(str(registration.checkpoint_path))

    assert sum(parameter.numel() for parameter in agent.parameters()) == 102_722
    assert (
        model_observation_abi_sha256(observation_space)
        == registration.observation_abi_sha256
    )
    assert model_action_abi_sha256(observation_space) == registration.action_abi_sha256
