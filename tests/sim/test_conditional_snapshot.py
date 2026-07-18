"""Fail-closed tests for the immutable conditional shard/manifest contract (INT-14).

Mirrors ``test_teacher0_partial_snapshot.py``: freeze/verify round-trip a
conditional shard, then fail closed on digest, condition-schema, claim-boundary,
provenance, file-set, trainer-source, and identity drift. Also exercises the
``ConditionalStrategyResult`` shape contract against the real INT-13 fixture
(canonical seam) and the uniform-determinization toy producer/loader.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.runners.run_belief_conditioned_snapshot import (
    CLAIM_BOUNDARY,
    CONDITION_SCHEMA,
    DATASET_SOURCE_FILES,
    TRAINER_SOURCE_FILES,
    SnapshotError,
    _snapshot_identity,
    freeze_conditional_snapshot,
    verify_conditional_snapshot,
)
from manabot.sim.conditional_distill import (
    CONDITION_COUNT_KEY,
    CONDITION_INDEX_KEY,
    CONDITION_KEYS,
    CONDITION_ROLES,
    CONDITION_SCORES_KEY,
    CONDITION_WEIGHT_KEY,
    ConditionalDistillError,
    assert_conforms_to_conditional_strategy_shape,
    build_conditional_dataset,
    generate_uniform_determinization_shard,
    load_conditional_shards,
    viewer_safe_to_condition_rows,
    with_neutral_condition,
)

INT_13_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "data"
    / "int-13-conditional-strategy-fixture-v1.json"
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _source_root(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    for relative in set(DATASET_SOURCE_FILES) | set(TRAINER_SOURCE_FILES):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"source:{relative}\n")
    return root


def _freeze(
    tmp_path: Path, *, shards: int = 2, games_per_shard: int = 8
) -> tuple[Path, Path, dict]:
    root = _source_root(tmp_path)
    dataset_dir = root / "dataset"
    build_conditional_dataset(
        dataset_dir,
        num_games=shards * games_per_shard,
        games_per_shard=games_per_shard,
        sims=2,
        seed=197,
    )
    snapshot_dir = tmp_path / "snapshot"
    manifest = freeze_conditional_snapshot(
        dataset_dir,
        snapshot_dir,
        shard_count=shards,
        trainer_root=root,
    )
    return root, snapshot_dir, manifest


# ---------------------------------------------------------------------------
# Freeze / verify round-trip
# ---------------------------------------------------------------------------


def test_freeze_copies_conditional_prefix_and_verifies(tmp_path: Path) -> None:
    root, snapshot_dir, manifest = _freeze(tmp_path, shards=2)

    verified = verify_conditional_snapshot(
        snapshot_dir,
        expected_identity=manifest["snapshot_identity_sha256"],
        trainer_root=root,
    )

    assert verified["schema_version"] == 2
    assert verified["claim_boundary"] == CLAIM_BOUNDARY
    assert verified["condition_schema"] == CONDITION_SCHEMA
    assert [item["shard_index"] for item in verified["shards"]] == [0, 1]
    assert verified["cutoff"]["shard_count"] == 2
    assert not (snapshot_dir / "shard_00002.npz").exists()


def test_freeze_fails_when_declared_prefix_not_durable(tmp_path: Path) -> None:
    root = _source_root(tmp_path)
    dataset_dir = root / "dataset"
    build_conditional_dataset(
        dataset_dir, num_games=8, games_per_shard=8, sims=2, seed=197
    )

    with pytest.raises(SnapshotError, match="not yet durable"):
        freeze_conditional_snapshot(
            dataset_dir,
            tmp_path / "snapshot",
            shard_count=2,
            trainer_root=root,
        )


def test_freeze_fails_on_source_digest_mismatch(tmp_path: Path) -> None:
    root = _source_root(tmp_path)
    dataset_dir = root / "dataset"
    build_conditional_dataset(
        dataset_dir, num_games=16, games_per_shard=8, sims=2, seed=197
    )
    (dataset_dir / "shard_00000.npz").write_bytes(b"mutated")

    with pytest.raises(SnapshotError, match="digest differs"):
        freeze_conditional_snapshot(
            dataset_dir,
            tmp_path / "snapshot",
            shard_count=2,
            trainer_root=root,
        )


def test_consumer_fails_on_snapshot_byte_mutation(tmp_path: Path) -> None:
    root, snapshot_dir, manifest = _freeze(tmp_path)
    (snapshot_dir / "shard_00000.npz").write_bytes(b"mutated")

    with pytest.raises(SnapshotError, match="digest mismatch"):
        verify_conditional_snapshot(
            snapshot_dir,
            expected_identity=manifest["snapshot_identity_sha256"],
            trainer_root=root,
        )


@pytest.mark.parametrize("mode", ["missing", "extra"])
def test_consumer_fails_on_snapshot_file_set_change(tmp_path: Path, mode: str) -> None:
    root, snapshot_dir, manifest = _freeze(tmp_path)
    if mode == "missing":
        (snapshot_dir / "shard_00001.json").unlink()
        match = "required file is missing"
    else:
        (snapshot_dir / "shard_99999.npz").write_bytes(b"extra")
        match = "file set mismatch"

    with pytest.raises(SnapshotError, match=match):
        verify_conditional_snapshot(
            snapshot_dir,
            expected_identity=manifest["snapshot_identity_sha256"],
            trainer_root=root,
        )


def test_consumer_fails_when_trainer_source_changes(tmp_path: Path) -> None:
    root, snapshot_dir, manifest = _freeze(tmp_path)
    (root / TRAINER_SOURCE_FILES[0]).write_text("changed\n")

    with pytest.raises(SnapshotError, match="trainer source identity changed"):
        verify_conditional_snapshot(
            snapshot_dir,
            expected_identity=manifest["snapshot_identity_sha256"],
            trainer_root=root,
        )


def test_consumer_requires_exact_declared_snapshot_identity(tmp_path: Path) -> None:
    root, snapshot_dir, _ = _freeze(tmp_path)

    with pytest.raises(SnapshotError, match="declared consumer input"):
        verify_conditional_snapshot(
            snapshot_dir,
            expected_identity="0" * 64,
            trainer_root=root,
        )


def test_consumer_fails_on_condition_schema_drift(tmp_path: Path) -> None:
    root, snapshot_dir, manifest = _freeze(tmp_path)
    snapshot_path = snapshot_dir / "snapshot.json"
    snapshot = json.loads(snapshot_path.read_text())
    snapshot["condition_schema"] = {
        **CONDITION_SCHEMA,
        "condition_source": "tampered",
    }
    snapshot["snapshot_identity_sha256"] = _snapshot_identity(snapshot)
    _write_json(snapshot_path, snapshot)

    with pytest.raises(SnapshotError, match="condition schema changed"):
        verify_conditional_snapshot(
            snapshot_dir,
            expected_identity=snapshot["snapshot_identity_sha256"],
            trainer_root=root,
        )


def test_consumer_fails_on_claim_boundary_drift(tmp_path: Path) -> None:
    root, snapshot_dir, manifest = _freeze(tmp_path)
    snapshot_path = snapshot_dir / "snapshot.json"
    snapshot = json.loads(snapshot_path.read_text())
    snapshot["claim_boundary"] = {**CLAIM_BOUNDARY, "strength_claim": True}
    snapshot["snapshot_identity_sha256"] = _snapshot_identity(snapshot)
    _write_json(snapshot_path, snapshot)

    with pytest.raises(SnapshotError, match="claim boundary changed"):
        verify_conditional_snapshot(
            snapshot_dir,
            expected_identity=snapshot["snapshot_identity_sha256"],
            trainer_root=root,
        )


def test_consumer_fails_on_shard_condition_provenance_drift(tmp_path: Path) -> None:
    root, snapshot_dir, manifest = _freeze(tmp_path, shards=1)
    snapshot_path = snapshot_dir / "snapshot.json"
    sidecar_path = snapshot_dir / "shard_00000.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["provenance"]["condition_label_format"] = "tampered"
    _write_json(sidecar_path, sidecar)
    # Update the recorded sidecar digest + identity so verify reaches the
    # provenance check rather than failing on digest mismatch first.
    from experiments.runners.run_belief_conditioned_snapshot import _file_sha256

    snapshot = json.loads(snapshot_path.read_text())
    snapshot["shards"][0]["json_sha256"] = _file_sha256(sidecar_path)
    snapshot["snapshot_identity_sha256"] = _snapshot_identity(snapshot)
    _write_json(snapshot_path, snapshot)

    with pytest.raises(SnapshotError, match="condition_label_format mismatch"):
        verify_conditional_snapshot(
            snapshot_dir,
            expected_identity=snapshot["snapshot_identity_sha256"],
            trainer_root=root,
        )


# ---------------------------------------------------------------------------
# ConditionalStrategyResult shape contract (canonical INT-13 seam)
# ---------------------------------------------------------------------------


def test_int13_fixture_conforms_to_pinned_shape() -> None:
    fixture = json.loads(INT_13_FIXTURE.read_text())
    assert_conforms_to_conditional_strategy_shape(fixture)


def test_int13_fixture_adapter_ingests_with_no_contract_change() -> None:
    fixture = json.loads(INT_13_FIXTURE.read_text())
    rows = viewer_safe_to_condition_rows(fixture)
    assert len(rows) == len(CONDITION_ROLES)
    weights = [r.condition_weight for r in rows]
    assert abs(sum(weights) - 1.0) < 1e-6
    for row in rows:
        assert 0 <= row.condition_index < len(CONDITION_ROLES)
        assert row.condition_scores.shape == (fixture["action_count"],)


def test_shape_rejects_wrong_condition_count() -> None:
    fixture = json.loads(INT_13_FIXTURE.read_text())
    fixture["viewer_safe"]["conditions"].append(fixture["viewer_safe"]["conditions"][0])
    with pytest.raises(ConditionalDistillError, match="expected 5 conditions"):
        assert_conforms_to_conditional_strategy_shape(fixture)


def test_shape_rejects_wrong_planner() -> None:
    fixture = json.loads(INT_13_FIXTURE.read_text())
    fixture["planner"] = "something_else"
    with pytest.raises(ConditionalDistillError, match="planner"):
        assert_conforms_to_conditional_strategy_shape(fixture)


# ---------------------------------------------------------------------------
# Toy producer + loader
# ---------------------------------------------------------------------------


def test_generate_uniform_determinization_shard_conforms(tmp_path: Path) -> None:
    out_path = tmp_path / "shard_00000.npz"
    summary = generate_uniform_determinization_shard(
        num_games=4, sims=2, seed=197, out_path=out_path
    )
    assert summary["condition_count"] == len(CONDITION_ROLES)
    with np.load(out_path) as data:
        for key in CONDITION_KEYS:
            assert key in data.files, f"shard missing {key}"
        count = data[CONDITION_INDEX_KEY].shape[1]
        assert count == len(CONDITION_ROLES)
        weights = data[CONDITION_WEIGHT_KEY]
        assert np.allclose(weights[0].sum(), 1.0)
        scores = data[CONDITION_SCORES_KEY]
        # uniform determinization: every condition shares the flat-MC scores
        base_scores = data["scores"]
        for k in range(count):
            assert np.allclose(scores[:, k, :], base_scores)


def test_load_conditional_shards_expands_per_condition(tmp_path: Path) -> None:
    out_path = tmp_path / "shard_00000.npz"
    generate_uniform_determinization_shard(
        num_games=4, sims=2, seed=197, out_path=out_path
    )
    dataset = load_conditional_shards([str(out_path)])
    with np.load(out_path) as data:
        expected_rows = int(data[CONDITION_COUNT_KEY].sum())
    rows = len(dataset["action"])
    assert rows == expected_rows
    assert set(("condition_index", "condition_weight")).issubset(dataset.keys())
    # condition_index cycles 0..K-1 per decision
    assert dataset["condition_index"][: len(CONDITION_ROLES)].tolist() == list(
        range(len(CONDITION_ROLES))
    )
    # weights are uniform 1/K
    assert np.allclose(
        dataset["condition_weight"][: len(CONDITION_ROLES)], 1.0 / len(CONDITION_ROLES)
    )


def test_load_conditional_shards_fails_on_missing_condition_key(
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "shard_00000.npz"
    generate_uniform_determinization_shard(
        num_games=4, sims=2, seed=197, out_path=out_path
    )
    with np.load(out_path) as data:
        arrays = {key: data[key] for key in data.files if key != CONDITION_INDEX_KEY}
    bad_path = tmp_path / "bad.npz"
    np.savez_compressed(bad_path, **arrays)
    with pytest.raises(ConditionalDistillError, match="missing required key"):
        load_conditional_shards([str(bad_path)])


def test_load_conditional_shards_fails_on_digest_mismatch(tmp_path: Path) -> None:
    out_path = tmp_path / "shard_00000.npz"
    generate_uniform_determinization_shard(
        num_games=4, sims=2, seed=197, out_path=out_path
    )
    with np.load(out_path) as data:
        arrays = {key: data[key] for key in data.files}
    import json as _json

    provenance = _json.loads(str(arrays["provenance"]))
    provenance["condition_schema_digest"] = "0" * 64
    arrays["provenance"] = np.array(_json.dumps(provenance))
    bad_path = tmp_path / "bad.npz"
    np.savez_compressed(bad_path, **arrays)
    with pytest.raises(ConditionalDistillError, match="condition_schema_digest"):
        load_conditional_shards([str(bad_path)])


def test_with_neutral_condition_masks_condition_input(tmp_path: Path) -> None:
    out_path = tmp_path / "shard_00000.npz"
    generate_uniform_determinization_shard(
        num_games=4, sims=2, seed=197, out_path=out_path
    )
    dataset = load_conditional_shards([str(out_path)])
    neutral = with_neutral_condition(dataset)
    assert np.all(neutral["condition_index"] == 0)
    assert np.allclose(neutral["condition_weight"], 1.0)
    # targets and observations are unchanged
    assert np.array_equal(neutral["scores"], dataset["scores"])
    assert np.array_equal(neutral["action"], dataset["action"])


def test_build_conditional_dataset_freezes_and_verifies(tmp_path: Path) -> None:
    root = _source_root(tmp_path)
    dataset_dir = root / "dataset"
    manifest = build_conditional_dataset(
        dataset_dir, num_games=16, games_per_shard=8, sims=2, seed=197
    )
    assert manifest["schema_version"] == 2
    assert len(manifest["shards"]) == 2
    snapshot_dir = tmp_path / "snapshot"
    snap = freeze_conditional_snapshot(
        dataset_dir, snapshot_dir, shard_count=2, trainer_root=root
    )
    verified = verify_conditional_snapshot(
        snapshot_dir,
        expected_identity=snap["snapshot_identity_sha256"],
        trainer_root=root,
    )
    assert verified["cutoff"]["shard_count"] == 2
    npz_paths = sorted(snapshot_dir.glob("shard_*.npz"))
    dataset = load_conditional_shards([str(p) for p in npz_paths])
    assert "condition_index" in dataset and "condition_weight" in dataset
