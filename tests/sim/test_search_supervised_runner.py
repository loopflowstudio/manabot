"""Evidence-integrity tests for the search-supervised experiment runner."""

from argparse import Namespace
from pathlib import Path

import pytest

from experiments.runners.run_search_supervised import _validate_dataset_manifest


def _args() -> Namespace:
    return Namespace(games=12, workers=2, seed=197, sims=8, games_per_shard=6)


def _manifest() -> dict:
    teacher = {"kind": "determinized_puct", "sims": 8, "worlds": 2, "c_puct": 1.5}
    return {
        "games": 12,
        "workers": 2,
        "seed": 197,
        "sims": 8,
        "games_per_shard": 6,
        "policy_target_kind": "visit_distribution",
        "value_target_kind": "root_value",
        "provenance": {"teacher_spec": teacher},
        "shards": [
            {"out_path": "/old/worktree/dataset/shard_00.npz"},
            {"out_path": "/old/worktree/dataset/shard_01.npz"},
        ],
    }


def _validate(manifest: dict, paths: list[Path] | None = None) -> None:
    _validate_dataset_manifest(
        manifest,
        args=_args(),
        teacher_spec={
            "kind": "determinized_puct",
            "sims": 8,
            "worlds": 2,
            "c_puct": 1.5,
        },
        policy_target_kind="visit_distribution",
        value_target_kind="root_value",
        shard_paths=paths or [Path("shard_00.npz"), Path("shard_01.npz")],
    )


def test_matching_dataset_manifest_can_resume_after_worktree_move() -> None:
    _validate(_manifest())


def test_legacy_flat_teacher_manifest_has_unambiguous_targets() -> None:
    manifest = _manifest()
    manifest.pop("policy_target_kind")
    manifest.pop("value_target_kind")
    flat_teacher = {"kind": "search", "sims": 8}
    manifest["provenance"]["teacher_spec"] = flat_teacher
    _validate_dataset_manifest(
        manifest,
        args=_args(),
        teacher_spec=flat_teacher,
        policy_target_kind="score_softmax",
        value_target_kind="terminal_outcome",
        shard_paths=[Path("shard_00.npz"), Path("shard_01.npz")],
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("games", 11),
        ("workers", 1),
        ("seed", 198),
        ("sims", 4),
        ("games_per_shard", 4),
        ("policy_target_kind", "score_softmax"),
        ("value_target_kind", "terminal_outcome"),
    ],
)
def test_resume_rejects_mismatched_generation_contract(
    field: str, replacement: object
) -> None:
    manifest = _manifest()
    manifest[field] = replacement
    with pytest.raises(SystemExit, match=field):
        _validate(manifest)


def test_resume_rejects_different_teacher_or_stale_shard() -> None:
    manifest = _manifest()
    manifest["provenance"]["teacher_spec"]["worlds"] = 1
    with pytest.raises(SystemExit, match="teacher_spec"):
        _validate(manifest)

    with pytest.raises(SystemExit, match="shards"):
        _validate(_manifest(), [Path("shard_00.npz"), Path("shard_02.npz")])
