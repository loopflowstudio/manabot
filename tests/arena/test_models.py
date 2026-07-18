import json
from pathlib import Path

from pydantic import ValidationError
import pytest

from manabot.arena.models import (
    ArenaContract,
    PlayerRegistration,
    canonical_sha256,
    file_sha256,
)
from manabot.arena.players import build_player

ROOT = Path(__file__).resolve().parents[2]


def test_checked_contract_has_exact_code_only_base() -> None:
    path = ROOT / "experiments/contracts/int-6-skill-arena-v1.json"
    assert file_sha256(path) == (
        "fc9cb76c0d80ad64951455ac6fede94b1355f383dde2f2964d0e578f62671a71"
    )
    contract = ArenaContract.model_validate(json.loads(path.read_text()))
    assert [player.player_id for player in contract.anchors] == [
        "random-v1",
        "scripted-greedy-v1",
        "flat-mc-4-v1",
        "flat-mc-16-v1",
        "flat-mc-64-v1",
    ]
    assert all(player.runner_kind == "code" for player in contract.anchors)
    assert len(contract.schedules["production"].deal_seeds) == 24
    assert contract.promotion.elo_margin == 25.0
    assert contract.key.anchor_cohort_sha256 == (
        "3a29f51662e37a02c493b1bffaf6e618a1f5b14322e701499160cf09ba659362"
    )
    assert contract.key.rating_prior_sha256 == (
        "ff4920af7c3c325eabcfe1e715c0abf7e1569393f30d572da80012c15dabc904"
    )
    assert {
        player.player_id: player.identity_sha256 for player in contract.anchors
    } == {
        "random-v1": "d1bbcda218602bcab299f01dee0b64fd795cab0289a77d4a53acee834f59febb",
        "scripted-greedy-v1": "7fad8c4167eb969bb6aa39393594893bab5ca3f56c1165c257029db8e20b4bc3",
        "flat-mc-4-v1": "7afbd87ae0134a2cd448944b35c69cb8c5c798b35552771aa921a7a3bf2604ee",
        "flat-mc-16-v1": "be11c9e146021a0b3c6121b3c9c63436bd51ac056ed8280b44e80bb818582e2e",
        "flat-mc-64-v1": "a885d0f6a81979d1adea9c9d7822ac93837b22d7c67d040fadc20a3803b6ec76",
    }


def test_dpuct_registration_requires_every_effective_parameter() -> None:
    candidate = json.loads(
        (ROOT / "experiments/candidates/int-6-dpuct-32-w4-v1.json").read_text()
    )
    candidate["player_spec"].pop("c_puct")
    with pytest.raises(ValidationError, match="fully explicit"):
        PlayerRegistration.model_validate(candidate)


def test_dpuct_registration_and_factory_echo_frozen_semantics() -> None:
    candidate = PlayerRegistration.model_validate(
        json.loads(
            (ROOT / "experiments/candidates/int-6-dpuct-32-w4-v1.json").read_text()
        )
    )
    player, observation_space = build_player(candidate, seed=7)
    assert observation_space is None
    assert player.simulations == 32
    assert player.worlds == 4
    assert player.c_puct == 1.5
    assert player.max_steps == 2000
    assert player.branch_driver_id == "full_clone/current_game_v1"
    assert not player.branch_audit


def test_base_contract_rejects_cohort_expansion_even_with_rehashed_identity() -> None:
    raw = json.loads(
        (ROOT / "experiments/contracts/int-6-skill-arena-v1.json").read_text()
    )
    learned = dict(raw["anchors"][0])
    learned.update(
        {
            "player_id": "learned-fixture-v1",
            "display_name": "learned fixture",
            "compute_class_id": "policy-cpu-batch1-v1",
        }
    )
    original_cohort_sha256 = raw["key"]["anchor_cohort_sha256"]
    raw["anchors"].append(learned)
    raw["key"]["anchor_cohort_sha256"] = canonical_sha256(raw["anchors"])
    assert raw["key"]["anchor_cohort_sha256"] != original_cohort_sha256
    with pytest.raises(ValidationError, match="exactly the five"):
        ArenaContract.model_validate(raw)


def test_checkpoint_candidate_missing_bytes_fails_closed() -> None:
    registration = PlayerRegistration(
        player_id="fixture-checkpoint-v1",
        display_name="fixture",
        role="challenger",
        runner_kind="checkpoint",
        player_spec={
            "kind": "checkpoint",
            "deterministic": True,
            "device": "cpu",
            "batch_size": 1,
        },
        compute_class_id="policy-cpu-batch1-v1",
        information_boundary="acting-viewer-history-only-v1",
        world="w2",
        content_suite="w2-interactive-mirror-v1",
        observation_abi_sha256="a" * 64,
        action_abi_sha256="c" * 64,
        matchup_sha256="d" * 64,
        checkpoint_sha256="b" * 64,
        checkpoint_bytes=1,
        parameter_count=1,
        training_seed=1,
        artifact_id="fixture-only",
        evidence_class="fixture",
        player_seed_derivation_id="arena-pair-deal-player-v1",
    )
    with pytest.raises(FileNotFoundError, match="unavailable"):
        build_player(registration, seed=1)
