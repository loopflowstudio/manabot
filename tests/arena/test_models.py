import json
from pathlib import Path

from pydantic import ValidationError
import pytest

from manabot.arena.models import ArenaContract, PlayerRegistration, canonical_sha256
from manabot.arena.players import build_player

ROOT = Path(__file__).resolve().parents[2]


def test_checked_contract_has_exact_code_only_base() -> None:
    contract = ArenaContract.model_validate(
        json.loads(
            (ROOT / "experiments/contracts/int-6-skill-arena-v1.json").read_text()
        )
    )
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
