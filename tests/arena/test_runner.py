from argparse import Namespace
import json
from pathlib import Path

import pytest
import torch

from experiments.runners.run_skill_arena import (
    ArenaError,
    diagnostic_decision_payload,
    diagnostic_profile_variants,
    preflight_candidate,
    preflight_int8_dependencies,
)
from manabot.arena.competency import run_competencies
from manabot.arena.match import play_cell
from manabot.arena.models import (
    ArenaContract,
    PlayerRegistration,
    ProfileRoots,
    file_sha256,
)
from manabot.arena.profile import profile_players, verify_profile
from manabot.arena.replay import read_trace
from manabot.env import ObservationSpace
from manabot.infra.hypers import AgentHypers, ObservationSpaceHypers
from manabot.model.agent import Agent
from manabot.verify.competency import SCENARIOS

ROOT = Path(__file__).resolve().parents[2]


def _int8_candidates() -> list[PlayerRegistration]:
    return [
        PlayerRegistration.model_validate(
            json.loads((ROOT / "experiments/candidates" / name).read_text())
        )
        for name in (
            "int-8-uniform-prior-puct-32-v1.json",
            "int-8-chosen-policy-prior-puct-32-v1.json",
            "int-8-visit-policy-prior-puct-32-v1.json",
        )
    ]


def _decision_fixture(
    *, chosen_delta: float, visit_delta: float
) -> tuple[
    list[PlayerRegistration],
    list[dict[str, object]],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    candidates = _int8_candidates()
    anchors = {f"anchor-{index}" for index in range(5)}
    rows = []
    for candidate, score in zip(
        candidates, (0.4, 0.4 + chosen_delta, 0.4 + visit_delta), strict=True
    ):
        for anchor in sorted(anchors):
            for block in range(2):
                for leg in range(2):
                    rows.append(
                        {
                            "player_a": candidate.player_id,
                            "player_b": anchor,
                            "deal_block": block,
                            "leg": leg,
                            "score_a": score,
                            "replay_passed": True,
                            "integrity": {"failures": 0},
                        }
                    )
    competencies = {"players": {}}
    for index, candidate in enumerate(candidates):
        competencies["players"][candidate.player_id] = {
            f"scenario-{scenario}": {
                "runs": [{"correct": index != 0} for _ in range(2)]
            }
            for scenario in range(5)
        }
    profile = {
        "players": {
            candidate.player_id: {
                "root_mutations": 0,
                "illegal_actions": 0,
                "playout_cap_rate": 0.0,
            }
            for candidate in candidates
        }
    }
    player_metrics = {
        candidates[0].player_id: {"p95_seconds": 1.0, "nodes_per_second": 100.0},
        candidates[1].player_id: {"p95_seconds": 1.05, "nodes_per_second": 95.0},
        candidates[2].player_id: {"p95_seconds": 1.2, "nodes_per_second": 80.0},
    }
    mechanism = {
        "selected_command_agreement": {
            "uniform32_vs_uniform128": {"aggregate": {"rate": 0.5}},
            "chosen32_vs_uniform128": {"aggregate": {"rate": 0.75}},
            "visit32_vs_uniform128": {"aggregate": {"rate": 0.55}},
        },
        "player_metrics": player_metrics,
    }
    return candidates, rows, competencies, profile, mechanism


def test_diagnostic_materializes_three_matched_budget_curves() -> None:
    variants = diagnostic_profile_variants(_int8_candidates())
    assert len(variants) == 9
    assert {variant.player_spec["sims"] for variant in variants} == {8, 32, 128}
    assert all(
        variant.player_seed_derivation_id == "arena-comparison-alias-player-v1"
        for variant in variants
    )


def test_int8_preflight_adds_current_identity_without_mutating_int6(
    tmp_path: Path,
) -> None:
    retained = (
        ROOT
        / "experiments/data/int-8-retained-int-4-smoke-v1/sha256"
        / "13868767846b7004f140cfade3652909347bdbb6708b69cb8c10b36ec2756eb0"
    )
    experiment_contract_path = (
        ROOT / "experiments/contracts/int-8-student-signal-guidance-v1.json"
    )
    experiment_contract = json.loads(experiment_contract_path.read_text())
    compatibility_path = tmp_path / "compatibility.json"
    compatibility_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "compatible",
                "evidence_class": "engineering_smoke_only_no_admission_claim",
                **experiment_contract["input"],
                "conversion_performed": False,
                "rewriting_performed": False,
                "retraining_performed": False,
                "substitution_performed": False,
            }
        )
    )
    args = Namespace(
        contract=ROOT / "experiments/contracts/int-6-skill-arena-v1.json",
        experiment_contract=experiment_contract_path,
        input_compatibility=compatibility_path,
        uniform_candidate=(
            ROOT / "experiments/candidates/int-8-uniform-prior-puct-32-v1.json"
        ),
        chosen_candidate=(
            ROOT / "experiments/candidates/int-8-chosen-policy-prior-puct-32-v1.json"
        ),
        chosen_checkpoint=(
            retained
            / "payload/training/chosen_policy_only-seed-197-9004b87e2be4a893.pt"
        ),
        visit_candidate=(
            ROOT / "experiments/candidates/int-8-visit-policy-prior-puct-32-v1.json"
        ),
        visit_checkpoint=(
            retained / "payload/training/visit_policy_only-seed-197-c2c8dcec02dbcf19.pt"
        ),
    )
    (
        _,
        frozen_contract_sha,
        runtime,
        _,
        _,
        authority,
        candidates,
        _,
    ) = preflight_int8_dependencies(args, validate_checkpoint_load=False)
    assert frozen_contract_sha == (
        "fc9cb76c0d80ad64951455ac6fede94b1355f383dde2f2964d0e578f62671a71"
    )
    assert authority["frozen_int6_contract_sha256"] == frozen_contract_sha
    assert authority["current_arena_implementation"]["source_sha256"]
    assert runtime["engine_extension_sha256"] == (
        "c95a85bba1128e6c3afdade5b5cf59dfeb3b1ec464fdd403cdf155a5cf834f8e"
    )
    assert len(candidates) == 3


def test_diagnostic_decision_selects_only_one_clearing_signal() -> None:
    candidates, rows, competencies, profile, mechanism = _decision_fixture(
        chosen_delta=0.2, visit_delta=0.05
    )
    decision = diagnostic_decision_payload(
        candidates=candidates,
        rows=rows,
        competencies=competencies,
        profile=profile,
        mechanism=mechanism,
        resource_caps={"passed": True},
        anchor_ids={f"anchor-{index}" for index in range(5)},
    )
    assert decision["decision"] == "next_corpus_chosen_action"
    assert decision["promotion_eligible"] is False
    assert decision["admission_eligible"] is False


def test_diagnostic_decision_kills_ambiguous_smoke_guidance() -> None:
    candidates, rows, competencies, profile, mechanism = _decision_fixture(
        chosen_delta=0.0, visit_delta=0.0
    )
    decision = diagnostic_decision_payload(
        candidates=candidates,
        rows=rows,
        competencies=competencies,
        profile=profile,
        mechanism=mechanism,
        resource_caps={"passed": True},
        anchor_ids={f"anchor-{index}" for index in range(5)},
    )
    assert decision["decision"] == "kill_retained_smoke_policy_guidance"


def _write_fixture_checkpoint(path: Path) -> int:
    observation_hypers = ObservationSpaceHypers()
    agent_hypers = AgentHypers()
    agent = Agent(ObservationSpace(observation_hypers), agent_hypers)
    torch.save(
        {
            "hypers": {
                "observation_hypers": observation_hypers.model_dump(),
                "agent_hypers": agent_hypers.model_dump(),
            },
            "model_state_dict": agent.state_dict(),
        },
        path,
    )
    return sum(parameter.numel() for parameter in agent.parameters())


def test_production_checkpoint_missing_bytes_fails_before_output(
    tmp_path: Path,
) -> None:
    contract = json.loads(
        (ROOT / "experiments/contracts/int-6-skill-arena-v1.json").read_text()
    )
    runtime = contract["runtime"]
    registration = {
        "player_id": "checkpoint-production-v1",
        "display_name": "checkpoint production",
        "role": "challenger",
        "runner_kind": "checkpoint",
        "player_spec": {
            "kind": "checkpoint",
            "deterministic": True,
            "device": "cpu",
            "batch_size": 1,
        },
        "compute_class_id": "policy-cpu-batch1-v1",
        "information_boundary": contract["key"]["viewer_boundary"],
        "world": "w2",
        "content_suite": contract["key"]["content_suite"],
        "observation_abi_sha256": runtime["observation_abi_sha256"],
        "action_abi_sha256": runtime["action_abi_sha256"],
        "matchup_sha256": runtime["matchup_sha256"],
        "checkpoint_sha256": "b" * 64,
        "checkpoint_bytes": 1,
        "parameter_count": 1,
        "training_seed": 1,
        "artifact_id": "immutable-production-id",
        "evidence_class": "production",
        "player_seed_derivation_id": "arena-pair-deal-player-v1",
    }
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(registration))
    out_dir = tmp_path / "must-not-exist"
    with pytest.raises(ArenaError, match="checkpoint candidate bytes are unavailable"):
        preflight_candidate(
            candidate_path,
            tmp_path / "missing.pt",
            contract=ArenaContract.model_validate(contract),
            profile_name="production",
        )
    assert not out_dir.exists()


def test_fixture_checkpoint_loads_replays_profiles_and_runs_competencies(
    tmp_path: Path,
) -> None:
    contract = ArenaContract.model_validate(
        json.loads(
            (ROOT / "experiments/contracts/int-6-skill-arena-v1.json").read_text()
        )
    )
    checkpoint_path = tmp_path / "fixture.pt"
    parameter_count = _write_fixture_checkpoint(checkpoint_path)
    registration_payload = {
        "player_id": "checkpoint-fixture-v1",
        "display_name": "checkpoint fixture",
        "role": "challenger",
        "runner_kind": "checkpoint",
        "player_spec": {
            "kind": "checkpoint",
            "deterministic": True,
            "device": "cpu",
            "batch_size": 1,
        },
        "compute_class_id": "policy-cpu-batch1-v1",
        "information_boundary": contract.key.viewer_boundary,
        "world": "w2",
        "content_suite": contract.key.content_suite,
        "observation_abi_sha256": contract.runtime["observation_abi_sha256"],
        "action_abi_sha256": contract.runtime["action_abi_sha256"],
        "matchup_sha256": contract.runtime["matchup_sha256"],
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "parameter_count": parameter_count,
        "training_seed": 17,
        "artifact_id": "fixture-checkpoint-seed-17",
        "evidence_class": "fixture",
        "player_seed_derivation_id": "arena-pair-deal-player-v1",
    }
    registration_path = tmp_path / "fixture.json"
    registration_path.write_text(json.dumps(registration_payload))
    registration, checkpoint_paths = preflight_candidate(
        registration_path,
        checkpoint_path,
        contract=contract,
        profile_name="smoke",
    )
    assert isinstance(registration, PlayerRegistration)
    rows, _, replay = play_cell(
        key=contract.key,
        player_a=registration,
        player_b=contract.anchors[0],
        deal_seeds=(781,),
        out_dir=tmp_path / "checkpoint-match",
        checkpoint_paths=checkpoint_paths,
    )
    assert replay["passed"]
    assert all(row["replay_passed"] for row in rows)
    _, source_trace, _ = play_cell(
        key=contract.key,
        player_a=contract.anchors[0],
        player_b=contract.anchors[1],
        deal_seeds=(782,),
        out_dir=tmp_path / "profile-source",
    )
    profile = profile_players(
        [registration],
        source_games=read_trace(Path(source_trace["path"])),
        profile_roots=ProfileRoots(
            source_cell="random-v1__scripted-greedy-v1",
            selection="deal-leg-revision-canonical-v1",
            warmup=0,
            measured=1,
            sampler_interval_ms=5,
        ),
        checkpoint_paths=checkpoint_paths,
    )
    verify_profile(profile)
    assert profile["players"][registration.player_id]["illegal_actions"] == 0
    competencies = run_competencies(
        [registration], seeds=(62001,), checkpoint_paths=checkpoint_paths
    )
    assert set(competencies["players"][registration.player_id]) == set(SCENARIOS)
