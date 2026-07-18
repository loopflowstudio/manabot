import json
from pathlib import Path

import pytest
import torch

from experiments.runners.run_skill_arena import main, preflight_candidate
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
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
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
    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                "challenge",
                "--contract",
                str(ROOT / "experiments/contracts/int-6-skill-arena-v1.json"),
                "--anchor-artifact",
                str(tmp_path / "anchors" / "manifest.json"),
                "--candidate",
                str(candidate_path),
                "--candidate-checkpoint",
                str(tmp_path / "missing.pt"),
                "--out-dir",
                str(out_dir),
                "--profile",
                "production",
            ]
        )
    assert exit_info.value.code == 2
    assert "checkpoint candidate bytes are unavailable" in capsys.readouterr().err
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
