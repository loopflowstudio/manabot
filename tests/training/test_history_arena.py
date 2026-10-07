"""Fixed-weight history ABI serving and replay; no training or scientific deals."""

from pathlib import Path

import pytest
import torch

from experiments.runners.history_input import recipes, runtime_bindings
from manabot.arena.match import SELECTED_SUITE, derive_seed, play_cell
from manabot.arena.models import (
    ArenaKey,
    PlayerRegistration,
    canonical_sha256,
    file_sha256,
)
from manabot.arena.replay import read_trace
from manabot.env import Match, ObservationSpace
from manabot.model.agent import Agent
from manabot.sim.distill import save_bc_checkpoint
import managym


def test_history_on_full_game_uses_own_abi_and_fixed_anchor(tmp_path: Path) -> None:
    # Do not silently skip during campaign preflight: unavailable native history
    # must fail admission before calibration.
    if not hasattr(managym.Observation, "encode_policy_history"):
        pytest.fail(
            "installed native extension lacks encode_policy_history; install the source-matched native build before campaign admission"
        )
    torch.set_num_threads(1)
    values = recipes()
    common, bindings = runtime_bindings(values)
    anchor = PlayerRegistration(
        player_id="scripted-greedy-fixed-anchor",
        display_name="scripted-greedy",
        role="anchor",
        runner_kind="code",
        player_spec={"kind": "scripted_greedy"},
        source_sha256=canonical_sha256(
            {
                "engine": common["engine_source_sha256"],
                "manabot": common["training_source_sha256"],
            }
        ),
        compute_class_id="scripted-greedy-cpu",
        information_boundary="acting-viewer",
        world=values[0].world,
        content_suite=SELECTED_SUITE,
        observation_abi_sha256=bindings[0].observation_abi_sha256,
        action_abi_sha256=common["action_abi_sha256"],
        matchup_sha256=common["matchup_sha256"],
        player_seed_derivation_id="arena-pair-deal-player-v1",
    )
    key = ArenaKey(
        world=values[0].world,
        content_suite=SELECTED_SUITE,
        viewer_boundary="acting-viewer",
        arena_version="training-regime-smoke-v1",
        rating_model_version="unrated",
        rating_prior_sha256=canonical_sha256({}),
        anchor_cohort_sha256=canonical_sha256([anchor.model_dump(mode="json")]),
        evaluation_compute_envelope_id="policy-cpu-one-thread-one-pass",
    )
    seeds: list[int] = []
    for recipe, binding in zip(values, bindings, strict=True):
        torch.manual_seed(811)
        space = ObservationSpace(recipe.observation)
        agent = Agent(space, recipe.agent)
        path = tmp_path / f"{recipe.id}.pt"
        save_bc_checkpoint(
            agent, space, path, player_configs=Match(recipe.match).to_rust()
        )
        candidate = PlayerRegistration(
            player_id=recipe.id,
            display_name=recipe.id,
            role="challenger",
            runner_kind="checkpoint",
            player_spec={
                "kind": "checkpoint",
                "deterministic": False,
                "device": "cpu",
                "batch_size": 1,
            },
            checkpoint_sha256=file_sha256(path),
            checkpoint_bytes=path.stat().st_size,
            parameter_count=sum(parameter.numel() for parameter in agent.parameters()),
            training_seed=811,
            artifact_id=f"history-fixture/{recipe.id}/raw",
            compute_class_id="policy-cpu-one-thread-one-pass",
            information_boundary="acting-viewer",
            world=recipe.world,
            content_suite=SELECTED_SUITE,
            observation_abi_sha256=binding.observation_abi_sha256,
            action_abi_sha256=common["action_abi_sha256"],
            matchup_sha256=common["matchup_sha256"],
            player_seed_derivation_id="arena-pair-deal-player-v1",
        )
        aliases = {candidate.player_id: "candidate", anchor.player_id: "reference"}
        seeds.append(
            derive_seed(
                key,
                (candidate.player_id, anchor.player_id),
                971811,
                candidate.player_id,
                comparison_seed_aliases=aliases,
            )
        )
        rows, trace, replay = play_cell(
            key=key,
            player_a=candidate,
            player_b=anchor,
            deal_seeds=(971811,),
            # This fixture checks ABI/serving/replay, not shared-runner latency.
            game_seconds=120,
            max_commands=10000,
            out_dir=tmp_path / recipe.id,
            checkpoint_paths={candidate.player_id: str(path)},
            comparison_seed_aliases=aliases,
        )
        assert replay["passed"] and trace["games"] == 4
        assert len(rows) == 4 and all(
            r["terminated"] and r["replay_passed"] and r["failure"] is None
            for r in rows
        ), [
            {
                key: row.get(key)
                for key in (
                    "leg",
                    "failure",
                    "terminated",
                    "game_seconds",
                    "replay_passed",
                )
            }
            for row in rows
        ]

        games = read_trace(Path(trace["path"]))
        assert all(
            g["observation_hypers"].get("policy_history_version", 0)
            == binding.policy_history_version
            for g in games
        )
    assert seeds[0] == seeds[1]
