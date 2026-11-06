"""Current-policy PUCT admission and search behavior, without fitting."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from manabot.env.match import Match
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig
from manabot.infra.identity import canonical_sha256, file_sha256
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import load_checkpoint, save_checkpoint
from manabot.sim.guidance import build_registered_player
from manabot.sim.native_game import execute_game
from manabot.sim.registration import PlayerRegistration, SearchSemantics
from manabot.sim.replay import replay_games
from manabot.sim.search.branch import SELECTED_BRANCH_DRIVER_ID
from manabot.sim.search.puct import (
    AgentLeafEvaluator,
    DeterminizedPuctPlayer,
    puct_source_sha256,
)
import managym
from tests.sim.test_local_update import small_match
from tests.sim.test_mcts import _fresh_engine_and_observation


def _flat_agent(*, categorical: bool = False) -> tuple[ActorCritic, ObservationSpace]:
    torch.set_num_threads(1)
    space = ObservationSpace()
    agent = ActorCritic(
        space,
        ModelConfig(
            hidden_dim=16,
            num_attention_heads=2,
            value_kind="categorical_wdl" if categorical else "scalar",
        ),
    )
    with torch.no_grad():
        for parameter in agent.parameters():
            parameter.zero_()
    return agent, space


@pytest.mark.parametrize("signed", [-1.0, 0.0, 1.0, 2.0])
def test_signed_critic_units_and_actor_inversion(signed: float) -> None:
    env, obs = _fresh_engine_and_observation(51)
    agent, space = _flat_agent()
    agent.value_semantic = "signed_outcome"
    with torch.no_grad():
        agent.value_head[-1].bias.fill_(signed)
    evaluator = AgentLeafEvaluator(agent, space)
    actor = int(env.current_agent_index())
    for hero in (actor, 1 - actor):
        leaf = evaluator.evaluate(
            env, obs, root_player=hero, node_player=actor, seed=0, max_steps=3
        )
        score = (signed + 1) / 2
        assert leaf.root_value == pytest.approx(score if hero == actor else 1 - score)


def test_categorical_critic_preserves_expected_win_score() -> None:
    env, obs = _fresh_engine_and_observation(51)
    agent, space = _flat_agent(categorical=True)
    with torch.no_grad():
        agent.value_head[-1].bias.copy_(torch.tensor([0.2, 0.3, 0.5]).log())
    actor = int(env.current_agent_index())
    leaf = AgentLeafEvaluator(agent, space).evaluate(
        env, obs, root_player=actor, node_player=actor, seed=0, max_steps=3
    )
    assert leaf.root_value == pytest.approx(0.5 + 0.3 / 2)


def test_checkpoint_semantics_are_required_and_cannot_contradict_architecture(
    tmp_path: Path,
) -> None:
    agent, space = _flat_agent()
    path = tmp_path / "unknown.pt"
    configs = Match(small_match()).to_rust()
    save_checkpoint(agent, space, path, player_configs=configs)
    loaded, _ = load_checkpoint(str(path))
    with pytest.raises(ValueError, match="explicit value semantic"):
        AgentLeafEvaluator(loaded, space)
    AgentLeafEvaluator(loaded, space, value_mode="neutral")
    for semantic in ("signed_outcome", "win_logit"):
        save_checkpoint(
            agent,
            space,
            path,
            player_configs=configs,
            extra={"value_semantic": semantic},
        )
        loaded, _ = load_checkpoint(str(path))
        assert loaded.value_semantic == semantic
    categorical, _ = _flat_agent(categorical=True)
    save_checkpoint(
        categorical,
        space,
        path,
        player_configs=configs,
        extra={"value_semantic": "win_logit"},
    )
    with pytest.raises(ValueError, match="conflict"):
        load_checkpoint(str(path))
    save_checkpoint(
        agent,
        space,
        path,
        player_configs=configs,
        extra={"value_semantic": "probability"},
    )
    with pytest.raises(ValueError, match="unsupported checkpoint value semantic"):
        load_checkpoint(str(path))


def test_nonfinite_values_and_incomplete_offers_fail_closed() -> None:
    env, obs = _fresh_engine_and_observation(51)
    agent, space = _flat_agent()
    agent.value_semantic = "signed_outcome"
    evaluator = AgentLeafEvaluator(agent, space)
    encoded = space.encode(obs)
    with pytest.raises(ValueError, match="complete legal offers"):
        evaluator.root_priors(encoded, action_count=len(encoded["actions_valid"]) + 1)
    with torch.no_grad():
        agent.value_head[-1].bias.fill_(float("nan"))
    with pytest.raises(ValueError, match="nonfinite"):
        evaluator.root_priors(obs, action_count=env.action_count())


def test_live_player_deadline_yields_no_decision() -> None:
    env, obs = _fresh_engine_and_observation(51)
    agent, space = _flat_agent()
    agent.value_semantic = "signed_outcome"
    player = DeterminizedPuctPlayer(
        4, worlds=1, max_steps=3, evaluator=AgentLeafEvaluator(agent, space)
    )
    wrapper = SimpleNamespace(_engine=env, match=Match(small_match()))
    player.start_game(wrapper, 0)
    player.deadline = 1.0
    with pytest.raises(TimeoutError):
        player.act(wrapper, space.encode(obs))
    assert player.stats.decisions == 0


def test_registered_puct_completes_games_and_replays(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(15)
    space = ObservationSpace()
    agent = ActorCritic(
        space,
        ModelConfig(hidden_dim=16, num_attention_heads=2),
    )
    path = tmp_path / "fixture.pt"
    save_checkpoint(
        agent,
        space,
        path,
        player_configs=Match(small_match()).to_rust(),
        extra={"value_semantic": "signed_outcome"},
    )
    identity = "0" * 64
    common = dict(
        role="challenger",
        compute_class_id="bounded-fixture",
        information_boundary="acting-viewer-only-v1",
        world=managym.WORLD_VERSION,
        content_suite="tiny-fixture",
        observation_abi_sha256=identity,
        action_abi_sha256=identity,
        matchup_sha256=identity,
        player_seed_derivation_id="fixture-seed",
    )
    candidate = PlayerRegistration(
        **common,
        player_id="puct-fixture",
        display_name="PUCT fixture",
        runner_kind="checkpoint",
        checkpoint_sha256=file_sha256(path),
        checkpoint_bytes=path.stat().st_size,
        parameter_count=sum(p.numel() for p in agent.parameters()),
        training_seed=15,
        artifact_id="untrained-fixture-15",
        evidence_class="fixture",
        player_spec=dict(
            kind="agent_puct",
            sims=2,
            worlds=1,
            c_puct=1.5,
            max_steps=2,
            device="cpu",
            batch_size=1,
            value_mode="learned",
            branch_audit=True,
            branch_driver_id=SELECTED_BRANCH_DRIVER_ID,
            implementation_source_sha256=puct_source_sha256(),
        ),
        search_call_seed_derivation_id="mcts-mix-player-seed-decision-v1",
        search_semantics=SearchSemantics(
            branch_audit=True,
            root_prior="checkpoint-canonical-policy/v1",
            leaf_evaluator="checkpoint-declared-value-win-score/v1",
        ),
    )
    bad = candidate.model_dump()
    bad["player_spec"]["worlds"] = 3
    with pytest.raises(ValueError, match="invalid agent PUCT"):
        PlayerRegistration.model_validate(bad)
    drift = candidate.model_copy(
        update={
            "player_spec": {
                **candidate.player_spec,
                "implementation_source_sha256": identity,
            }
        }
    )
    with pytest.raises(ValueError, match="source drift"):
        build_registered_player(drift, seed=1, checkpoint_path=str(path))
    wrong_bytes = candidate.model_copy(update={"checkpoint_sha256": identity})
    with pytest.raises(ValueError, match="byte identity drift"):
        build_registered_player(wrong_bytes, seed=1, checkpoint_path=str(path))
    opponent = PlayerRegistration(
        **common,
        player_id="random-fixture",
        display_name="Random",
        runner_kind="code",
        source_sha256=identity,
        player_spec={"kind": "random"},
    )
    games: list[dict[str, Any]] = []  # Arena's persisted JSON boundary.
    for leg in range(2):
        registrations = [candidate, opponent] if leg == 0 else [opponent, candidate]
        game: dict[str, Any] = dict(
            match_id=f"puct-fixture-{leg}",
            deal_seed=93,
            leg=leg,
            match_hypers=small_match().model_dump(),
            seat_players=[p.player_id for p in registrations],
            player_seeds={p.player_id: 7 for p in registrations},
            decisions=[],
            integrity=dict(
                private_exposures=0, root_mutations=0, illegal_actions=0, truncations=0
            ),
        )
        messages: list[tuple[str, object]] = []
        execute_game(
            game, registrations, {candidate.player_id: str(path)}, 1000, messages.append
        )
        assert game.get("terminated"), game.get("failure")
        assert all(v == 0 for v in game["integrity"].values())
        game["game_trace_sha256"] = canonical_sha256(game)
        games.append(game)
    assert replay_games(games).passed
