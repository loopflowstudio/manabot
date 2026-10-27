"""Current-policy PUCT admission and canonical-prefix behavior, without fitting."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

<<<<<<< HEAD
from manabot.arena.match import _execute_game
from manabot.arena.models import (
    PlayerRegistration,
    SearchSemantics,
    canonical_sha256,
    file_sha256,
)
from manabot.arena.players import build_player
from manabot.arena.replay import replay_games
from manabot.env import Match, ObservationSpace
from manabot.infra.hypers import AgentSpec
from manabot.model.agent import Agent
from manabot.sim.distill import save_bc_checkpoint
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.sim.local_compound import advance_compound, project_compound
from manabot.sim.mcts import (
=======
from manabot.env import Match, ObservationSpace
from manabot.infra.config import ModelConfig
from manabot.infra.identity import canonical_sha256, file_sha256
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import load_checkpoint, save_checkpoint
from manabot.sim.match import _execute_game
from manabot.sim.players import build_player
from manabot.sim.registration import PlayerRegistration, SearchSemantics
from manabot.sim.replay import replay_games
from manabot.sim.search.branch import SELECTED_BRANCH_DRIVER_ID
from manabot.sim.search.compound import advance_compound, project_compound
from manabot.sim.search.puct import (
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
    AgentLeafEvaluator,
    DeterminizedPuctPlayer,
    determinized_puct,
    puct_source_sha256,
)
<<<<<<< HEAD
from manabot.sim.search_branch import SELECTED_BRANCH_DRIVER_ID
=======
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
import managym
from tests.sim.test_local_update import small_match
from tests.sim.test_mcts import _fresh_engine_and_observation
from tests.training.test_compound import _agent, _block_root, _root, _waterbend_root


def _check() -> None:
    pass


<<<<<<< HEAD
def _flat_agent(*, categorical: bool = False) -> tuple[Agent, ObservationSpace]:
    torch.set_num_threads(1)
    space = ObservationSpace()
    agent = Agent(
        space,
        AgentSpec(
=======
def _flat_agent(*, categorical: bool = False) -> tuple[ActorCritic, ObservationSpace]:
    torch.set_num_threads(1)
    space = ObservationSpace()
    agent = ActorCritic(
        space,
        ModelConfig(
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
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
<<<<<<< HEAD
    save_bc_checkpoint(agent, space, path, player_configs=configs)
    loaded, _ = load_checkpoint_agent(str(path))
=======
    save_checkpoint(agent, space, path, player_configs=configs)
    loaded, _ = load_checkpoint(str(path))
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
    with pytest.raises(ValueError, match="explicit value semantic"):
        AgentLeafEvaluator(loaded, space)
    AgentLeafEvaluator(loaded, space, value_mode="neutral")
    for semantic in ("signed_outcome", "win_logit"):
<<<<<<< HEAD
        save_bc_checkpoint(
=======
        save_checkpoint(
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
            agent,
            space,
            path,
            player_configs=configs,
            extra={"value_semantic": semantic},
        )
<<<<<<< HEAD
        loaded, _ = load_checkpoint_agent(str(path))
        assert loaded.value_semantic == semantic
    categorical, _ = _flat_agent(categorical=True)
    save_bc_checkpoint(
=======
        loaded, _ = load_checkpoint(str(path))
        assert loaded.value_semantic == semantic
    categorical, _ = _flat_agent(categorical=True)
    save_checkpoint(
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
        categorical,
        space,
        path,
        player_configs=configs,
        extra={"value_semantic": "win_logit"},
    )
    with pytest.raises(ValueError, match="conflict"):
<<<<<<< HEAD
        load_checkpoint_agent(str(path))
    save_bc_checkpoint(
=======
        load_checkpoint(str(path))
    save_checkpoint(
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
        agent,
        space,
        path,
        player_configs=configs,
        extra={"value_semantic": "probability"},
    )
    with pytest.raises(ValueError, match="unsupported checkpoint value semantic"):
<<<<<<< HEAD
        load_checkpoint_agent(str(path))
=======
        load_checkpoint(str(path))
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f


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


@pytest.mark.parametrize("kind", ["attack", "blockers", "cast", "payment"])
def test_compound_tree_grows_without_flat_head_or_root_mutation(kind: str) -> None:
    env, _ = (
        _block_root(2, 2)
        if kind == "blockers"
        else _waterbend_root(6, lands=2)
        if kind == "payment"
        else _root(kind, 3)
    )
    torch.manual_seed(9)
    agent = _agent()
    # A compound search must be independent of both unused flat heads.
    with torch.no_grad():
        for parameter in agent.value_head.parameters():
            parameter.fill_(float("nan"))
    evaluator = AgentLeafEvaluator(agent, ObservationSpace())
    before = env.state_digest()
    first = determinized_puct(
        env, simulations=12, worlds=2, seed=91, max_steps=5, evaluator=evaluator
    )
    second = determinized_puct(
        env, simulations=12, worlds=2, seed=91, max_steps=5, evaluator=evaluator
    )
    assert env.state_digest() == before
    assert first.visit_counts.sum() == 12
    assert first.max_depth > 1
    np.testing.assert_array_equal(first.visit_counts, second.visit_counts)
    np.testing.assert_array_equal(first.q_values, second.q_values)
    assert np.isfinite(first.q_values).all()


def test_retained_payment_prefix_excludes_still_legal_actions() -> None:
    env, _ = _waterbend_root(6, lands=2)
    agent = _agent().eval()
    with torch.no_grad():
        agent.compound_decoder.value.weight.zero_()
        agent.compound_decoder.value.bias.fill_(-4)
    projection = project_compound(agent, env, None, _check)
    action = next(
        i for i, choice in enumerate(projection.choices) if choice.prefix == (0, 0, 1)
    )
    cursor = advance_compound(agent, projection, action, _check)
    assert cursor is not None
    env.execute_semantic_command_json(projection.choices[action].command.to_json())
    expected = project_compound(agent, env, cursor, _check)
    excluded = expected.probabilities == 0
    assert excluded.any()
    evaluator = AgentLeafEvaluator(agent, ObservationSpace())
    before = env.state_digest()
    result = determinized_puct(
        env,
        simulations=8,
        worlds=2,
        seed=13,
        max_steps=4,
        evaluator=evaluator,
        compound_cursor=cursor,
    )
    assert env.state_digest() == before
    assert np.all(result.visit_counts[excluded] == 0)
    assert result.visit_counts.sum() == 8
    assert evaluator.last_root_projection.cursor.root is cursor.root
    np.testing.assert_array_equal(evaluator.last_root_priors, expected.probabilities)
    swapped = env.clone_env()
    swapped.determinize(124, int(env.current_agent_index()))
    other = project_compound(agent, swapped, cursor, _check)
    np.testing.assert_array_equal(other.probabilities, expected.probabilities)


def test_live_player_retains_prefix_and_resets_at_game_boundary() -> None:
    env, _ = _root("attack", 3)
    agent = _agent()
    player = DeterminizedPuctPlayer(
        4,
        worlds=1,
        max_steps=3,
        evaluator=AgentLeafEvaluator(agent, ObservationSpace()),
    )
    wrapper = SimpleNamespace(_engine=env, match=Match(small_match()))
    before = env.state_digest()
    action = player.act(wrapper, {})
    assert env.state_digest() == before
    first = player.compound_cursor
    assert first is not None
    env.step(action)
    player.act(wrapper, {})
    second = player.compound_cursor
    assert second is not None
    assert second.root is first.root
    assert len(second.commands) == len(first.commands) + 1
    player.start_game(wrapper, 0)
    assert player.compound_cursor is None
    player.deadline = 1.0
    decisions = player.stats.decisions
    with pytest.raises(TimeoutError):
        player.act(wrapper, {})
    assert player.stats.decisions == decisions


@pytest.mark.parametrize("compound", [False, True])
def test_registered_puct_completes_games_and_replays(
    tmp_path: Path, compound: bool
) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(15)
    space = ObservationSpace()
<<<<<<< HEAD
    agent = Agent(
        space,
        AgentSpec(hidden_dim=16, num_attention_heads=2, compound_decisions=compound),
    )
    path = tmp_path / "fixture.pt"
    save_bc_checkpoint(
=======
    agent = ActorCritic(
        space,
        ModelConfig(hidden_dim=16, num_attention_heads=2, compound_decisions=compound),
    )
    path = tmp_path / "fixture.pt"
    save_checkpoint(
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
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
        build_player(drift, seed=1, checkpoint_path=str(path))
    wrong_bytes = candidate.model_copy(update={"checkpoint_sha256": identity})
    with pytest.raises(ValueError, match="byte identity drift"):
        build_player(wrong_bytes, seed=1, checkpoint_path=str(path))
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
            match_id=f"puct-fixture-{compound}-{leg}",
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
        _execute_game(
            game, registrations, {candidate.player_id: str(path)}, 1000, messages.append
        )
        assert game.get("terminated"), game.get("failure")
        assert all(v == 0 for v in game["integrity"].values())
        game["game_trace_sha256"] = canonical_sha256(game)
        games.append(game)
    assert replay_games(games).passed
