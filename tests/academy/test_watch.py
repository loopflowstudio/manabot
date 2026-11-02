"""Watch records choices once, preserves actor privacy, and forks either seat."""

from collections.abc import Iterator
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from academy import server
from academy.replay_index import restore_decision
from academy.watch import (
    WatchCreated,
    WatchRequest,
    create_watch,
    inspect_watch,
    step_watch,
)


@pytest.fixture
def watch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[WatchCreated]:
    monkeypatch.setattr(server.trace_store, "TRACES_DIR", tmp_path)
    result = create_watch(WatchRequest(players=["random", "random"], seed=42))
    yield result
    server._drop_session(result["id"], "test")


def test_watch_steps_are_exact_and_actor_safe(watch: WatchCreated) -> None:
    trace_id, token = watch["id"], watch["token"]
    client = TestClient(server.app)
    assert (
        client.post(f"/api/watch/{trace_id}/step", json={"revision": 0}).status_code
        == 403
    )
    actors: set[int] = set()
    for revision in range(20):
        result = step_watch(trace_id, token, revision)
        assert result["revision"] == revision + 1
        row = inspect_watch(trace_id, revision, token, reveal=False)
        actors.add(row["decision"]["viewer"])
        assert row["count"] == revision + 1
        assert row["observation"]["opponent"]["hand"] == []
        assert not row["revealed"]
        outputs = row["outputs"]
        assert sum(outputs["probabilities"]) == pytest.approx(1)
        assert outputs["probabilities"][outputs["action"]] > 0
        assert row == inspect_watch(trace_id, revision, token, reveal=False)
        spectator = inspect_watch(trace_id, revision, token, reveal=True)
        assert spectator["revealed"]
        assert spectator["decision"] == row["decision"]
        assert spectator["outputs"] == row["outputs"]
        assert (
            spectator["observation"]
            == server._live_record_for_trace(trace_id)
            .game.trace.events[revision]
            .analysis_observation
        )
        if revision == 0:
            assert spectator["observation"]["agent"]["hand"]
            assert spectator["observation"]["opponent"]["hand"]
            assert (
                client.get(f"/api/watch/{trace_id}/decisions/0?reveal=true").status_code
                == 403
            )
    assert actors == {0, 1}
    address = row["decision"]["address"]
    assert (
        client.get(
            f"/api/traces/{trace_id}/decisions/{address}",
            headers={"x-etude-participant-tokens": token},
        ).status_code
        == 403
    )
    assert (
        client.get(
            f"/api/traces/{trace_id}/decisions/{address}",
            headers={
                "x-etude-participant-token": token,
                "x-etude-participant-tokens": token,
            },
        ).status_code
        == 200
    )
    response = client.post(
        f"/api/watch/{trace_id}/step",
        headers={"x-etude-participant-token": token},
        json={"revision": 0},
    )
    assert response.status_code == 409
    assert len(server._live_record_for_trace(trace_id).game.canonical_decisions) == 20
    metadata = server._live_record_for_trace(trace_id).game.attempt_store.metadata(
        trace_id
    )
    assert [player["kind"] for player in metadata["players"]] == ["bot", "bot"]


def test_watch_roots_fork_both_seats_without_mutating_match(
    watch: WatchCreated,
) -> None:
    game = server._live_record_for_trace(watch["id"]).game
    for revision in range(12):
        step_watch(watch["id"], watch["token"], revision)
    original = game.env.state_digest()
    actors: set[int] = set()
    for index in range(12):
        row = inspect_watch(watch["id"], index, watch["token"])
        actor = row["decision"]["viewer"]
        address = row["decision"]["address"]
        branch = game.fork_study(address)
        actors.add(branch.viewer)
        assert branch.viewer == actor
        receipt = branch.return_to_recorded()
        expected = restore_decision(game.canonical_replay(), address, actor)
        assert receipt.frame == expected.frame
        assert receipt.command == expected.command
        command = row["decision"]["command"] | {"command_id": f"try.{index}"}
        result = game.retry_study(
            trace_id=watch["id"],
            replay=game.canonical_replay(),
            raw_address=address,
            raw_command=command,
        )
        assert result["retry"]["projection"]["agent"]["player_index"] == actor
        assert (
            game.return_from_study(result["attempt_id"])["command"]
            == row["decision"]["command"]
        )
        assert game.env.state_digest() == original

    assert actors == {0, 1}
    assert game.env.state_digest() == original


def test_search_captures_actual_scores(watch: WatchCreated) -> None:
    from academy.villain import choose_bot
    from manabot.sim.search.flat import FlatSearchPolicy

    game = server._live_record_for_trace(watch["id"]).game
    # Forced actions bypass search; reach a real choice before checking scores.
    for revision in range(64):
        if len(game.obs.action_space.actions) > 1:
            break
        step_watch(watch["id"], watch["token"], revision)
    assert len(game.obs.action_space.actions) > 1
    outputs = choose_bot(FlatSearchPolicy(sims=1, max_steps=2), game.env, game.obs)
    assert len(outputs.search_scores) == len(game.obs.action_space.actions)
    assert outputs.action == max(
        range(len(outputs.search_scores)), key=outputs.search_scores.__getitem__
    )


def test_arena_import_keeps_source_and_rejects_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import copy

    from academy.watch import import_arena
    from manabot.eval.match import play_cell
    from manabot.sim.replay import read_trace
    from tests.eval.arena_fixture import arena_fixture

    monkeypatch.setattr(server.trace_store, "TRACES_DIR", tmp_path / "academy")
    contract = arena_fixture()
    _, trace, receipt = play_cell(
        key=contract.key,
        player_a=contract.anchors[0],
        player_b=contract.anchors[1],
        deal_seeds=(777,),
        out_dir=tmp_path / "arena",
    )
    assert receipt["passed"]
    source = read_trace(Path(trace["path"]))[0]
    assert source.get("failure") is None
    original = copy.deepcopy(source)
    result = import_arena(source)
    assert source == original
    assert result["finished"]
    assert result["count"] == len(source["decisions"])
    inspected = inspect_watch(result["id"], 0, None)
    assert inspected["decision"]["frame"]["projection"]["opponent"]["hand"] == []
    assert not inspected["outputs"]["probabilities"]
    revealed = inspect_watch(result["id"], 0, None, reveal=True)
    assert revealed["revealed"]
    assert revealed["observation"]["opponent"]["hand"]
    # Saved acting-seat inspection outlives the controllers and retained roots.
    server._drop_session(result["id"], "test")
    second_seat = next(
        index
        for index, decision in enumerate(source["decisions"])
        if decision["actor"] == 1
    )
    saved = inspect_watch(result["id"], second_seat, None)
    response = TestClient(server.app).get(
        f"/api/traces/{result['id']}/decisions/{saved['decision']['address']}"
    )
    assert response.status_code == 200
    assert response.json()["viewer"] == 1
    source["decisions"][0]["command"]["offer_id"] += 1
    with pytest.raises(ValueError, match="verification failed"):
        import_arena(source)


@pytest.mark.parametrize("compound", [False, True])
def test_checkpoint_outputs_are_from_the_chosen_inference(
    tmp_path: Path,
    compound: bool,
) -> None:
    import torch

    from academy.villain import CheckpointVillain, choose_bot
    from manabot.env.observation import ObservationSpace
    from manabot.infra.config import ModelConfig
    from manabot.model.actor_critic import ActorCritic
    from manabot.model.world import checkpoint_world

    config = server._parse_game_config(
        {
            "hero_deck": "interactive",
            "villain_deck": "interactive",
            "villain_type": "random",
            "seed": 42,
        }
    )
    space = ObservationSpace()
    spec = ModelConfig(hidden_dim=16, compound_decisions=compound)
    agent = ActorCritic(space, spec)
    checkpoint = tmp_path / "agent.pt"
    torch.save(
        {
            "hypers": {
                "observation_hypers": space.encoder.config.model_dump(),
                "agent_hypers": spec.model_dump(),
            },
            "model_state_dict": agent.state_dict(),
            "world_binding": checkpoint_world(config.to_rust(), space),
            "bc": {"value_semantic": "signed_outcome"},
        },
        checkpoint,
    )
    policy = CheckpointVillain(
        str(checkpoint), deterministic=True, player_configs=config.to_rust()
    )
    engine = server.managym.Env(seed=42)
    observation, _ = engine.reset(config.to_rust())
    digest = engine.state_digest()
    decision = choose_bot(policy, engine, observation)
    assert decision.value is not None
    assert engine.state_digest() == digest
    if compound:
        assert decision.factors
        assert all(
            sum(factor.probabilities) == pytest.approx(1) for factor in decision.factors
        )
    else:
        assert len(decision.probabilities) == len(observation.action_space.actions)
        assert sum(decision.probabilities) == pytest.approx(1)
        assert decision.action == max(
            range(len(decision.probabilities)), key=decision.probabilities.__getitem__
        )
