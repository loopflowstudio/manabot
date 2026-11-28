"""A current-world, untrained smoke checkpoint exercises the real provider."""

from dataclasses import asdict
import math
from pathlib import Path
from typing import cast

import pytest
import torch

from manabot.backends.torch.checkpoint import save_checkpoint
from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig
from manabot.sim.policy_provider import BaselineFactory, BaselineSession
import managym
from managym.academy import server
from managym.academy.opponent import set_policy_registry
from managym.academy.watch import WatchRequest, create_watch, inspect_watch, step_watch
from managym.bots import passive_policy
from managym.policy import PolicyRegistry, PolicySpec, PolicyView
from managym.search import SearchConfig


def write_search_checkpoint(path: Path) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(15)
    space = ObservationSpace()
    agent = ActorCritic(
        space,
        ModelConfig(
            hidden_dim=8,
            num_attention_heads=2,
            value_kind="categorical_wdl",
            world="ur-lessons-vs-gw-allies",
        ),
    )
    save_checkpoint(
        agent,
        space,
        path,
        player_configs=[
            managym.authored_deck_setup("ur-lessons-vs-gw-allies", key)
            for key in ("ur_lessons", "gw_allies")
        ],
    )


def test_baseline_provider_search_runs_in_play_and_either_watch_seat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server.trace_store, "TRACES_DIR", tmp_path / "traces")
    path = tmp_path / "untrained-smoke.pt"
    write_search_checkpoint(path)
    factory = BaselineFactory(PolicySpec("manabot", str(path)))
    assert factory.describe().supports_search
    registry = PolicyRegistry(providers={})
    key = registry.register(factory)
    set_policy_registry(registry)
    spec = SearchConfig()
    game = server.GameSession()
    try:
        game.new_game(
            {
                "villain_type": "policy",
                "opponent_sha256": key,
                "mirror": asdict(spec),
                "seed": 42,
                "auto_pass": False,
            }
        )
        for _ in range(6):
            assert (
                game.env is not None and game.obs is not None and game.trace is not None
            )
            if any(event.bot_decision is not None for event in game.trace.events):
                break
            game.hero_action(passive_policy(game.env, game.obs))
        assert game.trace is not None
        assert any(
            event.bot_decision and event.bot_decision.mirror
            for event in game.trace.events
        )
        created = create_watch(
            WatchRequest(
                players=[f"policy:{key}", f"policy:{key}"], mirror=[spec, spec], seed=42
            )
        )
        try:
            actors: set[int] = set()
            for revision in range(8):
                step_watch(created["id"], revision)
                row = inspect_watch(created["id"], revision)
                assert row["outputs"]["mirror"]["worlds"] == 1
                assert row["outputs"]["mirror"]["depth"] == 1
                assert len(row["outputs"]["probabilities"]) == len(
                    row["decision"]["frame"]["offers"]
                )
                actors.add(row["decision"]["viewer"])
                if actors == {0, 1}:
                    break
            assert actors == {0, 1}
        finally:
            server.end_game(created["id"], "test")
    finally:
        game.close("test")
        set_policy_registry(None)


def test_baseline_session_encodes_the_card_catalog_once_per_loaded_policy(
    tmp_path: Path,
) -> None:
    """Search evaluates many forks; re-encoding every card program per call
    made one-world, depth-one decisions exceed the 30-second ceiling."""
    path = tmp_path / "untrained-smoke.pt"
    write_search_checkpoint(path)
    session = BaselineSession(PolicySpec("manabot", str(path)), seed=0)
    env = managym.Env(seed=3, skip_trivial=True)
    obs, _ = env.reset(
        [
            managym.authored_deck_setup("ur-lessons-vs-gw-allies", key)
            for key in ("ur_lessons", "gw_allies")
        ]
    )
    view = PolicyView(
        obs,
        int(obs.agent.player_index),
        "binding",
        tuple(str(i) for i in range(len(obs.action_space.actions))),
    )
    assert session.agent is not None
    expected = session.evaluate(view)
    cache = session.agent._program_cache
    assert cache is not None
    branch = session.fork(seed=1)
    assert branch.evaluate(view) == expected
    assert session.agent._program_cache is cache
    branch.close()
    assert session.evaluate(view) == expected
    agent = session.agent
    session.close()
    assert agent._program_cache is None


@pytest.mark.parametrize(
    "estimate", [-1.0149, 1.0149, float("nan"), float("inf"), -float("inf")]
)
def test_baseline_session_projects_signed_outcome_estimates_onto_their_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, estimate: float
) -> None:
    """A scalar head declared signed_outcome can overshoot; search must not
    fail the decision (and end a Play game) on a -1.01 estimate."""
    path = tmp_path / "untrained-smoke.pt"
    write_search_checkpoint(path)
    session = BaselineSession(PolicySpec("manabot", str(path)), seed=0)
    monkeypatch.setattr(
        "manabot.backends.torch.bot.evaluate_observation",
        lambda scope, space, observation: ((1.0,), estimate),
    )
    view = PolicyView(cast(managym.Observation, None), 1, "binding", ("0",))
    try:
        if not math.isfinite(estimate):
            with pytest.raises(ValueError, match="nonfinite value estimate"):
                session.evaluate(view)
        else:
            value = session.evaluate(view).value
            expected = -1.0 if estimate < 0 else 1.0
            assert (value.value, value.seat, value.semantic) == (
                expected,
                1,
                "signed_outcome",
            )
    finally:
        session.close()
