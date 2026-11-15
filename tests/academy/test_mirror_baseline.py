"""A current-world, untrained smoke checkpoint exercises the real provider."""

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from manabot.backends.torch.checkpoint import save_checkpoint
from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig
from manabot.sim.policy_provider import BaselineFactory
import managym
from managym.academy import server
from managym.academy.opponent import set_policy_registry
from managym.academy.watch import WatchRequest, create_watch, inspect_watch, step_watch
from managym.bots import passive_policy
from managym.mirror_search import MirrorSearch
from managym.policy import PolicyRegistry, PolicySpec


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
    spec = MirrorSearch()
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
                step_watch(created["id"], created["token"], revision)
                row = inspect_watch(created["id"], revision, created["token"])
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
            server._drop_session(created["id"], "test")
    finally:
        game.close("test")
        set_policy_registry(None)
