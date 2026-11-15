"""A timed-out bot leaves its native decision intact until an explicit retry."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from fastapi import HTTPException
import pytest

import managym
from managym import mirror_search
from managym.academy import server
from managym.academy.opponent import set_policy_registry
from managym.academy.trace import load_trace, prepare_trace_payload
from managym.academy.watch import WatchRequest, create_watch, inspect_watch, step_watch
from managym.bots import passive_policy
from managym.mirror_search import SearchSpec
from managym.policy import PolicyDecision, PolicyDescriptor, PolicyRegistry, PolicyView
from tests.academy.test_policy_runtime import IndependentFactory
from tests.semantic.test_mirror_policy_search import Clock, IndependentSession, _runner


class PlayableSession(IndependentSession):
    def choose(self, view: PolicyView) -> PolicyDecision:
        return PolicyDecision(view.choices[0], view.binding)


@dataclass
class SearchFactory:
    sessions: list[IndependentSession] = field(default_factory=list)

    def describe(self) -> PolicyDescriptor:
        return PolicyDescriptor(
            "Independent search", "independent", "test-search/v1", supports_search=True
        )

    def admit(self, setup: Sequence[managym.PlayerConfig]) -> None:
        assert len(setup) == 2

    def load(self, *, seat: int, seed: int) -> PlayableSession:
        session = PlayableSession(seed=seed)
        self.sessions.append(session)
        return session


def test_search_flag_cannot_bypass_missing_evaluator_and_forks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class IncorrectFlagFactory(IndependentFactory):
        def describe(self) -> PolicyDescriptor:
            return replace(super().describe(), supports_search=True, recurrent=False)

    monkeypatch.setattr(server.trace_store, "TRACES_DIR", tmp_path)
    registry = PolicyRegistry(providers={})
    factory = IncorrectFlagFactory()
    key = registry.register(factory)
    set_policy_registry(registry)
    original = set(server.SESSION_REGISTRY)
    try:
        with pytest.raises(HTTPException) as failure:
            create_watch(
                WatchRequest(
                    players=[f"policy:{key}", "random"], search=[SearchSpec(), None]
                )
            )
        assert failure.value.status_code == 400
        assert failure.value.detail["code"] == "search_unavailable"
        assert "isolated forks" in failure.value.detail["message"]
        assert factory.sessions[0].closed
        assert set(server.SESSION_REGISTRY) == original
    finally:
        set_policy_registry(None)


def test_independent_adapter_search_plays_watches_and_retains_private_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server.trace_store, "TRACES_DIR", tmp_path)
    registry = PolicyRegistry(providers={})
    factory = SearchFactory()
    key = registry.register(factory)
    set_policy_registry(registry)
    spec = SearchSpec(worlds=1, depth=1)
    game = server.GameSession(trace_dir=tmp_path)
    try:
        game.new_game(
            {
                "villain_type": "policy",
                "opponent_sha256": key,
                "search": [None, asdict(spec)],
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
        assert game.trace is not None and game.trace.config.opponent is not None
        assert game.trace.config.opponent["sha256"] != key
        assert game.trace.config.opponent["base_sha256"] == key
        assert "mirror search 1×1" in game.trace.config.opponent["name"]
        saved = load_trace(game.match_id)
        outputs = next(
            event["bot_decision"]
            for event in saved["events"]
            if event.get("bot_decision")
        )
        assert outputs["search"]["worlds"] == 1
        assert outputs["belief"]["model"] == "physical-compatible-hand/v1"
        assert sum(outputs["probabilities"]) == pytest.approx(1)
        assert outputs["probabilities"][outputs["action"]] > 0
        assert all(
            "bot_decision" not in event
            for event in prepare_trace_payload(saved)["events"]
            if event["actor"] == "villain"
        )
        assert game.record_players is not None
        assert game.record_players[1].version == game.trace.config.opponent["sha256"]
        with pytest.raises(HTTPException) as refusal:
            inspect_watch(game.match_id, 0, None)
        assert refusal.value.status_code == 409

        created = create_watch(
            WatchRequest(
                players=[f"policy:{key}", f"policy:{key}"], search=[spec, None], seed=42
            )
        )
        try:
            for revision in range(6):
                step_watch(created["id"], created["token"], revision)
                row = inspect_watch(created["id"], revision, created["token"])
                if row["outputs"] and row["outputs"].get("search"):
                    assert row["outputs"]["search"]["worlds"] == 1
                    assert (
                        row["identities"][0]["version"]
                        != row["identities"][1]["version"]
                    )
                    break
            else:
                pytest.fail("Watch never reached the searching seat")
        finally:
            server._drop_session(created["id"], "test")
    finally:
        game.close("test")
        set_policy_registry(None)
    assert all(session.closed for session in factory.sessions)


def test_play_search_timeout_survives_reconnect_and_stop_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = Clock()
    live = IndependentSession(clock=clock)
    runner = _runner(live, 1)
    runner.configure_search(SearchSpec(worlds=1, depth=1), seed=3)
    monkeypatch.setattr(mirror_search, "time", clock)
    monkeypatch.setattr(server, "build_villain_policy", lambda _config: runner)
    game = server.GameSession(trace_dir=tmp_path)
    try:
        game.new_game(
            {
                "hero_deck": {"Mountain": 12, "Gray Ogre": 12},
                "villain_deck": {"Forest": 12, "Llanowar Elves": 12},
                "villain_type": "passive",
                "seed": 3,
                "auto_pass": False,
            }
        )
        for _ in range(6):
            if game.search_retry:
                break
            assert game.env is not None and game.obs is not None
            game.hero_action(passive_policy(game.env, game.obs))
        assert game.search_retry is not None
        assert game.env is not None and game.trace is not None
        original = game.env.state_digest()
        events = len(game.trace.events)
        revision = game.revision
        assert game.search_retry["code"] == "search_deadline_exceeded"
        recovered = game._wire_message(reason="reconnect")
        assert recovered["search_retry"] == game.search_retry
        assert recovered["frame"]["prompt"] is None
        assert recovered["frame"]["offers"] == []
        assert recovered["actions"] == []
        game.set_stops(None, None, None)
        assert len(game.trace.events) == events
        assert game.env.state_digest() == original
        with pytest.raises(ValueError, match="pending search changed"):
            game.retry_search("old-match", revision)
        assert game.env.state_digest() == original
        live.clock = None
        assert game.search_retry is not None
        assert game.search_retry["expected_revision"] == game.revision
        game.retry_search(game.match_id, game.revision)
        assert game.search_retry is None
        assert game.revision > revision
        assert len(game.trace.events) > events
        assert not live.evaluations
        assert all(child.closed for child in live.children)
    finally:
        game.close("test")
        runner.close()


def test_policy_replacement_releases_failed_candidate_and_preserves_live_game(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = PolicyRegistry(providers={})
    factory = SearchFactory()
    key = registry.register(factory)
    set_policy_registry(registry)
    game = server.GameSession(trace_dir=tmp_path)
    config = {
        "villain_type": "policy",
        "opponent_sha256": key,
        "seed": 42,
        "auto_pass": False,
    }
    try:
        game.new_game(config)
        assert game.env is not None
        original = game.env.state_digest()

        def refuse_reset(*, seed: int) -> managym.Env:
            raise ValueError("fixture reset refused")

        monkeypatch.setattr(server.managym, "Env", refuse_reset)
        with pytest.raises(ValueError, match="current game is unchanged"):
            game.new_game(config)
        assert factory.sessions[1].closed and not factory.sessions[0].closed
        assert game.env.state_digest() == original
    finally:
        game.close("test")
        set_policy_registry(None)
    assert factory.sessions[0].closed


def test_watch_releases_both_seats_when_persistence_and_cleanup_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server.trace_store, "TRACES_DIR", tmp_path)
    registry = PolicyRegistry(providers={})
    factory = SearchFactory()
    key = registry.register(factory)
    set_policy_registry(registry)
    created = create_watch(WatchRequest(players=[f"policy:{key}", f"policy:{key}"]))
    game = server.SESSION_REGISTRY[created["id"]].game

    def refuse_save(*, end_reason: str | None = None) -> None:
        raise OSError("fixture persistence failed")

    def fail_close() -> None:
        factory.sessions[1].closed = True
        raise RuntimeError("fixture cleanup failed")

    monkeypatch.setattr(game, "_persist_attempt", refuse_save)
    monkeypatch.setattr(factory.sessions[1], "close", fail_close)
    try:
        with pytest.raises(RuntimeError, match="cleanup failed"):
            server._drop_session(created["id"], "test")
        assert all(session.closed for session in factory.sessions)
        assert game.env is None and game.watch_policies is None
        monkeypatch.undo()
        game.close("test")
    finally:
        set_policy_registry(None)


def test_first_watch_search_timeout_survives_inspection_and_retries_same_choice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server.trace_store, "TRACES_DIR", tmp_path)
    clock = Clock()
    monkeypatch.setattr(mirror_search, "time", clock)
    registry = PolicyRegistry(providers={})
    factory = SearchFactory()
    key = registry.register(factory)
    set_policy_registry(registry)
    created = create_watch(
        WatchRequest(
            players=[f"policy:{key}", f"policy:{key}"],
            search=[SearchSpec(worlds=1, depth=1), None],
        )
    )
    game = server.SESSION_REGISTRY[created["id"]].game
    factory.sessions[0].clock = clock
    assert game.env is not None
    original = game.env.state_digest()
    try:
        with pytest.raises(mirror_search.SearchError) as failure:
            step_watch(created["id"], created["token"], 0)
        assert failure.value.reason == "search_deadline_exceeded"
        with pytest.raises(HTTPException) as pending:
            inspect_watch(created["id"], 0, created["token"])
        assert pending.value.detail["code"] == "search_deadline_exceeded"
        assert pending.value.detail["revision"] == 0
        assert game.env.state_digest() == original
        assert game.trace is not None and len(game.trace.events) == 0
        factory.sessions[0].clock = None
        result = step_watch(created["id"], created["token"], 0)
        assert result["revision"] == 1
        assert game.search_retry is None
    finally:
        server._drop_session(created["id"], "test")
        set_policy_registry(None)


def test_completed_play_search_inspection_needs_no_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server.trace_store, "TRACES_DIR", tmp_path)
    registry = PolicyRegistry(providers={})
    key = registry.register(SearchFactory())
    set_policy_registry(registry)
    game = server.GameSession()
    try:
        game.new_game(
            {
                "hero_deck": {"Mountain": 8, "Gray Ogre": 8},
                "villain_deck": {"Forest": 8, "Llanowar Elves": 8},
                "villain_type": "policy",
                "opponent_sha256": key,
                "search": [None, asdict(SearchSpec(worlds=1, depth=1))],
                "seed": 42,
                "auto_pass": False,
            }
        )
        for _ in range(200):
            assert game.env is not None and game.obs is not None
            if game.obs.game_over:
                break
            game.hero_action(passive_policy(game.env, game.obs))
        else:
            pytest.fail("Fixture did not finish its ordinary native game")
        trace = load_trace(game.match_id)
        ordinal = next(
            index
            for index, event in enumerate(trace["events"])
            if event.get("bot_decision")
        )
        set_policy_registry(PolicyRegistry(providers={}))
        inspected = inspect_watch(game.match_id, ordinal, None)
        assert inspected["finished"] and inspected["source_kind"] == "play"
        assert inspected["outputs"] == trace["events"][ordinal]["bot_decision"]
        assert inspected["outputs"]["search"]["worlds"] == 1
    finally:
        game.close("test")
        set_policy_registry(None)
