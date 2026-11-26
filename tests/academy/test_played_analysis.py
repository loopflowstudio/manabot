"""Analysis: a policy reads your decisions of a played game after the fact."""

from collections.abc import Iterator, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import random
import sqlite3
import time
from typing import Any

from fastapi.testclient import TestClient
import pytest

import managym
from managym.academy import analysis as analysis_module, server, trace
from managym.academy.analysis import (
    AnalysisStop,
    analyse_game,
    analysed_events,
    analysing_policies,
    policy_admissions,
)
from managym.academy.attempts import AttemptStore, RetainedAttempt
from managym.academy.opponent import set_policy_registry
from managym.academy.recording import record_of
from managym.academy.trace import academy_actions
from managym.policy import (
    PolicyDecision,
    PolicyDescriptor,
    PolicyEvaluation,
    PolicyRegistry,
    PolicyView,
    ValueEstimate,
)
from managym.replay import walk
from managym.search import SearchConfig, search_descriptor
from tests.academy.socket_messages import settled_message
from tests.academy.test_played_replay import BOLT, play


@dataclass
class ReadingSession:
    """Prefers later offers; values a position by the chooser's life lead."""

    seat: int
    fail_after: int | None = None
    calls: int = 0

    def observe(self, view: PolicyView) -> None:
        pass

    def evaluate(self, view: PolicyView) -> PolicyEvaluation:
        count = len(view.choices)
        weights = [i + 1 for i in range(count)]
        agent, opponent = view.observation.agent, view.observation.opponent
        lead = (int(agent.life) - int(opponent.life)) / 20
        return PolicyEvaluation(
            tuple(w / sum(weights) for w in weights),
            ValueEstimate(max(-1.0, min(1.0, lead)), view.seat, "signed_outcome"),
        )

    def choose(self, view: PolicyView) -> PolicyDecision:
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise ValueError("fixture failure")
        result = self.evaluate(view)
        return PolicyDecision(
            view.choices[0], view.binding, result.probabilities, result.value
        )

    def fork(self, *, seed: int) -> "ReadingSession":
        return ReadingSession(self.seat)

    def close(self) -> None:
        pass


@dataclass
class ReadingFactory:
    name: str = "Reader"
    fail_after: int | None = None
    sessions: list[ReadingSession] = field(default_factory=list)

    def describe(self) -> PolicyDescriptor:
        return PolicyDescriptor(
            self.name,
            "fixture",
            "reader-1",
            artifacts={"actor": self.name},
            supports_search=True,
        )

    def admit(self, setup: Sequence[managym.PlayerConfig]) -> None:
        pass

    def load(self, *, seat: int, seed: int) -> ReadingSession:
        session = ReadingSession(seat, self.fail_after)
        self.sessions.append(session)
        return session


@dataclass
class RecommendedFactory(ReadingFactory):
    name: str = "Recommended"

    def describe(self) -> PolicyDescriptor:
        return replace(super().describe(), recommended=True)


@dataclass
class SearchedFactory(ReadingFactory):
    def describe(self) -> PolicyDescriptor:
        return search_descriptor(super().describe(), SearchConfig(), source="fixture")


@pytest.fixture
def registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[PolicyRegistry]:
    monkeypatch.setattr(trace, "TRACES_DIR", tmp_path)
    registry = PolicyRegistry(providers={})
    set_policy_registry(registry)
    yield registry
    set_policy_registry(None)


def finished(tmp_path: Path, **config: Any) -> RetainedAttempt:
    game = server.GameSession(tmp_path)
    game.new_game(
        {
            "hero_deck": BOLT,
            "villain_deck": BOLT,
            "villain_type": "random",
            "seed": 11,
            "auto_pass": False,
            **config,
        }
    )
    play(game, random.Random(3))
    attempt = game.attempt_store.load_attempt(game.match_id)
    assert attempt is not None and attempt.trace["end_reason"] == "game_over"
    return attempt


def runner_for(
    registry: PolicyRegistry, factory: ReadingFactory, attempt: RetainedAttempt
):
    key = registry.register(factory)
    config = trace.GameConfig(**attempt.trace["config"])
    return registry.load(key, config.to_rust(), seat=config.human_seat, seed=1)


def test_an_analysis_reads_every_one_of_your_decisions(
    tmp_path: Path, registry: PolicyRegistry
) -> None:
    attempt = finished(tmp_path)
    runner = runner_for(registry, ReadingFactory(), attempt)
    decisions = list(analyse_game(attempt, 0, runner, seed=5))
    events = attempt.trace["events"]
    assert [d.event for d in decisions] == analysed_events(attempt.trace, 0)
    assert all(events[d.event]["actor"] == "hero" for d in decisions)
    for decision in decisions:
        offers = len(events[decision.event]["actions"])
        probabilities = decision.annotation.probabilities
        assert probabilities is not None and len(probabilities) == offers
        assert decision.semantic == "signed_outcome"
        assert len(decision.after) == offers
        # Open decklists: every offer has an estimate from your seat.
        assert all(a is not None and 0 <= a <= 1 for a in decision.after)


def test_closed_decklists_never_read_the_real_hidden_hand(
    tmp_path: Path, registry: PolicyRegistry
) -> None:
    attempt = finished(tmp_path, decklists="closed")
    runner = runner_for(registry, ReadingFactory(), attempt)
    decisions = list(analyse_game(attempt, 0, runner, seed=5))
    events = attempt.trace["events"]
    handover = [
        d
        for d in decisions
        if d.event + 1 < len(events)
        and events[d.event + 1]["actor"] == "villain"
        and events[d.event]["action"] == 0
    ]
    assert handover
    # Where the opponent chooses next no world can be re-dealt: no estimate.
    assert all(d.after[0] is None for d in handover)


def test_a_policy_failure_keeps_the_decisions_before_it(
    tmp_path: Path, registry: PolicyRegistry
) -> None:
    attempt = finished(tmp_path)
    runner = runner_for(registry, ReadingFactory(fail_after=3), attempt)
    read = []
    with pytest.raises(AnalysisStop, match="fixture failure"):
        for decision in analyse_game(attempt, 0, runner, seed=5):
            read.append(decision)
    assert len(read) == 3


def wait_complete(client: TestClient, trace_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        body = client.get(f"/api/traces/{trace_id}/analyses").json()
        if body["analyses"] and body["analyses"][-1]["status"] != "running":
            return body
        time.sleep(0.05)
    raise AssertionError("analysis did not finish")


def test_a_game_against_a_scripted_opponent_defaults_to_the_recommended_policy(
    tmp_path: Path, registry: PolicyRegistry
) -> None:
    registry.register(ReadingFactory())
    recommended = registry.register(RecommendedFactory())
    attempt = finished(tmp_path)
    choices = analysing_policies(
        trace.GameConfig(**attempt.trace["config"]), policy_admissions()
    )
    assert [policy["name"] for policy in choices.policies] == ["Reader", "Recommended"]
    assert choices.default == recommended


def test_review_analyses_a_game_and_saves_it_beside_the_game(
    tmp_path: Path, registry: PolicyRegistry
) -> None:
    key = registry.register(ReadingFactory())
    finished(tmp_path, villain_type="policy", opponent_sha256=key)
    store = AttemptStore(tmp_path / "play.sqlite")
    with store.connect() as db:
        trace_id, before = db.execute("SELECT id, trace_json FROM attempts").fetchone()
    client = TestClient(server.app)
    choices = client.get(f"/api/traces/{trace_id}/analyses/policies").json()
    # The opponent's own policy opens by default.
    assert choices["default"] == key
    assert client.get(f"/api/traces/{trace_id}/analyses").json()["analyses"] == []
    started = client.post(f"/api/traces/{trace_id}/analyses", json={"policy": key})
    assert started.status_code == 200, started.text
    body = wait_complete(client, trace_id)
    analysis = body["analyses"][-1]
    assert analysis["status"] == "complete", analysis["reason"]
    assert analysis["policy"]["id"] == key
    assert [d["event"] for d in analysis["decisions"]] == choices["events"]
    assert analysis["total"] == len(choices["events"])
    # A second analysis of the same game is its own row.
    client.post(f"/api/traces/{trace_id}/analyses", json={"policy": key})
    assert len(wait_complete(client, trace_id)["analyses"]) == 2
    # The game's own row is untouched.
    with store.connect() as db:
        (after,) = db.execute("SELECT trace_json FROM attempts").fetchone()
    assert after == before
    assert "analys" not in json.dumps(
        client.get(f"/api/traces/{trace_id}").json()
    ).replace("analysis_observation", "")


def test_following_an_analysis_does_not_re_execute_the_game(
    tmp_path: Path, registry: PolicyRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    finished(tmp_path)
    store = AttemptStore(tmp_path / "play.sqlite")
    with store.connect() as db:
        (trace_id,) = db.execute("SELECT id FROM attempts").fetchone()
    store.start_analysis(trace_id, 0, {"name": "Reader"}, 3)

    def walked(*_: object) -> None:
        raise AssertionError("the poll re-executed the game")

    monkeypatch.setattr(analysis_module, "walk", walked)
    monkeypatch.setattr(server, "policy_registry", walked)
    client = TestClient(server.app)
    (running,) = client.get(f"/api/traces/{trace_id}/analyses").json()["analyses"]
    assert running["status"] == "running"
    assert client.get("/api/traces/missing/analyses").status_code == 404


def test_analysis_by_search_is_refused(
    tmp_path: Path, registry: PolicyRegistry
) -> None:
    finished(tmp_path)
    key = registry.register(SearchedFactory())
    store = AttemptStore(tmp_path / "play.sqlite")
    with store.connect() as db:
        (trace_id,) = db.execute("SELECT id FROM attempts").fetchone()
    client = TestClient(server.app)
    assert all(
        p["sha256"] != key
        for p in client.get(f"/api/traces/{trace_id}/analyses/policies").json()[
            "policies"
        ]
    )
    refused = client.post(f"/api/traces/{trace_id}/analyses", json={"policy": key})
    assert refused.status_code == 409
    assert refused.json()["detail"] == "analysis_by_search_refused"


def test_an_analysis_left_running_by_a_stopped_server_reads_as_stopped(
    tmp_path: Path, registry: PolicyRegistry
) -> None:
    finished(tmp_path)
    store = AttemptStore(tmp_path / "play.sqlite")
    with store.connect() as db:
        (trace_id,) = db.execute("SELECT id FROM attempts").fetchone()
    analysis = store.start_analysis(trace_id, 0, {"name": "Reader"}, 3)
    store.add_analysed_decision(analysis, 0, "{}")
    with store.connect() as db:
        db.execute("UPDATE analyses SET process_id='gone'")
    (read,) = store.analyses(trace_id)
    assert read["status"] == "stopped" and read["reason"]
    assert read["decisions"] == [{}]


def test_see_its_line_plays_the_policys_first_choice_on_from_a_decision(
    tmp_path: Path, registry: PolicyRegistry
) -> None:
    key = registry.register(ReadingFactory())
    attempt = finished(tmp_path, villain_type="policy", opponent_sha256=key)
    store = AttemptStore(tmp_path / "play.sqlite")
    with store.connect() as db:
        trace_id, before = db.execute("SELECT id, trace_json FROM attempts").fetchone()
    client = TestClient(server.app)
    client.post(f"/api/traces/{trace_id}/analyses", json={"policy": key})
    analysis = wait_complete(client, trace_id)["analyses"][-1]
    # A decision where the person did not play the Reader's first choice (its last offer).
    decision = next(
        d
        for d in analysis["decisions"]
        if d["played"] != len(d["annotation"]["probabilities"]) - 1
    )
    path = f"/api/traces/{trace_id}/analyses/{analysis['id']}/line"
    line = client.post(path, json={"event": decision["event"]})
    assert line.status_code == 200, line.text
    body = line.json()
    first = len(decision["annotation"]["probabilities"]) - 1
    assert body["offer"] == first
    walker = walk(record_of(attempt))
    steps = iter(walker)
    for _ in range(decision["event"]):
        next(steps)
    offers = academy_actions(walker.observation, 0)
    # The line opens with the policy's choice, not the one that was played.
    assert body["steps"][0]["actor"] == "agent"
    assert body["steps"][0]["description"] == offers[first]["description"]
    assert body["end"] in {"turn", "game_over"}
    assert {step["actor"] for step in body["steps"]} <= {"agent", "opponent"}
    # Every board is your own view; a finished game names the opponent's moves.
    assert all(
        step["projection"]["agent"]["player_index"] == 0 for step in body["steps"]
    )
    assert all(step["description"] != "Opponent's move" for step in body["steps"])
    # A line saves nothing beside the game.
    with store.connect() as db:
        (after,) = db.execute("SELECT trace_json FROM attempts").fetchone()
    assert after == before
    assert len(client.get(f"/api/traces/{trace_id}/analyses").json()["analyses"]) == 1
    unanalysed = client.post(path, json={"event": -1})
    assert unanalysed.status_code == 404


# ———————————————— Analysis while you play


@dataclass
class Delivered:
    """The pilot's side of `publish`: every live message, each delivered at once."""

    messages: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, message: dict[str, Any]) -> "Future[bool]":
        self.messages.append(message)
        done: Future[bool] = Future()
        done.set_result(True)
        return done

    def reading(self, revision: int, timeout: float = 10.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for message in self.messages:
                assert "stopped" not in message, message
                if message.get("revision") == revision:
                    return message
            time.sleep(0.005)
        raise AssertionError(f"no reading for revision {revision}")


def commit_first(game: server.GameSession, index: int) -> dict[str, Any]:
    frame = game.current_message()["frame"]
    result = game.hero_command(
        {
            "command_id": f"live.{index}",
            "match_id": frame["match_id"],
            "expected_revision": frame["revision"],
            "prompt_id": frame["prompt"]["id"],
            "offer_id": frame["offers"][0]["id"],
            "answers": [],
        }
    )
    assert result["status"] == "accepted"
    return game.current_message()["frame"]


def test_shown_analysis_reads_each_decision_live_and_marks_it(
    registry: PolicyRegistry, tmp_path: Path
) -> None:
    reader = ReadingFactory()
    key = registry.register(reader)
    game = server.GameSession(tmp_path)
    game.new_game(
        {
            "hero_deck": BOLT,
            "villain_deck": BOLT,
            "villain_type": "random",
            "seed": 11,
            "auto_pass": False,
        }
    )
    commit_first(game, 0)
    commit_first(game, 1)
    delivered = Delivered()
    with pytest.raises(server.PlayRequestError, match="cannot read"):
        game.show_analysis(True, "0" * 64, delivered)
    state = game.show_analysis(True, None, delivered)
    assert state["shown"] and state["policy"]["sha256"] == key
    message = game.current_message()
    assert message["analysis"] == {"policy": state["policy"]}
    # The line so far: the two decisions made before analysis was shown.
    first = delivered.reading(message["frame"]["revision"])
    (history,) = [m["history"] for m in delivered.messages if "history" in m]
    assert delivered.messages.index(first) > 0
    earlier = [vars(event) for event in game.trace.events]
    assert [point["event"] for point in history] == analysed_events(
        {"config": vars(game.trace.config), "events": earlier}, 0
    )
    assert all(
        point["label"] == earlier[point["event"]]["action_description"]
        and point["semantic"] == "signed_outcome"
        and 0 <= point["score"] <= 1
        and point["turn"] >= 0
        for point in history
    )
    readings = []
    for index in range(2, 6):
        frame = message["frame"]
        if frame["status"] == "game_over":
            break
        reading = delivered.reading(frame["revision"])
        assert len(reading["annotation"]["probabilities"]) == len(frame["offers"])
        readings.append(reading)
        message = {"frame": commit_first(game, index)}
    game.show_analysis(False, None, delivered)
    assert "analysis" not in game.current_message()
    commit_first(game, 9)
    play(game, random.Random(4))

    attempt = game.attempt_store.load_attempt(game.match_id)
    assert attempt is not None
    events = attempt.trace["events"]
    shown = [i for i, event in enumerate(events) if event.get("analysis_shown")]
    assert shown == [reading["event"] for reading in readings]
    assert all(events[i]["actor"] == "hero" for i in shown)
    # The live reading is the reading an analysis after the game makes.
    after = {
        decision.event: decision
        for decision in analyse_game(
            attempt, 0, runner_for(registry, reader, attempt), seed=11
        )
    }
    for reading in readings:
        saved = after[reading["event"]]
        assert reading["annotation"] == saved.annotation.model_dump(mode="json")
        assert reading["after"] == saved.after


def test_a_game_played_without_analysis_stores_no_mark(
    registry: PolicyRegistry, tmp_path: Path
) -> None:
    registry.register(ReadingFactory())
    attempt = finished(tmp_path)
    with sqlite3.connect(tmp_path / "play.sqlite") as db:
        (stored,) = db.execute("SELECT trace_json FROM attempts").fetchone()
    assert "analysis_shown" not in stored
    assert attempt.trace["events"]


def test_showing_analysis_needs_a_policy_that_can_read_the_game(
    registry: PolicyRegistry, tmp_path: Path
) -> None:
    game = server.GameSession(tmp_path)
    game.new_game(
        {"hero_deck": BOLT, "villain_deck": BOLT, "villain_type": "random", "seed": 1}
    )
    state = game.show_analysis(True, None, Delivered())
    assert not state["shown"] and state["policies"] == []
    assert state["unavailable"] == "No registered policy can read this game."
    assert "analysis" not in game.current_message()


@dataclass
class CountedFactory(ReadingFactory):
    """Counts its admissions, noting whether a game's lock was held for each."""

    admitted_under_lock: list[bool] = field(default_factory=list)

    def admit(self, setup: Sequence[managym.PlayerConfig]) -> None:
        self.admitted_under_lock.append(
            any(record.lock.locked() for record in server.LIVE_GAMES.values())
        )


def test_reviewing_a_game_again_admits_no_policy_again(
    tmp_path: Path, registry: PolicyRegistry
) -> None:
    factory = CountedFactory()
    key = registry.register(factory)
    finished(tmp_path)
    finished(tmp_path, seed=12)
    store = AttemptStore(tmp_path / "play.sqlite")
    with store.connect() as db:
        ids = [row[0] for row in db.execute("SELECT id FROM attempts")]
    client = TestClient(server.app)
    for trace_id in (*ids, ids[0]):
        body = client.get(f"/api/traces/{trace_id}/analyses/policies").json()
        assert [p["sha256"] for p in body["policies"]] == [key]
    # Two games with the same decks: one admission between them.
    assert len(factory.admitted_under_lock) == 1
    started = client.post(f"/api/traces/{ids[0]}/analyses", json={"policy": key})
    assert started.status_code == 200, started.text
    # Only loading the runner admits again.
    assert len(factory.admitted_under_lock) == 2
    wait_complete(client, ids[0])
    # A newly registered policy is admitted for the same decks.
    other = CountedFactory(name="Other")
    registry.register(other)
    body = client.get(f"/api/traces/{ids[1]}/analyses/policies").json()
    assert len(body["policies"]) == 2 and len(other.admitted_under_lock) == 1


def test_showing_analysis_admits_policies_off_the_game_lock(
    tmp_path: Path, registry: PolicyRegistry
) -> None:
    factory = CountedFactory()
    key = registry.register(factory)
    server.LIVE_GAMES.clear()
    config = {
        "hero_deck": BOLT,
        "villain_deck": BOLT,
        "villain_type": "random",
        "seed": 11,
    }
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/play") as websocket:
            websocket.send_json({"type": "new_game", "config": config})
            assert settled_message(websocket)["type"] == "observation"
            for shown in (True, False, True):
                websocket.send_json({"type": "show_analysis", "shown": shown})
                message = settled_message(websocket)
                while message["type"] != "analysis_state":
                    message = settled_message(websocket)
                assert message["shown"] is shown
                assert [p["sha256"] for p in message["policies"]] == [key]
    server.LIVE_GAMES.clear()
    # Listed once for the game, while its lock was free; the other admissions
    # are each shown analysis's own load, on its worker.
    assert factory.admitted_under_lock[0] is False
    assert len(factory.admitted_under_lock) == 1 + len(factory.sessions) == 3


def test_failed_analysis_load_releases_game_slot(
    tmp_path: Path, registry: PolicyRegistry
) -> None:
    @dataclass
    class FlakyLoadFactory(ReadingFactory):
        fail_load: bool = True

        def load(self, *, seat: int, seed: int) -> ReadingSession:
            if self.fail_load:
                raise OSError("checkpoint unreadable")
            return super().load(seat=seat, seed=seed)

    factory = FlakyLoadFactory()
    key = registry.register(factory)
    finished(tmp_path)
    store = AttemptStore(tmp_path / "play.sqlite")
    with store.connect() as db:
        (trace_id,) = db.execute("SELECT id FROM attempts").fetchone()
    client = TestClient(server.app)
    first = client.post(f"/api/traces/{trace_id}/analyses", json={"policy": key})
    assert first.status_code == 409
    assert trace_id not in server._ANALYSING
    factory.fail_load = False
    second = client.post(f"/api/traces/{trace_id}/analyses", json={"policy": key})
    assert second.status_code == 200, second.text
    wait_complete(client, trace_id)
