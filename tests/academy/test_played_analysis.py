"""Analysis: a policy reads your decisions of a played game after the fact."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
import random
import sqlite3
import time
from typing import Any

from fastapi.testclient import TestClient
import pytest

import managym
from managym.academy import server, trace
from managym.academy.analysis import AnalysisStop, analyse_game, analysed_events
from managym.academy.attempts import AttemptStore, RetainedAttempt, participant_id
from managym.academy.opponent import set_policy_registry
from managym.policy import (
    PolicyDecision,
    PolicyDescriptor,
    PolicyEvaluation,
    PolicyRegistry,
    PolicyView,
    ValueEstimate,
)
from managym.search import SearchConfig, search_descriptor
from tests.academy.test_played_replay import BOLT, play

TOKEN = "t" * 36


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
class SearchedFactory(ReadingFactory):
    def describe(self) -> PolicyDescriptor:
        return search_descriptor(super().describe(), SearchConfig(), source="fixture")


@pytest.fixture
def registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[PolicyRegistry]:
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
    with sqlite3.connect(tmp_path / "play.sqlite") as db:
        db.execute("UPDATE attempts SET participant_id=?", (participant_id(TOKEN),))
    attempt = game.attempt_store.load_attempt(game.match_id)
    assert attempt is not None and attempt.trace["end_reason"] == "game_over"
    return attempt


def runner_for(registry: PolicyRegistry, factory: ReadingFactory, attempt: RetainedAttempt):
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
    assert [d.event for d in decisions] == analysed_events(attempt, 0)
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
        d for d in decisions if d.event + 1 < len(events)
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


def headers() -> dict[str, str]:
    return {"x-etude-participant-token": TOKEN}


def wait_complete(client: TestClient, trace_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        body = client.get(f"/api/traces/{trace_id}/analyses").json()
        if body["analyses"] and body["analyses"][-1]["status"] != "running":
            return body
        time.sleep(0.05)
    raise AssertionError("analysis did not finish")


def test_review_analyses_a_game_and_saves_it_beside_the_game(
    tmp_path: Path, registry: PolicyRegistry
) -> None:
    key = registry.register(ReadingFactory())
    attempt = finished(
        tmp_path, villain_type="policy", opponent_sha256=key
    )
    store = AttemptStore(tmp_path / "play.sqlite")
    with store.connect() as db:
        trace_id, before = db.execute("SELECT id, trace_json FROM attempts").fetchone()
    client = TestClient(server.app)
    listed = client.get(f"/api/traces/{trace_id}/analyses").json()
    # The opponent's own policy opens by default.
    assert listed["default"] == key and listed["analyses"] == []
    assert client.post(f"/api/traces/{trace_id}/analyses", json={"policy": key}).status_code == 403
    started = client.post(
        f"/api/traces/{trace_id}/analyses", json={"policy": key}, headers=headers()
    )
    assert started.status_code == 200, started.text
    body = wait_complete(client, trace_id)
    analysis = body["analyses"][-1]
    assert analysis["status"] == "complete", analysis["reason"]
    assert analysis["policy"]["id"] == key
    assert [d["event"] for d in analysis["decisions"]] == body["events"]
    assert analysis["total"] == len(body["events"])
    # A second analysis of the same game is its own row.
    client.post(f"/api/traces/{trace_id}/analyses", json={"policy": key}, headers=headers())
    assert len(wait_complete(client, trace_id)["analyses"]) == 2
    # The game's own row is untouched.
    with store.connect() as db:
        (after,) = db.execute("SELECT trace_json FROM attempts").fetchone()
    assert after == before
    assert "analys" not in json.dumps(
        client.get(f"/api/traces/{trace_id}", headers=headers()).json()
    ).replace("analysis_observation", "")


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
        for p in client.get(f"/api/traces/{trace_id}/analyses").json()["policies"]
    )
    refused = client.post(
        f"/api/traces/{trace_id}/analyses", json={"policy": key}, headers=headers()
    )
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
