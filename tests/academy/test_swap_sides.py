"""Mid-game seat swapping on a live Academy game."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
import random
import time
from typing import Any

from fastapi.testclient import TestClient
import pytest

import managym
from managym.academy import server, trace as trace_module
from managym.academy.analysis import (
    agent_line,
    analyse_game,
    analysed_events,
)
from managym.academy.attempts import participant_id
from managym.academy.numbers import game_numbers
from managym.academy.opponent import set_policy_registry
from managym.academy.recording import Recording, ReplayedViews
from managym.academy.trace import GameConfig, human_seat_at
from managym.policy import (
    PolicyDecision,
    PolicyDescriptor,
    PolicyRegistry,
    PolicyView,
    ValueEstimate,
)
from tests.academy.socket_messages import settled_message
from tests.academy.test_played_analysis import ReadingFactory
from tests.academy.test_played_replay import BOLT, play

TOKEN = "s" * 36
OTHER_BOLT = {"Mountain": 8, "Lightning Bolt": 12}


@dataclass
class RecurrentSession:
    """Recurrent fixture session that records every observed and chosen step."""

    seat: int
    observed_steps: list[int] = field(default_factory=list)
    chosen_steps: list[int] = field(default_factory=list)
    closed: bool = False

    def observe(self, view: PolicyView) -> None:
        assert view.seat == self.seat
        self.observed_steps.append(int(view.observation.turn.turn_number))

    def choose(self, view: PolicyView) -> PolicyDecision:
        assert view.seat == self.seat
        self.chosen_steps.append(int(view.observation.turn.turn_number))
        count = len(view.choices)
        probs = tuple(1.0 / count for _ in range(count))
        return PolicyDecision(
            view.choices[0],
            view.binding,
            probs,
            ValueEstimate(0.0, view.seat, "signed_outcome"),
        )

    def close(self) -> None:
        self.closed = True


@dataclass
class RecurrentFactory:
    name: str = "RecurrentFixture"
    sessions: list[RecurrentSession] = field(default_factory=list)

    def describe(self) -> PolicyDescriptor:
        return PolicyDescriptor(
            self.name,
            "fixture",
            "recurrent-1",
            artifacts={"actor": self.name},
            recurrent=True,
        )

    def admit(self, setup: Sequence[managym.PlayerConfig]) -> None:
        pass

    def load(self, *, seat: int, seed: int) -> RecurrentSession:
        session = RecurrentSession(seat)
        self.sessions.append(session)
        return session


@dataclass
class SeatOneOnlyFactory(ReadingFactory):
    """Refuses loading in seat 0 so swap-to-seat-1 refusal leaves the game intact."""

    name: str = "SeatOneOnly"

    def load(self, *, seat: int, seed: int):
        if seat == 0:
            raise ValueError("policy only supports seat 1")
        return super().load(seat=seat, seed=seed)


@pytest.fixture
def registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[PolicyRegistry]:
    monkeypatch.setattr(trace_module, "TRACES_DIR", tmp_path)
    server.SESSION_REGISTRY.clear()
    reg = PolicyRegistry(providers={})
    set_policy_registry(reg)
    yield reg
    set_policy_registry(None)
    server.SESSION_REGISTRY.clear()


def _play_steps(game: server.GameSession, rng: random.Random, count: int) -> None:
    """Play up to `count` human decisions via the Command path."""
    for i in range(count):
        if game.obs is None or game.obs.game_over:
            return
        prompt = game._publish_current_prompt()
        if prompt is None:
            return
        offer = rng.choice(prompt.offers)["id"]
        outcome = game.hero_command(
            {
                "command_id": f"step.{game.revision}.{i}",
                "match_id": game.match_id,
                "expected_revision": prompt.revision,
                "prompt_id": prompt.prompt_id,
                "offer_id": offer,
                "answers": [],
            }
        )
        assert outcome["status"] == "accepted"


def _make_publish(delivered: list[dict[str, Any]]):
    def publish(message: dict[str, Any]) -> Future[bool]:
        delivered.append(message)
        fut: Future[bool] = Future()
        fut.set_result(True)
        return fut

    return publish


def _wait_for_reading(
    delivered: list[dict[str, Any]], revision: int, timeout: float = 5.0
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for msg in delivered:
            if msg.get("revision") == revision:
                return msg
        time.sleep(0.01)
    raise AssertionError(f"Live analysis reading for revision {revision} did not arrive")


def test_swap_sides_end_to_end_with_auto_mana_analysis_and_review(
    tmp_path: Path, registry: PolicyRegistry
) -> None:
    factory = ReadingFactory(name="Reader")
    policy_key = registry.register(factory)
    game = server.GameSession(tmp_path)
    game.attempt_owner = participant_id(TOKEN)

    start_msg = game.new_game(
        {
            "hero_deck": BOLT,
            "villain_deck": OTHER_BOLT,
            "villain_type": "policy",
            "opponent_sha256": policy_key,
            "seed": 19,
            "mana_payment": "auto",
            "auto_pass": True,
            "stops": {"my": ["main1", "main2"], "opponent": ["end_step"]},
        }
    )
    assert game.trace is not None and game.env is not None and game.obs is not None
    original_config = game.trace.config.to_record()
    original_setup = game.trace.config.to_rust()
    original_players = [
        (p.name, dict(p.decklist), dict(p.sideboard), p.world) for p in original_setup
    ]
    original_match_id = game.match_id

    # Turn on live analysis before swapping so at least one pre-swap decision has analysis_shown=True.
    delivered: list[dict[str, Any]] = []
    publish = _make_publish(delivered)
    analysis_state = game.show_analysis(True, policy_key, publish)
    assert analysis_state["shown"] is True
    assert game.published_prompt is not None
    _wait_for_reading(delivered, game.revision)

    rng = random.Random(7)
    _play_steps(game, rng, 3)
    assert not game.obs.game_over

    # Snapshot pre-swap state & events.
    pre_swap_revision = game.revision
    pre_swap_prompt = game._publish_current_prompt()
    assert pre_swap_prompt is not None
    pre_swap_event_count = len(game.trace.events)
    assert pre_swap_event_count > 0
    assert any(e.analysis_shown for e in game.trace.events)
    pre_swap_events_snapshot = [
        (e.actor, e.bot_decision, e.committed_at, e.analysis_shown)
        for e in game.trace.events
    ]
    pre_swap_life = {
        int(game.obs.agent.player_index): int(game.obs.agent.life),
        int(game.obs.opponent.player_index): int(game.obs.opponent.life),
    }
    seat1_hand_before = [
        card.characteristics.name
        for card in game.env.observation_for_player(1).agent_cards
        if int(card.zone) == int(managym.ZoneEnum.HAND)
    ]

    # Swap sides!
    swap_msg = game.swap_sides(
        request_id="swap-1",
        match_id=original_match_id,
        expected_revision=pre_swap_revision,
    )

    assert game.match_id == original_match_id
    assert game.human_seat == 1
    assert game.revision > pre_swap_revision
    assert game.trace.seat_swaps == [pre_swap_event_count]
    assert game.trace.config.to_record() == original_config
    after_setup = game.trace.config.to_rust()
    assert [
        (p.name, dict(p.decklist), dict(p.sideboard), p.world) for p in after_setup
    ] == original_players
    assert after_setup.mana_payment == original_setup.mana_payment
    assert after_setup.mana_floats == original_setup.mana_floats
    assert after_setup.decklists == original_setup.decklists
    assert swap_msg["frame"]["match_id"] == original_match_id
    assert swap_msg["frame"]["projection"]["agent"]["player_index"] == 1
    assert swap_msg["deck_names"] == {
        "hero": start_msg["deck_names"]["villain"],
        "villain": start_msg["deck_names"]["hero"],
    }
    assert any(
        entry["actor"] == "system" and entry["text"] == "Swapped sides"
        for entry in swap_msg.get("log", [])
    )
    # Life totals did not jump on the swap boundary (before any immediate bot moves if life changed,
    # wait: if bot moved immediately, check seat 1's hand at least has cards and life is consistent).
    if len(game.trace.events) == pre_swap_event_count:
        assert swap_msg["frame"]["projection"]["agent"]["life"] == pre_swap_life[1]
        assert swap_msg["frame"]["projection"]["opponent"]["life"] == pre_swap_life[0]
        assert sorted(
            c["name"] for c in swap_msg["frame"]["projection"]["agent"]["hand"]
        ) == sorted(seat1_hand_before)

    # Pre-swap events kept exact actor, bot_decision, committed_at, and analysis_shown.
    assert [
        (e.actor, e.bot_decision, e.committed_at, e.analysis_shown)
        for e in game.trace.events[:pre_swap_event_count]
    ] == pre_swap_events_snapshot

    # Stale Command bound to the pre-swap revision is rejected.
    stale = game.hero_command(
        {
            "command_id": "stale-after-swap",
            "match_id": original_match_id,
            "expected_revision": pre_swap_prompt.revision,
            "prompt_id": pre_swap_prompt.prompt_id,
            "offer_id": pre_swap_prompt.offers[0]["id"],
            "answers": [],
        }
    )
    assert stale["status"] == "rejected"
    assert stale["rejection"]["code"] == "stale_revision"

    # Duplicate request_id with same bindings returns current message without swapping back.
    dup = game.swap_sides(
        request_id="swap-1",
        match_id=original_match_id,
        expected_revision=pre_swap_revision,
    )
    assert game.human_seat == 1
    assert game.trace.seat_swaps == [pre_swap_event_count]
    assert dup["frame"]["revision"] == game.revision

    # Reusing the same request_id with different bindings raises ValueError.
    with pytest.raises(ValueError, match="already used"):
        game.swap_sides(
            request_id="swap-1",
            match_id=original_match_id,
            expected_revision=game.revision,
        )

    # Live analysis followed the person to seat 1.
    if not game.obs.game_over:
        assert game._live_analysis is not None
        assert swap_msg.get("analysis") == {"policy": game._live_analysis.policy}
        _wait_for_reading(delivered, game.revision)

    # Finish the game.
    play(game, random.Random(13))
    assert game.obs is not None and game.obs.game_over

    attempt = game.attempt_store.load_attempt(original_match_id)
    assert attempt is not None
    assert attempt.trace["end_reason"] == "game_over"
    assert attempt.trace["seat_swaps"] == [pre_swap_event_count]
    assert human_seat_at(attempt.trace, 0) == 0
    assert human_seat_at(attempt.trace, pre_swap_event_count) == 1
    assert human_seat_at(attempt.trace, None) == 1

    # ReplayedViews re-executes completely across the swap.
    views = ReplayedViews(attempt, 0)
    assert views.board(len(attempt.trace["events"])) is not None
    assert views.stop is not None and views.stop.status == "complete"

    # game_numbers succeeds.
    numbers = game_numbers(attempt)
    assert numbers is not None and numbers.complete is True
    assert numbers.analysis_shown >= 1

    # Recording includes client decisions from BOTH seats.
    recording = Recording(original_match_id, attempt.trace, attempt)
    choice_viewers = {
        pos.decision.viewer
        for pos in recording.positions
        if pos.kind == "choice" and pos.decision is not None
    }
    assert choice_viewers == {0, 1}
    # Reading every position succeeds (including reveal=True for both hands).
    for pos in recording.positions:
        read = recording.read(pos.key, reveal=True)
        assert read["revealed"] is True

    # analyse_game reads all deliberate human decisions across both seats.
    config = GameConfig(**attempt.trace["config"])
    runners = {
        seat: registry.load(policy_key, config.to_rust(), seat=seat, seed=1)
        for seat in (0, 1)
    }
    try:
        analysed = list(analyse_game(attempt, None, runners, seed=5))
    finally:
        for r in runners.values():
            r.close()
    expected_events = analysed_events(attempt.trace)
    assert [d.event for d in analysed] == expected_events
    assert any(d.event < pre_swap_event_count for d in analysed)
    assert any(d.event >= pre_swap_event_count for d in analysed)

    # agent_line works both before and after the swap.
    pre_decision = next(d for d in analysed if d.event < pre_swap_event_count)
    post_decision = next(d for d in analysed if d.event >= pre_swap_event_count)
    for decision, expected_seat in ((pre_decision, 0), (post_decision, 1)):
        r = registry.load(policy_key, config.to_rust(), seat=expected_seat, seed=2)
        opp = server.build_villain_policy(config, seat=1 - expected_seat)
        try:
            line = agent_line(attempt, decision, expected_seat, r, opp)
            assert line.event == decision.event
            assert len(line.steps) > 0
        finally:
            server.close_policies((r, opp))

    # AttemptStore.record counts the game ONCE under the final seat (seat 1).
    owners = {game.attempt_owner}
    summaries = game.attempt_store.summaries(owners)
    assert len(summaries) == 1
    assert summaries[0]["id"] == original_match_id
    human_row = next(p for p in summaries[0]["players"] if p["kind"] == "human")
    bot_row = next(p for p in summaries[0]["players"] if p["kind"] == "bot")
    assert human_row["seat"] == 1
    assert bot_row["seat"] == 0
    rec = game.attempt_store.record(owners)
    assert len(rec) == 1
    assert rec[0]["games"] == 1
    winner = attempt.trace["winner"]
    assert rec[0]["wins"] == (1 if winner == 1 else 0)
    assert rec[0]["losses"] == (1 if winner == 0 else 0)

    # HTTP recording index exposes seat_swaps and starting_human_seat; Study decision restore
    # authorizes pre-swap and post-swap human decisions and refuses bot decisions.
    client = TestClient(server.app)
    headers = {"x-etude-participant-token": TOKEN}
    index_res = client.get(f"/api/traces/{original_match_id}/recording", headers=headers)
    assert index_res.status_code == 200
    index_body = index_res.json()
    assert index_body["starting_human_seat"] == 0
    assert len(index_body["seat_swaps"]) == 1
    assert index_body["seat_swaps"][0]["event"] == pre_swap_event_count

    pre_choice = next(
        pos for pos in recording.positions
        if pos.kind == "choice" and pos.decision and pos.decision.viewer == 0
    )
    post_choice = next(
        pos for pos in recording.positions
        if pos.kind == "choice" and pos.decision and pos.decision.viewer == 1
    )
    assert (
        client.get(
            f"/api/traces/{original_match_id}/decisions/{pre_choice.key}",
            headers=headers,
        ).status_code
        == 200
    )
    assert (
        client.get(
            f"/api/traces/{original_match_id}/decisions/{post_choice.key}",
            headers=headers,
        ).status_code
        == 200
    )
    # Bot decision is refused.
    bot_row_decision = next(
        row for row in recording.replay.decisions if row.source.value == "policy"
    )
    from managym.academy.replay_index import DecisionAddressV2

    bot_addr = DecisionAddressV2.from_decision(
        recording.replay, bot_row_decision
    ).serialize()
    assert (
        client.get(
            f"/api/traces/{original_match_id}/decisions/{bot_addr}",
            headers=headers,
        ).status_code
        == 404
    )

    # Play on from a post-swap decision lands in seat 1!
    play_on_game = server.GameSession(tmp_path)
    play_on_game.attempt_owner = game.attempt_owner
    play_on_msg = play_on_game.new_game(
        {"played_from": {"trace_id": original_match_id, "event": post_decision.event}}
    )
    assert play_on_game.human_seat == 1
    assert play_on_game.trace is not None
    assert play_on_game.trace.seat_swaps == [pre_swap_event_count]
    assert play_on_msg["frame"]["projection"]["agent"]["player_index"] == 1


def test_recurrent_policy_warms_up_from_step_zero_before_choosing(
    tmp_path: Path, registry: PolicyRegistry
) -> None:
    factory = RecurrentFactory()
    key = registry.register(factory)
    game = server.GameSession(tmp_path)
    game.new_game(
        {
            "hero_deck": BOLT,
            "villain_deck": OTHER_BOLT,
            "villain_type": "policy",
            "opponent_sha256": key,
            "seed": 23,
            "auto_pass": False,
        }
    )
    _play_steps(game, random.Random(2), 3)
    assert game.trace is not None and len(game.trace.events) >= 3
    steps_before_swap = len(game.trace.events)
    assert len(factory.sessions) == 1
    old_session = factory.sessions[0]

    game.swap_sides(
        request_id="swap-recurrent",
        match_id=game.match_id,
        expected_revision=game.revision,
    )

    assert old_session.closed is True
    assert len(factory.sessions) == 2
    shadow = factory.sessions[1]
    assert shadow.seat == 0
    # Shadow policy observed the initial state plus every step up to the swap before taking over.
    assert len(shadow.observed_steps) >= steps_before_swap + 1
    assert len(shadow.chosen_steps) >= 3


def test_swap_refusal_leaves_live_game_untouched_and_accepting_commands(
    tmp_path: Path, registry: PolicyRegistry
) -> None:
    factory = SeatOneOnlyFactory()
    key = registry.register(factory)
    game = server.GameSession(tmp_path)
    game.new_game(
        {
            "hero_deck": BOLT,
            "villain_deck": OTHER_BOLT,
            "villain_type": "policy",
            "opponent_sha256": key,
            "seed": 31,
            "auto_pass": False,
        }
    )
    _play_steps(game, random.Random(4), 2)
    assert game.trace is not None
    revision_before = game.revision
    prompt_before = game._publish_current_prompt()
    assert prompt_before is not None

    with pytest.raises(ValueError, match="swap_unavailable"):
        game.swap_sides(
            request_id="swap-refused",
            match_id=game.match_id,
            expected_revision=revision_before,
        )

    assert game.human_seat == 0
    assert game.revision == revision_before
    assert game.trace.seat_swaps == []
    # The live game still accepts a Command at the exact same revision and prompt.
    outcome = game.hero_command(
        {
            "command_id": "after-refused-swap",
            "match_id": game.match_id,
            "expected_revision": prompt_before.revision,
            "prompt_id": prompt_before.prompt_id,
            "offer_id": prompt_before.offers[0]["id"],
            "answers": [],
        }
    )
    assert outcome["status"] == "accepted"


def test_double_swap_produces_one_attempt_reexecutes_and_counts_under_seat_zero(
    tmp_path: Path, registry: PolicyRegistry
) -> None:
    game = server.GameSession(tmp_path)
    game.attempt_owner = participant_id(TOKEN)
    game.new_game(
        {
            "hero_deck": BOLT,
            "villain_deck": OTHER_BOLT,
            "villain_type": "random",
            "seed": 41,
            "auto_pass": False,
        }
    )
    rng = random.Random(9)
    _play_steps(game, rng, 2)
    assert game.trace is not None
    first_swap_at = len(game.trace.events)
    game.swap_sides(
        request_id="double-1",
        match_id=game.match_id,
        expected_revision=game.revision,
    )
    assert game.human_seat == 1
    _play_steps(game, rng, 2)
    second_swap_at = len(game.trace.events)
    game.swap_sides(
        request_id="double-2",
        match_id=game.match_id,
        expected_revision=game.revision,
    )
    assert game.human_seat == 0
    assert game.trace.seat_swaps == [first_swap_at, second_swap_at]

    play(game, random.Random(10))
    attempt = game.attempt_store.load_attempt(game.match_id)
    assert attempt is not None
    assert attempt.trace["seat_swaps"] == [first_swap_at, second_swap_at]

    views = ReplayedViews(attempt, 0)
    assert views.board(len(attempt.trace["events"])) is not None
    assert views.stop is not None and views.stop.status == "complete"

    owners = {game.attempt_owner}
    summaries = game.attempt_store.summaries(owners)
    assert len(summaries) == 1
    human_row = next(p for p in summaries[0]["players"] if p["kind"] == "human")
    assert human_row["seat"] == 0
    rec = game.attempt_store.record(owners)
    assert len(rec) == 1
    assert rec[0]["games"] == 1
    assert rec[0]["wins"] == (1 if attempt.trace["winner"] == 0 else 0)


def test_played_from_game_that_swaps_sides_stays_out_of_my_record(
    tmp_path: Path, registry: PolicyRegistry
) -> None:
    game = server.GameSession(tmp_path)
    game.attempt_owner = participant_id(TOKEN)
    game.new_game(
        {
            "hero_deck": BOLT,
            "villain_deck": OTHER_BOLT,
            "villain_type": "random",
            "seed": 53,
            "auto_pass": False,
        }
    )
    play(game, random.Random(11))
    source_id = game.match_id
    owners = {game.attempt_owner}
    before_record = game.attempt_store.record(owners)
    assert len(before_record) == 1

    source = game.attempt_store.load(source_id)
    assert source is not None
    first_hero_event = next(
        i for i, e in enumerate(source["events"]) if e["actor"] == "hero" and not e.get("auto")
    )

    game.new_game({"played_from": {"trace_id": source_id, "event": first_hero_event}})
    _play_steps(game, random.Random(12), 1)
    assert game.obs is not None and not game.obs.game_over
    game.swap_sides(
        request_id="played-from-swap",
        match_id=game.match_id,
        expected_revision=game.revision,
    )
    play(game, random.Random(14))

    assert game.attempt_store.record(owners) == before_record


def test_websocket_swap_sides_broadcasts_and_forbids_watcher(
    tmp_path: Path, registry: PolicyRegistry
) -> None:
    client = TestClient(server.app)
    with client.websocket_connect("/ws/play") as pilot_ws:
        pilot_ws.send_json(
            {
                "type": "new_game",
                "config": {
                    "hero_deck": BOLT,
                    "villain_deck": OTHER_BOLT,
                    "villain_type": "passive",
                    "seed": 61,
                    "auto_pass": False,
                },
            }
        )
        started = settled_message(pilot_ws)
        assert started["type"] == "observation"
        table = started["table"]
        assert table["access"]["identity"]["rules_viewer"] == 0
        invite_fragment = table["watcher_invite"]
        assert invite_fragment
        invite_token = invite_fragment.split("watch=")[1]

        with client.websocket_connect("/ws/play") as watcher_ws:
            watcher_ws.send_json(
                {
                    "type": "join_table",
                    "table_id": table["table_id"],
                    "invite_token": invite_token,
                }
            )
            watcher_joined = settled_message(watcher_ws)
            # Pilot also receives presence broadcast.
            pilot_presence = settled_message(pilot_ws)
            assert pilot_presence["type"] == "observation"

            # Watcher is forbidden from swapping sides.
            watcher_ws.send_json(
                {
                    "type": "swap_sides",
                    "grant_revision": watcher_joined["table"]["access"]["grant_revision"],
                    "request_id": "watcher-swap",
                    "match_id": started["frame"]["match_id"],
                    "expected_revision": started["frame"]["revision"],
                }
            )
            denied = settled_message(watcher_ws)
            assert denied["type"] == "control_error"
            assert denied["code"] == "forbidden"

            # Pilot swaps sides; both pilot and watcher receive updated rules_viewer == 1.
            pilot_ws.send_json(
                {
                    "type": "swap_sides",
                    "grant_revision": table["access"]["grant_revision"],
                    "request_id": "pilot-swap",
                    "match_id": started["frame"]["match_id"],
                    "expected_revision": started["frame"]["revision"],
                }
            )
            swapped_pilot = settled_message(pilot_ws)
            swapped_watcher = settled_message(watcher_ws)
            assert swapped_pilot["type"] == "observation"
            assert swapped_pilot["table"]["access"]["identity"]["rules_viewer"] == 1
            assert swapped_pilot["frame"]["projection"]["agent"]["player_index"] == 1
            assert swapped_watcher["type"] == "observation"
            assert swapped_watcher["table"]["access"]["identity"]["rules_viewer"] == 1
