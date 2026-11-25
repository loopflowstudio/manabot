"""Probe (MTG-482): can a live game's moves be replayed into a second session
with the person in the other seat, landing on the same native position?

Run from a checkout with a built engine:
    .venv/bin/python scratch/probe/swap_replay.py
"""

from pathlib import Path
import random
import re
import tempfile
import time
from typing import Any

from managym.academy import server
from managym.academy.recording import ReplayedViews
from managym.academy.trace import PlayedFrom
from managym.experience_projection import decision_surface_view
from tests.academy.test_played_replay import BOLT, play

PAIRS = ("deck", "deck_name", "sideboard", "source", "deck_pool")


def flipped(source: dict[str, Any]) -> dict[str, Any]:
    config = dict(source["config"])
    for name in PAIRS:
        hero, villain = f"hero_{name}", f"villain_{name}"
        if hero in config or villain in config:
            config[hero], config[villain] = config.get(villain), config.get(hero)
    config = {k: v for k, v in config.items() if v is not None or k == "seed"}
    config["human_seat"] = 1 - config.get("human_seat", 0)
    # Action descriptions name players by role ("Hero"/"Villain"), so they flip too.
    names = {"Hero": "Villain", "Villain": "Hero"}
    events = [
        {
            **event,
            "actor": "villain" if event["actor"] == "hero" else "hero",
            "action_description": re.sub(
                "Hero|Villain", lambda m: names[m[0]], event["action_description"]
            ),
        }
        for event in source["events"]
    ]
    return {**source, "config": config, "events": events}


class Swapped(server.GameSession):
    source: dict[str, Any]

    def _played_from_source(self, raw: Any, player_token: Any):  # noqa: ANN201
        return PlayedFrom(raw["trace_id"], raw["event"]), self.source

    def _play_recorded_prefix(self, events: list[dict[str, Any]], upto: int) -> None:
        self.upto = upto
        try:
            super()._play_recorded_prefix(events, upto)
        except ValueError as error:
            # The whole game replays; whoever chooses next, the position stands.
            if "decision was not reached" not in str(error):
                raise

    def _auto_play_villain(self) -> None:
        # Past the record the bot in the person's old seat chooses for itself.
        assert self.trace is not None
        hook = self._villain_offer_policy
        while hook is not None and len(self.trace.events) < self.upto:
            before = len(self.trace.events)
            self._one_recorded_villain_step()
            if len(self.trace.events) == before:
                return
        self._villain_offer_policy = None
        try:
            super()._auto_play_villain()
        finally:
            self._villain_offer_policy = hook

    def _one_recorded_villain_step(self) -> None:
        assert self.obs is not None
        if self.obs.game_over or server._is_hero_turn(self.obs, self.human_seat):
            return
        context = self._build_decision_context(self.obs, viewer=1 - self.human_seat)
        offer = int(self._villain_offer_policy(context))
        command = self._authority_command(context, offer, "policy")
        self._apply_bound_command(context, command, source="policy")


def run(commands: int, **config: Any) -> None:
    with tempfile.TemporaryDirectory() as directory:
        live = server.GameSession(Path(directory))
        live.new_game(
            {
                "hero_deck": BOLT,
                "villain_deck": {"Mountain": 12, "Lightning Bolt": 8},
                "villain_type": "random",
                "seed": 11,
                **config,
            }
        )
        play(live, random.Random(3), commands)
        assert live.obs is not None and not live.obs.game_over
        source = live.attempt_store.load(live.match_id)
        assert source is not None
        count = len(source["events"])
        swapped = Swapped(Path(directory))
        swapped.attempt_owner = live.attempt_owner
        swapped.source = flipped(source)
        start = time.perf_counter()
        swapped.new_game({"played_from": {"trace_id": live.match_id, "event": count}})
        elapsed = time.perf_counter() - start
        # The bot in the old seat has already moved on from the shared position,
        # so compare the recorded moves and that the new game plays to its end.
        trace = swapped.trace
        assert trace is not None
        same = all(
            mine.action == int(theirs["action"])
            for mine, theirs in zip(trace.events[:count], source["events"])
        )
        assert swapped.human_seat == 1 - live.human_seat
        human_view = decision_surface_view(swapped.env, swapped.human_seat)
        play(swapped, random.Random(5))
        attempt = swapped.attempt_store.load_attempt(swapped.match_id)
        assert attempt is not None
        views = ReplayedViews(attempt, swapped.human_seat)
        views.board(len(attempt.trace["events"]))
        print(
            f"{config or 'default'}: {count} moves replayed in {elapsed * 1000:.0f} ms;"
            f" same moves={same}; person now seat {swapped.human_seat}"
            f" (hand {len(human_view['agent']['hand'])});"
            f" ended {attempt.trace['end_reason']};"
            f" re-executes: {views.stop.status if views.stop else None}"
        )


if __name__ == "__main__":
    run(30, auto_pass=False)
    run(12, auto_pass=True)
    run(12, auto_pass=True, human_seat=1, decklists="closed")
    run(12, auto_pass=True, mana_payment="auto")
