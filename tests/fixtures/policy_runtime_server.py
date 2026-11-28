"""Isolated browser proof for direct objects, provider-backed Watch seats and
Study's analysis by a fixture policy (`study-review.spec.ts`)."""

import json
import os
from pathlib import Path
from threading import Event
from urllib.parse import parse_qs, urlsplit

from managym.academy import server
from managym.academy.launch import serve
from managym.academy.opponent import policy_registry
from managym.policy import PolicyDecision, PolicySpec, PolicyView
from tests.academy.test_play_modes import _write_tiny_checkpoint
from tests.academy.test_played_analysis import ReadingFactory
from tests.academy.test_played_replay import BOLT
from tests.academy.test_policy_runtime import IndependentFactory, IndependentSession

_LOAD_RELEASE = [Event(), Event()]
_LOAD_BLOCKED: set[int] = set()
_CHOOSE_RELEASE = [Event(), Event()]
_CHOOSE_BLOCKED: set[int] = set()


class ControlledSession(IndependentSession):
    def choose(self, view: PolicyView) -> PolicyDecision:
        if self.seat in _CHOOSE_BLOCKED:
            if not _CHOOSE_RELEASE[self.seat].wait(30):
                raise OSError("Controlled choice barrier expired")
        return super().choose(view)


class ControlledFactory(IndependentFactory):
    def load(self, *, seat: int, seed: int) -> IndependentSession:
        if seat in _LOAD_BLOCKED:
            if not _LOAD_RELEASE[seat].wait(30):
                raise OSError("Controlled load barrier expired")
        session = ControlledSession(seat)
        self.sessions.append(session)
        return session


@server.app.post("/api/__fixtures/watch-load/{seat}/{operation}")
def control_watch_load(seat: int, operation: str) -> dict[str, bool]:
    if seat not in (0, 1):
        raise ValueError("invalid seat")
    if operation == "block":
        _LOAD_RELEASE[seat].clear()
        _LOAD_BLOCKED.add(seat)
    else:
        _LOAD_BLOCKED.discard(seat)
        _LOAD_RELEASE[seat].set()
    return {"ok": True}


@server.app.post("/api/__fixtures/watch-choose/{seat}/{operation}")
def control_watch_choose(seat: int, operation: str) -> dict[str, bool]:
    if seat not in (0, 1):
        raise ValueError("invalid seat")
    if operation == "block":
        _CHOOSE_RELEASE[seat].clear()
        _CHOOSE_BLOCKED.add(seat)
    else:
        _CHOOSE_BLOCKED.discard(seat)
        _CHOOSE_RELEASE[seat].set()
    return {"ok": True}


def short_game(url: str) -> str:
    """The launch at `url` with Mountain and Lightning Bolt decks, whose games
    end in a few turns."""
    (launch,) = parse_qs(urlsplit(url).query)["launch"]
    config = server.LAUNCHES[launch].config
    config.update(hero_deck=dict(BOLT), villain_deck=dict(BOLT))
    server._parse_game_config(config)
    return url


def main() -> None:
    output = Path(os.environ["ACADEMY_POLICY_PROOF"])
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = output.parent / "baseline.pt"
    _write_tiny_checkpoint(checkpoint)
    refused = IndependentFactory("Refused configuration")
    with serve(
        port=int(os.environ["ACADEMY_POLICY_PORT"]),
        traces_dir=output.parent / "traces",
        policies={
            "independent": IndependentFactory(
                "Independent/long-run/tracker/source/snapshot"
            ),
            "refused-config": refused,
            "controlled": ControlledFactory("Controlled loading"),
            "baseline": PolicySpec("manabot", str(checkpoint)),
            # Gives probabilities and a signed-outcome value: Study's analysis.
            "reader": ReadingFactory(),
        },
    ) as academy:
        refused_url = academy.new_game(opponent="refused-config", seed=7).url
        refused.runner = "changed-after-launch"
        output.write_text(
            json.dumps(
                {
                    "refused": refused_url,
                    "play": academy.new_game(opponent="independent", seed=7).url,
                    "study": short_game(
                        academy.new_game(opponent="reader", seed=5).url
                    ),
                    # A launch starts one game, so each test that plays one
                    # to its end has its own.
                    "study_shown": short_game(
                        academy.new_game(opponent="reader", seed=5).url
                    ),
                    "watch": academy.watch(players=("independent", "baseline")).url,
                    "ids": {
                        name: policy_registry().descriptor(name).id
                        for name in (
                            "independent",
                            "baseline",
                            "refused-config",
                            "controlled",
                        )
                    },
                }
            )
        )
        print("ACADEMY_POLICY_READY", flush=True)
        academy.wait()


if __name__ == "__main__":
    main()
