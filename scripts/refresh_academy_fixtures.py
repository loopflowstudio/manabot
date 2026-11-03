"""Refresh the w17 Academy fixtures after a change to state digests or ids.

Each fixture pins digests of positions reached by recorded Commands (offer ids
and labels) from an ordinary GameSession. This replays those Commands through
the live server and rewrites only what the runtime derives: digests, revisions,
manifests and the retained frames. A label that no longer matches is a rules or
offer change, not something to refresh, and stops the script.

Run from the repository root after rebuilding the extension:

    uv run --extra dev python scripts/refresh_academy_fixtures.py

Refreshes `academy/fixtures/{learn,precision,precision-mana}-demo-w17.json`,
`choice-navigation-w17.json` and `trigger-order-w17.json`. The conformance and
parity receipts have their own writers (see `conformance/README.md`).
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any

from academy.play_demo import load_play_demo
from academy.server import GameSession
from manabot.verify.trigger_order import root

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "academy" / "fixtures"
DEMOS = {
    "learn": "learn-demo-w17.json",
    "precision": "precision-demo-w17.json",
    "precision-mana": "precision-mana-demo-w17.json",
}


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def _command(
    frame: dict[str, Any], offer_id: int, label: str, ordinal: int
) -> dict[str, Any]:
    offer = next(o for o in frame["offers"] if o["id"] == offer_id)
    if offer["label"] != label:
        raise ValueError(
            f"Command {ordinal}: offer {offer_id} is {offer['label']!r}, fixture says {label!r}"
        )
    return {
        "command_id": f"refresh.{ordinal}",
        "match_id": frame["match_id"],
        "expected_revision": frame["revision"],
        "prompt_id": frame["prompt"]["id"],
        "offer_id": offer["id"],
        "answers": [],
    }


def _replay(
    config: dict[str, Any], prefix: list[dict[str, Any]], trace_dir: Path
) -> tuple[GameSession, dict[str, Any], list[str]]:
    """Play `prefix` from a fresh game; returns the session, its last frame and
    the state digest before each Command."""
    session = GameSession(trace_dir=trace_dir)
    frame = session.new_game(config)["frame"]
    digests: list[str] = []
    for ordinal, step in enumerate(prefix):
        assert session.env is not None
        digests.append(session.env.state_digest())
        result = session.hero_command(
            _command(frame, step["offer_id"], step["label"], ordinal)
        )
        if result["status"] != "accepted":
            raise ValueError(f"Command {ordinal} rejected: {result}")
        frame = result["update"]["frame"]
    return session, frame, digests


def refresh_demo(name: str, trace_dir: Path) -> None:
    path = FIXTURES / DEMOS[name]
    fixture = json.loads(path.read_text())
    demo = load_play_demo(name)
    session, frame, digests = _replay(
        demo.config, [step.model_dump() for step in demo.prefix], trace_dir
    )
    assert session.env is not None
    for step, digest in zip(fixture["prefix"], digests, strict=True):
        step["before_digest"] = digest
    fixture["manifest"] = session.env.content_pack_manifest()
    fixture["decision_digest"] = session.env.state_digest()
    fixture["decision_revision"] = frame["revision"]
    if frame["action_space"] != fixture["decision_kind"]:
        raise ValueError(
            f"{name}: reached {frame['action_space']}, fixture pins {fixture['decision_kind']}"
        )
    _write(path, fixture)
    print(
        f"{path.name}: {len(digests)} Commands, decision at revision {frame['revision']}"
    )


def refresh_navigation(trace_dir: Path) -> None:
    path = FIXTURES / "choice-navigation-w17.json"
    fixture = json.loads(path.read_text())
    for family, position in fixture["positions"].items():
        session, frame, _ = _replay(position["config"], position["prefix"], trace_dir)
        assert session.env is not None
        if frame["action_space"] != family:
            raise ValueError(f"{family}: reached {frame['action_space']}")
        position["digest"] = session.env.state_digest()
        # Keep the retained frame's key set; the live frame supplies the values.
        position["frame"] = {
            key: frame.get(key, value) for key, value in position["frame"].items()
        }
        print(f"{path.name}: {family} at revision {frame['revision']}")
    _write(path, fixture)


def refresh_trigger_order() -> None:
    path = FIXTURES / "trigger-order-w17.json"
    fixture = json.loads(path.read_text())
    env, _ = root()
    capture = json.loads(env.capture_decision(0).to_json())
    fixture["world_version"] = capture["world_version"]
    fixture["content_digest"] = capture["content_digest"]
    fixture["root_digest"] = env.state_digest()
    _write(path, fixture)
    print(f"{path.name}: root {fixture['root_digest'][:12]}")


def main() -> None:
    with tempfile.TemporaryDirectory() as trace_dir:
        for name in DEMOS:
            refresh_demo(name, Path(trace_dir))
        refresh_navigation(Path(trace_dir))
    refresh_trigger_order()


if __name__ == "__main__":
    main()
