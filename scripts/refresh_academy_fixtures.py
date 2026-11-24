"""Refresh the live Academy fixtures after a change to state digests or ids.

Each fixture pins digests of positions reached by recorded Commands (offer ids
and labels) from an ordinary GameSession. This replays those Commands through
the live server and rewrites only what the runtime derives: digests, revisions,
manifests and the retained frames. A label that no longer matches is a rules or
offer change, not something to refresh, and stops the script.

Run from the repository root after rebuilding the extension:

    uv run --extra dev python scripts/refresh_academy_fixtures.py

Refreshes every play demo in `managym.academy.play_demo.DEMO_FILES`,
and `choice-navigation-opening-v1.json`. Conformance and parity receipts have
their own writers.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any

from managym.academy.play_demo import DEMO_FILES, load_play_demo, manifest_digest
from managym.academy.server import GameSession

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "managym" / "academy" / "fixtures"


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
    path = FIXTURES / DEMO_FILES[name]
    fixture = json.loads(path.read_text())
    demo = load_play_demo(name)
    session, frame, digests = _replay(
        demo.config, [step.model_dump() for step in demo.prefix], trace_dir
    )
    assert session.env is not None
    for step, digest in zip(fixture["prefix"], digests, strict=True):
        step["before_digest"] = digest
    fixture["manifest_digest"] = manifest_digest(session.env.content_pack_manifest())
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
    path = FIXTURES / "choice-navigation-opening-v1.json"
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


def main() -> None:
    with tempfile.TemporaryDirectory() as trace_dir:
        for name in DEMO_FILES:
            refresh_demo(name, Path(trace_dir))
        refresh_navigation(Path(trace_dir))


if __name__ == "__main__":
    main()
