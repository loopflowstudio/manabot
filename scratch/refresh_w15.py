"""Replay retained w14 inputs into fresh w15 receipts; never rewrite a source."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from academy import server
from academy.server import GameSession
from manabot.verify.trigger_order import root
import managym

FIXTURES = Path("academy/fixtures")


def write(name: str, value: dict[str, Any]) -> None:
    """JSON is the local fixture boundary; GameSession validates its config/Commands."""
    target = FIXTURES / name.replace("w14", "w15")
    assert not target.exists(), target
    target.write_text(json.dumps(value, indent=2) + "\n")


def replay(value: dict[str, Any], path: Path, *, demo: bool) -> None:
    session = GameSession(path)
    try:
        update = session.new_game(value["config"])
        assert session.env is not None
        if demo:
            assert value["manifest"]["world_version"] == "w14"
            manifest = session.env.content_pack_manifest()
            assert manifest == {**value["manifest"], "world_version": "w15"}
            value["manifest"] = manifest
        for ordinal, step in enumerate(value["prefix"]):
            frame = update["frame"]
            offer = next(o for o in frame["offers"] if o["id"] == step["offer_id"])
            assert offer["label"] == step["label"], (ordinal, offer, step)
            if demo:
                step["before_digest"] = session.env.state_digest()
            result = session.hero_command(
                {
                    "command_id": f"w15-prefix.{ordinal}",
                    "match_id": frame["match_id"],
                    "expected_revision": frame["revision"],
                    "prompt_id": frame["prompt"]["id"],
                    "offer_id": offer["id"],
                    "answers": [],
                }
            )
            assert result["status"] == "accepted", result
            update = result["update"]
        frame = update["frame"]
        if demo:
            assert frame["action_space"] == value.get("decision_kind", "LEARN")
            assert frame["revision"] == value["decision_revision"]
            value["decision_digest"] = session.env.state_digest()
        else:
            for field in ("revision", "action_space", "offers", "projection"):
                assert frame[field] == value["frame"][field], field
            value["digest"] = session.env.state_digest()
            value["frame"] = frame
    finally:
        session.close("fixture-recording")


def main() -> None:
    assert managym.WORLD_VERSION == "w15", managym.WORLD_VERSION
    with TemporaryDirectory() as temporary:
        path = Path(temporary)
        for name in (
            "learn-demo-explicit-w14.json",
            "precision-demo-w14.json",
            "precision-mana-demo-w14.json",
        ):
            value = json.loads((FIXTURES / name).read_text())
            replay(value, path, demo=True)
            write(name, value)
        native_env = managym.Env

        def collapsed_env(*, seed: int, skip_trivial: bool) -> managym.Env:
            return native_env(seed=seed, skip_trivial=True)

        for name in (
            "choice-navigation-w14.json",
            "choice-navigation-explicit-w14.json",
        ):
            value = json.loads((FIXTURES / name).read_text())
            if name == "choice-navigation-w14.json":
                server.managym.Env = collapsed_env
            try:
                for position in value["positions"].values():
                    replay(position, path, demo=False)
            finally:
                server.managym.Env = native_env
            write(name, value)
    env, _ = root()
    value = json.loads((FIXTURES / "trigger-order-w14.json").read_text())
    capture = json.loads(env.capture_decision(0).to_json())
    assert capture["content_digest"] == value["content_digest"]
    value["root_digest"] = env.state_digest()
    value["world_version"] = managym.WORLD_VERSION
    write("trigger-order-w14.json", value)
    print("Recorded three demos, both navigation cohorts and the trigger root for w15")


if __name__ == "__main__":
    main()
