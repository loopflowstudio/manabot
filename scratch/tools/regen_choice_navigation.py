"""Re-record the w17 choice-navigation positions in place by offer label."""

import json
import subprocess
import tempfile
from pathlib import Path

from academy.server import GameSession
import managym

SOURCE = Path("academy/fixtures/choice-navigation-w17.json")
TARGET = Path("academy/fixtures/choice-navigation-w17.json")

fixture = json.loads(SOURCE.read_text())
for family, position in fixture["positions"].items():
    session = GameSession(Path(tempfile.mkdtemp()))
    frame = session.new_game(position["config"])["frame"]
    prefix = []
    for ordinal, step in enumerate(position["prefix"]):
        offer = next(o for o in frame["offers"] if o["label"] == step["label"])
        prefix.append({"offer_id": offer["id"], "label": step["label"]})
        result = session.hero_command(
            {
                "command_id": f"navigation.{ordinal}",
                "match_id": frame["match_id"],
                "expected_revision": frame["revision"],
                "prompt_id": frame["prompt"]["id"],
                "offer_id": offer["id"],
                "answers": [],
            }
        )
        assert result["status"] == "accepted", (family, ordinal)
        frame = result["update"]["frame"]
    assert frame["action_space"] == position["frame"]["action_space"], family
    position["prefix"] = prefix
    position["digest"] = session.env.state_digest()
    position["frame"] = {key: frame[key] for key in position["frame"]}
    print(family, frame["revision"], fixture["positions"][family]["frame"]["revision"])
fixture["world_version"] = managym.WORLD_VERSION
fixture["source_base_commit"] = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], text=True
).strip()
TARGET.write_text(json.dumps(fixture, indent=2) + "\n")
