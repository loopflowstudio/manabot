"""Re-record the w17 choice-navigation positions in place by offer label."""

import json
from pathlib import Path
import subprocess
import tempfile

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
        # Labels can repeat (two copies of one card in hand); keep the recorded
        # offer id when it still carries the label, so the position's state
        # and digest are preserved and only the frame's observation moves.
        candidates = [o for o in frame["offers"] if o["label"] == step["label"]]
        offer = next(
            (o for o in candidates if o["id"] == step["offer_id"]), candidates[0]
        )
        if len(candidates) > 1:
            print(f"{family}: step {ordinal} {step['label']!r} x{len(candidates)}, kept id {offer['id']}")
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
