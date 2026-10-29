"""Re-record the w15 Academy play demos for the current world by offer label."""

import json
import sys
import tempfile
from pathlib import Path

from academy import server
from academy.play_demo import PlayDemo

FIXTURES = Path("academy/fixtures")


def regenerate(source: str, target: str) -> None:
    demo = PlayDemo.model_validate_json((FIXTURES / source).read_text())
    session = server.GameSession(trace_dir=Path(tempfile.mkdtemp()))
    update = session.new_game(dict(demo.config))
    prefix = []
    for ordinal, step in enumerate(demo.prefix):
        frame = update["frame"]
        offer = next(o for o in frame["offers"] if o["label"] == step.label)
        prefix.append(
            {
                "before_digest": session.env.state_digest(),
                "offer_id": offer["id"],
                "label": step.label,
            }
        )
        result = session.hero_command(
            {
                "command_id": f"regen.{ordinal}",
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
    assert frame["action_space"] == demo.decision_kind, frame["action_space"]
    out = json.loads((FIXTURES / source).read_text())
    out["manifest"] = session.env.content_pack_manifest()
    out["prefix"] = prefix
    out["decision_digest"] = session.env.state_digest()
    out["decision_revision"] = frame["revision"]
    (FIXTURES / target).write_text(json.dumps(out, indent=2) + "\n")
    print(target, frame["revision"], demo.decision_revision)


for name in sys.argv[1:]:
    regenerate(f"{name}-w15.json", f"{name}-w17.json")
