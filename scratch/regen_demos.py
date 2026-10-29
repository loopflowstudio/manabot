"""One-off: regenerate Academy play demos under native skipping.

Replays each explicit-mode demo (skip off, reference only) to tag every step
with its turn/step, then walks a skipping session and takes the matching step.
"""

import json
import sys
from pathlib import Path
import tempfile

import managym

from academy import server
from academy.play_demo import PlayDemo

FIXTURES = Path("academy/fixtures")
NATIVE = managym.Env


def _session(skip: bool) -> server.GameSession:
    if skip:
        server.managym.Env = NATIVE
    else:
        server.managym.Env = lambda *, seed: NATIVE(seed=seed, skip_trivial=False)
    return server.GameSession(Path(tempfile.mkdtemp()))


def _where(session: server.GameSession) -> tuple[int, str]:
    turn = session.obs.turn
    return int(turn.turn_number), str(turn.step)


def _command(session, frame, offer_id, cid):
    result = session.hero_command(
        {
            "command_id": cid,
            "match_id": frame["match_id"],
            "expected_revision": frame["revision"],
            "prompt_id": frame["prompt"]["id"],
            "offer_id": offer_id,
            "answers": [],
        }
    )
    assert result["status"] == "accepted", result
    return result["update"]["frame"]


def regenerate(source: str, target: str) -> None:
    demo = PlayDemo.model_validate_json((FIXTURES / source).read_text())
    ref = _session(skip=False)
    frame = ref.new_game(dict(demo.config))["frame"]
    tagged = []
    for i, step in enumerate(demo.prefix):
        tagged.append((_where(ref), step.label))
        frame = _command(ref, frame, step.offer_id, f"ref.{i}")
    goal = (_where(ref), sorted(o["label"] for o in frame["offers"]), len(ref.obs.stack_objects))
    ref.close("regen")

    new = _session(skip=True)
    frame = new.new_game(dict(demo.config))["frame"]
    prefix = []
    pointer = 0
    while True:
        here = _where(new)
        labels = {o["label"]: o["id"] for o in frame["offers"]}
        if (
            here == goal[0]
            and set(goal[1]) <= set(labels)
            and frame["action_space"] == demo.decision_kind
            and all(label == "Pass priority" for _, label in tagged[pointer:])
        ):
            break
        match = next(
            (
                k
                for k in range(pointer, len(tagged))
                if tagged[k][0] == here and tagged[k][1] in labels
            ),
            None,
        )
        if match is None:
            print(goal, (here, sorted(labels), len(new.obs.stack_objects)))
            raise SystemExit(f"{source}: no step at {here} offers {list(labels)}")
        label = tagged[match][1]
        prefix.append(
            {
                "before_digest": new.env.state_digest(),
                "offer_id": labels[label],
                "label": label,
            }
        )
        frame = _command(new, frame, labels[label], f"new.{len(prefix)}")
        pointer = match + 1
    out = json.loads((FIXTURES / source).read_text())
    out["prefix"] = prefix
    out["decision_digest"] = new.env.state_digest()
    out["decision_revision"] = frame["revision"]
    out["manifest"] = new.env.content_pack_manifest()
    (FIXTURES / target).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(target, len(demo.prefix), "->", len(prefix), "revision", frame["revision"])
    new.close("regen")


if __name__ == "__main__":
    for source, target in zip(sys.argv[1::2], sys.argv[2::2]):
        regenerate(source, target)
