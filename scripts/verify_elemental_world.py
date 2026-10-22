"""Bounded same-Command replay, privacy and fork checks for the selected world.

The cohort and seeds are fixed prospectively in docs/rules/elemental/world.md.
This correctness smoke uses uniform legal offers; it makes no strength claim.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random

import managym
from managym.decision import Command, DecisionFrame
from managym.experience_projection import serialize_observation

PACK = "elemental-world-v2"
DECKS = ("gr", "gw", "ur", "ub", "uw", "ug", "wr", "wb", "rb", "bg")
COHORT = tuple((key, key) for key in DECKS) + (
    ("gr", "ug"),
    ("ug", "gr"),
    ("ur", "gw"),
    ("gw", "ur"),
)


@dataclass(frozen=True)
class Result:
    left: str
    right: str
    seed: int
    commands: int
    terminal: bool
    command_sha256: str
    final_state: str
    fork_checks: int
    max_actions: int


def check(left: str, right: str, *, seed: int, cap: int) -> Result:
    configs = [
        managym.authored_deck_setup(PACK, "elemental_" + key) for key in (left, right)
    ]
    root = managym.Env(seed=seed, skip_trivial=False)
    root.reset(configs)
    replay = managym.Env(seed=seed, skip_trivial=False)
    replay.reset(configs)
    rng = random.Random(seed)
    tape = hashlib.sha256()
    fork_checks = 0
    max_actions = 0
    count = 0
    for count in range(cap):
        if root.observation_for_player(0).game_over:
            break
        frame_json = root.semantic_decision_frame_json()
        assert frame_json == replay.semantic_decision_frame_json(), (
            "replay frame differs"
        )
        frame = DecisionFrame.from_json(frame_json)
        max_actions = max(max_actions, len(frame.offers))
        offer = rng.choice(frame.offers)
        command = Command(f"elemental-{count}", frame.revision, int(offer["id"]))
        payload = command.to_json()
        for viewer in (0, 1):
            projection = serialize_observation(root.observation_for_player(viewer))
            assert not projection["opponent"]["hand"], "private opposing hand leaked"
            assert "sideboard" not in projection["opponent"], (
                "private opposing candidates leaked"
            )
            assert "library" not in projection["agent"], "library order leaked"
        if count % 50 == 0:
            digest = root.state_digest()
            branch = root.clone_env()
            branch.execute_semantic_command_json(payload)
            advanced = branch.state_digest()
            try:
                branch.execute_semantic_command_json(payload)
            except managym.AgentError as error:
                assert "stale_revision" in str(error)
            else:
                raise AssertionError("stale Command was accepted")
            assert branch.state_digest() == advanced, "stale rejection mutated branch"
            assert root.state_digest() == digest, "branch advanced its parent"
            fork_checks += 1
        assert root.execute_semantic_command_json(
            payload
        ) == replay.execute_semantic_command_json(payload)
        assert root.state_digest() == replay.state_digest(), "replay state differs"
        tape.update(payload.encode())
        tape.update(b"\n")
    else:
        count = cap
    terminal = bool(root.observation_for_player(0).game_over)
    return Result(
        left,
        right,
        seed,
        count,
        terminal,
        tape.hexdigest(),
        root.state_digest(),
        fork_checks,
        max_actions,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results: list[Result] = []
    for left, right in COHORT:
        result = check(left, right, seed=164, cap=30_000)
        results.append(result)
        print(
            f"{left}/{right}: {result.commands} Commands, terminal={result.terminal}, max offers={result.max_actions}",
            flush=True,
        )
    payload = {
        "world": managym.WORLD_VERSION,
        "pack": PACK,
        "policy": "uniform legal offers; random.Random(164)",
        "max_commands": 30_000,
        "results": [asdict(row) for row in results],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if not all(row.terminal for row in results):
        raise SystemExit("cohort reached its declared cap; see result rows")


if __name__ == "__main__":
    main()
