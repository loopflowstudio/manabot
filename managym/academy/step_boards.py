"""The person's board after each step of a batch, sent beside the frame.

Between two of the person's decisions the authority may play many steps (the
opponent's turn, the person's automatic passes) and publishes one frame, the
board after the last. The live table reads the opponent's moves out one beat at
a time; to show each beat on the board it happened on (an attacker that died is
seen attacking first), the session keeps the person's viewer-safe projection
after every step that added to their presentation track (`StepBoard`) and sends
them with the next frame as `boards` (`wire_boards`).

These are Academy presentation only: not part of the canonical frame, its hash,
the recording or the observation, and a client may ignore them. A board is the
same `decision_surface_view` the frame's `projection` is, validated the same
way, so it shows exactly what the person may see.

On the wire each board is a patch, and the patches chain backwards from the
frame: the newest board is `apply(patch, frame.projection)`, each older one is
its patch applied to the board after it. Consecutive boards differ by a move,
so a patch is small next to a whole board. A board equal to the one after it
merges into that one's range of events, and a board equal to the frame is not
sent at all: a beat whose events no board covers is shown on the frame.

Patch nodes (`patch`):

- `{"=": value}` replaces the value.
- `{"o": {key: node}, "x": [key, ...]}` edits an object: keys whose values
  changed or were added, and keys removed (either part may be absent).
- `{"a": [item, ...]}` rebuilds an array of objects that each carry a unique
  `id` (battlefield, hand, stack): each item is `{"k": id}` (the base's item of
  that id, unchanged), `{"k": id, "p": node}` (that item, patched) or
  `{"=": value}` (a new item). Other arrays that differ are replaced.

`academy/ui/src/lib/table/boards.ts` applies them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# The newest boards kept per batch: beats read out at most the last few moves.
MAX_STEP_BOARDS = 24


@dataclass(frozen=True)
class StepBoard:
    """The person's board after the step that added events `first`..`last` (presentation seqs)."""

    first: int
    last: int
    projection: dict[str, Any]


def keyed_ids(items: list[Any]) -> list[Any] | None:
    """The items' ids when every item is an object with a unique scalar `id`."""
    ids: list[Any] = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("id"), (int, str)):
            return None
        ids.append(item["id"])
    return ids if len(set(ids)) == len(ids) else None


def patch(base: Any, target: Any) -> dict[str, Any] | None:
    """The node that turns `base` into `target`, or None when they are equal."""
    if base == target:
        return None
    if isinstance(base, dict) and isinstance(target, dict):
        edits: dict[str, Any] = {}
        for key, value in target.items():
            if key not in base:
                edits[key] = {"=": value}
            elif (node := patch(base[key], value)) is not None:
                edits[key] = node
        removed = [key for key in base if key not in target]
        return {
            **({"o": edits} if edits else {}),
            **({"x": removed} if removed else {}),
        }
    if isinstance(base, list) and isinstance(target, list):
        base_ids, target_ids = keyed_ids(base), keyed_ids(target)
        if base_ids is not None and target_ids is not None:
            by_id = dict(zip(base_ids, base, strict=True))
            items: list[dict[str, Any]] = []
            for item_id, item in zip(target_ids, target, strict=True):
                if item_id not in by_id:
                    items.append({"=": item})
                elif (node := patch(by_id[item_id], item)) is None:
                    items.append({"k": item_id})
                else:
                    items.append({"k": item_id, "p": node})
            return {"a": items}
    return {"=": target}


def wire_boards(
    boards: list[StepBoard], projection: dict[str, Any]
) -> list[dict[str, Any]]:
    """`boards` (oldest first) as patches chained back from the frame's `projection`.

    Each entry is `{"first", "last", "patch"}`, oldest first, as `boards`.
    """
    out: list[dict[str, Any]] = []
    after = projection
    for board in reversed(boards):
        node = patch(after, board.projection)
        if node is None:
            # The same board as the one after it: that one covers these events too.
            if out:
                out[-1]["first"] = board.first
            continue
        out.append({"first": board.first, "last": board.last, "patch": node})
        after = board.projection
    out.reverse()
    return out
