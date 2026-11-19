"""JSON for records that grow by appending, encoded once per item.

A live game's record (`Trace`, its canonical replay) is saved after every step
and grows with the game: about 20 MB at 300 events. Re-encoding it whole on
each save makes a long game slower with every move. `encode_list` keeps each
item's JSON beside the item object it encodes and re-encodes an item only
when the object at its index is a different one; `splice` puts such texts
into an encoding of the rest. Together they produce exactly the bytes
`json.dumps` gives for the whole value (default separators), so stored
records are unchanged.

Items must not be mutated in place once encoded: identity is the only check.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import json
from typing import Any

# A string no encoded value contains: JSON escapes control characters.
PLACEHOLDER = "\x00append-json:{}\x00"


def encode_list[T](
    items: Sequence[T], encoded: list[tuple[T, str]], encode: Callable[[T], str]
) -> str:
    """`json.dumps(list)` text from each item's `encode`, updating `encoded` in place."""
    del encoded[len(items) :]
    for index, item in enumerate(items):
        if index < len(encoded) and encoded[index][0] is item:
            continue
        if index < len(encoded):
            encoded[index] = (item, encode(item))
        else:
            encoded.append((item, encode(item)))
    return "[" + ", ".join(text for _, text in encoded) + "]"


def splice(value: dict[str, Any], parts: dict[str, str]) -> str:
    """`json.dumps(value)` with the values at `parts`' keys replaced by their texts.

    Each key keeps its place in `value`; a key not in `value` is appended.
    """
    marked = dict(value)
    for key in parts:
        marked[key] = PLACEHOLDER.format(key)
    text = json.dumps(marked)
    for key, part in parts.items():
        text = text.replace(json.dumps(PLACEHOLDER.format(key)), part, 1)
    return text
