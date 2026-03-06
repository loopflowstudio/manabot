# 05: Mana Display

**Finish line:** Hero's available mana pool is visible during main phases in both live play and replay.

## What to build

The GUI now has Scryfall card images, replay, game log, opponent selector, and clickable board interactions. The remaining polish item is mana display.

Current blocker: the Python bindings used by `gui/server.py` do not expose mana pool state from the Rust engine. This requires a Rust/PyO3 change to serialize mana pool contents into the observation.

## Steps

1. Add mana pool to the Rust observation serialization (PyO3 bindings)
2. Include mana pool in `serialize_observation()` in `gui/server.py`
3. Add a `ManaDisplay` component to the frontend board
4. Show mana in both live play and replay views

## Context from shipped work

- Shared board components live in `frontend/src/lib/components/` — `GameBoard.svelte` is the passive root with callbacks
- `PlayerArea.svelte` renders per-player state (life, hand, battlefield, graveyard) — mana display fits here
- Types are in `frontend/src/lib/types.ts` — `PlayerState` will need a `mana_pool` field
- Live play still relies on frontend redaction for opponent hands; the shared board renders opponent hand backs from counts and ignores serialized hand contents. A later pass could redact hidden hand contents server-side.

## Done when

- Mana pool visible during main phases in live play
- Mana pool visible in replay
- No Scryfall or board regressions
