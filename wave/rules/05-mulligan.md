# 05: Mulligan

## Finish line

London mulligan (CR 103.4) is implemented. Players can mulligan opening hands
and choose cards to put on bottom. First player skips first draw.

## Changes

### Mulligan Flow

Implement London mulligan sequence:
1. Each player draws 7 cards
2. Starting with first player, each decides keep or mulligan
3. On mulligan: draw 7 again, then put N cards on bottom (N = number of
   mulligans taken)
4. Repeat until all players keep
5. First player skips draw on first turn (CR 103.8a — already should be
   implemented but verify)

### Action Space

Two new `ActionSpaceKind` variants:
- `Mulligan` — binary choice: keep or mulligan
- `MulliganBottomCards` — choose which card(s) to put on bottom of library

Bottom card selection is sequential (like attackers): for each card to
bottom, agent picks one from remaining hand, repeated N times.

### No Structural Dependencies

Mulligan is self-contained — no dependency on targeting, triggers, stack, or
events. Can be implemented independently of other stages.

### Trace Tests

- Keep 7 (no mulligan)
- Mulligan once, put 1 card on bottom, keep 6
- Mulligan twice, put 2 cards on bottom, keep 5
- Both players mulligan independently
- First player skips first draw after mulligan

### Training smoke test

Agent learns basic mulligan heuristics (keep playable hands, mulligan
unplayable ones). Compare win rate with/without mulligan enabled.

## Done when

- Mulligan works for both players with correct card-bottom sequencing.
- Trace tests for keep/mulligan/bottom sequences pass.
- Training smoke test passes.
