---
pm:
  linear_initiative: 21966203-a6bb-4e2c-a902-f43cbe813053
---

# Game

## Objective

Make Academy a lab for researchers: someone who has just trained a manabot can
load that checkpoint, watch it play or play against it, step through any of its
decisions, and see why it chose what it did, without writing glue code.

## Measures

1. **Load what you trained.** A checkpoint produced by `uv run manabot train`
   or a remote run opens in Academy from its path or run, with its world,
   content and producer identity shown and checked. A checkpoint from an
   incompatible world is refused with a message that names the mismatch.

2. **See a bot play.** A researcher can play against a checkpoint, or watch two
   checkpoints (or a checkpoint and a fixed anchor) play each other, on chosen
   decks, seats and deal seeds. Arena and Tournament games open as replays in
   the same table.

3. **Inspect a decision.** At any recorded decision the researcher sees the
   legal actions, the policy's probabilities over them, the value estimate,
   and, where they were used, the belief samples and search results that fed
   the choice. Inspection reads the same recorded state as play; it never
   reconstructs rules or reveals hidden cards the acting player could not see.

4. **Try another line.** From any recorded decision the researcher can fork the
   game, choose a different action or swap in another checkpoint, and compare
   what follows.

## Ownership

Game owns Academy as a research instrument: loading checkpoints, the table,
replay, decision inspection, and forks. Intelligence owns training, the arena
and strength claims; Game displays their checkpoints and games without
re-deriving them. Rules owns legality, hidden information and replay. Academy
is a client of manabot, and manabot never imports Academy.

<<<<<<< HEAD
Start from what already exists: one configured trained opponent
(`ACADEMY_PLAY_CANDIDATE`), the opponent selector, replay, Study forks, and
advice. Widen each toward the measures above before adding new surfaces.
=======
Start from what already exists: the trained-checkpoint opponent selector,
replay, Study forks, and advice. Widen each toward the measures above before adding new surfaces.
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f

## Limits

Academy is a local tool for inspecting bots, not a way to play Magic. It is not
a general Magic client, deck builder or collection manager, and hosted play is
not a goal. The selected matchup ([UR Lessons versus GW Allies](../../content/DECKS.md))
is the content boundary. Historical charters and decisions remain in
[memory](MEMORY.md) and the chapter archive.
