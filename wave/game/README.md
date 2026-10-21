# Game

Game owns Academy as a lab for researchers: the local tool for loading a
trained manabot, watching it play or playing against it, and inspecting its
decisions. The objective and measures are in [GOAL.md](GOAL.md).

The product stays narrow. It is not a general Magic client, deck builder,
collection manager or format-legality service, and it is not a way to play
Magic in place of other products. The selected matchup is the content
boundary.

## Principles

1. One authoritative, versioned experience protocol connects the rules engine,
   bots, replay and every client view.
2. The client renders what the server decides; it never reconstructs rules or
   reveals information the viewer could not see.
3. Play, replay and decision inspection are views of the same recorded game,
   not separate products.
4. Every checkpoint shown is identified exactly: its world, content and
   producer are checked before it plays.
5. Academy is a client of manabot. It reads manabot's checkpoints, arena games
   and evidence; manabot never imports Academy.

## What exists

- `./scripts/play` starts the server and table locally.
- One trained opponent can be configured through `ACADEMY_PLAY_CANDIDATE`
  ([local challenger](../../docs/local-trained-challenger.md)), chosen in the
  opponent selector.
- Finished games are recorded as traces and open as replays.
- Study forks a recorded decision and lets you try another action.
- Advice shows a policy's distribution at a live decision.

Reference designs: [experience protocol v1](../../docs/architecture/experience-protocol-v1.md)
and [Phase experience notes](../../docs/research/phase-experience.md). Earlier
charters are kept in [legacy-gui-charter.md](legacy-gui-charter.md) and
[MEMORY.md](MEMORY.md).

## Non-goals

- Hosted or public play
- Broad Commander or multiplayer UX
- Deck construction, collection browsing, or general format legality
- Supporting arbitrary cards through runtime natural-language parsing
- A parallel client-side rules engine
