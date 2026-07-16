---
pm:
  provider: linear
  linear_initiative: 21966203-a6bb-4e2c-a902-f43cbe813053
  linear_team: ef9e3b10-0953-4e92-aba7-0f587336f1cd
---

# Game

## Objective

Build Etude Fantasia toward the definitive Avatar Cube Team Sealed experience:
humans construct three decks from a shared pool, choose which seats they pilot,
play a three-by-three deck matchup matrix with human or manabot teammates and
opponents, and study every recorded game afterward. Construction, play, replay,
and Study are one player experience, not separate products.

The nearest product remains deliberately smaller: make one creator-selected
human-versus-manabot game complete, polished, recoverable, replayable, and
genuinely useful to Study. The north star guides interfaces and sequencing; it
does not justify filing the entire downstream feature tree before this loop
works.

## Measures

- One selected matchup launches through the release stack, reaches terminal,
  recovers safely, and preserves the same authoritative frames, offers,
  commands, semantic events, and replay identities across direct play and
  Study.
- Every historical player decision in a completed game is addressable and
  restorable; a player can inspect evidence, retry an exact position, follow a
  canonical continuation, and return without client-side rules or replay
  reconstruction.
- The player-facing table is fast, legible, accessible, visually authored, and
  portable, with repeatable release, recovery, performance, and visual gates.
- Versioned format and series identities can eventually represent the Avatar
  defaults without hardcoding them into match execution: a 540-card cube, two
  135-card team pools, three 40-card-minimum decks per team, unlimited basic
  lands, deck-specific sideboards, a full three-by-three game matrix, and a
  five-win clinch.
- Every played game in a future team series remains an independent canonical
  match and is recorded under one series identity for replay and Study.

## Bounds and sequencing

Study is a named Game mode. Game owns construction UX, play, presentation,
replay, recovery, series orchestration, and Study interaction. Rules owns
content, pool and deck legality, matches, viewer-safe state, and exact forks.
Intelligence owns manabot play, search, evidence, and later deck construction.

The first robot team may use fixed authored decks. Manabots initially pilot
decks without sideboarding. Building three legal decks from a shared sealed
pool is an important later Intelligence capability; drafting is not required.
Discord is the assumed human communication layer, so Etude does not build chat.

Advance by completing the nearest runnable player loop, then add the next
smallest behavior that makes the Avatar Team Sealed destination more real.
Do not create speculative lobby, social, tournament, or generalized cube-
platform infrastructure ahead of an exercised need.
