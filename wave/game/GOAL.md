---
pm:
  provider: linear
  linear_initiative: 21966203-a6bb-4e2c-a902-f43cbe813053
  linear_team: ef9e3b10-0953-4e92-aba7-0f587336f1cd
---

# Game

## Objective

Build Etude Fantasia as an AI-assisted Pro Tour testing house. Piloting,
watching, stating a read, comparing strategy, trying a line, having fun, and
continuing into Study are one shared surface over one authoritative match—not
separate play, replay, and analysis products.

The first AI experience is a decision surface where a player states an
explicit viewer-safe belief, sees one identified advisor's complete strategy
conditioned on it, changes the belief, and sees the delta. Match facts,
player-authored beliefs, model-inferred beliefs, and advice remain separately
labelled. The same decision, belief, advisor, compute class, and provenance
identity produces the same advice live or in Study. Only the acting pilot may
commit the live offered `Command`; watchers and isolated exploration never
pause or mutate the authoritative match.

The destination remains the definitive Avatar Cube Team Sealed experience:
humans construct three decks from a shared pool, choose which seats they pilot,
play a three-by-three deck matchup matrix with human or manabot teammates and
opponents, and study every recorded game afterward. The nearest product is
deliberately smaller: make one creator-selected human-versus-manabot game
complete, polished, recoverable, replayable, and genuinely useful to Study.
The north star guides interfaces and sequencing; it does not justify filing the
entire downstream feature tree before this loop works.

## Measures

- One selected matchup launches through the release stack, reaches terminal,
  recovers safely, and preserves the same authoritative frames, offers,
  commands, semantic events, and replay identities across direct play and
  Study.
- Every historical player decision in a completed game is addressable and
  restorable; a player can inspect evidence, Retry an exact position, follow a
  bounded canonical continuation, compare it, and return without client-side
  rules or replay reconstruction.
- At the same canonical decision reached live or through Study, at least two
  explicit viewer-safe belief scenarios use one pinned advisor and compute
  class and return reproducible, complete aligned action distributions,
  values, robustness, uncertainty, and deltas. Facts, authored beliefs,
  inferred beliefs, and advice remain distinct, and empty or mismatched
  evidence fails closed.
- A pilot and permitted watcher can inhabit the same canonical decision
  surface and compare viewer-safe belief and strategy artifacts without
  pausing or mutating the match. Only the pilot can submit its offered live
  `Command`; an isolated line returns to the identical recorded decision.
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

Study is a named Game mode and a time control over the same testing-house
surface. Game owns construction UX, player-authored belief and comparison UX,
viewer roles and capabilities, live play, presentation, replay, recovery,
series orchestration, isolated-line interaction, and Study navigation. The
ActionPanel remains the sole live Command path.

Rules owns content, pool and deck legality, matches, viewer-safe facts, typed
possible-world queries, and exact forks and returns. Intelligence owns manabot
play, priors, model-inferred beliefs, advisor/search identity, compute identity,
and attributable evidence. Game presents those inputs without inferring hidden
truth or generating strategy in the client.

The first robot team may use fixed authored decks. Manabots initially pilot
decks without sideboarding. Building three legal decks from a shared sealed
pool is an important later Intelligence capability; Game owns the construction
experience. Drafting is not required. Discord is the human communication layer,
so Etude owns shared decisions and artifacts but does not build chat.

Advance by completing the nearest runnable player loop, then add the next
smallest behavior that makes the Avatar Team Sealed destination more real. Do
not create speculative lobby, social, tournament, or generalized cube-platform
infrastructure ahead of an exercised need.
