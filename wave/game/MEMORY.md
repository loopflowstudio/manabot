# Game memory

## North star

- Etude Fantasia grows toward Avatar Cube Team Sealed: two teams build three
  decks from shared pools, play the full three-by-three deck matchup matrix to
  five wins, and can study every recorded game afterward.
- Study is a named Game mode, not an independent product wave. Construction,
  play, replay, Retry, and comparison are one player loop.
- The Avatar starting values—540 cube cards, 135 cards per team, three
  40-card-minimum decks, unlimited basics, deck-specific sideboards, and five
  wins—are versioned format parameters rather than engine constants.
- The first robot team may use fixed authored decks. Manabots initially pilot
  without sideboarding; sealed-pool deck construction is an important later
  Intelligence capability, while drafting is separate.
- Discord is the assumed human communication layer. Do not build chat.
- This destination guides interfaces and sequencing but does not justify a
  speculative Team Sealed backlog before one polished play-to-Study loop works.

## Decisions

- Renamed from `gui` to `game` on 2026-07-15 because the wave owns the full
  playing experience, not a rendering technology.
- The experience target is Phase-level or better smoothness, performance,
  polish, reliability, and portability for creator-selected decks.
- Preserve Etude Fantasia's differentiators: visible AI identity, decision inspection,
  research-grade traces, replay, and a deliberately tiny curated product.
- The authority seam is `ExperienceFrame`, `InteractionOffer`/`Command`,
  `PresentationEvent`, and `RecoveryEnvelope`.
- Commands bind to revision + prompt + offer and carry a stable command ID.
- Presentation consumes semantic events; it does not infer meaning by diffing
  arbitrary snapshots.
- Canonical replay exposes a stable address for every historical player
  decision. Study may rank highlights, but it does not define or reconstruct
  the replay timeline.
- Offline command queues must not replay gameplay decisions into a newer state.
- Curated assets are versioned content, not opportunistic runtime fetches.
- WASM is deferred until adapter benchmarks show a product benefit.

## Evidence

- `docs/research/phase-experience.md`
- `docs/architecture/experience-protocol-v1.md`
- Legacy implementation notes in `01-play-interface.md` and `05-polish.md`
- Previous charter in `legacy-gui-charter.md`

## Open tensions

- Keep protocol design ambitious without blocking a thin vertical slice.
- Preserve the useful existing Svelte/FastAPI table while replacing its
  snapshot/action seam incrementally.
- Treat visual authorship as a product requirement without creating a generic
  content platform.
