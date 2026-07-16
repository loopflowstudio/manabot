# Game memory

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
