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

## Live advice carry-forward (2026-09-24)

The [unfinished live-advice design](../../docs/plans/live-belief-advice.md)
preserves ETU-14's pending work and release gates. The current checkout has
`ed2` pre-command decision addresses, immutable pending roots promoted to Study,
the Rules conditioning-index binding, and a selected tracked-posterior resolver.
Its focused tests cover address promotion, Counterspell support partitions, and
unsupported-world failure. The production `/api/advice` route still serves the
fixture; these seams do not close live advice or fresh live/Study byte parity.

Keep the decision address and viewer projection prefix independent of the
chosen Command. Authored Has/Lacks conditions and the tracked posterior have
separate provenance; probabilities remain server-private. Tracking must consume
all semantic transitions in order, including auto-passes. Snapshot loss makes
advice unavailable, never stalls Commands. Clone inputs under the table lock,
then perform likelihood/search in bounded background lanes. Validate participant
lease, viewer/audience, decision, advisor, compute, and source identities;
unavailable replies and late-response handling must clear old evidence.

Reuse DecisionAdvice and ActionPanel. The release proof requires fresh isolated
live/Study computations with identical canonical bytes, belief-only strategy
deltas, unchanged search roots, and responsive pilot/watcher Commands during
recomputation. The retained design budgets remain command P95 <=100 ms over 20
Commands, broadcast lag <=1 update, fresh advice P95 <=2 s / RSS <=512 MiB, and
cache hit <=50 ms on its declared profile; they were not remeasured here.

The old INT-7 checkpoint pin and quoted serving costs need revalidation under
the new loader ABI. Never repair drift by changing frozen evidence or substituting
a compatible prior. The offline Linear snapshot has ETU-14 and Intelligence's
ETU-21 open, but refresh/mutation is blocked by PRD-44's repository PM migration.
The Game objective is unchanged; this branch does not claim its advice KR.
