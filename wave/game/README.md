# Game

Game owns the complete experience of playing Etude Fantasia. Its standard is not "a
working GUI" but a small, authored game whose smoothness, performance, visual
polish, reliability, and portability equal or exceed Phase for the exact decks
the creator chooses to support.

The product remains intentionally narrow. It is not a generic Commander
client, deck builder, collection manager, or format-legality service. Curated
content is an artistic constraint, not missing platform work.

## Product principles

1. One authoritative, versioned experience contract connects the rules engine,
   bots, replay, and every client surface.
2. The client renders semantic game meaning rather than reconstructing rules
   from mutable snapshots.
3. Every prompt is complete and legal by construction; every command is bound
   to an exact revision, prompt, and offer and is safe to retry.
4. Reconnect and replay are normal execution paths, not emergency patches.
5. Card art, crops, sounds, and interaction treatments ship as exact curated
   packs with deterministic fallbacks and no network dependency during play.
6. Direct play, decision inspection, and research traces are different views of
   the same match, not separate products with drifting semantics.
7. WASM is an adapter option to earn through measurement. Portability begins
   with a transport-neutral protocol and a tiny client.

## Portfolio

### Experience contract and recovery

Turn `ExperienceFrame + InteractionOffer/Command + PresentationEvent +
RecoveryEnvelope` into the stable authority boundary. Prove stale-command
rejection, idempotency, resume, reconnect, and replay from checkpoints.

### Semantic presentation and interaction

Build the event-driven table: prompt-specific interaction, meaningful animation
beats, audio/haptics seams, accessibility, and a decision inspector fed by the
same state and events as direct play.

### Curated packs and portable runtime

Produce exact deck/card asset manifests, offline-first loading, deterministic
fallbacks, process/worker adapters, packaging, and measured browser portability.
Adopt WASM only where it improves startup, latency, deployment, or offline use.

### Experience proof

Make polish computable: end-to-end flows for every curated prompt family,
reconnect and duplicate-command fault injection, replay equivalence, visual and
accessibility checks, frame/interaction latency budgets, and clean-machine
launch verification.

## Near-term sequence

1. Land the protocol types and one end-to-end Lightning Bolt interaction.
2. Add revision-safe commands and snapshot-plus-event recovery.
3. Drive presentation from semantic events while preserving the current table.
4. Package one exact curated matchup as the experience proof.
5. Benchmark process, worker, and only then WASM authority adapters.

Reference designs live in `docs/research/phase-experience.md` and
`docs/architecture/experience-protocol-v1.md`. The previous instrument-only charter is
preserved in `legacy-gui-charter.md`.

## Non-goals

- Broad Commander or multiplayer UX
- Deck construction, collection browsing, or general format legality
- Supporting arbitrary cards through runtime natural-language parsing
- A parallel client-side rules engine
- WASM as an architectural identity
