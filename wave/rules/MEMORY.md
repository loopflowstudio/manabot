# Rules memory

## Operating principle

> Build the world, play it, measure it, then isolate what surprised us.

Rules previously organized architectural invariants into independent proof and
benchmark Projects. That work produced valuable substrate, but the portfolio
now leads with playable curated worlds and real search/Study consumers.
Conformance, fuzzing, benchmarks, and micro-katas remain available when a
running slice exposes ambiguity or when a safety-critical invariant requires a
gate.

## Shipped substrate

- Immutable versioned `ContentPack` definitions are separated from compact
  mutable match facts. Environments and branches share definitions without
  copying card content.
- Stable `CardDefId` values and deterministic state witnesses reproduce seeded
  traces without allocation-address or debug-format identity.
- The selected two-deck content is compiled offline into checked-in typed IR and
  executed without card-name dispatch in the admitted interpreter boundary.
- Priority, targeting, and attacker declarations have uncapped typed offers and
  atomic revision-bound commands. The structured decoder preserved 6,435/6,435
  shared decisions and handled more than 32 legal choices.
- Viewer-safe semantic program projection exposes complete ragged programs with
  explicit structure, binding, schema, and pack identity and rejects unknown
  primitives or silent truncation.
- Object incarnation and LKI, proposed-event replacement/prevention, and
  trigger/SBA fixpoint behavior are explicit for the curated slice.
- Exact fork/root/sibling isolation, nested rollback, deterministic replay,
  viewer projection, stale-reference handling, differential execution,
  property/metamorphic checks, bounded fuzzing, and a pinned Phase overlap
  matrix exist as reusable instruments.

## Runtime evidence

- Compact full clone is a strong baseline because mutable state is already
  dense and immutable definitions are shared.
- Compact clone plus fine-grained undo did not clear its preregistered material
  win: flat throughput was slightly worse and peak RSS roughly 50% higher,
  largely because inverse-journal rollback cost more than copying the compact
  state.
- Dense event-page COW plus undo has been implemented and measured against both
  baselines. Its decision belongs to the real search and Study integrations;
  clone latency alone never selects a production representation.
- Revision-bound receipts must identify a reproducible relevant source closure
  and survive verification from another checkout. Incidental worktree paths,
  nested agent worktrees, and unrelated files are not source identity.

## Durable design decisions

- The destination is the creator's selected decks and matchups, not general
  Magic or Commander support.
- Steal Phase's invariants, not its shapes: incarnation and LKI, typed meaning,
  proposed-event replacement flow, explicit legal interaction, and viewer-safe
  projection.
- Prefer offline compiled typed IR over runtime natural-language parsing.
- Unknown primitives and oversized programs fail admission; they never become
  a generic unknown token or a silently truncated sequence.
- Structured offers are legal by construction. Flat enumerate-then-cap action
  lists remain compatibility adapters, not the destination.
- Use safe outer forks and dense transactional execution where real workload
  measurements justify them. Persistent collections, page COW, or undo are not
  adopted by doctrine.
- A historical Study identity must resolve to an exact viewer-safe fork,
  execute normal structured commands, preserve the recorded source, and return
  to the original state and event cursor in one action.
- A new diagnostic must name the play, search, or Study behavior it explains
  and the runtime decision it can change.

## Ownership boundaries

- **Game:** authored presentation, replay UX, recovery, accessibility,
  packaging, and client adapters.
- **Rules:** authoritative content, match state, semantics, legal interaction,
  events, viewer-safe projection, forks, and deterministic execution.
- **Intelligence:** policy, search algorithms, teacher data, and evaluation.
- **Study:** human-facing retry, reveal, comparison, and branch navigation.

## Open tensions

- Make the compiled semantic path the exercised production authority for a
  complete authored match rather than a parallel acceptance substrate.
- Integrate the selected branch runtime into actual Intelligence search and
  Study fork/return before optimizing another benchmark-only representation.
- Preserve dense-state speed as card semantics and event histories grow.
- Let creator-selected content pressure the IR naturally without turning each
  card into a one-off kernel exception.
- Keep exactness non-negotiable without allowing proof infrastructure to become
  a substitute for playing the world.
