# Rules memory

## Operating principle

> Build the world, play it, measure it, then isolate what surprised us.

Rules leads with playable curated worlds and real search/Study consumers.
Conformance, fuzzing, benchmarks, and micro-katas are instruments for observed
ambiguity or safety-critical invariants, not substitute products.

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

- Compact full clone is retained because mutable state is dense, immutable
  definitions are shared, and clone-plus-undo and page-COW did not justify a
  production split under measured workloads.
- RUL-2 ran `full_clone/current_game_v1` through the authored PUCT teacher with
  revision-bound Commands at world, child, and leaf. Selected/reference traces
  were exact with zero fallback; selected delivered 35.76 decisions/s at
  interactive p95 65.99 ms and 4.70 decisions/s saturated at p95 1343.29 ms
  with matched RSS.
- PR #136 recorded a checked seed-0 UR Lessons versus GW Allies release-stack
  authority trace: terminal revision 132, 132 Commands across nine prompt
  families, 26 admitted typed programs, ordered semantic/presentation events,
  and zero legacy fixed-action, card-name, candidate-cap, or client-legality
  fallback. Authored replay parity, workload budgets, and another
  creator-selected increment remain open.
- PRs #129 and #134 provide a Rules-owned Study fork over compact full clone.
  A historical viewer-safe decision can execute a structured command, preserve
  retained source and siblings, and return a consuming receipt with the exact
  canonical source digest, frame, offer, command, event cursor, and
  continuation. Retained-root drift fails closed.
- Revision-bound receipts identify a reproducible relevant source closure.
  Incidental worktree paths, nested agent worktrees, and unrelated files are
  not source identity.

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
- A historical Study identity resolves to an exact viewer-safe fork, executes
  normal structured commands, preserves the recorded source, and returns to the
  original state and event cursor in one action.
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

- Prove revision-by-revision live/headless/replay parity for the shipped
  authored trace without creating a parallel replay authority.
- Finish Study consumer evidence with fork/apply/return latency and peak RSS
  budgets plus focused multi-seed, nested, stale/pack, and privacy stress.
- Preserve dense-state speed as card semantics and event histories grow.
- Let the next creator-selected content increment pressure the IR naturally
  without turning each card into a one-off kernel exception.
- Keep exactness non-negotiable without allowing proof infrastructure to become
  a substitute for playing the world.
