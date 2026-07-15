# Rules memory

## Decisions

- The destination remains the creator's curated cube/decks; comprehensive Magic
  or Commander coverage is not the objective.
- Steal Phase's proven invariants, not its repository shapes: exact object
  incarnation and LKI, typed card meaning, proposed-event replacement flow,
  explicit legal interaction, and viewer-safe projections.
- Use immutable, versioned `ContentPack` definitions plus compact dense mutable
  `MatchState` facts.
- Prefer offline compilation into checked-in typed IR over runtime
  natural-language parsing.
- Replace flat enumerate-then-clone-and-validate action lists with structured
  offers that are legal by construction and admit structured policy decoders.
- The search contract likely combines safe snapshot forks outside a rollout with
  dense transactional execution and mark/rollback inside it.
- Benchmark three implementations at realistic worker × actor × rollout load:
  compact full clone; compact clone plus undo; dense page-COW fork plus undo.
  Decide on rollout throughput and peak RSS, not clone latency alone.
- Use a readable reference reducer and optimized executor as differential
  oracles. Explore Phase as a pinned conformance oracle and opponent pool, not
  as the primary training backend.
- Verification should become mechanical: conformance CI, gap-analysis worklist,
  property/metamorphic/differential/fuzz testing, and a content-change to
  kernel-change ratio that can trigger a redesign kill decision.

## Evidence

- `docs/research/semantic-kernel.md`
- `docs/research/manabot-vs-phase.md`
- Phase comparison pinned to phase-rs commit `553b97bd`
- Existing capability and card-pool audits under `wave/rules/`

## Open tensions

- Introduce identity and event invariants without a flag-day engine rewrite.
- Design the semantic IR from real curated cards without making every card a new
  kernel operation.
- Preserve dense-state rollout speed while making branching exact and safe.
- Keep the human experience protocol and learning observation ABI as separate
  projections of one authoritative match.
