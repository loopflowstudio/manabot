# Rules

Rules owns the semantic kernel shared by human play, replay, training, and
search. It is active again as of 2026-07-15 with a redesigned charter: make the
creator-selected decks exact, expressive, inspectable, and cheap to branch
without turning manabot into a general Magic platform.

The prior capability-ordered content roadmap still supplies the card pressure:
UR Lessons vs GW Allies, then TLA commons, then the curated cube's named tail.
Those pools are acceptance suites for the kernel, not a reason to accumulate
one-off card handlers. The detailed audit remains in `00-pool-audit.md`.

## Architecture principles

1. Immutable, versioned `ContentPack` definitions are separate from compact
   dense `MatchState` facts.
2. Rules boundaries carry typed identity domains, especially
   `ObjectRef { entity, incarnation }`, with explicit last-known information.
3. Curated card text compiles offline to checked-in typed semantic programs;
   runtime natural-language parsing is outside the trusted path.
4. The engine exposes structured legal offers and consumes atomic commands.
   It does not materialize a capped flat action list and clone the game merely
   to discover whether each candidate is legal.
5. Replacement and prevention semantics operate on proposed events before
   committed mutation; state-based actions and triggers reach an explicit
   fixpoint.
6. Product frames, presentation events, learning observations, and search
   snapshots are projections of one authoritative match—not one god-object
   serialized everywhere.
7. Optimization follows a written search-state contract and benchmarks. Dense
   state, safe forks, and undo are complementary tools.
8. Learning schemas bind symbolic semantics to checkpoints. Runtime registry
   IDs are transport, semantic programs are complete rather than silently
   truncated, and compatible table reordering does not create a new meaning.

## Portfolio

### Runtime state foundation

Separate immutable card definitions from mutable facts, establish stable IDs
and deterministic state hashing, and measure the current clone/step/RSS baseline
before choosing a branching representation.

### Identity and event semantics

Introduce incarnation-safe object references and LKI, then route zone changes,
damage, life, counters, and destruction through proposed-event replacement,
commit, trigger, and state-based-action stages.

### Semantic programs and choice ABI

Grow a typed effect/condition/selector/value IR from the curated deck suite and
replace flat action enumeration with legal-by-construction offers for priority,
targets, modes, payment, attackers, and blockers. Preserve an adapter for the
current policy ABI long enough to compare it with structured decoding.

Project viewer-safe IR into variable-length typed programs with stable symbolic
opcodes, explicit structure and masks, a checkpoint-bound `SemanticInputSpec`,
compiler-proven budgets, and cross-language parity fixtures. Test the result in
the four-arm ladder from card-ID/legacy actions through semantic programs and a
structured decoder on held-out cards or a held-out pack. Runtime-ID permutation,
checkpoint rebinding, identity ablation, legality, overflow, throughput, and RSS
are required controls—not optional polish.

### Search branching and verification

Specify `fork`, `mark`, `apply`, `rollback`, `snapshot`, and deterministic hash
semantics. Benchmark compact full clone, compact clone plus undo, and dense
page-COW fork plus undo at realistic worker × actor × rollout loads, using total
rollout throughput and peak RSS as the decision metrics.

Build independent evidence around the kernel: reference-versus-optimized
differential execution, property/metamorphic/fuzz tests, a pinned Phase oracle
for the overlapping cards where practical, coverage/gap generation, and the
ratio of new-card content changes to kernel changes.

## Near-term sequence

1. Establish definition/state separation and clone/rollout baselines.
2. Land one incarnation/LKI vertical slice before further zone mechanics.
3. Preserve the structured-offer and decoder evidence while building the
   viewer-safe typed-program projection and checkpoint-bound input spec.
4. Run the four-arm semantic transfer ladder, including held-out content,
   identity ablation, schema-rebind controls, and complete performance receipts.
5. Add proposed events before replacement-heavy cards.
6. Run the three-way search-state benchmark before adopting HAMTs, page-COW, or
   another representation by doctrine.
7. Turn conformance gaps into a machine-generated worklist and make regression
   evidence a CI gate.

Reference designs live in `docs/research/semantic-kernel.md` and
`docs/research/manabot-vs-phase.md`. The observation-schema history and semantic
input consequences are in `docs/research/metta-observation-robustness.md`.

## Non-goals

- Comprehensive Rules coverage independent of the selected decks
- Broad Commander, multiplayer, drafting, or casual-format support
- Deckbuilding and general format-legality infrastructure in this wave
- Runtime natural-language parsing of the full card catalog
- Porting Phase's consumer AI or adopting persistent HAMTs by default
- A flag-day rewrite that suspends a playable, trainable engine
