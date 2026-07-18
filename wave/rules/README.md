# Rules

Rules owns the semantic kernel shared by human play, replay, training, and
search. It is active again as of 2026-07-15 with a redesigned charter: make the
creator-selected decks exact, expressive, inspectable, and cheap to branch
without turning Etude into a general Magic platform.

The top-down contract is [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md).
The highest-priority Rules work is now to make managym the complete match,
Observation, replay, and possible-world authority described there. Content
breadth and representation optimization continue only through vertical slices
that exercise those shared contracts.

The prior capability-ordered content roadmap still supplies the card pressure:
UR Lessons vs GW Allies, then TLA commons, then the curated cube's named tail.
Those pools are acceptance suites for the kernel, not a reason to accumulate
one-off card handlers. The detailed audit remains in `00-pool-audit.md`.

## Architecture principles

1. One managym `MatchAuthority` owns match identity, revision, authoritative
   state, semantic Commands, committed events, replay, and exact forks.
2. Immutable, versioned `ContentPack` definitions are separate from compact
   dense `MatchState` facts.
3. Rules boundaries carry typed identity domains, especially
   `ObjectRef { entity, incarnation }`, with explicit last-known information.
4. Curated card text compiles offline to checked-in typed semantic programs;
   runtime natural-language parsing is outside the trusted path.
5. The engine exposes complete semantic `DecisionFrame` values and consumes
   atomic, revision-bound Commands. Positional action indices are private
   acceleration, never replay or cross-world identity.
6. Replacement and prevention semantics operate on proposed events before
   committed mutation; state-based actions and triggers reach an explicit
   fixpoint.
7. Canonical viewer `Observation` values compose current visible state,
   ordered visible events, and the complete current decision. Product frames,
   model tensors, and histories are derived projections.
8. managym defines viewer-relative `PossibleWorldSpace` and typed `WorldQuery`
   semantics and can materialize a legal branch. manabot owns probability over
   those worlds.
9. Optimization follows a written search-state contract and benchmarks. Dense
   state, safe forks, and undo are complementary tools.
10. Learning schemas bind symbolic semantics to checkpoints. Runtime registry
   IDs are transport, semantic programs are complete rather than silently
   truncated, and compatible table reordering does not create a new meaning.

## High-priority next tasks

These tasks take precedence over new content breadth and new branch
representations. Each must migrate a real consumer and delete its duplicate
meaning path; contract-only substrate is not complete.

### R1 — Make managym the match and Command authority

Introduce the semantic `MatchAuthority`, `DecisionFrame`, `Command`, and
`TransitionReceipt` boundary over the existing `Game`. A Command contains every
choice knowable at commitment time and validates against one exact revision.
Migrate Etude live play and one manabot execution/search path from separately
constructed offers and positional action meaning. Preserve current behavior
with characterization tests, then delete the migrated duplicate constructor.

Proof: the same recorded Command stream drives live play and deterministic
replay to byte-identical revisions, committed events, and viewer decisions;
stale, cross-match, incomplete, and tampered Commands fail closed.

### R2 — Land canonical Observation history

Make managym's canonical `Observation` contain typed identity, current
`ViewerState`, cursor-addressed ordered `ViewerEvent` values, and the current
semantic `DecisionFrame`. Define canonical serialization/digest and the
lossless per-viewer history contract. Consolidate Rust/Python tensor encoding
behind one managym implementation with an encoding receipt and hard failure on
legality- or meaning-affecting truncation.

Proof: Etude play/reconnect, manabot scalar/vector execution, replay, and one
dataset path consume equivalent Observations; viewer-equivalent states encode
identically; hidden-information and event-cursor tests pass.

### R3 — Establish the possible-world query kernel

Define the first exact `PossibleWorldSpace` for opponent-hand count hypotheses,
the typed/compositional `WorldQuery` grammar (`True`, semantic count predicate,
`All`, `Any`, `Not`), canonical query identity, the compatible-physical-deal
measure, filtering/support receipts, and deterministic world materialization.
Queries use semantic definition/type/tag identity, return explicit empty
support, preserve the viewer Observation, and cannot reveal whether actual
authority satisfies them.

Proof: `True`, `Has(Bolt)`, `Lacks(Bolt)`, land-count, complement, conjunction,
empty-support, copy-multiplicity, materialization, and no-truth-oracle cases are
covered through the real two-deck world and exact branch API.

### R4 — Prove one authority across all consumers

Complete a vertical receipt chain through Etude play, manabot conditional
search, exact replay, and Study fork/return. Every stage binds the same match,
revision, viewer, Observation, semantic Command, content, world schema, and
source identity. Search branches execute normal Commands; Study restores the
same historical Observation and cannot become a second replay engine.

Proof: one checked fixture runs end to end with zero legality, replay,
viewer-equivalence, branch-isolation, query-materialization, or receipt-link
mismatches.

## Continuing portfolio

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

## Execution order

R1 → R2 → R3 is the provider dependency order. R4 begins as soon as R1 has a
stable adapter and expands with each provider task. Intelligence may prototype
conditional planning against the adapter in parallel, but any change to
Command, Observation, possible-world, or materialization meaning remains a
Rules task.

After R1–R3 are exercised by real consumers, resume the next creator-selected
content increment, semantic-transfer experiment, and measured branch-backend
decision through the new authority. Existing incarnation/LKI, proposed-event,
structured-offer, semantic-program, and branch-driver work is retained
substrate rather than repeated as a prerequisite ladder.

Reference designs live in `docs/research/semantic-kernel.md` and
`docs/research/etude-vs-phase.md`. The observation-schema history and semantic
input consequences are in `docs/research/metta-observation-robustness.md`.

## Non-goals

- Comprehensive Rules coverage independent of the selected decks
- Broad Commander, multiplayer, drafting, or casual-format support
- Deckbuilding and general format-legality infrastructure in this wave
- Runtime natural-language parsing of the full card catalog
- Porting Phase's consumer AI or adopting persistent HAMTs by default
- A flag-day rewrite that suspends a playable, trainable engine
