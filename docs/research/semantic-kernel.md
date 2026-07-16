# Research: a semantic game kernel beyond Phase

Research date: 2026-07-15.

Compared revisions:

- Etude: `bbb5a0a38f8b90efeb87829b60847fb40c5d55d4` (2026-07-10).
- Phase: [`553b97bd5c9f1a28bf7a6ebe80f6cb3a0e296c0d`](https://github.com/phase-rs/phase/commit/553b97bd5c9f1a28bf7a6ebe80f6cb3a0e296c0d)
  (2026-07-15).

This is the technical follow-on to [the broad Etude/Phase comparison](etude-vs-phase.md).
Its question is not "how do we copy Phase?" It is: **which invariants should
Etude steal, which representations should it reject, and what smaller design
could support a more polished game and a more general learning system?**

## Executive conclusion

The Phase-like direction is directionally right and structurally wrong if taken
literally.

- **Typed card semantics: yes.** Do not keep semantics in card-specific Rust
  branches or strings. But compile authored definitions into a small, typed,
  immutable program rather than carrying a giant recursive AST on every runtime
  object.
- **Object incarnation: emphatically yes.** Make it stronger than Phase by
  banning bare storage IDs from rules references. A physical card, a storage
  entity, a rules object incarnation, a stack item, and last-known information
  are different identities.
- **Dynamic actions: yes, but not as a giant action enum.** The engine should
  expose a small grammar of decision schemas plus state-dependent candidates.
  The UI can build a draft interactively and the policy can decode it
  autoregressively; the completed rules action is committed atomically.
- **Clone-friendly state: yes, but persistent collections are not the default
  answer.** First remove immutable definitions from mutable state. Then use dense
  mutable data for play and batched environments, exact isolated forks at
  determinization/parallel boundaries, and reversible transactions where a
  search worker actually reuses prefixes. Benchmark compact clone, clone plus
  undo, and explicit dense page-COW for simultaneously retained positions.

The proposed destination is a **compiled semantic machine**:

1. a versioned, immutable `ContentPack` of card programs;
2. a compact, dense `MatchState` containing only facts that can change;
3. exact-incarnation references and explicit LKI snapshots;
4. a typed effect interpreter that can yield a generic `DecisionRequest`;
5. a proposed-event/replacement/commit pipeline;
6. atomic commands and factual domain events;
7. separate projections for the player experience and the learning ABI.

This is not an attempt to support arbitrary Commander. The initial content pack
can contain exactly the selected decks and tokens. That is an **admission
policy**, not an engine limitation. A later deck builder and format-legality
checker can consume catalog metadata and produce a validated match manifest
without entering the rules kernel.

The architectural maxim is: **steal Phase's invariants, not its shapes.**

## System understanding

### Architecture

#### Etude today

Etude already has the seed of a semantic machine, but four concerns are
collapsed together.

1. [`CardDefinition`](../managym/src/state/card.rs#L196) is both authoring data
   and runtime semantics. [`Card::from_definition`](../managym/src/state/card.rs#L281)
   clones names, type strings, abilities, effects, target requirements, costs,
   keywords, and rules text into every physical card.
2. [`Effect`](../managym/src/state/ability.rs#L92) is a small typed AST, but it
   mixes reusable operations (`DealDamage`, `DrawCards`, `IfKicked`) with
   card/mechanic-shaped operations (`TargetCreaturesDealPowerDamageToLastTarget`,
   `Earthbend`, `ExileUntilSourceLeaves`, `Learn`).
3. [`GameState`](../managym/src/flow/game.rs#L26) owns mutable match facts,
   immutable card definitions, event history, observation history, caches, RNG,
   and a card registry. Deriving `Clone` therefore clones much more than a
   branch actually changes.
4. [`Action`](../managym/src/agent/action.rs#L35) is dynamically enumerated at
   runtime, but the learning ABI reduces it to 14 action types, at most 32
   candidates, and two focus objects per action in
   [`ObservationEncoderConfig`](../managym/src/agent/observation_encoder.rs#L18).

Resolution is an interpreter in embryonic form. An
[`EffectFrame`](../managym/src/flow/decision.rs#L37) carries source, controller,
targets, and a queue of cloned effects. A resolving effect can yield one of five
[`Decision`](../managym/src/flow/decision.rs#L72) variants, after which the frame
is parked in `GameState`. This is a good seam. Its limitations are representational,
not conceptual:

- `primary_target()` makes target roles positional and lossy;
- control flow pushes cloned subtrees into a queue;
- `ForEachTarget` cannot suspend;
- each new choice shape needs a new `Decision`, `Action`, action-space, encoder,
  binding, UI, and resumption branch;
- most effects mutate state first and then emit a factual `GameEvent`, so there
  is no general "event as it would happen" object for replacement/prevention.

Identity is split between stable `CardId`, transient `PermanentId`, and a generic
`ObjectId`. A battlefield entry appends a new `Permanent` and obtains a fresh
`ObjectId`, which partially models incarnation, but rules references often retain
the stable `CardId` instead. Delayed triggers and exile links therefore decide
ad hoc whether they mean the physical card or one battlefield stint.

The search layout compounds clone cost. [`RolloutPool::from_game`](../managym/src/agent/rollout_pool.rs#L49)
clones once for each determinized world and then once for every
world × legal-action × rollout slot. That is a sound common-random-numbers
layout, but it currently deep-copies card semantics into every slot.

#### Phase today

Phase makes substantially more rules meaning explicit:

- [`CardFace`](https://github.com/phase-rs/phase/blob/553b97bd5c9f1a28bf7a6ebe80f6cb3a0e296c0d/crates/engine/src/types/card.rs#L128)
  distinguishes abilities, triggers, statics, replacements, casting options,
  restrictions, and other typed semantics.
- [`ObjectIncarnationRef`](https://github.com/phase-rs/phase/blob/553b97bd5c9f1a28bf7a6ebe80f6cb3a0e296c0d/crates/engine/src/types/identifiers.rs#L29)
  pairs the stable object store ID with a monotonic incarnation, and
  `ObjectIdentityBinding` adds the expected zone needed for LKI.
- [`Definitions<T>`](https://github.com/phase-rs/phase/blob/553b97bd5c9f1a28bf7a6ebe80f6cb3a0e296c0d/crates/engine/src/types/definitions.rs#L31)
  stores definition vectors behind `Arc` and copy-on-write mutation, explicitly
  because AI search clones `GameState` constantly.
- `GameState` uses persistent `im::HashMap` and `im::Vector` containers. The
  source says integer hashing plus HAMT lookup had once consumed roughly 35% of
  large-board resolution CPU, motivating a deterministic `FxBuildHasher`.
- Simulation apply skips frontend-only derived-state sweeps while preserving
  rules finalization.

Those are real lessons. The literal representation has also reached the scale
where extension is expensive:

- `ability.rs` is 23,209 lines and its central `Effect` begins at line 9,451;
- `game_state.rs` is 14,500 lines; `WaitingFor` begins at line 3,707 and
  `GameState` at line 7,217;
- `actions.rs` is 1,801 lines; `GameAction` begins at line 124;
- `GameObject` is a broad 2,810-line runtime aggregate containing printed
  characteristics, live characteristics, base characteristics, presentation
  metadata, mechanic state, and shared ability graphs.

Phase's README says "pure reducers" and "immutable state," but the public
[`apply`](https://github.com/phase-rs/phase/blob/553b97bd5c9f1a28bf7a6ebe80f6cb3a0e296c0d/crates/engine/src/game/engine.rs#L176)
takes `&mut GameState`. Search clones the persistent state and mutates the clone.
That is a reasonable implementation, but it matters when deciding what to copy.

The breadth creates explicit approximation pressure in AI candidate generation:

- generic selection looks only at a pool of 12 and returns at most 64
  candidates;
- mana combination enumeration stops at 64;
- trigger ordering is exhaustive only through four triggers, then offers
  identity and reverse order;
- library-search candidate generation caps its input because validating each
  candidate clones and applies a full state;
- default search branching is five;
- repeated casts and activations have AI-only caps which the source explicitly
  admits can remove legitimate lethal lines.

These caps are not evidence of poor engineering. They are evidence that
**materializing complete combinatorial actions before evaluation does not
scale**, even when state cloning is structurally shared.

#### Two useful outside poles

The mature open-source MTG engines show the two conventional content extremes.

- [Forge's card scripting API](https://github.com/Card-Forge/forge/wiki/Creating-a-custom-set)
  expresses cards in a compact external DSL such as `DB$ Branch`, `Token$`, and
  `ValidCard$`. It wins content velocity and modifiability, at the cost of a
  stringly authoring surface and a large interpreter vocabulary.
- [XMage](https://github.com/magefree/mage) reports more than 28,000 unique cards
  through Java card/effect classes. It wins escape-hatch expressiveness, at the
  cost of an enormous class graph and deep-copy/runtime-composition complexity.

Phase sits between them: Oracle text is lowered to a large typed Rust AST, with
Forge fallbacks. Etude should not choose one of these poles. Its selected-deck
scope makes it possible to insist on a smaller typed intermediate representation
and reject unsupported constructs at content-build time.

### Data flow

#### Current Etude flow

```text
Rust card constructor
    -> CardDefinition
    -> deep-copied Card per physical card
    -> mutable GameState
    -> enumerate Vec<Action>
    -> choose integer action index
    -> mutate state directly
    -> append GameEvent
    -> scan triggers / state-based actions
    -> flatten Observation + first 32 actions
    -> Python/Torch policy
```

This is admirably direct. Its scaling problem is that the same hand-authored
enums define rules expressiveness, product protocol, and neural ABI.

#### Current Phase flow

```text
MTGJSON + Oracle text + Forge fallback
    -> CardFace typed object graph + diagnostics
    -> copied/live GameObject definitions
    -> huge WaitingFor-specific candidate enumerator
    -> GameAction
    -> clone GameState for probe/search
    -> mutate clone through apply
    -> finalize rules state
    -> optionally derive display state
    -> heuristic policies + shallow planner
```

The parser/data pipeline is a product multiplier. The candidate/action/waiting
triple is the principal architectural tax.

#### Proposed flow

```text
hand-authored typed builder now                 catalog/importer later
              \                               /
               -> checked semantic compiler <-
               -> versioned immutable ContentPack
                              |
ValidatedMatchSpec + AdmissionManifest --------+
                              |
                        MatchState (mutable facts only)
                              |
                  legal Offer / DecisionRequest
                       /                    \
             UI ChoiceDraft          policy autoregressive decoder
                       \                    /
                         committed Command
                              |
                   begin reversible transaction
                              |
                typed program interpreter / rules loop
                              |
       ProposedEvent -> replacements/prevention -> committed mutation
                              |
             DomainEvent + triggers + SBA fixed point
                              |
                commit transaction + state hash
                         /                 \
              PresentationDelta       PerspectiveObservation
```

Only the middle—the content pack, match facts, decision protocol, commands,
and rules transaction—is authoritative. UI drafts, animation state, automatic
shortcuts, neural padding, and catalog legality are projections or clients.

### Key abstractions

#### 1. `ContentPack`: immutable semantics, once

The first and highest-return change is not a VM. It is separating definition
data from match data.

```rust
struct Game {
    content: Arc<ContentPack>,
    state: MatchState,
}

struct CardInstance {
    printed: CardDefId,
    owner: PlayerId,
    persistent: PersistentCardState,
}
```

`CardDefId`, `AbilityDefId`, `ProgramId`, `PredicateId`, and `TokenDefId` index
immutable arrays. Runtime objects carry IDs, not cloned strings or effect trees.
`ContentPack` has a schema version and content hash. Replays and snapshots name
that hash, so a semantic change cannot silently reinterpret an old game.

The pack should contain rules semantics and only the presentation metadata the
match client needs. Search/catalog fields—set, rarity, format legality, full
printing history—belong in a separate `CardCatalog`. This directly prevents the
Phase `CardFace` tendency to mix parser diagnostics, deck construction, printed
metadata, and runtime semantics.

Cards can still be authored ergonomically in Rust at first. The builder should
produce the same checked IR that a future Oracle importer would produce. The
authoring representation is allowed to be rich; the runtime representation
should be compact and closed.

#### 2. identity domains, not one universal ID

The April 17, 2026 [Magic Comprehensive Rules](https://media.wizards.com/2026/downloads/MagicCompRules%2020260417.pdf)
state in CR 400.7 that a zone-changing object becomes a new object with no
memory of its prior existence, subject to enumerated exceptions. CR 608.2h says
an effect uses current information only when the object remains in the expected
public zone, otherwise it uses last-known information.

That requires at least these domains:

```rust
struct CardDefId(u32);       // immutable semantic definition
struct EntityId(u32);        // match-local storage / physical piece
struct CardInstanceId(EntityId); // stable card identity across zones
struct Incarnation(u32);     // rules-object epoch
struct ObjectRef { entity: EntityId, incarnation: Incarnation }
struct StackRef(u32);        // exact spell/ability stack object
struct SnapshotId(u32);      // immutable LKI snapshot
```

The names can change; the separation cannot.

- `CardInstanceId` answers "which actual card is this?" Digital perpetual
  changes and ownership can follow it.
- `ObjectRef` answers "is this still the same rules object?" Targets, source
  references, attachments, combat assignments, delayed triggers, and events
  should use this by default.
- `SnapshotId` answers "what did that exact incarnation last look like in the
  expected zone?"
- explicit transition links answer the CR 400.7 exceptions where one part of a
  resolving effect may find the new object.

Phase has the right pair, but it still carries raw `ObjectId` through a great
deal of state and manually propagates optional `source_incarnation`. A stronger
Etude type rule is: **bare storage IDs never cross a rules boundary.** Looking
up `ObjectRef` validates the epoch. Following a physical card across zones is an
explicit operation with an explicit rules justification.

A Rust generational arena is useful for preventing stale IDs from referring to
reused storage; [`slotmap`](https://docs.rs/slotmap/latest/slotmap/) provides
versioned O(1) keys over vector-backed storage. But its storage generation is
not CR 400.7 incarnation. A card normally stays in storage when it changes zones,
and nevertheless becomes a new rules object.

For a single-creator first implementation, a monotonically allocated,
match-local `Vec` with no slot reuse may be better than adding a dependency.
It is dense, serializable, and ABA-free within the match. Add a separate rules
incarnation either way. Revisit storage reuse only if token-heavy benchmarks
show material memory pressure.

Zone movement should be one authority:

```rust
fn move_object(
    tx: &mut Transaction,
    old: ObjectRef,
    destination: ZoneRef,
) -> ZoneChange {
    // ZoneChange contains old LKI, new ObjectRef, movement facts, and
    // the narrowly-scoped link usable by CR 400.7 exceptions.
}
```

This is much safer than returning a `CardId` and asking every mechanic to decide
whether the old or new incarnation was intended.

#### 3. typed semantic programs, not card-shaped effects

Neither an untyped scripting language nor a variant per card is desirable. The
middle is a typed, validated effect program.

The semantic type system only needs rules-domain values:

```text
Bool, Int, Amount, PlayerRef, ObjectRef, CardInstanceRef, StackRef,
ZoneRef, Mana, Color, CardType, CounterType, EventRef, SnapshotRef,
List<T>, Optional<T>
```

The instruction vocabulary should be orthogonal:

```text
query/filter/map/count
bind source/controller/target/event field/LKI
choose one/many/number/order/distribution/mode
pay cost / test permission
propose move/damage/draw/life/counter/token/mana event
install continuous/replacement/delayed-trigger effect
if / jump / for-each / call / return
```

The authoring builder can remain pleasant and strongly typed:

```rust
ability.when(etb(this()))
    .choose(target(creature().opponent_controls()), "victim")
    .then(exile(var("victim")))
    .then(until(source_leaves(), return_linked_object()));
```

The compiler resolves names to typed slots, checks every branch, proves choice
cardinalities where possible, calculates read/write/event capabilities, and
emits compact instructions plus debug symbols. Runtime frames hold a program
counter and typed locals, not a queue of cloned `Effect` subtrees:

```rust
struct EffectFrame {
    program: ProgramId,
    pc: u32,
    controller: PlayerId,
    source: SourceBinding,
    locals: SmallVec<[Value; 8]>,
    continuation: Continuation,
}
```

A choice instruction yields without special-casing the mechanic. Resumption
writes the validated result into a typed slot and advances the PC. `ForEach` can
suspend naturally because the loop cursor is frame data. Modal targets have
named roles rather than "the first target."

This is "bytecode" in the same modest sense as a database expression program,
not a general scripting VM. There is no filesystem, allocation API, dynamic
code loading, or arbitrary recursion. The closed instruction set is portable to
native and WASM, serializable, inspectable by tests, and usable as structured
input to the policy.

An escape hatch may be necessary, but it should be explicit debt:

```rust
Op::Native(NativeOpId)
```

Every native op gets a coverage owner, capability declaration, and plan for
lowering or permanent justification. It must not accept an unrestricted `&mut
MatchState`, or the semantic guarantees evaporate.

#### 4. proposed events and transactional commit

Magic replacement and prevention rules operate on an event that *would* happen.
CR 616.1 can require the affected player to choose among multiple applicable
replacements. A direct "mutate, then emit" pipeline cannot model this generally.

Use two event families:

```rust
enum ProposedEvent {
    ZoneChange(...), Damage(...), Draw(...), LifeChange(...),
    CounterChange(...), CreateToken(...), ...
}

enum DomainEvent {
    ObjectMoved { old: ObjectRef, new: Option<ObjectRef>, lki: SnapshotId, ... },
    DamageDealt(...), CardDrawn(...), SpellCast(...), ...
}
```

The rules loop is:

1. An instruction proposes a semantic event.
2. Prohibitions and applicable replacement/prevention effects are computed.
3. If the rules require a choice, the interpreter yields a `DecisionRequest`.
4. The chosen replacement transforms or consumes the proposal; applicability is
   recomputed as required.
5. The final event mutates state through a transaction and emits factual domain
   events.
6. Triggers observe the committed event and LKI; state-based actions run to a
   fixed point; priority is offered when the rules loop is quiescent.

The transaction journal records the old value of every mutated cell, vector
length, zone edit, allocation watermark, RNG position, event-log length, and
incremental hash contribution. It provides:

```rust
let mark = game.mark();
game.apply(command)?;
let score = evaluate(&game);
game.undo_to(mark);
```

[Stockfish](https://github.com/official-stockfish/Stockfish/blob/master/src/position.cpp)
uses `do_move`/`undo_move` and a linked `StateInfo` chain so search updates a
dense position incrementally rather than cloning an entire board per node. MTG
has much richer and variable-sized state, so its undo log must be generic and
heavily tested; the principle still applies.

The important hybrid is:

- clone mutable state once for each hidden-information determinization and each
  parallel root worker;
- make/unmake depth-first speculative branches inside that world;
- keep ordinary gameplay and vector environments in-place;
- take periodic full snapshots for save/load, debugging, and replay seeking.

#### Search-state architecture: safe forks outside, dense execution inside

"Monte Carlo needs cheap safe forks" is correct, but **Monte Carlo is not one
memory-access pattern**. The storage choice depends on whether the search keeps
many simulations alive concurrently, walks one tree depth-first, or performs a
long independent playout and throws it away.

Etude currently has two importantly different Monte Carlo implementations:

- [`Env::flat_mc_scores`](../managym/src/agent/env.rs#L370) holds one
  determinized world and one simulation clone at a time. It clones a world,
  clones that world for an action/rollout pair, plays the simulation to
  terminal, scores it, and drops it. Total work is `W * A * R`, but peak live
  search states are approximately the root, one world, and one simulation—not
  `W * A * R`.
- [`RolloutPool`](../managym/src/agent/rollout_pool.rs#L31) deliberately holds
  every `W * A * R` simulation alive so all active positions can be sent
  through one policy-network batch per ply. Here the number of retained states
  really does scale with `W * A * R`. The existing experiments report live
  policy batches of roughly 40-500 slots in
  [`exp-07`](../experiments/exp-07-expert-iteration.md#throughput-reality-the-honest-part).

The current `Game` fork is *dense-ish*, not yet optimized for either path.
[`GameState`](../managym/src/flow/game.rs#L26) uses arrays and vectors for many
hot facts, which is a good starting shape, but its derived `Clone` also copies
card definition strings/effect trees, complete event vectors, registry data,
and other non-branch facts. [`Game`](../managym/src/flow/game.rs#L94) then adds a
cloned current action space and behavior trackers. Dense layout only becomes a
search optimization after immutable content and presentation/history are
removed from the mutable snapshot.

The current clone surface makes the distinction concrete:

| Current field family | Branch relevance | Redesign treatment |
|---|---|---|
| `cards: CardVec<Card>` | Owner and physical identity matter; names, text, types, abilities, and effect trees do not change | Dense entity facts hold `CardDefId`, owner, and incarnation; all printed/compiled semantics live in `ContentPack` |
| `card_registry: CardRegistry` | Used to instantiate tokens/copies; its `BTreeMap<String, CardDefinition>` is not match state | One shared immutable content lookup keyed by compact IDs, never cloned per game or simulation |
| `permanents`, `card_to_permanent` | Tapped, damage, counters, controller, and current incarnation are hot branch facts | Keep compact and direct-indexed; this is the genuinely search-friendly part of today's shape |
| player `deck` and `name` | The deck vector is populated during setup and then duplicates identity already represented in zones; display name is immutable | Move manifest/display data out; retain only life, mana, loss flags, and rules history in the mutable player record |
| ordered zones and reverse zone lookup | Library/hand order and zone membership are essential and frequently changed | Keep dense ordered IDs plus direct reverse membership; replace linear `retain` removal only if profiles justify position indices or gap structures |
| stack, combat, triggers, delayed links, suspended resolution | Essential continuation state | Keep in compact state, but refer to immutable programs and exact object incarnations rather than cloned effect subtrees |
| `events`, `pending_events`, `observation_events` | Pending trigger facts may be semantic; presentation/observation delivery is not always part of a branch | Keep the minimum committed/pending semantic ledger and transaction lengths; derive or suppress product projections in search |
| RNG, ID generator, incarnation/allocation state | Essential for reproducibility and safe object identity | Fork explicitly and restore exactly on rollback; separate determinization seed from rollout/rules RNG |
| current action space and behavior trackers on `Game` | Action offers are derivable except for real suspended choices; trackers are experiment telemetry | Recompute/cache offers by revision, store only semantic continuations, and keep telemetry outside branch state |

The desired ownership layers are:

```text
Arc<ContentPack>                         shared by everything; never mutated
        |
CanonicalMatch                          authoritative product state
        |
        +-- exact fork + determinize --> SearchWorld 0
        |                                  +-- dense worker + rollback
        |                                  +-- dense worker + rollback
        |
        +-- exact fork + determinize --> SearchWorld 1
                                           +-- dense worker + rollback
                                           +-- dense worker + rollback
```

The outer fork establishes isolation and a hidden-information world. The inner
state is owned, compact, and mutable. A worker may then use marks and rollback
where its search traversal actually reuses a prefix. This is not a compromise
between "mutable" and "immutable" state: it assigns each technique to the
scope where it pays.

##### A precise fork and rollback contract

The contract should be designed before choosing its storage implementation:

```rust
trait BranchState: Sized {
    type Mark: Copy;

    /// Exact logical snapshot with no shared mutable rules data.
    fn fork_exact(&self) -> Self;

    /// Hidden-information sampling is explicit, not a side effect of cloning.
    fn determinize(&mut self, viewer: PlayerId, seed: SearchSeed);

    /// Starts a nested, LIFO speculative scope in this owned branch.
    fn mark(&mut self) -> Self::Mark;

    /// Success commits one atomic rules command to this branch. Failure is a
    /// no-op: state, RNG, events, allocations, offers, and hashes are unchanged.
    fn apply(&mut self, command: &Command) -> Result<ApplyResult, RulesError>;

    /// Infallibly restores the exact logical state at `mark`.
    fn rollback(&mut self, mark: Self::Mark);

    fn semantic_hash(&self) -> StateHash;
}
```

The important guarantees are stronger than "the board looks the same":

1. `fork_exact` produces the same semantic hash, legal offers, viewer
   projection, pending decision, event boundary, and RNG state as its source.
   Mutating either fork can never mutate the other. Immutable `ContentPack`
   pages may be shared.
2. Determinization and rollout reseeding are separate operations. This keeps
   world sampling, common-random-number pairing across root actions, and rules
   randomness auditable rather than hiding them in `Clone`.
3. Marks are branch-local, revision-bound, nested, and LIFO. Using a mark from
   another branch or across a committed external revision is an error in debug
   builds. Search-local revision counters may roll back; externally published
   canonical revisions advance only on a real commit.
4. `rollback(mark)` restores entities and incarnations, exact zone order,
   stack/flow frames, pending choices and triggers, allocation/free-list
   watermarks, RNG position, event lengths, caches or cache-invalidity bits,
   state-hash contributions, and every rules-relevant history fact.
5. Every mutation enters through a journaling-aware authority. Returning raw
   `&mut Vec<_>` or `&mut EntityState` from the kernel would let a future card
   silently bypass rollback.
6. Canonical gameplay does not speculatively mutate and roll back client-visible
   state. Search branches use the contract; the product path commits atomic
   commands and publishes domain events normally.

The journal should start boring and explicit. Likely record families include
scalar before-images, entity-cell before-images, inverse ordered-zone edits,
vector old lengths, allocation watermarks, RNG checkpoints, flow-frame edits,
and event-log truncation points. Derived caches should preferably be invalidated
and recomputed; if a cache affects semantics or action ordering, its state must
also be restored. Copy-on-first-write dirty bits can coalesce repeated writes to
one cell within a transaction later, but they complicate nested marks and should
earn their way through profiling.

##### Which search shape wants which primitive?

| Search shape | Live-state shape | Likely best primitive | Why |
|---|---:|---|---|
| Current random flat MC | One world and one long playout | compact clone or checkpoint reset | A terminal playout mutates much of the state once. A full undo journal can record more bytes than the compact starting state and taxes every forward mutation. |
| One-ply probes / tactical ordering | One root, many short siblings | mark/apply/evaluate/rollback | Prefix reuse is high and each speculative delta is small. |
| MCTS / depth-first tree traversal | One mutable path per worker; tree stores statistics | exact worker fork plus make/unmake | The same prefixes are revisited many times; tree nodes need actions, priors, values, visits, and possibly belief metadata—not full `MatchState` copies. |
| Root-parallel or information-set search | One state per determinization/worker | exact compact fork, then local rollback | Hidden worlds and threads require hard isolation; rollback is useful only inside each owned world. |
| Current batched policy-rollout MC | `W * A * R` positions suspended between network calls | compact clones or explicit dense page-COW | All positions must coexist to preserve inference batching. Undo inside a slot does not eliminate the slot itself. |
| Beam/frontier search | A retained set of sibling leaves | compact clone or page-COW, with local rollback while expanding | The frontier is persistent by algorithm; make/unmake alone cannot represent all retained leaves. |

This table changes the default conclusion. Make/unmake is compelling for a
future MCTS worker, but it is **not automatically better for today's flat
terminal playout**. A 57-ply rollout that changes zones, events, triggers, RNG,
and allocations may build a journal larger than a compact snapshot. Likewise,
rollback cannot collapse `RolloutPool` to one state without also giving up its
central feature: simultaneous neural inference. For that path, cheap retained
forks or bounded replay/checkpointing matter more.

##### The three implementations to benchmark

The same public contract should support three deliberately different storage
strategies.

**1. Compact full clone (reference and serious contender).**

- `Arc<ContentPack>` and other immutable tables are shared.
- `MatchState` owns only mutable facts in dense arrays/vectors.
- `fork_exact` eagerly copies that compact state; stepping is ordinary in-place
  mutation with no journal tax.
- Sequential flat MC clones or resets one simulation per playout. Batched
  policy MC owns one compact clone per live slot.

This is the correctness oracle and should remain available even if another
strategy wins. If the mutable snapshot is tens of kilobytes and long rollouts
dirty most of it, a linear copy can beat pointer-heavy structural sharing and a
large inverse log. Modern memory bandwidth makes "copy the small thing" a real
design, not a fallback.

**2. Compact clone plus undo within each worker.**

- Fork a compact state once per determinization and parallel worker.
- Place a mark at the reusable root or tree prefix.
- Apply commands in-place, evaluate or descend, then roll back before exploring
  the next sibling.
- Keep the search tree separate from rules state. It stores statistics and
  semantic action identifiers, not snapshots.

This should dominate when branch deltas are small and prefixes are reused
often. It may lose on terminal rollouts because every mutation pays a journal
write and the journal high-water mark grows with path length. A useful variant
for flat playouts is *checkpoint reset*: keep one compact root image per worker
and overwrite/reset the worker after each long rollout instead of replaying a
huge inverse journal. Clone+undo and checkpoint reset should therefore be
reported separately even if they share most machinery.

**3. Explicit dense page-COW fork plus local undo.**

- Split mutable state into semantic chunks/pages: a small scalar header,
  entity-state pages, ordered-zone pages, flow/stack pages, and rare history
  pages. A fork copies a small page table and shares pages by reference.
- First write to a shared page copies that page; later writes are dense and
  in-place. An `Arc<Vec<EntityState>>` that clones the entire entity vector on
  first write is component-COW, not page-COW; both may be worth measuring, but
  they have different copy granularity.
- Inside an owned/uniquely referenced page, the same mark/rollback journal can
  reuse prefixes.

Use explicit application-level COW, not operating-system `fork`. OS COW is a
poor portability contract for threads, Windows, sandboxed browser/WASM, and
deterministic memory accounting. Explicit pages can preserve the same semantic
fork behavior in native and WASM builds; the ownership wrapper may be `Arc` in
a threaded native worker and `Rc` in a single-threaded WASM instance. The costs
are reference-count traffic, page-table
indirection, first-write page copies, possible false sharing between unrelated
facts on one page, and more complicated allocation/reclamation. Long rollouts
may eventually dirty every hot page, converging toward a full clone plus COW
overhead; early shallow branches and hundreds of simultaneously retained slots
are its best case. Rust's official [`Arc::make_mut`
documentation](https://doc.rust-lang.org/std/sync/struct.Arc.html#method.make_mut)
is the small-scale version of this mechanism—it clones an inner value only
when shared—and also notes the cost of atomic reference counting. A real
page-COW state would apply that ownership pattern to deliberately sized chunks,
not blindly wrap the whole `MatchState` in one `Arc`.

Page boundaries should follow mutation locality before they follow hardware
page size. For example, printed definition IDs never belong beside tapped,
damage, controller, counters, and incarnation fields; ordered library contents
should not share a page with frequently changing battlefield flags. Start with
component-level chunks that remain easy to inspect, record pages-copied per
command, and split only demonstrated hot chunks.

Existing measurements give a useful prior, not a verdict. In
[`exp-07`](../experiments/exp-07-expert-iteration.md#throughput-reality-the-honest-part),
engine-random tails cost roughly 0.2 ms per playout, while each batched policy
ply cost roughly 9-16 ms and inference dispatch dominated. That means a state
optimization could materially accelerate the current all-Rust flat search yet
barely move policy-search latency; for policy search its first win may instead
be lowering RSS enough to retain a larger, better-utilized batch. Definition
separation may also change the ratios substantially, so the redesigned state
must be remeasured rather than extrapolated from today's clone-heavy state.

##### Benchmark the search, not just `Clone`

Microbenchmarks for snapshot bytes, `fork_exact`, `mark`, `apply`, and
`rollback` are diagnostic. The decision benchmark must reproduce real search
shape. Use deterministic position fixtures from the admitted matchups:

- opening positions with small action spaces and almost no mutable history;
- interactive midgames with a stack, targets, triggers, and 8-16 plausible
  offers;
- token/counter-heavy and zone-order-heavy late games;
- a suspended mid-resolution choice;
- the largest admitted legal-offer frontier, even after the fixed 32-action ABI
  is removed.

Run at least these workload families:

1. **Sequential random flat MC:** the historical strength ladder of `N = W*R`
   equal to 16, 64, and 256 simulations per action, with the current `R = 4`.
   Current shapes are therefore `(W,R) = (4,4), (16,4), (64,4)`; sweep actual
   action count `A` rather than assuming the encoder cap.
2. **Simultaneous policy rollouts:** `N = 8, 16, 64`, normally `R = 1`, retaining
   every `W*A*R` state between policy calls. Include both policy-to-terminal and
   the deployed hybrid `K = 8` policy plies followed by random tails.
3. **Tree-shaped synthetic consumer:** 64, 256, and 1,024 visits with measured
   Etude legal offers and path lengths, one state per worker, 1/4/8 workers.
   This does not claim MCTS strength; it measures whether prefix reuse makes the
   transaction machinery valuable.
4. **Cross-game teacher batching:** multiple root games whose live leaves are
   co-batched, because [`pooled_datagen`](../../manabot/sim/pooled_datagen.py#L85)
   already makes cross-game inference throughput part of the actual system.

For each cell, report:

- end-to-end decisions/second and p50/p95/p99 decision latency;
- completed playouts/second and rules transitions/second;
- policy batch size, inference time, rules time, projection time, and time
  waiting to form a batch;
- peak and steady-state RSS, allocator bytes/calls, bytes eagerly cloned, and
  number of simultaneously live states;
- journal entries/bytes and high-water mark per branch;
- COW pages referenced, copied, and dirtied, plus bytes copied because of page
  false sharing;
- root-fork, determinization, action-application, rollback, and checkpoint-reset
  time separately;
- identical outcome totals, event traces, final semantic hashes, and RNG
  positions across implementations.

Peak RSS matters as much as throughput. A strategy that makes one rollout 5%
faster but prevents the useful 256-500-state policy batch from fitting is a
loss. Conversely, a memory-saving COW design that adds 15% rules overhead may
be invisible when policy dispatch dominates—but only the end-to-end measurement
can establish that.

The harness must use the same position, legal root actions, determinization
seeds, rollout seeds, and chosen rollout commands for all three strategies.
Warm allocations separately; report cold pool construction and steady-state
reuse; compile all variants with the same release profile. Record the engine
commit, content hash, target, allocator, thread count, and machine. Avoid using
state-layout-specific random choices that make two variants evaluate different
games.

##### Correctness gate and decision rule

Undo is a semantic feature with a performance hypothesis, not merely an
optimization. Differential tests should execute the compact-clone reference and
the optimized branch in lockstep:

1. clone root; mark optimized state;
2. apply the same generated legal command sequence to both;
3. after every command compare semantic hash, legal offers, all viewer
   projections, committed domain events, pending decisions, and RNG position;
4. roll the optimized state back through every nested mark;
5. compare it byte-for-byte where layout permits and semantically everywhere;
6. replay the same continuation from the restored state and require identical
   results.

Fuzz failed commands, replacement choices, token creation/removal, zone changes
that alter incarnation, shuffles, simultaneous events, cache invalidation,
allocation reuse, and rollback from every yielded `DecisionRequest`. Run the
suite under sanitizers/Miri where practical. One forgotten mutation is enough
to bias millions of training examples silently.

Pre-register a bias toward the simplest winner:

- Keep compact full clone if search is inference/action-generation bound or if
  optimized variants do not materially improve the real target workload.
- Adopt clone+undo for the worker/search shapes where it produces a substantial
  end-to-end gain (a reasonable initial bar is at least 20%) and passes the
  differential gate. It need not replace clones in flat terminal rollouts.
- Adopt page-COW only if retained-state memory is an observed constraint and it
  substantially lowers peak RSS (an initial bar might be 40% or more) without a
  meaningful end-to-end throughput regression (initially, no more than 10%).
  Tune these thresholds before seeing the result, then preserve the raw data.
- It is acceptable—and likely—that different consumers select different
  implementations behind one semantic contract.

The probable endpoint is therefore not "dense versus persistent." It is:

> immutable definitions shared globally; exact isolated snapshots at hidden
> worlds and parallel boundaries; compact dense mutation in the hot loop;
> rollback where prefixes are actually reused; and explicit page-COW only for
> workloads that must retain many related positions.

Do not build the journal first. Moving definitions behind `Arc<ContentPack>` may
remove most present clone cost. Add transaction undo when profiling a realistic
search shows that branch cloning, not inference or action enumeration, is the
bottleneck.

#### 5. a decision grammar, not an ever-growing `GameAction`

There are two different things currently called an action:

1. a **rules command**, such as cast this spell with these modes, targets,
   value of X, and payment; and
2. a **choice protocol step**, such as select another target or stop selecting.

They should not be the same type.

At priority the engine can expose offers:

```rust
struct Offer {
    id: OfferId,
    revision: StateRevision,
    actor: PlayerId,
    kind: OfferKind,       // cast, activate, play land, pass, special action
    source: Option<ObjectRef>,
    ability: Option<AbilityDefId>,
    command_schema: ChoiceSchema,
}
```

During a resolving effect it can expose the same generic choice schemas:

```rust
enum ChoiceSchema {
    One { role: RoleId, candidates: CandidateSetId, optional: bool },
    Many { role: RoleId, candidates: CandidateSetId, min: u16, max: u16,
           distinct: Distinctness, ordered: bool },
    Number { role: RoleId, min: i32, max: i32 },
    Distribution { amount: u32, buckets: CandidateSetId, constraints: ... },
    Permutation { items: CandidateSetId },
    Payment { cost: CostExprId, options: PaymentOptionSetId },
}
```

Candidates are typed references to objects, players, modes, mana units, or
semantic options. The client returns candidate IDs bound to the offer and state
revision; it cannot fabricate an object or submit a stale route.

For a human, `ChoiceDraft` lives outside canonical rules state. Clicking three
attackers does not perform three partial declare-attacker actions. It edits a UI
draft; Confirm sends one atomic `DeclareAttackers` command. Cancel discards the
draft without rules rollback.

For a policy, the same command can be decoded autoregressively:

```text
choose offer -> choose mode(s) -> choose target role 1 -> target role 2
-> choose X -> choose payment plan -> COMMIT
```

The partial decoder state belongs to policy inference, not `MatchState`, unless
the Comprehensive Rules truly pause resolution for another player's input.
This preserves atomic game semantics and keeps the RL horizon from growing with
mere UI mechanics.

This is where Phase's full-action enumeration should be rejected. AlphaStar's
policy sampled structured action arguments autoregressively and used a pointer
network over variable entities; its [paper](https://gwern.net/doc/reinforcement-learning/model-free/alphastar/2019-vinyals.pdf)
reports transformer/entity processing, autoregressive action arguments, and a
pointer network. [Pointer Networks](https://arxiv.org/abs/1506.03134) were
specifically introduced for output dictionaries whose size depends on the input.
That is almost exactly "choose one of the current legal objects."

The policy should therefore see:

- entity tokens for visible cards, permanents, stack items, players, and public
  effect objects;
- global turn/priority/mana/history tokens;
- definition/program embeddings shared by every instance of a card or ability;
- dynamic offer and candidate tokens with legality masks and semantic roles;
- pointer heads for object/player choices and small categorical heads for modes,
  booleans, or bounded numbers.

No legal command is truncated because it was candidate 33. Combinatorial
choices are generated as a sequence, not materialized as every subset or
permutation. Training records the log-probability and entropy of each argument,
while the environment applies the completed command once.

[OpenSpiel's `State` API](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/spiel.h#L3454-L3482)
provides the simpler baseline: `Clone()` returns a copy, `Child(action)` clones
then applies, and the interface separately permits a game-specific fast
`UndoAction`. Etude needs a richer structured command than OpenSpiel's integer
action, but should retain the clean distinction between a game state, its fork
and rollback capabilities, and the algorithm consuming them.

#### 6. state layout and zones

A plausible compact state is:

```rust
struct MatchState {
    revision: StateRevision,
    turn: TurnState,
    players: SmallVec<[PlayerState; 2]>,
    entities: Vec<EntityState>,
    locations: Vec<Location>,
    zones: ZoneStore,
    stack: Vec<StackItem>,
    flow: FlowState,
    continuous: ContinuousEffectStore,
    delayed: DelayedEffectStore,
    rng: RngState,
    fast_hash: u64,
}
```

Use structure-of-arrays only where profiling supports it. A compact array of
small `EntityState` records is easier to reason about than an ECS and is likely
more than fast enough for two curated decks. Separate cold/variable data—large
counter maps, perpetual modifications, remembered sets—from hot fields through
small handles.

Zones do not all have the same semantics:

- library, stack, and sometimes graveyard require stable order;
- battlefield and hand generally do not have rules-significant display order;
- exile can require groups, links, face state, and ordering provenance.

Use explicit zone implementations or capabilities rather than one vector API
that pretends all zones are equivalent. Ordered zones can remain `Vec<ObjectRef>`;
their modest O(n) middle removal is usually a good simplicity trade. Set-like
zones can use dense vectors plus reverse positions and swap-remove. UI ordering
is presentation state and should not become a `GameAction` as it does in Phase.

Etude's current `ZoneManager` uses `retain` for every removal, scans the full
zone, and stores only the zone—not the position—in its reverse index. This is
not urgent at 60-card decks, but the new identity migration is the right time to
centralize zone invariants and add O(1) location lookup.

#### 7. product and learning projections

The authoritative engine should not serialize its entire internal state to the
UI and ask React to infer what happened. It should produce:

```text
ViewSnapshot(viewer) + PresentationDelta(domain events, viewer)
```

`ViewSnapshot` enforces hidden information. `PresentationDelta` contains stable
animation facts: object moved from hand to stack, damage originated here, these
objects became tapped simultaneously, this choice opened, this trigger entered
the stack. The UI owns layout, grouping, hover, drag state, animation clocks,
and uncommitted choice drafts.

This separation is how Etude can equal Phase's smoothness while remaining
narrow. Phase's simulation path already learned that frontend derivation must be
skippable during search. The cleaner endpoint is that display state is never
part of the rules aggregate at all.

The learning projection is separately versioned:

```text
PerspectiveObservationVn(content_hash, feature_schema_hash)
```

It uses the same visibility authority but returns entity/candidate tensors or
ragged buffers. Card semantics enter through `CardDefId`/`ProgramId` embeddings,
not a handful of coarse numeric attributes. This prevents two mechanically
different cards with similar type/P/T/mana value from aliasing.

#### 8. curated now, deck building and legality later

The product boundary should be three artifacts:

```rust
struct AdmissionManifest {
    allowed_decks: Vec<DeckManifestId>,
    content_pack_hash: ContentHash,
}

struct DeckManifest {
    exact_cards: Vec<(CardDefId, u16)>,
    side_material: Vec<(CardDefId, u16)>,
}

struct ValidatedMatchSpec {
    decks: Vec<DeckManifest>,
    rules_profile: RulesProfileId,
    seed: u64,
}
```

Today, the launcher exposes only manifests selected by the creator. There is no
general deck editor and no promise that arbitrary catalog cards compile.

Eventually:

- a deck builder queries `CardCatalog`, not `ContentPack` or `MatchState`;
- format legality is a pure validator over deck manifests, printing metadata,
  format rules, and ban/restriction data;
- admitted cards must additionally have compiled semantics in the exact content
  pack;
- the match engine accepts only `ValidatedMatchSpec` and remains unaware of the
  catalog UI.

This lets deck building and legality arrive later without either being a
permanent non-goal or a source of present architectural sprawl.

## Tensions

### General semantics versus a small artistic surface

A general semantic kernel does not imply a general product. The danger is
mistaking representational elegance for a promise to support every printed
mechanic. The content compiler must be allowed to reject a card, and the
admission manifest must remain hand-curated.

The reward is that the chosen decks can become deeper without each new card
requiring new UI, action, observation, and cloning machinery.

### Typed IR versus authoring velocity

Rust constructors are safe but verbose. A text DSL is fast but tends toward
Forge-style string protocols. Oracle parsing is high leverage but probabilistic
and difficult to validate.

The compromise is a checked compiler boundary: multiple authoring frontends can
target one small typed IR. Hand-authored definitions remain the truth for the
selected decks. An importer may propose IR later, but admission requires
compilation, scenario tests, and review.

### Clone simplicity versus undo complexity

Cloning a compact state is easy to make correct. Generic undo is easy to make
fast and hard to make trustworthy. Persistent HAMTs make forks cheap but add
allocation, pointer chasing, hashing, and a less cache-friendly hot path.

Therefore the sequence matters:

1. share immutable definitions;
2. compact mutable state and remove presentation/history duplication;
3. measure clone bytes and time on representative positions;
4. add transactional undo only for the search scopes that need it;
5. benchmark persistent/page-COW alternatives rather than adopting them by
   analogy.

This is not one global storage election. The detailed
[search-state analysis](#search-state-architecture-safe-forks-outside-dense-execution-inside)
predicts that compact clone may remain best for long disposable terminal
playouts, make/unmake may win for tree workers, and explicit page-COW may earn a
place only in the simultaneous policy pool. All should implement the same
semantic fork/rollback contract so the benchmark can choose per consumer.

### Atomic rules actions versus ergonomic interaction

A smooth UI wants incremental selection and undo. The rules engine wants one
validated declaration. The policy wants an autoregressive decoder. These can
share a schema without sharing state: human and policy draft outside the game,
then submit one atomic command.

Actual mid-resolution choices are different; they yield the semantic
interpreter and belong in canonical state because save/load and remote play must
preserve them.

### A small instruction set versus Magic's exceptions

An overly abstract IR becomes an unreadable programming language. An overly
concrete IR recreates Phase's giant `Effect`. The instruction set should be
chosen by semantic reuse across the selected decks, not by theoretical purity.

The compiler can macro-expand named mechanics into primitives while preserving
debug/source metadata. If a mechanic truly changes the rules machine—layers,
replacement ordering, mutate piles—it may deserve a typed kernel subsystem
rather than twenty clever bytecode instructions.

### Perfect engine semantics versus useful learning throughput

Richer identity, events, and programs add work per transition. Conversely,
semantic aliasing and action truncation make cheap transitions scientifically
misleading. The objective is not minimum nanoseconds per `step`; it is maximum
correct, learnable decisions per second.

The design should expose counters separately: rules time, legal-offer time,
projection time, inference time, clone bytes, journal bytes, and candidate
decoder steps.

## Observations

### Complexity

1. **Etude's current complexity is cross-product complexity.** Adding a
   choice-bearing mechanic touches definition enums, target derivation,
   resolution, suspended decisions, action variants, action-space generation,
   PyO3 bindings, flat encoding, model heads, and UI. A semantic program and
   generic decision schema turn much of that into data.
2. **Phase's complexity is accretive complexity.** It has strong local typing,
   but global enums and aggregates grow with the Comprehensive Rules. The result
   is safe exhaustiveness inside files too large for one creator to hold in
   working memory.
3. **Full action enumeration is the wrong combinatorial boundary.** Phase's
   documented caps show that even clone-friendly state cannot save enumeration
   of all subsets, permutations, target assignments, and payment plans.
4. **Persistent data structures move cost rather than remove it.** They lower
   fork cost but make lookup and iteration less dense. Phase's own 35% lookup
   profiling note is unusually direct evidence.
5. **A VM can reduce code size but increase semantic opacity.** Debug symbols,
   typed slots, capability summaries, source spans, and trace tooling are not
   optional. Every scenario failure should print the card program, PC, locals,
   proposal/replacement chain, and transaction edits.

### Quality

1. **Phase's incarnation work is among its most valuable rules engineering.**
   It aligns directly with CR 400.7 and CR 608.2h and prevents an entire class of
   blink, delayed-trigger, source, and LKI bugs.
2. **Etude's fresh `PermanentId` is a useful partial model, not a durable
   identity contract.** Stable `CardId` references can still accidentally reach
   a later incarnation, and `PermanentId` is tied to one representation/zone.
3. **Etude's suspension seam is good.** Parking an effect frame proves the
   engine can support serializable continuations. Replacing the queue with a
   program counter is an evolution, not a rewrite of the rules loop concept.
4. **Direct mutation plus factual events is insufficient for replacements.** A
   proposed-event layer is foundational, not polish.
5. **Definition sharing is an immediate, low-risk win.** It reduces clone cost,
   makes runtime state more honest, and creates stable definition IDs for both
   UI and neural embeddings.
6. **Phase's broad state serialization is impressive but expensive to evolve.**
   A content hash, explicit snapshot schema, command log, and migrations at the
   boundary are safer than treating every runtime helper field as public save
   format.

### Potential

The proposed kernel creates capabilities neither project currently has cleanly:

- exhaustive support closure for the selected decks without a broad card UI;
- one legal-choice schema shared by native, browser, human, bot, and tests;
- no fixed action cap and no combinatorial pre-enumeration requirement;
- learned embeddings tied to reusable semantics instead of card names alone;
- deterministic replay across native and WASM from commands plus RNG state;
- exact scenario shrinking because programs, decisions, and events are data;
- differential execution between a simple reference clone/apply engine and a
  fast journaled engine;
- future importers and deck-building tools without changing match semantics;
- search that forks hidden worlds once and explores them efficiently;
- visual polish driven by authoritative domain events rather than state diffs.

The important AGI claim is modest: this does not make an MTG AGI. It makes the
environment's ontology less hostile to one. A learner can attend to objects,
semantic roles, programs, and dynamically legal choices instead of memorizing a
fixed 32-row action table.

## Failure modes to design against

### Building an "IR" that is just the current enum serialized

If every new card still adds an instruction, decision kind, and policy head,
nothing has been gained. Track the ratio of new content to kernel changes. The
target for an ordinary supported card is zero kernel changes.

### Letting `NativeOp` become the normal path

Native operations are tempting under schedule pressure. Require capability
declarations and coverage reports, and fail content-pack builds when unapproved
native ops appear.

### Conflating policy decoding with game transitions

Autoregressive selection must not expose partial target/mode choices to an
opponent or chance, alter triggers between arguments, or multiply rewards. The
environment receives one committed command.

### Using incarnation as storage generation

Zone changes must bump rules incarnation even when storage is unchanged.
Storage deletion/reuse must invalidate storage handles even when no Magic zone
change exists. Tests must exercise both independently.

### Journaling incomplete state

An undo system that forgets RNG position, event length, caches, allocation
watermarks, LKI stores, or incremental hashes will generate plausible but false
search states. Differential clone/apply versus apply/undo tests are mandatory.

### Treating replay as "deserialize whatever `GameState` is today"

Replays should contain a content hash, engine/schema version, initial validated
match spec, seed, and committed commands. Periodic snapshots are acceleration,
not the sole record.

### Allowing display order into rules state

Hand arrangement, card grouping, stack fan-out, hover, animation phase, and
selection drafts must remain client state. Otherwise search clones UI data and
remote clients can create meaningless rules diffs.

### Premature arbitrary-card parsing

The seductive path is to build the compiler frontend before the IR has proved
itself on the selected decks. Hand-author the first content pack and use its
friction to discover the right primitives. Parsing comes after semantics.

## Verification architecture

The new kernel should be built with a reference/fast duality.

1. **Content compilation tests**
   - type-check every program and predicate;
   - validate target/choice cardinality and role binding;
   - reject unknown or unapproved native ops;
   - produce a deterministic content hash;
   - report exact supported-card closure for every admitted deck, including
     tokens and conjured/copyable definitions.
2. **Scenario traces**
   - each card has executable arrange/act/assert scenarios;
   - each mechanic has interaction matrices, especially zone changes, LKI,
     replacement ordering, copied abilities, and simultaneous events;
   - trace snapshots include program PC, locals, decision, proposed events,
     replacements, committed events, and state hash.
3. **Reference versus optimized execution**
   - the reference path clones compact state and applies commands;
   - the optimized path journals and undoes;
   - property tests generate valid command sequences and compare state, visible
     projections, events, RNG, and hashes after every step and undo.
4. **Metamorphic rules tests**
   - permuting irrelevant storage order does not change legal commands;
   - cloning an identical public position with renamed storage IDs preserves
     outcomes;
   - a zone change invalidates exact old references but preserves only explicit
     physical-card links;
   - hiding information changes observations, never authoritative legality.
5. **Cross-target replay**
   - run the same command log natively and in WASM;
   - compare committed state hashes and viewer projections at checkpoints;
   - fuzz snapshot/load/resume at every `DecisionRequest`.
6. **Learning-contract tests**
   - every engine offer has at least one encodable legal completion;
   - decoder masks never admit an illegal candidate;
   - no legal offer disappears due to tensor truncation;
   - action log-probability equals the sum of the committed argument decisions;
   - terminal and truncation boundaries are tested independently.

## Open questions

1. **IR granularity:** which selected decks define the first corpus? The right
   op vocabulary depends more on their interaction closure than on abstract CR
   taxonomy.
2. **Continuous effects:** should the first kernel include a general layer
   evaluator, or only typed continuous modifiers required by the admitted decks?
   Incarnation and proposed events should precede either choice.
3. **Cost/payment semantics:** should mana-source selection be a first-class
   strategic decision for the policy, or can a deterministic solver supply
   Pareto-distinct payment plans and hide dominated microchoices?
4. **Search shape:** will Etude remain flat determinized rollout search, move
   to deeper tree search, or use search mostly as a teacher? This decides how
   soon make/unmake earns its complexity.
5. **Program embeddings:** learn from opcode/operand structure end-to-end,
   precompute a definition encoder, or initially use trainable `CardDefId`
   embeddings plus explicit mechanic features?
6. **History semantics:** which facts are durable state ("cast this turn"),
   which are derived from a bounded event ledger, and which require explicit
   linked provenance?
7. **State hashing:** is a fast incremental hash needed immediately for
   transpositions/loop detection, or only a strong canonical checkpoint hash
   for replay verification?
8. **Snapshot compatibility:** is the intended guarantee replay compatibility
   within one content pack, across engine versions, or both? Stronger guarantees
   materially constrain the runtime schema.
9. **Native/WASM determinism:** are floating-point values completely absent
   from rules state? They should be. RNG algorithm and stream position must be
   pinned as protocol.
10. **Catalog boundary:** does the eventual deck builder show known-but-
    unsupported cards, or only the active content pack? This is a product choice,
    not an engine one.

## Recommendations

### Establish the three-boundary architecture first

**Observation**: Runtime semantics, catalog/deck metadata, and product admission
are currently easy to conflate. Phase does conflate parts of them in `CardFace`.

**Cost**: Low. Define `ContentPack`, `CardCatalog`, `AdmissionManifest`, and
`ValidatedMatchSpec` interfaces before changing resolution.

**Benefit**: Preserves the selected-decks product now and makes deck building
and legality additive later.

**Verdict**: Do first. This is the architectural expression of "exactly the
decks I select" without turning it into technical debt.

### Move immutable definitions out of `GameState`

**Observation**: Every Etude `Card` deep-clones its definition, and every
search fork clones those copies again.

**Cost**: Medium. Introduce stable definition IDs, an `Arc<ContentPack>`, and
runtime card-instance state; update call sites to resolve definitions through
the pack.

**Benefit**: Immediate clone/memory reduction, a cleaner state schema, stable
semantic identities for UI and ML, and a foundation for compiled programs.

**Verdict**: Highest-leverage implementation slice. Benchmark clone bytes and
rollout-pool construction before and after.

### Introduce exact incarnation before adding more zone-sensitive mechanics

**Observation**: Current stable-card and fresh-permanent IDs do not state which
references follow a card and which die on zone change.

**Cost**: Medium to high because targets, events, delayed triggers, exile links,
combat, stack objects, and tests must migrate together.

**Benefit**: Prevents a viral class of future correctness bugs and gives LKI a
coherent home.

**Verdict**: Do early, immediately after immutable definitions. Use exact
`ObjectRef` by default and force physical-card following to be explicit.

### Generalize the existing suspended frame into a typed program interpreter

**Observation**: `EffectFrame` plus `Decision` already demonstrates resumable
resolution, but cloned effect queues, positional targets, and mechanic-shaped
variants will not scale.

**Cost**: High. Requires an IR design, compiler/builder, interpreter, trace
format, and migration of the selected card corpus.

**Benefit**: Turns ordinary new cards into data, makes choices uniformly
serializable, improves portability, and exposes semantics to the learner.

**Verdict**: Worth a bounded prototype, not a flag-day rewrite. Choose 8-12
cards that stress modes, multiple target roles, loops, LKI, and a suspending
`ForEach`; compile and execute them alongside the existing engine.

### Replace flat action materialization with offers and a choice grammar

**Observation**: Etude truncates dynamic actions; Phase caps combinatorial
enumeration. Both are symptoms of treating a structured command as a flat list.

**Cost**: High across engine ABI, PyO3, model, training data, search, and UI.

**Benefit**: No hard action cap, better transfer across cards, shared human/AI
semantics, and tractable subsets/permutations through autoregressive decoding.

**Verdict**: This is the correct long-term AI boundary. Prototype it first for
priority offers, target choice, and declare attackers. Keep completed rules
commands atomic.

### Add a proposed-event pipeline before broad replacement effects

**Observation**: Current mutation-then-event flow cannot represent events being
replaced or prevented before they occur.

**Cost**: High because zone moves, damage, draws, life, counters, and tokens
must route through one authority.

**Benefit**: A principled replacement/prevention system, reliable trigger facts,
better presentation events, and reversible mutations.

**Verdict**: Foundational. Do not implement replacement mechanics as more
special branches around current direct mutations.

### Prefer dense state plus measured transaction undo over persistent HAMTs

**Observation**: Phase's structural sharing reduces clone cost but pays in
lookup/allocation complexity; Etude's current state is small and dense but
clones immutable baggage.

**Cost**: Low for dense compact state; high for complete undo journaling.

**Benefit**: Cache-friendly play and batched training, cheap depth-first search,
and a simpler mental model for one creator.

**Verdict**: Compact first, profile second, journal third. Establish exact forks
per hidden world/root worker, but do not assume that undo replaces every clone.
Benchmark compact full clone, compact clone plus worker-local undo/checkpoint
reset, and explicit dense page-COW plus undo on sequential flat MC, simultaneous
policy MC, and a tree-shaped consumer. Persistent containers and page-COW are
competitors, not doctrine.

### Build product polish on domain events and viewer projections

**Observation**: Reliability and animation suffer when clients infer transitions
from large mutable snapshots, while search suffers when display derivation lives
inside rules apply.

**Cost**: Medium. Define viewer snapshots, presentation deltas, and client-owned
choice drafts; add native/WASM replay tests.

**Benefit**: Smooth animation, hidden-information safety, transport
independence, replay/debug tooling, and a smaller search hot path.

**Verdict**: Required to meet or exceed Phase's experience without importing
its broad interface.

### Keep arbitrary-card parsing and general deck UI deliberately later

**Observation**: The selected-deck corpus is an advantage: it allows semantics
and experience to become excellent before content breadth creates pressure for
heuristic parsing and generic interfaces.

**Cost**: Opportunity cost from not advertising broad coverage.

**Benefit**: Preserves focus, artistic control, and trustworthy ground truth.

**Verdict**: Correct product sequence. Build the catalog/manifest seams now;
build deck construction, legality, and importers only when the kernel and chosen
experience are already excellent.

## Suggested proof sequence

This is a research sequence, not yet an implementation plan:

1. **Definition separation spike**
   - move card definitions into an immutable pack;
   - keep current `Effect` and action system;
   - measure `Game` heap size, clone latency, rollout-pool construction, and
     steps/second on small/mid/large curated positions.
2. **Identity/LKI model spike**
   - introduce `ObjectRef { entity, incarnation }` and `SnapshotId` in a small
     vertical slice;
   - test blink, dies triggers, exile-return, delayed triggers, and stale target
     invalidation;
   - decide monotonic vector versus slotmap based on measured token churn.
3. **Semantic program spike**
   - implement the minimal typed value set, 10-15 orthogonal ops, named roles,
     PC/locals frame, and generic choose instruction;
   - dual-run 8-12 adversarial cards against the existing interpreter and
     compare event traces.
4. **Offer/action-decoder spike**
   - expose ragged priority/target/combat candidates;
   - build a pointer-scoring policy head and autoregressive stop token;
   - prove every current legal action is representable with no 32-action cap;
   - keep the old PPO ABI for an A/B baseline.
5. **Proposed-event and search-state contract spike**
   - route zone changes and damage through proposals;
   - add replacement choice and specify exact fork, nested mark, atomic apply,
     rollback, RNG, allocation, event, cache, and hash invariants;
   - retain compact clone/apply as the reference executor;
   - differential-test clone/apply versus mark/apply/undo through generated
     command sequences and every yielded decision.
6. **Three-way search-state benchmark**
   - implement compact full clone, compact clone plus worker-local
     undo/checkpoint reset, and explicit dense page-COW plus local undo behind
     the same contract;
   - measure the historical `N=16/64/256` random flat-MC ladder, simultaneous
     `N=8/16/64` policy rollouts, a tree-shaped consumer, and cross-game pooled
     datagen on representative opening/midgame/late-game positions;
   - report end-to-end throughput and tail latency, peak RSS, clone/journal/COW
     bytes, allocation behavior, policy batching, and semantic equivalence;
   - select per search consumer. Adopt undo or page-COW only where the
     pre-registered end-to-end and memory bars are met.
7. **Experience vertical slice**
   - one chosen matchup, native and browser/WASM;
   - deterministic command replay, viewer snapshots, event-driven animations,
     responsive choice drafts, reconnect/load at every decision;
   - use this—not broad card count—as the bar against Phase.

The proof should end with a kill decision. If the semantic program needs a new
kernel op for most cards, or the generic decision protocol makes the UI/model
less legible, keep the successful pieces—immutable definitions, identity,
events—and stop. The goal is a better game and research substrate, not fidelity
to an architectural aesthetic.
