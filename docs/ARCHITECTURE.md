# Etude Fantasia architecture

This is the top-down architecture for Etude Fantasia, manabot, and managym.
It names the central data structures, assigns one owner to each kind of truth,
and states the contracts that play, replay, search, learning, evaluation, and
Study must share.

The architecture is anchored by three decisions from the search-and-learning
review:

> “managym should be THE rules engine, but it also should be the rules engine
> that ETUDE SPECIFICALLY needs.”

> “Observation design is incredibly important for manabot and we should take
> those as key laws of physics.”

> “There is a language for talking about the world model and a grammar for
> making queries on it to filter possible worlds.”

The intended product proof is simple to describe: at a historical decision,
Etude can show the complete play distribution under the baseline belief, “they
have Lightning Bolt,” and “they have no lands,” while preserving the exact
rules position and never revealing which condition is actually true.

## The system in one picture

```text
                         authoritative world (managym)

 ContentPack ──> MatchAuthority ──> Observation(viewer) ──> DecisionFrame
                       │                    │                     │
                       │                    │                     v
                       │                    └── history        Command
                       │                                      │
                       ├── exact replay/fork <── TransitionReceipt
                       │
                       └── PossibleWorldSpace + WorldQuery + materialization
                                                   │
                                                   v
                        belief and planning (manabot)

                     ConditionalWorldPrior / BeliefState
                                                   │
                         ┌─────────────────────────┴────────────────────────┐
                         v                                                  v
               PolicyHead + ValueHead                        Planner + LeafEvaluator
                         └─────────────────────────┬────────────────────────┘
                                                   v
                                    ConditionalStrategyResult
                                      │        │         │
                                      v        v         v
                                    shards   arena   Study evidence

                            player experience (Etude)

             transport + interaction drafts + presentation + explanation
```

The ownership rule follows the repository's naming boundary:

- **managym is the world.** It owns content meaning, match state, legality,
  semantic decisions and Commands, committed events, viewer-safe
  Observations, deterministic replay, exact forks, and the meaning of possible
  worlds and queries.
- **manabot is the agent.** It owns memory, beliefs and priors over managym
  worlds, search, policy/value models, teacher evidence, datasets,
  checkpoints, self-play, opponent selection, evaluation, and promotion
  evidence.
- **Etude is the product.** It owns transport, interaction drafts, reconnect
  behavior, presentation, Study navigation, query construction and
  explanation, and research consent. It does not reconstruct rules meaning.

Telemetry, behavior trackers, profilers, caches, and user-interface state may
observe or accelerate these authorities. They never become an alternate source
of truth.

## Current state and convergence

The labels in this document are precise:

- **Current** means present on current main.
- **In flight** means implemented or designed on an active branch but not a
  contract of main.
- **Target** means accepted architecture to which the repository should
  converge.

| Boundary | Current main | In flight | Target |
|---|---|---|---|
| Match authority | managym `Game`/`GameState` own rules mutation, while the Etude server separately owns match IDs, revisions, accepted commands, and historical decision bookkeeping. | — | One managym `MatchAuthority` owns match identity, revision, execution, committed Commands/events, replay, observation, and forks. |
| Decisions | `ActionSpace` and `step(index)` are authoritative internally. `StructuredOfferSet`/`AtomicCommand` are a narrow bridge; Etude has a separate protocol `Command` and lowering context. | Structured command and compiled-semantic work continues in Rules. | A revision-bound semantic `DecisionFrame` and atomic `Command` are the common contract. Positional indices are private acceleration only. |
| Observation | managym projects viewer-safe snapshots and recent events. Rust and Python separately encode fixed tensors with caps. | Semantic-program projection and structured decoding extend the visible input. | One canonical managym `Observation` composes identity, current viewer state, ordered event increment, and current decision. Model tensors and memory are derived. |
| Possible worlds | `Game::determinize` uniformly samples a compatible hidden hand and libraries. | **INT-9** adds exact hand-range support, public-action likelihood updates, calibration evidence, and weighted sampling. | managym defines `PossibleWorldSpace`, `WorldQuery`, the reference compatible-deal measure, and materialization. manabot owns normalized priors and learned beliefs over that domain. |
| Planning | Flat Monte Carlo and determinized PUCT search private world branches and aggregate at the root. | INT-9 improves world weighting, not tree information consistency. | A viewer-rooted `PlanningProblem` accepts a belief and queries and returns aligned full strategy distributions. Determinized PUCT remains honestly named; later information-set planners are separate implementations. |
| Learning | Teacher traces, NPZ shards, masked policy/value training, and checkpoints work end to end; stronger provenance lives mostly in experiment runners. | Visit-teacher and selected-branch work continue to harden this path. | Query-conditioned teacher rows, belief-conditioned policy/value, shared artifact manifests, and checkpoint-bound history/belief schemas. |
| Evaluation | `play_games` provides deal-diverse, seat-balanced pairwise results and Wilson intervals. | **INT-6** designs a world-pinned, paired-deal arena with registered players, a full payoff matrix, ratings, uncertainty, and promotion gates. | Development matches remain non-admission. Only the versioned arena promotes immutable player artifacts. |
| Study | Etude identifies historical decisions and can exact-fork a retained managym root, execute a normal structured command, prove sibling/source isolation, and consume a source-bound return receipt. manabot can adapt selected policy/search outputs into strict `DecisionEvidence`; exact arbitrary historical model/evidence resolution remains incomplete. | — | managym resolves the exact replay position and return authority, manabot produces attributable baseline and conditional strategy evidence, and Etude presents it without becoming another replay or belief engine. |

The current implementation is not renamed into compliance. Migration is
complete only when consumers use the target contract and the duplicate meaning
constructor is deleted.

## Central types and APIs

These are semantic contracts. Exact module placement and wire encoding may
evolve without changing their ownership.

```rust
struct MatchAuthority {
    identity: MatchIdentity,
    revision: Revision,
    game: Game,
    commands: Vec<CommittedCommand>,
    events: Vec<CommittedEvent>,
}

struct Observation {
    identity: ObservationIdentity,
    state: ViewerState,
    events: Vec<ViewerEvent>,
    decision: Option<DecisionFrame>,
}

struct DecisionFrame {
    revision: Revision,
    actor: PlayerId,
    fingerprint: DecisionFingerprint,
    offers: Vec<SemanticOffer>,
}

struct Command {
    command_id: CommandId,
    expected_revision: Revision,
    offer_id: OfferId,
    answers: Vec<SemanticAnswer>,
}

struct TransitionReceipt {
    before_revision: Revision,
    after_revision: Revision,
    command: CommandId,
    events: Vec<EventIdentity>,
    next_decision: Option<DecisionFingerprint>,
}
```

`Command` is one atomic semantic commitment containing every choice knowable
at commitment time. Casting a spell includes modes, targets, divisions,
optional costs, and payment. Declaring attackers or blockers is one complete
declaration. A later Command exists only for a genuinely later choice caused
by resolution, new information, or another player.

```rust
struct PossibleWorldSpace {
    observation: ObservationIdentity,
    history: ViewerHistoryIdentity,
    schema: WorldSchemaIdentity,
}

enum WorldQuery {
    True,
    Count {
        zone: HiddenZone,
        selector: WorldSelector,
        comparison: Comparison,
        value: u16,
    },
    All(Vec<WorldQuery>),
    Any(Vec<WorldQuery>),
    Not(Box<WorldQuery>),
}

struct ConditionalWorldPrior {
    space: PossibleWorldSpaceIdentity,
    query: WorldQueryDigest,
    measure: WorldMeasureIdentity,
    normalization: NormalizationReceipt,
}

struct BeliefState {
    space: PossibleWorldSpaceIdentity,
    model: BeliefModelIdentity,
    normalized_distribution: WorldDistribution,
}
```

A world hypothesis is the hidden semantic truth relevant to decisions, not a
serialization of every private engine byte. Hypotheses quotient irrelevant
physical-copy and hidden-order differences. A query uses stable semantic
identity such as `CardDefId`, card type, or semantic tag; it cannot name a
private physical card.

For viewer history `H_v`, the actual private world `w*` is a member of
`Omega(H_v)`. A belief `b` is a normalized distribution over that same domain.
A query `Q` denotes a subset, and conditioning is:

```text
b_Q(w) = b(w) * 1[Q(w)] / b(Q)
```

Empty support is an explicit result. It never falls back to the unconditioned
belief, and checking a query never reveals whether `w*` satisfies it.

```python
@dataclass(frozen=True)
class PlanningProblem:
    observation: ObservationIdentity
    viewer_history: ViewerHistoryIdentity
    belief: BeliefStateIdentity
    queries: tuple[WorldQueryDigest, ...]
    planner: PlannerIdentity
    evaluator: LeafEvaluatorIdentity
    budget: SearchBudget
    seed_plan: SeedPlan

@dataclass(frozen=True)
class ConditionalStrategyResult:
    actions: tuple[SemanticActionIdentity, ...]
    conditions: tuple[ConditionResult, ...]
    provenance: SearchReceipt

@dataclass(frozen=True)
class ConditionResult:
    query: WorldQueryDigest
    condition_mass: float
    policy: tuple[float, ...]
    q_values: tuple[float | None, ...]
    root_value: float | None
    uncertainty: UncertaintyReceipt
    realized_budget: RealizedSearchBudget
```

Every condition aligns the same complete semantic root action set. Student
policy distributions and temperature-normalized PUCT visit distributions are
different typed outputs. Selecting an argmax action is a downstream operation,
not the search result itself.

The durable learning and evaluation identities are:

| Type | Owner | Identity and lifetime |
|---|---|---|
| `ViewerTrajectory` | managym truth, recorded by manabot | Append-only ordered Observations, Commands, and outcomes for one match and seat. Private audit truth is a separate access class. |
| `TeacherTrajectory` | manabot | Immutable root contexts, belief/query identities, sampled-world/search receipts, full targets, played Commands, and outcome. |
| `QuerySamplerSpec` | manabot | Versioned curriculum defining which queries may become labels and how roots/conditions are balanced. |
| `DatasetManifest` | manabot | Exact shard bytes plus world, source trajectories, schemas, target semantics, split, and generator identity. |
| `CheckpointManifest` | manabot | Exact weight bytes plus model, history/memory, belief, Observation/action ABI, dataset/run, target, seed, and inference-profile identities. |
| `OpponentPoolManifest` | manabot | Immutable registered opponents and a versioned selection policy. A sampled opponent is recorded per match. |
| `PlayerRegistration` | manabot | Immutable algorithm/checkpoint bytes, world, information boundary, compute class, and seed policy. |
| `ArenaKey` | manabot | World, content suite, viewer boundary, arena/rating versions, frozen anchor cohort, and compute class. |
| `DecisionEvidence` | manabot evidence consumed by Etude | Immutable alternatives, policy/visits, values, robustness, uncertainty, and typed unavailable fields bound to one exact replay decision. |
| `StudyReturnReceipt` | Etude adapter over retained managym authority | Consuming exact return to the canonical replay decision plus the captured managym source digest and a full-clone/structured-only execution receipt; fails closed if the retained root drifts or a rejected command mutates its child. |

## Laws of physics

### 1. Authority and ownership

There is one rules and match authority. Only managym commits legal state
transitions. Etude and manabot select Commands; neither patches state, invents
legality, reconstructs viewer filtering, or interprets replay independently.
Play, replay, Study, self-play, arena matches, and search must exercise this
same authority.

### 2. Identity and versioning

Identity is typed and scoped. An action position is valid only in one exact
decision frame. An `ObjectRef` includes incarnation and fails after a zone
change. Match revision, Observation schema, world schema, planner, dataset,
checkpoint, player, arena, and evidence identities are distinct domains.

An end-to-end manifest composes those identities; it does not replace them
with one untyped hash. A world/schema change creates a new version and requires
explicit artifact migration or regeneration.

### 3. Mutability, lifetime, and transitions

`MatchAuthority` is the mutable aggregate. Its legal transition is:

```text
(revision N, DecisionFrame N) + valid Command(expected=N)
    -> committed events + revision N+1 + next Observation
```

Stale Commands fail closed. Observations, receipts, queries, manifests,
datasets, checkpoints, player registrations, arena results, and Study evidence
are immutable values. Search roots are immutable; simulations mutate isolated
child branches. A fork cannot outlive or silently detach from the authority
identity from which it was made.

### 4. Observation and history

Observation is the complete legal input for one viewer at one revision:
current visible state, ordered newly visible events, and the complete current
semantic decision. managym must not discard viewer-visible information.
Viewer-equivalent authority states produce byte-identical canonical
Observations.

Viewer history is the lossless ordered Observation/event stream from match
start, including that viewer's hand, draws, private reveals, public events,
hidden-zone counts, and prior Commands. It excludes opponent private cards,
other seats' private observations, sampled-world seeds, and audit truth.

Agent memory is not part of the match protocol. Current-only, last-N,
recurrent, compressed, range-based, and full-history representations are
manabot strategies bound to checkpoints. Dataset truth retains the raw ordered
trajectory.

### 5. Hidden information and queries

Possible-world spaces are viewer-relative and revision-bound. Every
materialized hypothesis preserves the viewer Observation and satisfies its
query. Queries are deterministic, typed, compositional, canonically
serialized, and versioned. They never address physical private IDs or expose
whether actual authority satisfies them.

An eventual LLM may construct `WorldQuery` values through a schema-constrained
tool. It cannot inspect authority state, execute arbitrary predicates, or use
validation as a hidden-truth oracle.

### 6. Beliefs and conditional strategy

managym defines world meaning and the reference measure over compatible
physical deals. manabot owns epistemic and learned distributions over that
domain. Uniform determinization is a baseline prior, not truth.

The belief head consumes viewer history only. Actual hidden truth is a
supervised calibration target, never an inference feature. Both policy and
value are conditioned on `BeliefState`; otherwise PUCT priors, leaf values,
and root strategy describe different decision problems.

Query comparisons hold the visible state, legal offers, checkpoint, planner,
budget, and seed plan fixed. Their deltas are conditional comparisons, not
causal claims and not assertions about the actual hidden hand.

### 7. Determinism and exact replay

Rules replay pins content, engine, match seed, initial state, Commands, and
schema and must regenerate the same revisions, committed events, and canonical
Observations. RNG streams are explicit and derived from recorded seeds; thread
scheduling cannot select game outcomes.

Search replay additionally pins the root, viewer history, world distribution,
queries, sampled-world seed schedule, planner/evaluator, checkpoint bytes,
budget, source/runtime, and deterministic aggregation order. If a numerical
backend cannot promise bit-exact floating-point replay, the receipt must say so
and distinguish exact engine replay, exact recorded evidence, and statistical
search reproduction.

### 8. Serialization and provenance

Every durable value has a versioned canonical serialization and digest.
Locators and paths are not identity. Canonical encodings define field order,
enum meaning, collection ordering, numeric representation, and absent versus
zero. Unknown versions fail closed.

Artifacts bind exact bytes to their complete context. Atomic file replacement
prevents torn writes; immutability additionally requires content digests,
append-only manifests or ledgers, and rejection of mutation. Verification may
recompute from recorded inputs but never manufactures missing data.

### 9. Planner honesty

Current PUCT builds a separate perfect-information tree for each sampled world
and aggregates root statistics. It is **determinized PUCT**, not ISMCTS,
public-belief search, CFR, or a proof against strategy fusion. Better beliefs
improve the worlds searched; they do not make future tree decisions obey one
information-set strategy.

Budgets are semantic. Flat Monte Carlo playouts per root action and total PUCT
traversals across worlds are not comparable integers. Results report worlds,
traversals or playouts, realized latency, inference work, and perspective.

### 10. Concurrency and performance

Full clone is the correctness reference for branches. Undo, copy-on-write, and
other backends are admissible only when representation-neutral witnesses,
legal surfaces, viewer Observations, and transitions remain equivalent.

Root state is immutable during concurrent planning. Each worker owns an
isolated branch and RNG stream. Reductions have a deterministic order.
Variable-length Observations, offers, histories, and world supports are batched
with explicit lengths/masks; no fixed-cap fast path may silently lose legality
or viewer-visible meaning.

Possible-world filtering and materialization are lazy. Exact enumeration is a
useful reference while support is small, not an API assumption. Performance
receipts report p50/p95 latency, throughput, peak RSS, support/effective sample
size, and model batching behavior at the consumer boundary.

### 11. Learning evidence

Chosen action, score softmax, PUCT visit distribution, per-action Q, root
value, and terminal outcome are different targets. Missing values are absent,
not zero. Teacher audit traces and learning shards share provenance but retain
different shapes: one optimizes replay, the other batching.

The training seed is the method-level experimental unit. More decisions or
evaluation games narrow within-run noise; they do not create independent
evidence about a learning method.

### 12. Evaluation and promotion

A player is an immutable registration, not a nickname or mutable path. Arena
ratings are local coordinates: they never cross a changed world, content
suite, information boundary, cohort, estimator, or compute class.

Seat balance is not paired-deal evaluation. Admission uses the same exact deal
with seats swapped, retains the full payoff matrix and competency/system
results, estimates uncertainty over paired deal blocks, and treats disconnected
or dominance-separated rating data explicitly. No one-off win rate, Elo-like
number, experiment report, or Study artifact promotes a checkpoint.

## End-to-end flows

### 1. Authoritative execution and Commands

1. `MatchAuthority.observe(viewer, cursor)` returns the canonical Observation.
2. Its `DecisionFrame` lists all semantic legal offers for the actor.
3. Etude, a human, or a manabot chooses an offer and supplies all currently
   knowable answers in one Command.
4. `MatchAuthority.execute(command)` checks match, revision, actor, offer,
   answers, costs, and targets immediately before mutation.
5. managym commits events, advances the revision, and returns a
   `TransitionReceipt` and next Observation.

Forced engine progress may collapse decisions, but it remains committed and
observable through events. UI interaction drafts are not Commands and do not
mutate authority.

### 2. Observations and action representation

The canonical Observation remains ragged and semantic. A versioned managym
encoder may produce tensors with lengths, masks, symbolic vocabularies, and an
encoding receipt. The receipt records original/encoded counts and any clipping;
legality-affecting or meaning-affecting truncation is a hard error.

Models score semantic offers. A positional tensor row may be used inside one
batch, but trajectory, query, search, and Study evidence use semantic offer and
Command identity.

### 3. Hidden information and belief modeling

managym derives `Omega(H_v)` from the viewer's history and all public
constraints. The first reference representation can be exact opponent-hand
count vectors plus pinned hidden facts; a completion kernel materializes
physical copies and library chance when a search branch is required.

The canonical prior is uniform over compatible physical deals. It is not
uniform over collapsed count vectors. For remaining multiplicities `N_i` and a
hand count vector `k`:

```text
P(k | Q) proportional to 1[Q(k)] * product_i choose(N_i, k_i)
```

Thus `Has(Bolt)` is precisely the normalized compatible-deal measure restricted
to worlds where the Bolt count is at least one. A learned belief reweights the
same domain using viewer history.

### 4. Branching, determinization, PUCT, and leaf evaluation

A `PlanningProblem` binds one root Observation/history and distribution.
managym exact-forks authority and materializes sampled worlds. manabot applies
only semantic Commands through the branch API.

The current planner runs independent PUCT trees per determination and aligns
semantic root actions when aggregating visits and Q values. A leaf evaluator is
either an explicitly identified rollout policy or a checkpoint-bound,
belief-conditioned policy/value model. Results retain per-world diagnostics,
sampling receipts, uncertainty, and realized compute.

A future information-set-consistent planner implements the same outer planning
contract with different declared result semantics. It does not appear as a
boolean that silently changes what PUCT means.

### 5. Teacher evidence and training labels

The initial supervised conditional teacher uses a versioned
`QuerySamplerSpec`:

- always include `True` as the baseline;
- pair useful `Q` and `Not(Q)` conditions at the same root;
- sample `Has`/`Lacks` for strategically relevant definitions;
- sample land-count buckets and stable tags such as interaction, removal,
  counter, and threat;
- reject impossible, redundant, nearly vacuous, or initially vanishingly rare
  conditions;
- reserve limited conjunctions or held-out compounds for generalization.

The teacher searches the whole conditional distribution. The recorded actual
world need not satisfy a counterfactual query. Each row records the full
aligned action target, values, condition mass, belief/prior and query identity,
search provenance, and played Command. Actual hidden truth may be retained in
access-controlled audit data to supervise belief calibration separately.

### 6. Datasets, shards, and provenance

`TeacherTrajectory` is the replayable narrative. Shards are immutable batched
projections derived from it. A `DatasetManifest` binds every shard digest,
source trajectory/root, query and target schema, world/Observation/action
identities, generator source/runtime, seeds, split, and access class.

Training and validation split at the game or trajectory level, not at random
decisions from the same game. Loaders accept an expected manifest/context and
verify exact bytes before decoding. A permissive local helper, if retained,
lives in an explicit unverified namespace and cannot feed promotion.

### 7. Student policy/value learning and checkpoints

The desired model is:

```text
viewer Observation history -> BeliefHead -> BeliefState b

current visible state + semantic offers + b -> PolicyHead pi(a | O, b)
                                             -> ValueHead  V(O, b)
```

The first conditional policy/value student may consume canonical conditional
priors before a learned belief head is ready. Later training must include
history-informed non-uniform beliefs so serving on learned beliefs is not out
of distribution. Equivalent queries inducing the same `BeliefState` must
produce the same model result; query text is provenance, not an extra strategy
feature.

The concrete belief-head output representation remains a local model decision:
the first exact world may use a fixed categorical distribution, while a
structured hypothesis scorer is the likely scaling seam. Either choice emits
the same normalized semantic `BeliefState` contract.

The diagram requires the belief path but does not yet decide that it is the
only path from history to strategy. A separate recurrent or compressed
agent-memory input may also reach policy/value if experiments justify it. That
choice remains internal to manabot and is bound to the checkpoint; it never
moves memory into the match protocol.

Checkpoints are immutable, content-addressed artifacts. Loading validates the
world, Observation/action and world-hypothesis schemas, history/memory strategy,
model architecture, target semantics, and exact bytes before constructing a
player or leaf evaluator.

### 8. Self-play and opponent selection

Every actor in self-play is a `PlayerRegistration`. An
`OpponentPoolManifest` pins candidate bytes and a versioned selection policy;
each match records the selected opponent and selection probability. A mutable
“latest checkpoint” path is never an opponent identity.

Early data generation may use teacher mirrors or true self-play. As the
population grows, the pool may include the current champion, immutable
historical checkpoints, nearby-skill opponents, search teachers, and dedicated
exploiters. Selection is a manabot training policy, not an arena rating rule or
managym behavior. No recurrent memory, belief state, branch, or RNG stream is
shared between seats or matches.

### 9. Arena evaluation, skill estimation, and promotion

Development evaluation answers a pairwise question and remains useful. Arena
admission registers a candidate under one `ArenaKey`, plays a declared frozen
cohort on exact paired deals with seats swapped, and records Commands, outcomes,
competencies, calibration, latency, throughput, and cost.

INT-6's intended estimator is a regularized Bradley–Terry-style population
model with order effects and paired-deal block bootstrap. The estimator and
prior are part of `ArenaKey`; raw matrix, connectivity, residuals, and
uncertainty remain visible. “Elo” is acceptable product shorthand only when it
does not imply portability across arenas. Promotion is an immutable receipt
over all declared strength, safety, competency, and systems gates.

### 10. Projecting evidence into Study

Etude selects a historical decision and optional typed conditions. managym
resolves its exact replay, historical viewer, Observation, and legal semantic
offers. manabot resolves the exact checkpoint/planner artifacts or reports
typed unavailability, then emits `DecisionEvidence` or a
`ConditionalStrategyResult` bound to that context.

Current main already proves the authority-private fork/execute/return half:
`StudyForkProvider` retains a managym root, clones it for exploration, checks
source and sibling isolation, and returns a consuming `StudyReturnReceipt`
bound to the captured source digest. RUL-6 measures the production seam under
sequential and retained-sibling load and makes its full-clone,
`step_structured`-only, zero-fallback execution path explicit in that receipt.
The convergence work is to move the remaining match/replay identity into
managym and attach exact conditional manabot evidence to that proven return
seam.

Etude presents full distributions, values, robustness, uncertainty, and deltas
without private sampled worlds. Queries may overlap; displays label conditional
differences rather than causal effects. Study never substitutes nearby evidence
and never reveals whether an actual hidden hand satisfies a condition.

## Duplicate abstractions and migration boundaries

| Present duplication or gap | Scaling failure | Convergence action |
|---|---|---|
| managym rules state versus Etude-owned match revision, accepted-command, and replay bookkeeping | Live play, Study, and search can disagree about the same match. | Introduce managym `MatchAuthority`; migrate Etude transport and historical consumers; delete server-side semantic/replay authority. |
| `ActionSpace` positions, narrow `AtomicCommand`, Etude protocol `Command`, and multiple offer constructors | Cross-world actions and replay depend on positional or adapter-specific meaning. | Make managym semantic `DecisionFrame`/`Command` authoritative and retain indices only in revision-private bindings. |
| Snapshot-like Observation plus duplicated Rust/Python tensor encoders | Visible history and legal actions can be truncated or drift across scalar/vector paths. | Land composite Observation and one managym encoder with explicit receipts; derive all model views from it. |
| Uniform determinizers, raw-game search access, and INT-9 range types | Belief semantics, query semantics, and world completion can fork into separate ontologies. | Establish managym `PossibleWorldSpace`/`WorldQuery`/materializer, then adapt INT-9 and all planners to manabot distributions over it. |
| Action-index keyed search trees/results | Semantic actions fail to align across conditioned worlds. | Key durable root evidence by semantic offer identity and validate every materialized world's legal correspondence. |
| Dict-like player and opponent specs | Checkpoint, information, compute, and seed identity remain implicit. | Require `PlayerRegistration`, `OpponentPoolManifest`, and arena-local immutable registrations. |
| Permissive shard/checkpoint paths with stronger checks only in runners | Safe provenance is optional and cannot compose through training, arena, and Study. | Move manifest verification into shared loaders; keep runners as orchestration, not identity authorities. |
| Caller-supplied Study identity strings | Structurally valid evidence can belong to another replay. | Resolve exact context from managym replay authority and fail with typed unavailable fields when artifacts are absent. |

The integration order is dependency-shaped:

1. managym semantic Command, composite Observation, transition receipt, and
   exact replay authority;
2. minimal exact `PossibleWorldSpace`, typed `WorldQuery`, reference deal
   measure, and materialization;
3. conditional determinized PUCT and aligned teacher evidence;
4. conditional shards and policy/value student, then the learned belief head;
5. INT-9 adaptation, self-play pools, INT-6 arena admission, and Study
   conditional evidence.

Each step migrates at least one real Etude or manabot consumer and deletes its
duplicate constructor. Contract-only substrate that no running path consumes
does not complete a step.

## Roadmaps and detailed contracts

The immediate provider work is prioritized in the [Rules
roadmap](../wave/rules/README.md) and [Intelligence
roadmap](../wave/intelligence/README.md). Rules owns changes to managym
authority; Intelligence owns manabot beliefs, planning, learning, opponents,
and evaluation. Cross-wave work is integrated through runnable consumer slices,
not by duplicating provider APIs.

Detailed existing contracts remain authoritative within their narrower scope:

- [experience protocol v1](architecture/experience-protocol-v1.md) describes
  the current and transitional Etude wire path;
- [match-state hash v1](architecture/match-state-hash-v1.md) defines one
  deterministic state-hash domain;
- [presentation runtime](architecture/presentation-runtime.md) describes the
  product projection seam;
- [search branching contract](benchmarks/search-branching-contract-v1.md) and
  [selected branch driver](benchmarks/selected-branchdriver-teacher-v1.md)
  constrain branch implementations;
- [WORLDS.md](../WORLDS.md) governs current Observation/action world freezes;
- [manabot research](../manabot/RESEARCH.md) records the broader research and
  evidence program.

When a lower-level document conflicts with this file about ownership or the
target architecture, this file governs the convergence direction. Frozen
evidence, experiment IDs, contracts, and receipts retain their historical
meaning and are never rewritten to appear compliant.

## Non-goals

- A general-purpose Magic rules library independent of Etude's selected
  worlds.
- Runtime natural-language parsing or LLM authority over legality.
- Hidden decklists, multiplayer/general-sum play, Commander breadth, or broad
  format legality in the current architecture slice.
- A flag-day rewrite or a permanent compatibility layer with two semantic
  authorities.
- Claiming that exact beliefs solve strategy fusion, that one rating describes
  every matchup, or that a valid Study schema proves exact historical evidence
  is available.
