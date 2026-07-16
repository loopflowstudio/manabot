# Research: Etude Fantasia vs. Phase as MTG AI platforms

Research date: 2026-07-15.

Follow-on: [semantic game-kernel deep dive](semantic-kernel.md), covering
typed card programs, exact object incarnation, structured dynamic actions,
transactional state, a three-way clone/undo/page-COW search benchmark, product
projections, and the eventual deck/legality seam.

Compared revisions:

- Etude Fantasia (Etude): `bbb5a0a38f8b90efeb87829b60847fb40c5d55d4` (2026-07-10).
- phase: [`553b97bd5c9f1a28bf7a6ebe80f6cb3a0e296c0d`](https://github.com/phase-rs/phase/commit/553b97bd5c9f1a28bf7a6ebe80f6cb3a0e296c0d) (2026-07-15).

Method: source inspection of both engines and AI stacks, local test execution for
Etude, inspection of Phase's committed AI baselines and workflow definitions,
and verification of the pinned Phase commit's [successful CI
run](https://github.com/phase-rs/phase/actions/runs/29439693949). Dynamic project
and coverage figures are snapshots, not timeless facts.

## System understanding

The projects overlap at the engine/search boundary but are pursuing different
primary products.

- **Phase is a broad MTG application platform.** Its center of gravity is a
  comprehensive, serializable rules state that powers native, WASM, Tauri, PWA,
  WebSocket, and P2P clients. Its AI is a product opponent layered on top: exact
  candidate generation, many tactical rules, heuristic evaluation, and shallow
  budgeted search. It has no neural policy or self-play learning loop.
- **Etude is a narrow learning research platform.** Its center of gravity is a
  cheap two-player simulator, dense tensor observations, a batched Python/Torch
  boundary, PPO/self-play, and search-to-policy distillation. Its card and rules
  world is deliberately small.

The shortest accurate verdict is: **Phase is a real, audacious platform with a
vibecoded blast radius; it is not a vibecoded disaster, and its present AI is not
an MTG-AGI approach.** Phase is currently the stronger game/product substrate.
Etude is currently the stronger learning agenda. Neither is yet a credible
general MTG-AGI substrate without major work.

### Architecture

#### Etude

Etude has four relevant layers:

1. `managym/src`: a mutable Rust rules engine and agent environment. The engine
   owns legal action enumeration, a priority/stack/event loop, hidden-information
   determinization, random playouts, rollout pools, and a sequential vector
   environment.
2. `managym/src/python`: PyO3 bindings. The hot vector path writes observations
   into preallocated NumPy buffers rather than constructing Python objects for
   each environment step.
3. `manabot/`: Gym-style observation handling, the Torch agent, PPO, per-seat
   opponent routing, flat Monte Carlo players, batched policy rollouts, behavior
   cloning, and experiment verification.
4. `experiments/`, `wave/`, and `scratch/`: an unusually explicit research
   ledger. Hypotheses, budgets, results, negative findings, and caveats are
   recorded rather than collapsed into a product claim.

The rules engine is intentionally a two-player slice. `GameState.players` is a
two-element array, `ZoneManager` stores two vectors for each zone, and
`VectorEnv` rejects configurations other than two players. The July 9 card audit
reports 55 real registered cards: 45 conformant, 9 sanctioned simplifications,
one documented behavioral deviation, and no stubs. Full Comprehensive Rules
coverage is explicitly not the goal.

#### Phase

Phase is a much broader Rust workspace plus a large React/TypeScript client:

1. `crates/engine`: the authoritative game state, reducers, rules machinery,
   Oracle-text parser, MTGJSON/Forge card-data pipeline, visibility filtering,
   legal-action candidates, and serialization/migrations.
2. `crates/phase-ai`: policy registry, deck/archetype features, state evaluation,
   combat and special-case decision makers, hidden-state sampling, beam/rollout
   search, tuning tools, and regression/performance gates.
3. `crates/engine-wasm`, `server-core`, and `phase-server`: browser and server
   adapters around the engine, including per-viewer hidden-information filtering
   and authenticated actor checks.
4. `client/`: the product UI and its transport-independent adapter layer.

The [project README](https://github.com/phase-rs/phase) describes a native/WASM
engine, multiplayer clients, deck building, feeds, and 34,300+ cards. The live
badges at research time reported [31,351 of 35,397 cards](https://data.phase-rs.dev/badges/cards.json),
[88% card coverage](https://data.phase-rs.dev/badges/coverage.json), and
[190/190 keywords](https://data.phase-rs.dev/badges/keywords.json). "Playable"
has a precise but limited meaning: a card face exists and its typed definition
contains no `Unimplemented` marker. It does not mean the card has passed an
independent rules-conformance game suite.

Phase calls its design "pure reducers" and "immutable state with structural
sharing." The important implementation-level correction is that the public
entry point is `apply(state: &mut GameState, actor, action)`: it mutates a state
in place. Search obtains value semantics by cloning before mutation. The
`im::HashMap`, `im::Vector`, and `Arc<Vec<_>>` fields make those clones
copy-on-write and structurally shared; this is not a purely functional reducer
API. The README also names `rpds` as the persistent-container library, while the
current engine depends on and uses `im`; this is a small but telling example of
documentation lag under the project's change rate.

#### Side-by-side position

| Axis | Etude | Phase |
|---|---|---|
| Primary product | Learning/search research loop | Full MTG engine and playable app |
| Rules/card scope | Two players, 55 audited real cards | Multiplayer, dynamic card DB, 31k+ marker-free cards |
| Intelligence | PPO, self-play, behavior cloning, expert iteration, flat determinized MC, learned-policy rollouts | Tactical policy ensemble, heuristic evaluation, shallow beam/rollout search, deterministic special handlers |
| Learned component | Neural policy/value trained from games and search labels | A small set of scalar evaluation weights learned from 90.4M 17Lands turn samples; optional CMA-ES tuning of heuristic parameters |
| Runtime scaling target | Many small simultaneous environments and batched neural inference | One rich game, product serialization, and many cheap branch clones |
| State representation | Dense typed vectors and two-player arrays | Dynamic object graph/state machine with persistent containers |
| AI-facing representation | Fixed-cap, perspective-relative tensors | Full typed `GameState`; no stable ML observation ABI |
| Search | Flat MC over common determinized worlds; uniform or learned-policy rollouts | Branch-capped beam/alpha-beta-like continuation search with heuristic/rollout leaves |
| Project shape | Small, single-author research repo | Very large, maintainer-led community project with releases and deployed clients |

### Data flow

#### Etude game and learning path

The engine advances a mutable `Game` until it reaches a non-trivial decision.
The observation builder projects the current perspective into players, cards,
permanents, actions, recent events, validity masks, and action-focus indices.
The native encoder defaults are:

- 60 card slots per player, 38 features each;
- 40 permanent slots per player, 24 features each;
- 32 legal-action slots, 15 features each;
- two focus-object indices per action;
- 32 recent events, seven raw numeric fields each.

The Torch agent independently embeds players, cards, permanents, and action
types. Optional global multi-head attention exchanges information across the
fixed object sequence. Each action is combined with the embeddings of up to two
objects it focuses on, then scored. The value head mean-pools the object
sequence. PPO collects environment transitions, computes GAE, and performs the
standard clipped policy/value updates.

The research path extends beyond stock PPO:

- `Game::determinize` resamples opponent hand/library and reshuffles the acting
  player's unknown library while preserving legitimately revealed top cards.
- Flat MC evaluates every root action over `W` shared determinized worlds and
  `R` playouts. Sharing each sampled world across actions gives common random
  numbers and lowers comparison variance.
- `RolloutPool` materializes `W × actions × R` cloned games, applies the root
  action, and exposes all active decisions for a single batched policy forward
  pass per rollout ply.
- Search can label self-play states for behavior cloning, and a student can
  replace uniform random rollout decisions for policy-guided MC/expert
  iteration.
- `SeatRoutedCollector` supports random, frozen-network, and true-self-play
  opponent seats without changing the PPO update machinery.

This is a recognizable AlphaZero-adjacent research loop, though the current
search is flat rather than a recursively improved MCTS policy and the imperfect
information treatment is root determinization rather than a public-belief-state
solver.

#### Phase game and AI path

Phase loads MTGJSON/Oracle text into a typed card definition through a large
`nom` parser and synthesis pipeline, with selective Forge-script fallback. At
runtime a central object store and several ordered zone vectors form the world
state. `WaitingFor` encodes the current continuation/choice. The engine builds
typed candidate actions, clone-and-applies candidates for legality, and mutates
the authoritative state only through the engine boundary.

An AI decision roughly follows this pipeline:

1. Pre-empt mechanical or information-sensitive prompts with dedicated handlers
   (combat, mulligan, tutor, mana-color selection, secret guesses, and others).
2. Ask the engine for a `DecisionContext` and legal candidate actions.
3. Validate candidates by cloning and applying them.
4. Reject tactical self-harm and attach additive priors from a registry of about
   70 named tactical policy identities: board development, timing, interaction,
   archetype payoff, mulligan, combo, tutor, X-value, stack awareness, and so on.
5. Evaluate the world with hand-engineered features. Five fields per game phase
   use scalar weights fitted from 90.4M 17Lands samples; four fields remain
   hand-tuned. Additional policy parameters can be optimized with CMA-ES.
6. At Medium and above, run shallow, branch-capped continuation search with
   heuristic and rollout-backed leaves, then sample/argmax through a
   difficulty-dependent temperature.

The native presets make the approximation explicit:

| Difficulty | Depth | Nodes | Max branching | Rollout depth/samples | Hidden samples `K` |
|---|---:|---:|---:|---:|---:|
| Medium (default) | 2 | 24 | 5 | 1 / 1 | 0 |
| Hard | 3 | 48 | 5 | 2 / 1 | 2 |
| VeryHard | 3 | 64 | 5 | 2 / 2 | 3 |
| CEDH | 3 | 96 | 5 | 2 / 2 | 3 |

Interactive native search shares a 1.5-second deadline across samples. WASM
caps depth at two, reduces node counts, and caps each worker at two hidden-world
samples. Three-to-four-player games reduce ordinary search to depth two and
`K <= 1`; five-to-six-player Medium disables search. CEDH deliberately avoids
the ordinary four-player reduction.

This is sophisticated game-bot engineering, but it is not a general learning
system. New strategic concepts usually enter as Rust code, feature extractors,
or newly tuned scalar weights. Phase's breadth comes from encoded semantics and
human/LLM-written priors, not from a policy learning how arbitrary card text
composes through self-play.

#### Hidden information

Both projects clone a complete authoritative state and determinize before some
search. Their details differ materially.

- Etude uniformly re-deals the opponent's hand/library pool, reshuffles its
  own unknown library, and pins cards exposed by an active scry/look decision.
  This assumes decklists are known and does not maintain an action-conditioned
  range, but it does remove the directly stored hidden order.
- Phase Medium uses `K=0`, which its own comments call perfect-information
  search. That is the default difficulty, so default search can see the actual
  opponent hand and library.
- Phase Hard+ replaces unknown opponent hidden-zone object faces with samples
  from a deck model and averages each root action over `K` samples. Public and
  privately looked-at cards are pinned.
- Phase explicitly leaves the AI's **own library byte-identical** in every
  sample. A player knows its hand but normally not its library order. Because
  search simulates future game transitions and draws, leaving the stored order
  intact can leak future draws. This is an inference from the determinizer and
  search path, and is a real fairness gap even at Hard+.
- If Phase's reconstructed opponent pool is shorter than the hidden slots, its
  implementation leaves trailing slots with their real identity. The code calls
  this a rare bounded residual; it is another path to partial perfect
  information under inconsistent deck data.

Phase's multiplayer transport visibility is much stronger than these AI-search
issues might imply: the engine has a dedicated per-viewer state filter with
tests for hands, libraries, reveals, face-down cards, and private choices. The
fairness problem is inside the trusted AI's forward model, not necessarily a
network information leak.

### Key abstractions

#### State and identity

Etude uses dense, typed indices:

- `CardId(usize)` indexes `CardVec<Card>`.
- `PermanentId(usize)` indexes `PermanentVec<Option<Permanent>>`.
- `ObjectId(u32)` gives user-visible/runtime objects a stable identifier.
- `card_to_permanent` maps a card to its current battlefield record.
- `ZoneManager` has seven `[Vec<CardId>; 2]` zone arrays plus a dense reverse
  `card_zones` vector.

This layout is cache-friendly and easy to encode. Zone removal calls `retain`,
so a move is O(zone length). A `Game` clone deeply clones the vectors, card
definitions, registry, events, and continuations. At 60-card, two-player scale,
that simplicity is often a win; it becomes expensive at high search breadth or
large Commander boards. Object identity does not have Phase's explicit
zone-change incarnation epoch, so future expansion of last-known-information and
blink/self-reference rules will need more identity discipline.

Phase uses a richer dynamic graph:

- `ObjectId(u64)` keys a central
  `im::HashMap<ObjectId, GameObject, FxBuildHasher>`.
- Battlefield, stack, exile, command zone, and per-player hidden zones use
  persistent `im::Vector` collections.
- Every `GameObject` carries a monotonic `incarnation` that advances on real zone
  changes. `ObjectIncarnationRef { object_id, incarnation }` distinguishes a
  new object from a previous rules incarnation while retaining a stable storage
  ID.
- Ability and baseline definition lists are `Arc<Vec<_>>`; mutations use
  `Arc::make_mut`. A `Definitions<T>` wrapper withholds ordinary public
  iteration so runtime readers must pass through the engine's
  functioning-ability gates.
- Hot integer-key lookup uses `FxBuildHasher`; source comments report ordinary
  hashing plus HAMT lookup had reached about 35% of a large-board resolution's
  CPU time.
- Dirty flags, derived-display state, indexes, caches, and a simulation-specific
  apply path avoid recomputing frontend-only projections during search.

These are real scaling choices, not decoration. Phase has optimized for cheap
branch clones of a large state and for serialized product state. Etude has
optimized for cheap dense inference input and many small environments. The two
systems therefore agree on clone-and-apply search but choose nearly opposite
physical representations.

#### Actions and combinatorics

Both engines treat legal action enumeration as engine authority, which is the
right starting point for search and learning. Both also need caps.

Etude truncates the encoded action list to 32. The engine can hold more legal
actions, but the neural agent cannot select actions after the cap. Each encoded
action contains only a 14-way action-type one-hot plus focus-object indices; it
does not directly encode costs, modes, targets beyond two focus objects, or the
effect being chosen.

Phase retains dynamically typed `GameAction` values, but its AI candidate layer
controls explosions with domain-specific caps: a 12-object selection pool and
64 selection candidates, 64 mana combinations, 16 collect-evidence
combinations, all trigger permutations only through four triggers (then identity
and reverse), and other decision-specific limits. Search considers at most five
branches per node in the normal two-player presets.

Phase also imposes AI-only behavioral caps that are not Magic rules: at most
four non-mana activations from the same source per turn and at most three casts
of a card name per turn. The comments correctly document that these can suppress
legal lethal loops such as a many-counter Walking Ballista line. These are
pragmatic bot-pathology defenses and strong evidence that the current AI is a
consumer opponent, not a complete strategy solver.

#### Card semantics and observations

Phase's card representation is vastly more scalable in breadth. Oracle text is
lowered into typed effects, triggers, statics, replacements, conditions, costs,
and target filters. Common mechanics become reusable engine patterns; a card
database can be regenerated as the parser grows. The cost is a very large AST,
parser, synthesis layer, and continuation state.

Etude's card registration is manual and audited, which produces higher
confidence over a tiny slice. The engine has a typed effect DSL, but the neural
observation does not expose it. A card tensor includes zone, ownership, P/T,
mana value, broad types, a subset of keywords, token/Ally/Lesson flags, and
ward/kicker amounts. It does **not** include card identity, registry key,
colored mana cost, Oracle text, spell effects, triggered abilities, static
abilities, or activated-ability definitions. Permanents similarly expose
current tactical characteristics but not identity/effect semantics.

Consequently, two cards with the same coarse features are observationally
aliased even if their text boxes do opposite things. Legal actions can expose
some difference through available action types and focus objects, but not enough
to make a card-general policy. The observation architecture is currently the
largest blocker to calling Etude an MTG-AGI platform.

#### Neural architecture

The current Manabot agent is deliberately modest: two-layer typed MLP
projections, optional single-block multi-head self-attention, focus-aware action
embeddings, a per-action policy head, and a mean-pooled value head. The default
object sequence has 202 positions before actions (two players, 120 card slots,
and 80 permanent slots), so attention is O(202²) per environment before batch
effects.

Padding is zeroed after attention, but `MeanPoolingLayer` averages over every
fixed slot rather than only valid objects. The value magnitude therefore mixes
game content with a fixed-cap occupancy factor. This is learnable at one stable
capacity but is an avoidable representation coupling. The agent also ignores
the encoded recent-event tensor entirely.

Phase does not have a comparable neural representation. Its AI reads the rich
authoritative state directly, which avoids semantic aliasing for handcrafted
logic but supplies no compact, versioned, perspective-safe interface for
training a learned agent.

## Tensions

- **Breadth versus semantic confidence**: Phase can classify tens of thousands
  of faces as marker-free, but that coverage metric is weaker than executable
  conformance. Etude audits and trace-tests a tiny card slice, but cannot claim
  broad Magic capability.
- **Product opponent versus AGI research**: Phase's explicit policy modules and
  special handlers make a usable bot across many mechanics. They also bake in
  the solution and create a growing manual maintenance surface. Etude asks the
  policy to learn, but its observation omits the semantics required for that
  learning to generalize.
- **Rich state versus throughput**: Phase's persistent structures make rich
  state clones cheaper, but legal-action probes, layers, and continuations still
  make a single decision expensive. Etude's small dense state is much cheaper,
  but `VectorEnv` currently iterates environments sequentially and deep-clones
  worlds during search.
- **Dynamic correctness versus stable ML ABI**: Phase can keep extending
  `GameState` and `GameAction` without a tensor contract. Etude has a stable
  fixed shape, but every new rule/card concept either aliases an old feature or
  requires an ABI/model migration.
- **Functional branding versus mutable implementation**: Phase gets value-like
  search through persistent containers and clone-before-apply, but its actual
  reducer mutates `&mut GameState`. Documentation that says "with no mutation"
  obscures performance and rollback semantics a contributor needs to know.
- **LLM velocity versus review capacity**: Phase openly recruits LLM-authored
  card contributions and grew to roughly 6,000 commits within months. Typed
  enums, CR annotations, authority gates, and CI are a serious attempt to
  industrialize that velocity. Giant modules, hundreds of open issues/PRs, and a
  maintainer-dominated history show the review and integration pressure it
  creates.
- **Search strength versus information fairness**: Phase preserves default
  Medium's perfect information as a "strength floor" and leaves the AI's own
  library order intact even when sampling opponents. Etude's root
  determinization is fairer at the immediate state level but remains a weak
  approximation to action-conditioned beliefs.
- **Reproducibility versus moving worlds**: Both projects seed their RNG and
  maintain deterministic evaluation modes. Etude's own experiment notes show
  a learned ladder rating moved when the rules engine and observation schema
  changed. Phase's rapidly changing parser/card database has the same problem at
  much larger scale; a policy or benchmark needs an engine/card-data hash, not
  only a model/version label.

## Observations

### Complexity

The complexity difference is enormous.

At the compared Etude revision, `managym/src` is about 14.9k Rust lines and
the main `manabot` package about 11.7k Python lines. The rules engine is small
enough that the state machine and all core data structures can be understood in
one focused pass. Complexity concentrates in `flow/resolution.rs`, action
generation, the observation schema, Python/Rust dual encoders, and the multiple
rollout/training drivers.

The inspected Phase Rust crates contain about 1.28M lines including tests; the
client contains about 162k TypeScript/TSX lines. Source hotspots include a
46k-line parser test file, 45k-line casting test file, 30k-line Oracle effect
parser, 25k-line synthesis module, 23k-line ability type, 23k-line effect
dispatcher, 14.5k-line `GameState`, 9.7k-line engine boundary, and 5.7k-line AI
search module. Large files are not themselves proof of bad behavior, but here
they correspond to very wide enums, continuation state, and cross-cutting
matching logic. Changes are compiler-assisted but cognitively expensive.

Phase's `GameState` is serving at least four masters at once: authoritative
rules state, serializable save format, product display derivation, and cheap AI
cloning. It carries current flow, many pending/resume variants, public-state
dirty tracking, reveal state, caches, and legacy migration fields. This is the
central architectural hotspot.

Phase's parser strategy is both its moat and its largest correctness risk. It
turns natural-language Oracle text into executable typed semantics at corpus
scale. Parser success, however, can be syntactically plausible and semantically
wrong without leaving an `Unimplemented` node. The newest pinned commit itself
is a 313-addition parser fix plus integration test for named-card tutor
alternatives, illustrating both rapid repair and the long tail of grammar
interaction.

### Quality

#### Why Phase is not a disaster

The pinned commit passed a 12-minute CI matrix covering Rust format/lint/parser
gates, two Rust test shards, card-data generation/validation/coverage, draft
smoke tests, WASM and Tauri compilation, lobby tests, and frontend lint/type/test
checks. The source contains about 18.4k Rust `#[test]`/`#[tokio::test]`
annotations and extensive CR-numbered regression comments. It has explicit
engine-authority gates, per-viewer filtering, actor authorization, deterministic
measurement mode, state migration handling, AI performance counters, and
dual MIT/Apache licensing. Those are platform behaviors, not a toy demo.

The data structures show profiling-driven engineering: persistent HAMT/RRB
containers, `Arc`-shared ability lists, faster deterministic hashing, dirty
display derivation, simulation-specific apply, session caches, and explicit
multiplayer/WASM budget reductions. Several abstractions encode invariants in
types rather than relying solely on comments, especially object incarnation and
the `Definitions<T>` functioning-ability gate.

As a project, Phase also has real adoption signals. GitHub reported the public
repository was created in March 2026; the pinned page showed roughly 6,043
commits, 77 releases, 193 stars, 125 forks, and 86 contributor records returned
by the API. The lead maintainer authored about 3,732 of roughly 6,008
first-page-attributed contributions, so it is still highly maintainer-dependent.

#### Why confidence should remain bounded

The strongest Phase coverage number is based mainly on absence of typed
`Unimplemented` markers. It cannot detect a parser that confidently lowers text
to the wrong executable meaning. Tens of thousands of unit tests are valuable,
but many are local parser/engine regression tests produced alongside the same
implementation and do not constitute an independent oracle.

The AI gate is a regression tripwire, not evidence of playing strength. Its
committed baseline has three 10-game Medium mirror matchups (red aggro,
Affinity, and Enchantress). One baseline produced four draws in ten Affinity
games, including two zero-turn results. PR CI runs only the quick 10-game
setting. The workflow comments say the 100-game full suite had not completed
under its previous 90-minute limit since at least June 13 and raises the nightly
ceiling to five hours; nightly failure opens a drift issue rather than blocking
main. The performance gate tracks operation counts well, but neither gate
measures human strength, exploitability, or generalization across the card
corpus.

The project is explicitly AI-accelerated. Its README invites users to "lend
your LLM" to implement cards, and its contributor constitution contains agent
workflows and structural gates. The concern is not that AI wrote code; it is
that review evidence scales much more slowly than the generated surface. About
1.28M Rust lines, giant authority modules, and more than 500 open issues at four
months old create a large semantic blast radius.

Therefore the fair label is **real alpha platform, unusually well-guarded for
its velocity, with uncertain corpus-level semantic confidence**. It should not
yet be treated as a verified Magic oracle.

#### Etude quality

The local Rust suite passed 205 tests at this revision (12 unit, 2 conformance,
13 engine, 162 rules, 6 scenario, and 10 search tests). The frontend passed 24
tests in five files. The Python suite could not be executed from the current
environment because `pytest` is not installed/resolvable through the uv-managed
environment; that is an environment/dependency gap, not a test failure.

The strongest part of the project is its research honesty. Experiment reports
pre-register predictions, show confidence intervals and seeds, record negative
results, tag engine-world drift, and distinguish a lead from a theorem. The
July curriculum experiment, for example, found true self-play best or tied-best
across its five small-sample columns while explicitly retaining two-seed and
moving-world caveats.

The card conformance process is also appropriately scoped: 55 registered real
cards have a committed Scryfall fixture and shell tripwire, with semantic
simplifications and the Ancestral Recall deviation named rather than hidden.

There are three important rough edges:

1. **Observation aliasing/truncation** is architectural, not cosmetic. A policy
   cannot learn arbitrary card semantics from the current features, and more
   than 32 legal actions are silently unavailable to it.
2. **The stock PPO GAE indexing appears wrong at episode boundaries.** Rollout
   collection stores `dones_buf[t]` from the transition taken at observation
   `t`, but `_compute_gae` uses `dones[t + 1]` for non-final transitions. It
   should gate bootstrap from transition `t`; the current code can bootstrap
   through an auto-reset observation and cut the trace one step late/early
   around termination. The only direct GAE test uses all-false done flags, so it
   cannot detect this. This is a source-level finding that should be verified by
   a terminal-boundary unit test before trusting PPO comparisons.
3. **Project packaging is not yet public-platform quality.** The README is
   stale relative to the uv-only contributor rules, the current dev environment
   does not provide pytest, and no license file was found. Without an explicit
   license, outside reuse/contribution is legally ambiguous.

The value head's unmasked mean and unused event tensor are smaller but concrete
model-quality debts. The sequential Rust vector loop and full deep clones are
future performance opportunities, not demonstrated current blockers at the
55-card scale.

### Potential

The projects are complementary enough that their best ideas compose:

- Phase demonstrates the rules/data substrate needed for arbitrary-deck work:
  typed Oracle semantics, regeneration from MTGJSON, explicit object
  incarnation, persistent clone-friendly state, dynamic actions, visibility,
  save/replay determinism, and broad client/server reach.
- Etude demonstrates the intelligence loop Phase lacks: perspective-relative
  observations, batched environments, self-play, learned policy/value,
  search-as-teacher data, exploitability probes, and world-tagged experiment
  evidence.

The plausible MTG-AGI architecture is not "use Phase's AI." It is
**Phase-like semantics plus Manabot-like learning**, connected by a new API:

1. A versioned, perspective-safe observation that exposes executable card and
   ability semantics, not card-name one-hots or the complete private
   `GameState`.
2. A dynamic structured action representation that can score variable choices
   without silently truncating the legal set.
3. Headless batched stepping and profiling that separates engine resolution,
   legal-action enumeration, observation projection, clone cost, and neural
   inference.
4. Engine/card-data hashes on every dataset, checkpoint, and ladder result.
5. Differential/property/metamorphic rules tests in addition to
   implementation-authored regression tests.
6. Public-belief-state or recurrent opponent modeling once the perfect-state
   policy is competent; root determinization alone will not solve strategic
   imperfect information.

Phase could support that after substantial adaptation. Its persistent state and
typed action semantics are valuable, but its current AI boundary passes a huge
trusted `GameState`, uses numerous handcrafted policies, and is optimized for
one interactive decision rather than millions of training steps. Etude could
also grow there, but its manual card registry and coarse observation would need
replacement, not incremental feature accretion.

## Open questions

- What is Phase's measured headless throughput for `legal_actions + clone +
  apply` on representative Standard, Modern, and Commander states in release
  mode? The committed perf baseline counts operations but does not report the
  training-relevant distribution.
- How often do Phase's candidate caps exclude the human-optimal action in real
  game traces? There is no corpus-level recall metric for the AI action subset.
- Has Phase been differentially tested against another independent engine or a
  judge-authored position corpus? No such evidence was found in this pass.
- What does Phase's strongest configuration score against skilled humans, a
  simple external baseline, or an adversarial best-response learner? Its mirror
  regression gate does not answer this.
- How much of Phase's 88% marker-free corpus has at least one end-to-end runtime
  trace rather than parser-only coverage?
- Does Phase's AI ever exploit its unchanged own-library order in the current
  shallow search, and how much strength disappears when that order is correctly
  resampled? The data-flow permits the leak; its empirical frequency remains
  unmeasured.
- After correcting Manabot's terminal GAE semantics, do the published PPO
  rankings and self-play lead reproduce? Shared bugs preserve arm fairness only
  imperfectly because episode lengths and terminal frequencies differ by
  policy.
- What representation should be the stable semantic unit for Manabot: normalized
  ability AST nodes, effect-graph message passing, text embeddings grounded by
  typed execution, or a hybrid? This decision dominates card-generalization
  prospects.
- Is the intended next milestone broad two-player constructed play or actual
  multiplayer/Commander? The right state and batching structures differ enough
  that this should be explicit before borrowing Phase internals.

## Recommendations

### Do not copy or port Phase's current AI into Manabot

**Observation**: Phase's strength comes from dozens of explicit tactical
policies, deterministic special handlers, scalar evaluation, and shallow search.
That improves a consumer opponent but moves intelligence into authored code.

**Cost**: Porting would import a large maintenance surface and blur Manabot's
learning objective.

**Benefit**: Some immediate play-strength and mechanism coverage.

**Verdict**: Not worth it as the intelligence direction. Use selected Phase
policies as evaluation probes, scripted curriculum opponents, or labeled
competency tests—not as the target agent.

### Fix the Manabot learning contract before scaling compute

**Observation**: Terminal GAE indexing, coarse card aliasing, fixed action
truncation, unmasked value pooling, and unused event features make current
results harder to interpret and block broad transfer.

**Cost**: Low for a terminal-boundary GAE test/fix and masked pooling; medium to
high for a structured semantic observation/action ABI and checkpoint migration.

**Benefit**: Restores confidence in PPO evidence and makes future engine breadth
usable by learning rather than merely executable.

**Verdict**: Highest-priority work. Correctness first, then semantic
representation, then more training throughput.

### Borrow Phase's identity and clone ideas selectively

**Observation**: `ObjectId + incarnation`, Arc-shared immutable card/ability
definitions, and simulation-specific display suppression directly address rules
correctness and clone cost. Etude's dense typed IDs and arrays remain simpler
for its current scale.

**Cost**: Low to adopt incarnation semantics deliberately before more blink/LKI
rules; medium to share immutable definitions; high and risky to replace all
vectors with persistent containers.

**Benefit**: Avoids identity bugs and reduces search cloning as cards/abilities
become richer.

**Verdict**: Adopt incarnation and immutable-definition sharing as needs arrive.
Do not cargo-cult persistent HAMTs without a clone/profile benchmark showing the
dense layout has become the bottleneck.

### Evaluate Phase as an alternate rules backend with a bounded benchmark

**Observation**: Phase has far more rules/card breadth, but no training-grade
ML ABI and uncertain corpus-level conformance. A wholesale engine migration
would be speculative.

**Cost**: Medium: define 50-100 representative decision states, implement a
read-only perspective projection prototype, and benchmark action enumeration,
clone/apply, memory, determinism, and batchability against managym.

**Benefit**: Replaces architectural debate with evidence and may reveal a path
to arbitrary-deck experiments years earlier than manual card registration.

**Verdict**: Worth a spike only after the Manabot learning-contract fixes. Pin a
Phase commit and card-data hash; do not build against moving `main`.

### Treat Phase coverage as a lead generator, not an oracle

**Observation**: Marker-free typed parsing is impressive but cannot identify a
confidently wrong lowering. Phase's velocity makes independent validation more,
not less, important.

**Cost**: Medium to build metamorphic and cross-engine position suites; high for
wide independent judge review.

**Benefit**: Converts breadth into trustworthy training data and prevents a
learner from mastering simulator bugs.

**Verdict**: Required before using Phase-generated games as AGI-scale ground
truth. Start with high-frequency mechanics and adversarial parser compositions,
not random cards.

### Make Etude reusable as a project

**Observation**: The repo is effectively single-author, its README is stale,
pytest is absent from the active dev environment, and there is no explicit
license.

**Cost**: Low.

**Benefit**: Clear onboarding, reproducible verification, and legal permission
for collaboration or reuse.

**Verdict**: Worth doing independently of the technical direction.
