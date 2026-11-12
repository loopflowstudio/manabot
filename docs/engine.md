# The rules engine

managym is the rules of Magic as a training environment. It runs many games at
once, decides what is legal at every point, shows each seat only what that seat
may see, and can copy, replay or re-deal a game exactly. It never chooses among
a learner's legal actions. Forced single actions are played automatically, and
every other choice comes from outside, whether from a policy, a search player,
a script or a person; a native random opponent is an explicit opt-in.

This guide is for anyone who needs to trust what the engine guarantees.
Building and testing it is in [managym/README.md](../managym/README.md); the
cross-package contract is in the
[README](../README.md#contracts-the-code-enforces).

- [One decision](#one-decision)
- [Forks, replay and determinization](#forks-replay-and-determinization)
- [Batched games](#batched-games)
- [Hand snapshots](#hand-snapshots)
- [Encoded observations](#encoded-observations)
- [Cards as programs](#cards-as-programs)
- [Worlds](#worlds)
- [What we know](#what-we-know)
- [Where the code lives](#where-the-code-lives)

## One decision

A game is a sequence of decisions. Between them the engine does everything the
rules say happens on its own: turn steps, priority passing, resolving the
stack, state-based actions and triggers.

![The engine advances the game until a seat must choose, publishes the legal actions, builds that seat's observation, receives a chosen action from outside, executes it and returns to advancing. From any state it can fork an exact clone, replay, or determinize hidden cards.](images/engine-decision.svg)

### Forced decisions

A decision reaches a player only when it has at least two legal actions. With
one legal action the engine plays it and keeps going, so a forced choice never
costs a model a forward pass and never produces a training row.

From `Game::tick` in [`managym/src/flow/tick.rs`](../managym/src/flow/tick.rs):

```rust
if let Some(space) = action_space {
    if !self.skip_trivial || space.actions.len() > 1 {
        self.publish_action_space(space);
        return false;
    }
    self.skip_trivial_count += 1;
    if let Some(action) = space.actions.first() {
        if self.execute_action(action).is_err() {
            return true;
        }
    }
    continue;
}
```

- `skip_trivial` is on everywhere by default: training, evaluation, Academy and
  benchmarks. Turning it off is for explicit debugging and the constructed-position
  diagnostic fixtures only.
- Every decision count in these docs excludes forced choices. Decisions per
  game and learner decisions both mean real choices.
- The readable conformance path surfaces every singleton action so the
  optimized path can be checked against it
  ([`conformance.rs`](../managym/src/conformance.rs)).

### Who chooses

The published decision names its chooser once, from the legal actions
themselves. Nothing downstream remaps it.

From `publish_action_space` in [`managym/src/flow/tick.rs`](../managym/src/flow/tick.rs):

```rust
self.decision_epoch = self
    .decision_epoch
    .checked_add(1)
    .expect("decision epoch exhausted");
action_space.player = action_space
    .actions
    .first()
    .map(|action| action.player())
    .map(|player| self.decision_controller(player));
```

- `Action::player()` (and `Decision::player()`) say whose rules decision it is; `decision_controller`
  resolves who controls that player's choices, which matters for effects that
  grant control.
- The epoch advances on every publication, so a Command decoded from an
  earlier set of offers fails closed instead of acting on a stale decision.
- Env dispatch, observations and Academy all read this chooser directly.

### Observations

The acting seat gets an observation of what it may see: its hand, both
boards, the stack, the pending decision, public history and the legal actions.
The opponent's hand and both library orders are absent, not masked later.

From `Observation::for_player` in
[`managym/src/agent/observation.rs`](../managym/src/agent/observation.rs):

```rust
pub fn for_player(game: &Game, viewer: PlayerId) -> Self {
    assert!(
        viewer.0 < game.state.players.len(),
        "viewer must name a player"
    );
    let expose_actions = game
        .current_action_space
        .as_ref()
        .and_then(|space| space.player)
        == Some(viewer);
    Self::for_player_with_events(game, viewer, std::iter::empty(), expose_actions)
}
```

- The engine filters hidden information. No encoder, search player or browser
  re-derives what a viewer may know.
- Legal actions appear only in the chooser's observation.
- Decklists are open by default, set on `PlayerConfig.disclosure`. A closed
  opponent's list is absent from observations, search and possible worlds alike.
- Two states that look the same to a viewer produce byte-identical
  observations, which is what makes replay and the learnability checks
  meaningful.
- The same observation feeds the policy tensor, the LLM text and Academy. A
  mechanic that reaches one reaches all three in the same change
  ([observations and the policy network](observation-and-model.md)).

Consumers read the encoded arrays by name; see
[encoded observations](#encoded-observations).

### Executing an action

Whatever chose the action, the engine checks it again before anything changes.

From `Game::step` in [`managym/src/flow/tick.rs`](../managym/src/flow/tick.rs):

```rust
let selected_action = action_space.actions[action].clone();
if let Err(error) = self.execute_action(&selected_action) {
    self.current_action_space = Some(action_space);
    return Err(error);
}

Ok(self.finish_action_step())
```

- An action is an index into the published decision. Out of range is an error,
  and a failed execution restores the decision rather than leaving the game
  half-stepped.
- `finish_action_step` runs state-based actions and advances to the next real
  decision. Structured Commands, which carry targets and payments in one step,
  share this same boundary.
- The transition is `(revision N, decision N) + Command(expected=N) →
  committed events + revision N+1`. A Command against another revision fails
  closed ([contracts](../README.md#contracts-the-code-enforces)).

## Forks, replay and determinization

Search, replay verification and hidden-information sampling all need the engine
to produce the same game again, or a copy of it, exactly.

**Fork.** A search player clones the whole game and steps the clone. From
`Env::fork` in [`managym/src/agent/env.rs`](../managym/src/agent/env.rs)
(`selected_fork`, which the Python `fork_exact` binding calls, has the same
body):

```rust
/// Independent copy of this env's game for search rollouts.
///
/// Profiling and behavior tracking are disabled on the fork; stepping the
/// fork never mutates the original.
pub fn fork(&self) -> Result<Env, AgentError> {
```

Full clone is the only branching strategy. Undo journals and copy-on-write
pages were tried, failed their preregistered thresholds and were kept out of
production; their drivers have since been deleted
([decision record](https://github.com/loopflowstudio/manabot/blob/055dddab1/docs/benchmarks/search-branching-decision-v1.md)). A
search-state witness of rules facts alone checks that a fork left its source
untouched ([`search_state.rs`](../managym/src/search_state.rs)).

**Replay.** Every evaluation game is replayed from its seed and Commands, and
only games that regenerate the same states score. From `replay_games` in
[`manabot/sim/replay.py`](../manabot/sim/replay.py):

```python
def replay_games(
    games: list[dict[str, Any]], *, require_terminal: bool = True
) -> ReplayReceipt:
    """Verify saved Commands; diagnostics may explicitly request prefix replay.

    A passing prefix receipt proves only the retained trajectory, never game
    completion. Arena callers retain the complete-game requirement by default.
    """
```

Replay pins the deal seed, the match and observation configuration, the
initial state digest and every Command, and every pre- and post-state digest
must match. RNG streams derive from recorded seeds, so thread scheduling can't
select an outcome.

The digest is `GameState::deterministic_hash()`, a versioned BLAKE3 digest of
canonical JSON over every `GameState` field plus the content pack's schema and
digest (`MATCH_STATE_HASH_VERSION`, now 1). The struct is destructured without
`..`, so a new field fails to compile until its canonical treatment is chosen;
pointers, capacities, timings and observation projections are excluded, and
the RNG is represented by eight draws from a clone, never advanced.
`cargo test --test match_state_hash_tests` in `managym/` proves clone and
allocation independence, mutation sensitivity and RNG non-mutation. Consumers
compare version and digest together; a content change changes the hash on
purpose.

**Determinize.** A search over hidden information redraws what the acting seat
cannot see and keeps everything it can. From `Game::determinize` in
[`managym/src/flow/search.rs`](../managym/src/flow/search.rs):

```rust
pub fn determinize(&mut self, perspective: PlayerId, seed: u64) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let opponent = PlayerId((perspective.0 + 1) % 2);
    assert!(
        self.decklist_known(opponent, perspective),
        "determinize requires the opponent's open decklist"
    );
```

The unseen pool is the opponent's unknown hand cards, library and face-down
foretold cards; publicly known hand cards stay put. Both library orders are
reshuffled, since the viewer knows neither, and everything the viewer can see is
preserved. [`possible_worlds.rs`](../managym/src/possible_worlds.rs) defines the
same domain exactly, with compatible-deal counts as weights, for beliefs that
want more than a uniform redraw ([beside the loop](../README.md#beside-the-loop)).

## Batched games

Training and evaluation step many games in lockstep: one array of actions in,
one batch of observations out, so a single forward pass serves every active
game. `VectorEnv` ([`managym/src/agent/vector_env.rs`](../managym/src/agent/vector_env.rs))
owns the games and their reset seeds; the Python wrapper is
[`manabot/env/vector_env.py`](../manabot/env/vector_env.py). A finished slot
is scored and reset in place. By default an observation that exceeds its
declared capacity is an error; the `first-legal` action-overflow policy and
isolated capacity recovery are explicit opt-ins.

`opponent_policy="none"` (the default) surfaces both seats' decisions to the
caller. `opponent_policy="random"` plays the opponent's decisions natively and
returns seat 0's next decision. These are the only built-in vector policies;
`"passive"` is removed and raises an unsupported-policy `ValueError`, including
through CapturedVector and Torch. Retained failure evidence keeps its original
policy name, but replaying a passive-policy receipt requires its original source.

Collection was 75–83% of active time across the profiled training runs
([training time](../experiments/training-time.md)), so engine and inference
throughput are the cost levers. CI keeps an
[engine performance budget](development.md#engine-performance-budget): a
change more than 10% slower than the reference, or one that moves the
search/decision digest, fails.

### Choosing a Python API

The [scalar/batch API audit](python-api-audit.md) records the capability
comparison, source evidence and independent follow-ups.

Install the native API as the [engine-only `managym` distribution](engine-python.md)
and use `import managym`. Installing `manabot` adds training and keeps
`from manabot import managym` as a compatibility alias to the same engine.
The Torch wrapper and the capture wrapper are separate Python classes:

| Need | API |
|---|---|
| One game, explicit reset, readable observations | `managym.Env` |
| A fixed viewer's observation | `Env.observation_for_player(player_index)` |
| Revision-bound Commands, decision captures, forks and possible worlds | Scalar `Env`; these are not general `VectorEnv` operations |
| Batched readable observations and automatic reset | `managym.VectorEnv.reset_all` / `step` |
| Caller-owned NumPy buffers, per-slot setups and partial stepping | Native `VectorEnv` buffer methods |
| Private failure replay inputs around buffered execution | `manabot.env.failures.CapturedVector` |
| Torch observation/reward tensors on a selected device | `manabot.env.vector_env.VectorEnv` |

Scalar native `Env.step` returns the terminal state and leaves resetting to the
caller. Native vector steps reset finished slots before returning: reward and
`terminated` / `truncated` describe the completed game, while its returned
observation describes the replacement game. The Torch wrapper keeps this
behavior. Its observation tensors are reused; clone anything retained across
steps. The Gymnasium wrapper `manabot.env.env.Env` never resets on its own:
a terminal `step` returns the terminal observation and true done flags, and the
caller calls `reset`. Its former `auto_reset` argument is removed; passing it
raises `TypeError`, and step info no longer carries `true_terminated` /
`true_truncated`. Vector autoreset is the only automatic-reset contract.

Torch `VectorEnv.reset(seed=s)` restarts slot `i` at seed `s + i` for any
explicit seed, including the current one, so equal seeds reproduce deals.
`reset()` without a seed continues native seed succession. Its
`current_agent_indices()` names each slot's chooser.

`CapturedVector` delegates only native reads (seeds, generations, choosers,
frames, setups, diagnostics). Readable `reset_all` / `step` and other native
mutators raise `AttributeError`; use its `reset_*_into_buffers` and
`step_*_into_buffers` methods, which keep replay prefixes aligned. It starts a
new prefix whenever the native game generation changes, so a same-seed reset
or a replacement at the saturated maximum seed never extends the old game.

Native buffered stepping and `CapturedVector.step_into_buffers` accept an
`active` boolean per slot. Inactive games and observation rows stay unchanged;
the Python buffer bindings clear their rewards and done flags to zero/false.
Apply the same mask when collecting transitions. The Torch wrapper's `step`
advances every slot and does not accept an active mask.

Strict batch errors can occur after other slots have advanced. Do not retry
the same batch assuming nothing changed. `CapturedVector` retains private
pre-call replay inputs; explicit `step_isolated_into_buffers` instead reports
individual game failures for replacement. Buffer-integrity failures remain
fatal, and capacity recovery requires its separate opt-in. See
[game failures](training.md#game-failures).

### Hand snapshots

Read authoritative hand contents for explicit native seats at the current
read boundary in scalar or batched games. This is an opt-in hidden-information API:

```python
from manabot import managym
from manabot.env.match import Match

games = managym.VectorEnv(num_envs=2, seed=42, opponent_policy="none")
observations = games.reset_all(Match().to_rust())
snapshots = games.hand_snapshots([1, 0])

for slot, snapshot in enumerate(snapshots):
    print(slot, snapshot.player, snapshot.cards)
    # snapshot.cards is a name-count mapping, e.g. {"Mountain": 2, ...}.

game = managym.Env(seed=42)
game.reset(Match().to_rust())
hand = game.hand_snapshot(1)  # The same HandSnapshot contract for one game.
print(hand.player, hand.cards)

# These snapshots remain unchanged when the games advance.
```

`hand_snapshots(players: list[int])` returns one `HandSnapshot` per slot, in slot
order; the list index is the slot. Supply exactly one native seat (`0` or `1`)
per slot: `[1, 0]` reads seat 1 in slot 0 and seat 0 in slot 1. There is no implicit opponent. Seats are not
remapped under player-control effects: `current_agent_indices()` names the
chooser, which can differ from the turn or rules-decision owner.

Scalar `Env.hand_snapshot(player: int)` returns the same `HandSnapshot` record
for one native seat. The Gym `manabot.env.env.Env` wrapper exposes it directly.

Each snapshot contains:

| Field | Meaning |
|---|---|
| `player: int` | Requested native seat, unchanged by control effects |
| `game_id: int` | Opaque game generation; fresh for every reset, replacement, fork or materialized world, including identical deals |
| `seed: int` | The originating native reset seed; unchanged by scheduling a reset or reseeding a branch's RNG |
| `revision: int` | The current game's decision revision at the read |
| `cards: dict[str, int]` | Complete hand, by card name with positive counts; `{}` means an empty hand |

Save your collection/run identity with `(game_id, revision, player)` and, for
batches, the list index identifying the slot.
Game IDs are allocated process-wide, including across Torch native-vector
rebuilds; they are not portable replay identifiers, consecutive counters or
reproducible values. A seed alone cannot distinguish repeated resets to the same
deal. For replay, retain the match configuration and runtime identity separately.

`set_seed(s)` schedules the next reset; existing snapshots and subsequent reads
of the live game retain its original reset seed until reset succeeds. Forks and
materialized worlds inherit that originating reset seed and receive distinct
game IDs. The seed passed to possible-world materialization controls that
branch's randomness; it does not replace the snapshot's reset seed. Capture
origin and stale checks still apply independently of these snapshot fields.

Both methods synchronously read authoritative hand zones, with one native call
for the whole batch, without stepping, resetting, consuming randomness or
building viewer observations. Counts are detached snapshots; dictionary edits cannot affect the
game or retained snapshot. `CapturedVector` and Torch `VectorEnv` expose the same
method and CPU records. Ordinary reset/step calls do not collect snapshots.

Read before stepping to capture the current hands; copy reused observation
buffers too if retaining both. After a successful step, snapshots describe the
returned game states. Scalar games retain their terminal hands until explicit
reset. Vector done slots describe replacement games; terminal hands are not
retained by vectors. Partial stepping leaves inactive games unchanged, but
snapshots still include every slot: apply the same active mask when selecting results.

A wrong-length seat list, a seat outside `0..1` or reading before reset raises
an error. After a failed step, discard that call's results and recover or reset
before reading further snapshots. An empty hand never denotes a failure;
observation capacities never truncate snapshots.

Snapshots contain hidden information, including with closed decklists. They
remain separate from policy inputs, viewer-safe observations, shared decision
captures and viewer trace exports. Public `known_hand` counts describe only
revealed knowledge. Reading snapshots does not change deck disclosure or what a
player may observe, and requires no policy-input version or checkpoint migration.

## Encoded observations

The native encoder turns an observation into NumPy arrays, one layout per
engine release. manabot reads the same arrays as its policy input
(`ObservationConfig.policy_input_version`, v9); how its network reads them is in
[observations and the policy network](observation-and-model.md#policy-inputs).

### Reading arrays by name

To read a permanent's card definition without hard-coding column 24, ask the
native API for the layout:

```python
import managym

layout = managym.observation_layout()
definition = layout["agent_permanents"]["columns"]["definition"]
print(definition["index"], definition["kind"])
# 24 id
print(definition["encoding"])
# CardDefId + 1; 0 means absent
```

The scalar encoder writes the arrays the layout describes. This example starts
a game with fixed decks and reads only valid permanent rows (the first
decision can have none):

```python
game = managym.Env(seed=42)
observation, info = game.reset([
    managym.PlayerConfig("Alice", {"Mountain": 40, "Lightning Bolt": 20}),
    managym.PlayerConfig("Bob", {"Forest": 40, "Llanowar Elves": 20}),
])
arrays = game.encode_observation(observation)
valid = arrays["agent_permanents_valid"].astype(bool)
references = arrays["agent_permanents"][valid, definition["index"]]
definition_ids = [int(reference) - 1 for reference in references if reference != 0]
```

These are definition IDs for catalog joins, not object identities or numeric
features. Do not log-scale them or infer similarity from their magnitude.
Retain the installed world's content identity with saved arrays and catalog
references.

`observation_layout(*, policy_history=False)` returns a mapping keyed by the
encoder's array names. Each array describes its `axes`, `shape` and stored
`dtype`; a `None` extent in `shape` is a caller-selected capacity, not a
default. The seat axis is viewer-relative (agent first, opponent second).
Feature arrays have a `columns` mapping keyed by column name. Each column gives
its zero-based `index`, `kind`, `encoding` and `range`. Kinds distinguish
identities and references, categorical codes, flags, counts, numeric values and
ranks. The encoding explains scaling, join targets, index bases and absence
sentinels; the range describes stored values, including when no fixed bound
exists. A `tagged` kind needs its row-kind description before preprocessing.
Tagged fact, relation and effect rows describe meanings conditional on their
row kind in `row_kinds` (integer codes mapped to descriptions). Masks and
scalar-valued arrays have empty `columns` and describe their `element` meaning
without inventing a feature axis. Shared headers own the detail/member slices;
unused rows are zero. `policy_history=True` describes the 12-column public
history rows of `events` instead of the 7-column raw events; public history is
bounded to 32 rows, even though its allocated event capacity is selected
separately.

Read the stored encoding before adding preprocessing: permanent damage, power
and toughness are already divided by 10, flags are 0 or 1, and IDs stay exact.
Fact timestamp ranks are stored as ranks; the built-in model divides them by
128, independently of fact capacity. The layout describes the array values,
not a blanket normalization to apply to every column.

The layout comes from the column table the native encoder writes through, so
it cannot drift from the arrays. Reading it starts no game and imports no
Torch. Row capacities are a separate allocation choice: use
`ObservationSpace.shapes` when allocating caller-owned buffers, and retain all
its keys. Capacity errors need larger buffers and a compatible model; state is
never truncated.

A later engine release can change the layout. Save the package version and
world identity with collected arrays, and look columns up by name rather than
by position.

### Card definition IDs

A `CardDefId` is the definition's position in the installed catalog, sorted by
definition key. Adding a card renumbers every definition after it, so IDs are
stable only within one installed build. Do not key a saved embedding table or
dataset by raw IDs. Join each ID to a stable identity for the installed build:
`managym.content_pool("world")["definitions"]` maps every `card_def_id` to its
`registry_name`, definition `key` and `scryfall_id` (back faces have a name but
no key). For card features rather than identities, `managym.catalog_tokens()`
gives each definition's tokenized rules text, rows by `CardDefId`, so the
tokens describe cards a model has not seen.

### Deprecated names

Engine names describe what managym produces. These earlier spellings, named for
manabot's use, still work with a `DeprecationWarning` and are removed in 0.3.0:

| Deprecated | Use |
|---|---|
| `managym.learning_catalog()` / `LearningCatalog` | `managym.catalog_tokens()` / `CatalogTokens` |
| `managym.learning_token_value(name)` | `managym.catalog_token_value(name)` |
| `managym.learning_record_names()` | `managym.record_vocabulary()` |

## Cards as programs

A card is one Rust function returning the engine's own `CardDefinition`, written
in its rules types. There is no card text parser, no card-specific logic and no
intermediate language.

From [`managym/cards/src/lea.rs`](../managym/cards/src/lea.rs):

```rust
fn lightning_bolt() -> CardDefinition {
    let text_box = "Lightning Bolt deals 3 damage to any target.";

    CardDefinition {
        key: "lea.lightning_bolt",
        scryfall_id: "d573ef03-4730-45aa-93dd-e45ac1dbaf4a",
        name: "Lightning Bolt".into(),
        mana_cost: Some(ManaCost::parse("{R}")),
        types: CardTypes::new([CardType::Instant]),
        spell_effects: vec![Effect::deal_damage(
            3,
            ObjectSelector::Target(TargetSpec::AnyTarget),
        )],
        targeting: vec![TargetRequirement::one(TargetSpec::AnyTarget)],
        text_box: text_box.into(),
        ..Default::default()
    }
}
```

- An `Effect` is a verb over an `ObjectSelector`: deal damage to a target, tap
  each creature you control, sacrifice this. New behavior is a new rules
  variant with a directed test, never a special case for one card.
- Learning tokens derive from the same definition
  ([`managym/src/cards/tokens.rs`](../managym/src/cards/tokens.rs)), so the
  policy can read what a card does and a new variant reaches the model without
  a vocabulary edit.
- The world's card pool is every defined card. Known rules bugs are tracked as
  Tasks and don't revoke a card's playability, so playable doesn't mean
  bug-free.
- Each definition lives in the set file of its first printing and carries that
  printing's Scryfall id; `cargo test` in `managym/` covers all three crates.
- Tokens are not cards. A token is a `TokenDefinition` in
  [`managym/cards/src/tokens.rs`](../managym/cards/src/tokens.rs), in its own
  catalog under `token.*` keys, written in the same rules types. Effects name
  the token by key, so tokens that share a printed name stay distinct:

  ```rust
  Effect::create_tokens("token.spirit", 1)
  ```

  A token carries the Scryfall id of its first token printing, or none when it
  was never printed. Tokens are never deck entries or part of the card pool.

## Objects and zones

The engine's runtime types and their names come from the Comprehensive Rules.
A definition says what something is; an object is one instance of it in a
game, and a zone says where that object is. Take one Spirit token through a
turn:

| Step | What the rules say | What the engine holds |
| --- | --- | --- |
| `Effect::create_tokens("token.spirit", 1)` resolves | A token enters the battlefield under its creator's control (CR 111.2); it is a permanent (CR 110.1) | an object of kind `ObjectKind::Token` from `TokenDefinition` `token.spirit`, with a `Permanent` for its battlefield state |
| The Spirit is destroyed | It is put into its owner's graveyard and dies (CR 700.4); it is still a token | the same token, now in the graveyard; death triggers read its last known information |
| State-based actions are checked | A token in a zone other than the battlefield ceases to exist (CR 111.7, 704.5d) | the token leaves every zone and no longer exists |

At no point is the Spirit a card. An effect that says "card" (return target
creature card from your graveyard) never sees it, because "card" means only a
Magic card (CR 108.2); engine code asks `is_card()`.

### Objects

CR 109.1 lists every kind of object. Cards, tokens and copies of spells are
all held in `GameState::cards` as `state::card::Card` values, the engine's
object record, and `Card::kind()` returns which of them each one is
(`ObjectKind`). An object keeps its kind for its whole life; a zone change
makes a new object of the same kind (CR 400.7). The one change of kind is a
copy of a permanent spell, which becomes a token as it resolves (CR 111.12,
608.3f).

| Object | Rule | Engine representation |
| --- | --- | --- |
| Card | CR 108 | `ObjectKind::Card`, from a `CardDefinition` |
| Token | CR 111 | `ObjectKind::Token`, from a `TokenDefinition` or a copy effect that creates a token |
| Copy of a spell | CR 707.10 | `ObjectKind::SpellCopy`, made from a spell on the stack |
| Copy of a card | CR 707.12 | none yet; no defined card casts a copy of a card |
| Activated or triggered ability on the stack | CR 113, 602, 603 | `StackObject::ActivatedAbility`, `StackObject::TriggeredAbility` |
| Emblem | CR 114 | none yet; no defined card creates one |

A definition's `is_token` says only that its catalog row is a token
definition's; an object's kind is the authority for what it is.

Two more names are statuses of those objects in a zone, not kinds of their own:

- A **permanent** is a card or token on the battlefield (CR 110.1).
  `Permanent` holds its battlefield state (tapped, damage, counters,
  attachment) beside the card or token it is; `PermanentId` names it.
- A **spell** is a card, or a copy of a card or spell, on the stack
  (CR 112.1). `StackObject::Spell` holds its cast choices (targets, X,
  modes, kicker) beside the card or copy it is.

A Clone is still a card. A copy effect changes a card's or token's copiable
values (CR 707.2) and is recorded on that object (`copy`); it does not make a
copy object. A copy of a permanent spell becomes a token as it resolves
(CR 111.12, 608.3f).

### Zones

CR 400.1 names seven zones (ante aside). Each holds only the objects its rule allows:

| Zone | Rule | Holds |
| --- | --- | --- |
| Library | CR 401 | cards |
| Hand | CR 402 | cards |
| Graveyard | CR 404 | cards |
| Exile | CR 406 | cards |
| Battlefield | CR 403 | permanents: cards and tokens |
| Stack | CR 405 | spells (cards and copies) and abilities |
| Command | CR 408 | cards (commanders, CR 903.6) and emblems (CR 114.1); empty until a format or card puts something there |

A token or copy that leaves the battlefield or stack is briefly in another
zone until state-based actions are checked: a token in any zone but the
battlefield (CR 111.7, 704.5d), a copy of a spell in any zone but the stack, and
a copy of a card in any zone but the stack or battlefield (CR 704.5e). Library,
hand, graveyard and exile therefore hold cards plus, for that moment, those
objects. They keep their kind there; such a token can't return to the
battlefield (CR 111.8), exile grants and "card" selections skip it, and
`flow/sba.rs` removes it by its kind.

Outside the game is not a zone (CR 400.11). Sideboards are fixed rosters held
beside the game (see [Decks](#decks)); a companion starts there and is put into
its owner's hand by a special action (CR 702.139a, 116.2g), becoming a card in that
zone.

### Where the engine departs from the rules

- **One object table.** Cards, tokens and copies of spells share
  `GameState::cards`, indexed by `CardId`, and the record type is named
  `Card`; zone lists hold `CardId`s of any kind. `Card::kind()` carries the
  distinction the rules draw.
- **Shared zones are stored per player.** Battlefield, stack, exile and command
  are shared zones (CR 400.1); the engine keeps one list per player for each.
  Stack order is `GameState::stack_objects`.
- **Ante is not modeled** (CR 407).
- **Observations keep their names and columns.** The observation API's
  `CardData` rows cover cards, tokens and copies alike, and its `is_token`
  column is set for tokens and for copies of spells. Observations stay
  byte-identical to the layout frozen evidence was recorded with; the column
  reads "not a card", which is what a viewer can tell. The match-state hash
  likewise records a token's or copy's characteristics with `is_token` set.

## Resolution contracts

Three boundaries inside a transition have fixed contracts.

**Pre-priority stabilization.** `Game::stabilize_before_priority` is the sole
authority before priority is published: collect triggers from committed events,
commit one simultaneous batch of state-based actions, repeat until none fire,
put waiting triggers on the stack in APNAP order, and publish priority only
when nothing remains. State-based actions discovered together share one
pre-batch trigger snapshot and commit in `ObjectRef` order; a stale incarnation
is rejected rather than followed after a zone change.

**Proposed events.** Damage, life change, destruction, counter change and zone
movement are submitted as a typed `ProposedEvent` before any state mutates.
Replacement and prevention definitions from the immutable card definitions
apply once each in `(source ObjectRef, definition index)` order; the surviving
event commits and emits the ordinary facts, so triggers, observations and
presentation consume only committed events. Supported replacements are
prevent N damage, double damage, enters tapped and enters with counters. The
choice CR 616 gives a player among several replacements is not offered: the
order is deterministic so replay, training and search stay exact, and a card
that needs more stays unsupported rather than approximated.

**Optional payments.** A resolving optional cost is `Effect::OptionalPayment`
with a `PaymentCost` (mana, or waterbend taps against a generic obligation) and
`if_paid` / `if_declined` effect lists. `PaymentProgress` owns the payer, the
remaining cost and the exact objects already contributed, and completion
prepends exactly one branch to the effect frame; counter-unless-paid uses the
same progress with the spell's controller as payer. A mana ability activated
during payment stays committed if the payment is then declined, and waterbend
taps are cost payment, never tapping for mana. People and policies see the same
native offers (pay or not, waterbend, modal colors, discard), with no
human-only payment flow.

## Worlds

A world names rules meaning plus observation and action shape, and the running
engine always knows its own: `managym.WORLD_VERSION`
([`world.rs`](../managym/src/world.rs)). Checkpoints bind the world, the deck
setup and the input schema, and load only where the digest matches. Main
carries at most one open world; a rules or shape change extends it, and
freezing it is a deliberate step with re-run baselines ([WORLDS.md](../WORLDS.md)).

## Decks

A deck is a managym value: `managym.Deck` holds main and sideboard counts, and
`managym.Decklist` one submitted file with its source, filename, entries and
line issues. Parsing, name resolution and admission all run natively against
the installed world, so Academy and manabot share one importer, and a deck
needs only a world: defined non-token cards are supported, tokens and rules
fixtures are not, and set restrictions such as TLA-only are filters for
choosing decks (`managym.set_cards`), never admission. Construction is the
`development-v1` ruleset: a nonempty main deck, positive counts, at most 1,000
cards per player, no tournament size or copy limits, and no legality claim.

Files are UTF-8 text under 64 KiB, one `count name` per line, with optional
`Main`/`Deck`/`Sideboard` headings and `#` comments. Names match
case-insensitively after Unicode normalization; a double-faced card takes its
full or front name, never the back alone. Unknown or unsupported names are
reported by line and never substituted.

```python
import managym
from manabot.infra.config import MatchConfig

hero, villain = managym.admit_decklists(
    [("hero", managym.Decklist.read("hero.txt")),
     ("villain", managym.Decklist.read("villain.txt"))]
)
match = MatchConfig.from_import(hero, villain)
```

Failure raises `managym.DeckImportError` with each failing seat's line issues.
A serialized match keeps each seat's admitted source privately as provenance
and re-admits it on load, so a world that no longer admits the list rejects
the record. `managym.check_decks(decks)` is the cheap check without a game.

Training does not take deck files; it takes a **deck source** in the regime
(`deck_source`, with `agent.world = "world"` and no `match`). `UniformDecks`
(`uniform-distinct-v1`) draws two different decks per game uniformly over
ordered pairs from its RNG domain, so exact resume reproduces decks, seats and
seed; the TLA pilot's source is the frozen trophy corpus
([training](training.md#regimes-and-stages)). Every deck a source lists is
admitted when the regime validates, and again natively at each game root. A
checkpoint from such a run binds the world and nothing about the decks, so it
plays any deck the world admits, including ones it never saw. Checkpoints bound
to an authored setup refuse imported decks even when shapes agree
(`manabot.model.world.validate_agent_setup`). Academy imports a pair through
**Import decklist files**; the CLI takes `--hero-deck-file` and
`--villain-deck-file` ([Academy](academy.md)).

## What we know

- **Replay holds.** Evaluation cohorts replay exactly in practice. In the
  filter-scope screen all 3,600 monitoring and final games replayed
  ([technique screen](../experiments/ataraxos-technique-screen.md)).
- **The encoding hides no decision.** The learnability search found no clean
  input collisions, and the fixed suite runs on relevant PR updates
  ([learnability](../experiments/learnability-search.md)).
- **Collection dominates cost.** 75–83% of active training time across the
  profiled runs, 82.5% in the L4 run
  ([training time](../experiments/training-time.md)).
- **Hidden-pool ordering once leaked.** Reordering a hidden zone changed search
  evidence between viewer-equivalent states. Viewer equivalence, replay and
  leakage checks are now hard gates ([research program](../manabot/RESEARCH.md#earlier-the-search-era)).

## Where the code lives

| Part | Code |
| --- | --- |
| Turn, priority, stack, combat | [`managym/src/flow/`](../managym/src/flow/) |
| Decisions and forced play | `Game::tick`, `publish_action_space` in [`flow/tick.rs`](../managym/src/flow/tick.rs) |
| Observations | `Observation::for_player` in [`agent/observation.rs`](../managym/src/agent/observation.rs) |
| Env, fork, determinize | [`agent/env.rs`](../managym/src/agent/env.rs), [`flow/search.rs`](../managym/src/flow/search.rs) |
| Batched games | [`agent/vector_env.rs`](../managym/src/agent/vector_env.rs), [`manabot/env/vector_env.py`](../manabot/env/vector_env.py) |
| Replay verification and records | [`manabot/sim/replay.py`](../manabot/sim/replay.py), [`canonical_replay.rs`](../managym/src/canonical_replay.rs) (record shape) |
| Possible worlds | [`possible_worlds.rs`](../managym/src/possible_worlds.rs) |
| Cards and rules types | [`managym/cards/src/`](../managym/cards/src/), [`managym/state/`](../managym/state/) |
| Worlds | [`world.rs`](../managym/src/world.rs), [WORLDS.md](../WORLDS.md) |
| Python bindings | [`managym/src/python/`](../managym/src/python/), `manabot.managym` |
