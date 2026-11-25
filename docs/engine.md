# The rules engine

managym is the rules of Magic as a training environment. It runs many games at
once, decides what is legal at every point, shows each seat only what that seat
may see, and can copy, replay or re-deal a game exactly. It never chooses among
a learner's legal actions. Forced single actions are played automatically, and
every other choice comes from outside, whether from a policy, a search player,
a script or a person; a native opponent (`random`, `heuristic`) is an
explicit opt-in.

This guide is for anyone who needs to trust what the engine guarantees.
Building and testing it is in [managym/README.md](../managym/README.md); the
cross-package contract is in the
[README](../README.md#contracts-the-code-enforces).

- [One decision](#one-decision)
- [Forks, replay and determinization](#forks-replay-and-determinization)
- [Batched games](#batched-games)
- [Per-seat players](#per-seat-players)
- [Built-in heuristic](#built-in-heuristic)
- [Hand snapshots](#hand-snapshots)
- [Encoded observations](#encoded-observations)
- [Cards as programs](#cards-as-programs)
- [Worlds](#worlds)
- [Decks](#decks)
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
- Decklists are open by default, set on `GameSetup.decklists`. A closed
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
digest. Version 26 records each life gain's ordinal when it happens. An ability
that reads its resolution count keeps it for that turn under the source
incarnation, printed face and ability index. A trigger still on the stack
keeps that identity after the source leaves. A countered trigger adds zero.
It also records the winning seat for a card-instructed victory, private
opening progress and simultaneous loss causes. Version 30 adds a
`PermanentUntapped` event to the hashed event ledger for each permanent that
becomes untapped. Version 31 hashes the draw-by-mandatory-loop watch (CR
104.4b) on `PriorityState`. It counts each forced resolution after state-based
actions finish and waiting triggers reach the stack. A win or loss at that
point takes precedence over a loop draw. An empty stabilized stack, a new step, a
successful non-pass priority action, or a decision with several choices ends
the watch and cancels any pending count. After 64 resolutions it fingerprints
the state with life and poison totals, event logs, priority and identity allocation masked
(`GameState::loop_fingerprint`). A repeated fingerprint with no new low of
life or high of poison for a player who can lose ends the game as
`GameOutcome::Draw { mandatory_loop: true }`, and Env reports
`draw_reason: "mandatory_loop"`. A chain that spends counters, cards or
permanents never repeats a fingerprint, so it is never drawn; 10,000
resolutions draw a loop that grows something without bound.
Version 32 includes the last copiable face and casting choices of a departed
spell when a trigger still refers to it. A suspended repeated instruction
records the remaining count in its resolution frame.
Destructuring the state without `..` makes every new field an explicit decision
in the hash contract. The digest excludes pointers, capacities, timings and
observation projections. It reads eight draws from a clone of the RNG.
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
foretold cards, plus the printings under manifested permanents the viewer cannot inspect. The permanents keep their public identities across deals. Public hand minima and exact viewer-known library positions
reserve distinct definition witnesses; the remaining cards are shuffled.
Everything the viewer can see is preserved.
[`possible_worlds.rs`](../managym/src/possible_worlds.rs) defines the same domain exactly, with compatible-deal counts as weights, for beliefs that
want more than a uniform redraw ([beside the loop](../README.md#beside-the-loop)).

**Snapshot and continuation.** Save a live decision boundary with
`Env::snapshot`. The immutable `GameSnapshot` is `Send + Sync`, so several
workers can restore independent games after the source is dropped.

```rust
use managym::agent::{BranchMode, GameSnapshot, RestoreOptions};

let snapshot = env.snapshot()?;
let branch = snapshot.restore(RestoreOptions {
    mode: BranchMode::ShuffleLibraries { seed: 101 },
    forced_first_action: Some(0),
    ..Default::default()
})?;
let replay = branch.snapshot()?.to_replay_json()?;
let reopened = GameSnapshot::from_replay_json(&replay)?;
```

An exact restore continues the RNG, both decision logs, chooser policies and
issued object bindings. Each restored game has fresh projection caches and
a new capture generation. Choose a branch mode, optionally change decision
memory capture, then optionally commit a first action. Restore checks that
the action still identifies the same choice after sampling.

The private snapshot replay reproduces the original setup, external actions and
ordered interventions and verifies the resulting digest. See the
[Python snapshot guide](engine-python.md#save-and-restore-game-positions)
for the mode contracts, replay limits and Academy Watch import.

### Remembered library positions

Native knowledge retains public hand minima and exact card definitions at
library positions counted from the top. Each player has an opponent-visible
and an owner-visible position map. Inspection and chosen ordering record only
what that viewer sees. A draw shifts known positions; a known top card drawn
into hand becomes a hand minimum. A shuffle or an unspecified private reorder
forgets the affected order. Public and private maps independently retain ten
positions: eight nearest the top and two nearest the bottom among known facts.
The owner reads their union, so private retention cannot erase a public fact.

Facts never claim several possible locations. After a secret hand departure,
each independent hand minimum decreases by one (to a minimum of zero). Seeing
A, B and C, then returning two cards privately, therefore preserves no
individual hand fact. Joint deductions belong to the model. This information
loss is deliberate; an observation must never assert something false.

Observations, determinization and exact-world materialization read the same
native facts. Samplers reserve distinct definition witnesses for known library
positions before dealing the remaining hand and library. An unshared sampled
opponent's private position map resets to their public map. Clones preserve
both maps. Closed opposing decklists still refuse hidden-world sampling.

`BranchMode::ShuffleLibraries` works on both open and closed decks without
altering either hand, face-down exile, or future rules RNG. Because the
positional knowledge maps are bounded at ten entries, the engine separately
pins every library object that was inspected, revealed as a visible top, or
placed/repositioned in the library until a genuine rules shuffle clears that
seat's pins (and pins an entire library while a suspended search inspects it).
Only the remaining exchangeable slots permute in place.

## Batched games

Training and evaluation step many games in lockstep: one array of actions in,
one batch of observations out, so a single forward pass serves every active
game. `VectorEnv` ([`managym/src/agent/vector_env.rs`](../managym/src/agent/vector_env.rs))
owns the games and their reset seeds; the Python wrapper is
[`manabot/env/vector_env.py`](../manabot/env/vector_env.py). A finished slot
reports its terminal outcome once and parks until the caller loads a new game
or resets the vector; the Torch wrapper reloads finished slots automatically.
By default an observation that exceeds its
declared capacity is an error; the `first-legal` action-overflow policy and
isolated capacity recovery are explicit opt-ins.

Collection was 75–83% of active time across the profiled training runs
([training time](../experiments/reports/training-time.md)), so engine and inference
throughput are the cost levers. CI keeps an
[engine performance budget](development.md#engine-performance-budget): a
change more than 10% slower than the reference, or one that moves the
search/decision digest, fails.

### Per-seat players

Both `managym.Env` and `managym.VectorEnv` accept a two-seat
`player_policies` tuple. Each player policy is `"agent"` (external human, model
or tool), `"random"` or `"heuristic"`. The default `("agent", "agent")`
surfaces both seats' decisions. The native published chooser selects the seat,
including under control effects; built-ins receive only that chooser's live
viewer observation. Player policies are separate from game/deck configuration
and world identity.

For example, step a batch with an external learner in seat 0 and the built-in
heuristic in seat 1:

```python
import managym

games = managym.VectorEnv(num_envs=2, seed=42, player_policies=("agent", "heuristic"))
players = [
    managym.PlayerConfig(name, {"Forest": 20, "Llanowar Elves": 20})
    for name in ("learner", "heuristic")
]
observations = games.reset_all(players)
# Illustrate the action-index interface with the first legal action in each slot.
transitions = games.step([0, 0])
```

The supplied lists are small development decks. Each reset returns an
`(observation, info)` pair; each step returns an
`(observation, reward, terminated, truncated, info)` tuple per slot. Replace the
zero indices with choices from your learner.

With one external seat, reset and positional step advance built-ins until the
external chooser or a terminal state. Both scalar and native vector terminals
stay parked until explicitly reset or reloaded. Terminal rewards belong to the
external seat, including when a built-in finishes the game. Reset never silently
starts a second game if a built-in wins before an external choice.

With two built-in seats, reset only constructs a game. Call scalar `step()`
(with no action) or vector `step([-1, ...])` to execute exactly one published
built-in decision per game; terminal rewards belong to that decision's chooser.
This makes autonomous games bounded and inspectable.

Command and structured submission methods commit exactly their requested
external action, preserving atomic receipts and stale rejection. If the
resulting chooser is a built-in, call scalar `step()` to continue native play.
Supplying an external index for a built-in chooser raises an error. Native
trivial skipping still applies. Each seat owns independent heuristic memory;
reset clears both, even with the same seed, and forks copy both.

The Python training wrapper accepts the same tuple. The public vector
`opponent_policy` argument is deprecated, warns, and will be removed in 0.3.0:
`none` maps to `(agent, agent)`, other values to `(agent, value)`. Supplying both
arguments or an unknown name raises `ValueError`.
Encoding capacity failures remain errors under the configured overflow policy;
they do not switch a player policy to random play.
`"passive"` is removed and raises an unsupported-policy `ValueError`, including
through RecordedVectorEnv and Torch. Retained failure evidence keeps its original
policy name; replaying a passive-policy receipt requires its original source.

### Built-in heuristic

The scalar `HeuristicPlayer` adapter and arena `heuristic` anchor use the same
native rules as the built-in player policy. Direct native callers can create
`managym.Heuristic()`, call
`select_action(observation)` for each decision, and call `reset()` between
independent games. Keep one selector per seat. Selection returns an original
legal action index and rejects an empty action space. The optional
`max_actions` constructor argument restricts selection to that legal prefix;
it must be positive and is used only by explicitly bounded policies.

To evaluate an existing compatible checkpoint, use the
[local evaluation command](evaluation.md#quick-local-evaluation) with
`--opponent heuristic`.

The heuristic reads only information available to its chooser: visible cards,
legal offers and public combat history. It respects closed decklists and native
information sharing under control effects. The chooser may differ from whose
turn it is or whose resources pay for an action; control effects do not give it
private sideboard access. Hidden hands and library order are never strategic
inputs.

Its rules are deliberately approximate:

- Play lands and activate fetch lands, then cast the highest-mana-value useful
  spell, letting its own spell resolve before casting another. Estimate mana
  from untapped lands; pay kicker when that estimate covers spell plus kicker.
- Delay instants in its own upkeep and draw step. Save untargeted instants
  and draw/life abilities for the opponent's end step.
- Derive target polarity from card definitions: harmful effects target the
  opponent's side, beneficial effects its own. Hold removal for a kill, burn
  for lethal damage to a creature or player, and counters for opposing spells.
- Learn a Lesson, preferring removal while opposing creatures remain. Discard
  spare lands once five are in play, otherwise the costliest card; decline an
  optional discard without a spare land. Look-and-select favors a land while
  fewer than four are available across hand and battlefield.
- Keep enough tough defenders to avoid a lethal swing back. Avoid attacks
  where a blocker kills the attacker and survives, unless an available trick
  rescues it; attack for lethal unblockable damage. Plan one blocker per
  attacker, preferring kills that survive, trades up and free absorption, then
  chump to prevent lethal damage.
- Use its own pumps when they change a fight or make unblocked damage lethal;
  ignore hypothetical opposing combat tricks. Pay offered optional ward and
  “unless pays” costs, except a ward paid in life that would leave it at 0.

Choices are deterministic for the same visible state and decision history.
Equal-value casts use the first legal offer; equal-ranked discard and
look-and-select choices use the last. Combat rankings use visible permanent
IDs or observation order to resolve ties. Unhandled prompts use a fallback:
prefer land, spell, then pass at priority; attack or block when offered;
otherwise take the first non-decline choice, or the last choice when all
decline. These rules are a baseline, not a claim of optimal play or exact
combat search.

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
| Encoded fixed-viewer observations for all or selected slots, including nonchoosing seats | `VectorEnv.write_observations_for_players` ([fixed-seat observations](#fixed-seat-observations)) |
| Live snapshots and batched slot loading (`GameSnapshot`, `RestoreOptions`, `SlotLoad`) | `Env.snapshot` / `restore`; `VectorEnv.snapshot_slots`, `slot_metadata`, `load_slots_into_buffers` ([Batched slot loading and snapshots](engine-python.md#batched-slot-loading-and-snapshots)) |
| Revision-bound Commands, decision captures, search forks and possible worlds | Scalar `Env`; these are not general `VectorEnv` operations |
| Batched readable observations | `managym.VectorEnv.reset_all` / `step` |
| Caller-owned NumPy buffers, per-slot setups, sparse slot loading and partial stepping | Native `VectorEnv` buffer methods |
| Private failure replay inputs around buffered execution | `manabot.env.failures.RecordedVectorEnv` |
| Torch observation/reward tensors on a selected device with Gym-style automatic reset | `manabot.env.vector_env.VectorEnv` |

Scalar `Env.step` returns the terminal observation and waits for `reset`.
Native `VectorEnv` and `RecordedVectorEnv` return the terminal observation,
reward, flags and winner, then park that slot with chooser `-1`. Restart
all slots with the matching readable or buffered reset, or use
`load_slots_into_buffers` to replace individual slots.

Torch `VectorEnv.step` saves the rewards, flags and diagnostics before
refilling finished slots. Its observation tensors then show the replacement
games. Clone tensors you need to retain across steps.

Torch `VectorEnv.reset(seed=s)` restarts slot `i` at seed `s + i` for any
explicit seed, including the current one, so equal seeds reproduce deals.
`reset()` without a seed continues native seed succession. Its
`current_agent_indices()` names each slot's chooser.

`RecordedVectorEnv` records both readable and buffered steps and resets. Each
reset or successful slot load starts a new replay prefix, including when the
seed repeats or saturates at its maximum value.

Native buffered stepping and `RecordedVectorEnv.step_into_buffers` accept an
`active` boolean per slot. Inactive games and observation rows stay unchanged;
the Python buffer bindings clear their rewards and done flags to zero/false.
Apply the same mask when collecting transitions. The Torch wrapper's `step`
advances every slot and does not accept an active mask.

Strict batch errors can occur after other slots have advanced. Do not retry
the same batch assuming nothing changed. `RecordedVectorEnv` retains private
pre-call replay inputs; explicit `step_isolated_into_buffers` instead reports
individual game failures for replacement. Buffer-integrity failures remain
fatal, and capacity recovery requires its separate opt-in. See
[game failures](training.md#game-failures).

### Hand snapshots

Read authoritative hand contents for explicit native seats at the current
read boundary in scalar or batched games. This is an opt-in hidden-information API:

```python
import managym
from manabot.env.match import Match

games = managym.VectorEnv(num_envs=2, seed=42, player_policies=("agent", "agent"))
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
game or retained snapshot. `RecordedVectorEnv` and Torch `VectorEnv` expose the same
method and CPU records. Ordinary reset/step calls do not collect snapshots.

Read before stepping to capture the current hands; copy reused observation
buffers too if retaining both. After a successful step, snapshots describe the
current game states. Scalar games and native vector slots retain their terminal
hands while parked, until reset or reloaded with `load_slots_into_buffers`.
Torch `VectorEnv` reloads finished slots during `step`, so its done slots
describe replacement games. Partial stepping leaves inactive games unchanged,
but snapshots still include every slot: apply the same active mask when
selecting results.

A wrong-length seat list, a seat outside `0..1` or reading before reset raises
an error. After a failed step, discard that call's results and recover or reset
before reading further snapshots. An empty hand never denotes a failure;
observation capacities never truncate snapshots.

Snapshots contain hidden information, including with closed decklists. They
remain separate from policy inputs, viewer-safe observations, shared decision
captures and viewer trace exports. Public `known_hand` counts describe only
revealed knowledge. Reading snapshots does not change deck disclosure or what a
player may observe, and requires no policy-input version or checkpoint migration.

### Fixed-seat observations

Read seat 0 in one game and seat 1 in another, even when those seats are not
choosing. Allocate a separate batch from the observation space used by your
consumer:

```python
import managym
from manabot.env.match import Match
from manabot.env.observation import ObservationSpace

space = ObservationSpace()
games = managym.VectorEnv(num_envs=2, seed=42, player_policies=("agent", "agent"))
games.reset_all(Match().to_rust())
buffers = space.allocate(2)

players = [0, 1]
reads = games.write_observations_for_players(
    players, buffers, layout=space.layout, action_overflow=space.config.action_overflow
)
# Row i in every array describes players[i] in slot i.

# Read the opposite hands at the same boundary, without stepping in between.
hands = games.hand_snapshots([1 - player for player in players])
for slot, (read, hand) in enumerate(zip(reads, hands, strict=True)):
    assert (read.slot, read.game_id, read.revision) == (
        slot, hand.game_id, hand.revision
    )
    assert read.player == 1 - hand.player
```

`write_observations_for_players(players, buffers, action_overflow="error", *, layout)`
takes one entry per slot. Pass seat `0` or `1` to read it, or `None` to
leave its game and buffer row untouched. This permits reads beside terminal,
poisoned or unset slots. Each requested slot returns an `ObservationRead` in
ascending slot order.

| Field | Meaning |
|---|---|
| `slot: int` | The row in every array (equal to the list index when no slot is `None`) |
| `player: int` | Requested native seat, unchanged by control effects |
| `game_id: int`, `seed: int`, `revision: int` | The same values a `HandSnapshot` of that slot reports at this read boundary |

Pair a read with a hand snapshot at `read.slot` when game ID and
revision agree, selecting matching or opposite seats as needed. Save a
collection/run identity too: `game_id` distinguishes resets within the process,
even to the same seed, but is not a portable replay ID.

The arrays contain the same encoding as scalar
`Env.observation_for_player(player_index)` for the requested seat and selected
policy-input version. Native visibility rules apply, including closed decklists
and information shared through player-control effects. Requesting a seat does
not remap it to its controller or grant access to hidden information. Only the
published chooser sees legal actions. Hand snapshots are separate privileged
records and must stay out of viewer-safe inputs and exports.

`space.allocate(num_envs)` supplies every required writable, contiguous
NumPy array at the selected version's shapes, dtypes and capacities. No reward
or done arrays are needed. Keep these buffers separate from ordinary step
buffers and copy rows you need to retain before the next write.

`RecordedVectorEnv` and the Torch `VectorEnv` wrapper expose the same NumPy-buffer
method and CPU identity records. The Torch wrapper supplies its own layout.
Pass `layout=consumer_space.layout` for a consumer with different capacities.
Use `torch.from_numpy` and an explicit device transfer as needed. Clone shared CPU tensors you need to retain before the next write.

A fixed-seat read preserves the chooser's observations, legal choices and RNG.
On native `VectorEnv` and `RecordedVectorEnv`, a parked terminal slot describes
the finished game until reloaded (or pass `None` to skip it); on Torch
`VectorEnv`, a finished slot describes its replacement game. The seat list
selects rows independently of the preceding step's active mask. Ordinary steps
and resets only encode the chooser's view. Fixed-seat reads confer no Command
or decision-capture authority.

A wrong-length seat list, an invalid seat, reading before reset, or malformed
buffers raises an error. Encoding uses supplied capacities and the selected
policy-input version's admission rules. Overflow errors by default;
`action_overflow="first-legal"` permits only the existing action-prefix policy,
never truncation of other data. Discard all output rows after a failed read.
Correct the request, increase capacity or select an input version that supports
the state before retrying. After a failed step, recover or reset the games first.

## Encoded observations

One native encoder turns an observation into NumPy arrays, one layout per
engine release. Every caller's arrays come from it: a single observation, a
batch of search branches and the vector environment's step buffers. manabot
reads the same arrays as its policy input
(`ObservationConfig.policy_input_version`, v11); how its network reads them is
in [observations and the policy network](observation-and-model.md#policy-inputs).

### Encoding one observation

```python
import managym
from manabot.env.observation import ObservationSpace, encode_observation

game = managym.Env(seed=42)
observation, info = game.reset([
    managym.PlayerConfig("Alice", {"Mountain": 40, "Lightning Bolt": 20}),
    managym.PlayerConfig("Bob", {"Forest": 40, "Llanowar Elves": 20}),
])

space = ObservationSpace()
arrays = encode_observation(space, observation)
print(int(arrays["agent_cards_valid"].sum()))
# 7

restored = managym.Observation.from_json(observation.toJSON())
del game
print(int(encode_observation(space, restored)["agent_cards_valid"].sum()))
# 7
```

`encode_observation(space, obs)` returns freshly allocated arrays at the
space's shapes, so a caller may keep them. The space builds `space.layout`
from its `ObservationConfig`. The native layout resolves the shapes for the
configured input version, optional blocks and capacities. `space.shapes`
reads those shapes from the layout, and `space.allocate()` uses the layout's
allocator. The encoder accepts any `managym.Observation`, including the
one a step returned, a fixed seat's view from `observation_for_player`, or one
restored with `Observation.from_json`. No game needs to be open. A restored
observation retains the public history written by `toJSON()`. Older JSON
without that field loads with empty public history. Semantic `viewer_state`,
decision captures and search witnesses keep the canonical bytes without it; a
capture carries its own bounded `history`.

To write into arrays you own, call the function underneath:

```python
layout = managym.ObservationLayout(max_actions=256, policy_history=True)
buffers = layout.allocate()
managym.encode_observation_into(observation, layout, buffers)
```

The same layout allocates a batch with `layout.allocate(batch_size)`.
In manabot, pass `space.layout` to use the configured input version, history,
memory and capacities. Both encoders check every key, shape and dtype before
writing. Vector stepping also requires the reward and done arrays that
`allocate_vector_buffers(space, num_envs)` supplies. Register those arrays
with `games.set_buffers(step_buffers, layout=space.layout)`.

An observation that does not fit raises `managym.ObservationCapacity`, a
`ValueError` naming the field, the count and the capacity:

```python
from manabot.infra.config import ObservationConfig

small = ObservationSpace(ObservationConfig(max_cards_per_player=4))
try:
    encode_observation(small, observation)
except managym.ObservationCapacity as error:
    print(error)
# observation capacity exceeded: agent_cards plus sideboard 7 > 4
```

Catch that type to count or recover from a game that outgrew its buffers;
`encode_observation_into` and the batch `encode_observations_into` raise it,
including for more legal actions than slots without
`action_overflow="first-legal"`. A vector environment's step raises
`managym.AgentError` with the same message for an unrecovered failure (its
retained evidence names that class) and records `ObservationCapacity` as the
failure class when it recovers the game. Any other `ValueError` means the
buffers or the request are wrong: malformed arrays, or state the selected
input version cannot hold (`managym.input_v10_state(obs)` and
`managym.unrepresentable_state(obs)` name it). Nothing is truncated to fit.
Discard the arrays after a failure.

`Env.encode_observation` and `Env.encode_observation_into` are deprecated
aliases that warn and are removed in 0.3.0. Replace the first with
`encode_observation(space, obs)` and the second with
`managym.encode_observation_into`. In manabot, `ObservationSpace.encode(obs)`,
`space.encoder` (now the space itself) and `ObservationEncoder` (now
`ObservationSpace`) warn and are removed in 0.3.0 too.

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

A count or number also says whether the encoder already divided it:

```python
columns = layout["agent_permanents"]["columns"]
print(columns["power"]["kind"], columns["power"]["scale"])
# number 10
print(columns["exhausted_abilities"]["kind"], columns["exhausted_abilities"]["scale"])
# count 1
```

A category says how many values it has: `size` is its largest code. A code
above it is not an error, so a category can grow. In the rows the policy reads
column by column, 0 means absent, except a player's `seat`, where 0 is the
seat on the play:

```python
frame_kind = layout["relations"]["columns"]["frame_kind"]
print(frame_kind["kind"], frame_kind["size"])
# category 14
```

The encoder writes the arrays the layout describes. With `arrays` from
[the example above](#encoding-one-observation), read only valid permanent rows
(the first decision can have none):

```python
valid = arrays["agent_permanents_valid"].astype(bool)
references = arrays["agent_permanents"][valid, definition["index"]]
definition_ids = [int(reference) - 1 for reference in references if reference != 0]
```

These are definition IDs for catalog joins, not object identities or numeric
features. Do not log-scale them or infer similarity from their magnitude.
Retain the installed world's content identity with saved arrays and catalog
references.

`observation_layout(*, policy_history=False, turn_state=True, legacy=False, decision_memory=False)` returns a mapping keyed by the
encoder's array names. Each array describes its `axes`, `shape` and stored
`dtype`; a `None` extent in `shape` is a caller-selected capacity, not a
default. The seat axis is viewer-relative (agent first, opponent second).
Feature arrays have a `columns` mapping keyed by column name. Each column gives
its zero-based `index`, `kind`, `encoding` and `range`. The kind is one of a
closed set:

| Kind | The column holds |
| --- | --- |
| `flag` | 0 or 1 |
| `count` | A nonnegative amount, stored divided by `scale` |
| `mana` | Floating mana in a player pool, stored divided by `scale` |
| `number` | A signed amount, stored divided by `scale` |
| `rank` | A one-based order rank; 0 absent |
| `position` | A one-based position in a zone; 0 absent |
| `category` | A code from a small fixed set |
| `selector` | A one-based choice index (a mode, an ability, a frame index a relation row repeats); 0 absent |
| `reference` | A join to a row of another table: the padded object table, a continuation frame an action continues, the naming catalog; the encoding gives the base |
| `id` | An identity, such as a card definition or token value; never a magnitude |
| `tagged` | A value whose meaning depends on its row's kind |

`count`, `mana` and `number` columns also give `scale`, the divisor the encoder
applied: 1 for an unscaled value, otherwise 10, 20 or 60. No other kind has
one. The encoding explains join targets, index bases and absence
sentinels in prose; the range describes stored values, including when no fixed bound
exists. A `tagged` kind needs its row-kind description before preprocessing.
Tagged fact, relation and effect rows describe meanings conditional on their
row kind in `row_kinds` (integer codes mapped to descriptions); `facts` and
`effect_details` also name each kind in `row_kind_names` (`"counter"` → 1), and
`effect_details` maps its program kinds to catalog `TokenKind` codes in
`program_token_kinds`, so a reader looks kinds up by name rather than copying
numbers. Masks and
scalar-valued arrays have empty `columns` and describe their `element` meaning
without inventing a feature axis. Shared headers own the detail/member slices;
unused rows are zero. `policy_history=True` describes the 12-column public
history rows of `events` instead of the 7-column raw events; public history is
bounded to 32 rows, even though its allocated event capacity is selected
separately. `turn_state=True` adds policy input v10's `player_turn_state` (columns
`life_lost`, `spells_cast`, then `nonartifact_mana_white` through
`nonartifact_mana_colorless`) and `card_turn_state`.

Read the kind and `scale` before adding preprocessing: permanent damage, power
and toughness are already divided by 10, flags are 0 or 1, and IDs stay exact.
Timestamp ranks are stored as ranks. The layout describes the array values,
not a blanket normalization to apply to every column; the built-in model
chooses its treatment from the kind
([how the network reads a column](observation-and-model.md#how-the-network-reads-a-column)).

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
no key). Join on `key`: a printed name does not identify a definition, since
two tokens may share one and a token may share a card's. For card features
rather than identities, `managym.catalog_tokens()` gives each definition's
tokenized rules text, rows by `CardDefId`, with the same `keys` and
`registry_names` beside them, so the tokens describe cards a model has not
seen.

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
        printed: PrintedCharacteristics {
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
        },
        ..Default::default()
    }
}
```

`PrintedCharacteristics` is what a card and a token share: name, cost, types,
colors, keywords, abilities, text and power/toughness as printed. It is the
input to the layer system; `Game::characteristics(id)` returns the layered
view of a game object after continuous effects. A `CardDefinition` adds the
card’s identity (`key` and a required `scryfall_id`). Double-faced cards
register their `back_face` alongside the front. A front with `modal_back`
offers either printed face when the play permission allows it. Casting Photon
uses Photon’s cost and abilities, then the permanent enters back face up.
Monica's activated ability turns the same permanent over to Photon. The
modal permanent uses the mana value of the face it shows. Moving it to the
graveyard restores Monica Rambeau.

- An `Effect` is a verb over an `ObjectSelector`: deal damage to a target, tap
  each creature you control, sacrifice this. New behavior is a new rules
  variant with a directed test, never a special case for one card.
- Later instructions can read what an earlier instruction accomplished.
  Fumigate counts the creatures its destruction moved, including a creature
  exiled by a finality counter. Exsanguinate reads `LifeLostThisWay` after
  each opponent loses X life. A player whose life total cannot change
  contributes zero. A selection's child effects have their own results, and
  completing the selection restores the caller's results.
- A trigger is "whenever X does Y": an event over its actor, with the event's
  own parameters as fields. Monastery Swiftspear's prowess
  ([`managym/cards/src/ktk.rs`](../managym/cards/src/ktk.rs)) is the cast event
  with you as the actor:

  ```rust
  condition: TriggerCondition::casts(
      Actor::You,
      CardPredicate {
          not_card_types: vec![CardType::Creature],
          ..Default::default()
      },
  ),
  ```

  `casts` builds `Casts { who, predicate, nth: None }`. Changing `who` to
  `Actor::Opponent` or `Actor::AnyPlayer` gives "whenever an opponent casts" or
  "whenever a player casts"; `nth: Some(2)` gives "your second spell each
  turn". The same applies to draws (`Draws { who, nth }`), steps
  (`BeginningOf { step, whose }`), keyword actions such as searching, scrying
  or bending (`Performs { who, any_of }`), cards reaching a graveyard
  (`PutIntoGraveyard { whose, predicate }`) and putting counters
  (`PutsCounters { who, kind, on }`). Object events name a `TriggerSubject`
  instead, whose controller is the same kind of `Actor` ("a creature an
  opponent controls" is `Matching { controlled_by: Actor::Opponent, .. }`):
  `Dies`, `Sacrificed`, `Attacks { subject, per, attackers }`,
  `DealsCombatDamage { subject, to, per }` and
  `DealsDamage { kind, source, to }`, where `per` says whether "one or
  more" creatures trigger once per batch or once each, and `attackers` says
  whether the attack is alone or with at least one other creature.
  `DealsDamage` fires once per damage event of its `DamageKind` (any,
  combat or noncombat) to a `DamageRecipientFilter` (a player actor, or
  anything). Its source is any object, a spell included, matched as it is
  or last was with its controller ("a source you control" is
  `TriggerSubject::you_control(CardPredicate::default())`), and the damage
  dealt is "that many". `BecomesTapped` and `BecomesUntapped` fire on a
  transition only: every untap, the untap step's included, goes through
  `Game::untap_permanent`, and a stun counter removed instead is no untapping.
  A state trigger (CR 603.8) is `State(Condition)` ("when there are four or
  more page counters on this artifact"): it is asked of its source whenever a
  player would receive priority, never of an event, and doesn't trigger again
  while its occurrence waits or is on the stack. The ability
  description ("Whenever an opponent draws their second card each turn, ...")
  and the learning tokens come from one template per event, so a card that
  needs a new actor or count needs no new variant.
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
  catalog under `token.*` keys: a key, an optional Scryfall id and the same
  `PrintedCharacteristics`. Effects name the token by key, so tokens that
  share a printed name stay distinct:

  ```rust
  Effect::create_tokens("token.spirit", 1)
  ```

  A token carries the Scryfall id of its first token printing, or none when it
  was never printed. Tokens are never deck entries or part of the card pool.
- The engine registers each card, back face and token as one
  `ObjectDefinition`, the row a `CardDefId` names: its identity, its
  `PrintedCharacteristics` and whether it is a token. Token status is not a
  printed characteristic, so copy effects copy `printed` and never token
  status; a token copy of a card is a token because it was created as one.

## Objects and zones

The engine's runtime types and their names come from the Comprehensive Rules.
A definition says what something is; a card is the physical card a player
brought to the game (CR 108); an object is one appearance of a card, token or
copy in one zone (CR 109); and a zone says where that object is. An object
that changes zones becomes a new object with no memory of the old one (CR
400.7), so the engine has two ids: `ObjectId` names one appearance and stops
working when the object moves, and `CardId` names the card and follows it for
the whole game.

Flicker a Gray Ogre (exile it, then return it to the battlefield):

```rust
let ogre = game.state.permanents[permanent].as_ref().unwrap().object; // ObjectId
let card = game.state.card_of(ogre).unwrap();                         // CardId

let exiled = game.move_object(ogre, ZoneType::Exile);
let returned = game.move_object(exiled, ZoneType::Battlefield);

// The object that left is gone. Nothing that held its id can follow it.
assert_eq!(game.state.object(ogre).err(), Some(ObjectLookupError::Stale));
assert_eq!(game.lookup_current_permanent(ogre), Err(ObjectLookupError::Stale));
assert_ne!(returned, ogre);

// The card is the same card, and names the object it is now.
assert_eq!(game.state.object_of(card), returned);
assert_eq!(game.state.cards[card].owner, PlayerId(0));

// What the old object was is last known information (CR 608.2h).
assert_eq!(game.object_lki(ogre).unwrap().from_zone, ZoneType::Battlefield);
```

The returned Ogre has no damage, no counters, no Auras and no "until end of
turn" effects, and anything that targeted the old Ogre no longer has a target.
None of that is cleanup code; the new object never had them.

Cecil, Dark Knight transforms on the battlefield. If Cecil deals three damage
while you have 13 life, the trigger remembers three. On resolution you fall to
10, then untap Cecil and turn it over to Cecil, Redeemed Paladin. The same
object keeps its counters, attachments and any spell already targeting it.
If Cecil tramples over a blocker, one trigger counts the damage to both
recipients. You still lose that life if Cecil dies in combat.
When the Paladin attacks, its trigger gives the other attacking creatures
indestructible until end of turn.

Vanille’s first-main trigger offers a payment of {3}{B}{G} when you own
and control her and a creature named Fang, Fearless l'Cie. Paying exiles
both. The original cards form Ragnarok, Divine Deliverance. Both physical
card IDs then refer to the same battlefield object. If Lost Days puts
Ragnarok on the bottom of its owner’s library, the owner chooses which
card to draw first. Ragnarok stays on the battlefield until that choice
commits, then both cards move and the spell creates its Clue. A token
copy of Ragnarok leaves without an ordering choice.

When Manifest Dread resolves, its controller privately inspects the top two library
cards and chooses one to put onto the battlefield face down. The opponent sees a
nameless 2/2 creature and can see the other card in the graveyard. With one card
left, that card enters face down automatically. With an empty library, the spell
finishes resolving. Copying the face-down creature creates a face-up, nameless
2/2. It has no back face, even if the manifested card can transform. Copying
that token again preserves those characteristics. With priority, the controller can
show a manifested creature card and pay its mana cost to turn it face up. This
keeps its counters and public object reference. Spells already targeting it
still target the same permanent. A subsequent token copy sees the revealed
creature, including its back face when it can transform.

Esper Origins illustrates a zone change with a different face up. Cast it from hand
to surveil two and gain two life. Cast it from your graveyard for its flashback
cost and it also exiles itself, then enters as Summon: Esper Maduin with finality
and lore counters. The stack spell, exiled card and Saga each have their own
`ObjectId` and share one `CardId`. The Saga has mana value two and no mana cost.
Its first chapter reveals the top card to both players. A permanent card goes
to hand and any other card stays known on top.
Fiendish Panda’s death trigger compares each eligible graveyard card’s mana
value with the Panda’s power when it died, including its counters. The same
comparison applies when choosing a new target and when resolving the trigger.
Returning the Panda to the battlefield leaves that bound intact. If the chosen
card leaves the graveyard and returns before resolution, it has a new object
ID and stays in the graveyard. Heartfire Immolator likewise deals damage using
its power when sacrificed, including bonuses from prowess.

A token is an object with no card behind it. Take one Spirit token through a
turn:

| Step | What the rules say | What the engine holds |
| --- | --- | --- |
| `Effect::create_tokens("token.spirit", 1)` resolves | A token enters the battlefield under its creator's control (CR 111.2); it is a permanent (CR 110.1) | an object of kind `ObjectKind::Token` from `TokenDefinition` `token.spirit`, with a `Permanent` for its battlefield state |
| The Spirit is destroyed | It is put into its owner's graveyard and dies (CR 700.4); it is still a token | a new token object in the graveyard; death triggers read the battlefield object's last known information |
| State-based actions are checked | A token in a zone other than the battlefield ceases to exist (CR 111.7, 704.5d) | the token leaves every zone; its id is stale and nothing replaces it |

At no point is the Spirit a card. An effect that says "card" (return target
creature card from your graveyard) never sees it, because "card" means only a
Magic card (CR 108.2). The types say the same thing: the Spirit has an
`ObjectId` and `card_of` returns `None` for it.

```rust
// "Return target creature card from your graveyard to your hand."
let Some(card) = game.state.card_of(object) else {
    return; // a token, a copy of a spell, or an object that has since moved
};
let owner = game.state.cards[card].owner;
```

### Objects

Each runtime type takes its name and its contents from the rule that defines
it:

| Rules term | Rule | Engine type | Holds |
| --- | --- | --- | --- |
| Card | CR 108 | `Card` in `GameState::cards`, named by `CardId` | the physical card: owner and definition, for the whole game |
| Object | CR 109 | `Object` (`state::game_object`) in `GameState::objects`, named by `ObjectId` | one appearance in one zone: its characteristics as printed or copied, its copy effect and its `ObjectKind` |
| Token | CR 111 | `Object` with `ObjectKind::Token`, from a `TokenDefinition` or a copy effect | no card |
| Copy of a spell | CR 707.10 | `Object` with `ObjectKind::SpellCopy`, made on the stack | no card |
| Copy of a card | CR 707.12 | none yet; no defined card casts a copy of a card | no card |
| Permanent | CR 110 | `Permanent`, named by `PermanentId`, beside the object it is | battlefield state: tapped, damage, attachment |
| Spell | CR 112 | `StackObject::Spell`, beside the object it is | cast choices: targets, X, modes, kicker |
| Ability on the stack | CR 113, 602, 603 | `StackObject::ActivatedAbility`, `StackObject::TriggeredAbility` | source, effects and choices; not an `Object` |
| Emblem | CR 114 | none yet; no defined card creates one | no card |

`Card` is a small `Copy` value written once at setup: `owner`, `definition`
and the `PublicId` observations show for it. `GameState::cards` holds both
main decks and both sideboards and never grows.

An object's kind is fixed when the object is made. A copy of a permanent spell
resolves into a token (CR 111.12, 608.3f): the permanent is a new object of
kind `Token`, and neither it nor the copy has a `CardId`. A definition's
`is_token` says only that its catalog row is a token definition's; an object's
kind is the authority for what it is.

Cast Sun-Blessed Healer with kicker and its entry trigger lets you choose a
nonland permanent card with mana value two or less in your graveyard. Nullpriest
of Oblivion uses the same trigger to return a creature card of any mana value.
The trigger remembers kicker if the Cleric dies in response. Exiling and
returning the Cleric gives it a fresh entry, so it offers no second target.

A copy of a kicked permanent spell enters with the original kicker choice and
X. Only the cast spell has a payment receipt, including when its cost was zero.
A battlefield copy takes its characteristics from the permanent and starts with
its own entry choices. [Zendikar Rising release notes](https://magic.wizards.com/en/news/feature/zendikar-rising-release-notes-2020-09-10)

Fiendish Panda gets one counter each time its controller gains life. When it
dies, its last battlefield power limits which non-Bear creature cards it can
return. A Clone that dies as a 4-power Panda can return Driver of the Dead
(mana value four). It cannot return itself, even though it is a Shapeshifter
in the graveyard. Changing the trigger's target uses the same restrictions.
[Foundations release notes](https://magic.wizards.com/en/news/feature/foundations-release-notes)

Token creation replacements finish before the allocator assigns the token's
viewer id, owner and kind. Spell copies use `Object::spell_copy`. The zone
allocator supplies each copy's `ObjectId`.

A Clone is still a card. A copy effect changes an object's copiable values
(CR 707.2) and is recorded on that object (`copy`); it does not make a copy
object, and it does not survive the object leaving its zone.

### Identity

| Id | Names | Valid | Get it from |
| --- | --- | --- | --- |
| `ObjectId` | one appearance of a card, token or copy in one zone | until that object changes zones or ceases to exist | a zone list, a permanent, a spell, a target, an event, or the move that made the object |
| `CardId` | a physical card | the whole game | a decklist, sideboard, library or hand entry, or `card_of` |
| `PermanentId` | an object's battlefield state | while it is on the battlefield | the battlefield |
| `PublicId` | the match-unique number observations show for players, objects and abilities, also used to order timestamps | the whole game | `Card::public_id`, or the token or copy |

Two functions connect cards and objects, and both read current state:

```rust
/// The object this card currently is. A card outside the game (a sideboard)
/// is in no zone: this is the id it would enter with.
pub fn object_of(&self, card: CardId) -> ObjectId

/// The card behind this object; `None` for a token, a copy or a stale id.
pub fn card_of(&self, object: ObjectId) -> Option<CardId>
```

`GameState::live(id)` returns the id while it still names an object in a zone,
and `ZoneManager::successor(id)` returns the object `id` became by its one next
zone change while that object is still there (CR 400.7e, j). `GameState::card_id`
reads which card an object is or was without checking that the object still
exists. It serves a card outside the game (a sideboard is no zone), viewer
captures of departed objects and stale cast grants; rules code that acts on a
card uses `card_of`.

Every lookup by `ObjectId` checks that the id is current:

| `ObjectLookupError` | Meaning |
| --- | --- |
| `Stale` | the object changed zones or ceased to exist; ask last known information or the card instead |
| `WrongZone` | the object exists but is not where the caller requires (`lookup_current_permanent` on an object in hand) |
| `Missing` | no object ever had this id |

`state[id]` uses the same check and asserts that the caller has a current
appearance. Raw object storage has no `ObjectId` or `CardId` indexing.
Entry processing can inspect a newly allocated row before placement through
an entry-only accessor that still rejects retired generations.

An ability keeps one source identity through pending triggers, the stack and
child resolution frames. Its captured definition and the retained base
printing supply definition provenance after departure; these reads never
borrow a later appearance's copy or transformed face. Battlefield facts come
from LKI. When an ability creates a token copy of its source, it takes the
source's copiable values at resolution or, after departure, at the moment it
left. A Clone that returns as a different creature keeps its new form while
the earlier ability copies the departed form.

Stack inspection, target changes and resolution use the triggering face captured
at placement. For example, a Clone that copies an Electroduplicate token can
return to hand with its sacrifice ability still on the stack. Both players can
read the ability, and its source reference still names the departed Clone.

Copy exceptions become part of the new token’s copiable characteristics. A Rite
of Replication copy of an Electroduplicate Koma’s Coil inherits haste and its own
end-step sacrifice trigger. A temporary death ability granted to the first Coil
triggers only when that Coil dies.

A stale id is an ordinary outcome, not a bug: a spell whose target left fizzles
on it, and a delayed trigger watching one creature retires on it. Code picks
the id for what its rules text means:

- **"That creature", a target, an attacker, a source of damage** is an
  `ObjectId`. If the object left, the effect does nothing to its successor.
- **"That card", an owner, a decklist entry, "return it to its owner's hand"
  after it has already moved** is a `CardId`; `object_of` finds where it is.
- **What something was when it left** (power when it died, its controller,
  its counters) is `Game::object_lki(old_id)`, the `ObjectLkiLog` entry
  recorded as the object departs. Its name and supertypes come from the face
  it last showed (`Game::lki_face_printing`, a copy's or back face's when they
  changed) and its mana value from its front face
  (`Game::lki_mana_printing`, CR 712.8e); the entry stores a name only when an
  effect changed it (`renamed`).

An `ObjectId` is opaque. It cannot be built from a number, and tests take ids
from the game (a permanent's `object`, a zone list, a move's return value)
rather than writing literals. There is no conversion from `CardId` to
`ObjectId` other than `object_of`, because the answer changes during the game.

### Zone changes

`Game::move_object` and the `ProposedEvent::ZoneMove` it submits are the only
way an object comes from another. The move retires the old id, builds the new
object from its base (the card's printing, or a token's or copy's copiable
values) and returns the new `ObjectId`. Everything else about the old object
is dropped: damage, counters, attachments, copy effects, a transformed face,
continuous effects that named it, and every id held elsewhere.
A move that does not happen (the object is already in that zone, or the id is
already stale) changes nothing and returns the id it was given.

What the rules let the new object know arrives in one `CarryOver` value
(`flow/zones.rs`), built by the move from the departing object. A fact
crosses a zone change only as a field of it, with the rule that allows it:

| Rule | What crosses | How |
| --- | --- | --- |
| CR 400.7d | a permanent knows how its spell was cast: that it was cast, the value of X and the mana spent | `CarryOver::cast` |
| CR 400.7e, f | a zone-change trigger (dies, leaves the battlefield, an Aura put into the graveyard) finds the object in the public zone it moved to | `CardMoved.object_id` is the successor; `departed` names its exact pre-move appearance for LKI |
| CR 400.7g, h | a permission to cast a card, and the effect that granted it, find the spell on the stack | the grant holds the exact `ObjectId` in its permitted zone; casting returns the spell's `ObjectId`, and the spell records the zone it was cast from (`SpellOnStack.cast_from`) |
| CR 400.7i | an effect that lets a land be played finds the land on the battlefield | the entry batch's `EventResult.moved`, bound when the batch completes (an entry replacement may suspend it first) |
| CR 400.7j | an effect or cost that moves an object to a public zone finds it there ("exile it, then return it", "if you do") | the move's returned `ObjectId`, held by the effect frame (`Moved`, `CreatedThisWay`) or the exile link |
| CR 715.3d | resolving an Adventure can give its caster permission to cast the permanent spell from exile | the selected Adventure replacement grants access to the exact exile successor. Bilbo’s competing exile replacement creates no grant. |
| [Tarkir: Dragonstorm release notes](https://magic.wizards.com/en/news/feature/tarkir-dragonstorm-release-notes) | an Omen uses its own characteristics while casting and on the stack | Vizier permits Twinmaw Stormbrood from the library. Charring Bite resolves into its owner’s library and shuffles it. A counter or loss of its only target sends it to the graveyard. |
| CR 111.12, 608.3f | a copy of a permanent spell becomes a token | the new object's kind |

Yue’s hand permission checks the spell being announced, so it can offer
Spew Flame from Smaug. Solstice Revelations likewise checks Charring Bite’s
mana value when it exiles Twinmaw Stormbrood. During targeting, the caster
sees the chosen spell name in the decision context and its profile in the
continuation. Both viewers see the spell once it reaches the stack.

Omniscience offers the ordinary hand cast and a cast without paying the mana
cost whenever both are payable. Additional copies grant the same choice.
Kicker, bargain, and required sacrifices still apply to the free cast.
High Fae Trickster changes the timing of an otherwise legal cast. Harbinger of
the Tides can supply its own flash permission for two more mana, including
when Omniscience pays its printed cost. If another effect grants flash,
Harbinger also offers the cast without the surcharge.

Sphinx of Forgotten Lore grants its chosen graveyard card flashback at that
card’s mana cost until cleanup. The permission survives the Sphinx leaving
play. A card with printed flashback keeps both prices. Either flashback cast
exiles the spell when it leaves the stack.

On its controller’s turn, Iroh, Grand Lotus grants graveyard Lessons flashback
for {1} and other instants and sorceries flashback for their mana cost. A
creature with an Adventure in the graveyard still needs another permission to
cast either face.

Casting the top card with Vizier of the Menagerie or Planetarium of Wan Shi
Tong moves each remembered library position up by one. A player who scried
the second card knows the new top. That knowledge stays private.

When Etali, Primal Storm attacks, its controller exiles the top card of each
library. They can cast the eligible spells in either order for free, or stop
and leave the remaining cards in exile. Choosing Charring Bite from an exiled
Twinmaw Stormbrood asks for its target. Once the cast commits, Etali offers
the remaining spell before anyone receives priority. The last spell cast
resolves first.

Volcanic Torrent’s cascade stops at the first nonland card with mana value
below five. If it finds Scavenger Regent, the chooser can cast Regent, cast
Exude Toxin at X=0, or decline. The resulting spell must also have mana
value below five.

Targeting Scavenger Regent with an opposing spell or ability triggers ward.
The spell or ability’s controller can discard a card of their choice to keep the spell or ability
on the stack. Declining counters it.

Exude Toxin costs {X}{B}{B}. At X=1 its stack mana value is 3. Casting it
without paying its mana cost fixes X at zero, spends no mana, and gives it
stack mana value 2. Both viewers can inspect this value. The v11 stack row
keeps the selected printing’s base value in `mana_value` and reports the
on-stack value in `effective_mana_value`.

A spell cast from a prepared copy is a prepared spell; one cast with flashback
is exiled as it leaves the stack. Both facts belong to the cast, so a copy of
the spell has neither. Observations, the v11 stack row (`from_prepared`,
`flashback`), the text and Academy's stack entries show them to both viewers.

Entry replacements supply instructions for the new object, rather than facts
carried from the old one. An "enters as a copy" choice (CR 614.1c) supplies
copiable values before the other entry replacements are considered (CR
614.12). While those choices are pending, the entering row shows that
prospective copy to replacements and observations. The pending entry owns it;
minting consumes those values in place. Without a new copy choice, minting
uses the retained printing, including any fixture edits, and discards the old
appearance's copy effect.

When Zhao, the Moon Slayer has a conqueror counter, Dark Depths enters
as a Mountain with zero ice counters. Zhao’s separate ability makes it
enter tapped. If Zhao has also lost all abilities, the Mountain effect
still applies in layer 4 and Dark Depths enters untapped.

CR 400.7a to c (effects and prevention applied to a permanent spell continuing
on the permanent) carry nothing, because no supported effect applies to a
spell. Madness and stickers (CR 400.7k, m) are not modeled.

Each id handed across is itself an `ObjectId` and goes stale on the next move.
A creature exiled "until this leaves the battlefield" that leaves exile and is
exiled again by something else is a different object, so it stays in exile
when the first source leaves.

Card tests see the same rule. `force_permanent_on_battlefield`, `cast` and
`enter_and_settle` return the id of the object the card is when they return;
after a later move, use the id the move returned. A test that follows one card
through moves the engine made asks the support helpers `now`, `zone_of` and
`permanent_of` (`tests/support/inspect.rs`), which go through the card, and
moves it with `move_card` (`tests/support/game.rs`), which moves whatever
object the card is now and returns the one it becomes.

#### Names

A binding's name says which id it holds:

| Name | Holds | Use it for |
| --- | --- | --- |
| `card`, `card_id`, `cards` | `CardId` | card-only zone readers, decklists, sideboards, ownership, and effects whose text says "card" |
| `object`, `object_id`, `objects` | `ObjectId` | zone lists, spells, permanents, targets, events and last known information |

The parameter type says which of the two a function works on:

```rust
// Works on one object, wherever it is; rejects a stale id.
pub fn zone_of(&self, object: ObjectId) -> Option<ZoneType>

// Works on cards alone: a token or a spell copy cannot be passed.
pub(crate) fn play_land(&mut self, player: PlayerId, card: CardId) -> Result<(), AgentError>
```

A caller holding an `ObjectId` narrows once with `card_of` and passes the
`CardId` on; a caller holding a `CardId` that needs the object asks
`object_of` at the point of use and does not keep the answer across a move.
`Object::is_card()` remains for code that only asks the question.

Names that do not bind runtime ids keep `card`: definitions
(`CardDefinition`, `CardDefId`, `card.rs`, which describe a card rather than
name an object in a game), card-name strings, and the published observation
and Python names (`CardData`, `StackTargetData.card_id`, serialized `card`
fields, `source_incarnation`, `target_incarnation`), which keep their spelling
and values for recorded evidence.

#### Storage and recorded bytes

A new object does not get a new storage row. Each card, token or copy has one
slot in `GameState::objects`, a flat vector, and a zone change rebuilds the
object in that slot and advances the slot's generation. An `ObjectId` is the
slot and the generation it was issued for (8 bytes, `Copy`); the zone manager
owns each slot's generation, and a lookup compares generations and reads the
vector, with no hash map. The store grows only when
a token or copy is created, and a zone change allocates nothing.

Slots are private to the store, the zone lists and the match-state hash. Cards
take the first slots in setup order (main decks, then sideboards), followed by
tokens and copies in creation order, which is the numbering recorded states
already use. A field recorded as a row number writes its slot, and a field
recorded as `{"entity", "incarnation"}` writes slot and generation under those
keys. The match-state hash therefore cannot tell a current id from a stale one
in the fields that used to hold a bare row; a later hash version writes whole
ids. Version 17 adds, with whole ids, the object that produced each pooled
unit of mana and each unit a payment spent.
Observations, replays and the Python API are unchanged.

### Zones

CR 400.1 names seven zones (ante aside). Each holds only the objects its rule
allows. Library and hand entries record which occupants are cards; the other
lists hold `ObjectId`s, and card-only readers narrow them through
`GameState::zone_cards`:

| Zone | Rule | Holds | Engine list |
| --- | --- | --- | --- |
| Library | CR 401 | cards, transient non-card objects | ordered `CardZoneEntry`: an `ObjectId` and optional checked `CardId` |
| Hand | CR 402 | cards, transient non-card objects | ordered `CardZoneEntry`: an `ObjectId` and optional checked `CardId` |
| Graveyard | CR 404 | cards, transient non-card objects | `ObjectId` |
| Exile | CR 406 | cards, transient non-card objects | `ObjectId` |
| Battlefield | CR 403 | permanents: cards and tokens | `ObjectId` |
| Stack | CR 405 | spells (cards and copies) and abilities | `ObjectId` for spells; `GameState::stack_objects` includes abilities and owns stack order |
| Command | CR 408 | cards (commanders, CR 903.6) and emblems (CR 114.1); empty until a format or card puts something there | `ObjectId` |

Decklists and sideboards are outside the game and hold cards alone.

```rust
for card in game.state.zone_cards(ZoneType::Hand, player) {
    // card: CardId
}
for object in game.state.zones.zone_objects(ZoneType::Graveyard, player).iter() {
    // object: &ObjectId; a token that just died may be here
}
// One card, wherever it is now:
let zone = game.state.zones.zone_of(game.state.object_of(card));
```

A prepared permanent links its battlefield appearance to a spell copy in exile.
The copy belongs to the controller at preparation. Its casting permission follows
the permanent’s current controller. Both viewers see the current caster and the
`follows_source_controller` term before control changes. If Agent of Treachery takes Blossom-Blessed
Angel, its new controller can pay for the linked Seed Suture and cast it during
their main phase with an empty stack. Emergency Phytomedic’s controller can
still cast its own Seed Suture. Casting moves the copy to the stack and ends the
link, even if the spell is subsequently countered. A source that leaves the
battlefield loses its link, and the next state-based check removes the exile copy.
A returning Angel creates a fresh copy. A Command bound to the departed Angel
is stale even when submitted at the current decision revision.

Skycoach Waypoint can target any creature. When its ability resolves, a creature
with a prepare spell and no linked copy gains one. Preparing it again keeps the
same copy and permission. The three FRA creatures that prepare during upkeep
check the link both when upkeep begins and when their trigger resolves.
Infinite Coursework taps and unprepares its attached creature when the Aura’s
entry trigger resolves. Only that creature’s linked copy ceases to exist.

Emeritus of Truce gives the chosen player a flying Inkling before comparing
creature counts. Its trigger can prepare only the battlefield appearance that
triggered it. Its prepared Swords to Plowshares exiles the target creature
and uses its last known power for the life its controller gains.

An ordinary token or copy that leaves the battlefield or stack is briefly in another
zone until state-based actions are checked: a token in any zone but the
battlefield (CR 111.7, 704.5d), a copy of a spell in any zone but the stack, and
a copy of a card in any zone but the stack or battlefield (CR 704.5e). It keeps
its kind there; such a token can't return to the battlefield (CR 111.8), exile
grants skip it, and `flow/sba.rs` removes it by its kind. A selection that
reads a zone or the cards milled or discarded this way (`Effect::SelectCards`
over `Zone`, `TargetPlayersZone` or `CardsThisWay`) offers cards alone; a
selection over other objects, such as the token an effect just created
(`CreatedThisWay`), still offers that token.

Library and hand entries hold an appearance and its optional physical card.
Non-cards awaiting state-based actions have no card association. Keeping both
ids in the entry lets `zone_objects` borrow an appearance without treating a
physical card id as one. The single entry sequence owns membership and order;
there is no parallel list to
synchronize. `GameState::zone_cards` delegates to the borrowed view's `cards`
iterator, which reads stored card ids and skips visitors. Other zones keep
mixed `ObjectId` lists and narrow at that same boundary.

`zone_objects` borrows an ordered view of the underlying object ids without
allocating. Size, top, shuffles, hidden-world reassignment and observations
include visitors at their original positions. Reordering moves the typed entry;
it cannot insert an unchecked id. Every entry serializes as its slot
number, preserving the old zone bytes and match-state hash. Shuffles operate on the
whole entry sequence, preserving RNG consumption. This matters even at decisions:
a Treasure sacrificed for mana remains in the graveyard during its color choice,
before state-based actions run.

Outside the game is not a zone (CR 400.11). Sideboards are fixed rosters held
beside the game (see [Decks](#decks)); a companion starts there and is put into
its owner's hand by a special action (CR 702.139a, 116.2g), becoming a card in that
zone.

### Where the engine departs from the rules

- **Shared zones are stored per player.** Battlefield, stack, exile and command
  are shared zones (CR 400.1); the engine keeps one list per player for each.
  Stack order is `GameState::stack_objects`.
- **Drawing retains the existing mixed-zone behavior.** `Game::draw_cards`
  moves the top library object to hand and increments draw history even when
  that object is a transient token. Card-only selection skips that token, but
  drawing does not skip ahead to another object.
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
pre-batch trigger snapshot and commit in `ObjectId` order; a stale id
is rejected rather than followed after a zone change.

Time Stop and Sundial of the Infinite end the turn during resolution. The
engine drops earlier waiting triggers, exiles the stack, and checks state-based
actions while still in the interrupted step. A required legend choice pauses
the procedure there. New triggers wait until cleanup, after the active player
discards and marked damage clears. If a player gets priority in cleanup,
another cleanup step follows. Combat-only control ends when combat ends.
Turn-long control applies to every cleanup discard that turn.

The Dominion Bracelet locks its activation price using the equipped creature’s
power. When several Bracelets grant that ability, each offer identifies the
Bracelet it exiles. The payment receipt retains that object and the mana spent,
even if the ability is countered. Construct a Cosmic Cube puts its sacrifice
trigger on the stack when a placement crosses seven plan counters. A successful
sacrifice puts a separate targeted control trigger on the stack. Once either
card’s control effect resolves, the chooser makes decisions during the targeted
opponent’s next turn using that opponent’s resources.

**Proposed events.** Damage, life change, destruction, counter change and zone
movement are submitted as a typed `ProposedEvent` before any state mutates.
Replacement and prevention definitions from the immutable card definitions
apply once each in `(source ObjectId, definition index)` order; the surviving
event commits and emits the ordinary facts, so triggers, observations and
presentation consume only committed events. Supported replacements are
prevent N damage, double damage, enters tapped and enters with counters. The
choice CR 616 gives a player among several replacements is not offered: the
order is deterministic so replay, training and search stay exact, and a card
that needs more stays unsupported rather than approximated.

**Optional payments.** A resolving optional cost is `Effect::OptionalPayment`
with a `PaymentCost` (mana, waterbend taps against a generic obligation, or
life) and `if_paid` / `if_declined` effect lists. `PaymentProgress` owns the payer, the
`CostComponent` still owed and a `PaymentReceipt` of what has paid, the same
two records a cast or activation keeps: `CostComponent::resolving` is the one
place the definition's cost becomes what is owed, and a waterbend tap or a
life payment is the step an announced cost takes. Completion
prepends exactly one branch to the effect frame; counter-unless-paid uses the
same progress with the spell's controller as payer. A mana ability activated
during payment stays committed if the payment is then declined, and waterbend
taps are cost payment, never tapping for mana. A life cost ("Ward—Pay 7
life.") is one action: Pay is offered only to a player whose life total is at
least the amount (CR 119.4) and loses the life at once, under manual and
automatic payment alike; no mana ability is offered for it. People and policies see the same
native offers (pay or not, waterbend, modal colors, discard), with no
human-only payment flow.

## Worlds

The engine has no world label. `managym.world_identity()`
([`world.rs`](../managym/src/world.rs)) names the installed cards by their
digest; checkpoints, recordings and evaluation records written before the
label was removed still carry one, which nothing compares
([history](../WORLDS.md)). `managym.WORLD_VERSION` is a deprecated alias,
removed in 0.3.0. Which checkpoints load is
[compatibility](observation-and-model.md#running-a-checkpoint-on-a-later-revision);
which measurements line up is
[evaluation](evaluation.md#comparing-measurements-across-revisions).

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

Training does not take deck files; it takes a **deck source** in the
self-play settings (`SelfPlay.deck_source`, with no `match`). `UniformDecks`
(`uniform-distinct-v1`) draws two different decks per game uniformly over
ordered pairs from that game's native seed, so a seed reproduces decks and
seats; the TLA pilot's source is the frozen trophy corpus
([training](training.md#self-play-settings)). Every deck a source lists is
admitted when the settings validate, and again natively at each game root.
`UniformDecks(decklists="closed")` closes every game the source deals; each
game has one setting shared by both seats. Use `decklists="mixed"` for a
seeded, balanced per-game open/closed draw. A checkpoint from such a run binds nothing about the decks, so it
plays any deck the pool admits, including ones it never saw and cards added
after it was trained. Checkpoints bound
to an authored setup refuse imported decks even when shapes agree
(`manabot.model.world.validate_agent_setup`). Academy imports a pair through
**Import decklist files**; the CLI takes `--hero-deck-file` and
`--villain-deck-file` ([Academy](academy.md)).

### TLA decks with TLE bonus-sheet cards

TLA play boosters include a bonus sheet of 61 Avatar: The Last Airbender
Eternal (TLE) cards, collector numbers 1–61. TLA Limited decks use the same
importer, with no extra pack or flag. The engine defines all 61:

```python
import managym

decklist = managym.Decklist(
    "1 Avatar Enthusiasts\n1 Scout's Warning\n17 Plains\n",
    filename="tla-with-bonus-sheet.txt",
)
admitted, = managym.admit_decklists([("hero", decklist)])
print(managym.check_decks([admitted.deck]))
# [[]] — no admission issues
```

Every defined non-token card belongs to the world pool. This small example
checks `development-v1` admission, not Limited construction or booster
contents; use a complete deck for play.

All bonus-sheet cards now have definitions, including Heartbeat of Spring,
Release to Memory and Lita, Mechanical Engineer. Re-admitting the first 1,000
TLA PremierDraft trophy decks in the retained MTG-213 corpus admits all 1,000.
This check reparses each retained deck's original source against the current
world, preserving the corpus and its historical diagnostics.

Two bonus-sheet cards create tokens that need care when reading a board.
Release to Memory makes plain 1/1 colorless Spirits, which block and are
blocked normally; TLA's own Spirit token shares the name and can't block or be
blocked by non-Spirit creatures. The two are separate token definitions, and
observations and learning tokens tell them apart. Lita, Mechanical Engineer
makes a 5/5 colorless Zeppelin Vehicle with flying and crew 3, a token with no
printing and so no Scryfall id.

Bloodchief Ascension, Brought Back and Cityscape Leveler depend on state that
policy input v9 cannot hold: life lost this turn, cards put into a graveyard
from the battlefield this turn, and Powerstone mana that can't be spent on
nonartifact spells. Their decks admit and play under any input, but a v9 policy
refuses an observation that needs that state; train on them with input v10
([policy inputs](observation-and-model.md#input-v10-turn-history-and-powerstone-mana)).

`managym.set_cards(["tla"])` selects definitions by their first-printing set.
Adding `"tle"` still excludes reprints defined under earlier sets. Admit the
actual counted decklist against the world to include its bonus-sheet cards.

For unknown names in `DeckImportError`, check spelling and the installed
engine's definitions. A checkpoint loads when the engine still supplies its recorded input
contract ([what a checkpoint is bound to](observation-and-model.md#what-a-checkpoint-is-bound-to));
keep frozen runs on their recorded revision. New card support does not expand
saved training corpora or change their split assignments.

## What we know

- **Replay holds.** Evaluation cohorts replay exactly in practice. In the
  filter-scope screen all 3,600 monitoring and final games replayed
  ([technique screen](../experiments/reports/ataraxos-technique-screen.md)).
- **The encoding hides no decision.** The learnability search found no clean
  input collisions, and the fixed suite runs on relevant PR updates
  ([learnability](../experiments/reports/learnability-search.md)).
- **Collection dominates cost.** 75–83% of active training time across the
  profiled runs, 82.5% in the L4 run
  ([training time](../experiments/reports/training-time.md)).
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
