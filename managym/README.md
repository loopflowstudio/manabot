# managym

The deterministic Magic: The Gathering rules engine and training environment
behind Manabot. Rust, with PyO3 bindings consumed by `manabot` (the agent).
Install the `managym` distribution for `import managym` without the training
stack. `manabot` depends on the exact same version and keeps `manabot.managym`
as a compatibility alias. Rust sources stay in this directory. Start with the
[Python installation example](https://github.com/loopflowstudio/manabot/blob/main/docs/engine-python.md).
Both `Env` and `VectorEnv` accept `player_policies=("agent", "heuristic")`:
each native seat can use an external agent, random play or the built-in
heuristic, without installing `manabot`. See
[per-seat players](https://github.com/loopflowstudio/manabot/blob/main/docs/engine.md#per-seat-players).

## Recording, statistics and diagnostics

```python
capture = ScalarCapture(seed=7, games_path=Path("games.jsonl"))
capture.reset(game, players)
while not game.is_game_over():
    capture.step(game, choose(game), annotation={"value": 0.31})
capture.close()
```

`managym.recording` writes one small row per finished game: seed, setups,
actions and state digests. That one record is what `managym academy replay
games.jsonl` steps through, what `managym.statistics.reduce` computes card
performance, behaviour rates and your own reducers from, and what
`managym.export_arena` turns into an Arena-style log that outlives the engine
version. `record_fraction` samples games; a `journal_path` writes the same
record durably around every native call, for a call that may never return.
`managym.diagnostics` marks the phases of your loop and leaves the phase,
stacks and games in progress when one stalls.

Give the capture `statistics_path=Path("statistics/worker.jsonl")` for native
per-game card counts on every game, independent of record sampling. Read them
with `read_statistics` and `CardPerformance.from_statistics`. Low-level callers
can enable/read/drain counters directly on the Env.

All of it works without manabot, Torch or Etude. Records and statistics hold
hidden information: keep them private; Academy shows each viewer only what
that seat could see. See the
[engine guide](https://github.com/loopflowstudio/manabot/blob/main/docs/engine-python.md#record-games)
for options, costs, interrupted games and replay.

Decision memory is also opt-in: construct `Env` or `VectorEnv` with
`decision_memory=True` to retain viewer-safe past decisions. The training
wrappers derive this choice from positive observation record capacity; disabling
it leaves empty logs and skips capture. `decision_memory_include_own=True` enables
both capture and own-seat records. Search clones retain the existing log without
recording hypothetical choices. Saved diagnostic inputs retain the setting;
older inputs with no setting keep their original enabled behavior on replay.

## Play in the browser

The distribution includes Academy, a browser table for playing a game against
the engine's built-in opponents and stepping through recorded games:

```bash
uv venv --python 3.13
uv pip install --python .venv/bin/python --only-binary :all: 'managym[academy]'
uv run --no-project managym academy
```

This starts a local server and opens `http://127.0.0.1:8000` with a game
ready; Ctrl-C stops it. It needs CPython 3.12 or 3.13 and nothing else: no
manabot, Torch, Node or Rust. Finished and unfinished games are listed under
**Games** and open in **Replay**. `uv run --no-project python -m managym.academy` is the same
command, and `uv run --no-project managym academy --help` lists the port, host and storage
options. Cards show their Scryfall art with no extra step: the server fetches
each image the first time it is shown and keeps it in `~/.cache/managym/card-art`;
`ACADEMY_CARD_ART=off` turns that off ([card art](https://github.com/loopflowstudio/manabot/blob/main/docs/academy.md#card-art)).
Installing `manabot` in the same environment adds trained opponents.
The [Academy guide](https://github.com/loopflowstudio/manabot/blob/main/docs/academy.md#running-academy)
covers playing, replays, trained bots and launch troubleshooting.

### Start a game with your chosen decks

```bash
uv run --no-project managym academy play --deck ur_lessons --opponent-deck ./my-deck.txt \
    --opponent search --sims 64 --seed 7
```

The browser opens already in that game: UR Lessons against the decklist in
`my-deck.txt`, played by Search 64, with deal seed 7. Each deck is a bundled
deck name or a decklist file; `--opponent` is `search`, `random`, `passive`
or a registered policy alias/identity. A rejected decklist or unknown deck stops the command
with the reason before anything opens.
Bundled names take precedence over same-named files; use `./ur_lessons` to
select a file named `ur_lessons`.

The same choices are in the browser. Open **New game**, expand **Browse decks
and cards**, and assign a bundled deck to each native seat. Choose an opponent
such as **Search 64**, then click **Start game**.

For your own lists, expand **Import decklist files** and check **Use imported
decks for the next game**. Upload or paste both decks, one `count card name`
per line, with optional `Main` and `Sideboard` headings. Click **Check decklists**,
fix any reported issues, and select **Start game**. Imported decks use the
development format and supported cards. Registered policies check the complete
setup, including imported lists, against their own admission requirements. See the
[play guide](https://github.com/loopflowstudio/manabot/blob/main/docs/academy.md#play-a-game).

For several games against a model, **Rematch**, **Swap sides** and **Change
opponent** retain unrelated setup and group recordings in one session. See
[playing a series](https://github.com/loopflowstudio/manabot/blob/main/docs/academy.md#playing-a-series-against-a-model)
for deal modes, results and recovery. **Edit next game** opens the session draft:
**Save next setup** retains it, and **Start next game** starts a continuation.
Global **New game** opens a separate session setup; **Cancel** discards edits.
Changing both controllers to bots in the session editor offers **Start new
watched game**, preserving the Play session.

### Play against your own policy

Academy loads external bots without importing your training framework or Torch.
Pass a Python file, a saved `PolicySpec` JSON, or a `managym.bots` entry point:

```bash
# Plain (obs) -> int | probabilities function, or a managym.policy.Policy object
uv run --no-project managym play --opponent ./my_bot.py:choose --decklists closed

# Saved PolicySpec JSON (e.g. {"provider": "manabot", "artifact": ".../iter-0010.safetensors"})
uv run --no-project managym play --opponent ./iter-0010.policy.json --decklists closed

# Pit two policies against each other in Watch, or list discovered policies
uv run --no-project managym watch --bot1 ./my_bot.py:choose --bot2 heuristic
uv run --no-project managym opponents
```

When you pass a `.py` file without `:attr`, Academy looks for `policy` or `choose`
at module scope. For stateful/recurrent bots, probability/value outputs, or
custom model architectures, implement `managym.policy.Policy`
(`describe`, `admit`, `load -> PolicySession`) or register a provider under the
`managym.bots` entry-point group. From Python, pass a live policy directly to
`managym.academy.launch.serve(policies={"my-bot": policy})` or
`managym.academy.launch.play(opponent=policy)`. See
[independent Python policies](https://github.com/loopflowstudio/manabot/blob/main/docs/academy.md#independent-python-policies)
for the full contract.

### Watch a recorded game

```bash
uv run --no-project managym academy replay games.jsonl
uv run --no-project managym academy replay .runs/demo/games --game 3
```

Any game managym recorded opens with only `managym` installed: one your own
loop wrote through `managym.recording`, a training run's `games/` directory
(a recipe config's `record_games`) or an evaluation's. The browser opens in
**Replay** at the first recorded choice or this tab's saved position; use
**Next**, **Previous** or **Play** to follow choices, consequences and the
result. Each choice shows the probabilities, value and update the chooser
recorded, and **not retained** for what it did not. `--game` is the game's
position, counted from 0, or the start of its digest; run the command without
it to list the games. In the browser, the same is **Open a game file** under
**Watch** (`/replay?watch=1`).

Academy plays the recorded actions through the engine again and checks the
recorded state digest before every decision, without running the original
players or loading their checkpoints. A game cut off part-way, or one this
build does not reproduce, opens up to its last verified decision and says
where and why it stops. A game from another world is refused with the managym
version that recorded it. `managym academy replay <recording-id>` reopens a
game from **Games** in the same browser and address (including port), or when
it is shared and finished.

Evaluation command traces (`traces/<cell-id>.commands.jsonl.gz`) and games
exported by `manabot export-game`, both written by earlier releases, still
open; those two are verified by manabot, installed at the same release
version. The
[replay guide](https://github.com/loopflowstudio/manabot/blob/main/docs/academy.md#open-a-replay-directly)
covers selection, privacy and troubleshooting.

### Launch from Python

```python
from pathlib import Path

from managym.academy.launch import serve

with serve(port=0) as academy:
    game = academy.new_game(deck="ur_lessons", opponent_deck=Path("my-deck.txt"), seed=7)
    replay = academy.replay(Path("out/traces/c41.commands.jsonl.gz"), game=3)
    academy.open(game.url)
    academy.wait()  # until Ctrl-C; leaving the block stops the server
```

`serve` opens no browser; call `academy.open()` to open the bare table.
`game.url` and imported evaluation replay URLs open in any browser that can
reach the server; recording ids retain the access rules above. Failures raise
`AcademyLaunchError` with a `code`. See
[Launch from Python](https://github.com/loopflowstudio/manabot/blob/main/docs/academy.md#launch-from-python).

For building a deck before play, see [Limited deck building](../docs/deck-building.md):
`CardPool` holds the cards available, `format.Limited` holds the rules, and a
build started from both reports the copies still available and when stopping
is legal. The same rules read and write NumPy arrays over the card axis for
batched builders.
Format rules live in `src/format/` (`format::limited`); they build on the
definitions, decklists and world admission in `src/cards/`.

[Engine development principles](PRINCIPLES.md) records the lessons from
Allies–Lessons, Elemental and TLA: shared rule ownership, preserved choices and
timing, trustworthy learning evidence, and measured runtime/search performance.

## Architecture

managym is Manabot's authoritative world: match execution, semantic
Commands, viewer Observations, deterministic replay, exact forks, and the
meaning and materialization of possible worlds. The cross-package contracts
are in the [README](../README.md#contracts-the-code-enforces).

1. **`src/agent/`**: RL-facing API (`Env`, action spaces, observations)
2. **`src/flow/`**: Game progression (turns, priority, combat)
3. **`state/`**: the `managym-state` crate — rules vocabulary and game-state
   types (cards, zones, mana, effects), re-exported here as `managym::state`;
   **`src/state/`** adds players, decks and the canonical match-state hash
4. **`cards/`**: the `managym-cards` crate — card set modules (one per printed set)
   and their tests, reached by the engine only through `managym_cards::catalog()` and `token_catalog()`;
   **`src/cards/`**: `ContentPack`, rules fixtures, authored setups, pool support and learning tokens
5. **`src/format/`**: Format construction rules and card selection;
   `limited.rs` adds Limited pool counts and deck size rules to world admission
6. **`src/infra/`**: Logging and profiler infrastructure
7. **`src/python/`**: PyO3 bindings and Rust→Python conversions

Each card is a function in the file of its first printing's set under
`cards/src/`, returning the engine's own `CardDefinition` (key, Scryfall id and
`PrintedCharacteristics`) in its rules types (`Effect`, `Ability`, `TargetSpec`,
…). Tokens are not cards: each is a `TokenDefinition` in `cards/src/tokens.rs`
with the same `PrintedCharacteristics`, keyed `token.*` and created by that
key, with a Scryfall id unless it has no printing. The engine registers both
as `ObjectDefinition` rows with token status. In a game every object has a kind
(`ObjectKind::Card`, `Token` or `SpellCopy`): a token is never a card, every
zone change makes a new object with a new `ObjectId` while a `CardId` follows
the physical card, and
each zone holds only the objects the Comprehensive Rules allow it
([objects and zones](../docs/engine.md#objects-and-zones)). The engine runs these values
directly, and
the learning catalog (`src/cards/tokens.rs`, `managym.catalog_tokens`)
derives policy tokens from it. `src/world.rs` owns the card/deck
support check (defined cards, excluding tokens and rules fixtures) and the set
filter. Known rules bugs are tracked as Tasks without revoking playability. Keep
decklist data separate from card behavior; see
[the content contract](../content/semantic/README.md).

Dependencies flow: python → agent → flow → state/infra, and across crates
`managym` → `managym-cards` → `managym-state`. `cargo test` in this directory
runs all three (the workspace's default members).

Authored setups come from `AuthoredSetup::player_config` (Python:
`managym.authored_deck_setup(setup, deck_key)`, naming the setup by preset or
by the `pack_key` frozen evidence records). Carry that complete config
when swapping seats or rebuilding roots: deck and sideboard together identify
an authored setup. `PlayerConfig::new` intentionally gives custom setups an
empty sideboard; it does not infer one from card names or a matching main deck.
Sideboard rosters are fixed at setup, and remaining copies are derived from
absence from game zones. They are not part of the library or a new game zone.

Viewer observations carry remaining owner copies in `agent_sideboard`, using
private `candidate_id` values that join Learn action focus. These are outside-game
descriptors, not `CardData` with a fabricated zone or in-game `ObjectId`.
Both players expose definition-only initial and remaining sideboard counts;
only the owner gets copy candidates. Python observation JSON and validation
use the native Rust implementations so new visibility fields cannot drift.

The policy input factors resolved `EffectRecords` through `agent/shared_effects.rs`
before admission: complete equal payloads share one header and typed detail
slice, with one explicit membership per original record. Controller, exact
source appearance, timestamps, fixed affected sets and duplicate edges survive.
Public object IDs are joins; private object storage slots do not enter this view.
The native encoder writes bounded arrays of 128 × 18, 4096 × 3 and 2048 × 2
for every caller; every capacity fails explicitly with
`managym.ObservationCapacity`. Local facts exclude
active effects.
Projection never mutates or semantically coalesces the engine’s records.

Catalog naming uses `Decision::NameCard` over a shared immutable `NameVocabulary`.
The catalog builds sorted real names once, including back faces and excluding
tokens and rules fixtures. Decisions/forks share that allocation; canonical state
records its digest. A decision holds a `NameScope`; a scope over
`NAME_BRANCHING` (64) names offers a deterministic balanced partition into
contiguous groups, and choosing one only narrows the scope. Selecting a name
constructs one `ExileNamed` effect, then enters the existing private search
continuation. Public names and optional learning definition references are not
in-game card objects or deck admission. The policy input carries name joins and
group spans in `name_candidates[A,4]` at ordinary action capacity.
See [policy inputs](../docs/observation-and-model.md#catalog-naming-input).

Spell-level modes are announced before targets and payment. Independent target
clauses use `Effect::TargetClause`; effects within a clause share its targets.
Repeated specs in separate clauses may choose the same object. A resolving
frame retains its active clause and each chosen slot's resolution-start eligibility
across suspended decisions (CR 608.2b). Effects use those retained results; live
presence checks separately prevent following departed objects (CR 400.7).
A trigger's mode and target choices, changes of target, and resolution all
check source-relative restrictions. A power bound reads the live source's
effective power or its last battlefield power after departure. “Another”
excludes both graveyard arrivals of a departed melded source until each
changes zones again. Illegal slots remain positional, and zero optional
targets still resolve.
`ObjectSelector::Targets` applies one verb to all eligible slots of that clause
in one event batch. `ForEachTarget` instead runs a separate child program per
slot, retaining its position across choices. Leaves from the Vine's simultaneous
counter placement uses the former.
Graveyard targets
carry an exact `ObjectId`, so continuations cannot follow a physical card into
the object it later becomes. Mutate permanents at their event authority;
source-relative effects require the original battlefield object.

For a per-creature combat damage trigger or a life-loss trigger,
`EffectValue::TriggeringAmount` reads the amount of the event that fired it:
the damage dealt by that hit, or the life actually lost. The player ordering
simultaneous triggers sees each amount in its offer, and both players see it
on the stack. Quilled Greatwurm counts all combat damage each creature dealt
in that step, including trample damage split across recipients. Drake Hatcher
puts that many incubation counters on its source even if its power has since
changed. Its activation
spends three incubation counters during payment. Countering the ability
leaves them spent, and resolving it creates the flying Drake.

A trigger that asks about a spell after it leaves the stack reads the spell
as cast. After Charring Bite is countered, the waiting trigger still sees a
red sorcery with mana value two. Its pending description names Charring Bite
even though Twinmaw Stormbrood is a creature card again.

Death triggers check whether each permanent was a creature as it left the
battlefield. High-Society Hunter draws for each other nontoken creature that
dies alongside it, including a land that was copying a creature. Sacrificing
a creature token to its attack trigger puts a counter on Hunter without a draw.
That sacrifice also satisfies Cackling Prowler's end-step condition. Both
viewers can see whether a creature died this turn, including a token or a
land copying a creature. Exile does not count. The fact resets each turn.

Delayed abilities compose event matching, frequency, expiry, captured subjects
and effects. `CreateDelayedTrigger.event` only matches the event;
`DelayedFrequency::Once` consumes its watcher on the first match, even if the
occurrence is countered or later does nothing. `EachMatch` keeps waiting until
`DelayedExpiry::ThisTurn` removes the watcher at cleanup, or `Never` supplies no
stated expiry. Exact-object death watchers retire when the watched appearance
leaves, regardless of frequency. Expiry removes waiting watchers, never queued
or stacked occurrences. No printed trigger vocabulary is migrated.
`CreateDelayedTrigger.remember` resolves an ordinary `ObjectSelector` when the
ability is created. Its exact appearances travel through placement, stack and
suspended resolution; `ObjectSelector::Remembered` never follows a departed
object into a new appearance. Event context (such as the spell just cast) stays
separate. End-step timing can wait for either player's step or the controller's,
independently of what was captured. Viewer-safe relations and text expose
the remembered references, expiry and frequency. Relation row kind 9 carries
frequency (column 28: 1 once, 2 each match); column 48 keeps its expiry meaning.
Missing historical frequency is unknown.

Public hand knowledge is a minimum multiset of `CardDefId` counts on `Player`,
not a set of revealed physical cards. Exact library positions are retained
per viewer in `Player.known_library`, and projected as `PlayerData.known_library`
([library knowledge](../docs/engine.md#remembered-library-positions)).
Private moves discard uncertain locations; both world samplers preserve the
remaining facts. Search forks by full clone: clone the
`Game` before a normal Command `apply` and compare `search_state::witness` to
check that the source is unchanged. Exercise nested choices through Commands,
not by direct calls to resolution helpers.

## Build and test

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

**CI runs `cargo test` in debug — validate in debug before landing.** The
engine guards its invariants with `debug_assert!`, which compiles out of
release entirely, so a test can pass green in `--release` and still fail CI.

New registered real cards also need their Scryfall shell and Oracle text in
`tests/fixtures/scryfall_cards.json`; `cargo test --test conformance_tests`
checks every registration against that snapshot. For an accepted rules-world
change, record a new versioned conformance corpus and update the CI/test root
without rewriting earlier receipts. The current corpus is
[semantic-kernel-w17-v2](../conformance/semantic-kernel-w17-v2/).

After changing Rust, rebuild the Python extension into the uv-managed venv:

```bash
uv run --extra play maturin develop --release \
  --manifest-path managym/Cargo.toml --features python
```

Run this from the repository root with its CPython 3.13 development environment.
For a 3.12 compatibility check, use a separate environment and rebuild for that
interpreter; a `cp313` extension cannot be reused by `cp312`. Follow
[AGENTS.md](../AGENTS.md#commands) when replacing an already loaded extension.

The [pinned Comprehensive Rules audit](https://github.com/loopflowstudio/manabot/blob/cac6fa27d/docs/rules/audit-2026-10/README.md)
inventories every numbered entry, with explicit unknowns and scoped evidence.
Card-shell checks and replay agreement do not independently prove rules
correctness. Prefer directed tests with independently specified outcomes and
small bounded combinations before adding longer random-game cohorts.

Check the committed inventory offline from the repository root with
`uv run python scripts/audit_rules.py`. Supplying `--source <downloaded-txt>`
also checks the pinned byte digest and exact source inventory. Do not upgrade
child rules merely because their family has a test or a matching citation.
The Prism/Treasure audit probe now follows explicit nested payment decisions
and passes its final mana, sacrifice and empty-stack assertions. Its original
failure remains historical evidence; this bounded pass does not certify all
payment rules. See the audit's
[ownership review](https://github.com/loopflowstudio/manabot/blob/cac6fa27d/docs/rules/audit-2026-10/unification.md)
before using its historical Task attribution.

### Native allocator

Native builds use `mimalloc` by default (`default = ["mimalloc"]`). Pass
`--no-default-features` to Cargo to select the system allocator instead. For a
Python build, run the following commands from the repository root to build and
measure four-worker collection.

```bash
uv run --extra play maturin develop --release \
  --manifest-path managym/Cargo.toml --features python
uv run --no-sync python scripts/bench_sharded_collection.py --expect-allocator mimalloc
```

For the system comparison, pass `--no-default-features --features python` and
`--expect-allocator system`. `--no-sync` preserves the local build. The
benchmark checks the loaded extension's `managym.NATIVE_ALLOCATOR` before
timing. The separate Cargo feature `benchmark` counts Rust allocation requests
on either allocator and sets `managym.ALLOCATION_COUNTING` to true. Use it for
footprint diagnostics. Collection timings require counting to be disabled.

## Style

```rust
// filename.rs
// One-line purpose of file

use crate::flow::game::Game;
use crate::state::player::PlayerId;
```

Prefer explicit types and focused modules. Keep game behavior in enums +
`match` expressions instead of inheritance-like abstractions.

Bounded observation encoders reserve card rows after the owner's visible cards
for outside candidates. Their seven zone bits stay zero and the explicit outside
bit is set; action focus uses the same padded object table as the model. The
current default has 512 action rows. Card, permanent, action and focus capacity
excesses raise encoding errors by default; they must not be treated as a random-policy
fallback. An explicit `action_overflow="first-legal"` policy retains the first
`max_actions` legal actions with their original indices and Commands. The TLA
pilot selects 64 slots and records replay-linked cutoff counts; the native and
Academy legal action sets remain complete. Other capacity excesses still fail.
Complete outside program bindings and public known-hand definition
counts still require the semantic input path during the Learn migration.

The optional [policy-history v1 projection](../docs/observation-and-model.md#policy-inputs) is a bounded
derived suffix updated at the native event append. Observation reads share it;
clone and diagnostic injection must preserve its documented lifecycle.

Every match, authored or not, runs on the world's one catalog
(`cards::content_pack::default_content_pack`): every set-file definition in name
order, so authored setups are decklists over the world pool, not packs.
`PlayerConfig.submitted` (Python: `world="world"`, `managym.WORLD_SETUP`, set by
`Deck.player()`) submits a deck to the installed world: it must pass
`development-v1` admission against the world's supported cards (see [decklist
import](../docs/engine.md#decks)) or reset fails. Fixed setups and tests leave
it unset and run as given without submitted-deck construction checks.
Any other `world` value is rejected: authored content packs are retired.

Mana feasibility and production share typed costs and pools. `ManaClass`
retains each class's spending restriction and spend effect; `SpendingContext`
distinguishes casting, activation and resolving costs. A player's `ManaPool`
keeps, unit for unit, the object that produced its mana (`ManaOrigin`); units
of one class and color are interchangeable, so the oldest is spent first.
Among pooled units of different classes that can pay the same pip, the
automatic payer spends the most specific first (`ManaClass::spend_order`):
mana only some costs can use, narrowest restriction first, or mana whose rider
applies to this cost; then ordinary mana; last, mana whose rider this cost
would waste. Manual payment offers every class in its usual order.

A cast or activation is paid as one `Payment` (`flow/payment.rs`): the announced
`Operation`, the `CostComponent`s still owed in the order they are paid, and
the `PaymentReceipt` of what has been paid. `Game::cost_components` fixes the
total cost once announcement ends; `PendingChoice::Pay` offers the component
at the front of `owed`, and each choice shrinks or removes it and appends to
the receipt. The receipt has one entry per unit of mana spent, with its origin
and applied rider, and one `Contribution` per object tapped, sacrificed,
discarded or exiled for the cost, naming what was done (`How`) and the part of
the cost it paid (`Toward`), and the life paid. A chosen sacrifice is recorded
at once and leaves as the whole cost completes. The components after `Mana`
have no choice (`Life`, and `Source(How)`: the source's own tap, sacrifice, exile
or discard): they are paid from `owed` as the payment completes and
recorded on the same receipt, so a pending payment never reads the definition
again. A spell's receipt is shared by the stack object,
its resolution and the entering permanent's own triggers; an activated
ability keeps its own as `paid`.
Waterbend, convoke, improvise and delve pay part of a mana cost with something
other than mana, so they are the `Help` of the `Mana` component and no step of
their own: while it is owed the payer may tap a creature or artifact, or exile
a card from their graveyard, each paying {1} and adding a contribution, or pay
the rest with mana. One rule (`Game::help_taps`) says what a direct tap may
pay: by convoke an owed pip of the creature's color or {1}, by improvise {1},
and {1} of the announced waterbend part by any artifact or creature. Each
payment that leaves the rest payable is offered, and the payer chooses
(`CostComponent::TapPays`) when a tap has more than one: a white creature may
convoke {W} or {1}, and under Hama's waterbend a creature may convoke or
waterbend {1}; the receipt's `Toward` keeps which. That choice's mode names
the payment, the same in every decision (`TapPayment::mode`): 0-4 one
{W}/{U}/{B}/{R}/{G} pip by convoke, 5 {1} by convoke, 6 {1} by improvise and 7
{1} of the waterbend part. Only payable modes are offered, so they are sparse,
and the choice focuses the permanent just tapped. The payment model splits
a cost's generic mana into what those taps may pay (`help_obligations`), so
castability, offers and execution agree. `AdditionalCost::{Convoke, Improvise, Delve}` give a
spell the keyword and `StaticAbility::SpellsHave` gives it to other spells.
Offers and affordability ask the payment model with the help included
(`payable_with_help`, `while_paying`), so a spell castable only with help is
offered and a tap that would strand the cost is not; the automatic payer never
takes help, so when mana alone cannot pay the taps and exiles are the player's
picks.
Costs chosen while casting use the kicker choice: bargain
(`AdditionalCost::Bargain`) adds a `Sacrifice` component toward `Bargain` and
marks the spell kicked, and a conditional alternative
(`AdditionalCost::AlternativeMana`) replaces the mana cost while its condition
holds. `PrintedCharacteristics::cast_option` says which of these a card's
choice is, so the instruction, the offer label and the text observation say
"bargained" or "cast for its alternative cost" where the flag says `kicked`. "Forage or pay" (`AdditionalCost::ForageOr`) is one `Forage` component
whose step offers a Food to sacrifice, the graveyard cards to exile (the first
commits to three) and the mana instead. "Discard a card" as a spell's
additional cost (`AdditionalCost::Discard`) is the `Discard` component an
ability's discard cost uses, offering every card in hand but the spell being
cast. Crew and saddle are the same
`TapPower` component, toward `Crew` or `Saddle`; a saddled Mount carries a
`Modification::Saddled` record until end of turn. Station
(`ActivatedAbilityDefinition.station`) is a one-creature `Tap` component
toward `Station` that never offers the source; an activated ability's
resolving frame carries its receipt, and the station effect reads the tapped
creature's power from it (`EffectValue::PowerTappedToPay`). While the ability
is on the stack, both viewers see the amount used by the counter instruction
before replacements. It follows the tapped creature's power, down to zero,
and uses its last known power if it leaves the battlefield. A Spacecraft's
"N+" lines are static abilities conditioned on its charge counters.
The automatic payer pays only the mana component, and pays it with mana:
which creatures crew, saddle or station, every other object a cost takes, and
each convoke or improvise tap or delve exile when mana alone cannot pay, is
the player's choice under either `mana_payment` setting. With lands for the
whole cost it asks nothing. Waterbend taps are the exception and are still
made for the player.
Spending is a native unit/class decision whenever receipts or remaining pools
differ, including generic colors, restrictions, riders and combat expiry. Each
offer can complete every saved payment. Partial spending, nested activation
fees and their saved caller survive exact forks and Command replay; the spell's
receipt excludes those fees. Academy and policy inputs expose the same choices.
Chosen-color mana abilities pay their activation costs before their
immediate output decision. `Game::characteristics` derives battlefield grants;
those granted mana abilities never become activated stack objects. Waterbend
taps are cost payment, not tapping for mana. The view is the one reader of a
battlefield permanent's characteristics, legendary and ward included: the
legend rule reads its `is_legendary`, and every permanent observation characteristic reads it there (`tests/cards/observability.rs` classifies each
field as layered or printed by definition).

The cached raw layer board tracks discovered reads of mana pools and tapped
state, including nested predicates and granted statics. Independent pool/tap
writes retain the board and fixed activations; dynamic activation outputs and
costs refresh separately. Trigger kinds follow raw layer dependencies. Immutable
scans use `layer_views` (or `characteristics_many` when activation finalization
is needed), preserving caller order and validating the shared board once in
debug. Singular reads retain their cache checks. Opt-in `benchmark`
instrumentation attributes board, clone and dependency work without adding
counters to ordinary wheels.

A resolving spell remains on the stack until the zone-move authority commits
its departure. A spell cast with flashback carries that choice onto the stack.
A Zombie cast from the graveyard with Zul Ashur pays its normal cost, enters
the battlefield on resolution and returns to the graveyard if countered.
With Muldrotha in play, you can choose which permission casts the Zombie.
Casting through Muldrotha spends its creature allowance for the turn, even
if the spell is countered. Using Zul Ashur leaves that allowance available.
The zone-move authority applies flashback exile replacement and captures
cast provenance before removing the stack object. Do not remove spells early in
countering or finalization. Copied tokens detach immutable characteristics from
the source definition, retain its semantic identity, and expose their actual
legendary status in observations; canonical hashing includes their copied data.

Resolving optional costs use `Effect::OptionalPayment` and `PaymentProgress`.
Their [ownership and authoring contract](../docs/engine.md#resolution-contracts)
covers both ordinary mana and waterbend without a card-specific decision.
`OptionalManaPaymentX` first chooses an affordable X, then uses the same payment
progress. Its `WhenYouDo` branch creates a distinct occurrence retaining X;
`TargetRequirement.total_mana_value_x` bounds the whole chosen clause, including
retargeting. Named searches retain native permission to inspect the searched
zones only during the instruction; semantic capture uses that permission too.

Mana offers, waterbend affordability and automatic production use executable
witnesses from `flow/mana.rs`. A mana source is reserved while its printed
upfront cost is pending; `Decision::ManaUpfront` retains the parent decision
and frame. Only after that obligation finishes do its tap/life/sacrifice costs
commit and its output become a visible color decision. Mana abilities do not
create stack objects. The initial fixed-source witness keeps source order;
[automatic source preference](#automatic-source-preference) selects the taps.
A nested automatic prefix considers concrete pool remainders and the parent's
completion.
The announced activation's own tap, sacrifice and unpaid life fee remain
reserved throughout waterbend and nested mana decisions, including a saved
spending caller. A nested mana ability cannot consume those resources and
leave the announcement without a complete payment. Its own upfront life fee
is counted once, on that ability's output obligation.

Most payment questions never reach the search. With four untapped Islands and
an empty pool, "can this player pay `{2}{U}`?" is answered by adding: the pool
plus every source's output either covers the cost or does not, and the first
witness taps sources in source order until the pool pays, useful or not. That
direct answer applies when the question is one mana cost and no available
source has a mana ability beyond tapping itself for one fixed output: no color
choice, upfront cost, life payment, counted tap, sacrifice or limited use.
Restricted fixed outputs (CR 106.6) stay in the class, counted only toward the
purposes they may pay. A waterbend cost stays in it too: every source worth
mana is tapped for mana, artifacts and creatures worth none cover the generic
that taps may pay, and each candidate tap of one waterbend decision is judged
against one model instead of a new search. Automatic production never
activates abilities, so its single cost is direct on any board.

Legal actions computed for one decision build each player's sources once
(`PaymentScan`). Those sources are a function of the game alone, which cannot
change while the computation borrows it; the pending payment's reservations, a
sacrifice under consideration and payable life are applied per question. Debug
builds rebuild the sources on every reuse and assert they match.

The direct answer is exact or it is not used. Modal lands, filters, mana
abilities with other costs, counted taps, nested activations and the automatic
prefix of a nested activation go to the search described below, which stays
the one definition of a legal payment: debug builds run it beside every direct
answer and assert the same affordability and the same first witness. Automatic
source preference starts from that shared witness. Adding a kind of mana
source needs no change here unless it should join the direct class, which
requires a test comparing both paths on that source.

Payment search tracks an object's untapped resource, presence and used limited
abilities separately. Counted tap costs select distinct untapped matching
permanents through ordinary SelectCard decisions; The Cabbage Merchant can
activate twice by tapping four distinct Foods without tapping itself. Each
selected tap commits before the next selection and is unavailable to nested
costs or waterbending. Pending tap
counts use the continuation mode field; source and ability retain their usual
joins. Single-object costs keep their serialized definition shape.
Every registered mana ability must consume a finite resource. Search bounds
count repeatable outputs optimistically; every accepted payment still requires
an identity-preserving witness with explicit tap choices. Every saved counted-tap
cost must retain enough present, untapped candidates throughout the search;
production and spending cannot restore a consumed resource. The whole-payment
bound counts only outputs a pending cost could spend and takes the maximum of
competing uses of one tap. A separate conservation bound counts every mana
class and the net gain after future activation fees; saved outputs stay gross
supply beside their explicit unpaid costs. Restricted mana can therefore seed
a filter without being mistaken for mana the parent can spend. The color-set
bound applies the same supply to every set of owed colors, so one "U or R"
source cannot cover `{U}{R}`. Search
memoization shares feasibility across interchangeable sources (including
membership in every counted-tap set), while retaining source order and exact
identities in the first witness. An immutable branch state also serves as its
memo key; canonical permutations copy only when resource order changes.
The per-cost optimistic pool projects allowed classes onto color counts after
taking each class's output-mode maxima, without copying restriction metadata.
Concrete pools retain their exact classes throughout execution and search.
Spending alternatives are explored only after the pool covers the whole current
cost; a partial spend cannot interleave production, and its remaining pool must
cover the unpaid pips before saved callers are checked.
Repeated lands therefore contribute resource
counts instead of all identity subsets. A search checks the color-set bound at
its root and keeps identity keys until it has expanded 16 states; only then
does it classify the sources and apply both to every state, so ordinary small
payments pay nothing for them. Distinct
source behaviors, overlapping tap sets, nested obligations and concrete pool
remainders can still make exact search combinatorial; there is no general
large-board work limit or truncation of legal payments. The
[stress states](../docs/development.md#stress-states) measure payment and every
other engine layer on large duplicative and large diverse boards; run them when
changing this search.

Each Game caches at most 64 searched queries in FIFO order, retaining negative
answers and complete first witnesses. Exact keys include ordered source
identities and capabilities, tap resources, pool classes, payable life, ordered
caller obligations, source policy and the selected counted tap. Forks own
independent FIFOs but share immutable entries, source lists, output alternatives
and tap-candidate lists. Reservations detach source lists on writes. The cache
stays outside hashes and observations; debug builds recompute every hit.

Automatic-prefix searches start at the fewest taps whose largest fixed outputs
or waterbend contributions could cover the pool shortfall, conservatively
ignoring colors and tap competition. The cache and lower bound preserve complete
choices and first witnesses; neither substitutes a fallback for an infeasible
payment.

Spending restrictions (CR 106.6) are typed: `ManaRestriction::Only` holds
`SpendPredicate`s naming any subtype a spell must have or an activation's
source must have, `ExceptNonartifactSpells` is a Powerstone's exclusion and
`ExceptSpellsFromHand` is Heartwood Crafter's. Every cost carries a
`SpendingContext`: `Cast` with the spell's announced subtypes and types and
the zone it is cast from, `Activate` with the source's zone and subtypes, or
`Resolving`, which only exclusions admit. Observation data names each
restriction whole. The policy input's restricted-mana columns name exactly
Lesson and Shrine spells and Ally spells or sources (and, in v10, Powerstone
mana); every other restriction is a `pooled_mana_classes` class and a
`PooledMana` fact program, never a player column.

`Env.capture_decision(viewer, history_capacity=0)` supplies atomic decision capture
v6: the schema-8 semantic Observation, shared native continuation frames, bound public card
characteristics, visible object joins, content identity and optional public
history v1 (0 disables it; 1..32 admits a chronological suffix). Its SHA-256 covers
all fields except the digest itself using recursively key-sorted compact JSON.
`DECISION_CAPTURE_VERSION` owns this aggregate's inclusion/order/digest contract;
`SEMANTIC_DECISION_VERSION` owns embedded schema 8. Historical captures retain
their original schema; w14 deliberately changes the current action contract.

The opaque native capture binds an Env and reset generation. Use
`execute_captured_command_json(capture, command_json)` for atomic origin and
current-digest validation plus ordinary semantic execution. Portable capture JSON
is evidence, not execution authority; clones need their own live capture.

Frozen viewer JSON omits continuation frames. To reconstruct a complete
stateless policy observation, use
`Observation.from_json(viewer_json, continuation_json=frames_json)`; the frames
pass through the same canonical validator as retained decision captures.
Neither input creates an executable decision binding.


The policy input has configurable fact capacity (512 by default). It keeps
trigger-placement action joins and retained
resolution-start eligibility in native current stack/continuation relations, and
widens player and card rows for cast permissions, the hand-only lock, restricted
mana by spending subtype and announced X. Its
layout, checkpoint admission and measurement commands are documented in
[the observation input contract](../docs/observation-and-model.md#policy-inputs).

For a v11 batch, call
`encode_observation_batch(items, layout, action_overflow="error")` with `Env`
or `Observation` items.
`RolloutPool.set_observation_layout(layout, action_overflow="error")` pairs with
`encode_active_batch()` and `step_active_batch(actions)` to return
`(active_slots, ObservationBatch)`.
`VectorEnv.step_observation_batch(actions, layout, action_overflow="error")`
returns the batch from the step together with per-stream rewards, terminated
flags and truncated flags. Each stream emits its observation and step events
as it advances. A strict stepping error can leave streams advanced.

For repeated collection, call `VectorEnv.set_observation_layout(state, layout,
action_overflow)` with one writable outcome array per stream for each of
`rewards` (float64), `terminated` and `truncated` (uint8). Use the ordinary
buffered reset, replacement and step methods. After reset, `observation_rows()`
returns one `(batch, index)` per stream in slot order. Masked steps preserve
parked slots. Isolated stepping reports the failed slot and keeps every
successful sibling's observation. A failed snapshot load leaves the prior
roots and observations available.

`observations_for_players(players)` freezes the requested viewers at their
current decisions. Pass one seat or `None` per stream. The result contains the
selected slot identities and a batch in that order. Any batch's
`gather(coordinates)` copies just the selected live spans from compatible
batches, preserving the requested order and repeated rows.

Each `ObservationBatch` owns its snapshot across later steps. Its `tables()` maps
each value buffer to `(seat_axis, offsets, values)`, with immutable byte
strings for native-endian int32 row offsets and packed, native-endian records.
One span belongs to each batch item, or to each (item, seat) if `seat_axis`
is true. Subtract consecutive offsets for live counts, including all-zero
live rows. Companion validity masks derive from those counts. Object joins
retain the layout's coordinates. The default overflow policy rejects excess
actions. With `"first-legal"`, `action_counts` gives the retained prefix.

`ObservationBatch` supports `pickle`, `copy.copy` and `copy.deepcopy`. Pickling
transports the layout, batch size and immutable byte tables without copying or
densifying live spans, preserving shared buffers across batches (such as after
`blank_decision_memory()`). Unpickling validates the v11 layout, table keys,
`seat_axis` flags, monotonic in-bounds span offsets and packed byte lengths,
raising `ValueError` on corrupt data. `ObservationBatch` has no public
raw-bytes constructor; batches originate from native encoding or pickle
restoration.

`layout.csr_fields[key]` lists each record field's name, dtype and logical
column index in byte order. IDs, joins, enums, unscaled counts and full signed
words use int32. Flags use uint8 and scaled scalars use float32. In a dense
input a word occupies two 16-bit columns. The packed field takes their common
name and the first column's index. Both fact payloads use raw integer units.
Divide by `layout.csr_fact_divisors[kind]` for their logical numeric values.
Decision memory includes separate presence flags for optional selectors.

```python
fields = batch.layout.csr_fields["actions"]
dtype = np.dtype([(name, dtype) for name, dtype, _ in fields])
_, offsets, values = batch.tables()["actions"]
starts = np.frombuffer(offsets, dtype=np.int32)
actions = np.frombuffer(values, dtype=dtype)
first_game_modes = actions["mode"][starts[0]:starts[1]]
```

For element arrays, the single field describes each element. An action-focus
row contains `layout.shapes["action_focus"][1]` consecutive int32 elements,
with -1 for unused focus slots. Dense buffer allocation uses `layout.dtypes`.

For dense batched buffers, `VectorEnv.set_buffers`, `RolloutPool.set_buffers`,
`encode_observation_into`, `encode_observations_into`,
`ObservationLayout.axis_extents` and `ObservationLayout.object_references` all
emit a `DeprecationWarning` targeting removal in `0.3.0` (retained alongside
the deprecated v9/v10 fork; v11 callers should use `set_observation_layout` and
`encode_observation_batch`). `layout.axis_extents(buffers)` returns one int64
vector per variable axis. Each item includes its occupied rows and rows needed
by joins. Keep the buffers stable until the scan returns. A key or shape
mismatch returns `None`. For each of those axes, `layout.axis_buffers` lists
the buffer names and their unbatched row dimensions. For a pair `(key, dim)`,
`layout.shapes[key][dim]` is the configured capacity of that axis.
`layout.object_references` lists each buffer's object-reference columns and
their index bases. A `None` column means every element is a reference.

The presentation context and text capture derive from the same typed frames;
portable schema-8 viewer JSON is validated by the native observation types.

`Game::object_pt` reads contextual power and toughness in every zone: the layered view on
the battlefield, and characteristic-defining abilities with the spell's
controller or card's owner elsewhere (including outside the game). Zone
filters use `object_matches_predicate`; printed-only matching belongs to
structural definition checks and private layer stages. `LookAndSelect` names
whether the remainder is random; chosen order suspends on native
`OrderLibrary`, placing cards first-to-draw first through ordinary SelectCard
Commands. Bottom ordering exposes the remaining inspected cards; Surveil uses
the same owner to order kept cards on top, retaining the chosen prefix privately
until ordering finishes. Top ordering has obligation code 27 (mode is the
committed prefix length); code 23 retains its bottom-order meaning.

`EffectFrame::completion` owns the return destination: the interrupted scope,
a retained caller decision, a completed-operation event, root spell cleanup,
or root priority. Completion events fire before the caller's suffix; Scry and
Surveil wait for every partition and ordering choice. `continue_resolution` runs
any pending batch before exposing the next decision. An `EventBatch` on the active
frame retains every proposal, applied replacement identity, pre-departure facts
and result until replacement choices and consequent linked returns finish. The
affected player chooses each replacement; applicability is recomputed after every
transformation. No member
commits until the complete batch is ready. Native obligation code 29 names this
choice, with source-specific labels and focus.

Mana activation sacrifice effects retain their output and parent payment there
after upfront mana, tap and life costs commit. Returning from a linked-entry
replacement restores the saved decision without paying costs again or opening
priority. Combat, SBA departures, spell finalization, activated costs, token
entry and Surveil movement use this owner. SBA progress distinguishes stable,
changed and pending; a pending turn action cannot enter priority stabilization.
Immediate fixture/arithmetic helpers require completion without a question and
fail explicitly if used for a choice-bearing event. Sacrifice facts match their
pre-move subjects after replacement choices and before any batch member leaves;
trigger placement still waits for resolution. The policy input projects the
retained program, results, completion and proposals through fact-kind 14 as bare
name/value facts, with viewer-safe reference slots in relation kind 10; a frame
that retains nothing emits no occurrence. Academy's pending-resolution summary
and LLM text are rendered by the same walk only through `describe_effect_execution`
(decision context and capture), never by a policy observation.

A permanent's `damaged_permanents` lists the live permanents it dealt damage
to this turn. Multiple hits on one recipient give one entry. Either permanent
leaving and returning breaks the relation. Both viewers get the public
permanent IDs in the observation and Academy frame. Policy input v11 joins
their permanent rows in relation kind 11, including with policy history off.
For combat damage to a player, `combat_damage_to_players` lists the seats hit
by that exact source this turn. An activated ability on the stack retains
this information when its source leaves. V11 encodes each seat relative to
the viewer on the permanent and stack rows. Steel Hellkite checks those
players’ current nonlands and their mana values when its ability resolves.
Each Hellkite can activate once per turn, including for X=0 before any hit.
Arcanis uses `ObjectSelector::ExactSource` to return the battlefield object
that activated its ability. A zone change ends that identity. Avatar Destiny
uses `ObjectSelector::This` to return the Aura from the graveyard after the
enchanted creature dies.
An effect on the enchanted or equipped permanent uses `ObjectSelector::Attached`.
It reads the attachment at resolution. If the source left in response, its last
known attachment identifies the exact permanent it last enchanted or equipped.

`SelectCards` snapshots eligible exact objects from its selector, retains min/max
cardinality across choices, and runs the chosen effects in a child scope with
`ObjectSelector::Selected`. `CardsThisWay` names the scope's latest mill/discard
result independently of a later `Move`; the caller keeps those results while
selection effects run. Mill, optional discard/draw and basic/tutor searches use
ordinary `MillCards`, `If`, `SelectCards`, `Move` and `ShuffleLibrary` composition.
An unrestricted `Effect::search_for_card` requires a choice while the library
contains cards. Rune-Scarred Demon uses an unrevealed move into hand. Micromancer
reveals the chosen card, which gives both viewers knowledge of it in hand.
`ObjectSelector::Zone` records source-zone/predicate provenance; both hidden-world
materializers refresh unknown selected subsets throughout saved callers and child
scopes, preserving cardinality and exact viewer-known choices. Restricted partial
selections exclude incompatible count worlds. Foretold assignments use the same
subset constraints: exact weights count only compatible ordered fills, and the
materializer samples that conditional measure. Commitments are captured before
zone reassignment, so a sampled face-down slot cannot erase an earlier selection.
Frozen count-only belief samplers reject these additional constraints before
inference. `ObjectBinding` carries exact appearance, captured zone/predicate and
per-seat exact knowledge through waiting, queued, placing, stack and resolving
trigger lifetimes. Both samplers use one assignment across those captures and
active selections; known exact captures remain fixed, and departed appearances
never follow a later object. Capturing `Moved` uses its new zone. This binding
provenance is canonical state, independent of viewer-safe reference projection. Library inspection exposes membership in stable definition/object order,
never draw order. Prospective battlefield entrants are public during replacement
choices and remain pinned in both samplers.

At opponent priority, hidden-world sampling deals uninspectable manifested
printings from the same pool as hidden hand and library cards. A pending spell
keeps its battlefield target and that target’s exact appearance across deals.
The controller can still inspect the printing in the sampled world. Restoring
the original snapshot restores the actual printing.

`Effect::Move.library_end` states top/bottom placement without a shuffle; its
placement commits before linked returns can pause. `put_in_library` authors that
operation, and `ShuffleLibrary` can follow after all child work. Ordinary `move_to`
retains shuffle-into-library behavior. Shared selection has obligation code 28;
retired codes 5, 6 and 7 keep their historical meanings.

`SuspendedResolution::continuation` walks retained frames and decisions root first;
observation projection and hidden-world sampling share that chain for caller
ancestry and inspection permissions.

### Automatic source preference

Pay `{1}{G}` with an empty mana pool and two untapped Mountains and a Forest:
automatic payment taps one Mountain and the Forest. With equal follow-up
scores, it uses a basic Forest before a green mana creature to pay `{G}`.

Automatic payment removes unnecessary fixed-source taps and uses a bounded
mana-cost heuristic to prefer the remaining sources (`flow/mana/autotapper.rs`).
It prefers, in order: more affordable follow-up costs, more hand costs, then
more instant-timing hand costs. It considers at most 64 distinct hand-cost and
ability records. Targets and other costs are only partially checked, and
different follow-ups may need incompatible spending remainders: the score is
not a count of playable cards or a global optimum. Single-use abilities with
the paid cost are excluded, including another source's equal-cost ability.

Further ties prefer basic lands, other lands, lands with activated abilities,
other permanents, then creatures; within those tiers, colorless output is
preferred. The class search stops after 1024 arithmetic checks and keeps a
payable candidate. Payments with saved obligations only receive surplus
pruning. If fixed sources cannot pay, the planner may also select the color
of a simple tap activation such as Kyoshi Village and removes surplus taps
(keeping creatures, utility lands and dual lands untapped ahead of basics).
The preview shows that source and its output. In manual payment mode, sacrifices, life payments and abilities with other
effects remain explicit payment choices. Native spending choices determine
the exact mana receipt. Source preference has no setting.

### Manual mana sources

Under the default `mana_payment="manual"` (with `mana_floats="off"`), a cast,
activation or resolution payment opens a prompt only when the player can finish
it in two distinguishable states. Simple tap-for-mana abilities — dual lands
such as Kyoshi Village, any-color sources such as Birds of Paradise or Arcane
Signet, Great Divide Guide grants, and restricted simple taps such as Jasmine
Dragon Tea Shop or Hermitic Herbalist — follow the same rule as fixed-output
lands.

With Kyoshi Village and one Plains untapped, casting a `{1}{W}` spell finishes
at once with no payment prompt. With four identical untapped Mountains for
`{2}{R}`, or Kyoshi Village and two identical Plains for `{1}{G}`, the game
likewise pays without prompting — tapping Kyoshi Village and one Plains,
leaving the other Plains untapped.

With Kyoshi Village and three untapped Plains for `{1}{W}`, choosing between
the dual land and a Plains is a real difference, so the payment stops and
offers every useful source beside `PayCost`:

```text
0  ActivateAbility  Tap Kyoshi Village for {G}/{W}
1  TapForMana       Tap Plains for {W}
2  TapForMana       Tap Plains for {W}
3  TapForMana       Tap Plains for {W}
4  PayCost          Pay the cost
```

Selecting action 0 taps Kyoshi Village and skips the `{G}`/`{W}` modal because
either color finishes identically (one Plains tapped, two Plains untapped).
Tapping Kyoshi Village toward `{1}{G}` with a Plains and a Forest left keeps
the color modal, because `{G}` leaves a choice between Plains and Forest while
`{W}` forces Forest.

- **When a manual payment skips.** A pending mana payment completes
  automatically when no non-simple mana ability, help action (convoke,
  improvise, delve, waterbend) or extra float tap is available, automatic
  payment can fund the cost, no choice source floats unspent mana, spent-color
  counts cannot differ (converge spells keep the prompt when a source can
  contribute more than one color), useful modes on each source share one spend
  effect, and the useful sources produce exactly the mana needed. A payment
  with spare sources can also skip when each useful tap produces one unit,
  the number of taps is fixed, and every optional tap comes from a single
  class of game-indistinguishable, unreferenced permanents. With Sol Ring and
  two Mountains for Gray Ogre, **Pay the cost** leaves one Mountain untapped.
  Tapping both Mountains before Sol Ring leaves one colorless mana floating.
  Any distinguishing state — a land targeted on the stack,
  enchanted, animated, carrying counters, or referenced by a trigger or
  continuous effect — places that permanent in its own class and keeps the
  prompt. Plain non-creature, non-animating lands played this turn are
  interchangeable with earlier copies, while creatures, non-lands,
  self-animating lands (such as Soulstone Sanctuary), and permanents with
  `power_up` abilities stay distinct when summoning sickness or turn of entry
  differs.
- **What always stops.** Non-simple mana abilities (sacrifices such as
  Treasure, life payments such as pain lands, mana fees such as filter lands,
  counted taps, once-per-turn limits, non-mana side effects, or triggered mana
  bonuses that alter a choice ability's output), extra taps under
  `mana_floats="usable"` or `"all"`, and costs needing help taps keep the
  payment open.
- **Actions and wording.** When a payment stops, every useful permanent keeps
  its own offer (identical sources are not merged) and must make progress
  while leaving the announced costs and saved payments payable. Fixed-source
  taps stay `TapForMana` (policy action `TapForCost` with
  `pays_directly=False`, Command verb `tap_for_mana`); simple choice-of-color
  taps stay one `ActivateAbility` per ability rather than expanding into
  per-color actions, keeping policy input and protocol shapes unchanged.
  Native `action.description` labels simple choice taps to match fixed
  sources — `"Tap <Name> for {G}/{W}"` for 2–4 explicit modes,
  `"Tap <Name> for any color"` for all five single colors, and unit or
  restriction detail when needed (such as
  `"Tap Hermitic Herbalist for 2 mana (Lesson spells only)"`) — while
  non-simple mana abilities keep `"Activate ability: <Name>"`.
- **Color modals and resolution payments.** During an announced or resolution
  payment with `skip_trivial` enabled, `Decision::ManaChoice` skips
  automatically when every legal mode leaves the same remaining state or the
  same determined completion with no floating surplus; deliberate floats at
  priority always keep the color choice. Resolution payments (`TurnFaceUp`,
  `Ward`, and other `Decision::Payment` prompts) share the same per-source tap
  and forced/interchangeable skip rules for both basics and simple choice
  sources: when only one outcome exists, turning a creature face up pays
  immediately and ward offers only pay or decline; when there is a real source
  choice, both fixed and simple choice taps are offered alongside pay (and
  non-simple mana abilities such as Treasure remain explicit).

These choices are on by default for every consumer. They do not expose undo,
or production at priority unless a seat
[asks for it](#mana-taps-at-priority-and-during-payment).

### Mana taps at priority and during payment

The seat chooses when to offer mana abilities at priority, including sources
with a choice of colors. With `off`, Kyoshi Village can pay for a green or
white spell during payment and can activate its draw ability at priority.
Two independent settings follow each seat. `mana_payment="manual"` offers useful source taps and automatic completion. With `"auto"`, the engine chooses how to fund each accepted cost, including sources, their colors and the units to spend. Optional costs still offer pay or decline. A deliberately floated source retains its output choice. Both payment modes support all three float levels.

```python
import managym

setup = managym.GameSetup([first, second], mana_floats=("all", "off"), mana_payment=("auto", "manual"))
```

Use `mana_floats=("usable", "off")` to offer priority floats when a listed action can
spend mana, the activation has additional costs or effects, or tapping that
source can trigger an ability.
The engine inspects the listed actions without checking individual mana colors.
Manual payment with usable or all floats also offers extra payment-time taps.

With two untapped Forests and a creature spell in hand, the first seat's
priority space is what it was, followed by one tap per source:

```text
0  CastSpell     Badgermole Cub
1  PassPriority
2  TapForMana    Forest
3  TapForMana    Forest
```

Taking action 2 taps that Forest and adds `{G}` to the seat's pool. The seat
keeps priority and is offered the remaining Forest; casting then spends the
floating mana first. Unspent mana empties with the pool at the end of the step
or phase, as any floating mana does. There is no undo.

- **The space only grows at its end.** Every action offered without the option
  keeps its index, `PassPriority` included; taps follow in battlefield order,
  one per source, identical sources included (eight Forests are eight
  actions). Under `action_overflow="first-legal"` the taps are the first
  actions dropped.
- **Manual payments.** During cast, activation and resolution payments,
  extra fixed-source taps follow the ordinary payment choices. A tap must
  leave reserved resources available and the committed cost payable. Mana
  made while a cast or activation is being paid pays toward it at once: the
  cost owed shrinks and the receipt names the unit's source, as a convoke tap
  does. Mana that was floating before, a source's surplus and mana that can
  pay nothing still owed stay in the pool for the spend choice, and so does
  new mana when pooled mana of another class (a rider, a restriction) or
  another color could pay the same pip. A payment ends as soon as no mana
  is owed, whatever is still untapped: surplus waits for the priority that
  follows, and a cost whose mana is {0} from the start never opens its mana
  step.
- **Automatic payments.** Once the player accepts a cost, the engine selects
  the sources, their outputs, any waterbend contributions and the units to
  spend. The player still chooses X, optional costs and spell targets.
- **Skipped decisions.** With floats off, a priority window offering only
  a pass is skipped. With all floats, any legal mana activation can open the
  window. With usable floats, a sacrifice or tap trigger can open it even
  without a listed spend. `Env.skip_trivial_count()` includes forced passes,
  other skipped single choices and internal automatic payment actions.
- **Seat order and replay.** Both settings follow the chooser. Fixed-source
  taps and direct taps use the `TapForCost` policy action with the
  permanent as focus, told apart by `pays_directly`.
  Choice-output abilities use their activation action. Omitting both fields
  produces the same native setup, game digest and recording bytes as explicit
  manual/off for both seats. Invalid values raise `ValueError`.

### Inspecting automatic payment

Ask what automatic payment would tap for one legal action, without taking it:

```python
from managym.decision import DecisionFrame
from managym.payment import preview_payment

frame = DecisionFrame.from_json(env.semantic_decision_frame_json())
preview = preview_payment(
    env,
    viewer=frame.actor,
    revision=frame.revision,
    fingerprint=frame.fingerprint,
    action_index=pay_cost_index,  # position in the current action space
)
preview.state                       # "exact"
[s.label for s in preview.sources]  # ["Mountain", "Forest"]
preview.pool_after_spending         # the pool once {1}{G} is paid
```

Stepping that action next taps exactly those permanents in that order. The
preview is the first half of the payment itself (`Game::automatic_taps`, which
`produce_mana` calls and then taps), so it cannot propose a plan execution
does not use. The call is read-only: the state digest, RNG, revision, trace
and pending decision are the same before and after it.

`viewer`, `revision` and `fingerprint` bind the question to one decision. A
viewer who is not the chooser, a revision that has moved on or a fingerprint
from another offer set raises `managym.AgentError` and returns no plan.

| `state` | Meaning |
| --- | --- |
| `exact` | `sources` is the plan, in tapping order; empty when the cost is zero or the pool covers it. `pool_after_production` is always present. `pool_after_spending` is present only when every unit spend is forced. |
| `needs_choice` | X, an optional, alternative or additional cost, or an open target can still change the cost, or mana alone cannot pay a cost that taps or exiled cards may help pay. `explanation` names the choice; there are no sources. |
| `unavailable` | Automatic payment cannot fund the cost; only activated mana abilities could. |
| `not_applicable` | The action has no automatic payment to preview. This includes choosing which floating mana to spend and activating a mana ability (whether it has a color choice or its own fee). |

The preview reads the cost as the payment owes it (`Payment.owed`). While a
sacrifice, discard, exile or forage is still the payer's to choose, the state
is `needs_choice`; once those are paid and only mana is owed, the plan is
exact. The preview is a forecast only: what is committed to the cost is a fact
of the pending decision, `decision_context.paid`, one phrase per object in the
order committed ("tapped Gray Ogre to crew") with the object as it is now, and
it is there at every step of the payment, including one with no Pay offer. A
chosen sacrifice leaves as the payment completes, so until then its phrase is
"Gray Ogre, sacrificed when the payment completes". A waterbend paid as an
effect resolves lists its taps the same way. A cost with convoke, improvise or delve previews its
mana route like any other: automatic payment pays mana with mana and never
picks a creature, artifact or card. When mana alone cannot pay, the state is
`needs_choice` ("Mana alone cannot pay this cost. Choose what to tap or
exile.") until the payer's own taps and exiles leave a remainder mana covers.

A cast offer, or the target that completes it, previews the spell's cost; a
non-mana activated ability previews its fee; `PayCost` previews the pending
payment. A fixed source-tap offer previews tapping that source and then paying automatically,
with the chosen source first. Each source carries its object reference, name
and output; pools carry the six amounts and the restriction, spend-effect and
combat-expiry classes of the mana in them.

This is an inspection API for Academy and tools. It adds no action,
observation field or encoder column, and nothing in `Env.step`, `VectorEnv`,
search or training calls it.

### Previewing a whole combat declaration

The engine declares attackers and blockers one creature at a time. Ask how a
whole drafted declaration would be declared, without declaring any of it:

```python
from managym.combat import CombatPlan, PlannedBlock, preview_combat

frame = DecisionFrame.from_json(env.semantic_decision_frame_json())
ask = preview_combat(env, viewer=frame.actor, revision=frame.revision,
                     fingerprint=frame.fingerprint, plan=CombatPlan())
ask.declarable          # every creature the declaration will ask about, in order
plan = CombatPlan(attackers=[c.creature for c in ask.declarable[:2]])
preview = preview_combat(env, viewer=frame.actor, revision=frame.revision,
                         fingerprint=frame.fingerprint, plan=plan)
preview.refused         # None, or the first creature refused and why
[(s.action_index, s.revision) for s in preview.steps]
```

`Game::combat_preview` replays the engine's own per-creature answers to the
plan on a clone (`managym/src/flow/combat_preview.rs`). Each step is one
ordinary action at the decision it will be asked at, with that decision's
revision and fingerprint; stepping them in order declares exactly the plan.
Creatures with a single legal answer are answered by the engine without a
step, as in play. A block plan names blocker → attacker pairs
(`PlannedBlock`), and a blocker's `can_block` lists the attackers it may block
from here. The plan is refused, with a reason, when a planned creature is
never asked, a planned answer is not offered (a required block) or the
finished declaration differs from the plan (a lone block of a creature with
menace, which the engine removes once every blocker is declared).
`declared_attackers` and `declared_blocks` are the declaration as the engine
would finish it. During attacker selection both viewers can see the attackers
chosen so far. Raid becomes true for the attacking player when a nonempty
declaration finishes and lasts through the turn, even if those creatures leave.
Linden triggers for each white attacker her controller declares. With two white attackers
and an ability that doubles her triggers, each attacker accounts for two
occurrences. A creature that enters attacking can deal combat damage. Its
first declaration in a later combat still counts as its first attack for
Aurelia. Leaving and returning starts a new attack history for that creature.
Extra combats preserve raid and each creature’s history until the turn ends.
The same address rules as payment previews apply, and the
call is read-only in the same way.

At the combat damage step, the attacking player assigns damage among the
remaining blockers. Each choice names the attacker, the current blocker and
the amount. A trampler can assign damage to the defending player once every
blocker has been assigned lethal damage. For example, a 5-power trampler
blocked by an undamaged 2/2 can assign 2, 3, 4 or 5 to it. The defender takes
the rest. Without trample, that sole blocker takes all 5. The engine deals
the assignments together after the final choice.

When a combatant has first strike or double strike (CR 510.4), the first damage
step is exposed as `StepKind::FirstStrikeCombatDamage` (`COMBAT_FIRST_STRIKE_DAMAGE`),
followed by the regular `StepKind::CombatDamage` (`COMBAT_DAMAGE`) step. Each
permanent's `first_strike_participant` remembers whether it had first strike or
double strike when the first step began, including after its abilities change.
### Reading power during combat

Courageous Goblin and Ruby, Daring Tracker check for a creature with power
four or greater as attackers are declared. Once either ability triggers, removing that
creature leaves the bonus intact. Nessian Hornbeetle and Surrak, the Hunt
Caller also check their conditions as their beginning-of-combat abilities
resolve. Hornbeetle needs another creature with power four or greater.
Surrak needs total power eight or greater, counting negative power as well.

When Ashroot Animist or Halana and Alena, Partners has a trigger on the
stack, both viewers can inspect the source power in the stack label. The
amount follows changes to the source until resolution or departure. If the
source leaves, the trigger retains its last battlefield power even if the
card returns. Bulk Up similarly reads the target's power at resolution. A
creature at −2 power goes to −4, and flashback uses the same calculation.
Bite Down uses the power of the creature dealing damage at resolution.
If either target leaves and returns first, it deals no damage.
