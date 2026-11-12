# Building Limited decks

A pool is the cards a player has to build from. The Limited rules say which
of those cards a deck may use. A build is one deck being made from one pool,
one card at a time:

```python
from manabot import managym

pool = managym.CardPool({"Llanowar Elves": 6})  # The cards available.
limited = managym.format.Limited()               # The rules, for any pool.

build = limited.build(pool)
for _ in range(6):
    build.select("Llanowar Elves")
assert "Llanowar Elves" not in build.selectable  # All six copies are used.

for _ in range(33):
    build.select("Forest")                       # Basics are not in the pool.
assert not build.can_stop                        # Only 39 cards.

build.select("Forest")
assert build.can_stop                            # A legal 40-card main deck.
assert "Forest" in build.selectable              # Forty is a minimum.
assert build.deck == {"Llanowar Elves": 6, "Forest": 34}
assert limited.issues(pool, build.deck) == []
```

This illustrates construction legality, not a recommended deck. The same
native rules serve Python clients, learned builders and search-based builders.
The native authority is `managym/src/format/limited.rs`, built on world
eligibility and admission in `cards/`. Deck building happens before a game;
selecting a card is not an in-game Command.

## The Limited rules

A Limited main deck has at least 40 cards. Every non-basic must come from the
pool, at no more than its pool count. There is no four-copy limit. Basic lands
are unlimited by the pool, but must still be cards supported by the installed
world; `limited.basic_cards()` lists them. Pool and deck counts must be
positive integers; omit an absent card instead of supplying a zero count.
Unknown names, tokens and rules fixtures are not pool cards or deck entries.

`CardPool` holds only names and counts, and `Limited` holds only the rules;
neither changes while a deck is built. Constructing a `CardPool` with an
unsupported card or a malformed count raises `ValueError` naming the card.
`limited.issues(pool, deck)` returns the construction issues of a finished
main deck, an empty list when it is legal. It applies no size cap.

## Selecting cards and stopping

`limited.build(pool)` starts an empty deck. `build.select(card)` adds one
copy and commits it; there is no unselect. Cards may be selected in any order.
`build.selectable` lists the cards that can be selected next, and
`build.can_stop` is true exactly when the current deck passes the Limited
rules. Reaching 40 does not stop the build; the caller decides when to stop.
`build.deck` returns the deck so far as names and counts.

Selecting a card outside `selectable` raises `ValueError` and leaves the build
unchanged. Correct the caller rather than clipping counts or skipping cards
silently.

A build has a size cap, 60 by default: `limited.build(pool, max_cards=...)`,
never below 40. At the cap `selectable` is empty, and a legal deck can still
stop. The cap bounds a building episode; it is neither a Limited rule nor a
fact about the pool, so `issues` may accept a deck larger than a build's cap.

Each `selectable` read calls one stateless native function over the pool and
deck counts. Batched builders that score many decks at once compute their own
masks from `basic_cards()` and the pool counts, then check each finished deck
with `limited.issues`.

Native world admission still applies when starting a game with the resulting
deck. Construction legality does not establish checkpoint compatibility or
playing strength. Ordinary custom-game setup retains its existing construction
rules; selecting Limited is explicit.

## The TLA pool corpus

Restore the frozen corpus from the repository root:

```sh
uv run --extra artifacts python -m manabot.data fetch experiments/data/mtg282/pools-v2/corpus.json
```

Then load it and keep the pools the installed engine can play:

```python
from pathlib import Path

from manabot import managym
from manabot.pools import PoolCorpus

corpus = PoolCorpus.read(Path("experiments/data/mtg282/pools-v2/corpus.json"))
assert len(corpus.pools) == 1000

def playable(pool):
    return not managym.unsupported_cards([*pool.cards, *pool.human_build])

held_out = [p for p in corpus.pools if p.split == "held_out" and playable(p)]
training = [p for p in corpus.pools if p.split == "train" and playable(p)]
print(len(training), len(held_out))
```

```text
827 95
```

The corpus holds the first 1,000 trophy drafts in the retained 17Lands corpus
as 17Lands recorded them. It is plain source data: `manabot.pools` imports
nothing from the engine or from training, the file has no world field, and
loading does not check cards against the installed engine. Loading does not
download a fresh 17Lands dump or launch training.

Each pool has:

| Field | Meaning |
| --- | --- |
| `id` | The 17Lands draft id |
| `cards` | Main deck plus sideboard, basic lands removed, repeated names counted together |
| `human_build` | The human's main deck, including its basics |
| `split` | `"train"` or `"held_out"` |

The human build is from the draft's seventh win; provenance records which of
the player's builds that was. A 41-card human build stays 41 cards. Pool cards
not in the human build remain available outside its main deck. Two drafts
with identical lists remain two pools.

Whether a pool is playable belongs to whoever plays it. Today 78 pools use one
or more of 41 TLE bonus-sheet cards the engine does not define, so 922 pools
are playable: 827 training and 95 held out. Defining more cards makes more
pools playable and moves no pool between splits. **State the playable pool
count with any result measured on the corpus**; held-out results before and
after a card-support change cover different pools. A run's own population is
frozen in its regime, which embeds the deck lists it used.

The split ranks all 1,000 drafts by SHA256 of `mtg-282-held-out-v1:<id>`,
with the id as the tie-breaker, and holds out the first 100. Consumers load
that saved split; they do not make a fresh random split on each run. Training
draws from the training pools. To evaluate a builder, play its deck against
the human build of the **same held-out pool**, using a fixed reference pilot.
Evaluate pilot changes separately on fixed trophy lists.

To use pools in self-play, give each deck its pool and build:

```python
from manabot.training.decks import UniformDecks

population = UniformDecks(
    decks=tuple(managym.Deck(p.human_build) for p in training),
    pools=tuple(p.id for p in training),
    builds=(None,) * len(training),  # None: the human build.
)
```

`pools` and `builds` align with `decks`; each (pool, build) pair is unique.
`UniformDecks.entry_ids` keeps its existing meaning in saved regimes.

TLA Limited games outgrow the default observation capacities; a run on
these decks stops with a typed capacity error within its first few dozen
games. Copy the `observation` block and the `game_errors` policy (with
`recover_capacity: true`) of
`experiments/regimes/tla-frontier-c5fba4-w64.json`. Recovered games are
`resource_limit` attempts in the private ledger and produce no result row.

### Deriving and publishing

To derive from retained source text instead of restoring the artifact:

```sh
uv run --extra artifacts python -m manabot.data fetch experiments/data/mtg213/corpus-v1/corpus.json
uv run python -m experiments.tla_pools experiments/data/mtg213/corpus-v1/corpus.json experiments/data/mtg282/pools-v2/corpus.json
```

Derivation reparses the unchanged source decklists and saves all 1,000 pools,
their split and source provenance: digests, attribution and license. A
malformed list fails derivation with its draft, line and reason. Unknown card
names are kept as recorded. `--held-out` defaults to 100; changing it defines
a different split. An existing output file is never overwritten: choose a new
versioned path. The historical v10 pilot's 134-deck population and its
evidence keep their original identities.

Publish a derived corpus to the experiment artifact store with:

```sh
uv run --extra artifacts python -m manabot.data publish experiments/data/mtg282/pools-v2/corpus.json
```

Commit `experiments/data.manifest.json`, keeping the corpus bytes in the
artifact store rather than tracking them in Git. Fetching uses that manifest
and checks the restored bytes' digest; it requires artifact-store access or an
existing local cache. If a local file differs, preserve it separately and
restore the frozen file; do not edit the manifest to bless different bytes.
`PoolCorpus.read` checks that every pool has a valid split matching the hash
ranking and that its human build uses only its pool's cards plus basics.

## Join games back to builds

Self-play retains one public result row per completed native game, including
runs that do not configure game-error recovery. Each row names both seats'
pool and build, and the winning seat or a draw:

```json
{"schema_version": 1, "stream": 0, "game": 41,
 "seats": [{"pool": "bc6d…", "build": null},
           {"pool": "bc6d…", "build": "ep-17"}],
 "winner": 1}
```

Here a builder's deck from pool `bc6d…` beat that pool's human build. Seats
use native ordering, so seat 0 was on the play. `build: null` is the human
build; `pool: null` is a deck with no pool, such as a constructed list.
`winner: null` is a completed draw. Both seats belong to the same row, even
when both contribute learning samples. Joining on pools recovers which cards
each deck was built from; joining on deck names alone loses that distinction.

Read a completed or failed stage's results with `manabot.game_results`, which
imports nothing from training:

```python
from pathlib import Path

from manabot.game_results import read_game_results

rows = read_game_results(Path(stage_record.artifacts["game_results"]["path"]))
```

The stage record scopes the `stream`/`game` ordinals. Resume attempts keep
their own records; do not concatenate attempts as independent games. A failed
or interrupted stage can retain more rows than its `games` counter, which
counts games in completed updates.

Failed, interrupted and unfinished games do not become draws or losses in
this result view. Their diagnostics belong in the existing
[game-failure records](training.md#game-failures), which may contain private
replay inputs; result rows do not. Aggregate deck counters are useful
summaries but cannot recover pool or build joins. Missing results in an older
run cannot be reconstructed from those aggregates.

While a stage runs, it fsyncs each completed result to a `.partial` journal.
Completion or failure seals the rows under the metric run's
`artifacts/game-results/<sha256>.jsonl`. Completed games survive a later
optimizer failure.

## Several models in one regime

A regime can name separate models and declare, for each stage, the models it
trains and the models it reads frozen. For a builder stage with a frozen
pilot, the roles are:

| Model name | Stage use | Weights |
| --- | --- | --- |
| `builder` | Train from builds and game outcomes | Updated by this stage |
| `pilot` | Play the games that supply outcomes | Frozen for this stage |

A joint stage trains both from the same games: `trains=("builder", "pilot")`.
Names identify models, not smaller networks or shared parameters. The builder
has its own weights; initializing its card encoder from a pilot checkpoint
copies those weights without establishing live sharing. A deck-score model
can be named separately when a stage uses one. Declaring models does not by
itself construct a builder learner or start games.

Each stage's references must resolve to declared models, and a model is
either trained or frozen in a stage, not both. Frozen readers do not receive
optimizer updates from that stage. Artifacts passed between stages have
declared input and output types; a missing producer, forward reference or
incompatible artifact is an error to correct in the recipe before execution.
A new stage owns its execution in its own module. Existing self-play and
belief operations retain their entry points and behavior.

Saved single-agent regimes keep their serialized identity and interpretation.
Loading an old `agent` configuration does not silently rewrite it into a
multi-model recipe or change the model an existing stage trains. Use a new
regime when declaring additional model roles, and preserve the original
regime with its run evidence. See [training](training.md#regimes-and-stages)
for execution, checkpoint admission and recovery contracts.

Named-model recipes explicitly select schema 3. For example, this recipe
inspects two independent models without playing games or changing weights:

```python
from manabot import managym
from manabot.infra.config import MatchConfig, ModelConfig
from manabot.training.regime import ModuleStage, TrainingRegime

regime = TrainingRegime(
    schema_version=3,
    id="builder-foundations",
    world=managym.WORLD_VERSION,
    match=MatchConfig(),
    models={
        "builder": ModelConfig(world=managym.WORLD_SETUP),
        "pilot": ModelConfig(world=managym.WORLD_SETUP),
    },
    stages=[ModuleStage(
        id="inspect",
        module="manabot.training.stages.model_inventory",
        trains=("builder",),
        frozen=("pilot",),
    )],
)
```

Run this through the ordinary `execute_regime` entry point. Named checkpoints
are returned in `run.selected_models`; schema 3 does not select one arbitrary
model as the run's `selected_artifact`.

A module exports `STAGE`, a `StageDefinition` from
`manabot.training.stages.contract`: named Pydantic input types, one Pydantic
output type and an `execute(context, inputs)` callable. `ModuleStage.inputs`
maps each input name to an earlier module stage ID. `StageDefinition.run` parses
the typed inputs and validates and writes the result. The executor verifies
input bytes, records the stage's `result` artifact and exports each trained
model through ordinary checkpoint admission. `StageContext` provides the live
trainable models by name, detached frozen readers, an output directory and a
resource check to call during long operations. Modules are trusted Python
code, like experiment files.

Schema 3 currently supports module stages and existing self-play stages. A
self-play stage trains exactly one named model and may initialize it from
`initial_weights`; producer-policy imports, learning-state imports and belief
stages retain their single-agent entry points. Named-model process recovery is
not implemented. All named models share the regime's observation/history
contract.
