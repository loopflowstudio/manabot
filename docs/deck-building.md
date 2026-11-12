# Building Limited decks

Use managym to check a pool and a partial build before offering the next pick.
For example, a pool with six Llanowar Elves permits all six copies; Forests
come from the basic-land supply:

```python
from manabot import managym

rules = managym.LimitedRules(pool={"Llanowar Elves": 6})
choices = rules.choices(picks={"Llanowar Elves": 6, "Forest": 33})
assert not choices.can_stop                 # Only 39 cards.
assert "Llanowar Elves" not in choices.cards  # All six copies are used.
assert "Forest" in choices.cards

choices = rules.choices(picks={"Llanowar Elves": 6, "Forest": 34})
assert choices.can_stop                     # A legal 40-card main deck.
assert "Forest" in choices.cards            # Forty is a minimum.
```

This illustrates construction legality, not a recommended deck. The same
native rules serve Python clients, learned builders and search-based builders.
The native authority is `managym/src/format/limited.rs`, built on world
eligibility and admission in `cards/`.
Deck building happens before a game; its picks are not in-game Commands.

## Pools and reference builds

Restore the frozen corpus from the repository root:

```sh
uv run --extra artifacts python -m manabot.data fetch experiments/data/mtg282/pools-v1/corpus.json
```

Then load its saved training and evaluation populations:

```python
from pathlib import Path

from manabot.training.pools import PoolCorpus

corpus = PoolCorpus.read(Path("experiments/data/mtg282/pools-v1/corpus.json"))
assert len(corpus.entries) == 922
training = corpus.population("train")
evaluation = corpus.population("held_out")
assert len(training.decks) == 827
assert len(evaluation.decks) == 95
```

Both populations contain human reference decks with their population entry
IDs. Loading does not download a fresh 17Lands dump or launch training.

The TLA pool corpus pairs each trophy event with its human reference build.
The pool is the selected build's main deck plus sideboard, with basic lands
removed and repeated names counted together. The reference is the human's
main deck, including its basics. Preserve a 41-card reference as 41 cards.
Unpicked pool cards remain available outside the main deck.

Each entry retains the source event identity, population entry identity,
counted pool, reference build and split membership. Two events with identical
lists remain two entries. A deck's content identity answers whether lists
match; an event identity answers which human build and pool to compare with.

The source is the retained 17Lands trophy corpus produced by
[`experiments/tla_corpus.py`](../experiments/tla_corpus.py): the first 1,000
events with seven observed wins, using each event's seventh-win build. The
pool corpus includes the 922 events admitted by the derivation world. Its
provenance records the other 78 event IDs and their unknown card names: 41
distinct names from the TLE bonus sheet. An excluded event contributes neither
a pool nor a reference build; no cards are removed to make it eligible and no
replacement event is selected. Source digests, attribution and license remain
with the derived corpus. The historical v10 pilot's 134-entry population and
its evidence keep their original identities.

Freeze the held-out event IDs with the pool corpus before training. Consumers
load that saved partition; they do not make a fresh random split on each run.
Training draws from the training partition. To evaluate a builder, compare its
deck with the human reference from the **same held-out pool**, using a fixed
reference pilot. Evaluate pilot changes separately on fixed trophy lists.

The split ranks **all 1,000 source events** by SHA256 of
`mtg-282-held-out-v1:<event_id>`, with event ID as the tie-breaker, and reserves
the first 100 for evaluation. The saved held-out partition contains only the
admitted events among those 100; exclusions are not backfilled. The remaining
admitted events are training entries: this corpus has 95 held-out and 827
training events. Provenance saves the source count and reserved count alongside
the exclusions; loading checks that the entries and exclusions cover the source
and that the partition matches its hash ranking. Adding support for the 78 excluded
events therefore moves no existing assignment. Keep the source population,
hash rule and held-out count unchanged when deriving that expanded corpus.

To derive from retained source text instead of restoring the artifact:

```sh
uv run --extra artifacts python -m manabot.data fetch experiments/data/mtg213/corpus-v1/corpus.json
uv run python -m experiments.tla_pools experiments/data/mtg213/corpus-v1/corpus.json experiments/data/mtg282/pools-v1/corpus.json
```

Derivation reparses the unchanged source decklists before current-world
admission and saves the admitted entries, exclusions and explicit partition.
`--held-out` defaults to 100 and sets the number reserved from the full source
population, not a target count after admission. Changing it defines a different
split. An existing output file is never overwritten: choose a new versioned
path. Later TLE support requires a new corpus version; retain this version and
its exclusions with the runs that used it.

Publish a derived corpus to the experiment artifact store with:

```sh
uv run --extra artifacts python -m manabot.data publish experiments/data/mtg282/pools-v1/corpus.json
```

Commit `experiments/data.manifest.json`, keeping the corpus bytes in the
artifact store rather than tracking them in Git. Fetching uses that manifest
and checks the restored bytes' digest; it requires artifact-store access or an
existing local cache. If a local file differs, preserve it separately and
restore the frozen file; do not edit the manifest to bless different bytes.
`PoolCorpus.read` validates reference legality and the saved partition, and
rejects a corpus whose world identity differs from the installed runtime.
Use the matching runtime to consume frozen evidence, or derive a new corpus
version for the new world.

## Legal picks and stopping

The Limited ruleset requires at least 40 main-deck cards. Every non-basic must
come from the pool, at no more than its pool count. There is no four-copy limit.
Basic lands are unlimited by the pool, but must still be cards supported by
the installed world. Counts must be positive integers; omit an absent card
instead of supplying a zero count. Unknown names, tokens and rules fixtures
are not deck entries.

`LimitedRules.choices(picks=...)` returns the cards that can be added once and
`can_stop`. A partial build below 40 cards is valid input with `can_stop=False`.
An overdrawn pool, unsupported card or malformed count is an error, not a
smaller choice set. Correct the input rather than clipping counts or removing
cards silently. The Python interface delegates both legality and the choice
set to managym.

Picks commit one copy at a time, in any order. Stop is legal exactly when the
current list passes the Limited ruleset; reaching 40 does not automatically
stop. The picking interface has a separate 60-card default size cap, configurable
with `LimitedRules(pool=..., max_cards=...)` and never below 40. At the cap it
offers no further card picks; a valid list can still stop. The cap bounds a
building episode, not the definition of Limited's minimum deck size.

Native world admission still applies when starting the resulting game.
`rules.check(picks)` returns full-deck construction issues without applying
the episode cap; `LimitedRules.basic_cards()` returns the world's eligible
basic lands. Both use printed definitions, not a separate name allowlist.
Construction legality does not establish checkpoint compatibility or playing
strength. Ordinary custom-game setup retains its existing construction rules;
selecting Limited is explicit.

## Join games back to builds

Self-play retains one public result row per completed native game, including
runs that do not configure game-error recovery. Each row identifies both
population entries, which seat was on the play, and the winning seat or a
draw. Each seat can also carry a build-episode link; human reference lists have
no build episode. Both seats belong to the same game row, even when both
contribute learning samples.

For example, if event A's reference plays event B's reference, with B on the
play and A winning, the row preserves those two event entries, B's play
assignment and A's winning seat. Joining on the entries recovers the two
pools; joining on deck names alone loses that distinction. When a generated
deck replaces A's reference, A's seat also names its build episode so the
outcome can be attributed to that sequence of picks.

Failed, interrupted and unfinished games do not become draws or losses in
this result view. Their diagnostics belong in the existing
[game-failure records](training.md#game-failures), which may contain private
replay inputs. Public result rows contain outcome and build attribution, not
those replay inputs. Aggregate deck counters are useful summaries but cannot
recover event or episode joins. Missing results in an older run cannot be
reconstructed from those aggregates.

Read a completed or failed stage's retained results through
`read_game_results(Path(stage_record.artifacts["game_results"]["path"]))`
from `manabot.training.game_results`. Each row has `stream`, `game`, two
`seats` (`entry_id`, optional `build_episode`), `on_play` and `winner`.
Seats use native ordering, so `on_play` is zero; `winner=None` is a completed
draw. The stage record scopes the stream/game ordinals. Resume attempts keep
their own records; do not concatenate attempts as independent games.

While a stage runs, it fsyncs each completed result to a `.partial` journal.
Completion or failure seals the rows under the metric run's
`artifacts/game-results/<sha256>.jsonl`. Completed games survive a later
optimizer failure. `UniformDecks.build_episodes`, when supplied, aligns with
its unique `entry_ids`; an omitted episode means a human list.

## Several models in one regime

A regime can name separate models and declare, for each stage, the model it
trains and the models it reads frozen. For a builder stage, the roles are:

| Model name | Stage use | Weights |
| --- | --- | --- |
| `builder` | Train from build episodes and game outcomes | Updated by this stage |
| `pilot` | Play the games that supply outcomes | Frozen for this stage |

Names identify models, not smaller networks or shared parameters. The builder
has its own weights; initializing its card encoder from a pilot checkpoint
copies those weights without establishing live sharing. A deck-score model
can be named separately when a stage uses one. Declaring models does not by
itself construct a builder learner or start games.

Each stage's references must resolve to declared models. Frozen readers do not
receive optimizer updates from that stage. Artifacts passed between stages
have declared input and output types; a missing producer, forward reference or
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
        trains="builder",
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
input bytes, records the stage's `result` artifact and exports its trained model
through ordinary checkpoint admission. `StageContext`
provides the one live trainable model, detached frozen readers, an output
directory and a resource check to call during long operations. Modules are
trusted Python code, like experiment files.

Schema 3 currently supports module stages and existing self-play stages with
an explicit `trains` name. Self-play may initialize from `initial_weights`;
producer-policy imports, learning-state imports and belief stages retain their
single-agent entry points. Named-model process recovery is not implemented.
All named models share the regime's observation/history contract.
