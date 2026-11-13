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
assert "Llanowar Elves" not in build.available  # All six copies are used.

for _ in range(33):
    build.select("Forest")                      # Basics are not in the pool.
assert not build.can_stop                       # Only 39 cards.

build.select("Forest")
assert build.can_stop                           # A legal 40-card main deck.
assert build.available["Forest"] == 20          # Forty is a minimum; 60 is the cap.
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
`build.available` maps each card that can still be selected to the copies
that can still be added, and `build.can_stop` is true exactly when the current
deck passes the Limited rules. Reaching 40 does not stop the build; the
caller decides when to stop. `build.deck` returns the deck so far as names
and counts.

Selecting a card absent from `available` raises `ValueError` and leaves the
build unchanged. Correct the caller rather than clipping counts or skipping
cards silently.

A build has a size cap, 60 by default: `limited.build(pool, max_cards=...)`,
never below 40. At the cap `available` is empty, and a legal deck can still
stop. The cap bounds a building episode; it is neither a Limited rule nor a
fact about the pool, so `issues` may accept a deck larger than a build's cap.

## Rules as arrays

Learned and search-based builders work on many decks at once. The same rules
are available as NumPy arrays over the engine's card axis: index `i` is
`card_def_id` `i`, the `i`-th entry of `managym.card_definitions_json()`, with
`managym.CARD_DEFINITION_COUNT` entries. This is the index the policy's card
embedding already uses.

```python
import json

import numpy as np
from manabot import managym

limited = managym.format.Limited()
names = [d["name"] for d in json.loads(managym.card_definitions_json())]
elves, forest = names.index("Llanowar Elves"), names.index("Forest")

pools = np.zeros((2, managym.CARD_DEFINITION_COUNT), dtype=np.int32)
pools[:, elves] = 6
decks = np.zeros_like(pools)
decks[0, forest] = 34                       # Deck 0: 34 Forests.
decks[1, elves], decks[1, forest] = 6, 34   # Deck 1: the deck built above.

available, can_stop = limited.available(pools, decks)
print(available[:, elves], available[:, forest], can_stop)
print(limited.legal(pools, decks), limited.min_cards)
```

```text
[6 0] [26 20] [False  True]
[False  True] 40
```

| Call or field | Shape | Meaning |
| --- | --- | --- |
| `limited.min_cards` | int | The 40-card minimum |
| `limited.basics` | bool `[cards]` | Cards the pool does not limit |
| `pool.counts` | int `[cards]` | A `CardPool`'s counts; `managym.CardPool.from_counts(counts)` makes one |
| `limited.available(pools, decks, max_cards=60)` | int `[n, cards]`, bool `[n]` | Copies that can still be added, and whether each deck can stop |
| `limited.legal(pools, decks)` | bool `[n]` | Whether each deck passes the Limited rules, with no cap |

`pools` and `decks` are integer arrays of shape `[n, cards]`; a single
`[cards]` row is not broadcast. For a non-basic, `available` is its pool
count minus its deck count; for a basic, it is the slots left under the cap.
A deck at the cap has no copies available. The legal-pick mask is
`available > 0`, so a builder masks its logits with that and reads `can_stop`
for its stop action. `legal` matches `limited.issues(...) == []` and
`can_stop` matches `legal`, row by row.

`available` takes decks a build could have reached: each non-basic within its
pool count, only pool cards and basics, and no more than `max_cards` cards.
Any other row, a negative count, a mismatched shape or a card axis of the
wrong length raises `ValueError` naming the row. `legal` accepts any
non-negative counts and reports an illegal deck as `False`. A pool row, and
`CardPool.from_counts`, may not hold tokens, rules fixtures or other cards
the world does not support; both forms raise `ValueError`, as the name form
does.

`limited.basics` and `min_cards` are plain data, so a builder on a GPU can
compute `where(basics, slots_left, pools - decks)`, zeroed at the cap, without
a round trip; the native forms are the authority it is tested against. The
name form is the same native function: `build.available` and
`build.can_stop` read one row of `limited.available`. managym returns NumPy
arrays and never imports torch.

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
900 100
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

Whether a pool is playable belongs to whoever plays it. The engine now defines
every card used by these pools, so all 1,000 are playable: 900 training and
100 held out. Defining more cards makes more pools
playable and moves no pool between splits. **State the playable pool
count with any result measured on the corpus**; held-out results before and
after a card-support change cover different pools. A run's own population is
frozen in its regime, which embeds the deck lists it used.

The split ranks all 1,000 drafts by SHA256 of `mtg-282-held-out-v1:<id>`,
with the id as the tie-breaker, and holds out the first 100. Consumers load
that saved split; they do not make a fresh random split on each run. Training
draws from the training pools. To evaluate a builder, play its deck against
the human build of the **same held-out pool**, using a fixed reference pilot.
Evaluate pilot changes separately on fixed trophy lists.

To play human builds in self-play, name each deck by its pool:

```python
from manabot.training.decks import UniformDecks

population = UniformDecks(
    decks=tuple(managym.Deck(p.human_build) for p in training),
    entry_ids=tuple(p.id for p in training),
)
```

TLA Limited games outgrow the default observation capacities; a run on
these decks stops with a typed capacity error within its first few dozen
games. Copy the `observation` block and the `game_errors` policy (with
`recover_capacity: true`) of
`experiments/regimes/tla-frontier-c5fba4-w64.json`.

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
