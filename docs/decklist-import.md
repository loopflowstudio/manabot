# Import decklist files

Decks and decklists are managym concepts. `managym.Deck` is a deck's main and
sideboard counts; `managym.Decklist` is one submitted file (exact source,
filename, entries and line issues). Parsing, card-name resolution and
construction admission all run natively against the installed world, so
Academy and manabot share one importer. A deck needs only a world: its cards are
supported when the world defines them. Set restrictions such as TLA-only are
filters for choosing or generating decks (`managym.set_cards`), not admission. The initial `development-v1` ruleset is for development
play. It does not assert Standard or any other tournament legality.

Files are UTF-8 text, at most 64 KiB. Each entry is a positive integer, whitespace,
and a card name. Blank lines and full-line `#` comments are ignored. `Main`,
`Deck`, and `Sideboard` headings are case-insensitive and may end with a colon.
Entries before a heading belong to the main deck. Repeated entries are added
within their section. Names match case-insensitively with Unicode NFC and
whitespace normalization; punctuation is significant. Double-faced cards accept
the full “front // reverse” name or the front name, never a reverse face alone. Set codes, collector
numbers, inline comments and `4x` quantities are not part of this dialect.

```text
# My development deck
Main
10 Forest
10 Mountain
2 Avatar Enthusiasts
Sideboard
1 Island
```

The source, filename and line references survive validation errors. Unknown or
ambiguous names, unsupported cards, bad counts and construction issues are
reported without substituting cards. Card resolution is offline. Eligibility
comes from the installed world: a defined card is supported unless its set file
marks it incomplete. Authored setups (curated matchups, the Elemental roster)
are fixed setups, never import targets.

Development construction requires a nonempty main deck, positive quantities,
and eligible non-token cards in either section. There is no tournament deck-size,
copy-count or sideboard-size restriction. A separate operational limit permits
at most 1,000 cards per player across both sections. The sideboard supplies
outside-game copies to rules that use them; this flow does not offer sideboarding
between games. Engine observation capacity errors remain explicit.

In Academy, open **Import decklist files**, enable imported decks, and select
both files. The source boxes remain editable. **Check decklists** reports issues
by deck and line; **New Game** starts the admitted pair with a built-in opponent.
Uncheck imported decks to return to authored presets. The source is retained in
the private replay setup, and is excluded from viewer replay responses.

For a bounded ordinary manabot startup proof using the example files
(four decisions, then the configured step limit ends the run):

```sh
uv run manabot sim \
  --hero-deck-file docs/decklists/forest-mountain.txt \
  --villain-deck-file docs/decklists/islands.txt \
  --set sim.num_games=1 --set sim.num_threads=1 --set sim.max_steps=4
```

The CLI prints the construction ruleset and both admitted deck identities
before simulation. Programmatic
configuration is:

```python
import managym
from manabot.infra.config import MatchConfig

hero, villain = managym.admit_decklists(
    [
        ("hero", managym.Decklist.read("hero.txt")),
        ("villain", managym.Decklist.read("villain.txt")),
    ]
)
match = MatchConfig.from_import(hero, villain)
```

Failures raise `managym.DeckImportError`, whose `decklists` maps each failing
seat to its decklist with line issues. The serialized match holds each seat's
deck once and retains that seat's admitted decklist privately as provenance
(`hero_source`/`villain_source`). Loading the match re-imports each retained
source against the installed world, which must admit it to exactly that seat's
deck; a world that no longer admits the list rejects the record, as it would any
other deck. Reordering lines or comments does not change a deck's identity. Existing native Commands, reconnect, replay and forks retain
their ordinary authority checks.

Checkpoint players on imported setups pass the same execution admission as
every other match (`manabot.model.world.validate_agent_setup`). A checkpoint
bound to the world plays any decks the world admits; a checkpoint bound to authored setups, such as a curated matchup or the Elemental
roster, refuses imported decks even when tensor dimensions agree. Simulation
and Academy apply this check before the first game; `random` and `default`
need no checkpoint, and no training is required to use the importer.

## Training on decklists

A training run has a deck source: a way to get the decks for each new game,
not a list of files. Experiments are Python, so the experiment builds the
source however it likes and places it in the regime as `deck_source`, with
`agent.semantic_pack` set to `"world"` and no `match`: the source is the
regime's only record of its setups, and a regime that also authors a `match` is
rejected. Producer policies are admitted against the world, by checking every
deck the source lists, rather than against one pair. Deck stages require
`behavior: current-self`, one producer, ordinary roots and the default fixed
curriculum field. For example (this only constructs a configuration):

```python
import managym
from manabot.training.decks import UniformDecks
from manabot.training.regime import TrainingRegime

imports = managym.admit_decklists(
    [(path, managym.Decklist.read(path)) for path in ("white.txt", "blue.txt")]
)
values = recipe.model_dump(mode="json")
values.pop("match")
values["deck_source"] = UniformDecks(decks=tuple(i.deck for i in imports))
values["agent"]["semantic_pack"] = "world"
regime = TrainingRegime.model_validate(values)
```

`managym.check_decks(decks)` is the cheap upfront check: each deck's issues in
the installed world, in order, with no game allocated. Validating a regime runs
it over the source's decks, so an unsupported deck fails before training. Game
creation still admits every deck natively. Construction remains
`development-v1`, not tournament legality.

`UniformDecks` (`uniform-distinct-v1`) is the one source so far: at each native
game root it chooses two different decks from its distinct list, uniformly over
ordered pairs, so either deck is equally likely to be on the play. Draws use the
native reset seed in a separate RNG domain from shuffling and policy inference;
exact resume reproduces deck pair, seats and game seed, and rejects a run whose
source or recorded state differs. Storage is linear in the number of decks,
with no all-pairs table. All games use the world's semantic vocabulary.

Decks are submitted to the world, not registered with it. The native vector
may replace its training decks between roots; each replacement, and every game
creation, passes the world's native admission first, so an unsupported deck is
rejected before any game starts and running games keep their decks.

Policy exports from such a run bind the world — its version and card digest
(`world_digest`), rules versions, observation and learning schema — and nothing
about the deck source. The model can play any deck the world admits, including
decks it never trained on; unsupported cards and a changed world, rules or
input contract are still rejected. The run, not the checkpoint, records the
decks it used. Exact-setup exports for authored matches and the Elemental
roster keep their established binding.

Existing exposure counters report game starts, environment decisions, learner
transitions and terminal games separately. Uniform starts do not imply uniform
transition exposure. Failure receipts retain the actual setups each game was
created with, the run's deck-source digest and reset seed. Existing Elemental recipes
keep their original roster admission, mirror distribution, seed sequence and
serialization.

The bounded local proof uses supported development decks and performs no
weight updates. Real 17Lands ingestion awaits full TLA support; this does not
provide a 17Lands adapter, held-out dataset policy, adaptive curriculum, deck
generation or mirrors in deck sampling.
