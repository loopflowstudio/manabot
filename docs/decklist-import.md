# Import decklist files

Academy and manabot resolve files through the same native content pool and
construction admission. The initial `development-v1` ruleset is for development
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
comes from the selected versioned native pool, including its program/face/token
closure; registration alone does not make a card eligible.

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
  --content-pool tla-pool-v1 \
  --set sim.num_games=1 --set sim.num_threads=1 --set sim.max_steps=4
```

The CLI prints the admitted setup identity before simulation. Programmatic
configuration uses `manabot.decklist.import_files(...).match_config()`; the
serialized match retains source, resolved definitions, construction ruleset,
and exact world/content/compiled identities. Reordering lines does not change
the setup identity. A changed pool rejects a retained import instead of silently
reinterpreting it. Existing native Commands, reconnect, replay and forks retain
their ordinary authority checks.

Imported setups currently reject checkpoint opponents. A checkpoint trained on
a curated matchup is not admitted merely because tensor dimensions agree.
The simulation entry point supports `random` and `default`; no training is
required to use the importer.

## Reproducible training collections

Prepare a versioned local manifest from complete supported lists:

```sh
uv run python -m manabot.deck_collection --pool tla-pool-v1 --out decks-v1.json white.txt blue.txt red.txt
```

The command fails the entire preparation if any source is invalid or unsupported,
with its filename and line diagnostics. It does not overwrite an existing
manifest. It preserves UTF-8 source text, filenames, main decks and sideboards.
Lists with equal normalized contents share one deck ID, retaining all their
source records without increasing their sampling weight. At least two distinct
lists are required. Construction remains `development-v1`, not tournament legality.

In an ordinary `TrainingRegime`, set `deck_collection` to the manifest path and
`agent.semantic_pack` to its pool key. The representative `match` may be omitted;
it is derived from the admitted collection. Existing self-play stage settings
remain in use. Collection stages require `behavior: current-self`, one producer,
ordinary roots and the default fixed curriculum field. For example, adapting an
existing recipe in Python (this only constructs a configuration):

```python
from manabot.training.regime import TrainingRegime

values = recipe.model_dump(mode="json")
values.pop("match")
values["deck_collection"] = "decks-v1.json"
values["agent"]["semantic_pack"] = "tla-pool-v1"
regime = TrainingRegime.model_validate(values)
```

The recipe embeds the admitted manifest, so recovery does not reread a mutable
path. Membership, retained sources, content/world identity and learning-schema
hash are bound by the collection digest. Editing any of these requires preparing
a new collection; recovery under different membership or schema is rejected.

`uniform-distinct-v1` chooses two different deck IDs at each native game root,
uniformly over ordered pairs. Either deck is equally likely to be on the play.
Draws use the native reset seed in a separate RNG domain from shuffling and
policy inference; recovery reproduces deck pair, seats and game seed. Storage is
linear in the number of decks, with no all-pairs table. All games use the whole
pool's fixed semantic vocabulary. Policy exports retain the full collection
binding and admit its member decks; they contain private submitted sources.

Existing exposure counters report game starts, environment decisions, learner
transitions and terminal games separately. Uniform starts do not imply uniform
transition exposure. Failure receipts retain actual sampled deck IDs, complete
setups, collection digest and reset seed. Existing Elemental recipes keep their
original roster admission, mirror distribution, seed sequence and serialization.

The bounded local proof uses three supported development decks and performs no
weight updates. Real 17Lands ingestion awaits full TLA support; this does not
provide a 17Lands adapter, held-out dataset policy, adaptive curriculum or mirrors
in collection sampling.
