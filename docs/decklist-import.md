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

Checkpoint players on imported setups pass the same execution admission as
every other match (`manabot.model.world.validate_agent_setup`). A checkpoint
bound to the importing content pool plays any decks that pool admits; a
checkpoint bound to authored setups, such as a curated matchup or the Elemental
roster, refuses imported decks even when tensor dimensions agree. Simulation
and Academy apply this check before the first game; `random` and `default`
need no checkpoint, and no training is required to use the importer.

## Training on decklists

A training run supplies decks and samples matches from them. In an ordinary
`TrainingRegime`, set `decks` to the content pool and the decklist files, and
`agent.semantic_pack` to the same pool key, and omit `match`: the decks are
the regime's only record of its setups, and a regime that also authors a
`match` is rejected. Producer policies are admitted against the pool, by
checking every deck, rather than against one pair. Existing self-play stage
settings remain in use. Deck stages require `behavior: current-self`, one
producer, ordinary roots and the default fixed curriculum field. For example,
adapting an existing recipe in Python (this only constructs a configuration):

```python
from manabot.training.regime import TrainingRegime

values = recipe.model_dump(mode="json")
values.pop("match")
values["decks"] = {"pool": "tla-pool-v1", "files": ["white.txt", "blue.txt", "red.txt"]}
values["agent"]["semantic_pack"] = "tla-pool-v1"
regime = TrainingRegime.model_validate(values)
```

Loading checks the whole list before training: any invalid or unsupported
source fails with its filename and line diagnostics, and nothing is silently
filtered. The regime embeds the admitted decks and their UTF-8 source records,
so recovery never rereads a mutable path. Lists with equal normalized contents
are one deck, retaining every source record without increasing its sampling
weight. At least two distinct lists are required. Construction remains
`development-v1`, not tournament legality.

`uniform-distinct-v1` chooses two different decks at each native game root,
uniformly over ordered pairs. Either deck is equally likely to be on the play.
Draws use the native reset seed in a separate RNG domain from shuffling and
policy inference; recovery reproduces deck pair, seats and game seed, and
rejects a run whose decks, sources or recorded state differ. Storage is linear
in the number of decks, with no all-pairs table. All games use the whole pool's
fixed semantic vocabulary.

Decks are submitted to the world, not registered with it. The native vector
may replace its training decks between roots; each replacement, and every game
creation, passes the pool's native admission first, so an unsupported deck is
rejected before any game starts and running games keep their decks.

Policy exports from such a run bind the pool's world — its native content
descriptor, rules versions, observation and learning schema — and nothing
about the training decks, sources or sampling. The model can play any deck the
same pool admits, including decks it never trained on; unsupported cards and a
changed pool, rules or input contract are still rejected. The run, not the
checkpoint, records the decks it used. Exact-setup exports for authored matches
and the Elemental roster keep their established binding.

Existing exposure counters report game starts, environment decisions, learner
transitions and terminal games separately. Uniform starts do not imply uniform
transition exposure. Failure receipts retain the actual setups each game was
created with, the run's deck digest and reset seed. Existing Elemental recipes
keep their original roster admission, mirror distribution, seed sequence and
serialization.

The bounded local proof uses supported development decks and performs no
weight updates. Real 17Lands ingestion awaits full TLA support; this does not
provide a 17Lands adapter, held-out dataset policy, adaptive curriculum, deck
generation or mirrors in deck sampling.
