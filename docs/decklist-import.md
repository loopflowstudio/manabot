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
