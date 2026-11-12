# MTG-282 choices

Inherited MTG-271 notes are unrelated background, recoverable at
`1659b0e290f55daa632ec8f869adff4a3059d79f:scratch/questions.md`.

Sync resolution (2026-11-11): keep this branch's MTG-282 decisions here.
Main's unrelated MTG-285 release notes remain recoverable at
`668555c095c61709191ee7c4c59ad767b6903317:scratch/questions.md`.

The documentation step chose the reversible public API
`managym.LimitedRules(pool=..., max_cards=60).choices(picks=...)`, returning
`cards` and `can_stop`. Keep the cap separate from Limited's 40-card minimum.

Headless implementation assumption (2026-11-11): freeze 100 of the 1,000
source events for evaluation, selected by ascending SHA256 of
`mtg-282-held-out-v1:<event_id>`, ties by event ID. Save explicit membership,
selection rule and source provenance with the derived corpus before any
training. The Task did not specify a fraction or seed. Historical pilot
corpora and regimes remain untouched.

## Native format placement

Jack Heart, comment c17bdb50 (2026-11-11): "This probably needs a /format/
instead of cards?" Applied the full steer: `format/mod.rs` and
`format/limited.rs`, declared from `lib.rs`; bindings use `format::limited`.
No re-export from `cards`, which keeps definitions, packs, decklists and
world admission.

## Corpus admission counterexample — 2026-11-11

Re-admitting the retained source corpus (SHA256
`22f9b2b9411a69a85ff1270605dd68fba5941601859b752d2a4a0ecf5e4a6043`)
against this checkout's rebuilt w17 rejects 78/1,000 events, with 41 distinct
unsupported names. Examples: Scout's Warning (7 occurrences), Volcanic Torrent
(5), Blasphemous Act (5), Agent of Treachery (3), Noxious Gearhulk (3).
Historical Decklist issues must be removed by reparsing the unchanged source
text before explicit re-admission; the counts above use that path.

This contradicted the Task's premise that the current world admits all 1,000
and stopped publication pending a population decision (recorded below).
At that point no split had been published or used for training.

Jack Heart (2026-11-11), on the 41 rejected names: "oh these, are the tla
bonus cards". All 41 are reprints outside the defined TLA set (Scout's Warning
7 events, Blasphemous Act 5, Volcanic Torrent 5, Black Sun's Zenith 4, Humble
Defector 4, ...); 22 of them appear only in sideboards.

Decision (Jack Heart, 2026-11-11): "(1) is fine" — publish from the 922
admitted events, recording the 78 exclusions and their names in provenance.
The rejected names are TLE ("which are techincally part of TLE"); Task ed9b1d
covers defining all of TLE. The split must stay stable when those 78 are added.

Documentation choice (2026-11-11): use the steer's first permitted split rule,
retaining the existing ranking across all 1,000 source events and restricting
its first 100 to admitted events, without backfilling. This is the simpler
extension of the existing derivation and preserves assignments as TLE support
expands. The admitted held-out count is not assumed to equal 100; publication
must record the actual saved membership.

Publication (2026-11-11): saved membership is 827 training and 95 held out.
The 78 exclusions contain 41 distinct unknown names. Publishing with the
documented Doppler `manabot/prd` environment succeeded; an empty-cache remote
restore matched the manifest and loaded both populations. No open question
remains for implementation; broad validation belongs to gate.

## Docs assumptions after concept review — 2026-11-11 (headless, reversible)

- Plain types live at `manabot/pools.py` (`PoolCorpus`, `Pool`) and
  `manabot/game_results.py` (`read_game_results`, row types): "beside
  managym", importing no engine, training, torch or model code.
- New corpus path `experiments/data/mtg282/pools-v2/corpus.json`.
- `Pool` fields: `id` (17Lands draft id), `cards`, `human_build`, `split`;
  the seventh-win build index is provenance.
- Self-play seats get `{pool, build}` from new aligned `UniformDecks.pools`
  and `UniformDecks.builds`, replacing this branch's unmerged
  `build_episodes`; (pool, build) unique; `entry_ids` unchanged. A deck with
  no pool reports `pool: null`.
- `CardPool` rejects unsupported cards and non-positive counts at
  construction; `build.select` outside `selectable` raises `ValueError` and
  leaves the build unchanged; `limited.basic_cards()` retained for batched
  builders (answer 5 default).
- 4a: `ModuleStage.trains` is a tuple of names; self-play still trains
  exactly one; a model may not be both trained and frozen in one stage.

## Implementation assumptions — 2026-11-11 (headless, reversible)

- The engine-free reader needs basic-land names to check human builds, so
  derivation records them in `provenance.basic_lands`.
- Pooled players without `entry_ids` are named `pool` or `pool/build`; the
  collector joins result seats by player name.
- Module stages export each trained model as artifact `raw/<name>`.
- `CardPool` count errors name the card (binding parses counts itself).

## Sync onto main dac7f9005 — 2026-11-11

Main deleted `scratch/questions.md`; this branch keeps it until its pre-land
commit. Main moved engine stubs to `managym/__init__.pyi` with
`manabot/managym/__init__.pyi` a star re-export, so `CardPool` and
`format.pyi` moved to the engine package. The qa skill's training-recipe note
dropped its stale `policy_input_version=8` advice (main accepts only v9).

## Docs for tensor-native rules — 2026-11-12 (headless, reversible)

Steer 39c19262 (Jack Heart: "sadd it"): `limited.available(pool, deck,
max_cards=...)` -> int `[n, cards]` copies plus `can_stop` bool `[n]`;
`build.available` (`{name: copies}`) replaces `build.selectable`, no alias;
no separate bool-mask method. So the earlier `limited.selectable(...)` array
form is not built. Implied details settled in the docs:

- Card axis index = position in `managym.card_definitions_json()` (CardDefId
  order, 360 today, tokens included; tokens are never available). Docs map
  names through that existing call; no new name-index helper.
- Non-basic copies are pool minus deck, not clipped to slots left; basics get
  slots left; all zero at the cap.
- `available` raises `ValueError` naming the row for unreachable decks
  (over pool, non-pool non-basic, over cap, negative, bad shape); `legal`
  returns `False` for any non-negative illegal deck. No broadcasting of a
  single `[cards]` row.
- `managym.CardPool.from_counts(counts)` constructs from an array;
  `limited.basic_cards()` (names) stays for `experiments/tla_pools.py`.
- `limited.min_cards` is an int attribute; `limited.basics` bool `[cards]`.
