# PR D: decklist import and reproducible training on decklists

Full history, usage sketches and settled reviews: c1dd6c71c:scratch/f871a8-import-decklist-files.md
(earlier: e3b4f0583, d8fc9833c). Based on main: PR #394 (Task 892c47: typed
Rust cards, one open world w17, first-printing keys, support = set-file
definition) and T2 (87e012, c7979d310: 51 TLA cards extending w17) are merged;
last synced at 886e0b563. Main's T1/T2 notes ride in `scratch/questions.md`
below this Task's entries; they are main's, not this Task's.

## Accepted direction (Jack Heart)

- **2026-10-30: decks and decklists are managym.** Native `Deck {main,
  sideboard}` (`managym.Deck`) flattened into `PlayerConfig` (receipt keys
  `decklist`/`sideboard`); `admit_deck` takes `&Deck`. The decklist record
  (source text, filename, entries, line issues), parsing, name resolution and
  the import result live in Rust behind the managym Python surface;
  `manabot/decklist.py` and `catalog.DeckSetup` are removed. `TrainingDecks`
  holds `managym.Deck`. `MatchConfig` hero/villain fields stay. Training,
  checkpoint files and the Academy server stay Python.
- **2026-10-28: one deck concept.** Identity from counts alone; files are import
  provenance. Training submits decks to its world's native validation; fixed
  lists are checked upfront; no pre-registration defines compatibility.
- **Checkpoints bind worlds, not regimes.** A portable checkpoint holds weights,
  architecture/inference config, I/O contracts and the world binding; the run
  owns decks, sources, sampler, seeds and recovery. Pool-bound checkpoints play
  any deck the pool admits; authored/Elemental ones keep exact setups.
- **Scope.** One PR, stop before merge. Training matchups primary; Academy
  secondary. `uniform-distinct-v1` (two different decks, unbiased seats, native
  reset seed); curriculum, mirrors, 17Lands and the Academy 1a draft redesign
  are deferred (1a detail at c1dd6c71c).

## Accepted 2026-10-30 later (Jack Heart): a run has a way to get its next decks

Supersedes the run-owns-a-deck-list parts of the decisions below.

- **No content concept.** Beyond 1a: "content", "pool" and "pack" leave the
  deck, training and checkpoint surfaces. The world is the only concept. A
  checkpoint binds the world; the world's identity must still change when its
  cards change, but under the name world.
- **A run does not have a list of decks or files.** It has a deck source: a way
  to get the next decks for each new game. Decks may be sampled from many lists,
  generated at random on the fly, or (initially) drawn uniformly from a small
  list built however the experiment likes. Experiments are Python
  (`experiments/*.py` recipes), so the source is a Python object in the regime.
  Remove `TrainingDecks`-as-file-list and regime `decks: {files}`.
- **Cheap batch check.** One function answers "are all these decks valid in this
  world?" (e.g. `managym.check_decks(decks) -> per-deck issues`). Callers use it
  upfront when they want; game creation still admits every deck natively.
- **Recovery.** Exact resume must reproduce the same decks for the same games:
  a source is deterministic from its seed/state, or the run records each game's
  actual decks (it already does) and replays them. Checkpoints never know the source.

Implementation choices (not product decisions): the source protocol's exact
shape; how native autoreset obtains each next pair (e.g. Python fills a per-
stream queue, or the native uniform sampler remains one source implementation).

## Accepted 2026-10-30 (Jack Heart: 1a, 2a, 3a): decks need only a world

Since #394 the engine has one world (`world.rs`: w17; its cards are every
supported definition in `cardsets/`). Implement direction:

- **1a — no pool in deck admission.** Decks are checked against the installed
  world. Remove the pool/pack argument from `admit_deck`, `admit_decklists`,
  `Decklist::validate_deck`; `TrainingDecks.pool`; regime `decks: {pool, files}`
  becomes `decks: {files}`; drop `manabot sim --content-pool`; drop Academy
  `pack_key` and the **Content pool** field in `DeckImport.svelte`;
  `MatchConfig.from_import` takes no pack. A deck-run checkpoint binds the world
  (version, content digest, rules, I/O contract), not a `content_pool` with a
  pack key; no released checkpoint uses the branch's pool form, so no reader for
  it is needed.
- **2a — authored packs are fixed setups only.** Elemental,
  `ur-lessons-vs-gw-allies`, `mtg122-complexity-v1` are not deck-import or
  admission targets (stop registering them as pools in `pool.rs`). Their
  checkpoints keep setup bindings because their input vocabulary is the pack's.
  Frozen Elemental bindings and evidence are unchanged.
- **3a — set restrictions belong to deck making.** TLA-only and similar are
  filters used when choosing or generating decks (`world::set_cards`); admission
  only asks whether the world supports the cards.

Failure/recovery unchanged: unsupported cards reject on their line with the
whole input; saved imports re-admit against the installed world.

## Contract

- `counted-lines-v1`: exact source bytes (BOM/CRLF), filename, line
  diagnostics; front/full DFC names resolve, reverse-only/ambiguous/unknown/
  unsupported reject without substitution; whole input rejects on any issue.
- `development-v1`: nonempty main, positive counts, eligible non-token cards;
  no tournament legality. Limits: 64 KiB per source, 1,000 cards per player.
- Imported provenance stays private; Commands, stale rejection, reconnect,
  replay/fork identity and viewer safety unchanged. PR383/MTG-175 owns adaptive
  reassignment; PR389/19d240 owns shared simulation APIs.

## Implemented

`state::deck` (identity byte-equal to the retained Python digest);
`cards::deck_admission` (`admit_deck(&Deck)`, `check_decks(&[Deck])` reading
the world once, `DeckIssue.line`, crate-private `admit_in_world`; rules are
`DEVELOPMENT_RULESET`); `cards::decklist` (`Decklist`, `parse_decklist`,
`read_decklist`, `DeckImport {decklist, deck}`, `admit_decklists(labelled)`,
`Decklist::validate_deck(&Deck)`), all against the installed world only;
`python/deck_bindings.rs` (frozen `manabot.managym` classes, `check_decks`,
`WORLD_SETUP`, `Deck.player()` submits to the world).
`content_pack_for_match`: `"world"` admits each deck and runs the default pack;
an authored key runs only its own setups (main's rule); no fallback.
`MatchConfig` holds decks plus private `hero_source`/`villain_source`;
`from_import(hero, villain)` sets `content_pack="world"`; loading re-admits each
source. Academy `GameConfig` keeps only the two sources (JSON form); no
`pack_key` in payload, UI or validate response (which reports world version and
digest). Training: regime `deck_source: UniformDecks` (`uniform-distinct-v1`,
distinct decks in the experiment's order, native draw from the reset seed);
identities `deck_source_sha256`; collector/failure evidence record the source
digest and each game's actual decks. Deck-run checkpoints: `deck_world` binds
`world`, `world_digest` (card digest), rules, input and learning schema; no
pack, pool or source.

The authored-match parity receipt (`conformance/authored-match-parity-w17-v1`)
hashes source files; regenerate it (`uv run python -m academy.authored_match_parity
write`) after any further source edit, including formatting.

Compress check (2026-10-30): debug lib deck_admission/decklist/setups/
vector_env 27 passed, clippy/fmt clean; world/mirror/matchups/training-decks/
decklist/catalog/Academy-decklist/match/self-play/elemental/boundaries/
learning-state 122 passed; parity receipt rewritten and verified.

Check (2026-10-30, world-only + deck source): debug `cargo test --no-fail-fast`
all pass (after the 1a cut), lib deck_admission/decklist/setups/vector_env/pool/
world 40 passed and clippy clean after `check_decks`; decklist/training-deck/
Academy-decklist/checkpoint-catalog/mirror/matchups/match/world/boundaries/
self-play/elemental-uniform 91 passed; UI vitest 16 passed, svelte-check 0
errors; parity receipt rewritten, verifies. Gate owns the affected-area suites.

## Delete — do not maintain

Done (compress): the `authored-pack-v1` roster binding (`AUTHORED_PACK_BINDING`,
`_binds_roster`, `_setup_world(roster=)`, `bound_pack`) — a pool-era form; setup
bindings are main's again (exact pair; Elemental roster), world bindings are
`deck_world`. Main's mirror/matchup/world tests restored unchanged.
Done (this pass): pack/pool arguments of `admit_deck`/`admit_decklists`/
`validate_deck`/`development_rejection`; Python `admit_deck` (now
`check_decks`); `Deck.player(content_pack)`; explicit-pool admission for
authored keys in `content_pack_for_match`; `TrainingDecks` (`pool`, `sources`,
`from_decklists`, `from_files`); regime `decks: {pool, files}` and `read_decks`;
`training_decks_sha256`/`content_pool` identities; `pool_world`,
`admit_pool_decks`, checkpoint `content_pool`/`save_checkpoint(content_pool=)`;
`manabot sim --content-pool`; Academy `pack_key`, `GameConfig.content_pack`,
**Content pool** field. Earlier: `import_match`, `ImportedMatch`, `PoolIdentity`,
caller `ruleset`, `GameConfig.imported_decks`, `manabot/decklist.py`,
`catalog.DeckSetup`, `deck_collection.py`, checkpoint deck bindings (list at
c1dd6c71c).

## Handoff

Copied evidence (scratch/tla-evidence, scratch/tla-card-ledger.jsonl) is kept
out of staging by local exclusions; stage explicit owned paths and use
`lf commit --no-add`. PR #395 is open against main for Jack Heart's review; no
merge or auto-merge.
