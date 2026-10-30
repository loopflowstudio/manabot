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

## Concept review 2026-10-30 (proposed; awaiting Jack Heart)

Since #394 the engine has one world (`world.rs`: w17; its cards are every
supported definition in `cardsets/`). This branch still makes every deck caller
name a content pool, and for imported or training decks the only meaningful
value is `"world"`: `admit_deck(pack, &Deck)`, `admit_decklists(.., pack_key)`,
`Decklist::validate_deck(pack, ..)`, `TrainingDecks.pool`, regime
`decks: {pool, files}`, `manabot sim --content-pool`, Academy `pack_key` and a
free-text **Content pool** field in `DeckImport.svelte`, and checkpoint
`content_pool` bindings. Authored packs (Elemental, `ur-lessons-vs-gw-allies`,
`mtg122-complexity-v1`) are also registered as pools (`pool.rs:111`).

Proposal (not accepted): deck admission takes no pool; a deck is checked
against the installed world. Regime `decks: {files}`; no `--content-pool`; no
Academy pool field; a deck-run checkpoint binds the world (version, content
digest, rules and I/O contract). Authored packs stay fixed setups whose
checkpoints keep setup bindings. A set restriction such as TLA-only belongs to
how decks are chosen or generated (`world::set_cards`), not to admission.

Open questions (concept-review.html):
1. Drop the pool parameter from deck admission and the surfaces above?
2. Are authored packs only fixed setups, or also targets for imported/training decks?
3. Is a set restriction (TLA-only) a deck-making filter, or a run-level rule checked at admission?

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
`cards::deck_admission` (`admit_deck(pack, &Deck)`, `DeckIssue.line`,
crate-private `admit_in_pool`; the world's rules are `DEVELOPMENT_RULESET`, no
caller ruleset); `cards::decklist` (`Decklist`, `parse_decklist`, `read_decklist`,
`DeckImport {decklist, deck}`, `admit_decklists` (labelled; a match labels
`hero`/`villain`), `Decklist::validate_deck`); `python/deck_bindings.rs` (frozen
`manabot.managym` classes, `to_dict`/`from_dict`, pickle/copy, Pydantic core
schema; `DeckImportError.decklists`; `DEVELOPMENT_RULESET`).
Each deck is stored once: `MatchConfig` holds the decks plus private
`hero_source`/`villain_source: Decklist | None`; `MatchConfig.from_import(pack,
hero, villain)` builds imported matches, and loading re-admits each source
against the installed world (`validate_deck`). Academy's `GameConfig` keeps
`content_pack` and the two sources (JSON form); `to_rust`, rematch and the
validate endpoint re-admit against the installed world, and viewer export
strips the sources and decks. `TrainingDecks.from_decklists`/`from_files` admit
natively; `set_training_decks`, producer admission and `validate_agent_decks`
are unchanged in behavior.

The authored-match parity receipt (`conformance/authored-match-parity-w17-v1`)
hashes source files; regenerate it (`uv run python -m academy.authored_match_parity
write`) after any further source edit, including formatting.

Check (2026-10-30, concept-review corrections): debug `cargo test --no-fail-fast`
all pass; decklist/training-deck/Academy-decklist (incl. new rematch test)/
checkpoint-catalog/advice/match/matchups/mirror/boundaries/parity 86 passed;
UI vitest `game.svelte.test.ts` 16 passed, svelte-check 0 errors. Compress
(drop `import_match`): lib decklist tests, clippy, decklist/Academy/training-deck/
match/advice/matchups/mirror/boundaries 81 passed; parity receipt rewritten, verifies.
After the T2 sync: `academy.authored_match_parity verify` OK; decklist/
training-deck/Academy-decklist/match 39 passed. PR #395 CI pending on the sync.

## Delete — do not maintain

Done: `import_match` (Rust and Python; callers use `admit_decklists`), `ImportedMatch` (`setup_id`, `swapped`, `validate_current`,
`validate_setup`), `PoolIdentity`, the caller `ruleset` parameter (Rust, Python,
Academy `DeckImportInput`, UI payload), `GameConfig.imported_decks`, tracked
`.DS_Store`; earlier `manabot/decklist.py` and its Python parser tests,
`catalog.DeckSetup`, `Decklist.source_sha256`, `deck_collection.py`, checkpoint
deck bindings, `env.match.parse_deck`, `validate_imported_players` (list at
c1dd6c71c).

## Concept review (2026-10-30, after T2 sync)

Implementation matches the accepted direction: one `Deck`, decklists and
admission native, sources re-admitted against the installed world, no caller
ruleset, training decks checked upfront, checkpoints bound to the world only.
No open product choice. Gate nit: `manabot/managym/__init__.pyi` declares
`admit_deck` twice (keep the documented one beside `admit_decklists`).

## Handoff

Copied evidence (scratch/tla-evidence, scratch/tla-card-ledger.jsonl) is kept
out of staging by local exclusions; stage explicit owned paths and use
`lf commit --no-add`. PR #395 is open against main for Jack Heart's review; no
merge or auto-merge.
