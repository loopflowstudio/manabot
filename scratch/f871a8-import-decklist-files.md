# PR D: decklist import and reproducible training on decklists

Full history, usage sketches and settled reviews: c1dd6c71c:scratch/f871a8-import-decklist-files.md
(earlier: e3b4f0583, d8fc9833c). Stacked on PR #394 (Task 892c47, typed Rust
cards, one world w16, first-printing keys, support = set-file definition).

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
`semantic::deck_admission` (`DeckIssue.line`, `admit_in_pool`);
`semantic::decklist` (`Decklist`, `parse_decklist`, `read_decklist`,
`DeckImport {decklist, deck}`, `ImportedMatch` with `setup_id`/`swapped`/
`validate_current`/`validate_setup`, `admit_decklists`, `import_match`);
`python/deck_bindings.rs` (frozen `manabot.managym` classes, `to_dict`/
`from_dict`, pickle/copy, Pydantic core schema; `DeckImportError.decklists`).
`MatchConfig.from_import` builds imported matches; `TrainingDecks.from_decklists`
/`from_files` admit natively; `set_training_decks`, producer admission and
`validate_agent_decks` are unchanged in behavior.

Check (2026-10-30): debug `cargo test --no-fail-fast` fails only
`semantic_conformance_tests` (w15 manifest, ur-vs-gw-5eed digest; #394's);
clippy `--all-targets --all-features -D warnings` clean; decklist/training-deck/
Academy-decklist/self-play/uniform 30 passed; boundaries/catalog/world/recovery/
game-error/capture/matchups/mirror 143 passed; tests/academy + CLAUDE.md world
set: 30 failures, all w13/w15 fixtures, receipts or learning-schema 8≠7 against
w16 (#394's evidence, not regenerated here).

## Delete — do not maintain

Done: `manabot/decklist.py` and its Python parser/face/ambiguity tests (now in
`decklist.rs`); `catalog.DeckSetup`; earlier `deck_collection.py`, checkpoint
deck bindings, `env.match.parse_deck`, `validate_imported_players` (list at
c1dd6c71c).

## Handoff

Copied evidence (scratch/tla-evidence, scratch/tla-card-ledger.jsonl) is kept
out of staging by local exclusions; stage explicit owned paths and use
`lf commit --no-add`. Scratch sits just under its token target while #394's
tracked notes are present; keep this note short until #394 lands.
