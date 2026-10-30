# PR D: decklist import and reproducible training on decklists

Full history, usage sketches and settled reviews: c1dd6c71c:scratch/f871a8-import-decklist-files.md
(earlier: e3b4f0583, d8fc9833c). Based on main fa5e13b23 since PR #394 (Task
892c47: typed Rust cards, one open world w17, first-printing keys, support =
set-file definition) merged 2026-10-30; synced at cd86469b1.

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
`cards::deck_admission` (`admit_deck`, `DeckIssue.line`, crate-private
`admit_in_pool`); `cards::decklist` (`Decklist`, `parse_decklist`, `read_decklist`,
`DeckImport {decklist, deck}`, `ImportedMatch` with `setup_id`/`swapped`/
`validate_current`/`validate_setup`, `admit_decklists`, `import_match`);
`python/deck_bindings.rs` (frozen `manabot.managym` classes, `to_dict`/
`from_dict`, pickle/copy, Pydantic core schema; `DeckImportError.decklists`).
`MatchConfig.from_import` builds imported matches; `TrainingDecks.from_decklists`
/`from_files` admit natively; `set_training_decks`, producer admission and
`validate_agent_decks` are unchanged in behavior.

The authored-match parity receipt (`conformance/authored-match-parity-w17-v1`)
hashes every `managym/src/*.rs`; it was regenerated after the sync (source hash
only, behavior parity unchanged) and must be regenerated after further Rust edits.

Check (2026-10-30, after sync): debug `cargo test --no-fail-fast` all pass;
decklist/training-deck/Academy-decklist/match/matchups/mirror/world/boundaries
84 passed; tests/academy + CLAUDE.md world set 324 passed, 2 failed only on the
locally absent artifact `experiments/data/rul-9-played-workloads-v1.measurement.json`
(in the manifest; fetch from the artifact store, not a branch defect).

## Delete — do not maintain

Done: `manabot/decklist.py` and its Python parser/face/ambiguity tests (now in
`decklist.rs`); `catalog.DeckSetup`; unused `Decklist.source_sha256`; earlier `deck_collection.py`, checkpoint
deck bindings, `env.match.parse_deck`, `validate_imported_players` (list at
c1dd6c71c).

## Next implement direction (concept review, 2026-10-30, no person present)

Settled by the accepted decisions above; no new product choice.

1. **One copy of each deck; sources are provenance only.** Today an import is
   stored twice: `MatchConfig.hero_deck`/`villain_deck` (+ sideboards) and
   `imported_decks: ImportedMatch` (pool, ruleset, two `DeckImport`s each with a
   `Deck`), so `validate_setup` must prove they agree on every `MatchConfig`
   and trace load. Replace `ImportedMatch` with per-seat source records:
   `MatchConfig` keeps the decks and gains private
   `hero_source`/`villain_source: Decklist | None` (Academy trace the same).
   `import_match` returns the two `DeckImport`s (or is replaced by
   `admit_decklists` with seat labels); `MatchConfig.from_import` takes them.
   Delete `ImportedMatch`, `setup_id`, `swapped` (seat swap moves two fields),
   `validate_setup` and the trace's dict round-trip. Keep the source-truthfulness
   check at record load as "re-importing a retained source yields this seat's
   deck", one helper on `Decklist`/`DeckImport`, not a match-level record.
2. **The world's validation decides, not pool-identity equality.**
   `validate_current` rejects any imported record once `content_digest`
   changes (any card edit), while curated traces and training decks load and
   are re-admitted natively. Imported traces/rematches re-admit their sources
   against the installed world like everything else; drop `PoolIdentity` from
   the record. The validate endpoint may still report `world_version`/
   `content_digest` as information. Counterexample kept: a changed world can
   make an old tape diverge — that is the existing replay/canonical-replay
   check's job for every trace, not a decklist-only gate.
3. **The ruleset is the world's, not the caller's.** `development-v1` is the
   only ruleset, yet `admit_deck`, `admit_decklists`, `import_match` and
   Academy's `DeckImportInput.ruleset` accept a free string that can only
   error. Remove the parameter; keep `DEVELOPMENT_RULESET` as the named
   constant the validate response and CLI report. Adding a tournament format
   later is a review decision and would reintroduce a choice then.
4. **Remove the committed `.DS_Store`** (added by 2497dfa01 "lf pr open").

Must survive: whole-input rejection with retained source and line issues;
rematch prefills the editor from retained sources; viewer-safe trace export
strips decks when sources are present; Commands, stale rejection, reconnect and
replay/fork identity. Regenerate the authored-match parity source hash after
Rust edits; run debug `cargo test` and the decklist/Academy/match test sets.

## Handoff

Copied evidence (scratch/tla-evidence, scratch/tla-card-ledger.jsonl) is kept
out of staging by local exclusions; stage explicit owned paths and use
`lf commit --no-add`. PR #395 is open against main for Jack Heart's review; no
merge or auto-merge.
