# Decklist assumptions

Earlier entries (construction policy, limits, coordination with PR383/PR389,
inherited MTG-207/MTG-193 notes) are recorded in the plan or at
c1dd6c71c:scratch/questions.md.

- Excluding mirrors from `uniform-distinct-v1` is an assumption; at least two
  distinct lists are required. The policy is code-named, not a stored field.
- Checkpoint `bc` extra still carries run_id/stage_id/regime_digest producer
  links that predate this branch; producer admission depends on them. Moving
  them to run records is a separate change.
- 2026-10-30 (managym move): native parsing approximates Python's `casefold()`
  with Unicode `to_lowercase()` and splits lines at `\r\n`, `\n` and `\r` only.
  NFC uses the new `unicode-normalization` dependency. `DeckImport.document`
  is now `.decklist`, and `Deck` serializes as `{decklist, sideboard}`, so the
  unreleased `setup_id`/training-deck digests changed; `Deck.identity` keeps the
  retained value. `ImportedMatch.pool` is a dict in Python.
- 2026-10-30 (concept-review corrections): Academy's `GameConfig` is a
  dataclass persisted through `asdict`/JSON, so it keeps each seat's source in
  `Decklist` JSON form (`hero_source`/`villain_source: dict | None`) beside a new
  `content_pack`, and reads it with `Decklist.from_dict` on load; `MatchConfig`
  holds native `Decklist`s. The CLI now prints both `Deck.identity` values
  instead of the deleted `setup_id`; the validate endpoint drops `setup_id`.
  `validate_deck` compares only the re-admitted deck, so retained resolved
  entries (card ids) are provenance and may go stale without rejecting.
- 2026-10-30 (sync onto main fa5e13b23, #394 squashed): merged with the last
  integrated #394 commit 86ebe9ccf as base so #394's later changes win. Deck
  admission and decklists moved from the deleted `semantic` module into
  `managym/src/cards/` (`deck_admission`, `decklist`); `content_pack_for_match`
  lives in `cards/setups.rs`. `PoolIdentity` drops `compiled_semantics`;
  `DeckEntry.semantic_key` is `key`; the pool key is `"world"`.
- Old worlds run from their recorded git revision; no in-process rules shims.
- Superseded 2026-10-29 by Jack Heart's one-representation decision (w17, no
  JSON, tokens from `Effect`): the earlier choices to stay in w16, keep the IR's
  JSON shape and build tokens in `learning.py`.
- 2026-10-29: the 14 setup-used incomplete TLA cards keep their definitions and
  are listed in one `tla::INCOMPLETE` const (key, missing behavior) beside
  `cards()`, rather than a field on every definition. Since the one-representation
  rewrite every non-token definition outside `INCOMPLETE` is eligible. Crossroads of Destiny moved to `cards/fixtures.rs` but stays
  in the default pack, because Rust integration tests and `manabot/verify` build
  games from it. `set_cards` is the set filter (pool key since renamed `"world"`).
  Undefined TLA cards are no longer listed as unsupported pool rows.
- Not rewritten (dated evidence): the Academy curated pack `local_identity` values
  keep pre-rename keys.
- Full TLA completion remains the eight-PR plan; no merge or new Task authorized.

Main's notes at the syncs live at `6d069d6e5:scratch/questions.md` and `origin/main@633dca159:scratch/questions.md`.

- 2026-10-29 (one-representation rewrite): five cards were filed under a later
  printing and moved to their first printing's set (see plan). Three are TLA
  reprints (Corrupt Court Official, Epic Downfall, Lightning Strike), so
  `set_cards(["tla"])` no longer lists them; a TLA-legal filter would need
  printings, not first-printing key prefixes. Followed Jack Heart's
  first-printing rule; flag for f871a8 deck validation.
- 2026-10-29: `key` and `scryfall_id` are `#[serde(skip)]` — identity metadata,
  not rules — so content and state digests stay equal to w16's and digest-bound
  fixtures keep validating.
- 2026-10-29: learning token values are stable blake3 hashes of Rust/printed
  names (no vocabulary file); the policy builds its vocabulary from observed
  (kind, value) pairs as before.
- 2026-10-29: D1/f871a8 interface change: `ContentPool.compiled_semantics`
  (IR/source hashes) is gone; manifests carry `pack_key`; rows add
  `scryfall_id`; `IrError` is `ContentError`. `content_pool`/`pool_content`/
  `Eligibility` otherwise unchanged.
- 2026-10-29: the Python offline compiler (`manabot/semantic/compiler.py`) and its
  frozen-source tests stay as the reader/reproducer of frozen v1–w11 IR; nothing
  current reads it.
- 2026-10-29 (support reorg): tests that exercise several cards stay where
  they were (Corrupt Court Official with Ozai's Cruelty in `cards/tla.rs`, Epic
  Downfall with Power Word Kill in `cards/afr.rs`); Lightning Strike has no test
  of its own, so no `ptk.rs`/`ths.rs` test files exist yet.
- 2026-10-29 (concept review): world pool key becomes `"world"` (not per-world
  `"w17"`), describing the default pack; row `semantic_key` becomes `key`.
  Coordinate with f871a8 in one comment. The v1 coverage gate is left for a later
  Task rather than PR394.

- 2026-10-29 (assumption): superseded corpora now without consumers
  (`semantic-kernel-w15-v1`, `authored-match-w11-v1`, `authored-match-parity-w11-v1`,
  `public-commitment-parity-v1`) stay in the tree like w12–w14; archiving them
  to S3 is a separate publish step, not done here.
- 2026-10-29 (public-commitment parity, decision): bound to w17 by prefix. Only
  viewer 0's tracker fails (Command 68, South Pole Voyager revealed from library
  into hand); viewer 1 consumes the hero's Island discard (30) and
  `decline_learn` (102) over all 142 Commands. `TRACKED_COMMANDS = {0: 68,
  1: 142}`; identity-stream parity still covers the whole tape. Task 3be183
  (engine wave) owns the transport and removing the bound. No xfail.
