# TLA direct-Rust decisions

Reversible operator choices made headless (2026-10-28 unless dated); each is
noted in the PR. Jack Heart's 2026-10-29 direction superseded the qualification
and pool-membership choices that were here (history: `be2b7586b:scratch/questions.md`).

- Old worlds run from their recorded git revision; no in-process rules shims.
- Superseded 2026-10-29 by Jack Heart's one-representation decision (w17, no
  JSON, tokens from `Effect`): the earlier choices to stay in w16, keep the IR's
  JSON shape and build tokens in `learning.py`.
- 2026-10-29: the 14 setup-used incomplete TLA cards keep their definitions and
  are listed in one `tla::INCOMPLETE` const (key, missing behavior) beside
  `cards()`, rather than a field on every definition. Since the one-representation
  rewrite every non-token definition outside `INCOMPLETE` is eligible. Crossroads of Destiny moved to `cards/fixtures.rs` but stays
  in the default pack, because Rust integration tests and `manabot/verify` build
  games from it. D1 keeps the `tla-pool-v1` key; `set_cards` is the set filter.
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
