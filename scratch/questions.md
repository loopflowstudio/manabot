# Implementation decisions — 2026-10-20

- The supplied list contains 42 owned cards. Direct Scryfall card requests
  returned HTTP 429; its bulk index and compressed JSONL export provided all
  42 exact Oracle identities. The retrieved export is retained locally in
  `elemental-oracle.json`.
- Arbitrary scry ordering is now implemented; Gather the White Lotus is
  registered with directed reorder/privacy evidence.
- Native registrations are internal progress, not completed MTG-161 delivery.
  The Task remains unfinished until all cards, compiled semantics, new corpus
  and delivery checks are complete. No Task completion or publication occurred.

- The Elemental semantic pack admits two 46-card validation rosters (one copy
  of each owned card split across seats, plus five of each basic per roster).
  These exercise the catalog and all 42 definitions without selecting the
  final ten product decks, which remain the fourth Task's responsibility.
