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
