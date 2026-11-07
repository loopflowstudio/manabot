# 30df0c direction — 2026-11-06

Jack Heart, at demo review: validate that catalog record ids are not much worse
than hashed ids before long training, by coordinating with frontier Task c5fba4,
then start the frontier training with catalog on based on that result.

Split agreed with c5fba4 (steer c9d82ac9):
- c5fba4 owns the paid check: a matched w64 learning comparison (same recipe,
  seed, decks and updates; only `record_vocabulary` differs), with a criterion
  fixed in advance, as a gate before its 54-hour runs. A failure sends the
  embedding choice back to review. 30df0c uses no paid compute.
- 30df0c: hashed stays selectable for fresh regimes
  (`test_fresh_regimes_can_select_hashed_ids`); PR #437 says c5fba4's gate
  validates learning quality; the coverage sweep is kept
  (`manabot.verify.record_coverage`, `tests/model/test_record_coverage.py`).

The check runs both arms from main, so #437 merges first. Hashed remains a
config option, so a failed gate needs no revert. Merging is Jack's decision.
