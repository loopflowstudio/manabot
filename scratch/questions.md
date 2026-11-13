# Assumptions and source facts

2026-11-12: Use `token.plain_spirit` for Release to Memory's unrestricted
Spirit (concept review replaced `token.colorless_spirit`: TLA's Spirit is
colorless too), keeping TLA's `token.spirit` unchanged. Use `token.zeppelin` for Lita's
unprinted Vehicle. Both are reversible internal naming choices.

Scryfall API verified the first-printing card ids from the pinned inventory:
- https://api.scryfall.com/cards/b45c4bbf-2c61-4d49-96c8-34a0fb0d6d02
- https://api.scryfall.com/cards/f3a5a044-d474-43f4-95a9-e67808908e6c
The earliest 1/1 colorless Spirit token from the ordered printings search is
P04 (2004-01-01), id 46b60d95-b9bc-40f8-b986-bfa8e3eb74f3, empty Oracle text.
Lita's printed rules say Zeppelin is a 5/5 colorless artifact Vehicle with
flying and crew 3; it becomes a creature only when animated.

2026-11-12: The concept-review plan called existing flicker text "Return it
to the battlefield…". Inspection of main's `move_description` shows "Put it
onto the battlefield under its owner's control" for `Moved`. Preserve that
actual wording in the regression assertion, consistent with the instruction
not to reword existing cards; do not change the move verb.

2026-11-12: Use the descriptive `w17-token-cards` fixture suffix for the new
Academy catalog binding. Preserve the existing `w17` files unchanged; replay
only existing Commands and require their offer ids/labels to match.

2026-11-13: Main now includes MTG-297 (Heartbeat of Spring). Use the complete
61/61 bonus sheet and 1,000-deck admission as the combined acceptance outcome;
do not keep a known obsolete "only Heartbeat missing" expectation. Seed new
`w17-token-cards` fixtures from main's current w17 Commands before refreshing
the combined catalog, because main's mana spending redesign changed Commands.
The managed `lf sync` plan selected merge_target for this branch; preserve
history through that supported integration rather than raw git rebase.

## Historical precision notes from main (MTG-328)

# Assumptions and scope

- The numerical rejection at attempted update 141 is decisive under the frozen
  rule. The paid strength evaluation was omitted because it cannot admit BF16;
  the report states M = 75 and the absence of strength and trajectory comparisons.
- Keep `BF16_MIN_HIDDEN_DIM = 256` as the measured speed crossover, with
  `ladder_precision` returning float32 everywhere after rejection, as the Task
  specifies. Reopening BF16 admission needs another learning result.
- Earlier A/B choices and launch history remain in git `e181224e7` and the
  experiment report. No new rental, experiment, or numerical-guard change.
