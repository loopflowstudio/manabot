# Assumptions (MTG-347 kickoff, 2026-11-13)

- "Self-play" for the frequency count is read as mirror games per deck. A
  trained checkpoint is used where one loads on the current world; heuristic
  and random mirrors run everywhere. Simpler option taken: do not train a
  policy just to count.
- The Task asks for a small learning comparison; the 2026-11-13 comment hands
  learning-effect claims to MTG-265. Both kept: one exploratory A-vs-automatic
  comparison under USD25, reported without a claim.
- The automatic-spend switch for that comparison is branch scaffolding.
  Whether it lands is for review of the recommended design.
- "Payment outcome as one decision" is the leading hypothesis only; no design
  is chosen before the counts.
- Frequency slice: select the first ten 40-card TLA frontier population
  entries, in file order, with their source IDs and exact main/sideboard lists.
  This keeps the sample bounded and does not rewrite the frozen regime.
- Over-tap attribution is observational, not a prediction of post-fix play;
  the non-over-tapped subset retains the original decision denominator.
- Realign (2026-11-13): spend prompts are 0.2% of Elemental decisions, so the
  planned Elemental A-vs-B learning run is near-certain null. Left open in the
  plan (Elemental vs a Heartbeat-including recipe); default stays Elemental,
  as the Task brief names it, and the run waits until after part 3.

## Sync resolution (2026-11-13)

Retain main's MTG-313 `scratch/qa.html`: it records new card QA, rather
than the inherited MTG-324 page removed at kickoff. Preserve main's source
facts below as historical context, separate from MTG-347 assumptions.

## Source facts from main

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

## Historical precision notes from main

### Assumptions and scope

- The numerical rejection at attempted update 141 is decisive under the frozen
  rule. The paid strength evaluation was omitted because it cannot admit BF16;
  the report states M = 75 and the absence of strength and trajectory comparisons.
- Keep `BF16_MIN_HIDDEN_DIM = 256` as the measured speed crossover, with
  `ladder_precision` returning float32 everywhere after rejection, as the Task
  specifies. Reopening BF16 admission needs another learning result.
- Earlier A/B choices and launch history remain in git `e181224e7` and the
  experiment report. No new rental, experiment, or numerical-guard change.

## Implementation assumptions (2026-11-13)

- Keep the planned Elemental learning comparison (two seeds per arm, 32 matched
  updates). Its rarity of pool decisions is a limitation, not a reason to replace
  it with a competing compound-decision study.
- Etude SDK 0.2.16 refuses launcher schema 13; SDK 0.2.15 reads it. Use that
  compatible launcher without migrating or replacing the shared database.
- Fixed-source enumeration includes surplus tapping; cap hits retain lower-bound
  counts. Native saved payment obligations still gate every final remainder.

- Equal mana output alone does not establish strategic interchangeability (a
  mana creature may also attack, block or carry an effect). The diagnostic
  reports exact-source identity counts separately; the recommendation must not
  authorize a source-capability collapse without a rules equivalence argument.

## Measurement baseline after merge (2026-11-13)

The rebuilt current engine replays elemental_bg deal 0 to digest
`25e7d895b1660c391141b9e627c53deb89d7dfe61c1a9b79545c21035e9d6c54`
with capture both off and on, with identical payment counters. The retained
58e5923c53 trace ends at `4814ad1b380e005edc2f31a816d802db2284d61b53070bd18b55144c8e2abb50`.
Main's engine/card changes merged after counting. Preserve the original evidence;
regenerate the frequency baseline under the instrumented current engine in
`experiments/data/mana_spending/current/`, on the same 21 exact deck lists,
8 games and seeds. Use that baseline for strict shape replay digest checks.
This is a second measured baseline, not a rebind of the old trace or checkpoint.

- Shape C is measured as an explicit automatic/override entry at the payment
  boundary, followed by an outcome choice only for a nontrivial override. Its
  decision range includes that entry. Binding override intent earlier at cast
  selection would be another interaction design; it is not counted as free.
- Zero-mana payments keep their existing pool and add no fixed sources in this
  boundary measurement. Unresolved folded probes include every case without
  exactly one immediate cast-payment row, not only target/mode announcements.
