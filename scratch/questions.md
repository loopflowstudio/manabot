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
  Elemental pilot has little exposure to the changed decision. The final choice
  stayed Elemental, as the Task brief names it; the finished two-seed pilot
  reports that limitation without inferring equivalence.

## Sync resolution (2026-11-13)

Later sync onto main: retain this branch's notes despite main deleting the
file. Combine the payment-outcome module with main's `NonZeroUsize` import,
and retain both `mana_spending` and `mps_retest` in experiment discovery.

Main's own notes that the sync carried into this file (MTG-313 token naming,
Scryfall ids, the BF16 precision rejection) concern other Tasks; they are in
git `78397f2042:scratch/questions.md`.

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

## Non-redundant capture iteration (2026-11-13)

- Use the existing traversal and its cap for both surplus-inclusive and
  non-redundant counts. No selected source is restored in the omission probe,
  matching the recorded over-tap attribution on the post-tap board.
- Keep definition/state equivalence unresolved: PaymentModel holds outputs and
  resource transitions, not complete object state. Exact-identity loss is an
  upper bound on strategically meaningful loss and includes identical basics.
- Preserve current/shapes bytes; put the new paired measurement in
  current/non-redundant-shapes/.

The first paired-shapes command stopped during import, before any games or
evidence directory: installed etude-sdk 0.2.16 lacks `Seating`, while the merged
lock/source require 0.3.2. Sync the checkout's environment from its current lock
without rebuilding the native package. This does not rerun or migrate the
frozen pilot or its launcher database.

## Concept review (2026-11-13, at `59c82bc2be`)

- The recommended payment decision is taken to cover every mana payment, not
  only casts: nobody restricted the Task to spells, and a cast-only chooser
  leaves a second payment interaction for ability costs. The report prices the
  non-cast payments from published rows instead of asking a fourth question.
