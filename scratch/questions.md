# Questions and assumptions — MTG-239, 2026-11-03

- Jack Heart's Task directive is the scope. The design is
  [mtg-239-complete-five-defined.md](mtg-239-complete-five-defined.md).
- Existing inherited notes select no work. Their complete committed source is
  `22e4aab716f2b582d704b88dc5d22cb1547ce623:scratch/questions.md`; frozen failures
  and historical evidence remain unchanged.
- PR #414 delivered at `44388ad42` and is integrated here. Its card/state
  crates and sparse counter storage are the surviving owners.
- Headless choice: no new planeswalker/battle definitions or combat system.
  Existing type flags plus loyalty/defense counters support the requested
  any-target damage contract, tested with controlled shells.
- Jack Heart’s November 3 steer (comment 1f3876a7) resolves the policy
  discussion: preserve playability for known bugs, track defects as bug Tasks,
  finish these five fixes without broader policy work or new Tasks.
- Jack Heart accepted both follow-up concept-review decisions on November 3
  (comment 6b64d212): remove the empty bug-based admission exclusion and retain
  immediate ordering Commands with clearer draw-order wording. Keep automatic
  final placement; no draft, confirmation or undo lifecycle.
- Current w17 receipts are open-world evidence: Rallier’s corrected filter
  changes the authored match, and Accumulate Wisdom adds explicit Commands to
  three navigation prefixes. Prior bytes remain at the base git reference.

Sync completed with `lf sync --continue`; the combined-runtime receipt refresh
is retained at `6d30c9013`. The plan owns its proof limits and remaining gate,
demo and delivery work. No new product decision is unresolved.
