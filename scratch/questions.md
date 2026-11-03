# Questions and assumptions — MTG-239, 2026-11-03

- Jack Heart's Task directive is the scope. The design is
  [mtg-239-complete-five-defined.md](mtg-239-complete-five-defined.md).
- Existing inherited notes select no work. Their complete committed source is
  `22e4aab716f2b582d704b88dc5d22cb1547ce623:scratch/questions.md`; frozen failures
  and historical evidence remain unchanged.
- PR #414 is open, not a dependency handed off for this checkout. Keep the
  focused rules changes on the supplied base and reconcile its card/counter
  ownership when delivered. No competing catalog rewrite.
- Headless choice: no new planeswalker/battle definitions or combat system.
  Existing type flags plus loyalty/defense counters support the requested
  any-target damage contract, tested with controlled shells.
- Use `/Applications/Loopflow.app/Contents/MacOS/lf` for Work commands per Jack
  Heart's environment handoff (comment 14c4c6af). No new Graph or auth repair.

- Jack Heart’s November 3 steer (comment 1f3876a7) resolves the policy
  discussion: preserve playability for known bugs, track defects as bug Tasks,
  finish these five fixes without broader policy work or new Tasks.
- Current w17 receipts are open-world evidence: Rallier’s corrected filter
  changes the authored match, and Accumulate Wisdom adds explicit Commands to
  three navigation prefixes. Prior bytes remain at the base git reference.

## Sync onto main — 2026-11-03

- Main at `44388ad4211b4292cc161a3087eb1230b2005737` delivered the card/state
  crate split, scalable keywords and sparse counters. Preserve those owners and
  APIs while carrying all five MTG-239 fixes into them. Earlier MTG-238 notes
  remain at that commit’s `scratch/questions.md`.
- Current w17 receipts must be regenerated from the combined runtime: neither
  side’s hashes represent both changes. Preserve this branch’s explicit library
  ordering Commands and fixed match behavior; historical receipts stay unchanged.
