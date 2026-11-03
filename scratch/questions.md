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
- Current w17 receipts are open-world evidence: Rallier’s corrected filter
  changes the authored match, and Accumulate Wisdom adds explicit Commands to
  three navigation prefixes. Prior bytes remain at the base git reference.

## Sync onto main — 2026-11-03

- Main at `44388ad4211b4292cc161a3087eb1230b2005737` delivered the card/state
  crate split, scalable keywords and sparse counters. All five MTG-239 fixes
  now use those owners and APIs. Earlier MTG-238 notes
  remain at that commit’s `scratch/questions.md`.
- Current w17 receipts were regenerated from the combined runtime: neither
  side’s old hashes represented both changes. This branch’s explicit library
  ordering Commands and fixed match behavior survive; historical receipts stay
  unchanged.

Sync resolution completed locally with `lf sync --continue`. Rebuilt the native
extension and refreshed the mutable w17 fixtures/receipts through their producers
(14 semantic-kernel cases, 3,473 Commands; authored parity 135 Commands / 816
viewer checks; public parity 66 commitments / zero gaps). Focused post-sync
proof: three Lightning Strike tests passed in debug; 14 Academy parity, choice
navigation and current belief-receipt tests passed. The public receipt writer
finished after the merge commit, so its result is retained in a local follow-up
commit `6d30c9013`. The reconciled plan owns remaining validation and delivery;
no new product decision is unresolved.
