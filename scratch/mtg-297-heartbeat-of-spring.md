# MTG-297: Heartbeat of Spring

## Dependency status (checked 2026-11-12)

- MTG-286 (PR #469) is still OPEN. `origin/main` is at 92560a1e0, which has no
  v10 policy input and no MTG-286 mana-payment rework in `flow/mana.rs`.
- Steer (comment bed9671f): "check main for those before designing on top of
  them, and stop at design review if the dependency isn't on main yet."
- The production design is MTG-286's unreviewed proposal,
  `scratch/tle-production-design.md` on branch
  `jack/mtg-286-define-every-tle-card-so-tla-limited-pools` (at 5f6ae4633).
  It covers more than Heartbeat (Animal Attendant, Toph, Uncle's Musings,
  Henge/Citadel suffixes, spending receipts). Heartbeat's part: the trigger
  needs the land's actual produced types and either player's bonus sources;
  `TriggeredManaAbility.mana`, `flow/zones.rs::tap_bonus_sources(controller)`
  and tap-time firing can't supply them. Read the reviewed version, not this
  summary, when resuming.
- The brief says to copy that design here "before starting", and the steer
  says it "is copied"; it is not in this checkout. Copy the reviewed version
  from MTG-286's branch or main when implement resumes.

## Docs

There are no user-facing docs for this change. A card definition ships as code
plus directed tests. The tracked markdown that names mana triggers
(`content/DECKS.md` and experiment reports) is decklists and frozen evidence.
The proposed v10 `mana_choices` encoding table belongs to MTG-286's
observation review. Changed doc paths: none.

## Implement status (2026-11-12, rechecked at realign)

Blocked; no code written. Rechecked at implement and realign: `origin/main` still at
92560a1e0 and PR #469 (MTG-286) is OPEN, unmerged. Per the steer, this stops
at design review. The production design is also unreviewed and says "no new
triggering record is implemented before that review". Resume implement once
#469 is on main and the production/observation design is reviewed; rebase
onto that main first.
