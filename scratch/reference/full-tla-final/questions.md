# Full TLA — review decisions and assumptions, 2026-10-28

**Draft for review.** The accepted problem is Jack Heart's full TLA request;
Jack Heart accepted the eight-PR plan and authorized implementation/publication;
every merge requires his review. This research Graph only completes its handoff.

- **Inventory boundary:** TLA set code means all 286 gameplay identities and
  their required token/copy closure. TLE-only Beginner Box/Jumpstart/Scene and
  source-material cards remain separate even when products mix them. This is
  the simpler reversible interpretation of “TLA card file,” not an assertion
  that it implements retail Draft/Sealed. The official checklist and ledger
  preserve exclusions so a wider product decision can expand scope explicitly.
- **Accepted — Jack Heart, 2026-10-28:** option 1a, arbitrary valid TLA decks through the shared engine, with decklist-file input. File import can be its own parallel PR. No more than eight TLA PRs; at most two or three concurrent card groups; automatically build, Jack reviews each PR.
- **Format/admission details:** propose the initial two-player deck-construction rules and a documented decklist text dialect. Curated-only play is no longer an open option. Draft/Sealed collation/UI, Commander, multiplayer and new authored product decks remain separate. Secret of Bloodbending remains required in two-player play.
- **Proposed PR partition:** seven TLA PRs plus one decklist PR, eight combined. All 286 identities and required tokens now have one completion owner in the ledger; exact prerequisites are reconciled in the final plan. Shared engine owners and stacked integrations must preserve the two/three-workstream limit.
- **Proposed sequence:** content-authority deletion followed by airbend/exact
  zone permissions is the recommended first implementation investment. The parent operator launched T1 **892c47** and D1 **f871a8**; this research run
  starts no duplicates and declares no upstream dependency delivered.
- **Resolution dependency:** MTG-177's w15 draft needs acceptance. Its shared
  planning mirror is stale and earlier PR354 being merged proves only the
  earlier repair. Do not duplicate its target-resolution work.
- **Evidence limit:** full official inventory is verified; all face obligations
  were inspected, but a fresh official Oracle-text/independent clause audit
  is not complete. Scryfall is a pinned secondary transcription. Existing tests
  show bounded behavior, not complete card support across this larger world.
- **Operational limits:** no full-file latency/throughput target exists. Choose
  representative real consumer fixtures and budgets when schemas are ready;
  overflow must be explicit, never action omission. Rough effort is 28–50
  engineer-weeks, with high interaction uncertainty.

Unrelated inherited MTG-207 page-performance notes are preserved in git at
`c5d406e3d2a701525c15c8500a30f7da7896cc40:scratch/questions.md`; they do not direct
this Task. Historical audit findings remain under `scratch/reference/rules-audit/`.
