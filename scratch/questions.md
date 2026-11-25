# MTG-483 questions and assumptions

## Assumptions taken in kickoff after the grouped-input pivot (2026-11-25, nobody present)

- **Unstacked from MTG-431 onto main** (`lf sync main`, merge `e2224c3988`).
  The pile split exists only on MTG-431; it becomes a follow-up group kind.
  Undo: `lf sync MTG-431`.
- During that sync: `docs/academy.md` was restored to main's text (the earlier
  "Drafted decisions" section described the chunked runner and the MTG-431
  pile split), and `scratch/concept-review.html` was deleted as a review of
  the abandoned design. Both are in git history at `2f3d2833a4`.
- The group descriptor travels beside the frame on live messages, not inside
  `ExperienceFrame`: a frame field would move `frame_hash` for every combat
  frame and break retained replays. The direction said "on `ExperienceFrame` /
  Python API"; this reads that as "delivered with the frame".
- Built on `combat_preview`'s clone-and-step mechanism, not on
  `compound_offers` / `step_structured`, which stays untouched.
- First non-combat kinds: Scry, Surveil, OrderLibrary (one arrange panel).
- Apply is all-or-nothing; an incomplete plan is rejected.
- `Env.combat_preview_json` and `managym/combat.py` stay as `0.3.0`-deprecated
  wrappers because they shipped in 0.2.6.
- Per-step recorded command ids are derived from the group's (`<id>.<n>`) with
  `source: "client"`; no marker field is added to `ReplayDecision`.

## Open

- Pass 3 waits for MTG-441 (PR #613) on main. As of 2026-11-25 it has
  auto-merge on with `Academy UI` failing.
- Should a recorded step say it came from a group? Nothing reads that today,
  and adding it would change the replay schema.

## Follow-ups for ship-review to file or drop

- **Pile split as a group kind** once MTG-431 lands: `PartitionPiles` in
  `flow/group/`, its panel on `group_command`, and removal of MTG-431's
  `partition.ts` send loop and `FACE_UP_RE`. Includes the `ChoosePile`
  opponent-view check, which has no UI test.
- **"Pick" kind and panel:** SelectCards with max > 1, SearchDistinctNames,
  Discard of 2+, OpponentSplitsRevealed, KeepPermanents.
- **"Amounts" kind and panel:** DistributeCounters, DivideAmongTargets,
  AssignCombatDamage.
- **OrderTriggers** as a group only while no trigger asks a mode or target
  between picks.
- The `discard` instruction text says "or decline when offered" though no
  decline is offered (seen on the MTG-431 branch in
  `managym/src/decision_context.rs`; check main).
- Remove the `combat_preview` wrappers with the other `0.3.0` graphs.
