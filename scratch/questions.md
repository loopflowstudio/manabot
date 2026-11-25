# MTG-483 questions and assumptions

## Assumptions taken in kickoff and concept-review (2026-11-25, nobody present)

- **Unstacked from MTG-431 onto main** (`lf sync main`, merge `e2224c3988`).
  The pile split exists only on MTG-431; it becomes a follow-up group kind.
  Undo: `lf sync MTG-431`.
- During that sync, `docs/academy.md` was restored to main's text (the earlier
  "Drafted decisions" section described the abandoned chunked runner and the
  MTG-431 pile split; both are in git history at `2f3d2833a4`).
- **Live `group` travels beside `frame`**, not inside `ExperienceFrame`:
  adding a field to `ExperienceFrame` would move `frame_hash` for every combat
  frame and break retained replays. `protocol/experience-v1.schema.json` stays
  untouched, while `protocol/testing-house-v1.schema.json` + its fixture add
  the `group_command` WebSocket request (`test_shared_table_protocol.py`
  enforces exact schema equality).
- **Built on `combat_preview`'s clone-and-step mechanism**, not on
  `compound_offers` / `step_structured`, which stays untouched.
- **Tagged `GroupSpec` / `GroupPlan` / `GroupOutcome`** inside the shared
  `Group` / `GroupPreview` envelope (chosen in concept-review over a flat
  `Vec<Choice>` so `OrderLibrary` carries `end: Top | Bottom`, every kind
  carries already-committed choices, `GroupOutcome` serves all kinds, and
  future `Pick`/`Distribute` kinds extend cleanly).
- **Domain activation thresholds** (`>= 1` declarable creature for combat;
  `>= 2` remaining cards for Scry, Surveil, OrderLibrary) rather than
  `steps > 1`, so 1-creature combat preserves MTG-441 drafting/drag and 2-card
  `OrderLibrary` after Surveil 2 stays in `ArrangePanel` under `skip_trivial`.
- **First non-combat kinds:** Scry, Surveil, OrderLibrary (one arrange panel).
  Surveil starts in "Keep on top" with an explicit dynamic Confirm label; Scry
  and OrderLibrary start unplaced with Confirm disabled until all remaining
  cards are placed.
- **Scry `bottom` plan order** is top-to-bottom of the bottom group (matching
  `OrderLibrary::Bottom` and human reading order); the Scry compiler emits
  `ScryCard(to_bottom=true)` in reverse so `decision.rs:2806` commits that
  exact order.
- **Apply is all-or-nothing;** an incomplete plan is rejected without mutating
  the game.
- **`Env.combat_preview_json` and `managym/combat.py` stay** as
  `0.3.0`-deprecated wrappers because they shipped in 0.2.6.
- **Per-step recorded command ids** use `authority.group.<id>.<n>` with
  `source: "client"` (collision-free because client command ids cannot start
  with `authority.`, and no field is added to `ReplayDecision`).
- **Inter-step purity boundary** checks between step `i` and step `i+1` (only
  while `continues()` is true): no `CardDrawn`/`CardRevealed` in `state.events`,
  unchanged `state.rng`, and no new hidden-zone `ObjectId` visible to the
  viewer. Post-group draws (e.g., Preordain) happen after `!continues()` and
  complete cleanly.

## Open

- See numbered review questions (1a–4b) in `scratch/concept-review.html`.

## Follow-ups for ship-review to file or drop

- **Pile split as a group kind** once MTG-431 lands: `PartitionPiles` in
  `flow/group/`, its panel on `group_command`, and removal of MTG-431's
  `partition.ts` send loop and `FACE_UP_RE`. Includes the `ChoosePile`
  opponent-view check, which has no UI test.
- **"Pick" kind and panel:** SelectCards with max > 1, SearchDistinctNames,
  Discard of 2+, OpponentSplitsRevealed, KeepCreatures.
- **"Amounts" kind and panel:** DistributeCounters, DivideAmongTargets,
  AssignCombatDamage.
- **OrderTriggers** as a group only while no trigger asks a mode or target
  between picks.
- The `discard` instruction text says "or decline when offered" though no
  decline is offered (`managym/src/decision_context.rs:513`).
- Remove the `combat_preview` wrappers with the other `0.3.0` graphs.
