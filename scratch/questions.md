# Questions and assumptions

Notes inherited from stacked parents that have since merged into main (D1
decklist import, T2, afbd6a, 558daa, T3 and its characteristics view, T4,
MTG-234, the 2026-10-30 Clean-machine Play and action-identity CI fixes) are
retained at `1e9fd2d7c:scratch/questions.md` (lines 5–472). None selects work
here; main carries their outcomes.

## T5 (a7234f) — 2026-10-30 operator choices

- Started from main; stacked on T3 (`3cef39b8e`, its local main merge, not
  yet pushed) per Jack Heart's design review.
- Surveil reuses the LookAndSelect surface (select = graveyard, decline =
  keep) instead of a new action type, which would change `ACTION_TYPE_DIM`.
  Kept cards stay in order: exact for surveil 1, the only printed amount.
- Joo Dee's fixture entry was added from Scryfall's card API with the refresh
  script's fields, not a full bulk refresh (which reorders the file).
- `world` reports 239 eligible definitions (T3's 236 plus Joo Dee, Ran and
  Shaw, Phoenix Fleet Airship).
- 2026-10-30 (T5 implementation, headless):
  - Copy exceptions reuse T3's `Modification` rather than a separate
    `CopyException` enum, so exceptions and layered records share one
    vocabulary and token id.
  - The protocol field is `copy_of`, because pydantic's `BaseModel.copy`
    shadows `copy`.
  - The Airship's static is a self-relative `StaticAbility::CreatureWithNamesakes`,
    not a conditional `StaticTypeGrant`, since `check_static_condition` has no
    source.
  - `/verify` was not run before commits: that Loopflow skill finishes the Task
    after merge, and T5 stops at demo.
- 2026-10-30 (demo iteration): `ContinuationFrame.decision` means different
  things per frame kind (obligation code for kind 6, trigger ordering class for
  7/8, placement group for 9). The LLM text renderer had read all of them
  through one obligation table, where slot 14 said "order triggers" although
  the engine's 14 is `SearchDistinctLands`. The renderer now labels each kind
  separately (`_frame_decision`), so the text for trigger frames changes
  wording. No retained fixture holds that text.

## T5 sync onto main — 2026-11-02 (headless)

- Merged with T3's head `a18902e0e` as the base (main's T3 is its squash) so
  only T5's own diff met the characteristics cache and T4.
- T5's `PlayerTurnHistory`, player column, `update_turn_history` and
  `StaticCondition::SacrificedThisTurnAtLeast` were dropped for T4's struct,
  column position (after attacked) and `Condition`; the Airship trigger uses
  T4's `intervening_if`. Decision codes keep main's numbering, Surveil is 22.
- Singular wording ("sacrificed 1 permanent", "drew 1 card") replaced main's
  test expectation "drew 1 cards"; no retained fixture holds that text.

## T5 gate at the green head — 2026-11-02 (headless)

- The supported-card count in `WORLDS.md` was derived by addition (main's
  263 plus three), but the engine counts 314 eligible definitions. Main's
  figure was already stale: T3 recorded 236, T4 added 75 cards and T5 three,
  which is the 314 `set_cards` reports. The gate corrected the branch's
  statements to the measured number and did not touch main's history.
- `/verify` was not run before the gate commit, as for earlier T5 commits:
  that Loopflow skill checks a merged change and finishes the Task, and
  shipping is stopped.

## T5 concept-review decisions 1a/2a — 2026-11-02 (headless)

- `PermanentData.power`, `keywords`, `is_animated` and the card row's
  layered fields now agree for a battlefield card (both read
  `Game::characteristics`). The permanent row keeps its columns: Python
  readers, the text renderer and the policy input index them, and the
  decision changed `CardData`, not `PermanentData`. `PermanentData.name` /
  `creature_types` stay the "set by an Aura" markers (fact rows), while the
  card row's `name` is the layered name.
- `Characteristics.types` is the layer-4 owner; `is_creature` and `is_land`
  are kept as flags derived from it so the many readers of one bool stay as
  they are. A static grant of the Creature type (none exists) would now make
  the permanent a creature through `types`, as `is_land` already did for
  Land.
- Ward in the view is the printed ward unless abilities are lost. Nothing
  grants ward, so there is no granted-ward layer yet; the trigger that
  enforces ward is already suppressed by ability loss.
- Text wording: "printed keywords" became "keywords" on the card line, since
  the flags are layered on the battlefield; "card power/toughness" keeps its
  name as the card line's label.
- Evidence that embeds card rows with active effects moves; implement left
  the re-records to gate, which owns the receipts.
