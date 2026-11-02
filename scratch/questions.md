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
