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

# T6 (MTG-225) operator choices — 2026-11-02, headless

- Base: developed on main `4c8324a59` (T3+T4) while T5 #405 was mid-sync;
  T5's synced head is merged when it publishes, so the stack is kept.
- A double-faced card is one definition function; `back_face` holds the back
  face, which the pack registers as its own row right after the front.
  `back_face` serializes as the back face's name (a definition reference in
  tokens and digests), so a back face carries no `key`/`scryfall_id` of its
  own; pool rows mark it "not a deck entry".
- Back-face colors follow Scryfall's face data: no color indicator except
  Fire Lord Sozin (black), so Avatar Kuruk/Kyoshi/Roku/Yangchen and the Aang
  back faces are colorless here. Flag for review against the printed cards.
- Entering with a Saga's lore counter emits `CountersPut` (so chapter I
  triggers, CR 714.2b/122.6); other entry counters still emit nothing.
- Extra turns are observed as a per-player count (player column
  `PLAYER_EXTRA_TURNS_INDEX`, player width +1); the resume order is not a
  column.
- A transformed permanent's mana value is its front face's (CR 712.8e),
  read by `Card::mana_value_cost` for predicates and targets.
- Airbending a spell exiles it from the stack with the {2} recast grant
  (Aang, Swift Savior's "creature or spell").
- Copies of double-faced permanents follow CR 707.8a (post-2021): the token
  has both faces and is transformed if the original is, so it can transform.
- Foggy Swamp Visions' "X target creature cards" offers up to X targets
  (`max_x`), not exactly X; "your next end step" uses the next end step, the
  same for a sorcery. Its heuristic polarity is pinned "harmful" (exile).
- Additional combat phases are a stack on `TurnState` observed as a count on
  the active player's row; a pending phase's attacker limit is visible only
  once it applies (attack offers).
- Yangchen II's "you may have target opponent draw three" is a resolution
  `Modal` (draw both / nothing) inside its target clause, not a new variant.
- The Rise of Sozin II: the card name is a resolution `Modal`, one mode per
  deck-eligible name in the world (front faces, no tokens or fixtures). The
  search exiles as many as it finds up to four (graveyard, hand, library)
  rather than offering to find fewer.
- Fire Lord Sozin: targets are chosen as the trigger goes on the stack and X
  is then their total mana value, paid or declined at resolution — the
  reverse of "pay {X}, when you do choose targets", with the same outcomes.
- Bending events: "you waterbend" fires once per paid waterbend cost (spell,
  activation or resolution payment), "you firebend" once per resolved
  until-end-of-combat mana (firebending's only source), "you earthbend" and
  "you airbend" once per resolved instruction. The turn history keeps the four
  kinds as bits (`PlayerTurnHistory.bent`, four player columns).
- Koh: "has all activated and triggered abilities of the last chosen card" is
  a `HasAbilitiesOfSource` continuous-effect record whose source is the chosen
  definition (latest record wins). Inherited activated abilities follow the
  printed ones; an inherited activation resolves from the record while Koh
  remains, and inherited leaves-the-battlefield triggers do not fire as Koh
  leaves. "You may exile it" is a resolution `Modal`.
