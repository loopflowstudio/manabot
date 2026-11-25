# MTG-443 questions and assumptions

Choices made with nobody present during concept review (see `scratch/concept-review.html`); each is reversible.

- **Effective details via `decision_surface_view` (`include_details=True`), not `Observation` or `CardFace`**:
  Play, Watch replays (`server.py:1585`), StepBoards (`step_boards.py`), and Study (`study_branch.py`) all use `decision_surface_view`. Computing `keyword_abilities`, `attacks_despite_defender`, and `restrictions` from `Game::characteristics` and shared `Game` rules helpers via `env.permanent_details_json(viewer)` gives one CR 613 Layer 6 authority with zero RL tensor bloat (`PERMANENT_DIM` unchanged), zero frozen-evidence hash risk (`serialize_observation` default unchanged), and fixes token name collisions (`Soldier`, `Dragon`), Sokka's `TriggeredGrant(Prowess)`, Fire Nation Cadets' conditional Firebending 2, granted Ward, and static restrictions (`DoesntUntap`, `CantAttackOrBlock`, `CantBeBlockedIf`, `CantBlock`, `AttacksEachCombat`).
  Passed over: widening `PermanentData` / `PERMANENT_DIM` with a derived condition column, and adding name-keyed `CardFace.abilities`.
- **Attacking and Blocking chips removed from battlefield cards**: lean, tilt, and block line already carry them; both stay in Inspect and `aria-label`. `first_strike_participant` stays as a concise `"Struck first"` chip during the combat-damage window.
- **Hand/Stack Inspect keeps evergreen keywords + rules text**: `CardFace` stays untouched; rules text already names Prowess, Ward, and Firebending.
- **Abilities stay words; only summoning sickness becomes a 16×16 spiral mark**: matches Sepia Etude's typographic voice.

- Firebending X retains the printed X. Its mana depends on the board when its attack trigger resolves. Project one row per effective trigger so repeated abilities and distinct Ward costs survive.

- Regenerating the advice schema with its documented Pydantic exporter also picked up existing fields such as payment context. Keep the generated file reproducible.
- Warm local CPython 3.13 release build, 40 battlefield permanents, median of five batches: native details JSON 0.003 ms/call (100 calls per batch), full Academy projection 0.745 ms/call (10 calls per batch).

- A turned card with a two-line name can have less than 16 px of art above its power/toughness plate. Put the overflow count beside its name in that case.
- Use the resolved effect timestamp for provenance on a granted trigger or evergreen keyword. The record already carries the source and duration. Current evergreen rows include an empty grant map when all keywords are the card’s own.
