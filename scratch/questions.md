# MTG-434 questions and assumptions

Assumptions taken in the plan (all reversible before merge):

- Observation JSON writes step/phase as the public enum names
  (`COMBAT_FIRST_STRIKE_DAMAGE`), not serde variant names.
- `Card`'s flat characteristic accessors are removed with `Permanent`'s; the
  brief names only `PermanentData` / `SpellData`.
- `enable_behavior_tracking` and `opponent_policy` are also removed from the
  `manabot.env` wrappers that only forwarded them, although manabot shims
  belong to MTG-435.
- The JSON/protocol key `cast_permission` (single-grant card wire) stays; only
  the Python `Card.cast_permission` accessor is removed.
- `experiments/mana_taps.py` and `experiments/vector_statistics_cost.py` are
  edited to the replacement spellings (`"w17"` literal, `RecordedVectorEnv`).
- Decision-memory capture (`decision_memory.rs:1113-1114`) shares the frozen
  12-step policy index (`0..=11`, folding `FirstStrikeCombatDamage` onto 7,
  `Cleanup` = 11), keeping `managym/decision_memory.py` (`le=11`),
  `manabot/policy/decision_memory.py` (`0..=11`, `step / 12`), and stored
  memory checkpoints untouched.
- `RecordedGame.deal_seed` is removed alongside `leg`, `game_trace_sha256`, and
  `decks`; `scripts/release/prepare.py` and `managym/pyproject.toml` drop their
  vestigial `academy/` staging/excludes.

- Four frozen pending-selection captures predate permanent object rows. Compare
  the writer bytes before the existing object upgrade, then compare the public
  parse results of the integer and named captures after the upgrade.

Ideas outside this Task:

- Rename `Permanent.name` (effect-set name override) to `name_override` with a
  wire/API deprecation cycle so it does not collide conceptually with
  `Permanent.characteristics.name`.
- Distinguish first-strike combat damage in `DecisionContext` (instead of
  folding onto 7 in linear `step / 12`) and render step/phase names in
  `text_observation.py:1495-1496`.
- pytest has no `filterwarnings`; deprecated use inside the repo is silent.
  Erroring on our own `DeprecationWarning`s would catch stragglers before the
  next removal release.
- MTG-294 is superseded by this Task and can be closed when it merges.
