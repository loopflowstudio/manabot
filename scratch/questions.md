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

Pass 2 corrections:

- The `cast_permission` assertion in `tests/semantic/test_player_control.py` reads a serialized projection. It continues to test the single-grant wire field. Only the native card-name lookup in the same test file needed migration.
- `docs/development.md` identifies `manabot_slow_step.py` as the exact reported game. Its verified SHA-256 is `42c6bec5f6a94fbb189ef40a92c7f197489e75756d3e813eaff063bd996596fa`. Its historical constructor call stays in the fixture.
- The Temur fixture hit the opening Modal decision before it could inspect the legendary flag. The test now uses the established first-main fixture with a ready Hermitic Herbalist and reads Iroh on his permanent row.

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

Pass 3 corrections:

- Test the top-level import in the installed distribution. The checkout retains the frontend under `academy/ui` and Python treats that directory tree as a namespace.
- Hosted play copied `academy/__init__.py` explicitly in its Dockerfile. Remove that copy with the shim.
- The historical spelling in `learning_schema.json` is part of `LearningSchema.schema_hash`; preserve the schema bytes.
- Comparing the authored-match parity receipt to a fresh run showed exactly 350 changed viewer-state hashes and 350 changed semantic-observation hashes on each of live, headless and replay, plus source identity. The 174 Commands, all state witnesses, ordered events, presentation and stale rejection matched. Follow the existing separate-receipt convention for this wire change.
