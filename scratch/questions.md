# Open questions / assumptions

- Implemented the event-observation wave against the current engine surface, which does not yet have dedicated Lightning Bolt / Counterspell spell-resolution paths. `SpellCountered` remains in the stable schema but is not emitted by current gameplay paths.
- `Observation.recent_events` currently preserves the full decision-window event list; truncation to `max_events` happens in the Rust/Python encoders rather than inside `Observation::new`. This keeps encoder configs flexible while preserving the most recent `max_events` events in encoded tensors.
- Event ids in `recent_events` use the engine's typed id namespaces (`CardId`, `PermanentId`, `PlayerId`) keyed by `source_kind` / `target_kind`, rather than always converting to `ObjectId`. This avoids losing permanent identity after zone changes but is a slight divergence from the original design doc wording.
