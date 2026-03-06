# Event Observations — Validation

## Verify events appear in observations

```bash
cargo test --manifest-path managym/Cargo.toml event_observations
```

Expected: 4 tests pass:
- `observation_event_window_resets_after_drain`
- `spell_events_are_observable_for_creature_spells`
- `combat_damage_emits_damage_and_life_change_events`
- `triggered_abilities_emit_observable_trigger_events`

## Verify encoders match

```bash
pytest tests/env/test_observation.py -v -k event
pytest tests/verify/test_observation_parity.py -v
```

## Verify clippy clean

```bash
cargo clippy --manifest-path managym/Cargo.toml --all-targets --all-features -- -D warnings
```

## Schema contract

`EventType` and `EventEntityKind` discriminants are `#[repr(i32)]` with
stable values. Any new variant must be appended (not inserted) to
preserve the contract between Rust and Python encoders.
