# 08: Layers + Continuous Effects v1

## Finish line

A practical first slice of continuous/layered effects is implemented (starting
with P/T-relevant effects), with explicit documentation of unsupported cases.

## Context from stage 06

Stage 06 added `temp_power: i32` and `temp_toughness: i32` on `Permanent` for
Shivan Dragon's firebreathing (+1/+0 until EOT). These are added to base P/T
when computing effective stats (combat damage, lethal checks) and cleared in
cleanup alongside damage. This is a primitive layer 7c implementation — the
proper layer framework should subsume it, replacing ad-hoc temp modifiers with
a structured continuous-effect system.

Activated ability infrastructure exists: `ActivatedAbilityDefinition` on
`CardDefinition` with `SelfGetsUntilEot { power_delta, toughness_delta }` effect
kind, `PriorityActivateAbility` action, and stack-based resolution. Giant Growth
can reuse most of this scaffolding (spell resolution applies the same effect kind).

## Changes

### Minimal Layer Framework

Introduce layer application for the subset needed by current cards. Start
with layer 7 (P/T modification) and its sublayers:
- 7a: Characteristic-defining abilities (e.g., Tarmogoyf)
- 7b: Set P/T to specific value
- 7c: Modify P/T (+X/+Y effects, anthem effects)
- 7d: Switch P/T

Other layers (1-6) documented as not-yet-supported with explicit notes on
what cards would require them.

### Continuous Effect Tracking

Effects from resolving spells/abilities lock their affected objects at
creation time. Effects from static abilities continuously recalculate.

"Until end of turn" effects tracked and removed during cleanup (already
partially implemented via damage clearing).

### Driver Cards

Add cards that exercise layer 7:
- Giant Growth (pump spell: +3/+3 until end of turn, layer 7c)
- A lord effect if triggers support it (e.g., "other creatures you control
  get +1/+1", layer 7c static)

### Trace Tests

- Pump spell modifies P/T, expires at cleanup
- Multiple pump effects stack correctly
- Layer ordering between set-P/T and modify-P/T
- Document unsupported layer interactions explicitly

## Done when

- Supported layer subset is deterministic and tested.
- Unsupported behavior is explicitly documented in coverage artifact.
- Training smoke test passes.
