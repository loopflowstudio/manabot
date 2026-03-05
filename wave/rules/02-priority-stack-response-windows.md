# 02: Priority/Stack Response Windows

## Finish line

Priority and stack behavior supports real response windows (not just own-turn
sorcery speed), with focused tests for pass/resolve/advance semantics.

## Changes

- Align priority sequencing with CR 117 and stack resolution with CR 405.
- Allow instant-speed casting windows where legal.
- Add tests for:
  - active/nonactive pass flow
  - all-pass resolves top object
  - empty stack advances phase/step
  - post-resolution priority handoff

## Done when

- Response-window stack interactions are testable and deterministic.
- CR-cited tests for core 117/405 behaviors pass.
