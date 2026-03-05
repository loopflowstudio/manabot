# 03: Lightning Bolt (`any target`) ASAP

## Finish line

Lightning Bolt is implemented and tested as `any target` semantics with current
legal targets (players/creatures now), and it validates real instant-speed
interaction on the stack.

## Changes

- Add Lightning Bolt to card registry.
- Add target selection plumbing for noncombat damage spells.
- Keep target model extensible for planeswalkers/battles later.
- Add CR-cited tests for:
  - legal/illegal targets
  - stack resolution ordering
  - lethal player/creature outcomes
  - negative-path illegal targeting

## Done when

- Bolt can be cast in response windows.
- Rule tests covering damage/targeting edge cases pass.
