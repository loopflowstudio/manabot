# 04: Triggered Abilities v1 (Man-o'-War first)

## Finish line

Triggered abilities exist with queueing + stack placement semantics, validated
with Man-o'-War ETB and focused ordering tests.

## Changes

- Implement trigger detection and pending-trigger queue.
- Put triggers onto stack at the correct timing boundaries.
- Start with ETB triggers; use Man-o'-War as first driver card.
- Add tests for trigger timing vs priority/SBA loop.

## Done when

- Man-o'-War ETB behavior is deterministic and CR-cited.
- Triggered ability tests for success + negative-path constraints pass.
