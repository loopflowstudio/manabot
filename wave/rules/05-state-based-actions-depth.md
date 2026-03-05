# 05: State-Based Actions Depth

## Finish line

SBA handling is expanded and reliably integrated with trigger/priority loops,
including simultaneous-loss edge cases.

## Changes

- Expand SBA checks beyond current subset where needed by added mechanics.
- Add edge-case tests for simultaneous terminal conditions.
- Verify repeated SBA checks and interaction with queued triggers.

## Done when

- SBA loop behavior matches documented supported CR subset.
- Edge-case SBA tests pass and are rule-cited.
