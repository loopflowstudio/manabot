# Open questions / assumptions

- Verification rollout/evaluation loops now catch `PyAgentError` from `env.step(action)` and
  try action `0` once before treating the game as aborted. Assumption: continuing the run with
  a counted non-win is better than crashing the whole verification script.
- `build_hypers(match={...})` now treats `hero_deck`/`villain_deck` overrides as full replacements
  (not deep merges) so step 2 can truly run Mountain-only decks.
- Running `python -m manabot.verify.step0_env_sanity` currently fails hard criteria:
  heavy card/permanent truncation and random-vs-random hero win rate far from 50%. This
  appears to indicate either aggressive observation caps or action-selection robustness issues.
