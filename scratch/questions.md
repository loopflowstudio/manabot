# f31c9c skip forced decisions — assumptions, 2026-10-28

Direction (Jack Heart, 2026-10-28): native skipping on in every runner,
including Academy and the action benchmarks; only explicit debugging may turn
it off. MTG-193's landed notes are at `6d069d6e5:scratch/questions.md`.

- **Assumption — named diagnostic exceptions:** `manabot/verify/action_cases.py`
  and `manabot/verify/text_observation.py` keep `skip_trivial=False`. They
  build constructed positions that must stop at sole-offer continuations
  (CASE_SET-pinned action cases; equal-state/distinct-history collision proof)
  and play no game for a model or policy. Flipping them changes CASE_SET and
  removes the collision. `tests/env/test_skip_forced_decisions.py` allowlists
  exactly these two. If Jack Heart wants them on too, rebuild those fixtures
  and bump `CASE_SET`.
- **Academy demos regenerated, not migrated:** current-world fixtures
  `precision-demo-w15.json`, `precision-mana-demo-w15.json` and the new
  `learn-demo-w15.json` were rebuilt by replaying the explicit prefixes and
  aligning on turn/step and label (`scratch/regen_demos.py`). Explicit-mode
  `learn-demo-explicit-w15.json` and `choice-navigation-explicit-w15.json`
  are retained like earlier worlds' fixtures but no longer read. The Precision
  Treasure payment (activation, `Add {B}`, Pay) was entirely forced and is now
  automatic; tests assert the automated commit instead of the prompts.
- **Academy forced-step automation removed:** `_advance` and villain play no
  longer step single-action spaces (native skipping already did); the toggle
  reads **Use priority stops**. Old traces keep their recorded forced Auto steps.
- **Not run here:** Playwright specs (no `node_modules` in this checkout) —
  `decision-surface`, `learn`, `choice-navigation`, `deck-browser`, `stops`
  changed or depend on the new fixtures; leave to gate/CI.
- **Pre-existing local failures (also on base 6d069d6e5):** 8 Academy receipt /
  world-pin tests (`authored_match_*`, `public_commitment_parity`,
  `ur_lessons_increment`); `tests/llm/*` need `huggingface_hub`.
