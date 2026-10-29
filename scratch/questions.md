# f31c9c skip forced decisions — assumptions, 2026-10-28

Direction (Jack Heart, 2026-10-28): native skipping on in every runner,
including Academy and the action benchmarks; only explicit debugging may turn
it off. MTG-193's landed notes are at `6d069d6e5:scratch/questions.md`.

- **Settled (concept review, 2026-10-28) — the two verification fixtures are
  debugging use:** `manabot/verify/action_cases.py` and
  `manabot/verify/text_observation.py` keep `skip_trivial=False` as scripted
  native-window walkers: they drive both seats through constructed positions
  one native window at a time. Nothing they hand on is a forced decision: every
  `curriculum()` case offers a policy 2–3 candidates (attack/target/mode/
  ability/land 2, learn 3; 48 cases), and the text fixtures answer with a
  response stub, no model loaded. Neither is a default nor a user path, so Jack
  Heart's rule ("never ... any user use skip_trivial=false except debugging";
  "no code paths where its off by default") already covers them; this is
  applying his rule, not a product choice left open. Evidence: flipping both to
  `True` broke their walkers (`reach`/`_choose` no longer stop where the
  scripts expect; `action_cases.py:210`, `text_observation.py:292`, 9+
  identity/text tests), so flipping would mean rebuilding the fixtures and
  bumping `CASE_SET` with no effect on any person, model or policy. `AGENTS.md`
  states the debugging-only rule and points at the allowlist. Optional polish:
  `DIAGNOSTIC_EXCEPTIONS` and the test docstring in
  `tests/env/test_skip_forced_decisions.py` could say "debugging fixtures" to
  match `AGENTS.md`. Test-only constructed positions under `tests/` are outside
  the scan.
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
- **Priority automation is logged as automation:** with forced steps gone,
  `Auto:` log lines come from every automated priority pass (previously only
  single-action steps), per the rule that configured automation stays
  identifiable. `test_priority_automation_is_logged` pins it on the Elemental
  seed-0 deal the `deck-browser` spec uses.
- **Academy UI CI repaired (2026-10-28):** `trigger-order` no longer clicks the
  now-forced Deadly Dispute sacrifice; `decision-surface` captures the Command
  before reloading. Local release suite 24 passed, unit 107 passed, check clean
  (Node 24 via nvm; Homebrew node is broken on this machine).
- **Pre-existing local failures (also on base 6d069d6e5):** 8 Academy receipt /
  world-pin tests (`authored_match_*`, `public_commitment_parity`,
  `ur_lessons_increment`); `tests/llm/*` need `huggingface_hub`.
- Check (2026-10-28 realign): `uv run --extra dev pytest tests/env/test_skip_forced_decisions.py -q` — 4 passed in 0.88s.
