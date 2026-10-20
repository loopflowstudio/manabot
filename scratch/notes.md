# MTG-154 evidence notes (2026-10-20, macOS arm64, heavily loaded machine)

Scripts: scratch/sweep.py (gate fit, any seeds), scratch/real_states.py (real decisions), scratch/proto.py.

## Probe sweep, seeds 1-300, gate Config
- additive: 78/300 fail; 55 seeds stuck on attack/mode/ability (attack 34, mode 25, ability 25); held-out Learn 28.
- FiLM (zero init): 31/300 fail; 0 stuck; held-out Learn 29, held-out target 2.
- zero-init bilinear (rejected): 0 stuck but held-out Learn 18/100.
- FiLM + identity bug (selector/declared columns zeroed): 30/30 seeds fail -> gate still sensitive.

## Real authored decisions (choice_corpus rounds 0-2, 96 decisions, 10 families, life 5 vs 35)
- gate size additive: 1/30 seeds has one stuck decision; 29/30 fit all.
- gate size FiLM: 0/30 stuck; 30/30 fit all.
- production baseline spec: see real-prod-*.txt

## Strict gate (capacity layer) seeds 17/29/43 with FiLM
- main: 144/144 pass. PR #322 (CARD_DEFINITION_COUNT 64): 144/144 pass.
- PPO control timed out locally both times (load avg ~60 from other sessions; PPO agent unaffected by this change).

## Status (realign 2026-10-20)
- Done, committed (490b6181): FiLM policy, gate protocol learnability-2.
- Done, uncommitted: FiLM default for new runs (`new_agent_spec`, baseline-v2),
  `build_hypers` pinned additive, compound+FiLM error names the fix.
- Remaining: commit; PPO control + 30-seed claim confirmed in CI; publish PR
  (PR #322 waits). Gate on PR #322's branch passed locally (144/144).
- Check: `uv run pytest -q tests/model/test_policy_film.py tests/training/test_experiments.py tests/verify/test_first_light.py tests/training/test_belief_stages.py tests/config/test_load.py` -> 47 passed.
