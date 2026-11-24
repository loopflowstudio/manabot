Read when CI fails, a test fails locally, or the change touches CI or test timing (`.github/`).

- Tests that fail locally may fail on the base too. Classify them by running
  the base tree's own tests with this checkout's interpreter: from the base
  worktree, `PYTHONPATH=<worktree> <checkout>/.venv/bin/python -m pytest <files>`
  (print `managym.__file__` and `manabot.__file__` first). Viewer JSON changes
  need a per-decision base comparison of `semantic_observation_json(v)` for
  both seats and `search_witness_json()`: these bytes bind Academy frames,
  decision captures and search witnesses, and retained fixtures rarely cover
  live history.
- CI workflow or test-timing changes: the user surface is hosted CI on the
  pushed branch: `gh workflow run ci.yml --ref <branch>`. Read job and step
  timestamps with `gh run view <id> --json jobs`; read logs with
  `gh run view <id> --job <job-id> --log`. Compare install time with peers in
  the same run and inspect dependency logs for unexpected downloads.
  See `docs/development.md` (Job time) for the budget.
- PR CI failures while the run is still in progress: `gh run view --log-failed`
  refuses; read `gh api --allow-escape-sequences
  repos/loopflowstudio/manabot/actions/jobs/<job-id>/logs` and grep `^FAILED`.
  Check the PR's checks before QA ends; tests elsewhere in the tree can still
  assert the behavior the change replaced.
- Once a CI run has finished, `gh run view <run> --job <job-id> --log-failed`
  gives the pytest summary. CI names its test files in
  `.github/workflows/manabot-integration.yml`; `tests/sim/test_puct_compatibility.py`,
  `tests/sim/test_conditional_search.py` and `tests/verify/test_learning_regression.py`
  are not among them, and the first two fail on main (2026-11-18).
