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
  gives the pytest summary. CI names test directories in
  `.github/workflows/manabot-integration.yml` and `academy-integration.yml`;
  `tests/infra/test_ci_selection.py` fails when a `tests/**/test_*.py` file is
  outside every workflow.
- CI has no R2 secrets, so `experiment_data`-marked tests skip there. A test
  that passes locally can still fail CI because the bytes already sit in
  `experiments/data/`. Reproduce CI by moving the named paths aside and running
  with `MANABOT_DATA_CACHE=<empty .runs dir>`: the test must report `SKIPPED
  ... needs experiment data`, not error. Restore the files afterwards.
- Job selection: drive `.github/ci/select-tests.sh` as `ci.yml` does, with
  NUL-delimited paths on stdin:
  `printf '%s\0' tests/belief/test_tracker.py | bash .github/ci/select-tests.sh | grep =true`.
  Empty input selects every job; `--only <job>` rejects an unknown name with
  exit 1. `test_all_test_files_are_covered_by_ci_workflows` checks that each
  test file's selected jobs include a workflow that runs it.
