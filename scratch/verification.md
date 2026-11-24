# Verified on main

The snapshot claims hold. The Academy browser failure has its own follow-up.

PR [#581](https://github.com/loopflowstudio/manabot/pull/581) merged as
`82c5d1d1b7cfe0827816228f49d75e2e147ce584`. After fetching `origin/main`,
`git rev-parse 'HEAD^{tree}' 'origin/main^{tree}'` printed
`c0bac3287462bf92ee4cc7151203501e9298aab6` twice. I used the Task checkout
to build those exact sources and an isolated Etude home under `.runs/`.

```sh
CARGO_BUILD_JOBS=2 UV_NO_SYNC=1 uv run --extra play maturin build --release \
  -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels
CARGO_BUILD_JOBS=2 cargo test --locked --manifest-path managym/Cargo.toml \
  --lib agent:: -- --test-threads 2
UV_NO_SYNC=1 uv run --extra dev --extra artifacts python -m pytest -q \
  tests/env/test_vector_snapshot_slots.py \
  tests/env/test_game_snapshot.py \
  tests/training/test_snapshot_rollouts.py \
  tests/training/test_training_game_journals.py \
  tests/training/test_game_errors.py::test_replay_resource_limit_on_a_restored_memory_boundary \
  tests/test_package_boundaries.py
```

Maturin built `managym-0.2.4-cp313-cp313-macosx_11_0_arm64.whl`. I
atomically installed its CPython 3.13 extension into the Task checkout.

- Rust printed `99 passed; 0 failed`. The construction counter verified that
  strict and isolated stepping park terminal slots without constructing or
  encoding a fresh game. Both viewers started a new incremental memory write
  on each scalar restore and search fork.
- Python printed `6 failed, 203 passed`. The same-slot sibling overwrite matched
  every dense scalar observation buffer and left the neighboring slot untouched.
  Four-worker CRN shuffle matched the scalar restore, shared immutable history,
  and diverged after different actions. A separate four-worker determinization
  probe printed `4 determinized siblings match scalar; shared Chunk, Tail and
  ChunkFeatures; 4 serving segments`. The tests also exercised rejected loads,
  forced-terminal starts, fresh seed and evidence accounting, and cold
  reproduction of a fork failure with the retained Command and intervention.
- The two-worker PLR and four-sibling training fixture captured roots with GRU
  decision memory, saved and resumed the bank, and completed the learner update.
  The continuation fixture resumed from the refunded fresh seed after a branch.
  The package boundary checks passed.
- The six failures were the parameterizations of
  `test_executing_replay_requires_active_fresh_step_binding`. They reached
  turn 0 during opening hands and failed the existing `turn >= 1` assertion.
  This reproduces the fixture drift tracked in [MTG-426](loopflow://manabot/task/MTG-426).

I extracted the two Python examples verbatim from `docs/engine-python.md`
and `docs/training-loop.md`. The engine example, run using
`uv run --project "$project_dir" python snapshot_example.py` with the
checkout on `PYTHONPATH`, printed:

```text
0 42 0 False
1 45 0 False
[42, 47]
```

The CPU training run used an isolated `ETUDE_HOME` and disabled W&B.

```sh
uv run etude train snapshot_recipe.py --experiment verify-snapshot \
  --config '{"workers": 4}' --until updates=4 --seconds 120 --json
uv run etude train snapshot_recipe.py --experiment verify-snapshot \
  --resume --until updates=5 --seconds 120 --json
```

Both commands returned `"outcome": "finished"`. Etude reported snapshot
starts and horizon cuts of `[0, 0, 4, 4, 4]` with zero failed branches.
The saved bank held 64 native recipes at update 4 and 80 at update 5, all
with `replay_json` in `managym-snapshot-v1`.

```sh
gh run list --branch main --commit 82c5d1d1b7cfe0827816228f49d75e2e147ce584 \
  --event push --json databaseId,workflowName,status,conclusion,url
gh run view 45326996399 --json status,conclusion,headSha,jobs
gh api --allow-escape-sequences repos/loopflowstudio/manabot/actions/jobs/138218947874/logs
```

The exact-merge [push run](https://github.com/loopflowstudio/manabot/actions/runs/45326996399)
reported `in_progress` at 04:39 UTC on November 24. Academy UI had failed,
with Clean-machine Play, Engine Budget and evaluation still pending. The
Academy job printed `1 failed, 3 skipped, 67 passed`. In
`opponent-beats.spec.ts:76` it expected `data-board="step"` and received
`"frame"`. The same UI job passed on the identical PR source tree with
`68 passed`. The test starts an unseeded game and needs that game to reach
a specific presentation state. Its deterministic fixture belongs to
[MTG-427](loopflow://manabot/task/MTG-427), filed with `lf task create` and
confirmed in the tracker. The required PR check separately passed in run
`45325039555`.
