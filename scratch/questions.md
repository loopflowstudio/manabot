# MTG-176 assumptions

Inherited MTG-174 sync/research notes are retained at
`95ca2059ea443351cabd4e8c322991c36c0d8212:scratch/questions.md`.
They do not select current work or authorize paid execution.

# Material assumptions

- MTG-176 supplies the accepted recovery brief; its design is in
  `scratch/game-crash-recovery.md`.
- Window/warmup 100/100 are prospective engineering defaults, not numbers chosen
  by Jack Heart. Game error tolerance is opt-in; complete-game buffering is used
  even for its one-seat control, avoiding claims of historical unlearning.
- Native setup/reset failures and buffer/configuration failures fail closed;
  only settled gameplay errors/truncations are eligible for regather. External
  run/allocation deadlines remain hard stops, not replacement-game allowances.
- The supplied Task names MTG-175's seed but does not include its 34-action
  prefix. The combined regression uses a seeded native Mountain/Gray Ogre setup and
  a narrow fixture payment arithmetic defect/repair shared across capture, replay
  and restart. It does not reproduce or fix that original game.
- Automatic evidence is independent of opt-in tolerance. Strict paths retain
  exceptions and recipe identities; no recovery is implicitly enabled. Standalone
  consumers without an owning output scope write private `.runs/game-failures/`
  files and attach the path to the exception. This is the simpler local default,
  not a new storage service or public replay surface.
- A strict vector exception leaves other active rows' execution outcomes unknown.
  Retain their pre-call inputs for replay without inventing completed/crashed counts.
  Positional/pool hard aborts or hangs still need external process supervision.
- Concept review, 2026-10-22: replay evidence formats and existing CLI entry points
  stay separate. Give their binding/error checks consistent meaning; older missing
  optional metadata is explicitly unavailable rather than assumed to match. This
  is the simpler compatibility-preserving interpretation of reproducible evidence,
  not approval to rewrite historical receipts or relax restart admission. The implemented
  clarification is recorded in `scratch/game-crash-recovery.md`.
- Replay diagnostics use null for binding/error checks unavailable or not reached;
  the boundary flag remains false until execution there is attempted. Arena keeps
  its original failure label; automatic vector results retain the batch outcome
  at the top level and mark individual streams' original outcomes unknown.
  A strict batch exception does not identify which streams originally advanced.
  Rust/PyO3 exception-name equivalence is normalized only for comparison.


## MTG-176 sync recovery — 2026-10-22

- Restored code, tests and design from stash
  `d815bbfc94e03881681ca30283be0a4b3ccfea8d` after Loopflow sync reset the checkout
  to `95ca2059e`. Its untracked `scratch/questions.md` collided with main's tracked
  notes; the historical notes remain in the git reference above. The original
  stash is unchanged.
- The original proposed Wave memory remains in the stash. The 2026-10-22 realign
  step in the active pursue Flow reconciles Intelligence memory with delivered
  strict capture and the bounded remote integration proof. Integration
  uses the delivered native uniform sampler and records actual sampled setups.

- The remote accounting proof uses fake provider effects and CPU placement for
  the exact local recovery recipe. RunPod's production CUDA recovery rejection
  remains authoritative. Attempt limits here mean the existing permanent provider
  claim, one recovery child per parent, and no checkpoint-evaluation retries within
  the retained cumulative allowance; no new numeric process-retry policy is added.

# MTG-182 assumptions (2026-10-23)

- Any exception raised by the run's own supervision inside collection (deadline,
  wall allowance, cancellation, process memory limit) settles in-progress games
  as run interruptions: `deadline` for TimeoutError, `abandoned` otherwise. They
  count in `attempted`/`abandoned`/`abandoned_samples`, never in the error window,
  `crashed`, `discarded_samples` or replacement debt. Strict-path failure capture
  remains TimeoutError-only, as before.
- Existing `deadline` receipts in parent ledgers (e.g. MTG-181) now settle as
  abandoned on resume. Frozen run.json states are not rewritten.
- The report names a paused run's stop reason `allocation_deadline` and a
  TimeoutError-interrupted run's `deadline`; a real `game_error_rate_exceeded`
  still wins.
- Status (realign, 2026-10-23): all four Task outcomes implemented in
  `game_errors.py`, `collection.py` and `reports/experiment.py`;
  `docs/game-crash-recovery.md` updated to match. A KeyboardInterrupt
  (BaseException) inside collection still settles nothing, as before.
  Remaining: PR #359 CI and landing.
- Check: `uv run --extra dev pytest tests/training/test_game_errors.py tests/training/test_experiment_report.py tests/training/test_allocation.py -q` → 44 passed.
- Concept review (2026-10-23): no change. "Run interruption" is one concept
  with two receipt labels (`deadline`, `abandoned`) and one counter
  (`abandoned`), matching the brief's explicit outcomes. The broadened `except`
  wraps only `check()` and the deadline test, so engine/agent failures cannot
  reach it. `paused` is set only when the allocation ends
  (`training/execution.py`), so `allocation_deadline` is accurate. No open
  product choice.
