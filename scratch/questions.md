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

## Training consolidation defaults — 2026-10-22

The accepted consolidation left exact local defaults and obsolete preset handling
open. The implementation selects `local-smoke-v1` (two CPU updates, two streams,
16 decisions per stream, both seats, 120-second failure bound), using the baseline
model and damped objective through `execute_regime`. Explicit one-seat controls
retain exact quotas. Old training preset/override flags fail with migration
instructions because fixed-seat PPO and optional shaping cannot be translated
faithfully. Historical executables without current consumers are retired; their
results, pinned source and first-light report readers remain intact.
