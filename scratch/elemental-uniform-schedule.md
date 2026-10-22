# Historical uniform ten-deck sampling plan — 2026-10-21

The one-seat constraints below describe the frozen experiment, not current main's
both-seat default. Current delivery is reconciled in `elemental-mana-repair.md`;
the paid attempt's failed outcome is `experiments/elemental-uniform-baseline.md`.

Jack Heart's accepted outcome remains one actual ten-deck Elemental self-play training run under the inclusive $10 cap, an admitted checkpoint, bounded reproducible evaluation, and a published report against retained MTG-159 evidence. MTG-164 is completed: PR337 merged as6d7fee869d8210669ebbe152a87d33944603f635. Do not replay its delivery or create another Task. The first ten-dollar-baseline Flow ended after its router correctly found no ten-deck collector schedule. That missing executable capability is work to implement, not a need for new budget or user judgment.

## Accepted constraints

Read scratch/elemental-ten-dollar.md and elemental-baseline-preflight.md. Preserve all ten exact 40-card setups and outside-game Lesson pools, one learner seat per stream, the actual baseline64x2 policy with its historical film setting,256 streams and64 learner transitions/update where feasible, and the retained learning recipe. Do not quietly change batch size, add two-seat training, adaptive sampling, representation/PPO tuning, or a hard-coded world switch. Do not spend during implementation. Existing preflight baseline numbers are historical and have different cost/world boundaries; no direct cross-world Elo/checkpoint opponent claim.

## Required scheduling behavior

Implement a minimal explicit uniform sampling mode for the complete elemental-world-v1 authored roster: independently sample an ordered deck pair from all100 pairs, including mirrors, at each new game/root. Both seats come from exact native authored setups including Lesson pools. Keep256 streams and their existing fixed learner seat s%2; sample independently of that seat and report realized deck/pair/learner-seat transition and terminal-game exposure rather than claiming exact finite-sample balance. Every pair has probability1/100; no deterministic fixed pair-only run or permanently skewed initial stream allocation masquerading as uniform sampling. Match length can skew transition exposure despite uniform game-root sampling; state the sampling unit clearly in the prospective protocol and final report.

Use the current authoritative collector/vector environment rather than a parallel bespoke training loop. The native VectorEnv currently auto-resets with its original per-stream configs. Any sampling extension must actually change reset setups, remain deterministic under exact journal and current-game recovery, and work with current native/sharded producer boundaries or reject unsupported worker modes before allocation. Preserve the old fixed/cross/mirror paths byte-for-byte semantically. Avoid resetting or discarding an unfinished game at update boundaries, stale-policy trajectory reuse, or mutable hidden Python-only sampling state that recovery cannot reproduce.

A possible small design is a native optional exact matchup roster and a separate domain-seeded deterministic uniform selector keyed by each game's existing reset seed, used by ALL reset paths, including auto-reset and current-game restoration. Keep the selector independent of in-game RNG. Expose actual current matchup identity to the collector so terminal accounting attributes the old game and next observations use the newly sampled game. This is a proposal, not a requirement for a framework or general curriculum engine. An equivalent bounded implementation is welcome if its proof is equally explicit. Keep experiment seeds and source versions prospectively recorded.

## Verification before paid work

Test real setup changes across terminal resets, inclusion/reachability of all100 ordered pairs and mirrors, setup/Lesson-pool validity, learner-seat independence, deterministic seed behavior and exact replay/current-game continuation, and truthful exposure counters. A real collector CPU training update/checkpoint reload must use the complete roster. Include invalid roster/mode rejection and ensure existing two-deck collector behavior/recovery tests pass. If changing Rust, run configured debug cargo tests and rebuild from the repository root with uv per AGENTS.md before Python/native integration tests. Do not update frozen historical worlds/evidence or claim registration alone is enough.

## Continue the same experiment

After scheduling is implemented and checked, freeze the prospective protocol in git per experiments/README.md: question, numerical prediction, usefulness threshold, seed, source/native/world/roster/recipe, fixed evaluation seeds/anchors/cohort, kill criteria and inclusive cumulative cost cap. Inspect live jobs/prior charges, current rates and remote admission; no duplicate paid run. Single-run cap stays$10 including startup/probes/retries/paid evaluation with shutdown/export reserve and existing one-day maximum. Existing remote lifetime/credential guards may require safe cutoff below$10; do not bypass guards, reset cumulative accounting, or start a second run to improve results. No PR or merge is a prerequisite for freezing runnable source and running this experiment; preserve source bytes/digests so the remote job builds the exact implementation.

Use retained MTG-159 baseline/checkpoint rather than retraining it. Execute one actual run with existing monitoring/remote lifecycle, retain immutable checkpoint/cost/stop receipts, perform bounded CPU evaluation and publish a concise scientific result/comparison with method/checkpoint uncertainty, card/pair coverage and limits. Status reports, smoke tests, job acceptance and running jobs do not complete the Task. If a real unrepairable external blocker occurs, retain its precise evidence in this Task and stop without spending beyond the cap.

Delete — do not maintain: no existing public sampler is obsolete; preserve historical fixed/cross/mirror recipes and their exact semantics. Remove only transient duplicate scaffolding introduced during this repair. Do not build a second training executor.

## Corrected implementation entry point

The initial repair Flow reached a repository override `.lf/skills/implement.md` whose entire scope was unrelated Academy runtime checks; it made no sampler changes and its router correctly withheld paid execution. This does not satisfy the present design. That Flow39a22b21-b765-4be4-aade-7be42710ddc1 has ended. The corrected entry point is the task-specific elemental-uniform skill, followed by a capability router and the task-specific elemental-experiment execution skill. The generic override is retained for its existing callers; no global instructions or historical evidence were changed. The Task now has exported identifier MTG-174; its stable ID remains task_f721516493004baa9b5a9025b1ef81c8.


## Local implementation handoff — 2026-10-21

Implemented `matchup_curriculum="elemental-uniform"` in the existing
`TrainSelfPlay` → `SeatRoutedCollector` → native `VectorEnv` path. No rental,
paid execution, Task completion, PR or source/protocol freeze was performed.
Existing untracked Flow/skill/experiment notes were preserved.

- Native `set_uniform_roster` requires every exact `elemental-world-v1` setup
  once, including its Lesson pool and explicit content pack. Python admission
  requires an exact authored representative pair and complete Elemental policy
  input. The native roster order returned by `authored_deck_keys` is authoritative.
- Each new game's existing native reset seed drives two independent uniform
  index draws from a separate ChaCha8 RNG, domain XOR `0x656c656d70616972`.
  There is no persistent sampling RNG or consumption of shuffle/policy RNG.
  Every ordered pair, including mirrors, has probability 1/100 at game roots.
  Explicit resets, terminal auto-resets, buffered paths and current-game replay
  share that selector. Fixed/cross/mirror branches retain their setup/RNG behavior;
  no world, authored content, historical recipe or frozen evidence was changed.
- Uniform mode requires current-self, ordinary roots and **workers=1**. Unsupported
  worker/root modes fail before environment allocation; no sharded-uniform claim.
  Keep the accepted 256 streams, 64 transitions/update, learner seat `s % 2`.
  Unfinished games continue across updates; no sampled action is carried forward.
- `cumulative_matchup_exposure` in successful update diagnostics records actual
  ordered deck identities and learner seat, with game roots, surfaced environment
  decisions, learner transitions, terminal games and truncations. Keys are
  `deck_identity(seat0)/deck_identity(seat1)/learner-N`; `deck_identity` hashes
  main deck plus Lesson pool. Terminal counters use the old game's pair before
  refreshing to the auto-reset pair. Snapshot recovery retains counts and derives
  current identities from native seeds. Aggregate marginal deck/seat exposure
  from these keys; sampling is per game, never exact finite transition balance.

Validation:

- New native debug vector test reaches all 100 ordered pairs separately in both
  learner-seat groups across 4,000 actual roots, checks seeded roots against
  scalar native environments including Lesson offers, rejects incomplete/altered
  rosters, and checks changed terminal-reset observations against scalar execution.
  All 10 debug vector tests passed, including old fixed/reset/shard tests.
- New `tests/training/test_elemental_uniform.py` covers all-pair checkpoint
  admission, invalid configurations, both exact-journal and bounded current-game
  recovery through changed terminal roots, identical subsequent 64-transition
  batches, and independent native action-journal audits of every pair/seat counter.
  Two real CPU updates export/reload raw and EMA checkpoints and change weights.
- Local full-size CPU proof at `.runs/elemental-uniform-local-256x64/` retains
  `recipe.json`, `run/run.json`, VerifyStore, initial/raw/EMA checkpoints and
  `proof.json`: seed 17402, 64 hidden × 2 attention layers × 4 heads, scalar value,
  FiLM=false, 256 streams × 64 transitions, one existing Ataraxos update.
  Completed 16,384 learner transitions, 34,811 surfaced environment decisions,
  52 terminal games, 1,200 retained optimizer exposures; collection 68.15 s,
  learning 18.44 s. Both raw and EMA reload with all ten setups and changed weights.
  Realized roots: 308, including 35 mirror roots; 95 ordered pairs and 158
  pair/learner-seat cells observed. These are finite-sample counts, not a coverage
  guarantee or a scientific strength result. Raw SHA-256:
  `8cfc6070b81b28075864f32756bb5242bacc192eea392fc946025bf325e671d2`.
  The later zero-valued truncated-game field addition is covered by the final
  focused regression suite; this earlier full-size proof has no truncations.
- Release extension rebuilt from repository root with
  `uv run --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels`,
  then the wheel's cp312 `.so` installed at the AGENTS.md package path. `play`
  supplies maturin, absent without that extra. Final build follows all Rust edits.
- `cargo clippy --manifest-path managym/Cargo.toml --all-targets --all-features -- -D warnings`,
  cargo formatting, Ruff and diff whitespace checks passed.
- Final Python regression command:
  `uv run --extra dev --extra notebook --extra artifacts pytest tests/training/test_elemental_uniform.py tests/training/test_matchups.py tests/training/test_ataraxos_regime.py tests/model/test_world.py tests/training/test_recovery.py tests/training/test_active_recovery.py tests/training/test_producers.py tests/sim/test_net_opponent.py -q`
  — **73 passed**, including legacy fixed/cross/mirror collection, sharded
  producers, exact full-learning recovery and current-game recovery. Final native
  suite `cargo test --manifest-path managym/Cargo.toml -q` also completed:
  **484 passed, zero failures/ignored**, including debug invariants,
  clone/undo contracts and semantic conformance. Final targeted vector checks
  passed after the small fixed-mode query cleanup as well.

Execution boundary: no sampler capability blocker remains in the tested
single-producer path. Before paid work, the next step must freeze the prospective
scientific protocol and exact runnable source/native bytes, inspect fresh live
jobs and cumulative charges/rates, admit the guarded remote lifetime and inclusive
$10 budget, and pin fixed evaluation seeds/anchors. Those execution gates remain
unverified here. The bounded CPU proof is not the accepted paid baseline or Task
completion. No PR/merge is required to take that next authorized step.
