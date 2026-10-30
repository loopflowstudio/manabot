# 558daa — cull what the codebase no longer needs

Directive: Jack Heart, 2026-10-30 (Task 558daa). Was stacked on afbd6a; #403
merged into main (10eeb1f55) and this branch merged it (8db3b2fe8), so the PR
targets main and its diff is the cull alone.
Status: third pass implements Jack Heart's 2026-10-30 decisions on the held list (all in this one PR; open PRs rebase); awaiting gate/demo. Stop at review/demo; no merge.

## Approach

1. Protected set: every file touched by open PRs #383 #395 #399 #402
   (`gh api pulls/N/files`; #403 merged 2026-10-30, so its files are free). Do not delete or restructure those files; list
   held-back candidates below.
2. Area audits (read-only) produce candidates with evidence of no current
   reader: managym; manabot + tests; academy; scripts/docs/CI/conformance/
   experiments code/top-level.
3. Delete by area in separate commits; verify with debug cargo test, clippy,
   fmt, focused pytest, Academy UI check.
4. Delivery: all area commits go in ONE PR (Jack Heart, 2026-10-30). Local
   per-area commits are fine; do not split into multiple or stacked PRs.

## Jack Heart decisions, 2026-10-30 (held-back list)

1. Search branching: forking is the search strategy now. Cut the rejected branching drivers (ClonePlusUndoDriver/DensePageCowUndoDriver, Game COW helpers, paged event log, benchmark.rs, bin/branching_bench.rs, scripts/bench_branching.py + tests). Do not wait on #399.
   flow/undo.rs + journal_* calls: evaluate whether undo is correct and complete today (perfect or very close). If it is not, cut it too and record the evidence in the ledger. A follow-up Task for undo will be filed: Jack wants undo in Academy eventually. If it is near-perfect, keep it and say why.
2. CI: cut the duplicate semantic-conformance job (keep fuzz coverage somewhere).
3. Top-level wave/: remove it entirely; bring wave content up to current Loopflow standards under .lf/ (no legacy wave/ anywhere). Jack explicitly authorizes this write.
4. Do the seeds + checkpoint-hash form for semantic-kernel-w17-v1 (failures write the regenerated case on mismatch). Do not wait on #395/#399.
5. Do the docs consolidation into ~6 current-system files. Do not wait on open PRs; they will rebase.
6. Remove Academy's live belief runtime (academy/live_advice.py + server.py hooks); Jack is not confident the belief code works.
7. Remove: website landing-page assets, experiments/planning/, the old S3 publisher (experiment/artifacts.py; update CLAUDE.md/tests that name it), recipes/ataraxos.py + training/presets/ataraxos-mtg-v1.json.

Behavior limits and frozen-evidence rules still apply. Update the ledger and held-back list accordingly.

## Deletion ledger

Academy (b2e036c67):
- `academy/ui/src/lib/packs/{elemental-world,tla-ur-lessons-vs-gw-allies,tla-gw-allies-jeong-vs-ur-lessons}/v1/` → every reader uses v2; the one e2e import moved to v2 (identical roster/pack id).
- `academy/ui/tailwind.config.ts` → Tailwind v4 never loads it (no `@config`).
- `academy/ui/e2e/dense-board-captures/*.png` → outputs of the opt-in spec; now gitignored.
- `academy/__init__.py` `academy.app` lazy alias → every entry point uses `academy.server:app`.
- `academy/local_advice.py` (+ two test blocks) → only tests read it; no advisor uses it.
- `academy/advice.py` live/study advisor resolver chain (121 lines) → no caller.
- `academy/ui/src/lib/advice.ts` TS adapter twin + `advice.test.ts` → test-only mirror of Python `request_advice`.
- `presentationInspectorRows` → no inspector UI; recorded sequences moved from production `presentation.ts` to `presentation-fixtures.ts` (test-only), so the fixture JSON leaves the client bundle.
- `StopStepKey`, `academy/traces/.gitkeep` → unused.

Repository (b1edd343b):
- 35 `experiments/regimes/*.json` + `experiments/tournaments/size-floor-grid.json` → deleted in #340, resurrected by the #353 merge; nothing reads them (reports cite three by commit-pinned URL).
- `.clangd`, `.clang-tidy`, `.clang-format`, `.vscode/settings.json`; CMake/`ops` lines in `.dockerignore` → no C/C++ left.
- `scripts/benchmarks/`, `scripts/etude_native_job.sh`, `scripts/bench_policy_driver.py` → no runner; frozen reports keep the citations.
- `experiments/{elemental_181_new_heuristic,mtg186_report,mtg206_cards,heuristic_strength,gpu_training_profile,ataraxos_mini,seat_comparison_report}.py` → finished studies; reports stay.
- `protocol/fixtures/study-index-identity-curated.json` → no reader.
- `docs/plans/{etude-fantasia-manifest,corrected-baseline-operations,candidate-demo-comparison,live-belief-advice}.md`, `docs/research/phase-experience.md`, `docs/evidence/{learn-ci-repair-2026-09-25,learning-dashboard-2026-10-08}.md` → superseded plans/records with dead references.
- `paper/drafts/` → historical July drafts.
- Root `RELEASE_NOTES.md` → byte-identical to `release/v0.2.0/NOTES.md`.
- `academy/ui/e2e/visual-references/` (13 MB) → no consumer; archived byte-for-byte to `experiments/data/archive/academy/visual-references/` (133 manifest entries) before untracking.

managym:
- `examples/measure_content_pack.rs`, `bin/validate_study_artifact.rs`, `infra/log.rs` → nothing runs or imports them.
- `PriorityState::pass_priority`, `SemanticError::NoMatch`, `lib.rs` re-exports of `VectorEnv`/`MatchStateHash` → unused (`pass_count` stays: hashed).


manabot (8ab966b4a; ~3,300 code/test/doc lines + 26k lines of unread fixture JSON):
- `sim/search/teacher1_evidence.py` + test → finished INT-4 Teacher-1 pilot, test-only (`_fresh_env` moved to `tests/belief/fresh_env.py`).
- `sim/action/targeting.py` + test, `training/learning/paired.py` → never used outside tests.
- `verify/util.py` first-light evaluation harness (612 lines) → only frozen reports and its own tests named it.
- `env/single_agent_env.py` `SingleAgentEnv` and truncation helpers; `conditional.py` belief variant; `runtime.py` INT-7 evidence snapshot; `rating.payoff_matrix`; `value.spearman`, unused name sets; `store.py` first-light field lists; `sampling_fit` exact-reference cohort; `UniformRangePlayer`; `BeliefState.draw_unknown/remove_known/add_known`; `OBSERVATION_BUFFER_DTYPES` → test-only or unreferenced; surviving tests now exercise the live paths with the same expected values.
- Duplicate `file_sha256` / `canonical_sha256` (incl. `sim/search/runtime.py`) → import from `manabot.infra.identity`.
- `env/single_agent_env.py` → renamed `env/opponents.py` (only `PassivePolicy`/`RandomPolicy`/`build_opponent_policy` remain); `flat_mc` docstrings no longer cite the deleted `verify.util` harness.
- `configs/llm/openrouter-haiku.json`, `manabot/model/training_monitoring.md`, `tests/fixtures/resolution-eligibility-captures{,-v4}.json` → unreferenced.

Receipt change (7e61a050b): `conformance/authored-match-parity-w17-v1` receipt regenerated; only its `relevant_source` file hashes changed (it hashes every `managym/src/**/*.rs`). Parity result unchanged (142 commands, no divergence).

Second pass (protected set re-read from `gh api pulls/{383,395,399,402}/files`, 2026-10-30):
- manabot (38c692e14): `EvaluationContract` family (contract, profile, schedule, promotion, rating, roots, caps, `BASE_ANCHOR_IDS`) → tests read only the INT-6 fixture's key and anchors, now via typed `tests/eval/arena_fixture.py`; validator-only test dropped, cohort digest still asserted. `belief/__init__.py` map trimmed to the 15 names protected helpers read; three unprotected tests import defining modules.
- repository (ae9c5c2a5): `scripts/bench_study_branch.py` + test, `scripts/verify_elemental_world.py` (world.md cites it at a74f31828; frozen reports/preregistrations keep their citations); `semantic-conformance` CI job keeps only fuzz (every path selecting it also selects Rust Tests, which runs `semantic_conformance_tests` incl. `check_root`); stale corpus/scratch citations in `docs/rules/{optional-payments,learn-lesson}.md` → archive paths / plain checkpoint names.
- managym (05d5e2ac4): `ZoneManager::total_size`; parity receipt `relevant_source` re-hashed (result unchanged: 142 commands, no divergence).
- manabot (cd8778588): `semantic/policy.py` + test (test-only; no CLI path; boundary probe keeps the sim entry points).

Third pass — Jack Heart's decisions on the held list (2026-10-30, comment f57280ee; "do not wait on open PRs; they will rebase"):
- website/repository/manabot (350ed3370): website landing components, `content.yaml`, `static/` (blog uses none), `pyyaml`; `experiments/planning/` (plan doc links pinned at a74f31828); `manabot/experiment/artifacts.py` + its test (other artifact-store tests stay; training-monitoring doc rewritten); `recipes/ataraxos.py` → `tests/training/wdl_baseline.py`; `training/presets/ataraxos-mtg-v1.json`.
- Academy (9075fab0a): `academy/live_advice.py`, its server hooks and test, `resolve_live_tracked_posterior`, `resolve_live_advisor_decision` + `StudyForkProvider.resolve_advisor_decision`, `scripts/play.py` runtime import.
- Waves (465be9dcb): top-level `wave/` deleted; `.lf/wave/` adopts #402's layout (engine, training, eval) plus `academy` (legacy game charter/memory, live-advice section replaced by its retirement) and `engine/MEMORY.md` (legacy rules memory). Code/doc citations repointed; frozen experiment reports keep historical `wave/` text.
- managym (7c4036867): undo cut — not near-perfect: structured Command path (Academy's) unjournaled (`structured_offer.rs` takes the action space and clears attackers with no entries), `lose_game` and semantic object bindings unrestored, broad check only on ten Alpha-era cards; no product reader. Removed `flow/undo.rs`, ~100 journal calls, `Game.undo`, COW fork/paged event log (`EventLog` now a dense Vec wrapper), clone-plus-undo/page-COW drivers, `BranchDriver`/`BenchCommand`, `benchmark.rs` harness → `match_fixtures.rs` (fixtures only), `bin/branching_bench.rs`, `scripts/bench_branching.py` + test, zone/id-generator undo helpers. Undo-generic tests now assert fork isolation. Follow-up Task for Academy undo is Jack's to file (fit it on fork snapshots as an explicit shared action).
- conformance (d1cd00be4): `semantic-kernel-w17-v1` is seeds + per-case `trace_sha256` (schema 2; Phase matrix keeps schema 1); 14 tapes (2.3 MB) deleted; drift writes `target/semantic-conformance/<corpus>/<case>.json` (uploaded by Rust Tests); final state hashes unchanged.
- receipt (07865749b): authored-match parity `relevant_source` regenerated (deleted sources); result unchanged.
- CI: duplicate conformance steps already cut (ae9c5c2a5); fuzz kept.
- docs (7cad8dae5): 29 top-level/architecture/development docs → `development.md`, `training.md`, `evaluation.md`, `search-and-beliefs.md`, `llm.md`, `academy.md` (+ unchanged `ARCHITECTURE.md`; `experiment-metrics.md` kept, reports parse it); written by parallel agents against code, superseded detail linked at a74f31828; inbound links/anchors repointed repo-wide (frozen `docs/rules/audit-2026-10`, `.lf/chapters`, `experiments/data` untouched) and every anchor checked. Unverified claims were dropped, not guessed; `experiments/ataraxos-transfer.md` already imported a missing `learned_belief_local_search` (pre-existing).

## Kept on purpose

- All four w17 conformance corpora: each has a current reader (Rust conformance tests/CI; Academy parity scripts and tests).
- `x-etude-*` headers, `etude.*` storage keys, schema ids, `managym` import alias, service names (CLAUDE.md).
- `docs/rules/audit-2026-10/`, `docs/rules/tla/` (MTG-177 / TLA T2–T7 open), frozen evidence in `docs/evidence/*` that reports cite, `docs/benchmarks/*` preregistrations.
- Four `experiments/regimes/elemental-*.json`: drivers and protocol identity files read them.
- `academy/learn_lesson_evidence.py` (repository-archive records keeping it); Academy curated-combat fixtures (both read).
- Experiment-only manabot modules whose drivers are current (LLM stack, `reports.inference`, `experiment.study`, `eval.local`, `eval.benchmark`, `training.panels`) and documented manual CLIs (`checkpoint_scenarios`, `reports.snapshot`, `serving_benchmark`, `sampling_cohort`, `monitor_worker`, `python -m manabot.llm`).
- `PassivePolicy` (reached via `build_opponent_policy("passive")`); `PriorityState.pass_count` (hashed into the match-state hash).
- `scripts/generate_advice_fixture.py`, `generate_conditional_strategy_fixture.py`: sole generators of live fixtures.

## Held

Still held (outside Jack's 2026-10-30 decisions; files in open PRs):
- #395: belief demo cluster (`belief/demo.py`, `recipes/belief.py`, `belief/population.py`) — reached by `belief-demo`/`belief-learn-demo` in `cli.py`; `tests/belief/test_state.py` imports the demo. `match.parse_deck`. Remaining `belief/__init__.py` map (read by `tests/belief/{support,test_state}.py`). `advice.py` `int13_result_to_surface_deltas`/`policy_mass_is_non_uniform`. Retired `train --preset/--set` rejection in `cli.py`; `provenance.validate_runtime_fingerprints`; `server.py` `DEFAULT_DECK`/`hero_view` aliases; `Env::encode_observation_into`; `manabot/README.md` stale citations.
- #399: `action_cases.identity_removed`; `flat_mc.aggregate_records` (imported by `tests/sim/test_rollout.py`); `academy/semantic_boundary.py` (edits `authored_match_parity._source_manifest`); `protocol/advice-v1.schema.json`, `local_power/local_toughness`, `basic_land`.
- #395/#399: unused PyO3 surface in `python/bindings.rs` (`export_profile_baseline`, `compare_profile` + profiler baseline, `structured_search_offers`, `unsupported_cards`, `RolloutPool.active_count`); `players_starting_with_*`, `PlayerConfig::deck_list`.
- #383/#395: `eval/duel.py` + `tests/training/test_head_to_head.py`.
- Not decided: MTG-186 drivers (an Etude migration test imports them), `seat_comparison.py`, `verify/decision_profile.py` (only `wilson_interval` used).

## Checks

- Pre-existing at base (fail identically with this branch's changes stashed): `tests/belief/test_belief_agent.py::test_keystone_demo_changes_belief_tokens_and_hides_actual_truth` (zero deltas), `tests/belief/test_state.py::test_native_projection_preserves_canonical_receipt_bytes` (receipt digest), `tests/semantic/test_temur.py::test_temur_finishes_through_the_production_training_observation_path` (60 == 40, v5 player width); earlier: `tests/sim/test_conditional_search.py`, `tests/sim/test_search_runtime.py::test_seed_197_checkpoint_model_abis_match_the_registry`, `tests/verify/test_observation_parity.py::test_elemental_indestructible_encoding_preserves_outside_and_definition_columns`.
- Second pass (cd8778588): debug `cargo test` (opt-level 1, as CI) 665 passed / 0 failed; `cargo fmt --check`, clippy `--all-targets` clean; CLAUDE.md world-change suite 65 passed; focused pytest over touched tests + `tests/belief tests/semantic` 363 passed / 3 failed (the three above) / 1 skipped; Academy `npm run check` 0 errors.
- Third pass (after 7cad8dae5 + inventory anchors): debug `cargo test` (opt-level 1) 650 passed / 0 failed; `cargo fmt --check`; clippy `--all-targets` with and without `python,benchmark` clean; `semantic_conformance check` 14 cases / 3,401 commands; authored-match parity 142 commands, no divergence; website 21 passed. Focused pytest (world-change suite + `tests/academy tests/env tests/sim tests/infra`, touched training/model tests) 700 passed / 10 failed / 1 skipped → after repointing renamed-test anchors in `docs/rules/audit-2026-10/{index.jsonl,reviews.json}` the inventory passes; the other 9 are pre-existing: the two known (`test_conditional_search`, `test_search_runtime` seed 197) plus 7 that fail identically at a74f31828 with its own rebuilt extension (`test_public_commitment_parity` ×2: missing local `experiments/data/rul-9-played-workloads-v1.measurement.json`; `test_ur_lessons_increment` ×4: permanent width 32 == 29; `test_rollout::test_selects_valid_actions_only`: unadmitted card definition). Academy UI unchanged since the 0-error check.

