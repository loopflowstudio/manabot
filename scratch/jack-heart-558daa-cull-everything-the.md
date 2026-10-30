# 558daa — cull what the codebase no longer needs

Directive: Jack Heart, 2026-10-30 (Task 558daa). Stacked on afbd6a (#403).
Status: implemented; awaiting gate/demo. Stop at review/demo; no merge.

## Approach

1. Protected set: every file touched by open PRs #383 #395 #399 #402 #403
   (`gh api pulls/N/files`). Do not delete or restructure those files; list
   held-back candidates below.
2. Area audits (read-only) produce candidates with evidence of no current
   reader: managym; manabot + tests; academy; scripts/docs/CI/conformance/
   experiments code/top-level.
3. Delete by area in separate commits; verify with debug cargo test, clippy,
   fmt, focused pytest, Academy UI check.
4. Delivery: all area commits go in ONE PR (Jack Heart, 2026-10-30). Local
   per-area commits are fine; do not split into multiple or stacked PRs.

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

## Kept on purpose

- All four w17 conformance corpora: each has a current reader (Rust conformance tests/CI; Academy parity scripts and tests).
- `x-etude-*` headers, `etude.*` storage keys, schema ids, `managym` import alias, service names (CLAUDE.md).
- `docs/rules/audit-2026-10/`, `docs/rules/tla/` (MTG-177 / TLA T2–T7 open), frozen evidence in `docs/evidence/*` that reports cite, `docs/benchmarks/*` preregistrations.
- Four `experiments/regimes/elemental-*.json`: drivers and protocol identity files read them.
- `academy/learn_lesson_evidence.py` (repository-archive records keeping it); Academy curated-combat fixtures (both read).
- Experiment-only manabot modules whose drivers are current (LLM stack, `reports.inference`, `experiment.study`, `eval.local`, `eval.benchmark`, `training.panels`) and documented manual CLIs (`checkpoint_scenarios`, `reports.snapshot`, `serving_benchmark`, `sampling_cohort`, `monitor_worker`, `python -m manabot.llm`).
- `PassivePolicy` (reached via `build_opponent_policy("passive")`); `PriorityState.pass_count` (hashed into the match-state hash).
- `scripts/generate_advice_fixture.py`, `generate_conditional_strategy_fixture.py`: sole generators of live fixtures.

## Wanted to delete, held back

Held because an open PR edits the file (#383 #395 #399 #402 #403):
- Rejected search-branching drivers: `flow/undo.rs` (766 lines), ~100 `journal_*` calls in 17 `flow/*.rs`, `Game` COW helpers, `flow/event_log.rs` paged log, `search_state.rs` `ClonePlusUndoDriver`/`DensePageCowUndoDriver`, `benchmark.rs` (1,276), `bin/branching_bench.rs`, `scripts/bench_branching.py` + test, and their search_state_contract tests. Decision record: `docs/benchmarks/search-branching-decision-v1.md` (keep full clone). Largest remaining cut; needs `game.rs`/`benchmark.rs`/`flow/*` free of #399.
- `scripts/bench_study_branch.py` + test (RUL-6 finished; no CI).
- PyO3 surface with no Python caller: `export_profile_baseline`, `compare_profile` (+ profiler baseline code), `structured_search_offers`, `unsupported_cards`, `RolloutPool.active_count` (`python/bindings.rs`).
- Small Rust items: `basic_land`, `local_power/local_toughness`, `players_starting_with_*`, `PlayerConfig::deck_list`, `ZoneManager::total_size`, `Env::encode_observation_into`.
- `manabot/semantic/policy.py` (556, test-only), belief demo cluster (`belief/demo.py`, `recipes/belief.py`, `belief/population.py`, ~1,460, reached only by `belief-demo` CLI commands), retired `train --preset/--set` rejection in `cli.py`, `EvaluationContract` family in `eval/models.py`, `flat_mc.aggregate_records`, `provenance.validate_runtime_fingerprints`, `match.parse_deck`, `action_cases.identity_removed`, `belief/__init__.py` re-export map.
- `eval/duel.py` + `tests/training/test_head_to_head.py`: `manabot-integration.yml` names the test.
- `academy/semantic_boundary.py` (test-only; listed in `authored_match_parity._source_manifest`), `server.py` `DEFAULT_DECK`/`hero_view` aliases and `resolve_live_advisor_decision`, `advice.py` `int13_result_to_surface_deltas`/`policy_mass_is_non_uniform` (tests in protected `test_advice.py`).
- `protocol/advice-v1.schema.json` (nothing validates it), `scripts/verify_elemental_world.py` (cited by protected `docs/rules/elemental/world.md`; its output dir is archived).
- Top-level `wave/` (legacy copy of `.lf/wave/`; `wave/README.md` is in #402).
- CI: `semantic-conformance` job repeats the Rust job's test and `check`; only `fuzz` is unique (`ci.yml` held).
- Stale citations in protected docs to fix after their PRs land: `manabot/README.md` (`PairedLearners`, `sim/action/targeting.py`, `manabot/training/builders.py`), `docs/repository-archive.md` (targeting), `WORLDS.md`, `docs/rules/{optional-payments,learn-lesson}.md`, `docs/ARCHITECTURE.md`.

Needs Jack's decision (not deleted):
- Seeds + checkpoint-hash form for `semantic-kernel-w17-v1`: feasible (tapes are a pure function of `manifest.json`; `action_space_digest` already pins ordering), ~100–150 lines in `conformance.rs` + contract v2; failures stay actionable by writing the regenerated case to `target/semantic-conformance/` on mismatch. Not done: every w17 corpus file is in #395/#399.
- Docs consolidation into ~6 current-system files (architecture, development, training, evaluation, search/beliefs, LLM): cross-links run through held `docs/ARCHITECTURE.md`, `README.md`, `WORLDS.md`, `managym/README.md`.
- Live belief runtime in Academy (`academy/live_advice.py` + `server.py` hooks): computes posteriors on every interactive-mirror command that no response reads.
- Website landing-page assets (`/` redirects to the blog), `experiments/planning/`, MTG-186 drivers (an Etude migration test imports them), `seat_comparison.py` (sample file in `test_experiment_files.py`), `experiment/artifacts.py` (old S3 publisher; CLAUDE.md names its test), `verify/decision_profile.py` (only `wilson_interval` used), `recipes/ataraxos.py` (test fixture), `training/presets/ataraxos-mtg-v1.json`.
- Unrelated pre-existing failures seen at base: `tests/sim/test_conditional_search.py` (INT-13 root digest), `tests/sim/test_search_runtime.py::test_seed_197_checkpoint_model_abis_match_the_registry` (w17 world binding), `tests/verify/test_observation_parity.py::test_elemental_indestructible_encoding_preserves_outside_and_definition_columns` (55 == 41; file untouched by this branch).
