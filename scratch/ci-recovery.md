# MTG-239 CI recovery — 2026-11-03

PR #415 at ea316ff14 was published and waiting for demo when CI run
41711692234 exposed the failures below. Jack Heart's November 3 steer
(comment d35e6e94) returned the existing Task to the code Graph for repair.

- Rust lib tests: 124 passed, 3 failed. Failures are
  `agent::vector_env::tests::training_decks_are_admitted_distinct_replayable_and_replaceable`,
  `cards::deck_admission::tests::novel_decks_admit_against_the_world_without_an_authored_setup`,
  and `cards::setups::tests::world_selection_admits_novel_decks_and_never_falls_back`.
  Their negative fixtures still use now-playable Accumulate Wisdom/Rallier.
- Main integration: 865 passed, 1 skipped, 4 failed, all in
  `tests/training/test_training_decks.py`: source/batch validation, regime bad
  decks, checkpoint setup validation and replacement-root rejection. These
  still expect Accumulate Wisdom rejection.
- Repair: use the Ally token for these negative fixtures, retaining rejection
  assertions and unchanged roots after failed replacement. Native batch and
  training-source checks positively admit all five repaired cards; setup,
  checkpoint and replacement checks admit Rallier, Accumulate Wisdom and
  Cat-Gator. Existing unknown-card and rules-fixture tests remain unchanged.
- Engine Budget: timing passed (worst reply 5.280 → 4.986 seconds, total
  177.796 → 171.233 seconds, throughput 12874.6 → 12891.2 steps/sec). Three
  search digests, step count (99923 → 101650), and step digest differ. Both
  runs completed 512 games without caps; the old reference reproduced its
  committed digests. The unchanged benchmark decks contain Accumulate Wisdom,
  Rallier and Suki, whose new ordering choices and contextual library power
  affect actual play and search rollouts. Adopted this PR's artifact with
  `uv run scripts/engine_budget.py record --artifact /tmp/mtg-239-engine-budget-41711692234 --reason …`;
  the file records the full reason and retains the 10% tolerance. Documentation
  now cites this run and its same-run timing deltas. This is not evidence of
  equivalent gameplay or a speedup claim.
- Python unit, Academy integration/UI, clean-machine play, both conformance
  jobs and long integration passed. Full validation still needs the repaired
  head; reuse unaffected evidence appropriately.

Raw log can be refetched with `gh run view 41711692234 --log-failed`; this
machine also retains it at `/tmp/mtg-239-ci-41711692234.log`.

Implementation check result: debug `cargo test -p managym --lib` filters
`training_decks_are_admitted_distinct_replayable_and_replaceable` and
`novel_decks` passed (3 tests; `CARGO_PROFILE_DEV_OPT_LEVEL=1`, `-j 2`);
`uv run --extra dev --extra artifacts pytest tests/training/test_training_decks.py -q -k 'experiment_builds_a_source or regime_records_its_source or checkpoint_binds_world or later_roots_take_new_decks'`
passed (4). The root release wheel rebuilt and its cp312 extension was installed
atomically. Artifact provenance/digests and original same-run timing checks
match the recorded baseline; Cargo formatting and Python lint pass.

The repaired and simplified head `e5627dd98` is now published on PR #415.
CI run `41734880466` was in progress at the November 3 reconciliation; no
completed result for that head is inferred. Gate owns the affected suites,
including debug Cargo targets that the first run never reached after the lib
failures, and the fresh Engine Budget comparison. Publication owns subsequent
repairs and final reviewable context; demo owns the real-path walkthrough.
Main integration's later evaluation/smoke steps were skipped, despite the
separate long-integration job passing. These checks remain required. No demo
acceptance or additional shipping approval is inferred. Both accepted concept
decisions and frozen evidence are preserved.

Publication verification (November 3, source head `e5627dd98`): `lf sync
--manual` completed against main `44388ad42` without conflicts or code changes.
Run `41734880466` has passed the complete debug Rust suite, Academy UI
(unit/check/build/release browser interactions), both conformance jobs,
clean-machine play, Python unit, Academy integration and long integration.
Main integration also passed, including two-seat/Etude evaluation, the
Evaluation Perf report proof, enum smoke and training smoke. All 11 jobs in
run `41734880466` completed successfully; the package build/install workflow
`41734880090` also passed on Linux and macOS.

The fresh Engine Budget comparison passed with identical three search digests,
step digest, 101,650 steps and 512 games. On that runner, reference → candidate:
worst reply 3.626 → 3.617 seconds, total search 123.573 → 123.722 seconds,
throughput 17,286.7 → 17,490.5 steps/sec; tolerance remains 10%.

Local supplemental checks: 70 tests passed across
`tests/academy/{test_choice_navigation,test_learn_setup,test_authored_match_parity,test_public_commitment_parity}.py`,
`tests/semantic/{test_current_content,test_learning_projection}.py`, and
`tests/belief/test_state.py` (37.79 seconds). Cargo formatting, changed-Python
Ruff, and all-target/all-feature clippy passed. Clippy's first attempt could
not resolve a relative interpreter path; rerunning with
`PYO3_PYTHON=/Users/jack/src/manabot.mtg-239-complete-five-defined/.venv/bin/python`
and `CARGO_PROFILE_DEV_OPT_LEVEL=1`, `-j 2` passed. No runtime source changed
during publication verification. Demo remains a separate, pending Graph step.
