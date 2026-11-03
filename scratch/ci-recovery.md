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

Remaining owners: gate owns the affected suites, including debug Cargo targets
that CI never reached after the lib failures; publication owns the repaired
head and a fresh Engine Budget comparison; demo owns the real-path walkthrough.
Main integration's later evaluation/smoke steps were skipped, despite the
separate long-integration job passing. These checks remain required. No demo
acceptance or additional shipping approval is inferred. Both accepted concept
decisions and frozen evidence are preserved.
