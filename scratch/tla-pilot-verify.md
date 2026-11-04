# TLA pilot continuation — 2026-11-04 UTC

Headless demo at 19:31 UTC accepts the enabling repair for delivery:
[review record](tla-pilot-demo.md). Fresh evidence includes 11 passing behavior
checks and 717 valid self-play collection decisions through the exact v8 model,
observation and 134-deck source, plus four scalar evaluation-adapter decisions.
This adds no GPU update, checkpoint or paired score; empirical acceptance below
remains outstanding.

Jack Heart's accepted outcome remains preliminary GPU TLA self-play and paired
checkpoint evaluation on the frozen 134-deck trophy population, <=$1 per
allocation and <=$10 cumulative. Corpus/runners, Vast integration and an earlier
GPU learner milestone are delivered. An admitted trained checkpoint, complete
paired evaluation and Elemental-relative analysis remain outstanding.

## Current accepted design

Jack Heart's accepted 2026-11-04 steer 3b0f8826 selects deterministic first-64 **legal** actions with
logging and cutoff, superseding the 512-action pilot. Steer f61beeb7 separately
requires native legality/payability and direct unavailable-X rejection without
state changes. Neither cutoff nor tensor capacity can legalize an invalid action.
V8 preserves input-v7 shared effects, card128, all 134 decks, recipe, seeds and
budgets. The fixed heuristic anchor retains its existing full native semantics;
learned candidates and the initial-policy anchor use the same cutoff as training.

`ObservationConfig.action_overflow=first-legal` is explicit in v8's regime,
protocol and checkpoints. Default `error` is omitted on serialization, preserving
frozen configurations. Native `policy_action_count` owns prefix selection;
Python and native buffered encoders consume it before bounded writes. Row i
remains native action/Command i. Native authority and Academy remain complete.
No pagination or global engine action filtering is introduced.

The existing `casting_mana_cost` repair owns selected-cost announcement/payment:
airbend {2} fixes printed X to zero; any-type-mana casts still pay printed X;
additional waterbend X survives a free non-X printed cost. New direct native
submission tests use current command bindings and full search-state witnesses.
Seven untapped basics pay precisely X=0..5 for {X}{U}{U}; X=6/255 reject unchanged.
Airbend and any-type-mana rejection checks also pass. No further rules repair
was exposed by these tests.

`input_buffer_config` still derives configurable axes from supplied arrays and
validates companion shapes. The cutoff does not alter any card, permanent,
focus, fact, relation or shared-effect capacity behavior.

Diagnostics count consumed observation decisions and overflow by decision kind,
with dropped totals and up to four full seed/deck/action-prefix samples carrying
original/retained/dropped counts. Console summaries occur at powers of two;
collector/game close writes exact aggregates, including zero overflow. Evaluation
retains the report in game receipts. Deliberate cutoff is distinct from a fatal
observation/game truncation. Scalar, native vector and sharded collectors use
the same path; no policy mask or browser filter duplicates legality.

The active launcher and analysis/discovery imports select `tla_pilot_v8.py`.
Completed simplification removed buffer default overrides and printed-cost-only
X announcement; overflow totals derive from per-kind counts, collection uses
one buffer-policy dispatch and parity tests share one comparison helper. Frozen
v1–v8 protocols, regimes and evidence remain unchanged.

## Frozen execution and spending

V5 Run `e8c76bc2c06f`, source `97a2cfb963d7e3a3b2816f31c45dda72f34cdddd`,
ran on Vast RTX5090: three learner updates, 585 optimizer steps and 149260
transitions, then failed at 256 offered actions versus 64 before trained export.
The 256 offers were subsequently identified as illegal X0..255 during an airbend
cast of Wan Shi Tong. Source, failed attempt and initialization evaluation remain
frozen. The report retains 32 valid initial games (5 wins/27 losses), 256 missing
scheduled games and no improvement claim. All 114 v5 evidence files are published.

Prepared v6 Run `bcde93e9ff8c` at `ad2c3a010f53` and v7 Run `af3661790ac5`
at `b5406bb33abd98870af6ca07c3beed63a92b7c1c` were never submitted. Both are
superseded and must not launch. V7 artifacts and archive remain intact under
`.runs/mtg213/v7-continuation/` and `.runs/mtg213/runtime-b5406bb33/`.
The latter also passed the new retained-prefix rejection/parity diagnostic at 64
and 777 before the new cutoff direction arrived; it is evidence, not v8 authority.

Fresh 18:56:15 UTC inventory found no Vast instances, RunPod pods or unsettled
hardware. Accounted whole-rental cost remains $0.32710677365251912;
conservative historical reservations remain $3.9139916049382716/$10.
Earlier 18:44 quotes refused Vast's <=$0.81/hour ceiling and actual L4 capacity.
They establish no current availability and submitted nothing. Any v8 launch
requires fresh inventory, bills and an actual compatible quote.

Published isolated control client is `etude-sdk[s3]==0.2.14`; worker dependency
remains the recipe's 0.2.12. Credentials use nested Doppler `manabot/prd` then
`etude/prd`. V8 operational scripts under `.runs/mtg213/v8-continuation/` derive
from the retained v7 scripts; they preserve the prepared-run guard and budget
checks. The exact v8 source freeze and subsequent refusals are recorded below.

## Validation and remaining work

Retained implementation evidence: five directed debug `wan_shi_tong` tests and
one `first_legal_policy_prefix` test passed, followed by a rebuilt release
extension. The buffer/cutoff suites passed 61 tests; collection/automatic-capture
suites passed 41 after simplification. Protocol/discovery and Ruff checks remain
applicable. Inspection confirms the earlier feedback's missing rejection tests
are implemented, including full-state witnesses and actual payment for each
offered X. No executable change or paid execution occurred during reconciliation.
Gate owns broader affected suites, full debug/Clippy, current-world admission and
publication/delivery; these focused receipts do not replace those checks.

Reconciliation check: `git diff --check` — passed (plan and Training memory only).

Remaining: reconcile/quote for the selected frozen source, execute one bounded
GPU allocation, retain actual learner/checkpoint evidence and teardown/billing,
then complete the paired evaluation and qualified Elemental-relative report.
Historical missing slots remain missing; no new result fills v5 denominators. No product or budget
approval is outstanding; compatible supply remains an external uncertainty.

Earlier continuation detail is retained at
`ebb42d0df85c6638d47a160f1f888e021622c102:scratch/tla-pilot-verify.md`;
uncommitted notes immediately before v8 consolidation are preserved under
`.runs/mtg213/pre-v8-notes/`. Training memory records the accepted cutoff,
rejection evidence and dated v8 preflight.

## V8 preflight settlement — 19:12 UTC

V8 Run `f47be2d3c0d8` is frozen at
`900c207502bd28757d7cf66196bd464458e39402`. The source archive for later local
evaluation is `.runs/mtg213/runtime-900c20750/`; no games or initial checkpoint
were produced. All quotes refused before submission: RTX5090 exceeded the
USD0.81/hour ceiling; L4 and a separately configured RTX3090 supply reported
no available machines. The RTX3090 alternative retained >=6vCPU/48GBRAM/80GB
disk and the same ceilings. No new rental, watcher, training/evaluation process
or spending exists. Do not infer future availability from these dated reads.

Inventory at 19:10 identified unrelated RTX3090 rental 59148889, session
`0b7669dd3936`, Run `332a954fd935`, job `state_representation_gpu`; it was left
to its owner. The final 19:11:43 read found no Vast/RunPod rentals or unsettled
hardware. Accounted cost and conservative reservations remain unchanged.
Receipts and the exact prepared Run are retained under
`experiments/data/mtg213/gpu-v8-preflight-20261104/` with artifact publication.

Execution needs a fresh compatible quote. Run `f47be2d3c0d8` remains bound to
its original source. Execution of the simplified capture/collector code requires
a successor Run with a fresh source freeze; the prepared v8 Run remains evidence.
Broader gate and delivery remain outstanding, alongside the empirical pilot
acceptance.

## Concept review — 2026-11-04

Outcome: **nothing to change and no open product choice**. This review does not
select Graph navigation or claim pilot completion. Jack Heart's accepted
first-64 direction (3b0f8826), native payability requirement (f61beeb7), shared
effects and fixed-corpus/budget decisions already settle the experience. No
additional implementation direction or product approval is needed.

The consequential distinction is **legal actions versus policy support**.
Managym owns the complete payable action set and rejects unavailable Commands;
the pilot selects a prefix of that set for learned policies. Padding contributes
no probability. A cutoff is diagnostic information, not a failed game or proof
that omitted strategies are redundant. Academy retains the complete shared
decision surface. These concepts have separate jobs; merging them would hide
legal choices or misclassify valid pilot games.

The existing usage contract in `experiments/protocols/tla-trophy-213-v8.md`
already makes the interaction concrete: train with `max_actions=64` and explicit
`action_overflow=first-legal`; evaluate learned checkpoints with that same
configuration; inspect cutoff frequency and replay samples alongside outcomes.
The fixed heuristic remains the documented full-native baseline. No usage-doc
or skill rewrite is needed. This is implemented behavior, with empirical pilot
acceptance still pending, rather than a proposed redesign.

Actions → concepts → APIs:

- Start one bounded pilot → frozen corpus, regime/protocol and source-bound Run
  → `frozen()`, `runs(revision)`, `TrainingJob` and the existing Etude owner.
- Choose an action → complete native decision and explicit learned-policy
  support → shared selected-cost calculation, `policy_action_count`, native
  buffered/Python encoding and original Command binding. No second legality
  filter or page-navigation lifecycle is needed.
- Inspect learning → admitted checkpoint and expected paired comparisons
  → `comparison_schedule`, `eval`/`play`, shared `evaluate`, and the companion
  analysis. Cutoff reports belong to existing capture/game receipts, not a new
  experiment or a successful-learning status.
- Recover → retained failed attempt plus an explicitly sourced successor
  → existing capture, source freeze and rental reconciliation. A prepared Run
  cannot be rebound to later edits; historical missing comparisons remain missing.

Normal-path check: six legal actions remain six valid rows; 70 legal actions
yield rows 0–63 with six dropped, and selected row 63 executes native action 63.
Candidates and their initialization anchor use the same declared policy support.
Failure check: an unaffordable X must reject with state unchanged even if its
numeric index would fit 64 slots; increasing capacity cannot repair legality.
Recovery check: a capacity failure, absent checkpoint or refused quote supplies
no completed comparison. Preserve source/world, corpus, configuration, Commands,
attempt costs and expected schedule; use a fresh source identity for changed
code while keeping the original USD1/USD10 limits. No GPU process-recovery
capability is inferred from this workflow.

Counterexamples retained: a strategically necessary legal action can occur
after row 63 (accepted approximation, not redundancy); an over-capacity card or
effect block must still fail (action cutoff is not general observation loss);
and v5's learner updates without a trained export do not satisfy checkpoint or
learning acceptance. No new UI, resampling, automatic capacity expansion or
retry machinery follows from these cases.

Next proof remains the existing gate and execution work: preserve the directed
legality/prefix/parity/diagnostic witnesses, then use a fresh compatible quote
and correctly frozen source for bounded GPU execution, checkpoint admission,
paired evaluation and qualified analysis. This review ran no games, training,
provider queries or code tests. The companion page is
`scratch/concept-review.html`; no unresolved product questions are recorded.

## Publication validation — 2026-11-04

`lf sync` completed cleanly against main at `5dd11cec4`. The actual range includes
the configurable-buffer and selected-cost X repairs, explicit first-64 support,
prospective v6–v8 contracts and retained GPU/preflight evidence. Publication is
for those enabling changes; no completed v8 execution or learning result is claimed.

- Rust formatting and all-target/all-feature Clippy passed. The complete debug
  workspace suite (`CARGO_PROFILE_DEV_OPT_LEVEL=1`, `-j 2`) passed 957 tests,
  with two ignored. Clippy required the absolute root venv interpreter path;
  its first relative-path invocation failed before checks ran.
- Focused buffer/cutoff, pilot/discovery, active training/regimes, CI/release,
  artifact-tracking, current-world admission/monitoring, choice separability
  and package-boundary checks passed: 260 tests.
- Collection, automatic capture, cutoff, CI selection and Academy navigation
  checks passed: 105 tests (overlapping the first invocation).
- Ruff passed for changed Python modules. The updated main integration shard
  collected all 1,037 tests successfully. The new cutoff collector module is now
  explicitly included in CI. `git diff --check` passed.

Logs: `.runs/mtg213/publish-{cargo-test,python-tests,collection-tests,ci-collection}.log`.
These checks establish local behavior, not Linux CI, engine-budget comparison,
successful paid v8 execution, trained checkpoint admission or complete paired
evaluation. The prepared Run remains bound to its original source; later edits
require a successor source freeze before execution. No provider action occurred.

## Gate plan — 2026-11-04

Comparison: supplied checkout base `5dd11cec4` through HEAD `972d55673`, plus
working edits. The CI selector (`git diff --name-only -z --no-renames 5dd11cec4 |
bash .github/ci/select-tests.sh`) selects all jobs because the native engine
changed; it has no local content-keyed result cache. CI owns that full matrix.

Changed paths and focused checks:

- `managym/src/{flow,agent,python}`, `managym/cards/tests/cards/tla.rs`: X-cost
  legality, prefix transport and buffer binding. The preceding publication pass
  already ran the complete required debug suite (957 passing, two ignored),
  formatting and Clippy on these exact unchanged Rust sources. Preserve those
  results; no redundant full workspace run or extension rebuild is needed.
- `manabot/env/{env,vector_env,failures,observation,target_practice}.py`,
  `manabot/training/rollout/{collection,producers}.py`,
  `manabot/sim/native_game.py`, `manabot/infra/config.py` and native stubs:
  buffer parity, scalar/vector cutoff diagnostics, collection, automatic capture,
  active training, current-world admission/monitoring, choice navigation and
  package-boundary tests. Add regression coverage for scalar sample actors and
  vector reseed flushing before running this affected group once.
- `experiments/tla_pilot_v8.py`, analysis, v6–v8 regimes/protocols and evidence
  manifest: focused pilot/discovery/regime tests plus artifact-tracking guard.
- `.github/workflows/manabot-integration.yml`: selector/release tests and exact
  main-shard collection (the prior 1,037-test collection remains evidence; a new
  test in the existing cutoff module requires fresh collection).
- Markdown/HTML: inspect README/protocol/report consistency, `git diff --check`.
  No UI implementation changed; no new browser rendering or latency claim.

After tests, run Ruff on changed Python sources. Linux/platform jobs and
same-runner engine-budget comparison remain CI-owned, not locally certified.
Gate does not execute a GPU run or settle empirical pilot acceptance.

### Gate result

The enabling repair passes the local gate. Two diagnostic defects were fixed:
scalar cutoff samples now bind the current actor to their pending frame, with
the same decision-kind labels as vector reports; vector reseeding flushes the
retiring capture so zero-overflow decision totals survive. Regression coverage
checks the actor/frame/prefix and exact per-capture counts across two seeds.
README contracts and the report now identify v8 and the successor-source
requirement. No action selection, rules or frozen protocol/evidence changed.

- `uv run --extra dev --extra notebook --extra artifacts --extra reports --extra evaluation pytest tests/verify/test_native_buffer_capacities.py tests/training/test_action_cutoff.py tests/training/test_collection.py tests/training/test_automatic_game_capture.py tests/training/test_active_training.py tests/training/test_regimes.py tests/eval/test_tla_pilot.py tests/eval/test_experiment_files.py tests/infra/test_ci_selection.py tests/infra/test_release.py tests/infra/test_experiment_data_untracked.py tests/semantic/test_elemental_world.py tests/eval/test_evaluation_models.py tests/eval/test_mirror_admission.py tests/training/test_monitor_evaluation.py tests/model/test_choice_separability.py tests/academy/test_choice_navigation.py tests/test_package_boundaries.py -q` — 310 passed in 127.41s; `.runs/mtg213/gate-python-tests.log`.
- Exact `Integration tests` command read from `.github/workflows/manabot-integration.yml`, with `--collect-only -q` appended — 1,038 collected in 3.67s; exact command and output in `.runs/mtg213/gate-ci-collection.log`.
- `uv run --extra dev ruff check --fix` over existing changed `.py`/`.pyi` paths from `git diff --name-only --diff-filter=ACMR 5dd11cec4` — passed, no fixes; `.runs/mtg213/gate-ruff.log`.
- `git diff --check` — passed.

Native source is unchanged from `972d55673`; the prior 957-pass debug suite,
formatting, Clippy and rebuilt extension remain applicable. This pass starts no
second full native verification. The repository's CI selector correctly selects
all jobs for the branch, without a content-keyed local reuse facility.

At the final CI read, published HEAD `972d55673` had green Rust, Python unit,
Academy, protocol, semantic conformance, UI, clean-machine and package/install
checks. Main/long integration and Engine Budget were still running. CI on that
HEAD does not cover these uncommitted diagnostic edits; delivery must publish
and obtain checks for its resulting revision. No timeout or failed check was
encountered in this gate. All gate-owned processes exited; no provider action or
spending occurred. The Task remains open for its trained checkpoint, paired evaluation and
qualified learning report after delivery and a new source-bound Run.
