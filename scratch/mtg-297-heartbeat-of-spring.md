# MTG-297: Heartbeat, Animal Attendant, Toph, Uncle's Musings

Contract: `tle-production-design.md`; verbatim review and coordination:
`heartbeat-design.md`; implementation limits: `questions.md`. Detailed repair
and check history is preserved at
`ba3b05427:scratch/mtg-297-heartbeat-of-spring.md`; earlier implementation
receipts are at `8d2721ab9:scratch/mtg-297-heartbeat-of-spring.md`.

## Outcome and implementation

All four cards are defined with Scryfall ids, directed behavior tests,
observability and learning tokens. Production and spending remain in scope.
Badgermole Cub uses the production transaction; The Great Henge and Tarnished
Citadel retain their source-owned suffix behavior.

`managym/src/flow/mana.rs` captures `TappedForMana` at tap and selects live
trigger sources at production. Native unit choices preserve color, restriction,
rider, expiry and every saved caller. `ManaSpent` excludes nested activation
fees and reaches `ZoneEntry` and the entering permanent's own triggers. Toph
reads total mana spent; Uncle's Musings reads distinct non-colorless colors.
Attendant's rider applies through entry, not a new triggered ability.

The automatic first-remainder payment path is deleted. `PaymentSearch` emits
`PaymentStep::Spend` at the zero-cost transition; execution pools the whole
current cost before native unit spending. Upstream source-equivalence
memoization and supply bounds remain; `PaymentSearchState::usable` counts
allowed rider and combat classes. The earlier search regression visited 1,001
states, 1,000 already unable to meet a saved counted tap (run
`877a3189-d326-44e7-a45a-7f0aa7f63fbd`). Rejecting those states before exploring
earlier payments repaired it without changing the 1,000-state witness bound.

`Game::mana_spend_choices` stops feasibility after the first valid ordering
only when all usable units must be consumed: every ordering leaves the same
receipt and pool. Otherwise it retains the distinct feasible unit choices.
`cards/tests/cards/tle.rs` and `roe.rs` cover receipts, restrictions and expiry
with Command replay, stale rejection and forks.

Existing fact kinds 11/14 carry pools, offers and rule-relevant receipts in
v9/v10; no input shape or model kind was added. The public projection is
`pooled_mana_classes`, with absent qualifiers omitted. Its rows describe
subsets of the totals. Academy's `PlayerArea.svelte` says “including” and
`manabot/semantic/text_observation.py` says “Of the mana above” after QA found
wording that made those units appear additional. Shared encoding is covered
by `tests/semantic/test_pooled_mana_classes_input.py`.

## What the fixture failures taught us

The publication check recorded **568 passed, 75 failed, 492 skipped**
(run `9644df6c-0cbb-4ba7-b5e3-30fb71cb76f7`). Many failures passed priority
while a spending decision was pending; others inspected a stack or hand, or
replaced a pool, before casting had finished. Intermediate diagnostics ran
while repairs were arriving and do not certify a stable source revision.

Commit `eca2253e9` repairs those assumptions with explicit test-player
payments. `managym/tests/support/play.rs::spend_named` requires native
`Decision::ManaSpend`; `spend_colorless` spends ordinary step-expiring
colorless mana for funded generic costs. Hybrid pips select the intended
color; Azula selects combat red while retaining regular red. Repeated Sailor
draws preserve blue; Lesson casting completes before installing its resolution
pool. Casting, priority, mode and target helpers do not silently pay costs.
Existing outcome, nested-payment and decline assertions remain.

The repairs needed no engine behavior change. The same commit separately
moves the existing forced-payment truncation into `.take(limit)`, avoiding
unneeded feasibility searches without changing which choices survive. Shared
Heartbeat lookups and Prism's expiry selector were also consolidated.

The iteration feedback's 75-failure readiness gap is therefore historical:
stable affected-file debug runs below include those failures and the final
simplifications. This does not establish a complete CI-suite pass.

## Reused verification

Affected debug selection:

```sh
cargo nextest run --manifest-path managym/Cargo.toml --profile ci --build-jobs 2 --test-threads 2 --lib --test cards --test rules_tests --test match_state_hash_tests --test semantic_conformance_tests -E 'test(mana::tests) | test(chk::) | test(tle::) | test(roe::) | test(tla::) | test(observability::) | test(cr_106_mana) | test(cr_603_triggers) | test(cr_608_resolution) | test(cost_effect_completion_retains) | binary(match_state_hash_tests) | binary(semantic_conformance_tests)'
```

- Final selection: **643 passed, 492 skipped** in 189.825s; run
  `da6e9270-c440-4849-8560-4cd363887cd9`,
  `/tmp/mtg297-compress-native.log`. Conformance was slow at 184.694s,
  below the 300-second limit. Independent stable run
  `bbdbab2a-e47a-4e13-8bf3-c6effbf6be3b` also passed all 643 in 186.972s;
  hashes of all eight changed Rust files stayed fixed during that run.
- Clippy with `--all-targets --all-features -- -D warnings` passed
  (`/tmp/mtg297-compress-clippy.log`); workspace formatting passed.
- Release extension rebuilt and installed atomically; installed cp313 bytes
  matched the wheel. Shared pooled-class input tests: **4 passed** against
  that build (`/tmp/mtg297-compress-input.log`); no later runtime edit.
- Current-world Academy authority/parity, Learn setup and navigation:
  **29 passed**. Authored parity: 135 Commands, 136 checkpoints, 816 viewer
  checks, no divergence; stale-object/revision rejection retained.
  Public commitments: 135 Commands, 66 commitments, zero gaps.
- Semantic evidence: 14 cases, 2,715 Commands, 15 Phase rows; debug
  reproduction is included in the final selection above.
- Observation, action cutoff, native capacities and active training:
  **118 passed**. Arena admission/monitoring, choice separability and belief:
  **77 passed, one stale belief pin failed**; after refreshing only current-v3
  pins, belief state plus model world: **32 passed**. Historical receipts stay
  unchanged. Full original commands are in the archived plan.
- Frontend PlayerArea, protocol conformance and decision labels: **14 passed**.
  Changed-Python Ruff and workflow validation passed. Earlier shared-input,
  text and schema receipts remain in the archived plan.

MTG-213 corpus re-admission used original `Decklist.source` / `filename` and
preserved historical records: **998/1,000 admit**. All three Heartbeat events
(`af3d52fd84ed4bfab9e13dd56354de5a`, `55481e295a8d4f28ac505c982cd9af31`,
`d88c59ee998e456cad266dbebe4c0852`) admit. Only Lita events
`2937d550424b4ec1b44c9fc30c85c418` and `ed932f4c0dad48e3b868517f69696c4c`
fail; `docs/engine.md` matches. Lita and Release to Memory remain outside scope.

Headless Academy QA (`scratch/qa.html`) exercised all four cards, distinct
spend offers, rider qualification, Toph's receipt and reconnect at unchanged
revision 30 with identical offers. Bonus type choice, opponent-land bonuses
and Musings with graveyard targets were not reached in that browser run;
directed CHK/TLE tests cover them. No personal demo or approval occurred.
QA also noted verbose expiry/rider labels; no wording change was selected.
Spending the only usable rider units automatically is forced payment, consistent
with the shared trivial-decision rule, not an omitted strategic choice.

## Publication blocker (2026-11-12)

`lf sync` completed cleanly. Publication review compared the change with
`origin/main`; only QA-report trailing whitespace was changed, in `fc0574eec`.
The earlier selected debug run does not cover the full authored-rules check.

At `fc0574eec`, this command compiled successfully, then hit two five-minute
timeouts:

```sh
cargo nextest run --manifest-path managym/Cargo.toml --profile ci --build-jobs 2 --test-threads 2
```

- `agent::vector_env::tests::masked_mixed_setups_preserve_scalar_autoreset`:
  300.044 seconds.
- `agent::vector_env::tests::restored_game_seeds_preserve_future_autoreset_stride`:
  300.032 seconds.

Log: `/tmp/mtg297-publish-debug.log`. Both fixtures compare vector/scalar
execution over 1,000 steps. This run does not isolate a runtime regression
from fixture cost; no baseline comparison or performance acceptance occurred.
The remaining run was interrupted after the blockers were established;
nextest exited 100 and reaped its two running tests. Summary: 44 passed,
2 timed out, 2 interrupted; 1,171 of 1,219 tests not run, plus 2 excluded tests.
The interrupted tests were `shards_preserve_global_seeds_through_autoreset_and_restore`
and `terminal_step_returns_post_reset_observation`; they were not timeouts.

No PR was published. Diagnose and fix or split the timeout fixtures without
raising the watchdog or weakening their contracts, then finish debug coverage
before retrying publication. Same-runner Engine Budget remains unverified.

## Upstream and remaining work

Main's MTG-242 pool types, #504 search bounds, #506 sharding and v0.2.3
preparation are incorporated. Merge `ba3b05427` also brings in
`054e987a0`'s CPU CI dependency selection and bounded experiment smoke.
Its changes since `eca2253e9` touch workflows, the experiment fixture and
context/docs, not this Task's engine or Academy runtime. Its hosted timing
receipts belong to that CI work (`scratch/ci-under-8-minutes.md`), not proof
of this mana change's final CI result. No Wave is selected or edited here.

No further implementation mismatch or product decision was identified.
The fixed multi-type source and tap-predicate snapshot limitations remain
explicit in `questions.md`; neither is exercised by current content.
The deletion list in the contract is complete.

Remaining: review/gate and delivery, including the final revision's full debug
CI coverage and same-runner Engine Budget required by `docs/development.md`.
The 643-test selection is affected-file evidence, not the whole suite; no
performance acceptance or baseline update is inferred. A human demo remains
unperformed, without blocking headless checks. Publishing, landing and Task
completion belong to the caller.

Realignment changed context only and reused the recorded runs; no tests were
rerun. Rechecked the final debug and Clippy log summaries and the repair diff.

## QA re-run on ba3b05427 (2026-11-12)

Academy, headless (`scratch/qa.html`, top section). On the final extension:
reload during a spend kept the revision and offers; Musings returned a
graveyard Toph; both seats ran Heartbeat through to Game Over; no console
errors. Found automatic payment tapping lands in battlefield order beyond the
cost (Toph with 2 {G} floating left 2 {R} · 10 {G}). Main's build taps the
same way, so this is not a regression. Filed as MTG-342.
