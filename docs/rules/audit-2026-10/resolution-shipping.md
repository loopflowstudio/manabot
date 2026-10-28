# Resolution repair shipping acceptance — 2026-10-28

The repair fixes a target that gains hexproof during resolution incorrectly
escaping later instructions, and suppresses every instruction of an activated
ability when all its chosen targets are illegal at resolution entry. Spells,
activated abilities and triggers share one clause-specific authority. Suspended
frames retain its result; separate presence checks prevent following an object
into a new incarnation. Illegal slots keep their positions.

World w15/hash 10 integrates MTG-195's controller-selected trigger placement.
Policy/capture/text v4 carries retained eligibility. V2/v3 remain 32 columns;
v4 adds column 32, with validity still at column 31. Earlier checkpoints,
captures, corpora and experiment evidence retain their identities. No payment
ownership or authored card file changes are included.

Main `c5d406e3d` was integrated through supported `lf sync` recovery before
acceptance. Its evaluation changes were preserved; the only conflict was in
scratch assumptions, and both sets were retained. The comparison is the merge
base with current main, including working edits, not a two-tree subtraction.

## Acceptance

The local gate passed. [The receipt](resolution-shipping.json) retains exact
commands, source fingerprints, native wheel/extension hashes, results and failed
attempts. Python used the locked 3.12 environment with dev, notebook, artifacts,
reports, play and evaluation dependencies; test execution used two OpenMP threads.
Recorded Cargo acceptance commands used two jobs and ran one at a time. An early
dependency setup unexpectedly started another native build during debug tests;
that setup was stopped, dependencies were installed without the project, and the
final wheel was built separately. No acceptance result came from the stopped setup.

| Check | Result |
|---|---|
| `cargo test -j 2` with CI's debug optimization level 1 | 558 passed; two known proposals ignored; assertions and overflow checks retained |
| Resolution tests after test-only iterator cleanup | 8 passed; unchanged native suites retain their earlier result |
| `cargo fmt --check`; all-target/all-feature clippy with `-D warnings` | Passed |
| Release `maturin build`; atomic cp312 extension installation | Passed; wheel and loaded source-extension hashes retained |
| Model, observation, continuation and text suites | 253 passed, one unsupported attention/value-token combination skipped; 15 continuation cases passed again after formatting |
| World, admission, monitor and active-training suites | 58 passed across the initial run and the focused monitor-fixture correction |
| Shared inference | 17 passed, including explicit v1/v2/v3/v4 served checkpoints; one CUDA-only case skipped |
| Command, reconnect, stale rejection, optional-payment recovery, replay/Study, full choice navigation, Learn setup and trigger ordering | 85 passed |
| Training and game-error recovery | 26 passed |
| Local evaluation, including the actual offline CLI and crash/replay continuation | 19 passed |
| Inventory and CI-selection tests | 46 passed; offline index verification accounts for all 3,322 entries |
| Frontend check and build | Passed; zero Svelte errors/warnings |
| Focused browser proof | Eight choice-navigation/Learn cases passed on the ordinary server; two trigger-order cases passed on the built release fixture harness |
| Ruff lint/format on changed Python files; whitespace | Passed; only the changed continuation test was rerun after formatting |

Both browser runs used isolated ports and stores. Their servers stopped after
the checks. No subjective visual assessment is inferred from headless tests.

## Repaired failures and incomplete attempts

- The CR 608 fixture still named MTG-195's removed `enqueue_order` field. It now
  supplies the current trigger identity and ordinary ordering class.
- Clippy found three fixed-width iterator suggestions in the new resolution
  test. The assertions are unchanged; the final test and lint runs passed.
- Monitoring rejected a test policy with v2 input paired with a fresh v4 runtime
  fingerprint. Ordinary fixtures now use the current `ObservationConfig`;
  historical versions remain explicit. Runtime rejection was preserved.
- The first trigger browser invocation used the ordinary server, which correctly
  ignored the test-only root flag. The authored release configuration supplies
  `trigger_order_server.py`; both ordering outcomes and reconnect then passed.
- An unoptimized debug run was interrupted after passing the rules suite because
  search-state contracts were costly. The complete replacement run used CI's
  documented debug optimization level 1, with no assertions disabled.
- A 300-second aggregate local-evaluation wrapper expired after 18 passes, at
  `test_incremental_replay_retains_first_block_when_later_game_crashes`. That
  aggregate bound was shorter than the module's CLI proof's own 360-second
  allowance, before accounting for other cases. The unchanged module completed
  under a 600-second wrapper: the final replay case took 70.01 seconds, the
  offline CLI 65.54 seconds, and the whole pytest run 213.28 seconds. The original
  timeout remains in the receipt; no inner deadline or assertion was relaxed.

The measured command durations sum to **57.30 local minutes**, including failed,
interrupted and timed-out attempts. This sum is neither total gate wall time nor
CI runner cost, and excludes unmeasured provisioning. The shared host's observed
load average reached 259.74; no contention was induced. These are uncontrolled
correctness timings, not a before/after speedup claim. No sharding or required-CI
coverage change is introduced by this shipping gate.

## Checks still owned elsewhere at publication

The path selector requests all nine existing CI groups. Required Linux CI,
including its full integration, protocol, semantic coverage/conformance, frontend
and clean-machine checks, must pass on the published head through waited landing.
The PR's checks are the authority for those results. The broader macOS nightly
matrix is not certified by this local subset. CUDA shared inference still needs
a CUDA runner; the CPU CI matrix cannot certify that skipped case. No paid
compute allocation is selected to obtain it.

## Scope and limits

Jack Heart authorized publication and waited landing on October 28. This is a
new delivery after historical PR354, which remains unchanged. MTG-177 stays
open: 2,840 rules entries remain unknown, and the deathtouch-history and
earthbend-destination probes remain ignored, unresolved proposals. Synthetic
resolution fixtures demonstrate obligations, not admitted-deck reachability.
This delivery establishes no playing-strength baseline or release certification.

The [line accounting](line-accounting.json) retains **10,430 authored churn
lines**, including 118 for this gate and its temporary bounded-check helper.
That is 430 above the approximate 10k guideline; generated report/evidence and
inherited main changes are accounted separately. The [CI analysis](ci.md) remains
a dated observational study; this gate does not rerun a performance experiment
or claim measured runner savings. No paid compute or artificial load is used.
