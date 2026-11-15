# MTG-350 — bounded mana payment work

## Current outcome (2026-11-15)

The shared reservation/life repair is implemented on the Task base. Both new
regressions fail on the unchanged base and pass with the repair in debug.
The retained experimental trajectory remains an integration dependency; the
four audit workloads were measured, retain identical witnesses/work/digests,
and do **not** meet the 2x speed target. This is not Task completion. No cache
or new search bound is implemented. Continue with the remaining work below.

Jack Heart, 2026-11-15: "launch an auto task stacked on this to fix the yue bug?"
The Task comment separately requires "Keep legal actions, first witnesses and
the Engine Budget digests unchanged". No Wave is selected. The complete prior
design, contradictory exact-replay requirement, diagnosis and concept review
are retained in git at `592f499546:scratch/mtg-350-speed-up-the.md`.

## Contract and implementation

An offer must preserve completion of the announced payment and every saved
caller. `Payment` is the announced work; `PendingChoice::Waterbend` owns the
remaining cost and committed tap count. `Decision` / `SpendCompletion` own
nested callers. Reservations and ordered obligations are derived planning
inputs, never a parallel serialized payment lifecycle.

`payment_model` now reads the announced payment once for ManaPayment,
SacrificePayment and Waterbend. A shared internal helper reserves its source
and unpaid life fee; the same helper follows an announced payment held in
`SpendCompletion::Announced`, including nested Upfront/ManaChoice callers.
ManaUpfront life fees remain Output obligations, so they are not debited a
second time. Actual cost commits and caller restoration use the existing path.
No browser operation, automatic source preference, cancellation, fallback
action, policy layout, world identity or replay-schema change is introduced.
The existing README payment contract remains accurate; shipping docs unchanged.

Directed proof: Yue owes her own tap and waterbend {5}. With one floating mana
and Prism plus three Ogres, tap two Ogres, leaving {3}. Activating Prism spends
one and returns one, taps Prism and leaves only one Ogre. The remaining pool
plus that Ogre pays two, not three: Yue's promised tap cannot be spent twice.
The repaired authority removes this infeasible offer. With two floating mana,
the same Prism activation remains legal and completes. Both continuations
retain every previous tap, consume the complete pool, leave all permanents in
play and reach a legitimate decision with native trivial skipping on. Commands
at intermediate payments verify exact forks, source immutability and stale
rejection. Empty-hand Yue resolves through the forced decisions.

A synthetic payment-model fixture isolates a two-life announced fee with
Sunbaked Canyon's one-life mana fee. At two life the nested activation is
infeasible; at four life it is payable. The announced fee survives Waterbend,
spending and a saved upfront caller. Execution and a fork pay Canyon once,
then the announcement once, leave one life and no pooled mana, and complete
without an empty pending decision. Existing nested Prism/Treasure and
sacrifice/suspended-resolution regressions also pass.

## Frozen evidence and experimental integration dependency

Original engine source: `fd7a7ed36c21553d419504daafeb36f70b3ab4ed`.
Study source: `3b519a3e94edd5570d240e606d636fb9d358fd96`.
Required flags before import: `MANAGYM_EXPERIMENT_COMBAT_PICK=0`
`MANAGYM_EXPERIMENT_SOURCE_TAPS=1`.

Retained stream 24 / seed 37037 / input 196 (Add {G}, actor 0, turn 25 precombat main)
previously reproduced on a CPython 3.13 rebuild of the unchanged failing source.
It matched the complete original frame, fingerprint
`9b85ba019f7270c05cea9aa92779b40915bfc3a4a6febdc113aa305933880b91`, in
0.045749 s after reset, then exceeded the approximately 58-second subprocess
bound and was killed/reaped. This was **changed-runtime reproduction**, not
the original private binary. Rebuilt extension SHA256:
`9458a2ae4652d8098d9439740aafc88713a80dec29f161bc9b2356f955ad14ec`.

The diagnostic found an empty Waterbend space, not a stuck witness: remaining
{3}, tapped 2, announced Yue permanent 21, pool G, no suspended decision. Native
trivial skipping repeatedly executed no action. A source-reservation-only
snapshot diagnostic first changed offer 193 from one mana activation plus two
waterbend taps to two waterbend taps. Its continued positional inputs after
that divergence were **not** a corrected continuation and prove no fix of the
original game. The snapshot is diagnostic only, never shipping candidate code.

The Task base lacks source taps and automatic fixed-source preference. The
prior source-tap-only prerequisite import diverged at input 70 (input 2 versus
two priority actions), was removed, and is not part of the repair. Its inventory
was TapForMana, action/offer/focus projection, setup switch, fixed_tap_actions,
tap_for_mana_action, payment entry/dispatch and Mana::spendable_units plus its
test. Do not import another owner's source-preference work to force a match.

2026-11-15 implementation read: PRs #524, #538 and #540 remain OPEN/unmerged,
respectively at `b1ecacd974`, `67ce2568da`, `7b8707e806`. No compatible owned
prerequisite was obtained. Full candidate replay on the experimental path,
first-divergence identity proof, corrected continuation and the isolated-step
release target under 1 s remain unverified. The base-directed proof above does
not establish the exact retained removed activation's identity or feasibility.

Keep the original receipt/context bytes and runtime bindings unchanged:

- `scratch/yue-stall-receipt.jsonl`: SHA256
  `2a575ed1eba230fda450c2be7534182f6ba663922a35ecbeda9a963bf8d9c007`.
- `scratch/stream-24-context.json`: SHA256
  `120d22c30608762ac8f01e11d44fc906dc0ba5a58179c011cb3b1075458bc387`.

Once a compatible prerequisite is available, compare original and candidate
frames before every input; stop at the first changed offer, retain the common
prefix and both frames, identify choices by actor/kind/source/ability/parameters,
and select a legal continuation in the candidate's own frame/binding. Prove
removed offers infeasible from all resources and record candidate provenance
and new inputs separately. Never consume shifted numeric inputs or weaken
runtime admission / final-frame binding checks. A snapshot-only patch is not
Task completion. No paid training, rental, active-job control, population
change, deck exclusion, or merge of PRs #523/#524/#538/#540 is authorized here.

## Audit evidence — reservation repair only

Same checkout/machine, release benchmark feature; baseline production is
`592f499546` (new cfg(test) regressions do not enter the executable). Each row
is the harness median of three samples. Commands: build `bench_states` with
`CARGO_BUILD_JOBS=2 cargo build --release --features benchmark --example bench_states --manifest-path managym/Cargo.toml`,
then run the baseline executable and repaired executable consecutively with
`counted-tap diverse-mana mixed --sizes 16 --layers search`, followed by
`restricted-filter --sizes 4 --layers search`. All checkpoint rows were checked.

| Required cell/checkpoint | Before ms | Repair ms | Before/repair | Plans / calls / states (both) |
|---|---:|---:|---:|---|
| counted-tap 16, cast-choice | 1131.268 | 1934.778 | 0.58x | 8296 / 515556 / 478448 |
| diverse-mana 16, cast-choice | 392.008 | 763.329 | 0.51x | 3159 / 91915 / 80006 |
| mixed 16, cast-choice | 332.931 | 611.224 | 0.54x | 10213 / 51479 / 41282 |
| restricted-filter 4, priority | 234.289 | 236.766 | 0.99x | 1969 / 51900 / 23089 |

Every checkpoint has identical plans, calls, states, maximum calls, layer
counters, simulation counts and first-witness digest. Closing digests agree:
size 16 group `ee476afb928628d61f01469a7de6fe2eefa95121bdd1006ca1a270c3faa1abb3`;
restricted-filter 4 `adecfb769e039154e870dc7d36956035cb8c772ff25058ea5718ef70f6d43ecb`.
These are stress-harness digests, not an Engine Budget CI result.

Timing is unstable: a preceding quiescent pair measured counted-tap
1606.328/3144.517ms, diverse-mana 392.477/1025.662ms,
mixed 274.651/729.958ms and restricted-filter 106.478/326.195ms. Even
zero-payment-call rows moved (mixed block 14.960/37.442ms, then 16.509/25.402ms).
The rerun was justified by that unresolved timing difference. Do not attribute
all of it to this repair or claim no regression. No 2x target passes; the
slowdown needs controlled comparison/CI before publishing. An initial baseline
run overlapped a compiler and was discarded; the table uses subsequent
sequential runs, with no compiler/test runner during measurement.

The counts support pursuing repeated-plan reuse: counted-tap's largest single
plan uses 67 calls, mixed 13, despite thousands of plans. Restricted-filter's
priority plan reaches 2375 calls and still needs a safe nested-payment bound.
Local raw JSON lives in ignored `.lf/tmp/mtg350/audit-*.jsonl`; the measured
facts needed for later work are retained here.

## Remaining implementation

### Concept review, pass 2 (2026-11-15)

Design changed: any repeated-plan cache must identify the effective query
start, including a selected counted tap, rather than only the model and
unmodified obligations. `plan_after_tap` consumes a particular permanent's
untapped resource and decrements the first Tap obligation before searching.
Two choices on the same board can therefore have the same previously proposed
key but different completions. A tap that consumes the only remaining useful
producer is the counterexample; tapping an otherwise interchangeable helper
must still be offered when it completes the caller.

Next implement direction: preserve the current shared announcement repair;
diagnose timing before adding reuse. If reuse pays, key the complete ordered
model, effective initial resources/pool/life/ordered obligations/prefix budget,
and SourcePolicy. Either include the selected tap explicitly or derive that
effective start once; never reuse an ordinary plan for an after-tap query by
its unmodified arguments. Store complete first witnesses and negatives, with
debug recomputation against the same uncached query. Keep the existing exact
fork, mutation, retention and overhead requirements below. This clarifies a
derived query, adding no payment lifecycle, public API or persisted cache.

Normal play still offers every completion-preserving choice. An infeasible
branch disappears only with a resource proof; the nearby payable Prism branch
stays. Fork/reconnect retain committed taps, fees and the actual pending caller.
The base-directed regressions support that contract; they do not verify the
retained experimental game's first divergence. Preserve frozen evidence and
record a corrected continuation separately when the prerequisite is available.

No open product choice was found. The README's shared payment contract needs
no change. The four performance targets and experimental integration remain
implementation work, not reasons to redesign choices or weaken binding checks.

1. Resolve the experimental integration dependency as above; no source-only
   snapshot result can replace candidate integration.
2. Diagnose the observed timing difference. Then pursue the four original
   audit targets (2x lower median each); report misses and retain all legal
   choices, concrete first witnesses and Engine Budget digests. Do not update
   a moved budget digest automatically.
3. Nested-filter pruning: a conservative feasibility bound alongside existing
   bounds must account for upfront spending purposes and saved outputs. It
   may overestimate supply, never reject a payable branch. Compare complete
   first witnesses with an unpruned bounded reference. Keep
   `payment_bound_allows_restricted_mana_to_fund_a_filter`: subtracting fees
   from gross output is unsafe when restricted mana funds a filter.
4. Repeated plans: only if measurements support it, a per-Game exact-query
   cache, initially 64 FIFO entries, copied by value on Game clone. Key the
   complete ordered PaymentModel (source identities, abilities, outputs,
   restrictions, current resources, life), pool, ordered obligations and
   SourcePolicy. Store negatives and complete witnesses, compare values rather
   than a digest alone, and debug-recompute hits. Test pool/tap/life/caller
   mutations, source restrictions and exact forks. Report lookup/key cost,
   retained bytes and hit rates; remove it if key overhead outweighs reuse.
   Revisions alone omit pending callers and life. This cannot bound a first
   pathological query. No global cache, cutoff or fallback legality answer.
5. Gate owns CI and broad debug nextest/clippy. Current implementation has
   only run focused checks; no full suite or Engine Budget pass is claimed.
   Main has moved 24 commits past base, three touching `flow/mana.rs`
   (f6f99d91b7, 9632914976, b5478d8789); parent PR #523 lands first, then
   rebase and rerun the directed tests. PRs #523/#524/#538/#540 remain OPEN.

## Delete — do not maintain

The duplicated pending announced-payment source/life readers are replaced by
one shared internal helper. Do not retain a parallel payer, serialized
reservation state, cache prototype, fallback action, local stack probe or
source-tap prerequisite import. Preserve receipts, caller ownership, exact
cost/resource semantics and forced-decision skipping. The shared model and
original replay binding authority survive. No schema/data migration.

## Checks

- Before repair: both new directed debug tests FAIL on the old production
  code (infeasible Prism offer retained; Canyon offer exceeds combined life).
- `CARGO_BUILD_JOBS=2 cargo test --manifest-path managym/Cargo.toml -p managym --lib flow::mana -- --test-threads 2` — 28 passed, 0.48 s.
- Cards target filter `yue_reserves_her_tap` — 1 passed, 0.05 s; both fixture
  branches use native skipping and exact fork/stale-Command checks.
- Cards filter `committed_optional_payment_resumes_after_two_nested_mana_decisions` — 1 passed.
- Lib filter `retained_batch_linked_entry_suspends_actual_mana_cost_completion` — 1 passed.
- Release stress audit — all checkpoint witnesses/work/digests equal; 2x targets missed; timing regression unresolved as above.
- CPython 3.13.15 release wheel rebuilt from root, matching extension replaced
  atomically; native import passed. Candidate extension SHA256:
  `90793465845dd1292bb803b220b1238c03011a5dcc2dc03698c52dc4d2a61cc9`.
- `uv run --extra dev pytest tests/training/test_game_errors.py tests/model/test_record_coverage.py -q`
  — 33 passed, 3 failed in 24.55 s. All record-coverage tests passed. The three
  variants of `test_resource_limit_receipt_replays_its_retained_encoding_boundary`
  (ObservationCapacity, AgentError, None) expected `failed` but received
  `binding_mismatch` at line 190. A separate root rebuild of unchanged HEAD
  production code reproduces all three failures (1.21 s). This failure
  predates the repair; no fixture or binding check was changed. The candidate
  source and extension were restored afterward. Main's 452d7b5357 (not in
  base `ddc337f764`) renames that test and asserts `binding_mismatch` for the
  schema-7 receipt, so it resolves when the stack rebases; nothing to fix here.
- Workspace `cargo fmt --manifest-path managym/Cargo.toml --all --check`
  and `git diff --check` passed. `lf context`: scratch within both limits.
