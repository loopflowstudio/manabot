# MTG-350 — bounded mana payment work

## Current outcome (2026-11-15)

The reservation/life repair is preserved. The latest pass reuses owned mana
output modes and skips zero-use output scans. Latest sequential local speedups
against production baseline are 3.50x counted-tap, 2.72x diverse-mana, 2.03x
mixed and 2.91x restricted-filter, with matching witnesses and stress digests.
All four local 2x targets pass in both comparison orders; this is not Engine
Budget CI. The retained experimental trajectory still lacks a compatible
owned prerequisite, so its continuation and subsecond target remain unverified.
This is not Task completion.

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

## Performance implementation (2026-11-15)

`PaymentCache` belongs to Game, retains 64 FIFO searched queries, and copies
its FIFO by value on Game clone, sharing only immutable Arc query entries. It stores negatives and complete first witnesses.
Equality covers ordered sources (identities, abilities, restrictions, outputs,
untapped state), payable life, pool classes, ordered obligations, SourcePolicy
and selected counted-tap identity. Initial resources and prefix minimization
are deterministic from these inputs. Selected taps never alias ordinary plans
or another selected permanent. Debug hits recompute the same uncached query.
Arithmetic/direct plans bypass the cache; they do not justify full model-key
retention. Lookups inspect newest entries first without changing FIFO eviction.
Queries and keys share immutable ordered source lists from the existing
PaymentScan; reservations detach with Arc::make_mut. Value equality remains the
authority when source lists have different allocations. Source/tap/restriction
mutation tests also exercise this detachment. No mutable source list is shared
between keys or forks. The cache is outside persistence, hashes, worlds,
offers and policy inputs.

The conservation bound counts *all* classes, including restricted seeds that
cannot pay the parent, and adds max(0, gross output - upfront mana fee) for each
remaining possible activation. It overcounts competing non-tapping resources
and ignores life fees, so supply remains optimistic. Already committed saved
outputs contribute gross mana while their explicit Pay obligations remain owed.
This is separate from the existing purpose-aware gross bound: subtracting fees
from that original bound would reject payable restricted-seed filters.
Automatic-prefix budgets begin at the optimistic number of largest fixed/tap
outputs needed to cover the first cost's pool shortfall; no feasible budget is
skipped. Zero-cost Pay, completed Tap and single Output transitions advance
before repeated full-board bounds, keeping Spend/Output witness steps and life
accounting. The old late branches are removed. Limited-ability use flags are a sorted
sparse list of exact (source, ability) pairs instead of one allocated flag row
per source per state. Canonical memo keys preserve the former full flag-row
ordering; all 64 use combinations on three two-ability sources compare complete
grouped and identity witnesses. No limit on ability count is introduced.

Tests cover selected useful-producer versus helper taps, source order and
restrictions, tap/pool/life/caller changes, negatives, bounded FIFO retention,
independent exact Game forks and unchanged bindings. Small filter/pool/cost
combinations compare complete witnesses against the gross-only reference.
Prefix budgets compare with every original budget from zero. Forced internal
transitions retain their exact expected steps and reject insufficient life.
The README describes these surviving paths.

Ownership counterexample: the deep-copy implementation moved mixed block from
18.066 to 46.156 ms with zero payment calls. Its 203,776-byte full cache copy
was needless work on every fork. Storage is revised to Arc-owned immutable
entries inside an independently copied FIFO; insert/evict and Game mutation
remain independent. No shared mutable cache or new rules representation.

Timing diagnosis: the existing baseline and reservation-only executables were
run in both orders without any compiler or test runner. First pair's cast-choice
medians were baseline/repair 979.713/992.204 ms counted, 345.957/352.896 diverse,
256.101/280.962 mixed. Reversed pair still drifted. The earlier near-2x repair
slowdown is not reproducible as a stable code difference; no particular host
cause is established. Use sequential matched pairs and deterministic witnesses,
not the old absolute timings, for decisions.

Rejected implementation: optimistic activation-output reachability preserved
witnesses but reduced restricted priority calls only from 34,306 to 33,825,
while its pair measured 230.127 ms versus the cache-only 80.383 ms from the
preceding pair. Diverse also grew from 354.223 to 906.384 ms. Timing is noisy,
but the extra closure work had little pruning value. Its code and exclusive
fixture were removed; no dormant prototype is retained.

## Performance measurement before the allocation pass

Production baseline `592f499546`; final candidate is this diff. Same release
`benchmark` feature, machine and commands as the earlier audit, baseline then
candidate, no compiler/test runner during timing. Each cell is the harness
median of three actual workloads; raw rows are ignored
`.lf/tmp/mtg350/sparse-{base,candidate}-{16,4}.jsonl`.

| Required cell | Before ms | Candidate ms | Speedup | Cache hits / queried searches | Calls / expanded states |
|---|---:|---:|---:|---:|---:|
| counted-tap 16, cast-choice | 1134.397 | 326.998 | 3.47x | 4352 / 8296 (52.46%) | 240254 / 166804 |
| diverse-mana 16, cast-choice | 397.192 | 268.944 | 1.48x | 302 / 3157 (9.57%) | 82869 / 56149 |
| mixed 16, cast-choice | 299.451 | 216.858 | 1.38x | 2611 / 10197 (25.61%) | 36636 / 18942 |
| restricted-filter 4, priority | 114.469 | 73.580 | 1.56x | 382 / 1960 (19.49%) | 38401 / 13507 |

Only counted-tap meets 2x. Restricted priority's largest plan rises from 2375
to 2918 calls as forced transitions leave the expansion memo; total calls and
expanded states fall. Do not claim a bound on a first pathological query.
All 14 checkpoint rows match ordered first-witness digests, plan counts,
non-payment layer counters, simulation counts and cap hits. The two closing
digests remain exactly those recorded above. Search calls/states intentionally
change; these stress digests do not substitute for Engine Budget CI.
Mixed zero-payment-call block returns to parity (16.302/16.306 ms), versus the
rejected deep-copy cache's >2x cost. No other checkpoint has a material measured
regression in this final pair; noisy wall time still needs CI confirmation.

Before the allocation pass, standalone `payment_cache_profile` (release, benchmark, one test thread):
64 entries / 34 sources retain 29,952 newly materialized owned bytes including
the unique shared source payload, with 200 allocations; a fork adds 512 bytes.
This is a fixture measurement, not a global heap bound. Latest-key lookup,
full-key equality and witness cloning median 333 ns versus uncached search
30,833 ns (three samples). This excludes model construction and is not an
average miss cost. Lower reuse and remaining allocation work keep three targets
open; retain the cache because it materially improves the counted workload.

A one-second `/usr/bin/sample` of one real mixed search workload (not a timing
sample) collected 750 samples. Top-of-stack allocation/free symbols remain
prominent (`_xzm_free` 121); source reservation copies, obligation/pool clones,
restriction sets and hashing appear in the tree. This suggests allocation work
for the next investigation, not a quantitative attribution or permission for
an unsafe bound. Raw diagnostic: `.lf/tmp/mtg350/profile-mixed.txt`.

## Allocation implementation pass (2026-11-15)

Output alternatives and counted-tap candidate lists now share immutable payloads
between source abilities, obligations, memo states and cache entries. Every
comparison still uses complete values; branch progress, pool, life and resource
flags stay owned. The new directed mutation fixture changes one source's blue
output to green, verifies the old blue query fails and green succeeds, then
restores the source and gets the original cached blue witness. Reservations
check for an actual change before detaching the source list. The branching
match borrows its first obligation; output and activation branches construct
only the fields they need instead of cloning and overwriting whole states.
No bound or source/branch order changed in this pass.

`bench_states --allocations` is a separate one-operation diagnostic, documented
in development.md; timing mode still takes three samples with counting off.
It counts only the operation, excluding setup/root restoration. Actual workload
allocation traffic before/after **this pass** (both include the prior cache,
bounds and reservation repair; this is not the production baseline):

| Cell | Allocations before → after | Requested bytes before → after |
|---|---:|---:|
| counted-tap 16 cast-choice | 2,749,273 → 2,044,909 | 405,945,836 → 332,438,476 |
| diverse-mana 16 cast-choice | 3,995,558 → 3,263,096 | 518,216,026 → 441,801,544 |
| mixed 16 cast-choice | 5,738,907 → 3,989,509 | 664,480,854 → 465,827,236 |
| restricted-filter 4 priority | 1,563,135 → 1,001,763 | 191,665,077 → 133,339,247 |

These are allocation requests/traffic, not live heap or a global retention
bound. Mixed zero-payment block is exactly unchanged: 331,603 allocations and
71,848,920 bytes. All 14 rows have identical work counters and first witnesses,
and both closing stress digests agree. Raw diagnostic rows are ignored
`.lf/tmp/mtg350/traffic-{before,after}-{16,4}.jsonl`; before source is the prior
pass's final diff, captured as `before-transient.rs` there. The allocation-mode
harness is identical in both executables.

Intermediate timing: removing duplicate transient copies alone improved required
cells by 3–6% in its sequential pair. A first shared-payload pair was inconclusive
(mixed zero-payment block 15.453 → 46.834 ms). The justified reverse-order pair
had a steady 13.894/13.920 ms control and payload/transient cast choices of
217.022/228.302 ms diverse and 151.714/183.130 ms mixed; restricted priority
51.838/57.861 ms. Inline resource storage added no useful gain (counted cast
274.781 → 292.034 ms), and its code, exclusive spill fixture and dependency
were removed. Keep no dormant inline path.

A subsequent production-baseline/final pair measured counted 1270.890/408.373,
diverse 470.717/331.928, mixed 345.780/236.522 and restricted
142.157/74.486 ms (3.11x/1.42x/1.46x/1.91x); zero-payment mixed block drifted
17.928/21.585 ms. No target other than counted passes in that pair. This control
drift justifies the reverse-order final check below, not a host-cause claim.
Raw `.lf/tmp/mtg350/allocation-{base,final}-{16,4}.jsonl`.

Final justified reverse-order comparison: final then production baseline
`592f499546`, then this pass's before executable, sequentially on one machine,
with no compiler/test runner. Each row is a median of three actual workloads.
All 14 checkpoint witnesses, plan counts, non-payment counters and simulation
results match across all three; search work is unchanged within this pass.
Both closing stress digests still match the frozen values above.

| Required cell | Production baseline ms | Before this pass ms | Final ms | Baseline / final |
|---|---:|---:|---:|---:|
| counted-tap 16 cast-choice | 1003.259 | 283.033 | 269.737 | 3.72x |
| diverse-mana 16 cast-choice | 342.366 | 230.606 | 214.680 | 1.59x |
| mixed 16 cast-choice | 255.721 | 181.651 | 155.261 | 1.65x |
| restricted-filter 4 priority | 103.444 | 61.581 | 49.694 | 2.08x |

Mixed zero-payment block is steady: 13.938 / 13.814 / 14.032 ms in the same
column order. This supports retaining the allocation reductions: this pass
improves the required cells by 5%, 7%, 17% and 24% respectively without a fork
regression. Counted exceeds 2x; restricted crosses it in this pair but was 1.91x
in the preceding pair. Diverse and mixed still miss. These local stress results
are not Engine Budget CI or proof of a general bound. Raw rows:
`.lf/tmp/mtg350/allocation-reverse-{base,before,final}-{16,4}.jsonl`.

Dependency re-read: PRs #524/#538/#540 remain OPEN at the same heads on
2026-11-15. No compatible owned experimental prerequisite appeared, and none
was imported. Retained trajectory integration and the subsecond step target
remain unverified; the frozen receipts are unchanged.

### Allocation-pass checks

- `CARGO_BUILD_JOBS=2 cargo test --manifest-path managym/Cargo.toml -p managym --lib flow::mana -- --test-threads 2` — 34 passed, 0.67 s, debug on final code.
- `CARGO_BUILD_JOBS=2 cargo test --manifest-path managym/Cargo.toml -p managym-cards --test cards yue_reserves_her_tap -- --test-threads 2` — 1 passed, 0.02 s, debug.
- Release `bench_states` four-cell sequential pairs and separate `--allocations` runs — all 14 checkpoint witnesses and both closing digests match; detailed comparisons above. Counting and timing never overlap a compiler/test runner.
- Root `CARGO_BUILD_JOBS=2 uv run --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels` — built cp313 macOS arm64 wheel; matching extension replaced atomically; native import passed. SHA256 `332da42877aa9c0211d6cd8f5d859924fde3acd667976a4bcb915abc9352f37f`.
- `uv run --extra dev pytest tests/model/test_record_coverage.py -q` — 4 passed, 3.28 s.
- Workspace `cargo fmt --manifest-path managym/Cargo.toml --all --check`, `git diff --check`, frozen receipt/context SHA256 checks and `lf context` — passed; no broad-suite/Engine Budget claim.

## Projected-bound and memo pass (2026-11-15)

The per-cost upper bound now projects allowed classes onto ordinary color
counts after taking each class's maxima across output modes. Saturating sums
preserve the prior bound's color feasibility; no class metadata is copied.
Concrete search pools, spending units and exact Game-cache keys retain every
class. A directed test compares the original class-preserving bound across
ordinary/restricted/combat modes, differing purposes, hybrids and saturation.
Zero-use abilities need no optimistic output, and a pool already covering the
current cost needs no per-cost production upper bound.

The live branching state is moved into an immutable Arc and retained as its
own memo key. Identity keys and already canonical groups share it; a resource
permutation detaches through Arc::make_mut. Concrete children own their
mutations. This removes the independent memo copy of pool classes, resources
and obligations, without changing canonical equality or source order.

The existing search requires the whole current cost pooled before spending.
When it cannot be paid, generating each first unit only to reject the unpaid
remainder is unnecessary. One can_pay check now skips those doomed branches;
for funded pools, resource/obligation copies wait until the spent pool remainder
is payable. Native execution's complete spending offers are unchanged.

Actual workload allocation traffic, **before this pass → final**, both with
the reservation repair and prior optimizations (not production baseline):

| Cell | Allocations | Requested bytes |
|---|---:|---:|
| counted-tap 16 cast-choice | 2,044,909 → 1,778,149 | 332,438,476 → 257,592,036 |
| diverse-mana 16 cast-choice | 3,263,096 → 1,246,902 | 441,801,544 → 172,962,946 |
| mixed 16 cast-choice | 3,989,509 → 3,590,140 | 465,827,236 → 407,148,620 |
| restricted-filter 4 priority | 1,001,763 → 636,492 | 133,339,247 → 85,706,891 |

Mixed zero-payment block stays exactly 331,603 allocations / 71,848,920 bytes.
These are allocation requests/traffic, not live heap. Isolated source snapshots
and binaries under ignored `.lf/tmp/mtg350/` separate each cut:
`before-projected-bound.rs`, `projected-bound.rs`, `memo-bound.rs`,
`pooling-bound.rs`; `projected-*`, `memo-*` and `pooling-*` JSON rows.
Projection alone reduced diverse to 2,056,072 allocations; memo sharing reduced
it further to 1,491,082; rejecting incomplete-pool spending reduced it to the
final 1,246,902. Corresponding mixed counts were 3,665,206 → 3,591,608 →
3,590,140. Thus pool/metadata and memo work materially help diverse, while
mixed has much less allocation traffic in those cuts. All intermediate work
counters, checkpoint witnesses and closing stress digests match.

Intermediate timing pairs were confounded by unchanged-control drift: mixed
zero-payment block moved 37.777 → 22.801 ms (projection), 40.407 → 18.427 ms
(memo), then 17.966 → 28.174 ms (partial spending). They do not isolate speedups
or establish a host cause. Retention relies on lower allocation traffic and the
final sequential comparison below. No compiler/test runner ran during any
measurement, and counting was separate from timing.

Final comparison order: final candidate, production baseline `592f499546`,
then before-this-pass executable, size-16 group followed by restricted size 4.
Same machine, release benchmark feature, each row median of three actual
workloads. Commands use the same `--layers search` audit cells as above.

| Required cell | Production baseline ms | Before this pass ms | Final ms | Baseline / final |
|---|---:|---:|---:|---:|
| counted-tap 16 cast-choice | 994.720 | 285.915 | 276.404 | 3.60x |
| diverse-mana 16 cast-choice | 362.651 | 227.706 | 144.251 | 2.51x |
| mixed 16 cast-choice | 280.732 | 163.113 | 141.627 | 1.98x |
| restricted-filter 4 priority | 106.696 | 55.707 | 41.878 | 2.55x |

Mixed zero-payment block is 14.916 / 15.087 / 13.864 ms in column order: no
measured fork regression, though the control still varies by about 9%.
Diverse and restricted now clear 2x in this pair; mixed remains below it. All
14 checkpoint first witnesses, plan counts, non-payment counters and
simulation outputs match the baseline. All within-pass search work matches.
Both closing stress digests remain the frozen values above. These local
measurements are not Engine Budget CI or proof of a general complexity bound.
Raw rows: `.lf/tmp/mtg350/projection-{base,before,final,traffic}-{16,4}.jsonl`.

Dependency re-read (rechecked at realign): #523 OPEN at `ddc337f764`;
#524/#538/#540 remain OPEN at `b1ecacd974`, `67ce2568da`,
`7b8707e806`; no owned compatible source-tap prerequisite appeared or was
imported. Frozen receipt/context SHA256 checks still match. The exact retained
continuation and subsecond step target remain unverified.

Checks on final code:

- `CARGO_BUILD_JOBS=2 cargo test --manifest-path managym/Cargo.toml -p managym --lib flow::mana -- --test-threads 2` — 35 passed, 0.31 s, debug.
- `CARGO_BUILD_JOBS=2 cargo test --manifest-path managym/Cargo.toml -p managym-cards --test cards yue_reserves_her_tap -- --test-threads 2` — 1 passed, 0.03 s, debug.
- Workspace fmt check, `git diff --check` and both frozen evidence hashes — passed.
- Root `CARGO_BUILD_JOBS=2 uv run --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels` — built cp313 macOS arm64 wheel; matching extension replaced atomically and native import passed. SHA256 `239bc256eb0204c1bb0e61a94607fef837a0ea6524e57ca26de0668c1b85947d`.
- `uv run --extra dev pytest tests/model/test_record_coverage.py -q` — 4 passed, 3.43 s.
- `lf context` — scratch below both limits; no broad-suite/Engine Budget claim.

## Concept review after projected-bound/memo pass (2026-11-15)

Outcome for loop-or-next: **nothing to change and no open product choice**.
The current implementation preserves the design's distinction between committed
payment state and disposable feasibility work. The next implement direction
remains the mixed-workload investigation below; this review adds no redesign
and does not establish Task completion.

The concrete interaction remains: announce Yue, commit two waterbend taps,
then choose among native offers that can finish the remaining payment and its
saved callers. One floating mana excludes the infeasible Prism activation;
two retains it. `Payment` and `Decision` own that interaction, while
`payment_model` derives reservations and `PaymentModel` returns a complete first
witness. Projected color counts are valid only inside an optimistic check for
one spending purpose. They must never replace the exact pool, cached query,
spending offer or receipt. Shared memo states are immutable; concrete branches
and fork-local FIFO mutations remain independent.

Normal completion consumes the existing obligations once. Failure must leave
infeasible offers absent without committing speculative taps or fees. Recovery
uses the actual pending decision and existing Command binding; an exact fork
must retain its choices, and a stale Command must still fail. The existing
directed fixtures cover these cases. The restricted-seed filter is the key
counterexample to merging optimistic supply with concrete resources: mana that
cannot pay the parent can still fund a nested filter. Keep the separate gross
and all-class conservation bounds and the class-preserving concrete pool.

The remaining mismatch is measured latency, not an extra concept for a player:
mixed's latest 1.98x still misses 2x. Profile its remaining model/search work
before another storage change; retain the existing exact cache limit and
query equality unless measurements justify a new implementation proposal.
The README's current payment contract and allocation-diagnostic instructions
describe the implemented paths; no new usage or skill instruction is needed.
Persistence, public APIs, policy inputs and infrastructure gain no new state.
The experimental continuation still requires the compatible owned prerequisite
and its separate first-divergence proof; this review cannot resolve that
dependency. Review page: `scratch/concept-review.html`.

## Mixed production pass (2026-11-15)

One bounded `sample` of the actual mixed size-16 search workload showed both
witness work and source construction, including `mana_choices`,
`production_alternatives` and class cloning. It is a coarse profile, not a
percentage attribution to the required checkpoint. Raw sample/rows:
`.lf/tmp/mtg350/mixed-current-profile.{txt,jsonl}`.

`production_alternatives` now consumes owned modes. With no bonuses it retains
and deduplicates them in place; a deterministic bonus mutates the owned output.
Fixed production with no bonuses returns its output directly. First occurrence
order, exact classes and branching bonus combinations are unchanged. The gross
whole-payment bound also skips output scans when ability uses are zero, as the
other bounds already did. No keys, retention, rules or receipts changed.
Before source/binary: `before-production.rs` / `bench-states-projection-final`;
final binary: `bench-states-production-final`, all under `.lf/tmp/mtg350/`.

| Required cell | Baseline ms | Before pass ms | Final ms | Baseline / final |
|---|---:|---:|---:|---:|
| counted-tap 16 cast-choice | 1015.588 | 275.857 | 289.905 | 3.50x |
| diverse-mana 16 cast-choice | 382.944 | 147.047 | 140.889 | 2.72x |
| mixed 16 cast-choice | 278.208 | 138.678 | 137.064 | 2.03x |
| restricted-filter 4 priority | 106.920 | 40.326 | 36.685 | 2.91x |

This closing order was before, production baseline `592f499546`, final;
size-16 group then restricted size 4 per binary. Each row is the median of
three actual workloads, sequential on one machine with no compiler/test runner.
Mixed zero-payment block: 15.968 / 14.035 / 16.026 ms in table order, nearly
equal baseline/final. Within-pass timing gains are small in this closing pair;
counted is 5% slower. Retention also rests on lower allocation traffic.
The first order (final, baseline, before) measured 2.37x/2.23x/2.28x/3.35x,
but its control drifted 17.004 / 21.856 / 18.389 ms in table order, justifying
the reverse check. No host cause or general complexity bound is inferred.
Raw timing: `production-{base,before,final}-{16,4}.jsonl` and
`production-reverse-{base,before,final}-{16,4}.jsonl` in that directory.

Separate allocation counts, before → final (requests / requested bytes):
counted 1,778,149 / 257,592,036 → 1,745,509 / 256,286,436;
diverse 1,246,902 / 172,962,946 → 968,161 / 150,658,014;
mixed 3,590,140 / 407,148,620 → 2,436,070 / 306,000,252;
restricted 636,492 / 85,706,891 → 524,660 / 76,452,019.
Zero-payment block stays 331,603 / 71,848,920. Raw:
`production-traffic-{before,final}-{16,4}.jsonl`. These are not live heap.
All 14 checkpoint witnesses, plan counts, non-payment counters and simulation
outputs match baseline in both orders; all within-pass work is identical.
Both closing stress digests match the frozen values above.

Checks: debug `flow::mana` — 36 passed (0.35 s); cards
`yue_reserves_her_tap` — 1 passed (0.06 s); cards `heartbeat` — 7 passed.
Root cp313 release wheel rebuilt; matching extension replaced atomically and
native import passed, SHA256
`7afe6f46a8efbd9ee0d032042461ad747f418a0a052da37406a64dd0b8933480`.
`uv run --extra dev pytest tests/model/test_record_coverage.py -q` — 4 passed,
2.29 s. Workspace fmt check, `git diff --check` and `lf context` passed.
No experimental prerequisite was imported; frozen receipt/context hashes match.

## Concept review after mixed production pass (2026-11-15)

Outcome for loop-or-next: **nothing to change and no open product choice**.
The resumed review read the implemented owned-mode path and current evidence;
the earlier direction to investigate mixed is fulfilled by the production pass
above. All four local targets pass in both orders. Mixed's closing 2.03x is
close to the threshold, and the control drift in the first order still limits
the timing claim; no further redesign follows from it. Gate owns CI validation.

The interaction remains one announced payment with native pending choices.
Normal completion pays the announcement and nested fees once. An infeasible
activation is absent before committing its speculative resources; the payable
two-floating-mana Yue/Prism continuation remains available. Recovery retains
committed taps and the actual pending caller, with exact-fork choices and stale
Command rejection. Ordered production modes remain distinct by exact mana
class: absent or deterministic bonuses permit reusing owned values, while
branching bonuses still enumerate their combinations in first-occurrence order.
Deduplication must never merge different restrictions or change first witnesses.
The existing mode/reference fixture and directed debug checks support that
boundary; restricted-seed filters remain the counterexample to projecting
concrete resources or weakening exact keys.

There is no new public API, persisted lifecycle, policy input, browser action
or infrastructure contract. The existing README and allocation-diagnostic
instructions describe the implemented behavior; no usage or skill change is
needed. Keep the present design and its remaining validation/integration work
below. The experimental continuation and subsecond target remain unverified
pending the compatible owned prerequisite; local audit speedups do not satisfy
that separate proof. Review page: `scratch/concept-review.html`.

## Remaining implementation

The shared payment contract is unchanged; no open product choice was found.

1. Four local 2x audit targets now pass in both comparison orders (latest
   3.50x counted, 2.72x diverse, 2.03x mixed, 2.91x restricted). Gate still
   owns Engine Budget CI. Keep exact keys, the 64-entry FIFO, independent
   branch mutations and complete first witnesses. No inline-vector or
   reachability-closure revival.
2. Retained experimental integration: no compatible owned prerequisite exists.
   Keep frozen receipts and admission unchanged; no source-only diagnostic
   substitutes for first-divergence proof, a corrected candidate continuation
   and the subsecond isolated-step target.
3. Gate owns CI, broad debug nextest/clippy and Engine Budget. Parent #523 lands
   first; then rebase and rerun directed checks plus the four-cell witness
   parity. Main (2026-11-15, `f6f99d91b7`) replaced `ObjectRef` with
   `ObjectId`, `object_to_permanent` with `permanent_of`, and synthetic
   payment-model objects `ObjectId(index)` with `payment_object(index)` from a
   distinct allocator; `b5478d8789`/`9632914976` only touch `AgentError`
   spelling. This diff's new code uses none of these except the test-fixture
   `ObjectId(index)` lines in `flow/mana.rs` tests (`fn source` and the moved
   activation fixture), which must take `payment_object` on rebase rather than
   keep the old constructor. Do not change an Engine
   Budget reference to disguise a moved digest. The old schema-7 receipt test
   failures below are a base issue resolved on main (`452d7b5357`), not
   permission to change binding checks.

## Delete — do not maintain

Owned output-mode copies used only to add an absent/deterministic bonus are
removed; retain first-occurrence mode order and exact classes. Zero-use output
scans in the whole-payment bound are removed.
Duplicated announced-payment reservation readers and late forced-transition
branches are removed. Duplicate first/tail/overwritten search-state copies and
owned output/candidate payload copies are removed; keep only shared immutable
payloads. Temporary optimistic class-metadata pools, independent copies of
identity/already-canonical memo states and doomed partial-spending branch
copies are removed. Concrete pools retain all classes; no parallel bound or
memo representation is retained. Rejected inline resource-vector code and its exclusive spill fixture
are removed, with no dependency change. Rejected reachability closure code and its exclusive
fixture are removed. Dense per-source ability-use rows are replaced by sparse
flags; retain no parallel dense resources path. Preserve one payer/model,
shared announced caller ownership,
exact resource/fee semantics, forced-decision skipping, receipts and replay
binding authority. No parallel serialized lifecycle, global cache, cutoff,
fallback action, source-tap prerequisite import, schema/data migration or
unrelated source preference belongs here.

## Earlier reservation-only checks

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
