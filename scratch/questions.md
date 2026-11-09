# MTG-268 — batched local-update rollouts

## Second optimize step 6 — compress (2026-11-09)

Notes-only reduction. Reviewed the branch diff, retained design, writer/decoder,
Command and semantic-hash consumers, batched/model paths, regression coverage,
configuration and user docs. No worthwhile production simplification remains
within this cut: the explicit field adapters make lexical order, omission and
integer casts visible; a generic framework or broader derived-record rewrite
would add risk without simplifying the contract. Existing production, tests,
fixtures, harness, runtime and every HTML measurement row remain unchanged.
No build, test or timing rerun is needed for this notes-only edit.

Current step direction, verbatim:
> Second optimize checklist step6. Step5 ADVANCE recorded at top scratch/questions.md. Keep borrowed canonical writer: five alternating pairs showed 24.93%/23.93% lower 16-world latency with disjoint ranges and exact compatibility. Preserve one writer owner and narrow sorted-derived-record boundaries; rejected dtype and five-array trials remain rejected. Review code for worthwhile simplification without altering canonical bytes or re-opening optimization. Curate oversized repeated notes, preserving exact user direction, every historical measurement row/result, failed fixture correction, baseline provenance, trial/revert history and artifact locations. Full originals must remain recoverable. If production changes, run relevant exact-byte/debug checks and record rebuild needs for final measure; otherwise avoid redundant tests. Keep existing work. No Graph, publication, shipping or Task completion. Return source/runtime identity and final-measurement requirements.

Reversible curation choice: condense superseded implementation/protocol plans
and repeated identities; keep measurement sections and contrary findings.
Before editing, moved the complete 51,321-byte original to
`.cache/mtg-268/compress-step6/before-questions.md` (SHA-256
`30fd226008a7098fba69333703ae4a806c0b4733ea1f3e0da511006771ed0861`).
`before.json` records its hash plus all 50 other branch-file hashes and the
installed runtime. Earlier originals remain at the historical paths below.
`scratch/measure.html` is untouched, preserving every historical row/result.

Source/runtime still match the kept measurement: HEAD `0bbb055127a79aa0e1bde4f6aa36c792dab61317`
plus candidate patch `c9bbc1e8efeeff84c4f5786ffb6ae851136903e1754b61cd285987d7b9b8c03f`;
installed runtime `bff695ea265d07924bdf4f4bc205b7585e0b92e248ab7f4bf1d0b1a793179ad0`.
All 14 frozen candidate source/fixture hashes match. No runtime replacement or
rebuild was performed or made necessary by compression.

Final measurement follows sync and realign. Reconcile the resulting source,
lockfiles, harness, world/catalog, checkpoint and installed runtime first.
If native source changes, rebuild an ordinary release wheel from the root with
`CARGO_BUILD_JOBS=1 uv run --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels`,
retain build/source/wheel hashes, and atomically install the cp312 extension.
Native edits need the eleven independent byte/SHA-256/BLAKE3 cases and bounded
debug pending-entry/event/privacy/terminal checks before final timing. Verify
fresh runtime identity, no benchmark instrumentation or `preserve_order`, and
live/saved receipt plus native/Python encoder parity against the retained
merged references. Preserve original-world references separately.
Use the frozen harness/protocol and only ordinary-wheel latency; one thread,
fixed openings 42/43, seed 17, depth 20, no overlapping builds/tests, all samples
and failures retained. Do not pool the kept A/B result with later post-sync
measurements or reinterpret the two rejected trials. Full gate, MTG-183/w13
admission and the original near-singleton definition remain separate below.

Validation: `git diff --check` passes; SHA-256 preservation checks confirm all
50 other branch files, the runtime and every Markdown table row unchanged;
`lf context` is below both configured limits. No tests/builds/timings rerun.

## Second optimize step 5 — advance with borrowed writer (2026-11-09)

Decision: **advance**. Read every current/historical HTML table row, the complete
borrowed-writer result, and the retained writer plus its Command/hash call sites.
Keep the 24.93% / 23.93% reduction: both openings pass the declared >=5% and
disjoint-range rule with exact receipt/byte/hash compatibility. The smaller
array-only trial and dtype trial stay rejected, with their evidence retained.
Command/audit and network now cost about the same; no newly measured removable
substep supports another comparable bounded change. Snapshot/allocation remain
small. Extending borrowed serialization to derived nested records is the best
remaining idea, but its removable fraction is unmeasured and it expands the
ordering/omission contract; do not select it without new attribution. No witness
caching, hash substitution or audit reduction is justified. Finish compress,
sync, realign and final measurement; broader gate and original acceptance are
separate, with no shipping or Task completion authorization.

## Second optimize step 4 — borrowed JSON measurement (2026-11-09)

**Better; keep the candidate.** Median sixteen-world decision latency falls
24.93% at opening 42 and 23.93% at opening 43. Both candidate maxima are below
their baseline minima, and all correctness/work/identity checks pass. The
predeclared >=5% reduction AND disjoint-range rule passes at both openings.
This result exceeds the provisional 8–15% hypothesis; that estimate was not
an acceptance ceiling. Original near-singleton and trained-policy acceptance
remain unmet, and the full delivery gate remains separate.

Claim tested: “Borrowed canonical JSON emission reduces sixteen-world decision
latency by 8–15% at openings 42 and 43, with unchanged behavior.”
The supported claim is 24.93% / 23.93% lower median latency on these two fixed
initialized-policy openings. These are whole `teacher.search` latencies, not
serializer microbenchmarks. Only `after_seconds` enters this comparison.

Seconds per decision, median [minimum, maximum], n=5 per binary per opening:

| Opening | Baseline seconds | Candidate seconds | Median reduction | A/B median ratio |
| --- | --- | --- | --- | --- |
| 42 | 0.97506 [0.95206, 0.97882] | 0.73197 [0.71286, 0.75503] | 24.93% | 1.33211× |
| 43 | 1.20364 [1.19115, 1.23144] | 0.91563 [0.89085, 0.92580] | 23.93% | 1.31454× |

Savings are 0.24309 s / 0.28801 s per decision. Baseline is the immediate kept
batched implementation at `0bbb055127a79aa0e1bde4f6aa36c792dab61317`, not the
historical sequential implementation. No prior times or gains are pooled here.

### Protocol and actual work

Five adjacent serial pairs ran AB/BA/AB/BA/AB. Every invocation used a fresh
process after a hash-verified atomic binary swap, with the frozen command:
`uv run --extra dev python scripts/bench_local_update.py --worlds 16 --position-seeds 42 43 --native-reference .cache/mtg-268/merged-reference`.
All ten samples, stdout, stderr, timestamps and exit codes are retained;
none was retried, replaced or dropped. All exited zero, all measurement stderr
files are empty, and there are zero receipt/work/identity failures, deadline
rejections or missing samples. No builds, tests or other benchmarks overlapped
timing; no artificial load, games/evals, Graph launch, publication or shipping.

Both preserved builds are ordinary release wheels with only the Python Cargo
feature, no native benchmark instrumentation and no serde_json preserve_order.
They were built and verified before this series; their frozen build/install
journals are retained rather than rebuilding during measurement. Same local
Apple M4 Max, macOS 26.0.1, Python 3.12.12, Torch 2.10.0 and locked environment;
same warmed initialized 64-wide/two-layer v6 scalar policy, openings 42/43,
search seed 17, depth 20, sixteen worlds, one Torch CPU/inter-op thread and
OMP/OpenBLAS/MKL each one. The fixed harness runs sequential control then
batched search, with separate profiles after opening 42. Setup, warming,
swaps, comparisons and profiles are outside each latency region.

Each timed search executes 64 rollouts / 1,280 Commands at 42 and 80 / 1,600
at 43, all at depth 20. Per binary: 720 timed batched rollouts / 14,400 Commands,
the same timed sequential-control work, plus 640 separate profiled rollouts /
12,800 Commands. Total series work: 4,160 rollouts / 83,200 Commands. Every
batched opening-42 profile has 21 forwards and 1,280 Commands. Earlier retained
1/4/16-world runs establish workload sensitivity; no scaling grid was repeated.

Twenty live and twenty saved full receipt comparisons pass. Commands, canonical
hashes, audits, legal support, terminal flags, sampling and policy/world identity
match exactly under the unchanged comparator's existing runtime/source/time
exceptions and prediction tolerances (atol 1e-6, rtol 1e-5). Maximum absolute
base/value/target differences are 1.0242171000207634e-10 /
4.0978193283081055e-08 / 1.8624171926884259e-09. Numeric prediction equality is
toleranced, not claimed bitwise. No new exceptions were added.

The implementation handoff retains eleven independently baseline-produced exact
byte/SHA-256/BLAKE3 fixtures, five debug contracts, six live/saved receipt pairs
and 1,281 exact native/Python encodings. After the final swap, a fresh process
again passes all eleven exact-byte/SHA-256 fixtures and verifies the installed
runtime, uninstrumented build, w17/catalog, vector API and checkpoint identity.
These final checks contain no latency samples.

### Mechanism and limits

Separate ordinary-wheel profiles at opening 42/16, seconds per search,
median [minimum, maximum], n=5 per binary; these are not decision latency:

| Bucket | Baseline profile seconds | Candidate profile seconds |
| --- | --- | --- |
| native_step_and_audit | 0.40358 [0.39611, 0.41982] | 0.24599 [0.24254, 0.24865] |
| semantic_direct_hash | 0.15898 [0.15579, 0.16236] | 0.08189 [0.08062, 0.08289] |
| network | 0.24509 [0.23976, 0.25466] | 0.24235 [0.23529, 0.24324] |
| observation_encoding | 0.15772 [0.15716, 0.16646] | 0.15527 [0.15205, 0.15887] |
| session_snapshot | 0.00702 [0.00692, 0.00842] | 0.00679 [0.00653, 0.00712] |

Command/audit falls 39.05% and semantic direct hash 48.49%, with disjoint
ranges for both. Network and encoding ranges overlap. This localizes the change
to the two consumers of `Observation::to_json`: `ViewerProjectionWitness::new`
in `search_state.rs`, and `Game::semantic_viewer_input` in `decision.rs`.
The single borrowed writer in `observation_json.rs` removes handwritten Value
tree construction, repeated traversal and destruction, retaining sorted-Value
boundaries for derived records. The profile agrees with that mechanism; it
does not isolate each allocation or prove every saved millisecond's cause.

The earlier 21.99–23.75% materialization fraction yields an impossible
full-removal bound of 1/(1−0.2375) = 1.312× **for that instrumented category
and denominator**. It excludes stringify (another 10.88–11.88%) and some
unseparated cleanup; the borrowed writer also replaces final Value traversal.
It is not a strict bound on this broader rewrite or on ordinary-wheel latency.
The observed 1.332× / 1.315× does not justify subtracting instrumented seconds
from ordinary times. No new internal-timer campaign ran, and the fraction of
the gain due to construction versus traversal/cleanup remains unmeasured.

Command/audit (0.24599 s) and network (0.24235 s) are now similar in the
candidate's separate profile, with encoding next (0.15527 s). Snapshots remain
small (0.00679 s). These are profile attribution, not additive ordinary savings.
Five observations are not confidence intervals; alternating five pairs is not
fully order-balanced. Results cover one host and two full-depth opening searches,
not terminal-heavy positions, trained strength, other capacities or checkpoints.
The original “near today's one-world cost” requirement is not established by
this series. No new workload or broader optimization campaign was started.

### All paired observations

Positive reduction means candidate time is lower. JSON retains full precision.

| Pair | Order | Opening | Baseline seconds | Candidate seconds | Reduction |
| --- | --- | --- | --- | --- | --- |
| 1 | AB | 42 | 0.975062750 | 0.742224833 | +23.879% |
| 1 | AB | 43 | 1.231443333 | 0.915634583 | +25.645% |
| 2 | BA | 42 | 0.978821292 | 0.731967917 | +25.219% |
| 2 | BA | 43 | 1.203099958 | 0.922797958 | +23.298% |
| 3 | AB | 42 | 0.975891708 | 0.755026959 | +22.632% |
| 3 | AB | 43 | 1.203642208 | 0.890850417 | +25.987% |
| 4 | BA | 42 | 0.952060000 | 0.712864459 | +25.124% |
| 4 | BA | 43 | 1.210987084 | 0.893580500 | +26.211% |
| 5 | AB | 42 | 0.963193167 | 0.722468209 | +24.992% |
| 5 | AB | 43 | 1.191153708 | 0.925798625 | +22.277% |

### Kept source and runtime for step 5

Source is unchanged from the frozen candidate: base
`0bbb055127a79aa0e1bde4f6aa36c792dab61317` plus uncommitted patch SHA-256
`c9bbc1e8efeeff84c4f5786ffb6ae851136903e1754b61cd285987d7b9b8c03f`.
Kept installed runtime B:
`bff695ea265d07924bdf4f4bc205b7585e0b92e248ab7f4bf1d0b1a793179ad0`.
Preserved baseline runtime A:
`292cc8e190d8f29e6945ee587ecf1975e786e5517fd9da8d0c9bc58936d52881`.
Candidate `observation.rs` SHA-256:
`9d3f817f66fc6c31a3ccfe39c16f03b7240f7fa64ea44748ba68a7c88bd3ab80`.
Candidate `observation_json.rs` SHA-256:
`ac80ca5a5f2014102778e5074dbdac988b114479160559160de11723afd2090f`.
Frozen harness: `3e1c2bf414c4a7654fd6741eede6f979f704a14f9d4cbc2cd29cb76c4496d67a`.
Checkpoint: `57fd8f999d15426e9cebb73ce0f917fe987757f992bba2d5ab0b5c41542bddeb`.
World w17 catalog: `0798b2c7c7f54b7329043b13cb1742d74fe21b35373820580b76f979e6b7eb5f`.

All samples and journals, pre-series notes/report, summary, reporting scripts,
this report and final artifact hashes are in
`.cache/mtg-268/borrowed-json/measurement/`. The parent retains both binaries,
candidate wheel/patch, fixtures, source manifest, build/verification journals and
the original immutable `handoff.json`; `installations.jsonl` is append-only.
`step4-kept-identity` and `step4-kept-bytes` journals record final verification.
`measurement/disposition.json` supersedes the original handoff's unmeasured
status. All historical HTML rows and earlier implementation/evidence remain.
No server, watcher or measurement child remains running. No commit, shipping or
Task completion occurs here; step 5 receives the kept candidate and this result.


## Borrowed JSON implementation — frozen handoff (2026-11-09)

Historical pre-measurement handoff; verification below remains evidence.
Performance was subsequently measured and kept as recorded above. One borrowed
canonical writer in `observation_json` owns `Observation::to_json`. Handwritten
root/turn/action/card/mana/permission/stack/target/event fields stream directly
in lexical order, with no intermediate Value trees or collected arrays. The
existing derived records retain narrow sorted-Value boundaries. No caller,
hash algorithm, schema, projection, audit or dependency changed.

Reversible coverage choice: the ten old fixtures were verified against the
retained hashes before restoration. Added one baseline-produced sparse-optionals
case before changing production source: absent permission terms, all target
joins, signed event fields, numeric-map lexical ordering and present empty
shared effects. All eleven passed on the unchanged baseline, then candidate,
with exact bytes/SHA-256/BLAKE3. Generator, source hash and capture journal are
preserved; candidate code never generated expected bytes.

Verification (all command/stdout/stderr/exit journals in the handoff directory):

- Debug `cargo test` filters for `canonical_json_preserves_baseline_bytes_and_hashes`,
  `hidden_selection_public_entry_is_pinned_through_both_samplers`, and the three
  bounded `search_state_contract` event/privacy/terminal tests: **5 passed**.
- Root `uv run --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o .cache/mtg-268/borrowed-json/wheels`:
  **passed**, one build worker, ordinary `python` feature only. Default and
  Python feature trees confirm no `serde_json/preserve_order`. Wheel extension
  extracted, hash checked, written to a sibling temporary file and atomically
  renamed. Fresh processes verify eleven byte fixtures and uninstrumented
  w17/catalog/checkpoint identities on both binaries.
- `uv run --extra dev python scripts/bench_local_update.py --native-reference .cache/mtg-268/merged-reference --compare-encoders`:
  **6 live + 6 saved receipt cells pass; 1,281 exact native/Python encodings**.
  Baseline's preceding singleton preflight passes 2 live/2 saved cells and 81
  encodings. Same comparators/tolerances (atol 1e-6, rtol 1e-5); no new exceptions.
  Maximum base/value/target errors: 1.0242171000207634e-10 /
  4.0978193283081055e-08 / 3.1306484626902886e-09.
  Sixteen-world work stays 64/80 rollouts and 1,280/1,600 Commands at 42/43;
  the ordinary 42/16 profile has 21 forwards. All incidental times/profiles are
  retained and excluded from performance conclusions.
- `cargo fmt --check` and `git diff --check`: **passed**. User docs rechecked;
  their behavior, example and limits remain accurate. Full gate stays below.

### Frozen artifacts

`.cache/mtg-268/borrowed-json/handoff.json` retains the pre-measurement source,
wheel/member/binary hashes, fixtures, lockfiles, harness, six saved references,
identities, verification results and protocol. Beside it: `candidate.patch`,
`candidate-source.json`, both ordinary binaries, candidate wheel, `fixtures/`,
all run journals and atomic `install.py`. `measurement/disposition.json`
supersedes the handoff's unmeasured status. Identities and the completed paired
protocol are recorded once in the measurement above. No pending implementation
or trial-revert instruction remains from this handoff.

## Borrowed-writer selection — historical (2026-11-09)

After rejecting the dtype and five-array trials, step 5 selected borrowed
emission of the handwritten observation fields, keeping derived records behind
narrow sorted-Value boundaries. The provisional 8–15% hypothesis and >=5%
median reduction AND disjoint-range keep rule at both openings were declared
before measurement. The result above passes that rule. The baseline's 22–24%
materialization attribution motivated the experiment; it did not measure the
removable fraction. No generic serializer framework, derived-record redesign,
projection cache, hash substitution, audit reduction or new dependency was
selected. The independent baseline fixtures were reused after hash checks,
with sparse-optionals captured before changing production source.

## Second optimize step 4 — five-array move measurement (2026-11-09)

**No measurable difference; speedup inconclusive. Candidate reverted.** Both
openings have overlapping observed ranges. Opening 42's candidate median is
1.23% higher; opening 43's is 10.35% lower. Neither clears the predeclared rule:
each opening must improve at least 3% AND candidate maximum must be below
baseline minimum. The expected 3–6% gain is not established. The immediate
batched baseline, source `0bbb055127a79aa0e1bde4f6aa36c792dab61317`, is restored;
all earlier batching work remains. Original near-singleton acceptance is unmet.

Claim tested: “Moving the five JSON arrays reduces sixteen-world decision
latency by 3–6% at openings 42 and 43, with unchanged behavior.”
Seconds per decision below are median [minimum, maximum], n=5 per side per
opening. Only `after_seconds` enters this A/B comparison. No historical
sequential-to-batched gain or previous ordinary timing is pooled with it.

| Opening | Baseline seconds | Candidate seconds | Median reduction | A/B median ratio |
| --- | --- | --- | --- | --- |
| 42 | 1.09405 [1.01709, 1.91236] | 1.10751 [0.94544, 1.16997] | -1.23% | 0.98785× |
| 43 | 1.32530 [1.27261, 1.46090] | 1.18819 [1.06747, 1.43368] | +10.35% | 1.11539× |

Five adjacent serial pairs ran AB, BA, AB, BA, AB, each in a fresh process
after the preserved installer verified and atomically swapped the binary.
Exact command for every invocation:
`uv run --extra dev python scripts/bench_local_update.py --worlds 16 --position-seeds 42 43 --native-reference .cache/mtg-268/merged-reference`.
No builds, tests or other benchmarks overlapped the series; no artificial load,
games/evals, Graph launch, publication, shipping or extra profiling campaign.
All ten samples and errors are retained, without retries or replacements.
The baseline pair-3 opening-42 sample is 1.91236 s; its cause is unmeasured.
Five samples are not confidence intervals. This study cannot resolve a small
benefit; a separately declared repeated comparison on a quieter controlled host
would be needed. No further series is started here.

Same local Apple M4 Max/macOS 26.0.1, Python 3.12.12, Torch 2.10.0 and locked
checkout environment; ordinary release binaries, only python Cargo feature.
Both warmed initialized 64-wide/two-layer v6 scalar policies use position seeds
42/43, search seed 17, depth 20, sixteen worlds and one Torch CPU/inter-op thread;
OMP/OpenBLAS/MKL are explicitly one. The unchanged harness executes sequential
control then batched search internally, with separate profile searches after
opening 42. Alternating binaries limits drift but five pairs are not fully
order-balanced. Setup, warming, swaps, comparisons and profiles are outside
`teacher.search`, the only latency region. Prior 1/4/16 results established
that the harness distinguishes workload size; no scaling grid was repeated.

### Correctness and actual work

All ten invocations exited zero, with zero receipt/work/identity failures,
retries, deadline rejections or missing samples. Measurement stderr files are
empty. Twenty live and twenty saved full receipt comparisons passed, including
Commands, hashes, private audits, support, terminal flags and sampling identity.
Only existing time/runtime/source and floating prediction exceptions apply;
no comparator, fixture, receipt or tolerance changed (atol 1e-6, rtol 1e-5).
Maximum base/value/target errors: 1.0242171000207634e-10 /
4.0978193283081055e-8 / 1.8624171926884259e-9.
Step 3's ten independent baseline byte/SHA/BLAKE3 fixtures, debug privacy/event/
terminal/pending-entry tests and 1,281 exact encodings remain valid evidence.

Every timed search did 64 rollouts / 1,280 Commands at opening 42 and 80 /
1,600 at 43, all depth 20. Each wheel did 720 timed batched rollouts / 14,400
Commands plus the same sequential-control work. Separate harness profiles add
640 rollouts / 12,800 Commands per wheel; total series work is 4,160 rollouts /
83,200 Commands. Every batched opening-42 profile has 21 network calls and
1,280 Commands. The openings are neither terminal-heavy nor trained-strength
measurements, and supply no MTG-183/w13 admission or acceptance bridge.

### Limiter and mechanism, separate from latency

The retained native attribution maps witness and semantic JSON to
`Observation::to_json` in `managym/src/agent/observation.rs`, reached through
`ViewerProjectionWitness::new` in `search_state.rs` and
`Game::semantic_viewer_input` in `decision.rs`. Moving five prebuilt arrays
removes recursive Value serialization, copying and destruction at that owner;
the necessary initial tree and final serialization remain. Previous separate
timers put value-build at 18.05–19.63% of instrumented search. The design's
assumed removable 15–30% gives 2.7–5.9% of the whole path, not a measured
fraction. Even impossible removal of the full 23.75% materialization category
would yield only 1/(1−0.2375) = 1.312× there, not near-singleton performance.
No such profile subtraction is used to claim ordinary savings.

This series' already-produced ordinary profiles show Command/audit median
0.45120 s [0.41014, 0.50972] → 0.40057 s [0.35116, 0.52531] and semantic direct
hash 0.17526 s [0.16023, 0.19862] → 0.14429 s [0.13084, 0.18245], n=5 per side.
Both overlap: direction is consistent with the mechanism but does not isolate
array savings. Command/audit remains the largest median bucket. Network and
encoding variability also remain; no cause for that variability was measured.
Profile seconds are separate from ordinary latency and are never added as savings.

### Restoration and evidence

Reversed only the frozen `candidate.patch`, including its added serializer test
and fixtures. Both Rust files now match HEAD byte for byte; previous baseline
work is untouched. The candidate patch, binaries, wheel, fixtures and every
verification/sample remain preserved. The baseline was atomically reinstalled.
Fresh identity confirms uninstrumented w17, catalog
`0798b2c7c7f54b7329043b13cb1742d74fe21b35373820580b76f979e6b7eb5f`, vector-root
API and initialized checkpoint
`57fd8f999d15426e9cebb73ce0f917fe987757f992bba2d5ab0b5c41542bddeb`.
Bounded restored 42/43 singleton checks pass two live and two saved comparisons
and 81 exact native/Python encodings; their incidental times are excluded.

Restored runtime A: `292cc8e190d8f29e6945ee587ecf1975e786e5517fd9da8d0c9bc58936d52881`.
Rejected runtime B: `5d4f0a641152bbd8537c2fb349b23f4638ce7be3a93bbe6baa8e2d2d7714e84a`.
Frozen harness: `3e1c2bf414c4a7654fd6741eede6f979f704a14f9d4cbc2cd29cb76c4496d67a`.
Patch: `c670303d4c53c9eb04794465ebf7805e3c6adb40ffdd7bc7df2957cb680de6d9`.

New evidence: `.cache/mtg-268/json-array-move/measurement/` contains `before.json`,
both prior notes/report files, `measure.py`, exact `pair-{1..5}-{baseline,candidate}`
stdout/stderr/run journals and install journals, `runs.json`, `summary.json`,
`summarize.py`, preserved `fixtures/`, this `report.md`, `write_report.py` and
`final-artifacts.json`. The parent directory retains the frozen implementation
handoff and all earlier evidence; `installations.jsonl` is append-only.
`step4-restored-{install,identity,parity}.{out,err,run.json}` records restoration.
Historical HTML rows are preserved in full. No server/watcher or child remains
running. Broad gate checks and Task acceptance remain for later steps.

### All paired reductions

Positive means candidate time is lower; none is dropped.

- Opening 42, pairs 1–5: -4.368%, +1.107%, +39.252%, +8.133%, -8.890%.
- Opening 43, pairs 1–5: -8.178%, +16.475%, +8.910%, +12.949%, +24.502%.

## Second optimize step 3 — historical implementation handoff

The implementation was completed and verified, then rejected and reverted by
step 4 above. Full original notes are preserved verbatim in
`.cache/mtg-268/json-array-move/measurement/before-questions.md`; the immutable
step-3 manifest remains `../handoff.json` relative to that evidence directory.
Ten independent exact byte/hash fixtures, debug bounded checks, ordinary identity
checks, six live/saved receipt cells and 1,281 exact encodings passed before
measurement. Both ordinary binaries and the frozen patch remain preserved.

Failed-fixture correction: the first populated case put an agent-owned hand
card in the opponent list. Rust byte roundtrip passed, but native canonical
validation correctly rejected it. Ownership was corrected and the baseline
regenerated **before production edits**; corrected baseline debug and ordinary
checks passed. Original files and failure remain under
`.cache/mtg-268/json-array-move/initial-fixtures/` and
`baseline-wheel-bytes.{run.json,out,err}` in that same trial directory.
Pending-entrant capture used a temporary four-line write in the constructed
regression, removed after capture. Synthetic optional combinations isolate
serialization; they do not claim admitted card behavior.

## Five-array design — historical, rejected (2026-11-09)

The bounded trial moved only `action_space.actions`, `agent_cards`,
`opponent_cards`, `stack_objects` and `recent_events` into parent Values.
Locked serde_json 1.0.150 `json!` sends expression values through `to_value`,
recursively rebuilding owned strings/maps/vectors; ordinary/default and Python
feature trees had no `preserve_order`. Moving these arrays preserved sorted
keys but retained initial trees, inner helpers and final stringify.

The assumed removable 15–30% of the measured 18.05–19.63% value-build category
implied 2.7–5.9% of instrumented search: a provisional 3–6% ordinary target,
not a measured fraction. Conditional illustrations were 29–59 ms / 37–73 ms
at 42/43, approximately 1.03–1.06×. The 3% median reduction AND disjoint-range
keep rule at both openings was declared before the five AB/BA/AB/BA/AB pairs.
It failed; do not retry the five-array patch alone. The completed measurement
above owns actual work, results and restoration. Full design and commands are
in `.cache/mtg-268/compress-step6/before-questions.md`, under this heading's
former title “Second optimize step 2 — working design: move observation arrays”.

## Prior native attribution (2026-11-09)

Exact witnesses occupy 91.3–92.1% of native Command/audit. Five instrumented
42/16 profiles attribute 21.99–23.75% of full search to JSON Value build/drop
(3,908 calls; median 0.40120 s [0.35370, 0.42119]), including both witness and
semantic serializers. This is a category ceiling, not the removable-array
fraction or ordinary latency. Acting builds are already eliminated in this
workload. Authority Value construction is 4.19–4.42%; rules, audit and acting
byte clones are too small to close the target gap. One 43/16 diagnostic agrees
on mechanism (23.48% JSON materialization), not variance. Detailed profile totals
rose 1.489→1.815 s; cause unmeasured, all samples retained.

Seven instrumented invocations passed live/saved discrete comparisons with max
base/value/target errors 1.0242171e-10 / 4.09781933e-8 / 1.86241719e-9.
No latency claim follows from instrumentation. Full tables, limits, commands and
all artifacts remain in `scratch/measure.html`, the preserved pre-step-3 notes,
and `.cache/mtg-268/native-attribution-2/`. Temporary timers were removed and the
292cc8e1 ordinary binary restored. No experiment/world/receipt was rebound.

## Final post-sync measurement — optimize step 9 (2026-11-09)

**Better for equal work.** On synced `0bbb05512`, sixteen-world search takes
0.97911 s / 1.22335 s at openings 42/43, versus live sequential 3.33854 s /
4.21951 s: **3.41× / 3.45× faster**. Eight-world search takes 0.50590 s /
0.62788 s: **3.47× / 3.50× faster** for equal work. Every cell has disjoint
sequential/batched ranges. The original near-singleton outcome remains unmet;
this result neither supplies original-world singleton equivalence nor trained
strength. Dtype validation remains rejected and reverted.

Claim tested: “Batched local-update search is faster for equal work on two fixed
openings, with matching receipts.” All values below are seconds per decision,
median [minimum, maximum], n=5 per implementation per cell. Speedups divide
medians. All samples, stderr and exit codes are retained; none was retried,
replaced or dropped. Five samples are not confidence intervals.

### Primary 1/4/16-world grid

| Position | Worlds | Sequential seconds | Batched seconds | Speedup | Rollouts / Commands per search |
| --- | --- | --- | --- | --- | --- |
| 42 | 1 | 0.21538 [0.21306, 0.22106] | 0.08589 [0.08175, 0.08840] | 2.51× | 4 / 80 |
| 42 | 4 | 0.84463 [0.81816, 0.86687] | 0.26746 [0.26138, 0.27197] | 3.16× | 16 / 320 |
| 42 | 16 | 3.33854 [3.30155, 3.47404] | 0.97911 [0.96551, 0.99997] | 3.41× | 64 / 1280 |
| 43 | 1 | 0.26266 [0.26003, 0.27264] | 0.09705 [0.09452, 0.10423] | 2.71× | 5 / 100 |
| 43 | 4 | 1.05185 [1.03508, 1.09272] | 0.32409 [0.31470, 0.33016] | 3.25× | 20 / 400 |
| 43 | 16 | 4.21951 [4.10531, 4.36913] | 1.22335 [1.20339, 1.24545] | 3.45× | 80 / 1600 |

### Separate eight-world supplement

| Position | Worlds | Sequential seconds | Batched seconds | Speedup | Rollouts / Commands per search |
| --- | --- | --- | --- | --- | --- |
| 42 | 8 | 1.75341 [1.72520, 1.78252] | 0.50590 [0.50534, 0.51032] | 3.47× | 32 / 640 |
| 43 | 8 | 2.20035 [2.17964, 2.20853] | 0.62788 [0.62208, 0.63177] | 3.50× | 40 / 800 |

The primary grid has 30 live and 30 saved comparisons; each implementation
executes 945 timed rollouts / 18,900 Commands. The separate supplement has ten
live comparisons, no saved eight-world references, and 360 timed rollouts /
7,200 Commands per implementation. All ten measurement invocations exited zero;
zero parity failures, retries or deadline rejections. Preflight separately
passed six live/saved comparisons and 1,281 exact native/Python encodings;
its incidental timing is excluded. No missing or failed samples are encoded as
zero. The only stderr message is uv ignoring an unrelated VIRTUAL_ENV and using
this checkout's pinned environment.

Maximum absolute base/value/target differences across the forty live and thirty
saved measurement comparisons are 1.0242171000207634e-10 /
4.0978193283081055e-8 / 3.1306484626902886e-9, within atol 1e-6 / rtol 1e-5.
Complete discrete receipts, legal support, Commands, canonical hashes, private
audits, terminal flags, sampling identities and policy/world identities agree.
Only the comparator's existing runtime/source and measured numeric/time
exceptions apply. No reference or comparator exception was changed.

### Interpretation and retained evidence

The 8-world/current sequential-singleton ratios are 2.35×/2.39×; 16-world ratios
are 4.55×/4.66×. These are not original-world acceptance. All opening rollouts
reach depth 20, so terminal-heavy and trained positions remain unmeasured.
The ordinary 42/16 profile shows 1,280 Commands and 21 forwards; median buckets:
Command/audit 0.40657 s, network 0.24983 s, semantic direct hash 0.15986 s,
encoding 0.15919 s, snapshot 0.00736 s. Categories are profile attribution,
not ordinary savings. Snapshot removal cannot close the gap.

Full post-sync evidence lives in `.cache/mtg-268/final-measurement/`, including
`before.json`, wheel/build/install/identity journals, six-cell preflight, five
grid and five eight-world runs, summaries and preserved historical report.
All source/lock/receipt identities, historical full tables and detailed limits
remain in unchanged `scratch/measure.html` and the preserved pre-step-3 notes.
The harness is 3e1c2bf414c4a7654fd6741eede6f979f704a14f9d4cbc2cd29cb76c4496d67a.
Policy is 57fd8f999d15426e9cebb73ce0f917fe987757f992bba2d5ab0b5c41542bddeb,
world w17, catalog 0798b2c7c7f54b7329043b13cb1742d74fe21b35373820580b76f979e6b7eb5f.

## Implemented behavior and removed mechanisms

Keep independent policy RNGs, stable world/action receipt order, every receipt
field and private audit, native Commands, canonical hashes, full legal support,
terminal/sign semantics, sampling identities, atomic rejection and deadlines.
No input/checkpoint/schema change or batching flag. Semantic SHA-256 is rebuilt
under the Env lock on every read; witness BLAKE3 is not equivalent. Memory still
scales with worlds × supported root actions.

### Delete — do not maintain

No remaining deletion targets. Completed cuts:

- Handwritten nested Value builders and root assembly in `Observation::to_json`:
  one borrowed writer in `observation_json.rs` now owns canonical bytes. Preserve
  its explicit lexical field order and narrow sorted-derived-record boundaries;
  replacing them with a generic adapter would obscure the compatibility contract.
- Five-array production trial, reverted after its keep rule failed. The later
  borrowed writer reuses only its independent fixtures/test, with a new sparse
  case. The rejected production path stays removed; its full patch is archived.
- Sequential production rollouts and separate singleton inference: one batch
  implementation; git supplies the timing oracle.
- Per-ply Python semantic parsing, unused composite envelopes and repeated
  root/config/prepared-constraint parsing. Real composite consumers remain.
- Pre-Command viewer projections, duplicate equivalent acting-view serialization
  and revalidation of a Command just constructed under both locks.
- Globally empty CPU attention/fact columns, preserving coordinates, historical
  pooling and program order. No state-derived cache.
- Receipt-field allowlists, duplicate fact-word decoding, validity conversions,
  fact-program guards and native profiling lifecycles.
- Scalar Python-list conversion in native array returns and per-row Python
  encoding/stack copies in `predict_batch`. Keep the Python encoder oracle,
  selected capacities and independent writable arrays.

Compound execution was already removed (`docs/training.md#compound-decisions-removed`);
retain historical readers, not the directive's obsolete sequential fallback.
The rejected dtype-name experiment and its tests stay reverted.

## Historical evidence (condensed for context budget)

Full pre-step-3 notes are preserved at `.cache/mtg-268/json-array-move/before-questions.md`.
Committed earlier history is `0bbb05512:scratch/questions.md`; earlier complete
curation is `c0c3c2b4476755676115c602a97f57bacac6a7ea:scratch/questions.md`.
All historical tables and the new attribution remain in unchanged `scratch/measure.html`.
Dtype trial remains rejected: all six observed ranges overlapped despite a 60.7%
profiled encoding reduction. Smaller-cell medians rose; the 0.76723 s candidate
42/four-world sample remains unexplained. Its patch, binaries, failed setup
(traceback from a wrapper shadowing stdlib profile), all samples and restoration
proof stay in `.cache/mtg-268/dtype-validation/`. No retry is selected.
Pre-sync equal-work sixteen-world gains were 3.35×/3.62×, distinct from current
post-sync 3.41×/3.45×. Original-world receipts failed comparison after the catalog
merge; `.cache/mtg-268/reference/` and MTG-183/w13 remain frozen, separate from
`.cache/mtg-268/merged-reference/`. There is no cross-world acceptance bridge.

The directive says: "Batching should bring 8-16 worlds near today's one-world cost."
And: "Never run games/evals on the laptop; small fixed-position timing benches are fine, or use the Mac mini (mini-heart.local)."
No numeric definition of "near" or relaxation has been agreed. The 30-second
README value is a budget, not a latency promise. Original near-singleton acceptance
and trained-policy admission remain unmet. Earlier source/runtime reconciliation
is superseded by the successful post-sync rebuild above; model typed/v7+ integration
remains a gate concern. No Wave was selected or modified.

## Remaining gate and acceptance requirements

Gate still owns complete local-update suites and Command-evidence coverage;
model suites `test_value_aggregation`, `test_inference`, `test_agent`,
`test_layer_input`, `test_record_vocabulary`; upstream integration suites
`test_state_records` and `test_shared_effects`; `tests/training/test_active_training.py`,
`tests/env/test_observation.py`, `tests/training/test_action_cutoff.py` and
`tests/verify/test_native_buffer_capacities.py`. In particular, retained terminal
frontier, sign, deadline, legal-support and selected-capacity tests cover cases
absent from the two full-depth opening measurements. Reuse valid earlier results
with their source/runtime identities; the sync's two focused checks do not
replace these suites.

Native gate requires `cargo fmt --check`,
`cargo clippy --all-targets --all-features -- -D warnings` and the complete debug
suite (`cargo nextest run --profile ci`, or debug `cargo test`) from `managym/`,
with bounded workers and one Cargo operation at a time. Direct/composite hash
parity includes `hidden_selection_public_entry_is_pinned_through_both_samplers`
and `search_state_contract`. Applicable engine-budget CI checks remain required;
release-only success is insufficient. Suites that play games, train or evaluate,
including full debug engine tests, belong on the Mac mini or CI under the Task's
laptop restriction. The rebuilt ordinary wheel is restored/verified after any
debug binding validation, before final timing. There is no world/shape change
in this sync; if one is discovered, the documented current-world arena/monitoring
and checkpoint-admission checks become required too, without changing frozen
worlds. Gate also retains changed-file lint/format checks and `git diff --check`.

Open for demo: what maximum 8–16-world/original-singleton latency ratio,
checkpoint and representative positions define "near"? No numeric threshold or
relaxation has been agreed. The eight-world supplement is now measured (see step 9 above); the planned openings are not terminal-heavy or trained-strength evidence.
MTG-183/w13 admission remains unresolved. The README's 30 seconds is a budget,
not a promise. After the borrowed writer, Command/audit and network have similar
profiled costs; no newly identified safe removal closes the gap. Snapshots/allocation
cannot explain it.
Meeting a looser same-runtime target would require an explicit decision, not a
reinterpretation of the existing outcome.
