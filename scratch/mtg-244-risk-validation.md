# MTG-244 performance and learning risk validation

Jack Heart directed this Task on 2026-11-04: "ok. lets make suer we measuer
these risks and attempt to ameliorate them". This adds measurement and mitigation
to the current delivery, not a separate Task. Keep the live Graph and existing
review boundary. No merge or training experiment launch is authorized.

## Accepted evidence contract

Before review readiness, compare identical ordinary TLA and dependency step/search
workloads, complete payments, and all seven full-policy/encoding cases on the same
runner. Retain legal choices, full score/step digests, simulation/cap counts,
latency samples, allocation/operation attribution, input/parameter/rollout bytes
and process memory. Name exact sources, hardware and configurations. A changed
world is not an equivalent speedup. No artificial load or baseline reset.

Fix concrete findings and remeasure. Shared/local v8 programs must preserve
strategic distinctions, numeric meaning, boundaries/ownership and gradients;
padding and irrelevant remapped order must be invariant. Check native/Python/
text/token parity, all model arms, real collector/checkpoint selection and early
older-input rejection. Keep capacities complete, fresh v6 and frozen saved
versions. Full suites and the unchanged 10% Engine Budget belong to Gate.

Measured representation and computation do not establish sample efficiency,
win rate, generalization or accelerator performance. Training remains separately
authorized. Complete original acceptance and earlier evidence are retained in
blob `3078761c09e3675c043bde49627af12a47bbec31`.

## Retained representation findings

Before this source-plan cut, mitigations fixed cross-holder recurrence, packed
independent live programs, batched shared program reduction and used v8-only
signed log magnitudes. The predecessor's isolation assertion differed in 5/16
elements (max 0.3391394); numeric values 4096/8192 collapsed to identical outputs
before normalization, versus 0.0576074 separation afterwards. Shared B16 dispatch
fell from 16 GRU/pack/index-add and 416 linear calls to 1/1/1 and 71. Padding,
independent remapped order, meaningful program/timestamp order, source/target
joins, duplicates and gradient paths have focused proofs across all v8 arms.
Collector transition/checkpoint admission are exercised; no learning claim.

Exact earlier evidence: independent-program sources/receipts and full command
history are in blob `3078761c09e3675c043bde49627af12a47bbec31`;
cache/model mitigation bundle `b273985619373fe6c86996eae2e27b5fb440fd14`;
complete-payment/inventory bundle `88d6abd34d8b56a9606c3ec58696063955d08226`;
parity receipt `81fa8070f9ff772ff01b2eab6c210e2ac72e362b`.
Earlier timings on a host with load above 500 establish no speedup. Their exact
contrary rows and interrupted runs remain in those sources, not acceptance.

## Completed isolated predecessor comparison — 2026-11-04

Layer Risk run `41952171271` succeeded. Artifact:
`.cache/mtg-244-second-risk`, including `comparison.json`, CPU/dependency records,
source/harness/binary hashes and raw samples. Reference `ea316ff14`, before
`fb3555ffb`, after `beaacaa19`. GitHub Ubuntu VM, Intel Xeon 6975P-C; all work
sequential. Every full score/action/step/simulation digest matches. These numbers
measure earlier mitigations, **not the source-plan cut**.

| Workload | Reference | Before | beaacaa19 |
| --- | ---: | ---: | ---: |
| Original TLA, 512 games / 101,650 steps, zero caps | 5.447 s | 15.903 s | 7.715 s |
| Search, 120 decisions / 21,504 simulations, zero caps | 116.593 s | 415.910 s | 178.709 s |
| Worst of 120 search samples | 3.439 s | 14.007 s | 5.533 s |
| Dependency, 16 games / 689 steps | Not comparable | 0.1000 s | 0.0617 s |
| Dependency search, 8 decisions / 1,600 simulations | Not comparable | 10.831 s | 5.340 s |

The corrected producer now supplies credible evidence, but the engine regression
remains unaccepted. Separate Engine Budget run `41952171449` also fails:
166.656→230.730 s search, 4.883→7.045 s worst, 12,938.9→9,904.0 steps/s on its
AMD EPYC runner. Never compare wall time across these runners. Complete-payment
allocation traffic improved 1,176→468 allocations / 295,717→94,777 B (pool) and
2,882→541 / 830,804→98,473 B (lands), with all 16 full traces/resources matching.

Full policy: Torch 2.10 CPU float32, one thread, B16/D64/depth2/heads4/FiLM,
512 facts in every version, 512 actions, 60 cards/40 permanents per seat.
Two warmups/eight samples per operation in fresh processes. Median ms:

| Input/state | Inference before→after | Forward/backward before→after |
| --- | ---: | ---: |
| v6 ordinary | 117.10→122.74 | 288.82→310.53 |
| v7 ordinary | 122.25→122.33 | 287.75→290.86 |
| v8 ordinary | 115.02→122.84 | 282.49→292.88 |
| v6 shared | 122.63→126.28 | 310.92→315.36 |
| v7 shared | 126.02→130.91 | 325.54→328.18 |
| v8 shared | 129.72→125.29 | 348.78→324.94 |
| v8 program | 137.41→125.41 | 376.93→337.68 |

All seven native/Python observations, configs, storage and occupancies match.
Program peak process RSS improves 936.7→932.2 MB (imports/warmups included);
other cases span 917.0–933.7 MB. Native encoding remains 0.96–1.48 ms, Python
0.16–0.38 ms across cases. V8 program native encoding 1.469→1.454 ms, Python
0.249→0.251 ms. These small samples support the targeted program improvement,
not a general speedup: the ordinary controls retain variation/regressions.

Matched observation bytes are 133,616 / 165,360 / 208,368 for v6/v7/v8 with
512 facts. V7's historical 128-fact default is 146,928 B. B16 v8 inputs use
3,333,888 B; parameters use 7,761,416 B versus v7 7,555,592 and v6 4,357,128.
Rollout observation storage is N times the individual payload before overhead.
The program fixture has 21 live details and 11 facts, not a universal occupancy
bound. All 4,096 detail rows and explicit overflow remain. V8 program versus v7
ordinary adds about 2.5% inference and 16% forward/backward on this receipt;
those are different semantics, not an equal-input speed comparison. Accelerator
synchronization, sample efficiency, win rate and generalization remain unmeasured.

## Source-plan redesign: current local evidence

The 2026-11-04 cut caches copiable source objects and bound static plans by
immutable definition identity/incarnation/timestamp; evaluation objects detach on
writes. Conservative read/write summaries avoid impossible semantic probes;
plausible pairs still compare exact instructions and existence, with loop
fallback and rediscovery. Fixed activation views remain shared when finalization
would write nothing. Whole-board invalidation remains conservative. Debug checks
all skipped pairs and rebuilds source plans independently.

Before is `beaacaa19`; after is `614a38fc87942d4fa66726486e6ba366d86268fc` (exact patch retained in
`.cache/mtg-244-plan-final/source.patch`). Apple M4 Max/macOS 26.0.1; sequential
release `benchmark` builds, controlled existing fixtures, no artificial load.
Raw files: `.cache/mtg-244-plan-{before,after,final}`. Every step/search/full score
and complete-payment trace matches; 101,650 ordinary steps, 120 Search-64
decisions / 21,504 simulations, 689 dependency steps and 1,600 dependency search
simulations remain unchanged, zero caps.

| Cost | beaacaa19 | Current cut |
| --- | ---: | ---: |
| Ordinary board rebuilds | 205,709 | 205,709 |
| Ordinary context clones / semantic comparisons | 40,622 / 45,050 | 0 / 0 |
| Copiable source constructions | Rebuilt per board | 17,870 total |
| Dependency steps allocations / requested bytes | 853,919 / 200,939,692 | 596,201 / 120,592,958 |
| Dependency search allocations / requested bytes | 77,885,256 / 18,067,547,041 | 52,768,808 / 10,446,007,092 |
| Dependency search comparisons | 301,294 | 275,568 |
| Native step-process peak RSS | 10,289,152 B | 10,502,144 B |

Complete pool/land payments already rebuild zero boards at beaacaa19; the cut
retains exactly 468/541 allocations and 94,673/98,369 requested bytes locally.
The 104-byte difference from Linux receipts is platform allocation layout, not
a changed payment. Source retention costs about 213 KB peak RSS in this bounded
step run; it shares immutable data across forks and releases departed seeds on
the next board rebuild. Native heap traffic is not peak live memory.

Local latency is exploratory under variable load: first cut search 99.421→69.058 s;
final cut 85.520 s, worst 2.228 s versus predecessor 3.006 s. Steps 5.017→3.559 s;
the memory pass separately measured 4.145→3.319 s. This variance prevents budget
acceptance. Independent probes disappear in the ordinary workload; dependency
positions retain exact comparisons and per-target instruction/object allocation.
Per-component invalidation and instruction borrowing could remove more work,
but add dynamic dependency/ownership complexity; these receipts support the
source-sharing cut first. No baseline has moved.

Focused debug layer/cache and tiny-comparison checks pass (62.89 s versus
97.84 s locally); seven policy cases match. Local policy timings remain
exploratory. Exact commands, samples and prior figures remain in the bundle below.

Exact raw local and hosted receipts, binary hashes, commands, source patch and
verification logs are preserved in gzip tar blob
`49f516b3b42188a9e170e59a870907632d17fbcd`
(`git show <blob> > evidence.tar.gz`); local copy
`.cache/mtg-244-plan-evidence.tar.gz`. The installed predecessor wheel was retained
by hash before atomic replacement; Python model/encoder sources did not change
in this cut. All native workloads include matched choices/counts/digests, and
ordinary step receipts retain the first 128 step samples.

## Measured residual and final SBA cut

Source-plan checkpoint `a7b0ddaff` CI `41955382680` passes long integration,
including the previously failing tiny comparison. Engine Budget still fails:
reference/candidate steps 7.339/8.112 s (10.54% slower), search 155.396/169.924 s,
worst 4.567/5.007 s. Full digests/counts match. Artifact:
`.cache/mtg-244-plan-ci-budget`. This improvement is not accepted as completion.

A bounded one-second profile of the final ordinary workload found eager
`snapshot_trigger_sources` calls on unchanged SBA checks. The final cut defers
that snapshot until immediately before the first death/detachment/token-removal
mutation, retaining one pre-batch source set for all simultaneous events.
No rule read, choice, input capacity or debug assertion is suppressed.

Compared with `beaacaa19`, final dependency-step allocations/bytes are
563,724 / 113,331,651 (was 853,919 / 200,939,692); dependency-search totals are
49,115,896 / 9,475,236,036 (was 77,885,256 / 18,067,547,041). Pool payments now
use 436 allocations / 88,909 B, land payments 509 / 92,605 B, retaining all 16
complete traces. Every ordinary/dependency/full-score digest and count matches.
Local final search 58.861 s / 1.709 s worst and steps 3.050 s remain exploratory;
contrary dependency timing (2.164→2.754 s despite lower allocation) confirms
host variability. Per-target instructions and conservative full-board rebuilds
remain costs; deeper invalidation adds dynamic membership/dependency complexity
for a smaller remaining layer share. No promised speedup or budget exemption.

Eleven debug CR704 cases and the tiny comparison (53.24 s) pass; seven policy
cases match after atomic release-extension rebuild. Exact patch over `a7b0ddaff`,
binary hashes, samples, profiles, policy receipts and CI artifact are gzip tar blob
`6baa683f7ec901ea243538ec53e91108429f66ed`; local copy
`.cache/mtg-244-lazy-sba-evidence.tar.gz`. This supplements the source-plan bundle,
not its timing acceptance. Source-plan peak RSS above predates the lazy-SBA cut.

## Isolated ordinary gains at the lazy-SBA checkpoint

Run `41956687926`, job `127013635221`, measures `68ef4a63d` against pinned
`ea316ff14` on one GitHub Ubuntu 24.04 AMD EPYC 7763 VM (4 vCPU, release).
The unchanged 10% Engine Budget passes. Search total is 164.461→132.934 s
(19.17% less time), worst reply 4.818→3.860 s (19.88% less), and steps
7.772806→7.104065 s (8.60% less; 13,077.6→14,308.7 steps/s). All three full
score digests, action lists, 120 decisions, 21,504 simulations, zero caps and
101,650 steps match. This supports ordinary-workload gains beyond merely
repairing the guard. It does not isolate the lazy snapshot from the source-plan
cut, establish dependency/model costs, or certify the later simplification.

Raw artifact and provenance: `.cache/mtg-244-lazy-sba-ci-budget`, retained as
gzip tar blob `402a408e4d96fee1eba9c561f553116af591c19c`.
The existing comparator passes on the downloaded artifact; full score/action/
simulation equality was also checked directly. No reference or tolerance moved.
That head's long integration, Academy UI/integration, clean-machine Play,
Python Unit and both conformance jobs pass. Rust and main integration were
cancelled; the aggregate failed and run concluded cancelled. These are individual
job results, not a green aggregate.

The `ed25b277e` simplification now in published head `7851e6765` removes repeated
read walks, fixed-node assembly, redundant discovery and temporary static-source
vectors. It retains 16 complete payment traces and all dependency digests/counts;
dependency allocations fall 563,724→552,779 (steps) and
49,115,896→48,159,001 (search) against `68ef4a63d`. Pool/land counts stay 436/509.
Exact receipts and patch: gzip tar blob `928f19c27f4729601867502f72b3da7180a255e6`.
This is allocation evidence, not a new latency claim. Main's accompanying
pilot-evidence sync changes no engine/model source.

Source-plan Layer Risk `41955382652` also completed successfully during
reconciliation: `ea316ff14` / `fb3555ffb` / `a7b0ddaff`, one AMD EPYC 7763 VM.
Original steps are 7.784 / 20.955 / 7.898 s; search 167.890 / 544.698 /
157.258 s; worst 4.910 / 18.167 / 4.588 s. Dependency steps improve
0.1355→0.05865 s and search 14.2605→4.7791 s against `fb3555ffb`; the old
reference cannot represent that dependency comparison. All workloads, 16
payments and seven policy cases match through the comparator. V8 program
inference is 214.79→205.49 ms and forward/backward 506.58→451.98 ms under
the same B16/D64 CPU settings; ordinary controls vary, not uniformly improve.
This receipt does not erase that head's separate 10.54% Engine Budget failure:
runner-paired comparisons stand on their own. Complete samples/configs/memory
are in `.cache/mtg-244-source-plan-risk`, gzip tar blob
`2255e42dda6862d97635b359cf08c7f985311dd4`. It predates lazy SBA and simplification.

## Producer and remaining acceptance

Reproduction: `docs/development.md#bounded-layerinput-attribution`.
`scripts/measure_layer_risk.py` and `layer-risk.yml` retain exact historical
sources and all partial receipts; the comparator rejects missing cases,
nonfinite/incomplete samples, different configs/storage, incomplete seed/payment
identities and changed semantic digests. Eighteen injected-receipt checks cover
that contract. Inventory-only output cannot satisfy timing comparison.

Layer Risk `41958999383` completed every comparison for `e4e527621`; results
are in [CI repair](mtg-244-ci-repair.md). `c46e4f3d8` changes `layers.rs`, so
Layer Risk `42035237232` and CI `42035238111` own current evidence. An ordinary
budget pass does not replace dependency search, complete payments, seven policy
cases or suites. Residual costs still include conservative whole-board
invalidation and per-target instructions; further redesign needs cost evidence
and matched remeasurement. No baseline adoption, merge, training experiment or
learning-quality claim is authorized.
