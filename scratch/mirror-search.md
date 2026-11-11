# Mirror search continuation

## Current state — 2026-11-11 final measurement

**Better: keep compact/native.** Current post-compression binary `3f086259…`,
fresh full/Python controls on the same binary, six retained roots, five alternating
rounds at each of one and three startup threads. Depth 20, alpha 0.02, beta 0.01,
16 worlds. One-thread uniform **full → compact/native = 4.98×** on the sum of
three root medians (17.135 → 3.442 s); step 100 alone is **5.66×**.

| Step 100 | Full/Python median [range], s | Compact/native | Ratio |
|---|---|---|---|
| 1 thread, oracle | 11.539 [8.700–34.930] | 2.043 [1.585–6.136] | 5.65× |
| 1 thread, uniform | 9.737 [8.201–29.699] | 1.721 [1.516–5.384] | 5.66× |
| 3 threads, oracle | 10.169 [8.225–15.060] | 0.945 [0.802–2.206] | 10.76× |
| 3 threads, uniform | 10.239 [8.706–12.563] | 1.106 [0.893–1.837] | 9.26× |

Eleven of twelve root/thread comparisons have separated ranges. One-thread
step-286 uniform overlaps: no measurable difference there despite its favorable
median. No reliable sub-second step-100 result. One-minute host load 87.0–457.8;
all spikes retained. Equal-root ratios are local planning proxies, not measured
seven-worker M1 throughput, campaign weighting or ETA.

120 timed searches / 7,040 rollouts / 132,680 Commands; zero errors. All 144
external warm admissions match retained actual Commands, final observations,
leaf signs and values/targets with maximum numerical error zero. Timed and
profile receipts match their warmups. Twenty-four profiles and 24 companion
timings are excluded. Step-100 compact/native forward takes 85–95% of profiled
time; eliminating native encoding entirely saves at most 1.44%. Audit/witness,
viewer serialization/parsing and Python encoding dominate the full controls.

`scratch/measure.html` retains earlier tables and adds all current samples,
loads, profiles, identities and a ledger row. Raw evidence and reproduction
scripts are in `scratch/mirror-final-20261111/`. Frozen binary `7c22a1e1…` and
candidate `3f086259…` hashes remain unchanged. No new games, mini access, native
rebuild, campaign transition or production default change. No benchmark, server
or watcher remains running. The next action is committing and transitioning the
uniform stage separately; this run performed the final local measurement only.
Compound checks remain deferred to gate/CI for the retained psutil signature
failure. Main sync stays skipped to preserve w13. Earlier snapshots below retain
their original conditions and are not current work orders.

## Native encoder thread sweep — 2026-11-11 measured

**No measurable difference versus fresh two-thread controls; do not advance the
reliable <1 s target.** Five alternating startup 2/3/4-thread rounds, compact
evidence, iteration-4 separate native encoder, unchanged six roots and workload:

| Step 100 | 2 threads median [range], s | 3 threads | 4 threads |
|---|---|---|---|
| Oracle | 1.191 [1.020–2.447] | 0.931 [0.847–1.672] | 0.891 [0.724–2.124] |
| Uniform | 1.383 [1.027–1.752] | 0.924 [0.892–2.488] | 0.995 [0.738–2.080] |

Three/four-thread medians meet the threshold, but all ranges overlap controls
and include calls above one second. All twelve comparisons across the six roots
overlap; no sample removed. One-minute host load 52.0–101.7. Keep the existing
opt-ins and production defaults; no thread-count speedup is established here.

90 timed searches / 5,280 rollouts / 99,510 Commands; zero errors. All 108 external
admissions match retained actual Commands, final observations and leaf signs;
maximum numerical errors are zero. All timed/profile receipt identities match
their admitted warmups. Eighteen separate profiles and eighteen companion timings
are excluded from sweep medians. No new games or campaign/remote access.

Next limiter is CPU forward, especially multi-head attention. Three-thread
step-100 profiles: total 0.978–1.057 s, forward 0.825–0.906 s (84–86%), attention
kernel 0.563–0.611 s (~58% of total, nested within forward). Native encoding is
~0.019 s; removing all of it saves only ~2%. Repeat the same sweep without host
contention before choosing a thread setting or another inference optimization.
This is local, single-worker search evidence, not mini latency or seven-worker
throughput. The easier terminal roots remain measured, not artificially loaded.

All samples, loads, profiles and the new ledger row are in `scratch/measure.html`,
with earlier tables intact. Raw evidence, reproduction driver and hashes are in
`scratch/mirror-native-threads-20261111/`. The frozen binary (`7c22a1e1…`), selected
binary (`7a0d936d…`) and bound implementation sources are unchanged. No production
code/default change or native rebuild in this iteration. The sweep is finished;
no benchmark, server or watcher remains running.

## Direct object encoding — 2026-11-10 implemented and measured

**No measurable latency difference; <1 s remains unmet.** Bind each native
card keyword/type wrapper and permanent keyword wrapper once, and write features
directly into zeroed padded rows. Remove per-object arrays, copies and lambda
adapters. Shape, dtype, feature/object order, padding, validity and capacity errors
are unchanged. Keep this requested simpler allocation path without promoting it
as a measured speed win; no new production mode or thread default.

Golden tensors captured from the previous encoder cover all 303 actor views of
sealed smoke f38ccc90… plus its terminal view. Every pending frame/actor observation
was checked during capture. The small Command/tensor fixture replays without a
checkpoint or private search receipts; every element and byte matches. Five focused
regression/capacity checks pass.

Five alternating fresh control/candidate process rounds, compact evidence,
two intra-op threads, inter-op 16; all six retained roots, unchanged depth 20,
alpha 0.02, beta 0.01, 16 worlds, seeds, checkpoint and native bytes:

| Step 100 | Control median [range], s | Direct rows median [range], s |
|---|---|---|
| Oracle | 1.359 [1.336–3.720] | 1.610 [1.259–3.034] |
| Uniform | 2.223 [1.345–4.037] | 1.311 [1.268–3.598] |

All six paired ranges overlap. One-minute load spans 44.1–106.2; retain every
spike, and do not infer improvement from the favorable uniform median. The
candidate profiles show step-100 forward 2.52–2.53 s and encoding 0.91–0.95 s;
control profiles show forward 4.47–4.48 s and encoding 1.53–1.82 s. Both slices
shifted with contention, so those separate calls do not isolate an encoding gain.
Forward remains the largest slice; the prospective ~0.25 s encoding estimate
was not established under these host conditions.

60 timed searches / 3,520 rollouts / 66,340 Commands; zero benchmark errors.
All 72 warm admissions (including separate profile processes) match retained
actual Commands, final observations and leaf signs; all numeric errors are zero.
Timed/profile receipt identities match their admitted warmups. Twelve profile
calls and twelve companion timings are excluded from sweep medians. No new games,
mini access, remote-source change, checkpoint change or native rebuild.

Evidence and exact encoder hashes: `scratch/mirror-encoding-20261110/` contains
control/candidate source archives, the startup bootstrap, frozen-harness child
results, captures, receipts, profiles and `summary.json`. All samples, profile
components and the ledger row are in `scratch/measure.html`; older tables remain.
The existing search-source digest omits observation.py, so this comparison binds
encoder bytes separately instead of claiming that digest identifies the encoder.
At this snapshot the native encoder was still prospective; the later native
implementation and admission are recorded in the live plan. No benchmark,
server or watcher remains running.

## Startup inference threads — 2026-11-10 measured

**Better at step 100 with two threads; iterate on encoding.** The prepared
1/2/4/8 sweep completed five alternating rounds, each count in its own sequential
process, using all six fixed roots and fresh compact one-thread controls.
Depth 20, alpha 0.02, beta 0.01, 16 worlds; unchanged checkpoint, native binary,
search source, histories, seeds, cache policy and inter-op 16.

| Step 100 / sampler | 1 thread median [range], s | 2 threads | 4 threads | 8 threads |
|---|---|---|---|---|
| Oracle | 1.894 [1.876–2.167] | 1.362 [1.334–1.787] | 1.159 [1.076–2.015] | 1.302 [0.979–2.298] |
| Uniform | 1.894 [1.883–2.098] | 1.349 [1.344–1.394] | 1.183 [1.077–1.291] | 1.296 [1.077–1.555] |

Two threads improve step-100 oracle/uniform medians 28.1%/28.8%, with disjoint
ranges. Four threads improve uniform 37.5%; their oracle range overlaps the
control. Other overlapping comparisons are labeled no measurable difference,
not promoted from lower medians. No count establishes <1 s across the roots.
One-minute host load was 35.2–56.4; all 120 sample loads and all six-root tables
are in `scratch/measure.html`, with previous tables retained. These controls
supersede the old wall timings for this comparison only.

120 timed searches / 7,040 rollouts / 132,680 Commands; zero benchmark errors.
All 120 external admissions match actual Commands, final observations and
leaf signs; maximum errors for base, values, target and rollout values are zero
at unchanged atol=1e-6/rtol=1e-5. Every timed receipt matches its admitted warmup.
24 separate profiles, 24 warm admissions and 24 companion timings also pass
and are excluded from sweep medians. Existing Commands were replayed; no new games.

Step-100 profiles: forward 1.38–1.43 s at one thread, 0.84–0.85 s at two,
0.57–0.58 s at four. At four threads, encoding takes 0.534 s (~43% of the
1.25 s profiled path); native apply ~0.045 s. Object encoding accounts for
0.413 s, including card features 0.282 s and permanent features 0.082 s.
The 399 NumPy stacks take only 0.011–0.012 s (<1%): dropping stacking alone
cannot achieve the target. Prospective direct feature writes into preallocated
frontier arrays target 0.20–0.30 s of encoding, subject to tensor equality and
whole-search admission. No gain or sub-second result is claimed for that work.

Evidence: `scratch/mirror-threads-20261110/{result,summary}.json`, archived
child sources/receipts/captures/logs, and `scratch/mirror-thread-profile-20261110-{1,2,4,8}`.
Renderer: `scratch/summarize_mirror_threads.py`. The compact source digest remains
`b571e727ef1a000a34304f9dca56466ce6e13494aff44e0b76aecfc24e086bbc`.
No production default, search implementation, mini campaign or remote source
changed. Keep measurement thread selection for controlled follow-up; no evidence
here authorizes a seven-worker thread change.

Focused harness validation: 3 passed; Ruff and diff checks pass. The renderer
initially found multiple `forward` methods, then was corrected to select
`ActorCritic.forward` by its source definition. This reporting error changed no
measurement or admission, and required no search rerun. No benchmark, server or
watcher remains running.

## Compact rollout evidence — 2026-11-10 completed

Implemented the approved `rollout_evidence="compact"` flag. Full remains the
default; compact v4 omits per-step native witnesses and actor viewer hashes while
independently counting successful Commands and checking native counters. Full
v3 receipt config and historical registration bytes remain unchanged. No mini
access, campaign transition, game, model/input change or native rebuild.

Same six retained roots, MTG-183/native identities and seeds as the baseline;
CPU/one intra-op thread, inter-op 16, five alternating full/compact repeats:

| Step / sampler | Full median [range], s | Compact median [range], s | Reduction |
|---|---|---|---|
| 0 / oracle | 7.881 [7.538–9.382] | 3.074 [2.794–3.472] | 61.0% |
| 0 / uniform | 4.292 [3.855–7.975] | 1.782 [1.194–2.297] | 58.5% |
| 100 / oracle | 17.646 [11.934–22.191] | 3.155 [2.852–4.106] | 82.1% |
| 100 / uniform | 16.584 [11.086–20.636] | 3.533 [2.848–6.678] | 78.7% |
| 286 / oracle | 5.154 [3.463–7.855] | 0.942 [0.732–1.755] | 81.7% |
| 286 / uniform | 10.141 [8.791–11.690] | 2.513 [1.982–3.210] | 75.2% |

All paired ranges are disjoint. The earlier 8.5 s baseline is not the current
control: host contention shifted substantially (one-minute load 74.6–191.8).
This establishes a local fixed-root improvement, not production-mini latency,
seven-worker throughput or a strength result. The <1 s target remains unmet.

Untimed external admission captured every actual branch Command and both final
viewer observations; full/compact agree exactly. Every retained behavioral field,
sampled world, terminal sign, value and target agrees (maximum float error zero).
Full receipts also match the archived baseline except source identity/timing.
All 60 timed calls and 12 separate profiles match their mode's warm receipt digest;
3,520 timed rollouts / 66,340 Commands, zero benchmark errors. Root witnesses and
replayed pending frames remain unchanged. Only 286 existing Commands replayed.

Separate step-100 compact profiles: oracle 3.215 s total (forward 2.103 s,
encode 0.807 s, native apply 0.095 s); uniform 6.043 s total (forward 4.627 s,
encode 1.024 s, native apply 0.078 s). These are profiler calls, not the medians.
**Iterate next on explicit CPU forward thread count (1/2/4/8), then encoding
allocation/copying if needed**, using the admission gates and quantitative
hypotheses in `scratch/mirror-rollout-evidence-plan.md`. No thread or encoding
optimization was implemented in this step.

`scratch/measure.html` contains the paired timings, all component tables and
both ledger rows. Evidence is `scratch/mirror-compact-20261110/{result,summary}.json`,
with source archives, compressed receipts, external Command/final-observation
captures and pstats. Search source digest is
`b571e727ef1a000a34304f9dca56466ce6e13494aff44e0b76aecfc24e086bbc`.
Harness: `scratch/measure_mirror_compact.py`; renderer:
`scratch/summarize_mirror_compact.py`. Original baseline bytes are preserved.

62 focused checks pass, five game/fitting tests deliberately deselected; Ruff,
format and diff checks pass. The new native-precondition fixture initially used
an offer ID outside u32; corrected to a different valid-width ID and reran the
focused checks. No production change or benchmark rerun was needed for that test
fix. No local benchmark, server or watcher remains running.

## Fixed-root latency baseline — 2026-11-10 completed

Request: “Measure the baseline seconds per mirror local-update search decision
at depth 20, beta 0.01, 16 worlds on fixed roots (steps 0, 100, 286 of the sealed
smoke attempt, oracle and uniform), with a per-component breakdown.”

Worse than the <1-second target on this laptop; no optimization gain or mini
latency is established. Existing batched source at HEAD `7fd6c202f163`, native
`7c22a1e1…`, checkpoint `4f2ad656…`, full audit/hash evidence, alpha 0.02,
CPU/one torch intra-op thread (default inter-op 16), warm-cache search-only timing:

| Step | Oracle median [min–max], s | Uniform median [min–max], s | Work per call |
|---|---|---|---|
| 0 | 2.917 [2.819–3.193] | 2.941 [2.884–2.965] | 64 rollouts, 1,280 Commands |
| 100 | 8.510 [8.433–10.006] | 8.414 [8.370–8.894] | 80 rollouts, 1,600 Commands |
| 286 | 3.234 [3.208–3.294] | 3.275 [3.220–3.417] | 32 terminal rollouts, 437 Commands |

Five alternating repeats per cell, one excluded warmup and one separate cProfile
call per cell. Thirty measured searches / 1,760 rollouts / 33,170 Commands;
zero errors. Every repeat/profile matches its warmup's complete receipt digest
excluding timing. All six warmups also match the retained batching admission's
discrete receipts and floats exactly (max error zero). Replay checked the sealed
pending frames/observations and root immutability; no new games. Host is Apple
M4 Max; one-minute load 65–249 during timed calls. Oracle/uniform ranges overlap
at each root. These are not paired comparisons with historical mini timings.

Separate step-100 oracle profile: native apply with audit 4.881 s (57.0%),
viewer JSON 1.079 s (12.6%), viewer parse 0.460 s (5.4%), network forward 1.471 s
(17.2%), encode 0.569 s (6.6%), receipt assembly 0.017 s, other 0.085 s; recorded
profile total 8.562 s. Native apply includes actual game execution as well as
witnesses. A separate five-repeat alternating single-Command check gave audit
on median 2.834 ms [2.808–3.070], off 0.0276 ms [0.0223–0.0471], with matching
post-Command viewer observations. One-world step-100 uniform diagnostic did
5 rollouts / 100 Commands in 0.549 s; it is not a repeated speed claim.

Even subtracting *all* native apply and viewer JSON/parse from the step-100
profiles leaves 2.06–2.14 s. That is an optimistic profile arithmetic bound,
not a fast-mode measurement: actual game application and root checks must stay.
Audit/hash removal alone therefore does not establish the <1 s target; inference
and encoding remain material. Profiles are separate from reported wall samples.

All six component tables, raw timings, identities and limits are in
`scratch/measure.html` and `scratch/mirror-latency-baseline-20261110/{result,summary}.json`.
Full warm receipts and pstats sit alongside them. Frozen measurement harness:
`scratch/measure_mirror_roots.py`; renderer: `scratch/summarize_mirror_latency.py`.
The original `scratch/profile_mirror_root.py` is unchanged. The receipt counters,
update algebra, terminal/depth limits and historical comparisons passed; no
production code changed or rebuild was needed. No benchmark remains running.

Next action is planned in `scratch/mirror-rollout-evidence-plan.md`: explicit
full/compact evidence mode, new labeled receipt schema, independent Command
counter, preserved root validation and interleaved full/compact admission.
The mini campaign and remote source were not accessed or changed.

## Decision-latency optimization — 2026-11-10

Jack Heart: "how could we bring down the 11.6 seconds/search decision to <1sec
for that # of shuffles", then "is the bottleneck here iference time or game time
or something else", then "yeah" to implementing steps 1 and 2 below behind a
flag and measuring on the same fixed positions against current receipts.

Target: one depth-20, beta 0.01, 16-world oracle search decision under 1 s
(mini campaign currently ~11.6 s with seven workers contending).

cProfile of one fixed root (smoke attempt f38ccc90..., step 100, oracle, 16
worlds, 80 rollouts, 1,600 Commands, one torch thread, laptop at load average
55-70 so absolute seconds are rough), total ~8.5 s:
- native `SelectedBranchRuntime.apply_policy_choice`: 4.9 s. Session is opened
  with `audit=True` (local_update.py), so `selected_witness` runs before and
  after each Command. Direct timing, five repeats: 3.3 ms/Command audit on,
  0.027 ms audit off.
- `semantic_observation_json` + `Observation.from_json` per step, used only for
  `viewer_state_hash`: ~1.5 s (0.6 ms native JSON + ~0.3 ms parse).
- batched forward (21 calls): ~1.4 s; 80-row batch 68 ms on 1 thread, 25 ms on 8.
- Python `space.encode`: ~0.35 s (0.21 ms each).
- clone 0.005 ms, observation_for_player 0.006 ms.
The MTG-268 frontier batching port only touched the forward slice (0.93-1.45x).

Order of work: (1) flag to run rollouts without per-step witnesses; (2) drop or
natively compute the per-step viewer hash; then (3) forward threads/batch and
(4) encode. Steps 1-2 change receipt contents (`branch_audit_json`, per-step
hashes): a new implementation identity, not mixable unlabeled with the running
campaign. The running mini campaign (Etude attempt 3a2b62144b9a) and its remote
source must not be touched; no games on the laptop, fixed-position timing only.
Profile script: `scratch/profile_mirror_root.py <attempt.jsonl.gz> <step>`, run
with `PYTHONPATH=$PWD uv run --project /Users/jack/src/manabot.search --no-sync python`.

## Mini access — 2026-11-10

Jack Heart: “mini is at 100.96.227.95”. SSH to `jack@100.96.227.95`
works and reports `mini-heart.local`; use this address when `.local` resolution
is unavailable. This changes no Etude allocation or running experiment.

## Analysis handoff — 2026-11-09 22:32 UTC snapshot

The existing report now uses 78 sealed v2 outcomes (27/26/25 oracle at 4/8/16
worlds), retaining all 3,000 slots and 2,922 missing. The audit reconciles
12,890 Commands and 6,170 search receipts; no failed outcome or worker warning
is retained. Prior game, decision, attempt and log bytes remain identical prefixes. Eight
uncommitted attempt filenames were listed but not read or counted as failures.

Seven sequential workers and the original Etude attempt remain active. No
execution or resource change occurred; batching remains local and undeployed.
The report owns partial estimates and cost. Full-grid uncertainty, uniform-hand
effects, replay proof, training exposure and scientific acceptance remain
unresolved. Read the protocol's newest snapshot section and existing report;
the older 77-row batching inventory no longer leads the scientific snapshot.

Publication updated 97 manifest entries; all 224 mirror artifacts verify.
The report/summary rebuild byte-for-byte and all v1 scientific fields remain
unchanged. Seventeen focused tests, Ruff and the post-publication untracked-data
check pass. Prior broader collection failures remain unresolved. No local
server, watcher or new experiment process remains running.

## Earlier history

Earlier batching admission, campaign launch, reconciliation, v1 completion and
conversation quotes remain in `7fd6c202f163bc7dd0d507c7eab121086af84a68:scratch/mirror-search.md`.
The owning protocols are `experiments/protocols/search-mirror-v1.md` and
`search-mirror-v2.md`; they retain scientific limits and execution accounting.
V1 completed all 8,000 planned outcomes. V2 remains the existing mini campaign;
its immutable source and Etude attempt `3a2b62144b9a` must not be changed by this
latency optimization. Prior full-study import failures (Etude/duckdb) do not
authorize an environment upgrade.
