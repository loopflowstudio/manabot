# Recording and statistics cost (Task 80db9a)

## Question

What CUDA collection and update cost does each recording, statistics and
diagnostics option add to Elemental training, and what should stay on or be
sampled? This is exploratory, with no playing-strength claim.

## Current verdict (2026-11-17)

**Inconclusive: incremental option overhead is not measured.** A CUDA baseline
completed, but the next cell exposed a shared tracking identity in the driver.
That driver defect is repaired and locally admitted with distinct declarations.
Two subsequent A6000 creates were refused; complete provider inventory confirmed
both pods absent and the launcher reconciled both requests. L4 availability
reported no stock. There is no live measurement rental and no paired result.

The one completed edited-tree, recording-off cell at `978cec8bc2` ran on an
RTX A6000 / AMD EPYC 7543. It is not a pre-change baseline:
213,509 decisions, four updates and 1,029 completed games, zero crashes or
abandoned games. Collection: **1,790.827 decisions/s** (119.224 seconds total).
Collection plus learning: **43.239 seconds/update**. This is n=1, no range,
and no option-cost estimate. The repaired comparison at `a5afe273f3` must run
fresh matched baselines on its eventual host; do not reuse this one across hosts.

## Method

The frozen driver is `experiments/recording_cost.py` at
`1a3777d18987a8a7bc8fa3d5e2b797e2cbcd9ade`. Seven arms run sequentially on
one rented GPU host: all off; native statistics with its persisted sink;
game recording at 1%, 10% and 100%; escalated durable journals; and the standard
operation marker. Each uses seed 80161, three rotating repeats, four updates
per cell and a fresh process per cell. Compare collection decisions per second
and seconds per update with off in the same repeat, reporting median and full
range. Include first-update costs and check receipts for equal counts.

The Elemental recipe retains 256 streams and its 64×2 model. Every arm replaces
its frozen input v9 prospectively with current input v11; historical recipe
bytes stay unchanged. CUDA is required, with no CPU fallback. Journals include
escalated profiler scopes and recording fraction zero (diagnostic cursors,
without per-decision state digests). Statistics include writes inside collection.
The marker arm includes watchdog and signal handling. Arms keep the same native
game and training workloads and introduce no artificial load.

One job allows at most three execution hours under a $5 job cap; its quoted
window includes startup and must fit the $25 experiment commitment limit.
The final admission pass permits at most three further creates on RunPod
A6000/A40/L4, each under a fresh run ID after verified settlement of its
predecessor. A non-400 refusal or unknown state ends the pass. No Lium rental.

## Earlier outcome and limits (2026-11-16)

**Not measured.** The final pass admitted an A6000, but the first `off` cell
failed before its first update: the recipe requested tracking experiment
`self-play` inside a job launched as `recording_cost`, which Etude rejected as
`experiment_differs_from_launch`. No collection rates, update times, matched
ratios, spread or evidence-based default recommendation were produced.
The rental is terminated and its absence is confirmed. This ends the bounded
pass; the remaining creates were not used.

Keep the documented defaults as provisional: standard diagnostics, recording
off and native statistics off unless requested. This run supplies no cost-based
reason to change them or recommend a sampling fraction. A future measurement
needs its launched Job and training tracking experiment names aligned before
admission; the frozen driver and this failed attempt remain unchanged.

The all-off arm includes the capture's existing frame reads. This method
measures incremental option cost and cannot separately attribute those reads.
The pinned Etude transport discards provider HTTP error bodies, so a status-only
refusal cannot establish why creation failed; that gap belongs in Etude.

<details>
<summary>Dated admission and settlement log (2026-11-16, UTC)</summary>

Provider and storage credentials were injected from Doppler `etude/prd` and
`manabot/prd`, without printing values or changing auth. Quotes describe planned
windows, not bills or confirmed rentals.

- Initial availability: L4 unavailable; A40 listed available but admission
  refused capacity. A6000 quoted $0.69/hour and $2.2425 including startup,
  under the $5 cap. The CLI controller scanned retained storage before
  admission; submission used the public launch API with an aggregate cap check.
- **22:54:46, initial create:** launch `edf5992304ee`, requested run
  `recording_cost/6bffcb5ff678`, machine `mtg338-a6000-2e3aec59e889` failed
  with `ProviderUnavailable: POST pods: status 400`. Source preparation
  completed, but Etude reported "job command not started" and "no run recorded".
  The create result was initially unknown, so no acquisition was retried.
  At 22:55:46 a complete inventory read confirmed no `etude-2e3aec59e889` pod;
  `etude terminate` reconciled the request. Launcher cost: **$0.00**.
- Follow-up retained run and machine logs carried no response body. The pinned
  `etude.launch.providers.http.send_to_provider` returns an empty body on
  `HTTPError`; the refusal reason is unrecoverable. Earlier rentals saw 400
  when quoted stock disappeared before creation. Stock refusal is the likeliest
  interpretation here, not a recovered reason.
- **23:04:26:** A6000 again quoted available, 0 of 1 in use.
- **23:09–23:10, preflight only:** the prior machine remained unavailable;
  complete inventory at 23:09:57 found no matching pod, and availability at
  23:10:09 again quoted A6000. `runpodctl pod list` failed with
  `lookup rest.runpod.io: no such host`, so the then-required second read
  stopped the pass before any create. An earlier local availability invocation
  lacked storage credentials; injecting both Doppler projects allowed the read.
- **23:11:** `rest.runpod.io`, the host used by runpodctl 2.14.0, was NXDOMAIN
  from both 1.1.1.1 and 8.8.8.8 (SOA `runpod.io`); `api.runpod.io` resolved.
  The reason for the missing hostname is unknown. The amended settlement rule
  uses Etude's complete provider inventory plus refused/unavailable session
  status, and runs runpodctl as well only when its host resolves.
- **23:16:13–23:16:16, final-pass preflight:** the prior session was refused
  and a complete inventory again showed no matching pod. A6000 was available
  at $0.69/hour; A40 and L4 were unavailable. `rest.runpod.io` still did not
  resolve, so runpodctl was skipped under the amended rule.
- **23:16:36, final create preflight:** prior session still refused and absent
  from the complete inventory. A6000 again quoted $0.69/hour and a $2.2425
  startup-inclusive window. Fresh requested run `recording_cost/final-1-20261116`,
  launch `f901baac793d`, submitted at 23:16:38 with the same frozen source
  and $5 job cap.
- **23:19:48, admission:** machine `mtg338-a6000-8182b919f405` confirmed
  (provider id `8vvsbploeuzkc2`, attempt `880cb249e3a6`). Runtime setup exited
  0. The driver reached `START 0 off`, then raised:

  ```text
  ValueError: track(experiment='self-play') in a run etude launch started for 'recording_cost'; omit experiment= here, or name it on the launched Job
  etude.errors.EtudeError: experiment_differs_from_launch
  ```

  The first cell exited 1 before any update, the driver stopped, and Etude
  retained the failed attempt at 23:21:52. No other arm ran and no summary
  was produced. This is a tracking-label failure, not a provider refusal or
  a measured recording slowdown. No further create was attempted.
- **23:21:56–23:22:50, cleanup:** `etude terminate` returned the machine
  deleted at 23:21:56.629313; the complete provider inventory at 23:22:50
  confirmed no `etude-8182b919f405` pod. The session has `ended_at` and final
  run status has `rental_ended=true`, outcome `failed`. Launcher run cost:
  **$0.020903, estimated**, not a reconciled provider invoice.

</details>

## Continuation authorized 2026-11-17

Jack Heart: "run some measurement probably. what are the highest risk parts of the chagne".
The next comparison aligns both the launched Job and `execute_regime` on
`recording_cost`, uses five rotating repeats (35 cells), and retains receipt
error counters. It measures the current software including the QA repairs,
with a fresh prospective source commit; earlier attempts above are unchanged.
The $5 job cap and $25 total experiment cap remain. No rate is claimed before
successful CUDA receipts. The expected report is the incremental collection
and update cost of each option relative to off, with median and full range.

Continuation launch `c305376580de`, requested run
`recording_cost/measure-20261117-978cec8`, uses frozen source
`978cec8bc245c07f74fbadb4ea159797479e1d7a`. A6000 quote: $0.69/hour,
$2.2425 startup-inclusive window, $5 job cap. Machine
`mtg338-a6000-6cc7927810d1`, attempt `7c76315579b7`, admitted at
2026-11-17 02:33:42 UTC. Initial receipt retrieval hit a storage timeout;
no duplicate acquisition was requested. Measurement and settlement pending.

The continuation at `978cec8bc2` completed its baseline: 213,509 decisions,
four updates, 1,029 completed games, zero crashed/abandoned games;
1,790.827 collection decisions/s and 43.239 seconds/update. These are one
unpaired cell, not an overhead estimate. The second cell failed with
`run_identity_conflict`: the launcher tracking identity was shared across
subprocess cells although their configurations differed. RunPod's complete
inventory confirmed the pod absent after termination. A prospective repair
gives each cell a distinct tracking identity inside the one rental. A local
tracking-only admission creates both off and statistics declarations with
separate configs successfully; no training ran in that check.

Launcher cost for `recording_cost/measure-20261117-978cec8`:
**$0.05451162035, estimated**, rental ended. The repaired driver is frozen at
`a5afe273f3cd23c50dbf83a0a68e4e0033134303`. Launch `a01162215daa`, requested
run `recording_cost/measure-20261117-scoped`, was refused with HTTP 400;
complete provider inventory showed no `etude-c3b57f356fd2` pod, termination
reconciled the request, and launch status is failed with no recorded run.
A fresh request `recording_cost/measure-20261117-scoped2` uses the same source
and caps after availability again reported A6000 stock. No concurrent rental.

Launch `a02bdee53a41` (`recording_cost/measure-20261117-scoped2`) also received
HTTP 400 before a worker started. Complete inventory confirmed no
`etude-1e8800a0beab` pod; termination reconciled
`mtg338-a6000-1e8800a0beab`. At 2026-11-17 02:59:06 UTC,
`mtg338-l4` reported no stock. No further create was attempted. Both scoped
requests started no worker and produced no timings. The admitted run's
receipt and narrowed local logs are under `.runs/recording-cost-evidence/`
and `.runs/recording-cost-live.log`; launcher-retained artifacts are the durable
source. Comparison is blocked on rental capacity, with the repaired driver
frozen for the next attempt.

## Revision comparison, directed 2026-11-17

Jack Heart: "ok well we need to get a baseline v edited for thar to mean anythitn".
The primary comparison is now the last main revision merged into this checkout,
`5af8490a1c9bd5e9e7bda1d789e04a6f6de259fb`, versus the prospectively frozen
edited revision. The earlier "off" cell is explicitly an edited-tree option
baseline, not a before-change baseline, and cannot estimate the change's cost.

One rental builds both native extensions in release with the same pinned
Rust toolchain, interpreter and shared locked dependency environment. The
baseline is a digest-checked git archive supplied as a launcher Input;
`SourceBundle` verifies it and binds its own native extension. Each subprocess
checks its training implementation came from the selected source directory.
Both revisions read one identical current-world v11 recipe JSON, seed 80161,
256 streams, 64x2 model and four updates. Setup/build time is excluded from
collection/update timing but included in rental cost. Five pairs alternate
baseline/edited and edited/baseline; the baseline uses its unchanged defaults,
and the edited tree uses its unchanged defaults (standard diagnostics,
recording/statistics off). Cells have distinct tracking identities.

The comparison refuses a speed claim if decision counts or final policy tensor
hashes differ. Tensor hashes exclude checkpoint metadata, including revision
and run identity. Each hash includes names, dtypes, shapes and tensor bytes.
Report median and full range for collection decisions/s and update time;
the option comparisons are secondary and run separately with `comparison=options`.
No new overhead conclusion exists until the paired CUDA receipts complete.

Preflight for the revision comparison: built the baseline's CPython 3.13
release wheel locally (4m29s build, not a training measurement), installed its
matching extension into the extracted baseline, verified its source manifest,
and imported the common harness against the baseline training implementation.
The staged archive's size/digest and alternating five-pair order passed checks;
Ruff and `git diff --check` passed. No local training timing is used.
