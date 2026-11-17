# Recording and statistics cost (Task 80db9a)

## Question

What CUDA collection and update cost does each recording, statistics and
diagnostics option add to Elemental training, and what should stay on or be
sampled? This is exploratory, with no playing-strength claim.

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

## Outcome and limits

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
