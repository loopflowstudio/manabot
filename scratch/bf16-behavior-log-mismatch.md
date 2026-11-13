# MTG-345 — behavior log/probability mismatch that stopped BF16 at update 141

## Problem

BF16 run `2be568a65518` (w384l8, bucketed, seed 32401, L4) trained 140 updates,
then update 141 was rejected with `behavior_log_probability_mismatch`. While
that can happen BF16 cannot be a default and its 1.7–2.7× learner speedup at
width ≥ 256 stays unused. Outcome wanted: the cause, with numbers; a fix at the
cause; a cheap proof; a statement of whether the USD ~30 learning A/B is worth
rerunning. No change to the precision default or the `auto` resolver.

## Cause (found 2026-11-13 from retained evidence, on CPU)

The Task brief describes the check as collection's log-probability against the
learner's recomputation. It is not. The failing line
(`manabot/backends/torch/rl.py::check_behavior`, called by
`PackedActions.check` before every update) is

    isclose(logs.exp(), probabilities, atol=0, rtol=1e-5)

Both operands are collection evidence. `Collection._sample_learning_rows`
computes float32 `logs`, then `weights = logs.exp()` **on the device**, and
stores both. The check recomputes `logs.exp()` **on the host** and demands the
two `exp` results agree to 1e-5 relative with no absolute tolerance. The
learner never reads the stored probabilities (`updates/damped_policy.py` and
`ppo.py` read `rows.log_probabilities` only); their other readers are
`updates/health.py::batch_health` (two metrics) and incident artifacts.

The incident (`etude fetch 2be568a65518
training/self-play-numerical-incident/{incident.json,tensors.pt}`; digests
match `run.json`) holds the failing 32-row check chunk and a 2,048-entry
slice of the rejected batch's packed distributions:

- One entry failed, row 25589: legal log **−94.75248**, a float32 subnormal
  probability (subnormals start below log −87.34). Device `exp` stored
  7.070952e-42 = **5046** subnormal spacings; host `exp` gives 7.072353e-42 =
  **5047**. The exact value is 5046.5007 spacings, so the two results are the
  two neighbouring floats. One spacing at that size is 1.98e-4 relative, 20×
  the tolerance.
- Over the slice's 2,048 legal entries, host and device `exp` differ on 591:
  −2 ulp ×2, −1 ×421, +1 ×168. In the normal range that is ≤ 1.5e-7 relative
  and always passes. Below p ≈ 1.4e-40 (log < −91.8) one spacing exceeds 1e-5,
  so **any** one-spacing disagreement there rejects the update.
- That row's legal logs are max−{0, 6, 15, 93, 94.75}: BF16-quantized logits.
  The sampled action had log −0.0025; the failing entry was never sampled.

So the rejection is a float32 representability limit of a device-vs-host
comparison, not BF16 arithmetic in the log-softmax, gather, EMA or snapshot:
at the failing revision (`783ccdba`, `collection.py:1424`) logits were already
cast to float32 before `policy_logs`, as they are today. The brief's expected
fix (float32 head inside autocast) is already in place and changes nothing.

What BF16 contributed is reach. Per-update `numerical/legal_probability_min`
and `legal_logit_gap_max` of the four A/B runs (read from `r2://manabot/jobs`,
experiment `precision_ladder`):

| Run | Updates | Max legal-logit gap | Updates with a subnormal legal p | …with p < 1.4e-40 (at risk) | …with p == 0 |
|---|---:|---:|---:|---:|---:|
| float32 32401 `07f84834e617` | 188 | 90.6 | 9 (first 161) | 0 | 0 |
| float32 32402 `aa9518f6fc4b` | 170 | 72.6 | 0 | 0 | 0 |
| bf16 32401 `2be568a65518` | 140 | 124.1 | 40 (first 95) | 33 | 12 |
| bf16 32402 `6aa5ba4d5fac` | 290 | 103.5 | 8 (first 154) | 2 | 0 |

Float32 came within a factor 3 of the at-risk range (min p 4.37e-40) and
would be rejected the same way one logit further; the bug is
precision-independent. Whether BF16's larger gaps are themselves a learning
difference is the A/B's question, with two seeds per arm, not this Task's.

Local probe, all 2,188,903 float32 values in [−104, −87.3]: torch-CPU `exp`
differs from the correctly rounded value on 45,185, never by more than one
spacing; the old rule rejects 385 of them. MPS `exp` returns values up to
8.4M spacings away on 2,182,263 (it flushes subnormals), and the old rule
rejects 2,180,453. No fixed ulp tolerance is right for every device.

## Chosen approach: logs are the only stored behavior distribution

Delete the second copy. `PackedActions` keeps `log_probabilities`; nothing
stores device `exp` output, so no cross-device `exp` comparison exists.

- `manabot/training/rollout/actions.py`: `PackedActions` and `ActionRows` lose
  `probabilities`; `pack(observation, log_probabilities)`; `unpack`,
  `concatenate`, `row`, `nbytes` follow. Padding-mass admission stays on logs
  (`illegal_log_mass`).
- `manabot/backends/torch/rl.py::check_behavior(logs, actions, selected_logs,
  valid)`: keeps shape, finite legal logs, −inf padding, log normalization
  (atol 1e-5, unchanged), action range and legality, selected-log identity
  (unchanged tolerances). Drops `check_probabilities` on stored weights,
  `behavior_log_probability_mismatch` and `zero_sampled_probability` (a sampled
  action is legal with a finite log). `check_probabilities` stays for the
  reference distribution. The section comment above `check_behavior` follows.
- `collection.py::_sample_learning_rows`: sample from `logs.exp()` as now; stop
  copying `weights` to the host. `RolloutBatch.observations` drops the reshape.
- `updates/health.py::batch_health`: `legal_probability_min` and
  `legal_underflow_count` come from host `np.exp(logs)` (float32), same
  meaning: the legal probability as float32 represents it, zeros counted.
  `docs/training.md` "Sampling contract" wording follows ("zero sampling weight" →
  rounds to zero in float32).
- `training/state/incidents.py`: no `packed_actions/probabilities`.
- No tolerance is widened; float32 and BF16 run the same checks.
- Half of the per-legal-action float storage of a rollout goes away (9.1M
  transitions in the failed run).

Forbidden: a derived `probabilities` property kept for old callers, a
per-precision tolerance, an `atol` on the old comparison, or computing the
stored copy on the host "to match" (torch-CPU, NumPy and vectorized/scalar
paths already disagree with each other: 45,185 of 2.19M above).

### Alternative not taken

Keep both copies and accept a difference of k subnormal spacings. Smaller
diff, but k is a device property (CUDA showed 2 ulp in 2,048 normal-range
samples; MPS flushes to zero), it is the tolerance widening the brief rules
out, and it keeps two owners of one fact.

## Tests (CPU, in CI)

- `tests/training/test_numerical_health.py`: replace
  `test_behavior_mismatch_rejected_without_epsilon`'s first case with the
  incident row — legal logs max−{0, 6, 15, 93, 94.75}, action 1 — admitted by
  `PackedActions.check`; add gaps 87.5, 92, 103.9, 104, 124 admitted, and
  `batch_health` reporting the subnormal minimum and the zero count. Keep the
  `selected_log_mismatch`, illegal-action and normalization rejections.
- Update the callers of `pack` / `rows.probabilities`:
  `test_packed_action_rollouts.py`, `test_collection.py`,
  `test_two_seat_learning.py`, `test_action_cutoff.py`, `test_regimes.py`,
  `test_rl_objectives.py`, `tests/model/test_inference.py`,
  `experiments/collection_tail_graphs.py` (wraps `pack`).

Gate (later step; add existing discovery extras): `uv run --extra dev --extra artifacts --extra reports pytest tests/training/test_numerical_health.py tests/training/test_packed_action_rollouts.py tests/training/test_collection.py tests/training/test_two_seat_learning.py tests/training/test_action_cutoff.py tests/training/test_regimes.py tests/training/test_rl_objectives.py tests/model/test_inference.py tests/training/test_credit_assignment.py tests/training/test_precision.py tests/eval/test_experiment_files.py -q`
Required result: all pass; `uv run --extra dev ruff check manabot tests experiments` clean.

## Proof run (exploratory, USD 5 cap)

`experiments/bf16_behavior_logs.py` + `.md`. One L4, one Job, two phases:

1. **Probe** (minutes, writes `probe.json`): (a) device `exp` against host
   `exp` over every float32 in [−104, −87.3] — spacing-gap histogram and the
   count the old rule would reject; (b) under float32 and BF16, for
   bucketed-only and bucketed + live tokens, collection's forward and the
   learner's forward on the same weights and rows at two bucket sizes — max
   and quantiles of |Δ log p| of the chosen action. (b) is the quantity the
   brief asked about; it is reported, not gated, since no check compares it.
2. **Train**: `precision_ladder.arm("bf16")` unchanged (w384l8, bucketed, no
   live tokens, horizon 300), seed 32401, stop at the window (about 190 updates).
   The failed run paced 126 s/update, so 200 updates is 7.0 h; at an L4
   ceiling of USD 0.60/h that is USD 4.35 with setup, at 0.70/h the USD 5 cap
   ends near update 197. Launch with the job's own `max_usd=5`. Actual L4 offers are USD 0.69/h;
   Etude adds 900 seconds startup to the job window, so the implemented window
   is 6.99 hours, less setup/probe and finalization inside it. The 300-update
   schedule is unchanged.

Done when the run passes update 141 with `rejected_updates: 0` and reports
how many updates carried a legal p < 1.4e-40 and were admitted (33 of 140 in
the failed run). If it never reaches that range, say so: the CPU regression
test on the incident row is then the proof and the run shows only no
regression. Terminate the machine after reading results.

The report gives the table above as "before", the probe and the run's
`legal_probability_min` / gap / underflow series as "after", cost once, and
whether the evidence supports rerunning the A/B: the numerical gate's only
recorded BF16 failure has a precision-independent cause, so the A/B's
strength and trajectory questions are again open; the rerun needs a person's
go-ahead naming the amount. The cause sentence already exists in
`docs/training.md` "Reduced precision"; slice 3 links the report from it; `experiments/precision_ladder.md` is frozen evidence
and is not edited.

## Docs (written ahead of the code, 2026-11-13)

Intended behavior is already written in `docs/training.md` (Sampling contract,
the two Metrics rows, and the cause sentence in the precision section) and
`docs/learning.md` (row definition, "1. Play"). Implementation matches these;
slice 3 only adds the link to `experiments/bf16_behavior_logs.md` beside the
cause sentence. `docs/training-loop.md` needs no change.

## Slices

1. **This slice — implemented**: storage and check change with regressions.
   `collection_tail_graphs.py` already forwards `pack` arguments unchanged.
2. **Implemented**: experiment file, probe + real worker smoke on CPU (`uv run python -m
   experiments.bf16_behavior_logs smoke`), `tests/eval/test_experiment_files.py`.
3. **Running**: report, docs link and CPU evidence publication are done; the
   capped L4 proof is live (below). Publish its results and finish the report.

## Exclusions

Precision default and `auto` resolver; the full A/B; any change to the
selected-log or normalization tolerances; MLX (it has no behavior-evidence
check); explaining why BF16 policies reach larger logit gaps.

## Delete — do not maintain

Delete `PackedActions.probabilities`, `ActionRows.probabilities`, the device-to-host
weight copy, packed incident probability tensors, and the cross-device
`behavior_log_probability_mismatch` / `zero_sampled_probability` checks. Replace
their exclusive fixtures with log corruption and incident-row regressions.
Preserve log evidence, coordinates, ownership, all learner objectives, reference
distribution checks, incident capture and unchanged normalization/selected-log
tolerances. `test_credit_assignment.py` also consumes the deleted field.

## Implementation check (2026-11-13)

Focused CPU checks: numerical-health + packed-action files **47 passed**;
consumer run **92 passed**, then corrected masked-log/KL expectations and
installed existing discovery extras, with affected retry **35 passed**.
Final behavior/incident filter: **10 passed**. `ruff check` on changed Python
files and `git diff --check` pass. Gate still
owns the complete acceptance command above; CUDA proof has not run.


Compress (2026-11-13): `check_behavior` now calls `check_logs` for the log
half (one owner of shape, support, finiteness, padding and normalization) and
keeps only the action checks; a log failure's incident tensors are `logs` and
`valid` plus `minibatch_indexes`. `uv run --extra dev --extra artifacts
--extra reports pytest tests/training/test_numerical_health.py
tests/training/test_packed_action_rollouts.py -q`: **47 passed**.

Proof implementation (2026-11-13): experiment discovery/job construction and
CPU smoke (eight forward combinations, one real worker update, admitted
policy) pass; CPU receipts are in the artifact manifest. Job window 6.99 h
plus 900 s rental startup, job/cohort caps USD 5. Launch `ebd68228bef4` was
refused before recording a run ("job dollars do not cover the hours", fixed by
the window); two quotes then returned "provider reports no available
machines".

Compress 2 (2026-11-13): the proof worker called `bind_etude_source()` and
then `run_worker()`, which binds again and raises "source manifest already
exists" on any launched job (the smoke has no source bundle, so it passed).
The job command is now `probe && python -m manabot.experiment.job`: the probe
is its own process and the training worker is unmodified. The wrapper worker,
its duplicate deadline check and SIGTERM handler are gone.
`uv run python -m experiments.bf16_behavior_logs smoke`: completed, one
update, admitted policy; `tests/eval/test_experiment_files.py`: **4 passed**.

Launched 2026-11-13 17:07 UTC at `34b380a554`: run `b00f4500b78d`, cohort
`mtg345-bf16-proof`, machine `mtg324-l4-ab-e0f0acfca5a5`, deadline
2026-11-14 00:07 UTC. Probe exited 0 and training reached update 4 by 17:19
UTC. Open: read results past update 141 (rejections, admitted subnormal
updates, probe deltas), publish them, finish the report, and
`etude terminate mtg324-l4-ab-e0f0acfca5a5`. Control client:
`doppler run -p etude -c prd -- doppler run -p manabot -c prd
--preserve-env=RUNPOD_API_KEY -- uv run --with 'etude-sdk[s3,eval]==0.2.15'
etude ...` with `SSL_CERT_FILE` set to certifi's bundle. Gate still owns the
complete acceptance command.

## Added scope: 50-update learning pair (Task comments, 2026-11-13)

Jack Heart, comment `ae2552d3`: "after the USD 5 proof clears update 141, run a
matched BF16-vs-float32 learning comparison with the fix, capped at USD 10
total (training + any evaluation; separate from the proof's USD 5). ... Reuse
the machine already rented if it is free; terminate machines when done. Also
fold in the free evidence: MTG-324's four existing runs ... compare their
loss, KL, clip-fraction and value curves alongside the new pair. ... No
pre-registered verdict ...: float32 is the incumbent; report whether BF16 with
the fix learns the same per update and how much faster it is per wall-clock
hour, separate what the data supports from what it does not, then recommend
whether auto should pick BF16 on CUDA at w>=256. Do not change the auto default
in this PR unless the result clearly supports it; say so either way."

Jack Heart, comment `d3422d92`: "run each arm (BF16, float32, same seed) for
about 50 updates, not a time window; the USD 10 cap stays as a ceiling ...
Compare per-update curves over updates 1-50 (plus MTG-324's existing four runs
over the same range) and wall-clock per update; strength evaluation at update
50 only if it fits the cap."

Implemented: `experiments/precision_pair.py` (`runs`: both precision_ladder
arms, seed 32401, `stage.updates = 50` on the unchanged 300-step damping
horizon, 3 h window, USD 2.50 per run; `eval`: update-50 head-to-head, USD 1;
`python -m experiments.precision_pair FLOAT32_RUN,BF16_RUN` writes
`experiments/data/precision_pair/{updates.csv,curves.png}`). Per-update
metrics come from `etude.train.read.read(storage_from_uri("r2://manabot/jobs"),
experiment, run).points(metric, "updates")`; one read takes minutes.

Open: launch after the proof passes 141 (expected ~21:05 UTC; a background
watcher polls `etude status`), cohort cap USD 5; then curves, eval if it fits,
report `experiments/precision_pair.md`, publish data, terminate machines.
