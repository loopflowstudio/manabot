# MTG-338 notes

- Parent landed: #501 merged to main 2026-11-13T04:40Z (ef28f3a58) with the
  torch backend; #503 closed. Branch synced onto origin/main 6ce618b8a.
- `scratch/speed/{bench,compare,mtg325_speed}.py` are MTG-325's L4 harness,
  restored from c958ae2f1^ (Etude 8a376219e430). `compare.txt` is its pass-4
  baseline: PPO 0.256 steps/s, "Backward and gradient clipping" 46.1 s of
  61.4 s minibatches; damped 9.6 steps/s. Paths inside the harness may name
  pre-#501 modules; check before launching.
- First measure completed: `experiments/ppo_cuda.md` and `scratch/measure.html`.
  Clean Etude `4a289dc30a82`, source `2202ad632cb5`: median 0.26437 optimizer
  steps/s, range 0.26366–0.26474, five repeats; 16 steps/update, zero rejected
  or skipped steps. This predates the CUDA lookup candidate.
- Profile `e9e5a6bd4649`: indexed definition-table gradient accumulation takes
  78.3% of recorded first-step GPU kernel duration. Fact-source and continuation
  definition joins dominate. Explore a repeated-ID reduction; preserve CPU
  bitwise behavior and bound CUDA summation changes before alternating arms.
- Baseline CUDA weights already differ across repeats by up to 2.486e-5 absolute;
  this is observed drift, not an equivalence tolerance. Production threads=1 and
  seed_random/cuDNN settings were missing from the historical harness and are
  now recorded. Do not compare the old 0.256 rate as an improvement claim.
- All four task rentals are terminated. Nine artifacts under
  `experiments/data/ppo_cuda/` are published and remain untracked.

- Paired measure could not run: candidate/harness freeze `02c202956`, requested
  run `4e50a037bd12`, admission `947aca1850c9` failed before any command or run
  record. Provider: “no available machines”; availability at 2026-11-13
  10:58:25 UTC: “none available”, “0 of 1 in use”, “not rentable now”.
- Measure restored candidate runtime/tests/launcher/harness to `734f2cd5c`.
  Full design/harness survives in `02c202956`; no candidate speed or CUDA
  equivalence result exists. No comparison work is queued or running.
- For the next rental, nest Doppler `manabot/prd` (R2) and `etude/prd`
  (RunPod) on the command. Either alone misses a required credential.
- Decision-pass read-only availability, 2026-11-13 11:02:39 UTC:
  `mtg338-l4` again returned “none available”, “0 of 1 in use”,
  “not rentable now” at $0.69/hour (ceiling $0.75). No launch or queue created.

- Implementation pass 2 recovered the eight candidate/harness paths exactly
  from `02c202956`. Focused lookup tests: 8 passed, 8 CUDA skips (27.83 s).
  No rental, equivalence run or performance measurement was started.

- Measurement pass 2, 2026-11-13: availability at 11:08:05 UTC said “available”,
  “0 of 1 in use”, “rentable now” at $0.69/hour. Requested run `4bacd2187ed1`
  (source `02c202956`, cohort `mtg338-paired-pass2`, cap $4) then failed admission
  `db4502e2006a` at 11:10:12 UTC: “provider reports no available machines”.
  Receipt: “job command not started”, “no run recorded”. No rental or queue;
  no paired data or CUDA correctness evidence. Restored the eight candidate/
  harness paths to `734f2cd5c` again. A rentable availability response alone
  does not guarantee admission. Capped launch commands require `--cohort NAME`.

- Implementation pass 3 restored the same eight files byte for byte from
  `02c202956`. Focused lookup tests: 8 passed, 8 CUDA skips, 2 deselected
  (5.98 s). Ruff, compilation and both generated shell commands passed.
  No availability read, launch or equivalence/performance run occurred.

- Measurement pass 3: all eight candidate paths matched `02c202956`. Availability
  at 2026-11-13 11:16:14 UTC said “available”, “0 of 1 in use”, “rentable now”.
  One launch (cohort `mtg338-paired-pass3`, cap $4) requested `3ca9e26b83a0`;
  admission `797c99576241` failed at 11:17:14.552355 UTC: “provider reports no
  available machines”. Receipt: “job command not started”, “no run recorded”.
  Cohort status found no run. No retry, rental or queue; all paired checks remain
  unrun. Restored all eight candidate/harness paths to `734f2cd5c` under the
  measure keep rule; source remains recoverable from `02c202956`.

- Measurement pass 4: eight candidate/harness files matched `02c202956` exactly.
  The single availability read at 2026-11-13 11:22:45 UTC returned “none
  available”, “0 of 1 in use”, “not rentable now” at $0.69/hour. No launch was
  attempted; no rental or queue created. Restored the eight paths to `734f2cd5c`
  under the keep rule. No candidate timing or numerical audit result exists.

- Measurement pass 5: eight candidate/harness files matched `02c202956` exactly.
  The single availability read at 2026-11-13 11:26:32 UTC returned “none
  available”, “0 of 1 in use”, “not rentable now” at $0.69/hour. No launch,
  rental or queue. Restored the eight paths to `734f2cd5c` under the keep rule.
  CPU/CUDA audits and paired measurements remain unrun.

- Measurement pass 6: all eight candidate/harness paths matched `02c202956`.
  Single L4 availability read at 2026-11-13 11:29:45 UTC: “none available”,
  “0 of 1 in use”, “not rentable now” at $0.69/hour. No launch, rental or queue.
  Restored the eight paths to `734f2cd5c` under the keep rule. CPU/CUDA audits
  and paired timing remain unrun; no speedup or behavior equivalence established.

- Measurement pass 7: all eight candidate/harness paths matched `02c202956`.
  Single availability read at 2026-11-13 11:33:45 UTC: “none available”,
  “0 of 1 in use”, “not rentable now” at $0.69/hour. No launch, rental or queue.
  Restored the eight paths to `734f2cd5c`; numerical audits and timing remain
  unrun. No speedup or CUDA equivalence established.

- Implementation pass 8 recovered the eight frozen paths exactly from
  `02c202956`. Focused lookup checks: 8 passed, 8 CUDA skips, 2 deselected
  (1.01 s). Ruff on runtime/tests/launcher/speed harness, compilation of all
  eight files and both generated shell syntax checks passed. No capacity read,
  launch or measurement occurred; numerical and performance acceptance remain
  pending on L4. Existing report and measurement evidence were preserved.

- Measurement pass 8: all eight candidate/harness paths matched `02c202956`.
  Single availability read at 2026-11-13 11:36:42 UTC: “none available”,
  “0 of 1 in use”, “not rentable now” at $0.69/hour. No launch, rental or queue.
  Restored the eight paths to `734f2cd5c`; numerical audits and paired timing
  remain unrun. No speedup or CUDA equivalence established.

- Implementation pass 9 recovered all eight frozen paths exactly from
  `02c202956`. Focused lookup checks: 8 passed, 8 CUDA skips, 2 deselected
  (1.45 s). Ruff on runtime/tests/launcher/speed harness, compilation of all
  eight files and both generated shell syntax checks passed. No capacity read,
  launch or measurement occurred. Whole-update equivalence and paired L4 timing
  remain pending; no performance or CUDA correctness claim.

- Measurement pass 9: all eight candidate/harness paths matched `02c202956`.
  Single availability read at 2026-11-13 11:39:54 UTC: “none available”,
  “0 of 1 in use”, “not rentable now” at $0.69/hour. No launch, rental or queue.
  Restored the eight paths to `734f2cd5c`; numerical audits and paired timing
  remain unrun. No speedup or CUDA equivalence established.

- Implementation pass 10 recovered all eight frozen paths exactly from
  `02c202956`. Focused lookup checks: 8 passed, 8 CUDA skips, 2 deselected
  (2.13 s). Runtime/tests/launcher/speed-harness Ruff checks and compilation
  passed. Both generated commands passed `bash -n` after correcting the local
  validation accessor from `Run.command` to `Run.job.command`; no source change
  was needed. No capacity read or launch occurred. CPU whole-update equivalence,
  CUDA audits and paired timing remain pending.

- Measurement pass 10: all eight candidate/harness paths matched `02c202956`.
  Single availability read at 2026-11-13 11:43:03 UTC: “none available”,
  “0 of 1 in use”, “not rentable now” at $0.69/hour. No launch, rental or queue.
  Restored the eight paths to `734f2cd5c`; numerical audits and paired timing
  remain unrun. No speedup or CUDA equivalence established.

- Implementation pass 11 recovered all eight frozen paths exactly from
  `02c202956`. Focused lookup checks: 8 passed, 8 CUDA skips, 2 deselected
  (2.44 s). Runtime/tests/launcher/speed-harness Ruff, compilation and both
  generated shell syntax checks passed. No capacity read or launch occurred.
  Whole-update CPU equivalence, CUDA audits and paired timing remain pending.

## Stopped 2026-11-13 ~11:46Z (lf task interrupt + hold)

Optimize Graph 62419322 looped through measure pass 11: restore the candidate
from 02c202956, find no L4, revert, and optimize-plan chose "iterate" again.
A token audit of that loop: about 16.35M tokens, no new measurement. Do not
relaunch until L4 capacity is confirmed first; missing capacity is a stop,
never a reason to iterate. The worktree holds pass 11's uncommitted
re-restored candidate (semantic_cards.py F.embedding + tests).

## A6000 stand-in (2026-11-13 ~15:30Z, approved in conversation)

No L4/A40 stock on RunPod. Jack: "go for it" on a same-machine paired run on a
similar card. Added on-demand `mtg338-a6000` (RTX A6000, GA102 like the A40,
6 vCPU/48 GB, ceiling $0.75, same image as mtg338-l4). Direct launch
3bda3c3547c1 refused (POST pods 400, no pod). Queued run fb0212617dc7,
cohort `mtg338-a6000-paired`, revision 02c202956 `--paired`, cohort cap $4;
`etude wait` runs in the background from this session. Relative speedup only;
absolute steps/s is not comparable to the L4 0.264 baseline. After success,
`etude terminate` the machine and check `runpodctl pod list`.
- 15:58Z: A6000 run never rented; a ghost machine record from the refused
  create held the template's one slot ("1 of 1 in use", no RunPod pod).
  Cancelled fb0212617dc7, terminated the ghost. L4 read "rentable now"; direct
  launch 9f7e2484b917 refused (no machines). Queued L4 run 0c3252befa2d,
  cohort `mtg338-l4-paired`, cap $4; background `etude wait` from this session.
