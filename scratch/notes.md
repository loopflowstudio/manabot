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
