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
