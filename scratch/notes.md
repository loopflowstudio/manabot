# MTG-338 notes

- Parent landed: #501 merged to main 2026-11-13T04:40Z (ef28f3a58) with the
  torch backend; #503 closed. Branch synced onto origin/main 6ce618b8a.
- `scratch/speed/{bench,compare,mtg325_speed}.py` are MTG-325's L4 harness,
  restored from c958ae2f1^ (Etude 8a376219e430). `compare.txt` is its pass-4
  baseline: PPO 0.256 steps/s, "Backward and gradient clipping" 46.1 s of
  61.4 s minibatches; damped 9.6 steps/s. Paths inside the harness may name
  pre-#501 modules; check before launching.
