# MTG-212 unresolved inputs — 2026-10-28

Inherited notes remain in `scratch/questions.md`, now matching main.
Current evidence and implementation: [mtg212.md](mtg212.md).

- **Planning read gap:** the supplied `.lf/wave/training/GOAL.md` and
  `MEMORY.md` paths are absent. The current shared MTG-212 Task read succeeds;
  implementation follows its directive and the retained design.

- **Required before execution:** dated combined ratings and resolved best
  checkpoint; matching historical source/native/input package; justified maximum
  inclusive request charge and overhead; credentials. All four accessible
  `manabot` Doppler configurations lack `OPENROUTER_API_KEY`. No paid request or
  rental occurred. Missing ratings do not establish an ambiguous ranking.
- **Reversible engineering choices:** the driver defaults to an included 20-game
  pilot, a shuffled 1,000-game schedule with independent deal seeds, and
  descriptive Wilson win intervals. It stops at the first failed game and never
  automatically resumes or replaces attempts. These choices are explicit in the
  protocol, not attributed to Jack Heart; a treatment change cannot reset spend.
- **Context admission:** the provider owns its tokenizer. Local preflight checks
  full payload bytes and 255-choice cardinality; provider token-context rejection
  remains remote. The byte ceiling is not billing-bound evidence.
- **Wave memory edit:** the staged diff edits `.lf/wave/intelligence/MEMORY.md`,
  although this Task belongs to the training wave and implement does not grant wave
  writes. Retained as an earlier contribution; review should confirm or drop it.

## Jack Heart's direction (2026-10-28, task conversation)

- **Credentials:** `OPENROUTER_API_KEY` lives in Doppler `loopflow/dev_personal`
  (the only config holding it; `etude/*` and `manabot/*` lack it). Run with
  `doppler run -p loopflow -c dev_personal -- ...`; never copy or print the value.
- **Billing bound:** Jack Heart has no per-request cost requirement; only the
  $10 inclusive total binds. The per-request reservation is an engineering
  choice: price a full 262,144-token prompt at the quoted input rate plus a
  fee margin, and stop admission before the $10 total.
- **Checkpoint:** Jack Heart selected MTG-183's last checkpoint, u5950 (run
  `f3748bcabac8`, 1,071,166 bytes, sha256
  `4f2ad656ba036a33d527212a7ac2badf2c5b9db136916f730846b07eef2fbea4`, world w13).
  Do not wait for combined ratings; report it as "final", not "best". It needs
  the matching w13 historical execution package (no admission exception).
- **World:** Jack Heart (2026-10-28): run Jev on world w13, the checkpoint's own
  world. The campaign executes in the w13 historical package; main's w15 does not
  apply to this benchmark.
