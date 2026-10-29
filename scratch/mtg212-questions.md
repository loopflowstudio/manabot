# MTG-212 unresolved inputs — 2026-10-28

Inherited notes remain in `scratch/questions.md`, now matching main.
Current evidence and implementation: [mtg212.md](mtg212.md).

- **Planning read gap:** the supplied `.lf/wave/training/GOAL.md` and
  `MEMORY.md` paths are absent. The current shared MTG-212 Task read succeeds;
  implementation follows its directive and the retained design.

- **Execution inputs resolved** by Jack Heart's 2026-10-28 direction below;
  the w13 package, billing bound and probe are recorded in [mtg212.md](mtg212.md).
- **Assumption (reversible):** reported charges settle ×1.06 for OpenRouter's
  credit-purchase fee; the $0.02 per-request reservation is an engineering
  choice from the dated listing. A missing provider cost stops admission.
- **Assumption:** the w13 package is a local branch/worktree, not pushed; its
  patch against `d7b98a487` is published with the campaign evidence.
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
- **Connect-only resend (2026-10-28, reversible engineering choice):** campaign 1
  died on one `ConnectError` in its pilot. Campaign 2 resends only unsent
  requests (≤3) and replays the same frozen schedule from game one as a new
  cohort; campaign 1's 11 attempts stay separate and unpooled. Review may prefer
  fresh deal seeds for cohort 2, or a different treatment of transport failures.
- **Campaign 3 is the reported cohort (2026-10-29, reversible engineering
  choice):** it stopped at 127 attempts (124 valid) after three consecutive
  `RemoteProtocolError` games at 00:21–00:23Z, while another Task's Jev
  campaign shared the key. No cohort 4 was launched: it would need a fourth
  protocol revision and replay the schedule from game one. About $7.6 of the
  $10 remains; Jack Heart may authorize a further cohort (e.g. the unattempted
  schedule suffix as new, unpooled evidence).
