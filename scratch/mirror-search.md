# Mirror search continuation

## Task and workflow

This study is now MTG-269 (`task_ed311b47c8704f18965f6a0ae8cbb370`), with
this existing checkout and branch adopted. The current conversation is its
primary Session, `session_c73fe3a8e9e94b56b1ffe915b575a9c0`.
The current conversation requests: “lajnch the task, using the experiment
workflow” and “make sure thati t knows about existing work”.

Continue from the existing study, not a blank design. Reconcile current remote
completion and MTG-268 status before proposing execution: the snapshots below
are historical. The experiment graph and its domain skills were copied from
the main checkout into this frozen checkout; both graphs compile and the
repository checker's experiment boundary checks pass. No engine code was synced.
Do not let a generic software sync step merge current-world main into w13.
The workflow launch stops at its first interactive review; it supplies no new
rental, merge or auto-ship authority.

## Recovered study

Recovered 2026-11-08 from Claude session
`/Users/jack/.claude/projects/-Users-jack-src-manabot-search/5537a213-b123-46cf-a71e-dc3ac90c37f2.jsonl`.
The conversation lived in this checkout; its study code and evidence live in
`/Users/jack/src/manabot.w13-search`, branch `jack-heart/w13-search-scaling`.
Do not move the study onto the current engine: it uses MTG-183's frozen w13
checkpoint. The log records “We can use w13 here” and “you can use the mini if
it helps”. The 2026-11-08 continuation request is “and continue the work on
magic mirror search”.

At 11:43 UTC the mini had seven existing mirror workers running. A subsequent
local snapshot contains 4,899 unique games of 8,000 planned, zero invalid.
Remote evidence: `mini-heart.local:~/search-scaling/mirror-v1/`.
The noise study at `~/search-scaling/noise-v1/` finished all 300 positions.
No new games or rentals were launched during recovery.

The refreshed report is
`/Users/jack/src/manabot.w13-search/experiments/data/search-mirror-v1/report.html`.
Depth 20, beta 0.03, four shuffles leads this interim snapshot. Paired gains
over sampled policy: +10.13 points against heuristic (306 deals), +14.59
against policy (305 deals). Against PUCT-1024 oracle: +4.25 and +8.52 points.
These compare performance against common opponents, not mirror versus PUCT
head-to-head. Oracle hands, multiple comparisons and interim sampling limit
the interpretation; host-mixed PUCT timings are not a controlled speed comparison.

MTG-268 (`task_fae941fd7346493998abc55737dd2564`) already owns rollout batching
in `/Users/jack/src/manabot.mtg-268-batch-local-update`. At recovery its code
Graph was running implement, loop 1 pass 3. Leave that live work running.
The proposed next game settings are depth 20, beta 0.01, 8 and 16 shuffles,
after batching is validated and adapted to the frozen study. The prior assistant
proposed these; they have not run. Continue checking the existing study and
batching work before starting additional execution. No merge was requested here.

Report rebuild (analysis only; no games on the laptop):

```sh
cd /Users/jack/src/manabot.w13-search
uv run --project /Users/jack/src/manabot.search --no-sync python -m experiments.search_mirror_report --mirror experiments/data/search-mirror-v1/games --baseline /Users/jack/.manabot/search-scaling/hosts --baseline-summary experiments/data/search-scaling-v1/summary.json --noise experiments/data/search-mirror-v1/noise --out experiments/data/search-mirror-v1
```

Results belong in the artifact store; publish changed evidence before committing
the manifest. Never commit raw `experiments/data/` files.

Recovery publication: the 4,899-game snapshot, HTML and summary were published
to the existing artifact-store paths (16 manifest entries changed). Light and
dark report renders inspected; untracked-data check passed (1 test).
