# Protocol017 placement and operational evidence

This package records admissions and completed controls. It does not claim the
live 18,304-game Mini panel or subsequent scientific/visual review has finished.
The scientific report and final recommendation must establish those separately.

- The laptop attempt was explicitly cancelled by Jack Heart. Its 352 replayed
  games are retained and excluded from the replacement Mini scores. Its exit124
  is not evidence that the original scientific deadline expired. Original logs,
  admission, status and manifest are copied; partial game traces remain in the
  original retained attempt and are not included in this small package.
- The first Mini control used Background scheduling and recorded watchdog
  failures. It was stopped and retained. The final controller log differs from
  the original manifest because a traceback was appended after that manifest;
  `receipts/failed-control-retention.json` verifies all 27 final files, including
  the unchanged original manifest. No historical receipt was rewritten.
- A separately admitted Standard-scheduling control completed 32 games, with
  all recorded decisions, identities and replays matching the earlier control.
  Its parallel command interval was 165.435403 seconds.
  Complete failed and successful control trees are included separately.
- PR311/commit `cbf2ebde4fd58dbccdecf4192d610fc98b1c7bef` serializes supervisor event output. The exact
  patch and receipt are included. Scientific source remains
  `3c3681d2b26a7fc8167928fa63fe24a6cd3d058a`; no model, game or learning change was made.
- The Mini plan scales operational command/work limits from matched controls
  while preserving the full panel and 120-second game watchdog. Four commands
  run concurrently; command limit 15,000s,
  work limit 85,800s, retention240s. These are
  bounds, not an ETA. Launchd uses Standard scheduling and no automatic restart.
- The separate one-shot audit owner waits for successful full-panel completion
  before running the frozen analysis and renderer. Source and admission hashes
  are retained here. Its launch is not proof that the later audit succeeded.

The Mini move used direct SSH file transfer and no new GPU rental or S3 artifact
download. The renderer's historical budget paragraph describes rental accounting
and an ancillary reserve. That reserve is not measured S3 attribution or proof
that total all-inclusive campaign spending is within $50; shared storage/transfer
charges remain partly unattributed. Do not present quote/SDK costs as invoices.

Treat the frozen protocol's original laptop-sized limits and placement as
historical, with this explicit Mini plan/admission as the operational amendment.
Scientific endpoints, seeds, deal panel and frozen selection rule are unchanged.
No failed/cancelled control or partial laptop score is combined with Mini scores.
