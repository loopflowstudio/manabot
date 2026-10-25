# MTG-181 live plan

Frozen protocol: `experiments/protocols/elemental-181.md` (commit 9e47a7d6, amended
0e9c4427 for the 18101 retry). It stays the authority; this note tracks state.

## Exists (latest check at end)

- Three jobs running on L4 (direct job_status); no failures in the ledger.
  ≈115–145 updates/hour → ≈600–700 updates by the 5 h deadline (≈22:30 UTC),
  inside the 300–700 prediction. Cost includes the $0.05 failed 18101 probe.
- 18103 at update 3: 106,166 learner = environment decisions, 515 games, 0
  crashed; ≈35k learner decisions/update (protocol estimate ~56k).
- Follower (launchd, pid 23706, until allocation + 48 h) is playing the first
  grid (18103 update 0) at ≈74 min per grid versus the 45–55 min calibration.
  Its ledger refreshes job state only between grids (last 17:31), so ledger job
  state lags between grids. The first grid had 112/200 group directories at
  39 min, 151 at 56 min and 172 at 63 min, consistent with ≈74 min per grid. Expected
  cadence 0/200/400/600 + endpoint ≈ 5 grids/seed ≈ 15 grids ≈ 18.5 h, inside
  the 48 h window; protocol predicted 3–6 evaluations per seed.
- Report script committed (7a75f9cc7, 53afcf4b1): pydantic `Report`/`SeedRecord`/
  `PointRecord` written as `summary.json`; throughput uses active collection +
  learning seconds; unfinished evaluations report null duration.
- `elemental-181-identity.json` pins by SHA-256 `elemental_grid.py`,
  `elemental_two_seat.py`, `elemental_two_seat_follow.py`, `manabot/arena/match.py`
  (and the job/regime JSON). Do not edit them, even to simplify them, while
  the trial is live. Only the report script and the docs can change.
- Report destination: `experiments/data/mtg181/` (report.md + summary.json +
  follower evidence), published via `manabot.data publish`; the protocol gets a
  short Results pointer. MTG-159 comparison numbers: scale-ladder.md (9,250
  updates, 151.6M learner decisions, $9.88 pod, heuristic 0.11 → 0.69).

## Remains

- Let training reach terminal state with confirmed cleanup; the follower
  evaluates every cadence export and endpoint, then renders the Etude reports.
- Run the report script over `.runs/mtg181/follow`; write the results report (per-seed curves, seed spread,
  learner/environment decisions, collection/learning time, throughput, errors,
  inclusive cost, MTG-159 within-world/per-dollar comparison, next question).
- Publish the evidence via `manabot.data publish`; commit the manifest and
  report; verify with `tests/infra/test_experiment_data_untracked.py`.
- A failed seed is reported as failed, not replaced (protocol).

## Proposed Wave memory (not written; standalone realign)

- Renaming `Agent` → `ActorCritic` also rewrote the `User-Agent` HTTP header,
  and RunPod's REST API returned 403. Repository-wide renames need a wire-string check.
- A guardian probe create that returns HTTP 500 is an ambiguous claim. Counting
  it conservatively and retrying once inside the cap kept the seed.
- The partial local 10×10 calibration (27/200 groups, eight workers, tests
  sharing the machine) projected 45–55 min. The production grid with ten workers
  is running at about 74 min. The cause has not been diagnosed.

- Training terminal (22:23 UTC): all three `paused` with cleanup confirmed. 18101-r2 u616, 18103 u568 at the deadline, 0 crashed games. **18102 stopped at u474 on the kill criterion** `game_error_rate_exceeded` (256 crashed of 127,358 attempted, window rate 1.0); Its `run/game-attempts.jsonl` (127,358 attempts) shows all 256 non-completed attempts are `outcome=deadline`, `TimeoutError: collector exceeded training deadline`, one per stream at u474: the allocation deadline cut the in-flight games and the game-error window counted them. No engine or agent error. It is a deadline stop that the `stop_reason` mislabels, not a game-bug failure. Report it as such (follow-up: deadline abandonment should not enter the game-error window). 18102 is slower (u258 vs 313–321 at 20:02), so it reached the deadline in mid-update. All three records carry a sticky `snapshot upload failed (ValueError)` from some intermediate snapshot; final manifests are complete.
- Cleanup confirmation needs `RUNPOD_API_KEY` (Doppler `etude/prd`) only to write the deletion receipt; the launchd follower has none, so `doppler run -p etude -c prd -- … status.py` wrote the receipts at 22:25. After that, keyless status reads them.

Latest check (2026-10-24 15:30 UTC): follower pid 23706 alive (22 h); 10/13 grids complete, all 5,000 valid each. Means (n=3): u0 0.206, u200 0.789 (sd 0.002), u400 0.838 (sd 0.007; 18101 0.829, 18102 0.842, 18103 0.842); 18102 endpoint u474 0.844 (gain +0.638). 18103 u568 at 77/200 groups; then 18101 u600 and u616. Overnight grids slowed to ≈3–4 h (host on battery, load ≈30); daytime ≈75 min, so done ≈ 19:00 UTC if the host stays awake. Report script runs cleanly on the partial evidence. Inclusive ≈ $9.12 of $10, settled.

Direction (Jack Heart, 2026-10-24): stack this branch on #362 (`jack-heart/remote-on-etude`) so the evidence publishes to R2. Rebase only after the follower exits (its grid workers import from this checkout). #362 deletes `manabot/remote/job_client.py`, which the report script uses for settled job status; read that from the ledger (or #362's Etude client) instead.
- #362 merged 2026-10-24 19:22 UTC: rebase onto main, not a stack.
- Steer (Jack Heart via Etude storage-cost session, 2026-10-24): once the follower stops, copy the keepers from s3://etudefantasia/manabot/jobs/mtg181-* into R2 `manabot` (same relative path, no `manabot/` prefix; checkpoints, final run/run.json, latest manifest, control files, Etude metrics; verify size/sha256; skip superseded snapshots, training.sqlite, dashboards, source archives), then delete all versions and delete markers of those folders. R2 creds: Doppler manabot/prd. Report copied items and deleted bytes.
- Main shipped a stronger heuristic (639ab7770). The frozen anchor stays v1; offered Jack Heart a separate endpoint addendum against the new heuristic.
- 2026-10-25 ~00:30 UTC: follower exited; summary.json + report.md drafted in experiments/data/mtg181 (34 MB evidence). Synced onto main (07a091bca), conflicts resolved: took main's deletion of manabot/remote, kept our scratch/questions.md. Retired elemental_two_seat{,_follow,_report}.py (cite 739fb251d). S3→R2: 79 keepers verified by sha256 (257 MB); S3 mtg181-* deleted, 20,675 versions, 25.33 GB, 0 remaining. New-heuristic endpoint grids running (experiments/elemental_181_new_heuristic.py → .runs/mtg181/new-heuristic); then fill NEW_HEURISTIC_TABLE, publish, commit manifest, PR.
- 2026-10-25: Jack Heart asked for a UR-vs-GW-only seed compared with MTG-181 → filed MTG-186 ($10, one seed, matched to 181, Etude) and started its pursue Graph. MTG-186 will use experiments/data/mtg181/new-heuristic, so publish it.
- 2026-10-25 04:30 UTC: Jack Heart: stop using the laptop for evals → local addendum and MTG-186's local follower stopped; MTG-186 steered to Etude evals. Addendum resumed on the Mac mini (ssh mini-heart, ~/src/manabot.mtg181-eval at dde769954, pinned .so 0a4af210…, log .runs/mtg181/new-heuristic.log, output .runs/mtg181/new-heuristic-out); 18101 copied (re-summarized to 0.584), 18102 87/200 groups copied. Fetch results back into experiments/data/mtg181/new-heuristic when done.
