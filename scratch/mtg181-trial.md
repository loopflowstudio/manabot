# MTG-181 live plan

Frozen protocol: `experiments/protocols/elemental-181.md` (commit 9e47a7d6, amended
0e9c4427 for the 18101 retry). It stays the authority; this note tracks state.

## Exists (observed 2026-10-23 18:00 UTC)

- Three jobs running on L4: `-18101-r2` (update 49), `-18102` (50), `-18103` (63)
  at 17:57. ≈145 updates/hour → ≈650–700 updates by the 5 h deadline (≈22:30
  UTC), inside the 300–700 prediction. Conservative cost ≈ $0.18 including the
  $0.05 failed 18101 probe.
- 18103 at update 3: 106,166 learner = environment decisions, 515 games, 0
  crashed; ≈35k learner decisions/update (protocol estimate ~56k).
- Follower (launchd, pid 23706, until allocation + 48 h) is playing the first
  grid (18103 update 0) at ≈74 min per grid versus the 45–55 min calibration.
  Its ledger refreshes job state only between grids (last 17:31). At 18:10 the
  first grid had started 112/200 groups after 39 min, which is consistent with
  ≈70 min per grid. Expected
  cadence 0/200/400/600 + endpoint ≈ 5 grids/seed ≈ 15 grids ≈ 18.5 h, inside
  the 48 h window; protocol predicted 3–6 evaluations per seed.
- Report script committed (7a75f9cc7, 53afcf4b1): pydantic `Report`/`SeedRecord`/
  `PointRecord` written as `summary.json`; throughput uses active collection +
  learning seconds; unfinished evaluations report null duration.
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

Check (18:10 UTC): follower pid 23706 alive; first grid at 112/200 groups started; no failures recorded in the ledger.
Check (18:12 UTC): direct job_status — 18101-r2 running u82, 18102 running u78, 18103 running u92. 18101-r2 has left not-accepted (ledger is stale only). Long watch started (≈1.9 h window).
