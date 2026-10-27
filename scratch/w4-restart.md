# w4 restart brief (Jack Heart decision, 2026-10-25 ~21:10 UTC)

## What happened
w3 (`e7d47d8b220f`) hit `TimeoutError: training resource wall deadline exceeded`
at update 3,264 (14:54 UTC, $6.31). `segment_job` sets only `regime.wall_seconds`;
nothing stops the stage gracefully, so no final raw/EMA/Adam or completed run
record was exported. The update target (3,500) came from w2's old-system rate
(103.55/h); on Etude's L4 w3 ran ~68/h. Surviving: tracked monitoring exports
every 50 updates (`weights: raw`), last at 3,250
(`jobs/e7d47d8b220f/checkpoints/learner_transitions-178800486.pt`, sha256
ea3cf99b…, monitor-00000014.pt), and the interrupted run.json at
`jobs/e7d47d8b220f/attempts/4c9ada27b8d0/training/run.json` whose
`monitoring_checkpoints[14]` holds the update-3,250 coordinates
(games, decisions, learner_transitions, optimizer_exposures/updates, seconds).

## Decision (Jack Heart)
w4 continues from w3's raw weights at update 3,250: EMA := those raw weights,
Adam fresh. Declared in the protocol as a learner-state discontinuity at 3,250
(an amendment before w4 allocation). Seed, recipe, cadence, absolute iteration
(damped-policy schedule follows it) unchanged.

## Work
1. Training library: a declared, explicit way for a TrainSelfPlay stage to
   continue from an interrupted parent's monitoring export (raw only): admission
   checks the raw bytes against the parent record's monitoring receipt at that
   update, takes the origin coordinates from that receipt, sets EMA = raw (EMA
   metadata at that iteration) and a fresh Adam, and records which roles were
   reset. Do not weaken the existing full raw/EMA/Adam path. Tests.
2. `segment_job` / TrainingJob: a segment must stop gracefully before its wall
   budget and export its full learner state + completed record (so the next
   segment can use the normal continuation). E.g. stage active-seconds bound
   with margin, or stop at the deadline and finalize. Test that a segment that
   runs out of time ends `completed`/paused with raw/EMA/Adam exported.
3. Protocol amendment (before allocation) + `elemental-183-w4.json` sized per
   the 05:14 UTC amendment with w3's measured rate (730 updates over its stage
   seconds) and receipts w1 $7.2252, w2 $7.2358, w3 $6.31 (use the Etude receipt),
   lost launches (`a04845f2548b`, `3828021cbc80`, any other never-trained).
4. Do NOT launch w4; stop at a committed, tested spec. The operator session
   launches it.
