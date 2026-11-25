---
requires: an experiment module at a committed revision
produces: completed Etude runs recorded in `experiments/reports/<name>.md` by commit, run ID and cost
action_style: procedural
---
Run the experiment through Etude and record each run in `experiments/reports/<name>.md`.

1. Launch from a committed revision and record the commit and Etude run ID in
   the report. Those two are the freeze; freeze, hash or admit nothing else
   by hand.
2. Keep spending within authority. At or under USD25 for the whole experiment
   (training, evaluation and retries), launch with `--max-usd`. Above USD25,
   stop before renting unless a person has already approved the amount; state
   the amount needed and wait. Don't ask again for an amount already
   approved.
3. Inspect existing runs and resume unfinished work in Etude. For checkpoint
   curves, run `etude eval` once checkpoints are ready—declaring `eval_jobs`
   alone queues no evaluation. Confirm the intended checkpoint and deal
   coverage before reporting.
4. A fix that keeps the question (capacity limit, crash fix, device, machine
   type, retry) is fixed, relaunched and noted as one dated line in the
   report—never a new protocol or version. A change to a finding's question
   or what it measures gets a new dated opening paragraph, and earlier runs
   become exploratory.
5. Record the launcher's cost once per run; never reconstruct bills, reserves
   or settlements. Confirm rentals ended.
6. Keep failed games (seeds and Commands), numerical health, and artifacts
   published through `manabot.data` before committing any manifest. File
   reproduced engine defects as bug Tasks. No artificial load, no fallback
   policies, and no spending to chase a favorable result.
