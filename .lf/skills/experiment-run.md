---
requires: the experiment module at a committed revision
produces: completed runs, each recorded in the report by commit, run ID and cost
---
Run the experiment through Etude and the existing manabot owners.

1. The freeze is the commit and the Etude run ID. Launch from a committed
   revision and record both in the report. Nothing else is frozen, hashed or
   admitted by hand.
2. Spending. At or under USD25 for the whole experiment (training, evaluation
   and retries), launch with the launcher's own cap (`--max-usd`). Above that,
   a person's go-ahead naming the amount is required; when it is absent, stop
   before renting and say what amount is needed. An approval already given is
   not asked for again.
3. Look at the experiment's existing runs before launching. Resume or finish
   them; never duplicate a run because an earlier step stopped. Etude owns jobs
   and recovery.
4. A change that keeps the question (a capacity limit, a crash fix, a device,
   a machine type, a retry) is fixed, relaunched and recorded as one dated line
   in the report. It is not a new protocol or a new version. A change to a
   finding's question or to what it measures is a new dated opening
   paragraph, and the runs before it become exploratory.
5. Cost is the launcher's number for the run, recorded once per run. Do not
   reconstruct bills, reserves or settlements. Confirm the rentals ended.
6. Keep what the runs retain: failed games with their seeds and Commands,
   numerical health, and artifact bytes published through `manabot.data`
   before any manifest is committed. File reproduced engine defects as bug
   Tasks. No artificial load, no fallback policies, and no spending to obtain
   a favorable result.
