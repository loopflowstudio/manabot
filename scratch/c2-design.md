# C2 (2026-11-11): large-model packing on one A40

The owning text is the protocol section "Step 3 continuation C2"
(`experiments/protocols/training-throughput-b015d5-v1.md`), including its "Driver
status — 2026-11-11". Design review is recorded in comment 6788204b. The compress pass (2026-11-10)
simplified `campaign.py` without changing behavior; 40 tests still pass. Nothing has
been rented. Next: freeze the protocol and driver hashes, then execution review.

- Source is frozen C1 `b5e8f404…`, unchanged. Cells are the existing `C1_LEGS`:
  a384l8 K1 → K2 → w160l8 K1(u12) → K2 → conditional reruns and K4s.
- Driver and its files are in
  `experiments/data/training-throughput-b015d5-v1/c2/step-3-c1/` (artifact store,
  not git). Before renting: `run_campaign.sh preflight`. After renting by hand:
  `run_campaign.sh MACHINE`. It deletes the machine on every exit.
- Check: `uv run --locked --extra dev --extra artifacts pytest
  experiments/data/training-throughput-b015d5-v1/c2/step-3-c1/test_c2_campaign.py -q`
  — 40 passed (2026-11-11).
- Reservation: one A40, ≤4.0 h claim to provider deletion, ≤$2.50 all-in, quote
  ≤$0.60/h, ≤400 queued updates. Work stops 15 minutes before the deadline.
- SHA-256 at implementation (the freeze records the final ones):
  - `campaign.py` c5359b0186731e5d3e88e5f2f3a2ea8c86b60ad1c5528e5fa64f6cc576b28c8d
  - `identity_job.py` 5ebee6b10581cfdbb680b8d749c8bbeca039aef776b853b6638c469737e3d5ac
  - `run_campaign.sh` 8aab44cabaeaa3f9e349bc9e44f4f9d944e999300fc86aa4697cbe5ff6dd669d
  - `budget.json` fe62eee84e879ca0ec18b97f4dec8474a8dcae3ad3815bea23125e0ea39ed10c
  - C1 `preflight/remote_identity.py` e21eb933dd0c906a53c2176cba83d98463a9078c4bbffb5723c37a27be1f8374
  - `test_c2_campaign.py` eccdac6411b0701f0ce09cd2fa561fa71b0d08899400e66ec59e924c7c6e874f
