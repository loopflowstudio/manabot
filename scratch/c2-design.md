# C2 design (2026-11-11): large-model packing on one A40

The owning text is the protocol section "Step 3 continuation C2"
(`experiments/protocols/training-throughput-b015d5-v1.md`). Next step: design
review in the Task conversation, then execution review. Nothing has been rented.

- The source is frozen C1 `b5e8f404…`, unchanged. Cells are the existing `C1_LEGS`:
  a384l8 K1 → K2 → w160l8 K1(u12) → K2 → conditional K4s.
- Launch path: clone with no `origin` at `~/.etude-b015d5-step2/src/manabot-c2`. A
  K=4 queue took 4 s, against C1's 20–25 min. `b015d5-a40` is held at max 0; rent
  by hand; supervise the heartbeat; tear down independently.
- Reservation: one A40, ≤4.0 h, ≤$2.50 all-in, quote ≤$0.60/h. Expected
  ~$1.20–1.45. Worst case leaves step 3 at $8.32/$10 and the experiment at
  $10.42/$50.
- Software before execution: only the evidence-directory driver
  `c2/step-3-c1/campaign.py`. No manabot or Etude change.
- Untracked local preflight receipts are in
  `experiments/data/training-throughput-b015d5-v1/c2/step-3-c1/preflight/`. Publish
  them with the C2 evidence.
