# Learner admission and backpressure

This is a scheduling model, not a distributed learner. The complete preregistered
protocol and analysis are retained in `accelerator-probes/6f66accbba689162134fad90ae0fb39fcd73baa166a12efe4ab7bcb83360ad96.learner-notes.txt`.

Current code collects a frozen behavior batch, then updates policy/EMA
(`manabot/training/execution.py:1270–1355`). Ataraxos filtering is centralized,
with one optimizer step per retained timestep (`ataraxos.py:211–310`). Existing
research leaves nonzero policy-lag admission unselected (`docs/distributed-rl.md`).
Thus equal row counts do not imply equal learning work: the real 512-row probes
used 96 optimizer steps at 16 streams versus 24 at 64 streams over three updates.

`scratch/learner-flow-probe.py` calibrates one fixed 8×64 batch from the retained
L4 reference run: mean collection 0.395460 seconds, learning 0.250497 seconds.
It counts production-start policy versions, waiting time, completed learning,
stale/full discards, and queued/producer-held/active packets. Every admitted
packet must have version lag zero. Synthetic offered rates are independent of
the inference simulator; larger global cohorts require separate calibration.

Seventeen cases and serial timing, partial service, fractional-clock, zero-lag,
block/drop, conservation and replay checks passed. Final evidence is
`accelerator-probes/learner-flow-clock.json`. Earlier reports remain; a timestamp
roundoff fix admits exact-horizon completions within 1e-12 seconds, never stale
policy versions. Three invocations cost 0.703 seconds; legacy probe usage is 232.905/240 seconds.
No GPU allocation occurred.

Key counterexamples:

- With 4,096 complete-batch producers and queue capacity one, blocking retained
  up to **511 completed packets**: 509 held, one queued, one active. Dropping
  overflow bounded that peak at two but discarded 510 packets for overflow.
- Both modes learned only two of 8,193 produced batches; the rest were stale or
  overflow discards. At eight producers offering half nominal learner capacity,
  only two of 17 batches were learned. Speculation can waste work below capacity.
- The serial control completed 20 iterations with no discards and 38.78% learner
  duty. Version lag and wall-clock age are distinct; generated rows are not
  admitted learning rows. Pending work at the finite cutoff remains counted.

These are scheduling counterexamples, not steady-state training predictions.
Queue length alone cannot bound retention or make stale learning valid. Candidate
controls include global batch leases, other zero-lag quotas, or a separately
validated learning rule; none is selected. Next: measured global-cohort service
and integration with inference sharing, rather than summing independent
optimizers as one consistent learner.
