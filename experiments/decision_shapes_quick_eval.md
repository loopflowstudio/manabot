# Equal-update payment-model head-to-head

Exploratory check requested 2026-11-16 UTC: compare available policies at the
same number of updates, not a scored finding about learnability. Compare raw
update-50 checkpoints of seed26502: baseline4218e16c4678 and original
mana-taps9b87af18dcdb, both trained on the same900-deck population. The latter
ended at66updates due to its wall budget; the separate Yue-excluded169-update
calibration is not substituted into this comparison.

Eight deterministic pairs from the held-out100-deck corpus, excluding decks
with Yue in the main or sideboard for both policies, with four crossed
deck/seat legs per pair. Run32games under ordinary decisions and32under
source-tap decisions; the process-wide mode applies equally to both policies.
Report these modes separately: neither mode is a neutral common action space.
Shared native trivial-decision skipping and first-legal policy overflow remain
as trained. Keep all attempted, invalid, timed-out and unattempted games;
only complete replay-verified four-leg pairs enter a paired score.

Use the committed study engine/backend and exact world-bound checkpoints;
never rebind a checkpoint or insert the new production autotapper. Original
monitoring receipts bind both files to50updates. Files and summaries are Etude
outputs; detailed evaluation retains Commands, replay receipts and allocation
identity. No games run on the laptop. No baseline training restarts.

One manually acquired L4, at most two hours and USD2 including startup, with
one bounded job and cleanup afterward. The launcher's study rental total at
planning was USD32.11969780140406483881217442; this fits the existing USD55
total authorization. Do not automatically acquire replacements or retry a
failed admission without inspecting it. This small one-seed comparison has
limited precision and cannot establish a general training-method benefit.

2026-11-16 placement change: the user suggested the laptop or Mac mini for
this small panel. GPU run e9589a885adf was cancelled during setup, with no
games delivered. Move the same checkpoint pair and schedule to an isolated
source/runtime directory on mini-heart, using CPU inference and two active
games. Do not replace another checkout's environment. Source changes only
placement/device and concurrency; report it as the same exploratory check.
