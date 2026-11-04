# Accepted pilot resource-limit policy — 2026-11-04

Jack Heart accepted bounded per-game resource-limit recovery after observing
Phoenix Fleet Airship population growth ("i like that"). He then clarified:
"can we just drop the game and not everything else" and explicitly selected
"quite permissive, like 25%" for the error rate.

This amends the active MTG-213 repair. A capacity bump alone is insufficient.
On a recognized game-local resource/capacity overflow, retain replay evidence
and the precise reason, discard only that game's uncommitted training samples
for both seats, reset its stream, and continue other games in the same batch.
Preserve completed optimizer updates and unaffected streams, pending samples,
randomness and counters. The operator's earlier phrase "discard their unfinished
training batch" was incorrect and is superseded by this explicit clarification.
Never invent a win, loss, draw or terminal reward for the abandoned game.

Use a prospective, explicit pilot failure threshold of 0.25, stopping when the
measured rate is strictly greater than 25%, with the existing bounded rolling
window/warmup semantics unless implementation establishes a concrete reason to
change them. Pin window/warmup in the new protocol and report the denominator,
all failed attempts, consumed compute and reasons; distinguish resource limits
from rules/engine errors. Reuse the existing recovery/ledger owner. This does
not authorize swallowing arbitrary exceptions, malformed buffers or infrastructure
failures. Preserve fail-fast behavior for integrity faults and frozen policies.

Evaluation records a resource-limited game as failed/incomplete in its original
scheduled slot, never a replacement successful result or a fabricated outcome.
Training replacement games follow the declared sampler without excluding decks.
The report must disclose resulting selection bias and incomplete comparisons.

Keep the prospective permanent64 choice as an engineering capacity if useful;
do not escalate capacities indefinitely. Broader variable-length/grouped-token
architecture is a future design, not selected implementation here. Preserve v9
source/protocol/evidence. Implement in its unexecuted successor and validate
isolation with controlled fixtures and the retained Airship replay: an overflow
in one stream leaves other streams learnable, produces no partial-game gradient,
and threshold boundary behavior is exact. Maintain USD1.25 allocation / USD10
cumulative allowances, all attempts included. Continue the already live Graph;
do not launch a competing repair.
