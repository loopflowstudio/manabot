# TLA pilot with isolated resource-limit recovery

V10 is a prospective successor to executed v9 Run `2171156493ca`. V9 completed
five updates and failed at 41 permanents against capacity 40, before exporting a
trained checkpoint. Its protocol, regime, source and evidence remain unchanged;
its executable runner is retained at source `6da60684c`. Current discovery and
analysis use `tla_pilot_v10` and require fresh initialization and evaluation.

## Capacity and recovery

V10 selects `max_permanents_per_player=64` using the existing configurable axis.
The exact retained stream43 / seed32624 / 371-action prefix fits both 41 and 64
rows without dropping an object. This is an engineering allocation, not a bound
on possible battlefields. Phoenix Fleet Airship copies can double across
qualifying turns; card, fact, shared-effect/member and continuation capacities
remain finite too. The matched first-64 legal-action approximation remains
separate from object or encoding capacity failures.

Jack Heart accepted per-game resource-limit recovery and a permissive 25% error
rate on 2026-11-04 (comment ca1660e8). Schema4 pins `game_errors` in both the
protocol and hashed regime: `recover_capacity=true`, threshold .25, rolling
window 100 settled games, warmup 100, maximum 40,000 actions per game. The stop
condition is strictly greater than .25: 25/100 continues; 26/100 stops. The
window counts completed games plus engine failures, resource limits, truncations
and action limits, once per game. Run deadline/cancellation abandonments are
retained separately and do not enter this denominator. All compute still counts
against the rental and cumulative budgets.

Only the native encoder's typed capacity errors during an active game's step
are newly recoverable. The receipt records `resource_limit`, the exact reason,
seed, decks, action prefix, pre-command frame and observation configuration.
Only that game's uncommitted samples, from both seats, are discarded; completed
updates, earlier completed games and other streams survive. The ordinary sampler
replaces that stream without deck exclusion. No terminal reward, win, loss or
draw is synthesized. Existing engine-error accounting remains distinct.

Malformed buffers, invalid shared-effect provenance, setup/reset failures and
infrastructure failures remain fatal. Omitted `recover_capacity` stays false
and is omitted from serialization, preserving frozen regimes. A reset that
cannot fit its initial observation fails the run rather than looping through
unplayable roots. Replay of a resource receipt includes its retained encoder
boundary; native gameplay alone cannot reproduce an allocation failure.

Evaluation keeps resource-limited games failed/incomplete in their original
scheduled slots; training replacement does not fill evaluation gaps. Discarding
resource-heavy games biases the learned distribution toward games that fit the
runtime. The report must retain failures, reasons, discarded samples and missing
comparisons. No learning claim is available without the declared complete paired
analysis.

## Unchanged design and execution boundary

The 134-deck population, independent seat sampling, seeds, input-v7 shared
effects, card128, matched first64 support for all policies, learning method,
100-update target, milestones 25/50/75 and paired schedule remain unchanged.
V10 records the already accepted USD1.25 whole-rental ceiling; USD0.97/hour
placement and the cumulative USD10 cap include all earlier attempts. This is
not a new budget. Final-source freezing, fresh inventory/reservation accounting,
CUDA admission, paid training, evaluation and teardown remain execution work.
Do not reuse v9 initialization, infer a checkpoint from its counters, or launch
its unchanged failed configuration.
