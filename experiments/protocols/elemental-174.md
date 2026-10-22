# MTG-174: prospective uniform Elemental baseline

Question: does the retained MTG-159 learning recipe improve a policy within
the ten-deck Elemental world during one guarded allocation? This is one training
seed, not a test of method superiority or world balance.

Freeze: the git commit containing this protocol and its three JSON companions
is the runnable source. The launch plan must bind that exact commit/tree/lock;
no later main changes enter it. `elemental-174-identity.json` binds the local
native bytes, full world/observation/content identities, all ten exact native
setups/Lesson pools and the complete resolved learning configuration. Linux
build/toolchain/native digests must be retained separately from the Darwin
extension digest; different platform bytes are not claimed identical.

Recipe: `experiments/regimes/elemental-uniform-174-v1.json` retains the historical
64-dimensional, two-layer, four-head, scalar masked-mean, additive (FiLM=false)
policy, Adam and every MTG-159 Ataraxos move learning parameter. Seed **17402**;
256 streams, 64 learner transitions per update, one fixed learner seat `s % 2`.
The absolute update target remains 100,000; allocation expiry may safely pause
far earlier. No optimizer-state continuation or replacement training is allowed.
Necessary differences: w10/full Elemental semantic input and complete authored
roster; independently uniform ordered pairs at **game roots**, including mirrors,
replace the old two-deck cross-balanced schedule. No adaptive sampling or tuning.
Longer games may skew transition exposure. Actual pair/deck/learner-seat roots,
transitions, environment decisions, terminal/truncated games must be reported.

Predictions, fixed before scientific execution: the guarded allocation
will retain at least 25 updated iterations (409,600 learner decisions), at least
95/100 realized ordered root pairs and all ten decks, with no illegal or
non-finite step. Expected learner throughput is 1–4 million decisions/hour of
active learning. Endpoint score versus its own initialization will be at least
0.55; endpoint-minus-initialization score against scripted greedy will be at
least +0.05. Useful checkpoint threshold: endpoint-versus-initialization mean
at least 0.60 with a 95% whole-match bootstrap lower bound above 0.50. Refutation,
early safe stop, no useful effect and incomplete coverage remain reportable.

Evaluation: the exact cohorts are `elemental-174-cohorts.json`. Monitoring uses
seed 17410001, ten adjacent roster pairs (both orientations and both policy
seats), plus ten mirrors grouped in five conditions: 60 games/checkpoint.
Freeze exports and Etude evaluation cadence at **500 absolute updates**, plus
initialization and the actual final endpoint. Only these cadence checkpoints, initialization and the actual terminal endpoint
are monitoring points; no post-hoc checkpoint selection. The owner cancels
training if more than two monitoring points await completion; no point is
silently dropped. At the predicted 1–4M learner decisions/hour, 500 updates
span 2.05–8.19 hours. The measured engineering cost of 19–27 seconds/game
with two workers predicts 10–15 minutes per 60-game point before overhead.
A one-hour monitoring job cap and bounded backlog protect against slower
execution; they do not authorize retries or a different cohort. The live report
shows raw-policy score against scripted greedy, source/checkpoint identity,
pending/failed states, observation freshness, timing and inclusive cost.

Final scoring uses disjoint seed 17420001 and all 45 unordered cross pairs plus
the ten mirrors: 100 ordered pairs × both candidate seats = **200 games** per
comparison. Retain endpoint-versus-initialization, endpoint-versus-greedy and
initialization-versus-greedy (600 games total), Commands and replay receipts.
No held-out seed is used for monitoring. Scripted greedy is a legal weak anchor;
the old heuristic is unsupported as a comparable full-Elemental anchor. The
retained w6 MTG-159 weights are never loaded here. No cross-world Elo is computed.
Etude's fixed-seed whole-match bootstrap intervals quantify this checkpoint's
game/cohort noise, not uncertainty across training seeds. One deal per pair is
narrow; report paired seat/deck coverage and this limitation explicitly.

Execution and cost: one job `mtg174-elemental-uniform-001`, no second training
allocation or seed. The inclusive hard ceiling stays **$10**, including guardian
proof, startup, failures, storage, shutdown and any paid evaluation. No paid
evaluation is planned. Fresh S3/provider inventory and local receipts must show
no prior MTG-174 rental; retain those observations, not an assumed zero ledger.
Fresh credential-only admission supports the existing `manabot-issuer`
profile and existing worker role for **43,140 seconds**. Use that path with
normal scoped issuance and no account change. The default temporary-role
3,540-second bound is inapplicable to this existing direct issuer. Repeat
admission immediately before allocation; failure blocks launch.

`elemental-174-job.json` freezes L4 at at most $0.60/hour plus $0.02/hour storage,
43,140 total seconds, and the unchanged inclusive **$10** reservation.
The main allocation projects at most **$7.429667**. Reserve **$0.05**
for the supported separate 90-second guardian proof and observed deletion,
leaving **$2.520333** for observation/deletion uncertainty inside the same cap.
No paid calibration or evaluation is planned. Fresh control-store/provider
inventories and retained engineering receipts establish zero known prior
MTG-174 rental cost, not a provider invoice assertion. Failed attempts remain
cumulative. The superseded short plan is retained alongside this protocol
and in git; it was never executed. No automatic continuation/top-up.
Stop at the earlier allocation deadline, $10 cumulative ceiling or one day.
Reserves: setup 600 seconds; final checkpoint 120; upload 300; deletion 120.
The remote supervisor/guardian own absolute pause/export/delete deadlines.
Preserve failures and charges. No reset of cumulative accounting on reconnect.

Kill criteria: any non-finite loss/gradient/parameter, illegal offered action,
world/checkpoint identity mismatch, invalid/truncated evaluation game, missing
required export or failed evaluator, lost reporting ownership, or exhaustion of
the original time/cost allowance. The reporting owner requests cancellation on
failure; the independent remote guardian remains the hard backstop. Stop rather
than tune, bypass admission, substitute a seed or allocate a replacement.

Etude bridge: exact source 452bd8a84b08d4732e3cfbaca5e9833b502d4380,
wheel SHA256 3b46aca22eb1b62fbb3b21e6de24d2313375f8ceb1f74de030932fb24b2228c0
in isolated report controller/worker environments. Training retains SDK 0.1.1.
The native Etude training report joins actual linked typed evaluation runs at
absolute update coordinates, with distinct cohort/anchor identities, checkpoint
hashes, uncertainty and pending/failure/freshness. Engineering job174000000006
proved 4/4 replay-checked games at update0; it is not scientific evidence.
`elemental_live` is an OS-managed local owner with one frozen config/ledger,
attached to the admitted scientific job before rental. Its report location
is `<owner-directory>/reports/mtg174-elemental.html`; record the actual
resolved location and first successful scientific evaluated checkpoint.
The owner expires 24 hours after admission: at most 11.983 hours training
plus 12.017 hours bounded CPU scoring/drain time. Monitoring jobs allow one
hour; each of three final comparison jobs allows two hours (six hours total),
one attempt, two workers, 120 seconds/game, 600 seconds per four-game
subprocess and 10,000 Commands/game. Expected final600 games take 95–135
minutes at measured speed with two workers; replay/startup overhead and
checkpoint-vs-checkpoint inference can be slower. Caps are failure boundaries,
not promises of completion. Monitoring must be current while training runs;
more than two incomplete monitoring points kills training instead of growing
an unbounded backlog. Actual final endpoint monitoring and disjoint final
scoring remain required, including after a safe pause. All CPU work is local
through Etude; no rented CPU. Interrupted launches reconcile the same stable
run ID; accepted jobs are observed, never replaced. The durable owner refreshes
Etude's native report every30 seconds through the endpoint.

Publication requires confirmed provider absence, actual inclusive lifetime
receipts, a reloadable admitted endpoint, realized coverage and the bounded
evaluation results/limitations. Publish immutable receipts and raw bytes through
`manabot.data`, commit only manifest/report, and leave MTG-174 incomplete if an
external blocker prevents the accepted actual outcome. Historical comparison:
`experiments/scale-ladder.md`, its allocation/evaluation receipts and retained
MTG-159 checkpoint, with 0.11→0.69 against heuristic at update 9,250; 0.11 was
not head-to-head against initialization. Next question is separately authorized
adaptive sampling, before any hard-coded world switch; no paid follow-up here.
