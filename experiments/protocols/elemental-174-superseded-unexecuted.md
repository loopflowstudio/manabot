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

Predictions, fixed before scientific execution: the guarded short allocation
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
Freeze exports and Etude evaluation cadence at **25 absolute updates**, plus
initialization and the actual final endpoint. Every retained checkpoint must be
evaluated on this cohort; no post-hoc checkpoint selection. The live report
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
The current default temporary SSO/role credentials impose the existing **3,540
second** allocation limit. Use that supported default path, without switching
accounts or extending the lease. This is an intentionally safe short baseline,
not a claim of $10 worth of learning. The retained MTG-159 $9.88 point had a much
longer rental and excluded separately charged calibration/evaluation.

`elemental-174-job.json` freezes L4 at at most $0.60/hour plus $0.02/hour storage,
3,540 total seconds, and $1 job reservation: projected maximum **$0.609667**.
The supported 90-second guardian proof is inside the absolute allocation window;
conservatively reserve another **$0.021** for its observed lifetime. Retain a
further $0.369333 inside that $1 reservation for observation/deletion uncertainty;
the remaining $9 is unavailable for automatic continuation/top-up. This is an
admission bound and conservative lifetime estimate, not an invoice guarantee.
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

Etude bridge: pinned source 3929f25b023aaa3289e1cb4a0914a584a5c7873f, separately
installed controller and worker environments; training retains SDK 0.1.1.
`elemental_live` is an OS-managed local owner with one lock/config ledger and an
absolute deadline; it launches `etude.jobs.evaluation` tournaments, runs the
authoritative Manabot arena in its own environment, and renders Etude's report
plus a learning curve. No legacy saved-evaluation upload stands in for games.
Owner deadline: allocation deadline plus four hours for bounded local scoring;
each CPU evaluation job is capped at one hour, one attempt, two workers,
30 seconds/game and 10,000 Commands/game. No rented CPU is authorized. Crash
recovery observes the original evaluation identity and never launches a new
training job. Failed engineering runs remain retained at zero rental cost.

Publication requires confirmed provider absence, actual inclusive lifetime
receipts, a reloadable admitted endpoint, realized coverage and the bounded
evaluation results/limitations. Publish immutable receipts and raw bytes through
`manabot.data`, commit only manifest/report, and leave MTG-174 incomplete if an
external blocker prevents the accepted actual outcome. Historical comparison:
`experiments/scale-ladder.md`, its allocation/evaluation receipts and retained
MTG-159 checkpoint, with 0.11→0.69 against heuristic at update 9,250; 0.11 was
not head-to-head against initialization. Next question is separately authorized
adaptive sampling, before any hard-coded world switch; no paid follow-up here.
