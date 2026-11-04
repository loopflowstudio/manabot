# MTG-213 — decisions and assumptions (2026-11-04 UTC)

Headless demo boundary (19:31 UTC): interpret this review as acceptance of the
enabling repair before delivery, with paid pilot acceptance still owned by the
following verification step (`lf help code`: pursue → demo → deliver → verify).
This is an agent scope interpretation, not new human approval or Task completion.
Fresh proof and the advance rationale are in `scratch/tla-pilot-demo.md`.
## Next card-file milestone — 2026-11-04

Jack Heart selected full support for any Standard deck, initially prioritizing
the post-rotation pool. He requires increasing engine composability/modularity
so shared rules parts multiply across cards, and improving or preserving both
search and ordinary runtime performance. He requested one independent analysis
per Standard set, a retrospective of Allies/Lessons, Elemental and TLA delivery,
an audit of existing shortcuts that may warrant substantial refactors, and a
comparison with other large Magic engines. Research is active at
[Standard milestone research](../docs/research/standard-2026-11/README.md).
The working rotation assumption is early 2027. Implementation order, exact
tournament-play boundary and any future architecture changes remain proposals.
This request does not inherit historical T6 shipping or auto-merge instructions.
Jack Heart clarified the acceptance standard on 2026-11-04: imperfect rules
correctness is expected; a generally working engine that supports training is
valuable. The consequential failure is later having to invalidate conclusions
believed learned. Prioritize defects by training disruption, strategic distortion
and risk to experimental conclusions, and architectural work by the recurring
cost it removes. Do not turn all edge cases into prerequisites for training or
card admission. Retain exact versions and evidence so discovered defects can be
scoped to affected conclusions; neither a rules fix nor a bug automatically
invalidates every checkpoint or prior result.
Jack Heart confirmed Phase (`https://phase-rs.dev/`), XMage
(`magefree/mage`) and Forge (`Card-Forge/forge`). He sees Phase as particularly
useful for problems its wider coverage has exposed, while judging its code
less disciplined. He requested both regression/commit-history lessons and a
deep comparison of engine objects, state machines, continuous effects and
triggers across these systems. Source-level structural comparisons and proposed
changes must remain distinct from tested runtime or performance claims.

Jack Heart authorized three implementation Tasks from the current-engine audit,
then explicitly added one independent layers Task (not one Task per card pair).
All four were created and launched on the code Graph on 2026-11-04, with ordinary
interactive review before landing: MTG-241 composable execution, selection,
replacements and contextual reads; MTG-242 mana restrictions/resources; MTG-243
catalog-wide naming and policy scaling; MTG-244 layers. Their self-contained
briefs are under `docs/research/standard-2026-11/tasks/`. No merge or experiment
launch was authorized. All four primary Task conversations were opened and
visually confirmed in separate terminal blocks in the existing `manabot` Rex
session. `lf open --host rex` is unimplemented; opening used `rex run --session
manabot ... lf open <task-url> --host terminal`.

The requested durable engine guide is `managym/PRINCIPLES.md`, linked near the
top of its README. MTG-241's brief carries its full draft and owns including it
in delivery. It incorporates the five implementation lessons, imperfect-but-
trainable acceptance, and architectural ownership boundaries. MTG-244 includes
Beorn the Fierce/Kellan plus ten nominated layer interaction witnesses. The old
Blood Moon/Urza's Saga automatic-sacrifice ruling is obsolete; use current rules
and preserve the distinction between basic layer order, timestamps and actual
same-layer dependency.

Jack Heart subsequently requested fewer than 50 Tasks for the post-rotation
milestone, prioritizing trusted rules, performance and composability, then
coherent changes, large engine changes early, and full FDN before generally
newer sets. Only the first 2–3 milestones should be concrete, over at most eight
new Tasks excluding the four launched. Include frequent bounded training
validation for learning, runtime and replayable crashes, plus sparser deeper
strength/generalization assessments. This is planning, not experiment launch.
The proposed two-milestone/eight-Task sequence and rough later arc are in
[standard-milestone.md](standard-milestone.md), with the requested HTML API
walkthrough at [standard-milestone.html](standard-milestone.html). API sketches
are proposals; existing excerpts are revision-pinned. MTG-245 was subsequently created as a separate research Task (see below). Jack Heart then requested roughly 1.5–2×
more cards per implementation Task and more ambitious exploration of Adventure,
prepare, planeswalkers and other engine-shaping mechanics. The plan now calls
for broad cross-set mechanism cohorts, with explicit identity/permission/ability
comparisons inside implementation Tasks; full FDN remains the first set closure.
Training coverage follows the broader cohorts without automatically doubling
run budgets. Exact per-Task card counts remain to be assigned from the ledgers.

Jack Heart's accepted scope is all 16 T6 cards plus one catalog (2026-11-02). His 2026-11-03 steer requires completing the detected Sozin gaps and auditing earlier approximations, then publishing for review. No merge or auto-merge is authorized. The current plan is [mtg-225-tla-t6.md](mtg-225-tla-t6.md).

Current outcome, dependency and acceptance: `scratch/tla-pilot-verify.md`.
Results and cost receipts: `experiments/tla-trophy-pilot.md`. Training Wave
memory now records the accepted shared-effect repair and its evidence limits.

## Accepted direction

Jack Heart's comment 1540e0ad selects preliminary fixed-corpus TLA self-play
and evaluation now, superseding the deferred Task brief. Comment 6434b4dd
authorizes GPUs under a hard $10 cumulative cap and shipping after checks.
Comment d2108241 requires a launch milestone only after observed GPU learner
updates, including job/GPU/progress/rate/deadline evidence. Neither quotes nor
the CPU proof satisfy it.

Comment 42a12145 resolves the next repair: Jack Heart approved shared-effect
fixed tensors, encoding each effect once and routing it to affected object
embeddings before attention. Task 5da5f4 delivered the implementation in PR #419
and owns investigation of redundant activations. Preserve semantics; no competing representation
implementation or unchanged v3 retry. V4 consumed that contract; v5 executed it with fresh initialization/evaluation and the same budget/population;
its action-capacity failure and remaining acceptance are in the continuation note.

## Scientific and operational choices

- Population: CC BY 4.0 public TLA PremierDraft dump; the trophy API restricts
  reuse to 17Lands.com. First 1,000 events with seven observed wins in source
  order, retaining seventh-win main/sideboards and event/build/results. Missing
  losses remain unknown. Native whole-list admission yielded 134 eligible and
  866 rejected without resampling. This is historical trophy data, not a uniform
  draft sample. Live fetching and held-out-deck generalization are deferred.
- Recipe: MTG-181/183's 64x2/four-head damped policy, with explicit new-world,
  full-vocabulary and trophy-population deviations; no checkpoint or budget
  inheritance. The $1 allocation, 100-update target, 2,700-second watchdog and
  eight-pair/four-leg evaluation are frozen engineering choices.
- Credentials: nested Doppler `manabot/prd` then `etude/prd` supplies artifacts
  and RunPod. Configured GPU alternatives are authorized within $10; record
  hardware/allocation deviations before scoring. Quotes reserve nothing.
- Launch: freeze source once through `source.freeze`, retain the typed Run,
  quote/submit through `client.launch_cohort`. Etude 0.2.12 could not append
  cohort runs, so `mtg213-pilot-v1-correction1` used a separate $1 ceiling
  without creating a scientific sample. T6 and MTG-239 retain separate owners.
- Analysis: retain milestones after early endpoints and both comparison slots
  for unknown endpoints. Failed callbacks count as failed matches; unknown
  games remain missing and attempted games are a lower bound. The companion
  analysis CLI owns the joint criterion; Etude curves are descriptive.

## Frozen compatibility and ownership

All executed protocols, configurations and evidence remain immutable. V1's
140-fact failure motivated the prospective input-v6 hotfix: the operator first
selected 512, then 256 after Jack Heart's question (78f4c226) and explicit
confirmation (be5b1394). Comment 0c17ad58 was an operator-issued run reason,
not separate human confirmation. No v2 execution/scoring existed at amendment.
V5 input stays 128 rows; generic fresh v6 defaults to 512 and timestamp scaling
stays rank / 128. PR #415 changed the runtime digest; v3 bound it without
expanding the original 134 decks to the later 306-list admission audit.

V3's 270-fact failure superseded the larger-table assumption. PR #419 supplied
shared-effect input v7: 128 local facts, 128 × 18 headers, 512 × 3 details,
2048 × 2 memberships. V4 consumed it, then failed at 61 card/sideboard objects.
V5 prospectively raised the existing card capacity to 128 under workflow-fix
authority. These are bounded allocations, not universal coverage claims.
Schema-3 protocol shapes and hashes bind every axis. Native configurable-axis
repair (operator bbc34e15) and selected-cost X repair (operator a85a89e8) followed
v5's paid failure; the continuation note owns current implementation and Runs.
W17 remains open per WORLDS.md; source changes do not refresh historical pins.
Synthetic compatible-world fixtures never authorize training, replay or encoding
under historical identities.

Related Tasks keep their own ownership: 5da5f4 delivered shared effects and
investigates redundant activations; T6 owns its rules work; the primary
conversation released Etude 0.2.14. No competing repair or release is selected.
The isolated published control client may use Vast while frozen workers retain
0.2.12. VAST_API_KEY remains in Doppler etude/prd. Setup plus the one-hour job
must fit the whole-rental ceiling; prior placement used >=6 vCPUs, >=48GB RAM,
80GB disk, <=USD0.81/hour and 840 seconds setup. The existing SSH public key was
registered. Unrelated provider rentals remain with their owners.

Complete decision history, superseded source pins and prior local evidence
locations are retained at
`af4227616ef2cf46bb1f0b05e690feebb28ac2e8:scratch/questions.md`.
T6 decisions inherited during sync remain at
`5cd6ed3f6e54e81f9f39c5c2bb56f322dc1576c1:scratch/questions.md`;
they do not replace MTG-213's scope. The current continuation note and report
retain historical failures, missing evaluation slots and settled costs.

## Explicit first-64 pilot — 2026-11-04 UTC

Jack Heart's steer 3b0f8826 supersedes 512-action v7: use deterministic first 64
legal native actions with replay-linked overflow diagnostics in training and
evaluation. This is an explicit policy approximation, preserving complete native
and Academy authority and original index/Command identities. V8's serialized
configuration and protocol bind it; prepared v7 must not launch. Earlier
exploratory steers d5dfad91/d47cb28b did not select a method; 3b0f8826 does.

Superseded interpretation: v8 originally retained the heuristic's full native
policy. Jack Heart subsequently selected concept-review option 1b below: all
pilot evaluation participants use the same first-64 legal prefix.

V8 placement assumption at 19:12 UTC: after fresh RTX5090 price and L4 capacity
refusals, try a separate RTX3090 supply with the same >=6vCPU/48GBRAM/80GB disk,
<=USD0.81/hour and <=USD1 allocation. The existing GPU/budget authorization
permits this hardware substitution; preserve the exact source, recipe and job
wall cap, and record the quote/rate/deadlines before scoring. Another Task's
live RTX3090 rental is left untouched and is not charged to this pilot.

## Interactive concept review — 2026-11-04

Jack Heart invoked concept-review after PR #423 entered landing. This review
does not revoke shipping authorization or restart execution. The accepted
first-64 direction above remains in force. Implementation is evidence, not
additional approval. The review page is `scratch/concept-review.html`.

Experience and model: a manabot has a saved decision budget; managym still
defines the complete legal choices. Legality is repaired before selecting the
prefix. At an illustrative 80-choice decision the model scores rows 0..63,
executes the original Command, and records 16 omitted choices. Academy can
still display all 80. Padding is unavailable, and no page/continue action is
invented. The cutoff changes policy support, not game termination.

Accepted decision: Jack Heart answered **1b** on 2026-11-04. The heuristic,
learned candidate and initialization anchor must all use the same first-64
legal native-action prefix. This supersedes the earlier recommendation and v8
full-support heuristic interpretation. Implementation remains with the live
Graph (steer ffd29028); this review does not claim delivery. Bind the matched
benchmark prospectively; preserve frozen sources and executed evidence.

Reporting finding: `Env.step` and `step_semantic` record cutoff diagnostics for
every consumed environment decision. `ActionCutoffCounts` aggregates by kind,
not actor/policy, while the evaluation runner allows the heuristic to act from
the full raw observation. Therefore the aggregate is observation overflow across
the game, not the frequency at which the learned bot actually lost choices.
Bounded samples contain actors but cannot reconstruct actor-specific totals.
Before making a policy-impact claim, either label the existing metric precisely
or retain exact counts by acting participant and whether that participant uses
bounded support. This follows the existing honest-reporting requirement; it
does not select a new opponent policy. Proposed implementation direction only;
no executable changes are made in this review.

Draft usage guidance (proposed clarification): “A checkpoint saves its action
budget. Training and checkpoint evaluation use the same first-64 native prefix.
The engine's full legal set and human play are unchanged. The pilot heuristic
also chooses from that same prefix. Label whole-game and per-participant cutoff
counts distinctly; neither ordering neutrality nor rare strategic impact is
established by a low aggregate count.” Existing README setup guidance stays.

Draft operator guidance (proposed clarification): “A capacity cutoff is a logged
policy decision, not a failed game. A rules error remains a rules error. On an
interrupted run retain source, checkpoint configuration, Commands, partial counts
and costs; do not infer zero overflows from an absent final report or reconstruct
lost optimizer state from counters. Any changed selection rule needs a new
prospective experiment identity.” No separate skill installation is needed.

Next proof: construct a >64-action position where the unrestricted heuristic
prefers an excluded action; the matched-support heuristic must select its best
retained legal action. Restrict candidate selection before scoring/choosing;
never clamp or remap an excluded chosen action. Preserve original Command
identity. In a mixed match with overflow only on heuristic turns, aggregate
policy restrictions must be positive while learned-bot restrictions are zero.
Also challenge a prefix whose only useful action occurs
after row 63: it remains legal but inaccessible, illustrating the accepted
approximation. Retain the existing <=64, prefix/Command, illegal-X rejection,
masked distribution, close/reseed accounting and frozen-configuration proofs.
Hard termination can leave counts since the last flush missing; exact totals
require normal finalization. GPU learning gain and universal action fairness
are not established by these software proofs.

## CI integration — 2026-11-04

PR #423 recorded head `d2ff60e6a1c11aa77e9592d0ecc3aedd72e0c3a7`
required integration with main. The only textual conflict is this notes file.
Preserve MTG-213 direction and the concurrent concept review above; incoming T7
notes remain at `4463e57f5a92ebd3b60e2fb50c046b84ac2c0372:scratch/questions.md`.
Those historical T7 delivery boundaries do not replace this Task’s authorization.

The CI repair implements comment ffd29028 in prospective v9: matched first-64
heuristic support before ranking/fallback, an explicit benchmark identity and
the T7-integrated runtime. V8 prepared source/protocol/evidence remain frozen.
No paid run was launched. Aggregate diagnostics describe all participating
policies; actor-specific claims remain unsupported by aggregate counts.

CI repair validation: integrated main 4463e57f5a; 968 debug Rust tests passed
(two ignored), Rust formatting and Clippy passed, extension rebuilt atomically.
Python focused suite passed 141 tests, then 84 relevant tests passed after five
additional invalid-capacity cases; 96 broader admission/monitoring/boundary tests
passed. Ruff and diff whitespace checks passed. Recorded failure remains
d2ff60e6a1c11aa77e9592d0ecc3aedd72e0c3a7 / required-integration. No paid launch.
The frozen v5 protocol pins player/card widths 67/55; T6 uses 73/56. Synthetic scheduling/analysis fixtures pin these historical widths as well as world identity, while separate negative tests prove real-world and real-schema rejection. Frozen protocol/regime/evidence bytes stay unchanged.

## Growing card-pool training methodology — 2026-11-04

Created MTG-245 (task_af1330d001e4460f9eab6f9fef7400cc), “Design training and
evaluation for growing card pools.” Jack Heart authorized launching it in Rex,
and corrected the proposed replacement of Task 04: MTG-245 is separate research;
Task 04 must actually run training, and Task 08 remains the later actual run.
The near-term plan now has nine new Tasks. Research has no engine prerequisite;
training depends on the selected protocol and integrated snapshots. No training
or compute provisioning is authorized by this research launch. Full brief:
docs/research/standard-2026-11/tasks/growing-pool-training.md.

MTG-245 launched on the code Graph (d70ef881-3587-4355-8fc5-b5fbef861965),
confirmed running implement. Worktree: /Users/jack/src/manabot.mtg-245-design-training-and.
Opened its primary conversation in Rex manabot block
block:01a1a853-8fe4-7c6b-be5f-d064e00ea155. Corrected Task description saved
locally before launch; Linear writeback is pending due to service rate limiting.
