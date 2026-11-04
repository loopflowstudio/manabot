# MTG-213 — decisions and assumptions (2026-11-04 UTC)

Headless demo boundary (19:31 UTC): interpret this review as acceptance of the
enabling repair before delivery, with paid pilot acceptance still owned by the
following verification step (`lf help code`: pursue → demo → deliver → verify).
This is an agent scope interpretation, not new human approval or Task completion.
Fresh proof and the advance rationale are in `scratch/tla-pilot-demo.md`.

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

Reversible interpretation: the fixed heuristic baseline retains its established
full native policy. Learned candidate and initialization-anchor policies both
use the declared 64-slot cutoff. Evaluation reports measure observation cutoff
frequency and retain the full game trace; no baseline redesign was requested.

V8 placement assumption at 19:12 UTC: after fresh RTX5090 price and L4 capacity
refusals, try a separate RTX3090 supply with the same >=6vCPU/48GBRAM/80GB disk,
<=USD0.81/hour and <=USD1 allocation. The existing GPU/budget authorization
permits this hardware substitution; preserve the exact source, recipe and job
wall cap, and record the quote/rate/deadlines before scoring. Another Task's
live RTX3090 rental is left untouched and is not charged to this pilot.
