# MTG-213 — decisions and assumptions (2026-11-04 UTC)

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

## Frozen compatibility and provenance

V1's 140 distinct facts exceeded 128. The operator selected v6/512, then 256
following Jack Heart's question (78f4c226); be5b1394 relays his explicit approval
of the immediate larger-capacity hotfix. Comment 0c17ad58 was an operator-issued
run reason, not separate human confirmation. No v2 execution/scoring existed
at amendment (`.runs/mtg213/v2-pre-amendment-state.json`). V6 retains all columns,
timestamp rank / 128 and fatal overflow; v5 stays fixed at 128 and legacy
serialization omits `max_facts`. Generic fresh v6 defaults to 512.

PR #415 changed the native digest, so v3 bound the merged runtime with the same
134 entries, v6/256, recipe and schedule. The later 306-list admission audit
never expands this population. V3's 270-row failure invalidates its capacity
assumption; frozen v1/v2/v3 inputs and evidence remain reproducibility records.
The native/Python duration-parity repair does not refresh those runtime pins.

Historical details: `b0d921cec92784d9303aea3d0d904416e8607982:scratch/questions.md`
retains source-binding decisions, fixture repairs, documentation contributions,
sync resolution and links to older decision history. V2 source is
`d655d37996d2702cd5fee6ac37db11d855e87640`; pre-merge validation is at
`7b8bb32731205981b6be09097eb4ca4b39b3864f:experiments/tla-trophy-pilot.md`.
Uncommitted continuation notes before consolidation are preserved in
`.runs/mtg213/context-before-compress/`; their substantive findings remain in
the continuation note. The manifest retains all 120 added v3 evidence entries.

Shared-effect owner decisions remain at `fb5ad2c1372f9c967b550f0b8ae0f84883f75563:scratch/questions.md`; its memory section was preserved during integration.

Prospective v4 uses delivered v7 defaults explicitly: 128 local facts, 128 × 18 effect headers, 512 × 3 details, 2048 × 2 memberships. Protocol schema 3 serializes every ObservationSpace shape and checks it at runtime; capacities remain bounded engineering choices, not universal coverage. This is a new initialization, preserving the prior recipe and all 134 entries.

V4 failed before learning at 61 card objects including sideboard (40 original cards + 5 tokens + 16 sideboard). Under the existing workflow-bug authorization, the prospective v5 pilot explicitly raises the existing `max_cards_per_player` configuration to 128, retaining v7 and all shared-effect capacities. This is an operator allocation to accommodate generated objects, not a representation change, population exclusion or universal bound. V4 source/evidence remains frozen at `0b6c2bb854ba48b37fb7e5ac360c0c9d047cea34`.


## Main integration boundary — 2026-11-04 UTC

The [continuation note](tla-pilot-verify.md#source-and-repair-boundary)
owns the executed v5 Run, T6 runtime/schema mismatch and fixture limitations.
V5 is now closed as failed. Operator comment bbc34e15 (2026-11-04 17:49 UTC)
selects native configurable-axis repair and prospective continuation under
Jack Heart's existing authorization; no new product decision is required. The
successor source/capacity/protocol identity is now bound in the continuation note.
Synthetic fixtures never authorize actual training, encoding or replay under
a historical identity.

A main sync brought T6 scratch decisions into this checkout. They remain at
`5cd6ed3f6e54e81f9f39c5c2bb56f322dc1576c1:scratch/questions.md`; they do not
replace MTG-213's accepted scope or shipping authorization. This file restores
the pilot decisions from merged PR #420, with this integration boundary added.

## Vast continuation selected — 2026-11-04 UTC

Jack Heart requested releasing Vast support then using it for TLA. Steer
c38da514 transferred GPU continuation to Graph 3529715b, with release 0.2.14
owned by the primary conversation. The isolated published control client kept
frozen v5 unchanged. The selected RTX 5090 placement required >=6 vCPUs,
>=48GB RAM, 80GB disk, <=$0.81/hour and 840 seconds setup plus the one-hour job,
inside the $1 allocation/$10 cumulative limits. The existing SSH public key
was registered; VAST_API_KEY remains in Doppler etude/prd.

V5 ran once and failed after three updates; the continuation note owns settlement,
remaining acceptance and the prospective repair boundary. Release 0.2.14
completed at 17:40:58 UTC; release and provider handoff facts are consolidated
there. Their complete pre-edit text, together
with this file and the report, is preserved under
`.runs/mtg213/context-before-compress-20261104-1745/`.

## X-cost blocker found before v6 submission — 2026-11-04 UTC

Operator comment a85a89e8 (18:11 UTC), read during fresh launch reconciliation,
requires investigating the 256 X offers before any successor training. The
prepared v6 Run `bcde93e9ff8c` / source `ad2c3a010f53` was never submitted: a
new unrelated Vast instance failed the empty-inventory guard first.
Do not reuse this prepared Run after a rules repair.

Exact current-source replay identifies Wan Shi Tong in exile with an airbend
{2} grant, seven untapped basics and zero ordinary/combat/restricted mana. The
cast path asks for X from the printed {X}{U}{U}, while payment uses fixed {2},
so it admits 0..255 and can draw into a deck-out. CR 107.3b fixes X to zero for
that alternative cost. The independent buffer contract remains needed.

The shared selected-cost repair and directed payment/counter/draw tests are
implemented. The [current implementation cut](tla-pilot-verify.md#current-implementation-cut)
owns deletions and validation; the historical 256-offer observation remains
transport-only data. W17 remains open per WORLDS.md.


V7 continuation is now source-bound but unsubmitted: Run `af3661790ac5` at
`b5406bb33abd98870af6ca07c3beed63a92b7c1c`. Vast refused its price ceiling;
the configured L4 alternative refused actual capacity. No new spending or
scoring occurred. The continuation note owns dated inventory/cost receipts and
remaining acceptance. Do not re-freeze the prepared Run onto later documentation
commits or reuse the unsubmitted, pre-rules-repair v6 Run.
