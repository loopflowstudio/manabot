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
embeddings before attention. Task 5da5f4 delivered the implementation in PR #419 and owns redundant-
activation investigation. Preserve semantics; no competing representation
implementation or unchanged v3 retry. V4 consumed that contract; the prepared
v5 retains it with fresh initialization/evaluation and the same budget/population.

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

The T6 merge at `5dd11cec4100aa0e1f4a23c063dfd5778e2ba28f` changes the
runtime world and input schema. Frozen v5 pins player/card widths 67/55 and
64 actions; current source uses widths 73/56 and a fresh 512-action default.
The existing runtime and shape guards reject substitution. Synthetic scheduling
and analysis tests use explicit historical identity/shape fixtures; those
fixtures never authorize actual training, encoding or replay under that identity.
Prepared Run `e8c76bc2c06f` remains on source
`97a2cfb963d7e3a3b2816f31c45dda72f34cdddd`, as the continuation feedback selects.
No prospective v6 pilot or re-freeze onto main has been selected.

A main sync brought T6 scratch decisions into this checkout. They remain at
`5cd6ed3f6e54e81f9f39c5c2bb56f322dc1576c1:scratch/questions.md`; they do not
replace MTG-213's accepted scope or shipping authorization. This file restores
the pilot decisions from merged PR #420, with this integration boundary added.
