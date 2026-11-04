# Effective state and shared effect transport

Implementation plan · Task 5da5f4 · reconciled 2026-11-04 UTC

## Current direction and delivery boundary

Jack Heart's 2026-11-04 02:49 UTC steer prioritizes consolidation **before** fact admission
following MTG-213 v3's 270 > 256 failure. The old same-input neural candidates
cannot repair that gate. No scientific seed has run. The proposed comparison is
suspended pending transport acceptance and a newly frozen common-input protocol.
The $25 Task allowance remains,
separate from MTG-213's $10. Jack Heart’s 03:01 UTC “this sounds good. ship it” approves the bounded v7
repair and supersedes the earlier no-auto-merge restriction. Publication and
landing follow the live code Graph and its delivery/gate boundaries. No change
to fresh defaults or frozen pilot protocols is included.

MTG-213's enabling PR416 is now main `3b122de9078c985f846a0f3c5a3a42ebca3b1182`.
It is integrated here, including PR415's card/continuation corrections and
PR414's catalog changes. Their existing authority remains the baseline; this
repair preserves v6 configurable fact capacity, all v1–v6 layouts, defaults,
recipes, receipts and world identities. This repair
is explicit prospective **v7**, not a change to the frozen v3 experiment.
The transferred observation/stream/replay/audit are under
`.runs/mtg213-consolidation-input/`. The audit proves projected roundtrip only:
135 records = 9 timestamps × 15 objects; 270 keyword fragments become 9 complete
LoseKeywords payloads (both keywords) and 135 memberships. It cannot recover the
old projection's absent native controller or exact source reference.

## Bounded repair contract

One native viewer projection factors canonical `EffectRecords` before tensor
capacity checks. Each shared record contains the **complete** modification,
optional amounts (including absence versus zero), all keyword/subtype/color
arguments, duration, timestamp, source definition, controller and exact source
incarnation where the duration requires it. Membership edges retain the exact
current affected object/player and original record order/multiplicity. Expand
and sort by original order to recover the projected native records exactly.
Nothing is deleted from rules state. Static sources and copy exceptions stay
separate; no changing a fixed affected set into a live scope.

The v7 tensors have three bounded blocks: 128 × 18 shared headers, 512 × 3
typed detail tokens and 2,048 × 2 membership edges (31,744 padded bytes). Initial admission limits are 128
payloads, 512 detail tokens and 2,048 memberships, with explicit errors and no
truncation; these are engineering limits, not complete-game guarantees. Existing
local fact rows keep counters, subtypes, set names, used abilities and copies;
continuous effects no longer expand into that table in v7. V1–v6 encode the
original table unchanged. Native scalar/vector, Python reference, text rendering,
learning vocabulary, model joins and checkpoint admission must agree. Exact
object handles are routing keys, never learned numeric identities. Live data
must carry source/controller authority; historical absence is unavailable and
cannot be guessed from a matching definition.

The model consumes each payload once and sends it through its explicit
memberships. V7 `rows` and `typed_pool` both select the typed sum reducer;
`typed_gru` selects the within-record GRU. They are therefore only two distinct
v7 reducers, not the original three distinct v5 arms. No temporal hidden state
is introduced. Old v5 candidate checkpoint behavior is
frozen to its original version. New schema selection is explicit, with all block
shapes in architecture/config identity. Fresh production defaults stay v6.

**Semantic coalescing is a separate audit.** The bounded delivery uses lossless
factoring only. Intervening grants, different expiry/source control, new affected
sets, later entrants, copies, duplicate memberships and additive P/T/firebending
must retain their distinctions. Repeated removal can be idempotent in a fixed
layer interval, but deleting different timestamps without proving every future
continuation is unsound. No rules-record deletion is authorized by compact
observation storage.

## Completed implementation and remaining acceptance

Implemented in `d5a09a707`, then simplified in `0b1ece0b6`: native shared
projection and ordered expansion, v7 native scalar/in-place/vector/Python
transport, policy routing, text-6 rendering, checkpoint reload and ordinary
CPU learner/export. Retained 140/270-row Command prefixes exercise the new
native authority; the old projected JSON alone cannot supply missing provenance.
Focused debug tests, the root native rebuild and Python checks completed.
Evidence bytes and their manifest are published; README and HTML describe v7.

Remaining for bounded delivery:

1. Gate acceptance of affected suites: complete debug Rust tests and Clippy,
   Python integration including shared-effect transport, model/checkpoint tests,
   active-training and experiment-discovery checks, and the required current-world
   arena/monitoring suite. Engine Budget owns same-runner performance acceptance.
2. Demo/review of the documented contract, then publication and landing through
   the existing Graph. No PR or merge is established by this reconciliation.
3. MTG-213 adoption requires its own prospective protocol/runtime selection;
   unchanged v3 is not a valid retry and this Task does not select its capacity.
   Final provider charges remain distinct from the retained rental estimate.

Replicated learning remains authorized within the separate $25 total allowance,
but unimplemented as a v7 study. It needs newly declared arms, common input,
native counterexamples, training/evaluation schedule and inclusive allocation
before scoring. The original three-arm v5 draft cannot be relabeled as v7.
No strength claim or production default promotion follows from engineering proof.

V7's call into `effect_facts` and Python `push_modification` for active continuous
effects is removed;
local facts and copy exceptions remain. Per-holder active-effect expansion is
only the explicit v1–v6/frozen-reader compatibility projection. V7 model/text
consume the shared authority, not a reconstructed shadow list. Do not alter
canonical `EffectRecords`, frozen decoders or already published experiments.
The global `sim.rollout.OBS_KEYS` was removed: collection, batched sampling and
value scoring now iterate the selected ObservationSpace. Frozen v5 experimental
arms remain explicit v5 inputs; no new scientific comparison replaces their
unlaunched draft protocol yet.

Retired `scratch/representation_probe.py` (synthetic keyword/ragged-wire algebra)
and `scratch/observation-architecture-snapshot.html` (duplicate v6 walkthrough).
Their exact bytes remain at `d5a09a707:scratch/`; native shared-effect rules and
transport tests now own the regression proof, and `docs/observation-architecture.html`
owns the current description. Contrary native timestamp evidence remains below.
No remaining production deletion target: frozen v5 experimental encoders and
their checkpoint/calibration consumers remain supported.

Shared neural record validation, joins and reduction now live in `model/records.py`,
independent of the experimental v5 encoder module.
Native encoding computes timestamp ranks once and passes the same map to object,
local-fact and shared-effect writers, avoiding repeated sorting and legacy scans.

## Retained calibration and cost evidence

The complete earlier plan, schema alternatives and launch chronology are at
`8e6cc62df:scratch/5da5f4-design-effective-state.md`; predecessor context is at
`c823c9c8f:scratch/questions.md`. No uncommitted evidence has been removed.
Model/fixture implementation is in `a8c1cbbff182`; sparse-batch routing and draft
analysis in `7f54fca04`. All runs below are **excluded engineering calibration**.

- L4 source a8c1cbbff182, seed73200, rows/pool/GRU: all one-update CUDA runs
  completed 64 games and exported checkpoints. Collection+learning seconds:
  11.981 / 38.354 / 48.257. Runs 532222ba6ec2 / 7b9fcf7557a3 / dcbd40a43d09.
  Rental representation-5da5f4-6168654d2c3f deleted 02:35:17 UTC; full-window
  SDK estimate **$0.07083924** includes setup/idle. Initial source preparation
  b65b1912e41a failed before a rental; second preparation succeeded.
- The second canary never ran: L4 unavailable; Ada offer disappeared at launch
  5c8cef228edb; RTX3090 offer disappeared at launch4b8bfb5fef1e. No machines or
  jobs were created for these. All quote/launch records remain Task-local in
  `.runs/representation/etude`; no pilot allocation was accessed.
- CPU source8e6cc62df, same excluded seed, 64 streams, one update, one thread,
  4GiB, 300-second limits: rows/pool/GRU completed in21.83/24.16/26.90seconds;
  each completed64games. Receipts `.runs/representation/cpu-canary`.
- Old CUDA GRU checkpoint passed8CPU evaluation games on excluded deals
  83901–83902 in6.29seconds; `.runs/representation/canary-eval`.
- Native fixture receipts `.runs/representation/authority-proof`:36 forward/back
  attempts, no failures; Drill6×10 yields120facts and7×10 rejects140>128.
  Source-duration parity was independently corrected by PR416 too.

No training comparison score or architecture winner exists. No rental or local
training/evaluation process is left running. Paid total is an estimate, not an
invoice. The ancillary storage/rounding reserve is $6: R2 one roundedmillion each
ClassA/B ($4.86), up to10GB-month ($0.15), bounded container storage and margin.
Pricing sources: https://developers.cloudflare.com/r2/pricing/ and
https://docs.runpod.io/pods/pricing (read2026-11-04). Execution plus first month
retention is covered; existing artifact policy owns longer retention. The limits
are 10GB and 100,000 operations/class. The remaining $25 allowance is not a
spending target. Final full-window billing reconciliation remains outstanding;
existing result bytes have been published through `manabot.data`.

## Implementation state and verification boundary

Implemented native pre-admission factoring, scalar/in-place/vector/Python parity,
fixed v7 shapes, complete shared payload encoding and pre-attention membership
routing, text-6 shared rendering, checkpoint admission and ordinary collection /
learner / batched serving. Frozen v1–v6 inputs ignore the new projection. Native
rules storage, defaults and published experiments are unchanged. README and the
HTML now describe actual v7 allocation and model routing.

The chosen CSR layout preserves original edge ordinals and duplicate edges;
ordinals reconstruct native order but are not numeric neural inputs. Timestamp
rank carries semantic order. Sum preserves multiplicity; mean alone would erase
it. Count-aware normalization remains experimental. Summed/hash embeddings may
alias or saturate; transport losslessness is not neural injectivity.

At width64 the illustrated rows model has 1,879,170 parameters, including
799,616 in the shared-effect encoder. The v7 padded effect/local-fact allocation
is 37,888 bytes versus v6/256’s 12,288. The 270-fragment case uses 1,944 active
effect bytes versus 12,960 old active fragment bytes; no padded-memory or CUDA
throughput improvement is claimed. Persistent memory remains deferred below.

Verification receipts for the implementation remain published under
`experiments/data/state-representation-5da5f4/v7-implementation/` via the artifact
manifest. Those receipts bind the implementation's original source hashes;
they do not silently rebind to later simplification. The retained simplification
record at `0b1ece0b6:scratch/5da5f4-design-effective-state.md` reports 78 model/CI
selection passes, 4 shared-effect and 6 encoder debug passes, the root rebuild,
29 transport/parity/text passes, formatting/lint and HTML checks. The shared
record helpers preserve the architecture and numerical operations. These are
focused engineering results, not full gate acceptance or new v7 CUDA evidence.

Reconciliation check: `git diff --check` — passed; documentation-only edits reuse
the retained focused results, with broader acceptance owned by gate.

151 initial artifact entries (about 106 MB) were published, retaining all old
calibration directories, evaluation evidence and a sanitized allocation ledger.
Task-local Etude confirms the sole paid rental deleted and no live machine among
its on-demand supplies. Full-window estimate is $0.07083924286666666; ancillary
storage remains within the separately reserved $6. No MTG-213 budget was used.

Historical design/probe/protocol decisions remain at
`8e6cc62df:scratch/5da5f4-design-effective-state.md`. The original typed v5
experiments are calibration, not v7 evidence. New source-A/B paired native
positions encode identically under v6, differently under v7 and diverge after
source A leaves. The separate pre-existing lost-keyword timestamp counterexample
is retained in questions.md; it blocks semantic coalescing, not lossless transport.

## Unresolved broader design

The accepted bounded repair does not redesign the 64 × 49 relation table:
ordered stack entries, suspended execution/choices, waiting triggers and causal
joins retain their existing shared contract. Separating those structures and
variable-length state encoding remain proposals, not completed or accepted ABIs.
Effective permanent characteristics already reach policy input; dropping raw
provenance in favor of only final characteristics remains unsound for positions
with different future source/expiry behavior. The source-A/B fixture supplies
one such witness, not a complete sufficiency proof for every mechanic.

The remaining design choice is whether replicated learning on the stateless v7
baseline justifies a different reducer or a persistent-memory arm. No evidence
yet selects a winner. Sum scaling, hash collisions/saturation and count-aware
normalization remain learning questions. No additional product decision is
needed for the already approved bounded repair; broader schema or memory
promotion remains a later review decision.

## Concept review — 2026-11-04 UTC

Outcome: **nothing to change and no open product choice for the bounded v7
repair**. Jack Heart's 03:01 UTC approval already settles shared complete
payloads, explicit routing before attention, and lossless factoring without
unproved semantic merging. The implemented concept matches that decision.
This review does not establish gate acceptance, delivery or a learning winner.

The concrete interaction is: select v7 for a prospective training configuration,
play nine Drill activations against fifteen permanents, then request the next
decision. V6/256 rejects the 270 keyword fragments. V7 admits nine complete
payloads, eighteen detail tokens and 135 membership edges, while native rules
still hold 135 records. Each header owns offset/count slices; each edge names
one current object row and its original record ordinal. Identical affected sets
are not separately interned. That optional storage optimization is unnecessary
to the accepted experience. The policy receives the effects through affected
object vectors and retains its existing legal-action focus joins.

The one consequential mismatch was capacity being consumed by repeated
per-holder fragments even though the desired concept was a shared resolved
effect. `shared_effects::project` now derives that representation from
`EffectRecords`; native `encode` and Python `encode_shared_effects` admit its
fixed buffers; `SharedEffectEncoder.forward` routes learned vectors before
object attention. There is no new effect lifecycle or Python rules store.
`Game::characteristics` still owns effective characteristics, and native cleanup,
zone movement and control changes still retire the canonical records. The
within-record GRU resets on each call; it does not introduce restoreable memory.

Failure and recovery preserve the same boundary. Exceeding a header, detail,
membership or remaining local-fact capacity is an explicit failed observation,
never permission to truncate or silently switch schema. Existing game-failure
capture/tolerance owns the failed attempt. A historical capture without shared
provenance is unavailable as a v7 input. A retained Command prefix may be replayed
under explicitly admitted runtime/world identities to produce a new native
observation; that produces prospective evidence, not a rewritten original run.
A saved v7 policy reloads through its own configuration and input binding;
weights alone do not restore an interrupted game. Native clone/restore remains
the owner of game state, and shared observations are derived again from it.

Counterexamples remain decisive: two equal-looking source definitions can bind
different permanents and diverge when one leaves; duplicate additive memberships
must contribute twice; later entrants must not join a fixed affected set; grants
between removals and different expiries cannot be collapsed. The native and
transport tests cover these distinctions, while the retained LoseKeywords
timestamp defect in questions.md prevents treating those tests as a complete
rules-correctness claim. No semantic deletion follows from this review.

Usage wording in `manabot/README.md` and the architecture walkthrough already
states the accepted explicit version, fixed allocations and learned-compression
limits; no usage or skill-instruction redesign is needed. The next proof is the
existing gate's affected suites and Engine Budget. Replicated learning, relation
table redesign and persistent memory remain the separately identified research
work above, not unresolved choices needed to ship this repair. The review page
is `scratch/concept-review.html`; implementation is unchanged by this step.

## Draft history and memory contract

The preferred first arm has **no persistent policy state**. Current active effects,
known-hand minima, deck disclosure and pending obligations survive history eviction
in native state. Optional history keeps its declared chronological window; old
rows disappearing is explicit and does not change current legal decisions.
A within-observation GRU resets each forward call and is not advertised as memory.

A later persistent arm is unimplemented. Its proposed acceptance requires all of
these together:

- Memory belongs to `(match instance, stream generation, viewer seat, policy
  generation)`, not whichever player acted last in a vector slot. Updates ingest
  only that seat's authorized observations/events since its committed cursor,
  including public events between retained decisions. No opponent private frames,
  sampled possible worlds or future trajectory facts enter it.
- New game, reset, rematch or slot reuse starts empty. Ordinary turn boundaries
  preserve memory. Seat swaps route to the corresponding independent state; they
  do not relabel an existing player's private memory. A rejected/stale command
  commits no new policy transition. Native trivial skipping stays on.
- Forks clone memory and cursor at the fork point and evolve independently.
  Replay reproduces the same ordered viewer inputs and model weights. Departed
  object handles never reconnect to returned physical cards.
- Exact restoration stores hidden state, event cursor, viewer/match identity,
  observation schema, model/checkpoint hash and relevant RNG with committed
  learning state. A weights-only checkpoint is not a mid-game memory snapshot.
  If state is absent, exact restore requires replay from the true initial viewer
  prefix with matching weights; a last-32-event burn-in cannot claim equivalence.
  Missing prefix is an explicit unavailable restoration, not a zero-state fallback.
- Training uses contiguous same-seat sequences, episode/padding masks and declared
  gradient truncation/burn-in. Loss uses admitted learning rows only; burn-in
  reconstructs context without introducing opponent rewards. Parameter changes
  require a declared memory refresh policy. Existing seat-routed return/terminal
  rules remain unchanged. Serving and collection call the same transition API.

No recurrent arm proceeds on a mere random-weight forward smoke. Required
lifecycle evidence includes alternating seats, vector reset, invalid command,
fork, resume and history-window eviction. Explicit symbolic known information
stays native-owned; do not build a second Python knowledge store.
