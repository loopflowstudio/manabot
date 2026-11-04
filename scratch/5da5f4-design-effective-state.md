# Effective state and shared effect transport

Implementation plan · Task 5da5f4 · 2026-11-03/04

## Current direction and delivery boundary

Jack Heart's 02:49 UTC steer prioritizes consolidation **before** fact admission
following MTG-213 v3's 270 > 256 failure. The old same-input neural candidates
cannot repair that gate. No scientific seed has run. Suspend the proposed
comparison until the prospective transport is implemented and verified; freeze
new arm/protocol identities before scoring. The $25 Task allowance remains,
separate from MTG-213's $10. Jack Heart’s 03:01 UTC “this sounds good. ship it” approves the bounded v7
repair and supersedes the earlier no-auto-merge restriction. Publication and
landing follow the live code Graph and its delivery/gate boundaries. No change
to fresh defaults or frozen pilot protocols is included.

MTG-213's enabling PR416 is now main `3b122de9078c985f846a0f3c5a3a42ebca3b1182`.
It is merged here through Loopflow sync. Keep its v6 configurable fact capacity,
all v1–v6 layouts, defaults, recipes, receipts and world identities. This repair
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
memberships. Typed pooling and within-record GRU remain alternative reducers;
no temporal hidden state is introduced. Old v5 candidate checkpoint behavior is
frozen to its original version. New schema selection is explicit, with all block
shapes in architecture/config identity. Fresh production defaults stay v6.

**Semantic coalescing is a separate audit.** The bounded delivery uses lossless
factoring only. Intervening grants, different expiry/source control, new affected
sets, later entrants, copies, duplicate memberships and additive P/T/firebending
must retain their distinctions. Repeated removal can be idempotent in a fixed
layer interval, but deleting different timestamps without proving every future
continuation is unsound. No rules-record deletion is authorized by compact
observation storage.

## Implementation and acceptance

1. Add the native shared projection with exact provenance and expansion tests.
2. Route v7 around effect-fragment expansion, including scalar/vector buffers and
   Python parity. Retained 270-row projection is a regression input, not invented
   full native provenance. New Command fixtures supply the native roundtrip proof.
3. Connect shared payload tokens/memberships to the policy and text. Verify
   object addressability, source-A/B distinctions, multiplicity and checkpoint
   admission through ordinary training/export/reload.
4. Run focused debug Rust behavior tests, rebuild the pinned root extension,
   then focused parity/model checks. Gate owns affected suites and full acceptance.
5. Publish exact evidence bytes and document the prerequisite contract for MTG-213.
   Do not retry its unchanged v3 or choose another capacity on its behalf.
6. Deliver the bounded repair through the existing Graph. Replicated learning
   comparisons remain a separately frozen prospective protocol; no strength
   claim or production default promotion follows from the engineering proof.

**Delete — do not maintain:** v7's call into `effect_facts` and Python
`push_modification` for active continuous effects is removed in the same cut;
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
owns the current description. Keep the contrary native timestamp evidence below.
No remaining production deletion target: frozen v5 experimental encoders and
their checkpoint/calibration consumers remain supported.

Shared neural record validation, joins and reduction belong in `model/records.py`;
v7 must not depend on the experimental v5 encoder module for these primitives.
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
invoice. Reserve $6 for ancillary storage/rounding: R2 one roundedmillion each
ClassA/B ($4.86), up to10GB-month ($0.15), bounded container storage and margin.
Pricing sources: https://developers.cloudflare.com/r2/pricing/ and
https://docs.runpod.io/pods/pricing (read2026-11-04). Execution plus first month
retention is covered; existing artifact policy owns longer retention. Stop
before10GB or100,000 operations/class. The remaining $25 allowance is not a
spending target. Record final full-window charges and publish result bytes
through `manabot.data` before delivery.

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
manifest. The simplification preserves the architecture and numerical operations.

- `uv run --extra dev --extra artifacts pytest tests/model/{test_shared_effects,test_state_records}.py tests/infra/{test_ci_selection,test_release}.py -q` → 78 passed.
- `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test --manifest-path managym/Cargo.toml -j 2 -p managym --test rules_tests shared_effects` → 4 passed; the same invocation with `--lib agent::observation_encoder` → 6 passed in debug.
- Root `uv run --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels` → passed; cp312 extension replaced atomically.
- `uv run --extra dev --extra artifacts pytest tests/verify/test_shared_effect_transport.py tests/verify/test_observation_parity.py::{test_v5_source_dependent_duration_matches_native_authority,test_v5_encodes_every_observed_permanent_and_card_field_in_parity,test_versioned_fact_capacity_preserves_provenance_and_rejects_overflow,test_v6_vector_buffers_derive_fact_capacity} tests/semantic/test_text_observation.py::test_continuous_effect_records_are_stated_in_order_with_duration -q` → 29 passed against the rebuilt extension.

Ruff, `cargo fmt --manifest-path managym/Cargo.toml --check`, `git diff --check`,
and HTML source-hash/parameter-total checks pass. Gate owns affected suites,
Clippy, arena/monitoring and CI acceptance; its explicit integration invocation
must include `tests/verify/test_shared_effect_transport.py`. Publication and merge
remain downstream in the live Graph. Context remains below both configured limits.

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

## History and memory contract

The preferred first arm has **no persistent policy state**. Current active effects,
known-hand minima, deck disclosure and pending obligations survive history eviction
in native state. Optional history keeps its declared chronological window; old
rows disappearing is explicit and does not change current legal decisions.
A within-observation GRU resets each forward call and is not advertised as memory.

A later persistent arm requires all of these together:

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
