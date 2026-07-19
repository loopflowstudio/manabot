# Replace the Advice Fixture with the Live Belief-Conditioned Advisor

## Problem

Etude Fantasia already has the two halves of its first testing-house advice
loop, but they are not connected. The live table and Study render the same
`DecisionAdvice` component, yet both bootstrap a completed-match fixture. The
real `AdviceProvider` can resolve a retained decision, normalize explicit
beliefs, run the INT-15 conditional determinized-PUCT advisor, and emit
canonical viewer-safe bytes, but the in-progress prompt has no address and no
live posterior is registered with it.

The missing seam matters to both the pilot and watcher. They must be able to
state a read at the exact decision they are looking at, see advice conditioned
on the actual viewer-relative history, and later reopen that same decision in
Study without changing identities or evidence. Advice is observational:
ordinary revision-bound `Command`s remain the only way to advance the live
match, and exact Study forks remain the only mutable exploratory surface.

This design advances the Game KR that requires an addressable in-progress
decision and a player-authored viewer-safe belief resolved against the live
tracked posterior. It also preserves the wave measure that the same decision,
belief, advisor, compute class, and provenance produce the same advice live or
in Study.

## The demo

Run `./scripts/play`, choose the supported Interactive mirror, and start a
game. At a surfaced player-0 decision, the existing Decision Advice panel
shows the exact current-decision address and two reads, “Has Counterspell” and
“Lacks Counterspell”; switching only the read changes the conditional action
distribution and its value/uncertainty deltas. Commit through the ActionPanel,
open that same address in Study, and receive byte-identical advice while a
concurrent recomputation leaves the live match responsive and unchanged.

## Approach

### 1. Give the unplayed prompt a canonical address

The current `erd1` address cannot name an unplayed decision: it includes the
selected `offer_id`, `command_id`, and a digest of the completed
`ReplayDecision`, all of which are unknown before the pilot acts. Do not invent
a placeholder offer or command and do not alias a live address to a different
replay address.

Add `DecisionAddressV2` with the canonical `ed2.` encoding. Its identity is the
decision, not the eventual choice:

```text
version, replay_id, match_id, ordinal, viewer, revision, prompt_id,
presentation_cursor, frame_hash, decision_sha256
```

`decision_sha256` hashes canonical JSON containing those fields plus the
frame's content and asset-manifest identities. It excludes the played offer and
command. Existing `ReplayDecisionAddress`/`erd1.` remains parseable for frozen
fixtures and recordings; new live sessions and their replay projections use
`ed2.`. Python, Rust, and TypeScript parse the same closed union and reject
non-canonical encodings.

When `_publish_current_prompt()` first surfaces a player-0 prompt,
`GameSession` captures one immutable `PendingCanonicalDecision`: the address,
deep-copied `ExperienceFrame`, viewer presentation cursor, an isolated
`Env.clone_env()` root, source digest, and the next canonical ordinal. Repeated
wire projections reuse it. `_apply_bound_command()` must prove that the
committed `ReplayDecision` regenerates the identical `ed2.` address before it
promotes the captured root into the retained Study roots. A prompt invalidated
without a deliberate command is retired and thereafter fails as stale; it is
never silently rebound.

`AdvisorIdentity.source_replay_sha256` is defined for this path as the
canonical viewer projection prefix ending at the addressed decision facts,
before the selected command. The prefix is captured once with the pending
decision and retained with the committed row, so later replay growth cannot
change the advisor identity.

### 2. Author a viewer-safe condition over one exact tracked posterior

The selected production slice is the Interactive mirror under world `w2`,
where the player-0 possible-world space has both Counterspell conditions. A
local check on seed 197 found 10,832 canonical worlds: 4,820 satisfy
`Has(Counterspell)` and 6,012 satisfy `Lacks(Counterspell)`. Other worlds or a
decision where either condition has empty posterior support report an explicit
unsupported/unavailable state; they never fall back to the fixture.

`TableSnapshot` gains an optional viewer-specific `advice_context` containing
the current address, exact `ViewerIdentity`, dynamically bound
`AdvisorIdentity`, tracker status, and the two server-authored option labels.
The client sends only the returned opaque `belief_option_id` and current
address in `author_belief`; it does not construct a card query or belief
weights.

The server turns that option into a live `BeliefScenario` whose source binds:

- the exact `DecisionAddressV2` and advisor identity;
- the managym-owned typed `WorldQuery` and its query/canonical digests;
- the tracked posterior's world-space, model, checkpoint, viewer-history, and
  posterior digests;
- the author, table revision, and personal/table audience.

The scenario provenance remains `player_authored`; its source separately
carries `ModelInferredBeliefProvenance` for the posterior underneath the
condition. This keeps the player's read, model-inferred range, public facts,
and advisor output visibly and structurally distinct. Private probability
vectors never enter `TableSnapshot` or an advice response.

### 3. Track the live posterior server-side

Create one player-0 `BeliefTracker` for the supported match using the retained
INT-7 world-w2 checkpoint through `FrozenPolicyLikelihood`. The currently
available registration is
`int-7-visit_policy_only-seed-197-1673a237ef2460d0`, SHA-256
`1673a237ef2460d0e699667987c29fe6b42c28711bdb2041989f37692edbd1e6`.
Missing or drifted bytes disable belief/advice only; game launch and Commands
continue.

All live authority transitions, including automatic passes, must produce the
same semantic `TransitionReceipt` path as deliberate play. Before an opponent
commitment, Game captures the isolated likelihood root; after the transition,
it enqueues the after-root and receipt to a single ordered tracker worker. The
table lock never waits for likelihood evaluation. The worker publishes an
immutable posterior snapshot only when its observation revision and viewer
hash equal a registered decision root. Queue overflow, provider gaps, or
checkpoint failure mark later snapshots unavailable rather than stalling the
match or skipping history.

One concrete API gap must be closed without duplicating Rules logic:
`managym/src/possible_worlds.rs::PossibleWorldSpace::condition` already owns
the canonical matching worlds, but
`managym/src/python/bindings.rs::possible_world_support_json` exposes only the
summary receipt. Add the smallest private binding,
`possible_world_condition_json(viewer, space_identity, query_json)`, returning
the matching canonical indexes plus the existing receipt. Mirror it as
`managym.possible_worlds.PossibleWorldSpace.condition_indexes()`. The live
belief resolver multiplies the tracked posterior by that authoritative mask
and normalizes through `BeliefState`; it never re-evaluates card counts in
Etude or the browser.

### 4. Resolve live and Study through one provider

Production `POST /api/advice` requires the participant lease token. Under the
table lock it validates the participant, address, scenario audience and
source, advisor/compute identity, retained tracker snapshot, and exact decision
root; it clones/copies those immutable inputs and releases the lock before any
search begins. The production route never calls
`request_versioned_fixture_advice`. Checked fixtures remain available only
through an explicitly named reference loader/route used by offline tests.

A `TrackedPosteriorBeliefResolver` returns the private conditioned
`BeliefDistributionPayload` with separate base-posterior and authored-condition
receipts. A dynamically registered INT-15 advisor pins the current match,
decision prefix, content, source bundle, paired seed plan, and the declared
512-traversal/16-world compute class. Serving disables exhaustive branch-tape
collection but retains the root digest before/after invariant; the full audit
stays an offline release proof. This is necessary because the retained INT-15
measurement was about 94.7 seconds and 1.77 GB with full auditing, versus
0.74–0.93 seconds and about 304 MB without it.

Advice executes in one bounded background lane with exact-request in-flight
deduplication and canonical-byte caching. A saturated lane fails visibly
instead of building an unbounded queue. Recompute uses a fresh isolated clone
in the same lane. Neither path holds `SessionRecord.lock`, touches
`GameSession.env`, changes a revision, appends a trace event, or writes a
belief/table record.

Before the pilot commits, `LiveAdvisorDecisionResolver` resolves the pending
snapshot. After commit, `StudyAdvisorDecisionResolver` resolves the promoted
retained root. Both return the same frame, root state, decision prefix digest,
world space, belief snapshot, and identity. Both call the same
`AdviceProvider` and `serialize_advice_response`; mode and wall-clock data are
absent from the envelope. Fresh recomputation through each resolver must be
byte-identical, not merely structurally equal or satisfied by a shared cache.

### 5. Reuse the existing surface and keep mutation boundaries obvious

Update the TypeScript `advice-v1` twin and add a presentation adapter from
`AdvisorStrategyEvidence` to the existing `DecisionAdvice` regions. Keep the
component, placement, scenario radio interaction, action rows, values,
uncertainty, and deltas. Add only the provenance labels needed to distinguish
“Your read”, “Tracked posterior”, and “Advisor”. No second live/Study component
or client belief engine is introduced.

The client keys every request by address and request hash and ignores a late
response if the live current address changed. Unavailable responses clear all
prior evidence before rendering the reason. `DecisionAdvice` remains
advisory; it has no command callback. The ActionPanel remains the sole live
Command path, pilot capability checks remain server-authoritative, and watcher
or advice requests cannot enter match mutation methods.

## De-risking

| Question | Finding | Impact on design |
|----------|---------|-----------------|
| Can the existing replay address name the unplayed live prompt? | No. `ReplayDecisionAddress` includes the eventual offer, command, and full played-row digest; `GameSession._step_and_record()` creates it only while applying that command. | Add prompt-level `DecisionAddressV2`; prove it is unchanged when the played row is committed. |
| Can the tracked posterior support both authored Counterspell reads in the selected world? | Yes. Interactive mirror seed 197 exposes 10,832 worlds, split 4,820 Has and 6,012 Lacks, and initializes against the retained world-w2 checkpoint. | Admit Interactive mirror first and fail closed elsewhere. |
| Can Etude condition the posterior without duplicating Rules? | Not with the current binding. Rust `PossibleWorldSpace::condition` returns matching worlds, but Python exposes only `possible_world_support_json`. | Add one authority-private matching-index binding; no Etude/client hand filtering. |
| Is the likelihood checkpoint present and identity-pinned? | Yes in this checkout: INT-7 policy-only seed 197, SHA-256 `1673a237...bd1e6`, loads through `FrozenPolicyLikelihood`. INT-9's broader likelihood contract remains unresolved, so this is a narrow supported slice, not a general calibration claim. | Pin the exact retained registration and report missing/mismatch; do not claim calibrated general belief quality. |
| Can INT-15 search run inline with the WebSocket authority loop? | No. Full branch audit measured ~94.7 s/1.77 GB; even the ~0.8 s serving profile would block an inline async handler. | Clone under lock, search in a bounded background lane, and keep the exhaustive audit offline. |
| Can live and Study emit exact same bytes? | Yes. The provider already has one canonical serializer and separate live/Study decision resolvers; the missing invariant is a stable pre-command address/source prefix and retained posterior snapshot. | Bind both resolvers to the same captured decision record and test fresh recomputation with separate providers. |
| Does the current frontend consume versioned provider evidence? | No. It posts the legacy fixture request and `DecisionAdvice` reads legacy `DecisionEvidence`, even though the backend has `AdvisorStrategyEvidence`. | Add a TypeScript contract twin and adapter while retaining the component and interaction design. |
| Does the current REST advice route enforce table viewer identity? | No. It trusts the request body and has no participant token. | Require the participant lease for production advice and validate personal/table audience server-side. |

## Fail-closed behavior

Every unavailable response contains the exact requested address and identity,
no frame/evidence/strategy/deltas, and one typed reason. The UI clears stale
evidence and renders a readable reason.

| Condition | Result |
|-----------|--------|
| Tracker or evaluator checkpoint missing | `belief_artifact_unavailable` or `advisor_artifact_unavailable` |
| Checkpoint/source bytes drift | `belief_artifact_mismatch` or `advisor_artifact_mismatch` |
| Matchup/world not admitted | `world_unsupported` |
| Query not admitted or has zero tracked mass | `belief_query_unsupported` or `belief_invalid` |
| Tracker has not reached the decision or lost a transition | `belief_distribution_unavailable` |
| Retired/tampered address, frame, root, or source prefix | `decision_identity_mismatch` or `decision_root_unavailable` |
| Participant, rules viewer, author, or audience differs | `belief_viewer_mismatch` |
| Belief source points at another decision/advisor | `belief_decision_mismatch` or `belief_provenance_mismatch` |
| Planner, evaluator, compute, seed, ABI, or content differs | Existing exact `*_mismatch` reason, including `compute_mismatch` |
| Advice lane saturated | `advisor_busy` |
| Search produces partial/private/unaligned evidence | `private_projection_failure`, `action_identity_mismatch`, or `policy_mass_invalid` |

No case substitutes the fixture, compatible prior, another checkpoint, a
nearby decision, a default value, or partial rows.

## Alternatives considered

| Approach | Tradeoff | Why not |
|----------|----------|---------|
| Preselect a placeholder offer/command to manufacture an `erd1` address | Small local change. | The address would change when the pilot chooses a different offer, so live and Study would not name the same decision. |
| Map current prompts to the closest checked fixture | Reuses the existing UI and endpoint. | It fabricates identity/evidence, ignores live history, and violates the explicit no-fallback directive. |
| Compute a compatible prior or query mask in the client | Avoids a server resolver. | Duplicates Rules and belief authority and exposes private weights/hand ontology. |
| Run tracking and advice synchronously under the table lock | Simplest ordering. | A slow checkpoint or search would pause Commands, watcher broadcast, and reconnect. |
| Build a second live-advice component/API | Avoids adapting legacy frontend types. | Creates the live/Study fork the task is intended to remove. |

## Key decisions

- Address the prompt independently of the eventual command; preserve `erd1`
  only as a frozen compatibility format.
- Admit one honest Interactive-mirror/Counterspell slice and expose unsupported
  worlds instead of pretending the provider is general.
- Treat the player-authored query and model-inferred posterior as two distinct
  provenance layers in one `BeliefScenario`.
- Keep tracking ordered but asynchronous; loss of evidence degrades advice,
  never match authority.
- Use one canonical response serializer and prove fresh live/Study parity
  before relying on cache.
- Keep exhaustive branch auditing as release evidence, not a per-click serving
  cost; serving still checks source-root immutability.
- Authenticate advice with the same participant lease as the shared table.

Wild success is that pilot and watcher naturally save competing reads during
play, share the useful ones, and later reopen the exact moment without noticing
a mode boundary. The surprising win is that a slow recomputation can finish
after play advances and still be a valid artifact for that recorded decision.

Wild failure is a nominally “live” panel backed by a stale fixture, an address
that changes after the command, or a search job coupled to the match lock. Any
of those destroys trust: the numbers may look polished while referring to a
different decision or making the table wait. The identity, concurrency, and
empty-evidence tests below are therefore release gates, not polish.

## Scope

- In scope: `DecisionAddressV2` across Rust/Python/TypeScript; pending-decision
  capture and promotion; one Interactive-mirror player-0 tracker; the private
  managym conditioning-index binding; live authored Has/Lacks Counterspell
  scenarios; authenticated background advice; versioned frontend adaptation;
  explicit fixture reference access; live/Study parity and non-mutation tests.
- Out of scope: arbitrary natural-language ranges, every deck/world, watcher
  perspectives other than the shared player-0 view, calibrated belief-quality
  claims, a new planner, chat, fork/Retry/return redesign, client-side rules,
  or changing ActionPanel Command authority.

## Done when

The thin end-to-end proof is one deterministic Interactive-mirror decision and
one real browser flow:

1. `cargo test` passes in debug, including Rust/Python address parity and the
   new authority-private query-conditioning binding.
2. `uv run pytest tests/etude/test_live_advice.py tests/etude/test_testing_house.py`
   proves the surfaced address equals the later replay/Study address; Has and
   Lacks requests differ only in `BeliefScenario`, bind the tracked posterior,
   and produce non-identical strategy; fresh live and Study providers return
   identical canonical bytes; every listed mismatch returns no evidence.
3. The same test starts a recomputation, advances 20 ordinary Commands, and
   proves match digest/revision/trace changes are attributable only to those
   Commands, command P95 stays at or below 100 ms, and pilot/watcher broadcast
   lag remains at most one update.
4. `cd frontend && npm test && npm run check && npm run build` passes with the
   versioned advice adapter and stale-response clearing tests.
5. `cd frontend && npm run test:e2e -- testing-house-advice.spec.ts` drives the
   real stack: author each read, observe distribution/value/uncertainty deltas,
   commit through ActionPanel, reopen the identical address in Study, compare
   response SHA-256 values, and verify watcher/advice controls cannot mutate
   the rendered or server authority state.
6. Existing checked advice fixtures and their schemas remain byte-for-byte
   reference artifacts, but the production live flow contains no fixture
   request or fallback.

## Measure

- Baseline authority latency is the GAM-8 20-action P95 of 41 ms with maximum
  broadcast lag 1. Under concurrent advice, require P95 <= 100 ms and the same
  lag bound.
- Fresh no-audit advice on the selected decision must complete at P95 <= 2 s
  and peak process RSS <= 512 MiB on the retained measurement profile; an exact
  cache hit must complete <= 50 ms.
- The release receipt records request/response SHA-256, live/Study byte parity,
  posterior/history/checkpoint identities, nonzero scenario delta, root
  before/after equality, command latency, and broadcast lag. Runtime timings
  remain outside deterministic advice bytes.
