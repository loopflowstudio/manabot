---
pm:
  provider: linear
  linear_initiative: 2b6f3d99-9176-4b77-af42-710f0f30fde9
  linear_team: 10edaa86-7f64-4996-93b4-22077965e74e
---

# Rules

## Objective

Build small, compiled Magic worlds that are genuinely playable, searchable,
replayable, and explorable in Study. Rules advances by making creator-selected
matchups run end to end through general semantic machinery, measuring their
behavior and cost, and hardening the exact seams that real play exposes. It does
not pursue open-ended card or format breadth.

The product north star is a versioned Avatar Cube Team Sealed world, but the
current two-deck matchup remains the next executable step. Grow toward the cube
through playable content increments; do not replace the current vertical slice
with a speculative 540-card admission program.

Exact object identity, typed card programs, structured legal choices,
proposed-event mutation, viewer-safe state, and safe forks remain foundational.
They earn their complexity by powering a running world; they are not separate
proof ladders that must be completed before a playable vertical slice begins.

## Measures

- A selected Etude Fantasia matchup reaches terminal through compiled typed
  programs and structured `InteractionOffer`/`Command` values as the production
  rules authority, with no card-name dispatch, candidate cap, or client-side
  legality in the exercised path.
- Live play, deterministic replay, Intelligence search, and Study branching
  consume projections of the same authoritative match and reproduce the same
  semantic consequences at shared identities.
- The retained branch representation runs real search and interactive Study
  workloads with exact isolation and return, bounded p95 latency, competitive
  whole-rollout throughput, and measured peak RSS.
- Each creator-selected content increment is admitted as a playable vertical
  slice. The change records terminal traces, actual consumer performance, and
  whether it was content-only or required a reusable kernel change.
- Rules failures discovered through play, search, or Study become focused
  regression, differential, property, or fuzz cases that prevent recurrence;
  diagnostics are tied to an observed system failure or a safety-critical
  invariant.
- Viewer-private information, stale object incarnations, and replacement or
  trigger ordering remain exact across play, replay, search forks, and shared
  Study artifacts.

## Operating loop

Lead with a running slice:

1. compile the smallest selected card or mechanic increment;
2. play it through the real command and event path;
3. exercise it in replay, search, or Study where relevant;
4. measure correctness, latency, throughput, and memory;
5. when behavior is surprising or confounded, construct the smallest oracle,
   conformance case, benchmark cell, or kata that distinguishes the causes;
6. revise the runtime and play it again.

Correctness tests ship with every slice, but broad conformance programs and
representation benchmarks are instruments, not the product. Pre-register a
benchmark when choosing between expensive runtime representations; do not make
the benchmark harness itself the measured bet. Prefer the simplest runtime that
meets the real consumer workload.

The repository should make incomplete semantics fail visibly. Unsupported
content is rejected at compilation or admission rather than silently parsed,
truncated, capped, or routed through card-name exceptions. A readable reference
path remains valuable as an oracle, but production progress is demonstrated by
the optimized path running an authored world.

## Dependencies and bounds

Game owns presentation, replay UX, recovery, packaging, and client adapters.
Rules owns their authoritative frames, offers, commands, events, and state.
Intelligence owns policies and search algorithms but supplies real branching
and throughput workloads. Game owns Study branch navigation but consumes an
exact Rules fork and return provider.

Use immutable versioned `ContentPack` definitions plus compact dense mutable
facts unless a measured consumer demonstrates a better representation. Offline
compilation into checked-in typed IR remains preferred to runtime natural-
language parsing. Open-ended deck building, general format legality, Commander
breadth, and arbitrary-card support remain out of scope; the selected Team
Sealed preset's pool and submission legality are part of its eventual world.

Rules will eventually own versioned cube, sealed-deal, team-submission, deck
legality, matchup-matrix, and series identities. Avatar's initial values—540
cube cards, 135 cards per team, three 40-card-minimum decks, unlimited basic
lands, and a first-to-five three-by-three matrix—are format parameters rather
than engine constants. Do not schedule that orchestration before a complete
selected match runs through the authoritative world.

Concrete repository changes begin as Linear Tasks under a Rules Project. A
Project must culminate in a running world or consumer integration; tests,
contracts, reports, and benchmarks support that outcome rather than replacing
it.
