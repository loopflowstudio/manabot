# Engine-owned choices

Etude Fantasia and manabot share managym's `InteractionOffer` meanings. The
engine supplies legal alternatives and constructive continuation support;
models score those alternatives. Labels remain presentation. Acting-player
choices may contain owner-private information and are not a spectator payload.

Semantic schema 7 and world w5 add fixed offer parameters and role contexts.
`structured_search_offers()` remains one offer per canonical Action;
`compound_offers()` groups only certified independent choices. Both project
source and parameter relationships from the same native actions. Blocker roles
identify their actual blocker, including distinct same-name objects. Target
roles identify a requirement in a hashed public program. Modes and abilities
bind their ordinal to program bytes. Learn uses outside-candidate addresses,
never fabricated in-game ObjectRefs. Payment contexts carry remaining native
mana requirements and tap units; consumers do not calculate affordability.

`managym.choice.OfferProjection.from_json` parses the shared records into typed
immutable Python values after native validation. `ChoiceSupport` retains the
certified independent support for scoring without a live match. Native
`Env.compound_prefix_support(offers, prefix)` additionally checks the live root.
The wire projection binds schema, revision and factorization order 1.
The semantic-policy adapter retains all native fields for scoring and execution;
its protocol-v1 Etude presentation omits `details` and choice `context`, which that
older DTO does not admit. This does not change the presentation protocol.

The first token selects an offer. Subsequent bits include or exclude candidates
in native role/candidate order. Every allowed prefix has a completion; every
subset has exactly one tape. Native support rejects impossible, trailing and
unsupported prefixes without enumerating complete declarations. Final native
lowering still validates the answer and emits revision-bound Commands.

## Policy and search

`AgentSpec(compound_decisions=True, compound_features="objects")` selects the
object decoder. The shared padded object map joins sources, role subjects,
objects, players, stack spell cards and outside candidates to the existing
viewer encoder. Missing rows and capacity overflow fail before inference.
Addresses choose rows; they are not learned embeddings. The existing GRU scores
native verb symbols, public numeric parameters and those encoded objects,
conditioned on its prefix. Presentation text does not enter this feature path.

The default `compound_features="labels"` preserves the admitted label decoder's
parameter meanings and initialization. Flat focus tensors retain their existing
layout. All decoders now obtain subset support from managym. The new object
architecture has additional saved parameters and requires its explicit metadata.
Historical w4 artifacts require their original bindings/runtime; no loader check
is relaxed and no historical checkpoint or measurement is relabeled w5.

The sampling likelihood is the product of conditional probabilities along the
actual tape. Forced factors have zero actor log probability. `tape=` rescoring
checks the retained viewer-root/projection fingerprint; `tokens=` can score a
newly authored complete answer. Collection probabilities remain frozen while
rescoring uses current weights. Path entropy and retained-prefix reverse-KL sums
are estimators, not exact joint entropy or reverse KL.

Local search asks native support for disjoint prefixes corresponding to the next
Command and sums only their newly fixed conditional log probabilities. Unchosen
suffixes integrate to one. Skipped waterbend taps retain exact zero policy mass,
even when still engine-legal. ETU-100's v3 positive-base-support update, zero
counts/targets and unavailable unvisited Q remain unchanged.

## Execution boundaries

One runner owns one suffix. Reset, another engine/branch or a replaced policy
clears it; an unexpected revision or absent Command clears it and reports an
interruption. Earlier accepted Commands remain recorded by GameSession/Trace.
This does not promise crash atomicity across a declaration.

Atomic attackers, independent blockers, eligible single-target casts and
fixed-economics waterbend subsets group. Menace stays sequential. Waterbend with
competing mana abilities or relevant triggers stays sequential. Kicker, ward,
reveals, resolving modes and other unsupported groupings use ordinary atomic
offers and fresh observations. Same actor or prompt kind alone does not certify
a grouping. In particular, completing an attacker declaration can automatically
skip into the next actor's combat; lowering stops after the original declaration's
roles rather than following another prompt of the same kind. Compound EMA, belief-input policies and unsupported objective/value
combinations keep their existing explicit rejections.

## Bounded software workflow

```bash
uv run --extra notebook python -m experiments.runners.choice_contract \
  --out .runs/choice-workflow
```

The runner resolves four current-world object-decoder recipes from the existing
compound credit/target fixtures, uses one CPU thread and a 900-second process
ceiling, exports raw checkpoints and reuses the ordinary checkpoint arena and
exact replay/report path. Retain failed attempt directories. This is a software
fixture, not a training campaign, strength result or default-adoption decision.

The new conformance corpus is `conformance/semantic-kernel-w5-v1`: four games,
567 Commands, with prior corpora unchanged. It retains the source-main-deck,
empty-sideboard scope; the workflow above uses authored sideboards separately.

Component latency and full-game throughput can be measured separately:

```bash
uv run python -m experiments.runners.choice_latency \
  --out .runs/choice-latency.json
```

This uses 3/35/65 attacker candidates, two warmups and nine timing samples per
component, with separate tracemalloc peaks (Python allocations only). The
complete-game control is the current label decoder, not the pre-migration native
runtime. Do not infer a native-support speedup or strategic strength from it.
The object decoder remains opt-in; a median complete-player slowdown over 20%
requires review before adopting it by default.

## Retained evidence — 2026-10-06

The third one-thread workflow completed in 199.33 seconds: four two-game training
arms, eight admitted raw checkpoints and 56 exact-replayed arena games over 14
cells, including both deck/seat assignments. No cells failed. Receipts, exact
artifact identities, costs and the report are retained in
`.runs/etu107-object-workflow-3` in the Task checkout.

Both earlier attempts remain retained. Attempt 1 exported all arms but failed
arena registration because its world enum stopped at w4 (68.24 seconds). Attempt 2
exported all arms but failed two arena games when attacker lowering followed
automatic steps into another actor's declaration (277.71 seconds). The native fix
bounds lowering by the original role list; a directed debug regression reproduces
that transition. Those draft-runtime artifacts are not final-runtime compatibility
evidence. Costs and failures were not replaced by the successful attempt.

The focused Python suite passed 94 checks, including typed object joins,
label-independent verb meaning, sampling/rescoring, wide offers, hidden-world
invariance, exact retained-prefix payment zeros and ordinary export/reload.
Sixteen native debug structured checks passed; the separate w5 conformance corpus
records 567 Commands. Final gate/CI verification remains separate. This one-seed
fixture establishes software integration, not strength, calibration, independent
training-seed uncertainty or chapter acceptance.

Offline regeneration preserved identical `report.md`, `metrics.json` and
`cost-comparison.json` bytes. An additional 12 focused label/object local-search
checks passed, covering attacker/cast/blocker prefix normalization and payment
joint parity with exact zeros.

The retained timing run `.runs/etu107-choice-latency-2.json` completed in 1.58
seconds. Times below are median / nearest-rank p95 milliseconds; complete-player
Python peaks come from a separate traced call. The JSON also retains each
component's Python peak. Native/Torch allocations are excluded.

| Decoder | Candidates | Projection | Encoding | Decoding | Lowering | Complete player | Python peak, complete player |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| labels | 3 | 0.002 / 0.002 | 0.144 / 0.167 | 0.851 / 0.902 | 0.036 / 0.042 | 1.141 / 1.563 | 79,300 B |
| labels | 35 | 0.008 / 0.008 | 0.798 / 0.996 | 3.381 / 3.664 | 0.496 / 0.588 | 4.995 / 5.389 | 204,433 B |
| labels | 65 | 0.013 / 0.014 | 1.489 / 1.666 | 5.323 / 5.687 | 1.096 / 1.177 | 8.260 / 8.477 | 371,469 B |
| objects | 3 | 0.002 / 0.002 | 0.139 / 0.227 | 0.899 / 1.093 | 0.032 / 0.034 | 1.099 / 1.246 | 79,224 B |
| objects | 35 | 0.007 / 0.007 | 0.738 / 0.865 | 3.100 / 3.361 | 0.437 / 0.518 | 4.718 / 5.079 | 210,753 B |
| objects | 65 | 0.012 / 0.012 | 1.338 / 1.502 | 5.051 / 5.295 | 1.021 / 1.094 | 7.942 / 8.299 | 383,701 B |

Across three tiny complete games per decoder, median throughput was 856 Commands/s
for labels and 909 Commands/s for objects; median game times were 0.145 and 0.112
seconds. The untrained policies follow different trajectories, timing order is
fixed, and nine samples are a small overhead screen. No >20% median complete-player
slowdown appeared in these roots, but this does not establish a general speedup
or authorize default adoption. The first timing attempt failed in 0.21 seconds
because its 65-object fixture supplied only 20 creatures; its receipt remains in
`.runs/etu107-choice-latency-1.json`. The fixture now supplies enough native cards.
