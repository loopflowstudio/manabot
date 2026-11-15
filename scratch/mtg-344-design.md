# One policy token per game thing (MTG-344)

Status: one-token layout and the reviewed characteristics correction are on PR #521, synced onto main 83406110db (merge 84f61948c5, pushed). Next: gate (native rebuild, CI), then demo/review. Evidence: `scratch/encoding-study.md`
(read-only study of origin/main 452d7b5357; `file:line` citations there).

## Direction

Jack Heart, 2026-11-14, in the MTG-286 operator conversation:

- "it looks to me like those objects really should be cards"
- "we already have permanents and stack_objects"
- On making the card list hold only cards and moving battlefield and stack
  characteristics onto permanents and stack objects: "I think this is the
  right thing", "lets turn 521 into that"
- "lets think about how the tensors get used in the neural net. what setup is
  most beneficial for learning"

This replaces the earlier MTG-344 scope (rename `agent_cards` to
`agent_objects` with deprecated aliases). Once the list holds only cards,
`agent_cards`, `opponent_cards` and the Python `Card` record keep their names;
the rename and its aliases are dropped.

## Problem

The policy sees one battlefield creature as two attention tokens and a spell
on the stack as a card row, while stack abilities are no token at all.

| Today | Consequence for learning |
|---|---|
| A permanent has a card row (zone=battlefield) and a permanent row, with different public ids and no column joining them; only list order aligns them | The model must learn to pair two tokens, and cannot pair them reliably when two copies of one definition are on the battlefield |
| Both rows add the same catalog program embedding (`semantic_cards` and permanent `DEFINITION`) | Duplicate signal; attention mass split across two tokens for one thing |
| Types, legendary, token, Ally/Lesson, ward and mana value exist only on the card row; tapped, damage, counters, combat, attachments only on the permanent row | No single token holds the whole creature |
| Attack, block, activate and target-a-permanent focus the permanent row; sacrifice, crew and tap payments, ChooseObject, ChooseToExile, KeepCreatures, DistributeCounters, and PayCost/ChooseMode/Decline for a permanent's ability focus the card row | Actions on one object read different tokens depending on the action family |
| History v1 contexts resolve only to card rows | "This creature was dealt damage" lands on the token that does not hold damage |
| Stack abilities live only in `relations` headers, pooled into a shared context; targeting one focuses -1 | Responses to abilities and "change the target of target spell or ability" (TLA) read zeros |
| `max_cards_per_player` counts battlefield and stack rows | Card capacity is spent on duplicates (TLA needed 128) |

## Target

Every game thing the agent may see is exactly one attention token, and every
pointer (action focus, link, fact, effect member, relation reference, history
context) names that token.

| Segment | One row is | Holds |
|---|---|---|
| players | a player | unchanged |
| cards (per side) | a card outside the battlefield and stack: hand, visible library, revealed and searched cards, graveyard, exile, pending public entrants in their current zone, the agent's sideboard | unchanged columns; zone bits keep `battlefield` and `stack` at zero |
| permanents (per side, by controller) | a permanent | today's state columns plus the characteristics now only on the battlefield card row: six card types, legendary, token, Ally, Lesson, has_ward, ward cost, mana value; catalog program via `DEFINITION` as today |
| stack (new, one shared segment, top first) | a spell, activated ability or triggered ability on the stack | kind, is_mine (controller), definition (spell's card, or the ability source's definition), ability index, mode, X, kicked, copy flag, spell types and mana value, granted keywords; catalog program via definition |

Native `Observation` follows the same shape: `agent_cards`/`opponent_cards`
lose battlefield and stack entries; `PermanentData` gains the characteristic
fields (and `text_box`, so readers stop pairing by position);
`StackObjectData` gains the spell's characteristics and is encoded as the
stack segment. The engine's object model is untouched.

### Pointers

| Reference | Points at |
|---|---|
| Every action naming a battlefield object (including the `SelectCard` families and permanent-source PayCost/ChooseMode/Decline) | permanent row |
| CastSpell's later choices (PayCost, ChooseMode, Decline for the spell being cast), ChooseTarget of a spell or an ability, OrderReplacement for a stack source | stack row |
| Hand, graveyard, exile, library, sideboard actions | card row |
| `relations` headers | a stack header's SOURCE becomes its stack row; ability headers keep a separate source-permanent reference |
| History v1 contexts | card, permanent or stack row by where the object is |

No focus may be -1 for an object the agent can see; the encoder treats a
visible object with no row as an error (debug assert plus a test sweep over
the curated decks).

### Model

`manabot/backends/torch/policy.py`: a stack projection
beside the card and permanent projections, adding the catalog program by
definition; the stack segment joins the attention set and the ownership
template; `_join_permanent_links` and other hard-coded offsets read segment
starts from the layout. Card and permanent projections widen with their
columns. No other architecture change, so a comparison measures the
representation.

## Compatibility and the world

Permanent rows widen and the stack segment is a new buffer, so the world
binding refuses older checkpoints by buffer shape; nothing loads silently.
The authorized exception below preserves the deprecated v9/v10 fork while
fresh configurations select v11. The representation changes once, in one
layout, rather than in stages that repeatedly obsolete checkpoints.

`max_cards_per_player` keeps its default; the comparison reports peak card,
permanent and stack occupancy on the curated decks and TLA pools so defaults
can shrink with evidence. Stack capacity default: 32, typed Capacity error
beyond it.

## What else changes

| Reader | Change |
|---|---|
| Native and Python encoders, layout, `observability.rs` guards | new columns and segment in parity; guards for AddTypes/SetTypes, Nonlegendary, Ward, stack X move to permanent and stack columns |
| Native heuristic (`heuristic/mod.rs:72-125`) | read types and mana value from permanent rows instead of searching card rows by registry key |
| Rewards (`manabot/env/match.py:145-193`) | count lands and creatures from permanent rows |
| LLM text observation | permanent and stack lines carry types, cost and rules text; no card-label borrowing |
| Learning tokens projection | preserve frozen card roles and v9/v10 order for permanents and spells |
| Academy experience projection and server | read names, types, text and token status from permanents and stack objects; delete positional pairing |
| History v1, decision capture, relations | resolve contexts and sources into all three segments |
| Belief audit, competency, conformance parity, benches | check; most read hands |

## Measurement

- Exploratory comparison (CLAUDE.md): current layout vs this layout on the
  current recipe (Elemental, 10v10), matched updates, two seeds per arm,
  head-to-head plus the standard anchors; under the launcher's USD25 cap.
- Native encode time per observation and policy forward time at default
  capacities, both layouts, same machine.
- Engine Budget is unaffected (no engine change) and must stay green.

## Decisions (Jack Heart, 2026-11-14)

Verbatim: "budget is $25. Freezing is fine. Just do it once, not staged one
shared stack segment is fine. I don't know. Do whatever you think makes sense
in terms of the freeze making things unusable, can we use our depertication
technology to basically fork the code, have the new version, and have it be
keyed by world or something like that? Then mark it. I guess this really goes
into the task that we have about portability across checkpoints or across
worlds. Anyway I think we can just copy the code and fork it as we need to and
put it behind whatever kind of bridge we need. Just leave it for deletion
later but let's move forward on testing the new paradigm"

Resolved:
1. One layout change, not staged.
2. Freeze is fine. Existing checkpoints stay usable by forking: copy the
   current encoders (native and Python) and policy model input path into a
   frozen legacy module, selected by the checkpoint's bound world/input
   layout; the new layout is the fresh default. Mark the legacy fork
   deprecated with a stated removal release and leave it for deletion later.
   This is an explicit exception to the "no other version branch" rule for
   this freeze; the general mechanism belongs to MTG-353 (checkpoint
   portability across worlds).
3. One shared stack segment with a controller flag.
4. Order against MTG-341: proceed independently; this change reads public
   ids only. Whichever lands second rebases.
5. Comparison budget: USD25 total under the launcher's cap.

## One-token implementation

- Input v11 (`InputLayout::OneToken`, fresh default) carries v10 turn state,
  51-column permanent rows, and `stack` [32, 37] plus `stack_valid`. Object
  order is players, each side's cards, each side's permanents, then stack.
  Object ids alias their permanent/stack row for focus, history and joins.
- The authorized v9/v10 fork keeps only the differences in `legacy_input`
  modules: old card lists, permanent width, no stack aliases and card-only
  history. Facts, effects, relations and actions share the surviving encoder.
  Removal release: 0.4.0; MTG-353 owns broader portability.
- Torch attends stack rows with catalog programs and controller ownership.
  Since the sync, v11 stack features, routes, ownership and catalog
  references are built once in main's shared host preparation
  (`manabot/policy/prepare.py`, `Prepared.stack*`); Torch consumes them. MLX
  still admits v9 only (`manabot/backends/mlx/checkpoint.py:require_supported`),
  so a fresh default config does not run on MLX until it is ported.
- The characteristics correction below replaces the initial nested `.card`
  implementation; older implementation detail remains in HEAD `72a7b4df80`.

### Legacy fork identity (2026-11-14)

Native v9, v10 and v10 + history v1 encodings (C=128), every buffer, over
12,403 observations from 64 seeded random curated games (≤600 steps): SHA-256
at origin/main 452d7b5357 and on this branch with `InputLayout::Legacy`
are identical — v9 `316c3c20…`, v10 `aa8d499b…`, v10h `fccaf565…`; 0 errors.
Tool: a throwaway `examples/legacy_digest.rs` built in both trees (not kept).

### Costs measured (2026-11-14, laptop, M-series)

- Native `encode_into`, 6,935 curated observations, C=128, release, median /
  p90 µs: origin/main v10 5.88 / 8.42; v11 6.29 / 8.92; legacy fork v10
  13.04 / 20.29 (`legacy_input::view` clones the observation).
- Python-side `encode_observation_into` ≈109 µs for both layouts (PyO3
  conversion dominates). CPU torch forward, recipe model, batch 256, 4
  threads: v10 95.7 ms, v11 86.9 ms (10 repeats; noisy).
- Elemental rows over 8,507 observations (40 random games): per-side card
  rows p99 43 → 31 (max 48 → 44, sideboard dominated); permanents max 25;
  stack rows p99 1 → 3 (abilities now rows), max 5.

## Concept review (2026-11-14)

Jack Heart, after the observation that token permanents expose `permanent.card`
and cross-layout checkpoints cannot meet in one arena game: "This is also worth
fixing". On cross-layout matches: "i dont think thats important". On the wider
follow-up: "i do think we should launc a followup taks, which i tihink is already
defined, which is thinking through world and observation format versioning and
compatability".

### Proposed correction: characteristics are not a card

The tensor achieves the intended single token, but the published observation
still makes a token permanent or copied spell a `Card` inside `.card`.
`card_records()` consequently includes things that are not cards. This moves
the original naming problem into a nested record and duplicates permanent P/T
and keywords across the outer and inner records.

Use a shared observation characteristics record for name, rules text, types,
mana cost, effective P/T, keywords and ward. Cards, permanents and spells own
that record; object identity, token/copy status, ownership, zone and paid costs
belong to their enclosing object. Stack abilities keep their own kind, source
definition and source reference; they never acquire fictitious spell
characteristics. Do not confuse this viewer projection with the engine's full
`Characteristics` type or promise dynamic rules text.

Proposed usage-doc excerpt:

```python
permanent = obs.agent_permanents[0]
name = permanent.characteristics.name
power = permanent.characteristics.power
is_token = permanent.is_token
# A spell owns spell characteristics; an ability owns its source reference.
```

One layered view supplies the characteristics. Every visible live permanent
has them; historical JSON is normalized at the read boundary or explicitly
refused when its identity cannot be reconstructed. Existing published names
need warning aliases with a stated removal release; aliases must derive from
the same record rather than introduce dual writes. Preserve existing frozen
bytes; never rewrite receipts to mask this API correction. Tensor keys,
v11 layout, catalog programs and deprecated v9/v10 encodings remain unchanged.

Concrete walkthrough: inspect a token, select it to pay a sacrifice cost, and
resolve the resulting ability. Inspection, focus and history identify the one
permanent row. After the source leaves, the ability keeps its captured source
definition and effects; a returned object never reclaims the departed identity.
A stale Command commits nothing. Reconnect reconstructs the pending native
decision and its viewer-safe projection, rather than recovering a browser's
former card/permanent pairing.

Counterexamples to preserve: two identical token definitions with different
counters; a copied spell with distinct X; a controlled permanent owned by the
other seat; an ability whose source has left. A definition key alone cannot
join these objects. Historical stack reconstruction currently matches by
definition key; any ambiguity must be exposed instead of assigning one copy's
identity or X to another.

### Follow-up boundary and evidence

MTG-353 already covers checkpoint portability when cards are added. Broaden its
design to distinguish rules/world identity, observation format, policy tensor
layout, vocabulary identity, exact replay and adopted play, including the
v9/v10 fork's removal. Cross-layout arena matches are not required by MTG-344
and are not a priority for that follow-up unless its design establishes a need.

The exploratory comparison in `experiments/one_token_layout.md` does not
establish a learning improvement or equivalence: two training seeds per arm,
heuristic evaluation only, and unequal final update counts. At matched update
75 the reported scores are v10 0.385 and v11 0.398; reported game standard
errors do not measure training-seed uncertainty. Performance receipts support
similar learner throughput, modestly slower native encoding and a noisy CPU
forward speedup. Keep learning benefit explicitly unproven.

Next demonstration: the token-sacrifice walkthrough above, a same-definition
copy identity case, and old-checkpoint loading without altered weights or
evidence. No new agent skill behavior is needed; existing shared decision
authority and deprecation guidance still applies.

### Second concept pass: one effective view (2026-11-14)

The correction must remove competing meanings, not merely rename `.card`.
`Permanent.name` currently means a name override (`None` while printed applies),
while the nested card's name is effective. Keep the effective name in the
characteristics record; any retained override getter must keep its old meaning.
Likewise, P/T and keywords have one stored source, with old getters derived
from it. Token/copy status and object ids are not copiable characteristics.

Academy's `base_power`/`base_toughness` currently repeat effective P/T, not
printed or layer-7 base values. No production UI reader was found in this pass;
do not invent printed values or subtract counters to populate these fields.
Keep any retained serialized compatibility fields at their existing meaning;
new usage documentation and readers consume explicitly effective P/T.

Concrete proof to show: a printed 1/1 token with a +1/+1 counter reads effective
2/2 from the one characteristics record, has token status on the permanent,
and selection/history still address its single row. Sacrificing it leaves the
stacked ability intact; reconnect retains the native pending decision. A second
identical token without the counter remains a separate 1/1 with a distinct id.

For the initial API correction, preserve current information and both encoder
layouts byte for byte. Adding genuinely missing observable state is a separate
layout decision. The proposed naming clean-up alone is not evidence for
additional learnability or performance claims. Deprecations use the repository's
existing release policy; this review sets no new removal date.

Review outcome: the working design changed to require one authoritative
effective characteristics projection, derived compatibility getters, and
explicit separation of effective values from override/base labels. No new
product choice is needed to make that correction; no code or Graph was changed.

Jack Heart, after this review: "Sounds good. Fix it and then we'll come back".
Implementation scope is the characteristics correction above, preserving
tensor and legacy encoding contracts. Complete focused validation and publish
the correction to PR #521, then stop at the interactive review/demo. This
direction does not request merging or further training experiments. MTG-353
continues the separate compatibility design; mixed-layout matches stay out.

## Characteristics correction implementation

Delete — do not maintain (completed in this correction):
- Canonical permanent/stack `card` records and duplicated permanent P/T and
  keywords. Keep old Python accessors as derived warning aliases, removed in
  0.4.0 alongside this Task's existing legacy fork.
- Production card/permanent pairing and definition-only historical spell
  matching. Normalize unambiguous historical records at the JSON boundary;
  refuse missing or ambiguous identity.

Preserve schema-8 JSON spelling, omissions and bytes in boundary-only wire
records; preserve catalog programs, v11 buffers and deprecated v9/v10 rows.
Canonical live objects own one `CharacteristicsData`; object identity and
copy/token status remain on their enclosing records. Abilities own no spell
characteristics. `Permanent.name` retains its override meaning.

Validation: 8 focused native checks passed in debug (characteristics, frozen JSON, observability and heuristic); the release extension rebuilt; focused `uv run --extra dev --extra play pytest` passed 24 tests (accessors, named-layout parity, text coverage, sacrifice/copy play, learning projection, Academy reconnect/stale/fork recovery); Ruff and Rust formatting passed.

Installed-extension comparison: canonical JSON followed by sorted buffer
names/bytes matches before/after for v9, v10 and v11 over **88 observations**
(the game finished before the 160-observation limit). Deal seed 111; NumPy
choice seed 93; C=128; 20 Mountain / 10 Lightning Bolt / 10 Gray Ogre versus
20 Forest / 20 Llanowar Elves. SHA-256 per version:

- v9: `cac8b692b58e492c9ee1a909dc6ff1b91230fcb404ba59cc758e960a15c58929`
- v10: `d4baa07e4c00eca136417e89a1e0c7eff1347d8777882a5911f15bcdf9c7da9b`
- v11: `efaac464056ddbc52ba88238dc276c1365ae36038660ff462b6416234e72db28`

Remaining: gate/CI, the interactive token/copy walkthrough and retained-checkpoint
demonstration, then review. After the sync, the installed extension predates
main's `managym.ObservationRead` (`manabot.env.failures` needs it), so no
Python test has run on the merged tree; gate rebuilds the extension and reruns
the focused Rust (debug) and Python checks above. No merge or auto-merge. Publication belongs to the
caller's delivery step. No retained receipts or checkpoint weights changed.

## Compress pass (2026-11-14)

- Removed the card writer's second flat copy and temporary vector; schema-8
  card serialization reads the canonical record. Historical normalization
  mutates its owned observation instead of cloning it first.
- Compatibility spell conversion has one owner (`SpellData::from_legacy_card`).
  Decision capture iterates borrowed object records without collecting them.
- Academy shares the effective type-field projection. Verification consumers
  read permanents directly; deleted `battlefield_pairs`. Fixed the remaining
  single-spell competency path and obsolete reward stubs, with a counterspell
  probe assertion in the existing native scenario.
- No remaining deletion targets in this correction. Required v9/v10 support
  and Python warning aliases remain scheduled for removal in 0.4.0.
- Python validation: 91 representation/text/recovery tests passed, then 44
  reward/competency/scenario tests passed after final consumer edits. Ruff and
  formatting passed.
- `CARGO_BUILD_JOBS=2 cargo test --manifest-path managym/Cargo.toml -p managym --lib agent::observation -- --test-threads=2`: 15 passed; the corresponding `agent::heuristic` selection: 17 passed, both in debug.
- The release CPython 3.13/aarch64 wheel rebuilt with the prescribed root
  `uv run --extra play maturin build --release` command and its extension was
  installed atomically. `uv run --extra dev --extra play pytest tests/env/test_observation_characteristics.py tests/env/test_named_observation_layout.py tests/semantic/test_learning_projection.py -q`: 28 passed on that extension.
- Rust formatting, Ruff and `git diff --check` passed. Gate/CI and the
  interactive demonstration remain with their Graph steps.

## Resumed concept review (2026-11-14)

Reviewed the merged checkout at `89845f7531` against the direction above.
Outcome for loop-or-next: **nothing to change and no open product choice;
advance**. The characteristics correction already implements the previous
review's direction. The remaining gate and demonstration are validation, not
another design iteration. No code changed or execution restarted in this pass.

The review screen is `scratch/concept-review.html`. Keep the existing next
demonstration: distinct same-definition tokens, sacrifice with a departed
source, reconnect/stale rejection, copied-spell identity/X, and retained
checkpoint admission. Learning improvement remains unproven, and MTG-353 retains
broader compatibility design. The later QA record below supersedes the earlier
installed-extension mismatch for the scenarios it actually ran.

## Conversation concept review after QA (2026-11-14)

Read the implementation and QA at `c8cde7f381` while run `f3ae952d` was publishing.
No further representation change or open product choice was found. The public
API now expresses the domain distinctions directly: effective characteristics
on cards/permanents/spells; token/copy status and object identity outside those
characteristics; abilities with captured source provenance and no spell record.
Deprecated accessors derive their values, and historical identity ambiguity is
refused. Keep the interactive review boundary; this review neither advances it
nor authorizes merging.

Recorded QA (`scratch/qa.html`): 240 seeded games, 48,015 observations, 237 games
finished and no invariant problems; a full Academy game with distinct Clue
offers and reload restoring six pending offers; v9 checkpoint weights unchanged
and two games completing after sim used the checkpoint's own input config.
The departed-Clue source was inspected in LLM text. A browser screenshot of its
ability after sacrifice was not captured because priority automation resolved
it before capture. Tests recorded 44 focused passes plus five sim tests. These
are the worker's recorded results, not reruns by this review. Host load made
the QA timings unsuitable as performance evidence.

One separate recovery mismatch remains: `manabot/sim/sim.py` catches policy
errors, samples a random action and still records a game outcome. A completed
game alone therefore cannot demonstrate model compatibility or strength; require
zero inference errors for those claims. The representation correction does not
require a decision about interactive fallback policy. Leave this pre-existing
behavior visible as separate follow-up work; no Task was filed by concept-review.

Review outcome for loop-or-next: nothing further to change in this Task's
concepts and no open product choice. Publication, gate and the interactive
demonstration retain their existing authority. `scratch/concept-review.html`
shows the built API and the recovery/evidence limits.
