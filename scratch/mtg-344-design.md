# One policy token per game thing (MTG-344)

Status: design for review. No code yet. Evidence: `scratch/encoding-study.md`
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

`manabot/backends/torch/policy.py` (and the MLX mirror): a stack projection
beside the card and permanent projections, adding the catalog program by
definition; the stack segment joins the attention set and the ownership
template; `_join_permanent_links` and other hard-coded offsets read segment
starts from the layout. Card and permanent projections widen with their
columns. No other architecture change, so a comparison measures the
representation.

## Compatibility and the world

Permanent rows widen and the stack segment is a new buffer, so the world
binding refuses older checkpoints by buffer shape; nothing loads silently.
Under the repository rule a new layout replaces v9/v10 at a world freeze and
adds no version branch. This change therefore lands as the next world's
layout, in one change, rather than in three staged layouts (each stage would
obsolete checkpoints again, and stage 2 alone changes row meaning without
changing shapes).

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
| Learning tokens projection | add permanent and stack roles; stop skipping spells |
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

## Decisions for Jack

1. Land as one layout at the next world freeze (recommended), or stage it.
2. Whether that freeze is now: it obsoletes every checkpoint on main,
   including in-flight frontier runs.
3. One shared stack segment with a controller flag (recommended; the stack is
   one ordered zone, CR 405.1) or per-side segments like cards and permanents.
4. Order against MTG-341 (engine object identity split). Both touch
   `observation.rs`; this change reads public ids only, so it can land first
   and MTG-341 rebases, or wait.
5. Comparison budget: two seeds per arm under USD25, or more.

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

## Implementation (2026-11-14)

- Native `Observation`: card lists hold cards outside the battlefield and
  stack (pending public entrants keep their row in their current zone).
  `PermanentData.card` and `StackObjectData.card` (spells) are the object's
  `CardData`, `Option` only for historical JSON captures; `StackObjectData`
  gains `kicked`. `Observation::card_records()` lists every record once.
- Input v11 (`InputLayout::OneToken`, fresh default, includes v10 turn
  state): permanent rows append 13 columns (mana value, 6 types, legendary,
  token, Ally, Lesson, ward flag/cost) before validity (width 51); `stack`
  [32, 37] + `stack_valid`, one shared segment, top first, `is_mine` by
  controller, `source` link. Object table: players, cards, cards, perms,
  perms, stack. A battlefield object's card id and a spell's card id alias
  its permanent/stack row in `object_to_index`, so focus, relations and
  history v1 contexts name the one row without changing focus producers.
- Legacy fork (inputs v9/v10, deprecated, removal 0.4.0, MTG-353):
  `managym/src/agent/legacy_input.rs` and `manabot/env/legacy_input.py`
  rebuild the old card lists from the records (`view`); the encoder branches
  on `InputLayout::Legacy` only for the card source, permanent width (38),
  no stack/aliases, and card-row-only history. A full file copy was not made:
  facts, effects, relations and actions encode identically in both layouts,
  so the fork is the parts that differ. Selecting v9/v10 warns.
- Model: stack projection and catalog program by definition; stack ownership
  from `is_mine`; buckets/admission/live tokens learn the segment. Framework
  backends (MLX, `manabot.policy.prepare`) stay v9-only.

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
