# One token per game thing: v10 rows against input v11 (MTG-344)

**Exploratory.** This run asks how the new layout trains against the old one
on the current recipe, and what it costs; its numbers are not a finding.

## Question

Input v11 gives each game thing one row of the policy's object table: cards
outside the battlefield and stack, permanents that carry their own
characteristics, and one stack segment for spells and abilities. In v10, a
battlefield creature is two attention tokens (a card row and a permanent row
joined only by list order), a stack spell is a card row and a stack ability is
no token. Does a policy trained on v11 play the same Elemental games better or
worse than one trained on v10, at the same number of updates, and what does
v11 cost to encode and run?

## Method

- Arms: the ten-deck Elemental recipe (`elemental_variants.elemental_recipe`:
  two-seat self-play over uniform deck pairs, width 64, two attention layers,
  256 streams) with `observation.policy_input_version` 10 or 11 and nothing
  else changed (`require_single_axis`). Both carry the same turn state, so the
  arms see the same information in different rows.
- Two shared seeds per arm (34401, 34402), 200 updates, one L4 each.
- Evaluation: scheduled checkpoints against the native heuristic over the 50
  fixed Elemental pairings (four legs each), and v11 against v10 at the last
  update both runs of a seed pair checkpointed.
- Strongest confound: two seeds per arm. A difference smaller than the
  seed-to-seed spread of one arm says nothing.
- Cost cap: USD 25 for training and evaluation, under the launcher's cap.
- Encode and forward costs and row occupancy: `uv run python -m
  experiments.one_token_layout measure` on one laptop (below).

## Results

Pending.

## Cost

Pending.

## Limits

- Exploratory: two seeds, 200 updates of a small model.
- The legacy arm runs through the deprecated v9/v10 fork; its encodings match
  452d7b5357 byte for byte on 12,403 curated observations.
