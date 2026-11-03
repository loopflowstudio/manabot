# MTG-213 prospective fact-capacity pilot

Status: prospective preparation on 2026-11-03. Operator comment 78f4c226
selects **256 rows for the pilot**, following Jack Heart's question about 256,
under the existing bug-fix and $10 authorization. It supersedes the operator's
earlier 512-row choice in comment 5d6cf81e. Comment 0c17ad58 was an
operator-issued run reason, not a separate human confirmation. This allocation
is not proof that every legal state fits, a new allowance, or evidence of learning.
Comment be5b1394 subsequently relays Jack Heart's explicit confirmation of
the immediate larger-capacity hotfix, with broader representation design
assigned to a separate Task.

At 2026-11-03 23:20 UTC, the authoritative launcher contained only the two v1
pilot attempts. Local tracking and six retained pilot run receipts contained no
v2 execution or scoring. The configured `r2://manabot/jobs` store had 26,219
objects, none modified since 22:33 UTC (before v2 was introduced), and no v2
identity in its 23 run metadata files. The read receipt is
`.runs/mtg213/v2-pre-amendment-state.json`. Therefore the unexecuted prepared
v2 bindings were amended in place; the earlier 512 preparation remains in git
at `1af50386fdc4d96b936e81a8c32fae3054dc2c5c`.

## Prospective binding

The JSON protocol and regime named `tla-trophy-213-v2` own this execution.
Policy input **v6**, with serialized **max_facts=256**, replaces v5 for this
new experiment only. Generic fresh v6 configuration still defaults to 512.
All twelve fact columns, row order, timestamp ranks and model timestamp
normalization (rank / 128) stay unchanged. Native scalar and vector encoding, Python encoding, allocation, model shape checks, input-schema
bindings and checkpoint architecture receipts use the selected capacity.
Overflow remains fatal; no fact truncation, deck exclusion or quarantine bypass
is introduced. V5 retains exactly 128 rows and its prior serialization.

The source baseline is `2cb3d3d77a9c939534601ac4a70cb24f8561c926`, including
main's PR #414 (`44388ad4211b4292cc161a3087eb1230b2005737`), plus the reviewed
v6 repair. Launch must freeze the exact committed repair revision through
Etude; its source receipt, native artifact hash and job receipt identify the
actual build. The protocol rejects a different installed native identity:

- Open rules world: `w17`.
- Card digest: `fef123cfe085dd252067ee318f0b31790b2ee4a5cd0a1f7d0404a8580aec61b2`.
- Input identity: v6, 256 × 12 facts, other v5 shapes unchanged.

This explicitly selects a new runtime/input binding, not the historical
`2adbd0758` runtime whose card digest was
`17735e21aa9d59ebf17319f7ee43e16921189fde2f8886e38bfefc6dcca51e68`.
PR #414 changed serialized card definitions: 365 keyword objects became lists
of the same present keywords; another 13 locations changed rules vocabulary
(`ControlsMatching` to `ControlsAtLeast`, `BecomeArtifactCreature` to
`BecomeCreature`, conditional creature types to `TypeGrant`, and the land-type
counter condition). Comparing all 331 definitions isolates these differences;
the local audit is `.runs/mtg213/definition-diff.json`. No historical definition,
world, checkpoint, corpus or result is relabeled. A matching rules-world label
alone does not establish equivalence across these sources.

The retained stream 181 / seed 31482 replays all 87 actions under the selected
runtime and matches the retained pre-command frame. Its 140 distinct facts
encode identically through v6 native scalar, native vector and Python paths.
Both v5 encoders still raise `facts 140 > 128`. Local receipt:
`.runs/mtg213/v6-256-capacity-replay.json` explicitly binds the 256-row check.
The earlier `v6-capacity-replay.json` remains the 512-row repair receipt.
A controlled boundary fixture retains all 256 rows and rejects 258 in native
and Python encoding. These checks establish neither universal legal-state
coverage nor an end-to-end speedup. This is prospective advancement after
repair, not a changed outcome or credit for the failed GPU attempt.

## Population, learning and evaluation

Keep the exact [v1 corpus and scientific plan](tla-trophy-213-v1.md). The new
regime copies all **134 entries**, source IDs, complete decks/sideboards and
uniform independent seat weights byte-for-value from v1. All entries admit to
the selected current world; the 866 historical exclusions stay excluded. The
corpus SHA256 remains
`2173600970d0ab9efb44c620b082e8b15a2615d4e4d801ec0c015710dc333951`.
There is no new population, resampling, or held-out-deck claim.

The 64-wide/two-layer/four-head damped-policy recipe, CUDA placement, 256
streams, both-seat complete-game collection, seeds, 100-update target,
2,700-second watchdog and 25/50/75 milestones are unchanged. Initialize fresh;
never resume a v5 initialization. Evaluate the new initialization and eligible
checkpoints against the fixed source-pinned heuristic and this run's own
initialization. Keep all eight pairs, four legs, deals, 120-second/10,000-command
limits, missing/failure denominators and the joint bootstrap criterion from v1.

Etude's distinct experiment is **tla_pilot_v2**, discovered through
`experiments/tla_pilot_v2.py`. Its evaluator rejects v1 training runs, and its
paired benchmark/environment have v2 identities. The existing analysis owner,
`experiments/tla_pilot_analysis.py`, now reads v2. The old runner is removed
from the active checkout; historical v1 execution/analysis uses its pinned
source, available at `2cb3d3d77:experiments/tla_pilot.py`, and retained runtime
archives. Frozen v1 JSON, original attempts and results remain untouched.

## Allocation and delivery

The **$10 cumulative cap** covers v1 and v2 setup, failures, training and any
paid evaluation together. Before another allocation, reconcile the two
**$0.7375** reservations and provider billing; the last retained settlement
estimated **$0.06508817** spent, with both rentals ended. Neither that old
estimate nor a new operational cohort replenishes the budget. Use a fresh quote,
a $1 first job ceiling, the existing bounded setup/time/export windows, and
nested Doppler `manabot/prd` then `etude/prd`. No retry is automatic.

After gate validates the affected suites, the caller owns source freeze,
publication and the paid launch. Record the actual revision and allocation
before running. The launch milestone requires observed GPU learner updates;
local checks, provisioning and initialization do not satisfy it. Training and
trained-checkpoint evaluation remain outstanding at protocol preparation.
