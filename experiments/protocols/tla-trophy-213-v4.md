# TLA trophy pilot with shared effects

V4 consumes the shared-effect contract delivered in PR #419, merged as
`fb5ad2c1372f9c967b550f0b8ae0f84883f75563`. It explicitly selects policy input
**v7**, with **128 local facts**, **128 × 18 effect headers**, **512 × 3
shared details**, and **2048 × 2 memberships**. Schema 3 serializes every
ObservationSpace shape and rejects drift before training or evaluation. The
regime hash binds the complete configuration. These fixed capacities fail on
overflow; they do not guarantee every reachable position fits.

The runtime remains w17 with card digest
`dcb9d47c674a657176c2a9d56e016288d1e175729aa09cd60f0dc333633330ed`.
Effects reach affected object embeddings before attention. Complete records,
provenance, ordering and duplicate memberships survive factoring; repeated
activations are not semantically merged. Default padded effects and local facts
occupy 37,888 bytes, versus 12,288 for v3's v6/256. This is neither a memory
saving nor a learning-speed claim.

The same **134 entries**, sideboards, source-event weights, independent seat
draws, seeds, 100-update recipe, milestones, eight pairs and four legs remain.
V3's 171 completed-but-unoptimized games and 32 initialization evaluation games
cannot fill v4 slots. Use fresh initialization and paired evaluation. Preserve
all v1/v2/v3 protocols, regimes and receipts. Their runners and analysis remain
at their exact frozen source; v3 is at
`3b122de9078c985f846a0f3c5a3a42ebca3b1182`. The maintained runner is now
`experiments/tla_pilot_v4.py`, with the shared analysis migrated alongside it.

The [v3 protocol](tla-trophy-213-v3.md) and its inherited scientific/allocation
contract otherwise apply: **$10 cumulative paid cap**, **$1 allocation ceiling**,
2,700-second training watchdog, original failure allowance and paired-bootstrap
criterion. Reconcile all prior bills and reservations before freezing and
launching verified source through the existing rental owner. No extra allowance
comes from the representation Task. Report the launch milestone only after
observed GPU learner updates. Checkpoint admission, complete evaluation and
Elemental-relative learning analysis remain acceptance, including an honest
negative result. This protocol preparation supplies none of those outcomes.
