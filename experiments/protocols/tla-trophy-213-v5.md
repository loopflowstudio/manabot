# TLA trophy pilot with space for generated card objects

V5 preserves the [v4 contract](tla-trophy-213-v4.md), changing only the pilot
identity and the existing `max_cards_per_player` configuration from 60 to 128.
The protocol serializes every resulting tensor shape. Policy input remains v7,
with 128 local facts, 128 effect headers, 512 detail tokens and 2048 memberships.
The shared scalar/vector native binding derives card capacity from the supplied
ObservationSpace, correcting its prior hard-coded default of 60. Default layouts
remain unchanged. The selected model now has a larger card/object axis; this is a prospective
architecture binding with fresh initialization, not a compatible checkpoint
continuation or a change to the card definitions.

V4 run `baa8671d6480` passed CUDA admission but failed before learning when a
40-card deck created five tokens and retained its 16 sideboard cards: 61 card
objects exceeded 60. Stream 22's 300-action prefix and exact pre-command frame
are retained. The shared encoder legitimately includes those generated objects;
none are discarded. 128 is an operator capacity choice under the accepted
workflow-repair authorization, not proof that all reachable games fit. Overflow
remains fatal.

All 134 entries, independent sampling, seeds, learning settings, milestones,
paired evaluation and cumulative $10 budget remain unchanged. V4's source is
`0b6c2bb854ba48b37fb7e5ac360c0c9d047cea34`; preserve its protocol, failed attempt,
initialization and 32-game evaluation. Its results never fill v5 slots. The active
runner and analysis move together to `experiments/tla_pilot_v5.py`.

At the 06:07 UTC settlement all four pilot rentals had ended and provider
inventory was empty. Earlier bills total $0.17054831; v4's whole-rental estimate
is $0.04314118 pending its bill. Retain all four $0.7375 reservations ($2.95)
conservatively when quoting the next $1 allocation. A new identity supplies no
new budget. GPU choice, exact verified source and reservation are recorded before
scoring; the ordinary CUDA gate, checkpoint admission, complete evaluation and
Elemental-relative analysis remain required. No learner update is established
by this preparation.
