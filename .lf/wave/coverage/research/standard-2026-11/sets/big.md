# The Big Score (BIG): artifact interactions as shared-engine witnesses

Reviewed 2026-11-04 against `.cache/standard-research/big.json`: 30 Oracle
identities, all single-faced; every full rules text is represented in
[the ledger](big.jsonl). Its source page was retrieved at
2026-11-04T19:07:40.594656Z, SHA256
`d60256b78eff3df3cb131496935bf2a27b2f1bda6fd4095eb0b50284c7e01e38`.
This is the rotating-set working inventory, not an independent legality claim.
No matching exact-name definitions currently exist in `managym/cards/src`.
Reprints belong in their first-printing modules, particularly Grand Abolisher,
Rest in Peace and Torpor Orb, rather than a duplicate BIG definition.

BIG should follow surviving-set mechanisms, while contributing compact acceptance
witnesses now. Eight cards have plausible existing compositions; that assessment
is source inspection, not executed proof. The remaining 22 need shared vocabulary
or structural extensions, often small but strategically consequential. No games,
benchmarks, production changes or new Tasks were performed.

## Existing authorities and ordered batches

**BIG-01 — definition composition (8).** Start with Harvester of Misery, Pest
Control, Legion Extruder, Sandstorm Salvager, Oltec Matterweaver, Bristlebud Farmer,
Fomori Vault and Lotus Ring. `state/src/ability.rs` already has simultaneous
`Destroy`, predicate selectors, `MillAndSelect`, dynamic `LookAndSelect`, modal
triggers, token copies and temporary effects. `ActivationZone::Hand` plus
`discard_source` supports Harvester and cycling. Farmer composes optional sacrifice
with mill selection; its candidate construction in `flow/resolution.rs` needs a
replacement-aware zone-validity check before claiming interaction completeness.

Lotus Ring reuses `StaticAbility::AttachedHas`, the layered activated-ability list,
and `mana_choices()` recognition of fixed modal outputs. Its sacrifice belongs
to the equipped creature, not the Equipment. A concrete payment witness must
check the grant, summoning sickness and three-of-one-color production. This is
not evidence that every mana grant needs new machinery.

**BIG-02 — contextual values and events (5).** Ancient Cornucopia, Simulacrum
Synthesizer, Hostile Investigator, Generous Plunderer and Loot share missing
queries or event descriptors. Generalize values by subject and aggregate:
triggering-spell color cardinality; defending-player artifact count; distinct card
types of other nonland permanents; and continuously counted artifacts. Synthesizer's
Construct is a static buff, not a one-time counter placement or a power CDA.
`StaticPtBuff` currently has specialized graveyard/color multipliers but no
battlefield-count multiplier. Investigator needs discard events, while
`OnceEachTurn` already owns trigger consumption. Cornucopia instead reuses
`MayOnceEachTurn`: declining must preserve its option; acceptance also needs to
suppress later triggers, beyond the present resolution-only guard. Plunderer's `WhenYouDo`
exists, but tapped token entry needs a distinct property from
`tapped_and_attacking`.

**BIG-03 — copy snapshots and grants (5).** Duplicator, Molten Duplication,
Territory Forge, Vaultborn Tyrant and Nexus of Becoming should share the copy
and layer authorities. `CreateTokenCopy` currently addresses source or restricted
target forms; `flow/zones.rs::create_token_copy` clones stored card characteristics.
It is not a generic copier of triggering, remembered or selected hand objects.
Preserve copiable snapshots through sacrifice and delayed resolution, including
original tokens that cease existing. Add artifact types through the existing
copy-exception vocabulary; temporary haste remains a separate effect. Tyrant also
needs the nontoken source condition. Forge requires activated-only abilities from
its linked exiled card: `HaveAbilitiesOf` currently grants both activated and
triggered abilities and is not an equivalent substitute. Coordinate with MTG244.

**BIG-04 — costs and mana (4).** Lost Jitte, Transmutation Font, Tarnation Vista
and Greed's Gambit exercise different shared obligations. Add charge to the counter vocabulary. Counter removal is an
activation cost, not a resolving `RemoveCounters` instruction. Font needs three
simultaneous sacrifices with distinct current names; Gambit's three sacrifices
are instead a resolving instruction that does as much as possible. Vista needs
an as-enters color choice and dynamic mana output, bounded to five colors.
Reuse payment witnesses and distinct-name selection work in MTG242/MTG243;
do not enumerate arbitrary token subsets or invent card-specific payment paths.

**BIG-05 — linked selection and play (3).** Cage, Omenpath Journey and Memory
Vessel require private selections, source links and explicit timing. Existing
`ExileLinked` and `CastExiledWithSource` are useful infrastructure but their
specific grants do not express hideaway or immediate free land play.
`PlayWindow` currently distinguishes this turn from through the next turn;
Vessel expires when the next turn begins, and prohibits playing hand cards.
Its source exile is an activation cost. Journey combines distinct-name search,
linked exile and seeded random choice. MTG241 should supply composable selection
and contextual reads; it must not silently substitute a private search for a
random choice.

**BIG-06 — event restrictions and replacements (5).** Rest in Peace,
Worldwalker Helm, Torpor Orb, Grand Abolisher and Sword of Wealth and Power
need extensions at shared event or legality authorities. The current
`ReplacementEffect` vocabulary lacks general graveyard redirection and token
creation replacement. Helm requires a set of created objects, rather than one
`frame.created` permanent. Orb suppresses by the entering object's event cause;
Abolisher restricts eligible activations, including mana abilities. Sword needs
parameterized protection and the existing delayed spell-copy machinery. Its
combat trigger can reuse the attachment match behind `EnchantedCreature`, with
the Equipment contract clarified rather than a duplicate matcher. MTG241 owns the relevant composability work.

## Closure, observations and discriminating evidence

Add exact Gnome, Golem, Bat, Construct, Map and Blood definitions; existing Clue,
Food and Treasure are reusable. Map entails explore, including reveal, land-to-hand,
counter placement and optional graveyard placement. Blood entails discard as an
activation cost. There are no back faces to close. Tokens must be actual token
definitions, never ordinary cards passed to token creation.

Focused witnesses from the [official release notes](https://magic.wizards.com/en/news/feature/outlaws-of-thunder-junction-release-notes)
include Cage retaining inspection rights for former controllers and playing during
resolution; Duplicator remembering a departed copy; Helm propagating creation
instructions to its added Map; and Orb suppressing artifact-entry triggers when
the entrant is a creature while leaving replacements intact. Test Cornucopia
against a countered multicolored spell and a declined earlier opportunity. These
notes are dated rules evidence; full interaction correctness remains unproven.

Expose chosen colors, linked public cards, viewer-authorized hideaway knowledge,
copy exceptions, delayed snapshots, use limits and pending cost selections through
the native observation, Python/tensor parity and learning tokens. Preserve hidden
identities and exact incarnations across reconnect, stale rejection and forks.
Performance risks are repeated aggregate scans, token-driven trigger multiplication,
name-constrained selection branching and snapshot cloning. Use revision-cached
queries and immutable shared snapshots where justified. Future bounded paired
workloads should measure both ordinary steps and forks; retain existing Engine
Budget digests and cumulative cost accounting without synthetic load.
