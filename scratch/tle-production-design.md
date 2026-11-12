# TLE production and spending — proposed replacement contract

This is a reviewable design, not implemented support or a completed observation
review. It replaces the open production questions in the main plan. The authored
rules/observation review must resolve the items below before dependent writers.
The Merchant counted-tap implementation and ordinary definitions do not depend
on this replacement. Keep the current 1v1/no-commander scope.

## Counterexamples and owners

Heartbeat of Spring needs the land's actual produced types and either player's
applicable bonus sources. The existing `TriggeredManaAbility.mana`,
`flow/zones.rs::tap_bonus_sources(controller)` and tap-time firing cannot supply
that. `Decision::ManaChoice` also drops the production source. Animal Attendant
needs a persistent entry effect on the mana actually spent, not just a color or
spending restriction.

The completed paragraph allocation adds three constraints to that proposal:

- Toph, Greatest Earthbender counts **all mana spent to cast**, including
  additional mana costs, but not waterbend taps, life, sacrifice or mana spent
  activating another mana ability on the way to that cast.
- Uncle's Musings counts the **distinct colors** of that same spending, excluding
  colorless. A generic cost's automatic remainder choice can change its result.
- Tarnished Citadel deals damage and The Great Henge gains life inside a mana
  ability. A production evaluator that accepts only a final AddMana instruction
  cannot silently discard these suffixes. Damage is not a life payment.

`state/mana.rs`, `flow/mana.rs`, `flow/decision.rs`, `flow/play.rs`,
`flow/zones.rs` and `state/stack_object.rs` own this cut. Production and spending
must use one transition implementation for feasibility witnesses and execution.
No card-name branches, secondary planner or provenance side table.

The Great Henge and Tarnished Citadel are cut first, without this proposal:
their suffix runs as the existing pre-`ManaChoice` child frame (plan, "Next
implement direction"). This transaction later absorbs that step; it does not add
a second suffix path.

## Production transaction

Replace the source-free mana output continuation with a production occurrence
owned by the existing resolution frame. It captures the activation's exact
source `ObjectRef` and LKI, controller, whether the source's own tap symbol was
paid, chosen effect program, produced amounts/types, pending immediate mana
triggers and saved caller. Counted Food taps and waterbend taps never mark the
Food/land as the producing source. Capture before paying a sacrificing source's
cost, so subsequent disappearance cannot change its identity or characteristics.

Execute the mana ability's complete chosen program without stack/priority.
At each AddMana instruction commit the chosen output and retain its **actual**
six-type vector; continue its suffix once. Only real production by an activation
that tapped its source qualifies as tapping that permanent for mana. Ordinary
becomes-tapped triggers remain at the tap transition; move the triggered-mana
bonus authority out of that transition. Do not fire a production trigger merely
because a tap cost was paid or count a zero output as producing a type.

Replace `TriggeredManaAbility`'s fixed-only output with typed output semantics:
fixed `Mana` (Badgermole Cub) or one chosen type from the triggering production
(Heartbeat). Its source predicate and player scope explicitly select controlled
or any player's source. Enumerate eligible trigger sources from the board at
the production event, capturing their programs/controller then; resolve them
immediately after the producing mana ability completes. The triggering producer
chooses Heartbeat's type and receives the mana even when an opponent controls
the Heartbeat. Preserve the actual controller separately. Multiple output types
offer each actually produced type exactly once; multiple Heartbeats are distinct
occurrences, not one multiplied output choice. A bonus is not a new tap-for-mana
event. Existing Badgermole behavior remains scoped to its controller.

If several immediate abilities can have order-dependent outcomes, retain an
ordinary native ordering decision before resolving them; never sort away a
strategic order. The same captured occurrences and output alternatives enter
payment search. Witness execution may advance uniquely determined operations;
it stops at every nontrivial output/order/spending choice. One legal alternative
uses existing trivial-decision skipping. Committed costs and outputs are never
replayed when a suffix or child payment suspends.

The reviewer must settle exact production-event timing/source snapshots against
the repository's pinned rules and compare either-player triggers, source
sacrifice and multiple output instructions. The proposal deliberately names this
boundary; no new triggering record is implemented before that review.

## Pool equivalence and spending receipt

Reshape the current `Mana` pool's restricted-class storage into one canonical
class representation. Ordinary amounts keep their allocation-free fast case;
nonordinary classes are keyed by restriction, spend-effect payload and expiry.
Keep regular/combat expiry distinct throughout planning and consumption. A class
may merge only when every rule-visible property matches. Retain source identity
only where a payload refers to it; production occurrences always retain exact
source identity. Animal Attendant's captured counter effect does not require the
Attendant to remain present. Do not add a second pool of raw unit ids or an
external map keyed by source.

A spending transition returns both its remainder and a typed receipt containing
six exact amounts and applicable captured spend effects, with their multiplicity.
Nested upfront activation costs produce receipts for that activation only. The
outer spell's receipt collects its own mana costs once across interruptions and
travels from `Payment::Spell` to `SpellOnStack` and the entering permanent's cast
provenance. Toph reads the sum; converge reads W/U/B/R/G presence; neither reads
printed cost or the aggregate pool difference across nested production.

Animal Attendant contributes an entry counter for each of its mana spent on a
qualifying non-Human creature spell. Evaluate its spend qualification against the
spell as cast and retain the resulting entry replacement with that spell. Entry
replacement ordering stays with the zone-entry authority. Counters on entry do
not become an ETB trigger. If the spell is countered the receipt cannot affect a
later reanimation of that physical card; copied spells have no original spending
receipt unless a rule explicitly says otherwise. A resolved entering permanent
retains only provenance needed by its own ETB instructions, tied to that entry.

Replace the automatic first-remainder commitment with semantic spending choices
whenever alternative receipts or remaining pools differ. This includes colored
mana used for a generic cost, restricted versus unrestricted amounts, regular
versus combat expiry and provenance-bearing versus ordinary mana of one color.
Offer one unit/class choice at a time, filtered by feasibility for **all** saved
obligations, to avoid materializing every distribution. An equivalent class's
units are indistinguishable; do not manufacture choices among them. Group large
class sets through native choice narrowing if necessary; narrowing spends nothing.
Only equivalent results may be automatic. No browser-only selection or undo of
committed costs. Preserve current output/Command identity for unchanged offers.

Finite resource search still owns source taps, sacrifices, counted taps and
limited-use marks. Every considered activation consumes a finite resource; bonus
production belongs to that activation and cannot recursively retap its source.
Replacing the pool key must update search memoization, both optimistic bounds,
parent obligations and exact witness replay together. Ordinary cost parsing and
printed-cost learning tokens stay one implementation. Historic manifests and
checkpoint payload omissions must not change merely because the in-memory pool
representation changes.

## Proposed policy input and public projection

Do not retrofit these meanings into frozen v1–v9. Propose explicit input **v10**,
retaining v9's object, relation, shared-effect and naming layouts and fresh v6
default. One new `mana_choices[A,4]` block distinguishes the layout at native
buffer binding and supplies action joins: one-based class/production record,
color code (W/U/B/R/G/C = 1..6), amount, and choice operation (spend/produce/order
or group). Nonmana actions are zero. The schema must include group membership
before any grouping implementation; no unexplained numeric group identifiers.

Project canonical mana records once in native authority for both viewers.
Public pool state includes all class counts, restrictions, expiries and captured
spend effects. Pending production/spending continuations belong only to the
acting viewer, consistent with the existing contract. Public stack spell
receipts and entering-object cast provenance are observable where they affect
rules. Hidden source cards get no invented live object join; use already-public
LKI/definition provenance and explicit absent-object references. Never expose
raw storage ids, private hand contents or retained source pointers.

Use v10 fact kinds following 14 for (a) canonical pool-class headers/counts,
(b) exact tokenized restrictions and spend-effect programs and (c) cast receipts.
Their concrete field/ordinal grammar, units and join owner must be written in
the encoding table during observation review **before implementation**. Existing
kind-14 continuation programs and kind-10 reference relations carry the captured
production/decision program where their current meaning suffices; do not encode
pool classes as fake permanents or overload remaining-tap `mode`. Split signed
32-bit token values exactly as v8 does. Native scalar/vector, Python reference,
semantic capture, text and model tokens must share this projection and capacity
failure behavior. Register every hashed name in `record_vocabulary.rs`.

This intentionally leaves a concrete review question: can the existing fact
layout carry all class headers, programs, receipts and action joins without
ambiguity, or does v10 need one dedicated bounded record block? Do not add both
as parallel stores. Review the worked witnesses below and fix one table; until
then production/spending runtime writers are blocked. This is a narrower open
question than the former unspecified production model, not an approved schema.

Admission rejects v1–v9 before either encoder writes when novel production,
spend provenance or observable receipts are present/required. Preexisting simple
color choices must retain their legacy projection if still representable. Cards
needing unavailable information reject at shared admission; they never become
silently zeroed tensors. All capacity limits come from supplied buffers; overflow
is a typed error. No frozen evidence/configuration rewrite or fresh default bump.

## Required witnesses and deletion cut

Directed witnesses: Heartbeat controlled by either seat, an opponent's land,
colorless and two-type output, two Heartbeats choosing different types, a source
sacrificed in its mana activation, Badgermole's old scope, and waterbend/other-Food
taps producing no bonus. Henge life gain and Citadel preventable damage happen
once before resuming the caller. Test mixed ordinary/Attendant green mana with
one generic obligation, qualifying/nonqualifying spells, two Attendant units,
Attendant departure, countering/reanimation, partially spent/cleared pools,
nested Prism fees, and regular/combat pool choice. Toph and converge witnesses
must distinguish equal printed costs with different actual spending. Every
suspension needs fork/replay/Command rejection and v10 scalar/vector/Python/text
parity, plus v1–v9 rejection with sentinel buffers unchanged.

Delete with the production cut: `TriggeredManaAbility.mana`'s fixed-only role,
`tap_mana_bonus`, `tap_mana_bonus_from`, `tap_bonus_sources`,
`fire_triggered_mana_abilities`, and source-free `ManaChoice` construction.
Move Badgermole and any other existing consumers in the same cut; retain their
behavior tests on the production authority. Delete the old bonus-only tests if
exclusive to the removed implementation, preserving directed card outcomes.

Delete with spending: restricted-class-only pool equivalence and unconditional
first-remainder selection in `spend_mana`/`spend_mana_to_remainder`. Preserve
ordinary/restricted/hybrid and combat-expiry behavior through the replacement,
not an adapter. Keep one mana planner, definition serializer and semantic pool.
Do not refactor either predecessor before this review/cut is ready.
