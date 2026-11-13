# MTG-297 production and spending contract

The original MTG-286 proposal is preserved at
`5f6ae4633:scratch/tle-production-design.md`; its reviewed MTG-297 copy is at
`2af8bc650:scratch/tle-production-design.md`. Review answers and timing decisions
are in `heartbeat-design.md`; implementation status is in
`mtg-297-heartbeat-of-spring.md`. This contract covers the complete production and spending change.

## Production

`state/card.rs::TriggeredManaAbility` names the tapping actor, source predicate
and output: fixed mana (Badgermole Cub), or one actually produced type
(Heartbeat of Spring), colorless included. The tapping player chooses and
receives Heartbeat's mana, regardless of its controller. No production means
no bonus; a bonus never creates a second tap-for-mana event.

`flow/mana.rs` owns production and its payment-search alternatives. Capture the
tapped permanent before sacrifice; select live trigger sources when output is
produced (CR 106.12a); resolve their captured occurrences immediately afterward,
off the stack (CR 605.4a). Waterbend and Food taps do not qualify. Multiple
Heartbeats retain separate choices. If immediate effects become order-dependent,
use a native ordering decision rather than choosing their order automatically.

`Decision::ManaChoice.production` retains the output's source, or the bonus's
source, tapped permanent, actual produced mana and pending bonuses. Its saved
caller resumes once. The Great Henge/Citadel non-mana instructions use the one
existing child-frame path (`ActivatedAbilityDefinition::mana_suffix`), including
source attribution and preventable damage; suspension never repays costs.

The planner includes every output/bonus combination in the existing finite
resource search. Execution stops at nontrivial choices; automatic fixed sources
must have only one total. Source taps, sacrifices, counted taps and limited-use
marks remain separate finite resources, including across nested payments.

## Pools, spending and receipts

Build on MTG-242's `ManaRestriction` and `SpendingContext`. `ManaClass` keys
restriction and spend effect; ordinary mana retains its allocation-free case.
Equivalent units merge, with regular/combat expiry kept separate. Animal
Attendant's captured rider does not retain or require its source.

`ManaSpent` records exact W/U/B/R/G/C amounts and applicable entry counters for
one payment. The spell's receipt includes additional casting costs, excludes
non-mana costs and nested mana-ability fees, and travels through the stack,
resolution and its entering permanent's own triggers. Toph reads its total;
Uncle's Musings reads distinct non-colorless colors. Attendant qualifies against
the spell as cast and contributes one entry counter per applicable unit, through
`ZoneEntry`, not an ETB trigger. Countering/reanimation cannot transfer the
receipt to a later entry; copied spells have no original payment receipt.

Spending is a native unit/class choice whenever alternative receipts or
remaining pools differ. One unit is offered at a time, feasible for every saved
obligation: colors for generic costs, restrictions, spend effects and expiry all
matter. Equivalent units add no choices. Native narrowing, if needed, commits
no spending. Preserve Command, replay and fork identity; no browser-only
selection or undo of committed costs. Memoization, bounds and executable
witnesses use the same pool semantics. Automatic production pools the complete
current cost before a spending choice starts; nested output and parent costs
remain in feasibility throughout that choice.

The merged `flow/mana.rs::PaymentSearch` may group interchangeable production
resources and prune impossible supply with total/color bounds. Every saved
counted-tap cost must still have enough present, untapped candidates; reject a
branch as soon as it cannot meet that cost. Pool class and expiry remain in
every search key; `PaymentSearchState::usable` counts allowed
classes, including unrestricted riders and combat mana. A feasibility witness
stops automatic production at `PaymentStep::Spend`; the native decision then
chooses units, rather than committing the witness's hypothetical remainder.

## Shared observation

Review superseded the proposed new v10 block, fact kinds and v1–v9 rejection.
Use existing layouts in v9/v10, with native/Python scalar/vector/text/token
parity and typed capacity errors from supplied buffers. No new version branch,
model embedding, raw storage identity or hidden information is needed.

- A production choice uses the existing obligation frame: output `source` is
  the mana ability's source; bonus `source` is its trigger, `context` the tap.
- Public pooled classes are kind-11 `EffectProgram`s on the player under
  `PooledMana`, one class/expiry program per occurrence. Counts also join the matching
  ordinary/restricted aggregate. Keep exact restrictions, effects and expiry
  understandable to both viewers. Combat classes (including ordinary combat
  mana) retain their expiry in the program; Academy displays the rider,
  restriction and expiry. A spending obligation uses code 31 in the existing
  kind-6 frame and joins native mode offers to unit programs in kind 14.
- A rule-relevant cast receipt is a kind-14 execution occurrence (`mana spent`)
  on its public stack or acting resolution frame. Register every runtime name
  in `record_vocabulary.rs`. A future rule reading another object's receipt
  would require projecting that object's receipt too.

## Witnesses

Preserve directed outcomes for either-seat Heartbeat, chosen/colorless/two-type
outputs, independent Heartbeats, sacrificed producer/removed trigger source,
Badgermole scope, waterbend/Food exclusion, and Henge/Citadel suffix exactly once.
Spending covers ordinary/Attendant mixtures, qualification, multiple units,
source departure, clearing/partial spending, countering/reanimation, nested
filter fees and regular/combat expiry. Toph/converge must distinguish equal
printed costs with different actual spending. Every new suspension needs shared
observation parity, exact fork/replay and stale Command rejection.

## Delete — do not maintain

Completed: fixed-only `TriggeredManaAbility.mana`; `tap_mana_bonus`,
`tap_mana_bonus_from`, `tap_bonus_sources`, `fire_triggered_mana_abilities`;
source-free runtime `ManaChoice`; restriction-only pool equivalence.
Their consumers use production and `ManaClass` directly.

Completed with spending choices: `Mana::pay`, `payment_options`,
`apply_payment_remainder`, automatic class ordering, `Game::spend_mana` and
`spend_mana_to_remainder`. Ordinary/restricted/hybrid and combat behavior use
one unit transition model and the shared decision surface.
Compression also removed the unused `Mana::set_class` remainder writer and
the single-use `ManaClass::effect_applies` forwarding method. No deletion
target remains open. The unpublished `spend_effect_mana` /
`SpendEffectManaData` / `SpendEffectMana` surface is replaced throughout by
`pooled_mana_classes` / `PooledManaClassData` / `PooledManaClass`; no alias is
kept. `PooledManaView`, token programs and buffer layouts survive unchanged.

## Pooled-mana presentation decision (concept review, 2026-11-12)

Implemented in `fcc7a514c`. The former `PlayerData.spend_effect_mana` /
`SpendEffectManaData` also carried ordinary combat mana with no spend effect.
Its comment said "The field name predates expiry projection", although the
surface was new in this change. Renaming before publication avoids a
compatibility alias; the behavior and learning representation are unchanged.

- The player's `pooled_mana_classes` has one row per `ManaClass` that the plain
  totals cannot state (a spend effect or combat expiry). Rust
  `agent/observation.rs::PooledManaClassData` and PyO3 `PooledManaClass` own
  the native and Python shapes; `PooledManaView` keeps its name. Encoders,
  bindings, `experience*.py`, text, Academy and all six schemas use the new
  field. Token programs, fact kinds and buffers are unchanged.
- No restriction or no effect is absent (`Option<String>` / `str | None`,
  omitted when serialized), not an empty string.
- Text observation states each row the same way (amounts, then restriction,
  effect and expiry when present) instead of titling an effect-less row
  "Combat mana:".
- Academy shows the aggregate "N mana expires at end of combat" line only when
  a frame has no class rows, as `lesson_mana` already falls back. The class rows
  otherwise carry expiry themselves (`PlayerArea.svelte`).
- The obsolete "predates" comment is removed.

The implementation and compression checks are recorded in
`mtg-297-heartbeat-of-spring.md`. Remaining work is gate acceptance there:
current-world evidence, affected checks and corpus admission. No open product
choice; the presentation repair is no longer pending implementation.

## Concept review after the pooled-class rename (2026-11-12)

Reviewed `fcc7a514c` against this contract: pool rows, spend offers
(`flow/action.rs`, "Spend {G} (…)"), text and Academy name one `ManaClass` by
the same four facts, and no old name survives. Nothing to change; no open
product choice. Noted, not directed: a non-combat class and a non-combat spend
offer both say "expires at step end", where plain totals and "Add" labels
leave ordinary expiry unstated. The wording is the class's default expiry
(CR 106.4); retention (`keeps_red_mana`) stays its own line, as it already
does for combat mana.
