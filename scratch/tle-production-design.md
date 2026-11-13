# MTG-297 production and spending contract

The original MTG-286 proposal is preserved at
`5f6ae4633:scratch/tle-production-design.md`; its reviewed MTG-297 copy is at
`2af8bc650:scratch/tle-production-design.md`. Review answers and timing decisions
are in `heartbeat-design.md`; implementation status is in
`mtg-297-heartbeat-of-spring.md`. This contract includes unfinished work.

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

**Still required:** replace automatic first-remainder commitment whenever
alternative receipts or remaining pools differ. Offer one unit/class choice at
a time, feasible for every saved obligation: colors for generic costs,
restrictions, spend effects and expiry all matter. Equivalent units add no
choices. Native narrowing, if needed, commits no spending. Preserve Command,
replay and fork identity; no browser-only selection or undo of committed costs.
Memoization, bounds and executable witnesses must use the same pool semantics.

## Shared observation

Review superseded the proposed new v10 block, fact kinds and v1–v9 rejection.
Use existing layouts in v9/v10, with native/Python scalar/vector/text/token
parity and typed capacity errors from supplied buffers. No new version branch,
model embedding, raw storage identity or hidden information is needed.

- A production choice uses the existing obligation frame: output `source` is
  the mana ability's source; bonus `source` is its trigger, `context` the tap.
- Public spend-effect pools are kind-11 `EffectProgram`s on the player under
  `PooledMana`, one class program per occurrence. Counts also join the matching
  ordinary/restricted aggregate. Keep exact restrictions, effects and expiry
  understandable to both viewers; Academy's rider display remains unfinished.
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

Remaining with spending choices: unconditional first-remainder commitment in
`Mana::pay` / `Game::spend_mana` / `spend_mana_to_remainder`, including automatic
class/expiry preferences. Replace those choices with the shared decision
surface, retaining ordinary/restricted/hybrid and combat behavior in one planner.
Do not polish that predecessor while its replacement remains pending.
