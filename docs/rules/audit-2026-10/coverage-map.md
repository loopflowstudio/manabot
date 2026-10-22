# Where the rules are strong, narrow or still open

The strongest evidence is in small, ordinary two-player interactions: basic
mana arithmetic, lethal-damage checks, object identity and several combat
keywords. The largest gaps are complete player choices, general continuous
effects, the long tail of mechanics and unsupported formats. We have no basis
to call an entire CR chapter thoroughly implemented.

This map summarizes inspected code and retained tests through `c514d11b`.
“Strong” means comparatively thorough **within the stated subset**, not full
family conformance. “Unreviewed” is a knowledge gap, not proof of missing code.
Test files below are evidence pointers; only the selected checks in the
[validation receipts](unification-validation.json) were rerun in this increment.

## Strongest bounded areas

| Kind of rule | What is relatively thorough | Where that confidence stops |
|---|---|---|
| Mana-pool arithmetic — CR 106/118 | 1,152 finite generic/specific payment cases with an independent counting oracle, plus 12 restricted-mana contexts. | Pool arithmetic is stronger than end-to-end payment. The retained Prism/Treasure failure prevents calling the whole payment system sound. |
| Lethal state-based actions — CR 704 | 192 toughness/damage/deathtouch/indestructible cases across both controllers; existing simultaneous-death and stabilization tests. | Does not cover every SBA, regeneration, poison or all replacement combinations. |
| Object identity — CR 400.7 | Leaving/reentering invalidates incarnations; source last-known information and an old target not following a returned creature have directed witnesses. | Stack targets using CardId and the full list of identity exceptions remain unproved. |
| Common combat keywords — CR 702 | Positive/negative cases for flying/reach, haste, vigilance, defender, trample, deathtouch, lifelink, menace, first/double strike; selected cross-keyword interactions. | This is a small keyword subset, not most of CR 702. Hexproof is inconsistent across target wordings. |

Evidence: [mana tests](../../../managym/tests/rules/cr_106_mana.rs),
[SBA tests](../../../managym/tests/rules/cr_704_sba.rs),
[identity tests](../../../managym/tests/rules/cr_400_identity.rs),
[trigger/identity witnesses](../../../managym/tests/rules/cr_603_triggers.rs),
[keyword tests](../../../managym/tests/rules/cr_702_keywords.rs).

## Working subsets with important limits

| Kind of rule | What exists | Gap or next question |
|---|---|---|
| Priority, turn steps and lands — CR 117/305/500–514 | Ordinary two-player priority, passing, main-phase land timing, land allowance reset, cleanup exceptions. | Broader special actions, extra/skipped turns and duration interactions are not established. |
| Casting and activated abilities — CR 601/602/605 | Staged modes/targets/payment, stacked activations and resumable optional costs. | Payment offer/execution disagreement remains retained evidence; unusual cost symbols and general mana-ability classification need review. Payment repair stays with existing owners. |
| Targets and resolution — CR 115/608 | Shared predicate, explicit mana-value boundaries, mixed/all-invalid targets and leave/reenter examples. | Confirmed hexproof bypass in `SpellOrPermanent`; not all targeting exceptions are reviewed. |
| Combat — CR 508–510 | Attacker subsets beyond a small flat list, multiple blockers, damage and strike steps. | General attack taxes, requirements/restrictions, all damage assignments and nonplayer defenders remain unresolved. |
| Trigger processing — CR 603 | Event matching, source snapshots, stack/response window, targets and APNAP ordering. | Confirmed absence of same-controller ordering choice; a survival-versus-loss witness shows its strategic consequence. |
| Replacements/prevention — CR 614–616 | Proposed-event pipeline; entry modifiers and selected prevention/doubling outcomes. | Deterministic replacement ordering is narrower than MTG's affected-player choices. General ordering/composition is not established. |
| Continuous effects — CR 611–613 | Authored additive P/T, selected characteristic-defining abilities, animation and keyword grants. | No general timestamp/dependency or text-changing engine; additive examples cannot establish noncommuting interactions. |

Evidence: [family reviews](reviews.json) supply scope, candidate code owners and
test paths. [Authority adjudication](unification.md) contains the newer exact
findings and counterexamples, including areas where no duplicate authority was
found. Earlier payment-owner identifiers in family notes are stale; consult
the refreshed ownership section before coordinating work.

## Large gaps and unadmitted territory

| Kind of rule | Current assessment | Consequence |
|---|---|---|
| Setup and format procedures — CR 100/103 | Fixed two-player setup and opening hand; no mulligan procedure or general tournament deck enforcement. | Useful authored-game setup does not establish normal tournament setup. |
| Counters, card parts and unusual types — CR 122/200–315 | +1/+1 counters and authored card shells; many characteristic exceptions, loyalty/defense procedures and unusual types unresolved. | A stored type tag or Scryfall shell match does not prove gameplay support. |
| Keyword actions/abilities and additional mechanics — CR 701–733 | Finite authored mechanics; most numbered obligations still unreviewed. | Determine support per mechanic and current card use; do not infer absence or correctness from names. |
| Alternate wins/losses and special zones | Ordinary life/draw losses have evidence; poison loss and ante zone were inspected as missing; broader alternate wins/loops unresolved. | Separate genuinely absent representation from unreviewed behavior. |
| Multiplayer and casual variants — CR 800–905 | Outside current admission, including two-player variants without a variant state machine. | Explicit product/format admission work, not defects in the ordinary two-player contract. |

The index contains **3,166 numbered rules/subrules**, plus 156 headings. After
the latest reviewed target entries, **2,970 total entries remain unknown**.
These counts measure review status, not a percentage of MTG implemented. The
original inventory was mostly family triage; a precise whole-rules coverage
percentage would be misleading.

## What this suggests doing next

Repair the demonstrated shared-targeting inconsistency first, with new-world
and admission evidence. Then restore missing player choices for trigger and
replacement ordering through the shared native decision surface. Continue
reviewing current-card interactions before expanding formats or building a
general engine for unused mechanics. Keep closing evidence gaps in strong
subsets too: agreement between consumers cannot rule out a shared rules bug.

Full detail stays in the [index](index.jsonl), [original evidence](evidence.md)
and [draft review guide](README.md). No new test run or conformance claim is
implied by this map.
