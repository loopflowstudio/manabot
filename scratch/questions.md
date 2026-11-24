# Open assumptions and deferred choices

Settled rationale for built slices lives in git history (this file before `realign` on 2026-11-24).

- `ObjectSelector::Zone(ZoneType)` (owner-scoped zone cards, kept for hidden-zone sampling) and `ZoneMatching { zone, owner: You, predicate }` overlap. Only `fin.rs:153` and `tle.rs:937` use `Zone` over a public zone. Folding them changes those cards' learning tokens, so it stays out of MTG-420; revisit if a third public-zone form appears.
- `can_add_clause_target` owns "that share a creature type". Only activation offers look ahead for a remaining partner while the minimum is unmet; no printed trigger or spell uses a shared-type clause yet (Secret Tunnel is an activation). Trigger-mode and activation availability count `min` candidates without the partner rule, matching `legal_spell_modes`.
- `Condition::AttachedCountersAtLeast` sits beside `SourceCountersAtLeast`; a general `CountersAtLeast { on }` would rename TLA learning tokens. Left until a third subject appears.
- `agent::observation_json::tests::historical_captures_move_battlefield_and_spell_rows_onto_their_records` fails at its last assertion on base `25ba418a2a` too. It belongs to the merged observation/legacy-input work; left for gate.
- CR 603.6a entry-attached Aura suppression: no catalog card triggers on an enchantment, Aura or any permanent entering, so `ZoneEntry.attached_to` has no card witness. Witness it when such a card is defined.
- `TriggerCondition::SourceDealsCombatDamage` is a separate variant rather than a `combat` field on `SourceDealsDamage` (which "parameters are fields" would prefer) because the field would change Cecil's serialized learning token. Revisit if a third source-damage trigger appears.
