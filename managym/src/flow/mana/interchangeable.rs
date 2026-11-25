//! Manual payment may exchange two taps only when the rules cannot tell
//! which source stayed untapped. The affordability search has a weaker
//! equivalence: it only needs to know whether the mana can be produced.

use super::{Game, Mana, ManaSource, Obligation, PaymentModel, UsefulTap};
use crate::{
    flow::trigger::DelayedTriggerEvent,
    state::{game_object::Target, stack_object::StackObject},
};

impl Game {
    /// Every tap supplies exactly one unit. Removing a representative tests
    /// whether every member of its class is needed. Once those taps are
    /// accounted for, only one class can supply the balance unambiguously.
    pub(super) fn determined_source_counts<'a>(
        &self,
        model: &PaymentModel,
        pool: &Mana,
        obligations: &[Obligation],
        needed: usize,
        taps: Vec<UsefulTap<'a>>,
    ) -> Option<Vec<UsefulTap<'a>>> {
        let _scan = self.board_scan();
        let mut classes: Vec<Vec<UsefulTap>> = Vec::new();
        // Only unbound permanents can represent a class with multiple members.
        let mut representatives: Vec<usize> = Vec::new();
        for tap in taps {
            if self.unbound_mana_source(tap.source) {
                if let Some(&index) = representatives
                    .iter()
                    .find(|&&index| self.same_mana_permanent(classes[index][0].source, tap.source))
                {
                    classes[index].push(tap);
                    continue;
                }
                representatives.push(classes.len());
            }
            classes.push(vec![tap]);
        }

        let mut mandatory = Vec::new();
        let mut optional = None;
        for class in classes {
            if model.payable_after_tap(
                class[0].source.permanent,
                pool.clone(),
                obligations.to_vec(),
            ) {
                if optional.replace(class).is_some() {
                    return None;
                }
            } else {
                mandatory.extend(class);
            }
        }
        let count = needed.checked_sub(mandatory.len())?;
        let optional = optional?;
        mandatory.extend_from_slice(optional.get(..count)?);
        mandatory.sort_unstable_by_key(|tap| tap.source.permanent);
        Some(mandatory)
    }

    fn same_mana_permanent(&self, a: &ManaSource, b: &ManaSource) -> bool {
        let left = self.state.permanents[a.permanent].as_ref().unwrap();
        let right = self.state.permanents[b.permanent].as_ref().unwrap();
        let l = &self.state[a.object];
        let r = &self.state[b.object];
        let turn = self.state.turn.turn_number;
        l.owner == r.owner
            && l.kind() == r.kind()
            && l.definition_id == r.definition_id
            && l.form == r.form
            && left.controller == right.controller
            && left.summoning_sick == right.summoning_sick
            && left.entered_this_turn(turn) == right.entered_this_turn(turn)
            && left.spell_entry == right.spell_entry
            && a.same_payment_profile(b)
            && self.characteristics(a.permanent) == self.characteristics(b.permanent)
    }

    fn unbound_mana_source(&self, source: &ManaSource) -> bool {
        let state = &self.state;
        let id = source.permanent;
        let object = source.object;
        let permanent = state.permanents[id].as_ref().unwrap();
        if permanent.tapped
            || permanent.manifested
            || state[object].is_face_down()
            || permanent.attacking
            || permanent.damage != 0
            || permanent.dealt_damage_this_turn
            || permanent.deathtouch_damage
            || permanent.has_counters()
            || permanent.transforms != 0
            || permanent.attached_to.is_some()
            || !permanent.spent_abilities.is_empty()
            || !permanent.activated_this_turn.is_empty()
        {
            return false;
        }
        let target = Target::Permanent(id);
        let turn = &state.turn;
        if turn.attack_count(object) != 0
            || turn.used_trigger_grants.count(object) != 0
            || turn.damage_pairs.recipients(object).next().is_some()
            || turn.damage_pairs.combat_players(object).next().is_some()
            || turn
                .ability_triggers_this_turn
                .keys()
                .chain(turn.ability_resolutions_this_turn.keys())
                .any(|key| key.0 == object)
            || state.combat.as_ref().is_some_and(|combat| {
                combat.attackers.contains(&id)
                    || combat.blockers_to_declare.contains(&id)
                    || combat.attackers_to_declare.contains(&id)
                    || combat.first_strikers.contains(&id)
                    || combat
                        .attacker_to_blockers
                        .iter()
                        .any(|(a, bs)| *a == id || bs.contains(&id))
            })
        {
            return false;
        }

        if state
            .permanents
            .iter()
            .flatten()
            .any(|p| p.attached_to == Some(object))
            || state
                .effects
                .iter()
                .any(|e| e.affects_object(object) || e.source_object == Some(object))
            || state
                .cast_permissions
                .grants
                .iter()
                .any(|g| g.source == Some(object) || g.object == object)
            || state.exile_links.iter().any(|link| link.source == object)
            || self
                .paying()
                .is_some_and(|(_, payment, _)| payment.names_mana_source(id, object))
            || state.cast_caller.is_some()
            || state
                .suspended_decision
                .as_ref()
                .is_some_and(|s| s.may_name_mana_source(id, object))
        {
            return false;
        }

        if state.stack_objects.iter().any(|s| {
            s.targets().contains(&target)
                || s.source_object() == object
                || s.source_ref() == Some(object)
                || match s {
                    StackObject::Spell(spell) => spell.spent.names_object(object),
                    StackObject::ActivatedAbility(a) => {
                        a.attached == Some(object)
                            || a.granted_by.as_ref().is_some_and(|g| g.grantor == object)
                            || a.paid.names_object(object)
                    }
                    StackObject::TriggeredAbility(t) => {
                        t.context == Some(target)
                            || t.remembered.iter().any(|b| b.object == object)
                            || t.source_lki
                                .as_ref()
                                .is_some_and(|lki| lki.attached_to == Some(object))
                            || t.spent.names_object(object)
                    }
                }
        }) {
            return false;
        }

        let placement = state
            .trigger_placement
            .iter()
            .flat_map(|p| p.remaining.iter().chain(&p.placing));
        !state.pending_triggers.iter().chain(placement).any(|t| {
            t.source_object() == object || t.source_ref() == Some(object) || t.attacker == Some(id)
                || t.context == Some(target) || t.targets.contains(&target)
                || t.remembered.iter().any(|b| b.object == object)
                || t.source_lki.as_ref().is_some_and(|lki| lki.attached_to == Some(object))
                || t.spent.names_object(object)
        }) && !state.delayed_triggers.iter().any(|t| {
            t.source_object() == object || t.source_ref() == Some(object)
                || t.remembered.iter().any(|b| b.object == object)
                || matches!(t.event, DelayedTriggerEvent::Leaves { watched, .. } if watched == object)
        })
    }
}
