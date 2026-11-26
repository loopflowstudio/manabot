//! Manual payment may exchange two taps only when the rules cannot tell
//! which source stayed untapped. The affordability search has a weaker
//! equivalence: it only needs to know whether the mana can be produced.

use super::{Game, Mana, ManaCost, ManaSource, Obligation, PaymentModel, UsefulTap};
use crate::{
    flow::{statics::Characteristics, trigger::DelayedTriggerEvent},
    state::{
        ability::Effect,
        card::{CardType, StaticAbility},
        effect::Modification,
        game_object::Target,
        stack_object::StackObject,
    },
};

impl Game {
    /// Every tap supplies exactly one unit. Removing a representative tests
    /// whether every member of its class is needed. Once those taps are
    /// accounted for, only one class can supply the balance unambiguously.
    ///
    /// Checks cheap profile optionality before scanning board bindings or
    /// computing layered characteristics: for a determined count to exist,
    /// partitioning `taps` by payment profile must yield one optional profile
    /// of size `>= spare + 1` (with every other profile mandatory), and every
    /// member of that optional profile must be unbound and mutually
    /// interchangeable.
    pub(super) fn determined_source_counts<'a>(
        &self,
        model: &PaymentModel,
        pool: &Mana,
        cost: &ManaCost,
        needed: usize,
        taps: Vec<UsefulTap<'a>>,
        need_taps: bool,
    ) -> Option<Vec<UsefulTap<'a>>> {
        if self.state.cast_caller.is_some() {
            return None;
        }
        let tap_count = taps.len();
        let spare = tap_count.checked_sub(needed)?;
        if spare == 0 {
            return None;
        }
        // The unique optional class leaves `spare` untapped members and
        // contributes at least one tap (`count >= 1`), so it must have at
        // least `spare + 1` interchangeable members sharing a payment profile.
        let min_opt = spare + 1;
        let mut profile_groups: Vec<Vec<usize>> = Vec::new();
        for (index, tap) in taps.iter().enumerate() {
            if let Some(group) = profile_groups
                .iter_mut()
                .find(|group| taps[group[0]].source.same_payment_profile(tap.source))
            {
                group.push(index);
            } else {
                profile_groups.push(vec![index]);
            }
        }
        if !profile_groups.iter().any(|group| group.len() >= min_opt) {
            return None;
        }

        // Fast arithmetic affordability check when every source is fixed-only.
        let fixed_total = model
            .sources
            .iter()
            .all(|s| s.abilities.is_empty())
            .then(|| model.fixed_total(pool));
        let payable_without = |source: &ManaSource| -> bool {
            let obligations = || vec![Obligation::mana(cost.clone())];
            if let Some(total) = &fixed_total {
                let mut without = total.clone();
                if let Some(output) = &source.fixed {
                    without.spend(output);
                }
                let fast = without.can_pay(cost);
                debug_assert_eq!(
                    fast,
                    model.payable_after_tap(source.permanent, pool.clone(), obligations()),
                    "direct determined_source_counts differs from model"
                );
                fast
            } else {
                model.payable_after_tap(source.permanent, pool.clone(), obligations())
            }
        };

        // Check small (mandatory-candidate) groups first so a common board
        // like `Forest + Plains × 3` exits on its first arithmetic check.
        let mut optional: Option<Vec<usize>> = None;
        for group in profile_groups.iter().filter(|group| group.len() < min_opt) {
            if payable_without(taps[group[0]].source) {
                return None;
            }
        }
        for group in profile_groups
            .into_iter()
            .filter(|group| group.len() >= min_opt)
        {
            if payable_without(taps[group[0]].source) {
                if optional.replace(group).is_some() {
                    return None;
                }
            }
        }
        let opt_indices = optional?;

        // Every member of the unique optional profile must share cheap static
        // permanent identity and have no local/turn/combat binding.
        let opt_sources: Vec<&ManaSource> = opt_indices.iter().map(|&i| taps[i].source).collect();
        if !self.same_mana_identity_cheap(&opt_sources)
            || !opt_sources
                .iter()
                .all(|source| self.locally_unbound_mana_source(source))
        {
            return None;
        }

        // Hoist board-wide reference scans once across the candidate slice.
        if !self.all_unbound_board_wide(&opt_sources) {
            return None;
        }

        // Layered characteristics, summoning sickness, and power-up timing
        // run only after arithmetic and binding checks have passed.
        let _scan = self.board_scan();
        if !self.same_layered_mana_permanents(&opt_sources) {
            return None;
        }
        if !need_taps {
            return Some(Vec::new());
        }

        // `opt_indices.len() >= spare + 1`, so `1 <= count < opt_indices.len()`.
        let count = opt_indices.len() - spare;
        let skip = &opt_indices[count..];
        let mut result: Vec<_> = taps
            .into_iter()
            .enumerate()
            .filter_map(|(index, tap)| (!skip.contains(&index)).then_some(tap))
            .collect();
        result.sort_unstable_by_key(|tap| tap.source.permanent);
        Some(result)
    }

    fn same_mana_identity_cheap(&self, sources: &[&ManaSource]) -> bool {
        let Some((&first, rest)) = sources.split_first() else {
            return true;
        };
        let left = self.state.permanents[first.permanent].as_ref().unwrap();
        let l = &self.state[first.object];
        rest.iter().all(|other| {
            let right = self.state.permanents[other.permanent].as_ref().unwrap();
            let r = &self.state[other.object];
            l.owner == r.owner
                && l.kind() == r.kind()
                && l.definition_id == r.definition_id
                && l.form == r.form
                && left.controller == right.controller
                && left.spell_entry == right.spell_entry
        })
    }

    fn same_layered_mana_permanents(&self, sources: &[&ManaSource]) -> bool {
        let Some((&first, rest)) = sources.split_first() else {
            return true;
        };
        let left = self.state.permanents[first.permanent].as_ref().unwrap();
        let view = self.characteristics(first.permanent);
        let turn = self.state.turn.turn_number;
        let first_entered = left.entered_this_turn(turn);
        let mut sickness_matters: Option<bool> = None;
        let mut power_up_matters: Option<bool> = None;
        for other in rest {
            let right = self.state.permanents[other.permanent].as_ref().unwrap();
            let other_view = self.characteristics(other.permanent);
            if !std::sync::Arc::ptr_eq(&view, &other_view) && view != other_view {
                return false;
            }
            // Summoning sickness matters for creatures (CR 302.6), non-lands
            // (such as Vehicles or Spacecraft), and self-animating lands
            // (Soulstone Sanctuary, Treetop Village, Crawling Barrens). Earthbend
            // grants haste (CR 702.182a) and no catalog card animates another
            // land without haste, so plain lands played this turn are
            // interchangeable with earlier copies.
            if left.summoning_sick != right.summoning_sick {
                let matters = *sickness_matters.get_or_insert_with(|| {
                    view.is_creature() || !view.is_land() || can_become_creature(&view)
                });
                if matters {
                    return false;
                }
            }
            if first_entered != right.entered_this_turn(turn) {
                let matters = *power_up_matters.get_or_insert_with(|| {
                    view.activated_abilities
                        .iter()
                        .any(|ability| ability.power_up)
                });
                if matters {
                    return false;
                }
            }
        }
        true
    }

    fn locally_unbound_mana_source(&self, source: &ManaSource) -> bool {
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
            || permanent.loyalty_activated_this_turn
        {
            return false;
        }
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
                combat.attack_targets.iter().any(|&(attacker, target)| {
                    attacker == id
                        || target == Some(crate::flow::combat::AttackTarget::Permanent(object))
                })
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
        true
    }

    fn all_unbound_board_wide(&self, sources: &[&ManaSource]) -> bool {
        let state = &self.state;
        let has_object = |object| sources.iter().any(|s| s.object == object);
        let has_permanent = |id| sources.iter().any(|s| s.permanent == id);
        let has_target =
            |target: Target| matches!(target, Target::Permanent(id) if has_permanent(id));

        if let Some((_, payment, _)) = self.paying() {
            if sources
                .iter()
                .any(|s| payment.names_mana_source(s.permanent, s.object))
            {
                return false;
            }
        }
        if let Some(suspended) = &state.suspended_decision {
            if sources
                .iter()
                .any(|s| suspended.may_name_mana_source(s.permanent, s.object))
            {
                return false;
            }
        }
        if state
            .permanents
            .iter()
            .flatten()
            .any(|p| p.attached_to.is_some_and(&has_object))
        {
            return false;
        }
        if state.effects.iter().any(|e| {
            e.source_object.is_some_and(&has_object)
                || sources.iter().any(|s| e.affects_object(s.object))
        }) {
            return false;
        }
        if state
            .cast_permissions
            .grants
            .iter()
            .any(|g| g.source.is_some_and(&has_object) || has_object(g.object))
        {
            return false;
        }
        if state.exile_links.iter().any(|link| has_object(link.source)) {
            return false;
        }
        if state.stack_objects.iter().any(|s| {
            s.targets().iter().copied().any(&has_target)
                || has_object(s.source_object())
                || s.source_ref().is_some_and(&has_object)
                || match s {
                    StackObject::Spell(spell) => sources
                        .iter()
                        .any(|src| spell.spent.names_object(src.object)),
                    StackObject::ActivatedAbility(a) => {
                        a.attached.is_some_and(&has_object)
                            || a.granted_by.as_ref().is_some_and(|g| has_object(g.grantor))
                            || sources.iter().any(|src| a.paid.names_object(src.object))
                    }
                    StackObject::TriggeredAbility(t) => {
                        t.context.is_some_and(&has_target)
                            || t.remembered.iter().any(|b| has_object(b.object))
                            || t.source_lki
                                .as_ref()
                                .is_some_and(|lki| lki.attached_to.is_some_and(&has_object))
                            || sources.iter().any(|src| t.spent.names_object(src.object))
                    }
                }
        }) {
            return false;
        }
        let placement = state
            .trigger_placement
            .iter()
            .flat_map(|p| p.remaining.iter().chain(&p.placing));
        if state.pending_triggers.iter().chain(placement).any(|t| {
            has_object(t.source_object())
                || t.source_ref().is_some_and(&has_object)
                || t.attacker.is_some_and(&has_permanent)
                || t.context.is_some_and(&has_target)
                || t.targets.iter().copied().any(&has_target)
                || t.remembered.iter().any(|b| has_object(b.object))
                || t.source_lki
                    .as_ref()
                    .is_some_and(|lki| lki.attached_to.is_some_and(&has_object))
                || sources.iter().any(|src| t.spent.names_object(src.object))
        }) {
            return false;
        }
        if state.delayed_triggers.iter().any(|t| {
            has_object(t.source_object())
                || t.source_ref().is_some_and(&has_object)
                || t.remembered.iter().any(|b| has_object(b.object))
                || matches!(t.event, DelayedTriggerEvent::Leaves { watched, .. } if has_object(watched))
        }) {
            return false;
        }
        true
    }
}

fn can_become_creature(view: &Characteristics) -> bool {
    view.activated_abilities
        .iter()
        .any(|ability| ability.effects.iter().any(Effect::may_become_creature))
        || view.triggered_abilities.iter().any(|trigger| {
            trigger
                .ability
                .effects()
                .iter()
                .any(Effect::may_become_creature)
        })
        || view.static_abilities.iter().any(|ability| match ability {
            StaticAbility::Continuous(continuous) => continuous
                .modifications
                .iter()
                .any(Modification::grants_creature_type),
            StaticAbility::TypeGrant(grant) => grant.types.contains(&CardType::Creature),
            _ => false,
        })
}
