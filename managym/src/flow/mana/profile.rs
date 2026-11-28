//! Board facts that determine whether a frozen payment model is sufficient.

use super::*;
use crate::state::card::{LayerValue, StaticAbility};
use crate::state::{effect::Modification, predicate::CardPredicate};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::flow) struct BoardArithmeticProfile {
    pub(super) triggers: bool,
    pub(super) stateful_triggers: bool,
    stateful_statics: bool,
    pub(super) writes_non_pt: bool,
    stateful_abilities: bool,
    damage_replacements: bool,
    life_replacements: bool,
    counter_or_zone_replacements: bool,
    pub(in crate::flow) reads_tapped: bool,
    pub(in crate::flow) reads_mana: bool,
    pub(in crate::flow) reads_life: bool,
}

impl BoardArithmeticProfile {
    pub(in crate::flow) fn of(view: &Characteristics, replacements: &[ReplacementEffect]) -> Self {
        let mut profile = Self {
            triggers: !view.triggered_mana_abilities.is_empty(),
            stateful_triggers: view.triggered_mana_abilities.iter().any(|trigger| {
                trigger.predicate
                    != (CardPredicate {
                        card_type: trigger.predicate.card_type,
                        ..Default::default()
                    })
            }),
            stateful_abilities: view.activated_abilities.iter().any(|ability| {
                ability.is_mana_ability()
                    && (ability.has_dynamic_mana_output()
                        || ability.condition.is_some()
                        || ability.cost_reduction_per.is_some()
                        || ability.reduce_by_source_power
                        || ability.tap_cost.is_some())
            }),
            ..Default::default()
        };
        if profile.stateful_abilities {
            for ability in &view.activated_abilities {
                let reads = crate::flow::layers::mana_ability_reads(ability);
                profile.reads_mana |= reads.mana;
                profile.reads_tapped |= reads.tapped;
                profile.reads_life |= reads.life;
            }
        }
        for ability in &view.static_abilities {
            profile.add_static(ability);
        }
        for replacement in replacements {
            profile.add_replacement(replacement);
        }
        profile
    }

    pub(in crate::flow) fn add_static(&mut self, ability: &StaticAbility) {
        match ability {
            StaticAbility::Continuous(effect) => {
                let pure_pt = effect.modifications.iter().all(|part| {
                    matches!(
                        part,
                        Modification::ModifyPt {
                            power: LayerValue::Fixed(_),
                            toughness: LayerValue::Fixed(_)
                        } | Modification::SetBasePt {
                            power: None | Some(LayerValue::Fixed(_)),
                            toughness: None | Some(LayerValue::Fixed(_)),
                        } | Modification::SwitchPt
                    )
                });
                self.stateful_statics |= effect.condition.is_some() || !pure_pt;
                self.writes_non_pt |= effect.modifications.iter().any(|part| {
                    !matches!(
                        part,
                        Modification::ModifyPt { .. }
                            | Modification::SetBasePt { .. }
                            | Modification::SwitchPt
                    )
                });
            }
            StaticAbility::ActivatedAbilitiesCostLess { predicate, .. } => {
                self.stateful_statics = true;
                self.writes_non_pt = true;
                self.reads_tapped |= crate::flow::layers::payment_predicate_reads(predicate).tapped;
            }
            StaticAbility::Ward(_)
            | StaticAbility::HexproofFrom(_)
            | StaticAbility::Protection(_)
            | StaticAbility::CostsLessByGreatestPower(_)
            | StaticAbility::AdditionalLandPlay
            | StaticAbility::AdditionalLandPlayEachPlayer => {}
            _ => {
                self.stateful_statics = true;
                self.writes_non_pt = true;
            }
        }
    }

    pub(in crate::flow) fn add_replacement(&mut self, replacement: &ReplacementEffect) {
        match replacement {
            ReplacementEffect::ModifyDamage { .. } => self.damage_replacements = true,
            ReplacementEffect::ModifyLifeGain { .. } => self.life_replacements = true,
            ReplacementEffect::ExileIfWouldDie { .. }
            | ReplacementEffect::GraveyardInstead { .. }
            | ReplacementEffect::ModifyCounters { .. } => self.counter_or_zone_replacements = true,
            ReplacementEffect::AsEntersChoose { .. }
            | ReplacementEffect::AdventureExile
            | ReplacementEffect::DiscardedInstead { .. }
            | ReplacementEffect::EntersPrepared
            | ReplacementEffect::PreventCombatDamage
            | ReplacementEffect::EntersTapped
            | ReplacementEffect::EntersTappedUnlessControls { .. }
            | ReplacementEffect::EntersWithCounters { .. }
            | ReplacementEffect::UnspentManaBecomesRed
            | ReplacementEffect::MultiplyTokens { .. }
            | ReplacementEffect::EntersAsCopy { .. } => {}
        }
    }

    pub(in crate::flow) fn add_effect(&mut self, modification: &Modification) {
        match modification {
            Modification::ExileIfWouldDie | Modification::ExileIfWouldLeave => {
                self.counter_or_zone_replacements = true;
            }
            _ => {}
        }
    }

    pub(in crate::flow) fn merge(&mut self, other: Self) {
        self.triggers |= other.triggers;
        self.stateful_triggers |= other.stateful_triggers;
        self.stateful_statics |= other.stateful_statics;
        self.writes_non_pt |= other.writes_non_pt;
        self.stateful_abilities |= other.stateful_abilities;
        self.damage_replacements |= other.damage_replacements;
        self.life_replacements |= other.life_replacements;
        self.counter_or_zone_replacements |= other.counter_or_zone_replacements;
        self.reads_tapped |= other.reads_tapped;
        self.reads_mana |= other.reads_mana;
        self.reads_life |= other.reads_life;
    }

    pub(super) fn stateful_tap(self) -> bool {
        self.reads_tapped || self.reads_mana
    }

    pub(super) fn sensitive_tap(self) -> bool {
        self.stateful_triggers || self.writes_non_pt
    }

    pub(super) fn stateful_delve(self) -> bool {
        self.stateful_abilities || self.stateful_statics || self.stateful_triggers
    }

    pub(super) fn activation_mutates(self, facts: ActivationFacts, help: Help) -> bool {
        facts.counters
            || facts.unverified_suffix
            || facts.life
                && (self.reads_life
                    || self.stateful_statics
                    || self.stateful_triggers
                    || self.damage_replacements
                    || self.life_replacements)
            || facts.sacrifices
                && (facts.rule_source
                    || self.stateful_abilities
                    || self.stateful_statics
                    || self.stateful_triggers
                    || self.counter_or_zone_replacements
                    || help.delve.is_some())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct ActivationFacts {
    pub(super) counters: bool,
    pub(super) life: bool,
    pub(super) sacrifices: bool,
    pub(super) rule_source: bool,
    pub(super) unverified_suffix: bool,
    pub(super) reads: bool,
}
