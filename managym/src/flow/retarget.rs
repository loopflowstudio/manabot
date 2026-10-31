//! Copying spells and abilities on the stack (CR 707.10) and changing their
//! targets (CR 115.7).
//!
//! A copy is a new stack object with the original's choices; a copied spell
//! is a new token card that ceases to exist off the stack (CR 704.5e). Each
//! new or changed target is chosen through `Decision::Retarget`, one target
//! slot at a time, from the targeted object's controller's perspective.

use crate::{
    flow::{decision::Decision, game::Game},
    state::{
        ability::{effect_target_requirements, Ability, TargetRequirement},
        game_object::{CardId, PlayerId, Target},
        stack_object::StackObject,
    },
};

impl Game {
    /// The stack index of a spell or ability target.
    pub(crate) fn stack_index_of(&self, object: Target) -> Option<usize> {
        match object {
            Target::StackSpell(card) => self.find_spell_on_stack_index(card),
            Target::StackAbility(id) => self.find_ability_on_stack_index(id),
            _ => None,
        }
    }

    /// The targeting requirement a stack object's target slot was chosen for.
    pub(crate) fn stack_target_requirement(
        &self,
        index: usize,
        slot: usize,
    ) -> Option<TargetRequirement> {
        match &self.state.stack_objects[index] {
            StackObject::Spell(spell) => {
                let requirements =
                    self.state.cards[spell.card].target_requirements_for_mode(spell.mode);
                requirements
                    .get(*spell.target_req_indices.get(slot)?)
                    .cloned()
            }
            StackObject::ActivatedAbility(ability) => {
                let definition = match ability.granted_by {
                    Some(grant) => self.state.cards[CardId::from(grant.grantor.entity)]
                        .attached_grants()
                        .nth(grant.index)?,
                    None => self.state.cards[ability.source_card]
                        .activated_abilities
                        .get(ability.ability_index)?,
                };
                definition.target_requirement()
            }
            StackObject::TriggeredAbility(triggered) => {
                let effects = match &triggered.inline_effects {
                    Some(effects) => effects.clone(),
                    None => {
                        let card = triggered
                            .source_lki
                            .as_ref()
                            .map_or(triggered.source_card, |lki| lki.card);
                        let Ability::Triggered { effects, .. } = self.state.cards[card]
                            .abilities
                            .get(triggered.ability_index)?;
                        effects.clone()
                    }
                };
                effect_target_requirements(&effects)
                    .get(*triggered.target_req_indices.get(slot)?)
                    .cloned()
            }
        }
    }

    /// The targets slot `slot` of the stack object could change to: legal for
    /// its requirement from its controller's perspective, not its current
    /// target, not chosen for another slot of the same requirement, and not
    /// its own source when the requirement says "another".
    pub(crate) fn retarget_candidates(&self, decision: &Decision) -> Vec<Target> {
        let Decision::Retarget { object, slot, .. } = decision else {
            return Vec::new();
        };
        let Some(index) = self.stack_index_of(*object) else {
            return Vec::new();
        };
        let Some(requirement) = self.stack_target_requirement(index, *slot) else {
            return Vec::new();
        };
        let stacked = &self.state.stack_objects[index];
        let controller = stacked.controller();
        let targets = stacked.targets();
        let indices: Vec<usize> = match stacked {
            StackObject::Spell(spell) => spell.target_req_indices.clone(),
            StackObject::TriggeredAbility(triggered) => triggered.target_req_indices.clone(),
            StackObject::ActivatedAbility(_) => vec![0; targets.len()],
        };
        let current = targets.get(*slot).copied();
        let source = stacked.source_ref();
        let source_card = stacked.source_card();
        let self_target = stacked.as_target();
        self.legal_targets_for_spec(&requirement.spec, controller)
            .into_iter()
            .filter(|target| Some(*target) != current && *target != self_target)
            .filter(|target| {
                !targets
                    .iter()
                    .zip(&indices)
                    .enumerate()
                    .any(|(other, (chosen, req))| {
                        other != *slot && *req == indices[*slot] && chosen == target
                    })
            })
            .filter(|target| {
                !requirement.another
                    || !matches!((target, source), (Target::Permanent(id), Some(source))
                        if self.permanent_object_ref(*id) == Some(source))
            })
            .filter(|target| {
                self.within_source_bound(*target, &requirement.spec, source_card, source)
            })
            .collect()
    }

    /// The first `Decision::Retarget` for `object` from `slot` on that has a
    /// candidate; `may_keep` lets each target stay.
    pub(crate) fn next_retarget(
        &self,
        player: PlayerId,
        object: Target,
        slot: usize,
        may_keep: bool,
    ) -> Option<Decision> {
        let index = self.stack_index_of(object)?;
        let count = self.state.stack_objects[index].targets().len();
        (slot..count)
            .map(|slot| Decision::Retarget {
                player,
                object,
                slot,
                may_keep,
            })
            .find(|decision| !self.retarget_candidates(decision).is_empty())
    }

    /// Change slot `slot` of the stack object to `target`; the newly
    /// targeted permanent sees it become a target (ward, CR 702.21).
    pub(crate) fn change_stack_target(&mut self, object: Target, slot: usize, target: Target) {
        let Some(index) = self.stack_index_of(object) else {
            return;
        };
        self.journal_stack();
        let controller = self.state.stack_objects[index].controller();
        match &mut self.state.stack_objects[index] {
            StackObject::Spell(spell) => spell.targets[slot] = target,
            StackObject::ActivatedAbility(ability) => ability.targets[slot] = target,
            StackObject::TriggeredAbility(triggered) => triggered.targets[slot] = target,
        }
        self.emit_permanents_targeted(&[target], object, controller);
    }

    /// Copy a spell or ability on the stack under `controller` (CR 707.10);
    /// returns the copy as a target.
    pub(crate) fn copy_stack_object(
        &mut self,
        object: Target,
        controller: PlayerId,
    ) -> Option<Target> {
        let index = self.stack_index_of(object)?;
        let id = self.state.id_gen.next_id();
        let copy = match self.state.stack_objects[index].clone() {
            StackObject::Spell(mut spell) => {
                let mut card = self.state.cards[spell.card].clone();
                card.id = id;
                card.owner = controller;
                card.is_token = true;
                let copied = CardId(self.state.cards.len());
                self.state.cards.push(card);
                self.state.card_to_permanent.push(None);
                self.state
                    .object_incarnations
                    .push(crate::state::game_object::Incarnation::INITIAL);
                self.journal_zone_move(copied);
                self.state
                    .zones
                    .move_card(copied, controller, crate::state::zone::ZoneType::Stack);
                spell.id = id;
                spell.card = copied;
                spell.controller = controller;
                StackObject::Spell(spell)
            }
            StackObject::ActivatedAbility(mut ability) => {
                ability.id = id;
                ability.controller = controller;
                StackObject::ActivatedAbility(ability)
            }
            StackObject::TriggeredAbility(mut triggered) => {
                triggered.id = id;
                triggered.controller = controller;
                StackObject::TriggeredAbility(triggered)
            }
        };
        let target = copy.as_target();
        self.push_to_stack(copy);
        Some(target)
    }
}
