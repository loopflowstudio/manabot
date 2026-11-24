//! Copying spells and abilities on the stack (CR 707.10) and changing their
//! targets (CR 115.7).
//!
//! A copy is a new stack object with the original's choices; a copied spell
//! is a new `ObjectKind::SpellCopy` object that ceases to exist off the
//! stack (CR 704.5e), or becomes a token as a permanent spell resolves. Each
//! new or changed target is chosen through `Decision::Retarget`, one target
//! slot at a time, from the targeted object's controller's perspective.

use crate::{
    flow::{decision::Decision, game::Game},
    state::{
        ability::{effect_target_requirements, TargetRequirement},
        game_object::{AbilitySource, PlayerId, Target},
        stack_object::StackObject,
    },
};

impl Game {
    /// The targeting requirement a stack object's target slot was chosen for.
    pub(crate) fn stack_target_requirement(
        &self,
        index: usize,
        slot: usize,
    ) -> Option<TargetRequirement> {
        match &self.state.stack_objects[index] {
            StackObject::Spell(spell) => {
                let requirements =
                    self.state[spell.object].target_requirements_for_mode(spell.mode);
                requirements
                    .get(*spell.target_req_indices.get(slot)?)
                    .cloned()
            }
            StackObject::ActivatedAbility(ability) => effect_target_requirements(&ability.effects)
                .get(ability.target_req_indices.get(slot).copied().unwrap_or(0))
                .cloned(),
            StackObject::TriggeredAbility(triggered) => {
                let effects = match &triggered.inline_effects {
                    Some(effects) => effects.as_slice(),
                    None => super::resolution::stacked_trigger_face(triggered).abilities
                        [triggered.ability_index]
                        .effects(),
                };
                effect_target_requirements(effects)
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
        let (indices, x) = match stacked {
            StackObject::Spell(s) => (&s.target_req_indices, s.x),
            StackObject::ActivatedAbility(a) => (&a.target_req_indices, a.x),
            StackObject::TriggeredAbility(t) => (&t.target_req_indices, t.x),
        };
        let clause = |slot: usize| indices.get(slot).copied().unwrap_or(0);
        let current = targets.get(*slot).copied();
        let source = Some(AbilitySource::new(
            stacked.source_object(),
            stacked.source_ref(),
        ));
        let self_target = stacked.as_target();
        // Both distinctness and the shared mana-value budget apply only to
        // the other targets of this clause, never to the slot being replaced.
        let other_targets = targets
            .iter()
            .enumerate()
            .filter(|(other, _)| other != slot && clause(*other) == clause(*slot))
            .map(|(_, target)| *target);
        self.legal_clause_targets(&requirement, controller, source)
            .filter(|target| Some(*target) != current && *target != self_target)
            .filter(|target| {
                self.can_add_clause_target(&requirement, x, other_targets.clone(), *target)
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
        let id = self.state.id_gen_mut().next_id();
        let copy = match self.state.stack_objects[index].clone() {
            StackObject::Spell(mut spell) => {
                spell.object = self.create_spell_copy(spell.object, id, controller);
                spell.id = id;
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
