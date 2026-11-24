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

/// Activated abilities without an explicit slot mapping use their first clause.
fn requirement_index(stacked: &StackObject, slot: usize) -> Option<usize> {
    match stacked {
        StackObject::Spell(spell) => spell.target_req_indices.get(slot).copied(),
        StackObject::TriggeredAbility(trigger) => trigger.target_req_indices.get(slot).copied(),
        StackObject::ActivatedAbility(ability) => {
            Some(ability.target_req_indices.get(slot).copied().unwrap_or(0))
        }
    }
}

impl Game {
    /// The targeting requirement a stack object's target slot was chosen for.
    pub(crate) fn stack_target_requirement(
        &self,
        index: usize,
        slot: usize,
    ) -> Option<TargetRequirement> {
        let stacked = &self.state.stack_objects[index];
        let clause = requirement_index(stacked, slot)?;
        match stacked {
            StackObject::Spell(spell) => {
                let requirements =
                    self.state[spell.object].target_requirements_for_mode(spell.mode);
                requirements.get(clause).cloned()
            }
            StackObject::ActivatedAbility(ability) => effect_target_requirements(&ability.effects)
                .get(clause)
                .cloned(),
            StackObject::TriggeredAbility(triggered) => {
                effect_target_requirements(&self.stacked_trigger_effects(triggered)?)
                    .get(clause)
                    .cloned()
            }
        }
    }

    /// The targets slot `slot` of the stack object could change to: legal for
    /// its requirement from its controller's perspective, not its current
    /// target, not chosen for another slot of the same requirement, and not
    /// its own source when the requirement says "another".
    pub(crate) fn retarget_candidates(&self, decision: &Decision) -> Vec<Target> {
        match decision {
            Decision::Retarget { object, slot, .. } => self.retarget_candidates_for(*object, *slot),
            _ => Vec::new(),
        }
    }

    pub(crate) fn retarget_candidates_for(&self, object: Target, slot: usize) -> Vec<Target> {
        let Some(index) = self.stack_index_of(object) else {
            return Vec::new();
        };
        let Some(requirement) = self.stack_target_requirement(index, slot) else {
            return Vec::new();
        };
        let stacked = &self.state.stack_objects[index];
        let controller = stacked.controller();
        let targets = stacked.targets();
        let clause = requirement_index(stacked, slot);
        let x = stacked.x();
        let current = targets.get(slot).copied();
        let source = AbilitySource::new(stacked.source_object(), stacked.source_ref());
        let self_target = stacked.as_target();
        // Both distinctness and the shared mana-value budget apply only to
        // the other targets of this clause, never to the slot being replaced.
        let other_targets: Vec<Target> = targets
            .iter()
            .enumerate()
            .filter(|(other, _)| *other != slot && requirement_index(stacked, *other) == clause)
            .map(|(_, target)| *target)
            .collect();
        self.targets_for_source(&requirement, controller, source)
            .filter(|target| {
                Some(*target) != current
                    && *target != self_target
                    && !other_targets.contains(target)
                    && self.fits_target_clause(
                        &requirement,
                        x,
                        other_targets.iter().copied(),
                        *target,
                    )
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
            .find(|&slot| !self.retarget_candidates_for(object, slot).is_empty())
            .map(|slot| Decision::Retarget {
                player,
                object,
                slot,
                may_keep,
            })
    }

    /// Change slot `slot` of the stack object to `target`; the newly
    /// targeted permanent sees it become a target (ward, CR 702.21).
    pub(crate) fn change_stack_target(&mut self, object: Target, slot: usize, target: Target) {
        let Some(index) = self.stack_index_of(object) else {
            return;
        };
        let controller = self.state.stack_objects[index].controller();
        self.state.stack_objects[index].targets_mut()[slot] = target;
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
                // A stack copy was never cast using the original’s permission.
                spell.flashback = false;
                spell.from_prepared = false;
                spell.exile_instead_of_graveyard = false;
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
