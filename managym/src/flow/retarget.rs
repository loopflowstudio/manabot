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
        game_object::{Object, PlayerId, Target},
        stack_object::StackObject,
    },
};

impl Game {
    /// The targets slot `slot` of the stack object could change to: legal for
    /// its requirement from its controller's perspective, not its current
    /// target, not chosen for another slot of the same requirement, and not
    /// its own source when the requirement says "another".
    pub(crate) fn retarget_candidates(&self, object: Target, slot: usize) -> Vec<Target> {
        let Some(index) = self.stack_index_of(object) else {
            return Vec::new();
        };
        let stacked = &self.state.stack_objects[index];
        let targets = stacked.targets();
        let Some(&current) = targets.get(slot) else {
            return Vec::new();
        };
        let indices = stacked.target_req_indices();
        let clause = match indices.get(slot) {
            Some(&clause) => clause,
            // Single-clause activations store no indices.
            None if matches!(stacked, StackObject::ActivatedAbility(_)) => 0,
            None => return Vec::new(),
        };
        let requirements = self.stack_target_requirements(stacked);
        let Some(requirement) = requirements.get(clause) else {
            return Vec::new();
        };
        // The clause rules apply only to the other targets of this clause,
        // never to the slot being replaced.
        let other_targets = targets
            .iter()
            .enumerate()
            .filter(|(other, _)| {
                *other != slot && indices.get(*other).copied().unwrap_or(0) == clause
            })
            .map(|(_, target)| *target);
        self.legal_clause_targets(requirement, stacked.controller(), stacked.effect_source())
            .filter(|target| *target != current && *target != object)
            .filter(|target| {
                self.can_add_clause_target(requirement, stacked.x(), other_targets.clone(), *target)
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
            .find(|slot| !self.retarget_candidates(object, *slot).is_empty())
            .map(|slot| Decision::Retarget {
                player,
                object,
                slot,
                may_keep,
            })
    }

    /// A copy becomes a source of targeting events once its final targets have
    /// been chosen (CR 707.10c), including targets the controller kept.
    pub(crate) fn next_copy_target(
        &mut self,
        player: PlayerId,
        copy: Target,
        slot: usize,
    ) -> Option<Decision> {
        let next = self.next_retarget(player, copy, slot, true);
        if next.is_none() {
            self.emit_copy_targets(copy);
        }
        next
    }

    pub(crate) fn emit_copy_targets(&mut self, copy: Target) {
        let index = self.stack_index_of(copy).expect("newly created stack copy");
        let stacked = &self.state.stack_objects[index];
        let targets = stacked.targets().to_vec();
        let controller = stacked.controller();
        self.emit_permanents_targeted(&targets, copy, controller);
    }

    /// Copy a spell or ability on the stack under `controller` (CR 707.10);
    /// returns the copy as a target.
    pub(crate) fn copy_stack_object(
        &mut self,
        object: Target,
        controller: PlayerId,
    ) -> Option<Target> {
        let (original, departed) = match self.stack_index_of(object) {
            Some(index) => (self.state.stack_objects[index].clone(), None),
            None => {
                let Target::StackSpell(object) = object else {
                    return None;
                };
                let lki = self.state.spell_lki.get(&object)?;
                let saved = lki.copiable.as_ref()?;
                let source =
                    Object::from_copiable_values(saved.spell.id, controller, &saved.values);
                (StackObject::Spell(saved.spell.clone()), Some(source))
            }
        };
        let id = self.state.id_gen_mut().next_id();
        let copy = match original {
            StackObject::Spell(mut spell) => {
                let source = departed.unwrap_or_else(|| self.state[spell.object].clone());
                spell.object = self.create_spell_copy(source, id, controller);
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
