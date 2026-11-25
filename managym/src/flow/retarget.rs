//! Copying spells and abilities on the stack (CR 707.10) and changing their
//! targets (CR 115.7).
//!
//! A copy is a new stack object with the original's choices; a copied spell
//! is a new `ObjectKind::SpellCopy` object that ceases to exist off the
//! stack (CR 704.5e), or becomes a token as a permanent spell resolves.
//!
//! New targets are chosen through `Decision::Retarget`, one slot at a time,
//! from the stack object's controller's perspective, but judged as one
//! assignment (CR 115.7e): an unchanged target may stay even if illegal,
//! and every change must be legal and leave each target that was legal
//! still legal. A slot offers a change, or keeping its target, only when
//! some choice for the later slots completes a legal assignment, so swaps
//! and coupled changes are reachable. The stack object keeps its original
//! targets until the last slot is decided; the final assignment and its
//! targeting events are then published together.

use crate::{
    flow::{decision::Decision, game::Game},
    state::{
        ability::TargetRequirement,
        game_object::{AbilitySource, Object, PlayerId, Target},
        stack_object::StackObject,
    },
};

/// A stack object's targets as a change of targets sees them.
struct TargetChange {
    original: Vec<Target>,
    /// The targeting clause of each slot.
    clauses: Vec<usize>,
    requirements: Vec<TargetRequirement>,
    x: u8,
    /// The legal objects each slot could change to, other than its own
    /// target and the stack object itself.
    candidates: Vec<Vec<Target>>,
    /// Whether each original target was legal in the original assignment.
    originally_legal: Vec<bool>,
}

/// What `Decision::Retarget` publishes once every slot is decided.
pub(crate) enum RetargetStep {
    Decide(Decision),
    Finished(Vec<Target>),
}

impl Game {
    fn target_change(&self, object: Target) -> Option<TargetChange> {
        let stacked = &self.state.stack_objects[self.stack_index_of(object)?];
        let original = stacked.targets().to_vec();
        let indices = stacked.target_req_indices();
        // Single-clause activations store no indices.
        let clauses: Vec<usize> = (0..original.len())
            .map(|slot| indices.get(slot).copied().unwrap_or(0))
            .collect();
        let requirements = self.stack_target_requirements(stacked);
        let controller = stacked.controller();
        let source: AbilitySource = stacked.effect_source();
        let candidates = original
            .iter()
            .zip(&clauses)
            .map(|(current, clause)| {
                requirements
                    .get(*clause)
                    .map_or_else(Vec::new, |requirement| {
                        self.legal_clause_targets(requirement, controller, source)
                            .filter(|target| target != current && *target != object)
                            .collect()
                    })
            })
            .collect();
        let mut change = TargetChange {
            original,
            clauses,
            requirements,
            x: stacked.x(),
            candidates,
            originally_legal: Vec::new(),
        };
        change.originally_legal = (0..change.original.len())
            .map(|slot| {
                change
                    .requirements
                    .get(change.clauses[slot])
                    .is_some_and(|requirement| {
                        self.target_fits_clause(
                            change.original[slot],
                            requirement,
                            controller,
                            source,
                        )
                    })
                    && self.slot_rules_hold(&change, &change.original, slot)
            })
            .collect();
        Some(change)
    }

    /// The clause rules and relation of `slot` against the other slots
    /// `assignment` already holds.
    fn slot_rules_hold(&self, change: &TargetChange, assignment: &[Target], slot: usize) -> bool {
        let clause = change.clauses[slot];
        let Some(requirement) = change.requirements.get(clause) else {
            return false;
        };
        let target = assignment[slot];
        let others = || {
            assignment
                .iter()
                .enumerate()
                .filter(move |(other, _)| *other != slot)
                .map(|(other, target)| (change.clauses[other], *target))
        };
        let mates = others()
            .filter(|(other, _)| *other == clause)
            .map(|(_, target)| target);
        self.can_add_clause_target(requirement, change.x, mates, target)
            && self.relation_admits(requirement, clause, target, others())
    }

    /// Whether the last slot of `assignment` is acceptable: a change obeys
    /// every rule, and an unchanged target that was legal stays legal.
    fn last_slot_acceptable(&self, change: &TargetChange, assignment: &[Target]) -> bool {
        let slot = assignment.len() - 1;
        (assignment[slot] == change.original[slot] && !change.originally_legal[slot])
            || self.slot_rules_hold(change, assignment, slot)
    }

    fn assignment_acceptable(&self, change: &TargetChange, assignment: &[Target]) -> bool {
        (1..=assignment.len()).all(|len| {
            let slot = len - 1;
            (assignment[slot] == change.original[slot] && !change.originally_legal[slot])
                || self.slot_rules_hold(change, assignment, slot)
        })
    }

    /// Whether some choice for the remaining slots completes `assignment`.
    /// Keeping is tried first, so the usual answer is found at once; each
    /// prefix is pruned by its last slot, as every rule is pairwise or
    /// grows with the clause.
    fn completes(&self, change: &TargetChange, assignment: &mut Vec<Target>) -> bool {
        let slot = assignment.len();
        if slot == change.original.len() {
            return self.assignment_acceptable(change, assignment);
        }
        for option in
            std::iter::once(change.original[slot]).chain(change.candidates[slot].iter().copied())
        {
            assignment.push(option);
            let found =
                self.last_slot_acceptable(change, assignment) && self.completes(change, assignment);
            assignment.pop();
            if found {
                return true;
            }
        }
        false
    }

    /// The changes slot `chosen.len()` admits after `chosen`. With
    /// `may_keep` ("choose new targets") the later slots may change too;
    /// otherwise ("change the target") they keep their targets.
    fn retarget_changes(
        &self,
        change: &TargetChange,
        chosen: &[Target],
        may_keep: bool,
    ) -> Vec<Target> {
        let slot = chosen.len();
        let Some(candidates) = change.candidates.get(slot) else {
            return Vec::new();
        };
        candidates
            .iter()
            .copied()
            .filter(|candidate| {
                let mut assignment: Vec<Target> =
                    chosen.iter().copied().chain([*candidate]).collect();
                if may_keep {
                    self.completes(change, &mut assignment)
                } else {
                    assignment.extend_from_slice(&change.original[slot + 1..]);
                    self.assignment_acceptable(change, &assignment)
                }
            })
            .collect()
    }

    fn keep_admits_completion(&self, change: &TargetChange, chosen: &[Target]) -> bool {
        let mut assignment: Vec<Target> = chosen.to_vec();
        assignment.push(change.original[chosen.len()]);
        self.last_slot_acceptable(change, &assignment) && self.completes(change, &mut assignment)
    }

    /// The targets slot `slot` of the stack object could change to while
    /// the earlier slots keep theirs and the later ones may change.
    pub(crate) fn retarget_candidates(&self, object: Target, slot: usize) -> Vec<Target> {
        let Some(change) = self.target_change(object) else {
            return Vec::new();
        };
        if slot >= change.original.len() {
            return Vec::new();
        }
        self.retarget_changes(&change, &change.original[..slot], true)
    }

    /// The actions of `Decision::Retarget` after `chosen`: the changes the
    /// next slot admits, and whether it may keep its target.
    pub(crate) fn retarget_offer(
        &self,
        object: Target,
        chosen: &[Target],
        may_keep: bool,
    ) -> (Vec<Target>, bool) {
        let Some(change) = self.target_change(object) else {
            return (Vec::new(), false);
        };
        if chosen.len() >= change.original.len() {
            return (Vec::new(), false);
        }
        let keep = may_keep && self.keep_admits_completion(&change, chosen);
        (self.retarget_changes(&change, chosen, may_keep), keep)
    }

    /// The next `Decision::Retarget` after `chosen`, keeping each slot that
    /// offers no change; or, with every slot decided, the final targets.
    /// "Change the target" (`may_keep` false) changes only the first slot
    /// that can change and keeps the rest.
    pub(crate) fn next_retarget(
        &self,
        player: PlayerId,
        object: Target,
        mut chosen: Vec<Target>,
        may_keep: bool,
    ) -> Option<RetargetStep> {
        let change = self.target_change(object)?;
        while chosen.len() < change.original.len() {
            if !self.retarget_changes(&change, &chosen, may_keep).is_empty() {
                return Some(RetargetStep::Decide(Decision::Retarget {
                    player,
                    object,
                    chosen,
                    may_keep,
                }));
            }
            chosen.push(change.original[chosen.len()]);
        }
        may_keep.then_some(RetargetStep::Finished(chosen))
    }

    /// Continue a copy's choice of new targets; once every slot is decided
    /// it publishes them and the copy becomes a source of targeting events
    /// (CR 707.10c), including targets the controller kept.
    pub(crate) fn next_copy_target(
        &mut self,
        player: PlayerId,
        copy: Target,
        chosen: Vec<Target>,
    ) -> Option<Decision> {
        match self.next_retarget(player, copy, chosen, true)? {
            RetargetStep::Decide(decision) => Some(decision),
            RetargetStep::Finished(targets) => {
                self.publish_targets(copy, targets, true);
                None
            }
        }
    }

    /// Continue after a slot is decided: "choose new targets" moves to the
    /// next slot; "change the target" keeps every later target.
    pub(crate) fn retarget_after(
        &mut self,
        player: PlayerId,
        object: Target,
        mut chosen: Vec<Target>,
        may_keep: bool,
    ) -> Option<Decision> {
        if may_keep {
            return self.next_copy_target(player, object, chosen);
        }
        let index = self
            .stack_index_of(object)
            .expect("retargeted stack object");
        chosen.extend_from_slice(&self.state.stack_objects[index].targets()[chosen.len()..]);
        self.publish_targets(object, chosen, false);
        None
    }

    /// Write the final targets of `object` and emit targeting events: for
    /// every target of a new copy, or for each changed target.
    pub(crate) fn publish_targets(&mut self, object: Target, targets: Vec<Target>, copy: bool) {
        let index = self
            .stack_index_of(object)
            .expect("retargeted stack object");
        let controller = self.state.stack_objects[index].controller();
        let slots = self.state.stack_objects[index].targets_mut();
        let targeted: Vec<Target> = targets
            .iter()
            .zip(slots.iter())
            .filter(|(new, old)| copy || new != old)
            .map(|(new, _)| *new)
            .collect();
        slots.copy_from_slice(&targets);
        self.emit_permanents_targeted(&targeted, object, controller);
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
