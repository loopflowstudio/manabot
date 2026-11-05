//! Rebind retained private selections to a sampled world's zone membership.
//!
//! The selected count and predicate are commitments, not disposable offers.
//! One exact-object mapping covers active decisions, saved callers and inherited
//! child bindings so the same selection cannot acquire two meanings on resume.

use std::collections::BTreeMap;

use rand::{seq::SliceRandom, Rng};
use rand_chacha::ChaCha8Rng;

use crate::{
    flow::{
        decision::{Decision, EffectFrame, FrameCompletion, ResolutionRef, SuspendedResolution},
        game::Game,
    },
    state::{
        ability::ObjectSelector,
        binding::{ObjectBinding, ZoneSelection},
        game_object::{CardId, ObjectRef, PlayerId},
        stack_object::StackObject,
        zone::ZoneType,
    },
};

pub(crate) type Requirements = BTreeMap<ObjectRef, Vec<ZoneSelection>>;

/// One name allocation to face-down slots and its ordered physical-copy count.
/// Retained selections condition this measure by feasibility, not by the number
/// of ways a player could have chosen the private subset.
struct ForetoldAllocation {
    counts: BTreeMap<String, u32>,
    ways: u128,
}

trait SelectionSampling {
    fn hidden_from(&self, game: &Game, viewer: PlayerId) -> bool;
    fn matches_slot(&self, game: &Game, owner: PlayerId, zone: ZoneType, card: CardId) -> bool;
}

impl SelectionSampling for ZoneSelection {
    fn hidden_from(&self, game: &Game, viewer: PlayerId) -> bool {
        matches!(self.zone, ZoneType::Hand | ZoneType::Library)
            && !game.shares_information(viewer, self.owner)
    }

    fn matches_slot(&self, game: &Game, owner: PlayerId, zone: ZoneType, card: CardId) -> bool {
        self.owner == owner
            && self.zone == zone
            && game.card_matches_predicate(card, &self.predicate)
    }
}

impl Game {
    /// Capture current membership once. Re-capturing Remembered preserves its
    /// original knowledge, while Moved starts from the new incarnation/zone.
    pub(crate) fn capture_bindings(
        &self,
        what: &ObjectSelector,
        frame: &EffectFrame,
    ) -> Vec<ObjectBinding> {
        self.select(what, frame)
            .into_iter()
            .map(|object| {
                if let Some(binding) = frame
                    .remembered
                    .iter()
                    .find(|binding| binding.object == object)
                {
                    let mut binding = binding.clone();
                    for viewer in [PlayerId(0), PlayerId(1)] {
                        binding.known_to[viewer.0] |=
                            self.visible_reference(object, viewer).object.is_some();
                    }
                    return binding;
                }
                let card = CardId::from(object.entity);
                let zone = self.state.zones.zone_of(card);
                let selection = zone
                    .filter(|zone| matches!(zone, ZoneType::Hand | ZoneType::Library))
                    .map(|zone| ZoneSelection {
                        owner: self.state.cards[card].owner,
                        zone,
                        predicate: frame
                            .selected_from
                            .as_ref()
                            .filter(|source| {
                                frame.selected.contains(&object)
                                    && source.zone == zone
                                    && self.card_matches_predicate(card, &source.predicate)
                            })
                            .map(|source| source.predicate.clone())
                            .unwrap_or_default(),
                    });
                ObjectBinding {
                    object,
                    selection,
                    known_to: [PlayerId(0), PlayerId(1)].map(|viewer| {
                        self.visible_reference(object, viewer).object.is_some()
                            || frame.selected_from.as_ref().is_some_and(|source| {
                                self.shares_information(viewer, source.owner)
                                    && frame
                                        .selected
                                        .iter()
                                        .any(|selected| selected.entity == object.entity)
                            })
                    }),
                }
            })
            .collect()
    }

    /// All owners of a captured binding, including a detached placement round.
    pub(crate) fn visit_captured_bindings(&self, mut visit: impl FnMut(&ObjectBinding)) {
        for trigger in &self.state.delayed_triggers {
            trigger.remembered.iter().for_each(&mut visit);
        }
        for trigger in &self.state.pending_triggers {
            trigger.remembered.iter().for_each(&mut visit);
        }
        if let Some(placement) = &self.state.trigger_placement {
            for trigger in placement.remaining.iter().chain(placement.placing.iter()) {
                trigger.remembered.iter().for_each(&mut visit);
            }
        }
        for object in &self.state.stack_objects {
            if let StackObject::TriggeredAbility(trigger) = object {
                trigger.remembered.iter().for_each(&mut visit);
            }
        }
        if let Some(suspended) = &self.state.suspended_decision {
            for part in suspended.continuation() {
                if let ResolutionRef::Frame(frame) = part {
                    frame.remembered.iter().for_each(&mut visit);
                }
            }
        }
    }

    /// Exact objects already known to this viewer must not be exchanged for a
    /// same-definition copy. Departed incarnations impose no current-zone slot.
    pub(crate) fn sampling_fixed_objects(&self, viewer: PlayerId) -> Vec<ObjectRef> {
        let mut fixed = self.pending_public_entries();
        self.visit_captured_bindings(|binding| {
            if binding.known_to[viewer.0]
                && self.current_object_ref(binding.object.entity.into()) == Some(binding.object)
            {
                fixed.push(binding.object);
            }
        });
        fixed.sort_unstable();
        fixed.dedup();
        fixed
    }

    /// Prospective battlefield entries are public even while their move awaits
    /// replacement choices. Neither sampler may exchange their exact objects.
    pub(crate) fn pending_public_entries(&self) -> Vec<ObjectRef> {
        let mut entries = Vec::new();
        if let Some(suspended) = &self.state.suspended_decision {
            for part in suspended.continuation() {
                if let ResolutionRef::Frame(frame) = part {
                    if let Some(batch) = &frame.batch {
                        batch.public_entries(&mut entries);
                    }
                }
            }
        }
        entries.retain(|object| self.current_object_ref(object.entity.into()) == Some(*object));
        entries.sort_unstable();
        entries.dedup();
        entries
    }

    pub(crate) fn sampling_fixed_library_cards(
        &self,
        owner: PlayerId,
        viewer: PlayerId,
    ) -> Vec<CardId> {
        self.sampling_fixed_objects(viewer)
            .into_iter()
            .map(|object| object.entity.into())
            .filter(|card| self.state.zones.contains(*card, ZoneType::Library, owner))
            .collect()
    }

    /// Exact ordered foretold fills that leave the selected subsets realizable.
    /// Unlike a feasibility-only filter, this removes the mass of impossible
    /// foretold assignments within an otherwise compatible hand-count world.
    pub(crate) fn selection_foretold_weight(
        &self,
        viewer: PlayerId,
        hand: &BTreeMap<String, u32>,
    ) -> u128 {
        self.selection_foretold_allocations(viewer, hand)
            .iter()
            .map(|allocation| allocation.ways)
            .sum()
    }

    /// Order one compatible foretold fill before the remaining shuffled pool.
    /// The existing slot mover still owns grants and exact exile membership.
    pub(crate) fn prepare_selection_foretell(
        &self,
        viewer: PlayerId,
        hand: &BTreeMap<String, u32>,
        pool: &mut Vec<CardId>,
        rng: &mut ChaCha8Rng,
    ) -> bool {
        let allocations = self.selection_foretold_allocations(viewer, hand);
        let total: u128 = allocations.iter().map(|allocation| allocation.ways).sum();
        if total == 0 {
            return false;
        }
        let mut draw = rng.gen_range(0..total);
        let allocation = allocations
            .into_iter()
            .find(|allocation| {
                if draw < allocation.ways {
                    true
                } else {
                    draw -= allocation.ways;
                    false
                }
            })
            .expect("draw is within the compatible measure");
        let mut counts = allocation.counts;
        let mut chosen = Vec::new();
        pool.retain(|card| {
            let Some(count) = counts.get_mut(&self.state.cards[*card].name) else {
                return true;
            };
            if *count == 0 {
                return true;
            }
            *count -= 1;
            chosen.push(*card);
            false
        });
        debug_assert!(counts.values().all(|count| *count == 0));
        chosen.shuffle(rng);
        chosen.append(pool);
        *pool = chosen;
        true
    }

    fn selection_foretold_allocations(
        &self,
        viewer: PlayerId,
        hand: &BTreeMap<String, u32>,
    ) -> Vec<ForetoldAllocation> {
        let requirements = self.hidden_selection_requirements(viewer);
        let owner = PlayerId(1 - viewer.0);
        let mut pool = BTreeMap::<String, Vec<CardId>>::new();
        for card in self.unseen_cards(owner, viewer) {
            pool.entry(self.state.cards[card].name.clone())
                .or_default()
                .push(card);
        }
        if hand
            .iter()
            .any(|(name, count)| *count as usize > pool.get(name).map_or(0, Vec::len))
        {
            return Vec::new();
        }
        let bearers: Vec<_> = pool
            .iter()
            .filter(|(_, cards)| self.state.cards[cards[0]].foretell().is_some())
            .map(|(name, cards)| {
                (
                    name.clone(),
                    cards.len() as u32 - hand.get(name).copied().unwrap_or(0),
                )
            })
            .collect();
        let public = self.sampling_fixed_objects(viewer);
        let count = self.foretold_cards(owner).len() as u32;
        let mut allocations = Vec::new();
        enumerate_foretold(&bearers, count, 1, &mut BTreeMap::new(), &mut allocations);
        let orders: u128 = (1..=count).map(u128::from).product();
        allocations.retain_mut(|allocation| {
            let mut slots = Vec::new();
            for (name, cards) in &pool {
                let in_hand = hand.get(name).copied().unwrap_or(0) as usize;
                let foretold = allocation.counts.get(name).copied().unwrap_or(0) as usize;
                let pinned_hand = cards
                    .iter()
                    .filter(|card| {
                        self.state.zones.contains(**card, ZoneType::Hand, owner)
                            && public.iter().any(|object| object.entity == (**card).into())
                    })
                    .count();
                for index in pinned_hand..cards.len() - foretold {
                    slots.push((
                        if index < in_hand {
                            ZoneType::Hand
                        } else {
                            ZoneType::Library
                        },
                        cards[0],
                    ));
                }
            }
            let mut candidates: Vec<Vec<usize>> = requirements
                .values()
                .map(|constraints| {
                    slots
                        .iter()
                        .enumerate()
                        .filter(|(_, (zone, card))| {
                            constraints
                                .iter()
                                .all(|source| source.matches_slot(self, owner, *zone, *card))
                        })
                        .map(|(index, _)| index)
                        .collect()
                })
                .collect();
            if let Some(suspended) = &self.state.suspended_decision {
                for decision in suspended.decisions() {
                    if let Decision::SelectCards {
                        from: Some(source),
                        min,
                        selected,
                        ..
                    } = decision
                    {
                        if source.owner == owner
                            && matches!(source.zone, ZoneType::Hand | ZoneType::Library)
                        {
                            let eligible: Vec<_> = slots
                                .iter()
                                .enumerate()
                                .filter(|(_, (zone, card))| {
                                    source.zone == *zone
                                        && self.card_matches_predicate(*card, &source.predicate)
                                })
                                .map(|(index, _)| index)
                                .collect();
                            for _ in selected.len()..*min {
                                candidates.push(eligible.clone());
                            }
                        }
                    }
                }
            }
            allocation.ways *= orders;
            complete_assignment(&candidates).is_some()
        });
        allocations
    }

    pub(crate) fn hidden_selection_requirements(&self, viewer: PlayerId) -> Requirements {
        let mut requirements = Requirements::new();
        let public = self.sampling_fixed_objects(viewer);
        let mut retain = |source: &Option<ZoneSelection>, selected: &[ObjectRef]| {
            let Some(source) = source
                .as_ref()
                .filter(|source| source.hidden_from(self, viewer))
            else {
                return;
            };
            for object in selected {
                let card = object.entity.into();
                // Moved selections keep their old incarnation and no longer
                // bind a hidden source slot. Public results must never be sampled.
                if self.current_object_ref(card) == Some(*object)
                    && !public.contains(object)
                    && matches!(
                        self.state.zones.zone_of(card),
                        Some(ZoneType::Hand | ZoneType::Library)
                    )
                {
                    // Inherited child bindings repeat the same constraint;
                    // retain each requirement once for the shared assignment.
                    let constraints = requirements.entry(*object).or_default();
                    if !constraints.contains(source) {
                        constraints.push(source.clone());
                    }
                }
            }
        };
        if let Some(suspended) = &self.state.suspended_decision {
            for part in suspended.continuation() {
                match part {
                    ResolutionRef::Frame(frame) => retain(&frame.selected_from, &frame.selected),
                    ResolutionRef::Decision(Decision::SelectCards { from, selected, .. }) => {
                        retain(from, selected)
                    }
                    _ => {}
                }
            }
        }
        // Known exact captures are already excluded by the fixed-object set,
        // including when another owner holds an unknown alias of that capture.
        self.visit_captured_bindings(|binding| {
            retain(&binding.selection, std::slice::from_ref(&binding.object));
        });
        requirements
    }

    pub(crate) fn has_hidden_selection(&self, viewer: PlayerId) -> bool {
        !self.hidden_selection_requirements(viewer).is_empty()
            || self
                .state
                .suspended_decision
                .as_ref()
                .is_some_and(|suspended| {
                    suspended.decisions().any(|decision| {
                        matches!(decision, Decision::SelectCards { from: Some(source), min, .. }
                        if *min > 0 && source.hidden_from(self, viewer))
                    })
                })
    }

    /// False means this deal cannot honor earlier choices. The caller rejects
    /// the world; neither selected counts nor public commitments are truncated.
    pub(crate) fn refresh_hidden_selections(
        &mut self,
        viewer: PlayerId,
        rng: &mut ChaCha8Rng,
        requirements: &Requirements,
    ) -> bool {
        if self
            .state
            .suspended_decision
            .as_ref()
            .is_some_and(|suspended| {
                suspended.decisions().any(|decision| match decision {
                    Decision::SelectCards {
                        from: Some(source),
                        min,
                        ..
                    } => {
                        self.state
                            .zones
                            .zone_cards(source.zone, source.owner)
                            .iter()
                            .filter(|card| self.card_matches_predicate(**card, &source.predicate))
                            .count()
                            < *min
                    }
                    _ => false,
                })
            })
        {
            return false;
        }
        let public = self.sampling_fixed_objects(viewer);
        let old: Vec<_> = requirements.keys().copied().collect();
        let mut candidates = Vec::with_capacity(old.len());
        for constraints in requirements.values() {
            let source = &constraints[0];
            let mut eligible: Vec<_> = self
                .state
                .zones
                .zone_cards(source.zone, source.owner)
                .iter()
                .copied()
                .filter(|card| {
                    constraints.iter().all(|constraint| {
                        constraint.matches_slot(self, source.owner, source.zone, *card)
                    })
                })
                .filter_map(|card| self.current_object_ref(card))
                .filter(|object| !public.contains(object))
                .collect();
            eligible.shuffle(rng);
            candidates.push(eligible);
        }
        let Some(assigned) = complete_assignment(&candidates) else {
            return false;
        };
        let mapping: BTreeMap<_, _> = assigned
            .into_iter()
            .map(|(object, index)| (old[index], object))
            .collect();
        if !mapping.is_empty() {
            for trigger in &mut self.state.delayed_triggers {
                rebind_captures(&mut trigger.remembered, &mapping);
            }
            for trigger in &mut self.state.pending_triggers {
                rebind_captures(&mut trigger.remembered, &mapping);
            }
            if let Some(placement) = &mut self.state.trigger_placement {
                for trigger in placement
                    .remaining
                    .iter_mut()
                    .chain(placement.placing.iter_mut())
                {
                    rebind_captures(&mut trigger.remembered, &mapping);
                }
            }
            for object in &mut self.state.stack_objects {
                if let StackObject::TriggeredAbility(trigger) = object {
                    rebind_captures(&mut trigger.remembered, &mapping);
                }
            }
        }
        if let Some(mut suspended) = self.state.suspended_decision_mut().take() {
            self.rebind_suspended_selection(&mut suspended, viewer, &mapping);
            *self.state.suspended_decision_mut() = Some(suspended);
        }
        true
    }

    fn rebind_suspended_selection(
        &self,
        suspended: &mut SuspendedResolution,
        viewer: PlayerId,
        mapping: &BTreeMap<ObjectRef, ObjectRef>,
    ) {
        self.rebind_frame_selection(&mut suspended.frame, viewer, mapping);
        self.rebind_decision_selection(&mut suspended.decision, viewer, mapping);
    }

    fn rebind_frame_selection(
        &self,
        frame: &mut EffectFrame,
        viewer: PlayerId,
        mapping: &BTreeMap<ObjectRef, ObjectRef>,
    ) {
        // Eligibility belongs to the joint mapping, not this alias's selector.
        // A selection from Remembered has no zone marker of its own.
        rebind(&mut frame.selected, mapping);
        // A child may remember or otherwise alias a selected exact object.
        // The one mapping must follow those aliases as well as `Selected`.
        rebind_captures(&mut frame.remembered, mapping);
        rebind(&mut frame.moved, mapping);
        rebind(&mut frame.this_way, mapping);
        self.rebind_completion_selection(&mut frame.completion, viewer, mapping);
    }

    fn rebind_completion_selection(
        &self,
        completion: &mut FrameCompletion,
        viewer: PlayerId,
        mapping: &BTreeMap<ObjectRef, ObjectRef>,
    ) {
        match completion {
            FrameCompletion::Frame(parent) | FrameCompletion::Event { frame: parent, .. } => {
                self.rebind_frame_selection(parent, viewer, mapping)
            }
            FrameCompletion::Decision(parent) => {
                self.rebind_suspended_selection(parent, viewer, mapping)
            }
            FrameCompletion::StateBased { next } => {
                self.rebind_completion_selection(next, viewer, mapping)
            }
            _ => {}
        }
    }

    fn rebind_decision_selection(
        &self,
        decision: &mut Decision,
        viewer: PlayerId,
        mapping: &BTreeMap<ObjectRef, ObjectRef>,
    ) {
        match decision {
            Decision::SelectCards {
                from,
                candidates,
                selected,
                ..
            } => {
                rebind(selected, mapping);
                if let Some(source) = from
                    .as_ref()
                    .filter(|source| source.hidden_from(self, viewer))
                {
                    *candidates = self
                        .selection_zone_cards(source.owner, source.zone)
                        .into_iter()
                        .filter(|card| self.card_matches_predicate(*card, &source.predicate))
                        .filter_map(|card| self.current_object_ref(card))
                        .collect();
                } else {
                    rebind(candidates, mapping);
                }
            }
            Decision::ManaUpfront {
                resume: Some(parent),
                ..
            }
            | Decision::ManaChoice {
                resume: Some(parent),
                ..
            } => self.rebind_decision_selection(parent, viewer, mapping),
            _ => {}
        }
    }
}

// Bipartite matching preserves overlapping predicates without a greedy
// choice rejecting a deal that has a complete assignment. Both count-domain
// admission and exact-object rebinding use this same matching rule.
fn complete_assignment<T: Copy + Ord>(candidates: &[Vec<T>]) -> Option<BTreeMap<T, usize>> {
    let mut assigned = BTreeMap::new();
    for index in 0..candidates.len() {
        if !assign(index, candidates, &mut assigned, &mut Vec::new()) {
            return None;
        }
    }
    Some(assigned)
}

fn assign<T: Copy + Ord>(
    index: usize,
    candidates: &[Vec<T>],
    assigned: &mut BTreeMap<T, usize>,
    seen: &mut Vec<T>,
) -> bool {
    for object in &candidates[index] {
        if seen.contains(object) {
            continue;
        }
        seen.push(*object);
        if assigned
            .get(object)
            .copied()
            .is_none_or(|previous| assign(previous, candidates, assigned, seen))
        {
            assigned.insert(*object, index);
            return true;
        }
    }
    false
}

fn rebind(objects: &mut [ObjectRef], mapping: &BTreeMap<ObjectRef, ObjectRef>) {
    for object in objects {
        if let Some(replacement) = mapping.get(object) {
            *object = *replacement;
        }
    }
}

/// Enumerate name allocations, counting physical subsets without enumerating
/// their permutations. Slot order contributes the factorial once, at the caller.
fn enumerate_foretold(
    names: &[(String, u32)],
    remaining: u32,
    ways: u128,
    counts: &mut BTreeMap<String, u32>,
    out: &mut Vec<ForetoldAllocation>,
) {
    let Some(((name, available), rest)) = names.split_first() else {
        if remaining == 0 {
            out.push(ForetoldAllocation {
                counts: counts.clone(),
                ways,
            });
        }
        return;
    };
    for count in 0..=remaining.min(*available) {
        if count > 0 {
            counts.insert(name.clone(), count);
        }
        enumerate_foretold(
            rest,
            remaining - count,
            ways * crate::possible_worlds::binom(*available, count),
            counts,
            out,
        );
        counts.remove(name);
    }
}

fn rebind_captures(bindings: &mut [ObjectBinding], mapping: &BTreeMap<ObjectRef, ObjectRef>) {
    for binding in bindings {
        if let Some(object) = mapping.get(&binding.object) {
            binding.object = *object;
        }
    }
}
