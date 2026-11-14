//! Rebind retained private selections to a sampled world's zone membership.
//!
//! The selected count and predicate are commitments, not disposable offers.
//! `HiddenCommitments` reads every retained commitment from the root once for
//! one viewer. Both samplers deal hidden zones their own way, then apply one
//! exact-object mapping across active decisions, saved callers and captured
//! bindings, so the same selection cannot acquire two meanings on resume.

use std::collections::BTreeMap;

use rand::{seq::SliceRandom, Rng};
use rand_chacha::ChaCha8Rng;

use crate::{
    flow::{
        decision::{
            Decision, EffectFrame, FrameCompletion, ResolutionRef, SpendCompletion,
            SuspendedResolution,
        },
        game::{Game, GameState},
    },
    state::{
        ability::ObjectSelector,
        binding::{ObjectBinding, ZoneSelection},
        game_object::{ObjectId, PlayerId},
        stack_object::StackObject,
        zone::ZoneType,
    },
};

/// What a sampled world must honor for one viewer, read from the root once.
pub(crate) struct HiddenCommitments {
    viewer: PlayerId,
    /// Exact objects no sampler may exchange for a same-definition copy:
    /// prospective public entrants, captures this viewer knows and library
    /// cards it inspects. Sorted and deduplicated.
    pinned: Vec<ObjectId>,
    /// Unknown hidden-zone objects a selection retains, each with every
    /// membership it must still satisfy. A deal assigns them distinct slots.
    slots: BTreeMap<ObjectId, Vec<ZoneSelection>>,
    /// Active selections over a zone, in continuation order, with the count
    /// their zone must still be able to supply.
    minimums: Vec<SelectionMinimum>,
}

struct SelectionMinimum {
    source: ZoneSelection,
    min: usize,
    selected: usize,
    hidden: bool,
}

/// One name allocation to face-down slots and its ordered physical-copy count.
/// Retained selections condition this measure by feasibility, not by the number
/// of ways a player could have chosen the private subset.
struct ForetoldAllocation {
    counts: BTreeMap<String, u32>,
    ways: u128,
}

fn hidden_from(game: &Game, source: &ZoneSelection, viewer: PlayerId) -> bool {
    matches!(source.zone, ZoneType::Hand | ZoneType::Library)
        && !game.shares_information(viewer, source.owner)
}

fn matches_slot(
    game: &Game,
    source: &ZoneSelection,
    owner: PlayerId,
    zone: ZoneType,
    object: ObjectId,
) -> bool {
    source.owner == owner
        && source.zone == zone
        && game.object_matches_predicate(object, &source.predicate)
}

impl HiddenCommitments {
    /// Whether a deal can fail to honor earlier choices. Unconstrained roots
    /// need no retry clone and no feasibility weighting.
    pub(crate) fn constrains(&self) -> bool {
        !self.slots.is_empty()
            || self
                .minimums
                .iter()
                .any(|minimum| minimum.hidden && minimum.min > 0)
    }

    pub(crate) fn pins(&self, object: ObjectId) -> bool {
        self.pinned.binary_search(&object).is_ok()
    }

    pub(crate) fn pins_object(&self, object_id: ObjectId) -> bool {
        self.pinned.contains(&object_id)
    }

    /// Pinned cards currently in `owner`'s library: they rejoin the sampled
    /// library after the deal instead of entering the unseen pool.
    pub(crate) fn pinned_library_objects(&self, game: &Game, owner: PlayerId) -> Vec<ObjectId> {
        pinned_library_objects(game, owner, &self.pinned)
    }

    /// `owner`'s cards this viewer cannot see; see [`Game::unseen_objects`].
    pub(crate) fn unseen_objects(&self, game: &Game, owner: PlayerId) -> Vec<ObjectId> {
        unseen_objects(game, owner, self.viewer, &self.pinned)
    }
}

fn pinned_library_objects(game: &Game, owner: PlayerId, pinned: &[ObjectId]) -> Vec<ObjectId> {
    pinned
        .iter()
        .copied()
        .filter(|object| game.state.zones.contains(*object, ZoneType::Library, owner))
        .collect()
}

fn unseen_objects(
    game: &Game,
    owner: PlayerId,
    viewer: PlayerId,
    pinned: &[ObjectId],
) -> Vec<ObjectId> {
    let pinned_library = pinned_library_objects(game, owner, pinned);
    let zones: &[ZoneType] = if owner == viewer {
        &[ZoneType::Library]
    } else {
        &[ZoneType::Hand, ZoneType::Library]
    };
    let mut objects: Vec<ObjectId> = zones
        .iter()
        .flat_map(|zone| game.state.zones.zone_objects(*zone, owner).iter().copied())
        .filter(|object| !pinned_library.contains(object))
        .collect();
    if !game.shares_information(viewer, owner) {
        objects.extend(
            game.foretold_cards(owner)
                .map(|card| game.state.object_of(card)),
        );
    }
    objects
}

impl GameState {
    /// Captured bindings held by trigger occurrences outside the suspended
    /// continuation: waiting, queued, in the detached placement round, and
    /// on the stack.
    fn trigger_bindings(&self) -> impl Iterator<Item = &ObjectBinding> {
        let placing = self
            .trigger_placement
            .iter()
            .flat_map(|placement| placement.remaining.iter().chain(placement.placing.iter()));
        let stacked = self.stack_objects.iter().filter_map(|object| match object {
            StackObject::TriggeredAbility(trigger) => Some(trigger),
            _ => None,
        });
        self.delayed_triggers
            .iter()
            .flat_map(|trigger| &trigger.remembered)
            .chain(
                self.pending_triggers
                    .iter()
                    .flat_map(|trigger| &trigger.remembered),
            )
            .chain(placing.flat_map(|trigger| &trigger.remembered))
            .chain(stacked.flat_map(|trigger| &trigger.remembered))
    }

    fn trigger_bindings_mut(&mut self) -> impl Iterator<Item = &mut ObjectBinding> {
        let placing = self.trigger_placement.iter_mut().flat_map(|placement| {
            placement
                .remaining
                .iter_mut()
                .chain(placement.placing.iter_mut())
        });
        let stacked = self
            .stack_objects
            .iter_mut()
            .filter_map(|object| match object {
                StackObject::TriggeredAbility(trigger) => Some(trigger),
                _ => None,
            });
        self.delayed_triggers
            .iter_mut()
            .flat_map(|trigger| &mut trigger.remembered)
            .chain(
                self.pending_triggers
                    .iter_mut()
                    .flat_map(|trigger| &mut trigger.remembered),
            )
            .chain(placing.flat_map(|trigger| &mut trigger.remembered))
            .chain(stacked.flat_map(|trigger| &mut trigger.remembered))
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
                let zone = self.state.zones.zone_of(object);
                let selection = zone
                    .filter(|zone| matches!(zone, ZoneType::Hand | ZoneType::Library))
                    .map(|zone| ZoneSelection {
                        owner: self.state.objects[object].owner,
                        zone,
                        predicate: frame
                            .selected_from
                            .as_ref()
                            .filter(|source| {
                                frame.selected.contains(&object)
                                    && source.zone == zone
                                    && self.object_matches_predicate(object, &source.predicate)
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
                                    && frame.selected.contains(&object)
                            })
                    }),
                }
            })
            .collect()
    }

    /// Every retained capture: trigger occurrences first, then the frames of
    /// the suspended continuation, outermost first.
    pub(crate) fn captured_bindings(&self) -> impl Iterator<Item = &ObjectBinding> {
        let frames = self.state.suspended_decision.iter().flat_map(|suspended| {
            suspended
                .continuation()
                .into_iter()
                .filter_map(|part| match part {
                    ResolutionRef::Frame(frame) => Some(&frame.remembered),
                    ResolutionRef::Decision(_) => None,
                })
        });
        self.state.trigger_bindings().chain(frames.flatten())
    }

    /// Exact objects already known to this viewer must not be exchanged for a
    /// same-definition copy. Departed incarnations impose no current-zone slot.
    pub(crate) fn pinned_objects(&self, viewer: PlayerId) -> Vec<ObjectId> {
        let mut pinned = self.pending_public_entries();
        pinned.extend(
            self.captured_bindings()
                .filter(|binding| {
                    binding.known_to[viewer.0]
                        && self.state.live(binding.object) == Some(binding.object)
                })
                .map(|binding| binding.object),
        );
        if let Some(suspended) = &self.state.suspended_decision {
            for (owner, object_ids) in suspended
                .decisions()
                .filter_map(Decision::library_inspection)
            {
                if self.shares_information(viewer, owner) {
                    pinned.extend(
                        object_ids
                            .iter()
                            .filter_map(|object_id| self.state.live(*object_id)),
                    );
                }
            }
        }
        pinned.sort_unstable();
        pinned.dedup();
        pinned
    }

    /// `owner`'s cards that `viewer` cannot see: the library, and for an
    /// opponent also the hand and face-down foretold exile. For an opponent
    /// this is exactly the possible-world pool, including any publicly known
    /// hand cards. Pinned library cards are excluded; they never move.
    pub fn unseen_objects(&self, owner: PlayerId, viewer: PlayerId) -> Vec<ObjectId> {
        unseen_objects(self, owner, viewer, &self.pinned_objects(viewer))
    }

    /// Prospective battlefield entries are public even while their move awaits
    /// replacement choices. Neither sampler may exchange their exact objects.
    pub(crate) fn pending_public_entries(&self) -> Vec<ObjectId> {
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
        entries.retain(|object| self.state.live(*object).is_some());
        entries.sort_unstable();
        entries.dedup();
        entries
    }

    /// Read every retained commitment for `viewer` from this root. Slots are
    /// collected before any hidden-zone reassignment; known exact captures are
    /// pinned rather than slotted, even when another owner holds an unknown
    /// alias of the same capture.
    pub(crate) fn hidden_commitments(&self, viewer: PlayerId) -> HiddenCommitments {
        let pinned = self.pinned_objects(viewer);
        let mut slots = BTreeMap::<ObjectId, Vec<ZoneSelection>>::new();
        let mut minimums = Vec::new();
        let mut retain = |source: &Option<ZoneSelection>, selected: &[ObjectId]| {
            let Some(source) = source
                .as_ref()
                .filter(|source| hidden_from(self, source, viewer))
            else {
                return;
            };
            for object in selected {
                // Moved selections keep their old identity and no longer
                // bind a hidden source slot. Public results must never be sampled.
                if pinned.binary_search(object).is_err()
                    && matches!(
                        self.state.zones.zone_of(*object),
                        Some(ZoneType::Hand | ZoneType::Library)
                    )
                {
                    // Inherited child bindings repeat the same constraint;
                    // retain each requirement once for the shared assignment.
                    let constraints = slots.entry(*object).or_default();
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
                    ResolutionRef::Decision(Decision::SelectCards {
                        from,
                        selected,
                        min,
                        ..
                    }) => {
                        retain(from, selected);
                        if let Some(source) = from {
                            minimums.push(SelectionMinimum {
                                source: source.clone(),
                                min: *min,
                                selected: selected.len(),
                                hidden: hidden_from(self, source, viewer),
                            });
                        }
                    }
                    ResolutionRef::Decision(_) => {}
                }
            }
            for decision in suspended.decisions() {
                let Some((owner, object_ids)) = decision.library_inspection() else {
                    continue;
                };
                let source = Some(ZoneSelection {
                    owner,
                    zone: ZoneType::Library,
                    predicate: Default::default(),
                });
                let objects: Vec<_> = object_ids
                    .iter()
                    .filter_map(|object_id| self.state.live(*object_id))
                    .collect();
                retain(&source, &objects);
                if let Decision::LookAndSelect {
                    predicate,
                    min_select,
                    selected,
                    ..
                } = decision
                {
                    let matches =
                        |object: &&ObjectId| self.object_matches_predicate(**object, predicate);
                    let known_matching = objects
                        .iter()
                        .filter(matches)
                        .filter(|object| pinned.binary_search(object).is_ok())
                        .count();
                    let needed = min_select
                        .saturating_sub(*selected)
                        .saturating_sub(known_matching);
                    let required: Vec<_> = objects
                        .iter()
                        .filter(matches)
                        .filter(|object| pinned.binary_search(object).is_err())
                        .take(needed)
                        .copied()
                        .collect();
                    retain(
                        &Some(ZoneSelection {
                            owner,
                            zone: ZoneType::Library,
                            predicate: predicate.clone(),
                        }),
                        &required,
                    );
                }
            }
        }
        for binding in self.captured_bindings() {
            retain(&binding.selection, std::slice::from_ref(&binding.object));
        }
        HiddenCommitments {
            viewer,
            pinned,
            slots,
            minimums,
        }
    }

    /// Exact ordered foretold fills that leave the selected subsets realizable.
    /// Unlike a feasibility-only filter, this removes the mass of impossible
    /// foretold assignments within an otherwise compatible hand-count world.
    pub(crate) fn selection_foretold_weight(
        &self,
        commitments: &HiddenCommitments,
        hand: &BTreeMap<String, u32>,
    ) -> u128 {
        self.selection_foretold_allocations(commitments, hand)
            .iter()
            .map(|allocation| allocation.ways)
            .sum()
    }

    /// Order one compatible foretold fill before the remaining shuffled pool.
    /// The existing slot mover still owns grants and exact exile membership.
    pub(crate) fn prepare_selection_foretell(
        &self,
        commitments: &HiddenCommitments,
        hand: &BTreeMap<String, u32>,
        pool: &mut Vec<ObjectId>,
        rng: &mut ChaCha8Rng,
    ) -> bool {
        let allocations = self.selection_foretold_allocations(commitments, hand);
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
        pool.retain(|object| {
            let Some(count) = counts.get_mut(&self.state.objects[*object].name) else {
                return true;
            };
            if *count == 0 {
                return true;
            }
            *count -= 1;
            chosen.push(*object);
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
        commitments: &HiddenCommitments,
        hand: &BTreeMap<String, u32>,
    ) -> Vec<ForetoldAllocation> {
        let owner = PlayerId(1 - commitments.viewer.0);
        let mut pool = BTreeMap::<String, Vec<ObjectId>>::new();
        for object in commitments.unseen_objects(self, owner) {
            pool.entry(self.state.objects[object].name.clone())
                .or_default()
                .push(object);
        }
        if hand
            .iter()
            .any(|(name, count)| *count as usize > pool.get(name).map_or(0, Vec::len))
        {
            return Vec::new();
        }
        let bearers: Vec<_> = pool
            .iter()
            .filter(|(_, objects)| self.state.objects[objects[0]].foretell().is_some())
            .map(|(name, objects)| {
                (
                    name.clone(),
                    objects.len() as u32 - hand.get(name).copied().unwrap_or(0),
                )
            })
            .collect();
        let count = self.foretold_cards(owner).count() as u32;
        let mut allocations = Vec::new();
        enumerate_foretold(&bearers, count, 1, &mut BTreeMap::new(), &mut allocations);
        let orders: u128 = (1..=count).map(u128::from).product();
        allocations.retain_mut(|allocation| {
            let mut slots = Vec::new();
            for (name, objects) in &pool {
                let in_hand = hand.get(name).copied().unwrap_or(0) as usize;
                let foretold = allocation.counts.get(name).copied().unwrap_or(0) as usize;
                let pinned_hand = objects
                    .iter()
                    .filter(|object| {
                        self.state.zones.contains(**object, ZoneType::Hand, owner)
                            && commitments.pins_object(**object)
                    })
                    .count();
                for index in pinned_hand..objects.len() - foretold {
                    slots.push((
                        if index < in_hand {
                            ZoneType::Hand
                        } else {
                            ZoneType::Library
                        },
                        objects[0],
                    ));
                }
            }
            let mut candidates: Vec<Vec<usize>> = commitments
                .slots
                .values()
                .map(|constraints| {
                    slots
                        .iter()
                        .enumerate()
                        .filter(|(_, (zone, object))| {
                            constraints
                                .iter()
                                .all(|source| matches_slot(self, source, owner, *zone, *object))
                        })
                        .map(|(index, _)| index)
                        .collect()
                })
                .collect();
            for minimum in &commitments.minimums {
                let source = &minimum.source;
                if source.owner != owner
                    || !matches!(source.zone, ZoneType::Hand | ZoneType::Library)
                {
                    continue;
                }
                let eligible: Vec<_> = slots
                    .iter()
                    .enumerate()
                    .filter(|(_, (zone, object))| {
                        source.zone == *zone
                            && self.object_matches_predicate(*object, &source.predicate)
                    })
                    .map(|(index, _)| index)
                    .collect();
                for _ in minimum.selected..minimum.min {
                    candidates.push(eligible.clone());
                }
            }
            allocation.ways *= orders;
            complete_assignment(&candidates).is_some()
        });
        allocations
    }

    /// Give every retained commitment its exact object in this dealt world and
    /// pin inspected library prefixes back in place. False means the deal cannot
    /// honor earlier choices: the caller rejects the world; neither selected
    /// counts nor public commitments are truncated.
    pub(crate) fn rebind_commitments(
        &mut self,
        commitments: &HiddenCommitments,
        rng: &mut ChaCha8Rng,
    ) -> bool {
        if commitments.minimums.iter().any(|minimum| {
            let source = &minimum.source;
            self.state
                .zones
                .zone_objects(source.zone, source.owner)
                .iter()
                .filter(|object_id| self.object_matches_predicate(**object_id, &source.predicate))
                .count()
                < minimum.min
        }) {
            return false;
        }
        let old: Vec<_> = commitments.slots.keys().copied().collect();
        let mut candidates = Vec::with_capacity(old.len());
        for constraints in commitments.slots.values() {
            let source = &constraints[0];
            let mut eligible: Vec<_> = self
                .state
                .zones
                .zone_objects(source.zone, source.owner)
                .iter()
                .copied()
                .filter(|object_id| {
                    constraints.iter().all(|constraint| {
                        matches_slot(self, constraint, source.owner, source.zone, *object_id)
                    })
                })
                .filter_map(|object_id| self.state.live(object_id))
                .filter(|object| !commitments.pins(*object))
                .collect();
            eligible.shuffle(rng);
            candidates.push(eligible);
        }
        let Some(assigned) = complete_assignment(&candidates) else {
            return false;
        };
        let rebinding = Rebinding::new(
            commitments.viewer,
            assigned
                .into_iter()
                .map(|(object, index)| (old[index], object)),
        );
        rebind(
            self.state
                .trigger_bindings_mut()
                .map(|binding| &mut binding.object),
            &rebinding.objects,
        );
        if let Some(mut suspended) = self.state.suspended_decision_mut().take() {
            self.rebind_suspended_selection(&mut suspended, &rebinding);
            *self.state.suspended_decision_mut() = Some(suspended);
        }
        self.repin_revealed_library_cards();
        true
    }

    fn rebind_suspended_selection(
        &self,
        suspended: &mut SuspendedResolution,
        rebinding: &Rebinding,
    ) {
        self.rebind_frame_selection(&mut suspended.frame, rebinding);
        self.rebind_decision_selection(&mut suspended.decision, rebinding);
    }

    fn rebind_frame_selection(&self, frame: &mut EffectFrame, rebinding: &Rebinding) {
        // Eligibility belongs to the joint mapping, not this alias's selector.
        // A selection from Remembered has no zone marker of its own, and a
        // child may remember or otherwise alias a selected exact object.
        let objects = &rebinding.objects;
        rebind(frame.selected.iter_mut(), objects);
        rebind(
            frame
                .remembered
                .iter_mut()
                .map(|binding| &mut binding.object),
            objects,
        );
        rebind(frame.moved.iter_mut(), objects);
        rebind(frame.this_way.iter_mut(), objects);
        rebind(frame.created.iter_mut(), objects);
        self.rebind_completion_selection(&mut frame.completion, rebinding);
    }

    fn rebind_completion_selection(&self, completion: &mut FrameCompletion, rebinding: &Rebinding) {
        match completion {
            FrameCompletion::Frame(parent) | FrameCompletion::Event { frame: parent, .. } => {
                self.rebind_frame_selection(parent, rebinding)
            }
            FrameCompletion::Decision(parent) => self.rebind_suspended_selection(parent, rebinding),
            FrameCompletion::StateBased { next } => {
                self.rebind_completion_selection(next, rebinding)
            }
            _ => {}
        }
    }

    fn rebind_decision_selection(&self, decision: &mut Decision, rebinding: &Rebinding) {
        match decision {
            Decision::SelectCards {
                from,
                candidates,
                selected,
                ..
            } => {
                rebind(selected.iter_mut(), &rebinding.objects);
                if let Some(source) = from
                    .as_ref()
                    .filter(|source| hidden_from(self, source, rebinding.viewer))
                {
                    *candidates = self
                        .selection_zone_cards(source.owner, source.zone)
                        .into_iter()
                        .filter(|card| {
                            self.object_matches_predicate(
                                self.state.object_of(*card),
                                &source.predicate,
                            )
                        })
                        .filter_map(|card| self.state.live(self.state.object_of(card)))
                        .collect();
                } else {
                    rebind(candidates.iter_mut(), &rebinding.objects);
                }
            }
            Decision::Scry {
                looked,
                remaining,
                top,
                bottom,
                ..
            } => {
                for object_ids in [looked, remaining, top, bottom] {
                    rebind(object_ids.iter_mut(), &rebinding.objects);
                }
            }
            Decision::Surveil {
                looked, graveyard, ..
            } => {
                rebind(looked.iter_mut(), &rebinding.objects);
                rebind(graveyard.iter_mut(), &rebinding.objects);
            }
            Decision::LookAndSelect { looked, .. } => rebind(looked.iter_mut(), &rebinding.objects),
            Decision::OrderLibrary { object_ids, .. } => {
                rebind(object_ids.iter_mut(), &rebinding.objects)
            }
            Decision::ManaSpend { completion, .. } => {
                if let SpendCompletion::Upfront(parent) = completion.as_mut() {
                    self.rebind_decision_selection(parent, rebinding);
                }
            }
            Decision::ManaUpfront {
                resume: Some(parent),
                ..
            }
            | Decision::ManaChoice {
                resume: Some(parent),
                ..
            } => self.rebind_decision_selection(parent, rebinding),
            _ => {}
        }
    }
}

/// One joint exact-object assignment, also keyed by physical card for the
/// inspected-library decisions that retain cards rather than objects.
struct Rebinding {
    viewer: PlayerId,
    objects: BTreeMap<ObjectId, ObjectId>,
}

impl Rebinding {
    fn new(viewer: PlayerId, mapping: impl Iterator<Item = (ObjectId, ObjectId)>) -> Self {
        Self {
            viewer,
            objects: mapping.collect(),
        }
    }
}

fn rebind<'a, K: Copy + Ord + 'a>(
    items: impl Iterator<Item = &'a mut K>,
    mapping: &BTreeMap<K, K>,
) {
    for item in items {
        if let Some(replacement) = mapping.get(item) {
            *item = *replacement;
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
