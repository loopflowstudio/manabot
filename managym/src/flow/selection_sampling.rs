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
        decision::{Decision, EffectFrame, FrameCompletion, ResolutionRef, SuspendedResolution},
        game::{Game, GameState},
    },
    state::{
        ability::ObjectSelector,
        binding::{ObjectBinding, ZoneSelection},
        game_object::{CardId, ObjectRef, PlayerId},
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
    pinned: Vec<ObjectRef>,
    /// Unknown hidden-zone objects a selection retains, each with every
    /// membership it must still satisfy. A deal assigns them distinct slots.
    slots: BTreeMap<ObjectRef, Vec<ZoneSelection>>,
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
    card: CardId,
) -> bool {
    source.owner == owner
        && source.zone == zone
        && game.card_matches_predicate(card, &source.predicate)
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

    pub(crate) fn pins(&self, object: ObjectRef) -> bool {
        self.pinned.binary_search(&object).is_ok()
    }

    pub(crate) fn pins_card(&self, card: CardId) -> bool {
        self.pinned
            .iter()
            .any(|object| object.entity == card.into())
    }

    /// Pinned cards currently in `owner`'s library: they rejoin the sampled
    /// library after the deal instead of entering the unseen pool.
    pub(crate) fn pinned_library_cards(&self, game: &Game, owner: PlayerId) -> Vec<CardId> {
        pinned_library_cards(game, owner, &self.pinned)
    }

    /// `owner`'s cards this viewer cannot see; see [`Game::unseen_cards`].
    pub(crate) fn unseen_cards(&self, game: &Game, owner: PlayerId) -> Vec<CardId> {
        unseen_cards(game, owner, self.viewer, &self.pinned)
    }
}

fn pinned_library_cards(game: &Game, owner: PlayerId, pinned: &[ObjectRef]) -> Vec<CardId> {
    pinned
        .iter()
        .map(|object| object.entity.into())
        .filter(|card| game.state.zones.contains(*card, ZoneType::Library, owner))
        .collect()
}

fn unseen_cards(
    game: &Game,
    owner: PlayerId,
    viewer: PlayerId,
    pinned: &[ObjectRef],
) -> Vec<CardId> {
    let pinned_library = pinned_library_cards(game, owner, pinned);
    let zones: &[ZoneType] = if owner == viewer {
        &[ZoneType::Library]
    } else {
        &[ZoneType::Hand, ZoneType::Library]
    };
    let mut cards: Vec<CardId> = zones
        .iter()
        .flat_map(|zone| game.state.zones.zone_cards(*zone, owner).iter().copied())
        .filter(|card| !pinned_library.contains(card))
        .collect();
    if !game.shares_information(viewer, owner) {
        cards.extend(game.foretold_cards(owner));
    }
    cards
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
    pub(crate) fn pinned_objects(&self, viewer: PlayerId) -> Vec<ObjectRef> {
        let mut pinned = self.pending_public_entries();
        pinned.extend(
            self.captured_bindings()
                .filter(|binding| {
                    binding.known_to[viewer.0]
                        && self.current_object_ref(binding.object.entity.into())
                            == Some(binding.object)
                })
                .map(|binding| binding.object),
        );
        if let Some(suspended) = &self.state.suspended_decision {
            for (owner, cards) in suspended
                .decisions()
                .filter_map(Decision::library_inspection)
            {
                if self.shares_information(viewer, owner) {
                    pinned.extend(
                        cards
                            .iter()
                            .filter_map(|card| self.current_object_ref(*card)),
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
    pub fn unseen_cards(&self, owner: PlayerId, viewer: PlayerId) -> Vec<CardId> {
        unseen_cards(self, owner, viewer, &self.pinned_objects(viewer))
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

    /// Read every retained commitment for `viewer` from this root. Slots are
    /// collected before any hidden-zone reassignment; known exact captures are
    /// pinned rather than slotted, even when another owner holds an unknown
    /// alias of the same capture.
    pub(crate) fn hidden_commitments(&self, viewer: PlayerId) -> HiddenCommitments {
        let pinned = self.pinned_objects(viewer);
        let mut slots = BTreeMap::<ObjectRef, Vec<ZoneSelection>>::new();
        let mut minimums = Vec::new();
        let mut retain = |source: &Option<ZoneSelection>, selected: &[ObjectRef]| {
            let Some(source) = source
                .as_ref()
                .filter(|source| hidden_from(self, source, viewer))
            else {
                return;
            };
            for object in selected {
                let card = object.entity.into();
                // Moved selections keep their old incarnation and no longer
                // bind a hidden source slot. Public results must never be sampled.
                if self.current_object_ref(card) == Some(*object)
                    && pinned.binary_search(object).is_err()
                    && matches!(
                        self.state.zones.zone_of(card),
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
                let Some((owner, cards)) = decision.library_inspection() else {
                    continue;
                };
                let source = Some(ZoneSelection {
                    owner,
                    zone: ZoneType::Library,
                    predicate: Default::default(),
                });
                let objects: Vec<_> = cards
                    .iter()
                    .filter_map(|card| self.current_object_ref(*card))
                    .collect();
                retain(&source, &objects);
                if let Decision::LookAndSelect {
                    predicate,
                    min_select,
                    selected,
                    ..
                } = decision
                {
                    let matches = |object: &&ObjectRef| {
                        self.card_matches_predicate(object.entity.into(), predicate)
                    };
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
        pool: &mut Vec<CardId>,
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
        commitments: &HiddenCommitments,
        hand: &BTreeMap<String, u32>,
    ) -> Vec<ForetoldAllocation> {
        let owner = PlayerId(1 - commitments.viewer.0);
        let mut pool = BTreeMap::<String, Vec<CardId>>::new();
        for card in commitments.unseen_cards(self, owner) {
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
                            && commitments.pins_card(**card)
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
            let mut candidates: Vec<Vec<usize>> = commitments
                .slots
                .values()
                .map(|constraints| {
                    slots
                        .iter()
                        .enumerate()
                        .filter(|(_, (zone, card))| {
                            constraints
                                .iter()
                                .all(|source| matches_slot(self, source, owner, *zone, *card))
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
                    .filter(|(_, (zone, card))| {
                        source.zone == *zone
                            && self.card_matches_predicate(*card, &source.predicate)
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
                .zone_cards(source.zone, source.owner)
                .iter()
                .filter(|card| self.card_matches_predicate(**card, &source.predicate))
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
                .zone_cards(source.zone, source.owner)
                .iter()
                .copied()
                .filter(|card| {
                    constraints.iter().all(|constraint| {
                        matches_slot(self, constraint, source.owner, source.zone, *card)
                    })
                })
                .filter_map(|card| self.current_object_ref(card))
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
                        .filter(|card| self.card_matches_predicate(*card, &source.predicate))
                        .filter_map(|card| self.current_object_ref(card))
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
                for cards in [looked, remaining, top, bottom] {
                    rebind(cards.iter_mut(), &rebinding.cards);
                }
            }
            Decision::Surveil {
                looked, graveyard, ..
            } => {
                rebind(looked.iter_mut(), &rebinding.cards);
                rebind(graveyard.iter_mut(), &rebinding.cards);
            }
            Decision::LookAndSelect { looked, .. } => rebind(looked.iter_mut(), &rebinding.cards),
            Decision::OrderLibrary { cards, .. } => rebind(cards.iter_mut(), &rebinding.cards),
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
    objects: BTreeMap<ObjectRef, ObjectRef>,
    cards: BTreeMap<CardId, CardId>,
}

impl Rebinding {
    fn new(viewer: PlayerId, mapping: impl Iterator<Item = (ObjectRef, ObjectRef)>) -> Self {
        let objects: BTreeMap<_, _> = mapping.collect();
        let cards = objects
            .iter()
            .map(|(old, new)| (CardId::from(old.entity), CardId::from(new.entity)))
            .collect();
        Self {
            viewer,
            objects,
            cards,
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
