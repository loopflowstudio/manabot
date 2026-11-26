//! Viewer-safe library facts and the shared physical-witness placement used
//! by determinization and exact-world materialization. Unspecified reorders
//! forget facts; only explicit inspection records the resulting positions.

use std::collections::BTreeMap;

use rand::{seq::SliceRandom, Rng};

use crate::{
    flow::{decision::Decision, game::Game, selection_sampling::HiddenCommitments},
    state::{
        card::CardDefId,
        game_object::{ObjectId, PlayerId},
        knowledge::LibraryKnowledge,
        zone::ZoneType,
    },
};

impl Game {
    /// `owner`'s library from the top: position 0 is the next card drawn.
    pub(crate) fn library_from_top(&self, owner: PlayerId) -> impl Iterator<Item = ObjectId> + '_ {
        self.state
            .zones
            .zone_objects(ZoneType::Library, owner)
            .iter()
            .rev()
            .copied()
    }

    fn library_definition_at(&self, owner: PlayerId, position: usize) -> Option<CardDefId> {
        self.library_from_top(owner)
            .nth(position)
            .map(|object| self.state[object].definition_id)
    }

    /// Current positions and definitions of those `objects` in `owner`'s library.
    fn library_positions_of(
        &self,
        owner: PlayerId,
        objects: &[ObjectId],
    ) -> Vec<(usize, CardDefId)> {
        self.library_from_top(owner)
            .enumerate()
            .filter(|(_, object)| objects.contains(object))
            .map(|(position, object)| (position, self.state[object].definition_id))
            .collect()
    }

    pub(crate) fn library_facts(
        &self,
        owner: PlayerId,
        viewer: PlayerId,
    ) -> BTreeMap<usize, CardDefId> {
        let knowledge = &self.state.players[owner.0].known_library;
        let mut facts = knowledge.public.clone();
        if self.shares_information(viewer, owner) {
            let mut private = knowledge.own.clone();
            if self.looks_at_library_top(owner) {
                if let Some(object) = self.state.zones.top(ZoneType::Library, owner) {
                    private.insert(0, self.state[object].definition_id);
                }
            }
            LibraryKnowledge::bound(&mut private);
            facts.extend(private);
        }
        facts
    }

    /// Called only for cards the owner has inspected, in their resulting
    /// order. True positions are private; this never changes the public view.
    pub(crate) fn remember_library_objects(&mut self, owner: PlayerId, objects: &[ObjectId]) {
        self.state.library_order_pins_mut(owner).extend(objects);
        let facts = self.library_positions_of(owner, objects);
        let knowledge = self.state.known_library_mut(owner);
        knowledge.own.extend(facts);
        LibraryKnowledge::bound(&mut knowledge.own);
    }

    /// Cards put into one library position together: their owner arranges
    /// them and need not reveal the order (CR 401.4), so neither position
    /// stays public while the owner remembers both.
    pub(crate) fn arrange_library_objects(&mut self, owner: PlayerId, objects: &[ObjectId]) {
        let positions = self.library_positions_of(owner, objects);
        let knowledge = self.state.known_library_mut(owner);
        for (position, _) in positions {
            knowledge.public.remove(&position);
        }
        self.remember_library_objects(owner, objects);
    }

    /// Both seats learn the top position even if the revealed card stays there.
    pub(crate) fn reveal_library_top(&mut self, owner: PlayerId) -> Option<ObjectId> {
        let object = self.state.zones.top(ZoneType::Library, owner)?;
        let definition = self.state[object].definition_id;
        let knowledge = self.state.known_library_mut(owner);
        knowledge.public.insert(0, definition);
        LibraryKnowledge::bound(&mut knowledge.public);
        self.emit(crate::flow::event::GameEvent::CardRevealed { owner, definition });
        Some(object)
    }

    /// Both seats learn the positions of revealed library cards (Genesis Wave).
    pub(crate) fn reveal_library_objects(&mut self, owner: PlayerId, objects: &[ObjectId]) {
        self.state.library_order_pins_mut(owner).extend(objects);
        let facts = self.library_positions_of(owner, objects);
        let knowledge = self.state.known_library_mut(owner);
        knowledge.public.extend(facts);
        LibraryKnowledge::bound(&mut knowledge.public);
    }

    pub(crate) fn refresh_visible_library_tops(&mut self) {
        let visible = self.visible_library_tops();
        for owner in [PlayerId(0), PlayerId(1)] {
            if visible[owner.0] {
                if let Some(object) = self.state.zones.top(ZoneType::Library, owner) {
                    let definition = self.state[object].definition_id;
                    self.state.library_order_pins_mut(owner).insert(object);
                    let knowledge = self.state.known_library_mut(owner);
                    knowledge.own.insert(0, definition);
                    LibraryKnowledge::bound(&mut knowledge.own);
                }
            }
        }
    }

    /// Reserve distinct physical witnesses before dealing a hand or foretold
    /// slots. Lowest-slot witnesses preserve the existing canonical convention.
    pub(crate) fn reserve_library_facts(
        &self,
        owner: PlayerId,
        viewer: PlayerId,
        pool: &mut Vec<ObjectId>,
        commitments: &HiddenCommitments,
    ) -> Option<Vec<(usize, ObjectId)>> {
        let mut facts = self.library_facts(owner, viewer);
        if self.shares_information(viewer, owner) {
            if let Some(suspended) = &self.state.suspended_decision {
                for (_, objects) in suspended
                    .decisions()
                    .filter_map(Decision::inspected_library_objects)
                    .filter(|(player, _)| *player == owner)
                {
                    facts.extend(self.library_positions_of(owner, objects));
                }
            }
        }
        if let Some(suspended) = &self.state.suspended_decision {
            for decision in suspended.decisions() {
                if let Decision::ChoosePile {
                    owner: pile_owner,
                    face_up,
                    ..
                } = decision
                {
                    if *pile_owner == owner {
                        facts.extend(self.library_positions_of(owner, face_up));
                    }
                }
            }
        }
        if facts.is_empty() {
            return Some(Vec::new());
        }
        let mut reserved = Vec::new();
        for (position, definition) in facts {
            let original = self.library_from_top(owner).nth(position)?;
            // Shared inspections already pin their exact objects through
            // HiddenCommitments; ordinary facts reserve definition witnesses.
            if commitments.pins(original)
                || position == 0
                    && self.shares_information(viewer, owner)
                    && self.looks_at_library_top(owner)
            {
                if let Some(index) = pool.iter().position(|object| *object == original) {
                    pool.remove(index);
                }
                reserved.push((position, original));
                continue;
            }
            let index = pool
                .iter()
                .enumerate()
                .filter(|(_, object)| {
                    self.state[**object].retained_base().0 == definition
                        && !commitments.pins(**object)
                })
                .min_by_key(|(_, object)| object.slot())
                .map(|(index, _)| index)?;
            reserved.push((position, pool.remove(index)));
        }
        Some(reserved)
    }

    /// Fill the remaining slots without changing the order of the random
    /// remainder. Indices in the returned vector still run bottom to top.
    pub(crate) fn place_library_facts(
        remainder: Vec<ObjectId>,
        reserved: Vec<(usize, ObjectId)>,
    ) -> Option<Vec<ObjectId>> {
        if reserved.is_empty() {
            return Some(remainder);
        }
        let size = remainder.len() + reserved.len();
        let mut positions = vec![None; size];
        for (from_top, object) in reserved {
            let index = size.checked_sub(from_top + 1)?;
            if positions[index].replace(object).is_some() {
                return None;
            }
        }
        let mut remainder = remainder.into_iter();
        for position in &mut positions {
            if position.is_none() {
                *position = remainder.next();
            }
        }
        positions.into_iter().collect()
    }

    pub(crate) fn shuffle_known_library(
        &mut self,
        owner: PlayerId,
        viewer: PlayerId,
        commitments: &HiddenCommitments,
        rng: &mut impl Rng,
    ) {
        let hidden_exile = self.hidden_face_down_exile(owner, viewer);
        if self.library_facts(owner, viewer).is_empty()
            && self.state.suspended_decision.is_none()
            && hidden_exile.is_empty()
        {
            self.state
                .zones
                .shuffle_canonical(ZoneType::Library, owner, rng);
            return;
        }
        let mut pool = self
            .state
            .zones
            .zone_objects(ZoneType::Library, owner)
            .to_vec();
        pool.extend(hidden_exile);
        pool.sort_unstable_by_key(|object| object.slot());
        let reserved = self
            .reserve_library_facts(owner, viewer, &mut pool, commitments)
            .expect("library facts are true in the source world");
        pool.shuffle(rng);
        assert!(
            self.refill_hidden_exile_slots(owner, viewer, &mut pool),
            "face-down cards are in their own pool"
        );
        let library = Self::place_library_facts(pool, reserved).expect("valid library positions");
        let hand = self
            .state
            .zones
            .zone_objects(ZoneType::Hand, owner)
            .to_vec();
        self.state.reassign_hidden(owner, hand, library);
    }

    /// Sampling changes the unshared opponent's private deal, so only facts
    /// the sampling viewer received survive in that opponent's private view.
    pub(crate) fn refresh_sampled_library_knowledge(&mut self, viewer: PlayerId) {
        let opponent = PlayerId(1 - viewer.0);
        if !self.shares_information(viewer, opponent) {
            let public = self.state.players[opponent.0].known_library.public.clone();
            // Rebind surviving facts to the sampled physical witnesses. The
            // displaced opponent's private placements belong to the old deal.
            if !self.state.library_order_pins[opponent.0].is_empty() || !public.is_empty() {
                let mut pins = self.pinned_objects(viewer);
                pins.retain(|object| {
                    self.state
                        .zones
                        .contains(*object, ZoneType::Library, opponent)
                });
                if let Some(suspended) = &self.state.suspended_decision {
                    for (_, objects) in suspended
                        .decisions()
                        .filter_map(Decision::inspected_library_objects)
                        .filter(|(owner, _)| *owner == opponent)
                    {
                        pins.extend_from_slice(objects);
                    }
                }
                pins.extend(
                    self.library_from_top(opponent)
                        .enumerate()
                        .filter_map(|(rank, object)| public.contains_key(&rank).then_some(object)),
                );
                *self.state.library_order_pins_mut(opponent) = pins.into_iter().collect();
            }
            self.state.known_library_mut(opponent).own = public;
        }
    }

    pub(crate) fn library_knowledge_holds(&self, viewer: PlayerId) -> bool {
        [PlayerId(0), PlayerId(1)].into_iter().all(|owner| {
            self.library_facts(owner, viewer)
                .into_iter()
                .all(|(position, definition)| {
                    self.library_definition_at(owner, position) == Some(definition)
                })
        })
    }

    pub(crate) fn assert_library_knowledge(&self) {
        #[cfg(debug_assertions)]
        for owner in [PlayerId(0), PlayerId(1)] {
            let player = &self.state.players[owner.0];
            for (definition, minimum) in &player.known_hand {
                let count = self
                    .state
                    .zones
                    .zone_objects(ZoneType::Hand, owner)
                    .iter()
                    .filter(|object| self.state[**object].definition_id == *definition)
                    .count();
                assert!(
                    *minimum as usize <= count,
                    "false known hand minimum for {owner:?}"
                );
            }
            let knowledge = &player.known_library;
            for facts in [&knowledge.public, &knowledge.own] {
                for (position, definition) in facts {
                    assert_eq!(
                        self.library_definition_at(owner, *position),
                        Some(*definition),
                        "false library fact for {owner:?} at {position}"
                    );
                }
            }
        }
    }
}
