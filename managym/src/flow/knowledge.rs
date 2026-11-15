//! Viewer-safe library facts and the shared physical-witness placement used
//! by determinization and exact-world materialization. Unspecified reorders
//! forget facts; only explicit inspection records the resulting positions.

use std::collections::BTreeMap;

use rand::{seq::SliceRandom, Rng};

use crate::{
    flow::game::Game,
    state::{
        card::CardDefId,
        game_object::{ObjectId, PlayerId},
        knowledge::LibraryKnowledge,
        zone::ZoneType,
    },
};

impl Game {
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
        let facts: Vec<_> = self
            .state
            .zones
            .zone_objects(ZoneType::Library, owner)
            .iter()
            .rev()
            .enumerate()
            .filter(|(_, object)| objects.contains(object))
            .map(|(position, object)| (position, self.state[*object].definition_id))
            .collect();
        let knowledge = self.state.known_library_mut(owner);
        knowledge.own.extend(facts);
        LibraryKnowledge::bound(&mut knowledge.own);
    }

    pub(crate) fn refresh_visible_library_tops(&mut self) {
        for owner in [PlayerId(0), PlayerId(1)] {
            if self.looks_at_library_top(owner) {
                if let Some(object) = self.state.zones.top(ZoneType::Library, owner) {
                    let definition = self.state[object].definition_id;
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
    ) -> Option<Vec<(usize, ObjectId)>> {
        let mut facts = self.library_facts(owner, viewer);
        if self.shares_information(viewer, owner) {
            if let Some(suspended) = &self.state.suspended_decision {
                for decision in suspended.decisions() {
                    let domain = match decision {
                        crate::flow::decision::Decision::OrderLibrary {
                            player,
                            object_ids,
                            ..
                        } => Some((*player, object_ids.as_slice())),
                        _ => decision.library_inspection(),
                    };
                    if let Some((player, objects)) = domain.filter(|(player, _)| *player == owner) {
                        for (position, object) in self
                            .state
                            .zones
                            .zone_objects(ZoneType::Library, player)
                            .iter()
                            .rev()
                            .enumerate()
                        {
                            if objects.contains(object) {
                                facts.insert(position, self.state[*object].definition_id);
                            }
                        }
                    }
                }
            }
        }
        if facts.is_empty() {
            return Some(Vec::new());
        }
        let commitments = self.hidden_commitments(viewer);
        let mut reserved = Vec::new();
        for (position, definition) in facts {
            let original = self
                .state
                .zones
                .zone_objects(ZoneType::Library, owner)
                .iter()
                .rev()
                .nth(position)
                .copied()?;
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
                    self.state[**object].definition_id == definition && !commitments.pins(**object)
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
        rng: &mut impl Rng,
    ) {
        if self.library_facts(owner, viewer).is_empty() && self.state.suspended_decision.is_none() {
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
        pool.sort_unstable_by_key(|object| object.slot());
        let reserved = self
            .reserve_library_facts(owner, viewer, &mut pool)
            .expect("library facts are true in the source world");
        pool.shuffle(rng);
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
            self.state.known_library_mut(opponent).own = public;
        }
    }

    pub(crate) fn library_knowledge_holds(&self, viewer: PlayerId) -> bool {
        [PlayerId(0), PlayerId(1)].into_iter().all(|owner| {
            let library = self.state.zones.zone_objects(ZoneType::Library, owner);
            self.library_facts(owner, viewer)
                .iter()
                .all(|(position, definition)| {
                    library
                        .iter()
                        .rev()
                        .nth(*position)
                        .is_some_and(|object| self.state[*object].definition_id == *definition)
                })
        })
    }

    pub(crate) fn assert_library_knowledge(&self) {
        #[cfg(debug_assertions)]
        for owner in [PlayerId(0), PlayerId(1)] {
            let library = self.state.zones.zone_objects(ZoneType::Library, owner);
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
                        library
                            .iter()
                            .rev()
                            .nth(*position)
                            .map(|object| self.state[*object].definition_id),
                        Some(*definition),
                        "false library fact for {owner:?} at {position}"
                    );
                }
            }
        }
    }
}
