use std::array;

use rand::seq::SliceRandom;

use super::game_object::{ObjectId, PlayerId};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
#[repr(i32)]
pub enum ZoneType {
    Library = 0,
    Hand = 1,
    Battlefield = 2,
    Graveyard = 3,
    Stack = 4,
    Exile = 5,
    Command = 6,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct ZoneManager {
    library: [Vec<ObjectId>; 2],
    hand: [Vec<ObjectId>; 2],
    battlefield: [Vec<ObjectId>; 2],
    graveyard: [Vec<ObjectId>; 2],
    stack: [Vec<ObjectId>; 2],
    exile: [Vec<ObjectId>; 2],
    command: [Vec<ObjectId>; 2],
    #[serde(rename = "card_zones")]
    object_zones: Vec<Option<ZoneType>>,
}

impl Default for ZoneManager {
    fn default() -> Self {
        Self {
            library: array::from_fn(|_| Vec::new()),
            hand: array::from_fn(|_| Vec::new()),
            battlefield: array::from_fn(|_| Vec::new()),
            graveyard: array::from_fn(|_| Vec::new()),
            stack: array::from_fn(|_| Vec::new()),
            exile: array::from_fn(|_| Vec::new()),
            command: array::from_fn(|_| Vec::new()),
            object_zones: Vec::new(),
        }
    }
}

impl ZoneManager {
    /// The player whose `zone` holds `card`.
    fn holder_in(&self, object: ObjectId, zone: ZoneType) -> Option<PlayerId> {
        [PlayerId(0), PlayerId(1)]
            .into_iter()
            .find(|player| self.zone_objects(zone, *player).contains(&object))
    }

    fn ensure_slot(&mut self, object: ObjectId) {
        if self.object_zones.len() <= object.0 {
            self.object_zones.resize(object.0 + 1, None);
        }
    }

    pub fn zone_of(&self, object: ObjectId) -> Option<ZoneType> {
        self.object_zones.get(object.0).copied().flatten()
    }

    fn remove_object_in_place(objects: &mut Vec<ObjectId>, object: ObjectId) {
        objects.retain(|c| *c != object);
    }

    fn remove_from_zone(&mut self, object: ObjectId, zone: ZoneType) {
        let Some(holder) = self.holder_in(object, zone) else {
            return;
        };
        let idx = holder.0;
        match zone {
            ZoneType::Library => Self::remove_object_in_place(&mut self.library[idx], object),
            ZoneType::Hand => Self::remove_object_in_place(&mut self.hand[idx], object),
            ZoneType::Battlefield => {
                Self::remove_object_in_place(&mut self.battlefield[idx], object)
            }
            ZoneType::Graveyard => Self::remove_object_in_place(&mut self.graveyard[idx], object),
            ZoneType::Stack => Self::remove_object_in_place(&mut self.stack[idx], object),
            ZoneType::Exile => Self::remove_object_in_place(&mut self.exile[idx], object),
            ZoneType::Command => Self::remove_object_in_place(&mut self.command[idx], object),
        }
    }

    fn insert_to_zone(&mut self, object: ObjectId, holder: PlayerId, zone: ZoneType) {
        let idx = holder.0;
        match zone {
            ZoneType::Library => self.library[idx].push(object),
            ZoneType::Hand => self.hand[idx].push(object),
            ZoneType::Battlefield => self.battlefield[idx].push(object),
            ZoneType::Graveyard => self.graveyard[idx].push(object),
            ZoneType::Stack => self.stack[idx].push(object),
            ZoneType::Exile => self.exile[idx].push(object),
            ZoneType::Command => self.command[idx].push(object),
        }
    }

    /// Remove an object from all zones entirely (CR 704.5d — tokens cease to
    /// exist). The object remains in `GameState::objects` but is no
    /// longer in any zone.
    pub fn remove_object(&mut self, object: ObjectId) {
        if let Some(zone) = self.zone_of(object) {
            self.remove_from_zone(object, zone);
            self.object_zones[object.0] = None;
        }
    }

    /// Move `card` from wherever it is to the end of `holder`'s `to_zone`.
    /// The holder is the card's owner (CR 108.3), except on the
    /// battlefield, where it is the permanent's controller (CR 110.2): each
    /// player's battlefield lists the permanents they control.
    pub fn move_object(&mut self, object: ObjectId, holder: PlayerId, to_zone: ZoneType) {
        self.ensure_slot(object);
        if let Some(old_zone) = self.zone_of(object) {
            self.remove_from_zone(object, old_zone);
        }
        self.insert_to_zone(object, holder, to_zone);
        self.object_zones[object.0] = Some(to_zone);
    }

    pub fn top(&self, zone: ZoneType, player: PlayerId) -> Option<ObjectId> {
        let objects = self.zone_objects(zone, player);
        objects.last().copied()
    }

    pub fn contains(&self, object: ObjectId, zone: ZoneType, player: PlayerId) -> bool {
        self.zone_objects(zone, player).contains(&object)
    }

    pub fn size(&self, zone: ZoneType, player: PlayerId) -> usize {
        self.zone_objects(zone, player).len()
    }

    pub fn zone_objects(&self, zone: ZoneType, player: PlayerId) -> &Vec<ObjectId> {
        let idx = player.0;
        match zone {
            ZoneType::Library => &self.library[idx],
            ZoneType::Hand => &self.hand[idx],
            ZoneType::Battlefield => &self.battlefield[idx],
            ZoneType::Graveyard => &self.graveyard[idx],
            ZoneType::Stack => &self.stack[idx],
            ZoneType::Exile => &self.exile[idx],
            ZoneType::Command => &self.command[idx],
        }
    }

    pub fn zone_objects_mut(&mut self, zone: ZoneType, player: PlayerId) -> &mut Vec<ObjectId> {
        let idx = player.0;
        match zone {
            ZoneType::Library => &mut self.library[idx],
            ZoneType::Hand => &mut self.hand[idx],
            ZoneType::Battlefield => &mut self.battlefield[idx],
            ZoneType::Graveyard => &mut self.graveyard[idx],
            ZoneType::Stack => &mut self.stack[idx],
            ZoneType::Exile => &mut self.exile[idx],
            ZoneType::Command => &mut self.command[idx],
        }
    }

    pub fn shuffle<R: rand::Rng + ?Sized>(
        &mut self,
        zone: ZoneType,
        player: PlayerId,
        rng: &mut R,
    ) {
        self.zone_objects_mut(zone, player).shuffle(rng);
    }

    /// Shuffle an unordered hidden zone from a canonical starting order.
    ///
    /// Belief-state sampling must depend on the viewer-visible card multiset
    /// and seed, never on the authority's current private ordering.
    pub fn shuffle_canonical<R: rand::Rng + ?Sized>(
        &mut self,
        zone: ZoneType,
        player: PlayerId,
        rng: &mut R,
    ) {
        let objects = self.zone_objects_mut(zone, player);
        objects.sort_unstable_by_key(|object| object.0);
        objects.shuffle(rng);
    }

    /// Put hidden card `new` where `player`'s face-down exiled card `old`
    /// lies. `old` is left zoneless until the caller reassigns it to the
    /// hand or library with [`ZoneManager::reassign_hidden`].
    pub fn replace_face_down_exile(&mut self, player: PlayerId, old: ObjectId, new: ObjectId) {
        if let Some(slot) = self.exile[player.0]
            .iter_mut()
            .find(|object| **object == old)
        {
            *slot = new;
        }
        self.ensure_slot(new);
        self.object_zones[new.0] = Some(ZoneType::Exile);
        self.object_zones[old.0] = None;
    }

    /// Reassign `player`'s hand and library to exactly `new_hand` and
    /// `new_library`, updating `object_zones` for every object. Used by exact
    /// possible-world materialization: the caller chooses which physical
    /// cards land in which hidden zone, so the resulting branch is a valid
    /// authoritative state rather than a uniform sample. The union of
    /// `new_hand` and `new_library` must be exactly the prior hand ∪ library.
    pub fn reassign_hidden(
        &mut self,
        player: PlayerId,
        new_hand: Vec<ObjectId>,
        new_library: Vec<ObjectId>,
    ) {
        let idx = player.0;
        for &object in &new_hand {
            self.ensure_slot(object);
            self.object_zones[object.0] = Some(ZoneType::Hand);
        }
        for &object in &new_library {
            self.ensure_slot(object);
            self.object_zones[object.0] = Some(ZoneType::Library);
        }
        self.hand[idx] = new_hand;
        self.library[idx] = new_library;
    }
}
