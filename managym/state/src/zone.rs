//! Ordered zone membership. Library and hand entries carry an appearance and
//! its optional physical card; other zones hold object ids. Borrowed views
//! and serialization project the same numeric ids without another stored list.

use rand::seq::SliceRandom;

use super::game_object::{CardId, Object, ObjectId, ObjectVec, PlayerId};

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

/// An occupant of a card zone. Non-cards are visitors until state-based actions;
/// they keep their position but can never be returned by a card-only scan.
#[derive(Clone, Copy, Debug)]
struct CardZoneEntry {
    object: ObjectId,
    card: Option<CardId>,
}

// Keep the previous entry budget while storing both physical-card identity
// and appearance. `Option<CardId>` uses the nonzero roster index's niche.
const _: () = assert!(std::mem::size_of::<CardZoneEntry>() <= 16);

impl serde::Serialize for CardZoneEntry {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serde::Serialize::serialize(&self.object, serializer)
    }
}

impl CardZoneEntry {
    fn new(objects: &ObjectVec<Object>, object: ObjectId) -> Self {
        Self {
            object,
            card: objects.card_id(object),
        }
    }

    fn object(&self) -> &ObjectId {
        &self.object
    }
}

/// One zone's backing list: typed entries for library and hand, object ids
/// elsewhere. Every mutation goes through this so both shapes stay in step.
enum ZoneStore<'a> {
    Cards(&'a mut Vec<CardZoneEntry>),
    Objects(&'a mut Vec<ObjectId>),
}

/// Borrowed ordered objects, without allocating a second zone representation.
#[derive(Clone, Copy, Debug)]
pub struct ZoneObjects<'a> {
    entries: &'a [CardZoneEntry],
    objects: &'a [ObjectId],
}

impl<'a> ZoneObjects<'a> {
    pub fn iter(self) -> impl DoubleEndedIterator<Item = &'a ObjectId> + Clone {
        self.entries
            .iter()
            .map(CardZoneEntry::object)
            .chain(self.objects)
    }

    /// Read stored checked ids in card zones; narrow mixed zones once here.
    pub fn cards(self, objects: &'a ObjectVec<Object>) -> impl Iterator<Item = CardId> + 'a {
        self.entries.iter().filter_map(|entry| entry.card).chain(
            self.objects
                .iter()
                .filter_map(|object| objects.card_id(*object)),
        )
    }

    pub fn len(self) -> usize {
        self.entries.len() + self.objects.len()
    }

    pub fn is_empty(self) -> bool {
        self.len() == 0
    }

    pub fn contains(self, object: impl Into<ObjectId>) -> bool {
        let object: ObjectId = object.into();
        self.iter().any(|candidate| *candidate == object)
    }

    pub fn last(self) -> Option<&'a ObjectId> {
        self.iter().next_back()
    }

    pub fn to_vec(self) -> Vec<ObjectId> {
        self.iter().copied().collect()
    }
}

impl std::ops::Index<usize> for ZoneObjects<'_> {
    type Output = ObjectId;
    fn index(&self, index: usize) -> &ObjectId {
        if self.entries.is_empty() {
            &self.objects[index]
        } else {
            self.entries[index].object()
        }
    }
}

impl PartialEq for ZoneObjects<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct ZoneManager {
    library: [Vec<CardZoneEntry>; 2],
    hand: [Vec<CardZoneEntry>; 2],
    battlefield: [Vec<ObjectId>; 2],
    graveyard: [Vec<ObjectId>; 2],
    stack: [Vec<ObjectId>; 2],
    exile: [Vec<ObjectId>; 2],
    command: [Vec<ObjectId>; 2],
    #[serde(rename = "card_zones")]
    object_zones: Vec<Option<ZoneType>>,
}

impl ZoneManager {
    /// The player whose `zone` holds `object`.
    fn holder_in(&self, object: ObjectId, zone: ZoneType) -> Option<PlayerId> {
        [PlayerId(0), PlayerId(1)]
            .into_iter()
            .find(|player| self.zone_objects(zone, *player).contains(object))
    }

    fn ensure_slot(&mut self, object: ObjectId) {
        if self.object_zones.len() <= object.0 {
            self.object_zones.resize(object.0 + 1, None);
        }
    }

    pub fn zone_of(&self, object: impl Into<ObjectId>) -> Option<ZoneType> {
        let object: ObjectId = object.into();
        self.object_zones.get(object.0).copied().flatten()
    }

    fn remove_from_zone(&mut self, object: ObjectId, zone: ZoneType) {
        let Some(holder) = self.holder_in(object, zone) else {
            return;
        };
        match self.store_mut(zone, holder) {
            ZoneStore::Cards(entries) => entries.retain(|entry| *entry.object() != object),
            ZoneStore::Objects(objects) => objects.retain(|candidate| *candidate != object),
        }
    }

    /// Remove an object from all zones entirely (CR 704.5d — tokens cease to
    /// exist). The object remains in `GameState::objects` but is no
    /// longer in any zone.
    pub fn remove_object(&mut self, object: impl Into<ObjectId>) {
        let object: ObjectId = object.into();
        if let Some(zone) = self.zone_of(object) {
            self.remove_from_zone(object, zone);
            self.object_zones[object.0] = None;
        }
    }

    /// Move `object` from wherever it is to the end of `holder`'s `to_zone`.
    /// The holder is the object's owner (CR 108.3, 111.2), except on the
    /// battlefield, where it is the permanent's controller (CR 110.2): each
    /// player's battlefield lists the permanents they control.
    pub fn move_object(
        &mut self,
        objects: &ObjectVec<Object>,
        object: impl Into<ObjectId>,
        holder: PlayerId,
        to_zone: ZoneType,
    ) {
        let object: ObjectId = object.into();
        self.ensure_slot(object);
        if let Some(old_zone) = self.zone_of(object) {
            self.remove_from_zone(object, old_zone);
        }
        match self.store_mut(to_zone, holder) {
            ZoneStore::Cards(entries) => entries.push(CardZoneEntry::new(objects, object)),
            ZoneStore::Objects(zone) => zone.push(object),
        }
        self.object_zones[object.0] = Some(to_zone);
    }

    pub fn top(&self, zone: ZoneType, player: PlayerId) -> Option<ObjectId> {
        self.zone_objects(zone, player).last().copied()
    }

    pub fn contains(&self, object: impl Into<ObjectId>, zone: ZoneType, player: PlayerId) -> bool {
        self.zone_objects(zone, player).contains(object)
    }

    pub fn size(&self, zone: ZoneType, player: PlayerId) -> usize {
        self.zone_objects(zone, player).len()
    }

    pub fn zone_objects(&self, zone: ZoneType, player: PlayerId) -> ZoneObjects<'_> {
        let idx = player.0;
        let mut view = ZoneObjects {
            entries: &[],
            objects: &[],
        };
        match zone {
            ZoneType::Library => view.entries = &self.library[idx],
            ZoneType::Hand => view.entries = &self.hand[idx],
            ZoneType::Battlefield => view.objects = &self.battlefield[idx],
            ZoneType::Graveyard => view.objects = &self.graveyard[idx],
            ZoneType::Stack => view.objects = &self.stack[idx],
            ZoneType::Exile => view.objects = &self.exile[idx],
            ZoneType::Command => view.objects = &self.command[idx],
        }
        view
    }

    fn store_mut(&mut self, zone: ZoneType, player: PlayerId) -> ZoneStore<'_> {
        let idx = player.0;
        match zone {
            ZoneType::Library => ZoneStore::Cards(&mut self.library[idx]),
            ZoneType::Hand => ZoneStore::Cards(&mut self.hand[idx]),
            ZoneType::Battlefield => ZoneStore::Objects(&mut self.battlefield[idx]),
            ZoneType::Graveyard => ZoneStore::Objects(&mut self.graveyard[idx]),
            ZoneType::Stack => ZoneStore::Objects(&mut self.stack[idx]),
            ZoneType::Exile => ZoneStore::Objects(&mut self.exile[idx]),
            ZoneType::Command => ZoneStore::Objects(&mut self.command[idx]),
        }
    }

    /// Reorder an existing occupant; `index` is its position after removal,
    /// clamped to the end. This preserves its kind and zone membership.
    pub fn reposition(
        &mut self,
        zone: ZoneType,
        player: PlayerId,
        object: impl Into<ObjectId>,
        index: usize,
    ) {
        let object: ObjectId = object.into();
        let Some(old) = self
            .zone_objects(zone, player)
            .iter()
            .position(|id| *id == object)
        else {
            return;
        };
        fn reinsert<T>(occupants: &mut Vec<T>, old: usize, index: usize) {
            let occupant = occupants.remove(old);
            occupants.insert(index.min(occupants.len()), occupant);
        }
        match self.store_mut(zone, player) {
            ZoneStore::Cards(entries) => reinsert(entries, old, index),
            ZoneStore::Objects(objects) => reinsert(objects, old, index),
        }
    }

    pub fn swap(&mut self, zone: ZoneType, player: PlayerId, a: usize, b: usize) {
        match self.store_mut(zone, player) {
            ZoneStore::Cards(entries) => entries.swap(a, b),
            ZoneStore::Objects(objects) => objects.swap(a, b),
        }
    }

    pub fn shuffle<R: rand::Rng + ?Sized>(
        &mut self,
        zone: ZoneType,
        player: PlayerId,
        rng: &mut R,
    ) {
        match self.store_mut(zone, player) {
            ZoneStore::Cards(entries) => entries.shuffle(rng),
            ZoneStore::Objects(objects) => objects.shuffle(rng),
        }
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
        match self.store_mut(zone, player) {
            ZoneStore::Cards(entries) => entries.sort_unstable_by_key(|entry| entry.object().0),
            ZoneStore::Objects(objects) => objects.sort_unstable_by_key(|object| object.0),
        }
        self.shuffle(zone, player, rng);
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
        objects: &ObjectVec<Object>,
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
        self.hand[idx] = new_hand
            .into_iter()
            .map(|object| CardZoneEntry::new(objects, object))
            .collect();
        self.library[idx] = new_library
            .into_iter()
            .map(|object| CardZoneEntry::new(objects, object))
            .collect();
    }
}
