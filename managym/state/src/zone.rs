//! Ordered zone membership. Library and hand entries carry checked card ids
//! or transient non-card occupants; other zones hold object ids. Borrowed views
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
#[derive(Clone, Copy, Debug, serde::Serialize)]
#[serde(untagged)]
enum CardZoneEntry {
    Card(CardId),
    Transient(ObjectId),
}

impl CardZoneEntry {
    fn new(objects: &ObjectVec<Object>, object: ObjectId) -> Self {
        match objects.card_id(object) {
            Some(card) => Self::Card(card),
            None => Self::Transient(object),
        }
    }

    fn object(&self) -> &ObjectId {
        match self {
            Self::Card(card) => card.as_ref(),
            Self::Transient(object) => object,
        }
    }
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
        self.entries
            .iter()
            .filter_map(|entry| match entry {
                CardZoneEntry::Card(card) => Some(*card),
                CardZoneEntry::Transient(_) => None,
            })
            .chain(
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
        match zone {
            ZoneType::Library => self.library[holder.0].retain(|entry| *entry.object() != object),
            ZoneType::Hand => self.hand[holder.0].retain(|entry| *entry.object() != object),
            _ => self
                .mixed_objects_mut(zone, holder)
                .retain(|candidate| *candidate != object),
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
        match to_zone {
            ZoneType::Library => self.library[holder.0].push(CardZoneEntry::new(objects, object)),
            ZoneType::Hand => self.hand[holder.0].push(CardZoneEntry::new(objects, object)),
            _ => self.mixed_objects_mut(to_zone, holder).push(object),
        }
        self.object_zones[object.0] = Some(to_zone);
    }

    pub fn top(&self, zone: ZoneType, player: PlayerId) -> Option<ObjectId> {
        let objects = self.zone_objects(zone, player);
        objects.last().copied()
    }

    pub fn contains(&self, object: impl Into<ObjectId>, zone: ZoneType, player: PlayerId) -> bool {
        let object: ObjectId = object.into();
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

    fn mixed_objects_mut(&mut self, zone: ZoneType, player: PlayerId) -> &mut Vec<ObjectId> {
        let idx = player.0;
        match zone {
            ZoneType::Library | ZoneType::Hand => unreachable!("typed card zone"),
            ZoneType::Battlefield => &mut self.battlefield[idx],
            ZoneType::Graveyard => &mut self.graveyard[idx],
            ZoneType::Stack => &mut self.stack[idx],
            ZoneType::Exile => &mut self.exile[idx],
            ZoneType::Command => &mut self.command[idx],
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
        match zone {
            ZoneType::Library | ZoneType::Hand => {
                let entries = if zone == ZoneType::Library {
                    &mut self.library[player.0]
                } else {
                    &mut self.hand[player.0]
                };
                let entry = entries.remove(old);
                entries.insert(index.min(entries.len()), entry);
            }
            _ => {
                let objects = self.mixed_objects_mut(zone, player);
                let object = objects.remove(old);
                objects.insert(index.min(objects.len()), object);
            }
        }
    }

    pub fn swap(&mut self, zone: ZoneType, player: PlayerId, a: usize, b: usize) {
        match zone {
            ZoneType::Library => self.library[player.0].swap(a, b),
            ZoneType::Hand => self.hand[player.0].swap(a, b),
            _ => self.mixed_objects_mut(zone, player).swap(a, b),
        }
    }

    pub fn shuffle<R: rand::Rng + ?Sized>(
        &mut self,
        zone: ZoneType,
        player: PlayerId,
        rng: &mut R,
    ) {
        match zone {
            ZoneType::Library => self.library[player.0].shuffle(rng),
            ZoneType::Hand => self.hand[player.0].shuffle(rng),
            _ => self.mixed_objects_mut(zone, player).shuffle(rng),
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
        match zone {
            ZoneType::Library => {
                self.library[player.0].sort_unstable_by_key(|entry| entry.object().0)
            }
            ZoneType::Hand => self.hand[player.0].sort_unstable_by_key(|entry| entry.object().0),
            _ => self
                .mixed_objects_mut(zone, player)
                .sort_unstable_by_key(|object| object.0),
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
